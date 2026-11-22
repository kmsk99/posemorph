"""Small ONNX CPU adapter; weights live outside the source tree."""

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

POSE_FILE = "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
POSE_URL = (
    "https://huggingface.co/Tau-J/RTMPose/resolve/cd4d709/rtmposev1/onnx_sdk/"
    + POSE_FILE
)
DETECT_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"

MODEL_SHA256 = {
    "detector": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    "pose": "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c",
}

HUMANART_URLS = {
    "detector": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.zip",
}
HUMANART_SHA256 = {
    "detector": "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
    "pose": "443c883202be9afa2dac1cb03257e9475f4c5205aea3b8be97086293ee12b237",
}
POSE_BACKENDS = ("body7", "body7-rgb", "humanart")


def cached_model(url, cache):
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    dest = cache / name
    if not dest.exists():
        print(f"Downloading {name}", flush=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
    if dest.suffix == ".zip":
        target = cache / (dest.stem + ".onnx")
        if not target.exists():
            with zipfile.ZipFile(dest) as archive:
                member = next(n for n in archive.namelist() if n.endswith(".onnx"))
                partial = target.with_suffix(".onnx.part")
                partial.write_bytes(archive.read(member))
                partial.replace(target)
        dest = target
    return dest


class Detector:
    def __init__(self, cache=None, backend="body7"):
        if backend not in POSE_BACKENDS:
            raise ValueError(f"Unknown pose backend: {backend}")
        import onnxruntime as ort

        cache = cache or os.environ.get("POSEMORPH_CACHE", "~/.cache/posemorph")
        self.backend = backend
        self.pose_rgb = backend != "body7"
        self.detector_format = "nms_xyxy" if backend == "humanart" else "yolox_raw"
        urls = (
            HUMANART_URLS
            if backend == "humanart"
            else {"detector": DETECT_URL, "pose": POSE_URL}
        )
        expected = HUMANART_SHA256 if backend == "humanart" else MODEL_SHA256
        if backend == "humanart":
            cache = Path(cache).expanduser() / "artpose"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.models = {}
        for name, url in urls.items():
            path = cached_model(url, cache)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != expected[name]:
                raise ValueError(
                    f"Cached model checksum mismatch: {path}; remove it and retry"
                )
            self.models[name] = {
                "url": url,
                "sha256": checksum,
                "backend": backend,
                "runtime": "onnxruntime-cpu",
                "threads": 1,
                "input_color": "RGB" if name == "pose" and self.pose_rgb else "BGR",
                "license_status": "experimental: Human-Art data non-commercial; weight commercial scope unresolved"
                if backend == "humanart"
                else "published Apache-2.0",
            }
            setattr(
                self,
                name,
                ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                ),
            )

    def boxes(self, image):
        h, w = image.shape[:2]
        ratio = min(640 / h, 640 / w)
        resized = cv2.resize(image, (int(w * ratio), int(h * ratio)))
        padded = np.full((640, 640, 3), 114, np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        data = padded.transpose(2, 0, 1)[None].astype(np.float32)
        predictions = self.detector.run(
            None, {self.detector.get_inputs()[0].name: data}
        )
        outputs = predictions[0][0]
        if getattr(self, "detector_format", "yolox_raw") == "nms_xyxy":
            # Official MMDeploy graph already decodes boxes and performs NMS.
            # Its coordinates are input-space xyxy, not raw YOLOX centre/log-size.
            if outputs.ndim != 2 or outputs.shape[-1] != 5:
                raise ValueError("Expected HumanArt detector output [1,N,5]")
            labels = np.asarray(predictions[1][0]).reshape(-1)
            valid = (outputs[:, 4] >= 0.25) & (labels == 0)
            boxes = outputs[valid, :4] / ratio
            boxes = np.clip(boxes, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            return boxes.astype(float).tolist()
        grids, strides = [], []
        for stride in [8, 16, 32]:
            x, y = np.meshgrid(np.arange(640 // stride), np.arange(640 // stride))
            grids.append(np.column_stack([x.ravel(), y.ravel()]))
            strides.append(np.full((x.size, 1), stride))
        strides = np.vstack(strides)
        center = (outputs[:, :2] + np.vstack(grids)) * strides
        extent = np.exp(outputs[:, 2:4]) * strides
        scores = outputs[:, 4] * outputs[:, 5]  # COCO person, never other classes
        valid = scores >= 0.25
        boxes = np.column_stack([center - extent / 2, extent]) / ratio  # xywh
        indices = cv2.dnn.NMSBoxes(
            boxes[valid].tolist(), scores[valid].tolist(), 0.25, 0.45
        )
        result = []
        for index in np.asarray(indices).ravel():
            x, y, bw, bh = boxes[valid][index]
            result.append(
                [max(0.0, x), max(0.0, y), min(w - 1.0, x + bw), min(h - 1.0, y + bh)]
            )
        return result

    def keypoints(self, image, box):
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        extent = np.array([x2 - x1, y2 - y1]) * 1.25
        extent[0] = max(extent[0], extent[1] * 192 / 256)
        extent[1] = extent[0] * 256 / 192
        matrix = np.array(
            [
                [192 / extent[0], 0, 96 - center[0] * 192 / extent[0]],
                [0, 256 / extent[1], 128 - center[1] * 256 / extent[1]],
            ],
            np.float32,
        )
        crop = cv2.warpAffine(image, matrix, (192, 256))
        # The official SDK pipeline.json explicitly specifies to_rgb=true.
        # body7 preserves v0/v1's historical BGR behavior for reproducibility;
        # body7-rgb isolates this correction from HumanArt model/domain changes.
        if getattr(self, "pose_rgb", False):
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = (crop.astype(np.float32) - [123.675, 116.28, 103.53]) / [
            58.395,
            57.12,
            57.375,
        ]
        data = crop.transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = self.pose.run(None, {self.pose.get_inputs()[0].name: data})
        points = np.column_stack([sx[0].argmax(-1), sy[0].argmax(-1)]) / 2
        points = points / [192, 256] * extent + center - extent / 2
        self.raw_scores = np.minimum(sx[0].max(-1), sy[0].max(-1))
        scores = np.clip(self.raw_scores, 0, 1)
        return points, scores

    def __call__(self, image):
        h, w = image.shape[:2]
        boxes = self.boxes(image)
        fallback = not boxes
        boxes = boxes or [[0, 0, w - 1, h - 1]]
        people = []
        for box in boxes:
            points, scores = self.keypoints(image, box)
            cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
            area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
            rank = area * np.exp(-2 * ((cx - 0.5) ** 2 + (cy - 0.45) ** 2))
            people.append(
                {
                    "box": box,
                    "keypoints": points.tolist(),
                    "scores": scores.tolist(),
                    "raw_scores": self.raw_scores.tolist(),
                    "prominence": float(rank),
                    "box_source": "full_image_fallback" if fallback else "detector",
                }
            )
        return sorted(people, key=lambda p: -p["prominence"])
