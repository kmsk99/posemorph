import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_pipeline import make_fixture

from posemorph import detector, pipeline
from posemorph.cli import main


def test_mmdeploy_boxes_are_already_xyxy_and_nms_filtered():
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, names, inputs):
            boxes = np.array(
                [
                    [
                        [64, 128, 320, 512, 0.9],
                        [0, 0, 640, 640, 0.95],
                        [10, 10, 20, 20, 0.1],
                        [650, 650, 700, 700, 0.9],
                    ]
                ],
                np.float32,
            )
            return boxes, np.array([[0, 1, 0, 0]])

    model = detector.Detector.__new__(detector.Detector)
    model.detector = Session()
    model.detector_format = "nms_xyxy"
    # A raw-YOLOX decode on these coordinates would explode; don't decode twice.
    np.testing.assert_allclose(
        model.boxes(np.zeros((100, 100, 3), np.uint8)), [[10, 20, 50, 80]]
    )


def test_artpose_rgb_conversion_and_legacy_bgr_are_explicit():
    captured = []

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, names, inputs):
            captured.append(inputs["input"][0, :, 128, 96].copy())
            return np.ones((1, 17, 384), np.float32), np.ones((1, 17, 512), np.float32)

    model = detector.Detector.__new__(detector.Detector)
    model.pose = Session()
    image = np.full((100, 100, 3), [0, 0, 255], np.uint8)
    model.pose_rgb = False
    model.keypoints(image, [0, 0, 99, 99])
    model.pose_rgb = True
    points, scores = model.keypoints(image, [0, 0, 99, 99])
    assert captured[0][0] == pytest.approx(-123.675 / 58.395)
    assert captured[1][0] == pytest.approx((255 - 123.675) / 58.395)
    assert points.shape == (17, 2) and np.all(scores == 1)


def test_backend_change_invalidates_detection_cache(tmp_path, monkeypatch):
    source, poses = make_fixture(tmp_path)
    person = json.loads(poses.read_text())["0.jpg"][0]
    calls = []

    class FakeDetector:
        def __init__(self, cache=None, backend="body7"):
            self.backend = backend
            self.models = {"pose": {"backend": backend}}

        def __call__(self, image):
            calls.append(self.backend)
            return [dict(person, box_source="detector")]

    monkeypatch.setattr(detector, "Detector", FakeDetector)
    out = tmp_path / "out"
    pipeline.detect(source, out, pose_backend="body7")
    assert calls == ["body7"] * 4
    pipeline.detect(source, out, pose_backend="body7")
    assert len(calls) == 4
    pipeline.detect(source, out, pose_backend="humanart")
    assert calls[-4:] == ["humanart"] * 4 and len(calls) == 8
    assert pipeline.load(out, "detect")["models"]["pose"]["backend"] == "humanart"
    pipeline.detect(source, out, pose_backend="body7-rgb")
    assert calls[-4:] == ["body7-rgb"] * 4
    with pytest.raises(SystemExit):
        main(["detect", str(source), "--out", str(out), "--pose-backend", "unknown"])
