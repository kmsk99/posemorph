import json
import shutil

import cv2
import numpy as np
import pytest

from posemorph.cli import main
from posemorph.geometry import EDGES
from posemorph.pipeline import load


def make_fixture(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    points = np.array(
        [
            [50, 20],
            [47, 18],
            [53, 18],
            [43, 20],
            [57, 20],
            [65, 35],
            [35, 35],
            [78, 49],
            [22, 49],
            [86, 65],
            [14, 65],
            [60, 70],
            [40, 70],
            [64, 100],
            [36, 100],
            [68, 125],
            [32, 125],
        ],
        float,
    )
    poses = {}
    for i in range(4):
        p = points.copy()
        p[9, 1] -= i * 4
        image = np.full((144, 100, 3), 240, np.uint8)
        for a, b in EDGES:
            cv2.line(
                image,
                tuple(p[a].astype(int)),
                tuple(p[b].astype(int)),
                (40, 60, 100),
                2,
            )
        for point in p:
            cv2.circle(image, tuple(point.astype(int)), 2, (80, 0, 200), -1)
        filename = f"{i}.jpg"
        cv2.imwrite(str(source / filename), image)
        scores = np.ones(17)
        if i == 3:
            scores[11] = 0.1
        poses[filename] = [
            {"box": [5, 5, 95, 140], "keypoints": p.tolist(), "scores": scores.tolist()}
        ]
    pose_file = tmp_path / "poses.json"
    pose_file.write_text(json.dumps(poses))
    return source, pose_file


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_end_to_end_cut_and_resume(tmp_path):
    source, poses = make_fixture(tmp_path)
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                str(source),
                "--out",
                str(out),
                "--poses",
                str(poses),
                "--size",
                "90x160",
                "--hold",
                "2",
                "--fps",
                "10",
            ]
        )
        == 0
    )
    result = load(out, "report")
    assert result["input_count"] == 4
    assert result["selected_count"] == 3
    assert result["drop_reasons"] == {"low_hip_confidence": 1}
    assert result["videos"]["cut"]["frames"] == 6
    assert (out / "contact-sheet.jpg").exists()
    cap = cv2.VideoCapture(str(out / "cut.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 90
    cap.release()
    assert (
        main(
            [
                "render",
                "--out",
                str(out),
                "--mode",
                "morph",
                "--hold",
                "1",
                "--transition",
                "2",
            ]
        )
        == 0
    )
    assert load(out, "render-morph")["frames"] == 7
    # Fresh upstream selection invalidates downstream geometry/order.
    assert main(["select", "--out", str(out), "--max-images", "2"]) == 0
    assert main(["render", "--out", str(out)]) == 1
    # Detect can reuse imported poses without importing ONNX or needing network.
    assert main(["detect", str(source), "--out", str(out), "--poses", str(poses)]) == 0
    assert load(out, "detect")["complete"]


def test_person_override_and_empty_selection(tmp_path):
    source, pose_file = make_fixture(tmp_path)
    poses = json.loads(pose_file.read_text())
    poses["0.jpg"].append({**poses["0.jpg"][0], "scores": [0.01] * 17})
    pose_file.write_text(json.dumps(poses))
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"0.jpg": 1}))
    out = tmp_path / "out"
    assert (
        main(
            [
                "detect",
                str(source),
                "--out",
                str(out),
                "--poses",
                str(pose_file),
                "--person-overrides",
                str(overrides),
            ]
        )
        == 0
    )
    assert load(out, "detect")["records"][0]["chosen"] == 1
    assert main(["select", "--out", str(out)]) == 0
    assert load(out, "select")["usable_before_cluster_and_limit"] == 2


def test_bad_size():
    with pytest.raises(SystemExit):
        main(["align", "--out", "unused", "--size", "91x160"])


@pytest.mark.model
def test_real_detector(tmp_path):
    from posemorph.detector import Detector

    model = Detector()
    people = model(np.full((256, 192, 3), 128, np.uint8))
    assert len(people) >= 1
    assert np.asarray(people[0]["keypoints"]).shape == (17, 2)
    assert set(model.models) == {"detector", "pose"}


def test_hidden_sidecars_and_source_changes(tmp_path):
    source, poses = make_fixture(tmp_path)
    (source / "._0.jpg").write_bytes(b"not an image: macOS resource fork")
    out = tmp_path / "out"
    assert main(["detect", str(source), "--out", str(out), "--poses", str(poses)]) == 0
    assert len(load(out, "detect")["records"]) == 4
    assert main(["select", "--out", str(out)]) == 0
    (source / "0.jpg").write_bytes(b"changed since detection")
    assert main(["align", "--out", str(out), "--size", "90x160"]) == 1


def test_simcc_scores_are_bounded():
    from posemorph.detector import Detector

    class FakeSession:
        def get_inputs(self):
            class Input:
                name = "image"

            return [Input()]

        def run(self, names, data):
            x = np.full((1, 17, 384), -1.0, np.float32)
            y = np.full((1, 17, 512), -1.0, np.float32)
            x[:, :, 100], y[:, :, 100] = 1.4, 1.2
            return x, y

    detector = Detector.__new__(Detector)
    detector.pose = FakeSession()
    points, scores = detector.keypoints(
        np.zeros((256, 192, 3), np.uint8), [0, 0, 191, 255]
    )
    assert points.shape == (17, 2)
    np.testing.assert_array_equal(scores, np.ones(17))


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe required",
)
def test_short_audio_is_padded_and_color_is_tagged(tmp_path):
    import subprocess
    import wave

    source, poses = make_fixture(tmp_path)
    audio = tmp_path / "short.wav"
    samples = (np.sin(np.arange(800) / 8000 * 2 * np.pi * 440) * 4000).astype("<i2")
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(samples.tobytes())
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                str(source),
                "--out",
                str(out),
                "--poses",
                str(poses),
                "--size",
                "90x160",
                "--fps",
                "10",
                "--hold",
                "2",
                "--audio",
                str(audio),
                "--audio-start",
                ".02",
            ]
        )
        == 0
    )
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(out / "cut.mp4"),
            ]
        )
    )
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert video["pix_fmt"] == "yuv420p"
    assert (
        video["color_space"]
        == video["color_transfer"]
        == video["color_primaries"]
        == "bt709"
    )
    assert video["color_range"] == "tv"
    assert abs(float(info["format"]["duration"]) - 0.6) < 0.06
    assert float(audio_stream["duration"]) >= 0.55
