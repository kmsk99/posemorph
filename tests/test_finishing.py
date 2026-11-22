import json
import shutil
import wave

import cv2
import numpy as np
import pytest
from test_pipeline import make_fixture

from posemorph.cli import main
from posemorph.finishing import (
    ToneNormalizer,
    audio_onsets,
    camera_frame,
    cut_timing,
    parse_curve,
    parse_zoom,
    read_beats,
)
from posemorph.pipeline import load


def test_curve_is_time_based_and_each_image_appears_once():
    curve = "0:0.25,0.6:0.2,0.9:0.1,1:0.1"
    holds, info = cut_timing(176, 30, 6, curve, duration=35)
    assert len(holds) == 176 and min(holds) >= 1 and sum(holds) == 1050
    assert np.mean(holds[:30]) > np.mean(holds[-30:]) * 1.5
    assert info["cut_times"][-1] == 35
    assert cut_timing(4, 30, 6)[0] == [6] * 4
    for invalid in ["0:.2", "0:.2,1:-.1", "0:.2,.5:.1,.4:.2,1:.1", "0:nan,1:.1"]:
        with pytest.raises(ValueError):
            parse_curve(invalid)
    with pytest.raises(ValueError, match="at least one frame"):
        cut_timing(50, 30, 6, duration=1)


def test_snap_beats_monotonic_and_audio_offset(tmp_path):
    path = tmp_path / "beats.json"
    path.write_text(json.dumps({"beats": [1, 2.21, 2.39, 2.61, 2.79]}))
    beats = read_beats(path, audio_start=2)
    holds, info = cut_timing(5, 100, 20, beats=beats, tolerance=0.025)
    assert sum(holds) == 100 and min(holds) > 0 and len(holds) == 5
    np.testing.assert_allclose(info["cut_times"][1:-1], [0.21, 0.39, 0.61, 0.79])
    assert info["snapped_cuts"] == 4
    holds, _ = cut_timing(40, 30, 1, beats=np.arange(0, 2, 0.005), tolerance=0.1)
    assert len(holds) == 40 and min(holds) == 1
    path.write_text('{"beats": ["bad"]}')
    with pytest.raises(ValueError, match="--beats"):
        read_beats(path)


def test_camera_uses_one_global_torso_center():
    image = np.zeros((80, 80, 3), np.uint8)
    image[40, 40] = 255
    image[40, 50] = 255
    assert camera_frame(image, 0, (1, 2), (40, 40)) is image
    frame = camera_frame(image, 1, (1, 2), (40, 40))
    assert frame[40, 40].max() == 255
    assert frame[40, 60].max() == 255
    assert frame[40, 50].max() == 0
    for invalid in [".9:1", "1:inf", "1:1:2"]:
        with pytest.raises(ValueError):
            parse_zoom(invalid)


def test_tone_reduces_flicker_preserves_colour_and_is_optional():
    yy, xx = np.indices((64, 64))
    image = np.stack([60 + xx, 90 + yy, 120 + xx], axis=-1).astype(np.uint8)
    variants = [
        np.clip(image.astype(float) * f, 0, 255).astype(np.uint8)
        for f in [1, 0.55, 1.45, 0.7, 1.3]
    ]
    tone = ToneNormalizer()
    for image in variants:
        result = tone.apply(image, True)
        a = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        b = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(float)
        assert np.mean(abs(a[:, :, 1:] - b[:, :, 1:])) < 3
    info = tone.report()
    assert (
        info["after"]["mean_luminance_jump"]
        < info["before"]["mean_luminance_jump"] * 0.8
    )
    assert ToneNormalizer().apply(variants[0], False) is variants[0]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_synthetic_onsets_and_finished_video(tmp_path):
    audio = tmp_path / "pulses.wav"
    sr = 11025
    samples = np.zeros(sr * 2, np.float32)
    rng = np.random.default_rng(1)
    for t in [0.25, 0.75, 1.25, 1.75]:
        start = round(sr * t)
        samples[start : start + 512] = rng.normal(0, 0.25, 512) * np.hanning(512)
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes((samples * 32767).astype("<i2").tobytes())
    onsets = audio_onsets(audio, 0, 2)
    for t in [0.25, 0.75, 1.25, 1.75]:
        assert min(abs(np.array(onsets) - t)) < 0.08
    assert len(audio_onsets(audio, 0.5, 1)) == 2
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
                "--mirror",
                "auto",
                "--order",
                "phase",
                "--hook",
                "2",
                "--audio",
                str(audio),
                "--speed-curve",
                "0:.5,1:.2",
                "--duration",
                "2",
                "--zoom",
                "1:1.12",
                "--normalize-tone",
            ]
        )
        == 0
    )
    result = load(out, "render")
    assert result["frames"] == 60 and sum(result["timing"]["holds"]) == 60
    assert result["timing"]["beat_source"] == "spectral_flux_onsets"
    assert result["zoom"] == [1, 1.12] and result["normalize_tone"]
    cap = cv2.VideoCapture(str(out / "cut.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
    assert cap.read()[0]
    cap.release()
    assert (
        main(
            [
                "render",
                "--out",
                str(out),
                "--mode",
                "morph",
                "--speed-curve",
                "0:.2,1:.1",
            ]
        )
        == 1
    )
