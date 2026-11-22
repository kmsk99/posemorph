import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import make_fixture

from posemorph import pipeline
from posemorph.cli import main
from posemorph.quality import (
    alignment_geometry,
    alignment_summary,
    automatic_count,
    head_center,
    rendered_alignment_summary,
)
from posemorph.relevance import contrast_score, read_prompts


def test_prompt_validation_and_group_contrast(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"positive":["an angel on a demon"],"negative":["a portrait"]}')
    assert len(read_prompts(prompts)["positive"]) == 1
    assert contrast_score([3, 1], 1) == pytest.approx(0.880797)
    assert contrast_score([3, 3, 1], 2) == contrast_score([3, 1], 1)
    prompts.write_text('{"positive":[],"negative":["portrait"]}')
    with pytest.raises(ValueError, match="nonempty"):
        read_prompts(prompts)


def test_optional_worker_contract_without_torch_or_network(tmp_path, monkeypatch):
    source, poses = make_fixture(tmp_path)
    out = tmp_path / "out"
    pipeline.detect(source, out, poses)
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"positive":["a scene"],"negative":["text"]}')

    def fake_run(command, check):
        request = json.loads(Path(command[-2]).read_text())
        response = {
            "records": [
                {**r, "score": 0.4 if r["view"] == "aligned" else 0.9}
                for r in request["records"]
            ],
            "model": "synthetic",
        }
        Path(command[-1]).write_text(json.dumps(response))
        assert command[0] == "custom-python"

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = pipeline.relevance(out, prompts, "custom-python")
    assert (
        len(result["records"]) == 4
        and pipeline.load(out, "relevance")["model"] == "synthetic"
    )
    assert result["records"][0]["score"] == 0.4
    assert len(result["records"][0]["views"]) == 2


def test_quality_gates_ranking_auto_count_and_staleness(tmp_path):
    source, poses = make_fixture(tmp_path)
    out = tmp_path / "out"
    pipeline.detect(source, out, poses)
    detected = pipeline.load(out, "detect")
    rows = [
        {"file": r["file"], "stamp": r["stamp"], "score": score}
        for r, score in zip(detected["records"], [0.9, 0.2, 0.95, 0.9])
    ]
    pipeline.save(
        out,
        "relevance",
        {"inputs": pipeline.inputs(out, "detect"), "records": rows, "seconds": 0},
    )
    selected = pipeline.select(
        out,
        min_relevance=0.5,
        min_pose_conf=0.4,
        duration=0.4,
        fps=10,
        hold=2,
        size=(90, 160),
        alignment="head-torso",
    )
    assert [r["file"] for r in selected["records"]] == ["0.jpg", "2.jpg"]
    assert Counter(r["reason"] for r in selected["dropped"]) == {
        "low_relevance": 1,
        "low_hip_confidence": 1,
    }
    assert (
        selected["max_images"] == 2
        and selected["max_images_source"] == "duration_and_curve"
    )
    capped = pipeline.select(
        out, min_relevance=0.5, max_images=1, size=(90, 160), alignment="head-torso"
    )
    candidates = selected["records"]
    assert (
        capped["records"][0]["file"]
        == max(candidates, key=lambda r: r["quality"]["rank_score"])["file"]
    )
    assert sum(r["reason"] == "max_images" for r in capped["dropped"]) == 1
    with pytest.raises(ValueError, match="settings differ"):
        pipeline.align(out, (180, 320))
    pipeline.save(out, "relevance", {"records": rows, "seconds": 0})
    with pytest.raises(ValueError, match="stale"):
        pipeline.load(out, "select")
    assert automatic_count(35.2, "0:.25,.6:.2,.9:.1,1:.1") > 150
    assert automatic_count(None) is None


def test_each_pose_box_head_and_upscale_gate(tmp_path):
    source, poses = make_fixture(tmp_path)
    out = tmp_path / "out"
    pipeline.detect(source, out, poses)
    data = pipeline.load(out, "detect")
    # Independent gates use a permissive base threshold so the named gate fires.
    data["records"][1]["scores"][9] = 0.1
    data["records"][2]["box"] = [40, 30, 45, 35]
    pipeline.save(out, "detect", data)
    selected = pipeline.select(
        out, min_pose_conf=0.35, min_person_area=0.1, size=(90, 160)
    )
    assert Counter(r["reason"] for r in selected["dropped"]) == {
        "low_pose_confidence": 1,
        "small_person_box": 1,
        "low_hip_confidence": 1,
    }
    for threshold, reason in [
        ("max_head_error", "head_alignment_error"),
        ("max_upscale", "excessive_upscale"),
    ]:
        data = pipeline.load(out, "detect")
        # A perfect target-sized pose survives a zero-error/zero-upscale gate.
        from posemorph.geometry import target_torso

        p = np.asarray(data["records"][0]["keypoints"], float)
        p[[0, 5, 6, 11, 12]] = target_torso((90, 160))
        data["records"][0]["keypoints"] = p.tolist()
        other = np.asarray(data["records"][1]["keypoints"], float)
        data["records"][1]["keypoints"] = (
            np.array([50, 50]) + 0.7 * (other - [50, 50])
        ).tolist()
        pipeline.save(out, "detect", data)
        result = pipeline.select(out, size=(90, 160), **{threshold: 0.001})
        assert any(r["reason"] == reason for r in result["dropped"])


def test_head_torso_alignment_reduces_real_geometric_variance(tmp_path):
    _, poses = make_fixture(tmp_path)
    record = json.loads(poses.read_text())["0.jpg"][0]
    before, after = [], []
    for stretch in [0.7, 1.0, 1.5]:
        p = np.asarray(record["keypoints"], float)
        p[11:13, 1] *= stretch
        for method, dest in [("similarity", before), ("head-torso", after)]:
            _, transformed, geometry = alignment_geometry(
                p, record["scores"], 0, (90, 160), method
            )
            dest.append(
                dict(
                    keypoints=transformed.tolist(),
                    scores=record["scores"],
                    head_index=0,
                    **geometry,
                )
            )
    old = alignment_summary(before, (90, 160))
    new = alignment_summary(after, (90, 160))
    assert new["head_jitter_px"]["mean"] < 1e-10
    assert new["torso_scale_variance_px2"] < 1e-10
    assert old["head_jitter_px"]["mean"] > 1
    assert old["torso_scale_variance_px2"] > 1
    np.testing.assert_allclose(
        head_center(np.asarray(after[0]["keypoints"]), record["scores"], 0), [45, 40]
    )
    assert (
        main(["select", "--out", str(tmp_path / "missing"), "--min-pose-conf", "nan"])
        == 1
    )


def test_rendered_jitter_includes_global_zoom_and_cut_boundaries(tmp_path):
    _, poses = make_fixture(tmp_path)
    person = json.loads(poses.read_text())["0.jpg"][0]
    records = [
        dict(person, head_index=0, head_center=[10, 0]),
        dict(person, head_index=0, head_center=[20, 0]),
    ]
    result = rendered_alignment_summary(
        records,
        {
            "mode": "cut",
            "hold": 2,
            "timing": {"holds": [2, 1]},
            "zoom": [1, 2],
            "zoom_center": [0, 0],
        },
    )
    # Rendered centers are x=10, 15, 40: jumps 5 then 25, cut jump 25.
    assert result["all_frame_head_jitter_px"]["mean"] == 15
    assert result["all_frame_head_jitter_px"]["p90"] == 23
    assert result["cut_boundary_head_jitter_px"]["mean"] == 25
