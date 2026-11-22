import json

import pytest
from test_pipeline import make_fixture

from posemorph import pipeline
from posemorph.evaluation import detection_summary


def test_copying_a_manifest_cannot_override_its_new_stage_identity(tmp_path):
    first = pipeline.save(tmp_path, "render-cut", {"frames": 678})
    copied = pipeline.save(tmp_path, "render-cut-020", first)
    assert copied["stage"] == "render-cut-020"
    assert pipeline.load(tmp_path, "render-cut-020")["frames"] == 678
    malformed = pipeline.save(
        tmp_path, "check", {"schema": -1, "stage": "old", "created": "old"}
    )
    assert malformed["schema"] == pipeline.SCHEMA
    assert malformed["created"] != "old"


def test_detection_summary_does_not_call_fallback_a_detection():
    r = {"people": [{}], "scores": [0.8] * 17, "box_source": "detector"}
    fallback = {**r, "scores": [0.1] * 17, "box_source": "full_image_fallback"}
    summary = detection_summary(
        {
            "records": [r, fallback, {"error": "bad image", "people": []}],
            "seconds": 1,
            "complete": True,
        }
    )
    assert summary["actual_box_count"] == 1 and summary["fallback_count"] == 1
    assert summary["pose_record_count"] == 2 and summary["error_count"] == 1
    assert summary["required_confidence_ge_035_count"] == 1
    assert summary["confidence"]["required_min"]["median"] == pytest.approx(0.45)


def test_static_random_audit_is_repeatable_and_does_not_modify_inputs(tmp_path):
    source, poses = make_fixture(tmp_path)
    out = tmp_path / "out"
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    pipeline.detect(source, out, poses)
    result = pipeline.evaluate(out, [out], sample_size=3, seed=7)
    files = [x["file"] for x in result["random_audit"]["samples"]]
    assert len(set(files)) == 3
    again = pipeline.evaluate(out, [out], sample_size=3, seed=7)
    assert files == [x["file"] for x in again["random_audit"]["samples"]]
    assert (out / "random-30-contact-sheet.jpg").exists()
    assert all((out / p).exists() for p in again["random_audit"]["pages"])
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    notes = out / "qa" / "visual-review.json"
    notes.write_text(json.dumps({"detect_sha256": "different"}))
    stale = pipeline.evaluate(out, [out], sample_size=3, seed=7)
    assert stale["visual_review_status"] == "stale_notes_not_used"
    notes.write_text(
        json.dumps(
            {
                "detect_sha256": pipeline.digest(out / "detect.json"),
                "reference_detect_sha256": stale["reference_detect_sha256"],
                "sample_files": files,
                "summary": {"synthetic": True},
            }
        )
    )
    fresh = pipeline.evaluate(out, [out], sample_size=3, seed=7)
    assert fresh["visual_review_status"] == "notes_match_current_detection_and_sample"
    custom = pipeline.evaluate(
        out, sample_size=1, target_duration=1.1, seconds_per_cut=0.2
    )
    assert custom["required_cut_count"] == 6
    with pytest.raises(ValueError, match="finite and positive"):
        pipeline.evaluate(out, target_duration=0)
