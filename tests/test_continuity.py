import json

import cv2
import numpy as np
import pytest
from test_pipeline import make_fixture

from posemorph import pipeline
from posemorph.cli import main
from posemorph.continuity import (
    SWAP,
    action_phase,
    continuity_metrics,
    material_features,
    material_labels,
    mirror_plan,
    phase_path,
    reflect_pose,
)
from posemorph.geometry import distance_matrix, pose_distance, transform_points


@pytest.fixture
def record(tmp_path):
    _, poses = make_fixture(tmp_path)
    return json.loads(poses.read_text())["0.jpg"][0]


def test_reflection_swaps_labels_and_reduces_mirror_distance(record):
    p = np.asarray(record["keypoints"], float)
    p[9] = [90, 5]
    record = dict(record, keypoints=p.tolist())
    reflected, scores = reflect_pose(p, record["scores"], 100)
    other = dict(record, keypoints=reflected.tolist(), scores=scores.tolist())
    majority, plan = mirror_plan([record, record, other], "auto")
    assert majority == 1
    assert [c["flipped"] for c in plan] == [False, False, True]
    restored, restored_scores = reflect_pose(reflected, scores, 100)
    assert pose_distance(record, other) > 0.2
    assert (
        pose_distance(record, {"keypoints": restored, "scores": restored_scores}) < 1e-8
    )
    np.testing.assert_allclose(restored, p)
    assert not any(c["flipped"] for c in mirror_plan([record, other], "off")[1])


def arm_arc(record, count=36):
    records = []
    for angle in np.linspace(-1.35, 4.4, count):
        p = np.asarray(record["keypoints"], float)
        direction = np.array([np.cos(angle), -np.sin(angle)])
        p[7] = p[5] + 20 * direction
        p[9] = p[5] + 40 * direction
        records.append(dict(record, keypoints=p.tolist()))
    return records


def test_phase_trajectory_cycles_hook_and_metrics(record):
    records = arm_arc(record)
    phases = [action_phase(r)["phase"] for r in records]
    assert np.all(np.diff(phases) > 0)
    shuffled = [records[i] for i in np.random.default_rng(7).permutation(len(records))]
    distances = distance_matrix(shuffled)
    path, details = phase_path(shuffled, distances, hook=4)
    assert sorted(path) == list(range(len(records)))
    assert np.all(np.diff([details[i]["bin"] for i in path]) >= 0)
    first_bin = [
        i for i, d in enumerate(details) if d["bin"] == min(x["bin"] for x in details)
    ]
    local = distances[np.ix_(first_bin, first_bin)]
    center = first_bin[int(np.argmin(local.mean(1)))]
    assert distances[center, path[0]] <= np.quantile(distances[center, first_bin], 0.25)
    assert (
        continuity_metrics(path, distances)["mean"]
        < continuity_metrics(list(range(len(records))), distances)["mean"]
    )
    repeated, cycle_details = phase_path(shuffled, distances, cycles=3)
    assert sorted(repeated) == list(range(len(records)))
    bins = np.array([details[i]["bin"] for i in repeated])
    assert (np.diff(bins) < 0).sum() == 2
    assert {d["cycle"] for d in cycle_details} == {0, 1, 2}
    with pytest.raises(ValueError, match="first phase"):
        phase_path(records, distance_matrix(records), start=len(records) - 1)


def test_material_runs_keep_phase_order(record):
    records = arm_arc(record)
    images = [
        np.full((60, 40, 3), (20, 40, 150), np.uint8),
        np.full((60, 40, 3), (150, 130, 20), np.uint8),
    ]
    features = [material_features(images[i % 2]) for i in range(len(records))]
    labels = material_labels(features)
    assert labels[0] != labels[1]
    path, details = phase_path(records, distance_matrix(records), groups=labels)
    ordered_labels = labels[path]
    assert np.count_nonzero(np.diff(ordered_labels)) == 1
    for group in set(labels):
        assert np.all(
            np.diff([details[i]["bin"] for i in path if labels[i] == group]) >= 0
        )


def test_multiple_roots_mirror_manifest_and_baseline(tmp_path):
    source, poses = make_fixture(tmp_path)
    # Reuse a valid single-root baseline as the old run.
    old = tmp_path / "old"
    assert main(["detect", str(source), "--poses", str(poses), "--out", str(old)]) == 0
    assert main(["select", "--out", str(old)]) == 0
    assert main(["align", "--out", str(old), "--size", "90x160"]) == 0
    assert main(["order", "--out", str(old)]) == 0
    second = tmp_path / "second"
    second.mkdir()
    cv2.imwrite(str(second / "0.jpg"), cv2.imread(str(source / "0.jpg")))
    pose_data = json.loads(poses.read_text())
    data = {f"01/{name}": people for name, people in pose_data.items()}
    p = np.asarray(pose_data["0.jpg"][0]["keypoints"], float)
    p[10, 1] = 0
    data["02/0.jpg"] = [dict(pose_data["0.jpg"][0], keypoints=p.tolist())]
    poses.write_text(json.dumps(data))
    out = tmp_path / "new"
    assert (
        main(
            [
                "detect",
                str(source),
                str(second),
                "--poses",
                str(poses),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert main(["select", "--out", str(out)]) == 0
    assert (
        main(["align", "--out", str(out), "--size", "90x160", "--mirror", "auto"]) == 0
    )
    aligned = pipeline.load(out, "align")
    assert aligned["flip_count"] >= 1
    original = {r["file"]: r for r in pipeline.load(out, "select")["records"]}
    for r in aligned["records"]:
        points = np.asarray(original[r["file"]]["keypoints"])
        if r["flipped"]:
            points = points[SWAP]
        np.testing.assert_allclose(
            transform_points(points, r["matrix"]), r["keypoints"], atol=1e-6
        )
    assert main(["order", "--out", str(out), "--order", "phase", "--hook", "3"]) == 0
    assert (
        main(["report", "--out", str(out), "--baseline-order", str(old / "order.json")])
        == 0
    )
    report = pipeline.load(out, "report")
    assert report["input_count"] == 5
    assert report["continuity"]["v0_comparison"]["shared_images"] == 3
    assert report["alignment_consistency"]["coordinate_size"] == [90, 160]
    baseline = report["continuity"]["baselines"]["old"]
    assert baseline["full_before_alignment"]["coordinate_size"] == [90, 160]
    assert baseline["full_before_continuity"]["transitions"] == 2
    assert len(pipeline.load(out, "order")["poses"]) == 4
    # Re-alignment changes reflected poses and must invalidate the order too.
    pipeline.align(out, (90, 160), "off")
    with pytest.raises(ValueError, match="stale"):
        pipeline.load(out, "order")
