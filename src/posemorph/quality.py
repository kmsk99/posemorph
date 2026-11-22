"""Quality gates and comparable alignment diagnostics (no learned dependencies)."""

import numpy as np

from .finishing import parse_curve
from .geometry import similarity, target_torso, transform_points


def pose_confidence(record):
    scores = np.asarray(record["scores"])
    return float(min(scores[:5].max(), scores[[5, 6, 9, 10, 11, 12]].min()))


def person_area(record):
    x1, y1, x2, y2 = record["box"]
    w, h = record["size"]
    return float(
        max(0, min(x2, w) - max(x1, 0)) * max(0, min(y2, h) - max(y1, 0)) / (w * h)
    )


def automatic_count(duration, curve=None, fps=30, hold=6):
    if duration is None:
        return None
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("--duration must be finite and positive")
    if curve:
        points = parse_curve(curve)
        fractions = np.linspace(0, 1, 10001)
        density = 1 / np.interp(fractions, points[:, 0], points[:, 1])
        count = duration * np.trapezoid(density, fractions)
    else:
        count = duration * fps / hold
    return max(1, min(round(duration * fps), round(float(count))))


def head_center(points, scores, fallback):
    indices = np.array([0, 1, 2])
    weights = np.asarray(scores)[indices] * [2, 1, 1]
    weights *= np.asarray(scores)[indices] >= 0.25
    return (
        np.average(points[indices], axis=0, weights=weights)
        if weights.sum()
        else points[fallback]
    )


def alignment_geometry(points, scores, head_index, size, method="similarity"):
    points, scores = np.asarray(points, float), np.asarray(scores)
    target = target_torso(size)
    anchors = [head_index, 5, 6, 11, 12]
    matrix = similarity(points[anchors], target, scores[anchors])
    initial = transform_points(points, matrix)
    pre_head_error = float(np.linalg.norm(initial[head_index] - target[0]))
    if method == "head-torso":
        shoulder, hip = points[[5, 6]].mean(0), points[[11, 12]].mean(0)
        down = hip - shoulder
        length = np.linalg.norm(down)
        if length <= 1e-6:
            raise ValueError("Degenerate torso for head-torso alignment")
        angle = np.pi / 2 - np.arctan2(down[1], down[0])
        scale = np.linalg.norm(target[3:].mean(0) - target[1:3].mean(0)) / length
        linear = scale * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        center = head_center(points, scores, head_index)
        matrix = np.column_stack([linear, target[0] - linear @ center])
    transformed = transform_points(points, matrix)
    return (
        matrix,
        transformed,
        {
            "head_error_px": float(np.linalg.norm(transformed[head_index] - target[0])),
            "pre_head_error_px": pre_head_error,
            "head_position": transformed[head_index].tolist(),
            "head_center": transform_points(
                [head_center(points, scores, head_index)], matrix
            )[0].tolist(),
            "torso_scale_px": float(
                np.linalg.norm(
                    transformed[[11, 12]].mean(0) - transformed[[5, 6]].mean(0)
                )
            ),
            "alignment_error_px": float(
                np.sqrt(np.mean(np.sum((transformed[anchors] - target) ** 2, axis=1)))
            ),
            "scale": float(np.linalg.norm(matrix[:, 0])),
        },
    )


def alignment_summary(records, size, common_size=None):
    """Cut-boundary jumps, not per-frame zeros inside held images.

    Reference all comparisons to a common output resolution when requested.
    """
    common_size = size if common_size is None else common_size
    scale = np.array(common_size) / np.array(size)
    heads, torsos, landmarks = [], [], []
    for r in records:
        p = np.asarray(r["keypoints"])
        landmark = p[r["head_index"]] * scale
        center = (
            np.asarray(
                r.get("head_center", head_center(p, r["scores"], r["head_index"]))
            )
            * scale
        )
        heads.append(center)
        landmarks.append(landmark)
        torso = (p[[11, 12]].mean(0) - p[[5, 6]].mean(0)) * scale
        torsos.append(np.linalg.norm(torso))
    heads, torsos, landmarks = np.array(heads), np.array(torsos), np.array(landmarks)

    def stats(values):
        return {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
        }

    return {
        "coordinate_size": list(common_size),
        "scope": "between consecutive shown images at cut boundaries, before global zoom",
        "head_jitter_px": stats(np.linalg.norm(np.diff(heads, axis=0), axis=1)),
        "selected_face_landmark_jitter_px": stats(
            np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
        ),
        "head_position_variance_px2": np.var(heads, axis=0).tolist(),
        "torso_scale_jitter_px": stats(np.abs(np.diff(torsos))),
        "torso_scale_variance_px2": float(np.var(torsos)),
    }


def rendered_alignment_summary(records, render):
    """Propagate detected geometry through holds/zoom; no pixel-based tracking."""
    if render["mode"] != "cut":
        return None
    holds = render.get("timing", {}).get("holds", [render["hold"]] * len(records))
    points, lengths = [], []
    for record in records:
        p = np.asarray(record["keypoints"])
        points.append(
            record.get(
                "head_center", head_center(p, record["scores"], record["head_index"])
            )
        )
        lengths.append(np.linalg.norm(p[[11, 12]].mean(0) - p[[5, 6]].mean(0)))
    points = np.repeat(np.asarray(points), holds, axis=0)
    lengths = np.repeat(np.asarray(lengths), holds)
    zoom = render.get("zoom", [1, 1])
    factors = np.linspace(*zoom, len(points))
    center = np.asarray(render.get("zoom_center", [0, 0]))
    points = center + factors[:, None] * (points - center)
    lengths *= factors
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    torso_jumps = np.abs(np.diff(lengths))
    cuts = np.cumsum(holds)[:-1] - 1

    def stats(values):
        return {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
        }

    return {
        "scope": "predicted landmarks after holds and global zoom; not independent image tracking",
        "all_frame_head_jitter_px": stats(jumps),
        "cut_boundary_head_jitter_px": stats(jumps[cuts]),
        "all_frame_torso_scale_jitter_px": stats(torso_jumps),
        "cut_boundary_torso_scale_jitter_px": stats(torso_jumps[cuts]),
    }
