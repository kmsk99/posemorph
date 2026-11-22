"""Static-pose heuristics, with uncertainty retained for human review."""

from itertools import pairwise

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .geometry import order_path

# Reflection changes anatomical labels as well as coordinates.
SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def handedness(record):
    p, s = np.asarray(record["keypoints"], float), np.asarray(record["scores"])
    shoulder, hip = p[[5, 6]].mean(0), p[[11, 12]].mean(0)
    down = hip - shoulder
    scale = max(np.linalg.norm(down), 1e-6)
    down /= scale
    if min(s[9], s[10]) >= 0.2:
        difference = float((p[10] - p[9]) @ down / scale)
        if abs(difference) >= 0.15:
            return {
                "sign": 1 if difference > 0 else -1,
                "confidence": float(min(s[9], s[10]) * min(abs(difference), 1)),
                "cue": "raised_wrist",
            }
    # Weak profile cue, relative to the shoulder axis rather than screen x.
    across = p[5] - p[6]
    across /= max(np.linalg.norm(across), 1e-6)
    offset = float((p[0] - shoulder) @ across / scale)
    if s[0] >= 0.4 and abs(offset) > 0.2:
        return {
            "sign": 1 if offset > 0 else -1,
            "confidence": float(0.25 * s[0] * min(abs(offset), 1)),
            "cue": "torso_profile",
        }
    return {"sign": 0, "confidence": 0.0, "cue": "ambiguous"}


def mirror_plan(records, mode):
    cues = [handedness(r) for r in records]
    vote = sum(c["sign"] * c["confidence"] for c in cues)
    majority = 1 if vote >= 0 else -1
    return majority, [
        dict(c, flipped=bool(mode == "auto" and c["sign"] == -majority)) for c in cues
    ]


def reflect_pose(points, scores, width):
    p = np.asarray(points, float)[SWAP].copy()
    p[:, 0] = width - 1 - p[:, 0]
    return p, np.asarray(scores)[SWAP]


def action_phase(record, wrist=9):
    """Unwrap an arm arc: low/outside -> overhead -> across/down.

    A still image cannot identify velocity or the true weapon hand. This is an
    explicit geometric hypothesis, not an inferred temporal action label.
    """
    p, s = np.asarray(record["keypoints"], float), np.asarray(record["scores"])
    if s[wrist] < 0.2:
        wrist = 19 - wrist
    shoulder_index, elbow = (5, 7) if wrist == 9 else (6, 8)
    center = p[[5, 6]].mean(0)
    down = p[[11, 12]].mean(0) - center
    scale = max(np.linalg.norm(down), 1e-6)
    down /= scale
    outside = p[shoulder_index] - center
    outside -= down * (outside @ down)
    if np.linalg.norm(outside) < 1e-6:
        outside = np.array([down[1], -down[0]]) * (1 if wrist == 9 else -1)
    outside /= max(np.linalg.norm(outside), 1e-6)
    origin = p[shoulder_index]
    vectors = [p[wrist] - origin, p[elbow] - origin, p[wrist] - p[elbow]]
    weights = np.array([0.7, 0.2, 0.1]) * [s[wrist], s[elbow], min(s[wrist], s[elbow])]
    angles = np.array([np.arctan2(-v @ down, v @ outside) for v in vectors])
    vector = np.sum(weights * np.exp(1j * angles))
    angle = float(np.angle(vector))
    if angle < -np.pi / 2:
        angle += 2 * np.pi
    phase = float(np.clip((angle + np.pi / 2) / (2 * np.pi), 0, 1))
    height = float((origin - p[wrist]) @ down / scale)
    # Height helps distinguish raised arms with similar bent-elbow angles.
    phase = float(np.clip(phase + 0.025 * np.tanh(height), 0, 1))
    return {
        "phase": phase,
        "wrist": wrist,
        "wrist_height": height,
        "upper_arm_angle": float(angles[1]),
        "forearm_angle": float(angles[2]),
        "confidence": float(min(s[shoulder_index], s[wrist], s[elbow])),
        "method": "static_arm_arc",
    }


def material_features(image):
    small = cv2.resize(image, (96, 96))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(float)
    edge = cv2.Canny(small, 80, 160).mean() / 255
    return np.r_[lab.mean((0, 1)) / 255, lab.std((0, 1)) / 128, edge]


def material_labels(features):
    if len(features) < 3:
        return np.zeros(len(features), int)
    return (
        fcluster(
            linkage(np.asarray(features), method="ward"),
            min(3, len(features)),
            criterion="maxclust",
        )
        - 1
    )


def phase_path(
    records, distances, wrist=9, cycles=1, hook=0, start=None, groups=None, phase_bins=3
):
    if cycles > len(records):
        raise ValueError("--cycles cannot exceed selected image count")
    details = [action_phase(r, wrist) for r in records]
    bins = np.minimum(
        (np.array([d["phase"] for d in details]) * phase_bins).astype(int),
        phase_bins - 1,
    )
    groups = np.zeros(len(records), int) if groups is None else np.asarray(groups)
    path = []
    # Distribute each phase across cycles, never repeat an image.
    for group in sorted(set(groups)):
        buckets = [[[] for _ in range(phase_bins)] for _ in range(cycles)]
        offset = 0
        for phase_bin in range(phase_bins):
            members = [
                i
                for i in range(len(records))
                if groups[i] == group and bins[i] == phase_bin
            ]
            members.sort(key=lambda i: (details[i]["phase"], i))
            for j, i in enumerate(members):
                buckets[(j + offset) % cycles][phase_bin].append(i)
            offset += len(members)
        for cycle_index, cycle in enumerate(buckets):
            for members in cycle:
                if not members:
                    continue
                if not path:
                    if start is not None and start not in members:
                        raise ValueError(
                            "--start must belong to the first phase bin/material run"
                        )
                    local_start = members.index(start) if start is not None else None
                else:
                    local_start = int(np.argmin(distances[path[-1], members]))
                local = order_path(
                    distances[np.ix_(members, members)],
                    local_start,
                    hook if not path else 0,
                )
                path.extend(members[i] for i in local)
                for i in members:
                    details[i]["cycle"] = cycle_index
    for i, d in enumerate(details):
        d["bin"] = int(bins[i])
        d["material_group"] = int(groups[i])
    # Optimize the two boundaries of each phase block together. Independent
    # bin searches can otherwise end at a pose far from the next bin's start.
    labels = [(d["material_group"], d["cycle"], d["bin"]) for d in details]
    fixed = 1 if start is not None or hook else 0
    for _ in range(30):
        improved = False
        for i in range(fixed, len(path) - 1):
            for j in range(i + 1, len(path)):
                if labels[path[j]] != labels[path[i]]:
                    break
                old = new = 0.0
                if i:
                    old += distances[path[i - 1], path[i]]
                    new += distances[path[i - 1], path[j]]
                if j + 1 < len(path):
                    old += distances[path[j], path[j + 1]]
                    new += distances[path[i], path[j + 1]]
                if new < old - 1e-9:
                    path[i : j + 1] = reversed(path[i : j + 1])
                    improved = True
        if not improved:
            break
    return path, details


def continuity_metrics(path, distances):
    values = np.array([distances[a, b] for a, b in pairwise(path)])
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "transitions": int(values.size),
    }
