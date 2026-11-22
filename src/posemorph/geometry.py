"""Similarity alignment, rotation-invariant pose distance, and landmark warps."""

from itertools import pairwise

import cv2
import numpy as np
from scipy.spatial import Delaunay

TORSO = np.array([0, 5, 6, 11, 12])
WEIGHTS = np.array(
    [1, 0.15, 0.15, 0.15, 0.15, 3, 3, 2, 2, 1.5, 1.5, 3, 3, 0.6, 0.6, 0.4, 0.4]
)
EDGES = [
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def transform_points(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def similarity(source, target, weights=None):
    """Weighted least-squares proper similarity (never reflects anatomy)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    w = np.ones(len(source)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    a, b = (source * w[:, None]).sum(0), (target * w[:, None]).sum(0)
    x, y = source - a, target - b
    covariance = (x * w[:, None]).T @ y
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    denominator = np.sum(w[:, None] * x * x)
    if denominator < 1e-10:
        raise ValueError("Degenerate pose: landmarks have no extent")
    scale = np.sum(w[:, None] * (x @ rotation) * y) / denominator
    if scale <= 1e-8:
        raise ValueError("Degenerate similarity scale")
    linear = scale * rotation.T
    return np.column_stack([linear, b - linear @ a])


def target_torso(size):
    w, h = size
    return np.array(
        [
            [0.5 * w, 0.25 * h],
            [0.61 * w, 0.34 * h],
            [0.39 * w, 0.34 * h],
            [0.57 * w, 0.55 * h],
            [0.43 * w, 0.55 * h],
        ]
    )


def normalize_pose(points, scores):
    """Torso-centred unit pose; rotation is removed pairwise by Procrustes."""
    points = np.asarray(points, float)
    w = WEIGHTS * np.asarray(scores)
    center = np.average(points[TORSO], axis=0, weights=w[TORSO])
    x = points - center
    scale = np.sqrt(np.average(np.sum(x[TORSO] ** 2, axis=1), weights=w[TORSO]))
    return x / max(scale, 1e-6)


def pose_distance(a, b):
    x, y = np.asarray(a["keypoints"]), np.asarray(b["keypoints"])
    sa, sb = np.asarray(a["scores"]), np.asarray(b["scores"])
    w = WEIGHTS * np.minimum(sa, sb) * ((sa >= 0.2) & (sb >= 0.2))
    x, y = normalize_pose(x, sa), normalize_pose(y, sb)
    u, _, vt = np.linalg.svd((x * w[:, None]).T @ y)
    r = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.sum(w[:, None] * (x @ r - y) ** 2) / max(w.sum(), 1e-9)))


def distance_matrix(records):
    d = np.zeros((len(records), len(records)))
    for i in range(len(records)):
        for j in range(i):
            d[i, j] = d[j, i] = pose_distance(records[i], records[j])
    return d


def path_length(path, distances):
    return float(sum(distances[a, b] for a, b in pairwise(path)))


def order_path(distances, start=None, hook=0):
    n = len(distances)
    if n == 0:
        return []
    prefix = [start] if start is not None else []
    if hook:
        center = int(np.argmin(distances.mean(axis=1))) if start is None else start
        nearby = np.flatnonzero(
            distances[center] <= np.quantile(distances[center], 0.25)
        )
        # Reserve at most N compatible hooks, connected by nearest neighbours.
        ranked = sorted(nearby, key=lambda i: (-distances[i].mean(), i))
        reserved = set(map(int, ranked[:hook])) - set(prefix)
        if not prefix and reserved:
            first = int(ranked[0])
            prefix.append(first)
            reserved.remove(first)
        while reserved:
            nxt = min(reserved, key=lambda i: (distances[prefix[-1], i], i))
            prefix.append(nxt)
            reserved.remove(nxt)
    path = prefix or [int(np.argmin(distances.mean(axis=1)))]
    remaining = set(range(n)) - set(path)
    while remaining:
        nxt = min(remaining, key=lambda j: (distances[path[-1], j], j))
        path.append(nxt)
        remaining.remove(nxt)
    # Open-path 2-opt; preserve a requested start and the hook prefix.
    fixed = max(1, len(prefix))
    for _ in range(20):
        improved = False
        for i in range(fixed, n - 1):
            for j in range(i + 1, n):
                old = distances[path[i - 1], path[i]]
                new = distances[path[i - 1], path[j]]
                if j + 1 < n:
                    old += distances[path[j], path[j + 1]]
                    new += distances[path[i], path[j + 1]]
                if new < old - 1e-9:
                    path[i : j + 1] = reversed(path[i : j + 1])
                    improved = True
        if not improved:
            break
    return path


def border_anchors(size):
    w, h = size
    return np.array(
        [
            [0, 0],
            [(w - 1) / 2, 0],
            [w - 1, 0],
            [w - 1, (h - 1) / 2],
            [w - 1, h - 1],
            [(w - 1) / 2, h - 1],
            [0, h - 1],
            [0, (h - 1) / 2],
        ],
        float,
    )


def warp_image(image, source, target):
    """Inverse piecewise-affine remap, triangulated in target space.

    Border anchors keep the output filled. Each valid landmark is mapped exactly;
    coincident targets are deduplicated. Work is bounded to triangle rectangles.
    """
    h, w = image.shape[:2]
    source = np.vstack([border_anchors((w, h)), source])
    target = np.vstack([border_anchors((w, h)), target])
    _, unique = np.unique(np.round(target, 3), axis=0, return_index=True)
    source, target = source[unique], target[unique]
    triangles = Delaunay(target).simplices
    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    for indices in triangles:
        dst, src = target[indices], source[indices]
        lo = np.maximum(np.floor(dst.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(dst.max(0)).astype(int), [w - 1, h - 1])
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        bary = np.linalg.solve(
            np.vstack([dst.T, np.ones(3)]),
            np.vstack([xx.ravel(), yy.ravel(), np.ones(xx.size)]),
        )
        mask = np.all(bary >= -1e-6, axis=0)
        mapped = src.T @ bary[:, mask]
        map_x[yy.ravel()[mask], xx.ravel()[mask]] = mapped[0]
        map_y[yy.ravel()[mask], xx.ravel()[mask]] = mapped[1]
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def morph_frames(a, b, pa, pb, scores_a, scores_b, count):
    valid = np.minimum(scores_a, scores_b) >= 0.3
    # Exclude landmarks outside the output crop; boundary anchors own its edges.
    h, w = a.shape[:2]
    for p in (pa, pb):
        valid &= (p[:, 0] > 1) & (p[:, 0] < w - 2) & (p[:, 1] > 1) & (p[:, 1] < h - 2)
    sa, sb = pa[valid], pb[valid]
    for frame in range(1, count + 1):
        t = frame / (count + 1)
        target = (1 - t) * sa + t * sb
        wa, wb = warp_image(a, sa, target), warp_image(b, sb, target)
        yield cv2.addWeighted(wa, 1 - t, wb, t, 0)
