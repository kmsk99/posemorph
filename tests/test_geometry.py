import cv2
import numpy as np
import pytest

from posemorph.geometry import (
    distance_matrix,
    morph_frames,
    order_path,
    path_length,
    pose_distance,
    similarity,
    transform_points,
    warp_image,
)


@pytest.fixture
def pose():
    return np.array(
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


def test_known_similarity_recovered(pose):
    theta = 0.43
    matrix = np.column_stack(
        [
            1.7
            * np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            ),
            [12, -23],
        ]
    )
    recovered = similarity(
        pose, transform_points(pose, matrix), np.linspace(0.5, 1, 17)
    )
    np.testing.assert_allclose(recovered, matrix, atol=1e-10)


def test_distance_invariant_to_similarity(pose):
    a = {"keypoints": pose, "scores": np.ones(17)}
    matrix = np.array([[0, -2, 230], [2, 0, -100]])
    b = {"keypoints": transform_points(pose, matrix), "scores": np.ones(17)}
    assert pose_distance(a, b) < 1e-10


def test_order_reduces_pose_path(pose):
    rng = np.random.default_rng(7)
    records = []
    for amount in rng.permutation(np.linspace(-25, 25, 25)):
        points = pose.copy()
        points[[7, 9], 1] += amount
        points[[8, 10], 1] -= amount
        records.append({"keypoints": points, "scores": np.ones(17)})
    d = distance_matrix(records)
    result = order_path(d)
    assert sorted(result) == list(range(len(records)))
    assert path_length(result, d) < 0.35 * path_length(list(range(len(records))), d)
    hooked = order_path(d, start=3, hook=3)
    assert hooked[0] == 3
    assert sorted(hooked) == list(range(len(records)))
    assert path_length(hooked, d) < path_length(list(range(len(records))), d)
    # A hook must not reserve the most distant outliers as a jumping prefix.
    safe_hook = order_path(d, hook=3)
    center = int(np.argmin(d.mean(1)))
    assert d[center, safe_hook[0]] <= np.quantile(d[center], 0.25)
    nearby = np.flatnonzero(d[center] <= np.quantile(d[center], 0.25))
    expected = sorted(nearby, key=lambda i: (-d[i].mean(), i))[:3]
    assert set(safe_hook[:3]) == set(expected)


def test_piecewise_warp_lands_on_intermediate_targets():
    a = np.array([[25, 25], [75, 25], [50, 75]], float)
    b = np.array([[32, 32], [68, 20], [55, 70]], float)
    image = np.zeros((100, 100, 3), np.uint8)
    for point in a:
        cv2.circle(image, tuple(point.astype(int)), 2, (255, 255, 255), -1)
    middle = (a + b) / 2
    warped = warp_image(image, a, middle)
    for x, y in middle:
        neighborhood = warped[
            round(y) - 2 : round(y) + 3, round(x) - 2 : round(x) + 3, 0
        ]
        assert neighborhood.max() >= 200
    # Also check the second source and actual morph interpolation.
    image_b = np.zeros_like(image)
    for point in b:
        cv2.circle(image_b, tuple(point.astype(int)), 2, (255, 255, 255), -1)
    frames = list(morph_frames(image, image_b, a, b, np.ones(3), np.ones(3), 1))
    for x, y in middle:
        assert (
            frames[0][round(y) - 1 : round(y) + 2, round(x) - 1 : round(x) + 2, 0].max()
            >= 180
        )


def test_identity_warp_preserves_image():
    image = np.random.default_rng(2).integers(0, 256, (64, 48, 3), dtype=np.uint8)
    p = np.array([[15, 15], [25, 40]], float)
    np.testing.assert_array_equal(warp_image(image, p, p), image)
