"""Geometry: conversions must round-trip and IoU must match hand-computed values."""

import numpy as np
import pytest

from cv_mot.boxes import (
    clip_tlwh,
    giou_matrix,
    iou_matrix,
    tlwh_to_xyah,
    tlwh_to_xyxy,
    xyah_to_tlwh,
    xyxy_to_tlwh,
)


@pytest.mark.parametrize("box", [[10, 20, 30, 40], [0, 0, 1, 1], [5.5, 6.25, 100.0, 200.0]])
def test_conversions_round_trip(box):
    arr = np.array([box], dtype=float)
    assert np.allclose(xyxy_to_tlwh(tlwh_to_xyxy(arr)), arr)
    assert np.allclose(xyah_to_tlwh(tlwh_to_xyah(arr)), arr)


def test_single_box_keeps_1d_shape():
    assert tlwh_to_xyxy(np.array([1.0, 2.0, 3.0, 4.0])).shape == (4,)


def test_xyah_is_centre_and_aspect():
    assert np.allclose(tlwh_to_xyah(np.array([10.0, 20.0, 40.0, 80.0])), [30.0, 60.0, 0.5, 80.0])


def test_iou_known_values():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],   # identical
            [5.0, 0.0, 15.0, 10.0],   # half overlap -> 50/150
            [10.0, 0.0, 20.0, 10.0],  # edge touching
            [20.0, 20.0, 30.0, 30.0], # disjoint
        ]
    )
    assert np.allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0, 0.0])


def test_iou_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 50, (7, 2))
    boxes_a = np.hstack([a, a + rng.uniform(1, 40, (7, 2))])
    b = rng.uniform(0, 50, (5, 2))
    boxes_b = np.hstack([b, b + rng.uniform(1, 40, (5, 2))])
    m = iou_matrix(boxes_a, boxes_b)
    assert m.shape == (7, 5)
    assert np.all((m >= 0) & (m <= 1))
    assert np.allclose(m, iou_matrix(boxes_b, boxes_a).T)


def test_iou_handles_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)
    assert iou_matrix(np.ones((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_degenerate_zero_area_box_is_not_nan():
    m = iou_matrix(np.array([[5.0, 5.0, 5.0, 5.0]]), np.array([[0.0, 0.0, 10.0, 10.0]]))
    assert np.isfinite(m).all() and m[0, 0] == 0.0


def test_giou_separates_disjoint_boxes_that_iou_cannot():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    near = giou_matrix(a, np.array([[12.0, 0.0, 22.0, 10.0]]))[0, 0]
    far = giou_matrix(a, np.array([[200.0, 0.0, 210.0, 10.0]]))[0, 0]
    assert iou_matrix(a, np.array([[12.0, 0.0, 22.0, 10.0]]))[0, 0] == 0.0
    assert near > far


def test_clip_keeps_boxes_inside_frame():
    out = clip_tlwh(np.array([[-20.0, -10.0, 100.0, 50.0]]), 640, 480)
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert out[0, 2] == 80.0 and out[0, 3] == 40.0


def test_clip_never_produces_negative_size():
    out = clip_tlwh(np.array([[700.0, 500.0, 50.0, 50.0]]), 640, 480)
    assert out[0, 2] >= 0 and out[0, 3] >= 0
