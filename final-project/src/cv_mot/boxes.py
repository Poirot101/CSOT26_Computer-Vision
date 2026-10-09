"""Bounding-box representations and vectorised overlap computations.

Three box conventions appear in this project and mixing them up is the single
most common source of silently-wrong tracking output:

``tlwh``
    ``(left, top, width, height)``  -- the MOTChallenge file format.
``xyxy``
    ``(left, top, right, bottom)``  -- what Ultralytics YOLO returns.
``xyah``
    ``(centre_x, centre_y, aspect_ratio, height)`` -- the Kalman filter state
    space, where ``aspect_ratio = width / height``.

Every conversion here is a pure function over ``numpy`` arrays and accepts
either a single box of shape ``(4,)`` or a batch of shape ``(N, 4)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "tlwh_to_xyxy",
    "xyxy_to_tlwh",
    "tlwh_to_xyah",
    "xyah_to_tlwh",
    "clip_tlwh",
    "iou_matrix",
    "giou_matrix",
    "box_areas",
]


def _as_2d(boxes: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1), True
    return arr, False


def _restore(arr: np.ndarray, squeeze: bool) -> np.ndarray:
    return arr[0] if squeeze else arr


def tlwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(l, t, w, h)`` -> ``(l, t, r, b)``."""
    arr, squeeze = _as_2d(boxes)
    out = arr.copy()
    out[:, 2] = arr[:, 0] + arr[:, 2]
    out[:, 3] = arr[:, 1] + arr[:, 3]
    return _restore(out, squeeze)


def xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    """``(l, t, r, b)`` -> ``(l, t, w, h)``."""
    arr, squeeze = _as_2d(boxes)
    out = arr.copy()
    out[:, 2] = arr[:, 2] - arr[:, 0]
    out[:, 3] = arr[:, 3] - arr[:, 1]
    return _restore(out, squeeze)


def tlwh_to_xyah(boxes: np.ndarray) -> np.ndarray:
    """``(l, t, w, h)`` -> ``(cx, cy, w / h, h)``."""
    arr, squeeze = _as_2d(boxes)
    h = np.maximum(arr[:, 3], 1e-6)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0] + arr[:, 2] / 2.0
    out[:, 1] = arr[:, 1] + arr[:, 3] / 2.0
    out[:, 2] = arr[:, 2] / h
    out[:, 3] = arr[:, 3]
    return _restore(out, squeeze)


def xyah_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    """``(cx, cy, w / h, h)`` -> ``(l, t, w, h)``."""
    arr, squeeze = _as_2d(boxes)
    out = np.empty_like(arr)
    width = arr[:, 2] * arr[:, 3]
    out[:, 2] = width
    out[:, 3] = arr[:, 3]
    out[:, 0] = arr[:, 0] - width / 2.0
    out[:, 1] = arr[:, 1] - arr[:, 3] / 2.0
    return _restore(out, squeeze)


def clip_tlwh(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp ``tlwh`` boxes to the image, keeping width/height non-negative.

    Kalman extrapolation during an occlusion regularly pushes a predicted box
    partly outside the frame. MOTChallenge evaluators compare raw coordinates,
    so a box hanging off the edge costs IoU against the clipped ground truth.
    """
    arr, squeeze = _as_2d(boxes)
    x1 = np.clip(arr[:, 0], 0.0, width)
    y1 = np.clip(arr[:, 1], 0.0, height)
    x2 = np.clip(arr[:, 0] + arr[:, 2], 0.0, width)
    y2 = np.clip(arr[:, 1] + arr[:, 3], 0.0, height)
    out = np.stack([x1, y1, np.maximum(x2 - x1, 0.0), np.maximum(y2 - y1, 0.0)], axis=1)
    return _restore(out, squeeze)


def box_areas(boxes_xyxy: np.ndarray) -> np.ndarray:
    arr, _ = _as_2d(boxes_xyxy)
    return np.maximum(arr[:, 2] - arr[:, 0], 0.0) * np.maximum(arr[:, 3] - arr[:, 1], 0.0)


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Returns an ``(len(a), len(b))`` matrix. Empty inputs yield a correctly
    shaped empty array rather than raising, which keeps the association code
    free of special cases for the first frame and for fully-lost frames.
    """
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = box_areas(a)[:, None]
    area_b = box_areas(b)[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def giou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise generalised IoU, rescaled from ``[-1, 1]`` to ``[0, 1]``.

    GIoU keeps a usable gradient between boxes that do not overlap at all,
    which matters when a track has been coasting through an occlusion and its
    predicted box has drifted clear of the true one. Used as an optional
    association cost; plain IoU remains the default.
    """
    a = np.asarray(a_xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_xyxy, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    iou = iou_matrix(a, b)
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    enclosing = wh[..., 0] * wh[..., 1]

    area_a = box_areas(a)[:, None]
    area_b = box_areas(b)[None, :]
    inter = iou * (area_a + area_b) / (1.0 + iou)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        giou = iou - np.where(enclosing > 0, (enclosing - union) / enclosing, 0.0)
    return (giou + 1.0) / 2.0
