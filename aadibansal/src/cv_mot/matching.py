"""Data association: cost construction and optimal assignment.

The assignment itself is the Hungarian (Jonker-Volgenant) algorithm. The
interesting engineering is in how the cost matrix is built, because the cost is
where the domain knowledge lives.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.optimize

from .boxes import giou_matrix, iou_matrix, tlwh_to_xyxy

__all__ = [
    "linear_assignment",
    "iou_distance",
    "giou_distance",
    "embedding_distance",
    "fuse_score",
    "fuse_motion",
    "gate_cost_matrix",
]


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a rectangular assignment problem with a cost ceiling.

    Returns ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is an
    ``(M, 2)`` integer array of index pairs.

    Pairs whose cost exceeds ``thresh`` are rejected *after* the global solve
    rather than being removed beforehand. That ordering matters: the optimal
    global assignment can legitimately route through a pair that is individually
    mediocre, and filtering first would change which solution is found.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    # Sentinel keeps the problem feasible while making over-threshold pairs
    # unattractive to the solver.
    guard = thresh + 1e-4
    solvable = np.where(cost_matrix > guard, guard, cost_matrix)
    rows, cols = scipy.optimize.linear_sum_assignment(solvable)

    keep = cost_matrix[rows, cols] <= thresh
    matches = np.stack([rows[keep], cols[keep]], axis=1) if keep.any() else np.empty((0, 2), dtype=int)

    unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), matches[:, 1])
    return matches, unmatched_a, unmatched_b


def _to_xyxy(items: Sequence) -> np.ndarray:
    """Accept tracks (objects with ``.tlwh``) or raw ``tlwh`` arrays."""
    if len(items) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    first = items[0]
    if hasattr(first, "tlwh"):
        boxes = np.asarray([t.tlwh for t in items], dtype=np.float64)
    else:
        boxes = np.asarray(items, dtype=np.float64).reshape(-1, 4)
    return tlwh_to_xyxy(boxes)


def iou_distance(tracks_a: Sequence, tracks_b: Sequence) -> np.ndarray:
    """``1 - IoU`` cost matrix."""
    return 1.0 - iou_matrix(_to_xyxy(tracks_a), _to_xyxy(tracks_b))


def giou_distance(tracks_a: Sequence, tracks_b: Sequence) -> np.ndarray:
    """``1 - GIoU`` cost matrix, for recovering tracks that have drifted apart."""
    return 1.0 - giou_matrix(_to_xyxy(tracks_a), _to_xyxy(tracks_b))


def embedding_distance(tracks: Sequence, detections: Sequence, metric: str = "cosine") -> np.ndarray:
    """Appearance cost between track feature banks and detection features."""
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost.size == 0:
        return cost

    det_features = np.asarray([d.curr_feature for d in detections], dtype=np.float64)
    track_features = np.asarray([t.smooth_feature for t in tracks], dtype=np.float64)

    if metric == "cosine":
        a = track_features / np.maximum(np.linalg.norm(track_features, axis=1, keepdims=True), 1e-12)
        b = det_features / np.maximum(np.linalg.norm(det_features, axis=1, keepdims=True), 1e-12)
        cost = np.clip(1.0 - a @ b.T, 0.0, 2.0)
    elif metric == "euclidean":
        diff = track_features[:, None, :] - det_features[None, :, :]
        cost = np.linalg.norm(diff, axis=2)
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown metric: {metric!r}")
    return cost


def fuse_score(cost_matrix: np.ndarray, detections: Sequence) -> np.ndarray:
    """Weight an IoU cost by detector confidence.

    A high-confidence detection that overlaps a track is stronger evidence than
    a marginal one with the same overlap, so similarity is multiplied by score.
    This is the ByteTrack ``fuse_score`` trick and it chiefly helps when a real
    pedestrian and a detector artefact compete for the same track.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    scores = np.asarray([d.score for d in detections], dtype=np.float64)
    fused = iou_sim * scores[None, :]
    return 1.0 - fused


def fuse_motion(
    kf,
    cost_matrix: np.ndarray,
    tracks: Sequence,
    detections: Sequence,
    only_position: bool = False,
    lambda_: float = 0.98,
    gate_value: float = 1e5,
) -> np.ndarray:
    """Blend an appearance cost with Kalman motion plausibility.

    Used only on the appearance-enabled path. Pairs outside the 95% confidence
    ellipse are hard-gated; the rest get a convex combination of appearance and
    (scaled) Mahalanobis distance.
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = __import__("cv_mot.kalman", fromlist=["CHI2_INV95"]).CHI2_INV95[gating_dim]
    measurements = np.asarray([d.to_xyah() for d in detections], dtype=np.float64)

    out = cost_matrix.copy()
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position=only_position
        )
        out[row, gating_distance > gating_threshold] = gate_value
        out[row] = lambda_ * out[row] + (1.0 - lambda_) * gating_distance
    return out


def gate_cost_matrix(
    kf,
    cost_matrix: np.ndarray,
    tracks: Sequence,
    detections: Sequence,
    only_position: bool = True,
    gate_value: float = 1e5,
) -> np.ndarray:
    """Hard Mahalanobis gate without blending."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = __import__("cv_mot.kalman", fromlist=["CHI2_INV95"]).CHI2_INV95[gating_dim]
    measurements = np.asarray([d.to_xyah() for d in detections], dtype=np.float64)

    out = cost_matrix.copy()
    for row, track in enumerate(tracks):
        d = kf.gating_distance(track.mean, track.covariance, measurements, only_position=only_position)
        out[row, d > gating_threshold] = gate_value
    return out
