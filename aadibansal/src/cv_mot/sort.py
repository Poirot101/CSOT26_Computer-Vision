"""SORT baseline, kept for honest ablation.

This is the Week 4 pipeline: one confidence threshold, IoU association in a
single pass, Kalman motion model, no appearance. It exists so that every claim
about ByteTrack's benefit can be stated as a measured delta on the same
detections rather than asserted.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
(arXiv:1602.00763).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import matching
from .kalman import KalmanFilterXYAH
from .track import BaseTrack, STrack, TrackState

__all__ = ["SortConfig", "SortTracker"]


@dataclass
class SortConfig:
    det_thresh: float = 0.5
    """Single confidence threshold -- the limitation ByteTrack removes."""
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3


class SortTracker:
    """Single-pass IoU tracker."""

    def __init__(self, config: SortConfig | None = None) -> None:
        self.cfg = config or SortConfig()
        self.kalman_filter = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[STrack] = []
        self.frame_id = 0
        BaseTrack.reset_counter()

    def update(
        self,
        boxes_tlwh: np.ndarray,
        scores: np.ndarray,
        features: np.ndarray | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> list[STrack]:
        self.frame_id += 1
        cfg = self.cfg

        boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        keep = scores >= cfg.det_thresh
        detections = [STrack(boxes_tlwh[i], scores[i]) for i in np.flatnonzero(keep)]

        STrack.multi_predict(self.tracks, self.kalman_filter)
        dists = matching.iou_distance(self.tracks, detections)
        matches, u_track, u_det = matching.linear_assignment(dists, 1.0 - cfg.iou_threshold)

        for itrack, idet in matches:
            self.tracks[itrack].update(detections[idet], self.frame_id)
        for i in u_track:
            self.tracks[i].mark_lost()
        for i in u_det:
            detections[i].activate(self.kalman_filter, self.frame_id)
            self.tracks.append(detections[i])

        self.tracks = [
            t for t in self.tracks if self.frame_id - t.end_frame <= cfg.max_age
        ]
        return [
            t
            for t in self.tracks
            if t.state == TrackState.Tracked
            and (t.tracklet_len >= cfg.min_hits or self.frame_id <= cfg.min_hits)
        ]
