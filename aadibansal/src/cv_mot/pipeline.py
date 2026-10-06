"""Wiring: detector -> tracker -> MOT rows -> optional post-processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .interpolate import interpolate_tracks
from .mot_io import SequenceInfo


class TrackerLike(Protocol):
    def update(
        self,
        boxes_tlwh: np.ndarray,
        scores: np.ndarray,
        features: np.ndarray | None = ...,
        img_size: tuple[int, int] | None = ...,
    ) -> list: ...


@dataclass
class RunStats:
    frames: int = 0
    detections: int = 0
    output_rows: int = 0
    interpolated_rows: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def run_sequence(
    seq: SequenceInfo,
    detector,
    tracker_factory: Callable[[], TrackerLike],
    interpolate: bool = False,
    interpolate_max_gap: int = 20,
    clip_to_frame: bool = True,
    progress_every: int = 0,
) -> tuple[np.ndarray, RunStats]:
    """Track one sequence end to end.

    A fresh tracker is constructed here rather than reused across sequences, so
    identity numbering restarts at 1 as the submission format requires.

    Returns ``(rows, stats)`` where ``rows`` is an ``(N, 10)`` MOT array.
    """
    tracker = tracker_factory()
    stats = RunStats()
    img_size = (seq.width, seq.height) if clip_to_frame and seq.width else None

    records: list[list[float]] = []
    start = time.perf_counter()
    for frame_id, det in enumerate(detector.run(seq), start=1):
        stats.detections += len(det)
        tracks = tracker.update(det.boxes_tlwh, det.scores, det.features, img_size=img_size)
        for t in tracks:
            left, top, width, height = t.output_tlwh
            records.append([frame_id, t.track_id, left, top, width, height, -1, -1, -1, -1])
        stats.frames = frame_id
        if progress_every and frame_id % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"    {seq.name}: frame {frame_id}/{seq.length} "
                f"({frame_id / max(elapsed, 1e-9):.1f} fps)",
                flush=True,
            )

    stats.seconds = time.perf_counter() - start
    rows = np.asarray(records, dtype=np.float64) if records else np.zeros((0, 10))
    stats.output_rows = len(rows)

    if interpolate and len(rows):
        before = len(rows)
        rows = interpolate_tracks(rows, max_gap=interpolate_max_gap)
        stats.interpolated_rows = len(rows) - before

    return rows, stats
