"""Gap interpolation as a post-processing step.

An online tracker coasts through an occlusion without emitting boxes: the
identity survives in memory, but those frames are reported as misses. Since the
track is confirmed on both sides of the gap, the person's position in between is
known to good accuracy by linear interpolation.

This recovers detections that the tracker already "knew about" and costs nothing
at inference time. The ByteTrack authors use the same trick on MOT17.

Two guards keep it honest:

* only gaps shorter than ``max_gap`` are filled -- over a long occlusion a
  straight line is a bad model of where someone walked;
* interpolation never *creates* an identity, it only fills between two observed
  endpoints of one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["interpolate_tracks"]


def interpolate_tracks(
    rows: np.ndarray, max_gap: int = 20, min_track_len: int = 2
) -> np.ndarray:
    """Fill short frame gaps within each identity.

    Parameters
    ----------
    rows:
        ``(N, >=6)`` MOT rows: ``frame, id, l, t, w, h, ...``.
    max_gap:
        Largest gap (in frames) to bridge.
    min_track_len:
        Identities shorter than this are left untouched.

    Returns
    -------
    A new array, sorted by ``(frame, id)``, with interpolated rows added.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return rows.reshape(0, 10)
    if rows.shape[1] < 10:
        pad = -np.ones((rows.shape[0], 10 - rows.shape[1]))
        rows = np.hstack([rows, pad])

    out = [rows]
    for tid in np.unique(rows[:, 1]):
        track = rows[rows[:, 1] == tid]
        track = track[np.argsort(track[:, 0])]
        if len(track) < min_track_len:
            continue
        frames = track[:, 0].astype(int)
        gaps = np.flatnonzero(np.diff(frames) > 1)
        for g in gaps:
            f0, f1 = frames[g], frames[g + 1]
            gap = f1 - f0
            if gap - 1 > max_gap:
                continue
            b0, b1 = track[g, 2:6], track[g + 1, 2:6]
            for step in range(1, gap):
                alpha = step / gap
                box = b0 + alpha * (b1 - b0)
                out.append(
                    np.array([[f0 + step, tid, box[0], box[1], box[2], box[3], -1, -1, -1, -1]])
                )

    merged = np.vstack(out)
    order = np.lexsort((merged[:, 1], merged[:, 0]))
    return merged[order]
