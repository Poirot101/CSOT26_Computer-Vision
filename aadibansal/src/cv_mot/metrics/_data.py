"""Sequence alignment and MOTChallenge preprocessing.

The preprocessing step here is not optional bookkeeping -- it changes scores by
a large margin, and getting it wrong is the usual reason a local number
disagrees with a leaderboard.

MOTChallenge ground truth annotates far more than the pedestrians you are
scored on. It also marks people on vehicles, static (sitting/standing-still)
people, explicit "distractor" regions such as mannequins and posters, and
reflections. Those are *neither* targets nor errors: a detector that fires on a
shop-window reflection has not made a mistake the benchmark wants to punish, it
has found something the annotators deliberately excluded.

So the official protocol is:

* **Ground truth to score against**: class 1 (pedestrian) with the ignore flag
  unset.
* **Predictions that overlap a distractor class** (2 person-on-vehicle,
  7 static-person, 8 distractor, 12 reflection) are *deleted* rather than
  counted as false positives.

Skipping the second rule inflates FP heavily on these sequences and makes a
good tracker look broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize

from ..boxes import iou_matrix, tlwh_to_xyxy

__all__ = ["SequenceData", "build_sequence_data", "DISTRACTOR_CLASSES", "PEDESTRIAN_CLASS"]

PEDESTRIAN_CLASS = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)
"""person_on_vehicle, static_person, distractor, reflection."""

_EPS = np.finfo("float").eps


@dataclass
class SequenceData:
    """Per-frame aligned ground truth and predictions, after preprocessing.

    ``gt_ids`` / ``pred_ids`` hold *contiguous, zero-based* identity indices --
    the metrics allocate dense matrices over them, so the raw file IDs (which
    can be sparse and large) are remapped first.
    """

    gt_ids: list[np.ndarray]
    pred_ids: list[np.ndarray]
    similarity: list[np.ndarray]
    num_gt_ids: int
    num_pred_ids: int
    num_gt_dets: int
    num_pred_dets: int
    num_frames: int
    name: str = ""


def _group_by_frame(
    rows: np.ndarray, num_frames: int, num_cols: int
) -> list[np.ndarray]:
    """Bucket ``(frame, ...)`` rows into a dense per-frame list."""
    out: list[np.ndarray] = [np.zeros((0, num_cols)) for _ in range(num_frames)]
    if rows.size == 0:
        return out
    frames = rows[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    rows, frames = rows[order], frames[order]
    bounds = np.searchsorted(frames, np.arange(1, num_frames + 2))
    for f in range(num_frames):
        out[f] = rows[bounds[f] : bounds[f + 1]]
    return out


def build_sequence_data(
    gt_rows: np.ndarray,
    pred_rows: np.ndarray,
    num_frames: int,
    name: str = "",
    distractor_iou: float = 0.5,
    apply_preprocessing: bool = True,
) -> SequenceData:
    """Align and preprocess one sequence.

    Parameters
    ----------
    gt_rows:
        ``(N, 9)`` array: ``frame, id, l, t, w, h, conf, class, visibility``.
    pred_rows:
        ``(M, >=6)`` array: ``frame, id, l, t, w, h, ...``.
    num_frames:
        From ``seqinfo.ini``; frames with no rows on either side still count.
    apply_preprocessing:
        Set ``False`` to score raw rows, which is useful for unit tests where
        the distractor machinery would obscure the property under test.
    """
    gt_rows = np.asarray(gt_rows, dtype=np.float64).reshape(-1, 9 if gt_rows.size else 9)
    pred_rows = np.asarray(pred_rows, dtype=np.float64)
    if pred_rows.size == 0:
        pred_rows = np.zeros((0, 10))

    gt_by_frame = _group_by_frame(gt_rows, num_frames, 9)
    pred_by_frame = _group_by_frame(pred_rows, num_frames, pred_rows.shape[1])

    gt_ids_per_frame: list[np.ndarray] = []
    pred_ids_per_frame: list[np.ndarray] = []
    sims: list[np.ndarray] = []

    for f in range(num_frames):
        g = gt_by_frame[f]
        p = pred_by_frame[f]

        g_boxes = tlwh_to_xyxy(g[:, 2:6]) if g.size else np.zeros((0, 4))
        p_boxes = tlwh_to_xyxy(p[:, 2:6]) if p.size else np.zeros((0, 4))
        # Similarity is computed against *all* ground truth, distractors
        # included, because distractor suppression needs those overlaps.
        sim_all = iou_matrix(g_boxes, p_boxes)

        if apply_preprocessing:
            to_remove: np.ndarray = np.zeros(0, dtype=int)
            if g.size and p.size:
                scores = sim_all.copy()
                scores[scores < distractor_iou - _EPS] = 0.0
                rows, cols = scipy.optimize.linear_sum_assignment(-scores)
                matched = scores[rows, cols] > _EPS
                rows, cols = rows[matched], cols[matched]
                is_distractor = np.isin(g[rows, 7].astype(int), DISTRACTOR_CLASSES)
                to_remove = cols[is_distractor]

            pred_keep = np.setdiff1d(np.arange(p.shape[0]), to_remove)
            gt_keep = np.flatnonzero(
                (g[:, 6] != 0) & (g[:, 7].astype(int) == PEDESTRIAN_CLASS)
            ) if g.size else np.zeros(0, dtype=int)
        else:
            pred_keep = np.arange(p.shape[0])
            gt_keep = np.arange(g.shape[0])

        gt_ids_per_frame.append(g[gt_keep, 1].astype(int) if g.size else np.zeros(0, dtype=int))
        pred_ids_per_frame.append(p[pred_keep, 1].astype(int) if p.size else np.zeros(0, dtype=int))
        sims.append(sim_all[np.ix_(gt_keep, pred_keep)] if sim_all.size else np.zeros((len(gt_keep), len(pred_keep))))

    # Remap sparse file IDs onto contiguous indices.
    def remap(id_lists: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        unique = np.unique(np.concatenate(id_lists)) if any(len(x) for x in id_lists) else np.zeros(0, dtype=int)
        lookup = {int(v): i for i, v in enumerate(unique)}
        return [np.array([lookup[int(v)] for v in ids], dtype=int) for ids in id_lists], len(unique)

    gt_ids_per_frame, num_gt_ids = remap(gt_ids_per_frame)
    pred_ids_per_frame, num_pred_ids = remap(pred_ids_per_frame)

    return SequenceData(
        gt_ids=gt_ids_per_frame,
        pred_ids=pred_ids_per_frame,
        similarity=sims,
        num_gt_ids=num_gt_ids,
        num_pred_ids=num_pred_ids,
        num_gt_dets=int(sum(len(x) for x in gt_ids_per_frame)),
        num_pred_dets=int(sum(len(x) for x in pred_ids_per_frame)),
        num_frames=num_frames,
        name=name,
    )
