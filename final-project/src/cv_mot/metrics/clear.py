"""CLEAR metrics: MOTA, MOTP, identity switches, recall and precision.

Reported for diagnosis rather than for the leaderboard. The pair (MOTA, IDF1) is
genuinely informative during development: MOTA rising while IDF1 falls means a
change bought detections at the cost of identity stability, which on this
benchmark is a losing trade.

Reference: Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
Performance: The CLEAR MOT Metrics", 2008.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize

from ._data import SequenceData

__all__ = ["clear_sequence", "combine_clear"]

_EPS = np.finfo("float").eps


def clear_sequence(data: SequenceData, threshold: float = 0.5) -> dict[str, float]:
    """Compute CLEAR counts for one sequence."""
    res = {k: 0.0 for k in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MOTP_sum", "Frag")}
    res["num_gt_dets"] = float(data.num_gt_dets)

    if data.num_gt_dets == 0:
        res["CLR_FP"] = float(data.num_pred_dets)
        res.update(MOTA=0.0, MOTP=0.0, Recall=0.0, Precision=0.0)
        return res

    # Last predicted ID seen for each gt ID, used to detect switches.
    prev_match = np.full(data.num_gt_ids, np.nan)
    prev_frame_match = np.full(data.num_gt_ids, np.nan)

    for gt_ids, pr_ids, sim in zip(data.gt_ids, data.pred_ids, data.similarity):
        if len(gt_ids) == 0:
            res["CLR_FP"] += len(pr_ids)
            continue
        if len(pr_ids) == 0:
            res["CLR_FN"] += len(gt_ids)
            prev_frame_match[:] = np.nan
            continue

        # Bias the assignment heavily toward whatever ID already owned this
        # person, so that MOTA does not record a switch merely because two
        # equally-good assignments existed.
        keeps_identity = (pr_ids[None, :] == prev_match[gt_ids][:, None]).astype(np.float64)
        score_mat = 1000.0 * keeps_identity + sim
        score_mat[sim < threshold - _EPS] = 0.0

        rows, cols = scipy.optimize.linear_sum_assignment(-score_mat)
        matched = score_mat[rows, cols] > _EPS
        rows, cols = rows[matched], cols[matched]

        matched_gt = gt_ids[rows]
        matched_pr = pr_ids[cols]

        had_previous = ~np.isnan(prev_match[matched_gt])
        switched = had_previous & (prev_match[matched_gt] != matched_pr)
        res["IDSW"] += float(switched.sum())

        res["CLR_TP"] += len(rows)
        res["CLR_FN"] += len(gt_ids) - len(rows)
        res["CLR_FP"] += len(pr_ids) - len(cols)
        res["MOTP_sum"] += float(sim[rows, cols].sum())

        prev_match[matched_gt] = matched_pr
        new_frame_match = np.full(data.num_gt_ids, np.nan)
        new_frame_match[matched_gt] = matched_pr
        # A fragment is a track that was present last frame and is now absent.
        res["Frag"] += float(
            np.sum(~np.isnan(prev_frame_match) & np.isnan(new_frame_match))
        )
        prev_frame_match = new_frame_match

    tp, fn, fp, idsw = res["CLR_TP"], res["CLR_FN"], res["CLR_FP"], res["IDSW"]
    res["MOTA"] = 1.0 - (fn + fp + idsw) / max(data.num_gt_dets, _EPS)
    res["MOTP"] = res["MOTP_sum"] / max(tp, _EPS)
    res["Recall"] = tp / max(tp + fn, _EPS)
    res["Precision"] = tp / max(tp + fp, _EPS)
    return res


def combine_clear(per_sequence: list[dict]) -> dict[str, float]:
    tp = sum(r["CLR_TP"] for r in per_sequence)
    fn = sum(r["CLR_FN"] for r in per_sequence)
    fp = sum(r["CLR_FP"] for r in per_sequence)
    idsw = sum(r["IDSW"] for r in per_sequence)
    motp_sum = sum(r["MOTP_sum"] for r in per_sequence)
    num_gt = sum(r["num_gt_dets"] for r in per_sequence)
    return {
        "MOTA": 1.0 - (fn + fp + idsw) / max(num_gt, _EPS),
        "MOTP": motp_sum / max(tp, _EPS),
        "Recall": tp / max(tp + fn, _EPS),
        "Precision": tp / max(tp + fp, _EPS),
        "IDSW": idsw,
        "CLR_TP": tp,
        "CLR_FN": fn,
        "CLR_FP": fp,
        "Frag": sum(r["Frag"] for r in per_sequence),
    }
