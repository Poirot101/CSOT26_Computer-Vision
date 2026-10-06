"""HOTA: Higher Order Tracking Accuracy.

HOTA exists because MOTA and IDF1 disagree about what tracking is for. MOTA is
dominated by detection count; IDF1 is dominated by identity consistency. HOTA
separates the two explicitly and then recombines them as a geometric mean:

    HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)

and averages over localisation thresholds ``alpha`` from 0.05 to 0.95, so the
final number does not hinge on one arbitrary IoU cutoff.

The subtle part is the matching. A purely per-frame greedy or Hungarian match on
IoU would let the same ground-truth person be served by different predicted IDs
in different frames with no penalty. HOTA instead first estimates, globally, how
well each (gt_id, pred_id) pair corresponds across the whole sequence, and then
biases the per-frame assignment toward those globally-consistent pairs. That is
what makes AssA measure association rather than luck.

Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking", IJCV 2021.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize

from ._data import SequenceData

__all__ = ["ALPHAS", "hota_sequence", "combine_hota"]

ALPHAS = np.arange(0.05, 0.99, 0.05)
_EPS = np.finfo("float").eps


def hota_sequence(data: SequenceData) -> dict[str, np.ndarray | float]:
    """Compute per-alpha HOTA components for one sequence.

    Returns raw counts alongside the ratios so that multiple sequences can be
    combined by summing counts -- averaging per-sequence HOTA would weight a
    500-frame sequence the same as a 1500-frame one.
    """
    n_alpha = len(ALPHAS)
    res: dict[str, np.ndarray] = {
        key: np.zeros(n_alpha, dtype=np.float64)
        for key in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "LocA")
    }

    if data.num_gt_dets == 0 or data.num_pred_dets == 0:
        res["HOTA"] = np.zeros(n_alpha)
        res["DetA"] = np.zeros(n_alpha)
        res["AssA"] = np.zeros(n_alpha)
        res["HOTA_FN"] = np.full(n_alpha, float(data.num_gt_dets))
        res["HOTA_FP"] = np.full(n_alpha, float(data.num_pred_dets))
        res["LocA"] = np.ones(n_alpha)
        return res

    n_gt, n_pr = data.num_gt_ids, data.num_pred_ids

    # --- Pass 1: global correspondence estimate between identity pairs -------
    potential = np.zeros((n_gt, n_pr), dtype=np.float64)
    gt_id_count = np.zeros((n_gt, 1), dtype=np.float64)
    pr_id_count = np.zeros((1, n_pr), dtype=np.float64)

    for gt_ids, pr_ids, sim in zip(data.gt_ids, data.pred_ids, data.similarity):
        if sim.size:
            # Normalising each overlap by the total overlap mass of its row and
            # column keeps a single ambiguous frame in a dense crowd from
            # dominating the global estimate.
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > _EPS
            sim_iou[mask] = sim[mask] / denom[mask]
            potential[gt_ids[:, None], pr_ids[None, :]] += sim_iou
        gt_id_count[gt_ids] += 1
        pr_id_count[0, pr_ids] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        global_alignment = np.where(
            (gt_id_count + pr_id_count - potential) > 0,
            potential / np.maximum(gt_id_count + pr_id_count - potential, _EPS),
            0.0,
        )

    matches_counts = [np.zeros((n_gt, n_pr), dtype=np.float64) for _ in ALPHAS]

    # --- Pass 2: alignment-biased per-frame assignment -----------------------
    for gt_ids, pr_ids, sim in zip(data.gt_ids, data.pred_ids, data.similarity):
        if sim.size:
            score_mat = global_alignment[gt_ids[:, None], pr_ids[None, :]] * sim
            rows, cols = scipy.optimize.linear_sum_assignment(-score_mat)
        else:
            rows = cols = np.zeros(0, dtype=int)

        for a, alpha in enumerate(ALPHAS):
            if len(rows):
                keep = sim[rows, cols] >= alpha - _EPS
                ar, ac = rows[keep], cols[keep]
            else:
                ar = ac = np.zeros(0, dtype=int)
            n_match = len(ar)
            res["HOTA_TP"][a] += n_match
            res["HOTA_FN"][a] += len(gt_ids) - n_match
            res["HOTA_FP"][a] += len(pr_ids) - n_match
            if n_match:
                res["LocA"][a] += sim[ar, ac].sum()
                matches_counts[a][gt_ids[ar], pr_ids[ac]] += 1

    # --- Combine into DetA / AssA / HOTA ------------------------------------
    det_a = np.zeros(n_alpha)
    ass_a = np.zeros(n_alpha)
    for a in range(n_alpha):
        mc = matches_counts[a]
        tp = res["HOTA_TP"][a]
        denom = np.maximum(gt_id_count + pr_id_count - mc, 1.0)
        # Per-pair association IoU: TPA / (TPA + FNA + FPA).
        pair_ass = mc / denom
        ass_a[a] = (mc * pair_ass).sum() / max(tp, 1.0)
        det_a[a] = tp / max(tp + res["HOTA_FN"][a] + res["HOTA_FP"][a], 1.0)
        res["LocA"][a] = res["LocA"][a] / max(tp, 1.0)

    res["DetA"] = det_a
    res["AssA"] = ass_a
    res["HOTA"] = np.sqrt(det_a * ass_a)
    return res


def combine_hota(per_sequence: list[dict]) -> dict[str, float]:
    """Pool sequences by summing counts, then recompute the ratios.

    This is detection-weighted, matching how MOTChallenge reports an overall
    score across a benchmark's sequences.
    """
    n_alpha = len(ALPHAS)
    tp = np.zeros(n_alpha)
    fn = np.zeros(n_alpha)
    fp = np.zeros(n_alpha)
    loca_weighted = np.zeros(n_alpha)
    ass_weighted = np.zeros(n_alpha)

    for res in per_sequence:
        tp += res["HOTA_TP"]
        fn += res["HOTA_FN"]
        fp += res["HOTA_FP"]
        loca_weighted += np.asarray(res["LocA"]) * res["HOTA_TP"]
        ass_weighted += np.asarray(res["AssA"]) * res["HOTA_TP"]

    det_a = tp / np.maximum(tp + fn + fp, 1.0)
    ass_a = ass_weighted / np.maximum(tp, 1.0)
    loca = loca_weighted / np.maximum(tp, 1.0)
    hota = np.sqrt(det_a * ass_a)
    return {
        "HOTA": float(hota.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()),
        "LocA": float(loca.mean()),
    }
