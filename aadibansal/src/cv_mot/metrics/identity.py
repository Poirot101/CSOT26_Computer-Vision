"""IDF1 and the identity-based counts.

IDF1 asks a different question from MOTA: not "how many boxes did you get
right", but "how consistently did you keep one identity on one person". It does
this by choosing a single global one-to-one mapping between ground-truth IDs and
predicted IDs for the whole sequence, then counting detections that agree with
that mapping.

The consequence worth internalising: a tracker that splits one person into two
IDs halfway through loses roughly half that person's detections to IDFN *and*
IDFP, because only one of the two predicted IDs can be mapped. This is why the
leaderboard weights IDF1 double -- it is the metric that actually punishes
fragmentation.

Reference: Ristani et al., "Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking", ECCVW 2016.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize

from ._data import SequenceData

__all__ = ["identity_sequence", "combine_identity"]

_EPS = np.finfo("float").eps


def identity_sequence(data: SequenceData, threshold: float = 0.5) -> dict[str, float]:
    """Compute IDTP / IDFN / IDFP / IDF1 for one sequence."""
    if data.num_gt_dets == 0 or data.num_pred_dets == 0:
        return {
            "IDTP": 0.0,
            "IDFN": float(data.num_gt_dets),
            "IDFP": float(data.num_pred_dets),
            "IDF1": 0.0,
            "IDP": 0.0,
            "IDR": 0.0,
        }

    n_gt, n_pr = data.num_gt_ids, data.num_pred_ids
    potential = np.zeros((n_gt, n_pr), dtype=np.float64)
    gt_id_count = np.zeros((n_gt, 1), dtype=np.float64)
    pr_id_count = np.zeros((1, n_pr), dtype=np.float64)

    for gt_ids, pr_ids, sim in zip(data.gt_ids, data.pred_ids, data.similarity):
        if sim.size:
            hits = sim >= threshold - _EPS
            np.add.at(potential, (gt_ids[:, None], pr_ids[None, :]), hits.astype(np.float64))
        gt_id_count[gt_ids] += 1
        pr_id_count[0, pr_ids] += 1

    # Minimise total identity error over the one-to-one mapping. Expressing the
    # objective as FN + FP (rather than maximising hits) is what MOTChallenge's
    # reference implementation does, and it differs from the naive formulation
    # when the ID counts are unbalanced.
    fn_mat = np.broadcast_to(gt_id_count, (n_gt, n_pr)) - potential
    fp_mat = np.broadcast_to(pr_id_count, (n_gt, n_pr)) - potential
    rows, cols = scipy.optimize.linear_sum_assignment(fn_mat + fp_mat)

    idtp = float(potential[rows, cols].sum())
    idfn = float(data.num_gt_dets - idtp)
    idfp = float(data.num_pred_dets - idtp)

    idf1 = idtp / max(idtp + 0.5 * idfn + 0.5 * idfp, _EPS)
    idp = idtp / max(idtp + idfp, _EPS)
    idr = idtp / max(idtp + idfn, _EPS)
    return {"IDTP": idtp, "IDFN": idfn, "IDFP": idfp, "IDF1": idf1, "IDP": idp, "IDR": idr}


def combine_identity(per_sequence: list[dict]) -> dict[str, float]:
    """Pool by summing counts -- detection-weighted, like the benchmark."""
    idtp = sum(r["IDTP"] for r in per_sequence)
    idfn = sum(r["IDFN"] for r in per_sequence)
    idfp = sum(r["IDFP"] for r in per_sequence)
    return {
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
        "IDF1": idtp / max(idtp + 0.5 * idfn + 0.5 * idfp, _EPS),
        "IDP": idtp / max(idtp + idfp, _EPS),
        "IDR": idtp / max(idtp + idfn, _EPS),
    }
