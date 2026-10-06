"""Metrics are validated against values derivable by hand.

These are the tests that matter most: a tracker tuned against a wrong evaluator
is tuned against nothing.
"""

import numpy as np
import pytest

from cv_mot.metrics import ALPHAS, evaluate, leaderboard_score


def make_gt(n_ids: int = 4, n_frames: int = 60, cls: int = 1, conf: int = 1) -> np.ndarray:
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(1, n_ids + 1):
        x, y = rng.uniform(0, 800), rng.uniform(0, 800)
        for f in range(1, n_frames + 1):
            rows.append([f, pid, x + 2 * f, y + f, 60, 140, conf, cls, 1])
    return np.array(rows, dtype=float)


def as_pred(rows: np.ndarray) -> np.ndarray:
    return np.c_[rows[:, :6], -np.ones((len(rows), 4))]


def score(gt, pred, n_frames=60, **kw):
    _, overall = evaluate({"s": (gt, pred, n_frames)}, **kw)
    return overall


def test_perfect_prediction_scores_one():
    gt = make_gt()
    o = score(gt, as_pred(gt))
    for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "Score"):
        assert o[key] == pytest.approx(1.0)
    assert o["IDSW"] == 0


def test_empty_prediction_scores_zero():
    o = score(make_gt(), np.zeros((0, 10)))
    assert o["HOTA"] == 0.0 and o["IDF1"] == 0.0 and o["Score"] == 0.0


def test_identity_switch_halves_association_not_detection():
    """Renaming every ID at the midpoint: detection perfect, association halved."""
    gt = make_gt()
    swapped = gt.copy()
    swapped[:, 1] = np.where(swapped[:, 0] > 30, swapped[:, 1] + 100, swapped[:, 1])
    o = score(gt, as_pred(swapped))
    assert o["DetA"] == pytest.approx(1.0)
    assert o["AssA"] == pytest.approx(0.5, abs=0.02)
    assert o["IDF1"] == pytest.approx(0.5, abs=0.02)


def test_duplicate_detections_halve_precision():
    gt = make_gt()
    dup = np.vstack([gt, np.c_[gt[:, 0], gt[:, 1] + 500, gt[:, 2:]]])
    o = score(gt, as_pred(dup))
    assert o["IDP"] == pytest.approx(0.5, abs=0.02)
    assert o["IDR"] == pytest.approx(1.0, abs=0.02)


def test_deta_counts_alphas_passing_the_iou_threshold():
    """A 20px shift on a 60px-wide box gives IoU exactly 0.5.

    DetA then averages 1.0 over alphas <= 0.5 and 0.0 above, i.e. 10/19.
    """
    gt = make_gt()
    shifted = gt.copy()
    shifted[:, 2] += 20
    o = score(gt, as_pred(shifted))
    expected = np.mean(ALPHAS <= 0.5 + 1e-12)
    assert o["DetA"] == pytest.approx(expected, abs=1e-6)
    assert o["IDF1"] == pytest.approx(1.0)  # IDF1 uses a single 0.5 threshold


def test_shift_past_half_iou_destroys_idf1():
    gt = make_gt()
    shifted = gt.copy()
    shifted[:, 2] += 40  # IoU 0.2 < 0.5
    assert score(gt, as_pred(shifted))["IDF1"] == 0.0


def test_mota_penalises_switches_and_can_go_negative():
    gt = make_gt(n_ids=2, n_frames=10)
    noise = gt.copy()
    noise[:, 1] = np.arange(len(noise)) + 1000  # a brand-new ID every row
    o = score(gt, as_pred(noise), n_frames=10)
    assert o["MOTA"] < 1.0
    assert o["IDF1"] < 0.2


def test_half_the_frames_halves_detection_accuracy():
    gt = make_gt()
    o = score(gt, as_pred(gt[gt[:, 0] <= 30]))
    assert o["DetA"] == pytest.approx(0.5, abs=0.02)


def test_distractor_overlapping_predictions_are_not_false_positives():
    """The core MOTChallenge preprocessing rule.

    The same surplus prediction is *forgiven* when it lands on a distractor
    annotation and *penalised* when it lands on empty ground. Note the control
    cannot be "preprocessing off", because with preprocessing off the distractor
    row is itself treated as a target and the surplus box simply matches it.
    """
    gt = make_gt(n_ids=2, n_frames=20)
    # A static-person annotation (class 7, ignore flag set) that a detector
    # legitimately fires on.
    distractor = np.array([[f, 900, 1500.0, 500.0, 60, 140, 0, 7, 1] for f in range(1, 21)])
    gt_all = np.vstack([gt, distractor])

    on_distractor = as_pred(np.vstack([gt, distractor]))
    stray = np.array([[f, 901, 50.0, 950.0, 60.0, 140.0, -1, -1, -1, -1] for f in range(1, 21)])
    on_empty_ground = np.vstack([as_pred(gt), stray])

    forgiven = score(gt_all, on_distractor, n_frames=20)
    penalised = score(gt_all, on_empty_ground, n_frames=20)

    assert forgiven["IDF1"] == pytest.approx(1.0)
    assert forgiven["IDFP"] == 0
    assert penalised["IDFP"] == 20
    assert penalised["IDF1"] < forgiven["IDF1"]


def test_ignored_ground_truth_is_not_required():
    gt = make_gt(n_ids=2, n_frames=20)
    ignored = np.array([[f, 901, 100.0, 900.0, 60, 140, 0, 1, 1] for f in range(1, 21)])
    o = score(np.vstack([gt, ignored]), as_pred(gt), n_frames=20)
    assert o["IDF1"] == pytest.approx(1.0)


def test_leaderboard_score_weights_idf1_twice():
    assert leaderboard_score(1.0, 0.0) == pytest.approx(2 / 3)
    assert leaderboard_score(0.0, 1.0) == pytest.approx(1 / 3)
    assert leaderboard_score(0.9, 0.6) == pytest.approx((2 * 0.9 + 0.6) / 3)


def test_metrics_are_order_invariant():
    gt = make_gt(n_ids=3, n_frames=15)
    pred = as_pred(gt)
    rng = np.random.default_rng(7)
    a = score(gt, pred, n_frames=15)
    b = score(gt, pred[rng.permutation(len(pred))], n_frames=15)
    assert a["IDF1"] == pytest.approx(b["IDF1"])
    assert a["HOTA"] == pytest.approx(b["HOTA"])
