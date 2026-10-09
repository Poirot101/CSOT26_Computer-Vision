"""Assignment must be optimal, threshold-respecting, and empty-safe."""

import numpy as np

from cv_mot.matching import fuse_score, iou_distance, linear_assignment


class _Det:
    def __init__(self, tlwh, score):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.score = score


def test_picks_the_globally_optimal_pairing():
    # Greedy would take (0,0)=0.10 then be forced into (1,1)=0.90 for 1.00.
    # The optimal pairing is (0,1)+(1,0) = 0.20 + 0.15 = 0.35.
    cost = np.array([[0.10, 0.20], [0.15, 0.90]])
    matches, _, _ = linear_assignment(cost, thresh=0.95)
    assert cost[matches[:, 0], matches[:, 1]].sum() == 0.35


def test_rejects_pairs_above_threshold():
    matches, ua, ub = linear_assignment(np.array([[0.9]]), thresh=0.5)
    assert len(matches) == 0 and ua.tolist() == [0] and ub.tolist() == [0]


def test_accepts_pair_exactly_at_threshold():
    matches, _, _ = linear_assignment(np.array([[0.5]]), thresh=0.5)
    assert len(matches) == 1


def test_handles_empty_matrices():
    for shape in [(0, 0), (0, 3), (3, 0)]:
        matches, ua, ub = linear_assignment(np.zeros(shape), thresh=0.5)
        assert len(matches) == 0
        assert len(ua) == shape[0] and len(ub) == shape[1]


def test_rectangular_leaves_surplus_unmatched():
    matches, ua, ub = linear_assignment(np.array([[0.1, 0.2, 0.3]]), thresh=0.9)
    assert len(matches) == 1 and len(ua) == 0 and len(ub) == 2


def test_every_index_appears_at_most_once():
    rng = np.random.default_rng(3)
    matches, _, _ = linear_assignment(rng.uniform(0, 1, (6, 8)), thresh=1.0)
    assert len(set(matches[:, 0])) == len(matches)
    assert len(set(matches[:, 1])) == len(matches)


def test_iou_distance_is_one_minus_iou():
    a = [_Det([0, 0, 10, 10], 1.0)]
    b = [_Det([0, 0, 10, 10], 1.0), _Det([100, 100, 10, 10], 1.0)]
    d = iou_distance(a, b)
    assert np.isclose(d[0, 0], 0.0) and np.isclose(d[0, 1], 1.0)


def test_fuse_score_penalises_low_confidence():
    dets = [_Det([0, 0, 10, 10], 1.0), _Det([0, 0, 10, 10], 0.2)]
    cost = np.array([[0.2, 0.2]])
    fused = fuse_score(cost, dets)
    assert fused[0, 1] > fused[0, 0]
