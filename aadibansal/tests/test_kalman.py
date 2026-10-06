"""The filter must track constant velocity and stay numerically sane."""

import numpy as np

from cv_mot.kalman import CHI2_INV95, KalmanFilterXYAH


def _run_constant_velocity(vx: float, steps: int = 30):
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 200.0]))
    for i in range(1, steps + 1):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + vx * i, 100.0, 0.5, 200.0]))
    return mean, cov


def test_learns_constant_velocity():
    mean, _ = _run_constant_velocity(10.0)
    assert abs(mean[4] - 10.0) < 1.0
    assert abs(mean[5]) < 1.0


def test_tracks_position_closely():
    mean, _ = _run_constant_velocity(10.0, steps=30)
    assert abs(mean[0] - (100.0 + 10.0 * 30)) < 5.0


def test_covariance_stays_symmetric_positive_definite():
    _, cov = _run_constant_velocity(7.0)
    assert np.allclose(cov, cov.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_prediction_inflates_uncertainty():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.4, 100.0]))
    _, cov2 = kf.predict(mean, cov)
    assert np.trace(cov2) > np.trace(cov)


def test_update_reduces_uncertainty():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.4, 100.0]))
    mean, cov = kf.predict(mean, cov)
    _, cov2 = kf.update(mean, cov, np.array([50.0, 50.0, 0.4, 100.0]))
    assert np.trace(cov2) < np.trace(cov)


def test_multi_predict_matches_single_predict():
    kf = KalmanFilterXYAH()
    m1, c1 = kf.initiate(np.array([10.0, 20.0, 0.5, 60.0]))
    m2, c2 = kf.initiate(np.array([300.0, 400.0, 0.3, 120.0]))
    means, covs = kf.multi_predict(np.stack([m1, m2]), np.stack([c1, c2]))
    for i, (m, c) in enumerate([(m1, c1), (m2, c2)]):
        em, ec = kf.predict(m, c)
        assert np.allclose(means[i], em)
        assert np.allclose(covs[i], ec)


def test_multi_predict_handles_empty():
    kf = KalmanFilterXYAH()
    means, covs = kf.multi_predict(np.zeros((0, 8)), np.zeros((0, 8, 8)))
    assert len(means) == 0 and len(covs) == 0


def test_noise_scales_with_object_height():
    """A tall (near) object must be allowed to move further per frame."""
    kf = KalmanFilterXYAH()
    _, small = kf.predict(*kf.initiate(np.array([0.0, 0.0, 0.5, 20.0])))
    _, large = kf.predict(*kf.initiate(np.array([0.0, 0.0, 0.5, 400.0])))
    assert large[0, 0] > small[0, 0]


def test_gating_distance_grows_with_displacement():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 200.0]))
    d = kf.gating_distance(
        mean, cov, np.array([[100.0, 100.0, 0.5, 200.0], [160.0, 100.0, 0.5, 200.0]]), only_position=True
    )
    assert d[0] < d[1]


def test_chi2_table_is_monotonic():
    values = [CHI2_INV95[k] for k in sorted(CHI2_INV95)]
    assert values == sorted(values)
