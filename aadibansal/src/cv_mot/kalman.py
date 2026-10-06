"""A constant-velocity Kalman filter over ``(cx, cy, aspect, height)``.

The 8-dimensional state is ``(cx, cy, a, h, vx, vy, va, vh)``: box centre,
aspect ratio, height, and their first derivatives. Motion is linear with unit
time step; the measurement is the four positional components.

Two design choices are worth stating explicitly because they are not obvious
and they matter for dense crowds:

1.  **Noise scales with object height.** A pedestrian 400 px tall moves many
    more pixels per frame than one 40 px tall, so a fixed covariance is wrong
    for both. Standard deviations are therefore proportional to ``h``, which
    is the usual SORT/DeepSORT formulation and is what makes one parameter set
    work across the whole depth range of a scene.
2.  **Aspect ratio is treated as nearly static.** Its process noise is held
    small and constant, because a walking person's width/height ratio is
    roughly fixed while their pixel size is not. Letting aspect drift freely
    makes boxes degenerate during long occlusions.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

__all__ = ["KalmanFilterXYAH", "CHI2_INV95"]

# 0.95 quantile of the chi-square distribution, indexed by degrees of freedom
# (1-based). Used for Mahalanobis gating of implausible assignments.
CHI2_INV95: dict[int, float] = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilterXYAH:
    """Constant-velocity filter in ``xyah`` space.

    Parameters
    ----------
    std_weight_position:
        Positional noise as a fraction of box height. Larger values trust the
        detector more and the motion model less.
    std_weight_velocity:
        Velocity noise as a fraction of box height.
    aspect_std:
        Fixed standard deviation used for the aspect-ratio component, which is
        deliberately decoupled from ``h``.
    """

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
        aspect_std: float = 1e-2,
    ) -> None:
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = float(std_weight_position)
        self._std_weight_velocity = float(std_weight_velocity)
        self._aspect_std = float(aspect_std)

    # ---------------------------------------------------------------- helpers
    def _position_std(self, height: np.ndarray | float) -> np.ndarray:
        h = np.asarray(height, dtype=np.float64)
        return np.stack(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                np.full_like(h, self._aspect_std),
                self._std_weight_position * h,
            ],
            axis=-1,
        )

    def _velocity_std(self, height: np.ndarray | float) -> np.ndarray:
        h = np.asarray(height, dtype=np.float64)
        return np.stack(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                np.full_like(h, self._aspect_std * 1e-3),
                self._std_weight_velocity * h,
            ],
            axis=-1,
        )

    # ------------------------------------------------------------------- core
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated detection.

        Velocity is initialised to zero with a deliberately wide covariance:
        a brand-new track has no evidence about where the object is heading,
        so the filter must not commit to "stationary".
        """
        measurement = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[measurement, np.zeros(4)]

        h = measurement[3]
        std = np.r_[
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            self._aspect_std,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            self._aspect_std * 1e-2,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame."""
        h = mean[3]
        motion_cov = np.diag(
            np.square(np.r_[self._position_std(h), self._velocity_std(h)])
        )
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`predict` over a batch of tracks.

        Dense sequences carry hundreds of simultaneous tracks; doing this one
        at a time dominates the per-frame cost once the detector is warm.
        """
        if len(means) == 0:
            return means, covariances
        means = np.asarray(means, dtype=np.float64)
        covariances = np.asarray(covariances, dtype=np.float64)

        h = means[:, 3]
        std = np.concatenate([self._position_std(h), self._velocity_std(h)], axis=1)
        motion_cov = np.stack([np.diag(s) for s in np.square(std)], axis=0)

        means = means @ self._motion_mat.T
        covariances = (
            self._motion_mat @ covariances @ self._motion_mat.T + motion_cov
        )
        return means, covariances

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project a state into measurement space."""
        h = mean[3]
        innovation_cov = np.diag(np.square(self._position_std(h)))
        mean_out = self._update_mat @ mean
        cov_out = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean_out, cov_out

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct a state with an associated detection."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from a state to each measurement.

        ``only_position`` restricts the comparison to ``(cx, cy)``, which is the
        safer choice in crowds: a partially-occluded detection has a badly
        wrong height, and gating on height would then reject the correct match.
        """
        proj_mean, proj_cov = self.project(mean, covariance)
        measurements = np.atleast_2d(np.asarray(measurements, dtype=np.float64))
        if only_position:
            proj_mean, proj_cov = proj_mean[:2], proj_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(proj_cov)
        d = measurements - proj_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)
