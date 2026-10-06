"""Track state machine and the per-track bookkeeping shared by all trackers."""

from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np

from .boxes import tlwh_to_xyah, xyah_to_tlwh
from .kalman import KalmanFilterXYAH

__all__ = ["TrackState", "STrack", "BaseTrack"]


class TrackState(IntEnum):
    """Lifecycle of a single identity.

    ``New``
        Created this frame, not yet confirmed. Emitting these immediately would
        turn every detector false positive into a one-frame identity.
    ``Tracked``
        Confirmed and matched in the current frame.
    ``Lost``
        Confirmed but unmatched; coasting on the motion model. Still eligible
        for re-association, which is how occlusions are survived.
    ``Removed``
        Retired permanently.
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Process-wide identity counter.

    Identities must be unique within a sequence, so the counter is reset
    explicitly between sequences rather than relying on object lifetime.
    """

    _count: int = 0

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_counter() -> None:
        BaseTrack._count = 0


class STrack(BaseTrack):
    """A single tracked target.

    Holds the Kalman state, the lifecycle flags, and an optional appearance
    feature bank. Instances double as *detections* before they are activated:
    the tracker constructs one per detection each frame and only the matched or
    newly-initiated ones acquire an identity.
    """

    __slots__ = (
        "_tlwh",
        "out_tlwh",
        "score",
        "cls",
        "kalman_filter",
        "mean",
        "covariance",
        "is_activated",
        "state",
        "track_id",
        "frame_id",
        "start_frame",
        "tracklet_len",
        "curr_feature",
        "smooth_feature",
        "features",
        "feature_momentum",
    )

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        cls: int = 0,
        feature: np.ndarray | None = None,
        feature_history: int = 50,
        feature_momentum: float = 0.9,
    ) -> None:
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        # Box actually emitted for this frame. Kept separate from the Kalman
        # state: clipping to the image is an output concern, and writing a
        # clipped box back into the filter would corrupt the motion model.
        self.out_tlwh: np.ndarray | None = None
        self.score = float(score)
        self.cls = int(cls)

        self.kalman_filter: KalmanFilterXYAH | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        self.is_activated = False
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        self.curr_feature: np.ndarray | None = None
        self.smooth_feature: np.ndarray | None = None
        self.features: deque = deque(maxlen=feature_history)
        self.feature_momentum = float(feature_momentum)
        if feature is not None:
            self.update_feature(feature)

    # ------------------------------------------------------------- appearance
    def update_feature(self, feature: np.ndarray) -> None:
        """Fold a new appearance embedding into the running average.

        An exponential moving average is used rather than the raw latest
        embedding: a single frame where the person is half-occluded produces a
        contaminated feature, and the EMA keeps one bad frame from destroying
        an otherwise good identity model.
        """
        feature = np.asarray(feature, dtype=np.float64)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        self.curr_feature = feature
        if self.smooth_feature is None:
            self.smooth_feature = feature.copy()
        else:
            self.smooth_feature = (
                self.feature_momentum * self.smooth_feature
                + (1.0 - self.feature_momentum) * feature
            )
            smooth_norm = np.linalg.norm(self.smooth_feature)
            if smooth_norm > 0:
                self.smooth_feature = self.smooth_feature / smooth_norm
        self.features.append(feature)

    # ----------------------------------------------------------------- geometry
    @property
    def tlwh(self) -> np.ndarray:
        """Current box as ``(left, top, width, height)``, from the filter state."""
        if self.mean is None:
            return self._tlwh.copy()
        return xyah_to_tlwh(self.mean[:4])

    @property
    def output_tlwh(self) -> np.ndarray:
        """Box to write to the submission file.

        Falls back to the filter's box when no clipped output has been set.
        """
        return self.tlwh if self.out_tlwh is None else self.out_tlwh.copy()

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def to_xyah(self) -> np.ndarray:
        return tlwh_to_xyah(self.tlwh)

    # ---------------------------------------------------------------- lifecycle
    @staticmethod
    def multi_predict(tracks: list["STrack"], kf: KalmanFilterXYAH) -> None:
        """Advance a batch of tracks by one frame.

        Lost tracks have their height-velocity zeroed before prediction. Letting
        ``vh`` keep integrating while there is no measurement makes a coasting
        box grow or shrink without bound, and after a dozen frames it no longer
        overlaps anything -- the identity is then unrecoverable even when the
        person reappears exactly where predicted.
        """
        if not tracks:
            return
        means = np.asarray([t.mean.copy() for t in tracks])
        covs = np.asarray([t.covariance.copy() for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                means[i][7] = 0.0
        means, covs = kf.multi_predict(means, covs)
        for i, t in enumerate(tracks):
            t.mean, t.covariance = means[i], covs[i]

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int) -> None:
        """Promote a detection to a tracked identity."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # A track born on the very first frame is trusted immediately: there is
        # no previous frame that could have confirmed it, and withholding it
        # loses real detections at sequence start.
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        """Recover a lost identity from a fresh detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feature is not None:
            self.update_feature(new_track.curr_feature)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.cls = new_track.cls
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Standard measurement update for a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feature is not None:
            self.update_feature(new_track.curr_feature)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"STrack(id={self.track_id}, frames={self.start_frame}-{self.end_frame}, state={self.state.name})"
