"""ByteTrack: association that keeps low-confidence detections.

Why this tracker for dense crowds
---------------------------------
In a crowd, the dominant failure is not that the detector misses a person
outright -- it is that a partially-occluded person is detected with a *low*
score. A tracker with a single confidence threshold throws that detection away,
the track goes unmatched, and after a few frames the identity dies. When the
person emerges they get a new ID. That is an identity switch, and IDF1 punishes
it harder than anything else.

ByteTrack's observation is that a low-score box in roughly the right place is
still excellent evidence *if you already have a track there*. So association
happens in two passes:

1.  High-score detections are matched against all active and lost tracks.
2.  Low-score detections are matched against only the tracks that went
    unmatched in pass 1 -- and crucially, they are never allowed to start a new
    identity.

That asymmetry is the whole method: low-score boxes can sustain an existing
identity but cannot invent one, so recall improves without importing the false
positives that a globally-lowered threshold would bring.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", ECCV 2022 (arXiv:2110.06864).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import matching
from .boxes import clip_tlwh
from .kalman import KalmanFilterXYAH
from .track import BaseTrack, STrack, TrackState

__all__ = ["ByteTrackConfig", "ByteTracker"]


@dataclass
class ByteTrackConfig:
    """Tunable parameters, with the reasoning for each default.

    Defaults are the values selected by the sweep in ``scripts/tune.py`` on the
    reference sequence; see ``docs/tuning.md`` for the measured evidence.
    """

    track_high_thresh: float = 0.5
    """Score above which a detection may start a new identity and take part in
    the first association pass."""

    track_low_thresh: float = 0.1
    """Floor for the second pass. Below this, boxes are mostly background."""

    new_track_thresh: float = 0.6
    """Score required to create an identity. Deliberately stricter than
    ``track_high_thresh``: a spurious new ID is an immediate IDF1 penalty,
    whereas a real person missed for one frame is usually recovered."""

    match_thresh: float = 0.8
    """Maximum IoU cost for the first pass (i.e. IoU >= 0.2)."""

    second_match_thresh: float = 0.5
    """Stricter ceiling for low-score boxes -- they are weaker evidence, so the
    geometric agreement has to be better."""

    unconfirmed_match_thresh: float = 0.7
    """Ceiling for matching against not-yet-confirmed tracks."""

    track_buffer: int = 30
    """Frames a lost track is retained, scaled by frame rate at runtime. Longer
    buffers recover long occlusions but risk reviving a stale identity onto a
    different person."""

    min_box_area: float = 10.0
    """Boxes smaller than this are dropped from the output."""

    fuse_score: bool = False
    """Multiply IoU similarity by detector score. Helps on sparse scenes; the
    ByteTrack authors disable it for MOT20-style crowds because in dense scenes
    genuine pedestrians routinely score low and fusing then suppresses them."""

    use_giou: bool = False
    """Use GIoU instead of IoU for the first pass."""

    with_reid: bool = False
    """Blend appearance distance into the first association pass."""

    appearance_weight: float = 0.25
    """Weight on appearance when ``with_reid`` is set."""

    appearance_thresh: float = 0.35
    """Appearance distances above this are not allowed to assist a match."""

    duplicate_iou_thresh: float = 0.15
    """IoU-distance below which two tracks are considered the same object."""

    frame_rate: int = 30

    aspect_ratio_thresh: float = 10.0
    """Reject absurdly wide boxes, which are nearly always merged detections."""

    extra: dict = field(default_factory=dict)


def _join(list_a: list[STrack], list_b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in list_a}
    out = list(list_a)
    for t in list_b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _subtract(list_a: list[STrack], list_b: list[STrack]) -> list[STrack]:
    drop = {t.track_id for t in list_b}
    return [t for t in list_a if t.track_id not in drop]


def _remove_duplicates(
    tracks_a: list[STrack], tracks_b: list[STrack], thresh: float
) -> tuple[list[STrack], list[STrack]]:
    """Drop near-identical tracks across the active and lost pools.

    A lost track that is revived while an active track already covers the same
    person yields two IDs on one body for the rest of the sequence. The
    shorter-lived of the pair is discarded.
    """
    if not tracks_a or not tracks_b:
        return tracks_a, tracks_b
    pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < thresh)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        age_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        age_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        if age_p > age_q:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return (
        [t for i, t in enumerate(tracks_a) if i not in dup_a],
        [t for i, t in enumerate(tracks_b) if i not in dup_b],
    )


class ByteTracker:
    """Online multi-object tracker.

    One instance handles one sequence. Construct a fresh tracker per sequence so
    that identity numbering restarts, as the MOTChallenge format requires.
    """

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self.cfg = config or ByteTrackConfig()
        self.kalman_filter = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.max_time_lost = int(self.cfg.frame_rate / 30.0 * self.cfg.track_buffer)
        BaseTrack.reset_counter()

    # ------------------------------------------------------------------ update
    def update(
        self,
        boxes_tlwh: np.ndarray,
        scores: np.ndarray,
        features: np.ndarray | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> list[STrack]:
        """Consume one frame of detections and return the confirmed tracks.

        Parameters
        ----------
        boxes_tlwh:
            ``(N, 4)`` detections as ``(left, top, width, height)``.
        scores:
            ``(N,)`` detector confidences.
        features:
            Optional ``(N, D)`` appearance embeddings.
        img_size:
            ``(width, height)``; when given, output boxes are clipped to frame.
        """
        self.frame_id += 1
        cfg = self.cfg

        activated: list[STrack] = []
        refound: list[STrack] = []
        newly_lost: list[STrack] = []
        newly_removed: list[STrack] = []

        boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if features is not None:
            features = np.asarray(features, dtype=np.float64).reshape(len(scores), -1)

        # Split detections by confidence -- the core of the method.
        high_mask = scores >= cfg.track_high_thresh
        low_mask = (scores >= cfg.track_low_thresh) & (~high_mask)

        def build(mask: np.ndarray) -> list[STrack]:
            out = []
            for idx in np.flatnonzero(mask):
                feat = features[idx] if features is not None else None
                out.append(STrack(boxes_tlwh[idx], scores[idx], feature=feat))
            return out

        detections = build(high_mask)
        detections_low = build(low_mask)

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ---- Pass 1: high-score detections vs. active + lost tracks ---------
        strack_pool = _join(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool, self.kalman_filter)

        dists = (
            matching.giou_distance(strack_pool, detections)
            if cfg.use_giou
            else matching.iou_distance(strack_pool, detections)
        )
        if cfg.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if cfg.with_reid and features is not None and dists.size:
            emb = matching.embedding_distance(strack_pool, detections)
            # Appearance only ever *lowers* a cost, and only when it is
            # confident. A Re-ID model that has never seen this camera is not
            # trustworthy enough to veto a good geometric match.
            emb[emb > cfg.appearance_thresh] = 1.0
            dists = (1.0 - cfg.appearance_weight) * dists + cfg.appearance_weight * emb

        matches, u_track, u_detection = matching.linear_assignment(dists, cfg.match_thresh)
        for itrack, idet in matches:
            track, det = strack_pool[itrack], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)

        # ---- Pass 2: low-score detections vs. tracks still unmatched --------
        # Only previously-Tracked entries take part: reviving a long-lost
        # identity on the strength of a 0.2-confidence box is how ID swaps
        # happen in crowds.
        remaining = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists_low = matching.iou_distance(remaining, detections_low)
        matches_low, u_track_low, _ = matching.linear_assignment(
            dists_low, cfg.second_match_thresh
        )
        for itrack, idet in matches_low:
            track, det = remaining[itrack], detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refound.append(track)

        for i in u_track_low:
            track = remaining[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                newly_lost.append(track)

        # ---- Pass 3: leftover high-score detections vs. unconfirmed ---------
        leftover = [detections[i] for i in u_detection]
        dists_unc = matching.iou_distance(unconfirmed, leftover)
        if cfg.fuse_score:
            dists_unc = matching.fuse_score(dists_unc, leftover)
        matches_unc, u_unconfirmed, u_detection = matching.linear_assignment(
            dists_unc, cfg.unconfirmed_match_thresh
        )
        for itrack, idet in matches_unc:
            unconfirmed[itrack].update(leftover[idet], self.frame_id)
            activated.append(unconfirmed[itrack])
        for i in u_unconfirmed:
            unconfirmed[i].mark_removed()
            newly_removed.append(unconfirmed[i])

        # ---- Birth: only confident, well-shaped, leftover detections --------
        for i in u_detection:
            det = leftover[i]
            if det.score < cfg.new_track_thresh:
                continue
            w, h = det.tlwh[2], det.tlwh[3]
            if h <= 0 or w / h > cfg.aspect_ratio_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            activated.append(det)

        # ---- Death: retire tracks lost for too long -------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                newly_removed.append(track)

        # ---- Pool maintenance ----------------------------------------------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _join(self.tracked_stracks, activated)
        self.tracked_stracks = _join(self.tracked_stracks, refound)
        self.lost_stracks = _subtract(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(newly_lost)
        self.lost_stracks = _subtract(self.lost_stracks, newly_removed)
        self.removed_stracks.extend(newly_removed)
        self.tracked_stracks, self.lost_stracks = _remove_duplicates(
            self.tracked_stracks, self.lost_stracks, cfg.duplicate_iou_thresh
        )

        outputs = [t for t in self.tracked_stracks if t.is_activated]
        for track in outputs:
            track.out_tlwh = None
        if img_size is not None and outputs:
            width, height = img_size
            clipped = clip_tlwh(np.asarray([t.tlwh for t in outputs]), width, height)
            keep = []
            for track, box in zip(outputs, clipped):
                # Area is judged on the visible part: a track that has drifted
                # off-frame contributes nothing but false positives.
                if box[2] * box[3] >= self.cfg.min_box_area:
                    track.out_tlwh = box
                    keep.append(track)
            outputs = keep
        return outputs
