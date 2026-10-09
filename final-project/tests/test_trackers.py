"""Tracker behaviour, stated as the properties a submission depends on."""

import numpy as np
import pytest

from cv_mot.bytetrack import ByteTrackConfig, ByteTracker
from cv_mot.sort import SortConfig, SortTracker


def walk(n_frames=30, n_people=2, speed=5.0):
    """Generate people walking right at constant speed, all inside a 1080p frame.

    Lanes are laid out on a grid so that the generator stays valid for large
    ``n_people``; a box placed outside the frame would be clipped away by the
    tracker and the test would be measuring the generator, not the tracker.
    """
    per_row = 6
    for f in range(1, n_frames + 1):
        boxes = []
        for i in range(n_people):
            col, row = i % per_row, i // per_row
            boxes.append([60.0 + speed * f + 300.0 * col, 60.0 + 150.0 * row, 50.0, 120.0])
        yield f, np.array(boxes, dtype=float)


def run(tracker, frames, scores_fn=None):
    history = []
    for f, boxes in frames:
        scores = scores_fn(f, boxes) if scores_fn else np.full(len(boxes), 0.9)
        out = tracker.update(boxes, scores, img_size=(1920, 1080))
        history.append({t.track_id: tuple(np.round(t.tlwh, 1)) for t in out})
    return history


def test_assigns_stable_ids_to_clean_tracks():
    hist = run(ByteTracker(), walk())
    ids = {i for frame in hist for i in frame}
    assert ids == {1, 2}
    # Every frame after confirmation reports both people.
    assert all(len(frame) == 2 for frame in hist[3:])


def test_ids_are_one_based():
    hist = run(ByteTracker(), walk(n_people=3))
    assert min(i for frame in hist for i in frame) == 1


def test_low_confidence_dip_does_not_lose_the_identity():
    """The property ByteTrack exists for."""

    def scores(f, boxes):
        s = np.full(len(boxes), 0.9)
        if 10 <= f <= 16:
            s[1] = 0.25  # occluded: detector confidence collapses
        return s

    hist = run(ByteTracker(), walk(n_frames=30), scores)
    assert {i for frame in hist for i in frame} == {1, 2}
    # The second person is reported in every frame of the dip.
    assert all(2 in frame for frame in hist[10:17])


def test_sort_loses_detections_where_bytetrack_does_not():
    """The measurable difference between the baseline and the method."""

    def scores(f, boxes):
        s = np.full(len(boxes), 0.9)
        if 10 <= f <= 16:
            s[1] = 0.25
        return s

    byte_hist = run(ByteTracker(), walk(n_frames=30), scores)
    sort_hist = run(SortTracker(SortConfig(det_thresh=0.5)), walk(n_frames=30), scores)
    byte_reported = sum(len(f) for f in byte_hist)
    sort_reported = sum(len(f) for f in sort_hist)
    assert byte_reported > sort_reported


def test_low_score_detections_cannot_create_identities():
    """Second-pass boxes sustain tracks but must never start one."""
    tracker = ByteTracker(ByteTrackConfig(track_high_thresh=0.5, new_track_thresh=0.6))
    for f in range(1, 11):
        out = tracker.update(np.array([[100.0 + 5 * f, 200.0, 50.0, 120.0]]), np.array([0.2]), img_size=(1920, 1080))
    assert out == []


def test_single_frame_false_positive_is_not_emitted():
    """A one-frame blip mid-sequence must never reach the output.

    Tracks born on frame 1 are exempt and *are* emitted immediately: there is no
    earlier frame that could confirm them, so withholding them would discard
    real people at sequence start.
    """
    tracker = ByteTracker()
    for f, boxes in walk(n_frames=5, n_people=1):
        tracker.update(boxes, np.array([0.9]), img_size=(1920, 1080))

    before = {t.track_id for t in tracker.tracked_stracks}
    out = tracker.update(
        np.array([[60.0 + 5 * 6, 60.0, 50.0, 120.0], [1500.0, 900.0, 30.0, 70.0]]),
        np.array([0.9, 0.95]),
        img_size=(1920, 1080),
    )
    assert {t.track_id for t in out} == before  # the blip is withheld


def test_track_dies_after_buffer_expires():
    cfg = ByteTrackConfig(track_buffer=5, frame_rate=30)
    tracker = ByteTracker(cfg)
    for f, boxes in walk(n_frames=10, n_people=1):
        tracker.update(boxes, np.array([0.9]), img_size=(1920, 1080))
    for _ in range(20):
        tracker.update(np.zeros((0, 4)), np.zeros(0), img_size=(1920, 1080))
    assert all(t.state.name == "Removed" for t in tracker.lost_stracks + tracker.tracked_stracks) or not tracker.lost_stracks


def test_reappearance_within_buffer_keeps_the_same_id():
    tracker = ByteTracker(ByteTrackConfig(track_buffer=60))
    for f in range(1, 11):
        tracker.update(np.array([[100.0 + 5 * f, 200.0, 50.0, 120.0]]), np.array([0.9]), img_size=(1920, 1080))
    for _ in range(5):
        tracker.update(np.zeros((0, 4)), np.zeros(0), img_size=(1920, 1080))
    # Reappears where constant-velocity motion predicts.
    out = tracker.update(np.array([[175.0, 200.0, 50.0, 120.0]]), np.array([0.9]), img_size=(1920, 1080))
    assert [t.track_id for t in out] == [1]


def test_empty_frames_are_handled():
    tracker = ByteTracker()
    for _ in range(5):
        assert tracker.update(np.zeros((0, 4)), np.zeros(0), img_size=(640, 480)) == []


def test_identity_counter_resets_between_sequences():
    t1 = ByteTracker()
    run(t1, walk(n_people=2))
    t2 = ByteTracker()
    hist = run(t2, walk(n_people=2))
    assert min(i for frame in hist for i in frame) == 1


def test_output_boxes_are_clipped_to_the_frame():
    """A track walking off the edge must still report an in-frame box.

    Asserts a track is actually emitted first -- otherwise an empty output would
    satisfy the bounds check vacuously, which is how a no-op clip hid here once.
    """
    tracker = ByteTracker()
    out = []
    for f in range(1, 8):
        # Walks right until it straddles the right edge of a 640x480 frame.
        out = tracker.update(
            np.array([[480.0 + 10 * f, 300.0, 120.0, 150.0]]), np.array([0.95]), img_size=(640, 480)
        )
    assert len(out) == 1, "expected a live track to clip"
    left, top, w, h = out[0].output_tlwh
    assert left >= 0 and top >= 0
    assert left + w <= 640 + 1e-6 and top + h <= 480 + 1e-6
    assert w > 0 and h > 0
    # The unclipped filter state should genuinely exceed the frame, so the
    # assertions above are testing clipping rather than a box that already fit.
    assert out[0].tlwh[0] + out[0].tlwh[2] > 640


def test_crossing_people_keep_distinct_ids():
    """Two people walking through each other must not merge."""
    tracker = ByteTracker()
    ids_seen = set()
    for f in range(1, 41):
        a = 100.0 + 15 * f
        b = 700.0 - 15 * f
        boxes = np.array([[a, 200.0, 50.0, 120.0], [b, 205.0, 50.0, 120.0]])
        out = tracker.update(boxes, np.full(2, 0.9), img_size=(1920, 1080))
        ids_seen |= {t.track_id for t in out}
    assert len(ids_seen) == 2


@pytest.mark.parametrize("n_people", [1, 5, 30])
def test_scales_to_many_simultaneous_tracks(n_people):
    tracker = ByteTracker()
    hist = run(tracker, walk(n_frames=15, n_people=n_people))
    assert len(hist[-1]) == n_people
