# Week 4 — YOLO detection, Kalman filtering, Hungarian matching and SORT

This is where the final project's code actually begins.

## What the original repo contains

`Week 4/` (618 MB) — four MOT-format sequences, 2,925 frames total:

| Sequence | Frames | Resolution | Scored GT rows | Peak pedestrians/frame | Peak annotations/frame | Identities |
|---|---|---|---|---|---|---|
| `01` | 525 | 1920×1080 | 5,257 | 13 | 21 | 43 |
| `02` | 750 | 1920×1080 | 11,450 | 32 | 50 | 178 |
| `03` | 1050 | 1920×1080 | 47,557 | 52 | 109 | 141 |
| `ref` | 600 | 1920×1080 | 17,833 | 36 | 56 | 74 |

The two peak columns differ because `gt.txt` annotates vehicles and occluders as
well as people. **Pedestrians/frame is the figure to compare against the final
project's "246 pedestrians per frame"**; annotations/frame counts every row.

Each folder holds `img/` (individual `.jpg` frames), `seqinfo.ini` (fps,
resolution, frame count) and `gt.txt`.

### Two findings about this data that the readme does not state

**1. These are MOT17 training sequences.** The frame counts and resolutions match
exactly: `01`=MOT17-09 (525), `02`=MOT17-13 (750), `03`=MOT17-04 (1050),
`ref`=MOT17-02 (600). This matters a great deal — it means the official
MOTChallenge evaluation protocol applies verbatim, and published ByteTrack
numbers are a sanity reference for what "good" looks like.

**2. `gt.txt` does not match its documented format.** The readme states the
`class` column is "always 1 (pedestrian)". It is not. Measured:

```
01   classes {1: 5257, 8: 1575, 9: 1050, 12: 948}
02   classes {1: 11450, 3: 4484, 10: 2542, 11: 680, 4: 103, 8: 4}
03   classes {1: 47557, 10: 23100, 11: 18900, 4: 11550, 7: 4798, 3: 1050, 5: 1050}
ref  classes {1: 17833, 7: 5271, 4: 1559, 9: 1781, 8: 1200, 2: 1549}
```

Taking the readme at face value and scoring against all rows produces badly wrong
numbers. The handling this forced is documented in [../METRICS.md](../METRICS.md).

**3. All four sequences have `gt.txt`, not just `ref`.** The readme describes `ref`
as the sequence for self-evaluation, but `01`, `02` and `03` are annotated too.
That makes honest local evaluation on every sequence possible, which is what
[../RESULTS.md](../RESULTS.md) reports.

## The concepts

**From classification to detection.** A classifier emits one label per image. A
detector emits a variable-length set of (box, class, score) per image. The
variable length is the hard part — it is why detection needs non-maximum
suppression and why its metric (mAP) is more involved than accuracy.

**IoU** — intersection over union, the overlap measure underpinning everything
downstream. Implemented vectorised in `cv_mot/boxes.py::iou_matrix`.

**NMS** — a detector fires multiple times on one object; NMS keeps the highest
scoring box and suppresses overlapping ones. Handled inside Ultralytics via the
`iou` parameter.

**The Kalman filter** answers "where will this object be next frame?" It
maintains a mean and covariance over state, alternating *predict* (advance by the
motion model, uncertainty grows) and *update* (fold in a measurement,
uncertainty shrinks). Implemented in `cv_mot/kalman.py` over state
`(cx, cy, aspect, height, vx, vy, va, vh)`.

**The Hungarian algorithm** solves the assignment problem optimally: given a cost
matrix of tracks × detections, find the one-to-one pairing of minimum total cost.
Greedy matching is not equivalent — `tests/test_matching.py` contains a case
where greedy scores 1.00 and optimal scores 0.35. Implemented in
`cv_mot/matching.py::linear_assignment`.

**SORT's lifecycle** — birth (a new detection starts a tentative track),
confirmation (survives `min_hits` frames), coasting (unmatched, predicted
forward), death (unmatched for `max_age` frames). Implemented in `cv_mot/sort.py`.

## Where this lands in the final project

| Week 4 concept | Module |
|---|---|
| YOLOv8 inference | `cv_mot/detect.py::YoloDetector` |
| IoU | `cv_mot/boxes.py::iou_matrix` |
| Kalman filter | `cv_mot/kalman.py::KalmanFilterXYAH` |
| Hungarian matching | `cv_mot/matching.py::linear_assignment` |
| SORT | `cv_mot/sort.py::SortTracker` |
| Track lifecycle | `cv_mot/track.py::TrackState` |
| MOT file format | `cv_mot/mot_io.py` |

SORT is retained in the final project **as a baseline**, not as dead code. Every
claim about ByteTrack's benefit in [../RESULTS.md](../RESULTS.md) is a measured
delta against it on identical detections.

## The coordinate-format trap

The readme warns about it and it is worth restating because it fails *silently*:
YOLO emits `xyxy`; the submission format wants `tlwh`. A file written in `xyxy`
parses perfectly and scores near zero. Two defences were built:

- `cv_mot/boxes.py` keeps the three conventions (`tlwh`, `xyxy`, `xyah`) as named
  functions with round-trip tests, so no conversion is written inline.
- `scripts/validate_submission.py` flags it statistically — if box width
  correlates with the left edge at r > 0.9, the file is almost certainly `xyxy`.

## What was learned

Reading the data beats reading the documentation about the data. Both substantive
findings here — that these are MOT17 sequences, and that `gt.txt` contradicts the
readme's description of it — came from a few minutes of counting columns, and both
changed the implementation materially.
