# Code walkthrough

A tour of `src/cv_mot/`, module by module: what each does, the one decision in it
worth defending, and what was rejected.

## Data flow

```
 Week 4/data/<seq>/img/*.jpg
          │
          ▼
 detect.py ──────────── YoloDetector            (ultralytics, person class only)
          │             GroundTruthDetector     (oracle, for diagnosis)
          │             CachedDetections        (.npz replay, for sweeps)
          │
          ▼  Detections(boxes_tlwh, scores)  — one per frame
          │
 bytetrack.py ───────── ByteTracker.update()
          │              ├── pass 1: high-score dets  ×  active + lost tracks
          │              ├── pass 2: low-score dets   ×  still-unmatched tracks
          │              └── pass 3: leftovers        ×  unconfirmed tracks
          │                    uses: kalman.py (predict/update)
          │                          matching.py (cost matrices + Hungarian)
          │                          track.py (STrack lifecycle)
          │                          boxes.py (IoU, conversions, clipping)
          ▼  list[STrack]  — confirmed identities this frame
          │
 pipeline.py ────────── run_sequence()  → (N, 10) MOT rows
          │
          ▼
 interpolate.py ─────── fill short gaps inside each identity
          │
          ▼
 mot_io.py ──────────── write_mot_file()  → outputs/<seq>.txt
          │
          ▼
 metrics/ ───────────── evaluate()  → HOTA, IDF1, CLEAR, Score
```

---

## `boxes.py` — three conventions, named once

Three box formats appear and confusing them is the most common silent failure in
this problem:

| name | meaning | used by |
|---|---|---|
| `tlwh` | `(left, top, width, height)` | the MOT file format |
| `xyxy` | `(left, top, right, bottom)` | Ultralytics YOLO output |
| `xyah` | `(centre_x, centre_y, width/height, height)` | the Kalman state |

**Decision: conversions are named functions with round-trip tests, never written
inline.** YOLO emits `xyxy`; the submission wants `tlwh`. A file written in `xyxy`
parses perfectly and scores near zero — no exception, no warning. Centralising the
conversions means the mistake can only be made in one place, and
`test_boxes.py` asserts every round trip.

`iou_matrix` is fully vectorised and returns correctly-shaped empty arrays for
empty inputs, which keeps the association code free of first-frame special cases.
`clip_tlwh` clamps to the image while keeping width and height non-negative.

`giou_matrix` is included but unused by default: GIoU keeps a usable gradient
between boxes that do not overlap at all, which could help re-acquire a track that
has coasted clear of its true position. It measured no better than IoU here, so it
stayed off and stayed documented.

---

## `kalman.py` — constant velocity, noise scaled by height

State is `(cx, cy, a, h, vx, vy, va, vh)`. Two non-obvious choices:

**Noise is proportional to box height.** A pedestrian 400 px tall crosses many
more pixels per frame than one 40 px tall; a single fixed covariance is wrong for
both. Scaling every standard deviation by `h` is what lets one parameter set work
across the full depth range of a frame simultaneously.

**Aspect ratio is treated as nearly static** — `aspect_std` is a small constant,
deliberately excluded from the height scaling. A walking person's width-to-height
ratio is roughly fixed while their pixel size is not. Letting aspect drift freely
makes boxes degenerate during long occlusions.

`multi_predict` vectorises prediction over all tracks, which matters once a dense
frame carries hundreds of them. `gating_distance(..., only_position=True)` is the
safer default in crowds: a partially-occluded detection has a badly wrong height,
and gating on height would reject the correct match.

**Rejected:** learning a motion model. Constant velocity is a prior that fits
walking people, needs no training data, and cannot overfit — the same reasoning
that makes convolution the right prior for images.

---

## `matching.py` — cost construction, then optimal assignment

The assignment is Hungarian (`scipy.optimize.linear_sum_assignment`). The
engineering is in the cost matrix.

**Decision: the cost ceiling is applied *after* the global solve, not before.**

```python
guard    = thresh + 1e-4
solvable = np.where(cost_matrix > guard, guard, cost_matrix)
rows, cols = linear_sum_assignment(solvable)
keep = cost_matrix[rows, cols] <= thresh
```

The ordering matters. The optimal global assignment can legitimately route
through a pair that is individually mediocre; removing over-threshold pairs first
would change which solution is found. `test_matching.py` contains the case that
makes this concrete — greedy matching scores 1.00 where optimal scores 0.35.

`fuse_score` multiplies IoU similarity by detector confidence; `fuse_motion`
blends appearance distance with Mahalanobis plausibility and hard-gates pairs
outside the 95% ellipse. Both are available; `fuse_score` measured as not worth
enabling here (see [TUNING.md](TUNING.md)).

---

## `track.py` — the lifecycle

```
New ──confirmed──▶ Tracked ──unmatched──▶ Lost ──buffer expires──▶ Removed
                      ▲                     │
                      └──── re-matched ─────┘
```

`Lost` is the state that makes occlusion survivable: a confirmed track that goes
unmatched keeps coasting on the motion model and stays eligible for
re-association. Without it, every occlusion ends an identity.

**Decision: lost tracks have their height-velocity zeroed before prediction.**

```python
if t.state != TrackState.Tracked:
    means[i][7] = 0.0
```

Letting `vh` keep integrating with no measurement makes a coasting box grow or
shrink without bound. After a dozen frames it no longer overlaps anything, so the
identity is unrecoverable *even when the person reappears exactly where
predicted*. Zeroing it keeps the box the right size while the centre coasts.

**Decision: appearance features use an exponential moving average**, not the
latest embedding. One half-occluded frame yields a contaminated embedding; an EMA
stops a single bad frame destroying an otherwise good identity model.

`STrack` instances double as detections before activation — the tracker builds one
per detection each frame, and only the matched or newly-initiated ones acquire an
identity. This is why `tlwh` falls back to `_tlwh` when `mean is None`.

### `tlwh` vs `output_tlwh` — a bug worth keeping visible

`tlwh` is derived from the Kalman mean. `out_tlwh` is the box actually written to
file, set when the tracker clips to the frame.

They are separate because an earlier version wrote the clipped box back into
`_tlwh`, which the `tlwh` property *ignores* once a track is active — so clipping
was a silent no-op for the whole pipeline. The submission validator caught it. See
[LEARNINGS.md](LEARNINGS.md).

Clipping must not feed back into the filter state either: a clipped box is not an
observation of where the person is, only of where they are visible.

---

## `bytetrack.py` — the method

Three association passes, explicitly labelled in `update()`. The core asymmetry:

- **Pass 1**: high-score detections against active **and** lost tracks.
- **Pass 2**: low-score detections against only the tracks still unmatched — and
  these may **never create** an identity.
- **Pass 3**: leftover high-score detections against not-yet-confirmed tracks.

Low-score boxes can *sustain* an identity but not *invent* one. That single
asymmetry is the whole method: recall improves without importing the false
positives a globally-lowered threshold would bring.

**Decision: pass 2 considers only tracks that were `Tracked`**, not `Lost` ones.
Reviving a long-lost identity on the strength of a 0.2-confidence box is precisely
how ID swaps happen in crowds.

`_remove_duplicates` drops near-identical tracks across the active and lost pools,
keeping the longer-lived one. Without it, a revived lost track that overlaps an
existing active track yields two IDs on one body for the rest of the sequence.

Every parameter in `ByteTrackConfig` carries a docstring explaining what it trades
off, and `fuse_score` documents *why* it defaults to off for crowds.

---

## `sort.py` — the baseline, kept deliberately

Single threshold, one association pass, no appearance. This is the Week 4
pipeline, retained so that every claim about ByteTrack is a measured delta on
identical detections rather than an assertion. It is not dead code — it is the
control.

---

## `detect.py` — three detectors behind one interface

| class | purpose |
|---|---|
| `YoloDetector` | the real pipeline; Ultralytics restricted to COCO class 0 |
| `GroundTruthDetector` | oracle with tunable recall, noise and confidence model |
| `CachedDetections` | replays saved `.npz` detections |

**Decision: the detector's `conf` floor defaults to 0.05**, far below any tracker
threshold. ByteTrack's second pass needs low-score boxes to exist; filtering them
here silently disables the method. This is called out in the docstring because it
is the easiest way to accidentally neutralise the whole approach.

**Decision: the oracle exists and its outputs are always labelled as oracle.**
Tracking error has two independent sources — the detector missing people, and the
association assigning them wrongly. Perfect detections isolate the second. This
produced the project's most useful single measurement: with perfect boxes the
tracker scores IDF1 0.985 with zero ID switches, so remaining error is not a
tracker bug (see [RESULTS.md](RESULTS.md)).

`CachedDetections` is what makes tuning affordable: detection is 0.43 s/frame,
association ~2 ms, so a 60-trial sweep goes from hours to seconds.

---

## `mot_io.py` — the format

Reads `seqinfo.ini`, reads `gt.txt`, writes the required 10 columns with `conf`
and the three world-coordinate columns set to `-1`.

**Decision: frame files are sorted by the integer in the filename**, not
lexicographically. Zero-padding is conventional but not guaranteed, and one
unpadded name would silently reorder a sequence — producing a plausible-looking
file with meaningless tracks.

Missing or empty files read as a `(0, 10)` array so callers can treat "no
detections" uniformly.

---

## `interpolate.py` — recovering what the tracker already knew

An online tracker coasts through an occlusion without emitting boxes: the identity
survives in memory but those frames are reported as misses. Since the track is
confirmed on both sides of the gap, the position between is known to good accuracy.

Two guards keep it honest: only gaps shorter than `max_gap` are filled (a straight
line is a poor model of a long walk), and interpolation never *creates* an
identity — it only fills between two observed endpoints of one. Worth +0.008
Score, with identity switches slightly down.

---

## `pipeline.py` — wiring

`run_sequence` constructs a **fresh tracker per sequence** so identity numbering
restarts at 1, as the submission format requires. Returns rows plus `RunStats`
(frames, detections, output rows, interpolated rows, fps).

Detections are consumed as an iterator, so a 1050-frame sequence never needs to
fit in memory.

---

## `metrics/` — the evaluator

| file | contents |
|---|---|
| `_data.py` | per-frame alignment, ID remapping, MOTChallenge distractor preprocessing |
| `hota.py` | HOTA / DetA / AssA / LocA over 19 α thresholds |
| `identity.py` | IDF1 / IDP / IDR |
| `clear.py` | MOTA / MOTP / IDSW / Frag |
| `evaluate.py` | orchestration and `(2·IDF1 + HOTA)/3` |

**Decision: sequences are pooled by summing counts, then recomputing ratios** —
not by averaging per-sequence scores, which would weight a 525-frame sequence the
same as a 1050-frame one.

**Decision: IDs are remapped to contiguous zero-based indices** before scoring,
because the metrics allocate dense `num_gt_ids × num_pred_ids` matrices and raw
file IDs are sparse.

See [METRICS.md](METRICS.md) for the formulas and the validation table.

---

## `cli.py` — five verbs

| command | purpose |
|---|---|
| `detect` | run YOLO once, cache boxes to `.npz` |
| `track` | produce submission files |
| `evaluate` | score against `gt.txt` |
| `tune` | grid search, replaying cached detections |
| `ablate` | measure each design decision as a delta |

`tune` and `ablate` are first-class commands rather than scratch scripts, because
the evidence in [RESULTS.md](RESULTS.md) and [TUNING.md](TUNING.md) has to be
regenerable.

---

## Test suite

68 tests across five files. The emphasis is on properties with
hand-derivable answers rather than on coverage:

| file | what it pins down |
|---|---|
| `test_boxes.py` | round trips, IoU against hand-computed values, empty and degenerate inputs |
| `test_kalman.py` | learns constant velocity, covariance stays symmetric positive-definite, predict inflates / update shrinks uncertainty, noise scales with height |
| `test_matching.py` | optimal beats greedy, threshold boundaries, empty and rectangular matrices |
| `test_metrics.py` | the validation table in [METRICS.md](METRICS.md) |
| `test_trackers.py` | identity stability, ByteTrack survives a confidence dip where SORT loses boxes, low-score boxes cannot create identities, clipping, crossing pedestrians |
| `test_io_and_interpolate.py` | exact 10-column format, sort order, interpolation bounds and idempotence |

```bash
python -m pytest tests -q     # 68 passed
```
