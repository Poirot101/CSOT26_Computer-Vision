# Tuning methodology

## Protocol

1. **Build the evaluator first.** Without a local HOTA/IDF1 implementation,
   tuning optimises nothing. See [METRICS.md](METRICS.md).
2. **Cache detections once.** Detection costs 0.43 s/frame on 4 CPU cores;
   association costs ~2 ms. Caching turns a 60-trial sweep from 21 hours of
   repeated inference into about two minutes.
3. **Tune on `ref` only**, leaving `01`, `02`, `03` as near-held-out.
4. **Optimise the leaderboard metric**, `(2·IDF1 + HOTA)/3` — not MOTA, which
   rewards different behaviour.
5. **Ablate afterwards** to confirm each component still earns its place at the
   tuned operating point.

```bash
python -m cv_mot detect --data "../Week 4/data" --out ../cache/yolov8s_1280 \
    --weights yolov8s.pt --imgsz 1280 --conf 0.05
python -m cv_mot tune --data "../Week 4/data" --seqs ref \
    --dets ../cache/yolov8s_1280 --grid-high 0.2 0.3 0.4 0.5 0.6 --grid-new 0.3 0.5 0.7
```

---

## The detector's confidence floor is not a tuning parameter

`--conf 0.05` at detection time is set deliberately **below** every tracker
threshold and is never raised. ByteTrack's second association pass needs
low-score boxes to exist; filtering them at the detector silently disables the
method and no amount of tracker tuning recovers it.

This is the easiest way to accidentally neutralise ByteTrack, so it is called out
in the `YoloDetector` docstring as well.

---

## Grid search: the two confidence thresholds

60 configurations over `track_high_thresh` × `new_track_thresh`, on real
detections. Score shown; best per row in bold.

| `high` \ `new` | 0.3 | 0.5 | 0.7 |
|---|---|---|---|
| 0.2 | 0.3733 | 0.3680 | 0.3292 |
| 0.3 | 0.3686 | **0.3686** | 0.3179 |
| **0.4** | **0.3856** | 0.3788 | 0.3255 |
| 0.5 | 0.3735 | 0.3735 | 0.3320 |
| 0.6 | 0.3569 | 0.3569 | 0.3231 |

**Optimum `track_high_thresh = 0.4`.** The curve is unimodal with a clear peak,
not a plateau — 0.4 beats both 0.3 (−0.017) and 0.5 (−0.012).

### Why the optimum is lower than the default

ByteTrack's reference default is 0.5 (and many pipelines use 0.6). Measured here
the optimum is 0.4, and the direction is explained by section 3 of
[RESULTS.md](RESULTS.md): detector recall falls from 0.32 at a 0.4 floor to 0.28
at 0.5. On data where 39% of scored annotations are under 10% visible, every
recoverable box matters more than the false positives it brings.

The brief predicted this ("the optimal threshold for dense crowds is almost always
lower than the default"). The value of measuring rather than accepting that hint
is knowing *where* to stop: pushing to 0.2 makes things **worse**, not better.

### An artefact worth understanding, not copying

The raw grid's best cell was `new_track_thresh = 0.3` with `high = 0.4`, and
0.3 scores *identically* to 0.4:

| `new_track_thresh` (at `high=0.4`) | IDF1 | Score |
|---|---|---|
| 0.30 | 0.4100 | 0.3856 |
| 0.40 | 0.4100 | 0.3856 |
| 0.45 | 0.4073 | 0.3840 |
| 0.50 | 0.4008 | 0.3788 |

Identical because track birth only ever considers detections that already passed
`track_high_thresh = 0.4`, so any `new_track_thresh ≤ 0.4` rejects nothing. The
two settings are the same tracker.

The config therefore records **0.4, not 0.3** — the value that states the actual
behaviour ("every high-score detection may start a track") rather than an
arbitrary number below a floor. Reporting 0.3 would have implied a tuned
threshold where there is none.

The real finding is that **being permissive about track birth helps here**, which
is the opposite of the usual advice. With recall this limited, a missed person
costs more than a spurious track.

---

## Second-pass floor: `track_low_thresh`

| `track_low_thresh` | IDF1 | DetA | AssA | **Score** |
|---|---|---|---|---|
| 0.05 | 0.3925 | 0.2993 | 0.3803 | 0.3733 |
| **0.10** | 0.4100 | 0.2926 | 0.3916 | **0.3856** |
| 0.20 | 0.3902 | 0.2826 | 0.3873 | 0.3697 |
| 0.30 | 0.3587 | 0.2749 | 0.3947 | 0.3483 |

0.10 is optimal and the shape is informative. Dropping to 0.05 *raises* DetA
(0.2926 → 0.2993) but *lowers* IDF1 — the extra boxes below 0.1 are mostly
background, and attaching them to tracks corrupts identities faster than it adds
coverage. Kept at the reference default of 0.1, now with evidence.

---

## Memory: `track_buffer`

Frames a lost track is retained and remains eligible for re-association.

| `track_buffer` | IDF1 | AssA | IDSW | **Score** |
|---|---|---|---|---|
| 15 | 0.3976 | 0.3892 | 65 | 0.3767 |
| **30** | 0.4100 | 0.3916 | 53 | **0.3856** |
| 60 | 0.3968 | 0.3935 | 48 | 0.3777 |
| 90 | 0.3981 | 0.3953 | 48 | 0.3788 |

A genuine trade-off, and the metrics disagree:

- **Identity switches keep falling** with a longer buffer (65 → 48). Longer memory
  really does recover more occlusions.
- **IDF1 peaks at 30.** Beyond that, stale tracks get revived onto the *wrong*
  person — a switch that IDSW does not count, because from CLEAR's perspective the
  identity was consistent, while IDF1's global mapping sees the corruption.

`IDSW` alone would have selected 60 or 90. Optimising the actual leaderboard
metric selects 30. This is a concrete case where tuning against the wrong metric
would have cost real points.

---

## Post-processing: interpolation gap

| `interpolate_max_gap` | rows | IDF1 | DetA | **Score** |
|---|---|---|---|---|
| 0 (off) | 8739 | 0.4100 | 0.2926 | 0.3856 |
| 5 | 8811 | 0.4108 | 0.2937 | 0.3863 |
| 10 | 8937 | 0.4056 | 0.2957 | 0.3833 |
| **20** | 9323 | 0.4183 | 0.3037 | **0.3939** |
| 40 | 9722 | 0.4149 | 0.3103 | 0.3928 |

20 frames (two thirds of a second at 30 fps) is best. The non-monotonic dip at 10
is small and within the noise this single-sequence protocol can resolve — the
honest statement is that anything in 5–40 helps and 20 is the measured best, not
that 10 is specifically bad.

Beyond 40 frames DetA still climbs while Score falls: a straight line stops being
a good model of where a person walked, so the filled boxes miss and become false
positives.

---

## Final configuration

`configs/final.json`:

```json
{
  "track_high_thresh": 0.4,
  "track_low_thresh": 0.1,
  "new_track_thresh": 0.4,
  "match_thresh": 0.8,
  "second_match_thresh": 0.5,
  "unconfirmed_match_thresh": 0.7,
  "track_buffer": 30,
  "fuse_score": false,
  "min_box_area": 10.0,
  "aspect_ratio_thresh": 10.0
}
```

| parameter | value | justification |
|---|---|---|
| `track_high_thresh` | 0.4 | grid optimum; below the 0.5 default because recall is the binding constraint |
| `track_low_thresh` | 0.1 | swept; 0.05 adds background, 0.2 discards usable boxes |
| `new_track_thresh` | 0.4 | equals `high`: every high-score detection may start a track |
| `match_thresh` | 0.8 | reference default (IoU ≥ 0.2); not sensitive in the ranges tried |
| `second_match_thresh` | 0.5 | stricter than the first pass — low-score boxes are weaker evidence, so geometry must agree better |
| `unconfirmed_match_thresh` | 0.7 | reference default |
| `track_buffer` | 30 | swept; longer lowers IDSW but costs IDF1 via wrong revivals |
| `fuse_score` | `false` | measured +0.0006 Score and +7 IDSW on real detections — not worth it |
| `min_box_area` | 10.0 | drops tracks that have coasted off-frame |
| `aspect_ratio_thresh` | 10.0 | rejects absurdly wide boxes, which are merged detections |

---

## Honest limitations of this tuning

- **Single-sequence selection.** The configuration was chosen on `ref` alone, so
  the `01`/`02`/`03` numbers in [RESULTS.md](RESULTS.md) are *near*-held-out, not
  truly held-out. `ref` was chosen because it is the hardest sequence (59% of its
  scored people are under half visible), which should make the configuration
  conservative rather than over-fitted — but this is an argument, not evidence.
  A leave-one-sequence-out protocol over all four would be the correct fix and was
  not run.
- **Coordinate-wise, not joint.** After the 2-D grid, remaining parameters were
  swept one at a time from the grid optimum. Interactions between, say,
  `track_buffer` and `track_low_thresh` are unexplored.
- **Differences under ~0.005 Score are not resolvable** with one 600-frame
  sequence and no repeats. The ByteTrack-vs-SORT gap (+0.022) and the
  interpolation gain (+0.008) are comfortably above that; the score-fusion result
  (+0.0006) is not, which is a second reason to leave it off rather than claim it
  hurts.
