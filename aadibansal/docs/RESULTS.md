# Results

Every number here was produced by a command shown alongside it. All measurements
are on the four Week 4 sequences (MOT17 train: `01`=MOT17-09, `02`=MOT17-13,
`03`=MOT17-04, `ref`=MOT17-02), scored with the local implementation of
HOTA / IDF1 / CLEAR in `cv_mot/metrics/` under MOTChallenge preprocessing.

Hardware: **4 CPU cores, no GPU.** A GPU cuts the detection stages roughly 20×
and changes none of the scores.

**Leaderboard metric** is `Score = (2 · IDF1 + HOTA) / 3`.

---

## 1. Headline result

```bash
python -m cv_mot track --data "../Week 4/data" --dets ../cache/yolov8s_1280 \
    --config configs/final.json --interpolate --interpolate-max-gap 20 --out outputs
python -m cv_mot evaluate --data "../Week 4/data" --preds outputs
```

| sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | **Score** |
|---|---|---|---|---|---|---|---|
| `01` | 0.4615 | 0.5515 | 0.3893 | 0.5199 | 0.5868 | 40 | 0.5004 |
| `02` | 0.4157 | 0.3732 | 0.4676 | 0.5310 | 0.4333 | 76 | 0.4926 |
| `03` | 0.5063 | 0.4938 | 0.5233 | 0.5849 | 0.4884 | 67 | 0.5587 |
| `ref` | 0.3449 | 0.3035 | 0.3963 | 0.4182 | 0.3118 | 52 | 0.3937 |
| **OVERALL** | **0.4621** | 0.4410 | 0.4885 | **0.5418** | 0.4487 | 235 | **0.5152** |

Pooled by summing detection counts across sequences, as MOTChallenge reports an
overall score — not by averaging per-sequence scores, which would weight a
525-frame sequence the same as a 1050-frame one.

All four files pass format validation:

```
OK    01.txt  6072 rows, frames 1-525, 73 identities
OK    02.txt  6481 rows, frames 1-750, 125 identities
OK    03.txt  39393 rows, frames 1-1050, 141 identities
OK    ref.txt 9323 rows, frames 1-600, 95 identities
```

`ref` is the hardest sequence and the one tuning was done on — see
[TUNING.md](TUNING.md) on why that ordering matters.

---

## 2. Where the error actually is

This is the measurement that directed every subsequent decision.

An oracle detector (`cv_mot/detect.py::GroundTruthDetector`) synthesises
detections from `gt.txt`, so detector error can be dialled to zero and the
association logic measured in isolation.

| detections fed to the tracker | IDF1 | HOTA | DetA | AssA | IDSW |
|---|---|---|---|---|---|
| perfect boxes, **uniform** confidence 0.9 | **0.9853** | 0.9880 | 0.9974 | 0.9787 | **0** |
| perfect boxes, **realistic** confidence | 0.7425 | 0.7412 | 0.7218 | 0.7612 | 9 |

Two conclusions:

1. **The tracker is not the problem.** Given perfect boxes it scores IDF1 0.985
   with zero identity switches. The small residual gap is the three-frame
   confirmation delay at track birth.
2. **Confidence handling costs ~24 IDF1 points on its own.** The boxes are
   *identical* between those two rows; only the score attached to each one
   changes. Realistic confidence is modelled as `0.15 + 0.80 × visibility`,
   reproducing the real phenomenon that occluded people score low.

That is why the effort went into ByteTrack and threshold tuning rather than
appearance features.

---

## 3. Detector recall ceiling — the real limit

`DetA` on real detections tops out near 0.44, far below the oracle's 0.997. That
gap is the detector, and it is worth quantifying rather than assuming.

Matching cached YOLO detections to scored ground truth at IoU ≥ 0.5 with optimal
per-frame assignment (i.e. the best any tracker could possibly do), on `ref`:

| detector confidence floor | recall ceiling | matched / GT |
|---|---|---|
| ≥ 0.05 | **0.5618** | 10018 / 17833 |
| ≥ 0.10 | 0.4966 | 8855 / 17833 |
| ≥ 0.20 | 0.4233 | 7548 / 17833 |
| ≥ 0.30 | 0.3703 | 6603 / 17833 |
| ≥ 0.40 | 0.3199 | 5704 / 17833 |
| ≥ 0.50 | 0.2772 | 4944 / 17833 |
| ≥ 0.60 | 0.2370 | 4226 / 17833 |

**Even at a 0.05 floor the detector can only reach 56% recall.** No association
logic can recover a person the detector never proposed, so `ref`'s DetA of 0.30
sits against a hard ceiling of 0.56.

Broken down by how visible each annotated person actually is:

| ground-truth visibility | detector recall | GT rows |
|---|---|---|
| 0.0 – 0.1 | **0.234** | 7037 |
| 0.1 – 0.3 | 0.454 | 1956 |
| 0.3 – 0.5 | 0.618 | 1600 |
| 0.5 – 0.8 | 0.783 | 2093 |
| 0.8 – 1.0 | **0.943** | 5147 |

The detector is excellent on people it can see (94%) and poor on people it mostly
cannot (23%) — which is the expected and correct behaviour. The difficulty is that
**7037 of 17833 scored annotations on `ref` are less than 10% visible**, and the
benchmark counts every one of them. This is the single largest reason the absolute
scores here are what they are.

Reproduce: the script in [the notebook](../notebooks/tracking_pipeline.ipynb),
section 4, or the recall-ceiling snippet in [LEARNINGS.md](LEARNINGS.md).

---

## 4. Ablation — every decision as a measured delta

Same cached detections throughout, so differences are attributable to the
tracker alone.

### On real YOLO detections (`ref`)

```bash
python -m cv_mot ablate --data "../Week 4/data" --seqs ref \
    --dets ../cache/yolov8s_1280 --config configs/final.json
```

| variant | IDF1 | HOTA | DetA | AssA | IDSW | **Score** | Δ |
|---|---|---|---|---|---|---|---|
| SORT baseline (single threshold) | 0.3792 | 0.3322 | 0.2557 | 0.4344 | 50 | 0.3636 | — |
| ByteTrack (two-pass association) | 0.4100 | 0.3366 | 0.2926 | 0.3916 | 53 | 0.3856 | **+0.0220** |
| + score fusion | 0.4082 | 0.3423 | 0.2941 | 0.4032 | 60 | 0.3862 | +0.0006 |
| + gap interpolation | 0.4183 | 0.3451 | 0.3037 | 0.3967 | 52 | **0.3939** | **+0.0083** |

**Reading these honestly:**

- **ByteTrack's gain is real and comes from recall**: DetA rises 0.2557 → 0.2926
  (+14% relative). Note `AssA` *falls* (0.4344 → 0.3916) — recovering more
  low-confidence boxes also creates more association opportunities to get wrong.
  The leaderboard metric still prefers it decisively because IDF1 rewards the
  recovered identities more than it punishes the extra errors.
- **Score fusion is not worth it**: +0.0006 Score while identity switches rise
  53 → 60. Left **off** in the final configuration. This matches the ByteTrack
  authors' own choice to disable it for MOT20-style crowds: in dense scenes real
  pedestrians routinely score low, so multiplying similarity by confidence
  suppresses exactly the detections the method exists to keep.
- **Interpolation is a genuine free gain**: +0.0083 Score, DetA up, and IDSW
  slightly *down*. It only fills gaps between two observed endpoints of one
  identity, so it cannot invent tracks.

### On oracle detections (`ref`) — association isolated

```bash
python -m cv_mot ablate --data "../Week 4/data" --seqs ref --oracle
```

| variant | IDF1 | HOTA | DetA | AssA | IDSW | **Score** |
|---|---|---|---|---|---|---|
| SORT baseline | 0.4926 | 0.4415 | 0.4138 | 0.4710 | 45 | 0.4755 |
| ByteTrack | 0.5158 | 0.4606 | 0.4375 | 0.4849 | 36 | 0.4974 |
| + score fusion | 0.5396 | 0.4734 | 0.4383 | 0.5113 | 33 | **0.5175** |
| + gap interpolation | 0.5228 | 0.4678 | 0.4454 | 0.4913 | 36 | 0.5044 |

ByteTrack beats SORT on **every** metric here, including a drop in identity
switches from 45 to 36.

**Score fusion helps on the oracle but not on real detections.** The oracle's
confidences are a clean monotonic function of visibility, so multiplying by them
is informative. Real YOLO confidence is noisier and the signal does not survive.
This is a good illustration of why the oracle is a diagnostic and not a substitute
for the real thing — and why the final configuration follows the real-detection
measurement.

---

## 5. Threshold sensitivity

On oracle detections, sweeping the confidence threshold with everything else
fixed (see [TUNING.md](TUNING.md) for the real-detection sweep):

| `track_high_thresh` | IDF1 | DetA | AssA | **Score** |
|---|---|---|---|---|
| 0.60 | 0.7191 | 0.6651 | 0.8123 | 0.7244 |
| 0.50 | 0.7425 | 0.7218 | 0.7612 | 0.7421 |
| 0.40 | 0.8354 | 0.7973 | 0.8432 | 0.8302 |
| 0.30 | 0.8901 | 0.8549 | 0.8724 | 0.8813 |
| **0.25** | 0.9256 | 0.8918 | 0.9106 | **0.9174** |
| 0.20 | 0.9172 | 0.9045 | 0.9030 | 0.9127 |
| 0.16 | 0.9145 | 0.9267 | 0.8950 | 0.9132 |

**+19.3 Score points** from a single parameter. `DetA` rises monotonically as the
threshold falls, but `AssA` peaks and then declines — past the optimum the extra
boxes cause mis-associations faster than they add detections. The best score sits
where those curves cross.

---

## 6. Runtime

Detector throughput, 4 CPU cores, measured over 8 frames of 1920×1080:

| weights | `imgsz` | s/frame | dets/frame | all 2925 frames |
|---|---|---|---|---|
| `yolov8n.pt` | 640 | 0.06 | 37.9 | ~3 min |
| `yolov8n.pt` | 1280 | 0.19 | 46.4 | ~9 min |
| `yolov8s.pt` | 1280 | **0.43** | 49.8 | **~21 min** |

`yolov8s.pt @ 1280` was used for all reported results.

Cached detection counts at `conf=0.05`:

| sequence | detections | per frame |
|---|---|---|
| `01` | 9,676 | 18.4 |
| `02` | 18,302 | 24.4 |
| `03` | 68,291 | 65.0 |
| `ref` | 25,683 | 42.8 |

Tracker throughput, excluding detection (detections replayed from cache):

| sequence | fps |
|---|---|
| `01` | 579 |
| `02` | 666 |
| `03` | 212 |
| `ref` | 433 |

The tracker is comfortably real-time; detection is the entire cost. This is why
detections are cached once and every tuning trial replays them — it reduces a
parameter sweep from hours of inference to seconds of association.

---

## 7. What is not measured here

Stated plainly so the numbers are not over-read:

- **The real test sequences are absent.** The brief describes four test sequences
  reaching 246 people/frame; this repository only contains the Week 4 sequences
  (peak 109/frame). Everything above is Week 4 data. Scores on the denser test
  set will be lower.
- **Tuning used `ref` only.** `01`, `02` and `03` scores are therefore
  near-honest held-out numbers, but the configuration was selected on `ref`, so
  there is some selection bias. See [TUNING.md](TUNING.md).
- **No Re-ID model was run.** The appearance path is implemented and unit-tested
  but no Re-ID weights were evaluated, because section 2 shows association is not
  the binding constraint on this data.
- **Frame-clipping has almost no score effect** (0.5154 → 0.5152 when enabled) but
  is required for a valid submission. It was enabled after the validator caught
  that the original clipping code was a silent no-op — see
  [LEARNINGS.md](LEARNINGS.md).
