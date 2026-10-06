# Dense-Crowd Multi-Object Tracking

**CAIC Summer of Technology 2026 — Computer Vision Track, Final Project**

A multi-object pedestrian tracking pipeline for crowded scenes: YOLOv8 detection →
ByteTrack association → gap interpolation → HOTA / IDF1 / CLEAR evaluation.

The metrics are implemented from their source papers rather than taken from a
library, because the whole project depends on being able to measure its own
decisions. The trackers are implemented from the papers too, with SORT retained as
a baseline so every claim is a measured delta rather than an assertion.

---

## Results

Four MOT17 sequences (2,925 frames), scored with MOTChallenge preprocessing.
Leaderboard metric is `Score = (2 · IDF1 + HOTA) / 3`.

| sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | **Score** |
|---|---|---|---|---|---|---|---|
| `01` | 0.4615 | 0.5515 | 0.3893 | 0.5199 | 0.5868 | 40 | 0.5004 |
| `02` | 0.4157 | 0.3732 | 0.4676 | 0.5310 | 0.4333 | 76 | 0.4926 |
| `03` | 0.5063 | 0.4938 | 0.5233 | 0.5849 | 0.4884 | 67 | 0.5587 |
| `ref` | 0.3449 | 0.3035 | 0.3963 | 0.4182 | 0.3118 | 52 | 0.3937 |
| **OVERALL** | **0.4621** | 0.4410 | 0.4885 | **0.5418** | 0.4487 | 235 | **0.5152** |

Ablation on `ref`, identical detections throughout:

| variant | Score | Δ |
|---|---|---|
| SORT baseline (single threshold) | 0.3636 | — |
| ByteTrack (two-pass association) | 0.3856 | **+0.0220** |
| + score fusion | 0.3862 | +0.0006 → **rejected** |
| + gap interpolation | **0.3939** | **+0.0083** |

68 unit tests pass; all four submission files pass format validation.

---

## The three measurements that drove every decision

**1. The tracker is not the bottleneck.** Fed perfect boxes from a ground-truth
oracle, it scores **IDF1 0.985 with zero identity switches**.

**2. Confidence handling costs 24 IDF1 points on its own.** Keeping the boxes
perfect and changing *only* the confidence values to realistic ones drops IDF1
from 0.985 to 0.743. This is what justified ByteTrack and hard threshold tuning.

**3. Detection is the real ceiling.** Optimal per-frame assignment on the cached
detections recovers only **56%** of scored ground truth — because 39% of the
people being scored are less than 10% visible. Detector recall is 94% on
fully-visible people and 23% on barely-visible ones.

Full evidence in [docs/RESULTS.md](docs/RESULTS.md).

---

## Install

```bash
pip install -e ".[detect,dev]"     # core + ultralytics/torch + pytest
pip install -e .                   # tracking and metrics only (numpy + scipy)
```

Python ≥ 3.10. A GPU is optional — all results here were produced on 4 CPU cores.

## Quickstart

```bash
# 1. Cache detections once (~21 min on 4 CPU cores, ~1 min on a GPU)
python -m cv_mot detect --data "../Week 4/data" --out ../cache/yolov8s_1280 \
    --weights yolov8s.pt --imgsz 1280 --conf 0.05

# 2. Track -> submission files
python -m cv_mot track --data "../Week 4/data" --dets ../cache/yolov8s_1280 \
    --config configs/final.json --interpolate --interpolate-max-gap 20 --out outputs

# 3. Score against gt.txt
python -m cv_mot evaluate --data "../Week 4/data" --preds outputs

# 4. Check the format before submitting
python scripts/validate_submission.py --data "../Week 4/data" outputs/*.txt
```

Everything above, plus the ablation and tuning sweeps, is in
[`scripts/reproduce.sh`](scripts/reproduce.sh).

### Other commands

```bash
python -m cv_mot tune   --data "../Week 4/data" --seqs ref --dets ../cache/yolov8s_1280
python -m cv_mot ablate --data "../Week 4/data" --seqs ref --dets ../cache/yolov8s_1280
python -m cv_mot ablate --data "../Week 4/data" --seqs ref --oracle   # no inference
python -m pytest tests -q
```

To run on the real test sequences, point `--data` at them — no code change is
needed, and `01.txt`…`04.txt` are produced from whatever sequence folders exist.

---

## Repository map

```
aadibansal/
├── src/cv_mot/
│   ├── boxes.py          tlwh / xyxy / xyah conversions, vectorised IoU + GIoU
│   ├── kalman.py         constant-velocity filter, noise scaled by box height
│   ├── matching.py       cost matrices + Hungarian assignment
│   ├── track.py          STrack and the New/Tracked/Lost/Removed lifecycle
│   ├── bytetrack.py      the method: three-pass association
│   ├── sort.py           the baseline, kept for honest ablation
│   ├── detect.py         YOLO wrapper, ground-truth oracle, detection cache
│   ├── interpolate.py    gap filling as post-processing
│   ├── mot_io.py         seqinfo / gt.txt / submission format
│   ├── pipeline.py       detector -> tracker -> rows
│   ├── cli.py            detect | track | evaluate | tune | ablate
│   └── metrics/          HOTA, IDF1, CLEAR + MOTChallenge preprocessing
├── tests/                68 tests
├── docs/                 see below
├── notebooks/tracking_pipeline.ipynb
├── scripts/              reproduce.sh, validate_submission.py
├── configs/final.json    the tuned configuration
├── outputs/              submission files + JSON results
└── writeup.txt           the 200-word deliverable
```

The original course material (`Week 1`–`Week 4`, `Final Project`, root
`README.md`) is one level up and **unmodified**.

---

## Documentation

| document | contents |
|---|---|
| [docs/BUILD_LOG.md](docs/BUILD_LOG.md) | every construction step, in order, including the bugs |
| [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | module-by-module tour and the decision in each |
| [docs/METRICS.md](docs/METRICS.md) | HOTA / IDF1 / CLEAR from the papers, and the validation table |
| [docs/RESULTS.md](docs/RESULTS.md) | every measurement with the command that regenerates it |
| [docs/TUNING.md](docs/TUNING.md) | tuning protocol, sweeps, and honest limitations |
| [docs/LEARNINGS.md](docs/LEARNINGS.md) | findings, the mistakes, and what to try next |
| [docs/weeks/](docs/weeks/) | one document per course week, tying syllabus to code |

---

## Method, in one paragraph

SORT applies a single confidence threshold. In a crowd the dominant event is not a
missed person but a *partially occluded* person detected with low confidence —
SORT discards that box, the track dies, and the person gets a new identity on
reappearance. ByteTrack's observation is that a low-score box in roughly the right
place is strong evidence **if a track is already there**. Association therefore
runs in two passes: high-score detections against all active and lost tracks, then
low-score detections against only the tracks still unmatched — and those may never
*create* an identity. That asymmetry is the whole method: recall improves without
importing the false positives a globally-lowered threshold would bring.

---

## Honest limitations

- **The real test sequences are not in this repository.** The brief describes four
  sequences reaching 246 people/frame; only the Week 4 sequences (peak 109/frame)
  are present. All reported numbers are Week 4 data, and scores on the denser set
  will be lower.
- **Tuning used `ref` only**, so the `01`/`02`/`03` numbers are *near*-held-out
  rather than truly held-out. Leave-one-sequence-out would be the correct fix.
- **No Re-ID model was evaluated.** The appearance path is implemented and tested
  but unused, because measurement (1) above shows association is not the binding
  constraint on this data.
- **Differences below ~0.005 Score are not resolvable** by a single 600-frame
  sequence without repeats.

---

## References

- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection
  Box*, ECCV 2022 — [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- Bewley et al., *Simple Online and Realtime Tracking*, ICIP 2016 —
  [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)
- Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object
  Tracking*, IJCV 2021
- Ristani et al., *Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking*, ECCVW 2016
- Bernardin & Stiefelhagen, *Evaluating Multiple Object Tracking Performance: The
  CLEAR MOT Metrics*, 2008
