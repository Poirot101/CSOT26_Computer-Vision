# Metrics

Implemented from the source papers in `cv_mot/metrics/`, with no dependency on
TrackEval. The leaderboard metric is

```
Score = (2 · IDF1 + HOTA) / 3
```

so IDF1 carries two thirds of the weight: **association quality matters twice as
much as the balanced HOTA**, and tuning should never be driven by MOTA.

## Why the evaluator was built before the tracker

Every tuning decision is a comparison between two numbers. If those numbers come
from a wrong evaluator, the tuning optimises the wrong thing and the error is
invisible — the sweep still produces a confident-looking optimum. So `metrics/`
was written and validated first, against cases whose answers can be derived by
hand.

---

## IDF1 — identity-based F1

> Ristani et al., *Performance Measures and a Data Set for Multi-Target,
> Multi-Camera Tracking*, ECCVW 2016.

IDF1 does not ask "how many boxes were right". It asks "how consistently did one
identity stay on one person". Mechanically:

1. For every (ground-truth ID, predicted ID) pair, count the frames in which they
   coincide with IoU ≥ 0.5. Call that matrix `potential`.
2. Choose **one global one-to-one mapping** between GT IDs and predicted IDs for
   the entire sequence, minimising total identity error.
3. `IDTP` = matched detections under that mapping;
   `IDFN` = total GT − IDTP; `IDFP` = total predictions − IDTP.
4. `IDF1 = IDTP / (IDTP + ½·IDFN + ½·IDFP)`.

The consequence worth internalising: **a tracker that splits one person into two
IDs halfway through loses roughly half that person's detections to both IDFN and
IDFP**, because only one of the two predicted IDs can be mapped. That is exactly
why the leaderboard weights it double — it is the metric that punishes
fragmentation.

### Implementation detail that changes the number

The assignment minimises `IDFN + IDFP` over the mapping, rather than maximising
`IDTP` directly:

```python
fn_mat = gt_id_count  - potential
fp_mat = pred_id_count - potential
rows, cols = linear_sum_assignment(fn_mat + fp_mat)
```

These are **not** equivalent when the ID counts are unbalanced, because each
candidate pair carries different `gt_id_count` and `pred_id_count`. The
`FN + FP` form is what the MOTChallenge reference implementation uses, so it is
what is implemented here.

---

## HOTA — Higher Order Tracking Accuracy

> Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object
> Tracking*, IJCV 2021.

HOTA exists because MOTA and IDF1 disagree about what tracking is for: MOTA is
dominated by detection counts, IDF1 by identity consistency. HOTA separates them
and recombines them geometrically:

```
HOTA_α = sqrt(DetA_α · AssA_α)
HOTA   = mean over α ∈ {0.05, 0.10, ..., 0.95} of HOTA_α
```

Averaging over 19 localisation thresholds means the score does not hinge on one
arbitrary IoU cutoff.

- `DetA_α = TP / (TP + FN + FP)` — pure detection quality.
- `AssA_α = (1/TP) · Σ_c TPA(c) / (TPA(c) + FNA(c) + FPA(c))` — for each matched
  detection, how well its GT ID and predicted ID correspond across the whole
  sequence.

### The matching is the subtle part

Naive per-frame IoU matching would let the same person be served by different
predicted IDs in different frames at no cost. HOTA prevents this in two passes:

**Pass 1 — global correspondence.** For every (GT ID, predicted ID) pair,
accumulate a normalised overlap mass across all frames:

```python
denom   = sim.sum(0) + sim.sum(1) - sim      # row + column mass
sim_iou = sim / denom
potential[gt_id, pred_id] += sim_iou
global_alignment = potential / (gt_count + pred_count - potential)
```

Normalising by row and column mass stops one ambiguous frame in a dense crowd
from dominating the estimate.

**Pass 2 — alignment-biased per-frame assignment.** Hungarian matching on
`global_alignment × similarity`, solved **once** per frame and then thresholded at
each α. Biasing toward globally-consistent pairs is what makes `AssA` measure
association rather than luck.

### A property of HOTA that surprises people

**`AssA` is not independent of recall.** A missed detection counts as an
association false negative (`FNA`), so dropping detections lowers `AssA` even
when every reported identity is perfect. Measured here: predicting only the first
half of a sequence perfectly gives `DetA = 0.50` **and** `AssA = 0.50`, not
`AssA = 1.0`.

This is correct per the paper's formulation, not a bug — but it means "`AssA` is
the association metric, `DetA` is the detection metric" is a simplification, and
`AssA` moving should not be read as association logic changing on its own.

---

## CLEAR — MOTA, MOTP, IDSW

> Bernardin & Stiefelhagen, *Evaluating Multiple Object Tracking Performance*,
> 2008.

```
MOTA = 1 − (FN + FP + IDSW) / num_gt_detections
```

Reported for diagnosis, never optimised. MOTA is unbounded below and is dominated
by detection counts, so a change that trades identity stability for raw recall
can raise MOTA while lowering the leaderboard score.

The pair `(MOTA, IDF1)` is genuinely useful during development: MOTA up with IDF1
down means a change bought detections at the cost of identities — on this
benchmark, a losing trade.

Matching biases heavily toward whatever ID already owned each person
(`score = 1000 · keeps_identity + similarity`) so that a switch is only recorded
when the identity genuinely changed, not when two equally good assignments existed.

---

## MOTChallenge preprocessing — the step that is easy to miss

**This changes scores by a large margin.** Omitting it is the usual reason a local
number disagrees with a leaderboard.

MOTChallenge ground truth annotates far more than the pedestrians you are scored
on. It also marks people on vehicles, static people, explicit "distractor" regions
such as mannequins and posters, and reflections. Those are *neither* targets nor
errors: a detector firing on a shop-window reflection has not made the mistake the
benchmark wants to punish — it found something the annotators deliberately
excluded.

The protocol:

- **Ground truth to score against**: class 1 (pedestrian) with the ignore flag
  unset.
- **Predictions overlapping a distractor class** at IoU ≥ 0.5 are **deleted**, not
  counted as false positives. Distractor classes: 2 (person on vehicle),
  7 (static person), 8 (distractor), 12 (reflection).

Crucially, the overlap matrix for distractor suppression is computed against
**all** ground truth rows including the distractors, *before* the scored subset is
selected. Doing it in the other order finds nothing to suppress.

### The course readme is wrong about this data

`Week 4/readme.md` states `gt.txt`'s class column is "always 1 (pedestrian)".
Measured class counts:

```
01   {1: 5257,  8: 1575,  9: 1050, 12: 948}
02   {1: 11450, 3: 4484,  10: 2542, 11: 680, 4: 103, 8: 4}
03   {1: 47557, 10: 23100, 11: 18900, 4: 11550, 7: 4798, 3: 1050, 5: 1050}
ref  {1: 17833, 7: 5271,  9: 1781,  8: 1200, 2: 1549, 4: 1559}
```

In every sequence the count of `conf == 1` rows exactly equals the count of
`class == 1` rows, so the two filters coincide here — but the distractor rows are
real and must be handled. Taking the readme at face value and scoring against all
rows produces badly wrong numbers.

`build_sequence_data(..., apply_preprocessing=False)` exists so tests can disable
this when the machinery would obscure the property under test.

---

## Validation

Correctness is established against cases whose answers are derivable by hand, in
`tests/test_metrics.py`. Predicted vs measured:

| case | expectation | measured |
|---|---|---|
| perfect prediction | all metrics 1.0, IDSW 0 | ✅ exact |
| empty prediction | all metrics 0.0 | ✅ exact |
| every ID renamed at the midpoint | DetA 1.0, AssA 0.5, IDF1 0.5 | ✅ 1.000 / 0.500 / 0.500 |
| every box duplicated under a new ID | IDP 0.5, IDR 1.0 | ✅ exact |
| only the first half of frames predicted | DetA 0.5 | ✅ 0.500 |
| 20 px shift on a 60 px-wide box → IoU exactly 0.5 | DetA = 10/19 = 0.5263 | ✅ 0.5263 |
| same, IDF1 at its single 0.5 threshold | 1.0 | ✅ 1.000 |
| 40 px shift → IoU 0.2 | IDF1 0.0 | ✅ 0.000 |
| surplus box on a distractor annotation | IDFP 0, IDF1 1.0 | ✅ exact |
| same box on empty ground | IDFP 20 | ✅ 20 |
| row order permuted | identical scores | ✅ exact |

The 20 px case is the sharpest test: a 60 px-wide box shifted 20 px has IoU
exactly `40/80 = 0.5`, so it matches at every α ≤ 0.5 — 10 of the 19 thresholds —
and `DetA` must be exactly `10/19`. It is.

### A test that was wrong, and what it taught

The first version of the distractor test used "preprocessing disabled" as its
control and failed. The control was invalid: with preprocessing off, the
distractor row is itself treated as a *target*, so the surplus prediction simply
matches it and is not a false positive either way.

The correct control is the same surplus box on **empty ground** — forgiven at
IDFP 0 when it lands on a distractor, penalised at IDFP 20 when it does not. The
implementation had been right; the test had been measuring nothing.
