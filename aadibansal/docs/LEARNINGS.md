# Learnings

What this project actually taught, including the parts that went wrong.

---

## 1. Build the measurement apparatus before the thing being measured

The evaluator (`cv_mot/metrics/`) was written before any tracker. This felt like
a detour and was the single highest-leverage decision in the project.

Every tuning decision is a comparison between two numbers. If those numbers come
from a wrong evaluator, the sweep still produces a confident-looking optimum — it
is just the optimum of the wrong function, and nothing in the output reveals that.
Local HOTA/IDF1 meant every later choice (ByteTrack over SORT, threshold 0.4,
interpolation on, score fusion off) rests on a measurement instead of an argument.

**Generalises to:** write the metric before the model; write the assertion before
the fix.

---

## 2. Read the data, not the documentation about the data

Two findings, both from a few minutes of counting columns, both of which changed
the implementation:

**`gt.txt` contradicts the course readme.** The readme says the class column is
"always 1 (pedestrian)". It is not — the data is full MOTChallenge format with
distractor classes (static people, mannequins, reflections) that must be handled
by *deleting* overlapping predictions rather than counting them as false
positives. Scoring against all rows gives badly wrong numbers.

**The sequences are MOT17 train sequences.** Frame counts and resolutions match
exactly: `01`=MOT17-09, `02`=MOT17-13, `03`=MOT17-04, `ref`=MOT17-02. That meant
the official protocol applied verbatim and published ByteTrack numbers were
available as a sanity reference.

**Also:** all four sequences ship with `gt.txt`, not just `ref` as the readme
implies — so honest local evaluation was possible on everything.

---

## 3. Diagnose before treating — the oracle experiment

The most useful hour went into `GroundTruthDetector`, which synthesises detections
of known quality so detector error can be dialled to zero.

| detections | IDF1 | IDSW |
|---|---|---|
| perfect boxes, uniform confidence | **0.9833** | **0** |
| perfect boxes, realistic confidence | 0.7410 | 9 |

The boxes are *identical* between those rows. Only the confidence attached to each
changes, and that alone costs 24.2 IDF1 points.

Two things follow immediately:

1. The tracker is not broken — 0.983 with zero switches says the Kalman model and
   the association logic are sound. Any remaining error is elsewhere.
2. **Confidence handling, not box quality or association cleverness, is the
   dominant lever on this data.**

That is why the effort went to ByteTrack and threshold tuning, and why appearance
re-identification — the literature's more sophisticated answer — stayed optional.
Reading the papers alone would have suggested the opposite priority.

---

## 4. Know what your ceiling is before optimising toward it

`DetA` on real detections tops out near 0.44 and `ref` only reaches 0.30. Rather
than assume the tracker was wasting detections, I measured the best any tracker
could possibly do with these boxes — optimal per-frame assignment at IoU ≥ 0.5:

| detector confidence floor | recall ceiling |
|---|---|
| ≥ 0.05 | **0.5618** |
| ≥ 0.50 | 0.2772 |

**56% is the hard ceiling at the most permissive floor.** Broken down by how
visible each person actually is:

| visibility | recall | GT rows |
|---|---|---|
| 0.0 – 0.1 | 0.234 | 7037 |
| 0.8 – 1.0 | 0.943 | 5147 |

The detector is excellent on people it can see and poor on people it mostly
cannot — correct behaviour. The problem is that **7037 of 17833 scored
annotations on `ref` are under 10% visible**, and the benchmark counts every one.

Without this measurement I would have spent the remaining time tuning association
parameters against a detection-bound problem.

---

## 5. HOTA's AssA is not independent of recall

Expected `AssA` to measure association alone. It does not: a missed detection
counts as an association false negative, so dropping detections lowers `AssA` even
when every reported identity is perfect.

Measured — predicting only the first half of a sequence, perfectly:
`DetA = 0.50` **and** `AssA = 0.50`, not `AssA = 1.0`.

This is correct per the paper, and it is why the ablation table reads oddly at
first glance: enabling ByteTrack *lowers* `AssA` (0.4344 → 0.3916) while raising
`DetA` and the overall score. Recovering more low-confidence boxes creates more
association opportunities to get wrong. Without understanding this coupling I
would have read that as a regression and reverted a change worth +0.022 Score.

---

## 6. Three test failures, and what each revealed

The first full run had three failures. All three were *my tests* being wrong, but
each diagnosis was worth having.

**Test asserted a frame-1 track should be withheld.** It is emitted by design —
there is no earlier frame that could confirm it, so withholding loses real people
at sequence start. The test was rewritten to check a blip *mid-sequence*, which is
the case that actually matters.

**Test placed 30 people across a 1920 px frame at 300 px spacing.** Everyone past
index 6 was outside the frame and correctly clipped away. The test was measuring
its own generator, not the tracker. Fixed with a grid layout.

**Test used "preprocessing disabled" as the control for distractor handling.** An
invalid control: with preprocessing off, the distractor row is itself treated as a
*target*, so the surplus prediction matches it and is not a false positive either
way. The correct control is the same surplus box on empty ground — forgiven at
IDFP 0 on a distractor, penalised at IDFP 20 off it.

**Lesson:** when a test fails, establish whether the code or the test is wrong
before touching either. All three here would have been "fixed" destructively by
changing the implementation to match a bad expectation.

---

## 7. A test that passed while testing nothing

Worse than the failures. `test_output_boxes_are_clipped_to_the_frame` passed for
the whole project while the clipping it tested was a **no-op**.

The bug: `ByteTracker` wrote the clipped box to `STrack._tlwh`, but the `tlwh`
property derives from the Kalman mean whenever a track is active, so `_tlwh` was
ignored. Nothing was ever clipped.

The test passed because its trajectory walked the track *fully* off-frame, so the
minimum-area filter dropped it, `out` was empty, and the `for t in out:` assertion
loop never executed. A vacuous pass.

It surfaced only when `scripts/validate_submission.py` ran on the real output and
reported negative coordinates and boxes past the image bounds in all four files.

Fixes: an explicit `out_tlwh` slot separate from the filter state (clipping is an
output concern and must not feed back into the motion model), and a test that
asserts a track was actually emitted *and* that the unclipped state genuinely
exceeded the frame — so it cannot pass vacuously again.

**Lessons:** a loop over a possibly-empty collection is not an assertion; and an
independent output check catches what unit tests miss, because the validator knew
nothing about my assumptions.

---

## 8. Tune against the metric you are scored on

`track_buffer` made this concrete. Identity switches keep falling as the buffer
lengthens (65 at 15 frames → 48 at 90), because longer memory really does recover
more occlusions. But IDF1 **peaks at 30**: beyond that, stale tracks get revived
onto the wrong person — corruption that IDSW does not count, because from CLEAR's
view the identity stayed consistent, while IDF1's global mapping sees it.

Optimising `IDSW` would have selected 60 or 90. Optimising the actual leaderboard
metric selects 30.

---

## 9. Transfer learning did the heaviest lifting

The pipeline achieves usable recall on surveillance footage using a detector with
**zero task-specific training** — COCO-pretrained YOLOv8 applied directly. On 4
CPU cores with no GPU, fine-tuning was not an option, and the project works anyway
because pretrained features generalise.

Knowing *when not to train* mattered more here than any architectural knowledge.

---

## What I would try next, in order of expected value

1. **A pedestrian-specific detector.** Section 4 says detection is the binding
   constraint by a wide margin: the ceiling is 56% recall and we achieve DetA 0.44.
   Fine-tuning YOLOv8 on crowded pedestrian data, or using a detector trained for
   MOT (the ByteTrack authors' YOLOX-X), is worth more than every tracker change
   combined. Nothing else on this list competes — and the `yolov8m` result in (2)
   is direct evidence that this axis pays, since a merely *larger generic* detector
   already beats every tracker-side change except ByteTrack itself.
2. **A larger backbone — now measured, and it works.** Swapping `yolov8s` for
   `yolov8m` at the same resolution, changing nothing else, gains **+0.0146 Score**
   on `ref` (DetA +0.0196, AssA unchanged). That is two thirds of the entire
   ByteTrack-over-SORT gain, from one flag. See [RESULTS.md](RESULTS.md) §6.
   Adopting it for the submission needs detection re-run on all four sequences
   (~45 min on CPU) and a re-tune. `yolov8l`/`yolov8x` at 1536 is the next step.

   The surprise: `yolov8m` has a *lower* raw recall ceiling yet scores better,
   because its boxes are better calibrated and more of them survive the operating
   threshold. **Recall at the threshold you actually use is the objective, not raw
   recall** — a distinction I would have missed without measuring both.
3. ~~**Leave-one-sequence-out tuning.**~~ **Done** — `scripts/loso_tuning.py`.
   Every fold independently selects `track_high_thresh = 0.4`, with mean regret
   0.0074 Score against the hindsight-optimal per-sequence choice, so the headline
   threshold is not an artefact of tuning on `ref`. The secondary sweeps
   (`track_low_thresh`, `track_buffer`, interpolation gap) are still
   single-sequence and remain worth cross-validating.
4. **Re-ID appearance features.** Implemented but unevaluated. Expected to help
   less than (1)–(3) here *because* of the oracle result — but it is the right fix
   for the residual failure mode once detection improves, since the 9 ID switches
   in the perfect-box oracle run are exactly what appearance targets.
5. **OC-SORT's observation-centric update.** Addresses a real weakness visible in
   this code: a track coasting through a long occlusion accumulates velocity error,
   and OC-SORT re-smooths the trajectory from the re-detection backwards instead of
   trusting the drifted state. Cheap to add on top of the existing Kalman filter.
6. **Camera motion compensation.** Listed in the brief; these sequences are
   static, so expected gain is near zero. Worth implementing only for completeness.

---

## The one-line summary

The score came from understanding where the error was, not from implementing the
most sophisticated method available. The oracle experiment and the recall-ceiling
measurement each took under an hour and redirected everything that followed.
