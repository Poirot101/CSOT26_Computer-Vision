# Week 5 — Advanced localisation: DeepSORT, ByteTrack and occlusion

## What the original repo contains

No `Week 5/` directory exists. The week is described only in the root
`README.md`:

> * Overcoming baseline tracking limitations (handling occlusions, integrating
>   appearance/deep features, and processing low-score detections).
> * Exploring modern frameworks like DeepSORT and ByteTrack.
> * Benchmarking system continuity using industry-standard multi-object metrics.

Those three bullets map one-to-one onto the three largest pieces of work in this
project: `bytetrack.py` (low-score detections), the optional Re-ID path
(appearance), and `metrics/` (industry-standard benchmarking).

## The concepts

### Why SORT breaks in crowds

SORT associates on motion alone, through a single confidence threshold. Two
failure modes follow, and they are different:

**Failure 1 — the low-score detection.** When person A walks behind person B, the
detector still fires on A but with low confidence, because it can only see part of
them. SORT's threshold discards that box. The track goes unmatched, coasts, and
dies. When A emerges they receive a **new identity**. This is the dominant failure
mode on dense data, and it is a *thresholding* problem, not a motion-model problem.

**Failure 2 — the identity swap.** Two people cross. Both have plausible
predicted positions near both detections. Motion alone cannot disambiguate, and
the Hungarian solve may pick the wrong pairing. This is an *information* problem:
motion does not carry enough signal.

The two need different fixes, which is why there are two families of modern
tracker.

### ByteTrack — fixes failure 1

The insight: a low-score box in roughly the right place is excellent evidence
**if a track is already there**. The threshold's job is not to decide what is
real; it is to decide what may *start* a new identity.

Association runs in two passes:

1. High-score detections against all active **and** lost tracks.
2. Low-score detections against only the tracks still unmatched — and these may
   **never create** an identity.

That asymmetry is the whole method. Low-score boxes sustain existing identities
without importing the false positives that globally lowering the threshold would
bring. Implemented in `cv_mot/bytetrack.py`; the three passes are explicitly
labelled in `ByteTracker.update`.

### DeepSORT — fixes failure 2

Adds an appearance embedding per detection from a Re-ID network, and matches on a
combination of motion and appearance distance. A track remembers *what the person
looks like*, so identity survives occlusions long enough that motion prediction
has drifted uselessly.

Implemented as an **optional** path (`ByteTrackConfig.with_reid`,
`matching.embedding_distance`, the feature bank in `STrack.update_feature`) with
two deliberate design choices:

- Features are an **exponential moving average**, not the latest embedding. One
  half-occluded frame yields a contaminated embedding; the EMA prevents a single
  bad frame from destroying an identity model.
- Appearance may only ever **lower** a cost, and only when confident
  (`appearance_thresh`). A Re-ID model that has never seen this camera is not
  trustworthy enough to veto a good geometric match.

### Why ByteTrack was chosen as the primary method

Measured, not assumed. The oracle experiment in [../RESULTS.md](../RESULTS.md)
feeds the tracker **perfect boxes** and varies only the confidence values:

| detections | IDF1 | ID switches |
|---|---|---|
| perfect boxes, uniform high confidence | 0.9853 | 0 |
| perfect boxes, realistic confidence | 0.7425 | 9 |

With perfect boxes the tracker scores 0.985 with **zero** switches — so failure 2
is not what is costing points here. Changing only the confidence values costs ~24
IDF1 points. Failure 1 dominates on this data by a wide margin, so the effort went
to ByteTrack, and appearance remained optional.

This is the reasoning the final project asks for: pick the method that addresses
the failure mode your data actually exhibits, and be able to show which one that
is.

### Benchmarking continuity

"Industry-standard multi-object metrics" means HOTA, IDF1 and the CLEAR family,
implemented from their papers in `cv_mot/metrics/` — see
[../METRICS.md](../METRICS.md).

## Where this lands in the final project

| Week 5 concept | Module |
|---|---|
| Two-pass low-score association | `cv_mot/bytetrack.py::ByteTracker.update` |
| Track states incl. `Lost` for occlusion | `cv_mot/track.py::TrackState` |
| Appearance embeddings (optional) | `cv_mot/matching.py::embedding_distance`, `STrack.update_feature` |
| Motion + appearance fusion | `cv_mot/matching.py::fuse_motion` |
| HOTA / IDF1 / CLEAR | `cv_mot/metrics/` |
| Duplicate-track suppression | `cv_mot/bytetrack.py::_remove_duplicates` |

## What was learned

Reading three tracker papers and then **measuring which failure mode the data
exhibits** produced a different decision than reading the papers alone would have.
The literature's most sophisticated option (appearance-based re-identification)
was not the one this dataset needed most; the simpler one was. Diagnosis before
treatment.
