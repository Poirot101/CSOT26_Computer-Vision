# Clean-checkout verification

The project was cloned fresh from the private repository into an empty directory
and run end to end, to confirm it works independently of the environment it was
built in.

```bash
git clone --depth 1 https://github.com/Poirot101/dense-crowd-mot /home/user/dense-crowd-mot
cd /home/user/dense-crowd-mot/final-project
pip install -e .
```

Clone: **3,006 files**, commit `bbe80eb`. Install succeeded and registered the
`cv-mot` console script. `pip install -e .` installs only `numpy` and `scipy` —
the tracking and metrics code has no heavier dependency, and `ultralytics`/`torch`
are needed only for live detection (`.[detect]`).

---

## 1. Test suite

```
$ python -m pytest tests -q
77 passed
```

68 are unit tests of the library; 9 assert that the documentation matches the
measured results in `outputs/*.json` — see `tests/test_docs_consistency.py`.
Those were added after a verification pass found the oracle figures in six
documents had been measured before a bug fix and never re-run. Each was checked
to fail on a deliberately corrupted document before being kept.

## 2. Tracking, via the installed console script

Cached detections ship with the repository, so this runs without any inference:

```
$ cv-mot track --data "../Week 4/data" --dets ../cache/yolov8s_1280 \
      --config configs/final.json --interpolate --interpolate-max-gap 20 --out <out>

01: 525 frames,  9676 dets,  6072 rows (+413 interpolated), 541.5 fps
02: 750 frames, 18302 dets,  6481 rows (+287 interpolated), 646.4 fps
03: 1050 frames, 68291 dets, 39393 rows (+678 interpolated), 210.4 fps
ref: 600 frames, 25683 dets,  9323 rows (+584 interpolated), 444.9 fps
```

## 3. Evaluation — reproduces the reported scores exactly

```
sequence      HOTA    DetA    AssA    IDF1    MOTA   IDSW   Score
01          0.4615  0.5515  0.3893  0.5199  0.5868     40  0.5004
02          0.4157  0.3732  0.4676  0.5310  0.4333     76  0.4926
03          0.5063  0.4938  0.5233  0.5849  0.4884     67  0.5587
ref         0.3449  0.3035  0.3963  0.4182  0.3118     52  0.3937
OVERALL     0.4621  0.4410  0.4885  0.5418  0.4487    235  0.5152
```

Identical to [RESULTS.md](RESULTS.md) to four decimal places. The pipeline is
deterministic given fixed detections — no seeded randomness anywhere in the
association path.

## 4. Format validation

```
OK    01.txt  6072 rows, frames 1-525, 73 identities
OK    02.txt  6481 rows, frames 1-750, 125 identities
OK    03.txt  39393 rows, frames 1-1050, 141 identities
OK    ref.txt 9323 rows, frames 1-600, 95 identities
all files valid
```

## 5. Live detector path, with no cache

The cached detections shipped with the repository, so the detector path was
exercised separately on a 40-frame slice to confirm it is not stale:

```
$ cv-mot track --data <40-frame slice> --weights yolov8s.pt --imgsz 1280 \
      --conf 0.05 --config configs/final.json --interpolate --out <out>

01: 40 frames, 507 dets, 325 rows (+19 interpolated), 2.1 fps
OK    01.txt  325 rows, frames 1-40, 10 identities
```

Ultralytics downloaded `yolov8s.pt` automatically. Output rows are correctly
formatted `tlwh`:

```
1,1,1698.78,387.90,160.76,340.57,-1,-1,-1,-1
1,2,0.38,340.47,114.68,561.78,-1,-1,-1,-1
1,3,249.25,456.29,105.88,249.67,-1,-1,-1,-1
```

2.1 fps is the CPU detector cost (0.43 s/frame), not the tracker, which runs at
200–650 fps on cached detections.

---

## What this does and does not establish

**Establishes:** the repository is self-contained; a fresh checkout installs,
tests, tracks, evaluates and validates with no manual setup, and reproduces the
reported numbers exactly.

**Does not establish:** correctness on the real test sequences, which are not in
this repository (see [RESULTS.md](RESULTS.md) §7), or agreement with the official
TrackEval implementation — the metrics here were validated against hand-derivable
cases (see [METRICS.md](METRICS.md)) rather than differentially against TrackEval,
which was not available in this environment.
