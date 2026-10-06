#!/usr/bin/env bash
# Full reproduction, from a clean checkout to a validated submission.
#
# Every number in docs/RESULTS.md comes from these commands. Runtimes are for
# 4 CPU cores with no GPU; a GPU cuts the detection stages by roughly 20x.
set -euo pipefail

DATA="${DATA:-../Week 4/data}"
CACHE="${CACHE:-../cache}"
OUT="${OUT:-outputs}"
PY="${PY:-python3}"

export PYTHONPATH="src:${PYTHONPATH:-}"

echo "==> 1/6  unit tests (68 tests, ~5s)"
$PY -m pytest tests -q

echo
echo "==> 2/6  tracker ablation on oracle detections (no inference; isolates association)"
$PY -m cv_mot ablate --data "$DATA" --seqs ref --oracle \
    --json "$OUT/ablation_oracle.json"

echo
echo "==> 3/6  cache detections  (yolov8s @ 1280, ~21 min on 4 CPU cores)"
$PY -m cv_mot detect --data "$DATA" --out "$CACHE/yolov8s_1280" \
    --weights yolov8s.pt --imgsz 1280 --conf 0.05 --batch 4

echo
echo "==> 4/6  tune the tracker on the reference sequence (cached dets, ~2 min)"
$PY -m cv_mot tune --data "$DATA" --seqs ref --dets "$CACHE/yolov8s_1280" \
    --grid-high 0.3 0.4 0.5 0.6 --grid-new 0.5 0.6 0.7 \
    --json "$OUT/tuning.json"

echo
echo "==> 5/6  ablation on real detections"
$PY -m cv_mot ablate --data "$DATA" --seqs ref --dets "$CACHE/yolov8s_1280" \
    --json "$OUT/ablation_yolo.json"

echo
echo "==> 6/6  final submission + validation"
$PY -m cv_mot track --data "$DATA" --dets "$CACHE/yolov8s_1280" \
    --config configs/final.json --interpolate --out "$OUT"
$PY -m cv_mot evaluate --data "$DATA" --preds "$OUT" --json "$OUT/final_scores.json"
$PY scripts/validate_submission.py --data "$DATA" "$OUT"/*.txt

echo
echo "done. submission files in $OUT/"
