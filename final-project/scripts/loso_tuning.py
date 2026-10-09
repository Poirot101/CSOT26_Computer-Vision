#!/usr/bin/env python3
"""Leave-one-sequence-out tuning, to check the configuration is not over-fitted.

The configuration in ``configs/final.json`` was selected on the ``ref`` sequence
alone, which makes the scores on ``01``/``02``/``03`` only *near* held-out --
documented as a limitation in docs/TUNING.md. This script answers the question
that limitation raises:

    If the threshold had been chosen on the other three sequences instead,
    would it have landed in the same place?

For each sequence it selects the best ``track_high_thresh`` on the *other three*
and reports the score that choice achieves on the held-out one. A configuration
that generalises picks a similar value every time, and loses little against the
best value chosen with hindsight on the held-out sequence itself.

Usage::

    python scripts/loso_tuning.py --data "../Week 4/data" --dets ../cache/yolov8s_1280
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_mot.bytetrack import ByteTrackConfig, ByteTracker  # noqa: E402
from cv_mot.detect import CachedDetections  # noqa: E402
from cv_mot.metrics import evaluate  # noqa: E402
from cv_mot.mot_io import discover_sequences, read_mot_file  # noqa: E402
from cv_mot.pipeline import run_sequence  # noqa: E402


def score_sequences(seqs, dets_dir: Path, cfg: ByteTrackConfig, interpolate: bool, gap: int):
    payload = {}
    for seq in seqs:
        seq_cfg = replace(cfg, frame_rate=seq.frame_rate)
        rows, _ = run_sequence(
            seq,
            CachedDetections(dets_dir / f"{seq.name}.npz"),
            lambda c=seq_cfg: ByteTracker(c),
            interpolate=interpolate,
            interpolate_max_gap=gap,
        )
        payload[seq.name] = (read_mot_file(seq.gt_path), rows, seq.length)
    return evaluate(payload)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dets", required=True, type=Path)
    ap.add_argument("--grid", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/final.json")
    ap.add_argument("--interpolate", action="store_true", default=True)
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    base = ByteTrackConfig(**json.loads(args.config.read_text()))
    sequences = discover_sequences(args.data)
    sequences = [s for s in sequences if s.gt_path.exists()]
    print(f"sequences: {[s.name for s in sequences]}\n")

    # Score every (sequence, threshold) pair once; folds are then just lookups.
    table: dict[str, dict[float, float]] = {}
    for seq in sequences:
        table[seq.name] = {}
        for thresh in args.grid:
            cfg = replace(base, track_high_thresh=thresh, new_track_thresh=thresh)
            overall = score_sequences([seq], args.dets, cfg, args.interpolate, args.gap)
            table[seq.name][thresh] = overall["Score"]
        row = "  ".join(f"{t}:{table[seq.name][t]:.4f}" for t in args.grid)
        print(f"  {seq.name:>4}  {row}")

    print(f"\n{'held out':>10}{'chosen on others':>19}{'held-out score':>17}{'best possible':>15}{'regret':>9}")
    print("-" * 70)
    folds = []
    for held in sequences:
        others = [s for s in sequences if s.name != held.name]
        # Pick the threshold with the best mean score across the other sequences.
        means = {t: sum(table[o.name][t] for o in others) / len(others) for t in args.grid}
        chosen = max(means, key=means.get)
        got = table[held.name][chosen]
        best = max(table[held.name].values())
        regret = best - got
        folds.append({"held_out": held.name, "chosen": chosen, "score": got,
                      "best_possible": best, "regret": regret})
        print(f"{held.name:>10}{chosen:>19}{got:>17.4f}{best:>15.4f}{regret:>9.4f}")

    chosen_values = sorted({f["chosen"] for f in folds})
    mean_regret = sum(f["regret"] for f in folds) / len(folds)
    print("-" * 70)
    print(f"thresholds selected across folds: {chosen_values}")
    print(f"mean regret vs. hindsight-optimal: {mean_regret:.4f} Score")
    if len(chosen_values) == 1:
        print(f"\nEvery fold selects {chosen_values[0]} -- the choice does not depend on which\n"
              f"sequence it was tuned on, so it is not an artefact of tuning on ref.")
    else:
        print(f"\nFolds disagree ({chosen_values}); the threshold is sequence-dependent and the\n"
              f"single-sequence choice in configs/final.json carries real selection risk.")

    if args.json:
        args.json.write_text(json.dumps({"per_sequence": {k: {str(t): v for t, v in d.items()}
                                                          for k, d in table.items()},
                                         "folds": folds,
                                         "mean_regret": mean_regret}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
