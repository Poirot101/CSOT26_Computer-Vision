#!/usr/bin/env python3
"""Check submission files against the required format before handing them in.

Catches the failures that silently cost a leaderboard score rather than raising
an error: 0-based frames, ``xyxy`` coordinates left unconverted, a ``conf``
column that is not ``-1``, non-integer identities, and duplicate (frame, id)
pairs.

Usage::

    python scripts/validate_submission.py outputs/01.txt outputs/02.txt ...
    python scripts/validate_submission.py --data "../Week 4/data" outputs/*.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_mot.mot_io import read_seqinfo  # noqa: E402


def validate(path: Path, seq_dir: Path | None) -> list[str]:
    problems: list[str] = []
    text = path.read_text().strip()
    if not text:
        return [f"{path.name}: file is empty"]

    lines = text.split("\n")
    for lineno, line in enumerate(lines[:], start=1):
        parts = line.split(",")
        if len(parts) != 10:
            problems.append(f"{path.name}:{lineno}: expected 10 columns, found {len(parts)}")
            break

    rows = np.loadtxt(path, delimiter=",", ndmin=2)

    if rows[:, 0].min() < 1:
        problems.append(f"{path.name}: frame numbers must be 1-based (min is {rows[:, 0].min():.0f})")
    if rows[:, 1].min() < 1:
        problems.append(f"{path.name}: track ids must be 1-based (min is {rows[:, 1].min():.0f})")
    if not np.allclose(rows[:, 0], np.round(rows[:, 0])):
        problems.append(f"{path.name}: frame column must hold integers")
    if not np.allclose(rows[:, 1], np.round(rows[:, 1])):
        problems.append(f"{path.name}: id column must hold integers")
    if not np.all(rows[:, 6:] == -1):
        problems.append(f"{path.name}: columns 7-10 (conf, x, y, z) must all be -1")

    if np.any(rows[:, 4] <= 0) or np.any(rows[:, 5] <= 0):
        problems.append(f"{path.name}: width/height must be positive -- xyxy left unconverted?")

    # A width that tracks the left edge is the signature of xyxy output.
    # Needs variation in both columns for the correlation to be meaningful.
    if len(rows) > 2 and rows[:, 2].std() > 1e-9 and rows[:, 4].std() > 1e-9:
        if np.corrcoef(rows[:, 2], rows[:, 4])[0, 1] > 0.9:
            problems.append(
                f"{path.name}: width correlates with left edge (r>0.9) -- boxes look like xyxy, not tlwh"
            )

    pairs = rows[:, :2].astype(np.int64)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        problems.append(f"{path.name}: duplicate (frame, id) pairs present")

    if seq_dir is not None and seq_dir.exists():
        info = read_seqinfo(seq_dir)
        if rows[:, 0].max() > info.length:
            problems.append(
                f"{path.name}: frame {rows[:, 0].max():.0f} exceeds sequence length {info.length}"
            )
        margin = 1.0
        if rows[:, 2].min() < -margin or rows[:, 3].min() < -margin:
            problems.append(f"{path.name}: negative coordinates present")
        if (rows[:, 2] + rows[:, 4]).max() > info.width + margin:
            problems.append(f"{path.name}: boxes extend past image width {info.width}")
        if (rows[:, 3] + rows[:, 5]).max() > info.height + margin:
            problems.append(f"{path.name}: boxes extend past image height {info.height}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=None, help="sequence root, for frame-count and bounds checks")
    args = ap.parse_args()

    all_problems = []
    for path in args.files:
        seq_dir = (args.data / path.stem) if args.data else None
        problems = validate(path, seq_dir)
        rows = len(path.read_text().strip().split("\n")) if path.read_text().strip() else 0
        if problems:
            all_problems.extend(problems)
            print(f"FAIL  {path.name}  ({rows} rows)")
            for p in problems:
                print(f"        {p}")
        else:
            data = np.loadtxt(path, delimiter=",", ndmin=2)
            print(
                f"OK    {path.name}  {rows} rows, "
                f"frames {int(data[:,0].min())}-{int(data[:,0].max())}, "
                f"{len(np.unique(data[:,1]))} identities"
            )

    print()
    if all_problems:
        print(f"{len(all_problems)} problem(s) found")
        return 1
    print("all files valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
