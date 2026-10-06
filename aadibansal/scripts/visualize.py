#!/usr/bin/env python3
"""Render tracking output over the frames, as a video or contact sheet.

The Week 4 brief asks you to *watch* the output and find where matching fails:

> Watch your output video closely. Look for moments where matching fails -- such
> as when objects overlap or pass behind barriers.

Metrics say how much is wrong; only looking says what kind of wrong. An identity
switch and a missed detection both lower IDF1, but they look completely
different on screen, and the fix differs.

Usage::

    # annotated video of a sequence
    python scripts/visualize.py --data "../Week 4/data" --seq ref \
        --preds outputs/ref.txt --out /tmp/ref.mp4

    # just the frames where an identity switch happens, as a contact sheet
    python scripts/visualize.py --data "../Week 4/data" --seq ref \
        --preds outputs/ref.txt --switches-only --out /tmp/switches.jpg

With ``--gt`` the ground-truth boxes are drawn in white alongside predictions,
which is how you tell a tracker failure from a detector failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_mot.boxes import iou_matrix, tlwh_to_xyxy  # noqa: E402
from cv_mot.mot_io import read_mot_file, read_seqinfo  # noqa: E402

# Distinct, reasonably colour-blind-safe palette, cycled by track id.
PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (153, 153, 153),
]


def colour_for(track_id: int) -> tuple[int, int, int]:
    r, g, b = PALETTE[int(track_id) % len(PALETTE)]
    return (b, g, r)  # OpenCV uses BGR


def find_switch_frames(pred: np.ndarray, gt: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    """Frames where a ground-truth identity changes which predicted id covers it.

    Uses greedy best-IoU matching per frame, which is enough to *locate*
    interesting frames; the scoring metrics do the rigorous matching.
    """
    gt = gt[(gt[:, 6] != 0) & (gt[:, 7].astype(int) == 1)] if gt.shape[1] >= 8 else gt
    owner: dict[int, int] = {}
    switches: list[int] = []

    frames = np.union1d(pred[:, 0].astype(int), gt[:, 0].astype(int))
    for f in frames:
        g = gt[gt[:, 0].astype(int) == f]
        p = pred[pred[:, 0].astype(int) == f]
        if len(g) == 0 or len(p) == 0:
            continue
        sim = iou_matrix(tlwh_to_xyxy(g[:, 2:6]), tlwh_to_xyxy(p[:, 2:6]))
        for gi in range(len(g)):
            pj = int(np.argmax(sim[gi]))
            if sim[gi, pj] < iou_thresh:
                continue
            gid, pid = int(g[gi, 1]), int(p[pj, 1])
            if gid in owner and owner[gid] != pid:
                switches.append(int(f))
            owner[gid] = pid
    return sorted(set(switches))


def draw_frame(img, pred_rows: np.ndarray, gt_rows: np.ndarray | None, frame_no: int, label: str):
    import cv2

    if gt_rows is not None:
        for row in gt_rows:
            x, y, w, h = row[2:6]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)

    for row in pred_rows:
        tid = int(row[1])
        x, y, w, h = row[2:6]
        colour = colour_for(tid)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), colour, 2)
        text = str(tid)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(int(y) - th - 4, 0)
        cv2.rectangle(img, (int(x), top), (int(x) + tw + 4, top + th + 4), colour, -1)
        cv2.putText(img, text, (int(x) + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    banner = f"{label}  frame {frame_no}  tracks {len(pred_rows)}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gt", action="store_true", help="overlay ground truth in white")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--switches-only", action="store_true",
                    help="render only frames containing an identity switch, as a contact sheet")
    ap.add_argument("--cols", type=int, default=3, help="contact-sheet columns")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv is required: pip install opencv-python-headless", file=sys.stderr)
        return 2

    seq = read_seqinfo(args.data / args.seq)
    pred = read_mot_file(args.preds)
    if pred.size == 0:
        print(f"{args.preds} is empty", file=sys.stderr)
        return 1
    gt = read_mot_file(seq.gt_path) if (args.gt or args.switches_only) and seq.gt_path.exists() else None
    gt_eval = gt[(gt[:, 6] != 0) & (gt[:, 7].astype(int) == 1)] if gt is not None else None

    frames = seq.frame_paths()
    if args.switches_only:
        if gt is None:
            print("--switches-only needs gt.txt", file=sys.stderr)
            return 1
        wanted = find_switch_frames(pred, gt)
        if not wanted:
            print("no identity switches found by greedy matching")
            return 0
        print(f"{len(wanted)} frames contain an identity switch; rendering first {args.cols ** 2}")
        wanted = wanted[: args.cols ** 2]
    else:
        end = args.end or seq.length
        wanted = list(range(args.start, min(end, len(frames)) + 1))
        if args.max_frames:
            wanted = wanted[: args.max_frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    writer = None

    for frame_no in wanted:
        img = cv2.imread(str(frames[frame_no - 1]))
        if img is None:
            continue
        p = pred[pred[:, 0].astype(int) == frame_no]
        g = gt_eval[gt_eval[:, 0].astype(int) == frame_no] if (args.gt and gt_eval is not None) else None
        img = draw_frame(img, p, g, frame_no, f"{args.seq}")
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale)

        if args.switches_only:
            rendered.append(img)
        else:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps or seq.frame_rate, (w, h),
                )
            writer.write(img)

    if writer is not None:
        writer.release()
        print(f"wrote {args.out} ({len(wanted)} frames @ {args.fps or seq.frame_rate} fps)")
    elif rendered:
        cols = args.cols
        rows = (len(rendered) + cols - 1) // cols
        h, w = rendered[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, tile in enumerate(rendered):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
        cv2.imwrite(str(args.out), sheet)
        print(f"wrote {args.out} ({len(rendered)} frames, {rows}x{cols} sheet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
