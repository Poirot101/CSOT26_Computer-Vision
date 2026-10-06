"""Command line interface.

::

    python -m cv_mot detect   --data <root> --weights yolov8s.pt --imgsz 1280
    python -m cv_mot track    --data <root> --dets cache/ --out outputs/
    python -m cv_mot evaluate --data <root> --preds outputs/
    python -m cv_mot tune     --data <root> --dets cache/ --seqs ref
    python -m cv_mot ablate   --data <root> --dets cache/ --seqs ref
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .bytetrack import ByteTrackConfig, ByteTracker
from .detect import CachedDetections, GroundTruthDetector, YoloDetector, save_detections
from .metrics import evaluate, leaderboard_score
from .mot_io import discover_sequences, read_mot_file, read_seqinfo, write_mot_file
from .pipeline import run_sequence
from .sort import SortConfig, SortTracker

HEADER = f"{'sequence':<10}{'HOTA':>8}{'DetA':>8}{'AssA':>8}{'IDF1':>8}{'MOTA':>8}{'IDSW':>7}{'Score':>8}"


def _print_table(results: dict, overall: dict) -> None:
    print(HEADER)
    print("-" * len(HEADER))
    for name, res in results.items():
        s = res.summary
        print(
            f"{name:<10}{s['HOTA']:>8.4f}{s['DetA']:>8.4f}{s['AssA']:>8.4f}"
            f"{s['IDF1']:>8.4f}{s['MOTA']:>8.4f}{int(s['IDSW']):>7d}{s['Score']:>8.4f}"
        )
    if overall:
        print("-" * len(HEADER))
        print(
            f"{'OVERALL':<10}{overall['HOTA']:>8.4f}{overall['DetA']:>8.4f}{overall['AssA']:>8.4f}"
            f"{overall['IDF1']:>8.4f}{overall['MOTA']:>8.4f}{int(overall['IDSW']):>7d}{overall['Score']:>8.4f}"
        )


def _detector_from_args(args, seq_name: str):
    if getattr(args, "oracle", False):
        return GroundTruthDetector(
            recall=args.oracle_recall,
            noise_std=args.oracle_noise,
            visibility_scores=True,
        )
    if getattr(args, "dets", None):
        return CachedDetections(Path(args.dets) / f"{seq_name}.npz")
    return YoloDetector(
        weights=args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device, batch=args.batch
    )


def _config_from_args(args) -> ByteTrackConfig:
    cfg = ByteTrackConfig()
    if getattr(args, "config", None):
        cfg = replace(cfg, **json.loads(Path(args.config).read_text()))
    overrides = {
        k: getattr(args, k)
        for k in (
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "match_thresh",
            "second_match_thresh",
            "track_buffer",
            "fuse_score",
        )
        if getattr(args, k, None) is not None
    }
    return replace(cfg, **overrides)


# --------------------------------------------------------------------- detect
def cmd_detect(args) -> None:
    sequences = discover_sequences(args.data, args.seqs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(
        weights=args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device, batch=args.batch
    )
    for seq in sequences:
        target = out_dir / f"{seq.name}.npz"
        n = save_detections(target, detector, seq)
        print(f"{seq.name}: cached {n} detections ({n / max(seq.length,1):.1f}/frame) -> {target}", flush=True)


# ---------------------------------------------------------------------- track
def cmd_track(args) -> None:
    sequences = discover_sequences(args.data, args.seqs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = _config_from_args(args)

    for seq in sequences:
        cfg = replace(base_cfg, frame_rate=seq.frame_rate)
        if args.tracker == "sort":
            factory = lambda: SortTracker(SortConfig(det_thresh=cfg.track_high_thresh))
        else:
            factory = lambda c=cfg: ByteTracker(c)
        rows, stats = run_sequence(
            seq,
            _detector_from_args(args, seq.name),
            factory,
            interpolate=args.interpolate,
            interpolate_max_gap=args.interpolate_max_gap,
            progress_every=args.progress_every,
        )
        n = write_mot_file(out_dir / f"{seq.name}.txt", rows[:, :6])
        print(
            f"{seq.name}: {stats.frames} frames, {stats.detections} dets, "
            f"{n} rows (+{stats.interpolated_rows} interpolated), {stats.fps:.1f} fps",
            flush=True,
        )


# ------------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> None:
    sequences = discover_sequences(args.data, args.seqs)
    payload = {}
    for seq in sequences:
        if not seq.gt_path.exists():
            print(f"{seq.name}: no gt.txt, skipping")
            continue
        pred = read_mot_file(Path(args.preds) / f"{seq.name}.txt")
        payload[seq.name] = (read_mot_file(seq.gt_path), pred, seq.length)
    if not payload:
        print("nothing to evaluate")
        return
    results, overall = evaluate(payload, apply_preprocessing=not args.no_preprocessing)
    _print_table(results, overall)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"per_sequence": {k: v.summary for k, v in results.items()}, "overall": overall},
                indent=2,
            )
        )


# ----------------------------------------------------------------------- tune
def cmd_tune(args) -> None:
    """Grid search on sequences that have ground truth.

    Searching against cached detections keeps each trial to tracker cost alone,
    which is what makes an exhaustive sweep practical.
    """
    sequences = discover_sequences(args.data, args.seqs)
    grid = {
        "track_high_thresh": args.grid_high,
        "new_track_thresh": args.grid_new,
        "match_thresh": args.grid_match,
        "track_buffer": args.grid_buffer,
    }
    keys = list(grid)
    trials = list(itertools.product(*(grid[k] for k in keys)))
    print(f"{len(trials)} configurations over {[s.name for s in sequences]}\n")

    rows = []
    best = None
    for values in trials:
        params = dict(zip(keys, values))
        payload = {}
        for seq in sequences:
            cfg = replace(
                _config_from_args(args), frame_rate=seq.frame_rate, **params
            )
            out, _ = run_sequence(
                seq,
                _detector_from_args(args, seq.name),
                lambda c=cfg: ByteTracker(c),
                interpolate=args.interpolate,
                interpolate_max_gap=args.interpolate_max_gap,
            )
            payload[seq.name] = (read_mot_file(seq.gt_path), out, seq.length)
        _, overall = evaluate(payload)
        record = {**params, **{k: round(overall[k], 5) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "Score")}}
        rows.append(record)
        flag = ""
        if best is None or overall["Score"] > best["Score"]:
            best = record
            flag = "  <-- best"
        print(
            "  ".join(f"{k}={v}" for k, v in params.items())
            + f"   IDF1={overall['IDF1']:.4f} HOTA={overall['HOTA']:.4f} Score={overall['Score']:.4f}{flag}",
            flush=True,
        )

    print("\nbest:", json.dumps(best, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"trials": rows, "best": best}, indent=2))


# --------------------------------------------------------------------- ablate
def cmd_ablate(args) -> None:
    """Measure each design decision as a delta on the same detections."""
    sequences = discover_sequences(args.data, args.seqs)
    base = _config_from_args(args)

    variants: list[tuple[str, object, bool]] = [
        ("SORT baseline (single threshold)", SortConfig(det_thresh=base.track_high_thresh), False),
        ("ByteTrack (two-pass association)", base, False),
        ("  + score fusion", replace(base, fuse_score=True), False),
        ("  + gap interpolation", base, True),
    ]

    table = []
    for label, cfg, interp in variants:
        payload = {}
        for seq in sequences:
            if isinstance(cfg, SortConfig):
                factory = lambda c=cfg: SortTracker(c)
            else:
                factory = lambda c=replace(cfg, frame_rate=seq.frame_rate): ByteTracker(c)
            out, _ = run_sequence(
                seq,
                _detector_from_args(args, seq.name),
                factory,
                interpolate=interp,
                interpolate_max_gap=args.interpolate_max_gap,
            )
            payload[seq.name] = (read_mot_file(seq.gt_path), out, seq.length)
        _, overall = evaluate(payload)
        table.append((label, overall))
        print(
            f"{label:<38} IDF1={overall['IDF1']:.4f} HOTA={overall['HOTA']:.4f} "
            f"DetA={overall['DetA']:.4f} AssA={overall['AssA']:.4f} "
            f"IDSW={int(overall['IDSW']):<5d} Score={overall['Score']:.4f}",
            flush=True,
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps([{"variant": l, **{k: v for k, v in o.items()}} for l, o in table], indent=2)
        )


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cv_mot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--data", required=True, help="root directory containing sequence folders")
        sp.add_argument("--seqs", nargs="*", default=None, help="sequence names (default: all)")

    def detector_opts(sp):
        sp.add_argument("--dets", default=None, help="directory of cached .npz detections")
        sp.add_argument("--oracle", action="store_true", help="use ground-truth-derived detections")
        sp.add_argument("--oracle-recall", type=float, default=1.0)
        sp.add_argument("--oracle-noise", type=float, default=0.0)
        sp.add_argument("--weights", default="yolov8s.pt")
        sp.add_argument("--imgsz", type=int, default=1280)
        sp.add_argument("--conf", type=float, default=0.05)
        sp.add_argument("--device", default=None)
        sp.add_argument("--batch", type=int, default=4)

    def tracker_opts(sp):
        sp.add_argument("--config", default=None, help="JSON file of ByteTrackConfig overrides")
        sp.add_argument("--track-high-thresh", type=float, default=None)
        sp.add_argument("--track-low-thresh", type=float, default=None)
        sp.add_argument("--new-track-thresh", type=float, default=None)
        sp.add_argument("--match-thresh", type=float, default=None)
        sp.add_argument("--second-match-thresh", type=float, default=None)
        sp.add_argument("--track-buffer", type=int, default=None)
        sp.add_argument("--fuse-score", action="store_true", default=None)
        sp.add_argument("--interpolate", action="store_true")
        sp.add_argument("--interpolate-max-gap", type=int, default=20)

    sp = sub.add_parser("detect", help="run the detector and cache boxes")
    common(sp)
    sp.add_argument("--out", default="cache")
    sp.add_argument("--weights", default="yolov8s.pt")
    sp.add_argument("--imgsz", type=int, default=1280)
    sp.add_argument("--conf", type=float, default=0.05)
    sp.add_argument("--device", default=None)
    sp.add_argument("--batch", type=int, default=4)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("track", help="produce submission files")
    common(sp)
    detector_opts(sp)
    tracker_opts(sp)
    sp.add_argument("--tracker", choices=["bytetrack", "sort"], default="bytetrack")
    sp.add_argument("--out", default="outputs")
    sp.add_argument("--progress-every", type=int, default=0)
    sp.set_defaults(func=cmd_track)

    sp = sub.add_parser("evaluate", help="score predictions against gt.txt")
    common(sp)
    sp.add_argument("--preds", required=True)
    sp.add_argument("--no-preprocessing", action="store_true", help="skip MOTChallenge distractor handling")
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("tune", help="grid search tracker parameters")
    common(sp)
    detector_opts(sp)
    tracker_opts(sp)
    sp.add_argument("--grid-high", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6])
    sp.add_argument("--grid-new", type=float, nargs="*", default=[0.5, 0.6, 0.7])
    sp.add_argument("--grid-match", type=float, nargs="*", default=[0.8])
    sp.add_argument("--grid-buffer", type=int, nargs="*", default=[30])
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_tune)

    sp = sub.add_parser("ablate", help="measure each design decision")
    common(sp)
    detector_opts(sp)
    tracker_opts(sp)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_ablate)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
