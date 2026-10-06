"""Detection sources.

Two implementations share one interface:

:class:`YoloDetector`
    The real pipeline -- Ultralytics YOLO restricted to the ``person`` class.

:class:`GroundTruthDetector`
    An *oracle* that synthesises detections from ``gt.txt`` with controllable
    recall, localisation noise and score distribution.

The oracle is not a shortcut around building the real thing; it answers a
question the real detector cannot. Tracking error has two independent sources --
the detector missing people, and the association logic assigning them wrongly.
Feeding the tracker known-quality detections isolates the second, so a change to
the association code can be measured without re-running inference and without
detector noise masking the effect. Every number reported from the oracle is
labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .mot_io import SequenceInfo, read_mot_file

__all__ = ["Detections", "Detector", "YoloDetector", "GroundTruthDetector", "CachedDetections"]

COCO_PERSON_CLASS = 0


@dataclass
class Detections:
    """One frame of detections."""

    boxes_tlwh: np.ndarray
    scores: np.ndarray
    features: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.scores)


class Detector(Protocol):
    """Anything that can yield per-frame detections for a sequence."""

    def run(self, seq: SequenceInfo) -> Iterator[Detections]: ...


class YoloDetector:
    """Ultralytics YOLO restricted to pedestrians.

    Parameters
    ----------
    weights:
        Checkpoint name or path. Larger backbones raise recall on small,
        distant pedestrians, which is exactly where dense sequences lose points.
    imgsz:
        Inference resolution. Raising this is usually the single most effective
        change for crowds: a person 30 px tall in a 1080p frame is 18 px at 640
        and is frequently missed, but is resolvable at 1280.
    conf:
        Detector-side floor. Kept deliberately *below* the tracker's own
        thresholds so that ByteTrack's second association pass has low-score
        boxes to work with -- filtering them here would defeat the method.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.05,
        iou: float = 0.7,
        device: str | None = None,
        max_det: int = 1000,
        batch: int = 1,
        verbose: bool = False,
    ) -> None:
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.max_det = max_det
        self.batch = max(1, int(batch))
        self.verbose = verbose
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    def run(self, seq: SequenceInfo) -> Iterator[Detections]:
        model = self._load()
        frames = seq.frame_paths()
        for start in range(0, len(frames), self.batch):
            chunk = [str(p) for p in frames[start : start + self.batch]]
            results = model.predict(
                chunk,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=[COCO_PERSON_CLASS],
                device=self.device,
                max_det=self.max_det,
                verbose=self.verbose,
            )
            for res in results:
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    yield Detections(np.zeros((0, 4)), np.zeros(0))
                    continue
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
                scores = boxes.conf.cpu().numpy().astype(np.float64)
                tlwh = xyxy.copy()
                tlwh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
                tlwh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
                yield Detections(tlwh, scores)


class GroundTruthDetector:
    """Oracle detector derived from ``gt.txt``.

    Simulates a detector of a chosen quality so that association logic can be
    measured in isolation. Only pedestrian-class, non-ignored annotations are
    used, matching what the evaluator scores.

    Parameters
    ----------
    recall:
        Fraction of annotations emitted per frame.
    noise_std:
        Gaussian jitter on box coordinates, as a fraction of box height.
    visibility_scores:
        When ``True``, a detection's confidence is its annotated visibility.
        This is the realistic setting: it reproduces the actual phenomenon
        ByteTrack targets -- occluded people scoring low -- instead of assigning
        uniform confidence.
    drop_invisible_below:
        Annotations less visible than this are omitted entirely, standing in for
        a detector that cannot see a near-fully-occluded person at all.
    """

    def __init__(
        self,
        recall: float = 1.0,
        noise_std: float = 0.0,
        visibility_scores: bool = True,
        drop_invisible_below: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.recall = float(recall)
        self.noise_std = float(noise_std)
        self.visibility_scores = visibility_scores
        self.drop_invisible_below = float(drop_invisible_below)
        self.seed = int(seed)

    def run(self, seq: SequenceInfo) -> Iterator[Detections]:
        rows = read_mot_file(seq.gt_path)
        rng = np.random.default_rng(self.seed)

        keep = (rows[:, 6] != 0) & (rows[:, 7].astype(int) == 1)
        rows = rows[keep]
        if self.drop_invisible_below > 0:
            rows = rows[rows[:, 8] >= self.drop_invisible_below]

        by_frame: dict[int, np.ndarray] = {}
        for f in np.unique(rows[:, 0]).astype(int):
            by_frame[f] = rows[rows[:, 0].astype(int) == f]

        for frame in range(1, seq.length + 1):
            block = by_frame.get(frame)
            if block is None or len(block) == 0:
                yield Detections(np.zeros((0, 4)), np.zeros(0))
                continue

            n = len(block)
            sel = rng.random(n) < self.recall
            block = block[sel]
            if len(block) == 0:
                yield Detections(np.zeros((0, 4)), np.zeros(0))
                continue

            boxes = block[:, 2:6].copy()
            if self.noise_std > 0:
                scale = (boxes[:, 3:4] * self.noise_std)
                boxes = boxes + rng.normal(0.0, 1.0, boxes.shape) * scale
                boxes[:, 2:4] = np.maximum(boxes[:, 2:4], 1.0)

            if self.visibility_scores:
                vis = np.clip(block[:, 8], 0.0, 1.0)
                # Map visibility onto a plausible confidence range: a fully
                # visible person reads ~0.95, a barely-visible one ~0.15.
                scores = 0.15 + 0.80 * vis
            else:
                scores = np.full(len(block), 0.9)
            yield Detections(boxes, scores)


class CachedDetections:
    """Replay detections saved by :func:`save_detections`.

    Detector inference dominates runtime, so sweeps run against a cache: the
    detector is executed once per configuration and the tracker is then tuned
    over hundreds of parameter combinations at no additional inference cost.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def run(self, seq: SequenceInfo) -> Iterator[Detections]:
        with np.load(self.path, allow_pickle=False) as blob:
            boxes = blob["boxes"]
            scores = blob["scores"]
            frames = blob["frames"].astype(int)
        for frame in range(1, seq.length + 1):
            mask = frames == frame
            yield Detections(boxes[mask], scores[mask])


def save_detections(path: str | Path, detector: Detector, seq: SequenceInfo) -> int:
    """Run a detector once and persist every box for later reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_boxes, all_scores, all_frames = [], [], []
    for frame_idx, det in enumerate(detector.run(seq), start=1):
        if len(det):
            all_boxes.append(det.boxes_tlwh)
            all_scores.append(det.scores)
            all_frames.append(np.full(len(det), frame_idx))
    boxes = np.vstack(all_boxes) if all_boxes else np.zeros((0, 4))
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    frames = np.concatenate(all_frames) if all_frames else np.zeros(0)
    np.savez_compressed(path, boxes=boxes, scores=scores, frames=frames)
    return len(scores)
