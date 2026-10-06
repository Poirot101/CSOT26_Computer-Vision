"""Reading and writing the MOTChallenge sequence layout.

Layout expected by this project::

    <root>/<seq>/img/000001.jpg ...
    <root>/<seq>/seqinfo.ini
    <root>/<seq>/gt.txt            # optional; present for train/ref sequences

Submission files are written with the exact 10-column format the course
requires, with ``conf`` and the three world-coordinate columns set to ``-1``.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["SequenceInfo", "read_seqinfo", "read_mot_file", "write_mot_file", "discover_sequences"]


@dataclass
class SequenceInfo:
    name: str
    path: Path
    img_dir: Path
    frame_rate: int
    length: int
    width: int
    height: int
    ext: str = ".jpg"

    @property
    def gt_path(self) -> Path:
        return self.path / "gt.txt"

    def frame_paths(self) -> list[Path]:
        """Frame files in numeric order.

        Sorted by the integer in the filename rather than lexicographically:
        zero-padding is conventional but not guaranteed, and a single
        unpadded name would silently reorder the sequence.
        """
        files = sorted(
            self.img_dir.glob(f"*{self.ext}"),
            key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
        )
        return files


def read_seqinfo(seq_dir: str | Path) -> SequenceInfo:
    """Parse ``seqinfo.ini``, falling back to counting files if it is absent."""
    seq_dir = Path(seq_dir)
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        parser = configparser.ConfigParser()
        parser.read(ini)
        s = parser["Sequence"]
        img_dir = seq_dir / s.get("imDir", "img")
        return SequenceInfo(
            name=s.get("name", seq_dir.name),
            path=seq_dir,
            img_dir=img_dir,
            frame_rate=int(s.get("frameRate", 30)),
            length=int(s.get("seqLength", 0)),
            width=int(s.get("imWidth", 0)),
            height=int(s.get("imHeight", 0)),
            ext=s.get("imExt", ".jpg"),
        )

    img_dir = seq_dir / "img"
    frames = sorted(img_dir.glob("*.jpg"))
    width = height = 0
    if frames:
        from PIL import Image

        with Image.open(frames[0]) as im:
            width, height = im.size
    return SequenceInfo(
        name=seq_dir.name,
        path=seq_dir,
        img_dir=img_dir,
        frame_rate=30,
        length=len(frames),
        width=width,
        height=height,
    )


def read_mot_file(path: str | Path) -> np.ndarray:
    """Read a comma-separated MOT file into a float array.

    Returns an empty ``(0, 10)`` array for a missing or empty file so that
    callers can treat "no detections" uniformly.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    return rows


def write_mot_file(path: str | Path, records: list[tuple] | np.ndarray) -> int:
    """Write tracking output in the required 10-column submission format.

    Each record is ``(frame, track_id, left, top, width, height)``. Coordinates
    are written with two decimals; frame and id as integers, both 1-based.
    Returns the number of lines written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = np.asarray(records, dtype=np.float64).reshape(-1, 6)
    order = np.lexsort((records[:, 1], records[:, 0]))
    records = records[order]
    with path.open("w", encoding="utf-8") as fh:
        for frame, tid, left, top, width, height in records:
            fh.write(
                f"{int(frame)},{int(tid)},{left:.2f},{top:.2f},{width:.2f},{height:.2f},-1,-1,-1,-1\n"
            )
    return len(records)


def discover_sequences(root: str | Path, names: list[str] | None = None) -> list[SequenceInfo]:
    """Find sequence directories under ``root``."""
    root = Path(root)
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / "img").is_dir())
    if names:
        wanted = set(names)
        candidates = [p for p in candidates if p.name in wanted]
    return [read_seqinfo(p) for p in candidates]
