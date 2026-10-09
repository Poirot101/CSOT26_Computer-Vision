"""Top-level evaluation: run all metrics and compute the leaderboard score."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._data import SequenceData, build_sequence_data
from .clear import clear_sequence, combine_clear
from .hota import combine_hota, hota_sequence
from .identity import combine_identity, identity_sequence

__all__ = ["SequenceResult", "evaluate", "leaderboard_score"]


def leaderboard_score(idf1: float, hota: float) -> float:
    """The course metric: ``(2 * IDF1 + HOTA) / 3``.

    IDF1 carries two thirds of the weight, so association quality matters twice
    as much as the balanced HOTA. Any tuning decision should be judged against
    this, not against MOTA.
    """
    return (2.0 * idf1 + hota) / 3.0


@dataclass
class SequenceResult:
    name: str
    hota: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    clear: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, float]:
        hota_mean = float(np.mean(self.hota["HOTA"]))
        return {
            "HOTA": hota_mean,
            "DetA": float(np.mean(self.hota["DetA"])),
            "AssA": float(np.mean(self.hota["AssA"])),
            "IDF1": self.identity["IDF1"],
            "IDP": self.identity["IDP"],
            "IDR": self.identity["IDR"],
            "MOTA": self.clear["MOTA"],
            "MOTP": self.clear["MOTP"],
            "IDSW": self.clear["IDSW"],
            "Score": leaderboard_score(self.identity["IDF1"], hota_mean),
        }


def evaluate(
    sequences: dict[str, tuple[np.ndarray, np.ndarray, int]],
    apply_preprocessing: bool = True,
) -> tuple[dict[str, SequenceResult], dict[str, float]]:
    """Evaluate a set of sequences.

    Parameters
    ----------
    sequences:
        ``{name: (gt_rows, pred_rows, num_frames)}``.
    apply_preprocessing:
        Apply MOTChallenge distractor handling (strongly recommended; see
        :mod:`cv_mot.metrics._data`).

    Returns
    -------
    ``(per_sequence_results, overall)`` where ``overall`` pools detections
    across sequences rather than averaging per-sequence scores.
    """
    results: dict[str, SequenceResult] = {}
    hota_list, id_list, clear_list = [], [], []

    for name, (gt_rows, pred_rows, num_frames) in sequences.items():
        data: SequenceData = build_sequence_data(
            gt_rows, pred_rows, num_frames, name=name, apply_preprocessing=apply_preprocessing
        )
        h = hota_sequence(data)
        i = identity_sequence(data)
        c = clear_sequence(data)
        results[name] = SequenceResult(name=name, hota=h, identity=i, clear=c)
        hota_list.append(h)
        id_list.append(i)
        clear_list.append(c)

    overall: dict[str, float] = {}
    if hota_list:
        overall.update(combine_hota(hota_list))
        overall.update(combine_identity(id_list))
        overall.update(combine_clear(clear_list))
        overall["Score"] = leaderboard_score(overall["IDF1"], overall["HOTA"])
    return results, overall
