"""Documentation must agree with the measurements it reports.

Every number quoted in the docs was produced by a run whose output is saved
under ``outputs/``. Those two can drift: a code change alters the results, the
JSON is regenerated, and the prose silently keeps the old figures. That happened
once in this project -- the oracle numbers in six documents were measured before
a bug fix and were stale by ~0.002 IDF1 for a while.

These tests make documentation validity mechanical rather than a thing someone
remembers to re-check:

* every metric row in ``outputs/*.json`` appears verbatim in the docs that quote it
* every relative markdown link resolves
* the writeup stays inside its word limit
* the submission files match the row and identity counts the docs claim

They are intentionally strict about *formatting* (four decimal places) because
that is how the tables are written; if a table is reformatted these tests need
updating, which is the point -- the numbers get re-checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")


def _load(name: str):
    path = OUTPUTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return json.loads(path.read_text())


def _docs(*names: str) -> dict[str, str]:
    return {n: (ROOT / n).read_text() for n in names if (ROOT / n).exists()}


def _strip_markup(line: str) -> str:
    """Remove emphasis and code markers so labels match however they are styled.

    Without this, a row written ``| **OVERALL** | ...`` fails to match the label
    ``OVERALL`` -- and a label that never matches makes the whole check *skip*
    that row rather than fail it. That is exactly the vacuous-pass failure mode
    these tests exist to prevent, so normalise first and never skip on a miss.
    """
    return line.replace("**", "").replace("*", "").replace("`", "")


def _mentions(text: str, label: str) -> list[str]:
    """Lines that reference ``label`` as a standalone token."""
    pattern = rf"(^|[|\s]){re.escape(label)}([|\s]|$)"
    return [line for line in text.splitlines() if re.search(pattern, _strip_markup(line))]


def _row_present(text: str, label: str, values: list[str]) -> bool:
    """Is there a line mentioning ``label`` that carries every value?"""
    rows = _mentions(text, label)
    return any(all(v in _strip_markup(row) for v in values) for row in rows)


# --------------------------------------------------------------- final scores
def test_final_scores_match_every_doc_that_quotes_them():
    fs = _load("final_scores.json")
    expected = {
        seq: [f"{vals[m]:.4f}" for m in METRICS]
        + [str(int(vals["IDSW"])), f"{vals['Score']:.4f}"]
        for seq, vals in fs["per_sequence"].items()
    }
    overall = fs["overall"]
    expected["OVERALL"] = [f"{overall[m]:.4f}" for m in METRICS] + [
        str(int(overall["IDSW"])),
        f"{overall['Score']:.4f}",
    ]

    problems = []
    confirmed_overall = False
    for name, text in _docs(
        "README.md", "docs/RESULTS.md", "docs/VERIFICATION.md", "docs/BUILD_LOG.md"
    ).items():
        for label, values in expected.items():
            if not _mentions(text, label):
                continue  # this doc genuinely does not report that sequence
            if not _row_present(text, label, values):
                problems.append(f"{name}: row {label!r} does not carry {values}")
            elif label == "OVERALL":
                confirmed_overall = True
    assert not problems, "stale numbers in docs:\n" + "\n".join(problems)
    assert confirmed_overall, (
        "no document carries the OVERALL row -- the check would pass vacuously"
    )


# ------------------------------------------------------------------ ablations
@pytest.mark.parametrize("blob", ["ablation_yolo.json", "ablation_oracle.json"])
def test_ablation_rows_appear_in_results(blob):
    rows = _load(blob)
    text = (ROOT / "docs" / "RESULTS.md").read_text()
    missing = []
    for row in rows:
        values = [f"{row[m]:.4f}" for m in ("IDF1", "HOTA", "DetA", "AssA")] + [
            f"{row['Score']:.4f}"
        ]
        if not any(all(v in _strip_markup(line) for v in values) for line in text.splitlines()):
            missing.append(f"{blob}: {row['variant'].strip()!r} -> {values}")
    assert not missing, "ablation rows absent from docs/RESULTS.md:\n" + "\n".join(missing)


# -------------------------------------------------------- oracle experiments
def test_oracle_ceiling_and_sweep_are_quoted_consistently():
    oe = _load("oracle_experiments.json")
    uniform = f"{oe['oracle_ceiling']['uniform']['IDF1']:.4f}"
    realistic = f"{oe['oracle_ceiling']['visibility']['IDF1']:.4f}"
    peak = f"{max(t['Score'] for t in oe['threshold_sweep']):.4f}"

    docs = _docs(
        "docs/RESULTS.md",
        "docs/LEARNINGS.md",
        "docs/BUILD_LOG.md",
        "docs/weeks/week5.md",
        "docs/weeks/final-project.md",
    )
    for value, label in ((uniform, "oracle ceiling IDF1"), (realistic, "realistic-confidence IDF1")):
        assert any(value in t for t in docs.values()), f"{label} {value} quoted nowhere"

    # Wherever the sweep peak is discussed, it must be the current peak.
    results = docs["docs/RESULTS.md"]
    assert peak in results, f"sweep peak {peak} missing from RESULTS.md"


def test_confidence_cost_claim_matches_the_measurement():
    oe = _load("oracle_experiments.json")
    cost = oe["confidence_cost_idf1"]
    text = (ROOT / "docs" / "RESULTS.md").read_text()
    # Docs state this as a points figure, e.g. "24.2 IDF1 points".
    stated = re.findall(r"(\d+\.\d)\s+IDF1 points", text)
    assert stated, "RESULTS.md no longer states the confidence cost in IDF1 points"
    assert any(abs(float(s) - cost * 100) < 0.1 for s in stated), (
        f"RESULTS.md states {stated} IDF1 points; measured {cost * 100:.1f}"
    )


# ------------------------------------------------------------------- hygiene
def test_every_relative_markdown_link_resolves():
    broken = []
    for md in sorted(ROOT.rglob("*.md")):
        if ".git" in md.parts:
            continue
        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", md.read_text()):
            target = match.group(2).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).resolve().exists():
                broken.append(f"{md.relative_to(ROOT)}: {target}")
    assert not broken, "broken relative links:\n" + "\n".join(broken)


def test_writeup_is_within_the_word_limit():
    words = (ROOT / "writeup.txt").read_text().split()
    assert len(words) <= 200, f"writeup.txt is {len(words)} words, limit is 200"


def test_submission_files_match_documented_counts():
    """Row and identity counts quoted in the docs must match the files."""
    results = (ROOT / "docs" / "RESULTS.md").read_text()
    quoted = dict(
        (seq, (int(rows), int(ids)))
        for seq, rows, ids in re.findall(
            r"OK\s+(\w+)\.txt\s+(\d+) rows, frames [\d-]+, (\d+) identities", results
        )
    )
    if not quoted:
        pytest.skip("RESULTS.md does not quote validator output")

    for seq, (rows_claimed, ids_claimed) in quoted.items():
        path = OUTPUTS / f"{seq}.txt"
        if not path.exists():
            pytest.skip(f"{seq}.txt not generated")
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        assert len(data) == rows_claimed, f"{seq}.txt has {len(data)} rows, docs say {rows_claimed}"
        n_ids = len(np.unique(data[:, 1]))
        assert n_ids == ids_claimed, f"{seq}.txt has {n_ids} identities, docs say {ids_claimed}"


def test_final_config_is_the_one_documented_in_tuning():
    """configs/final.json must match the table in docs/TUNING.md."""
    cfg = json.loads((ROOT / "configs" / "final.json").read_text())
    text = (ROOT / "docs" / "TUNING.md").read_text()
    for key, value in cfg.items():
        # _mentions strips backticks from the line, so the label must be bare.
        rendered = "false" if value is False else ("true" if value is True else str(value))
        assert _row_present(text, key, [rendered]), (
            f"docs/TUNING.md does not document {key} = {value}"
        )
