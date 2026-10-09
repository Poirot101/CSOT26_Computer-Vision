"""Submission format and post-processing."""

import numpy as np
import pytest

from cv_mot.interpolate import interpolate_tracks
from cv_mot.mot_io import read_mot_file, read_seqinfo, write_mot_file


def test_submission_format_is_exactly_ten_columns(tmp_path):
    rows = np.array([[1, 1, 10.5, 20.25, 30.0, 40.0], [2, 1, 11.0, 21.0, 30.0, 40.0]])
    path = tmp_path / "01.txt"
    write_mot_file(path, rows)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parts = line.split(",")
        assert len(parts) == 10
        assert parts[6:] == ["-1", "-1", "-1", "-1"]
        int(parts[0]), int(parts[1])  # frame and id must be integers


def test_rows_are_sorted_by_frame_then_id(tmp_path):
    rows = np.array([[2, 5, 0, 0, 1, 1], [1, 9, 0, 0, 1, 1], [1, 2, 0, 0, 1, 1]])
    path = tmp_path / "x.txt"
    write_mot_file(path, rows)
    out = read_mot_file(path)
    assert out[:, 0].tolist() == [1, 1, 2]
    assert out[:2, 1].tolist() == [2, 9]


def test_missing_file_reads_as_empty(tmp_path):
    assert read_mot_file(tmp_path / "nope.txt").shape == (0, 10)


def test_empty_file_reads_as_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert read_mot_file(p).shape == (0, 10)


def test_round_trip_preserves_coordinates(tmp_path):
    rows = np.array([[1, 7, 123.46, 234.57, 50.5, 120.25]])
    p = tmp_path / "r.txt"
    write_mot_file(p, rows)
    back = read_mot_file(p)
    assert np.allclose(back[0, :6], rows[0], atol=0.01)


def test_seqinfo_is_parsed(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "seqinfo.ini").write_text(
        "[Sequence]\nname=07\nimDir=img\nframeRate=25\nseqLength=120\nimWidth=1280\nimHeight=720\nimExt=.jpg\n"
    )
    info = read_seqinfo(tmp_path)
    assert (info.name, info.frame_rate, info.length, info.width, info.height) == ("07", 25, 120, 1280, 720)


def test_interpolation_fills_short_gaps_linearly():
    rows = np.array([[1, 1, 0, 0, 10, 20], [5, 1, 40, 0, 10, 20]], dtype=float)
    out = interpolate_tracks(rows, max_gap=10)
    assert out[:, 0].astype(int).tolist() == [1, 2, 3, 4, 5]
    assert out[out[:, 0] == 3][0, 2] == pytest.approx(20.0)


def test_interpolation_refuses_long_gaps():
    rows = np.array([[1, 1, 0, 0, 10, 20], [100, 1, 40, 0, 10, 20]], dtype=float)
    out = interpolate_tracks(rows, max_gap=20)
    assert len(out) == 2


def test_interpolation_never_invents_an_identity():
    rows = np.array([[1, 1, 0, 0, 10, 20], [5, 1, 40, 0, 10, 20], [3, 2, 5, 5, 10, 20]], dtype=float)
    out = interpolate_tracks(rows, max_gap=10)
    assert set(out[:, 1].astype(int)) == {1, 2}


def test_interpolation_is_idempotent():
    rows = np.array([[1, 1, 0, 0, 10, 20], [5, 1, 40, 0, 10, 20]], dtype=float)
    once = interpolate_tracks(rows, max_gap=10)
    twice = interpolate_tracks(once, max_gap=10)
    assert len(once) == len(twice)


def test_interpolation_handles_empty_input():
    assert interpolate_tracks(np.zeros((0, 6))).shape[0] == 0
