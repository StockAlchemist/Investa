"""Tests for the nightly split-consistency check.

The check is only worth scheduling if it stays quiet. These cover the two things
that decide that: it must not fire on ordinary trading, and it must not re-report
what it already told you.
"""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import check_split_consistency as chk  # noqa: E402


def _db(path, bars, splits):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE daily_ohlcv (symbol TEXT, date TEXT, close REAL,
                                  interval TEXT DEFAULT '1d',
                                  PRIMARY KEY (symbol, date, interval));
        CREATE TABLE corporate_action (symbol TEXT, date TEXT, kind TEXT,
                                       value REAL, PRIMARY KEY (symbol, date, kind));
        """
    )
    conn.executemany(
        "INSERT INTO daily_ohlcv (symbol, date, close) VALUES (?, ?, ?)", bars
    )
    conn.executemany(
        "INSERT INTO corporate_action (symbol, date, kind, value) VALUES (?, ?, 'split', ?)",
        splits,
    )
    conn.commit()
    return conn


def _series(symbol, start_day, closes):
    return [(symbol, f"2026-07-{start_day + i:02d}", c) for i, c in enumerate(closes)]


def test_a_properly_applied_split_is_silent(tmp_path):
    """
    The whole premise: an applied split is invisible in the series. If this
    fired, the check would report every split ever and be worthless.
    """
    path = str(tmp_path / "m.db")
    conn = _db(path, _series("OK", 1, [10.0, 10.1, 10.2, 10.15, 10.3]), [("OK", "2026-07-03", 4.0)])
    try:
        assert chk.check(conn, None) == []
    finally:
        conn.close()


def test_an_unapplied_split_is_caught_on_the_ex_date(tmp_path):
    path = str(tmp_path / "m.db")
    # 40 -> 10 across the ex-date: the 4:1 is sitting in the data.
    conn = _db(
        path,
        _series("BAD", 1, [40.0, 40.4, 10.0, 10.1, 10.2]),
        [("BAD", "2026-07-03", 4.0)],
    )
    try:
        found = chk.check(conn, None)
        assert [f.shape for f in found] == ["unapplied"]
        assert found[0].symbol == "BAD"
    finally:
        conn.close()


def test_a_ratio_step_far_from_the_ex_date_is_labelled_differently(tmp_path):
    """
    A step matching the ratio three weeks from the ex-date is the interleaved
    shape, not an unapplied split. Calling it 'unapplied' sends the reader
    looking for a boundary that is not there.
    """
    path = str(tmp_path / "m.db")
    conn = _db(
        path,
        _series("MIX", 1, [40.0, 40.4, 10.0, 10.1, 10.2, 10.3, 10.4]),
        [("MIX", "2026-07-25", 4.0)],
    )
    try:
        shapes = {f.shape for f in chk.check(conn, None)}
        assert "unapplied" not in shapes
        assert "ratio-step-off-ex-date" in shapes
    finally:
        conn.close()


def test_a_single_bar_on_the_wrong_basis_is_reported_as_mixed(tmp_path):
    path = str(tmp_path / "m.db")
    conn = _db(
        path,
        _series("STRAY", 1, [10.0, 10.1, 40.4, 10.2, 10.3]),
        [("STRAY", "2026-07-03", 4.0)],
    )
    try:
        found = [f for f in chk.check(conn, None) if f.shape == "mixed"]
        assert len(found) == 1
        assert "2026-07-03" in found[0].detail
    finally:
        conn.close()


def test_small_ratios_are_ignored_however_the_price_moved(tmp_path):
    """
    A 1.05 ratio matches any day the stock moved 5%. Flagging those produced 515
    false positives out of 586 on the first attempt, five on held symbols.
    """
    path = str(tmp_path / "m.db")
    conn = _db(
        path,
        _series("SMALL", 1, [10.5, 10.0, 10.4, 10.1, 10.6]),
        [("SMALL", "2026-07-03", 1.05)],
    )
    try:
        assert chk.check(conn, None) == []
    finally:
        conn.close()


# --- the seen-state, which is what makes it schedulable --------------------


def test_findings_are_reported_once(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setattr(chk, "state_path", lambda: state_file)

    finding = chk.Finding("BAD", "2026-07-03", 4.0, "unapplied", "detail")
    assert chk.load_state() == {}

    chk.save_state({finding.key: finding.ex_date})
    assert finding.key in chk.load_state()


def test_the_key_separates_shapes_on_one_symbol(tmp_path):
    """
    A symbol can be both unapplied and mixed. Collapsing them would hide the
    second one for as long as the first stayed unfixed.
    """
    a = chk.Finding("X", "2026-07-03", 4.0, "unapplied", "")
    b = chk.Finding("X", "2026-07-03", 4.0, "mixed", "")
    assert a.key != b.key


def test_a_corrupt_state_file_does_not_break_the_run(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    with open(state_file, "w") as fh:
        fh.write("{not json")
    monkeypatch.setattr(chk, "state_path", lambda: state_file)
    assert chk.load_state() == {}
