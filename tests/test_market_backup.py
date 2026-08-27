"""Tests for archive snapshot and restore.

An untested restore path is a belief about a backup, not a backup. These run
against throwaway databases and exercise the whole round trip, plus the two
guards that stop a restore from being worse than the failure it recovers from.
"""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import backup_market_archive as backup  # noqa: E402
import restore_market_archive as restore  # noqa: E402


def _make_db(path: str, *, intraday_rows: int = 5) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE daily_ohlcv (symbol TEXT, date TEXT, close REAL,
                                  PRIMARY KEY (symbol, date));
        CREATE TABLE corporate_action (symbol TEXT, date TEXT, kind TEXT,
                                       value REAL, PRIMARY KEY (symbol, date, kind));
        CREATE TABLE fund_nav (fund_code TEXT, date TEXT, nav REAL,
                               PRIMARY KEY (fund_code, date));
        CREATE TABLE intraday_ohlcv (symbol TEXT, timestamp TEXT, close REAL,
                                     PRIMARY KEY (symbol, timestamp));
        """
    )
    conn.executemany(
        "INSERT INTO daily_ohlcv VALUES (?, ?, ?)",
        [("NLY", f"2002-06-{d:02d}", 77.6 + d) for d in range(1, 20)],
    )
    conn.execute(
        "INSERT INTO corporate_action VALUES ('AAPL','2020-08-31','split',4.0)"
    )
    conn.execute("INSERT INTO fund_nav VALUES ('SCBRM1','2002-02-15',9.9995)")
    conn.executemany(
        "INSERT INTO intraday_ohlcv VALUES (?, ?, ?)",
        [
            ("AAPL", f"2026-08-25T10:{m:02d}:00", 300.0 + m)
            for m in range(intraday_rows)
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def source_db(tmp_path):
    path = str(tmp_path / "market_data.db")
    _make_db(path)
    return path


# --- snapshot --------------------------------------------------------------


def test_core_snapshot_drops_intraday_and_keeps_everything_else(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    archive, manifest = backup.build_archive(source_db, dest, "core", "20260825_000000")

    assert os.path.exists(archive)
    assert manifest["tables"]["daily_ohlcv"] == 19
    assert manifest["tables"]["corporate_action"] == 1
    assert manifest["tables"]["fund_nav"] == 1
    assert "intraday_ohlcv" not in manifest["tables"]
    assert manifest["excluded_tables"] == ["intraday_ohlcv"]


def test_full_snapshot_keeps_intraday(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    _, manifest = backup.build_archive(source_db, dest, "full", "20260825_000000")

    assert manifest["tables"]["intraday_ohlcv"] == 5
    assert manifest["excluded_tables"] == []


def test_snapshot_records_a_checksum(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    _, manifest = backup.build_archive(source_db, dest, "core", "20260825_000000")
    assert len(manifest["sha256_snapshot"]) == 64


# --- round trip ------------------------------------------------------------


def test_restore_reproduces_the_data(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    archive, _ = backup.build_archive(source_db, dest, "core", "20260825_000000")

    target = str(tmp_path / "restored" / "market_data.db")
    assert restore.restore(archive, target, force=False) == 0

    # Not mode=ro: the restore leaves the file in WAL, and a strict read-only
    # URI cannot open a WAL database that has no -shm alongside it.
    conn = sqlite3.connect(target)
    try:
        assert conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0] == 19
        assert conn.execute(
            "SELECT close FROM daily_ohlcv WHERE symbol='NLY' AND date='2002-06-01'"
        ).fetchone()[0] == pytest.approx(78.6)
        assert conn.execute(
            "SELECT nav FROM fund_nav WHERE fund_code='SCBRM1'"
        ).fetchone()[0] == pytest.approx(9.9995)
    finally:
        conn.close()


# --- guards ----------------------------------------------------------------


def test_restore_refuses_to_overwrite_without_force(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    archive, _ = backup.build_archive(source_db, dest, "core", "20260825_000000")

    existing = str(tmp_path / "live.db")
    _make_db(existing)
    before = os.path.getsize(existing)

    assert restore.restore(archive, existing, force=False) == 1
    assert os.path.getsize(existing) == before  # untouched


def test_forced_restore_copies_the_old_database_aside(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    archive, _ = backup.build_archive(source_db, dest, "core", "20260825_000000")

    existing = str(tmp_path / "live.db")
    _make_db(existing)

    assert restore.restore(archive, existing, force=True) == 0
    aside = [p for p in os.listdir(tmp_path) if p.startswith("live.db.replaced_")]
    assert len(aside) == 1, "the replaced database must be kept"


def test_a_corrupt_snapshot_is_not_installed(source_db, tmp_path, monkeypatch):
    dest = str(tmp_path / "backups")
    archive, _ = backup.build_archive(source_db, dest, "core", "20260825_000000")

    monkeypatch.setattr(
        restore.sqlite3,
        "connect",
        lambda *a, **k: (_ for _ in ()).throw(
            sqlite3.DatabaseError("file is not a database")
        ),
    )
    target = str(tmp_path / "restored.db")
    with pytest.raises(sqlite3.DatabaseError):
        restore.restore(archive, target, force=False)
    assert not os.path.exists(target)


# --- rotation --------------------------------------------------------------


def test_rotation_keeps_the_newest_and_drops_the_rest(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    for day in range(1, 6):
        backup.build_archive(source_db, dest, "core", f"2026082{day}_000000")

    dropped = backup.rotate(dest, "core", retain=2, dry_run=False)
    remaining = sorted(os.listdir(dest))

    assert len(dropped) == 3
    assert len(remaining) == 2
    assert remaining[-1].endswith("20260825_000000.tar.gz")


def test_rotation_never_empties_the_directory(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    backup.build_archive(source_db, dest, "core", "20260825_000000")

    backup.rotate(dest, "core", retain=0, dry_run=False)
    assert len(os.listdir(dest)) == 1


def test_rotation_ignores_the_other_mode(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    backup.build_archive(source_db, dest, "full", "20260820_000000")
    for day in (21, 22, 23):
        backup.build_archive(source_db, dest, "core", f"202608{day}_000000")

    backup.rotate(dest, "core", retain=1, dry_run=False)
    remaining = os.listdir(dest)

    assert any("_full_" in f for f in remaining), (
        "a core rotation must not touch full snapshots"
    )
    assert sum("_core_" in f for f in remaining) == 1


# --- incremental -----------------------------------------------------------


def test_incremental_keeps_small_tables_whole_and_trims_bars(
    source_db, tmp_path, monkeypatch
):
    """
    The point of the mode: bounded by the delta, not by the archive. The small
    tables are the irreplaceable part and are tiny, so they ride along in full.
    """
    monkeypatch.setattr(backup, "INCREMENTAL_DAYS", 7)
    dest = str(tmp_path / "backups")
    _, manifest = backup.build_archive(
        source_db, dest, "incremental", "20260826_000000"
    )

    # 2002 bars are far outside the window; the actions and NAVs are not trimmed.
    assert manifest["tables"]["daily_ohlcv"] == 0
    assert manifest["tables"]["corporate_action"] == 1
    assert manifest["tables"]["fund_nav"] == 1
    assert "intraday_ohlcv" not in manifest["tables"]
    assert manifest["bar_window_days"] == 7


def test_incremental_keeps_bars_inside_the_window(source_db, tmp_path, monkeypatch):
    import datetime as _dt

    conn = sqlite3.connect(source_db)
    recent = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
    conn.execute("INSERT INTO daily_ohlcv VALUES ('NLY', ?, 99.0)", (recent,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(backup, "INCREMENTAL_DAYS", 7)
    dest = str(tmp_path / "backups")
    _, manifest = backup.build_archive(
        source_db, dest, "incremental", "20260826_000000"
    )
    assert manifest["tables"]["daily_ohlcv"] == 1


def test_incremental_is_far_smaller_than_core(source_db, tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "INCREMENTAL_DAYS", 7)
    dest = str(tmp_path / "backups")
    inc, _ = backup.build_archive(source_db, dest, "incremental", "20260826_000000")
    core, _ = backup.build_archive(source_db, dest, "core", "20260826_000001")
    assert os.path.getsize(inc) <= os.path.getsize(core)


def test_rotation_is_per_mode_across_all_three(source_db, tmp_path):
    dest = str(tmp_path / "backups")
    backup.build_archive(source_db, dest, "full", "20260820_000000")
    backup.build_archive(source_db, dest, "core", "20260821_000000")
    for day in (22, 23, 24):
        backup.build_archive(source_db, dest, "incremental", f"202608{day}_000000")

    backup.rotate(dest, "incremental", retain=1, dry_run=False)
    names = os.listdir(dest)
    assert sum("_incremental_" in f for f in names) == 1
    assert sum("_core_" in f for f in names) == 1
    assert sum("_full_" in f for f in names) == 1


def test_restore_re_enables_wal(source_db, tmp_path):
    """
    VACUUM INTO writes a `delete`-journal database, so a restored file arrives
    without WAL however the original was set up. Under DELETE journalling one
    writer blocks every reader — a live restore here left a background worker
    locking the archive out of every other process.
    """
    dest = str(tmp_path / "backups")
    archive, _ = backup.build_archive(source_db, dest, "core", "20260826_000000")

    # The snapshot itself is delete-journalled; that is what makes this matter.
    target = str(tmp_path / "restored" / "market_data.db")
    assert restore.restore(archive, target, force=False) == 0

    conn = sqlite3.connect(target)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()
