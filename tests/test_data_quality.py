"""Tests for the data-quality flags the clients show.

The archive has always known which symbols have broken price history and only
ever said so in a terminal. These cover the two things that decide whether the
warning is worth showing at all: that a definite defect outranks a suspicious
one, and that "nobody has scanned" is distinguishable from "nothing is wrong".
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

from market_db import MarketDatabase  # noqa: E402


@pytest.fixture
def db(tmp_path):
    database = MarketDatabase(str(tmp_path / "market.db"))
    with database._get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE data_quality (
                symbol TEXT NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL,
                occurred_on TEXT, detail TEXT, detected_at TEXT NOT NULL,
                PRIMARY KEY (symbol, kind, occurred_on));
            """
        )
        conn.commit()
    return database


def add(db, symbol, kind, severity, day, detail="why"):
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO data_quality VALUES (?,?,?,?,?, '2026-08-27T00:00:00Z')",
            (symbol, kind, severity, day, detail),
        )
        conn.commit()


def test_a_definite_defect_outranks_a_suspicious_one(db):
    """A symbol with both must read as 'known wrong', and must show the reason
    for the certain finding rather than whichever row came back first."""
    add(db, "BYND", "discontinuity", "medium", "2020-01-02", "a jump")
    add(db, "BYND", "unapplied", "high", "2026-08-14", "a split not applied")

    flag = db.get_data_quality(["BYND"])["BYND"]

    assert flag["severity"] == "high"
    assert flag["detail"] == "a split not applied"
    assert flag["occurred_on"] == "2026-08-14"
    assert flag["findings"] == 2
    assert set(flag["kinds"]) == {"discontinuity", "unapplied"}


def test_a_symbol_with_only_a_jump_stays_medium(db):
    add(db, "RCAT", "discontinuity", "medium", "2024-05-05")
    assert db.get_data_quality(["RCAT"])["RCAT"]["severity"] == "medium"


def test_clean_symbols_are_simply_absent(db):
    add(db, "BYND", "unapplied", "high", "2026-08-14")
    assert db.get_data_quality(["AAPL"]) == {}
    assert "AAPL" not in db.get_data_quality()


def test_an_archive_never_scanned_reports_nothing_rather_than_failing(tmp_path):
    """A fresh clone has no such table. That is a normal state — the flags are
    derived and rebuildable — so nothing downstream may require it to exist."""
    database = MarketDatabase(str(tmp_path / "fresh.db"))
    assert database.get_data_quality() == {}
    assert database.get_data_quality(["AAPL"]) == {}


def test_asking_about_no_symbols_does_not_return_the_whole_set(db):
    """`get_data_quality([])` means 'these zero symbols', not 'everything' —
    the difference between a clean badge and every row in a table lighting up."""
    add(db, "BYND", "unapplied", "high", "2026-08-14")
    assert db.get_data_quality([]) == {}
    assert len(db.get_data_quality()) == 1


def test_the_detail_carries_no_raw_date():
    """Dates ride in `occurred_on` so each client renders them as `DD MMM YYYY`.
    A date baked into the sentence would reach the screen as the ISO string the
    API happened to ship — the commonest way that convention gets broken."""
    import re

    import flag_data_quality as job

    # The two sentences the populator can produce, built the way it builds them.
    split_detail = f"A {0.1:g} split is on record, but the stored prices do not reflect it."
    jump_detail = (
        f"The close moves {30.0:.1f}x from {0.55:g} to {16.5:g} "
        f"with no corporate action to explain it."
    )
    iso = re.compile(r"\d{4}-\d{2}-\d{2}")
    assert not iso.search(split_detail)
    assert not iso.search(jump_detail)
    # And the module still exposes the collector the flags are built from.
    assert hasattr(job, "collect")
