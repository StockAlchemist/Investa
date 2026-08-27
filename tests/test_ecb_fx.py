"""Tests for the ECB reference-rate provider and the FX store it writes to.

Three things here would fail silently in production if they were wrong.

*The cross.* The ECB quotes everything per euro; the archive stores everything
per US dollar. Get the division backwards and `THB=X` reads 0.03 instead of
32.7 — a rate that is still a plausible-looking number, still writes cleanly,
and values a Thai holding a thousandfold wrong.

*The fill-only contract.* The ECB fixes at 14:15 CET and Yahoo takes a close, so
they disagree by ~0.2% on an ordinary day. If a fill ever overwrote a stored
rate, every historical portfolio figure would move by that much for no gain.

*The numpy coercion.* sqlite3 has no adapter for a numpy scalar, but numpy
scalars implement the buffer protocol, so one stores as a BLOB rather than
raising — which is how `USD=X` came to hold its synthetic 1 as
b'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import ecb_fx_provider as ecb  # noqa: E402
from market_db import MarketDatabase  # noqa: E402

# One real day of the feed, 25 Aug 2026.
ROW = {"USD": 1.1662, "JPY": 185.7, "GBP": 0.8555, "THB": 38.176, "CNY": 7.8366}

HIST_CSV = b"""Date,USD,JPY,GBP,THB,CNY,
2005-04-04,1.2916,138.95,0.68580,50.622,N/A,
2005-04-01,1.2954,139.14,0.68690,N/A,N/A,
"""

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01"
                 xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
  <Cube>
    <Cube time="2026-08-25">
      <Cube currency="USD" rate="1.1662"/>
      <Cube currency="THB" rate="38.176"/>
    </Cube>
    <Cube time="2026-08-24">
      <Cube currency="USD" rate="1.1700"/>
      <Cube currency="THB" rate="38.200"/>
    </Cube>
  </Cube>
</gesmes:Envelope>
"""


# --- pair arithmetic -------------------------------------------------------


@pytest.mark.parametrize(
    "pair,expected",
    [
        ("THB=X", ("USD", "THB")),
        ("thb=x", ("USD", "THB")),
        ("USDTHB=X", ("USD", "THB")),
        ("THBUSD=X", ("THB", "USD")),
        ("USD=X", ("USD", "USD")),
        ("AAPL", None),
        ("", None),
        ("TOOLONGX=X", None),
    ],
)
def test_split_pair(pair, expected):
    assert ecb.split_pair(pair) == expected


def test_cross_is_quote_per_base():
    """THB=X is baht per dollar (~32.7), never dollars per baht (~0.03)."""
    assert ecb.pair_rate(ROW, "THB=X") == pytest.approx(38.176 / 1.1662)
    assert ecb.pair_rate(ROW, "THB=X") == pytest.approx(32.7345, abs=1e-3)
    # The spelled-out pair is the same series under another name.
    assert ecb.pair_rate(ROW, "USDTHB=X") == ecb.pair_rate(ROW, "THB=X")
    # And the reverse is its reciprocal, not a repeat of it.
    assert ecb.pair_rate(ROW, "THBUSD=X") == pytest.approx(
        1 / ecb.pair_rate(ROW, "THB=X")
    )


def test_euro_is_its_own_unit():
    """EUR never appears as a column in the feed; it is the denominator."""
    assert ecb.pair_rate(ROW, "EUR=X") == pytest.approx(1 / 1.1662)
    assert ecb.pair_rate(ROW, "EURUSD=X") == pytest.approx(1.1662)


def test_identity_pair_is_flat_one():
    """USD=X is what stored a BLOB; here it is a rate like any other."""
    assert ecb.pair_rate(ROW, "USD=X") == 1.0


def test_uncovered_currency_is_none_not_zero():
    assert ecb.pair_rate(ROW, "SGD=X") is None
    assert ecb.pair_rate({"USD": 1.16, "THB": 0.0}, "THB=X") is None


# --- parsing ---------------------------------------------------------------


def test_hist_csv_skips_not_available():
    """THB predates CNY on the list; 'N/A' must not become a rate."""
    rates = ecb.parse_hist_csv(HIST_CSV)
    assert rates["2005-04-04"]["THB"] == 50.622
    assert "CNY" not in rates["2005-04-04"]
    assert "THB" not in rates["2005-04-01"]
    # So the pair exists on one day and not the other.
    assert ecb.pair_series(rates, "THB=X") == [
        ("2005-04-04", pytest.approx(50.622 / 1.2916))
    ]


def test_hist_csv_rejects_a_page_that_is_not_the_feed():
    with pytest.raises(ecb.ECBFXError):
        ecb.parse_hist_csv(b"<html>maintenance</html>")
    with pytest.raises(ecb.ECBFXError):
        ecb.parse_hist_csv(b"")


def test_xml_reads_every_day_not_just_the_first():
    rates = ecb.parse_xml(XML)
    assert set(rates) == {"2026-08-25", "2026-08-24"}
    series = ecb.pair_series(rates, "THB=X")
    assert [day for day, _ in series] == ["2026-08-24", "2026-08-25"]


def test_xml_rejects_junk():
    with pytest.raises(ecb.ECBFXError):
        ecb.parse_xml(b"not xml at all <<<")


def test_supported_pairs_filters_to_what_the_feed_can_price():
    rates = ecb.parse_xml(XML)
    assert ecb.supported_pairs(rates, ["THB=X", "SGD=X", "USD=X"]) == [
        "THB=X",
        "USD=X",
    ]


def test_fetch_history_reads_the_csv_inside_the_zip(monkeypatch):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("eurofxref-hist.csv", HIST_CSV)

    provider = ecb.ECBFXProvider()
    monkeypatch.setattr(provider, "_get", lambda url: buffer.getvalue())
    assert provider.fetch_history()["2005-04-04"]["USD"] == 1.2916


def test_a_transport_failure_is_an_ecbfxerror(monkeypatch):
    provider = ecb.ECBFXProvider()

    def boom(url):
        raise ecb.ECBFXError("ECB request failed (timeout)")

    monkeypatch.setattr(provider, "_get", boom)
    with pytest.raises(ecb.ECBFXError):
        provider.fetch_recent()


# --- the store -------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    return MarketDatabase(str(tmp_path / "market_data.db"))


def read_fx(db, pair):
    with db._get_connection() as conn:
        return {
            row[0]: (row[1], row[2])
            for row in conn.execute(
                "SELECT date, rate, source FROM daily_fx WHERE pair = ?", (pair,)
            )
        }


def test_fill_only_never_moves_a_stored_rate(db):
    db.upsert_fx_rows("THB=X", [("2026-08-25", 32.668)], source="yahoo")
    written = db.upsert_fx_rows(
        "THB=X",
        [("2026-08-25", 32.7345), ("2026-08-26", 32.70)],
        source="ecb",
        fill_only=True,
    )

    stored = read_fx(db, "THB=X")
    assert written == 1
    assert stored["2026-08-25"] == (32.668, "yahoo"), "the stored day must be untouched"
    assert stored["2026-08-26"] == (32.70, "ecb")


def test_overwrite_is_available_when_asked_for(db):
    db.upsert_fx_rows("THB=X", [("2026-08-25", 32.668)], source="yahoo")
    db.upsert_fx_rows("THB=X", [("2026-08-25", 32.7345)], source="ecb", fill_only=False)
    assert read_fx(db, "THB=X")["2026-08-25"] == (32.7345, "ecb")


def test_a_numpy_scalar_stores_as_a_number_not_a_blob(db):
    """The USD=X defect: numpy's buffer protocol makes sqlite3 store 8 bytes."""
    db.upsert_fx_rows("USD=X", [("2026-08-25", np.int64(1))], source="ecb")
    rate = read_fx(db, "USD=X")["2026-08-25"][0]
    assert isinstance(rate, float) and rate == 1.0


def test_junk_rates_are_dropped_not_stored(db):
    written = db.upsert_fx_rows(
        "THB=X",
        [
            ("2026-08-20", None),
            ("2026-08-21", float("nan")),
            ("2026-08-22", -1.0),
            ("2026-08-23", "not a number"),
            ("2026-08-24", 32.5),
        ],
        source="ecb",
    )
    assert written == 1
    assert set(read_fx(db, "THB=X")) == {"2026-08-24"}


def test_dataframe_path_still_works_and_records_its_source(db):
    frame = pd.DataFrame(
        {"Close": [32.6, 32.7]},
        index=pd.to_datetime(["2026-08-24", "2026-08-25"]),
    )
    db.upsert_fx("THB=X", frame)
    stored = read_fx(db, "THB=X")
    assert stored["2026-08-24"] == (32.6, "yahoo")
    assert len(stored) == 2


def test_writes_still_land_on_a_database_predating_the_source_column(tmp_path):
    """An un-migrated archive takes rates; it just cannot say where they came from."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE daily_fx (pair TEXT, date TEXT, rate REAL, "
        "interval TEXT DEFAULT '1d', PRIMARY KEY (pair, date, interval))"
    )
    conn.commit()
    conn.close()

    db = MarketDatabase(path)
    assert db.upsert_fx_rows("THB=X", [("2026-08-25", 32.7)], source="ecb") == 1
    with db._get_connection() as check:
        assert check.execute("SELECT rate FROM daily_fx").fetchone()[0] == 32.7


# --- the ingester's plan ---------------------------------------------------


def test_plan_separates_gaps_from_disagreements(tmp_path):
    import backfill_fx_rates as job

    path = str(tmp_path / "market_data.db")
    db = MarketDatabase(path)
    db.upsert_fx_rows(
        "THB=X",
        [("2026-08-24", 38.200 / 1.1700 * 1.001)],  # Yahoo, 0.1% off the fix
        source="yahoo",
    )
    rates = ecb.parse_xml(XML)

    plan = job.build_plan(path, "THB=X", rates, start=None, overwrite=False)

    assert [day for day, _ in plan.fills] == ["2026-08-25"], "only the missing day"
    assert plan.repairs == []
    assert plan.overlap == 1
    assert plan.diffs[0] == pytest.approx(0.1, abs=0.01)


def test_plan_repairs_a_stored_blob_even_without_overwrite(tmp_path):
    import backfill_fx_rates as job

    path = str(tmp_path / "market_data.db")
    MarketDatabase(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO daily_fx (pair, date, rate, interval) VALUES (?,?,?,?)",
        ("USD=X", "2026-08-25", memoryview(b"\x01" + b"\x00" * 7), "1d"),
    )
    conn.commit()
    conn.close()

    plan = job.build_plan(path, "USD=X", ecb.parse_xml(XML), None, overwrite=False)

    assert [day for day, _ in plan.repairs] == ["2026-08-25"]
    assert plan.overlap == 0, "a BLOB is not a second opinion to compare against"
    assert plan.fills == [], "a flat 1.0 series is not history worth storing"
