"""Tests for the Tiingo adjudication path.

This code rewrote 34,120 bars of the price archive, and it shipped two defects
that only measurement caught. Both are pinned here, because both looked correct
in review:

*The identity check compared price levels.* It rejected BYND and CURX at 96% —
precisely the symbols needing repair, because each had a recent split the
archive had not applied, which is what makes levels disagree. A level test
refuses to look at any symbol whose problem is recent.

*The evidence was a ±15-day window* around the flagged ex-date, inherited from
the hand-collected IBKR workflow. An unapplied split puts *every* earlier bar on
the wrong basis, so the window repaired the edge and left the middle: 19 of 25
symbols stayed flagged.

The third thing under test never broke, and would be the worst if it did:
`split_adjusted` converts Tiingo's raw closes onto the archive's basis. Get it
backwards and the repair tool sees every pre-split bar differing by exactly the
split ratio and rewrites whole histories onto the raw basis — a migration
wearing a repair's clothes.
"""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import ingest_tiingo_reference as job  # noqa: E402
import tiingo_provider as tg  # noqa: E402


def bar(day, close, split=1.0, div=0.0):
    return {"date": f"{day}T00:00:00.000Z", "close": close, "splitFactor": split, "divCash": div}


# --- the basis conversion ---------------------------------------------------


def test_a_forward_split_adjusts_earlier_bars_down():
    """AAPL's real 4:1. Raw 500.04 the Friday before is 125.01 on today's basis,
    which is exactly what the archive stores for that day."""
    rows = [
        bar("2020-08-27", 500.04),
        bar("2020-08-28", 499.23),
        bar("2020-08-31", 129.04, split=4.0),
        bar("2020-09-01", 134.18),
    ]
    adj = tg.split_adjusted(rows)
    assert adj["2020-08-27"] == pytest.approx(125.01)
    assert adj["2020-08-28"] == pytest.approx(124.8075)
    # The ex-date bar and everything after it are already on the new basis.
    assert adj["2020-08-31"] == pytest.approx(129.04)
    assert adj["2020-09-01"] == pytest.approx(134.18)


def test_a_reverse_split_adjusts_earlier_bars_up():
    """BYND's 1:30. The direction that made 0.5560 the wrong answer for a bar
    the reference put at 16.68."""
    rows = [
        bar("2026-07-30", 0.556),
        bar("2026-08-13", 0.407),
        bar("2026-08-14", 13.47, split=0.0333333333),
    ]
    adj = tg.split_adjusted(rows)
    assert adj["2026-07-30"] == pytest.approx(16.68, abs=0.01)
    assert adj["2026-08-14"] == pytest.approx(13.47)


def test_splits_compound_backwards():
    """Two splits: a bar before both carries the product, not the nearer one."""
    rows = [
        bar("2020-01-02", 100.0),
        bar("2021-01-04", 50.0, split=2.0),
        bar("2022-01-03", 12.5, split=4.0),
    ]
    adj = tg.split_adjusted(rows)
    assert adj["2020-01-02"] == pytest.approx(100.0 / 8)
    assert adj["2021-01-04"] == pytest.approx(50.0 / 4)
    assert adj["2022-01-03"] == pytest.approx(12.5)


def test_the_ex_date_bar_is_not_adjusted_by_its_own_split():
    """Off by one here and every series is wrong by a whole ratio on one day."""
    rows = [bar("2024-06-07", 1208.88), bar("2024-06-10", 121.79, split=10.0)]
    adj = tg.split_adjusted(rows)
    assert adj["2024-06-10"] == pytest.approx(121.79)
    assert adj["2024-06-07"] == pytest.approx(120.888)


@pytest.mark.parametrize("junk", [None, "", "n/a", 0])
def test_an_unusable_split_factor_is_treated_as_no_split(junk):
    rows = [bar("2024-01-02", 10.0), bar("2024-01-03", 10.5, split=junk)]
    adj = tg.split_adjusted(rows)
    assert adj["2024-01-02"] == pytest.approx(10.0)


def test_rows_out_of_order_still_adjust_correctly():
    """The feed is oldest-first, but nothing should depend on that."""
    rows = [bar("2020-08-31", 129.04, split=4.0), bar("2020-08-27", 500.04)]
    assert tg.split_adjusted(rows)["2020-08-27"] == pytest.approx(125.01)


# --- provider failures ------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.response


def provider_with(response):
    p = tg.TiingoProvider(api_key="test-token")
    p._session = FakeSession(response)
    return p


def test_a_404_is_its_own_error_because_non_us_is_expected():
    """SET listings are not carried; that is a skip, not a failure."""
    p = provider_with(FakeResponse(404, text="not found"))
    with pytest.raises(tg.TiingoSymbolUnknown):
        p.fetch_prices("PTT")


def test_a_429_explains_the_meter():
    p = provider_with(FakeResponse(429, text="too many"))
    with pytest.raises(tg.TiingoError) as exc:
        p.fetch_prices("AAPL")
    assert exc.value.status == 429
    assert "idempotent" in str(exc.value)


def test_no_token_is_a_setup_error_not_an_outage(monkeypatch):
    monkeypatch.setattr(tg, "TIINGO_API_KEY", None)
    p = tg.TiingoProvider(api_key=None)
    assert not p.is_configured()
    with pytest.raises(tg.TiingoNotConfiguredError):
        p.fetch_prices("AAPL")


def test_a_non_list_payload_is_refused():
    p = provider_with(FakeResponse(200, body={"detail": "surprise"}))
    with pytest.raises(tg.TiingoError):
        p.fetch_prices("AAPL")


def test_calls_are_counted_because_the_tier_meters_them():
    p = provider_with(FakeResponse(200, body=[bar("2024-01-02", 10.0)]))
    p.fetch_prices("AAPL")
    p.fetch_prices("MSFT")
    assert p.calls_made == 2


# --- the archive side -------------------------------------------------------


@pytest.fixture
def archive(tmp_path):
    path = str(tmp_path / "market_data.db")
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
    conn.commit()
    return path, conn


def add_bars(conn, symbol, rows):
    conn.executemany(
        "INSERT OR REPLACE INTO daily_ohlcv (symbol, date, close, interval) VALUES (?,?,?, '1d')",
        [(symbol, d, c) for d, c in rows],
    )
    conn.commit()


def test_disputed_days_returns_every_wrong_bar_not_a_window(archive):
    """The regression that left 19 of 25 symbols still flagged.

    An unapplied split puts every earlier bar on the wrong basis. A window
    around the ex-date repairs the edge and leaves the middle.
    """
    path, conn = archive
    days = [f"2026-07-{d:02d}" for d in range(1, 29)]
    # Archive holds the pre-split basis throughout; reference is 30x higher.
    add_bars(conn, "BYND", [(d, 0.5) for d in days])
    closes = {d: 15.0 for d in days}

    disputed = job.disputed_days(conn, "BYND", closes)

    assert disputed == days, "every disagreeing bar is evidence, not just the recent ones"


def test_disputed_days_ignores_bars_the_two_providers_agree_on(archive):
    """`reference_price` is evidence about a disagreement. A bar they agree on
    is not evidence about anything."""
    path, conn = archive
    add_bars(conn, "X", [("2026-01-02", 10.0), ("2026-01-03", 0.5), ("2026-01-06", 10.1)])
    closes = {"2026-01-02": 10.0, "2026-01-03": 15.0, "2026-01-06": 10.1}

    assert job.disputed_days(conn, "X", closes) == ["2026-01-03"]


def test_disputed_days_skips_days_the_reference_does_not_carry(archive):
    path, conn = archive
    add_bars(conn, "X", [("2026-01-02", 1.0), ("2026-01-03", 1.0)])
    assert job.disputed_days(conn, "X", {"2026-01-02": 30.0}) == ["2026-01-02"]


def test_disputed_days_is_capped(archive, monkeypatch):
    monkeypatch.setattr(job, "MAX_REFERENCE_BARS", 5)
    path, conn = archive
    days = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(40)]
    add_bars(conn, "X", [(d, 1.0) for d in days])
    disputed = job.disputed_days(conn, "X", {d: 30.0 for d in days})
    assert len(disputed) == 5
    assert disputed == sorted(days)[-5:], "keeps the most recent evidence"


def test_identity_holds_for_a_symbol_whose_split_was_never_applied(archive):
    """The BYND regression, and the reason the test is on returns.

    Levels differ by 30x on every pre-split bar — which is the defect being
    adjudicated, not evidence of a different company. The daily moves match.
    """
    path, conn = archive
    days = [f"2026-06-{d:02d}" for d in range(1, 29)]
    moves = [1.0, 1.02, 0.99, 1.05, 0.97] * 6
    archive_series, ref_series, a, b = [], {}, 0.5, 15.0
    for d, m in zip(days, moves):
        a, b = a * m, b * m
        archive_series.append((d, a))
        ref_series[d] = b
    add_bars(conn, "BYND", archive_series)

    ok, why = job.identity_check(conn, "BYND", ref_series)

    assert ok, why
    assert "match" in why


def test_identity_fails_for_a_genuinely_different_series(archive):
    """The mirror case: two deep histories under one ticker, different companies."""
    path, conn = archive
    days = [f"2026-06-{d:02d}" for d in range(1, 29)]
    add_bars(conn, "X", [(d, 10.0 + i) for i, d in enumerate(days)])
    # Unrelated path: same scale, uncorrelated moves.
    ref = {d: 10.0 + (i * 7 % 13) for i, d in enumerate(days)}

    ok, why = job.identity_check(conn, "X", ref)

    assert not ok
    assert "different listing" in why


def test_identity_needs_enough_overlap_to_mean_anything(archive):
    path, conn = archive
    add_bars(conn, "X", [("2026-06-01", 1.0), ("2026-06-02", 1.0)])
    ok, why = job.identity_check(conn, "X", {"2026-06-01": 1.0, "2026-06-02": 1.0})
    assert not ok
    assert "common" in why or "comparable" in why


def test_identity_ignores_the_pair_that_straddles_an_ex_date(archive):
    """The one day an unapplied series legitimately jumps."""
    path, conn = archive
    days = [f"2026-06-{d:02d}" for d in range(1, 29)]
    archive_series = [(d, 1.0) for d in days]
    ref = {d: 1.0 for d in days}
    # The archive jumps 30x on the ex-date; the reference does not.
    archive_series[14] = (days[14], 30.0)
    add_bars(conn, "X", archive_series)
    conn.execute(
        "INSERT INTO corporate_action (symbol, date, kind, value) VALUES ('X', ?, 'split', 0.0333)",
        (days[14],),
    )
    conn.commit()

    ok, why = job.identity_check(conn, "X", ref)
    assert ok, why


# --- the guard that was missing ---------------------------------------------


def test_a_reference_missing_one_of_our_splits_is_refused(archive):
    """The guard whose absence cost 15,588 bars.

    KGEI: the archive records a 1:10 reverse split, Tiingo carries no split for
    the symbol at all. Every earlier bar then disagrees by exactly 10x — which
    is, by construction, one of the symbol's own recorded ratios. So the repair
    "explains" the difference, divides by it, and lands the bar on the
    reference's *unadjusted* basis. Nothing downstream catches it: the ratio
    matches, the repaired value lands on the reference, and the identity check
    passes because the two series move together. Only the event logs disagree.
    """
    path, conn = archive
    conn.execute(
        "INSERT INTO corporate_action (symbol, date, kind, value) "
        "VALUES ('KGEI', '2022-05-19', 'split', 0.1)"
    )
    conn.commit()

    ok, why = job.split_coverage_check(conn, "KGEI", [bar("2022-05-20", 1.0)])

    assert not ok
    assert "2022-05-19" in why and "different basis" in why


def test_a_reference_carrying_our_splits_is_accepted(archive):
    path, conn = archive
    conn.execute(
        "INSERT INTO corporate_action (symbol, date, kind, value) "
        "VALUES ('BYND', '2026-08-14', 'split', 0.0333333333)"
    )
    conn.commit()

    ok, why = job.split_coverage_check(
        conn, "BYND", [bar("2026-08-14", 13.47, split=0.0333333333)]
    )

    assert ok, why


def test_the_same_split_dated_a_day_apart_still_counts_as_carried(archive):
    """Providers date one event a day apart routinely — BOTJ is 03-07 at Yahoo
    and 03-08 at Tiingo. That is not a missing split, and refusing on it would
    reject symbols whose reference is perfectly good."""
    path, conn = archive
    conn.execute(
        "INSERT INTO corporate_action (symbol, date, kind, value) "
        "VALUES ('BOTJ', '2005-03-07', 'split', 1.5)"
    )
    conn.commit()

    ok, why = job.split_coverage_check(
        conn, "BOTJ", [bar("2005-03-08", 16.0, split=1.5)]
    )

    assert ok, why


def test_refusing_a_symbol_purges_its_stored_evidence(tmp_path):
    """Reverting bad bars is not enough on its own, and this cost two rounds.

    `repair_bars_against_reference.py` acts on whatever sits in
    `reference_price`. Evidence left on file after its bars were reverted gets
    re-applied by the next run — which is how a repair pass for a single symbol
    silently redid fifteen others. So a refusal has to delete, not merely
    decline to add.
    """
    import sqlite3 as sq

    path = str(tmp_path / "market.db")
    conn = sq.connect(path)
    conn.executescript(
        """
        CREATE TABLE reference_price (symbol TEXT, date TEXT, close REAL,
            source TEXT, fetched_at TEXT, PRIMARY KEY (symbol, date, source));
        """
    )
    conn.executemany(
        "INSERT INTO reference_price VALUES (?,?,?,?,?)",
        [("KGEI", "2020-01-02", 1.46, "tiingo", "now"),
         ("KGEI", "2020-01-03", 1.47, "tiingo", "now"),
         ("BYND", "2026-07-23", 16.8, "tiingo", "now")],
    )
    conn.commit()
    conn.close()

    removed = job.purge_reference(path, "KGEI")

    check = sq.connect(path)
    left = {r[0] for r in check.execute("SELECT DISTINCT symbol FROM reference_price")}
    assert removed == 2
    assert left == {"BYND"}, "only the refused symbol's evidence is dropped"
