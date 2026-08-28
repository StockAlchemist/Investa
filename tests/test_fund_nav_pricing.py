"""Pricing a holding the market has no quote for.

Five Thai fund positions have no ticker, so they sit on the exclusion list and
the price feed never reaches them. Three things can price them, and the order
matters more than it looks:

  1. the SEC's published NAV — a real number, kept current;
  2. the manual override — a real number, typed once and then left. SCBRCTECH
     sat 15% above its NAV that way while the graph beside it, which had read
     the published series since the fund store landed, drew the right one;
  3. the ledger's own transaction prices — what the position actually traded
     at, which for a monthly contribution plan is the fund's NAV on each
     contribution date.

(3) is not a consolation prize. ES-GQG is a provident-fund sub-policy whose
NAVs the SEC publishes month-end only, so its override carries metadata and no
price on purpose, and the contributions are the best series that exists for it.
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import fund_nav_sync  # noqa: E402
import portfolio_logic  # noqa: E402
from market_db import MarketDatabase  # noqa: E402
from portfolio_analyzer import _build_summary_rows  # noqa: E402
from sec_thailand_provider import FundMatch, SECThailandError  # noqa: E402

REPORT_DATE = date(2026, 8, 27)
PRICE_COL = "Price (THB)"


@pytest.fixture
def db(tmp_path):
    database = MarketDatabase(str(tmp_path / "market_test.db"))
    with database._get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_nav (
                fund_code TEXT NOT NULL, date TEXT NOT NULL, nav REAL NOT NULL,
                currency TEXT, source TEXT NOT NULL,
                PRIMARY KEY (fund_code, date))
            """
        )
        conn.commit()
    return database


def _transactions(symbol, rows):
    """A minimal buy ledger: rows is [(iso_date, qty, price)].

    `original_index` is not decoration — the last-transaction fallback and the
    IRR both sort on it to break ties within a day, and its absence is swallowed
    as a generic fallback error that prices the position at zero.
    """
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d, _q, _p in rows],
            "Symbol": [symbol] * len(rows),
            "Quantity": [q for _d, q, _p in rows],
            "Price/Share": [p for _d, _q, p in rows],
            "Type": ["Buy"] * len(rows),
            "Account": ["Eastspring"] * len(rows),
            "Commission": [0.0] * len(rows),
            "Total Amount": [q * p for _d, q, p in rows],
            "original_index": list(range(len(rows))),
        }
    )


def _summary_row(symbol, *, manual_prices, published_navs, transactions):
    holdings = {
        (symbol, "Eastspring"): {
            "qty": 100.0,
            "total_cost_local": 2000.0,
            "local_currency": "THB",
            "total_cost_display_historical_fx": 2000.0,
        }
    }
    rows, _, _, _ = _build_summary_rows(
        holdings=holdings,
        current_stock_data={},
        current_fx_rates_vs_usd={"THB": 1.0, "USD": 1.0},
        current_fx_prev_close_vs_usd={"THB": 1.0, "USD": 1.0},
        display_currency="THB",
        default_currency="THB",
        transactions_df=transactions,
        report_date=REPORT_DATE,
        shortable_symbols=set(),
        # These holdings have no ticker, which is exactly why they are excluded
        # — and why a fix that only reached the quote path would never fire.
        user_excluded_symbols={symbol},
        user_symbol_map={},
        manual_prices_dict=manual_prices,
        published_navs=published_navs,
    )
    return rows[0]


# --- price precedence ------------------------------------------------------


def test_published_nav_outranks_the_manual_override():
    row = _summary_row(
        "SCBRCTECH",
        manual_prices={"SCBRCTECH": 6.6853},
        published_navs={"SCBRCTECH": ("2026-08-25", 5.6477)},
        transactions=_transactions("SCBRCTECH", [("2026-07-27", 100.0, 6.0)]),
    )
    assert row[PRICE_COL] == pytest.approx(5.6477)
    assert "Published NAV (2026-08-25)" in row["Price Source"]


def test_manual_override_still_prices_a_fund_with_no_published_nav():
    row = _summary_row(
        "SOMEFUND",
        manual_prices={"SOMEFUND": 12.5},
        published_navs={},
        transactions=_transactions("SOMEFUND", [("2026-07-27", 100.0, 11.0)]),
    )
    assert row[PRICE_COL] == pytest.approx(12.5)
    assert "Manual Fallback" in row["Price Source"]


def test_no_nav_and_no_override_falls_back_to_the_last_transaction():
    """The ES-GQG shape: metadata-only override, priced from its own trades."""
    row = _summary_row(
        "ES-GQG",
        manual_prices={},
        published_navs={},
        transactions=_transactions(
            "ES-GQG",
            [("2026-05-28", 50.0, 24.3008), ("2026-07-27", 50.0, 23.6843)],
        ),
    )
    assert row[PRICE_COL] == pytest.approx(23.6843)


def test_a_stale_override_cannot_win_by_being_larger():
    """Direction is not the test — precedence is.

    An override *below* the NAV must lose too, or the rule is really "take the
    bigger number", which would be wrong the moment a fund falls.
    """
    row = _summary_row(
        "SCBRMS&P500",
        manual_prices={"SCBRMS&P500": 22.6468},
        published_navs={"SCBRMS&P500": ("2026-08-25", 22.9211)},
        transactions=_transactions("SCBRMS&P500", [("2026-07-27", 100.0, 22.0)]),
    )
    assert row[PRICE_COL] == pytest.approx(22.9211)


# --- the NAV lookup --------------------------------------------------------


def test_latest_fund_navs_takes_the_newest_row_per_fund(db):
    db.upsert_fund_nav("SCBRM1", [("2026-08-24", 15.3931), ("2026-08-26", 15.3957)])
    db.upsert_fund_nav("SCBCHA-SSF", [("2026-08-26", 9.383)])

    latest = db.get_latest_fund_navs()
    assert latest["SCBRM1"] == ("2026-08-26", 15.3957)
    assert latest["SCBCHA-SSF"] == ("2026-08-26", 9.383)


def test_published_nav_prices_returns_only_symbols_actually_held(db, monkeypatch):
    db.upsert_fund_nav("SCBRM1", [("2026-08-26", 15.3957)])
    db.upsert_fund_nav("SCBRMS&P500", [("2026-08-25", 22.9211)])
    monkeypatch.setattr("market_db.MarketDatabase", lambda *a, **k: db)

    # Mixed case on the way in: the engine normalizes symbols, the override file
    # keeps whatever the user typed. Keys come back as the caller spelled them,
    # because that is what the summary will look up.
    prices = portfolio_logic.published_nav_prices({"scbrm1", "AAPL", "ES-GQG"})
    assert prices == {"scbrm1": ("2026-08-26", 15.3957)}


def test_published_nav_prices_survives_a_missing_store(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no such table: fund_nav")

    monkeypatch.setattr("market_db.MarketDatabase", boom)
    assert portfolio_logic.published_nav_prices({"SCBRM1"}) == {}


# --- keeping the series current --------------------------------------------


class FakeProvider:
    """Resolves anything, and replays one NAV row per requested window."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.windows = {}

    def resolve_fund(self, code):
        if code in self.fail_for:
            raise SECThailandError("lookup exploded")
        return FundMatch(f"P_{code}", None, "abbr", {}, [])

    def fetch_nav(self, proj_id, start, end, fund_class_name=None):
        self.windows[proj_id] = (start, end)
        return [{"date": end.isoformat(), "nav": 42.0}]


def test_top_up_refetches_a_trailing_window_not_just_the_gap(db):
    db.upsert_fund_nav("SCBRM1", [("2026-08-20", 15.0)])
    provider = FakeProvider()

    written = fund_nav_sync.top_up(
        lookback_days=10, today=date(2026, 8, 27), db=db, provider=provider
    )

    assert written == {"SCBRM1": 1}
    # From last stored date *minus* the lookback: the SEC restates NAVs for a
    # few days after publishing, so resuming at 2026-08-20 exactly would keep
    # whichever provisional value happened to land first.
    assert provider.windows["P_SCBRM1"] == (date(2026, 8, 10), date(2026, 8, 27))


def test_top_up_never_backfills_a_fund_with_nothing_stored(db):
    db.upsert_fund_nav("SCBRM1", [("2026-08-20", 15.0)])
    provider = FakeProvider()

    fund_nav_sync.top_up(today=date(2026, 8, 27), db=db, provider=provider)

    # SCBRMS&P500 has no rows; twenty-five years of history is the backfill
    # script's job, not a routine tick's.
    assert set(provider.windows) == {"P_SCBRM1"}


def test_top_up_skips_provident_sub_policies(db):
    db.upsert_fund_nav("ES-GQG", [("2026-08-20", 24.0)])
    provider = FakeProvider()

    assert fund_nav_sync.top_up(today=date(2026, 8, 27), db=db, provider=provider) == {}
    assert provider.windows == {}


def test_top_up_contains_one_funds_failure(db):
    db.upsert_fund_nav("SCBRM1", [("2026-08-20", 15.0)])
    db.upsert_fund_nav("SCBCHA-SSF", [("2026-08-20", 9.0)])
    provider = FakeProvider(fail_for={"SCBRM1"})

    written = fund_nav_sync.top_up(today=date(2026, 8, 27), db=db, provider=provider)

    # One unreachable fund must not stop the others catching up.
    assert written == {"SCBCHA-SSF": 1}


def test_top_up_applies_the_scbrctech_alias(db):
    """The local code and the SEC's differ by one letter; a miss is silent."""
    db.upsert_fund_nav("SCBRCTECH", [("2026-08-20", 5.7)])
    provider = FakeProvider()

    fund_nav_sync.top_up(today=date(2026, 8, 27), db=db, provider=provider)

    assert "P_SCBRMCTECH" in provider.windows
