"""Phase 13e: portfolio_logic's summary in Rust (rust/investa_core/src/
pl_summary.rs and pl_helpers.rs) against the original frozen in
tests/reference/portfolio_logic.py.

A deterministic fake market provider stands in for the network: quotes with
missing, zero and NumPy prices, a critical-fetch flag, historical FX tables in
both Yahoo conventions (USDTHB=X and a GBP=X quoted as USD per pound), a
currency with no history at all (the current-quote fallback), and metadata
from the batch, from the per-symbol fetch, or not at all. A fake NAV store
covers the published-NAV path. The ledgers carry transfers into and out of
selected accounts, splits, "All Accounts" rows, a missing original_index, and
every account-filter shape (none, a valid list, an unknown account, a tuple).
Manual overrides carry prices (valid, text, zero) and sector, geography and
exchange. The summary runs twice per book with caching on, so the FIFO cache
is read as well as filled.

The whole seven-part result, the caller's frame afterwards, and the log
records must agree. Two log texts are masked: the elapsed-time line, and the
tracebacks of the two `traceback.format_exc()` lines (the port's has no frame
for the summary function, which is no longer Python); their last line, the
exception itself, is still compared.
"""

import os
import re
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import market_db  # noqa: E402
import portfolio_logic as new  # noqa: E402
from reference import portfolio_logic as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

REPORT = date(2024, 6, 14)
ACCOUNTS = ["IBKR", "E*Trade", "SCBS", "Kim Eng"]
SYMBOLS = ["AAPL", "MSFT", "PTT.BK", "NOFEED", "SCBRMS&P500", "$CASH", "VOD.L", "SAP.DE"]
CCY = {"AAPL": "USD", "MSFT": "USD", "PTT.BK": "THB", "NOFEED": "USD",
       "SCBRMS&P500": "THB", "$CASH": "USD", "VOD.L": "GBP", "SAP.DE": "EUR"}  # fmt: skip


class FakeProvider:
    def __init__(self, seed, flavour):
        self.seed = seed
        self.flavour = flavour

    def get_current_quotes(self, internal_stock_symbols, required_currencies, user_symbol_map, user_excluded_symbols):
        rng = np.random.default_rng(self.seed)
        data = {}
        for sym in sorted(internal_stock_symbols):
            r = rng.random()
            if sym in ("NOFEED", "SCBRMS&P500") or r < 0.15:
                continue
            data[sym] = {
                "price": [float(rng.uniform(5, 300)), 0.0, np.float64(42.5)][int(rng.integers(0, 3))],
                "change": float(rng.normal(0, 1)) if r < 0.8 else None,
                "changesPercentage": float(rng.normal(0, 1)),
                "dividendRate": 0.5,
                "name": f"{sym} Inc",
            }
        fx = {"USD": 1.0, "THB": 35.0, "GBP": 0.79, "EUR": 0.92}
        prev = {"USD": 1.0, "THB": 35.3, "GBP": 0.8}
        if self.flavour == "no-gbp":
            fx.pop("GBP")
        return data, fx, prev, self.flavour == "critical", rng.random() < 0.3

    def get_historical_fx_rates(self, fx_pairs_yf, start_date, end_date, use_cache, cache_key):
        days = pd.date_range(start_date, end_date, freq="B")
        out = {}
        for pair in sorted(fx_pairs_yf):
            if pair == "EUR=X" and self.flavour == "no-eur-history":
                continue
            base = {"USDTHB=X": 35.0, "GBP=X": 1.27, "EUR=X": 0.92}.get(pair, 2.0)
            col = "Close" if pair != "GBP=X" else "price"
            out[pair] = pd.DataFrame(
                {col: base * (1 + 0.01 * np.sin(np.arange(len(days))))}, index=days
            )
        return out, self.flavour == "hist-error"

    def _ensure_metadata_batch(self, yf_symbols):
        if self.flavour == "meta-error":
            raise RuntimeError("metadata service down")
        return {s: {"sector": "Tech", "quoteType": "EQUITY", "country": None} for s in sorted(yf_symbols) if s != "MSFT"}

    def get_fundamental_data(self, yf_symbol):
        return {"sector": "Software", "industry": "Systems"} if yf_symbol == "MSFT" else None


class FakeMarketDatabase:
    def get_latest_fund_navs(self):
        return {"SCBRMS&P500": ("2024-06-13", 12.5), "NOFEED": ("2024-06-12", np.nan)}


def _ledger(rng):
    rows = []
    days = pd.bdate_range("2022-01-03", "2024-06-14")
    for i in range(int(rng.integers(10, 70))):
        sym = str(rng.choice(SYMBOLS))
        acc = str(rng.choice(ACCOUNTS + ["All Accounts"] if rng.random() < 0.05 else ACCOUNTS))
        typ = str(rng.choice(["buy", "buy", "buy", "sell", "dividend", "fees", "transfer", "split", "tax"]))
        if sym == "$CASH":
            typ = str(rng.choice(["deposit", "withdrawal"]))
        rows.append(
            {
                "Date": days[int(rng.integers(0, len(days)))],
                "Symbol": sym,
                "Account": acc,
                "To Account": str(rng.choice(ACCOUNTS)) if typ == "transfer" else np.nan,
                "Type": typ,
                "Quantity": float(rng.choice([1.0, 5.0, 10.0, 40.0])),
                "Price/Share": float(rng.choice([0.0, 12.5, 101.0, 250.0, np.nan])),
                "Commission": float(rng.choice([0.0, 1.0, np.nan])),
                "Total Amount": float(rng.choice([np.nan, 100.0, 1250.0])),
                "Local Currency": CCY[sym],
                "Split Ratio": 2.0 if typ == "split" else 0.0,
                "Note": None,
                "original_index": i,
            }
        )  # fmt: skip
    df = pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True)
    if rng.random() < 0.1:
        df = df.drop(columns=["original_index"])
    return df


def _masked(out):
    result, logs, warned = out
    masked = []
    for level, msg in logs:
        if msg.startswith("Total Summary Calc Time:"):
            msg = "Total Summary Calc Time: <t>"
        elif "Traceback (most recent call last)" in msg or (level == "ERROR" and msg.endswith("\n")):
            msg = "<traceback> " + msg.strip().splitlines()[-1]
        masked.append((level, msg))
    return result, masked, warned


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    for mod in (new, ref):
        monkeypatch.setattr(mod, "get_latest_trading_date", lambda: REPORT)
        monkeypatch.setattr(mod, "get_est_today", lambda: REPORT)
        mod._FIFO_CACHE.clear()
    monkeypatch.setattr(market_db, "MarketDatabase", FakeMarketDatabase)


@pytest.mark.parametrize("seed", range(8))
def test_portfolio_summary_matches(caplog, seed):
    rng = np.random.default_rng(13600 + seed)
    for trial in range(10):
        df = _ledger(rng)
        flavour = str(rng.choice(["plain", "plain", "critical", "no-gbp", "no-eur-history", "hist-error", "meta-error"]))
        include = [None, [], ["IBKR"], ["SCBS", "E*Trade"], ["Nowhere"], ("IBKR",)][int(rng.integers(0, 6))]
        overrides = [None, {}, {"NOFEED": {"price": 42.0, "sector": "Funds", "exchange": "SET"},
                                "AAPL": {"price": "n/a", "geography": "US"},
                                "MSFT": {"price": 0, "asset_type": "ETF"}}][int(rng.integers(0, 3))]  # fmt: skip
        kwargs = dict(
            fmp_api_key=None,
            display_currency=str(rng.choice(["USD", "THB"])),
            show_closed_positions=bool(rng.random() < 0.5),
            include_accounts=include,
            manual_overrides_dict=overrides,
            user_symbol_map={"PTT.BK": "PTT.BK"},
            user_excluded_symbols={"NOFEED", "SCBRMS&P500"},
            default_currency="USD",
            account_currency_map={"SCBS": "THB"},
            account_interest_rates={"IBKR": 4.0} if rng.random() < 0.5 else None,
            interest_free_thresholds={"IBKR": 1000.0},
            account_cash_mode_map={"IBKR": "Auto"} if rng.random() < 0.5 else None,
            db_mtime=float(rng.choice([0.0, 1.0])),
        )
        for repeat in range(2):  # the second pass reads the FIFO cache
            a, b = df.copy(), df.copy()

            def run(mod, frame):
                return mod.calculate_portfolio_summary(
                    frame, frame, {99}, {99: "bad row"},
                    market_provider=FakeProvider(seed * 100 + trial, flavour), **kwargs
                )  # fmt: skip

            got = _masked(call(lambda: run(new, b), caplog))
            want = _masked(call(lambda: run(ref, a), caplog))
            assert got == want, (seed, trial, repeat, flavour)
            assert canon(a) == canon(b), (seed, trial, "frame left behind")


def test_empty_and_missing_inputs(caplog):
    for frame in (None, pd.DataFrame()):
        got = _masked(call(lambda: new.calculate_portfolio_summary(frame, frame, set(), {}), caplog))
        want = _masked(call(lambda: ref.calculate_portfolio_summary(frame, frame, set(), {}), caplog))
        assert got == want
    df = _ledger(np.random.default_rng(3)).drop(columns=["Account"])
    for include in (None, ["IBKR"]):
        got = _masked(call(lambda: new.calculate_portfolio_summary(df.copy(), None, None, None, include_accounts=include, market_provider=FakeProvider(1, "plain")), caplog))  # fmt: skip
        want = _masked(call(lambda: ref.calculate_portfolio_summary(df.copy(), None, None, None, include_accounts=include, market_provider=FakeProvider(1, "plain")), caplog))  # fmt: skip
        assert got == want


def test_helpers_match():
    rng = np.random.default_rng(5)
    for _ in range(20):
        df = _ledger(rng)
        assert new._transactions_fingerprint(df) == ref._transactions_fingerprint(df)
    for empty in (True, False):
        assert canon(new.get_default_metrics_dict("THB", REPORT, ["A"], empty)) == canon(
            ref.get_default_metrics_dict("THB", REPORT, ["A"], empty)
        )
    syms = {"SCBRMS&P500", "nofeed ", "AAPL"}
    assert canon(new.published_nav_prices(syms)) == canon(ref.published_nav_prices(syms))
    assert new.published_nav_prices(set()) == {} == ref.published_nav_prices(set())
    assert re.fullmatch(r"[0-9a-f]{32}", new._transactions_fingerprint(df))
