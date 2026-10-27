"""Phase 14c: portfolio_history's input preparation, L1 holdings cache and
daily valuation in Rust (rust/investa_core/src/ph_valuation.rs) against the
originals frozen in tests/reference/portfolio_history.py.

- Input preparation over random ledgers with every include/exclude shape
  (none, valid, unknown, overlapping, a tuple), transfer-source backfill,
  splits (valid, zero, text, duplicated across accounts), cash rows, user
  symbol maps and exclusions, and a CSV path that exists, does not, or is
  absent; with and without an app cache directory.
- The L1 daily-holdings cache: a miss that computes and saves, then a hit
  that loads, in separate directories per side, and no cache directory.
- Daily valuation over random holdings grids with and without last prices,
  price tables in every column layout and index kind, published NAVs from a
  fake fund store, manual prices, and FX tables under every pair spelling
  (THB=X, USDTHB=X, a USD-per-unit EURUSD=X, a "rate" column) or missing.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
import market_db  # noqa: E402
import portfolio_history as new  # noqa: E402
from reference import portfolio_history as ref  # noqa: E402
from test_rust_history_helpers_parity import FakeFunds  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

ACCOUNTS = ["IBKR", "SCBS", "Kim Eng", "All Accounts"]
SYMBOLS = ["AAPL", "MSFT", "PTT.BK", "$CASH", "SCBRMS&P500", "VOD.L"]
CCY = {"AAPL": "USD", "MSFT": "USD", "PTT.BK": "THB", "$CASH": "USD", "SCBRMS&P500": "THB", "VOD.L": "GBP"}


def _ledger(rng):
    rows = []
    for i in range(int(rng.integers(1, 60))):
        sym = str(rng.choice(SYMBOLS))
        typ = str(rng.choice(["buy", "buy", "sell", "transfer", "split", "Stock Split", "dividend", "deposit"]))
        rows.append(
            {
                "Date": pd.Timestamp("2022-01-03") + pd.Timedelta(days=int(rng.integers(0, 700))),
                "Symbol": sym,
                "Account": str(rng.choice(ACCOUNTS + ["ibkr "])),
                "To Account": str(rng.choice(ACCOUNTS)) if typ == "transfer" else None,
                "Type": typ,
                "Quantity": float(rng.choice([1.0, 10.0, 25.0])),
                "Price/Share": float(rng.choice([0.0, 12.5, 101.0])),
                "Commission": float(rng.choice([0.0, 1.0])),
                "Total Amount": float(rng.choice([0.0, 125.0])),
                "Split Ratio": [2.0, 0.0, 3.0, np.nan][int(rng.integers(0, 4))] if "plit" in typ else 0.0,
                "Local Currency": CCY[sym],
                "original_index": i,
            }
        )  # fmt: skip
    df = pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True)
    if rng.random() < 0.1:
        df = df.drop(columns=["To Account"])
    if rng.random() < 0.1:
        df = df.drop(columns=["Split Ratio"])
    return df


def _masked(out, *dirs):
    result, logs, warned = out
    logs = [(lv, _strip(msg, dirs)) for lv, msg in logs]
    return result, logs, warned


def _strip(text, dirs):
    for d in dirs:
        text = text.replace(str(d), "<dir>")
    return text


@pytest.mark.parametrize("seed", range(8))
def test_prepare_historical_inputs_match(caplog, monkeypatch, tmp_path, seed):
    rng = np.random.default_rng(14900 + seed)
    csv = tmp_path / "tx.csv"
    csv.write_text("Date,Symbol\n2024-01-01,AAPL\n")
    for trial in range(25):
        df = _ledger(rng)
        cache = [str(tmp_path / "cache"), None][int(rng.integers(0, 2))]
        monkeypatch.setattr(config, "get_app_cache_dir", lambda c=cache: c)
        include = [None, [], ["IBKR"], [" scbs", "Kim Eng"], ["Nowhere"]][int(rng.integers(0, 5))]
        exclude = [None, [], ["SCBS"], ["IBKR", "Kim Eng"], ("SCBS",), ["Nowhere"]][int(rng.integers(0, 6))]
        path = [None, str(csv), str(tmp_path / "missing.csv")][int(rng.integers(0, 3))]
        kwargs = dict(
            user_symbol_map={"PTT.BK": "PTT.BK"} if rng.random() < 0.5 else None,
            user_excluded_symbols={"SCBRMS&P500"} if rng.random() < 0.5 else None,
            original_csv_file_path=path,
            interval=str(rng.choice(["1d", "1h"])),
        )
        outs = []
        for mod in (ref, new):
            a = df.copy()
            out = call(
                lambda: mod._prepare_historical_inputs(
                    a, None, {7}, {7: "x"}, {"SCBS": "THB", "Kim Eng": "THB"}, "USD", include, exclude,
                    pd.Timestamp("2022-01-03").date(), pd.Timestamp("2024-06-14").date(), ["SPY", "QQQ"], "THB",
                    "v23", "raw_prefix", "daily_prefix", **kwargs,
                ),
                caplog,
            )  # fmt: skip
            outs.append((_masked(out, tmp_path), canon(a)))
        assert outs[0] == outs[1], (seed, trial)


@pytest.mark.parametrize("seed", range(4))
def test_all_daily_holdings_match(caplog, monkeypatch, tmp_path, seed):
    rng = np.random.default_rng(15000 + seed)
    for trial in range(6):
        df = _ledger(rng)
        df["Account"] = df["Account"].str.upper().str.strip()
        maps = new.generate_mappings(df)
        symbol_to_id, _, account_to_id, _, type_to_id, _, _ = maps
        modes = {"IBKR": "Auto", "scbs": "manual"} if rng.random() < 0.5 else None
        no_cache = rng.random() < 0.2
        outs = []
        for side, mod in (("a", ref), ("b", new)):
            cache = None if no_cache else tmp_path / f"{trial}{side}"
            monkeypatch.setattr(config, "get_app_cache_dir", lambda c=cache: None if c is None else str(c))
            runs = []
            for _ in range(2):  # a miss that saves, then a hit that loads
                out = call(
                    lambda: mod._get_or_calculate_all_daily_holdings(
                        df.copy(), pd.Timestamp("2022-01-03").date(), pd.Timestamp("2024-06-14").date(),
                        symbol_to_id, account_to_id, type_to_id, modes,
                    ),
                    caplog,
                )  # fmt: skip
                runs.append(_masked(out, tmp_path / f"{trial}{side}"))
            files = [] if cache is None else sorted(p.name for p in (cache / "all_holdings_cache_new").iterdir())
            outs.append((runs, files))
        assert outs[0] == outs[1], (seed, trial)


def _fx_tables(rng, grid):
    days = grid if rng.random() < 0.5 else grid.tz_localize(None)
    out = {}
    if rng.random() < 0.8:
        out["USDTHB=X" if rng.random() < 0.4 else "THB=X"] = pd.DataFrame({"Close": 35 + rng.normal(0, 0.2, len(days))}, index=days)
    if rng.random() < 0.6:
        out["EURUSD=X"] = pd.DataFrame({"price": 1.08 + rng.normal(0, 0.01, len(days))}, index=days)
    if rng.random() < 0.6:
        g = pd.DataFrame({"rate": 0.79 + rng.normal(0, 0.01, len(days))}, index=days)
        g.iloc[::5] = np.nan
        out["GBP=X"] = g
    if rng.random() < 0.1:
        out["THB=X"] = pd.DataFrame({"Volume": [1.0]}, index=days[:1])
    return out


@pytest.mark.parametrize("seed", range(8))
def test_value_daily_holdings_match(caplog, monkeypatch, seed):
    monkeypatch.setattr(market_db, "MarketDatabase", FakeFunds)
    FakeFunds.mode = "ok"
    rng = np.random.default_rng(15100 + seed)
    caplog.set_level("DEBUG")
    id_to_symbol = {0: "AAPL", 1: "MSFT", 2: "$CASH", 3: "SCBRMS&P500", 4: "NOFEED", 5: "VOD.L"}
    id_to_account = {0: "IBKR", 1: "SCBS", 2: "KIM ENG", 3: "LONDON"}
    for trial in range(25):
        days = int(rng.integers(5, 40))
        grid = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
        qty = rng.choice([0.0, 0.0, 3.0, 10.0], size=(days, 6, 4))
        last = None if rng.random() < 0.3 else rng.choice([np.nan, 50.0, 0.0], size=(days, 6, 4))
        cash = rng.choice([0.0, 100.0, -20.0], size=(days, 4))
        prices = {}
        for yf in ("AAPL", "MSFT", "VOD.L"):
            if rng.random() < 0.8:
                col = str(rng.choice(["price", "Close", "Adj Close", "Open"]))
                idx = grid if rng.random() < 0.5 else grid.tz_localize(None).date
                frame = pd.DataFrame({col: rng.uniform(10, 300, days)}, index=idx)
                frame.iloc[int(rng.integers(0, days))] = np.nan
                prices[yf] = frame
        fx = _fx_tables(rng, grid)
        acc_ccy = {"SCBS": "THB", "kim eng ": "THB", "London": str(rng.choice(["GBP", "EUR", "JPY"]))}
        include = set(rng.choice([0, 1, 2, 3], int(rng.integers(0, 5))).tolist())
        target = str(rng.choice(["USD", "THB", "EUR"]))
        manual = {"NOFEED": {"price": 9.5}} if rng.random() < 0.6 else None
        yf_map = {"AAPL": "AAPL", "MSFT": "MSFT", "VOD.L": "VOD.L"}
        outs = []
        for mod in (ref, new):
            p = {k: v.copy() for k, v in prices.items()}
            f = {k: v.copy() for k, v in fx.items()}
            out = call(
                lambda: mod._value_daily_holdings_vectorized(
                    grid, qty.copy(), None if last is None else last.copy(), cash.copy(), p, f, target,
                    acc_ccy, id_to_symbol, id_to_account, yf_map, include, "USD", "1d", manual,
                ),
                caplog,
            )  # fmt: skip
            outs.append((out, canon(p), canon(f)))
        assert outs[0] == outs[1], (seed, trial)
