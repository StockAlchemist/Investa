"""Phase 15a: valuation_history in Rust (rust/investa_core/src/vh.rs) against
the original frozen in tests/reference/valuation_history_p15.py.

The EDGAR reads (`split_consistent_series`, `get_concept_values` under
`as_of`) and the price store are faked with random, point-in-time-varying
figures: losses, missing operating cash flow or capex, share series that are
too short (so the fallbacks run), period keys that are not dates, prices with
gaps and non-positive closes, a store that raises, and a symbol with no
prices. Results and logs must agree.
"""

import contextlib
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import edgar_provider  # noqa: E402
import market_db  # noqa: E402
import valuation_history as new  # noqa: E402
from reference import valuation_history_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402


class FakeEdgar:
    def __init__(self, rng):
        self.rng = rng
        self.as_of_date = None
        years = range(2004, 2025)
        self.periods = [f"{y}-12-31" for y in years]
        self.shares = {p: float(rng.uniform(5e8, 2e9)) for p in self.periods}
        self.mode = int(rng.integers(0, 4))

    @contextlib.contextmanager
    def as_of(self, when):
        prev, self.as_of_date = self.as_of_date, when
        try:
            yield
        finally:
            self.as_of_date = prev

    def split_consistent_series(self, cik, concept):
        if cik == "BROKEN":
            raise RuntimeError("fact store offline")
        if self.mode == 0 and concept == "shares_diluted":
            return {p: v for p, v in list(self.shares.items())[-3:]}
        if self.mode == 1 and concept == "shares_diluted":
            return {}
        if self.mode == 2:
            return {**self.shares, "not-a-date": 1.0}
        return dict(self.shares)

    def get_concept_values(self, cik, concepts):
        seed = hash((self.as_of_date or "now", cik)) % (2**32)
        r = np.random.default_rng(seed)
        out = {}
        for c in concepts:
            vals = {}
            for p in self.periods:
                x = r.random()
                if x < 0.1:
                    continue
                vals[p] = float(r.normal(5e9, 4e9)) if c != "capex" else float(r.uniform(0, 3e9))
            out[c] = vals if r.random() < 0.95 else None
        return out


class FakePrices:
    mode = "ok"

    def get_ohlcv(self, symbol, start, end):
        if self.mode == "raise":
            raise OSError("db locked")
        if symbol == "NOPRICE":
            return None
        days = pd.date_range(start, end, freq="7D")
        close = 50 + np.abs(np.cumsum(np.random.default_rng(len(symbol)).normal(0, 2, len(days))))
        close[::13] = np.nan
        close[5] = -1.0
        return pd.DataFrame({"Close": close}, index=days)


@pytest.mark.parametrize("seed", range(10))
def test_bands_match(caplog, monkeypatch, seed):
    rng = np.random.default_rng(15600 + seed)
    monkeypatch.setattr(market_db, "MarketDatabase", FakePrices)
    for trial in range(8):
        fake = FakeEdgar(rng)
        for name in ("as_of", "split_consistent_series", "get_concept_values"):
            monkeypatch.setattr(edgar_provider, name, getattr(fake, name))
        FakePrices.mode = "raise" if rng.random() < 0.1 else "ok"
        symbol = str(rng.choice(["AAPL", "msft", "NOPRICE"]))
        cik = "BROKEN" if rng.random() < 0.1 else "0000320193"
        today = [date(2024, 6, 14), date(2024, 2, 29), date(2012, 3, 1)][int(rng.integers(0, 3))]
        got = call(lambda: new.bands(symbol, cik, today), caplog)
        want = call(lambda: ref.bands(symbol, cik, today), caplog)
        assert got == want, (seed, trial)


def test_helpers_match():
    rng = np.random.default_rng(1)
    for _ in range(200):
        pop = list(rng.normal(20, 5, int(rng.integers(0, 20))))
        v = float(rng.normal(20, 6))
        a, b = new._percentile_of(v, pop), ref._percentile_of(v, pop)
        assert (np.isnan(a) and np.isnan(b)) or a == b
        pct = float(rng.uniform(-5, 105))
        assert new._describe_percentile(pct) == ref._describe_percentile(pct)
    assert new._describe_percentile(float("nan")) == ref._describe_percentile(float("nan"))
    prices = pd.Series([1.0, 0.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    for when in (date(2019, 1, 1), date(2020, 2, 15), date(2021, 1, 1)):
        assert new._price_on_or_before(prices, when) == ref._price_on_or_before(prices, when)
    assert new._price_on_or_before(None, date(2020, 1, 1)) == ref._price_on_or_before(None, date(2020, 1, 1))
