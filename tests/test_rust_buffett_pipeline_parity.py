"""Phase 15f: buffett_pipeline in Rust (rust/investa_core/src/bp.rs) against
the original frozen in tests/reference/buffett_pipeline_p15.py.

The archive (closes, cached share counts, data-quality flags) and the
provider's share-count batches are faked: missing closes, stale counts,
batches that fail, zero and text share counts, a quality scan that raises,
and high/low severity flags. `fetch_market_data`, the price-quality gate and
the exclusion records are compared directly; a whole run over a slice of the
real universe is compared in the real-data check (it needs the fact store).
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_pipeline as new  # noqa: E402
import market_data  # noqa: E402
import market_db  # noqa: E402
from reference import buffett_pipeline_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402


class FakeDb:
    def __init__(self, rng, upserts, mode):
        self.rng, self.upserts, self.mode = rng, upserts, mode

    def get_latest_closes(self, symbols):
        r = np.random.default_rng(len(symbols))
        return {s: float(r.uniform(5, 300)) for s in symbols if r.random() < 0.85}

    def get_share_counts(self, symbols, max_age_days=30):
        r = np.random.default_rng(len(symbols) + 1)
        return {s: float(r.uniform(1e7, 1e9)) for s in symbols if r.random() < 0.5}

    def upsert_share_counts(self, counts, when):
        self.upserts.append((sorted(counts.items()), when))

    def get_data_quality(self, symbols):
        if self.mode == "raise":
            raise RuntimeError("no scan yet")
        return {s: {"severity": ["high", "low"][i % 2]} for i, s in enumerate(symbols) if i % 3 == 0}


class FakeProvider:
    def __init__(self, fail_every):
        self.calls = 0
        self.fail_every = fail_every

    def get_ticker_details_batch(self, symbols):
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise ConnectionError("rate limited")
        return {s: {"sharesOutstanding": [1e8, 0, None, "2e8"][hash(s) % 4]} for s in sorted(symbols)}


@pytest.mark.parametrize("seed", range(6))
def test_market_data_and_gates_match(caplog, monkeypatch, seed):
    rng = np.random.default_rng(16400 + seed)
    for trial in range(6):
        symbols = [f"S{i:04d}" for i in range(int(rng.integers(0, 500)))]
        mode = str(rng.choice(["ok", "raise"]))
        fail_every = int(rng.choice([0, 2, 3]))
        outs = []
        for mod in (ref, new):
            upserts = []
            monkeypatch.setattr(market_db, "MarketDatabase", lambda: FakeDb(rng, upserts, mode))
            monkeypatch.setattr(market_data, "get_shared_mdp", lambda: FakeProvider(fail_every))
            got = call(lambda: mod.fetch_market_data(symbols), caplog)
            companies = [
                SimpleNamespace(symbol=s, cik=f"{i:010d}", name=s, model="generic", gate_failures=[] if i % 4 else ["thin"],
                                period_count=i % 12, coverage=0.5)
                for i, s in enumerate(symbols + [""])
            ]  # fmt: skip
            gated = call(lambda: mod._apply_price_quality_gate(companies), caplog)
            excl = call(lambda: mod._exclusion_records(companies), caplog)
            outs.append((got, upserts, gated, excl))
        assert outs[0] == outs[1], (seed, trial, mode, fail_every)
