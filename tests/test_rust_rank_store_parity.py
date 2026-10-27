"""Phase 11c: buffett_store.BuffettRankStore in Rust
(rust/investa_core/src/rank_store.rs) against the original frozen in
tests/reference/buffett_store.py.

Twin stores take the same random sequence of calls; after each, the return
values, every table's rows, the log records and the warnings must agree
(timestamps masked). The calls cover:
- runs started, finished and pruned, with `keep` from 0 up and stale windows
  that do or do not reach the unfinished runs;
- score frames with NumPy scalars, NaN, text, missing columns and non-string
  index labels, and exclusion records;
- the paged readers, with model filters, searches holding LIKE wildcards
  (`%`, `_`, `\\`) or only spaces, and pages past the end.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_store as new  # noqa: E402
from reference import buffett_store as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

SYMS = ["AAPL", "MSFT", "KO", "A_B", "100%", "C\\D"]


def tables(store):
    c = sqlite3.connect(store.db_path)
    try:
        names = [
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return [
            (n, canon(c.execute(f'SELECT * FROM "{n}" ORDER BY rowid').fetchall()))
            for n in names
        ]
    finally:
        c.close()


def same(result):
    """A result with each store's own directory written alike."""
    return repr(result).replace("/new/ranks.db", "/ref/ranks.db")


def scores(rng):
    rows = []
    for sym in rng.choice(SYMS, int(rng.integers(0, 6)), replace=False):
        r = {"cik": str(rng.integers(1, 99999)), "name": f"{sym} Corp", "model": str(rng.choice(["generic", "bank"])),
             "rank": np.int64(rng.integers(1, 50)), "composite_score": float(rng.normal(50, 10)),
             "quality_score": np.float64(rng.normal(50, 10)), "value_score": [np.nan, 3.5, None][int(rng.integers(0, 3))],
             "confidence": "high", "period_count": int(rng.integers(1, 20)), "latest_period": pd.Timestamp("2025-12-31")}  # fmt: skip
        if rng.random() < 0.2:
            del r["quality_score"]
        rows.append((sym, r))
    frame = pd.DataFrame([r for _, r in rows], index=[s for s, _ in rows])
    if rng.random() < 0.1 and len(frame):
        frame.index = range(len(frame))
    return frame


@pytest.mark.parametrize("seed", range(8))
def test_random_sequences_match(tmp_path, caplog, seed):
    rng = np.random.default_rng(8600 + seed)
    (tmp_path / "ref").mkdir()
    (tmp_path / "new").mkdir()
    a = ref.BuffettRankStore(str(tmp_path / "ref" / "ranks.db"))
    b = new.BuffettRankStore(str(tmp_path / "new" / "ranks.db"))
    runs = []
    for step in range(60):
        r = rng.random()
        if r < 0.15:
            params = {"weights": [0.6, 0.4], "n": int(rng.integers(1, 9))}
            ga, gb = (
                call(lambda: a.start_run(100, params), caplog),
                call(lambda: b.start_run(100, params), caplog),
            )
            assert same(ga) == same(gb), (seed, step)
            runs.append(a.latest_run_id() or 1)
            continue
        run = int(rng.integers(1, 8))
        if r < 0.3:
            f = scores(rng)
            ops = [lambda s: s.save_scores(run, f.copy())]
        elif r < 0.38:
            recs = [
                {
                    "symbol": str(x),
                    "cik": "1",
                    "reasons": "no_fundamentals",
                    "period_count": 2,
                }
                for x in rng.choice(SYMS, 3)
            ]
            ops = [lambda s: s.save_exclusions(run, [dict(x) for x in recs])]
        elif r < 0.48:
            ops = [lambda s: s.finish_run(run, 10, 5)]
        elif r < 0.53:
            keep, stale = int(rng.integers(0, 4)), float(rng.choice([6, 0, -1]))
            ops = [lambda s: s.prune_runs(keep, stale)]
        else:
            search = [None, "", "  ", "aa", "%", "_", "\\", "corp"][
                int(rng.integers(0, 8))
            ]
            model = [None, "bank", "generic"][int(rng.integers(0, 3))]
            rid = [None, run, 0][int(rng.integers(0, 3))]
            ops = [
                lambda s: s.latest_run_id(),
                lambda s: s.get_ranked(rid, 3, int(rng_offsets[0]), model, search),
                lambda s: s.count_ranked(rid, model, search),
                lambda s: s.get_exclusions(rid, 5, 0, search),
                lambda s: s.count_exclusions(rid, search),
                lambda s: s.get_run(rid),
                lambda s: s.get_scores_frame(rid),
                lambda s: s.get_symbol_history(str(rng_sym[0]), 5),
            ]
            rng_offsets = [rng.integers(0, 4)]
            rng_sym = [rng.choice(SYMS).lower()]
        for op in ops:
            ga, gb = call(lambda: op(a), caplog), call(lambda: op(b), caplog)
            assert same(ga) == same(gb), (seed, step, ga, gb)
        assert tables(a) == tables(b), (seed, step)


def test_helpers_match():
    for v in [
        None,
        np.nan,
        1,
        np.int64(3),
        np.float64(2.5),
        "x",
        pd.Timestamp("2025-01-01"),
        pd.NaT,
        [1, 2],
        np.array([1]),
        True,
        1.5,
        b"b",
    ]:
        assert canon(call(lambda: ref._coerce(v), _NoLog())) == canon(
            call(lambda: new._coerce(v), _NoLog())
        ), v
    for t in ["a%b_c\\d", "", "plain"]:
        assert ref._escape_like(t) == new._escape_like(t)


class _NoLog:
    records = []

    def clear(self):
        pass

    def at_level(self, level):
        import contextlib

        return contextlib.nullcontext()
