"""Phase 11d: edgar_provider.EdgarFactStore and the company session in Rust
(rust/investa_core/src/ef_store.rs) against the original frozen in
tests/reference/edgar_provider.py.

Twin stores hold the same random facts: restatements, same-day ties between
two filings, NULL values, quarterly spans with and without a start. Each side
then answers the same random calls through its own module — readers with
random tag sets, as-of dates, inside and outside a company session (nested
too), plus writes and the ingest logs. Return values (dict order included),
tables, logs and warnings must agree.
"""

import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import edgar_provider as new  # noqa: E402
from reference import edgar_provider as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

TAGS = new._all_chain_tags()[:30] + ["NotAChainTag"]


def facts(rng):
    rows, qrows = [], []
    for cik in ("0001", "0002"):
        for tag in TAGS:
            if rng.random() < 0.3:
                continue
            for year in range(2012, 2012 + int(rng.integers(1, 8))):
                for k in range(int(rng.integers(1, 3))):
                    filed = f"{year + 1 + k}-02-1{int(rng.integers(0, 3))}"
                    val = None if rng.random() < 0.05 else float(rng.normal(1e9, 3e8))
                    rows.append(
                        (
                            cik,
                            tag,
                            f"{year}-12-31",
                            None,
                            val,
                            "USD",
                            "10-K",
                            f"a{year}{k}",
                            filed,
                        )
                    )
                    if rng.random() < 0.1:
                        rows.append(
                            (
                                cik,
                                tag,
                                f"{year}-12-31",
                                None,
                                float(rng.normal(1e9, 1e8)),
                                "USD",
                                "10-K/A",
                                f"b{year}{k}",
                                filed,
                            )
                        )
                    for q in range(int(rng.integers(0, 3))):
                        start = "" if rng.random() < 0.2 else f"{year}-0{3 * q + 1}-01"
                        qrows.append(
                            (
                                cik,
                                tag,
                                start,
                                f"{year}-0{3 * q + 3}-30",
                                val,
                                "USD",
                                "10-Q",
                                f"q{year}{k}{q}",
                                [filed, None][int(rng.random() < 0.05)],
                            )
                        )
    return rows, qrows


@pytest.mark.parametrize("seed", range(5))
def test_random_calls_match(tmp_path, caplog, seed):
    rng = np.random.default_rng(8700 + seed)
    rows, qrows = facts(rng)
    stores = []
    for side, mod in (("ref", ref), ("new", new)):
        (tmp_path / side).mkdir()
        st = mod.EdgarFactStore(str(tmp_path / side / "facts.db"))
        st.upsert_facts(rows)
        st.upsert_quarterly_facts(qrows)
        stores.append((mod, st))
    for step in range(80):
        cik = str(rng.choice(["0001", "0002", "0003"]))
        tags = list(rng.choice(TAGS, int(rng.integers(0, 10)), replace=False))
        as_of = None if rng.random() < 0.6 else f"{int(rng.integers(2013, 2022))}-06-30"
        session = rng.random() < 0.5
        nested = rng.random() < 0.2
        names = [
            "get_many_tag_series",
            "get_tag_series_by_filing",
            "get_many_tag_spans",
            "get_many_tag_spans_by_filing",
            "get_tag_revisions",
        ]
        name = str(rng.choice(names + ["get_tag_series", "has_data", "write", "state"]))
        outs = []
        for mod, st in stores:

            def op():
                if name == "get_tag_series":
                    return st.get_tag_series(cik, tags[0] if tags else "X", as_of)
                if name == "has_data":
                    return st.has_data(cik)
                if name == "write":
                    st.mark_ingested(cik, len(tags))
                    st.mark_quarterly_ingested(cik, len(tags))
                    return st.upsert_facts(rows[: len(tags)])
                if name == "state":
                    return st.quarterly_ingest_state(cik), sorted(st.ingested_ciks())
                return [
                    getattr(st, name)(cik, tags, as_of),
                    getattr(st, name)(cik, tags[: len(tags) // 2], as_of),
                ]

            def in_session():
                if not session:
                    return op()
                with mod.company_session(cik):
                    if nested:
                        with mod.company_session(cik):
                            return op()
                    return op()

            out = call(in_session, caplog)
            outs.append(repr(out).replace("/new/facts.db", "/ref/facts.db"))
        assert outs[0] == outs[1], (seed, step, name, outs)
        assert ref._session_var.get() is None and new._session_var.get() is None
        dumps = []
        for _, st in stores:
            c = sqlite3.connect(st.db_path)
            tabs = [
                r[0]
                for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            dumps.append(
                [
                    (
                        t,
                        canon(
                            c.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall()
                        ),
                    )
                    for t in tabs
                    if "log" not in t
                ]
            )
            c.close()
        assert dumps[0] == dumps[1], (seed, step)
