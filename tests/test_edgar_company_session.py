"""edgar_provider.company_session: reads inside a session equal reads outside.

The session shares one connection and loads each tag once, answering later
subsets from what it holds. On a random store (restatements, same-day ties
between two filings, NULL values, several companies), every reader call made
inside a session must equal the same call made without one. That holds for
subsets, the prefetched tags, as-of dates, a nested session and another
company's reads, and a caller mutating what it got back must not change later
answers. A session's connection never crosses threads.
"""

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import edgar_provider as E  # noqa: E402

TAGS = E._all_chain_tags()[:40] + ["NotAChainTag"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = E.EdgarFactStore(db_path=str(tmp_path / "facts.db"))
    monkeypatch.setattr(E, "_shared_store", s)
    rng = np.random.default_rng(7700)
    rows = []
    for cik in ("0001", "0002"):
        for tag in TAGS:
            if rng.random() < 0.3:
                continue
            for year in range(2010, 2010 + int(rng.integers(1, 12))):
                end = f"{year}-12-31"
                for k in range(int(rng.integers(1, 4))):  # restatements
                    filed = f"{year + 1 + k}-02-{10 + int(rng.integers(0, 3)):02d}"
                    val = None if rng.random() < 0.05 else float(rng.normal(1e9, 3e8))
                    rows.append(
                        (
                            cik,
                            tag,
                            end,
                            None,
                            val,
                            "USD",
                            "10-K",
                            f"acc-{year}-{k}-a",
                            filed,
                        )
                    )
                    if rng.random() < 0.1:  # a second filing the same day
                        rows.append(
                            (
                                cik,
                                tag,
                                end,
                                None,
                                float(rng.normal(1e9, 3e8)),
                                "USD",
                                "10-K/A",
                                f"acc-{year}-{k}-b",
                                filed,
                            )
                        )
    s.upsert_facts(rows)
    return s


def _calls(rng):
    for _ in range(40):
        tags = list(rng.choice(TAGS, int(rng.integers(1, 12)), replace=False))
        as_of = None if rng.random() < 0.6 else f"{int(rng.integers(2011, 2024))}-06-30"
        yield str(rng.choice(["series", "by_filing"])), tags, as_of


def _read(store, cik, reader, tags, as_of):
    if reader == "series":
        return store.get_many_tag_series(cik, tags, as_of=as_of)
    return store.get_tag_series_by_filing(cik, tags, as_of=as_of)


def _canon(x):
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    return x


def test_session_reads_equal_plain_reads(store):
    rng = np.random.default_rng(7701)
    calls = list(_calls(rng))
    plain = [_read(store, "0001", *c) for c in calls]
    with E.company_session("0001"):
        inside = [_read(store, "0001", *c) for c in calls]
        other = _read(
            store, "0002", "series", TAGS, None
        )  # another company: no session
        with E.company_session("0001"):  # nested: the outer session serves
            nested = [_read(store, "0001", *c) for c in calls]
    assert [_canon(x) for x in inside] == [_canon(x) for x in plain]
    assert [_canon(x) for x in nested] == [_canon(x) for x in plain]
    assert _canon(other) == _canon(_read(store, "0002", "series", TAGS, None))
    # Within a tag the order is part of the answer (periods ascending, filings newest first).
    for a, b in zip(inside, plain):
        for tag in b:
            assert (list(a[tag].items()) if isinstance(b[tag], dict) else a[tag]) == (
                list(b[tag].items()) if isinstance(b[tag], dict) else b[tag]
            )


def test_mutating_an_answer_does_not_leak(store):
    with E.company_session("0001"):
        first = store.get_many_tag_series("0001", TAGS)
        for series in first.values():
            series.clear()
        filings = store.get_tag_series_by_filing("0001", TAGS)
        for lst in filings.values():
            for f in lst:
                f.clear()
            lst.clear()
        assert _canon(store.get_many_tag_series("0001", TAGS)) == _canon(
            E.EdgarFactStore(store.db_path).get_many_tag_series("0001", TAGS)
        )
        assert _canon(store.get_tag_series_by_filing("0001", TAGS)) == _canon(
            E.EdgarFactStore(store.db_path).get_tag_series_by_filing("0001", TAGS)
        )


def test_concept_values_and_statements_match(store):
    plain = (E.get_concept_values("0001"), E.get_statements("0001"))
    with E.company_session("0001"):
        inside = (E.get_concept_values("0001"), E.get_statements("0001"))
    assert _canon(inside[0]) == _canon(plain[0])
    for name in plain[1]:
        a, b = inside[1][name], plain[1][name]
        assert (
            a.equals(b)
            and list(a.index) == list(b.index)
            and list(a.columns) == list(b.columns)
        )


def test_another_thread_reads_without_the_session(store):
    import contextvars

    errors = []
    with E.company_session("0001"):
        store.get_many_tag_series("0001", TAGS[:3])
        ctx = contextvars.copy_context()  # what asyncio.to_thread would carry over

        def work():
            try:
                assert E._session_for("0001") is None
                store.get_many_tag_series("0001", TAGS[:3])
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        t = threading.Thread(target=ctx.run, args=(work,))
        t.start()
        t.join()
    assert not errors
    assert E._session_var.get() is None
