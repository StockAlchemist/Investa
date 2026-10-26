"""Phase 15g: the glue of edgar_provider in Rust (rust/investa_core/src/ep.rs)
against the original frozen in tests/reference/edgar_provider_p15.py.

Synthetic companyfacts documents stand in for filers: random fiscal calendars,
10-Qs tagging the quarter, the year-to-date figure or both, 10-Ks with two
years of comparatives, restatements, a stock split that later filings restate
and earlier ones do not, period ends mistyped by a few days, missing rungs,
8-Ks and amendments, and tags no concept reads. Both parsers run over them and
the rows go into one real fact store, which every reader then answers from —
concept values and provenance, the split-consistent series, revisions, split
factors, the annual and quarterly statements, sector concepts — with and
without an `as_of` date. The pure helpers also take hand-made odd inputs
(invalid and aware dates, None and zero values, Unicode digits, object and
duplicate frames), and the ingest paths run against separate stores, a fake
SEC fetch and a synthetic bulk archive. Results, logs and stores agree.
"""

import calendar
import copy
import datetime as dt
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import edgar_concepts  # noqa: E402
import edgar_provider as new  # noqa: E402
from reference import edgar_provider_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call, canon, tables  # noqa: E402

CHAINS = edgar_concepts.all_concepts()
BALANCE_TAGS = {t for n in edgar_concepts.BALANCE_CONCEPTS for t in CHAINS[n]} | {
    t for n in ("deposits", "loans", "real_estate_net") for t in CHAINS.get(n, [])
}
ALL_TAGS = [t for chain in CHAINS.values() for t in chain]
SHARE_TAGS = [t for n in ("shares_diluted", "shares_basic", "shares_outstanding") for t in CHAINS[n]]
EPS_TAGS = [t for n in ("eps_diluted", "eps_basic") for t in CHAINS[n]]


def _mend(year, month):
    while month > 12:
        year, month = year + 1, month - 12
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _payload(rng, cik):
    """One filer's companyfacts document."""
    chosen = set(rng.choice(ALL_TAGS, size=int(rng.integers(8, 70)), replace=False))
    if rng.random() < 0.8:
        chosen |= set(rng.choice(SHARE_TAGS, size=2, replace=False)) | {EPS_TAGS[0]}
    fy_month = int(rng.choice([12, 12, 6, 9, 3]))
    split = None
    if rng.random() < 0.5:
        split = (dt.date(int(rng.integers(2012, 2023)), int(rng.integers(1, 13)), 15), float(rng.choice([2, 4, 7, 1.5])))
    first, last = int(rng.integers(2009, 2016)), int(rng.integers(2018, 2025))
    us_gaap = {"SomeCompanyExtensionTag": {"units": {"USD": [{"end": "2020-12-31", "val": 1, "form": "10-K", "filed": "2021-02-01"}]}}}
    for tag in chosen:
        instant = tag in BALANCE_TAGS or tag.startswith("CommonStockShares")
        per_share = tag in EPS_TAGS or "PerShare" in tag
        shares = (tag in SHARE_TAGS or "Shares" in tag) and not per_share
        unit = "USD/shares" if per_share else "shares" if shares else "USD"
        base = float(rng.uniform(1e6, 5e9)) if not per_share else float(rng.uniform(0.1, 5))
        quarters = {}  # true (start, end) -> value on the period's own basis
        this_year = None
        entries = []

        def basis(end, filed):
            if not split or not (shares or per_share):
                return 1.0
            day, ratio = split
            if end < day <= filed:
                return ratio if shares else 1 / ratio
            return 1.0

        def emit(start, end, value, form, filed, accn):
            if rng.random() < 0.03:  # a period end mistyped in one filing
                end = end + dt.timedelta(days=int(rng.choice([-3, -2, -1, 1, 2])))
            val = value * basis(end, filed)
            if rng.random() < 0.05:
                val *= float(rng.choice([1.03, 0.9, 1.2]))  # a restatement
            val = round(val) if (rng.random() < 0.3 and not per_share) else round(val, 4)
            e = {"end": end.isoformat(), "val": val, "accn": accn, "form": form, "filed": filed.isoformat()}
            if start is not None:
                e["start"] = start.isoformat()
            entries.append(e)

        for year in range(first, last + 1):
            fy_end = _mend(year, fy_month)
            fy_start = _mend(year - 1, fy_month) + dt.timedelta(days=1)
            q_ends = [_mend(year - 1, fy_month + 3 * k) for k in (1, 2, 3)] + [fy_end]
            if rng.random() < 0.1:  # a 16-week fourth quarter
                q_ends[2] = q_ends[2] - dt.timedelta(days=28)
            q_vals = [base * (1 + 0.02 * (year - 2009)) * float(rng.uniform(0.8, 1.2)) for _ in range(4)]
            if split and fy_end >= split[0] and (shares or per_share):
                q_vals = [v * (split[1] if shares else 1 / split[1]) for v in q_vals]
            starts = [fy_start] + [q + dt.timedelta(days=1) for q in q_ends[:3]]
            for k in range(4):
                quarters[(starts[k], q_ends[k])] = q_vals[k]
            prior_year, this_year = (this_year if year > first else None), [(starts[k], q_ends[k], q_vals[k]) for k in range(4)]
            for k in range(3):  # the 10-Qs
                filed = q_ends[k] + dt.timedelta(days=int(rng.integers(25, 45)))
                accn = f"{cik:010d}-{year % 100:02d}-Q{k + 1}"
                form = "10-Q/A" if rng.random() < 0.03 else "10-Q"
                if instant:
                    emit(None, q_ends[k], base * float(rng.uniform(0.9, 1.1)), form, filed, accn)
                    continue
                if k == 0 or rng.random() < 0.75:
                    emit(starts[k], q_ends[k], q_vals[k], form, filed, accn)
                if k > 0 and rng.random() < 0.85:
                    emit(fy_start, q_ends[k], sum(q_vals[: k + 1]), form, filed, accn)
                if prior_year and rng.random() < 0.4:  # last year's quarter as a comparative
                    emit(*prior_year[k], form, filed, accn)
            filed = fy_end + dt.timedelta(days=int(rng.integers(40, 80)))
            accn = f"{cik:010d}-{year % 100:02d}-K"
            for back in range(3 if rng.random() < 0.8 else 1):  # comparatives
                y_end = _mend(year - back, fy_month)
                y_start = _mend(year - back - 1, fy_month) + dt.timedelta(days=1)
                if year - back < first:
                    break
                total = sum(v for (s, e), v in quarters.items() if y_start <= s and e <= y_end)
                if instant:
                    emit(None, y_end, base * float(rng.uniform(0.9, 1.1)), "10-K", filed, accn)
                elif total:
                    emit(y_start, y_end, total, "10-K", filed, accn)
            if rng.random() < 0.05:
                emit(fy_start, fy_end, base, "8-K", filed, accn + "-8")
        us_gaap[tag] = {"label": tag, "units": {unit: entries}}
    return {"cik": cik, "entityName": f"Co {cik}", "facts": {"dei": {}, "us-gaap": us_gaap}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = new.EdgarFactStore(str(tmp_path / "facts.db"))
    monkeypatch.setattr(new, "_shared_store", s)
    monkeypatch.setattr(ref, "_shared_store", s)
    return s


def _both(caplog, name, *args, **kwargs):
    a = call(lambda: getattr(ref, name)(*copy.deepcopy(args), **kwargs), caplog)
    b = call(lambda: getattr(new, name)(*copy.deepcopy(args), **kwargs), caplog)
    assert a == b, (name, args, kwargs)
    return a


@pytest.mark.parametrize("seed", range(8))
def test_filers_read_alike(caplog, store, seed):
    rng = np.random.default_rng(17000 + seed)
    ciks = []
    for i in range(4):
        cik = int(rng.integers(1, 2_000_000))
        payload = _payload(rng, cik)
        wanted = None if i % 2 else set(rng.choice(ALL_TAGS, size=40, replace=False))
        for parser in ("parse_company_facts", "parse_company_quarterly_facts"):
            _both(caplog, parser, payload, wanted)
        store.upsert_facts(new.parse_company_facts(payload))
        store.mark_ingested(f"{cik:010d}", 1)
        store.upsert_quarterly_facts(new.parse_company_quarterly_facts(payload))
        ciks.append(f"{cik:010d}")
    ciks.append("0000000000")  # a filer the store has never seen

    concepts = list(CHAINS)
    for cik in ciks:
        for as_of in (None, "2016-06-30", "2021-03-01"):
            with new.as_of(as_of):
                for subset in (None, [], ["revenue", "shares_diluted", "nope"], list(rng.choice(concepts, 8))):
                    _both(caplog, "get_concept_values", cik, subset)
                    _both(caplog, "get_concept_provenance", cik, subset)
                    _both(caplog, "get_quarterly_concept_values", cik, subset)
                for concept in ("shares_diluted", "shares_basic", "shares_outstanding", "eps_diluted", "revenue", "nope"):
                    _both(caplog, "split_consistent_series", cik, concept)
                    _both(caplog, "_values_by_filing", cik, concept)
                for mc in (0.01, 0.0, 0.5):
                    _both(caplog, "revisions", cik, None, mc)
                _both(caplog, "revisions", cik, ["revenue", "eps_diluted", "equity"], 0.01)
                for name in ("split_adjustment_factors", "get_statements", "get_quarterly_statements", "_quarterly_basis_factors"):
                    _both(caplog, name, cik)
                for model in ("bank", "reit", "generic"):
                    _both(caplog, "get_sector_concepts", cik, model)
                values = new.get_concept_values(cik)
                factors = new.split_adjustment_factors(cik)
                for f in (factors, {}, {p: 0.0 for p in factors}):
                    a, b = copy.deepcopy(values), copy.deepcopy(values)
                    assert call(lambda: ref._apply_split_adjustment(a, f), caplog) == call(lambda: new._apply_split_adjustment(b, f), caplog)
                    assert canon(a) == canon(b)
    for concepts_arg in (("shares_diluted", "eps_basic"), (), ("nope",)):
        _both(caplog, "_tags_for", concepts_arg)
    _both(caplog, "_all_chain_tags")
    _both(caplog, "concept_labels")


def test_parsers_take_odd_documents(caplog):
    good = {"end": "2020-12-31", "start": "2020-01-01", "val": 5, "form": "10-K", "accn": "a", "filed": "2021-01-05"}
    tag = ALL_TAGS[0]
    odd_entries = [
        good,
        {**good, "form": None},
        {**good, "form": "10-Q", "start": "2020-10-01"},
        {**good, "end": ""},
        {**good, "end": None},
        {**good, "start": ""},
        {**good, "start": None},
        {**good, "start": "2020-13-01"},
        {**good, "start": 20200101},
        {**good, "start": "2020-01-01T00:00:00+00:00"},
        {**good, "start": "2020-01-01T00:00:00"},
        {**good, "start": "2019-12-01"},
        {**good, "start": "2020-12-31"},
        {**good, "start": "2021-01-01"},
        {k: v for k, v in good.items() if k != "accn"},
        {"end": "2020-12-31", "form": "10-K/A"},
    ]
    payloads = [
        {"cik": 320193, "facts": {"us-gaap": {tag: {"units": {"USD": odd_entries}}, "NotWanted": {"units": {"USD": [good]}}}}},
        {"cik": "320193", "facts": {"us-gaap": {tag: {"units": {}}, ALL_TAGS[1]: {}}}},
        {"cik": None, "facts": {"us-gaap": {tag: {"units": {"USD": [good]}}}}},
        {"facts": {"us-gaap": {tag: {"units": {"USD": [good]}}}}},
        {"cik": 1, "facts": {}},
        {"cik": 1},
        {"cik": 1, "facts": {"us-gaap": {}}},
        [1, 2],
        "text",
        {"cik": 1, "facts": {"us-gaap": {tag: {"units": {"USD": [{**good, "form": ["10-K"]}]}}}}},  # unhashable form
        {"cik": 1, "facts": {"us-gaap": {tag: {"units": {"USD": [good, 7]}}}}},
    ]
    for payload in payloads:
        for wanted in (None, {tag}, set()):
            for parser in ("parse_company_facts", "parse_company_quarterly_facts"):
                _both(caplog, parser, payload, wanted)


def test_pure_helpers_take_odd_inputs(caplog):
    dates = [None, "", "2020-01-01", "2020-12-31", "2021-01-03", "2020-02-30", "2020-01-01T00:00:00+00:00", 5, "x"]
    for s in dates:
        for e in dates:
            _both(caplog, "_span_days", s, e)
            _both(caplog, "_is_annual_duration", s, e)
    for values in ([], [1], [3, 1], [2.5, 1, 4], [float("nan"), 1, 2], [1, "a"], [np.float64(2), 1, 3, 7]):
        _both(caplog, "_median", values)

    rng = np.random.default_rng(17100)
    for trial in range(300):
        ends = sorted({f"2020-{m:02d}-{d:02d}" for m, d in zip(rng.integers(1, 13, 12), rng.integers(1, 29, 12))})
        ends += ["2020-02-30", "bad", ""][: int(rng.integers(0, 4))]
        starts = ["", "2020-01-01", "2019-10-01", "2019-07-01", "2020-04-01", None, "2020-13-01"]
        spans = {}
        for _ in range(int(rng.integers(0, 25))):
            key = (str(rng.choice(np.array(starts, dtype=object))) if rng.random() < 0.9 else None, str(rng.choice(ends)) if ends else "2020-03-31")
            value = [float(rng.normal(100, 50)), int(rng.integers(-5, 50)), 0, 0.0, None, float("nan"), np.float64(3.5)][int(rng.choice(7, p=[0.6, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05]))]
            spans[key] = (value, "USD")
        tagged = {"A": spans, "B": dict(list(spans.items())[: len(spans) // 2])}
        _both(caplog, "_canonical_period_ends", tagged)
        canonical = ref._canonical_period_ends(tagged) if all(k[1] for k in spans) else {}
        _both(caplog, "_apply_canonical_ends", spans, canonical)
        for additive in (True, False, 0, "yes"):
            _both(caplog, "_derive_quarterly_series", spans, additive)
        filings = [((f"2021-0{k + 1}-01", f"acc{k}"), dict(list(spans.items())[k::3])) for k in range(3)]
        _both(caplog, "_filing_basis_factors", filings)
        factors = call(lambda: ref._filing_basis_factors(filings), caplog)
        if factors[0][0] == "ok":
            f = ref._filing_basis_factors(filings)
            for inverse in (False, True):
                _both(caplog, "_spans_on_latest_basis", filings, f, inverse)
                _both(caplog, "_spans_on_latest_basis", filings, {k: 0 for k in f}, inverse)

        series = {str(rng.choice(ends)) if ends else "2020-01-01": (float(rng.normal()), "USD") for _ in range(5)}
        _both(caplog, "resolve_concept", {"T1": series, "T2": {}, "T3": {"2020-03-31": (None, "USD")}}, ["T0", "T2", "T1", "T3"])
    _both(caplog, "resolve_concept", {"T1": {"2020-03-31": (1, "USD", "x")}}, ["T1"])
    _both(caplog, "resolve_concept", {}, None)
    _both(caplog, "_filed_ratio", [{"a": 0, "b": 2}, {"a": 4, "b": 2}], "a", "b")
    _both(caplog, "_filed_ratio", [{"a": 1}], "a", "b")
    _both(caplog, "_filed_ratio", [{"a": "x", "b": 2}], "a", "b")


def test_frames_and_derived_rows(caplog):
    rng = np.random.default_rng(17200)
    periods = ["2019-12-31", "2020-12-31", "2021-12-31", "2022-06-30", "FY2020", "２０２１-12-31", "2021-02-30", 2020]
    labels = dict(list(new._INCOME_LABELS.items()) + list(new._BALANCE_LABELS.items()) + list(new._CASHFLOW_LABELS.items()))
    for trial in range(200):
        values = {}
        for concept in list(labels) + list(new._NEGATE_ON_EMIT) + ["short_term_debt", "long_term_debt"]:
            if rng.random() < 0.5:
                continue
            chosen = rng.choice(len(periods), size=int(rng.integers(0, 6)), replace=False)
            values[concept] = {periods[i]: [float(rng.normal(1e6, 1e6)), int(rng.integers(0, 100)), np.float64(2.0), float("nan")][int(rng.choice(4, p=[0.7, 0.15, 0.1, 0.05]))] for i in chosen}
        _both(caplog, "_period_stamps", values)
        for which in (new._INCOME_LABELS, new._BALANCE_LABELS, {}):
            _both(caplog, "_frame_from_concepts", values, which)
        _both(caplog, "_frame_from_concepts", values, new._CASHFLOW_LABELS, new._NEGATE_ON_EMIT)
        frames = []
        for mod in (ref, new):
            inc = mod._frame_from_concepts(values, mod._INCOME_LABELS)
            bal = mod._frame_from_concepts(values, mod._BALANCE_LABELS)
            cf = mod._frame_from_concepts(values, mod._CASHFLOW_LABELS, negate=mod._NEGATE_ON_EMIT)
            if trial % 7 == 0 and not bal.empty:
                bal = bal.astype(object)
            cf = None if trial % 5 == 0 else cf
            got = call(lambda: mod._add_derived_rows(inc, bal, values, cf), caplog)
            frames.append((got, canon(inc), canon(bal), canon(cf)))
        assert frames[0] == frames[1], trial

    cols = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"])
    cases = [
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=cols), "A", [(cols[0], 9.0), (cols[0], 8.0)]),
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=cols), "B", [(cols[1], 9.0), (cols[2], 7.5)]),
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=cols), "B", [(cols[1], 9)]),
        (pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["A", "A"], columns=cols), "A", [(cols[1], 9.0)]),
        (pd.DataFrame([[1, 2, 3]], index=["A"], columns=cols), "B", [(cols[1], 9.0)]),
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=[cols[0], cols[0], cols[1]]), "B", [(cols[1], 9.0)]),
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=cols), "B", []),
        (pd.DataFrame([[1.0, 2.0, 3.0]], index=["A"], columns=cols), "B", [(cols[1], 9.0, 1)]),
    ]
    for frame, label, cells in cases:
        a, b = frame.copy(), frame.copy()
        assert call(lambda: ref._write_row(a, label, cells), caplog) == call(lambda: new._write_row(b, label, cells), caplog)
        assert canon(a) == canon(b), (label, cells)


def test_as_of_block(store):
    assert new._effective_as_of(None) is None
    with new.as_of("2015-01-01") as got:
        assert got is None
        assert new._effective_as_of(None) == "2015-01-01"
        with new.as_of(None):
            assert new._effective_as_of(None) is None
        assert new._effective_as_of("2020-01-01") == "2020-01-01"
    with pytest.raises(KeyError):
        with new.as_of("2016-01-01"):
            raise KeyError("inside")
    assert new._effective_as_of(None) is None
    assert ref.get_store() is new.get_store() is store


def test_ingest_paths(caplog, tmp_path, monkeypatch):
    rng = np.random.default_rng(17300)
    payloads = {f"{c:010d}": _payload(rng, c) for c in map(int, rng.integers(1, 9_000_000, 6))}
    archive = tmp_path / "companyfacts.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for cik, payload in payloads.items():
            z.writestr(f"CIK{cik}.json", json.dumps(payload))
        z.writestr("CIK0000000042.json", "{not json")
        z.writestr("README.txt", "ignore me")
        z.writestr("CIK0000000043.json", json.dumps({"facts": {}}))

    def fetch(url, **kwargs):
        cik = url.rsplit("CIK", 1)[1][:10]
        return payloads.get(cik)

    for side, mod in (("ref", ref), ("new", new)):
        monkeypatch.setattr(mod, "sec_get_json", fetch)
    targets = list(payloads)[:3] + ["0000000099"]
    results = {}
    for side, mod in (("ref", ref), ("new", new)):
        s = new.EdgarFactStore(str(tmp_path / f"{side}.db"))  # its path, in the logs, is masked below
        monkeypatch.setattr(mod, "_shared_store", s)
        out = []
        for ciks, limit, every in ((None, None, 250), (set(targets), None, 1), (None, 3, 2), ({"0000000042"}, None, 1)):
            out.append(call(lambda: mod.ingest_bulk(str(archive), ciks=ciks, limit=limit, progress_every=every), caplog))
        out.append(call(lambda: list(mod.iter_bulk_documents(str(archive), ciks=set(targets))), caplog))
        out.append(call(lambda: list(mod.iter_bulk_documents(str(tmp_path / "missing.zip"))), caplog))
        it = mod.iter_bulk_documents(str(archive))
        out.append(canon(next(it)))
        for cik in targets:
            out.append(call(lambda: mod.ingest_company(int(cik)), caplog))
            out.append(call(lambda: mod.ingest_company_quarterly(cik), caplog))
            out.append(call(lambda: mod.ingest_company_quarterly(cik), caplog))  # within the cooldown
            out.append(call(lambda: mod.ingest_company_quarterly(cik, max_age_days=0, empty_retry_hours=0), caplog))
        out.append(tables(s))
        results[side] = [repr(x).replace(f"{side}.db", "<side>.db") for x in out]
    for i, (a, b) in enumerate(zip(results["ref"], results["new"])):
        assert a == b, i
