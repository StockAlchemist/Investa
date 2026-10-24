"""The Rust IRR/NPV (rust/investa_core/src/irr.rs) against what it replaced.

tests/reference/finance.py freezes finutils.calculate_npv/calculate_irr as
they were before the port. Results must be bitwise equal. The two SciPy root
finders the IRR relies on are also compared directly with SciPy, including the
paths that raise (no convergence, same-sign brackets, NaN mid-search), and so
is the CPython sum() the IRR's pre-checks use.
"""

import os
import struct
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from scipy import optimize

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils  # noqa: E402
import investa_core  # noqa: E402
from reference import finance as ref  # noqa: E402


def _bits(x):
    return struct.pack("<d", float(x))


def _same(a, b):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or _bits(a) == _bits(b)


def _flows_case(rng):
    """A random IRR input: dates (sometimes datetimes) and mixed-type flows."""
    n = int(rng.integers(2, 40))
    start = date(2005, 1, 1) + timedelta(days=int(rng.integers(0, 5000)))
    gaps = rng.integers(0, 400, n)
    gaps[0] = 0
    if rng.random() < 0.05:
        gaps[:] = 0
    dates = [start + timedelta(days=int(g)) for g in np.cumsum(gaps)]
    if rng.random() < 0.1:
        dates = [
            datetime(d.year, d.month, d.day, int(rng.integers(0, 24)), int(rng.integers(0, 60)))
            for d in dates
        ]
    scale = float(rng.choice([1.0, 100.0, 1e4, 1e7, 1e9]))
    flows = list(rng.normal(0, scale, n))
    flows[0] = -abs(flows[0]) if rng.random() < 0.9 else abs(flows[0])
    if rng.random() < 0.3:
        flows[-1] = abs(flows[-1]) * float(rng.choice([0.5, 1.0, 1.5, 3.0, 20.0]))
    for i in range(n):
        r = rng.random()
        if r < 0.15:
            flows[i] = 0.0
        elif r < 0.2:
            flows[i] = 1e-10
    if rng.random() < 0.05:
        flows = [-abs(f) for f in flows]
    if rng.random() < 0.05:
        flows[-1] = 1e-7
    style = rng.random()
    if style < 0.5:
        flows = [float(f) for f in flows]
    elif style < 0.8:
        flows = [np.float64(f) if rng.random() < 0.3 else float(f) for f in flows]
    else:
        flows = [np.float64(f) for f in flows]
    return dates, flows


@pytest.mark.parametrize("seed", range(4))
def test_irr_matches_reference(seed):
    rng = np.random.default_rng(100 + seed)
    for _ in range(500):
        dates, flows = _flows_case(rng)
        want = ref.calculate_irr(list(dates), list(flows))
        got = finutils.calculate_irr(list(dates), list(flows))
        assert _same(got, want), (got, want, flows[:5])


def test_irr_input_edge_cases_match_reference():
    d = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 6, 30)]
    cases = [
        ([], []),
        (d[:1], [-1.0]),
        (d, [-100.0, 50.0]),  # length mismatch
        ([d[0]] * 3, [-100.0, 10.0, 120.0]),  # one date
        (d, [-100.0, float("inf"), 120.0]),
        (d, [-100.0, float("nan"), 120.0]),
        (d, [-100.0, np.int64(5), 120.0]),  # not int/float to isinstance
        (d, [-100, 5, 120]),  # plain ints
        (d, [-100.0, True, 120.0]),
        (d, [0.0, 0.0, 0.0]),
        (d, [0.0, -100.0, 0.0]),
        (d, [100.0, -50.0, 10.0]),
        (d, [-100.0, -50.0, -10.0]),
        (["2020-01-01", "2021-01-01", "2022-01-01"], [-100.0, 10.0, 120.0]),
        ([d[0], datetime(2021, 1, 1), d[2]], [-100.0, 10.0, 120.0]),  # mixed types
        (list(reversed(d)), [-100.0, 10.0, 120.0]),  # unsorted
    ]
    for dates, flows in cases:
        want = ref.calculate_irr(list(dates), list(flows))
        got = finutils.calculate_irr(list(dates), list(flows))
        assert _same(got, want), (dates, flows, got, want)


@pytest.mark.parametrize("seed", range(2))
def test_npv_matches_reference(seed):
    rng = np.random.default_rng(200 + seed)
    for _ in range(400):
        dates, flows = _flows_case(rng)
        if any(isinstance(d, datetime) for d in dates):
            continue
        for rate in (-1.5, -1.0, -0.999999, -0.5, 0.0, 0.07, 3.0, 1e3, 1e8, float("nan")):
            want = ref.calculate_npv(rate, list(dates), list(flows))
            got = finutils.calculate_npv(rate, list(dates), list(flows))
            assert _same(got, want), (rate, got, want)


def test_npv_length_mismatch_still_raises():
    with pytest.raises(ValueError):
        finutils.calculate_npv(0.1, [date(2020, 1, 1)], [1.0, 2.0])


def _npv_problem(rng):
    n = int(rng.integers(2, 25))
    days = np.concatenate([[0], np.cumsum(rng.integers(1, 300, n - 1))]).astype(np.int64)
    dates = [date(2010, 1, 1) + timedelta(days=int(x)) for x in days]
    flows = list(rng.normal(0, 100, n))
    flows[0] = -abs(flows[0]) - 1
    flows[-1] = abs(flows[-1]) * float(rng.uniform(0.2, 4))
    return days, dates, flows


def test_secant_newton_matches_scipy():
    rng = np.random.default_rng(300)
    hits = {"converged": 0, "raised": 0}
    for _ in range(600):
        days, dates, flows = _npv_problem(rng)
        x0 = float(rng.choice([0.1, 0.0, -0.2, 0.5, -0.5, -0.95, 5.0]))
        maxiter = int(rng.choice([1, 2, 5, 50]))
        try:
            want = optimize.newton(
                ref.calculate_npv, x0=x0, args=(dates, flows), tol=1e-6, maxiter=maxiter
            )
        except Exception:
            want = None
        got = investa_core.newton_npv(days.tolist(), flows, x0, 1e-6, maxiter)
        hits["raised" if want is None else "converged"] += 1
        assert (want is None) == (got is None), (want, got)
        if want is not None:
            assert _same(got, want), (got, want)
    assert hits["raised"] > 50 and hits["converged"] > 50, hits


def test_brentq_matches_scipy():
    rng = np.random.default_rng(400)
    hits = {"converged": 0, "raised": 0}
    for _ in range(600):
        days, dates, flows = _npv_problem(rng)
        a = float(rng.uniform(-0.99, 0.0))
        b = float(rng.uniform(0.0, 50.0))
        if rng.random() < 0.2:
            a, b = b, b * 2 + 1  # likely the same sign at both ends
        maxiter = int(rng.choice([1, 3, 10, 100]))
        try:
            want = optimize.brentq(
                ref.calculate_npv, a=a, b=b, args=(dates, flows),
                xtol=1e-6, rtol=1e-6, maxiter=maxiter,
            )  # fmt: skip
        except Exception:
            want = None
        got = investa_core.brentq_npv(days.tolist(), flows, a, b, 1e-6, 1e-6, maxiter)
        hits["raised" if want is None else "converged"] += 1
        assert (want is None) == (got is None), (a, b, maxiter, want, got)
        if want is not None:
            assert _same(got, want), (got, want)
    assert hits["raised"] > 50 and hits["converged"] > 50, hits


def test_brentq_stops_on_a_nan_value_like_scipy():
    """SciPy's wrapper raises when f returns NaN mid-search."""
    days = [0, 365]
    flows = [-100.0, 150.0]
    # A bracket reaching below -1 makes npv NaN at that end.
    with pytest.raises(ValueError):
        optimize.brentq(
            ref.calculate_npv, a=-2.0, b=1.0, args=([date(2020, 1, 1), date(2021, 1, 1)], flows)
        )
    assert investa_core.brentq_npv(days, flows, -2.0, 1.0, 1e-6, 1e-6, 100) is None


def test_scan_grid_is_numpys():
    grid = np.concatenate(
        [
            -1.0 + np.logspace(-8, 0, 40),
            np.linspace(0.1, 10.0, 34)[1:],
            np.logspace(1.05, 5, 15),
        ]
    )
    rust = np.array(investa_core.irr_scan_grid())
    assert rust.tobytes() == grid.tobytes()


def test_py_sum_matches_cpython():
    rng = np.random.default_rng(500)
    for _ in range(3000):
        n = int(rng.integers(0, 12))
        items = []
        for _ in range(n):
            r = rng.random()
            v = float(rng.normal(0, 10 ** float(rng.integers(-3, 17))))
            if r < 0.5:
                items.append(v)
            elif r < 0.65:
                items.append(int(rng.integers(-1000, 1000)))
            elif r < 0.7:
                items.append(bool(rng.integers(0, 2)))
            else:
                items.append(np.float64(v))
        want = sum(items)
        kinds = [0 if type(x) is float else 1 if isinstance(x, int) else 2 for x in items]
        got = investa_core.py_sum([float(x) for x in items], kinds)
        if not items:
            assert got is None and want == 0
        else:
            assert _same(got, want), (items, got, want)
