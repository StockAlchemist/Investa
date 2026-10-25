"""Monte Carlo valuation models (Rust per-sample arithmetic) against the originals.

Both sides draw from NumPy's legacy global generator, seeded identically, so
the results must be bitwise equal — and the generator must end in the same
state, which proves each shim consumes exactly the draws the original did (the
models share one seeded stream per valuation, so a shim that drew one sample
more or less would shift every model after it).
"""

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import financial_ratios as new  # noqa: E402
from reference import monte_carlo as ref  # noqa: E402


def _same(a, b, path="result"):
    assert type(a) is type(b) or (
        isinstance(a, float) and isinstance(b, float)
    ), f"{path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (np.isnan(a) and np.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _run_both(name, seed, *args, **kwargs):
    np.random.seed(seed)
    want = getattr(ref, name)(*args, **kwargs)
    want_state = np.random.get_state()
    np.random.seed(seed)
    got = getattr(new, name)(*args, **kwargs)
    got_state = np.random.get_state()
    _same(got, want, name)
    assert got_state[1].tobytes() == want_state[1].tobytes(), f"{name}: RNG stream"
    assert got_state[2:] == want_state[2:], f"{name}: RNG position/gauss cache"
    return want


def _ticker(rng):
    r = rng.random()
    shares = None if r < 0.05 else (0 if r < 0.08 else int(rng.integers(1_000_000, 20_000_000_000)))
    info = {"sharesOutstanding": shares}
    if rng.random() < 0.8:
        info["totalCash"] = int(rng.integers(0, 10**11)) if rng.random() < 0.5 else float(rng.uniform(0, 1e11))
    if rng.random() < 0.8:
        info["totalDebt"] = int(rng.integers(0, 10**11))
    return info


def _num(rng, scale=1.0, nan_rate=0.03):
    if rng.random() < nan_rate:
        return float("nan")
    return float(rng.normal(0, scale))


@pytest.mark.parametrize("seed", range(3))
def test_discounted_models_match(seed):
    rng = np.random.default_rng(1100 + seed)
    for case in range(120):
        years = int(rng.choice([-1, 0, 1, 2, 5, 10]))
        tg = float(rng.choice([0.0, 0.02, 0.03, -0.01]))
        it = int(rng.choice([10, 60, 500, 10000]))
        g, d = _num(rng, 0.2), abs(_num(rng, 0.1)) + float(rng.choice([0.0, 0.05]))
        s = int(rng.integers(0, 2**31))
        info = _ticker(rng)
        _run_both("run_monte_carlo_dcf", s, info, _num(rng, 1e9), g, d, years, tg, it)
        _run_both("run_monte_carlo_dcfo", s, info, _num(rng, 1e9), g, d, years, tg, it)
        _run_both("run_monte_carlo_dni", s, _num(rng, 10), g, d, years, tg, it)
        _run_both("run_monte_carlo_ddm", s, abs(_num(rng, 3)), g, d, years, tg, it)


@pytest.mark.parametrize("seed", range(3))
def test_multiple_models_match(seed):
    rng = np.random.default_rng(1200 + seed)
    for case in range(150):
        it = int(rng.choice([10, 60, 500, 10000]))
        s = int(rng.integers(0, 2**31))
        eps = _num(rng, 8)
        _run_both("run_monte_carlo_mean_pe", s, eps, abs(_num(rng, 25)), it)
        _run_both("run_monte_carlo_peg", s, eps, _num(rng, 20), float(rng.choice([0.0, 1.5])), float(rng.choice([1.0, 1.3])), it)
        _run_both("run_monte_carlo_mean_pb", s, _num(rng, 40), abs(_num(rng, 3)), it)
        _run_both("run_monte_carlo_mean_ps", s, _num(rng, 60), abs(_num(rng, 4)), it)
        _run_both("run_monte_carlo_psg", s, abs(_num(rng, 60)), abs(_num(rng, 25)), float(rng.uniform(0, 1)), float(rng.choice([1.0, 0.8])), it)
        _run_both("run_monte_carlo_graham", s, eps, _num(rng, 12), abs(_num(rng, 5)), it)
        _run_both("run_monte_carlo_epv", s, _ticker(rng), _num(rng, 1e9), float(rng.uniform(0, 0.4)), abs(_num(rng, 0.1)), it)
        _run_both("run_monte_carlo_lynch", s, eps, _num(rng, 20), float(rng.choice([0.0, 2.0])), it)


def test_models_share_one_stream_like_a_valuation():
    """A full valuation calls the models back to back on one seeded stream."""
    rng = np.random.default_rng(1300)
    for _ in range(20):
        info = _ticker(rng)
        info["sharesOutstanding"] = int(rng.integers(10**7, 10**10))
        seed = int(rng.integers(0, 2**31))
        results = {}
        for side, mod in (("ref", ref), ("new", new)):
            np.random.seed(seed)
            results[side] = [
                mod.run_monte_carlo_dcf(info, 5e8, 0.08, 0.1),
                mod.run_monte_carlo_ddm(1.2, 0.03, 0.09),
                mod.run_monte_carlo_mean_pe(4.0, 18.0),
                mod.run_monte_carlo_epv(info, 7e8, 0.21, 0.1),
                mod.run_monte_carlo_graham(4.0, 7.0, 4.5),
            ]
        _same(results["new"], results["ref"], "valuation")


def test_mc_stats_matches_on_awkward_samples():
    rng = np.random.default_rng(1400)
    samples = [
        np.full(100, 42.0),  # one value: histogram range widened by 0.5
        np.concatenate([np.full(60, 1.0), [np.nan] * 5, [-3.0] * 5, [np.inf] * 2]),
        rng.lognormal(3, 1.5, 49),  # under 50 valid
        rng.lognormal(3, 1.5, 50),
        np.array(rng.choice([1.0, 2.0, 3.0], 5000)),  # values on bin edges
        rng.normal(100, 1e-9, 1000),  # a very narrow range
    ]
    for v in samples:
        _same(new._generate_mc_stats(v), ref._generate_mc_stats(v), "mc_stats")
