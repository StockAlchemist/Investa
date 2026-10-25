"""Phase 3b: the deterministic valuation arithmetic (Rust) against the original.

tests/reference/financial_ratios.py is the module before phase 3b. Every
ported function — WACC, the growth blend, the FCF normalizers, the four
projection models, EPV, the reliability and the model blend — must return
the same thing, bitwise and type for type, on random statements and profiles.
NumPy-scalar arguments are mixed in on purpose: they change how the original's
sum() added its terms and what type it returned.
"""

import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import financial_ratios as new  # noqa: E402
from reference import financial_ratios as ref  # noqa: E402

ITEMS = [
    "Total Revenue", "Net Income", "Operating Income", "Interest Expense",
    "Tax Provision", "Pretax Income", "Diluted EPS", "Free Cash Flow",
    "Operating Cash Flow", "Capital Expenditure", "Cash Dividends Paid",
    "Total Debt", "Total Cash", "Ordinary Shares Number",
]  # fmt: skip


def _same(a, b, path="result"):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, (float, np.floating)):
        assert (np.isnan(a) and np.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _statements(rng):
    """Random annual statements, newest column first like yfinance's."""
    n = int(rng.integers(0, 15))
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cols = sorted(
        pd.Timestamp(f"{2025 - i}-12-31") for i in range(n)
    )[::-1]
    scale = float(rng.choice([1e6, 1e8, 1e10]))
    growth = float(rng.normal(0.06, 0.1))
    base = np.array([scale * (1 + growth) ** (n - i) for i in range(n)])

    def row(mult, noise=0.2, sign=1.0, nan_rate=0.1):
        v = sign * base * mult * (1 + rng.normal(0, noise, n))
        v[rng.random(n) < nan_rate] = np.nan
        return v

    fin = pd.DataFrame(
        {
            "Total Revenue": row(1.0),
            "Net Income": row(0.1, 0.6),
            "Operating Income": row(0.15, 0.5),
            "Interest Expense": row(0.01, 0.5, nan_rate=0.3),
            "Tax Provision": row(0.03, 0.5),
            "Pretax Income": row(0.13, 0.5),
            "Diluted EPS": row(1e-8, 0.4),
        },
        index=cols,
    ).T
    cf_rows = {
        "Operating Cash Flow": row(0.14, 0.4),
        "Capital Expenditure": row(0.05, 0.4, sign=-1.0),
        "Cash Dividends Paid": row(0.03, 0.2, sign=-1.0, nan_rate=0.4),
    }
    if rng.random() < 0.5:
        cf_rows["Free Cash Flow"] = row(0.09, 0.8)
    cf = pd.DataFrame(cf_rows, index=cols).T
    bs = pd.DataFrame(
        {
            "Total Debt": row(0.3, 0.3, nan_rate=0.2),
            "Total Cash": row(0.2, 0.3),
            "Ordinary Shares Number": row(1e-3, 0.01),
        },
        index=cols,
    ).T
    return fin, bs, cf


def _info(rng):
    def maybe(v, p=0.8):
        return v if rng.random() < p else None

    price = float(rng.uniform(5, 500))
    info = {
        "symbol": "TEST",
        "beta": maybe(float(rng.uniform(-0.5, 3.5)) if rng.random() < 0.95 else float("nan")),
        "marketCap": maybe(int(rng.integers(10**7, 10**12))),
        "totalDebt": maybe(int(rng.integers(0, 10**11)) if rng.random() < 0.8 else 0),
        "totalCash": maybe(float(rng.uniform(0, 1e11))),
        "sharesOutstanding": maybe(int(rng.integers(10**6, 10**10))),
        "totalRevenue": maybe(float(rng.uniform(1e7, 1e11))),
        "freeCashflow": maybe(float(rng.normal(1e9, 2e9))),
        "operatingCashflow": maybe(float(rng.normal(2e9, 2e9))),
        "trailingEps": maybe(float(rng.normal(3, 4))),
        "dividendRate": maybe(float(rng.uniform(0, 5)), 0.5),
        "payoutRatio": maybe(float(rng.uniform(0, 2)), 0.6),
        "currentPrice": price,
        "sector": str(rng.choice(["Technology", "Financial Services", "Real Estate", "Utilities"])),
        "industry": str(rng.choice(["Software", "Banks - Regional", "REIT - Retail", "Utilities"])),
    }
    if rng.random() < 0.5:
        info["_earnings_estimate"] = {
            "0y": {"growth": float(rng.normal(0.1, 0.1))},
            "+1y": {"growth": float(rng.normal(0.1, 0.1)) if rng.random() < 0.8 else None},
        }
    return info


def _maybe_numpy(rng, v):
    """Sometimes hand the function a NumPy scalar, as some callers do."""
    if v is None or rng.random() < 0.7:
        return v
    return np.float64(v)


@pytest.mark.parametrize("seed", range(6))
def test_cost_of_capital_growth_and_normalizers_match(seed):
    rng = np.random.default_rng(2100 + seed)
    for _ in range(80):
        fin, bs, cf = _statements(rng)
        info = _info(rng)
        _same(new.calculate_wacc(info, fin, bs), ref.calculate_wacc(info, fin, bs), "wacc")
        for item in ("Net Income", "Operating Cash Flow"):
            source = cf if item == "Operating Cash Flow" else fin
            _same(
                new.blended_growth_estimate(source, ticker_info=info, item_name=item),
                ref.blended_growth_estimate(source, ticker_info=info, item_name=item),
                "growth",
            )
        _same(new.estimate_fcf_margin(fin, cf), ref.estimate_fcf_margin(fin, cf), "margin")
        _same(new.through_cycle_fcf_margin(fin, cf), ref.through_cycle_fcf_margin(fin, cf), "cycle")
        _same(new.normalized_base_fcf(info, fin, cf), ref.normalized_base_fcf(info, fin, cf), "base_fcf")


@pytest.mark.parametrize("seed", range(6))
def test_projection_models_and_epv_match(seed):
    rng = np.random.default_rng(2200 + seed)
    for _ in range(60):
        fin, bs, cf = _statements(rng)
        info = _info(rng)
        years = int(rng.choice([0, 1, 2, 5, 10]))
        tg = _maybe_numpy(rng, float(rng.choice([0.0, 0.02, 0.03])))
        disc = _maybe_numpy(rng, None if rng.random() < 0.5 else float(rng.uniform(0.05, 0.25)))
        growth = _maybe_numpy(rng, None if rng.random() < 0.5 else float(rng.normal(0.08, 0.1)))
        kw = dict(discount_rate=disc, growth_rate=growth, projection_years=years, terminal_growth_rate=tg)
        base = _maybe_numpy(rng, None if rng.random() < 0.6 else float(rng.uniform(0.1, 1e9)))
        _same(
            new.calculate_intrinsic_value_dcf(info, fin, bs, cf, fcf=base, **kw),
            ref.calculate_intrinsic_value_dcf(info, fin, bs, cf, fcf=base, **kw),
            "dcf",
        )
        div = _maybe_numpy(rng, None if rng.random() < 0.6 else float(rng.uniform(0.1, 4)))
        _same(
            new.calculate_intrinsic_value_ddm(info, fin, cf, bs, base_dividend=div, **kw),
            ref.calculate_intrinsic_value_ddm(info, fin, cf, bs, base_dividend=div, **kw),
            "ddm",
        )
        _same(
            new.calculate_intrinsic_value_dcfo(info, fin, bs, cf, base_cfo=base, **kw),
            ref.calculate_intrinsic_value_dcfo(info, fin, bs, cf, base_cfo=base, **kw),
            "dcfo",
        )
        eps = _maybe_numpy(rng, None if rng.random() < 0.5 else float(rng.normal(3, 3)))
        _same(
            new.calculate_intrinsic_value_dni(info, fin, bs, base_eps=eps, **kw),
            ref.calculate_intrinsic_value_dni(info, fin, bs, base_eps=eps, **kw),
            "dni",
        )
        _same(
            new.calculate_earnings_power_value(info, fin, bs, cf, discount_rate=disc),
            ref.calculate_earnings_power_value(info, fin, bs, cf, discount_rate=disc),
            "epv",
        )


def test_reliability_and_blend_match():
    rng = np.random.default_rng(2300)
    keys = sorted(
        set(ref.MODEL_BLEND_PRIORS_OPERATING)
        | set(ref.MODEL_BLEND_PRIORS_FINANCIAL)
        | set(ref.MODEL_BLEND_PRIORS_REIT)
    )
    for _ in range(3000):
        models = {}
        for k in keys:
            if rng.random() < 0.3:
                continue
            iv = float(rng.choice([rng.uniform(-10, 400), np.nan, np.inf]) if rng.random() < 0.1 else rng.uniform(1, 400))
            res = {"intrinsic_value": iv}
            if rng.random() < 0.7:
                b = float(rng.uniform(0.5, 1.5) * iv)
                res["mc"] = {"bear": b * float(rng.uniform(0.3, 1.1)), "base": b, "bull": b * float(rng.uniform(0.9, 3))}
                if rng.random() < 0.05:
                    res["mc"]["base"] = None
            models[k] = res
        info = _info(rng)
        for res in models.values():
            _same(new._model_reliability(res), ref._model_reliability(res), "reliability")
        _same(new.blend_intrinsic_values(models, info), ref.blend_intrinsic_values(models, info), "blend")
