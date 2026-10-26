"""Phase 15k: the recommendation, the blend and the comprehensive valuation
of financial_ratios in Rust (rust/investa_core/src/fr_blend.rs) against the
original frozen in tests/reference/financial_ratios_p15.py.

The recommendation and the blend take random model maps (values missing,
None, zero, negative, NaN; Monte Carlo bands present, partial or text) under
every sector and industry. The comprehensive valuation runs end to end on
random statements and profiles with random overrides, ETFs and funds,
penny stocks, micro caps, and price histories, from the same NumPy state on
both sides. The per-symbol entry point runs against a fake provider and
config manager (sparse info, failing quotes, failing fetches, exclusions,
overrides). Results, logs, warnings, inputs and the generator state agree.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import financial_ratios as new  # noqa: E402
from reference import financial_ratios_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402
from test_rust_models_parity import _info  # noqa: E402
from test_rust_ratio_glue_parity import _both  # noqa: E402
from test_rust_ratios_parity import _case  # noqa: E402

KEYS = ["dcf", "dcfo", "dni", "mean_pe", "peg", "mean_pb", "mean_ps", "psg", "graham", "ddm", "epv", "lynch"]
SECTORS = ["Technology", "Financial Services", "Real Estate", "", None, "Utilities"]
INDUSTRIES = ["REIT - Retail", "Banks - Regional", "Insurance - Life", "Software", "", None]


def _models(rng):
    out = {}
    for k in KEYS:
        r = rng.random()
        if r < 0.2:
            continue
        if r < 0.3:
            out[k] = {"error": f"{k} refused"}
            continue
        iv = [float(rng.lognormal(3, 1)), None, 0.0, -5.0, float("nan")][int(rng.choice(5, p=[0.75, 0.1, 0.05, 0.05, 0.05]))]
        res = {"intrinsic_value": iv, "model": f"Model {k}"}
        m = rng.random()
        if m < 0.6 and iv:
            base = float(iv) if iv == iv else 10.0
            res["mc"] = {"bear": base * 0.7, "base": base, "bull": base * 1.4}
        elif m < 0.7:
            res["mc"] = {"bear": "x", "base": 1.0, "bull": 2.0}
        elif m < 0.8:
            res["mc"] = {"base": 1.0}
        out[k] = res
    return out


def _profile(rng):
    info = {"sector": SECTORS[int(rng.integers(0, len(SECTORS)))], "industry": INDUSTRIES[int(rng.integers(0, len(INDUSTRIES)))]}
    for k, v in (("trailingEps", 2.0), ("freeCashflow", 1e9), ("operatingCashflow", 1e9), ("revenueGrowth", 0.2),
                 ("dividendYield", 0.03), ("payoutRatio", 0.7)):  # fmt: skip
        if rng.random() < 0.7:
            info[k] = [v, -v, 0, None, "x"][int(rng.choice(5, p=[0.6, 0.15, 0.1, 0.1, 0.05]))]
    return info


@pytest.mark.parametrize("seed", range(6))
def test_recommend_and_blend(caplog, seed):
    rng = np.random.default_rng(21000 + seed)
    for trial in range(120):
        models, info = _models(rng), _profile(rng)
        _both(caplog, "recommend_best_valuation_method", info, models)
        _both(caplog, "recommend_best_valuation_method", info, models, None, None, True)
        _both(caplog, "blend_intrinsic_values", models, info)
        _both(caplog, "_blend_profile", info)
        for k in KEYS:
            _both(caplog, "_model_reliability", models.get(k))
        for profile in ("reit", "financial", "operating"):
            _both(caplog, "_ddm_describes_the_company", info, profile)
    for info in ({"symbol": "AAPL"}, {"shortName": "Apple"}, {"longName": "Apple Inc."}, {}, {"symbol": 5}, {"symbol": ""}):
        _both(caplog, "_stable_mc_seed", info)


def test_deterministic_draws_restore_the_stream():
    for mod in (ref, new):
        np.random.seed(3)
        expected = np.random.random(3).tolist()
        np.random.seed(3)
        with mod._deterministic_draws({"symbol": "MSFT"}):
            inside = np.random.random(2).tolist()
        assert np.random.random(3).tolist() == expected
        assert inside == np.random.RandomState(mod._stable_mc_seed({"symbol": "MSFT"})).random(2).tolist()
        np.random.seed(3)
        with pytest.raises(KeyError):
            with mod._deterministic_draws({"symbol": "MSFT"}):
                raise KeyError("inside")
        assert np.random.random(3).tolist() == expected


OVERRIDES = ["dcf_discount_rate", "dcf_growth_rate", "dcf_terminal_growth", "dcf_projection_years", "dcf_fcf",
             "target_fcf_margin", "graham_growth_rate", "graham_eps", "graham_bond_yield", "ddm_discount_rate",
             "ddm_growth_rate", "ddm_terminal_growth", "ddm_projection_years", "ddm_base_dividend", "dcfo_base_cfo",
             "mean_pe_multiple", "mean_pb_multiple", "mean_ps_multiple", "peg_target", "psg_target", "epv_discount_rate",
             "lynch_growth_rate"]  # fmt: skip
OVERRIDE_VALUES = {"dcf_projection_years": [5, 10, "7"], "ddm_projection_years": [5, 12]}


def _comprehensive_case(rng, trial):
    fin, bs, cf, _ppy, prices = _case(rng)
    info = _info(rng)
    info["quoteType"] = ["EQUITY", "EQUITY", "ETF", "MUTUALFUND", "equity"][trial % 5]
    if trial % 7 == 0:
        info["currentPrice"] = 0.5
    if trial % 11 == 0:
        info["marketCap"] = 1e7
    if rng.random() < 0.3:
        info["navPrice"] = [None, 0, 25.0][int(rng.integers(0, 3))]
    overrides = None
    if rng.random() < 0.4:
        overrides = {}
        for k in rng.choice(OVERRIDES, 4, replace=False):
            overrides[str(k)] = OVERRIDE_VALUES.get(str(k), [0.09, 0.05, 12.0])[int(rng.integers(0, 2))]
    return info, fin, bs, cf, overrides, prices


@pytest.mark.parametrize("seed", range(6))
def test_comprehensive_valuation(caplog, seed):
    rng = np.random.default_rng(21100 + seed)
    for trial in range(25):
        info, fin, bs, cf, overrides, prices = _comprehensive_case(rng, trial)
        outs = []
        for mod in (ref, new):
            np.random.seed(trial)
            outs.append(call(lambda: mod.get_comprehensive_intrinsic_value(info, fin, bs, cf, overrides, 400, prices), caplog))
            outs.append(canon(np.random.random()))  # the caller's stream is put back
        assert outs[:2] == outs[2:], (seed, trial)


class FakeConfigManager:
    def __init__(self, overrides):
        self.manual_overrides = overrides


class FakeMdp:
    def __init__(self, rng, mode, info, statements, prices):
        self.rng, self.mode, self.info, self.statements, self.prices = rng, mode, info, statements, prices
        self.calls = []

    def get_fundamental_data(self, symbol, force_refresh=False):
        self.calls.append(("info", symbol, force_refresh))
        if self.mode == "raise":
            raise ConnectionError("yahoo down")
        if self.mode == "sparse" and not force_refresh:
            return {"symbol": symbol}
        if self.mode == "empty":
            return {}
        return dict(self.info)

    def get_current_quotes(self, symbols, currencies, symbol_map, excluded):
        self.calls.append(("quotes", tuple(symbols), sorted(currencies)))
        if self.mode == "quote_fail":
            raise TimeoutError("quote timeout")
        return {symbols[0]: {"price": 42.0}}, None, None, None, None

    def _statement(self, kind, symbol, period, force_refresh=False):
        self.calls.append((kind, symbol, period, force_refresh))
        return self.statements[kind]

    def get_financials(self, symbol, period, force_refresh=False):
        return self._statement("fin", symbol, period, force_refresh)

    def get_balance_sheet(self, symbol, period, force_refresh=False):
        return self._statement("bs", symbol, period, force_refresh)

    def get_cashflow(self, symbol, period, force_refresh=False):
        return self._statement("cf", symbol, period, force_refresh)

    def get_historical_data(self, symbols, start, end):
        self.calls.append(("history", tuple(symbols)))
        return ({symbols[0]: self.prices} if self.prices is not None else {}), {}


def test_intrinsic_value_for_symbol(caplog):
    rng = np.random.default_rng(21200)
    for trial in range(30):
        info, fin, bs, cf, _ov, prices = _comprehensive_case(rng, trial)
        info.update({"quoteType": "EQUITY", "lastFiscalYearEnd": 1, "mostRecentQuarter": 1})
        if isinstance(prices, pd.Series):
            prices = prices.to_frame("Close")
        mode = ["ok", "sparse", "raise", "empty", "quote_fail", "ok"][trial % 6]
        cm = None
        if trial % 3:
            cm = FakeConfigManager({"user_symbol_map": {}, "user_excluded_symbols": ["BAD"],
                                    "valuation_overrides": {"AAPL": {"dcf_growth_rate": 0.05}}})  # fmt: skip
        symbol = ["AAPL", "BAD", "aapl", "MSFT"][trial % 4]
        pre = (fin, None, cf) if trial % 2 else (None, None, None)
        outs = []
        for mod in (ref, new):
            mdp = FakeMdp(rng, mode, info, {"fin": fin, "bs": bs, "cf": cf}, prices)
            np.random.seed(trial)
            got = call(lambda: mod.get_intrinsic_value_for_symbol(symbol, mdp, cm, 300, trial % 4 == 0, *pre), caplog)
            outs.append((got, mdp.calls))
        assert outs[0] == outs[1], (trial, mode, symbol)
