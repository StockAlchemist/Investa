"""Yahoo's dividendYield arrives in two units, with no flag saying which.

    VICI  -> dividendYield: 6.71     (percent:  6.71%)
    SCHD  -> dividendYield: 0.033    (fraction: 3.3%)

Across the cached fundamentals corpus 93.6% are percent-encoded and 6.0% are
fraction-encoded, and the ranges overlap on [0.01, 3.08] — so magnitude alone
cannot settle it. `robust_dividend_yield` resolves it by corroborating against
the indicated rate / price (step 1) or trailingAnnualDividendYield (step 2),
and only guesses from magnitude when neither exists.

These tests pin the resolution order, the guess of last resort, and the
fraction contract every client depends on.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from finutils import _RAW_YIELD_FRACTION_CUTOFF, robust_dividend_yield


# --- Corroborated paths --------------------------------------------------


def test_rate_over_price_wins_over_a_percent_encoded_raw_value():
    """VICI: raw 6.71 is a percent, and rate/price says so."""
    out = robust_dividend_yield({
        "dividendRate": 1.8,
        "currentPrice": 26.83,
        "dividendYield": 6.71,
    })
    assert out == pytest.approx(0.0671, abs=1e-4)


def test_rate_over_price_wins_over_a_fraction_encoded_raw_value():
    """SCHD: raw 0.033 is a fraction. Same answer, opposite encoding."""
    out = robust_dividend_yield({
        "dividendRate": 0.88,
        "currentPrice": 26.67,
        "dividendYield": 0.033,
    })
    assert out == pytest.approx(0.033, abs=1e-3)


def test_monthly_distributor_still_resolves(monkeypatch):
    """IDBOX, a monthly-paying bond fund: `dividendRate` is the *monthly*
    distribution, so rate/price is 12x low. The encoding still resolves,
    because the two readings sit 100x apart and 12x cannot bridge that."""
    out = robust_dividend_yield({
        "yield": 0.030299999,
        "dividendYield": 3.03,
        "dividendRate": 0.0290781,
        "regularMarketPrice": 10.41,
    })
    assert out == pytest.approx(0.0303, abs=1e-4)


def test_trailing_yield_used_when_price_is_missing():
    out = robust_dividend_yield({
        "trailingAnnualDividendYield": 0.0665,
        "dividendYield": 6.71,
    })
    assert out == pytest.approx(0.0665)


def test_result_is_always_a_fraction():
    """The contract every client relies on: never a percent number."""
    for info in (
        {"dividendRate": 1.8, "currentPrice": 26.83, "dividendYield": 6.71},
        {"trailingAnnualDividendYield": 0.0665},
        {"dividendYield": 6.71},
        {"dividendYield": 0.033},
    ):
        out = robust_dividend_yield(info)
        assert out is not None and 0 < out < 1.0, info


# --- Magnitude fallback (no corroborating signal) ------------------------


@pytest.mark.parametrize("raw,expected", [
    (0.033, 0.033),      # fraction -> 3.3%
    (0.0007, 0.0007),    # fraction -> 0.07%
    (0.05, 0.05),        # fraction -> 5%
])
def test_small_raw_values_are_read_as_fractions(raw, expected):
    assert robust_dividend_yield({"dividendYield": raw}) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    (6.71, 0.0671),      # percent -> 6.71%
    (0.47, 0.0047),      # percent -> 0.47%
    (68.18, 0.6818),     # percent -> 68.18% (real corpus maximum)
])
def test_larger_raw_values_are_read_as_percentages(raw, expected):
    assert robust_dividend_yield({"dividendYield": raw}) == pytest.approx(expected)


def test_cutoff_is_the_documented_value():
    """The three clients hard-code the same number; drifting apart would make
    the same security render differently per platform."""
    assert _RAW_YIELD_FRACTION_CUTOFF == 0.10


def test_previous_cutoff_regression():
    """0.30 used to send these the wrong way. Both must now resolve as percent."""
    assert robust_dividend_yield({"dividendYield": 0.15}) == pytest.approx(0.0015)
    assert robust_dividend_yield({"dividendYield": 0.25}) == pytest.approx(0.0025)


# --- Degenerate input ----------------------------------------------------


@pytest.mark.parametrize("info", [
    {},
    None,
    {"dividendYield": 0},
    {"dividendYield": None},
    {"dividendYield": "n/a"},
    {"dividendRate": 0, "currentPrice": 0},
])
def test_unusable_input_returns_none(info):
    assert robust_dividend_yield(info) is None


def test_zero_price_does_not_divide_by_zero():
    out = robust_dividend_yield({"dividendRate": 1.8, "currentPrice": 0, "dividendYield": 6.71})
    assert out == pytest.approx(0.0671, abs=1e-4)


# --- Whole-corpus consistency -------------------------------------------


CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "fundamentals_cache"
)


def _reference_fraction(info):
    """An independent yield estimate, as a fraction, or None."""
    try:
        rate = float(info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0)
        price = float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 0
        )
        if rate > 0 and price > 0:
            return rate / price
    except (TypeError, ValueError):
        pass
    try:
        trailing = float(info.get("trailingAnnualDividendYield") or 0)
        return trailing if trailing > 0 else None
    except (TypeError, ValueError):
        return None


@pytest.mark.skipif(not os.path.isdir(CACHE_DIR), reason="fundamentals cache not present")
def test_corpus_output_never_disagrees_with_its_reference_by_100x():
    """Run every cached record through the function and check the output sits
    in the same order of magnitude as an independent estimate.

    A magnitude bound cannot do this job: legitimate yields on collapsed penny
    stocks genuinely exceed 100% (ICON pays a $1.40 indicated rate on a $0.94
    share; TSEOF, down 99.7%, computes to 307%), so "output < 1.0" flags real
    data. A *unit* error is what we care about, and it always shows up as a
    ~100x disagreement with the reference.

    The tolerance is wide because the reference itself is approximate: for
    monthly distributors `dividendRate` is the monthly payment, not the annual
    one, so rate/price runs up to 12x low (IDBOX, a monthly bond fund, is the
    corpus example). 30x separates that from a genuine 100x unit error.
    """
    import glob
    import json

    checked = 0
    off_by_a_factor_of_100 = []
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        if "_" in os.path.basename(path):
            continue  # financials/balance-sheet blobs, not info dicts
        try:
            with open(path) as f:
                info = json.load(f).get("data")
        except Exception:
            continue
        if not isinstance(info, dict) or not info.get("dividendYield"):
            continue
        out = robust_dividend_yield(info)
        reference = _reference_fraction(info)
        if out is None or reference is None:
            continue
        checked += 1
        ratio = out / reference
        if not (1 / 30.0 < ratio < 30.0):
            off_by_a_factor_of_100.append(
                (os.path.basename(path), info.get("dividendYield"), out, reference, round(ratio, 2))
            )

    assert checked > 100, f"corpus too small to be meaningful ({checked})"
    assert not off_by_a_factor_of_100, (
        f"{len(off_by_a_factor_of_100)} records disagree with their reference by ~100x "
        f"(a unit error): {off_by_a_factor_of_100[:10]}"
    )
