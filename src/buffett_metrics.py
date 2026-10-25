# -*- coding: utf-8 -*-
"""
Per-company Buffett quality metrics, derived from EDGAR fundamentals.

Every metric here obeys two rules from the ranking's design principles:

**Durability over level (P4).** Almost nothing is measured on the latest year
alone. A 30% return on equity in one year is noise; 15% in each of twelve years
is evidence of a moat. So the profitability, predictability and growth metrics
are medians, dispersions and long-window CAGRs, not point values. Only genuine
balance-sheet *state* — what the company owes today — is read from the latest
period, because that is what it actually is.

**Missing data must never look good (P7).** Every metric returns `None` rather
than a default when its inputs are absent, and `coverage` reports what fraction
resolved. A company whose filings do not support a metric must be demoted for
uncertainty, never credited with a neutral score. This matters most at the top
of the ranking, where a thinly-reported micro-cap would otherwise outrank a
well-documented compounder.

Three model variants exist because their economics differ, not their quality
standards: `generic` (FCF businesses), `bank` (deposit-takers and insurers,
where leverage is an input rather than a risk) and `reit` (where GAAP
depreciation obscures cash earnings). See `edgar_sic` for routing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import edgar_provider
import investa_core
from edgar_concepts import BANK_CONCEPTS, REIT_CONCEPTS

# A company needs at least this many annual periods before consistency
# measures mean anything. Below it, a single good year would dominate.
MIN_PERIODS_FOR_CONSISTENCY = 5

# The durability window. EDGAR reaches ~17 years; ten is the span over which a
# competitive advantage either holds or does not.
DURABILITY_WINDOW = 10


@dataclass
class CompanyMetrics:
    """Everything the ranker needs about one company, plus its data provenance."""

    cik: str
    symbol: str
    name: str
    model: str  # generic | bank | reit

    period_count: int = 0
    latest_period: Optional[str] = None
    # The oldest filed year, so a reader can see the span the metrics rest on
    # rather than inferring it from a count.
    first_period: Optional[str] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    gate_failures: List[str] = field(default_factory=list)
    coverage: float = 0.0
    notes: List[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[float]:
        return self.metrics.get(name)

    @property
    def passes_gates(self) -> bool:
        return not self.gate_failures


# --- small numeric helpers --------------------------------------------------
# All of these return None rather than raising or substituting a default, so a
# missing input propagates as "unknown" instead of silently becoming a score.


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_divide(numerator, denominator) -> Optional[float]:
    num = _finite(numerator)
    den = _finite(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _cagr(series_oldest_first: Sequence[Optional[float]]) -> Optional[float]:
    """
    Compound annual growth rate as a fraction (0.12 = 12%/yr).

    Returns None when the starting value is not positive: a company that grew
    out of losses has no meaningful growth *rate*, and forcing one would reward
    the most erratic businesses in the universe. Computed in Rust, which the
    metric functions use directly.
    """
    return investa_core.buffett_cagr([_raw_number(v) for v in series_oldest_first])


def _total_debt(concepts: Dict[str, Dict[str, float]], period: str) -> Optional[float]:
    """
    Interest-bearing debt at one period end, or None when it is not reported.

    The `or 0.0` shortcut is deliberately avoided here. Treating an untagged
    debt line as zero hands the company a perfect leverage score — the single
    most dangerous data error in this system, because "no debt" is the best
    result the financial-strength pillar can award. Absent must stay absent so
    P7 can demote it, and only a genuinely reported zero counts as debt-free.
    Computed in Rust, which the metric functions use directly.
    """
    return investa_core.buffett_total_debt(_for_rust(concepts), str(period))


# The ratio labels the metrics read. Every use passes them through `_finite`,
# so they are converted that way once, here.
_RATIO_LABELS = (
    "Return on Equity (ROE) (%)",
    "Return on Assets (ROA) (%)",
    "Gross Profit Margin (%)",
    "Net Profit Margin (%)",
    "Debt-to-Equity Ratio",
    "Current Ratio",
)


def _raw_number(value) -> Optional[float]:
    """A concept value as Rust takes it: None stays None, numbers become floats
    with NaN and infinity kept (several metrics read raw values), anything
    else raises as the arithmetic on it would have."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)) or hasattr(value, "__float__"):
        return float(value)
    raise TypeError(f"concept value is not a number: {value!r}")


def _for_rust(
    concepts: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        name: {str(period): _raw_number(v) for period, v in (series or {}).items()}
        for name, series in (concepts or {}).items()
    }


def _ratios_for_rust(
    ratios_by_period: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        str(period): {
            label: _finite(row.get(label)) for label in _RATIO_LABELS if label in row
        }
        for period, row in (ratios_by_period or {}).items()
    }


def _from_rust(rows) -> Dict[str, Optional[float]]:
    """Rust's (key, value, is_int) rows back into Python's metrics dict."""
    return {key: (int(value) if is_int else value) for key, value, is_int in rows}


# --- generic model ----------------------------------------------------------


def _compute_generic_metrics(
    concepts: Dict[str, Dict[str, float]], ratios_by_period: Dict[str, Dict[str, float]]
) -> Dict[str, Optional[float]]:
    """FCF businesses: returns on capital, balance-sheet state, predictability,
    growth and capital allocation over the durability window.

    Computed in Rust (investa_core.buffett_generic_metrics); the metric
    definitions and their rationale are in rust/investa_core/src/buffett.rs.
    Original frozen in tests/reference/buffett_metrics.py.
    """
    return _from_rust(
        investa_core.buffett_generic_metrics(
            _for_rust(concepts), _ratios_for_rust(ratios_by_period)
        )
    )


# --- bank model -------------------------------------------------------------


def _compute_bank_metrics(
    concepts: Dict[str, Dict[str, float]],
    sector: Dict[str, Dict[str, float]],
    ratios_by_period: Dict[str, Dict[str, float]],
) -> Dict[str, Optional[float]]:
    """
    Bank and insurer quality.

    Deliberately omits debt-to-equity, interest coverage and the current ratio:
    for a deposit-taker those describe the business model, not its risk. A bank
    with 10x leverage is normal; an industrial with 10x leverage is nearly dead.
    Solvency is measured instead by the equity-to-assets ratio, and credit
    discipline by provisions relative to the loan book.

    Computed in Rust (investa_core.buffett_bank_metrics).
    """
    return _from_rust(
        investa_core.buffett_bank_metrics(
            _for_rust(concepts), _for_rust(sector), _ratios_for_rust(ratios_by_period)
        )
    )


# --- REIT model -------------------------------------------------------------


def _compute_reit_metrics(
    concepts: Dict[str, Dict[str, float]],
    sector: Dict[str, Dict[str, float]],
    ratios_by_period: Dict[str, Dict[str, float]],
) -> Dict[str, Optional[float]]:
    """
    REIT quality, measured on funds from operations rather than net income.

    A REIT depreciates buildings that in practice hold or gain value, so GAAP
    earnings systematically understate what the business generates.
    FFO = net income + real-estate depreciation − gains on property sales, the
    NAREIT definition. Share-count growth carries unusual weight here: REITs
    fund themselves by issuing equity, so per-share discipline separates the
    good ones from the merely large.

    Computed in Rust (investa_core.buffett_reit_metrics).
    """
    return _from_rust(
        investa_core.buffett_reit_metrics(
            _for_rust(concepts), _for_rust(sector), _ratios_for_rust(ratios_by_period)
        )
    )


# --- ratio bridge -----------------------------------------------------------


def _ratios_by_period(cik: str) -> Dict[str, Dict[str, float]]:
    """
    Run the existing ratio engine and re-key its output by period string.

    Reuses `financial_ratios.calculate_key_ratios_timeseries` rather than
    recomputing ROE/leverage here, so the ranking and the rest of the app can
    never disagree about what a company's ROE was.
    """
    import financial_ratios

    statements = edgar_provider.get_statements(cik)
    frame = financial_ratios.calculate_key_ratios_timeseries(
        statements.get("financials"), statements.get("balance_sheet")
    )
    if frame is None or frame.empty:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for period, row in frame.iterrows():
        try:
            key = period.strftime("%Y-%m-%d")
        except AttributeError:
            key = str(period)
        result[key] = {k: v for k, v in row.to_dict().items() if k != "Period"}
    return result


# --- entry point ------------------------------------------------------------

# Which metrics count toward the coverage score for each model. Coverage is the
# fraction of these that resolved, and it can only ever demote a company (P7).
_COVERAGE_KEYS = {
    "generic": [
        "roe_median",
        "roic_median",
        "gross_margin_median",
        "debt_to_equity",
        "fcf_margin_median",
        "revenue_cagr",
        "owner_earnings_cagr",
        "book_value_per_share_cagr",
        "share_count_cagr",
    ],
    "bank": [
        "roe_median",
        "roa_median",
        "equity_to_assets_median",
        "provision_rate_median",
        "net_interest_margin_median",
        "efficiency_ratio_median",
        "book_value_per_share_cagr",
        "share_count_cagr",
    ],
    # Insurers run the same residual-income model but are not judged on lending
    # metrics they have no reason to report. Including net interest margin or
    # loan-loss provisioning here would mark every insurer down for a data gap
    # that reflects its business model, not its quality.
    "insurer": [
        "roe_median",
        "roa_median",
        "equity_to_assets_median",
        "book_value_per_share_cagr",
        "revenue_cagr",
        "share_count_cagr",
    ],
    "reit": [
        "ffo_margin_median",
        "ffo_on_equity_median",
        "ffo_cagr",
        "debt_to_ffo",
        "ffo_per_share_cagr",
        "share_count_cagr",
        "real_estate_cagr",
    ],
}


def compute_metrics(cik: str, symbol: str, name: str, model: str) -> CompanyMetrics:
    """See `_compute_metrics`; its EDGAR reads share one per-company session."""
    with edgar_provider.company_session(cik):
        return _compute_metrics(cik, symbol, name, model)


def _compute_metrics(cik: str, symbol: str, name: str, model: str) -> CompanyMetrics:
    """
    Derive one company's full metric set. Never raises — a company that cannot
    be measured comes back with `gate_failures` explaining why.
    """
    result = CompanyMetrics(cik=cik, symbol=symbol, name=name, model=model)

    try:
        concepts = edgar_provider.get_concept_values(cik)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"Metrics: concept load failed for {symbol}: {exc}")
        result.gate_failures.append("fundamentals_unavailable")
        return result

    if not concepts or "net_income" not in concepts:
        result.gate_failures.append("no_fundamentals")
        return result

    periods = sorted(set(concepts.get("net_income", {})))
    result.period_count = len(periods)
    result.latest_period = periods[-1] if periods else None
    result.first_period = periods[0] if periods else None

    # The share count is the one series here a stock split can corrupt. A 10-K
    # restates the two prior years for the split and nothing restates the years
    # before it, so the assembled series steps by the split ratio partway
    # through the window: Apple's reads as +11.8%/yr of issuance across a decade
    # in which it retired a quarter of its shares, and NVIDIA's as +49.7%.
    # Every per-share rate below divides by this — share-count CAGR, book value
    # per share, FFO per share — so it is rebuilt once, here, from same-filing
    # ratios that no split can distort. Companies that never split come back
    # unchanged.
    if concepts.get("shares_diluted"):
        try:
            concepts["shares_diluted"] = edgar_provider.split_consistent_series(
                cik, "shares_diluted"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                f"Metrics: share-count reconstruction failed for {symbol}: {exc}"
            )

    try:
        ratios = _ratios_by_period(cik)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"Metrics: ratio computation failed for {symbol}: {exc}")
        ratios = {}

    if model in ("bank", "insurer"):
        # Insurers share the residual-income metric set; the lending-specific
        # entries simply resolve to None and are excluded from their coverage.
        sector = edgar_provider.get_concept_values(cik, list(BANK_CONCEPTS))
        result.metrics = _compute_bank_metrics(concepts, sector, ratios)
    elif model == "reit":
        sector = edgar_provider.get_concept_values(cik, list(REIT_CONCEPTS))
        result.metrics = _compute_reit_metrics(concepts, sector, ratios)
    else:
        result.metrics = _compute_generic_metrics(concepts, ratios)

    coverage_keys = _COVERAGE_KEYS.get(model, _COVERAGE_KEYS["generic"])
    resolved = sum(1 for key in coverage_keys if result.metrics.get(key) is not None)
    result.coverage = resolved / len(coverage_keys) if coverage_keys else 0.0

    return result
