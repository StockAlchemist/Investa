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
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import edgar_provider
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


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in (_finite(x) for x in values) if v is not None]
    return statistics.median(clean) if clean else None


def _stdev(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in (_finite(x) for x in values) if v is not None]
    return statistics.pstdev(clean) if len(clean) >= 2 else None


def _cagr(series_oldest_first: Sequence[Optional[float]]) -> Optional[float]:
    """
    Compound annual growth rate as a fraction (0.12 = 12%/yr).

    Returns None when the starting value is not positive: a company that grew
    out of losses has no meaningful growth *rate*, and forcing one would reward
    the most erratic businesses in the universe.
    """
    clean = [v for v in (_finite(x) for x in series_oldest_first) if v is not None]
    if len(clean) < 2:
        return None
    start, end = clean[0], clean[-1]
    years = len(clean) - 1
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0


def _total_debt(
    concepts: Dict[str, Dict[str, float]], period: str
) -> Optional[float]:
    """
    Interest-bearing debt at one period end, or None when it is not reported.

    The `or 0.0` shortcut is deliberately avoided here. Treating an untagged
    debt line as zero hands the company a perfect leverage score — the single
    most dangerous data error in this system, because "no debt" is the best
    result the financial-strength pillar can award. Absent must stay absent so
    P7 can demote it, and only a genuinely reported zero counts as debt-free.
    """
    short_term = _finite(concepts.get("short_term_debt", {}).get(period))
    long_term = _finite(concepts.get("long_term_debt", {}).get(period))
    if short_term is None and long_term is None:
        return None
    return (short_term or 0.0) + (long_term or 0.0)


def _aligned(
    values_by_period: Dict[str, float], periods: List[str]
) -> List[Optional[float]]:
    """Pull one concept onto a shared period axis, preserving gaps as None."""
    return [values_by_period.get(period) for period in periods]


def _common_periods(concepts: Dict[str, Dict[str, float]], required: Sequence[str]) -> List[str]:
    """
    Annual period ends where every `required` concept has a value, oldest first.

    Intersecting rather than unioning keeps ratios internally consistent — a
    margin must never divide one year's profit by another year's revenue.
    """
    period_sets = [set(concepts.get(name, {})) for name in required]
    if not period_sets or any(not s for s in period_sets):
        return []
    common = set.intersection(*period_sets)
    return sorted(common)


# --- generic model ----------------------------------------------------------


def _owner_earnings(concepts: Dict[str, Dict[str, float]], periods: List[str]) -> List[Optional[float]]:
    """
    Cash an owner could withdraw: operating cash flow less capital expenditure.

    Buffett's "owner earnings" subtracts *maintenance* capex, which no filer
    discloses separately. Using total capex is the conservative substitute — it
    understates earnings for companies investing in growth, which is the right
    direction to err in a quality screen.
    """
    ocf = concepts.get("operating_cash_flow", {})
    capex = concepts.get("capex", {})
    result: List[Optional[float]] = []
    for period in periods:
        operating = _finite(ocf.get(period))
        spend = _finite(capex.get(period))
        if operating is None:
            result.append(None)
        else:
            # EDGAR reports capex as a positive outflow.
            result.append(operating - (spend or 0.0))
    return result


def _compute_generic_metrics(
    concepts: Dict[str, Dict[str, float]], ratios_by_period: Dict[str, Dict[str, float]]
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}

    all_periods = sorted(set(concepts.get("net_income", {})) | set(concepts.get("revenue", {})))
    window = all_periods[-DURABILITY_WINDOW:]

    # --- Returns on capital -------------------------------------------------
    roe_series = [ratios_by_period.get(p, {}).get("Return on Equity (ROE) (%)") for p in window]
    roa_series = [ratios_by_period.get(p, {}).get("Return on Assets (ROA) (%)") for p in window]
    gross_series = [ratios_by_period.get(p, {}).get("Gross Profit Margin (%)") for p in window]
    net_margin_series = [ratios_by_period.get(p, {}).get("Net Profit Margin (%)") for p in window]

    metrics["roe_median"] = _median(roe_series)
    metrics["roa_median"] = _median(roa_series)
    metrics["gross_margin_median"] = _median(gross_series)
    metrics["net_margin_median"] = _median(net_margin_series)

    clean_roe = [v for v in (_finite(x) for x in roe_series) if v is not None]
    above_15 = sum(1 for v in clean_roe if v >= 15.0)
    metrics["roe_years_above_15"] = above_15 / len(clean_roe) if clean_roe else None
    # The count and its denominator alongside the share. Scoring uses the share;
    # a reader needs "7 of 10 years", because the denominator is what says
    # whether the record is long enough to mean anything. Nothing scores these:
    # pillars and coverage both name their inputs explicitly.
    metrics["roe_years_above_15_count"] = float(above_15) if clean_roe else None
    metrics["roe_observation_years"] = float(len(clean_roe)) if clean_roe else None

    # ROIC: NOPAT over capital actually employed. Cash is netted off because
    # a large idle balance would otherwise depress the return on the operating
    # business that the moat question is really about.
    roic_values = []
    for period in window:
        ebit = _finite(concepts.get("pretax_income", {}).get(period))
        interest = _finite(concepts.get("interest_expense", {}).get(period)) or 0.0
        tax = _finite(concepts.get("tax_provision", {}).get(period))
        pretax = ebit
        if ebit is None:
            continue
        ebit = ebit + interest
        tax_rate = _safe_divide(tax, pretax) if pretax and pretax > 0 else None
        if tax_rate is None or not (0.0 <= tax_rate <= 0.6):
            tax_rate = 0.21  # US statutory rate; filers with odd effective rates
        nopat = ebit * (1.0 - tax_rate)

        equity = _finite(concepts.get("equity", {}).get(period))
        debt = (_finite(concepts.get("short_term_debt", {}).get(period)) or 0.0) + (
            _finite(concepts.get("long_term_debt", {}).get(period)) or 0.0
        )
        cash = _finite(concepts.get("cash", {}).get(period)) or 0.0
        if equity is None:
            continue
        invested = equity + debt - cash
        value = _safe_divide(nopat, invested)
        if value is not None and -2.0 < value < 3.0:
            roic_values.append(value * 100.0)
    metrics["roic_median"] = _median(roic_values)

    # --- Financial strength (balance-sheet state: latest period) ------------
    latest = all_periods[-1] if all_periods else None
    if latest:
        latest_ratios = ratios_by_period.get(latest, {})
        metrics["debt_to_equity"] = _finite(latest_ratios.get("Debt-to-Equity Ratio"))
        metrics["current_ratio"] = _finite(latest_ratios.get("Current Ratio"))

        cash = _finite(concepts.get("cash", {}).get(latest)) or 0.0
        total_debt = _total_debt(concepts, latest)
        interest = _finite(concepts.get("interest_expense", {}).get(latest))
        pretax = _finite(concepts.get("pretax_income", {}).get(latest))

        metrics["total_debt"] = total_debt
        metrics["net_debt"] = (total_debt - cash) if total_debt is not None else None

        if interest and interest > 0 and pretax is not None:
            metrics["interest_coverage"] = (pretax + interest) / interest
        elif total_debt == 0 or (total_debt is None and not interest):
            # Genuinely debt-free: either a reported zero, or no debt tag *and*
            # no interest expense, which together are strong evidence rather
            # than a data gap. Coverage is undefined, not zero — a debt-free
            # balance sheet must not fail the leverage gate.
            metrics["interest_coverage"] = None
            metrics["debt_free"] = 1.0
        else:
            # Debt exists but interest expense is missing: genuinely unknown.
            metrics["interest_coverage"] = None

    owner_earnings = _owner_earnings(concepts, window)
    latest_owner_earnings = next(
        (v for v in reversed(owner_earnings) if v is not None), None
    )
    metrics["owner_earnings_latest"] = latest_owner_earnings
    metrics["net_debt_to_owner_earnings"] = (
        _safe_divide(metrics.get("net_debt"), latest_owner_earnings)
        if latest_owner_earnings and latest_owner_earnings > 0
        else None
    )

    # --- Predictability -----------------------------------------------------
    metrics["roe_stdev"] = _stdev(roe_series)
    observed_owner_earnings = [v for v in owner_earnings if v is not None]
    metrics["negative_owner_earnings_years"] = sum(
        1 for v in observed_owner_earnings if v < 0
    )
    metrics["owner_earnings_years"] = (
        float(len(observed_owner_earnings)) if observed_owner_earnings else None
    )

    revenue_series = _aligned(concepts.get("revenue", {}), window)
    growth_rates = []
    for previous, current in zip(revenue_series, revenue_series[1:]):
        rate = _safe_divide((current or 0) - (previous or 0), previous)
        if rate is not None and abs(rate) < 5.0:
            growth_rates.append(rate)
    metrics["revenue_growth_stdev"] = _stdev(growth_rates)

    fcf_margins = []
    for period, owner in zip(window, owner_earnings):
        revenue = _finite(concepts.get("revenue", {}).get(period))
        margin = _safe_divide(owner, revenue)
        if margin is not None and -2.0 < margin < 2.0:
            fcf_margins.append(margin)
    metrics["fcf_margin_median"] = _median(fcf_margins)
    metrics["fcf_margin_stdev"] = _stdev(fcf_margins)

    # --- Growth -------------------------------------------------------------
    metrics["revenue_cagr"] = _cagr([v for v in revenue_series if v is not None])
    metrics["owner_earnings_cagr"] = _cagr([v for v in owner_earnings if v is not None])

    equity_series = _aligned(concepts.get("equity", {}), window)
    shares_series = _aligned(concepts.get("shares_diluted", {}), window)
    book_per_share = []
    for equity, shares in zip(equity_series, shares_series):
        value = _safe_divide(equity, shares)
        if value is not None and value > 0:
            book_per_share.append(value)
    metrics["book_value_per_share_cagr"] = _cagr(book_per_share)

    # --- Capital allocation -------------------------------------------------
    clean_shares = [v for v in shares_series if v is not None and v > 0]
    metrics["share_count_cagr"] = _cagr(clean_shares)

    dividends = _aligned(concepts.get("dividends_paid", {}), window)
    payout_ratios = []
    for dividend, owner in zip(dividends, owner_earnings):
        if dividend is None or owner is None or owner <= 0:
            continue
        ratio = abs(dividend) / owner
        if 0.0 <= ratio < 3.0:
            payout_ratios.append(ratio)
    metrics["payout_ratio_median"] = _median(payout_ratios)

    # Incremental return on incremental capital: the sharpest test of whether
    # management creates value with retained earnings. Preferred over the
    # "$1 retained → $1 of market value" formulation because it needs no price
    # history, so it cannot be contaminated by look-ahead bias.
    if len(window) >= MIN_PERIODS_FOR_CONSISTENCY:
        first_period, last_period = window[0], window[-1]
        ebit_start = _finite(concepts.get("pretax_income", {}).get(first_period))
        ebit_end = _finite(concepts.get("pretax_income", {}).get(last_period))
        equity_start = _finite(concepts.get("equity", {}).get(first_period))
        equity_end = _finite(concepts.get("equity", {}).get(last_period))
        if None not in (ebit_start, ebit_end, equity_start, equity_end):
            delta_capital = equity_end - equity_start
            if delta_capital > 0:
                incremental = (ebit_end - ebit_start) / delta_capital
                if -2.0 < incremental < 5.0:
                    metrics["incremental_roic"] = incremental * 100.0

    return metrics


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
    """
    metrics: Dict[str, Optional[float]] = {}

    all_periods = sorted(set(concepts.get("net_income", {})))
    window = all_periods[-DURABILITY_WINDOW:]

    roe_series = [ratios_by_period.get(p, {}).get("Return on Equity (ROE) (%)") for p in window]
    roa_series = [ratios_by_period.get(p, {}).get("Return on Assets (ROA) (%)") for p in window]
    metrics["roe_median"] = _median(roe_series)
    metrics["roa_median"] = _median(roa_series)
    metrics["roe_stdev"] = _stdev(roe_series)

    clean_roe = [v for v in (_finite(x) for x in roe_series) if v is not None]
    # Banks clear a lower bar than industrials: sustained mid-teens ROE on a
    # conservatively funded book is an excellent bank.
    above_12 = sum(1 for v in clean_roe if v >= 12.0)
    metrics["roe_years_above_12"] = above_12 / len(clean_roe) if clean_roe else None
    metrics["roe_years_above_12_count"] = float(above_12) if clean_roe else None
    metrics["roe_observation_years"] = float(len(clean_roe)) if clean_roe else None

    # Solvency: equity as a share of assets, the leverage measure that actually
    # binds for a bank.
    equity_to_assets = []
    for period in window:
        ratio = _safe_divide(
            concepts.get("equity", {}).get(period), concepts.get("total_assets", {}).get(period)
        )
        if ratio is not None and 0 < ratio < 1:
            equity_to_assets.append(ratio * 100.0)
    metrics["equity_to_assets_median"] = _median(equity_to_assets)
    metrics["equity_to_assets_latest"] = equity_to_assets[-1] if equity_to_assets else None

    # Credit discipline: provisions against the loan book, and how much that
    # rate swings. A lender whose provisioning is stable through a cycle is
    # underwriting consistently.
    provision_rates = []
    for period in window:
        rate = _safe_divide(
            sector.get("provision_for_credit_losses", {}).get(period),
            sector.get("loans", {}).get(period),
        )
        if rate is not None and -0.1 < rate < 0.5:
            provision_rates.append(rate * 100.0)
    metrics["provision_rate_median"] = _median(provision_rates)
    metrics["provision_rate_stdev"] = _stdev(provision_rates)

    # Net interest margin against earning assets, proxied by total assets.
    nim_values = []
    for period in window:
        margin = _safe_divide(
            sector.get("net_interest_income", {}).get(period),
            concepts.get("total_assets", {}).get(period),
        )
        if margin is not None and 0 < margin < 0.25:
            nim_values.append(margin * 100.0)
    metrics["net_interest_margin_median"] = _median(nim_values)
    metrics["net_interest_margin_stdev"] = _stdev(nim_values)

    # Efficiency ratio: costs per dollar of revenue. Among the most durable
    # signals of a well-run bank.
    efficiency = []
    for period in window:
        net_interest = _finite(sector.get("net_interest_income", {}).get(period))
        noninterest_income = _finite(sector.get("noninterest_income", {}).get(period))
        noninterest_expense = _finite(sector.get("noninterest_expense", {}).get(period))
        if None in (net_interest, noninterest_expense):
            continue
        revenue = net_interest + (noninterest_income or 0.0)
        ratio = _safe_divide(noninterest_expense, revenue)
        if ratio is not None and 0 < ratio < 2:
            efficiency.append(ratio * 100.0)
    metrics["efficiency_ratio_median"] = _median(efficiency)

    net_income_series = _aligned(concepts.get("net_income", {}), window)
    observed_net_income = [v for v in net_income_series if v is not None]
    metrics["loss_years"] = sum(1 for v in observed_net_income if v < 0)
    metrics["net_income_years"] = (
        float(len(observed_net_income)) if observed_net_income else None
    )

    equity_series = _aligned(concepts.get("equity", {}), window)
    shares_series = _aligned(concepts.get("shares_diluted", {}), window)
    book_per_share = []
    for equity, shares in zip(equity_series, shares_series):
        value = _safe_divide(equity, shares)
        if value is not None and value > 0:
            book_per_share.append(value)
    metrics["book_value_per_share_cagr"] = _cagr(book_per_share)

    # Top-line growth from banking revenue, not the generic `revenue` concept:
    # measured coverage of `revenue` within banks is only 60%, because most
    # banks report net interest income and fee income rather than a single
    # revenue line. Falls back to `revenue` when the components are absent.
    banking_revenue = []
    for period in window:
        net_interest = _finite(sector.get("net_interest_income", {}).get(period))
        fees = _finite(sector.get("noninterest_income", {}).get(period))
        if net_interest is not None:
            banking_revenue.append(net_interest + (fees or 0.0))
        else:
            banking_revenue.append(_finite(concepts.get("revenue", {}).get(period)))
    metrics["revenue_cagr"] = _cagr([v for v in banking_revenue if v is not None])

    clean_shares = [v for v in shares_series if v is not None and v > 0]
    metrics["share_count_cagr"] = _cagr(clean_shares)

    return metrics


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
    """
    metrics: Dict[str, Optional[float]] = {}

    all_periods = sorted(set(concepts.get("net_income", {})))
    window = all_periods[-DURABILITY_WINDOW:]

    ffo_series: List[Optional[float]] = []
    for period in window:
        net_income = _finite(concepts.get("net_income", {}).get(period))
        if net_income is None:
            ffo_series.append(None)
            continue
        depreciation = _finite(
            concepts.get("depreciation_amortization", {}).get(period)
        ) or 0.0
        gains = _finite(sector.get("gain_on_sale_real_estate", {}).get(period)) or 0.0
        ffo_series.append(net_income + depreciation - gains)

    observed_ffo = [v for v in ffo_series if v is not None]
    metrics["ffo_latest"] = next((v for v in reversed(ffo_series) if v is not None), None)
    metrics["ffo_cagr"] = _cagr(observed_ffo)
    metrics["negative_ffo_years"] = sum(1 for v in observed_ffo if v < 0)
    metrics["ffo_years"] = float(len(observed_ffo)) if observed_ffo else None

    ffo_margins = []
    ffo_on_equity = []
    for period, ffo in zip(window, ffo_series):
        if ffo is None:
            continue
        revenue = _finite(concepts.get("revenue", {}).get(period))
        margin = _safe_divide(ffo, revenue)
        if margin is not None and -2 < margin < 2:
            ffo_margins.append(margin * 100.0)
        equity_return = _safe_divide(ffo, concepts.get("equity", {}).get(period))
        if equity_return is not None and -1 < equity_return < 2:
            ffo_on_equity.append(equity_return * 100.0)
    metrics["ffo_margin_median"] = _median(ffo_margins)
    metrics["ffo_margin_stdev"] = _stdev(ffo_margins)
    metrics["ffo_on_equity_median"] = _median(ffo_on_equity)

    # Leverage: debt against FFO is the REIT analogue of debt/EBITDA. Equity
    # ratios mislead because property is carried at depreciated cost.
    latest = window[-1] if window else None
    if latest:
        total_debt = _total_debt(concepts, latest)
        metrics["total_debt"] = total_debt
        if total_debt is not None:
            ffo_latest = metrics.get("ffo_latest")
            metrics["debt_to_ffo"] = (
                _safe_divide(total_debt, ffo_latest)
                if ffo_latest and ffo_latest > 0
                else None
            )
            metrics["debt_to_real_estate"] = _safe_divide(
                total_debt, sector.get("real_estate_net", {}).get(latest)
            )

    shares_series = _aligned(concepts.get("shares_diluted", {}), window)
    clean_shares = [v for v in shares_series if v is not None and v > 0]
    metrics["share_count_cagr"] = _cagr(clean_shares)

    ffo_per_share = []
    for ffo, shares in zip(ffo_series, shares_series):
        value = _safe_divide(ffo, shares)
        if value is not None and value > 0:
            ffo_per_share.append(value)
    metrics["ffo_per_share_cagr"] = _cagr(ffo_per_share)

    metrics["real_estate_cagr"] = _cagr(
        [v for v in _aligned(sector.get("real_estate_net", {}), window) if v is not None]
    )

    return metrics


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
            logging.warning(f"Metrics: share-count reconstruction failed for {symbol}: {exc}")

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
