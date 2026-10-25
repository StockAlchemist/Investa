"""Frozen reference for the Rust Monte Carlo models in rust/investa_core/src/mc.rs.

financial_ratios._generate_mc_stats and the twelve run_monte_carlo_* functions
exactly as they were before the port (commit 30511ba3). The clip bounds are
imported from financial_ratios, which still defines them.
tests/test_rust_mc_parity.py holds the port to this, bitwise.

Never edit these to follow a change in Rust. A deliberate behaviour change
lands in both, with the parity test updated in the same commit.
"""

# ruff: noqa
import logging
from typing import Any, Dict

import numpy as np

from financial_ratios import (
    MAX_DISCOUNT_RATE,
    MAX_PROJECTED_GROWTH,
    MIN_DISCOUNT_RATE,
    MIN_PROJECTED_GROWTH,
)


def _generate_mc_stats(intrinsic_values: np.ndarray) -> Dict[str, Any]:
    """Computes standard percentiles, standard deviation, and a 40-bin smoothed histogram for Monte Carlo samples."""
    try:
        valid = intrinsic_values[np.isfinite(intrinsic_values) & (intrinsic_values > 0)]
        if len(valid) < 50:
            return {}

        counts, edges = np.histogram(valid, bins=40)
        midpoints = (edges[:-1] + edges[1:]) / 2

        # 7-point Gaussian smoothing kernel
        kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        smoothed_counts = np.convolve(counts, kernel, mode="same")

        histogram = [
            {"price": float(p), "count": float(c)}
            for p, c in zip(midpoints, smoothed_counts)
        ]

        return {
            "bear": float(np.percentile(valid, 10)),
            "conservative": float(np.percentile(valid, 25)),
            "base": float(np.percentile(valid, 50)),
            "bull": float(np.percentile(valid, 90)),
            "std_dev": float(np.std(valid)),
            "histogram": histogram,
        }
    except Exception as e:
        logging.error(f"Error computing MC stats: {e}")
        return {}


def run_monte_carlo_dcf(
    ticker_info: Dict[str, Any],
    base_fcf: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Discounted Free Cash Flow (DCF)."""
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {}

        growth_sigma = max(0.06, abs(base_growth) * 0.25)
        growth_samples = np.random.normal(base_growth, growth_sigma, iterations)
        discount_sigma = max(0.015, abs(base_discount) * 0.15)
        discount_samples = np.random.normal(base_discount, discount_sigma, iterations)

        growth_samples = np.clip(
            growth_samples, MIN_PROJECTED_GROWTH, MAX_PROJECTED_GROWTH
        )
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        years = np.arange(1, projection_years + 1)
        fade_factors = (
            (years - 1) / (projection_years - 1)
            if projection_years > 1
            else np.array([0])
        )
        yearly_growths = (
            growth_samples[:, None]
            - (growth_samples[:, None] - terminal_growth) * fade_factors
        )

        fcf_projections = base_fcf * np.cumprod(1 + yearly_growths, axis=1)
        pv_projections = fcf_projections / (1 + discount_samples[:, None]) ** (
            years - 0.5
        )
        sum_pv_fcf = np.sum(pv_projections, axis=1)

        last_fcf = fcf_projections[:, -1]
        safe_discount = np.maximum(discount_samples, terminal_growth + 0.01)
        terminal_values = (last_fcf * (1 + terminal_growth)) / (
            safe_discount - terminal_growth
        )
        pv_terminal_values = (
            terminal_values / (1 + discount_samples) ** projection_years
        )

        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0
        enterprise_values = sum_pv_fcf + pv_terminal_values
        equity_values = enterprise_values + cash - debt
        intrinsic_values = equity_values / shares

        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo DCF failed: {e}")
        return {}


def run_monte_carlo_dcfo(
    ticker_info: Dict[str, Any],
    base_cfo: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Discounted Cash from Operations (D-CFO)."""
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {}

        cfo_per_share = base_cfo / shares
        growth_sigma = max(0.05, abs(base_growth) * 0.25)
        growth_samples = np.random.normal(base_growth, growth_sigma, iterations)
        discount_sigma = max(0.015, abs(base_discount) * 0.15)
        discount_samples = np.random.normal(base_discount, discount_sigma, iterations)

        growth_samples = np.clip(growth_samples, -0.05, 0.25)
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        years = np.arange(1, projection_years + 1)
        fade_factors = (
            (years - 1) / (projection_years - 1)
            if projection_years > 1
            else np.array([0])
        )
        yearly_growths = (
            growth_samples[:, None]
            - (growth_samples[:, None] - terminal_growth) * fade_factors
        )

        cfo_projections = cfo_per_share * np.cumprod(1 + yearly_growths, axis=1)
        pv_projections = cfo_projections / (1 + discount_samples[:, None]) ** (
            years - 0.5
        )
        sum_pv_cfo = np.sum(pv_projections, axis=1)

        last_cfo = cfo_projections[:, -1]
        safe_discount = np.maximum(discount_samples, terminal_growth + 0.01)
        terminal_values = (last_cfo * (1 + terminal_growth)) / (
            safe_discount - terminal_growth
        )
        pv_terminal_values = (
            terminal_values / (1 + discount_samples) ** projection_years
        )

        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0
        net_cash_per_share = (cash - debt) / shares
        intrinsic_values = sum_pv_cfo + pv_terminal_values + net_cash_per_share

        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo D-CFO failed: {e}")
        return {}


def run_monte_carlo_dni(
    base_eps: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Discounted Net Income (D-NI)."""
    try:
        growth_sigma = max(0.05, abs(base_growth) * 0.25)
        growth_samples = np.random.normal(base_growth, growth_sigma, iterations)
        discount_sigma = max(0.015, abs(base_discount) * 0.15)
        discount_samples = np.random.normal(base_discount, discount_sigma, iterations)

        growth_samples = np.clip(growth_samples, -0.05, 0.25)
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        years = np.arange(1, projection_years + 1)
        fade_factors = (
            (years - 1) / (projection_years - 1)
            if projection_years > 1
            else np.array([0])
        )
        yearly_growths = (
            growth_samples[:, None]
            - (growth_samples[:, None] - terminal_growth) * fade_factors
        )

        eps_projections = base_eps * np.cumprod(1 + yearly_growths, axis=1)
        pv_projections = eps_projections / (1 + discount_samples[:, None]) ** (
            years - 0.5
        )
        sum_pv_eps = np.sum(pv_projections, axis=1)

        last_eps = eps_projections[:, -1]
        safe_discount = np.maximum(discount_samples, terminal_growth + 0.01)
        terminal_values = (last_eps * (1 + terminal_growth)) / (
            safe_discount - terminal_growth
        )
        pv_terminal_values = (
            terminal_values / (1 + discount_samples) ** projection_years
        )

        intrinsic_values = sum_pv_eps + pv_terminal_values
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo D-NI failed: {e}")
        return {}


def run_monte_carlo_mean_pe(
    eps: float,
    applied_pe: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Mean P/E Valuation."""
    try:
        eps_samples = np.random.normal(eps, max(0.1, abs(eps) * 0.15), iterations)
        pe_samples = np.random.normal(
            applied_pe, max(1.5, applied_pe * 0.18), iterations
        )

        eps_samples = np.clip(eps_samples, 0.01, eps * 2.5)
        pe_samples = np.clip(pe_samples, 5.0, 45.0)

        intrinsic_values = eps_samples * pe_samples
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo Mean P/E failed: {e}")
        return {}


def run_monte_carlo_peg(
    eps: float,
    applied_growth_pct: float,
    dividend_yield_pct: float = 0.0,
    target_peg: float = 1.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for PEG Ratio Fair Value."""
    try:
        eps_samples = np.random.normal(eps, max(0.1, abs(eps) * 0.15), iterations)
        growth_samples = np.random.normal(
            applied_growth_pct, max(1.5, abs(applied_growth_pct) * 0.25), iterations
        )
        peg_samples = np.random.normal(
            target_peg, max(0.1, target_peg * 0.15), iterations
        )

        eps_samples = np.clip(eps_samples, 0.01, eps * 2.5)
        growth_samples = np.clip(growth_samples, 2.0, 40.0)
        peg_samples = np.clip(peg_samples, 0.5, 2.5)

        fair_pe = np.clip(
            peg_samples * (growth_samples + dividend_yield_pct), 5.0, 35.0
        )
        intrinsic_values = eps_samples * fair_pe
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo PEG failed: {e}")
        return {}


def run_monte_carlo_mean_pb(
    book_value_per_share: float,
    applied_pb: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Mean P/B Ratio Valuation."""
    try:
        bv_samples = np.random.normal(
            book_value_per_share, max(0.1, book_value_per_share * 0.08), iterations
        )
        pb_samples = np.random.normal(
            applied_pb, max(0.15, applied_pb * 0.18), iterations
        )

        bv_samples = np.clip(bv_samples, 0.01, book_value_per_share * 2.0)
        pb_samples = np.clip(pb_samples, 0.4, 8.0)

        intrinsic_values = bv_samples * pb_samples
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo Mean P/B failed: {e}")
        return {}


def run_monte_carlo_mean_ps(
    sales_per_share: float,
    applied_ps: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Mean P/S Ratio Valuation."""
    try:
        sps_samples = np.random.normal(
            sales_per_share, max(0.1, sales_per_share * 0.10), iterations
        )
        ps_samples = np.random.normal(
            applied_ps, max(0.2, applied_ps * 0.18), iterations
        )

        sps_samples = np.clip(sps_samples, 0.01, sales_per_share * 2.0)
        ps_samples = np.clip(ps_samples, 0.4, 15.0)

        intrinsic_values = sps_samples * ps_samples
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo Mean P/S failed: {e}")
        return {}


def run_monte_carlo_psg(
    sales_per_share: float,
    applied_growth_pct: float,
    gross_margin: float,
    target_psg: float = 1.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Price-to-Sales Growth (PSG) Valuation."""
    try:
        sps_samples = np.random.normal(
            sales_per_share, max(0.1, sales_per_share * 0.10), iterations
        )
        growth_samples = np.random.normal(
            applied_growth_pct, max(2.0, applied_growth_pct * 0.25), iterations
        )
        gm_samples = np.random.normal(
            gross_margin, max(0.02, gross_margin * 0.08), iterations
        )
        target_psg_samples = np.random.normal(
            target_psg, max(0.1, target_psg * 0.15), iterations
        )

        sps_samples = np.clip(sps_samples, 0.01, sales_per_share * 2.0)
        growth_samples = np.clip(growth_samples, 5.0, 50.0)
        gm_samples = np.clip(gm_samples, 0.20, 0.95)
        target_psg_samples = np.clip(target_psg_samples, 0.5, 2.0)

        fair_ps = np.clip(
            (growth_samples / 10.0) * gm_samples * target_psg_samples * 1.5, 0.5, 15.0
        )
        intrinsic_values = sps_samples * fair_ps
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo PSG failed: {e}")
        return {}


def run_monte_carlo_graham(
    eps: float, base_growth: float, base_bond_yield: float, iterations: int = 10000
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Graham's Formula."""
    try:
        growth_samples = np.random.normal(
            base_growth, max(1.0, abs(base_growth) * 0.2), iterations
        )
        yield_samples = np.random.normal(
            base_bond_yield, max(0.3, abs(base_bond_yield) * 0.1), iterations
        )

        growth_samples = np.clip(growth_samples, 0.0, 25.0)
        yield_samples = np.clip(yield_samples, 0.5, 10.0)

        intrinsic_values = (eps * (8.5 + 2 * growth_samples) * 4.4) / yield_samples
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo Graham failed: {e}")
        return {}


def run_monte_carlo_ddm(
    base_dividend: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Dividend Discount Model (DDM)."""
    try:
        growth_sigma = max(0.03, abs(base_growth) * 0.25)
        growth_samples = np.random.normal(base_growth, growth_sigma, iterations)
        discount_sigma = max(0.015, abs(base_discount) * 0.15)
        discount_samples = np.random.normal(base_discount, discount_sigma, iterations)

        growth_samples = np.clip(growth_samples, -0.05, 0.15)
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        years = np.arange(1, projection_years + 1)
        fade_factors = (
            (years - 1) / (projection_years - 1)
            if projection_years > 1
            else np.array([0])
        )
        yearly_growths = (
            growth_samples[:, None]
            - (growth_samples[:, None] - terminal_growth) * fade_factors
        )

        div_projections = base_dividend * np.cumprod(1 + yearly_growths, axis=1)
        pv_projections = div_projections / (1 + discount_samples[:, None]) ** (
            years - 0.5
        )
        sum_pv_div = np.sum(pv_projections, axis=1)

        last_div = div_projections[:, -1]
        safe_discount = np.maximum(discount_samples, terminal_growth + 0.01)
        terminal_values = (last_div * (1 + terminal_growth)) / (
            safe_discount - terminal_growth
        )
        pv_terminal_values = (
            terminal_values / (1 + discount_samples) ** projection_years
        )

        intrinsic_values = sum_pv_div + pv_terminal_values
        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo DDM failed: {e}")
        return {}


def run_monte_carlo_epv(
    ticker_info: Dict[str, Any],
    normalized_ebit: float,
    tax_rate: float,
    base_discount: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Earnings Power Value (EPV)."""
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {}

        ebit_samples = np.random.normal(
            normalized_ebit, max(1.0, abs(normalized_ebit) * 0.15), iterations
        )
        discount_samples = np.random.normal(
            base_discount, max(0.01, abs(base_discount) * 0.12), iterations
        )

        ebit_samples = np.clip(ebit_samples, 0.01, normalized_ebit * 2.5)
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        nopat_samples = ebit_samples * (1 - tax_rate)
        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0

        enterprise_values = nopat_samples / discount_samples
        equity_values = enterprise_values + cash - debt
        intrinsic_values = np.maximum(0.01, equity_values / shares)

        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo EPV failed: {e}")
        return {}


def run_monte_carlo_lynch(
    eps: float,
    growth_rate_pct: float,
    dividend_yield_pct: float = 0.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Peter Lynch Fair Value."""
    try:
        eps_samples = np.random.normal(eps, max(0.1, abs(eps) * 0.15), iterations)
        growth_samples = np.random.normal(
            growth_rate_pct, max(1.5, abs(growth_rate_pct) * 0.25), iterations
        )

        eps_samples = np.clip(eps_samples, 0.01, eps * 2.5)
        growth_samples = np.clip(growth_samples, 2.0, 35.0)

        multipliers = np.clip(growth_samples + dividend_yield_pct, 5.0, 25.0)
        intrinsic_values = eps_samples * multipliers

        return _generate_mc_stats(intrinsic_values)
    except Exception as e:
        logging.error(f"Monte Carlo Lynch failed: {e}")
        return {}
