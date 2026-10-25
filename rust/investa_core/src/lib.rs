//! Investa's calculation core.
//!
//! The Python backend keeps the web layer, data fetching and orchestration; the
//! calculations it hands arrays to live here. Each module is a port of a Python
//! (formerly Numba) kernel and keeps that kernel's arithmetic exactly, so the
//! golden tests and every cached history stay valid.

pub mod analyzer;
pub mod buffett;
pub mod fifo;
pub mod holdings;
pub mod irr;
mod irr_grid;
pub mod mc;
pub mod projection;
mod py;
pub mod rank;
pub mod records;
pub mod stats;
pub mod util;
pub mod valuation;

use pyo3::prelude::*;

/// Hash of this crate's sources, folded into the history-cache version.
pub const SOURCE_HASH: &str = env!("INVESTA_CORE_SOURCE_HASH");

#[pymodule]
fn investa_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SOURCE_HASH", SOURCE_HASH)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(py::calculate_holdings, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::calculate_daily_holdings_chronological,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::process_core, m)?)?;
    m.add_function(wrap_pyfunction!(py::dividend_amount, m)?)?;
    m.add_function(wrap_pyfunction!(py::npv, m)?)?;
    m.add_function(wrap_pyfunction!(py::calculate_irr, m)?)?;
    m.add_function(wrap_pyfunction!(py::irr_scan_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py::py_sum, m)?)?;
    m.add_function(wrap_pyfunction!(py::brentq_npv, m)?)?;
    m.add_function(wrap_pyfunction!(py::newton_npv, m)?)?;
    m.add_function(wrap_pyfunction!(py::np_sum, m)?)?;
    m.add_function(wrap_pyfunction!(py::pd_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py::pd_min, m)?)?;
    m.add_function(wrap_pyfunction!(py::pd_var, m)?)?;
    m.add_function(wrap_pyfunction!(py::pd_std, m)?)?;
    m.add_function(wrap_pyfunction!(py::max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(py::drawdown_series, m)?)?;
    m.add_function(wrap_pyfunction!(py::pct_change, m)?)?;
    m.add_function(wrap_pyfunction!(py::volatility, m)?)?;
    m.add_function(wrap_pyfunction!(py::sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py::sortino_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py::scoreboard_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_dcf, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_discounted, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_base_times_multiple, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_peg, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_psg, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_graham, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_epv, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_lynch, m)?)?;
    m.add_function(wrap_pyfunction!(py::mc_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py::np_median, m)?)?;
    m.add_function(wrap_pyfunction!(py::np_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py::scalar_projection, m)?)?;
    m.add_function(wrap_pyfunction!(py::wacc, m)?)?;
    m.add_function(wrap_pyfunction!(py::blend_growth, m)?)?;
    m.add_function(wrap_pyfunction!(py::plausible_margins, m)?)?;
    m.add_function(wrap_pyfunction!(py::model_reliability, m)?)?;
    m.add_function(wrap_pyfunction!(py::blend, m)?)?;
    m.add_function(wrap_pyfunction!(py::np_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(py::buffett_generic_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(py::buffett_bank_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(py::buffett_reit_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(py::py_pstdev, m)?)?;
    m.add_function(wrap_pyfunction!(py::buffett_total_debt, m)?)?;
    m.add_function(wrap_pyfunction!(py::buffett_cagr, m)?)?;
    m.add_function(wrap_pyfunction!(py::winsorised_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(py::row_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py::weighted_renormalised, m)?)?;
    m.add_function(wrap_pyfunction!(py::projection, m)?)?;
    m.add_function(wrap_pyfunction!(py::walk_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py::summarize_errors, m)?)?;
    m.add_function(wrap_pyfunction!(py::band_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py::trend_sma, m)?)?;
    m.add_function(wrap_pyfunction!(py::stress_window, m)?)?;
    m.add_function(wrap_pyfunction!(py::fifo_lots, m)?)?;

    // portfolio_analyzer's TYPE_* constants, so Python can check they agree.
    let types = PyModule::new(m.py(), "analyzer_types")?;
    use analyzer::types as t;
    for (name, value) in [
        ("TYPE_BUY", t::BUY),
        ("TYPE_SELL", t::SELL),
        ("TYPE_DEPOSIT", t::DEPOSIT),
        ("TYPE_WITHDRAWAL", t::WITHDRAWAL),
        ("TYPE_DIVIDEND", t::DIVIDEND),
        ("TYPE_FEES", t::FEES),
        ("TYPE_SPLIT", t::SPLIT),
        ("TYPE_TRANSFER", t::TRANSFER),
        ("TYPE_SHORT_SELL", t::SHORT_SELL),
        ("TYPE_BUY_TO_COVER", t::BUY_TO_COVER),
        ("TYPE_TAX", t::TAX),
        ("TYPE_INTEREST", t::INTEREST),
        ("TYPE_SPIN_OFF", t::SPIN_OFF),
        ("TYPE_UNKNOWN", t::UNKNOWN),
    ] {
        types.add(name, value)?;
    }
    m.add_submodule(&types)?;
    Ok(())
}
