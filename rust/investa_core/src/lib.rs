//! Investa's calculation core.
//!
//! The Python backend keeps the web layer, data fetching and orchestration; the
//! calculations it hands arrays to live here. Each module is a port of a Python
//! (formerly Numba) kernel and keeps that kernel's arithmetic exactly, so the
//! golden tests and every cached history stay valid.

pub mod analyzer;
pub mod asof;
pub mod buffett;
pub mod ca;
pub mod cashflows;
pub mod db_portfolio;
pub mod db_screener;
pub mod dl;
pub mod ef_store;
pub mod fifo;
pub mod history;
pub mod holdings;
pub mod irr;
mod irr_grid;
pub mod mc;
pub mod mdb;
pub mod pa_fifo;
pub mod pa_holdings;
pub mod pa_reports;
pub mod pa_small;
pub mod projection;
mod py;
pub mod pyconn;
pub mod pyhelp;
pub mod rank;
pub mod rank_store;
pub mod ratios;
pub mod records;
pub mod sql;
pub mod sqlite_read;
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
    m.add_function(wrap_pyfunction!(py::value_holdings_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py::portfolio_twr, m)?)?;
    m.add_function(wrap_pyfunction!(py::benchmark_accumulated, m)?)?;
    m.add_function(wrap_pyfunction!(py::money_weighted, m)?)?;
    m.add_class::<py::PyPriceTable>()?;
    m.add_function(wrap_pyfunction!(py::holding_flows, m)?)?;
    m.add_function(wrap_pyfunction!(py::mwr_flows, m)?)?;
    m.add_function(wrap_pyfunction!(py::key_ratios, m)?)?;
    m.add_function(wrap_pyfunction!(py::key_ratio_fields, m)?)?;
    m.add_function(wrap_pyfunction!(py::sqlite_bind, m)?)?;
    m.add_function(wrap_pyfunction!(py::read_daily_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(py::sql_execute, m)?)?;
    m.add_function(wrap_pyfunction!(py::sql_executemany, m)?)?;
    m.add_function(wrap_pyfunction!(py::sql_db_filename, m)?)?;
    m.add_function(wrap_pyfunction!(pyconn::conn_execute, m)?)?;
    m.add_function(wrap_pyfunction!(pyconn::conn_executemany, m)?)?;
    m.add_class::<pyconn::SqlRows>()?;
    m.add_function(wrap_pyfunction!(pyconn::conn_read_frame, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_apply_total_amount_sign,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_add_transaction, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_update_transaction, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_delete_transaction, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_get_all_watchlists, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_create_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_get_or_create_favorites,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_watchlist_ids_for_symbol,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_rename_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_delete_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_add_to_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_remove_from_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_get_watchlist, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_load_all_transactions, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_check_empty_and_csv_exists,
        m
    )?)?;

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
    m.add_function(wrap_pyfunction!(db_portfolio::db_get_database_path, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_get_global_screener_db_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_ensure_screener_schema,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_open_screener_db_conn, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_is_path_on_cloud_drive,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_get_db_connection, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_create_transactions_table,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_initialize_database, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_portfolio::db_initialize_global_database,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_connect_readonly, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_upsert_screener_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_screener::db_refresh_screener_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_get_cached_screener_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_get_screener_results_by_universe,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_get_all_distinct_screener_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(db_screener::db_is_synthetic_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_update_ai_review_in_cache,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        db_screener::db_update_intrinsic_value_in_cache,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_iso_days, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_future_split_factors, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_dividend_factors, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_adjust_frame, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_actions, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_set_price_basis, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_fx, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_fx_rows, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_has_column, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_fund_nav, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_share_counts, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_upsert_intraday, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_rust_reader_bound, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_read_daily_frames, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_actions, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_price_basis, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_ohlcv_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_intraday, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_data_quality, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_fx, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_fund_nav, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_latest_fund_navs, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_fund_nav_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_share_counts, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_latest_closes, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_last_date, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_last_dates, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_first_dates, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_get_sync_metadata_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_check_integrity, m)?)?;
    m.add_function(wrap_pyfunction!(mdb::mdb_init_db, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_escape_like, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_coerce, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_init_db, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_start_run, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_finish_run, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_prune_runs, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_save_scores, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_save_exclusions, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_latest_run_id, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_get_ranked, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_count_ranked, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_get_exclusions, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_count_exclusions, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_get_run, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_get_scores_frame, m)?)?;
    m.add_function(wrap_pyfunction!(rank_store::rs_get_symbol_history, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_effective_as_of, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_session_for, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_get_many_tag_spans, m)?)?;
    m.add_function(wrap_pyfunction!(
        ef_store::ef_get_many_tag_spans_by_filing,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_get_tag_series, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_many_tag_series_query, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_get_many_tag_series, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_get_tag_series_by_filing, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_get_tag_revisions, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_has_data, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_init_db, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_upsert_facts, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_upsert_quarterly_facts, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_mark_ingested, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_mark_quarterly_ingested, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_ingested_ciks, m)?)?;
    m.add_function(wrap_pyfunction!(ef_store::ef_quarterly_ingest_state, m)?)?;
    m.add_class::<ef_store::CompanySession>()?;
    m.add_function(wrap_pyfunction!(dl::dl_normalize_header, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_map_headers, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_parse_date_robustly, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_load_and_clean, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_convert_csv_headers, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(dl::dl_write_excel, m)?)?;
    m.add_function(wrap_pyfunction!(db_portfolio::db_migrate_csv_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_apply_return_of_capital, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_apply_stock_dividend, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_apply_spin_off, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_parse_spinoff_description, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_build_spinoff_legs, m)?)?;
    m.add_function(wrap_pyfunction!(ca::ca_deduplicate_split_transactions, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_get_type_id, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_run_scenario_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_rebalancing_trades, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_calculate_hhi, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_health_score, m)?)?;
    m.add_function(wrap_pyfunction!(pa_small::pa_cash_interest_events, m)?)?;
    m.add_function(wrap_pyfunction!(pa_reports::pa_periodic_returns, m)?)?;
    m.add_function(wrap_pyfunction!(pa_reports::pa_dividend_history, m)?)?;
    m.add_function(wrap_pyfunction!(pa_reports::pa_realized_gains_history, m)?)?;
    m.add_function(wrap_pyfunction!(
        pa_holdings::pa_transactions_to_holdings,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pa_fifo::pa_fifo_lots_and_gains, m)?)?;
    Ok(())
}
