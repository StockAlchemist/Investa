//! Investa's calculation core.
//!
//! The Python backend keeps the web layer, data fetching and orchestration; the
//! calculations it hands arrays to live here. Each module is a port of a Python
//! (formerly Numba) kernel and keeps that kernel's arithmetic exactly, so the
//! golden tests and every cached history stay valid.

pub mod analyzer;
pub mod asof;
pub mod bm;
pub mod bp;
pub mod br;
pub mod buffett;
pub mod ca;
pub mod cashflows;
pub mod db_portfolio;
pub mod db_screener;
pub mod dl;
pub mod ef_store;
pub mod ep;
pub mod fifo;
pub mod fr;
pub mod fr_blend;
pub mod fr_growth;
pub mod fr_models;
pub mod fu;
pub mod fu_flows;
pub mod fu_fmt;
pub mod fu_fx;
pub mod history;
pub mod holdings;
pub mod irr;
mod irr_grid;
pub mod mc;
pub mod mdb;
pub mod pa_aggregate;
pub mod pa_fifo;
pub mod pa_holdings;
pub mod pa_reports;
pub mod pa_rows;
pub mod pa_small;
pub mod pc_flows;
pub mod ph_daily;
pub mod ph_performance;
pub mod ph_small;
pub mod ph_valuation;
pub mod pl_helpers;
pub mod pl_summary;
pub mod projection;
mod py;
pub mod pyconn;
pub mod pyhelp;
pub mod rank;
pub mod rank_store;
pub mod ratios;
pub mod records;
pub mod rm;
pub mod sql;
pub mod sqlite_read;
pub mod st;
pub mod stats;
pub mod tr;
pub mod util;
pub mod valuation;
pub mod vh;

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
    m.add_function(wrap_pyfunction!(pa_rows::pa_build_summary_rows, m)?)?;
    m.add_function(wrap_pyfunction!(pa_aggregate::pa_aggregate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(
        pl_helpers::pl_transactions_fingerprint,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pl_helpers::pl_default_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(pl_helpers::pl_published_nav_prices, m)?)?;
    m.add_function(wrap_pyfunction!(pl_helpers::pl_enrich_sector_geo, m)?)?;
    m.add_function(wrap_pyfunction!(pl_helpers::pl_prepare_historical_fx, m)?)?;
    m.add_function(wrap_pyfunction!(pl_summary::pl_portfolio_summary, m)?)?;
    m.add_function(wrap_pyfunction!(
        pc_flows::pc_daily_net_cash_flow_vectorized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pc_flows::pc_daily_net_cash_flow, m)?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_unadjust_prices, m)?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_daily_metrics_worker, m)?)?;
    m.add_function(wrap_pyfunction!(
        ph_small::ph_manual_prices_by_symbol_id,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_fund_nav_series, m)?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_align_prices_to_grid, m)?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_prune_holdings_cache, m)?)?;
    m.add_function(wrap_pyfunction!(
        ph_small::ph_intraday_symbols_for_window,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ph_small::ph_generate_mappings, m)?)?;
    m.add_function(wrap_pyfunction!(
        ph_valuation::ph_prepare_historical_inputs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ph_valuation::ph_value_daily_holdings, m)?)?;
    m.add_function(wrap_pyfunction!(ph_valuation::ph_all_daily_holdings, m)?)?;
    m.add_function(wrap_pyfunction!(
        ph_daily::ph_accumulated_gains_and_resample,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ph_daily::ph_load_or_calculate_daily_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ph_performance::ph_historical_performance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(vh::vh_percentile_of, m)?)?;
    m.add_function(wrap_pyfunction!(vh::vh_describe_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(vh::vh_price_on_or_before, m)?)?;
    m.add_function(wrap_pyfunction!(vh::vh_load_prices, m)?)?;
    m.add_function(wrap_pyfunction!(vh::vh_observations, m)?)?;
    m.add_function(wrap_pyfunction!(vh::vh_bands, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_format, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_labelled_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_by_fiscal_year, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_stress_response, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_money, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_revisions, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_ranked_row, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_bands, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_stress, m)?)?;
    m.add_function(wrap_pyfunction!(tr::tr_build, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_month_end_closes, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_next_month_end, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_evaluate_trend_signal, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_load_signal_prices, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_latest_closes, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_market_trend_signal, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_optional_float, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_industry_label, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_apply_sector_cap, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_ranking_positions, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_ranking_age_days, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_sleeve_label, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_build_allocation, m)?)?;
    m.add_function(wrap_pyfunction!(st::st_strategy_payload, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_finite, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_safe_divide, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_raw_number, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_for_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_ratios_for_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_generic_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_bank_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_reit_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_ratios_by_period, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_compute_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bm_compute_metrics_inner, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bv_pct_yield, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bv_compute_market_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bv_score_value, m)?)?;
    m.add_function(wrap_pyfunction!(bm::bv_build_value_frame, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_evaluate_gates, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_confidence_factor, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_metrics_to_frame, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_score_quality, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_combine, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_blend_scores, m)?)?;
    m.add_function(wrap_pyfunction!(br::br_rerank, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_resolve_universe, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_infer_model_from_facts, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_collect_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_fetch_market_data, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_apply_price_quality_gate, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_exclusion_records, m)?)?;
    m.add_function(wrap_pyfunction!(bp::bp_run, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_db_path, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_store, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_span_days, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_is_annual_duration, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_parse_company_facts, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_parse_company_quarterly_facts, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_canonical_period_ends, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_apply_canonical_ends, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_median, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_filing_basis_factors, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_derive_quarterly_series, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_ingest_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_ingest_company, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_ingest_company_quarterly, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_resolve_concept, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_concept_values, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_concept_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_values_by_filing, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_filed_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_split_consistent_series, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_revisions, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_all_chain_tags, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_tags_for, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_quarterly_basis_factors, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_spans_on_latest_basis, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_split_adjustment_factors, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_apply_split_adjustment, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_concept_labels, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_frame_from_concepts, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_statements, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_quarterly_concept_values, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_quarterly_statements, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_add_derived_rows, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_period_stamps, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_write_row, m)?)?;
    m.add_function(wrap_pyfunction!(ep::ep_get_sector_concepts, m)?)?;
    m.add_class::<ep::AsOf>()?;
    m.add_class::<ep::BulkDocuments>()?;
    m.add_function(wrap_pyfunction!(fr::fr_get_statement_value, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_extract_fcf_from_statement, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_is_level_row, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_is_consecutive_window, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_to_trailing_twelve_months, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_price_on_or_before, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_opt_number, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_scalar_projection, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_key_ratios_python, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_numeric_statement, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_period_columns, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_first_column, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_key_ratios_timeseries, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_historical_mean_multiples, m)?)?;
    m.add_function(wrap_pyfunction!(
        fr::fr_load_price_history_for_multiples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fr::fr_current_valuation_ratios, m)?)?;
    m.add_function(wrap_pyfunction!(fr::fr_calculate_wacc, m)?)?;
    m.add_class::<fr::StatementLookup>()?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_estimate_growth_rate, m)?)?;
    m.add_function(wrap_pyfunction!(
        fr_growth::fr_historical_growth_regression,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_blended_growth_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_enrich_ticker_info, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_estimate_fcf_margin, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_through_cycle_fcf_margin, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_normalized_base_fcf, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_generate_mc_stats, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_dcf, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_dcfo, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_dni, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_ddm, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_mean_pe, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_peg, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_mean_pb, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_mean_ps, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_psg, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_graham, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_epv, m)?)?;
    m.add_function(wrap_pyfunction!(fr_growth::fr_mc_lynch, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_dcf, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_graham, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_ddm, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_lynch, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_epv, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_dcfo, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_dni, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_mean_pe, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_peg, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_mean_pb, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_mean_ps, m)?)?;
    m.add_function(wrap_pyfunction!(fr_models::fr_iv_psg, m)?)?;
    m.add_function(wrap_pyfunction!(
        fr_blend::fr_recommend_best_valuation_method,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fr_blend::fr_stable_mc_seed, m)?)?;
    m.add_function(wrap_pyfunction!(fr_blend::fr_blend_profile, m)?)?;
    m.add_function(wrap_pyfunction!(fr_blend::fr_model_reliability, m)?)?;
    m.add_function(wrap_pyfunction!(fr_blend::fr_ddm_describes_the_company, m)?)?;
    m.add_function(wrap_pyfunction!(fr_blend::fr_blend_intrinsic_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        fr_blend::fr_comprehensive_intrinsic_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        fr_blend::fr_intrinsic_value_for_symbol,
        m
    )?)?;
    m.add_class::<fr_blend::DeterministicDraws>()?;
    m.add_function(wrap_pyfunction!(fu::fu_infer_periods_per_year, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_get_file_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_currency_symbol_from_code, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_dividend_details, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_robust_dividend_yield, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_is_cash_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_is_external_flow_row, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_external_flow_mask, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_irr_date_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_npv, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_calculate_irr, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_numeric_column, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_commission_column, m)?)?;
    m.add_function(wrap_pyfunction!(fu::fu_cell_text, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_values, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_drawdown_series, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_sortino_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_beta, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_returns, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_all_risk_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(rm::rm_benchmark_scoreboard, m)?)?;
    m.add_function(wrap_pyfunction!(
        fu_flows::fu_cash_flows_for_symbol_account,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fu_flows::fu_cash_flows_for_mwr, m)?)?;
    m.add_class::<fu_flows::CashFlowLedger>()?;
    m.add_function(wrap_pyfunction!(fu_fx::fu_conversion_rate, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fx::fu_price_table, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fx::fu_build_price_table, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fx::fu_historical_price, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fx::fu_rate_via_usd_bridge, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_map_to_yf_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_safe_sum, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_format_currency_value, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_format_percentage_value, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_format_large_number_display, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_format_integer_with_commas, m)?)?;
    m.add_function(wrap_pyfunction!(fu_fmt::fu_format_float_with_commas, m)?)?;
    Ok(())
}
