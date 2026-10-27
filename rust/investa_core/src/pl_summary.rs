//! `portfolio_logic.calculate_portfolio_summary`: the current summary, from
//! the account filter (with the transfer-source backfill) through the
//! holdings pass, the quotes, the FIFO lots and gains, the rows and the
//! totals, to the dividend and realized-gain overrides and the status line.
//!
//! Ported line for line from `src/portfolio_logic.py` (frozen in
//! `tests/reference/portfolio_logic.py`). It orchestrates Python objects (the
//! market provider, pandas frames, the FIFO cache under its lock), so every
//! call and every figure is Python's own, in the original's order. Siblings
//! and module state (`_FIFO_CACHE`, `SHORTABLE_SYMBOLS`, the analyzer
//! functions it imported) are looked up on `portfolio_logic` at call time, so
//! a patched one is the one used.
//!
//! One difference cannot be avoided: the two `logging.error(traceback
//! .format_exc())` lines print the caught exception's traceback, which no
//! longer has a frame for this function, now that it is not Python.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::{clean_currencies, kwargs, pl};
use crate::pyhelp::{es, log, pystr, setitem, unpack};

/// `"; ".join(status_parts)`
fn joined(parts: &[String]) -> String {
    parts.join("; ")
}

/// `"".join(traceback.format_exception(e))`, what `traceback.format_exc()`
/// printed inside the original's `except`.
fn format_exc(py: Python<'_>, e: &PyErr) -> PyResult<String> {
    let lines = py
        .import("traceback")?
        .call_method1("format_exception", (e.value(py),))?;
    pystr(py, "").call_method1("join", (lines,))?.extract()
}

/// The per-account defaults the overrides create for a closed account.
fn closed_account_metrics(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for k in [
        "total_realized_gain_display",
        "total_unrealized_gain_display",
        "total_dividends_display",
        "total_commissions_display",
        "total_taxes_display",
        "total_gain_display",
        "total_buy_cost_display",
        "total_return_pct",
        "market_value",
    ] {
        d.set_item(k, 0.0)?;
    }
    Ok(d)
}

/// `metrics[pct_key] = gain / cost * 100.0`, or 0.0 for no gain on no money.
fn set_return_pct<'py>(
    o: &Ops<'py>,
    metrics: &Bound<'py, PyAny>,
    pct_key: &str,
    gain: &Bound<'py, PyAny>,
    cost: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if o.abs_gt(cost, 1e-9)? {
        metrics.set_item(pct_key, gain.div(cost)?.mul(100.0)?)?;
    } else if o.abs(gain)?.le(1e-9)? {
        metrics.set_item(pct_key, 0.0)?;
    }
    Ok(())
}

type SummaryOut<'py> = Bound<'py, PyTuple>;

/// `calculate_portfolio_summary`: (overall metrics, summary frame, holdings,
/// per-account metrics, ignored indices, ignored reasons, status).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pl_portfolio_summary<'py>(
    py: Python<'py>,
    all_transactions_df_cleaned: &Bound<'py, PyAny>,
    original_transactions_df_for_ignored: &Bound<'py, PyAny>,
    ignored_indices_from_load: &Bound<'py, PyAny>,
    ignored_reasons_from_load: &Bound<'py, PyAny>,
    fmp_api_key: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    show_closed_positions: &Bound<'py, PyAny>,
    cache_file_path: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    manual_overrides_dict: &Bound<'py, PyAny>,
    user_symbol_map: &Bound<'py, PyAny>,
    user_excluded_symbols: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    market_provider: &Bound<'py, PyAny>,
    calc_method: &Bound<'py, PyAny>,
    account_interest_rates: &Bound<'py, PyAny>,
    interest_free_thresholds: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
    db_mtime: &Bound<'py, PyAny>,
) -> PyResult<SummaryOut<'py>> {
    let _ = (
        original_transactions_df_for_ignored,
        fmp_api_key,
        account_currency_map,
        calc_method,
    );
    let m = pl(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let b = py.import("builtins")?;
    let dc = o.fmt(display_currency)?;
    let col = |label: &str| format!("{label} ({dc})");
    let none = || py.None().into_bound(py);
    let all_df = all_transactions_df_cleaned;
    let has_col =
        |df: &Bound<'py, PyAny>, c: &str| -> PyResult<bool> { df.getattr("columns")?.contains(c) };

    log(
        py,
        "info",
        format!("Starting Portfolio Summary Calculation (Display: {dc})"),
    )?;
    let start_time = m.getattr("time")?.call_method0("time")?;
    let mut has_errors = false;
    let mut has_warnings = false;
    let mut status_parts: Vec<String> = Vec::new();
    let report_date = m.getattr("get_latest_trading_date")?.call0()?;

    let combined_ignored_indices = if ignored_indices_from_load.is_truthy()? {
        ignored_indices_from_load.call_method0("copy")?
    } else {
        PySet::empty(py)?.into_any()
    };
    let combined_ignored_reasons = if ignored_reasons_from_load.is_truthy()? {
        ignored_reasons_from_load.call_method0("copy")?
    } else {
        PyDict::new(py).into_any()
    };
    let manual_overrides = if manual_overrides_dict.is_none() {
        PyDict::new(py).into_any()
    } else {
        manual_overrides_dict.clone()
    };
    let symbol_map = if user_symbol_map.is_none() {
        PyDict::new(py).into_any()
    } else {
        user_symbol_map.clone()
    };
    let excluded = if user_excluded_symbols.is_none() {
        PySet::empty(py)?.into_any()
    } else {
        user_excluded_symbols.clone()
    };

    let defaults =
        |available: &Bound<'py, PyAny>, empty_case: bool| -> PyResult<Bound<'py, PyAny>> {
            let kw = PyDict::new(py);
            kw.set_item("is_empty_data_case", empty_case)?;
            m.getattr("get_default_metrics_dict")?
                .call((display_currency, &report_date, available), Some(&kw))
        };
    let early = |metrics: Bound<'py, PyAny>, status: String| -> PyResult<SummaryOut<'py>> {
        PyTuple::new(
            py,
            [
                metrics,
                none(),
                PyDict::new(py).into_any(),
                none(),
                combined_ignored_indices.clone(),
                combined_ignored_reasons.clone(),
                pystr(py, &status),
            ],
        )
    };

    if !m.getattr("MARKET_PROVIDER_AVAILABLE")?.is_truthy()?
        || !m.getattr("ANALYZER_FUNCTIONS_AVAILABLE")?.is_truthy()?
    {
        let msg = "Error: Critical dependencies (MarketDataProvider or Analyzer Functions) not available.";
        log(py, "error", msg)?;
        return early(
            defaults(PyList::empty(py).as_any(), false)?,
            format!("Finished with Errors [{msg}]"),
        );
    }

    // --- 1. The pre-loaded transactions ---
    if all_df.is_none() || all_df.getattr("empty")?.is_truthy()? {
        log(
            py,
            "error",
            "Error: No transaction data provided or data is empty.",
        )?;
        status_parts.insert(0, "No Data".into());
        return early(
            defaults(PyList::empty(py).as_any(), true)?,
            format!("Finished with Errors [{}]", joined(&status_parts)),
        );
    }
    if !has_col(all_df, "original_index")? {
        log(
            py,
            "warning",
            "'original_index' column not found in provided DataFrame. Adding it from DataFrame index.",
        )?;
        setitem(all_df, "original_index", all_df.getattr("index")?)?;
        has_warnings = true;
        status_parts.push("original_index missing".into());
    }
    let mut available_accounts = PyList::empty(py).into_any();
    if !all_df.is_none() && has_col(all_df, "Account")? && !all_df.getattr("empty")?.is_truthy()? {
        available_accounts = b.getattr("sorted")?.call1((b
            .getattr("list")?
            .call1((all_df.get_item("Account")?.call_method0("unique")?,))?,))?;
    }

    // --- 2. Filter the transactions ---
    let filtered: Bound<'py, PyAny>;
    let filter_desc: String;
    if !include_accounts.is_truthy()? {
        filtered = all_df.call_method0("copy")?;
        filter_desc = "All Accounts".into();
    } else if include_accounts.is_instance_of::<PyList>() {
        if !has_col(all_df, "Account")? {
            log(
                py,
                "error",
                "CRITICAL: 'Account' column missing in provided transaction data. Cannot filter.",
            )?;
            status_parts.push("Filter Error: No Account Column".into());
            return early(
                defaults(&available_accounts, false)?,
                format!("Finished with Errors [{}]", joined(&status_parts)),
            );
        }
        let available_in_df = b
            .getattr("set")?
            .call1((all_df.get_item("Account")?.call_method0("unique")?,))?;
        if has_col(all_df, "To Account")? {
            available_in_df.call_method1(
                "update",
                (all_df
                    .get_item("To Account")?
                    .call_method0("dropna")?
                    .call_method0("unique")?,),
            )?;
        }
        let valid_include = PyList::empty(py);
        for acc in include_accounts.try_iter()? {
            let acc = acc?;
            if available_in_df.contains(&acc)? {
                valid_include.append(acc)?;
            }
        }
        if valid_include.is_truthy()? {
            // A selected account that received a transfer brings the
            // transferred asset's history from its source account, so cost
            // basis survives the transfer.
            let index_kw = kwargs(py, &[("index", &all_df.getattr("index")?)])?;
            let series_false = || pd.getattr("Series")?.call((false,), Some(&index_kw));
            let from_mask = all_df
                .get_item("Account")?
                .call_method1("isin", (&valid_include,))?;
            let mut to_mask = series_false()?;
            if has_col(all_df, "To Account")? {
                to_mask = all_df
                    .get_item("To Account")?
                    .call_method1("isin", (&valid_include,))?
                    .call_method1("fillna", (false,))?;
            }
            let mut preceding = series_false()?;
            let transfers_in = all_df.get_item(&to_mask)?;
            if !transfers_in.getattr("empty")?.is_truthy()? {
                let events = transfers_in
                    .get_item(PyList::new(py, ["Symbol", "Account", "Date"])?)?
                    .call_method0("drop_duplicates")?;
                let masks = PyList::empty(py);
                for item in events.call_method0("iterrows")?.try_iter()? {
                    let row = unpack(&item?, 2)?;
                    let ev = &row[1];
                    let symbol = ev.get_item("Symbol")?;
                    let from_account = ev.get_item("Account")?;
                    let date = ev.get_item("Date")?;
                    let mask = all_df
                        .get_item("Symbol")?
                        .rich_compare(&symbol, CompareOp::Eq)?
                        .bitand(
                            all_df
                                .get_item("Account")?
                                .rich_compare(&from_account, CompareOp::Eq)?,
                        )?
                        .bitand(
                            all_df
                                .get_item("Date")?
                                .rich_compare(&date, CompareOp::Le)?,
                        )?;
                    masks.append(mask)?;
                }
                if masks.is_truthy()? {
                    let kw = kwargs(py, &[("axis", &1i64.into_pyobject(py)?.into_any())])?;
                    preceding = pd
                        .call_method("concat", (&masks,), Some(&kw))?
                        .call_method("any", (), Some(&kw))?;
                }
            }
            // Splits and "All Accounts" rows apply whatever account recorded them.
            let lower = |c: &str| -> PyResult<Bound<'py, PyAny>> {
                all_df
                    .get_item(c)?
                    .call_method1("astype", (&o.str_t,))?
                    .getattr("str")?
                    .call_method0("lower")
            };
            let split_mask = lower("Type")?
                .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?;
            let all_accounts_mask =
                lower("Account")?.rich_compare("all accounts", CompareOp::Eq)?;
            let combined = from_mask
                .bitor(&to_mask)?
                .bitor(&preceding)?
                .bitor(&split_mask)?
                .bitor(&all_accounts_mask)?;
            filtered = all_df.get_item(combined)?.call_method0("copy")?;
            let sorted = b.getattr("sorted")?.call1((&valid_include,))?;
            filter_desc = format!(
                "Accounts: {}",
                pystr(py, ", ")
                    .call_method1("join", (sorted,))?
                    .extract::<String>()?
            );
        } else {
            let kw = kwargs(py, &[("columns", &all_df.getattr("columns")?)])?;
            filtered = pd.getattr("DataFrame")?.call((), Some(&kw))?;
            filter_desc = "No Valid Accounts Included".into();
            status_parts.push("No valid accounts after filter".into());
            has_warnings = true;
        }
    } else {
        filtered = all_df.call_method0("copy")?;
        filter_desc = "All Accounts (Invalid Filter Type)".into();
        status_parts.push("Invalid include_accounts filter type".into());
        has_warnings = true;
    }
    log(
        py,
        "info",
        format!("Processing transactions for scope: {filter_desc}"),
    )?;
    if filtered.getattr("empty")?.is_truthy()?
        && (!has_warnings
            || !status_parts
                .iter()
                .any(|p| p == "No valid accounts after filter"))
    {
        log(
            py,
            "info",
            "No transactions after filtering for the selected scope.",
        )?;
    }

    // --- Historical FX for the holdings pass ---
    let fx_prep = m.getattr("_prepare_historical_fx_rates")?.call1((
        &filtered,
        all_df,
        display_currency,
        default_currency,
        market_provider,
        cache_file_path,
    ))?;
    let fx_prep = unpack(&fx_prep, 3)?;
    let (hist_fx_processing, hist_fx_usd) = (fx_prep[0].clone(), fx_prep[1].clone());
    if fx_prep[2].is_truthy()? {
        has_warnings = true;
    }

    // --- 3. The holdings pass ---
    let shortable = m.getattr("SHORTABLE_SYMBOLS")?;
    let processed = m.getattr("_process_transactions_to_holdings")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("transactions_df", &filtered),
                ("default_currency", default_currency),
                ("shortable_symbols", &shortable),
                ("historical_fx_lookup", &hist_fx_processing),
                ("display_currency_for_hist_fx", display_currency),
                ("report_date", &report_date),
                ("account_cash_mode_map", account_cash_mode_map),
            ],
        )?),
    )?;
    let processed = unpack(&processed, 9)?;
    let holdings_in = processed[0].clone();
    let taxes_local = processed[4].clone();
    let ignored_indices_proc = processed[5].clone();
    let ignored_reasons_proc = processed[6].clone();
    let transfer_costs = processed[7].clone();
    let warn_proc = processed[8].clone();

    // The IRR flows see a transfer at the cost it carried.
    let tx_for_summary = filtered.call_method0("copy")?;
    if transfer_costs.is_truthy()? {
        log(
            py,
            "debug",
            format!(
                "Patching {} transfer transactions with cost-based prices for IRR...",
                transfer_costs.len()?
            ),
        )?;
        let cost_series = pd.getattr("Series")?.call1((&transfer_costs,))?;
        let _mask = tx_for_summary
            .get_item("original_index")?
            .call_method1("isin", (transfer_costs.call_method0("keys")?,))?;
        let temp_df = tx_for_summary.call_method1("set_index", ("original_index",))?;
        let update = PyDict::new(py);
        update.set_item("Price/Share", &cost_series)?;
        temp_df.call_method1("update", (pd.getattr("DataFrame")?.call1((update,))?,))?;
        for item in transfer_costs.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let rows = tx_for_summary
                .get_item("original_index")?
                .rich_compare(&kv[0], CompareOp::Eq)?;
            tx_for_summary
                .getattr("loc")?
                .set_item(PyTuple::new(py, [rows, pystr(py, "Price/Share")])?, &kv[1])?;
        }
    }
    combined_ignored_indices.call_method1("update", (&ignored_indices_proc,))?;
    combined_ignored_reasons.call_method1("update", (&ignored_reasons_proc,))?;
    if warn_proc.is_truthy()? {
        has_warnings = true;
    }
    if ignored_reasons_proc.is_truthy()? {
        status_parts.push(format!(
            "Processing Issues: {}",
            ignored_reasons_proc.len()?
        ));
    }

    // --- 5. Current market data ---
    let held_symbols = PySet::empty(py)?;
    for key in holdings_in.call_method0("keys")?.try_iter()? {
        held_symbols.add(key?.get_item(0)?)?;
    }
    let all_stock_symbols = b.getattr("list")?.call1((&held_symbols,))?;
    let required = b
        .getattr("set")?
        .call1((PyList::new(py, [display_currency, default_currency])?,))?;
    for data in holdings_in.call_method0("values")?.try_iter()? {
        required.call_method1(
            "add",
            (o.get(&data?, "local_currency", default_currency.clone())?,),
        )?;
    }
    if all_df.is_instance(&pd.getattr("DataFrame")?)? && has_col(all_df, "Local Currency")? {
        for curr in all_df
            .get_item("Local Currency")?
            .call_method0("dropna")?
            .call_method0("unique")?
            .try_iter()?
        {
            let curr = curr?;
            if curr.is_instance(&o.str_t)? && curr.call_method0("strip")?.len()? == 3 {
                required
                    .call_method1("add", (curr.call_method0("strip")?.call_method0("upper")?,))?;
            }
        }
    }
    required.call_method1("discard", (py.None(),))?;
    let required = clean_currencies(&o, &required)?;

    let provider = if market_provider.is_none() {
        m.getattr("MarketDataProvider")?.call(
            (),
            Some(&kwargs(py, &[("current_cache_file", cache_file_path)])?),
        )?
    } else {
        market_provider.clone()
    };
    let quotes = provider.call_method(
        "get_current_quotes",
        (),
        Some(&kwargs(
            py,
            &[
                ("internal_stock_symbols", &all_stock_symbols),
                ("required_currencies", required.as_any()),
                ("user_symbol_map", &symbol_map),
                ("user_excluded_symbols", &excluded),
            ],
        )?),
    )?;
    let quotes = unpack(&quotes, 5)?;
    let (stock_data, fx_rates, fx_prev) = (quotes[0].clone(), quotes[1].clone(), quotes[2].clone());
    let published_navs = m
        .getattr("published_nav_prices")?
        .call1((b.getattr("set")?.call1((&all_stock_symbols,))?,))?;
    if quotes[3].is_truthy()? {
        has_errors = true;
        status_parts.push("Fetch Failed Critically".into());
    }
    if quotes[4].is_truthy()? {
        has_warnings = true;
        status_parts.push("Fetch Warnings".into());
    }
    if has_errors && status_parts.iter().any(|p| p == "Fetch Failed Critically") {
        log(
            py,
            "error",
            "WARNING: Error: Price/FX fetch failed critically via MarketDataProvider. Proceeding with partial data/fallbacks.",
        )?;
    }

    // --- 6. FIFO realized gains and lots (the capital-gains view's figures) ---
    let mut fifo_df = pd.getattr("DataFrame")?.call0()?;
    let mut open_lots = PyDict::new(py).into_any();
    let fifo = (|| -> PyResult<()> {
        log(
            py,
            "info",
            "Calculating FIFO Realized Gains & Lots for Dashboard...",
        )?;
        let fifo_input = all_df.call_method0("copy")?;
        let by: Vec<&str> = if has_col(&fifo_input, "original_index")? {
            vec!["Date", "original_index"]
        } else {
            vec!["Date"]
        };
        let kw = PyDict::new(py);
        kw.set_item("by", PyList::new(py, by)?)?;
        kw.set_item("inplace", true)?;
        fifo_input.call_method("sort_values", (), Some(&kw))?;

        // Cached on the transactions themselves when the caller wants caching.
        let fingerprint = if db_mtime.gt(0)? {
            m.getattr("_transactions_fingerprint")?
                .call1((&fifo_input,))?
        } else {
            none()
        };
        let cache_key = PyTuple::new(
            py,
            [
                fingerprint.clone(),
                display_currency.clone(),
                default_currency.clone(),
            ],
        )?;
        let cache = m.getattr("_FIFO_CACHE")?;
        let cached = with1(py, &m.getattr("_FIFO_CACHE_LOCK")?, |_| {
            let cached = if fingerprint.is_truthy()? {
                cache.call_method1("get", (&cache_key,))?
            } else {
                none()
            };
            if !cached.is_none() {
                cache.call_method1("move_to_end", (&cache_key,))?;
            }
            Ok(cached)
        })?
        .unwrap_or_else(none);
        if !cached.is_none() {
            log(py, "info", "Using cached FIFO Realized Gains & Lots...")?;
            let pair = unpack(&cached, 2)?;
            fifo_df = pair[0].clone();
            open_lots = pair[1].clone();
        } else {
            let result = m.getattr("calculate_fifo_lots_and_gains")?.call(
                (),
                Some(&kwargs(
                    py,
                    &[
                        ("transactions_df", &fifo_input),
                        ("display_currency", display_currency),
                        ("historical_fx_yf", &hist_fx_usd),
                        ("default_currency", default_currency),
                        ("shortable_symbols", &m.getattr("SHORTABLE_SYMBOLS")?),
                        (
                            "stock_quantity_close_tolerance",
                            &m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?,
                        ),
                        ("current_fx_rates_vs_usd", &fx_rates),
                    ],
                )?),
            )?;
            let pair = unpack(&result, 2)?;
            fifo_df = pair[0].clone();
            open_lots = pair[1].clone();
            if fingerprint.is_truthy()? {
                with1(py, &m.getattr("_FIFO_CACHE_LOCK")?, |_| {
                    cache.set_item(&cache_key, PyTuple::new(py, [&fifo_df, &open_lots])?)?;
                    while cache.len()? > m.getattr("_FIFO_CACHE_MAX")?.extract::<usize>()? {
                        let kw = PyDict::new(py);
                        kw.set_item("last", false)?;
                        cache.call_method("popitem", (), Some(&kw))?;
                    }
                    Ok(())
                })?;
            }
        }

        log(
            py,
            "info",
            format!("FIFO DF Shape: {}", o.fmt(&fifo_df.getattr("shape")?)?),
        )?;
        log(py, "info", format!("Open Lots Count: {}", open_lots.len()?))?;
        let fifo_empty = fifo_df.getattr("empty")?.is_truthy()?;
        if !fifo_empty && has_col(&fifo_df, "Realized Gain (Display)")? {
            let nans = fifo_df
                .get_item("Realized Gain (Display)")?
                .call_method0("isna")?
                .call_method0("sum")?;
            log(
                py,
                "info",
                format!("NaNs in Realized Gain (Display): {}", o.fmt(&nans)?),
            )?;
        }

        let gains_agg = if !fifo_empty {
            let kw = PyDict::new(py);
            kw.set_item("orient", "index")?;
            fifo_df
                .call_method1("groupby", (PyList::new(py, ["Symbol", "Account"])?,))?
                .get_item(PyList::new(
                    py,
                    ["Realized Gain (Display)", "Realized Gain (Local)"],
                )?)?
                .call_method0("sum")?
                .call_method("to_dict", (), Some(&kw))?
        } else {
            PyDict::new(py).into_any()
        };

        let date_t = m.getattr("date")?;
        let datetime_t = m.getattr("datetime")?;
        for item in holdings_in.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let key = unpack(&kv[0], 2)?;
            let (sym, acct, holding) = (&key[0], &key[1], &kv[1]);
            let lookup_key = PyTuple::new(
                py,
                [
                    o.str_t
                        .call1((sym,))?
                        .call_method0("upper")?
                        .call_method0("strip")?,
                    o.str_t
                        .call1((acct,))?
                        .call_method0("upper")?
                        .call_method0("strip")?,
                ],
            )?;

            // 1. The FIFO realized gains replace the average-cost ones.
            if gains_agg.contains(&lookup_key)? {
                let agg = gains_agg.get_item(&lookup_key)?;
                let display = o.get_f(&agg, "Realized Gain (Display)", 0.0)?;
                let local = o.get_f(&agg, "Realized Gain (Local)", 0.0)?;
                holding.set_item("realized_gain_display", &display)?;
                holding.set_item("realized_gain_local", &local)?;
                log(
                    py,
                    "debug",
                    format!(
                        "Overrode realized gain for {}/{} with FIFO value: {}",
                        o.fmt(sym)?,
                        o.fmt(acct)?,
                        o.fmt(&display)?
                    ),
                )?;
            } else {
                let _is_shortable = shortable.contains(sym)?;
                let _is_cash = sym.eq(m.getattr("CASH_SYMBOL_CSV")?)? || sym.eq("$CASH")?;
                let avg_cost_gain = o.get_f(holding, "realized_gain_display", 0.0)?;
                o.abs_gt(&avg_cost_gain, 1e-9)?;
            }

            // 2. The open lots, valued at today's price.
            if open_lots.contains(&lookup_key)? {
                let lots = open_lots.get_item(&lookup_key)?;
                let processed_lots = PyList::empty(py);
                let mut current_price = o.fl(0.0);
                // A published NAV outranks the override, as in the summary rows.
                let nav = published_navs.call_method1("get", (sym,))?;
                if !nav.is_none() {
                    current_price = nav.get_item(1)?;
                }
                let manual = if manual_overrides.is_truthy()? {
                    manual_overrides.call_method1("get", (sym,))?
                } else {
                    none()
                };
                if current_price.le(1e-9)? && manual.is_instance_of::<PyDict>() {
                    let manual_price = manual.call_method1("get", ("price",))?;
                    if !manual_price.is_none() && o.notna(&manual_price)? {
                        match o.float(&manual_price) {
                            Ok(v) => {
                                if v > 1e-9 {
                                    current_price = o.fl(v);
                                }
                            }
                            Err(e)
                                if e.is_instance_of::<PyValueError>(py)
                                    || e.is_instance_of::<PyTypeError>(py) => {}
                            Err(e) => return Err(e),
                        }
                    }
                }
                if current_price.le(1e-9)? && stock_data.contains(sym)? {
                    let fetched = o.get_f(&stock_data.get_item(sym)?, "price", 0.0)?;
                    if o.notna(&fetched)? {
                        current_price = fetched;
                    }
                }
                let local_curr = o.get(holding, "local_currency", default_currency.clone())?;
                let mut fx_now = m.getattr("get_conversion_rate")?.call1((
                    &local_curr,
                    display_currency,
                    &fx_rates,
                ))?;
                if o.isna(&fx_now)? {
                    fx_now = o.fl(0.0);
                }
                for lot in lots.try_iter()? {
                    let lot = lot?;
                    let qty = lot.get_item("qty")?;
                    let purchase_date = lot.get_item("purchase_date")?;
                    let cost_local = lot.get_item("cost_per_share_local_net")?;
                    let mut purchase_fx = lot.get_item("purchase_fx_to_display")?;
                    if o.isna(&purchase_fx)? {
                        purchase_fx = o.fl(0.0);
                    }
                    let cost_basis = qty.mul(&cost_local)?.mul(&purchase_fx)?;
                    let market_value = qty.mul(&current_price)?.mul(&fx_now)?;
                    let gain = market_value.sub(&cost_basis)?;
                    let gain_pct = if o.abs_gt(&cost_basis, 1e-9)? {
                        gain.div(&cost_basis)?.mul(100)?
                    } else {
                        o.fl(0.0)
                    };
                    let date_str = if purchase_date.is_instance(&date_t)?
                        || purchase_date.is_instance(&datetime_t)?
                    {
                        purchase_date.call_method1("strftime", ("%Y-%m-%d",))?
                    } else {
                        o.str_t.call1((&purchase_date,))?
                    };
                    let d = PyDict::new(py);
                    d.set_item("Date", date_str)?;
                    d.set_item("Quantity", &qty)?;
                    d.set_item("Cost Basis", &cost_basis)?;
                    d.set_item("Market Value", &market_value)?;
                    d.set_item("Unreal. Gain", &gain)?;
                    d.set_item("Unreal. Gain %", gain_pct)?;
                    d.set_item("purchase_fx", &purchase_fx)?;
                    d.set_item("cost_local", &cost_local)?;
                    processed_lots.append(d)?;
                }
                holding.set_item("lots", processed_lots)?;
            }
        }
        Ok(())
    })();
    match fifo {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Error calculating FIFO realized gains & lots for Dashboard: {}",
                    es(py, &e)
                ),
            )?;
            log(py, "error", format_exc(py, &e)?)?;
        }
        Err(e) => return Err(e),
    }

    // Manual prices for the rows.
    let manual_prices = PyDict::new(py);
    if manual_overrides.is_truthy()? {
        for item in manual_overrides.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let (symbol_key, values) = (&kv[0], &kv[1]);
            if values.is_instance_of::<PyDict>() && values.contains("price")? {
                let price = values.get_item("price")?;
                if !price.is_none() && o.notna(&price)? {
                    match o.float(&price) {
                        Ok(v) => manual_prices.set_item(symbol_key, v)?,
                        Err(e)
                            if e.is_instance_of::<PyValueError>(py)
                                || e.is_instance_of::<PyTypeError>(py) =>
                        {
                            log(
                                py,
                                "warning",
                                format!(
                                    "Could not convert manual price for {} to float: {}",
                                    o.fmt(symbol_key)?,
                                    o.fmt(&price)?
                                ),
                            )?;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }

    let or_empty = |v: &Bound<'py, PyAny>| -> Bound<'py, PyAny> {
        if v.is_none() {
            PyDict::new(py).into_any()
        } else {
            v.clone()
        }
    };
    let built = m.getattr("_build_summary_rows")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("holdings", &holdings_in),
                ("current_stock_data", &or_empty(&stock_data)),
                ("current_fx_rates_vs_usd", &or_empty(&fx_rates)),
                ("current_fx_prev_close_vs_usd", &or_empty(&fx_prev)),
                ("display_currency", display_currency),
                ("default_currency", default_currency),
                ("transactions_df", &tx_for_summary),
                ("report_date", &report_date),
                ("shortable_symbols", &m.getattr("SHORTABLE_SYMBOLS")?),
                ("user_excluded_symbols", &excluded),
                ("user_symbol_map", &symbol_map),
                ("manual_prices_dict", manual_prices.as_any()),
                ("published_navs", &published_navs),
                ("account_interest_rates", account_interest_rates),
                ("interest_free_thresholds", interest_free_thresholds),
                ("include_accounts", include_accounts),
            ],
        )?),
    )?;
    let built = unpack(&built, 4)?;
    let mut rows = built[0].clone();
    if built[2].is_truthy()? {
        has_errors = true;
    }
    if built[3].is_truthy()? {
        has_warnings = true;
    }
    if has_errors {
        log(
            py,
            "error",
            "Error: Failed critically during summary row building (likely FX).",
        )?;
        status_parts.push("Summary Build Error".into());
        let status = format!("Finished with Errors ({filter_desc})")
            + &if status_parts.is_empty() {
                String::new()
            } else {
                format!(" [{}]", joined(&status_parts))
            };
        return early(defaults(&available_accounts, false)?, status);
    } else if !rows.is_truthy()? && holdings_in.is_truthy()? {
        log(
            py,
            "warning",
            "Warning: Failed to generate any summary rows (FX or other issue during build).",
        )?;
        has_warnings = true;
        status_parts.push("Summary Build Generated No Rows".into());
    }

    // The lots viewer reads Title Case keys: copy them back onto the holdings.
    if rows.is_truthy()? && holdings_in.is_truthy()? {
        let upper_strip = |v: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            o.str_t
                .call1((v,))?
                .call_method0("upper")?
                .call_method0("strip")
        };
        let norm = PyDict::new(py);
        for k in holdings_in.call_method0("keys")?.try_iter()? {
            let k = k?;
            norm.set_item(
                PyTuple::new(
                    py,
                    [upper_strip(&k.get_item(0)?)?, upper_strip(&k.get_item(1)?)?],
                )?,
                &k,
            )?;
        }
        for row in rows.try_iter()? {
            let row = row?;
            let r_sym = row.call_method1("get", ("Symbol",))?;
            let r_acct = row.call_method1("get", ("Account",))?;
            if r_sym.is_truthy()? && r_acct.is_truthy()? {
                let key = PyTuple::new(py, [upper_strip(&r_sym)?, upper_strip(&r_acct)?])?;
                let original = norm.call_method1("get", (key,))?;
                if original.is_truthy()? && holdings_in.contains(&original)? {
                    let h = holdings_in.get_item(&original)?;
                    h.set_item("Symbol", &r_sym)?;
                    h.set_item("Account", &r_acct)?;
                    h.set_item("Quantity", row.call_method1("get", ("Quantity", 0))?)?;
                    h.set_item("Avg Cost", row.call_method1("get", (col("Avg Cost"), 0))?)?;
                    h.set_item(
                        "Market Value",
                        row.call_method1("get", (col("Market Value"), 0))?,
                    )?;
                }
            }
        }
    }

    rows = m.getattr("_enrich_summary_rows_with_sector_geo")?.call1((
        &rows,
        &provider,
        &manual_overrides,
        &symbol_map,
        &excluded,
    ))?;

    // --- 7. The frame and the totals ---
    let summary_unfiltered: Bound<'py, PyAny>;
    let mut overall: Bound<'py, PyAny>;
    let mut account_metrics: Bound<'py, PyAny> = PyDict::new(py).into_any();
    if !rows.is_truthy()? {
        log(
            py,
            "warning",
            "Portfolio summary list is empty. Returning empty results for aggregates.",
        )?;
        has_warnings = true;
        let d = PyDict::new(py);
        for (k, v) in [
            ("market_value", 0.0),
            ("cost_basis_held", 0.0),
            ("unrealized_gain", 0.0),
            ("realized_gain", 0.0),
            ("dividends", 0.0),
            ("commissions", 0.0),
            ("taxes", 0.0),
            ("total_gain", 0.0),
            ("total_cost_invested", 0.0),
            ("total_buy_cost", 0.0),
            ("portfolio_mwr", f64::NAN),
            ("day_change_display", 0.0),
            ("day_change_percent", f64::NAN),
        ] {
            d.set_item(k, v)?;
        }
        d.set_item(
            "report_date",
            report_date.call_method1("strftime", ("%Y-%m-%d",))?,
        )?;
        d.set_item("display_currency", display_currency)?;
        d.set_item("cumulative_investment", 0.0)?;
        d.set_item("total_return_pct", f64::NAN)?;
        d.set_item("est_annual_income_display", 0.0)?;
        overall = d.into_any();
        summary_unfiltered = pd.getattr("DataFrame")?.call0()?;
    } else {
        let full = pd.getattr("DataFrame")?.call1((&rows,))?;
        let conversion = (|| -> PyResult<()> {
            let marker = format!("({dc})");
            let mut numeric: Vec<Bound<'py, PyAny>> = vec![pystr(py, "Quantity")];
            for c in full.getattr("columns")?.try_iter()? {
                let c = c?;
                if c.contains(&marker)? {
                    numeric.push(c);
                }
            }
            for c in [
                "Unreal. Gain %",
                "Total Return %",
                "IRR (%)",
                "Day Change %",
                "Div. Yield (Cost) %",
                "Div. Yield (Current) %",
            ] {
                numeric.push(pystr(py, c));
            }
            let coerce = kwargs(py, &[("errors", &pystr(py, "coerce"))])?;
            for c in &numeric {
                if full.getattr("columns")?.contains(c)? {
                    let converted =
                        pd.call_method("to_numeric", (full.get_item(c)?,), Some(&coerce))?;
                    setitem(&full, c, converted)?;
                }
            }
            Ok(())
        })();
        match conversion {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => {
                has_warnings = true;
                log(
                    py,
                    "warning",
                    format!(
                        "Warning: Error during numeric conversion of summary columns: {}",
                        es(py, &e)
                    ),
                )?;
            }
            Err(e) => return Err(e),
        }
        let kw = PyDict::new(py);
        kw.set_item(
            "by",
            PyList::new(py, ["Account".to_string(), col("Market Value")])?,
        )?;
        kw.set_item("ascending", PyList::new(py, [true, false])?)?;
        kw.set_item("na_position", "last")?;
        kw.set_item("inplace", true)?;
        match full.call_method("sort_values", (), Some(&kw)) {
            Ok(_) => {}
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                has_warnings = true;
                log(
                    py,
                    "warning",
                    "Warning: Could not sort summary DataFrame by Account/Market Value.",
                )?;
            }
            Err(e) => return Err(e),
        }

        let agg = m.getattr("_calculate_aggregate_metrics")?.call(
            (),
            Some(&kwargs(
                py,
                &[
                    ("full_summary_df", &full),
                    ("display_currency", display_currency),
                    ("report_date", &report_date),
                    ("fx_rates", &fx_rates),
                    ("include_accounts", include_accounts),
                    ("all_available_accounts", &available_accounts),
                    ("transactions_df", &tx_for_summary),
                    ("historical_fx_rates", &hist_fx_processing),
                ],
            )?),
        )?;
        let agg = unpack(&agg, 4)?;
        overall = if agg[0].is_instance_of::<PyDict>() {
            agg[0].clone()
        } else {
            PyDict::new(py).into_any()
        };
        account_metrics = if agg[1].is_instance_of::<PyDict>() {
            agg[1].clone()
        } else {
            PyDict::new(py).into_any()
        };
        if agg[2].is_truthy()? {
            has_errors = true;
        }
        if agg[3].is_truthy()? {
            has_warnings = true;
        }

        let mut price_source_warnings = false;
        if has_col(&full, "Price Source")? {
            let non_cash = full.get_item(
                full.get_item("Symbol")?
                    .rich_compare(m.getattr("CASH_SYMBOL_CSV")?, CompareOp::Ne)?,
            )?;
            if !non_cash.getattr("empty")?.is_truthy()? {
                let kw = PyDict::new(py);
                kw.set_item("case", false)?;
                kw.set_item("na", false)?;
                if non_cash
                    .get_item("Price Source")?
                    .getattr("str")?
                    .call_method(
                        "contains",
                        ("Fallback|Excluded|Invalid|Error|Zero",),
                        Some(&kw),
                    )?
                    .call_method0("any")?
                    .is_truthy()?
                {
                    price_source_warnings = true;
                }
            }
        }
        if price_source_warnings {
            has_warnings = true;
            status_parts.push("Fallback Prices Used".into());
        }
        summary_unfiltered = full;
    }

    // --- The dividends tab's total replaces the rows' ---
    let dividends = (|| -> PyResult<()> {
        let history = m.getattr("extract_dividend_history")?.call(
            (),
            Some(&kwargs(
                py,
                &[
                    ("all_transactions_df", &filtered),
                    ("display_currency", display_currency),
                    ("historical_fx_yf", &hist_fx_usd),
                    ("default_currency", default_currency),
                    ("include_accounts", include_accounts),
                ],
            )?),
        )?;
        if !history.getattr("empty")?.is_truthy()? {
            let total = history
                .get_item("DividendAmountDisplayCurrency")?
                .call_method0("sum")?;
            overall.set_item("dividends", &total)?;
            overall.set_item("total_dividends_display", &total)?;
            let unrealized = o.get_f(&overall, "unrealized_gain", 0.0)?;
            let realized = o.get_f(&overall, "realized_gain", 0.0)?;
            let commissions = o.get_f(&overall, "commissions", 0.0)?;
            let taxes = o.get_f(&overall, "taxes", 0.0)?;
            let new_total = realized
                .add(&unrealized)?
                .add(&total)?
                .sub(&commissions)?
                .sub(&taxes)?;
            overall.set_item("total_gain", &new_total)?;
            let buy_cost = o.get_f(&overall, "total_buy_cost", 0.0)?;
            set_return_pct(&o, &overall, "total_return_pct", &new_total, &buy_cost)?;

            let by_account = history
                .call_method1("groupby", ("Account",))?
                .get_item("DividendAmountDisplayCurrency")?
                .call_method0("sum")?
                .call_method0("to_dict")?;
            for item in by_account.call_method0("items")?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                let (acct, amount) = (&kv[0], &kv[1]);
                if !account_metrics.contains(acct)? {
                    account_metrics.set_item(acct, closed_account_metrics(py)?)?;
                }
                let metrics = account_metrics.get_item(acct)?;
                metrics.set_item("total_dividends_display", amount)?;
                let gain = o
                    .get_f(&metrics, "total_realized_gain_display", 0.0)?
                    .add(o.get_f(&metrics, "total_unrealized_gain_display", 0.0)?)?
                    .add(amount)?
                    .sub(o.get_f(&metrics, "total_commissions_display", 0.0)?)?
                    .sub(o.get_f(&metrics, "total_taxes_display", 0.0)?)?;
                metrics.set_item("total_gain_display", &gain)?;
                let cost = o.get_f(&metrics, "total_buy_cost_display", 0.0)?;
                set_return_pct(&o, &metrics, "total_return_pct", &gain, &cost)?;
            }
            log(
                py,
                "info",
                format!(
                    "Overrode Dashboard Total Dividends: {} (Legacy Extraction)",
                    o.fmt(&total)?
                ),
            )?;
        } else {
            log(
                py,
                "info",
                "Dividend extraction returned empty. Using default aggregation.",
            )?;
        }
        Ok(())
    })();
    match dividends {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!("Error overriding dividends: {}", es(py, &e)),
            )?;
            log(py, "error", format_exc(py, &e)?)?;
        }
        Err(e) => return Err(e),
    }

    // --- The FIFO totals replace the rows' realized gains ---
    if !fifo_df.getattr("empty")?.is_truthy()? {
        let r = (|| -> PyResult<()> {
            let filter_list =
                include_accounts.is_truthy()? && include_accounts.is_instance_of::<PyList>();
            let wanted = PyList::empty(py);
            if filter_list {
                for a in include_accounts.try_iter()? {
                    wanted.append(o.str_strip(&a?)?.call_method0("upper")?)?;
                }
            }
            let mut for_sum = fifo_df.call_method0("copy")?;
            if filter_list {
                for_sum = for_sum.get_item(
                    for_sum
                        .get_item("Account")?
                        .call_method1("isin", (&wanted,))?,
                )?;
            }
            let total = for_sum
                .get_item("Realized Gain (Display)")?
                .call_method0("sum")?;
            if include_accounts.is_none() {
                log(
                    py,
                    "info",
                    format!(
                        "debug_agg: All Accounts View. Total FIFO Gain: {}",
                        o.fmt(&total)?
                    ),
                )?;
                let kw = PyDict::new(py);
                kw.set_item("ascending", false)?;
                let breakdown = for_sum
                    .call_method1("groupby", ("Account",))?
                    .get_item("Realized Gain (Display)")?
                    .call_method0("sum")?
                    .call_method("sort_values", (), Some(&kw))?;
                log(
                    py,
                    "info",
                    format!(
                        "debug_agg: Breakdown by Account: {}",
                        o.fmt(&breakdown.call_method0("to_dict")?)?
                    ),
                )?;
            } else {
                log(
                    py,
                    "info",
                    format!(
                        "debug_agg: Filtered View ({}). Total FIFO Gain: {}",
                        o.fmt(include_accounts)?,
                        o.fmt(&total)?
                    ),
                )?;
            }
            let old_realized = o.get_f(&overall, "realized_gain", 0.0)?;
            overall.set_item("realized_gain", &total)?;
            overall.set_item("total_realized_gain_display", &total)?;
            let new_total = total
                .add(o.get_f(&overall, "unrealized_gain", 0.0)?)?
                .add(o.get_f(&overall, "dividends", 0.0)?)?
                .sub(o.get_f(&overall, "commissions", 0.0)?)?
                .sub(o.get_f(&overall, "taxes", 0.0)?)?;
            overall.set_item("total_gain", &new_total)?;
            let buy_cost = o.get_f(&overall, "total_buy_cost", 0.0)?;
            set_return_pct(&o, &overall, "total_return_pct", &new_total, &buy_cost)?;
            log(
                py,
                "info",
                format!(
                    "Overrode Dashboard Total Realized Gain: {} -> {} (FIFO Local)",
                    o.fmt(&old_realized)?,
                    o.fmt(&total)?
                ),
            )?;

            let by_account = fifo_df
                .call_method1("groupby", ("Account",))?
                .get_item("Realized Gain (Display)")?
                .call_method0("sum")?
                .call_method0("to_dict")?;
            for item in by_account.call_method0("items")?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                let (acct, gain) = (&kv[0], &kv[1]);
                if filter_list {
                    let wanted_now = PyList::empty(py);
                    for a in include_accounts.try_iter()? {
                        wanted_now.append(o.str_strip(&a?)?.call_method0("upper")?)?;
                    }
                    if !wanted_now.contains(o.str_strip(acct)?.call_method0("upper")?)? {
                        continue;
                    }
                }
                if !account_metrics.contains(acct)? {
                    account_metrics.set_item(acct, closed_account_metrics(py)?)?;
                }
                let metrics = account_metrics.get_item(acct)?;
                let _old = o.get_f(&metrics, "total_realized_gain_display", 0.0)?;
                metrics.set_item("total_realized_gain_display", gain)?;
                let new_gain = gain
                    .add(o.get_f(&metrics, "total_unrealized_gain_display", 0.0)?)?
                    .add(o.get_f(&metrics, "total_dividends_display", 0.0)?)?
                    .sub(o.get_f(&metrics, "total_commissions_display", 0.0)?)?
                    .sub(o.get_f(&metrics, "total_taxes_display", 0.0)?)?;
                metrics.set_item("total_gain_display", &new_gain)?;
                let cost = o.get_f(&metrics, "total_buy_cost_display", 0.0)?;
                set_return_pct(&o, &metrics, "total_return_pct", &new_gain, &cost)?;
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "error",
                format!(
                    "Error overriding aggregate realized gains with FIFO: {}",
                    es(py, &e)
                ),
            )?,
            Err(e) => return Err(e),
        }
    }

    // --- 8. Closed positions ---
    let summary_final = if summary_unfiltered.getattr("empty")?.is_truthy()? {
        summary_unfiltered.clone()
    } else if !show_closed_positions.is_truthy()? {
        if has_col(&summary_unfiltered, "Quantity")? && has_col(&summary_unfiltered, "Symbol")? {
            let held = summary_unfiltered
                .get_item("Quantity")?
                .call_method0("abs")?
                .rich_compare(m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?, CompareOp::Ge)?;
            summary_unfiltered.get_item(held)?.call_method0("copy")?
        } else {
            log(
                py,
                "warning",
                "Could not filter closed positions, Quantity/Symbol columns missing from summary_df_unfiltered.",
            )?;
            has_warnings = true;
            summary_unfiltered.clone()
        }
    } else {
        summary_unfiltered.clone()
    };

    // --- 9a. Contribution % ---
    if !summary_final.getattr("empty")?.is_truthy()? && overall.is_truthy()? {
        let total_invested = o.get_f(&overall, "total_cost_invested", 0.0)?;
        let gain_col = col("Total Gain");
        if has_col(&summary_final, &gain_col)? && o.abs_gt(&total_invested, 1e-9)? {
            let v = summary_final
                .get_item(&gain_col)?
                .div(&total_invested)?
                .mul(100.0)?;
            setitem(&summary_final, "Contribution %", v)?;
        } else {
            setitem(&summary_final, "Contribution %", 0.0)?;
        }
    }

    // --- 9b. % of total ---
    if !summary_final.getattr("empty")?.is_truthy()? {
        let mv_col = col("Market Value");
        if has_col(&summary_final, &mv_col)? {
            let total = summary_final.get_item(&mv_col)?.call_method0("sum")?;
            if o.abs_gt(&total, 1e-9)? {
                let v = summary_final.get_item(&mv_col)?.div(&total)?.mul(100.0)?;
                setitem(&summary_final, "pct_of_total", v)?;
            } else {
                setitem(&summary_final, "pct_of_total", 0.0)?;
            }
        } else {
            setitem(&summary_final, "pct_of_total", 0.0)?;
        }
    }

    // --- 10. Metadata ---
    if overall.is_none() {
        overall = PyDict::new(py).into_any();
    }
    let accounts = PySet::empty(py)?;
    if !all_df.is_none() {
        for c in ["Account", "To Account"] {
            if has_col(all_df, c)? {
                accounts.call_method1(
                    "update",
                    (all_df
                        .get_item(c)?
                        .call_method0("dropna")?
                        .call_method0("unique")?,),
                )?;
            }
        }
    }
    let named = PySet::empty(py)?;
    for acc in accounts.iter() {
        if acc.is_truthy()?
            && acc.is_instance(&o.str_t)?
            && acc.call_method0("strip")?.is_truthy()?
        {
            named.add(acc)?;
        }
    }
    overall.set_item(
        "_available_accounts",
        b.getattr("sorted")?
            .call1((b.getattr("list")?.call1((named,))?,))?,
    )?;
    if display_currency.ne(default_currency)? && fx_rates.is_truthy()? {
        let rate = m.getattr("get_conversion_rate")?.call1((
            default_currency,
            display_currency,
            &fx_rates,
        ))?;
        overall.set_item(
            "exchange_rate_to_display",
            if o.notna(&rate)? { rate } else { none() },
        )?;
    }
    overall.set_item("_fx_rates_vs_usd", &fx_rates)?;

    // --- 11. Taxes ---
    let mut total_taxes = o.fl(0.0);
    if taxes_local.is_truthy()? {
        for item in taxes_local.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let rate =
                m.getattr("get_conversion_rate")?
                    .call1((&kv[0], display_currency, &fx_rates))?;
            if o.notna(&rate)? {
                total_taxes = total_taxes.add(kv[1].mul(&rate)?)?;
            }
        }
    }
    overall.set_item("total_taxes", &total_taxes)?;

    // --- The status line ---
    let end_time = m.getattr("time")?.call_method0("time")?;
    let prefix = if has_errors {
        "Finished with Errors"
    } else if has_warnings {
        "Finished with Warnings"
    } else {
        "Success"
    };
    let mut final_status = format!("{prefix} ({filter_desc})");
    if !status_parts.is_empty() {
        final_status += &format!(" [{}]", joined(&status_parts));
    }
    log(
        py,
        "info",
        format!(
            "Total Summary Calc Time: {} seconds",
            o.fmt_spec(&end_time.sub(&start_time)?, ".2f")?
        ),
    )?;

    // Source-only accounts leave the holdings too, as they left the rows.
    let mut holdings = holdings_in.clone();
    if include_accounts.is_truthy()? {
        let wanted = PySet::empty(py)?;
        for a in include_accounts.try_iter()? {
            wanted.add(o.str_strip(&a?)?.call_method0("upper")?)?;
        }
        let kept = PyDict::new(py);
        for item in holdings_in.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let k = &kv[0];
            let account = if k.is_instance_of::<PyTuple>() && k.len()? > 1 {
                k.get_item(1)?
            } else {
                pystr(py, "")
            };
            if wanted.contains(o.str_strip(&account)?.call_method0("upper")?)? {
                kept.set_item(k, &kv[1])?;
            }
        }
        holdings = kept.into_any();
    }

    PyTuple::new(
        py,
        [
            overall,
            summary_final,
            holdings,
            py.import("builtins")?
                .getattr("dict")?
                .call1((account_metrics,))?,
            combined_ignored_indices,
            combined_ignored_reasons,
            pystr(py, &final_status),
        ],
    )
}
