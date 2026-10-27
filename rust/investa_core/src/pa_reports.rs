//! `portfolio_analyzer`'s report extractors: periodic returns, dividend
//! history and realized capital gains history.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`). These are frame reshaping —
//! resampling, percentage changes, holiday filtering, per-row dividend
//! conversion — so they are the same pandas operations in the original's
//! order, side effects included (`calculate_periodic_returns` re-indexes its
//! input in place, as it did). Column stores go through `pyhelp::setitem`,
//! in-place arithmetic through `operator`, and module names (the FIFO core,
//! the FX bridge, the holiday calendar) are read from `portfolio_analyzer`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::pyhelp::{es, log, pystr, s, setitem};

fn pa(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("portfolio_analyzer")
}

fn kw<'py>(py: Python<'py>, pairs: &[(&str, Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

fn empty(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    x.getattr("empty")?.is_truthy()
}

/// `calculate_periodic_returns`: {interval: returns frame} for W, M, Y, D.
#[pyfunction]
pub fn pa_periodic_returns<'py>(
    py: Python<'py>,
    historical_df: &Bound<'py, PyAny>,
    benchmark_symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let pd = py.import("pandas")?;
    log(py, "debug", "--- Entering calculate_periodic_returns ---")?;
    let is_frame = historical_df.is_instance(&pd.getattr("DataFrame")?)?;
    if !is_frame || empty(historical_df)? {
        log(
            py,
            "warning",
            "Input historical_df is not a valid DataFrame or is empty. Returning empty dict.",
        )?;
        return Ok(PyDict::new(py));
    }
    log(
        py,
        "debug",
        format!(
            "Input historical_df shape: {}",
            s(&historical_df.getattr("shape")?)?
        ),
    )?;
    log(
        py,
        "debug",
        format!("Input benchmarks: {}", s(benchmark_symbols)?),
    )?;
    let out = PyDict::new(py);
    if !historical_df
        .getattr("index")?
        .is_instance(&pd.getattr("DatetimeIndex")?)?
    {
        match pd.call_method1("to_datetime", (historical_df.getattr("index")?,)) {
            Ok(idx) => historical_df.setattr("index", idx)?,
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                log(
                    py,
                    "error",
                    format!("Error converting index to DatetimeIndex: {}", es(py, &e)),
                )?;
                return Ok(out);
            }
            Err(e) => return Err(e),
        }
    }
    let portfolio_col = "Portfolio Accumulated Gain";
    let mut all_gain = vec![pystr(py, portfolio_col)];
    for b in benchmark_symbols.try_iter()? {
        all_gain.push(pystr(py, &format!("{} Accumulated Gain", s(&b?)?)));
    }
    let columns = historical_df.getattr("columns")?;
    let valid = PyList::empty(py);
    for c in all_gain {
        if columns.contains(&c)? {
            valid.append(c)?;
        }
    }
    if valid.is_empty() {
        log(
            py,
            "warning",
            "Cannot calculate periodic returns: No valid accumulated gain columns found.",
        )?;
        return Ok(out);
    }
    let operator = py.import("operator")?;
    for (interval_key, freq_code) in [("W", "W-FRI"), ("M", "ME"), ("Y", "YE"), ("D", "D")] {
        let computed = (|| -> PyResult<Bound<'py, PyAny>> {
            let cols = PyList::new(py, valid.iter())?;
            let agg = PyDict::new(py);
            for c in valid.iter() {
                agg.set_item(c, "last")?;
            }
            let daily = if columns.contains("Portfolio Daily Gain")? {
                "Portfolio Daily Gain"
            } else {
                "daily_gain"
            };
            if columns.contains(daily)? {
                cols.append(daily)?;
                agg.set_item(daily, "sum")?;
            }
            let net_flow_col = "Cumulative Net Flow";
            if columns.contains(net_flow_col)? {
                cols.append(net_flow_col)?;
                agg.set_item(net_flow_col, "last")?;
            }
            let resampled = historical_df
                .get_item(&cols)?
                .call_method1("resample", (freq_code,))?
                .call_method1("agg", (agg,))?;
            let mut factors = resampled.get_item(&valid)?;
            let mut value_change = if resampled.getattr("columns")?.contains(daily)? {
                resampled.get_item(daily)?
            } else {
                pd.call_method0("Series")?
            };
            let mut net_flow = pd.call_method(
                "Series",
                (),
                Some(&kw(
                    py,
                    &[("dtype", py.import("builtins")?.getattr("float")?)],
                )?),
            )?;
            if resampled.getattr("columns")?.contains(net_flow_col)? {
                net_flow = resampled.get_item(net_flow_col)?.call_method0("diff")?;
                if !empty(&net_flow)? {
                    net_flow.getattr("iloc")?.set_item(
                        0,
                        resampled
                            .get_item(net_flow_col)?
                            .getattr("iloc")?
                            .get_item(0)?,
                    )?;
                }
            }
            if interval_key == "D" {
                let mask = factors
                    .getattr("index")?
                    .getattr("dayofweek")?
                    .call_method1("__lt__", (5,))?;
                factors = factors.get_item(&mask)?;
                if !empty(&value_change)? {
                    value_change = value_change.get_item(&mask)?;
                }
                if !empty(&net_flow)? {
                    net_flow = net_flow.get_item(&mask)?;
                }
                let holidays = (|| -> PyResult<()> {
                    let idx = factors.getattr("index")?;
                    if !empty(&factors)?
                        && pd
                            .call_method1("notnull", (idx.call_method0("min")?,))?
                            .is_truthy()?
                    {
                        let cal = pa(py)?.getattr("USFederalHolidayCalendar")?.call0()?;
                        let days = cal.call_method(
                            "holidays",
                            (),
                            Some(&kw(
                                py,
                                &[
                                    ("start", idx.call_method0("min")?),
                                    ("end", idx.call_method0("max")?),
                                ],
                            )?),
                        )?;
                        let keep = idx
                            .call_method1("isin", (days,))?
                            .call_method0("__invert__")?;
                        factors = factors.get_item(&keep)?;
                        if !empty(&value_change)? {
                            value_change = value_change.get_item(&keep)?;
                        }
                        if !empty(&net_flow)? {
                            net_flow = net_flow.get_item(&keep)?;
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = holidays {
                    if !e.is_instance_of::<pyo3::exceptions::PyException>(py) {
                        return Err(e);
                    }
                    log(
                        py,
                        "warning",
                        format!("Failed to filter holidays: {}", es(py, &e)),
                    )?;
                }
            }
            let returns = if !empty(&factors)? {
                let mut r = factors.call_method(
                    "pct_change",
                    (),
                    Some(&kw(py, &[("fill_method", py.None().into_bound(py))])?),
                )?;
                let first = factors.getattr("iloc")?.get_item(0)?.sub(1.0)?;
                r.getattr("iloc")?.set_item(0, first)?;
                r = operator.call_method1("imul", (&r, 100.0))?;
                if !empty(&value_change)? {
                    setitem(&r, format!("Portfolio {interval_key}-Value"), &value_change)?;
                }
                if !empty(&net_flow)? {
                    setitem(&r, format!("Portfolio {interval_key}-NetFlow"), &net_flow)?;
                }
                r
            } else {
                pd.call_method(
                    "DataFrame",
                    (),
                    Some(&kw(py, &[("columns", factors.getattr("columns")?)])?),
                )?
            };
            let renamed = PyList::empty(py);
            for c in returns.getattr("columns")?.try_iter()? {
                renamed.append(c?.call_method1(
                    "replace",
                    (" Accumulated Gain", format!(" {interval_key}-Return")),
                )?)?;
            }
            returns.setattr("columns", renamed)?;
            Ok(returns)
        })();
        match computed {
            Ok(frame) => out.set_item(interval_key, frame)?,
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                log(
                    py,
                    "error",
                    format!(
                        "Error calculating {interval_key} periodic returns: {}",
                        es(py, &e)
                    ),
                )?;
                out.set_item(interval_key, pd.call_method0("DataFrame")?)?;
            }
            Err(e) => return Err(e),
        }
    }
    let keys = py
        .import("builtins")?
        .getattr("list")?
        .call1((out.keys(),))?;
    log(
        py,
        "debug",
        format!(
            "--- Exiting calculate_periodic_returns. Result keys: {} ---",
            s(&keys)?
        ),
    )?;
    Ok(out)
}

const DIVIDEND_COLUMNS: [&str; 9] = [
    "Date",
    "Symbol",
    "Account",
    "LocalCurrency",
    "DividendAmountLocal",
    "FXRateUsed",
    "DividendAmountDisplayCurrency",
    "TaxAmountLocal",
    "TaxAmountDisplayCurrency",
];

/// `extract_dividend_history`: dividends (and their withheld tax) converted
/// to the display currency at each payment date's rate, newest first.
#[pyfunction]
pub fn pa_dividend_history<'py>(
    py: Python<'py>,
    all_transactions_df: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    log(
        py,
        "info",
        format!(
            "Extracting dividend history for display in {}...",
            s(display_currency)?
        ),
    )?;
    let is_frame = all_transactions_df.is_instance(&pd.getattr("DataFrame")?)?;
    let shape = if is_frame {
        s(&all_transactions_df.getattr("shape")?)?
    } else {
        "Not a DF".to_string()
    };
    log(
        py,
        "debug",
        format!("  Input all_transactions_df shape: {shape}"),
    )?;
    log(
        py,
        "debug",
        format!(
            "  extract_dividend_history received include_accounts: {}",
            s(include_accounts)?
        ),
    )?;
    let fx_keys = if historical_fx_yf.is_truthy()? {
        s(&py
            .import("builtins")?
            .getattr("list")?
            .call1((historical_fx_yf.call_method0("keys")?,))?)?
    } else {
        "Empty".to_string()
    };
    log(py, "debug", format!("  Historical FX data keys: {fx_keys}"))?;
    if !is_frame || empty(all_transactions_df)? {
        log(
            py,
            "warning",
            "Dividend history: Input transactions DataFrame is empty.",
        )?;
        return pd.call_method0("DataFrame");
    }
    let lower_type = || -> PyResult<Bound<'py, PyAny>> {
        all_transactions_df
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")
    };
    let dividend_mask = lower_type()?
        .call_method1("__eq__", ("dividend",))?
        .call_method1(
            "__or__",
            (lower_type()?
                .call_method1("__eq__", ("interest",))?
                .call_method1(
                    "__and__",
                    (all_transactions_df
                        .get_item("Symbol")?
                        .getattr("str")?
                        .call_method0("upper")?
                        .call_method1("__eq__", ("$CASH",))?,),
                )?,),
        )?;
    let tax_mask =
        lower_type()?.call_method1("isin", (PyList::new(py, ["tax", "withholding tax"])?,))?;
    let mut dividends = all_transactions_df
        .get_item(&dividend_mask)?
        .call_method0("copy")?;
    let mut taxes = all_transactions_df
        .get_item(&tax_mask)?
        .call_method0("copy")?;
    log(
        py,
        "debug",
        format!(
            "  Initial dividend transactions (before account filter): {} rows.",
            dividends.len()?
        ),
    )?;
    if !empty(&dividends)? {
        log(
            py,
            "debug",
            format!(
                "    Accounts in initial dividend transactions: {}",
                s(&dividends.get_item("Account")?.call_method0("unique")?)?
            ),
        )?;
    }
    if !include_accounts.is_none() && include_accounts.is_instance_of::<PyList>() {
        if dividends.getattr("columns")?.contains("Account")? {
            dividends = dividends
                .get_item(
                    dividends
                        .get_item("Account")?
                        .call_method1("isin", (include_accounts,))?,
                )?
                .call_method0("copy")?;
        }
        if !empty(&taxes)? && taxes.getattr("columns")?.contains("Account")? {
            taxes = taxes
                .get_item(
                    taxes
                        .get_item("Account")?
                        .call_method1("isin", (include_accounts,))?,
                )?
                .call_method0("copy")?;
        }
    }
    log(
        py,
        "debug",
        format!("  Found {} dividend transactions.", dividends.len()?),
    )?;
    if empty(&dividends)? {
        log(py, "info", "No dividend transactions found.")?;
        return pd.call_method(
            "DataFrame",
            (),
            Some(&kw(
                py,
                &[("columns", PyList::new(py, DIVIDEND_COLUMNS)?.into_any())],
            )?),
        );
    }
    let tax_lookup = py
        .import("collections")?
        .getattr("defaultdict")?
        .call1((py.get_type::<pyo3::types::PyFloat>(),))?;
    let coerce = kw(py, &[("errors", pystr(py, "coerce"))])?;
    let to_numeric = |x: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if crate::pyhelp::numeric_passthrough(&x) {
            return Ok(x);
        }
        pd.call_method("to_numeric", (x,), Some(&coerce))
    };
    let abs = py.import("builtins")?.getattr("abs")?;
    for pair in crate::pyhelp::iterrows(&taxes)?.try_iter()? {
        let tx = pair?.get_item(1)?;
        let added = (|| -> PyResult<()> {
            let d = tx.get_item("Date")?.call_method0("date")?;
            let (sym, acc) = (tx.get_item("Symbol")?, tx.get_item("Account")?);
            let mut amt = abs.call1((to_numeric(tx.call_method1("get", ("Total Amount",))?)?,))?;
            if pd.call_method1("isna", (&amt,))?.is_truthy()? || amt.le(1e-9)? {
                amt = abs.call1((to_numeric(tx.call_method1("get", ("Price/Share",))?)?,))?;
            }
            if pd.call_method1("notna", (&amt,))?.is_truthy()? {
                let key = pyo3::types::PyTuple::new(py, [d, sym, acc])?;
                tax_lookup.set_item(&key, tax_lookup.get_item(&key)?.add(amt)?)?;
            }
            Ok(())
        })();
        if let Err(e) = added
            && !e.is_instance_of::<pyo3::exceptions::PyException>(py)
        {
            return Err(e);
        }
    }
    let results = PyList::empty(py);
    let bridge = pa(py)?.getattr("get_historical_rate_via_usd_bridge")?;
    for (i, pair) in crate::pyhelp::iterrows(&dividends)?.try_iter()?.enumerate() {
        let row = pair?.get_item(1)?;
        let processed = (|| -> PyResult<()> {
            let tx_date = row.get_item("Date")?.call_method0("date")?;
            let (symbol, account) = (row.get_item("Symbol")?, row.get_item("Account")?);
            let local_curr = row.call_method1("get", ("Local Currency", default_currency))?;
            let mut dividend_local = to_numeric(row.call_method1("get", ("Total Amount",))?)?;
            if pd.call_method1("isna", (&dividend_local,))?.is_truthy()? {
                let qty = to_numeric(row.call_method1("get", ("Quantity",))?)?;
                let price = to_numeric(row.call_method1("get", ("Price/Share",))?)?;
                if pd.call_method1("notna", (&qty,))?.is_truthy()?
                    && pd.call_method1("notna", (&price,))?.is_truthy()?
                {
                    dividend_local = qty.mul(price)?;
                } else {
                    log(
                        py,
                        "warning",
                        format!(
                            "Could not determine dividend amount for row: {}",
                            s(&row.call_method1("get", ("original_index", "N/A"))?)?
                        ),
                    )?;
                    return Ok(());
                }
            }
            let commission = to_numeric(row.call_method1("get", ("Commission",))?)?;
            if pd.call_method1("notna", (&commission,))?.is_truthy()? {
                dividend_local = py
                    .import("operator")?
                    .call_method1("isub", (&dividend_local, commission))?;
            }
            let key = pyo3::types::PyTuple::new(py, [&tx_date, &symbol, &account])?;
            let tax_local = tax_lookup.call_method1("get", (key, 0.0))?;
            let fx_rate =
                bridge.call1((&local_curr, display_currency, &tx_date, historical_fx_yf))?;
            let nan = py.import("numpy")?.getattr("nan")?;
            let fx_ok = pd.call_method1("notna", (&fx_rate,))?.is_truthy()?;
            let dividend_display =
                if fx_ok && pd.call_method1("notna", (&dividend_local,))?.is_truthy()? {
                    dividend_local.mul(&fx_rate)?
                } else {
                    nan.clone()
                };
            let tax_display = if fx_ok && pd.call_method1("notna", (&tax_local,))?.is_truthy()? {
                tax_local.mul(&fx_rate)?
            } else {
                nan
            };
            if i < 5 {
                log(
                    py,
                    "debug",
                    format!(
                        "    Processed dividend row (orig_idx: {}):",
                        s(&row.call_method1("get", ("original_index", "N/A"))?)?
                    ),
                )?;
                log(
                    py,
                    "debug",
                    format!(
                        "      Date: {}, Symbol: {}, Account: {}, LocalCurr: {}",
                        s(&tx_date)?,
                        s(&symbol)?,
                        s(&account)?,
                        s(&local_curr)?
                    ),
                )?;
                log(
                    py,
                    "debug",
                    format!(
                        "      DividendLocal: {}, FXRateUsed: {}, DividendDisplayCurr: {}",
                        s(&dividend_local)?,
                        s(&fx_rate)?,
                        s(&dividend_display)?
                    ),
                )?;
            }
            results.append(PyList::new(
                py,
                [
                    tx_date,
                    symbol,
                    account,
                    local_curr,
                    dividend_local,
                    fx_rate,
                    dividend_display,
                    tax_local,
                    tax_display,
                ],
            )?)?;
            Ok(())
        })();
        if let Err(e) = processed {
            if !e.is_instance_of::<pyo3::exceptions::PyException>(py) {
                return Err(e);
            }
            log(
                py,
                "error",
                format!(
                    "Error processing dividend row {}: {}",
                    s(&row.call_method1("get", ("original_index", "N/A"))?)?,
                    es(py, &e)
                ),
            )?;
        }
    }
    if results.is_empty() {
        log(
            py,
            "warning",
            "  No results after processing dividend transactions.",
        )?;
        return pd.call_method0("DataFrame");
    }
    let df = pd.call_method(
        "DataFrame",
        (results,),
        Some(&kw(
            py,
            &[("columns", PyList::new(py, DIVIDEND_COLUMNS)?.into_any())],
        )?),
    )?;
    df.call_method(
        "sort_values",
        (),
        Some(&kw(
            py,
            &[
                ("by", pystr(py, "Date")),
                (
                    "ascending",
                    pyo3::types::PyBool::new(py, false).to_owned().into_any(),
                ),
                (
                    "inplace",
                    pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                ),
            ],
        )?),
    )?;
    log(
        py,
        "info",
        format!("Extracted {} dividend records.", df.len()?),
    )?;
    if !empty(&df)? {
        log(
            py,
            "debug",
            format!(
                "  Returned df_dividends head:\n{}",
                s(&df.call_method0("head")?.call_method0("to_string")?)?
            ),
        )?;
        let nans = df
            .get_item("DividendAmountDisplayCurrency")?
            .call_method0("isna")?
            .call_method0("sum")?;
        log(
            py,
            "debug",
            format!(
                "  'DividendAmountDisplayCurrency' NaNs: {} out of {}",
                s(&nans)?,
                df.len()?
            ),
        )?;
    }
    Ok(df)
}

const GAIN_COLUMNS: [&str; 15] = [
    "Date",
    "Symbol",
    "Account",
    "Type",
    "Quantity",
    "Avg Sale Price (Local)",
    "Total Proceeds (Local)",
    "Total Cost Basis (Local)",
    "Realized Gain (Local)",
    "Sale/Cover FX Rate",
    "Total Proceeds (Display)",
    "Total Cost Basis (Display)",
    "Realized Gain (Display)",
    "LocalCurrency",
    "original_tx_id",
];

/// `extract_realized_capital_gains_history`: FIFO realized gains, filtered to
/// the user's accounts and dates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_realized_gains_history<'py>(
    py: Python<'py>,
    all_transactions_df: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    shortable_symbols: &Bound<'py, PyAny>,
    stock_quantity_close_tolerance: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    current_fx_rates_vs_usd: &Bound<'py, PyAny>,
    from_date: &Bound<'py, PyAny>,
    to_date: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    log(
        py,
        "info",
        format!(
            "Extracting realized capital gains history for display in {}...",
            s(display_currency)?
        ),
    )?;
    if !all_transactions_df.is_instance(&pd.getattr("DataFrame")?)? || empty(all_transactions_df)? {
        log(
            py,
            "warning",
            "Capital gains: Input transactions DataFrame is empty.",
        )?;
        return pd.call_method(
            "DataFrame",
            (),
            Some(&kw(
                py,
                &[("columns", PyList::new(py, GAIN_COLUMNS)?.into_any())],
            )?),
        );
    }
    let txs = all_transactions_df.call_method0("copy")?;
    txs.call_method(
        "sort_values",
        (),
        Some(&kw(
            py,
            &[
                (
                    "by",
                    PyList::new(py, ["Date", "original_index"])?.into_any(),
                ),
                (
                    "inplace",
                    pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                ),
            ],
        )?),
    )?;
    let fifo = pa(py)?.getattr("calculate_fifo_lots_and_gains")?;
    // `df_gains, _ = calculate_fifo_lots_and_gains(...)`: the tuple is gone
    // at once in Python, and with it the last reference to the unfiltered
    // frame — which pandas' SettingWithCopy check looks for through a weakref.
    let mut gains = {
        let got = fifo.call(
            (),
            Some(&kw(
                py,
                &[
                    ("transactions_df", txs),
                    ("display_currency", display_currency.clone()),
                    ("historical_fx_yf", historical_fx_yf.clone()),
                    ("default_currency", default_currency.clone()),
                    ("shortable_symbols", shortable_symbols.clone()),
                    (
                        "stock_quantity_close_tolerance",
                        stock_quantity_close_tolerance.clone(),
                    ),
                    ("current_fx_rates_vs_usd", current_fx_rates_vs_usd.clone()),
                ],
            )?),
        )?;
        got.get_item(0)?
    };
    if include_accounts.is_truthy()? && include_accounts.is_instance_of::<PyList>() {
        let upper = PyList::empty(py);
        for acc in include_accounts.try_iter()? {
            upper.append(acc?.str()?.call_method0("upper")?.call_method0("strip")?)?;
        }
        if !empty(&gains)? {
            gains = gains.get_item(gains.get_item("Account")?.call_method1("isin", (upper,))?)?;
        }
    }
    if !empty(&gains)? && gains.getattr("columns")?.contains("Date")? {
        if !pd
            .getattr("api")?
            .getattr("types")?
            .call_method1("is_datetime64_any_dtype", (gains.get_item("Date")?,))?
            .is_truthy()?
        {
            setitem(
                &gains,
                "Date",
                pd.call_method1("to_datetime", (gains.get_item("Date")?,))?,
            )?;
        }
        if from_date.is_truthy()? {
            let bound = pd.call_method1("Timestamp", (from_date,))?;
            gains = gains.get_item(gains.get_item("Date")?.call_method1("__ge__", (bound,))?)?;
        }
        if to_date.is_truthy()? {
            let bound = pd.call_method1("Timestamp", (to_date,))?;
            gains = gains.get_item(gains.get_item("Date")?.call_method1("__le__", (bound,))?)?;
        }
    }
    log(
        py,
        "info",
        format!("Extracted {} realized capital gains records.", gains.len()?),
    )?;
    Ok(gains)
}

#[allow(dead_code)]
fn unused(_: &Bound<'_, PyString>) {}
