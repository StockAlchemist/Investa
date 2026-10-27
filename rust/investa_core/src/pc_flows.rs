//! `portfolio_cashflows`: the daily net external cash flow the TWR and the
//! cost line are built on — over a date range (the vectorized pass the
//! history uses) and for a single day.
//!
//! Ported line for line from `src/portfolio_cashflows.py` (frozen in
//! `tests/reference/portfolio_cashflows.py`). Both are pandas and per-row
//! Python, so every operation is Python's own, in the original's order. A
//! frame rebound in the original (`df = df[mask]`) is reassigned here, not
//! shadowed, so the parent frame dies where it died and pandas'
//! SettingWithCopy check sees what it saw. The helpers are looked up on
//! `portfolio_cashflows` at call time.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyIndexError, PyKeyError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PySet, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, setitem, unpack};

/// Where the original appends a line for a transfer it could not price (a
/// scratch file of another tool; kept, see the port doc).
const MISSING_PRICE_LOG: &str = "/Users/kmatan/.gemini/antigravity/brain/19ed956f-9dc7-40d0-85b1-1aa49c84c1d9/scratch/missing_transfer_prices.txt";

fn pc(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("portfolio_cashflows")
}

/// `pd.Series(value, index=index)`
fn series<'py>(
    pd: &Bound<'py, PyModule>,
    value: impl IntoPyObject<'py>,
    index: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pd.py();
    pd.getattr("Series")?
        .call((value,), Some(&kwargs(py, &[("index", index)])?))
}

/// `s.astype(str).str.upper().str.strip()`
fn upper_strip_col<'py>(o: &Ops<'py>, s: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    s.call_method1("astype", (&o.str_t,))?
        .getattr("str")?
        .call_method0("upper")?
        .getattr("str")?
        .call_method0("strip")
}

/// `s.str.lower().str.strip()`
fn lower_strip_col<'py>(s: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    s.getattr("str")?
        .call_method0("lower")?
        .getattr("str")?
        .call_method0("strip")
}

/// `pd.to_numeric(s, errors="coerce")`
fn numeric<'py>(pd: &Bound<'py, PyModule>, s: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = pd.py();
    pd.call_method(
        "to_numeric",
        (s,),
        Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
    )
}

/// The target/USD or local/USD rate over the range, back- and forward-filled.
fn usd_rate_series<'py>(
    pd: &Bound<'py, PyModule>,
    o: &Ops<'py>,
    currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    date_range: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pd.py();
    if currency.call_method0("upper")?.eq("USD")? {
        return series(pd, 1.0, date_range);
    }
    let pair = format!("{}=X", o.fmt(&currency.call_method0("upper")?)?);
    if historical_fx_yf.contains(&pair)? {
        let df = historical_fx_yf.get_item(&pair)?;
        if !df.getattr("empty")?.is_truthy()? && df.getattr("columns")?.contains("price")? {
            let s = df.get_item("price")?;
            let kw = PyDict::new(py);
            kw.set_item("utc", true)?;
            s.setattr(
                "index",
                crate::pyhelp::to_datetime(pd, &s.getattr("index")?, Some(&kw))?,
            )?;
            return s
                .call_method1("reindex", (date_range,))?
                .call_method0("ffill")?
                .call_method0("bfill");
        }
    }
    series(pd, f64::NAN, date_range)
}

/// `_calculate_daily_net_cash_flow_vectorized`: (net external flow per day
/// in the target currency, whether an FX lookup fell back).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pc_daily_net_cash_flow_vectorized<'py>(
    py: Python<'py>,
    date_range: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    target_currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    included_accounts: &Bound<'py, PyAny>,
    historical_prices_yf_unadjusted: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let _ = default_currency;
    let m = pc(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let mut daily_net_flow = series(&pd, 0.0, date_range)?;
    let mut has_lookup_errors = false;

    if transactions_df.getattr("empty")?.is_truthy()? {
        return Ok((daily_net_flow, false));
    }
    let utc = kwargs(
        py,
        &[(
            "utc",
            &pyo3::types::PyBool::new(py, true).to_owned().into_any(),
        )],
    )?;
    let tx_dates = crate::pyhelp::to_datetime(&pd, &transactions_df.get_item("Date")?, Some(&utc))?;
    let mask_date = tx_dates.rich_compare(date_range.get_item(-1)?, CompareOp::Le)?;
    let df_period = transactions_df.get_item(mask_date)?.call_method0("copy")?;
    if df_period.getattr("empty")?.is_truthy()? {
        return Ok((daily_net_flow, false));
    }

    // Dates normalised to naive, then to the range's zone.
    let target_tz = date_range.getattr("tz")?;
    let naive = crate::pyhelp::to_datetime(&pd, &df_period.get_item("Date")?, Some(&utc))?
        .getattr("dt")?
        .call_method1("tz_localize", (py.None(),))?;
    setitem(&df_period, "Date", naive)?;
    if !target_tz.is_none() {
        let aware = df_period
            .get_item("Date")?
            .getattr("dt")?
            .call_method1("tz_localize", (&target_tz,))?;
        setitem(&df_period, "Date", aware)?;
    }

    let included = included_accounts.is_truthy()?;
    let included_set = PySet::empty(py)?;
    if included {
        for a in included_accounts.try_iter()? {
            included_set.add(
                o.str_t
                    .call1((a?,))?
                    .call_method0("upper")?
                    .call_method0("strip")?,
            )?;
        }
    }

    // --- 1. Cash deposits and withdrawals: always external ---
    let cash_mask = py
        .import("finutils")?
        .getattr("compute_external_flow_mask")?
        .call1((&df_period,))?;
    let mut df_cash = df_period.get_item(cash_mask)?.call_method0("copy")?;
    if included && !df_cash.getattr("empty")?.is_truthy()? {
        let keep = upper_strip_col(&o, &df_cash.get_item("Account")?)?
            .call_method1("isin", (&included_set,))?;
        df_cash = df_cash.get_item(keep)?;
    }

    let flows_list = PyList::empty(py);
    let out_cols = || PyList::new(py, ["Date", "_flow_local", "Local Currency"]);
    if !df_cash.getattr("empty")?.is_truthy()? {
        let total_amount_col = if df_cash.getattr("columns")?.contains("Total Amount")? {
            df_cash.get_item("Total Amount")?
        } else {
            series(&pd, 0.0, &df_cash.getattr("index")?)?
        };
        let total_amount = numeric(&pd, &total_amount_col)?
            .call_method1("fillna", (0.0,))?
            .call_method0("abs")?;
        let qty = numeric(&pd, &df_cash.get_item("Quantity")?)?
            .call_method1("fillna", (0.0,))?
            .call_method0("abs")?;
        let comm =
            numeric(&pd, &df_cash.get_item("Commission")?)?.call_method1("fillna", (0.0,))?;
        let cash_amt = np.call_method1(
            "where",
            (
                total_amount.rich_compare(1e-9, CompareOp::Gt)?,
                &total_amount,
                &qty,
            ),
        )?;
        let types = df_cash
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?;
        let is_dep = types.rich_compare("deposit", CompareOp::Eq)?;
        let types = df_cash
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?;
        let is_wd = types.rich_compare("withdrawal", CompareOp::Eq)?;
        let local_flow = series(&pd, 0.0, &df_cash.getattr("index")?)?;
        setitem(
            &local_flow,
            &is_dep,
            cash_amt.get_item(&is_dep)?.sub(comm.get_item(&is_dep)?)?,
        )?;
        setitem(
            &local_flow,
            &is_wd,
            cash_amt
                .get_item(&is_wd)?
                .neg()?
                .sub(comm.get_item(&is_wd)?)?,
        )?;
        setitem(&df_cash, "_flow_local", &local_flow)?;
        flows_list.append(df_cash.get_item(out_cols()?)?)?;
    }

    // --- 2. Transfers (and ledger clean-ups) across the selection's edge ---
    if included {
        let keywords = PyList::new(
            py,
            [
                "cleanup",
                "dust",
                "ghost",
                "artifact",
                "adjustment",
                "correction",
            ],
        )?;
        let has_note = df_period.getattr("columns")?.contains("Note")?;
        let mut cleanup_mask = series(&pd, false, &df_period.getattr("index")?)?;
        if has_note {
            let kw_for_closure = keywords.clone().unbind();
            // lambda x: any(kw in x for kw in cleanup_keywords)
            let any_keyword =
                PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<bool> {
                    let py = args.py();
                    let x = args.get_item(0)?;
                    for kw in kw_for_closure.bind(py).iter() {
                        if x.contains(kw)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                })?;
            cleanup_mask = df_period
                .get_item("Note")?
                .call_method1("fillna", ("",))?
                .getattr("str")?
                .call_method0("lower")?
                .call_method1("apply", (any_keyword,))?;
        }
        let trans_mask = lower_strip_col(&df_period.get_item("Type")?)?
            .rich_compare("transfer", CompareOp::Eq)?
            .bitor(&cleanup_mask)?;
        let df_trans = df_period.get_item(trans_mask)?.call_method0("copy")?;
        if !df_trans.getattr("empty")?.is_truthy()? {
            let index = df_trans.getattr("index")?;
            let src_in = upper_strip_col(&o, &df_trans.get_item("Account")?)?
                .call_method1("isin", (&included_set,))?;
            let mut dest_in = series(&pd, false, &index)?;
            if df_trans.getattr("columns")?.contains("To Account")? {
                dest_in = upper_strip_col(&o, &df_trans.get_item("To Account")?)?
                    .call_method1("isin", (&included_set,))?;
            }
            let mut multiplier = series(&pd, 0.0, &index)?;
            let is_real_transfer = lower_strip_col(&df_trans.get_item("Type")?)?
                .rich_compare("transfer", CompareOp::Eq)?;
            let out_mask = is_real_transfer
                .bitand(&src_in)?
                .bitand(dest_in.call_method0("__invert__")?)?;
            setitem(&multiplier, out_mask, -1.0)?;
            let in_mask = is_real_transfer
                .bitand(src_in.call_method0("__invert__")?)?
                .bitand(&dest_in)?;
            setitem(&multiplier, in_mask, 1.0)?;

            // A clean-up buy comes in, a clean-up sale goes out.
            let is_cleanup = cleanup_mask
                .call_method1("reindex", (&index,))?
                .call_method1("fillna", (false,))?
                .bitand(is_real_transfer.call_method0("__invert__")?)?;
            if is_cleanup.call_method0("any")?.is_truthy()? {
                let type_l = lower_strip_col(&df_trans.get_item("Type")?)?;
                let ins = is_cleanup.bitand(
                    type_l.call_method1("isin", (PyList::new(py, ["buy", "deposit"])?,))?,
                )?;
                setitem(&multiplier, ins, 1.0)?;
                let outs = is_cleanup.bitand(
                    type_l.call_method1("isin", (PyList::new(py, ["sell", "withdrawal"])?,))?,
                )?;
                setitem(&multiplier, outs, -1.0)?;
            }

            let nonzero = multiplier.rich_compare(0.0, CompareOp::Ne)?;
            let relevant = df_trans.get_item(&nonzero)?.call_method0("copy")?;
            let nonzero = multiplier.rich_compare(0.0, CompareOp::Ne)?;
            multiplier = multiplier.get_item(nonzero)?;

            if !relevant.getattr("empty")?.is_truthy()? {
                let qty = numeric(&pd, &relevant.get_item("Quantity")?)?
                    .call_method1("fillna", (0.0,))?
                    .call_method0("abs")?;
                let price = numeric(&pd, &relevant.get_item("Price/Share")?)?
                    .call_method1("fillna", (0.0,))?;
                let is_cash_mask = relevant
                    .get_item("Symbol")?
                    .getattr("str")?
                    .call_method0("upper")?
                    .rich_compare(
                        m.getattr("CASH_SYMBOL_CSV")?.call_method0("upper")?,
                        CompareOp::Eq,
                    )?;
                setitem(&price, is_cash_mask, 1.0)?;

                let missing = price.rich_compare(1e-9, CompareOp::Le)?;
                if missing.call_method0("any")?.is_truthy()? {
                    let unique_syms = relevant
                        .get_item(&missing)?
                        .get_item("Symbol")?
                        .call_method0("unique")?;
                    for sym in unique_syms.try_iter()? {
                        let sym = sym?;
                        let sym_mask = relevant
                            .get_item("Symbol")?
                            .rich_compare(&sym, CompareOp::Eq)?
                            .bitand(&missing)?;
                        let mut price_found = o.fl(0.0);

                        // A: the symbol's own price history.
                        let yf_sym = if internal_to_yf_map.is_truthy()? {
                            internal_to_yf_map.call_method1("get", (&sym,))?
                        } else {
                            py.None().into_bound(py)
                        };
                        if yf_sym.is_truthy()?
                            && historical_prices_yf_unadjusted.is_truthy()?
                            && historical_prices_yf_unadjusted.contains(&yf_sym)?
                        {
                            let r = (|| -> PyResult<()> {
                                let mut ps = historical_prices_yf_unadjusted
                                    .get_item(&yf_sym)?
                                    .get_item("price")?;
                                if !ps.getattr("empty")?.is_truthy()? {
                                    if !ps
                                        .getattr("index")?
                                        .getattr("is_monotonic_increasing")?
                                        .is_truthy()?
                                    {
                                        ps = ps.call_method0("sort_index")?;
                                    }
                                    if !ps
                                        .getattr("index")?
                                        .is_instance(&pd.getattr("DatetimeIndex")?)?
                                    {
                                        ps.setattr(
                                            "index",
                                            pd.call_method1(
                                                "to_datetime",
                                                (ps.getattr("index")?,),
                                            )?,
                                        )?;
                                    }
                                    let dates_needed = pd.call_method1(
                                        "to_datetime",
                                        (relevant.getattr("loc")?.get_item(PyTuple::new(
                                            py,
                                            [sym_mask.clone(), pystr(py, "Date")],
                                        )?)?,),
                                    )?;
                                    let kw = kwargs(py, &[("method", &pystr(py, "ffill"))])?;
                                    let prices =
                                        ps.call_method("reindex", (dates_needed,), Some(&kw))?;
                                    price_found = prices.getattr("iloc")?.get_item(0)?;
                                }
                                Ok(())
                            })();
                            match r {
                                Ok(()) => {}
                                Err(e) if e.is_instance_of::<PyException>(py) => log(
                                    py,
                                    "warning",
                                    format!(
                                        "Transfer price fallback failed for {}: {}",
                                        o.fmt(&sym)?,
                                        es(py, &e)
                                    ),
                                )?,
                                Err(e) => return Err(e),
                            }
                        }

                        // B: any other trade of the symbol.
                        if price_found.le(1e-9)? {
                            let match_tx = df_period.get_item(
                                df_period
                                    .get_item("Symbol")?
                                    .rich_compare(&sym, CompareOp::Eq)?
                                    .bitand(
                                        df_period
                                            .get_item("Price/Share")?
                                            .rich_compare(1e-9, CompareOp::Gt)?,
                                    )?,
                            )?;
                            if !match_tx.getattr("empty")?.is_truthy()? {
                                price_found = match_tx
                                    .getattr("iloc")?
                                    .get_item(0)?
                                    .get_item("Price/Share")?;
                            }
                        }

                        if price_found.gt(1e-9)? {
                            price.getattr("loc")?.set_item(&sym_mask, &price_found)?;
                        } else {
                            let line = format!("Missing price for transfer: {}\n", o.fmt(&sym)?);
                            let r = (|| -> PyResult<()> {
                                let f = py
                                    .import("builtins")?
                                    .getattr("open")?
                                    .call1((MISSING_PRICE_LOG, "a"))?;
                                with1(py, &f, |f| {
                                    f.call_method1("write", (&line,))?;
                                    Ok(())
                                })?;
                                Ok(())
                            })();
                            match r {
                                Ok(()) => {}
                                Err(e) if e.is_instance_of::<PyOSError>(py) => {}
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }

                let value = qty.mul(&price)?;
                let local_flow = value.mul(&multiplier)?;
                setitem(&relevant, "_flow_local", local_flow)?;
                flows_list.append(relevant.get_item(out_cols()?)?)?;
            }
        }
    }

    if !flows_list.is_truthy()? {
        return Ok((daily_net_flow, false));
    }

    // --- 3. Consolidate and convert ---
    let all_flows = pd.call_method1("concat", (&flows_list,))?;
    let grouped = all_flows
        .call_method1("groupby", (PyList::new(py, ["Date", "Local Currency"])?,))?
        .get_item("_flow_local")?
        .call_method0("sum")?
        .call_method0("reset_index")?;
    let final_flows = PyList::empty(py);
    for curr in grouped
        .get_item("Local Currency")?
        .call_method0("unique")?
        .try_iter()?
    {
        let curr = curr?;
        let sub = grouped
            .get_item(
                grouped
                    .get_item("Local Currency")?
                    .rich_compare(&curr, CompareOp::Eq)?,
            )?
            .call_method0("copy")?;
        if curr.eq(target_currency)? {
            final_flows.append(
                sub.call_method1("set_index", ("Date",))?
                    .get_item("_flow_local")?,
            )?;
            continue;
        }
        // The cross rate over the whole range, looked up per flow date.
        let target_rate = usd_rate_series(&pd, &o, target_currency, historical_fx_yf, date_range)?;
        let local_rate = usd_rate_series(&pd, &o, &curr, historical_fx_yf, date_range)?;
        let errstate = np.call_method(
            "errstate",
            (),
            Some(&kwargs(
                py,
                &[
                    ("divide", &pystr(py, "ignore")),
                    ("invalid", &pystr(py, "ignore")),
                ],
            )?),
        )?;
        let cross = with1(py, &errstate, |_| {
            let mut cross = target_rate.div(&local_rate)?;
            if cross
                .call_method0("isna")?
                .call_method0("any")?
                .is_truthy()?
            {
                cross = cross.call_method0("ffill")?.call_method0("bfill")?;
                cross = cross.call_method1("fillna", (1.0,))?;
            }
            Ok(cross)
        })?
        .expect("errstate does not suppress");
        let rate_map = pd
            .getattr("Series")?
            .call(
                (cross.getattr("values")?,),
                Some(&kwargs(
                    py,
                    &[("index", &cross.getattr("index")?.getattr("date")?)],
                )?),
            )?
            .call_method0("to_dict")?;
        let mut factors = sub
            .get_item("Date")?
            .getattr("dt")?
            .getattr("date")?
            .call_method1("map", (rate_map,))?;
        if factors
            .call_method0("isna")?
            .call_method0("any")?
            .is_truthy()?
        {
            factors = factors.call_method1("fillna", (1.0,))?;
            has_lookup_errors = true;
        }
        let converted = sub.get_item("_flow_local")?.mul(&factors)?;
        converted.setattr("index", sub.get_item("Date")?)?;
        final_flows.append(converted)?;
    }

    if final_flows.is_truthy()? {
        let total = pd.call_method1("concat", (&final_flows,))?;
        let kw = kwargs(py, &[("level", &0i64.into_pyobject(py)?.into_any())])?;
        let mut add = total
            .call_method("groupby", (), Some(&kw))?
            .call_method0("sum")?;

        // The flows' zone follows the range's.
        let dti = pd.getattr("DatetimeIndex")?;
        if daily_net_flow.getattr("index")?.is_instance(&dti)? {
            let target_tz = daily_net_flow.getattr("index")?.getattr("tz")?;
            if add.getattr("index")?.is_instance(&dti)? {
                let source_tz = add.getattr("index")?.getattr("tz")?;
                if !target_tz.is_none() {
                    let idx = if source_tz.is_none() {
                        add.getattr("index")?
                            .call_method1("tz_localize", (&target_tz,))?
                    } else {
                        add.getattr("index")?
                            .call_method1("tz_convert", (&target_tz,))?
                    };
                    add.setattr("index", idx)?;
                } else if !source_tz.is_none() {
                    add.setattr(
                        "index",
                        add.getattr("index")?
                            .call_method1("tz_localize", (py.None(),))?,
                    )?;
                }
            }
        }

        // Flows before the range land on its first day as one initial flow.
        let start_ts = date_range.get_item(0)?;
        let pre = add
            .getattr("index")?
            .rich_compare(&start_ts, CompareOp::Lt)?;
        if pre.call_method0("any")?.is_truthy()? {
            let initial = add.get_item(&pre)?.call_method0("sum")?;
            log(
                py,
                "info",
                format!(
                    "STABILIZATION: Found {} transactions before start date. Adding Initial Flow of {} to {}",
                    o.fmt(&pre.call_method0("sum")?)?,
                    o.fmt_spec(&initial, ".2f")?,
                    o.fmt(&start_ts)?
                ),
            )?;
            add = add.get_item(pre.call_method0("__invert__")?)?;
            if add.getattr("index")?.contains(&start_ts)? {
                let v = add.get_item(&start_ts)?.add(&initial)?;
                setitem(&add, &start_ts, v)?;
            } else {
                setitem(&add, &start_ts, &initial)?;
            }
        }
        let kw = kwargs(py, &[("fill_value", &o.fl(0.0))])?;
        daily_net_flow = daily_net_flow.call_method("add", (add,), Some(&kw))?;
    }
    Ok((daily_net_flow, has_lookup_errors))
}

/// The local price on `target_date` from a symbol's price history: the
/// per-day pass's fallback, as both its blocks spell it (the first of Close,
/// price, Adj Close, else the row's first column; the exact date first, then
/// as a Timestamp).
fn market_price_on<'py>(
    py: Python<'py>,
    pd: &Bound<'py, PyModule>,
    prices_df: &Bound<'py, PyAny>,
    target_date: &Bound<'py, PyAny>,
    current: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut price = current;
    let pick = |at: &Bound<'py, PyAny>, price: &mut Bound<'py, PyAny>| -> PyResult<()> {
        let columns = prices_df.getattr("columns")?;
        for c in ["Close", "price", "Adj Close"] {
            if columns.contains(c)? {
                *price = prices_df
                    .getattr("loc")?
                    .get_item(PyTuple::new(py, [at.clone(), pystr(py, c)])?)?;
                return Ok(());
            }
        }
        // for/else: no standard column
        if !prices_df.getattr("empty")?.is_truthy()? {
            *price = prices_df
                .getattr("loc")?
                .get_item(at)?
                .getattr("iloc")?
                .get_item(0)?;
        }
        Ok(())
    };
    if prices_df.getattr("index")?.contains(target_date)? {
        pick(target_date, &mut price)?;
    } else {
        let r = (|| -> PyResult<()> {
            let ts = pd.getattr("Timestamp")?.call1((target_date,))?;
            if prices_df.getattr("index")?.contains(&ts)? {
                pick(&ts, &mut price)?;
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e)
                if e.is_instance_of::<PyKeyError>(py)
                    || e.is_instance_of::<PyIndexError>(py)
                    || e.is_instance_of::<PyValueError>(py) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(price)
}

/// `_calculate_daily_net_cash_flow`: (net external flow on `target_date` in
/// the target currency, whether an FX lookup failed).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pc_daily_net_cash_flow<'py>(
    py: Python<'py>,
    target_date: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    target_currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    processed_warnings: &Bound<'py, PyAny>,
    included_accounts: &Bound<'py, PyAny>,
    historical_prices_yf_unadjusted: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let _ = (account_currency_map, default_currency, processed_warnings);
    let m = pc(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let daily_tx = transactions_df.get_item(
        transactions_df
            .get_item("Date")?
            .getattr("dt")?
            .getattr("date")?
            .rich_compare(target_date, CompareOp::Eq)?,
    )?;
    if daily_tx.getattr("empty")?.is_truthy()? {
        return Ok((o.fl(0.0), false));
    }
    let mut fx_failed = false;
    let mut net = o.fl(0.0);
    let fmt_date = o.fmt(target_date)?;
    let bridge = |local: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        m.getattr("get_historical_rate_via_usd_bridge")?.call1((
            local,
            target_currency,
            target_date,
            historical_fx_yf,
        ))
    };
    let lookup_price =
        |symbol: &Bound<'py, PyAny>, current: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            if historical_prices_yf_unadjusted.is_truthy()? && internal_to_yf_map.is_truthy()? {
                let yf = internal_to_yf_map.call_method1("get", (symbol,))?;
                if yf.is_truthy()? && historical_prices_yf_unadjusted.contains(&yf)? {
                    let prices_df = historical_prices_yf_unadjusted.get_item(&yf)?;
                    return market_price_on(py, &pd, &prices_df, target_date, current);
                }
            }
            Ok(current)
        };

    // 1. Deposits and withdrawals, cash or not.
    let types = lower_strip_col(&daily_tx.get_item("Type")?)?;
    let external = daily_tx
        .get_item(types.call_method1("isin", (PyList::new(py, ["deposit", "withdrawal"])?,))?)?
        .call_method0("copy")?;
    if !external.getattr("empty")?.is_truthy()? {
        for item in external.call_method0("iterrows")?.try_iter()? {
            let row = unpack(&item?, 2)?.remove(1);
            let tx_type = row
                .get_item("Type")?
                .call_method0("lower")?
                .call_method0("strip")?;
            let symbol = row.get_item("Symbol")?;
            let is_cash = m
                .getattr("is_cash_symbol")?
                .call1((&symbol,))?
                .is_truthy()?;
            let qty = numeric(&pd, &row.get_item("Quantity")?)?;
            let commission_raw = numeric(&pd, &row.call_method1("get", ("Commission",))?)?;
            let commission = if o.isna(&commission_raw)? {
                o.fl(0.0)
            } else {
                o.fl(o.float(&commission_raw)?)
            };
            let local_currency = row.get_item("Local Currency")?;
            let mut flow_local = o.fl(0.0);
            if o.isna(&qty)? {
                continue;
            }
            if is_cash {
                if tx_type.eq("deposit")? {
                    flow_local = o.abs(&qty)?.sub(&commission)?;
                } else if tx_type.eq("withdrawal")? {
                    flow_local = o.abs(&qty)?.neg()?.sub(&commission)?;
                }
            } else {
                let mut price = numeric(&pd, &row.call_method1("get", ("Price/Share",))?)?;
                if o.isna(&price)? || price.le(0)? {
                    price = lookup_price(&symbol, price)?;
                }
                if !o.isna(&price)? && price.gt(0)? {
                    if tx_type.eq("deposit")? {
                        flow_local = o.abs(&qty)?.mul(&price)?.sub(&commission)?;
                    } else if tx_type.eq("withdrawal")? {
                        flow_local = o.abs(&qty)?.mul(&price)?.neg()?.sub(&commission)?;
                    }
                } else {
                    log(
                        py,
                        "warning",
                        format!(
                            "Flow: Could not determine price for asset flow {} on {fmt_date}. Skipping.",
                            o.fmt(&symbol)?
                        ),
                    )?;
                    continue;
                }
            }
            let mut flow_target = flow_local.clone();
            if local_currency.ne(target_currency)? {
                let fx = bridge(&local_currency)?;
                if o.isna(&fx)? {
                    log(
                        py,
                        "warning",
                        format!(
                            "Hist FX Lookup CRITICAL Failure for cash flow: {}->{} on {fmt_date}. Aborting day's flow.",
                            o.fmt(&local_currency)?,
                            o.fmt(target_currency)?
                        ),
                    )?;
                    fx_failed = true;
                    net = o.nan();
                    break;
                }
                flow_target = flow_local.mul(&fx)?;
            }
            if o.isna(&net)? {
                break;
            }
            if o.notna(&flow_target)? {
                net = net.add(&flow_target)?;
            } else {
                net = o.nan();
                fx_failed = true;
                break;
            }
        }
    }
    if fx_failed {
        return Ok((o.nan(), true));
    }

    // 2. Transfers across the selection's edge.
    if included_accounts.is_truthy()? {
        let included = PySet::empty(py)?;
        for acc in included_accounts.try_iter()? {
            included.add(acc?.call_method0("strip")?.call_method0("upper")?)?;
        }
        let transfers = daily_tx
            .get_item(
                lower_strip_col(&daily_tx.get_item("Type")?)?
                    .rich_compare("transfer", CompareOp::Eq)?,
            )?
            .call_method0("copy")?;
        for item in transfers.call_method0("iterrows")?.try_iter()? {
            let row = unpack(&item?, 2)?.remove(1);
            let symbol = row.get_item("Symbol")?;
            let is_cash = m
                .getattr("is_cash_symbol")?
                .call1((&symbol,))?
                .is_truthy()?;
            let account = row.get_item("Account")?;
            let to_account = row.call_method1("get", ("To Account",))?;
            let qty = numeric(&pd, &row.get_item("Quantity")?)?;
            let mut price = o.fl(1.0);
            if !is_cash {
                price = numeric(&pd, &row.call_method1("get", ("Price/Share",))?)?;
            }
            let local_currency = row.get_item("Local Currency")?;
            if o.isna(&qty)? {
                continue;
            }
            if !is_cash && (o.isna(&price)? || price.le(0)?) {
                price = lookup_price(&symbol, price)?;
            }
            if !is_cash && (o.isna(&price)? || price.le(0)?) {
                continue;
            }
            let account_norm = o.str_strip(&account)?.call_method0("upper")?;
            let to_norm = if to_account.is_truthy()? {
                Some(o.str_strip(&to_account)?.call_method0("upper")?)
            } else {
                None
            };
            let from_in = included.contains(&account_norm)?;
            let to_in = match &to_norm {
                Some(t) if t.is_truthy()? => included.contains(t)?,
                _ => false,
            };
            let value = o.abs(&qty)?.mul(&price)?;
            let mut flow = o.fl(0.0);
            if from_in && !to_in {
                flow = value.neg()?;
            } else if !from_in && to_in {
                flow = value;
            }
            if o.abs_gt(&flow, 1e-9)? {
                let mut flow_target = flow.clone();
                if local_currency.ne(target_currency)? {
                    let fx = bridge(&local_currency)?;
                    if o.isna(&fx)? {
                        fx_failed = true;
                        net = o.nan();
                        break;
                    }
                    flow_target = flow.mul(&fx)?;
                }
                if o.notna(&flow_target)? {
                    net = net.add(&flow_target)?;
                }
            }
        }
    }
    Ok((net, fx_failed))
}
