//! `portfolio_logic`'s helpers around the summary: the transactions
//! fingerprint the FIFO cache is keyed on, the default metrics, the published
//! NAVs, the sector/geography enrichment and the historical FX table the
//! holdings pass reads.
//!
//! Ported line for line from `src/portfolio_logic.py` (frozen in
//! `tests/reference/portfolio_logic.py`). They are pandas and dict glue around
//! the market provider, so the pandas operations, the provider calls and the
//! set and dict building are Python's own, in the original's order (a set's
//! iteration order depends on how it was filled). Module state and siblings
//! are looked up on `portfolio_logic` at call time.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::pa_rows::Ops;
use crate::pyhelp::{es, log, pystr, setitem, unpack};

pub(crate) fn pl(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("portfolio_logic")
}

pub(crate) fn kwargs<'py>(
    py: Python<'py>,
    pairs: &[(&str, &Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, *v)?;
    }
    Ok(d)
}

/// `_transactions_fingerprint`: a content hash of `df`, values and index.
#[pyfunction]
pub fn pl_transactions_fingerprint<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let kw = PyDict::new(py);
    kw.set_item("index", true)?;
    let hashed = pd
        .getattr("util")?
        .call_method("hash_pandas_object", (df,), Some(&kw))?
        .getattr("values")?;
    let b = py.import("builtins")?;
    let names = b
        .getattr("map")?
        .call1((b.getattr("str")?, df.getattr("columns")?))?;
    let columns = pystr(py, "|")
        .call_method1("join", (names,))?
        .call_method0("encode")?;
    let data = hashed.call_method0("tobytes")?.add(columns)?;
    let kw = PyDict::new(py);
    kw.set_item("digest_size", 16)?;
    py.import("hashlib")?
        .call_method("blake2b", (data,), Some(&kw))?
        .call_method0("hexdigest")
}

/// `get_default_metrics_dict`: zeros for the empty-but-valid case, NaN for
/// no data or an error.
#[pyfunction]
pub fn pl_default_metrics<'py>(
    py: Python<'py>,
    display_curr_arg: &Bound<'py, PyAny>,
    report_date_arg: &Bound<'py, PyAny>,
    available_accs_list_arg: &Bound<'py, PyAny>,
    is_empty_data_case: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let empty = is_empty_data_case.is_truthy()?;
    let z = if empty { 0.0 } else { f64::NAN };
    let d = PyDict::new(py);
    for k in [
        "market_value",
        "cost_basis_held",
        "unrealized_gain",
        "realized_gain",
        "dividends",
        "commissions",
        "taxes",
        "total_gain",
        "total_cost_invested",
        "total_buy_cost",
    ] {
        d.set_item(k, z)?;
    }
    d.set_item("portfolio_mwr", f64::NAN)?;
    d.set_item("day_change_display", z)?;
    d.set_item("day_change_percent", f64::NAN)?;
    d.set_item(
        "report_date",
        report_date_arg.call_method1("strftime", ("%Y-%m-%d",))?,
    )?;
    d.set_item("display_currency", display_curr_arg)?;
    d.set_item("cumulative_investment", z)?;
    d.set_item("total_return_pct", f64::NAN)?;
    d.set_item("est_annual_income_display", z)?;
    d.set_item("fx_gain_loss_display", z)?;
    d.set_item("fx_gain_loss_pct", f64::NAN)?;
    d.set_item("_available_accounts", available_accs_list_arg)?;
    Ok(d)
}

/// `published_nav_prices`: the newest published NAV per held symbol, as
/// {SYMBOL: (nav_date, nav)}; a store that is not there yet answers nothing.
#[pyfunction]
pub fn pl_published_nav_prices<'py>(
    py: Python<'py>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if !symbols.is_truthy()? {
        return Ok(out);
    }
    let o = Ops::new(py)?;
    let latest = (|| {
        let module = py.import("market_db")?;
        let class = module.getattr("MarketDatabase").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(format!(
                "cannot import name 'MarketDatabase' from 'market_db' ({})",
                module
                    .getattr("__file__")
                    .and_then(|f| o.fmt(&f))
                    .unwrap_or_else(|_| "unknown location".into())
            ))
        })?;
        class.call0()?.call_method0("get_latest_fund_navs")
    })();
    let latest = match latest {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("Fund NAV store unavailable: {}", es(py, &e)),
            )?;
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    if !latest.is_truthy()? {
        return Ok(out);
    }
    let isfinite = py.import("numpy")?.getattr("isfinite")?;
    for symbol in symbols.try_iter()? {
        let symbol = symbol?;
        let key = o
            .str_t
            .call1((&symbol,))?
            .call_method0("upper")?
            .call_method0("strip")?;
        let row = latest.call_method1("get", (key,))?;
        if !row.is_truthy()? {
            continue;
        }
        let parts = unpack(&row, 2)?;
        let (nav_date, nav) = (&parts[0], &parts[1]);
        if !nav.is_none() && isfinite.call1((nav,))?.is_truthy()? && nav.gt(1e-9)? {
            out.set_item(
                &symbol,
                PyTuple::new(py, [nav_date.clone(), o.fl(o.float(nav)?)])?,
            )?;
        }
    }
    Ok(out)
}

/// `_enrich_summary_rows_with_sector_geo`: Sector, quoteType, Country,
/// Industry and exchange on every row, from the manual overrides first, then
/// the batch-prefetched metadata, then a per-symbol fetch.
#[pyfunction]
pub fn pl_enrich_sector_geo<'py>(
    py: Python<'py>,
    portfolio_summary_rows: &Bound<'py, PyAny>,
    market_provider: &Bound<'py, PyAny>,
    manual_overrides_effective: &Bound<'py, PyAny>,
    effective_user_symbol_map: &Bound<'py, PyAny>,
    effective_user_excluded_symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !portfolio_summary_rows.is_truthy()? {
        return Ok(portfolio_summary_rows.clone());
    }
    let m = pl(py)?;
    let pd = py.import("pandas")?;
    let tmp = pd.getattr("DataFrame")?.call1((portfolio_summary_rows,))?;
    if !tmp.getattr("columns")?.contains("Symbol")? {
        return Ok(portfolio_summary_rows.clone());
    }
    let symbols: Vec<Bound<'py, PyAny>> = tmp
        .get_item("Symbol")?
        .call_method0("unique")?
        .try_iter()?
        .collect::<PyResult<_>>()?;
    let sector_map = PyDict::new(py);
    let quote_type_map = PyDict::new(py);
    let country_map = PyDict::new(py);
    let industry_map = PyDict::new(py);
    let exchange_map = PyDict::new(py);

    let is_cash_row = |s: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(s.eq(m.getattr("CASH_SYMBOL_CSV")?)?
            || s.call_method1("startswith", ("Cash (",))?.is_truthy()?)
    };

    // Batch pre-fetch metadata for every symbol at once.
    let yf_symbols = PySet::empty(py)?;
    let internal_to_yf = PyDict::new(py);
    for s in &symbols {
        if is_cash_row(s)? {
            continue;
        }
        let yf = m.getattr("map_to_yf_symbol")?.call1((
            s,
            effective_user_symbol_map,
            effective_user_excluded_symbols,
        ))?;
        if yf.is_truthy()? {
            yf_symbols.add(&yf)?;
            internal_to_yf.set_item(s, &yf)?;
        }
    }
    let provider_ok = || -> PyResult<bool> {
        Ok(m.getattr("MARKET_PROVIDER_AVAILABLE")?.is_truthy()? && market_provider.is_truthy()?)
    };
    let mut batch_metadata = PyDict::new(py).into_any();
    if yf_symbols.is_truthy()? && provider_ok()? {
        match market_provider.call_method1("_ensure_metadata_batch", (&yf_symbols,)) {
            Ok(v) => batch_metadata = v,
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "warning",
                format!("Batch metadata pre-fetch failed: {}", es(py, &e)),
            )?,
            Err(e) => return Err(e),
        }
    }

    let o = Ops::new(py)?;
    let override_str = |ov: &Bound<'py, PyAny>, k: &str| -> PyResult<Bound<'py, PyAny>> {
        o.get(ov, k, pystr(py, ""))?.call_method0("strip")
    };
    for s in &symbols {
        let ov = manual_overrides_effective
            .call_method1("get", (s.call_method0("upper")?, PyDict::new(py)))?;
        let manual_asset_type = override_str(&ov, "asset_type")?;
        let manual_sector = override_str(&ov, "sector")?;
        let manual_geography = override_str(&ov, "geography")?;
        let manual_industry = override_str(&ov, "industry")?;
        let manual_exchange = override_str(&ov, "exchange")?;

        if is_cash_row(s)? {
            sector_map.set_item(s, "Cash")?;
            quote_type_map.set_item(s, "CASH")?;
            country_map.set_item(s, "Cash")?;
            industry_map.set_item(s, "Cash")?;
            exchange_map.set_item(s, "Cash")?;
            if manual_exchange.is_truthy()? {
                exchange_map.set_item(s, &manual_exchange)?;
            }
            continue;
        }
        if s.eq("SCBRM1")? {
            log(
                py,
                "info",
                format!(
                    "DEBUG_OVERRIDE_SCBRM1: ManualExchange={}, AllOverrides={}",
                    o.fmt(&manual_exchange)?,
                    o.fmt(&ov)?
                ),
            )?;
        }
        let yf = internal_to_yf.call_method1("get", (s,))?;
        let or_default = |manual: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            Ok(if manual.is_truthy()? {
                manual.clone()
            } else {
                pystr(py, "N/A (No YF/Manual)")
            })
        };
        sector_map.set_item(s, or_default(&manual_sector)?)?;
        quote_type_map.set_item(s, or_default(&manual_asset_type)?)?;
        country_map.set_item(s, or_default(&manual_geography)?)?;
        industry_map.set_item(s, or_default(&manual_industry)?)?;
        if manual_exchange.is_truthy()? {
            exchange_map.set_item(s, &manual_exchange)?;
        }

        let (ms, mat, mg, mi) = (
            manual_sector.is_truthy()?,
            manual_asset_type.is_truthy()?,
            manual_geography.is_truthy()?,
            manual_industry.is_truthy()?,
        );
        if yf.is_truthy()? && (!ms || !mat || !mg || !mi) {
            let mut info = batch_metadata.call_method1("get", (&yf,))?;
            if !info.is_truthy()? && provider_ok()? {
                info = market_provider.call_method1("get_fundamental_data", (&yf,))?;
            }
            if info.is_truthy()? && info.is_instance_of::<PyDict>() {
                if !ms {
                    sector_map.set_item(s, o.get(&info, "sector", pystr(py, "Unknown Sector"))?)?;
                }
                if !mat {
                    quote_type_map.set_item(s, o.get(&info, "quoteType", pystr(py, "UNKNOWN"))?)?;
                }
                if !mg {
                    let c = o.get(&info, "country", pystr(py, "Unknown Region"))?;
                    let c = if c.is_truthy()? {
                        c
                    } else {
                        pystr(py, "Unknown Region")
                    };
                    country_map.set_item(s, c)?;
                }
                if !mi {
                    industry_map
                        .set_item(s, o.get(&info, "industry", pystr(py, "Unknown Industry"))?)?;
                }
            } else {
                if !ms {
                    sector_map.set_item(s, "N/A (Fetch Error)")?;
                }
                if !mat {
                    quote_type_map.set_item(s, "N/A (Fetch Error)")?;
                }
                if !mg {
                    country_map.set_item(s, "Unknown Region")?;
                }
                if !mi {
                    industry_map.set_item(s, "N/A (Fetch Error)")?;
                }
            }
        }
    }

    let mapped = |map: &Bound<'py, PyDict>, fill: &str| -> PyResult<Bound<'py, PyAny>> {
        tmp.get_item("Symbol")?
            .call_method1("map", (map,))?
            .call_method1("fillna", (fill,))
    };
    setitem(&tmp, "Sector", mapped(&sector_map, "Unknown Sector")?)?;
    setitem(&tmp, "quoteType", mapped(&quote_type_map, "UNKNOWN")?)?;
    setitem(&tmp, "Country", mapped(&country_map, "Unknown Region")?)?;
    setitem(&tmp, "Industry", mapped(&industry_map, "Unknown Industry")?)?;
    let existing = if tmp.getattr("columns")?.contains("exchange")? {
        tmp.get_item("exchange")?
    } else {
        let nones = PyList::new(py, (0..tmp.len()?).map(|_| py.None()))?;
        pd.getattr("Series")?.call(
            (nones,),
            Some(&kwargs(py, &[("index", &tmp.getattr("index")?)])?),
        )?
    };
    let exchange = tmp
        .get_item("Symbol")?
        .call_method1("map", (&exchange_map,))?
        .call_method1("combine_first", (existing,))?;
    setitem(&tmp, "exchange", exchange)?;
    let kw = PyDict::new(py);
    kw.set_item("orient", "records")?;
    tmp.call_method("to_dict", (), Some(&kw))
}

/// `str(c).strip().upper()` for each `c` that names a three-letter currency,
/// in the order the original's set comprehension met them.
pub(crate) fn clean_currencies<'py>(
    o: &Ops<'py>,
    currencies: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    let py = o.py;
    let out = PySet::empty(py)?;
    let junk = PyList::new(py, ["", "<NA>", "NAN", "NONE", "N/A"])?;
    for c in currencies.try_iter()? {
        let c = c?;
        if !o.notna(&c)? || junk.contains(o.str_strip(&c)?)? || o.str_strip(&c)?.len()? != 3 {
            continue;
        }
        out.add(o.str_strip(&c)?.call_method0("upper")?)?;
    }
    Ok(out)
}

/// `_prepare_historical_fx_rates`: ({(date, currency): display per local},
/// the USD-based tables, whether some history could not be fetched).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pl_prepare_historical_fx<'py>(
    py: Python<'py>,
    transactions_df_filtered: &Bound<'py, PyAny>,
    all_transactions_df_cleaned: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    market_provider: &Bound<'py, PyAny>,
    cache_file_path: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>, bool)> {
    let m = pl(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let b = py.import("builtins")?;
    let mut usd_based = PyDict::new(py).into_any();
    let mut fx_warning = false;
    let out = PyDict::new(py);
    if transactions_df_filtered.getattr("empty")?.is_truthy()? {
        return Ok((out, usd_based, fx_warning));
    }
    log(
        py,
        "debug",
        "Preparing historical FX rates for transaction processing (current summary)...",
    )?;
    let unique_dates = b.getattr("sorted")?.call1((transactions_df_filtered
        .get_item("Date")?
        .getattr("dt")?
        .getattr("date")?
        .call_method0("unique")?,))?;

    // Currencies of every account, excluded ones too (transfers need them).
    let currencies = b.getattr("set")?.call1((all_transactions_df_cleaned
        .get_item("Local Currency")?
        .call_method0("unique")?,))?;
    currencies.call_method1("add", (display_currency,))?;
    currencies.call_method1("add", (default_currency,))?;
    let cleaned = clean_currencies(&o, &currencies)?;

    let min_tx_date = all_transactions_df_cleaned
        .get_item("Date")?
        .call_method0("min")?
        .call_method0("date")?;
    let pairs = PyList::empty(py);
    for lc in cleaned.iter() {
        if !lc.is_truthy()?
            || o.str_strip(&lc)?.eq("")?
            || lc.call_method0("upper")?.eq("USD")?
            || o.isna(&lc)?
        {
            continue;
        }
        let code = lc.call_method0("upper")?;
        if code.eq("THB")? {
            pairs.append("USDTHB=X")?;
        } else {
            pairs.append(format!("{}=X", o.fmt(&code)?))?;
        }
    }

    let provider = if market_provider.is_truthy()? {
        market_provider.clone()
    } else {
        m.getattr("MarketDataProvider")?.call(
            (),
            Some(&kwargs(py, &[("current_cache_file", cache_file_path)])?),
        )?
    };
    // The capital-gains view reckons to today, so the summary does too.
    let end_date = m.getattr("get_est_today")?.call0()?;
    let set_list = || -> PyResult<Bound<'py, PyAny>> {
        b.getattr("list")?
            .call1((b.getattr("set")?.call1((&pairs,))?,))
    };
    let fx_pairs = set_list()?;
    let sorted_pairs = b.getattr("sorted")?.call1((set_list()?,))?;
    let cache_key = format!(
        "PROC_FX_HIST_{}",
        pystr(py, "_")
            .call_method1("join", (sorted_pairs,))?
            .extract::<String>()?
    );
    let true_ = pyo3::types::PyBool::new(py, true).to_owned().into_any();
    let result = provider.call_method(
        "get_historical_fx_rates",
        (),
        Some(&kwargs(
            py,
            &[
                ("fx_pairs_yf", &fx_pairs),
                ("start_date", &min_tx_date),
                ("end_date", &end_date),
                ("use_cache", &true_),
                ("cache_key", &pystr(py, &cache_key)),
            ],
        )?),
    )?;
    let parts = unpack(&result, 2)?;
    usd_based = parts[0].clone();
    if parts[1].is_truthy()? {
        log(
            py,
            "warning",
            "Failed to fetch some historical FX rates needed for precise FX G/L calculation in current summary. FX G/L might be inaccurate.",
        )?;
        fx_warning = true;
    }

    let series_list = PyList::empty(py);
    if usd_based.is_truthy()? {
        for item in usd_based.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let (pair, df) = (&kv[0], &kv[1]);
            if df.is_none() || df.getattr("empty")?.is_truthy()? {
                continue;
            }
            let code = pair.call_method1("replace", ("=X", ""))?;
            if !df
                .getattr("index")?
                .is_instance(&pd.getattr("DatetimeIndex")?)?
            {
                df.setattr(
                    "index",
                    pd.call_method1("to_datetime", (df.getattr("index")?,))?,
                )?;
            }
            let columns = df.getattr("columns")?;
            let series = if columns.contains("Close")? {
                df.get_item("Close")?.call_method1("rename", (&code,))?
            } else if columns.contains("price")? {
                df.get_item("price")?.call_method1("rename", (&code,))?
            } else {
                log(
                    py,
                    "warning",
                    format!(
                        "Warning: neither 'Close' nor 'price' column found in historical FX for {}. Columns: {}",
                        o.fmt(pair)?,
                        o.fmt(&columns)?
                    ),
                )?;
                continue;
            };
            series.setattr("index", series.getattr("index")?.getattr("date")?)?;
            let kw = PyDict::new(py);
            kw.set_item("keep", "last")?;
            let dup = series
                .getattr("index")?
                .call_method("duplicated", (), Some(&kw))?
                .call_method0("__invert__")?;
            series_list.append(series.get_item(dup)?)?;
        }
    }

    if series_list.is_truthy()? {
        let kw = PyDict::new(py);
        kw.set_item("axis", 1)?;
        let mut master = pd.call_method("concat", (&series_list,), Some(&kw))?;
        let kw = PyDict::new(py);
        kw.set_item("inplace", true)?;
        master.call_method("sort_index", (), Some(&kw))?;
        master = master.call_method0("ffill")?.call_method0("bfill")?;

        // Normalise the column names (USDTHB=X -> THB).
        let rename_map = PyDict::new(py);
        for col in master.getattr("columns")?.try_iter()? {
            let col = col?;
            let clean = col
                .call_method1("replace", ("=X", ""))?
                .call_method0("upper")?;
            let final_code = if clean.len()? == 6 {
                if clean.call_method1("startswith", ("USD",))?.is_truthy()? {
                    clean.get_item(pyo3::types::PySlice::new(py, 3, isize::MAX, 1))?
                } else if clean.call_method1("endswith", ("USD",))?.is_truthy()? {
                    clean.get_item(pyo3::types::PySlice::new(py, 0, 3, 1))?
                } else {
                    clean
                }
            } else {
                clean
            };
            if final_code.ne(&col)? {
                rename_map.set_item(&col, final_code)?;
            }
        }
        if rename_map.is_truthy()? {
            let kw = PyDict::new(py);
            kw.set_item("columns", &rename_map)?;
            kw.set_item("inplace", true)?;
            master.call_method("rename", (), Some(&kw))?;
            log(
                py,
                "debug",
                format!("Normalized FX columns: {}", o.fmt(&rename_map)?),
            )?;
        }

        // Cover every transaction date (ffill and bfill).
        let set = b.getattr("set")?;
        let all_needed = b.getattr("sorted")?.call1((b.getattr("list")?.call1((set
            .call1((master.getattr("index")?,))?
            .bitor(set.call1((&unique_dates,))?)?,))?,))?;
        master = master
            .call_method1("reindex", (all_needed,))?
            .call_method0("ffill")?
            .call_method0("bfill")?;

        // A currency with no history at all takes today's rate as a constant.
        let existing = set.call1((master.getattr("columns")?,))?;
        let missing = PyList::empty(py);
        for c in cleaned.iter() {
            if c.is_truthy()? && c.call_method0("upper")?.ne("USD")? && !existing.contains(&c)? {
                missing.append(c)?;
            }
        }
        if missing.is_truthy()? {
            log(
                py,
                "warning",
                format!(
                    "Historical FX missing for {}. Attempting to fetch current rates as fallback.",
                    o.fmt(&missing)?
                ),
            )?;
            let missing_pairs = PyList::empty(py);
            for c in missing.iter() {
                missing_pairs.append(format!("{}=X", o.fmt(&c.call_method0("upper")?)?))?;
            }
            let r = (|| -> PyResult<()> {
                let quotes = provider.call_method(
                    "get_current_quotes",
                    (),
                    Some(&kwargs(
                        py,
                        &[
                            ("internal_stock_symbols", &PyList::empty(py).into_any()),
                            ("required_currencies", &set.call1((&missing_pairs,))?),
                            ("user_symbol_map", &PyDict::new(py).into_any()),
                            ("user_excluded_symbols", &set.call0()?),
                        ],
                    )?),
                )?;
                let q = unpack(&quotes, 5)?;
                let current = &q[1];
                for curr in missing.iter() {
                    let cu = curr.call_method0("upper")?;
                    if current.contains(&cu)? && o.notna(&current.get_item(&cu)?)? {
                        let rate = current.get_item(&cu)?;
                        log(
                            py,
                            "info",
                            format!(
                                "Using current FX rate {} for {} as historical fallback (constant).",
                                o.fmt(&rate)?,
                                o.fmt(&curr)?
                            ),
                        )?;
                        setitem(&master, &curr, rate)?;
                    } else {
                        log(
                            py,
                            "error",
                            format!(
                                "Could not get even current FX rate for {}. FX conversion will fail/default.",
                                o.fmt(&curr)?
                            ),
                        )?;
                    }
                }
                Ok(())
            })();
            match r {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "error",
                    format!("Error fetching current FX fallback: {}", es(py, &e)),
                )?,
                Err(e) => return Err(e),
            }
        }

        setitem(&master, "USD", 1.0)?;
        let display_col = display_currency.call_method0("upper")?;
        if !master.getattr("columns")?.contains(&display_col)? {
            if display_col.eq("USD")? {
                setitem(&master, &display_col, 1.0)?;
            } else {
                log(
                    py,
                    "warning",
                    format!(
                        "Display currency {} not found in FX data. FX conversion may fail.",
                        o.fmt(&display_col)?
                    ),
                )?;
            }
        }
        if master.getattr("columns")?.contains(&display_col)? {
            let display_rates = master.get_item(&display_col)?;
            let one = o.fl(1.0);
            for curr_col in master.getattr("columns")?.try_iter()? {
                let curr_col = curr_col?;
                let r = (|| -> PyResult<()> {
                    // Every rate as local per 1 USD, so display/local divides.
                    let raw_rate = master.get_item(&curr_col)?;
                    let avg_rate = raw_rate.call_method0("mean")?;
                    let mut is_major = false;
                    for c in ["EUR", "GBP", "AUD", "NZD"] {
                        if curr_col.eq(c)? {
                            is_major = true;
                            break;
                        }
                    }
                    // f"Vectorized FX: Inverting {kind}{curr_col} (Avg=...) to {to}."
                    let note = |kind: &str, to: &str| -> PyResult<()> {
                        log(
                            py,
                            "debug",
                            format!(
                                "Vectorized FX: Inverting {kind}{} (Avg={}) to {to}.",
                                o.fmt(&curr_col)?,
                                o.fmt_spec(&avg_rate, ".4f")?
                            ),
                        )
                    };
                    let rate_to_use = if is_major && avg_rate.gt(1.0)? {
                        let r = one.div(&raw_rate)?;
                        note("major ", "Local/USD")?;
                        r
                    } else if curr_col.eq("THB")? && avg_rate.lt(1.0)? {
                        let r = one.div(&raw_rate)?;
                        note("", "THB/USD")?;
                        r
                    } else if avg_rate.lt(0.1)? {
                        let r = one.div(&raw_rate)?;
                        note("low-rate ", "Local/USD")?;
                        r
                    } else {
                        raw_rate
                    };
                    let cross = display_rates.div(rate_to_use)?;
                    let keep = cross
                        .getattr("index")?
                        .call_method1("isin", (&unique_dates,))?;
                    let relevant = cross.getattr("loc")?.get_item(keep)?;
                    for item in relevant.call_method0("items")?.try_iter()? {
                        let kv = unpack(&item?, 2)?;
                        if o.notna(&kv[1])? {
                            out.set_item(
                                PyTuple::new(py, [kv[0].clone(), curr_col.clone()])?,
                                o.float(&kv[1])?,
                            )?;
                        }
                    }
                    Ok(())
                })();
                match r {
                    Ok(()) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => log(
                        py,
                        "warning",
                        format!(
                            "Error calculating vectorized FX for {}: {}",
                            o.fmt(&curr_col)?,
                            es(py, &e)
                        ),
                    )?,
                    Err(e) => return Err(e),
                }
            }
        }
    }

    // USD is always 1.
    for d in unique_dates.try_iter()? {
        let key = PyTuple::new(py, [d?, pystr(py, "USD")])?;
        if !out.contains(&key)? {
            out.set_item(key, 1.0)?;
        }
    }
    Ok((out, usd_based, fx_warning))
}
