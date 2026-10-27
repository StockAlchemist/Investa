//! `portfolio_history`'s input preparation and daily valuation: the account
//! filter and cache keys (`_prepare_historical_inputs`), the L1 daily-holdings
//! cache around the replay kernel (`_get_or_calculate_all_daily_holdings`),
//! and the price/FX alignment around the valuation kernel
//! (`_value_daily_holdings_vectorized`).
//!
//! Ported line for line from `src/portfolio_history.py` (frozen in
//! `tests/reference/portfolio_history.py`). The kernels were already Rust;
//! what is ported here is the pandas, NumPy, hashing and file glue around
//! them, so every operation is Python's own, in the original's order.
//! Siblings and module state are looked up on `portfolio_history` at call
//! time.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCFunction, PyDict, PyList, PySet, PySlice, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::ph_small::ph;
use crate::pl_helpers::{clean_currencies, kwargs};
use crate::pyhelp::{es, log, pystr, setitem, unpack};

fn tru(py: Python<'_>) -> Bound<'_, PyAny> {
    PyBool::new(py, true).to_owned().into_any()
}

/// `", ".join(items)`
fn join<'py>(py: Python<'py>, sep: &str, items: &Bound<'py, PyAny>) -> PyResult<String> {
    pystr(py, sep).call_method1("join", (items,))?.extract()
}

/// `s[:n]` on a Python str.
fn prefix<'py>(s: &Bound<'py, PyAny>, n: isize) -> PyResult<Bound<'py, PyAny>> {
    s.get_item(PySlice::new(s.py(), 0, n, 1))
}

/// `_prepare_historical_inputs`: the effective ledger, the account lists,
/// the symbols and FX pairs to fetch, the splits, and the cache keys.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_prepare_historical_inputs<'py>(
    py: Python<'py>,
    preloaded_transactions_df: &Bound<'py, PyAny>,
    original_transactions_df_for_ignored: &Bound<'py, PyAny>,
    ignored_indices_from_load: &Bound<'py, PyAny>,
    ignored_reasons_from_load: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    exclude_accounts: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    benchmark_symbols_yf: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    current_hist_version: &Bound<'py, PyAny>,
    raw_cache_prefix: &Bound<'py, PyAny>,
    daily_cache_prefix: &Bound<'py, PyAny>,
    user_symbol_map: &Bound<'py, PyAny>,
    user_excluded_symbols: &Bound<'py, PyAny>,
    original_csv_file_path: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let b = py.import("builtins")?;
    let none = || py.None().into_bound(py);
    let set = b.getattr("set")?;
    let sorted = b.getattr("sorted")?;
    let list = b.getattr("list")?;
    let normalize = m.getattr("_normalize_series")?;
    let empty_list = || PyList::empty(py).into_any();
    let empty_dict = || PyDict::new(py).into_any();
    let empty_return = || -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                none(),
                none(),
                PySet::empty(py)?.into_any(),
                empty_dict(),
                empty_list(),
                empty_list(),
                empty_list(),
                empty_list(),
                empty_list(),
                empty_dict(),
                empty_dict(),
                empty_dict(),
                pystr(py, ""),
                pystr(py, ""),
                none(),
                none(),
                pystr(py, ""),
            ],
        )
    };

    let symbol_map = if user_symbol_map.is_none() {
        empty_dict()
    } else {
        user_symbol_map.clone()
    };
    let excluded_symbols = if user_excluded_symbols.is_none() {
        PySet::empty(py)?.into_any()
    } else {
        user_excluded_symbols.clone()
    };
    let df = preloaded_transactions_df;
    let ignored_indices = if ignored_indices_from_load.is_truthy()? {
        ignored_indices_from_load.call_method0("copy")?
    } else {
        PySet::empty(py)?.into_any()
    };
    let ignored_reasons = if ignored_reasons_from_load.is_truthy()? {
        ignored_reasons_from_load.call_method0("copy")?
    } else {
        empty_dict()
    };
    if df.is_none() {
        log(
            py,
            "error",
            "ERROR in _prepare_historical_inputs: No preloaded transaction data provided.",
        )?;
        return empty_return();
    }
    let has_col =
        |d: &Bound<'py, PyAny>, c: &str| -> PyResult<bool> { d.getattr("columns")?.contains(c) };

    let all_accounts = if has_col(df, "Account")? && !df.getattr("empty")?.is_truthy()? {
        sorted.call1((df
            .get_item("Account")?
            .call_method0("unique")?
            .call_method0("tolist")?,))?
    } else {
        empty_list()
    };
    let available_set = set.call1((&all_accounts,))?;
    let included_sorted;
    let mut excluded_sorted = empty_list();
    let mut filter_desc = String::from("All Accounts");
    let mut valid_include: Option<Bound<'py, PyList>> = None;

    let included_df = if !include_accounts.is_truthy()? {
        included_sorted = sorted.call1((list.call1((&available_set,))?,))?;
        df.call_method0("copy")?
    } else {
        let available = set.call1((normalize
            .call1((df.get_item("Account")?,))?
            .call_method0("unique")?,))?;
        if has_col(df, "To Account")? {
            available.call_method1(
                "update",
                (normalize
                    .call1((df.get_item("To Account")?,))?
                    .call_method0("dropna")?
                    .call_method0("unique")?,),
            )?;
        }
        let normalized = PyList::empty(py);
        for acc in include_accounts.try_iter()? {
            normalized.append(acc?.call_method0("strip")?.call_method0("upper")?)?;
        }
        let valid = PyList::empty(py);
        for acc in normalized.iter() {
            if available.contains(&acc)? {
                valid.append(acc)?;
            }
        }
        if !valid.is_truthy()? {
            log(
                py,
                "warning",
                "WARN in _prepare_historical_inputs: No valid accounts to include.",
            )?;
            return empty_return();
        }

        // The selection plus the history of anything transferred into it.
        let index_kw = kwargs(py, &[("index", &df.getattr("index")?)])?;
        let series_false = || pd.getattr("Series")?.call((false,), Some(&index_kw));
        let from_mask = normalize
            .call1((df.get_item("Account")?,))?
            .call_method1("isin", (&valid,))?;
        let mut to_mask = series_false()?;
        if has_col(df, "To Account")? {
            to_mask = normalize
                .call1((df.get_item("To Account")?,))?
                .call_method1("isin", (&valid,))?
                .call_method1("fillna", (false,))?;
        }
        let mut preceding = series_false()?;
        let transfers_in = df.get_item(&to_mask)?;
        if !transfers_in.getattr("empty")?.is_truthy()? {
            let events = transfers_in
                .get_item(PyList::new(py, ["Symbol", "Account", "Date"])?)?
                .call_method0("drop_duplicates")?;
            let masks = PyList::empty(py);
            for item in events.call_method0("iterrows")?.try_iter()? {
                let ev = unpack(&item?, 2)?.remove(1);
                let symbol = ev.get_item("Symbol")?;
                let from_account = ev.get_item("Account")?;
                let date = ev.get_item("Date")?;
                let mask = df
                    .get_item("Symbol")?
                    .rich_compare(&symbol, CompareOp::Eq)?
                    .bitand(
                        df.get_item("Account")?
                            .rich_compare(&from_account, CompareOp::Eq)?,
                    )?
                    .bitand(df.get_item("Date")?.rich_compare(&date, CompareOp::Le)?)?;
                masks.append(mask)?;
            }
            if masks.is_truthy()? {
                let kw = kwargs(py, &[("axis", &1i64.into_pyobject(py)?.into_any())])?;
                preceding = pd
                    .call_method("concat", (&masks,), Some(&kw))?
                    .call_method("any", (), Some(&kw))?;
            }
        }
        let split_mask = df
            .get_item("Type")?
            .call_method1("astype", (&o.str_t,))?
            .getattr("str")?
            .call_method0("lower")?
            .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?;
        let all_accounts_mask = normalize
            .call1((df.get_item("Account")?,))?
            .rich_compare("ALL ACCOUNTS", CompareOp::Eq)?;
        let combined = from_mask
            .bitor(&to_mask)?
            .bitor(&preceding)?
            .bitor(&split_mask)?
            .bitor(&all_accounts_mask)?;
        let included = df.get_item(combined)?.call_method0("copy")?;
        included_sorted = sorted.call1((&valid,))?;
        filter_desc = format!("Included: {}", join(py, ", ", &included_sorted)?);
        valid_include = Some(valid);
        included
    };

    let effective =
        if !exclude_accounts.is_truthy()? || !exclude_accounts.is_instance_of::<PyList>() {
            included_df.call_method0("copy")?
        } else {
            let available = set.call1((df.get_item("Account")?.call_method0("unique")?,))?;
            if has_col(df, "To Account")? {
                available.call_method1(
                    "update",
                    (df.get_item("To Account")?
                        .call_method0("dropna")?
                        .call_method0("unique")?,),
                )?;
            }
            let valid_exclude = PyList::empty(py);
            for acc in exclude_accounts.try_iter()? {
                let acc = acc?;
                if available.contains(&acc)? {
                    valid_exclude.append(acc)?;
                }
            }
            if valid_exclude.is_truthy()? {
                log(
                    py,
                    "info",
                    format!(
                        "Hist Prep: Excluding accounts: {}",
                        join(py, ", ", &sorted.call1((&valid_exclude,))?)?
                    ),
                )?;
                let without = |accounts: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
                    included_df
                        .get_item(
                            included_df
                                .get_item("Account")?
                                .call_method1("isin", (accounts,))?
                                .call_method0("__invert__")?,
                        )?
                        .call_method0("copy")
                };
                // With a selection, only accounts in it are excluded; their
                // transfer sources stay.
                let effective = if include_accounts.is_truthy()? {
                    let really = set.call1((&valid_exclude,))?.call_method1(
                        "intersection",
                        (set.call1((valid_include.as_ref().expect("set when including"),))?,),
                    )?;
                    if really.is_truthy()? {
                        without(&really)?
                    } else {
                        included_df.call_method0("copy")?
                    }
                } else {
                    without(valid_exclude.as_any())?
                };
                excluded_sorted = sorted.call1((&valid_exclude,))?;
                let names = join(py, ", ", &excluded_sorted)?;
                if include_accounts.is_truthy()? {
                    filter_desc += &format!(" (Excluding: {names})");
                } else {
                    filter_desc = format!("All Accounts (Excluding: {names})");
                }
                effective
            } else {
                included_df.call_method0("copy")?
            }
        };

    if effective.getattr("empty")?.is_truthy()? {
        log(
            py,
            "warning",
            "WARN in _prepare_historical_inputs: No transactions remain after filtering.",
        )?;
        return PyTuple::new(
            py,
            [
                effective,
                original_transactions_df_for_ignored.clone(),
                ignored_indices,
                ignored_reasons,
                all_accounts,
                included_sorted,
                excluded_sorted,
                empty_list(),
                empty_list(),
                empty_dict(),
                empty_dict(),
                empty_dict(),
                pystr(py, ""),
                pystr(py, ""),
                none(),
                none(),
                pystr(py, &filter_desc),
            ],
        );
    }

    // Splits from the unfiltered ledger, so a filtered-out account's split counts.
    let coerce = kwargs(py, &[("errors", &pystr(py, "coerce"))])?;
    let split_tx = if has_col(df, "Split Ratio")? {
        let mask = df
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?
            .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?
            .bitand(df.get_item("Split Ratio")?.call_method0("notna")?)?
            .bitand(
                pd.call_method("to_numeric", (df.get_item("Split Ratio")?,), Some(&coerce))?
                    .rich_compare(0, CompareOp::Gt)?,
            )?;
        df.get_item(mask)?.call_method(
            "sort_values",
            (),
            Some(&kwargs(
                py,
                &[("by", &pystr(py, "Date")), ("ascending", &tru(py))],
            )?),
        )?
    } else {
        pd.getattr("DataFrame")?.call0()?
    };
    let splits_by_symbol = PyDict::new(py);
    if !split_tx.getattr("empty")?.is_truthy()? {
        let unique = split_tx.call_method(
            "drop_duplicates",
            (),
            Some(&kwargs(
                py,
                &[("subset", PyList::new(py, ["Symbol", "Date"])?.as_any())],
            )?),
        )?;
        // lambda r: {"Date": r["Date"].date(), "Split Ratio": float(r["Split Ratio"])}
        let to_split =
            PyCFunction::new_closure(py, None, None, |args, _| -> PyResult<Py<PyAny>> {
                let py = args.py();
                let r = args.get_item(0)?;
                let d = PyDict::new(py);
                d.set_item("Date", r.get_item("Date")?.call_method0("date")?)?;
                d.set_item(
                    "Split Ratio",
                    py.import("builtins")?
                        .getattr("float")?
                        .call1((r.get_item("Split Ratio")?,))?,
                )?;
                Ok(d.into_any().unbind())
            })?;
        let axis = kwargs(py, &[("axis", &1i64.into_pyobject(py)?.into_any())])?;
        for item in unique.call_method1("groupby", ("Symbol",))?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let rows = kv[1]
                .get_item(PyList::new(py, ["Date", "Split Ratio"])?)?
                .call_method("apply", (&to_split,), Some(&axis))?
                .call_method0("tolist")?;
            splits_by_symbol.set_item(&kv[0], rows)?;
        }
    }

    // Every symbol in the effective ledger, mapped to its feed.
    let symbols_portfolio = PyList::empty(py);
    let internal_to_yf = PyDict::new(py);
    for s in effective
        .get_item("Symbol")?
        .call_method0("unique")?
        .call_method0("tolist")?
        .try_iter()?
    {
        let s = s?;
        if m.getattr("is_cash_symbol")?.call1((&s,))?.is_truthy()? {
            continue;
        }
        let yf = m
            .getattr("map_to_yf_symbol")?
            .call1((&s, &symbol_map, &excluded_symbols))?;
        if yf.is_truthy()? {
            symbols_portfolio.append(&yf)?;
            internal_to_yf.set_item(&s, &yf)?;
        }
    }
    let yf_to_internal = PyDict::new(py);
    for (k, v) in internal_to_yf.iter() {
        yf_to_internal.set_item(v, k)?;
    }
    let symbols_portfolio = sorted.call1((list.call1((set.call1((&symbols_portfolio,))?,))?,))?;
    let symbols_all = sorted.call1((list.call1((set
        .call1((&symbols_portfolio,))?
        .bitor(set.call1((benchmark_symbols_yf,))?)?,))?,))?;

    let currencies = set.call1((effective
        .get_item("Local Currency")?
        .call_method0("unique")?,))?;
    let wanted = PySet::new(py, [display_currency, default_currency])?;
    let needed = clean_currencies(&o, &currencies.call_method1("union", (wanted,))?)?;
    let pairs = PyList::empty(py);
    for curr in needed.iter() {
        if !curr.is_truthy()? || curr.call_method0("upper")?.eq("USD")? {
            continue;
        }
        let upper = curr.call_method0("upper")?;
        if upper.eq("THB")? {
            pairs.append("USDTHB=X")?;
            pairs.append("THB=X")?;
        } else {
            pairs.append(format!("{}=X", o.fmt(&upper)?))?;
        }
    }
    let fx_pairs = sorted.call1((list.call1((set.call1((&pairs,))?,))?,))?;

    // Cache paths and keys.
    let os_path = m.getattr("os")?.getattr("path")?;
    let app_cache_dir = m.getattr("config")?.call_method0("get_app_cache_dir")?;
    let in_cache_dir = |name: &str| -> PyResult<Bound<'py, PyAny>> {
        if app_cache_dir.is_truthy()? {
            os_path.call_method1("join", (&app_cache_dir, name))
        } else {
            Ok(pystr(py, name))
        }
    };
    let raw_file_name = format!(
        "{}_{}_{}.json",
        o.fmt(raw_cache_prefix)?,
        o.fmt(&start_date.call_method0("isoformat")?)?,
        o.fmt(&end_date.call_method0("isoformat")?)?
    );
    let raw_file = in_cache_dir(&raw_file_name)?;
    let raw_key = format!(
        "ADJUSTED_v7::{}::{}",
        join(py, "_", &sorted.call1((&symbols_all,))?)?,
        join(py, "_", &fx_pairs)?
    );

    let mut daily_file = none();
    let mut daily_key = none();
    let keyed = (|| -> PyResult<()> {
        let hashlib = m.getattr("hashlib")?;
        let json = m.getattr("json")?;
        // The key changes whenever the transactions do.
        let tx_hash = if original_csv_file_path.is_truthy()?
            && os_path
                .call_method1("exists", (original_csv_file_path,))?
                .is_truthy()?
        {
            m.getattr("_get_file_hash")?
                .call1((original_csv_file_path,))?
        } else {
            if original_csv_file_path.is_truthy()? {
                log(
                    py,
                    "warning",
                    format!(
                        "Hist Prep: Original CSV path '{}' for hash not found. Hashing DataFrame contents instead.",
                        o.fmt(original_csv_file_path)?
                    ),
                )?;
            }
            let hash_cols = PyList::empty(py);
            for c in [
                "Date",
                "Type",
                "Symbol",
                "Quantity",
                "Price/Share",
                "Total Amount",
                "Commission",
                "Account",
                "Local Currency",
                "To Account",
            ] {
                if !preloaded_transactions_df.is_none() && has_col(preloaded_transactions_df, c)? {
                    hash_cols.append(c)?;
                }
            }
            if !preloaded_transactions_df.is_none()
                && !preloaded_transactions_df.getattr("empty")?.is_truthy()?
                && hash_cols.is_truthy()?
            {
                let values = pd
                    .getattr("util")?
                    .call_method(
                        "hash_pandas_object",
                        (preloaded_transactions_df.get_item(&hash_cols)?,),
                        Some(&kwargs(
                            py,
                            &[("index", &PyBool::new(py, false).to_owned().into_any())],
                        )?),
                    )?
                    .getattr("values")?
                    .call_method0("tobytes")?;
                prefix(
                    &hashlib
                        .call_method1("sha256", (values,))?
                        .call_method0("hexdigest")?,
                    16,
                )?
            } else {
                pystr(py, "EMPTY_DATAFRAME")
            }
        };
        let sort_keys = kwargs(py, &[("sort_keys", &tru(py))])?;
        let acc_map = json.call_method("dumps", (account_currency_map,), Some(&sort_keys))?;
        let included_str = json.call_method1("dumps", (&included_sorted,))?;
        let excluded_str = json.call_method1("dumps", (&excluded_sorted,))?;
        let user_map = json.call_method("dumps", (&symbol_map,), Some(&sort_keys))?;
        let user_excluded = json.call_method1(
            "dumps",
            (sorted.call1((list.call1((&excluded_symbols,))?,))?,),
        )?;
        let content = format!(
            "DAILY_RES_{}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{}::{}",
            o.fmt(current_hist_version)?,
            o.fmt(&start_date.call_method0("isoformat")?)?,
            o.fmt(&end_date.call_method0("isoformat")?)?,
            o.fmt(&tx_hash)?,
            o.fmt(display_currency)?,
            o.fmt(&acc_map)?,
            o.fmt(default_currency)?,
            o.fmt(&included_str)?,
            o.fmt(&excluded_str)?,
            o.fmt(&user_map)?,
            o.fmt(&user_excluded)?,
            o.fmt(&m.getattr("HISTORICAL_CALC_METHOD")?)?,
            o.fmt(interval)?
        );
        let key = prefix(
            &hashlib
                .call_method1("sha256", (pystr(py, &content).call_method0("encode")?,))?
                .call_method0("hexdigest")?,
            32,
        )?;
        daily_key = key.clone();
        let filename = format!("{}_{}.feather", o.fmt(daily_cache_prefix)?, o.fmt(&key)?);
        daily_file = in_cache_dir(&filename)?;
        log(
            py,
            "info",
            format!(
                "Hist Prep: Daily results cache file set to: {}",
                o.fmt(&daily_file)?
            ),
        )?;
        Ok(())
    })();
    match keyed {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => log(
            py,
            "warning",
            format!(
                "Hist Prep WARN: Could not generate daily results cache key/filename: {}.",
                es(py, &e)
            ),
        )?,
        Err(e) => return Err(e),
    }

    PyTuple::new(
        py,
        [
            effective,
            original_transactions_df_for_ignored.clone(),
            ignored_indices,
            ignored_reasons,
            all_accounts,
            included_sorted,
            excluded_sorted,
            symbols_all,
            fx_pairs,
            internal_to_yf.into_any(),
            yf_to_internal.into_any(),
            splits_by_symbol.into_any(),
            raw_file,
            pystr(py, &raw_key),
            daily_file,
            daily_key,
            pystr(py, &filter_desc),
        ],
    )
}

/// The `price`, `Close`, `Adj Close` (and, for FX, `rate`) column a frame
/// offers first.
fn first_column<'py>(df: &Bound<'py, PyAny>, with_rate: bool) -> PyResult<Option<&'static str>> {
    let columns = df.getattr("columns")?;
    let names: &[&'static str] = if with_rate {
        &["price", "Close", "Adj Close", "rate"]
    } else {
        &["price", "Close", "Adj Close"]
    };
    for c in names {
        if columns.contains(*c)? {
            return Ok(Some(c));
        }
    }
    Ok(None)
}

/// A currency's rate per USD over the range (linear across gaps, then
/// filled), from `CUR=X`, or `USDTHB=X` / `USDCUR=X` when that is missing;
/// `None` when there is no usable table.
fn rate_per_usd<'py>(
    py: Python<'py>,
    pd: &Bound<'py, PyModule>,
    o: &Ops<'py>,
    upper: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    date_range: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let code = o.fmt(upper)?;
    let mut pair = format!("{code}=X");
    if !historical_fx_yf.contains(&pair)? {
        if upper.eq("THB")? && historical_fx_yf.contains("USDTHB=X")? {
            pair = "USDTHB=X".into();
        } else if upper.ne("THB")? && historical_fx_yf.contains(format!("USD{code}=X"))? {
            pair = format!("USD{code}=X");
        }
    }
    if !historical_fx_yf.contains(&pair)? {
        return Ok(None);
    }
    let df = historical_fx_yf.get_item(&pair)?;
    if df.getattr("empty")?.is_truthy()? {
        return Ok(None);
    }
    let Some(col) = first_column(&df, true)? else {
        return Ok(None);
    };
    let mut s = df.get_item(col)?.call_method0("copy")?;
    s.setattr(
        "index",
        crate::pyhelp::to_datetime(
            pd,
            &s.getattr("index")?,
            Some(&kwargs(py, &[("utc", &tru(py))])?),
        )?,
    )?;
    // A USD-per-unit pair (EURUSD=X) is inverted to units per USD.
    if pair.ends_with("USD=X") && !pair.starts_with("USD") {
        s = o.fl(1.0).div(&s)?;
    }
    Ok(Some(
        s.call_method1("reindex", (date_range,))?
            .call_method(
                "interpolate",
                (),
                Some(&kwargs(py, &[("method", &pystr(py, "linear"))])?),
            )?
            .call_method0("ffill")?
            .call_method0("bfill")?,
    ))
}

/// `_value_daily_holdings_vectorized`: (value per day, whether an FX table
/// was missing, a status note).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_value_daily_holdings<'py>(
    py: Python<'py>,
    date_range: &Bound<'py, PyAny>,
    daily_holdings_qty_np: &Bound<'py, PyAny>,
    daily_last_prices_np: &Bound<'py, PyAny>,
    daily_cash_balances_np: &Bound<'py, PyAny>,
    historical_prices_yf_unadjusted: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    target_currency: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    id_to_symbol: &Bound<'py, PyAny>,
    id_to_account: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
    account_ids_to_include_set: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    manual_overrides_dict: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool, String)> {
    let _ = interval;
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let cash = m.getattr("CASH_SYMBOL_CSV")?;
    let shape = unpack(&daily_holdings_qty_np.getattr("shape")?, 3)?;
    let (num_days, num_symbols, num_accounts) = (
        shape[0].clone(),
        shape[1].extract::<usize>()?,
        shape[2].extract::<usize>()?,
    );
    let mut has_errors = false;
    let mut status_msg = String::new();
    let f64_kw = || kwargs(py, &[("dtype", &np.getattr("float64")?)]);
    let column_of = |sid: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [PySlice::full(py).into_any(), sid.clone()])
    };

    // 1. Prices on the grid.
    let prices = np.call_method(
        "full",
        (
            PyTuple::new(py, [num_days.clone(), shape[1].clone()])?,
            f64::NAN,
        ),
        Some(&f64_kw()?),
    )?;
    let by_id = PyDict::new(py);
    let dti = pd.getattr("DatetimeIndex")?;
    let utc = kwargs(py, &[("utc", &tru(py))])?;
    for sym_id in 0..num_symbols {
        let symbol = id_to_symbol.call_method1("get", (sym_id,))?;
        if symbol.eq(&cash)? {
            continue;
        }
        let yf = internal_to_yf_map.call_method1("get", (&symbol,))?;
        if yf.is_truthy()? && historical_prices_yf_unadjusted.contains(&yf)? {
            let price_df = historical_prices_yf_unadjusted.get_item(&yf)?;
            if !price_df.getattr("empty")?.is_truthy()?
                && let Some(col) = first_column(&price_df, false)?
            {
                {
                    let mut ps = price_df.get_item(col)?;
                    if !ps.getattr("index")?.is_instance(&dti)?
                        || ps.getattr("index")?.getattr("tz")?.is_none()
                    {
                        ps = ps.call_method0("copy")?;
                        ps.setattr(
                            "index",
                            crate::pyhelp::to_datetime(&pd, &ps.getattr("index")?, Some(&utc))?,
                        )?;
                    }
                    by_id.set_item(sym_id, ps)?;
                }
            }
        }
    }
    // 1b. Published NAVs for holdings with no market feed.
    let nav_ids = PyDict::new(py);
    for item in id_to_symbol.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        if kv[1].is_truthy()? && kv[1].ne(&cash)? && !by_id.contains(&kv[0])? {
            nav_ids.set_item(&kv[0], &kv[1])?;
        }
    }
    if nav_ids.is_truthy()? {
        let navs = m
            .getattr("_fund_nav_series")?
            .call1((&nav_ids, date_range))?;
        for item in navs.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            by_id.set_item(&kv[0], &kv[1])?;
        }
    }
    if by_id.is_truthy()? {
        let matrix = m
            .getattr("align_prices_to_grid")?
            .call1((&by_id, date_range))?;
        for sid in by_id.keys().iter() {
            prices.set_item(column_of(&sid)?, matrix.get_item(&sid)?.getattr("values")?)?;
        }
    }
    // 2. The user's NAV where neither the market nor a published NAV has one.
    let manual = m
        .getattr("manual_prices_by_symbol_id")?
        .call1((manual_overrides_dict, id_to_symbol))?;
    for item in manual.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let column = prices.get_item(column_of(&kv[0])?)?;
        np.call_method(
            "copyto",
            (&column, &kv[1]),
            Some(&kwargs(
                py,
                &[("where", &np.call_method1("isnan", (&column,))?)],
            )?),
        )?;
    }

    // 3. FX on the grid, per account.
    let fx = np.call_method(
        "full",
        (
            PyTuple::new(py, [num_days.clone(), shape[2].clone()])?,
            f64::NAN,
        ),
        Some(&f64_kw()?),
    )?;
    let map_norm = PyDict::new(py);
    for item in account_currency_map.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        map_norm.set_item(
            o.str_t
                .call1((&kv[0],))?
                .call_method0("upper")?
                .call_method0("strip")?,
            &kv[1],
        )?;
    }
    let fx_cache = PyDict::new(py);
    let target_upper = target_currency.call_method0("upper")?;
    let target_rate = if target_upper.eq("USD")? {
        pd.getattr("Series")?
            .call((1.0,), Some(&kwargs(py, &[("index", date_range)])?))?
    } else {
        match rate_per_usd(py, &pd, &o, &target_upper, historical_fx_yf, date_range)? {
            Some(s) => s,
            None => pd
                .getattr("Series")?
                .call((f64::NAN,), Some(&kwargs(py, &[("index", date_range)])?))?,
        }
    };
    let ones = || np.call_method("ones", (&num_days,), Some(&f64_kw()?));
    for acc_id in 0..num_accounts {
        if !account_ids_to_include_set.contains(acc_id)? {
            continue;
        }
        let acc_name = id_to_account.call_method1("get", (acc_id,))?;
        let local = map_norm.call_method1("get", (&acc_name, default_currency))?;
        let local_upper = local.call_method0("upper")?;
        let col = PyTuple::new(
            py,
            [
                PySlice::full(py).into_any(),
                acc_id.into_pyobject(py)?.into_any(),
            ],
        )?;
        if fx_cache.contains(&local)? {
            fx.set_item(&col, fx_cache.as_any().get_item(&local)?)?;
            continue;
        }
        if local.eq(target_currency)? {
            let rates = ones()?;
            fx.set_item(&col, &rates)?;
            fx_cache.set_item(&local, rates)?;
            continue;
        }
        let local_rate = if local_upper.eq("USD")? {
            Some(
                pd.getattr("Series")?
                    .call((1.0,), Some(&kwargs(py, &[("index", date_range)])?))?,
            )
        } else {
            rate_per_usd(py, &pd, &o, &local_upper, historical_fx_yf, date_range)?
        };
        let rates = match local_rate {
            None => {
                has_errors = true;
                ones()?
            }
            Some(local_rate) => {
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
                let rates = with1(py, &errstate, |_| {
                    let mut cross = target_rate.div(&local_rate)?;
                    let series = pd.getattr("Series")?;
                    if series
                        .call1((&cross,))?
                        .call_method0("isna")?
                        .call_method0("any")?
                        .is_truthy()?
                    {
                        cross = series
                            .call1((&cross,))?
                            .call_method0("bfill")?
                            .call_method0("ffill")?
                            .getattr("values")?;
                    }
                    Ok(cross)
                })?
                .expect("errstate does not suppress");
                let nan_mask = np.call_method1("isnan", (&rates,))?;
                if nan_mask.call_method0("any")?.is_truthy()? {
                    setitem(&rates, &nan_mask, 1.0)?;
                }
                rates
            }
        };
        fx.set_item(&col, &rates)?;
        fx_cache.set_item(&local, rates)?;
    }

    // 4–6. qty × price × FX plus cash, summed per day, in the grid kernel.
    let acc_mask = np.call_method(
        "zeros",
        (num_accounts,),
        Some(&kwargs(
            py,
            &[("dtype", &py.import("builtins")?.getattr("bool")?)],
        )?),
    )?;
    for acc_id in account_ids_to_include_set.try_iter()? {
        acc_mask.set_item(acc_id?, true)?;
    }
    let logging = py.import("logging")?;
    if !daily_last_prices_np.is_none()
        && num_accounts > 0
        && logging
            .call_method0("getLogger")?
            .call_method1("isEnabledFor", (logging.getattr("DEBUG")?,))?
            .is_truthy()?
    {
        let missing = np.call_method1(
            "flatnonzero",
            (np.call_method1("isnan", (&prices,))?.call_method(
                "any",
                (),
                Some(&kwargs(
                    py,
                    &[("axis", &0i64.into_pyobject(py)?.into_any())],
                )?),
            )?,),
        )?;
        for s_id in missing.try_iter()? {
            let name = id_to_symbol.call_method1("get", (s_id?, "Unknown"))?;
            if name.ne(&cash)? {
                log(
                    py,
                    "debug",
                    format!(
                        "Valuation: Symbol '{}' missing market data. Using fallback.",
                        o.fmt(&name)?
                    ),
                )?;
            }
        }
    }
    let asarray = |a: &Bound<'py, PyAny>| np.call_method("asarray", (a,), Some(&f64_kw()?));
    let last = if daily_last_prices_np.is_none() {
        py.None().into_bound(py)
    } else {
        asarray(daily_last_prices_np)?
    };
    let total = py
        .import("investa_core")?
        .getattr("value_holdings_grid")?
        .call1((
            asarray(daily_holdings_qty_np)?,
            last,
            &prices,
            &fx,
            asarray(daily_cash_balances_np)?,
            acc_mask.call_method0("tolist")?,
        ))?;
    let mut series = pd
        .getattr("Series")?
        .call((total,), Some(&kwargs(py, &[("index", date_range)])?))?;

    // Gaps are interpolated, so a restored price is not a one-day spike.
    if series
        .call_method0("isna")?
        .call_method0("any")?
        .is_truthy()?
    {
        let missing = series
            .getattr("index")?
            .get_item(series.call_method0("isna")?)?;
        log(
            py,
            "warning",
            format!(
                "Valuation: Missing data for {} days. Using linear interpolation to smooth jumps.",
                missing.len()?
            ),
        )?;
        status_msg += " (partial data interpolated)";
        series = series
            .call_method(
                "interpolate",
                (),
                Some(&kwargs(py, &[("method", &pystr(py, "linear"))])?),
            )?
            .call_method0("ffill")?
            .call_method1("fillna", (0.0,))?;
    }
    Ok((series, has_errors, status_msg))
}

/// `_get_or_calculate_all_daily_holdings`: the daily holdings, cash and
/// last prices for the whole ledger, from the L1 cache or the replay kernel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_all_daily_holdings<'py>(
    py: Python<'py>,
    all_transactions_df: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    symbol_to_id: &Bound<'py, PyAny>,
    account_to_id: &Bound<'py, PyAny>,
    type_to_id: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let os = m.getattr("os")?;
    let os_path = os.getattr("path")?;
    let index_true = kwargs(py, &[("index", &tru(py))])?;
    let tx_hash = m
        .getattr("hashlib")?
        .call_method1(
            "sha256",
            (pd.getattr("util")?
                .call_method(
                    "hash_pandas_object",
                    (all_transactions_df,),
                    Some(&index_true),
                )?
                .getattr("values")?,),
        )?
        .call_method0("hexdigest")?;
    let cache_key = format!(
        "ALL_HOLDINGS_{}_{}_{}_{}",
        o.fmt(&m.getattr("CURRENT_HIST_VERSION")?)?,
        o.fmt(&tx_hash)?,
        o.fmt(&start_date.call_method0("isoformat")?)?,
        o.fmt(&end_date.call_method0("isoformat")?)?
    );

    let cache_dir_base = m.getattr("config")?.call_method0("get_app_cache_dir")?;
    let files = if cache_dir_base.is_truthy()? {
        let dir = os_path.call_method1("join", (&cache_dir_base, "all_holdings_cache_new"))?;
        os.call_method(
            "makedirs",
            (&dir,),
            Some(&kwargs(py, &[("exist_ok", &tru(py))])?),
        )?;
        let f = |suffix: &str| os_path.call_method1("join", (&dir, format!("{cache_key}{suffix}")));
        Some((
            dir.clone(),
            f(".key")?,
            f("_qty.npy")?,
            f("_cash.npy")?,
            f("_prices.npy")?,
        ))
    } else {
        log(
            py,
            "warning",
            "Could not find cache directory. Layer 1 caching disabled; computing without cache.",
        )?;
        None
    };

    if let Some((_, key_file, qty_file, cash_file, prices_file)) = &files {
        let exists = |p: &Bound<'py, PyAny>| -> PyResult<bool> {
            os_path.call_method1("exists", (p,))?.is_truthy()
        };
        if exists(key_file)? && exists(qty_file)? && exists(cash_file)? && exists(prices_file)? {
            log(
                py,
                "info",
                "L1 Cache HIT: Loading pre-calculated daily holdings for all accounts.",
            )?;
            let loaded = (|| -> PyResult<Bound<'py, PyTuple>> {
                let qty = np.call_method1("load", (qty_file,))?;
                let cash = np.call_method1("load", (cash_file,))?;
                let last = np.call_method1("load", (prices_file,))?;
                PyTuple::new(py, [qty, cash, last])
            })();
            match loaded {
                Ok(t) => return Ok(t),
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "warning",
                    format!(
                        "L1 Cache Load Error: Failed to load numpy arrays: {}",
                        es(py, &e)
                    ),
                )?,
                Err(e) => return Err(e),
            }
        }
    }

    // Chronological, as the replay needs.
    let sorted_tx = all_transactions_df
        .call_method(
            "sort_values",
            (),
            Some(&kwargs(
                py,
                &[("by", PyList::new(py, ["Date", "original_index"])?.as_any())],
            )?),
        )?
        .call_method0("copy")?;
    let mut grid = pd.call_method(
        "date_range",
        (),
        Some(&kwargs(
            py,
            &[
                ("start", start_date),
                ("end", end_date),
                ("freq", &pystr(py, "D")),
            ],
        )?),
    )?;
    if grid.getattr("tz")?.is_none() {
        grid = grid.call_method1("tz_localize", ("UTC",))?;
    }
    let int64 = np.getattr("int64")?;
    let float64 = np.getattr("float64")?;
    let i64_kw = kwargs(py, &[("dtype", &int64)])?;
    let as_i64 = |v: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        np.call_method(
            "array",
            (v.getattr("values")?.call_method1("astype", ("int64",))?,),
            Some(&i64_kw),
        )
    };
    let date_ordinals = as_i64(grid.clone())?;
    let tx_dates = as_i64(crate::pyhelp::to_datetime(
        &pd,
        &sorted_tx.get_item("Date")?,
        Some(&kwargs(py, &[("utc", &tru(py))])?),
    )?)?;
    let ids = |s: Bound<'py, PyAny>, mapping: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        s.call_method1("map", (mapping,))?
            .call_method1("fillna", (-1,))?
            .getattr("values")?
            .call_method1("astype", (&int64,))
    };
    let upper_strip = |c: &str| -> PyResult<Bound<'py, PyAny>> {
        sorted_tx
            .get_item(c)?
            .call_method1("astype", (&o.str_t,))?
            .getattr("str")?
            .call_method0("upper")?
            .getattr("str")?
            .call_method0("strip")
    };
    let tx_symbols = ids(sorted_tx.get_item("Symbol")?, symbol_to_id)?;
    let tx_accounts = ids(upper_strip("Account")?, account_to_id)?;
    let to_accounts_series = if sorted_tx.getattr("columns")?.contains("To Account")? {
        upper_strip("To Account")?.call_method1("map", (account_to_id,))?
    } else {
        pd.getattr("Series")?.call(
            (-1,),
            Some(&kwargs(py, &[("index", &sorted_tx.getattr("index")?)])?),
        )?
    };
    let tx_to_accounts = to_accounts_series
        .call_method1("fillna", (-1,))?
        .getattr("values")?
        .call_method1("astype", (&int64,))?;
    let tx_types = ids(
        sorted_tx
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?
            .getattr("str")?
            .call_method0("strip")?,
        type_to_id,
    )?;
    let floats = |c: &str| -> PyResult<Bound<'py, PyAny>> {
        sorted_tx
            .get_item(c)?
            .call_method1("fillna", (0.0,))?
            .getattr("values")?
            .call_method1("astype", (&float64,))
    };
    let zeros = || {
        np.call_method(
            "zeros",
            (sorted_tx.len()?,),
            Some(&kwargs(py, &[("dtype", &float64)])?),
        )
    };
    let columns = sorted_tx.getattr("columns")?;
    let tx_qty = floats("Quantity")?;
    let tx_comm = floats("Commission")?;
    let tx_split = floats("Split Ratio")?;
    let price_col = if columns.contains("Price/Share")? {
        "Price/Share"
    } else {
        "Price"
    };
    let tx_prices = if columns.contains(price_col)? {
        floats(price_col)?
    } else {
        zeros()?
    };
    let tx_totals = if columns.contains("Total Amount")? {
        floats("Total Amount")?
    } else {
        zeros()?
    };

    let type_id =
        |name: &str| -> PyResult<Bound<'py, PyAny>> { type_to_id.call_method1("get", (name, -1)) };
    let names = [
        "split",
        "stock split",
        "buy",
        "deposit",
        "sell",
        "withdrawal",
        "short sell",
        "buy to cover",
        "transfer",
        "spin off",
        "dividend",
        "interest",
        "fees",
        "tax",
    ];
    let mut type_ids = Vec::with_capacity(names.len());
    for n in names {
        type_ids.push(type_id(n)?);
    }
    let cash_symbol_id = symbol_to_id.call_method1("get", (m.getattr("CASH_SYMBOL_CSV")?, -1))?;
    let num_symbols = symbol_to_id.len()?;
    let num_accounts = account_to_id.len()?;

    // Auto cash mode per account.
    let modes_map = if account_cash_mode_map.is_truthy()? {
        account_cash_mode_map.clone()
    } else {
        PyDict::new(py).into_any()
    };
    let cash_modes = np.call_method("zeros", (num_accounts,), Some(&i64_kw))?;
    for item in modes_map.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let upper = kv[0].call_method0("upper")?.call_method0("strip")?;
        if account_to_id.contains(&upper)?
            && o.str_t
                .call1((&kv[1],))?
                .call_method0("lower")?
                .eq("auto")?
        {
            cash_modes.set_item(account_to_id.get_item(&upper)?, 1)?;
        }
    }
    let shortable = PyList::empty(py);
    for s in m.getattr("SHORTABLE_SYMBOLS")?.try_iter()? {
        let s = s?;
        if symbol_to_id.contains(&s)? {
            shortable.append(symbol_to_id.get_item(&s)?)?;
        }
    }
    let shortable_ids = np.call_method("array", (shortable,), Some(&i64_kw))?;

    let mut args: Vec<Bound<'py, PyAny>> = vec![
        date_ordinals,
        tx_dates,
        tx_symbols,
        tx_to_accounts,
        tx_accounts,
        tx_types,
        tx_qty,
        tx_comm,
        tx_split,
        tx_prices,
        tx_totals,
        num_symbols.into_pyobject(py)?.into_any(),
        num_accounts.into_pyobject(py)?.into_any(),
    ];
    args.extend(type_ids);
    args.push(cash_symbol_id);
    args.push(m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?);
    args.push(shortable_ids);
    args.push(cash_modes);
    let result = m
        .getattr("_calculate_daily_holdings_chronological_numba")?
        .call1(PyTuple::new(py, args)?)?;
    let parts = unpack(&result, 3)?;

    if let Some((dir, key_file, qty_file, cash_file, prices_file)) = &files {
        let saved = (|| -> PyResult<()> {
            np.call_method1("save", (qty_file, &parts[0]))?;
            np.call_method1("save", (cash_file, &parts[1]))?;
            np.call_method1("save", (prices_file, &parts[2]))?;
            let f = py
                .import("builtins")?
                .getattr("open")?
                .call1((key_file, "w"))?;
            with1(py, &f, |f| {
                f.call_method1("write", ("valid",))?;
                Ok(())
            })?;
            Ok(())
        })();
        match saved {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "error",
                format!(
                    "L1 Cache SAVE Error: Failed to save numpy arrays: {}",
                    es(py, &e)
                ),
            )?,
            Err(e) => return Err(e),
        }
        m.getattr("_prune_holdings_cache")?.call1((dir,))?;
    }
    PyTuple::new(py, parts)
}
