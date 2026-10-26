//! `portfolio_analyzer._process_transactions_to_holdings`: the ledger
//! normalised, split-deduplicated and encoded for the replay kernel
//! (`process_core`), and its state unpacked into the holdings the summary
//! reads.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`). The column preparation is the
//! same pandas operations (it still normalises the caller's frame in place,
//! as the original did); the price and split-ratio rules over the encoded
//! float64 columns are native; the kernel call and the helpers are looked up
//! on `portfolio_analyzer`. The holdings' values are read off the state array
//! as NumPy scalars, so they keep the original's types.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PySet, PyTuple};

use crate::pyhelp::{log, pystr, setitem};

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

fn tru(py: Python<'_>, b: bool) -> Bound<'_, PyAny> {
    pyo3::types::PyBool::new(py, b).to_owned().into_any()
}

/// `pd.to_numeric(col, errors="coerce").fillna(0.0).astype(np.float64).values`
fn numeric<'py>(py: Python<'py>, col: Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    let pd = py.import("pandas")?;
    let v = pd
        .call_method(
            "to_numeric",
            (col,),
            Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
        )?
        .call_method1("fillna", (0.0,))?
        .call_method1("astype", (py.import("numpy")?.getattr("float64")?,))?
        .getattr("values")?;
    let arr: PyReadonlyArray1<'_, f64> = v.extract()?;
    Ok(arr.as_array().to_vec())
}

/// `series.map(mapping).fillna(-1).astype(np.int64).values`
fn ids<'py>(
    py: Python<'py>,
    series: Bound<'py, PyAny>,
    mapping: &Bound<'py, PyAny>,
    fill: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    series
        .call_method1("map", (mapping,))?
        .call_method1("fillna", (fill,))?
        .call_method1("astype", (py.import("numpy")?.getattr("int64")?,))?
        .getattr("values")
}

fn i64s(v: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let arr: PyReadonlyArray1<'_, i64> = v.extract()?;
    Ok(arr.as_array().to_vec())
}

fn arr<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyAny> {
    Array1::from_vec(v).into_pyarray(py).into_any()
}

type HoldingsOut<'py> = (
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PySet>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    bool,
);

/// `_process_transactions_to_holdings`: (holdings, realized, dividends,
/// commissions, taxes, ignored indices, ignored reasons, transfer costs,
/// warned).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_transactions_to_holdings<'py>(
    py: Python<'py>,
    transactions_df: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    shortable_symbols: &Bound<'py, PyAny>,
    historical_fx_lookup: &Bound<'py, PyAny>,
    display_currency_for_hist_fx: &Bound<'py, PyAny>,
    report_date: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
) -> PyResult<HoldingsOut<'py>> {
    let _ = (display_currency_for_hist_fx, report_date);
    log(
        py,
        "debug",
        "Starting Numba-optimized transaction processing (Array-based)...",
    )?;
    let empty_out = || -> PyResult<HoldingsOut<'py>> {
        Ok((
            PyDict::new(py),
            PyDict::new(py),
            PyDict::new(py),
            PyDict::new(py),
            PyDict::new(py),
            PySet::empty(py)?,
            PyDict::new(py),
            PyDict::new(py),
            false,
        ))
    };
    if transactions_df.getattr("empty")?.is_truthy()? {
        return empty_out();
    }
    let mode_map = if account_cash_mode_map.is_none() {
        PyDict::new(py).into_any()
    } else {
        account_cash_mode_map.clone()
    };
    let m = pa(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let builtins = py.import("builtins")?;
    let str_t = builtins.getattr("str")?;
    let cash = m.getattr("CASH_SYMBOL_CSV")?;

    // --- normalise columns early, on the caller's frame ---
    let stripped = |col: &str| -> PyResult<Bound<'py, PyAny>> {
        transactions_df
            .get_item(col)?
            .call_method1("astype", (&str_t,))?
            .getattr("str")?
            .call_method0("strip")
    };
    setitem(transactions_df, "Symbol", stripped("Symbol")?)?;
    let cash_for_closure = cash.clone().unbind();
    let normaliser =
        PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<Py<PyAny>> {
            let py = args.py();
            let x = args.get_item(0)?;
            let is_cash = py
                .import("portfolio_analyzer")?
                .getattr("is_cash_symbol")?
                .call1((&x,))?
                .is_truthy()?;
            Ok(if is_cash {
                cash_for_closure.clone_ref(py)
            } else {
                x.unbind()
            })
        })?;
    setitem(
        transactions_df,
        "Symbol",
        transactions_df
            .get_item("Symbol")?
            .call_method1("apply", (normaliser,))?,
    )?;
    setitem(transactions_df, "Account", stripped("Account")?)?;
    setitem(
        transactions_df,
        "Type",
        stripped("Type")?.getattr("str")?.call_method0("lower")?,
    )?;

    // --- id mappings ---
    let all_symbols = builtins
        .getattr("list")?
        .call1((transactions_df.get_item("Symbol")?.call_method0("unique")?,))?;
    if !all_symbols.contains(&cash)? {
        all_symbols.call_method1("append", (&cash,))?;
    }
    let accounts = PySet::new(
        py,
        transactions_df
            .get_item("Account")?
            .call_method0("unique")?
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    if transactions_df.getattr("columns")?.contains("To Account")? {
        let to = transactions_df
            .get_item("To Account")?
            .call_method0("dropna")?
            .call_method1("astype", (&str_t,))?
            .getattr("str")?
            .call_method0("strip")?
            .call_method0("unique")?;
        accounts.call_method1("update", (to,))?;
    }
    // Sorted, not set order: account ids decide the holdings' order.
    let all_accounts = builtins.getattr("sorted")?.call1((accounts,))?;
    let all_currencies = transactions_df
        .get_item("Local Currency")?
        .call_method1("astype", (&str_t,))?
        .getattr("str")?
        .call_method0("strip")?
        .getattr("str")?
        .call_method0("upper")?
        .call_method0("unique")?;
    let enumerate_map =
        |items: &Bound<'py, PyAny>| -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
            let fwd = PyDict::new(py);
            for (i, x) in items.try_iter()?.enumerate() {
                fwd.set_item(x?, i)?;
            }
            let rev = PyDict::new(py);
            for (k, v) in fwd.iter() {
                rev.set_item(v, k)?;
            }
            Ok((fwd, rev))
        };
    let (sym_map, rev_sym) = enumerate_map(&all_symbols)?;
    let (acc_map, rev_acc) = enumerate_map(&all_accounts)?;
    let (curr_map, rev_curr) = enumerate_map(&all_currencies)?;
    let num_syms = all_symbols.len()?;
    let num_accs = all_accounts.len()?;
    let mut df = transactions_df.call_method0("copy")?;

    // --- split deduplication ---
    let is_split = df
        .get_item("Type")?
        .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?;
    if is_split.call_method0("any")?.is_truthy()? {
        let splits = df.get_item(&is_split)?.call_method0("copy")?;
        let other = df.get_item(is_split.call_method0("__invert__")?)?;
        if !pd
            .getattr("api")?
            .getattr("types")?
            .call_method1("is_datetime64_any_dtype", (splits.get_item("Date")?,))?
            .is_truthy()?
        {
            setitem(
                &splits,
                "Date",
                pd.call_method(
                    "to_datetime",
                    (splits.get_item("Date")?,),
                    Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
                )?,
            )?;
        }
        let all_acc = splits
            .get_item("Account")?
            .getattr("str")?
            .call_method0("lower")?
            .call_method1("__eq__", ("all accounts",))?;
        setitem(
            &splits,
            "__split_priority",
            np.call_method1("where", (all_acc, 0, 1))?,
        )?;
        let dt = splits.get_item("Date")?.getattr("dt")?;
        let ym = dt
            .getattr("year")?
            .mul(100)?
            .add(dt.getattr("month")?)?
            .call_method1("astype", (np.getattr("int64")?,))?;
        setitem(&splits, "__split_ym", ym)?;
        splits.call_method(
            "sort_values",
            (),
            Some(&kw(
                py,
                &[
                    (
                        "by",
                        PyList::new(py, ["Date", "Symbol", "__split_priority"])?.into_any(),
                    ),
                    ("inplace", tru(py, true)),
                ],
            )?),
        )?;
        let deduped = splits
            .call_method(
                "drop_duplicates",
                (),
                Some(&kw(
                    py,
                    &[
                        (
                            "subset",
                            PyList::new(py, ["__split_ym", "Symbol", "Split Ratio"])?.into_any(),
                        ),
                        ("keep", pystr(py, "first")),
                    ],
                )?),
            )?
            .call_method(
                "drop",
                (),
                Some(&kw(
                    py,
                    &[(
                        "columns",
                        PyList::new(py, ["__split_priority", "__split_ym"])?.into_any(),
                    )],
                )?),
            )?;
        let by = PyList::new(py, ["Date", "original_index"])?.into_any();
        df = pd
            .call_method(
                "concat",
                (PyList::new(
                    py,
                    [other.call_method0("copy")?, deduped.call_method0("copy")?],
                )?,),
                Some(&kw(py, &[("ignore_index", tru(py, true))])?),
            )?
            .call_method("sort_values", (), Some(&kw(py, &[("by", by.clone())])?))?;
        df.call_method(
            "sort_values",
            (),
            Some(&kw(py, &[("by", by), ("inplace", tru(py, true))])?),
        )?;
        df = df
            .call_method(
                "reset_index",
                (),
                Some(&kw(py, &[("drop", tru(py, true))])?),
            )?
            .call_method0("copy")?;
    }
    let n = df.len()?;

    let neg1 = (-1i64).into_pyobject(py)?.into_any();
    let sym_ids = ids(py, df.get_item("Symbol")?, sym_map.as_any(), neg1.clone())?;
    let acc_ids = ids(py, df.get_item("Account")?, acc_map.as_any(), neg1.clone())?;
    let type_map = PyDict::new(py);
    for (k, v) in [
        ("buy", "TYPE_BUY"),
        ("sell", "TYPE_SELL"),
        ("deposit", "TYPE_DEPOSIT"),
        ("withdrawal", "TYPE_WITHDRAWAL"),
        ("dividend", "TYPE_DIVIDEND"),
        ("fee", "TYPE_FEES"),
        ("fees", "TYPE_FEES"),
        ("split", "TYPE_SPLIT"),
        ("stock split", "TYPE_SPLIT"),
        ("transfer", "TYPE_TRANSFER"),
        ("short sell", "TYPE_SHORT_SELL"),
        ("buy to cover", "TYPE_BUY_TO_COVER"),
        ("tax", "TYPE_TAX"),
        ("withholding tax", "TYPE_TAX"),
        ("interest", "TYPE_INTEREST"),
        ("spin off", "TYPE_SPIN_OFF"),
        ("spin-off", "TYPE_SPIN_OFF"),
        ("spinoff", "TYPE_SPIN_OFF"),
    ] {
        type_map.set_item(k, m.getattr(v)?)?;
    }
    let type_ids_obj = ids(
        py,
        df.get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?
            .getattr("str")?
            .call_method0("strip")?,
        type_map.as_any(),
        m.getattr("TYPE_UNKNOWN")?,
    )?;
    let type_ids = i64s(&type_ids_obj)?;
    let ty = |name: &str| -> PyResult<i64> { m.getattr(name)?.extract() };
    let (t_div, t_tax, t_fees, t_int, t_split) = (
        ty("TYPE_DIVIDEND")?,
        ty("TYPE_TAX")?,
        ty("TYPE_FEES")?,
        ty("TYPE_INTEREST")?,
        ty("TYPE_SPLIT")?,
    );

    let mut qtys = numeric(py, df.get_item("Quantity")?)?;
    let mut prices = numeric(py, df.get_item("Price/Share")?)?;
    let totals = numeric(py, df.get_item("Total Amount")?)?;
    let cash_sym: Vec<bool> = df
        .get_item("Symbol")?
        .call_method1("__eq__", (&cash,))?
        .getattr("values")?
        .extract::<PyReadonlyArray1<'_, bool>>()?
        .as_array()
        .to_vec();
    let zero_qty: Vec<bool> = qtys.iter().map(|q| q.abs() < 1e-9).collect();
    let has_total: Vec<bool> = totals.iter().map(|t| t.abs() > 1e-9).collect();
    // --- cash rows: trust Total Amount ---
    for i in 0..n {
        if cash_sym[i] && has_total[i] {
            if zero_qty[i] {
                qtys[i] = totals[i].abs();
                prices[i] = 1.0;
            } else if (qtys[i] - totals[i].abs()).abs() < 1e-4 {
                prices[i] = 1.0;
            }
        }
    }
    // --- income and expense rows priced from Total Amount ---
    let mut use_total = vec![false; n];
    for i in 0..n {
        let t = type_ids[i];
        let income = t == t_div || t == t_tax || t == t_fees || t == t_int;
        use_total[i] = (income || cash_sym[i])
            && (prices[i].abs() < 1e-9 || qtys[i].abs() < 1e-9)
            && has_total[i];
        if use_total[i] {
            prices[i] = totals[i];
        }
    }
    for i in 0..n {
        let div_total = type_ids[i] == t_div && has_total[i];
        if div_total {
            prices[i] = totals[i].abs();
        }
        if type_ids[i] == t_div && !use_total[i] && !div_total {
            prices[i] *= qtys[i].abs();
        }
    }
    let comms = numeric(py, df.get_item("Commission")?)?;
    let mut ratios = numeric(py, df.get_item("Split Ratio")?)?;
    for i in 0..n {
        if type_ids[i] == t_split && ratios[i].abs() <= 1e-9 && qtys[i].abs() > 1e-9 {
            ratios[i] = if qtys[i].abs() <= 20.0 {
                qtys[i]
            } else {
                -qtys[i].abs()
            };
        }
    }

    let to_acc_ids = if df.getattr("columns")?.contains("To Account")? {
        setitem(
            &df,
            "To Account",
            df.get_item("To Account")?
                .call_method1("astype", (&str_t,))?
                .getattr("str")?
                .call_method0("strip")?,
        )?;
        ids(
            py,
            df.get_item("To Account")?,
            acc_map.as_any(),
            neg1.clone(),
        )?
    } else {
        np.call_method(
            "full",
            (n, -1),
            Some(&kw(py, &[("dtype", np.getattr("int64")?)])?),
        )?
    };
    setitem(
        &df,
        "Local Currency",
        df.get_item("Local Currency")?
            .call_method1("astype", (&str_t,))?
            .getattr("str")?
            .call_method0("strip")?
            .getattr("str")?
            .call_method0("upper")?,
    )?;
    let invalid = df
        .get_item("Local Currency")?
        .call_method1("isin", (&curr_map,))?
        .call_method0("__invert__")?;
    if invalid.call_method0("any")?.is_truthy()? {
        if !curr_map.contains(default_currency)? {
            let k = curr_map.len();
            curr_map.set_item(default_currency, k)?;
            rev_curr.set_item(curr_map.len() - 1, default_currency)?;
        }
        df.getattr("loc")?
            .set_item((invalid, "Local Currency"), default_currency)?;
    }
    let local_curr_ids = df
        .get_item("Local Currency")?
        .call_method1("map", (&curr_map,))?
        .call_method1("astype", (np.getattr("int64")?,))?
        .getattr("values")?;
    let dates = df
        .get_item("Date")?
        .getattr("dt")?
        .getattr("date")?
        .getattr("values")?;
    let currs = df.get_item("Local Currency")?.getattr("values")?;
    let looked_up = PyList::empty(py);
    for (d, c) in dates.try_iter()?.zip(currs.try_iter()?) {
        looked_up.append(
            historical_fx_lookup.call_method1("get", (PyTuple::new(py, [d?, c?])?, 1.0))?,
        )?;
    }
    let fx_hist = np.call_method(
        "array",
        (looked_up,),
        Some(&kw(py, &[("dtype", np.getattr("float64")?)])?),
    )?;
    let shortable = PySet::empty(py)?;
    for sym in shortable_symbols.try_iter()? {
        let sym = sym?;
        if sym_map.contains(&sym)? {
            shortable.add(sym_map.get_item(&sym)?.expect("present"))?;
        }
    }
    let nb_shortable = np.call_method(
        "array",
        (builtins.getattr("sorted")?.call1((shortable,))?,),
        Some(&kw(py, &[("dtype", np.getattr("int64")?)])?),
    )?;
    let cash_sym_id = sym_map.call_method1("get", (&cash, -1))?;
    let mut modes = vec![0i64; num_accs];
    // `{str(k).strip(): v for k, v in map.items()}`, then the Auto accounts
    let normalized = PyDict::new(py);
    for item in mode_map.call_method0("items")?.try_iter()? {
        let item = item?;
        normalized.set_item(
            item.get_item(0)?.str()?.call_method0("strip")?,
            item.get_item(1)?,
        )?;
    }
    for (name, mode) in normalized.iter() {
        if acc_map.contains(&name)? && mode.eq("Auto")? {
            let idx: usize = acc_map.get_item(&name)?.expect("present").extract()?;
            modes[idx] = 1;
        }
    }
    let acc_cash_modes = Array1::from_vec(modes).into_pyarray(py).into_any();

    let core_args: Vec<Bound<'py, PyAny>> = vec![
        sym_ids,
        acc_ids,
        type_ids_obj,
        arr(py, qtys),
        arr(py, prices),
        arr(py, comms),
        arr(py, totals),
        arr(py, ratios),
        to_acc_ids,
        local_curr_ids,
        fx_hist,
        nb_shortable,
        m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?,
        cash_sym_id,
        n.into_pyobject(py)?.into_any(),
        num_syms.into_pyobject(py)?.into_any(),
        num_accs.into_pyobject(py)?.into_any(),
        acc_cash_modes,
    ];
    let core = m
        .getattr("_process_numba_core")?
        .call1(PyTuple::new(py, core_args)?)?;
    let (final_state, transfer_costs_nb) = (core.get_item(0)?, core.get_item(1)?);

    // --- tags ---
    let tags_map = py
        .import("collections")?
        .getattr("defaultdict")?
        .call1((builtins.getattr("set")?,))?;
    if df.getattr("columns")?.contains("Tags")? {
        let mask = df.get_item("Tags")?.call_method0("notna")?.call_method1(
            "__and__",
            (df.get_item("Tags")?.call_method1("__ne__", ("",))?,),
        )?;
        let tagged = df.get_item(mask)?;
        for pair in tagged.call_method0("iterrows")?.try_iter()? {
            let row = pair?.get_item(1)?;
            let key = PyTuple::new(py, [row.get_item("Symbol")?, row.get_item("Account")?])?;
            for t in row
                .get_item("Tags")?
                .str()?
                .call_method1("split", (",",))?
                .try_iter()?
            {
                let cleaned = t?.call_method0("strip")?;
                if cleaned.is_truthy()? {
                    tags_map.get_item(&key)?.call_method1("add", (cleaned,))?;
                }
            }
        }
    }

    let holdings = PyDict::new(py);
    let float_dd = || -> PyResult<Bound<'py, PyAny>> {
        py.import("collections")?
            .getattr("defaultdict")?
            .call1((builtins.getattr("float")?,))
    };
    let (realized, dividends, commissions, taxes) =
        (float_dd()?, float_dd()?, float_dd()?, float_dd()?);
    let currency_channel = final_state.get_item((
        py.import("builtins")?
            .getattr("slice")?
            .call1((py.None(),))?,
        py.import("builtins")?
            .getattr("slice")?
            .call1((py.None(),))?,
        11,
    ))?;
    let active = np.call_method1(
        "argwhere",
        (currency_channel.call_method1("__ne__", (-1.0,))?,),
    )?;
    let fields = [
        "qty",
        "total_cost_local",
        "realized_gain_local",
        "dividends_local",
        "commissions_local",
        "local_currency",
        "short_proceeds_local",
        "short_original_qty",
        "total_cost_invested_local",
        "cumulative_investment_local",
        "total_buy_cost_local",
        "total_cost_display_historical_fx",
        "realized_gain_display",
        "taxes_local",
        "tags",
    ];
    let slot = [0usize, 1, 2, 3, 4, 99, 5, 6, 7, 8, 9, 10, 12, 13, 98];
    for idx in active.try_iter()? {
        let idx = idx?;
        let (s_id, a_id) = (idx.get_item(0)?, idx.get_item(1)?);
        let val = final_state.get_item((&s_id, &a_id))?;
        let sym = rev_sym.get_item(&s_id)?.expect("id");
        let acc = rev_acc.get_item(&a_id)?.expect("id");
        let curr_id = builtins.getattr("int")?.call1((val.get_item(11)?,))?;
        let curr = rev_curr
            .get_item(&curr_id)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(curr_id.clone().unbind()))?;
        let key = PyTuple::new(py, [&sym, &acc])?;
        let entry = PyDict::new(py);
        for (field, k) in fields.iter().zip(slot) {
            let v = match k {
                99 => curr.clone(),
                98 => {
                    let tags = tags_map.call_method1("get", (&key, PySet::empty(py)?))?;
                    builtins
                        .getattr("sorted")?
                        .call1((builtins.getattr("list")?.call1((tags,))?,))?
                }
                _ => val.get_item(k)?,
            };
            entry.set_item(*field, v)?;
        }
        holdings.set_item(&key, entry)?;
        for (dd, k) in [
            (&realized, 2usize),
            (&dividends, 3),
            (&commissions, 4),
            (&taxes, 13),
        ] {
            dd.set_item(&curr, dd.get_item(&curr)?.add(val.get_item(k)?)?)?;
        }
    }
    let transfer_costs = PyDict::new(py);
    let orig_indices = df.get_item("original_index")?.getattr("values")?;
    let valid = np
        .call_method1(
            "where",
            (np.call_method1("isnan", (&transfer_costs_nb,))?
                .call_method0("__invert__")?,),
        )?
        .get_item(0)?;
    let count = orig_indices.len()?;
    for idx in valid.try_iter()? {
        let idx = idx?;
        if idx.lt(count)? {
            transfer_costs.set_item(
                orig_indices.get_item(&idx)?,
                transfer_costs_nb.get_item(&idx)?,
            )?;
        }
    }
    let dict = builtins.getattr("dict")?;
    Ok((
        holdings,
        dict.call1((realized,))?.cast_into()?,
        dict.call1((dividends,))?.cast_into()?,
        dict.call1((commissions,))?.cast_into()?,
        dict.call1((taxes,))?.cast_into()?,
        PySet::empty(py)?,
        PyDict::new(py),
        transfer_costs,
        false,
    ))
}
