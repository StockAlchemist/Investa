//! The growth, normalization and Monte Carlo glue of `financial_ratios`: the
//! measured and blended growth estimates, the ticker-info enrichment from the
//! statements, the FCF margins and normalized base FCF, the Monte Carlo
//! statistics, and the twelve Monte Carlo model wrappers.
//!
//! Ported line for line from `src/financial_ratios.py` (frozen in
//! `tests/reference/financial_ratios_p15.py`). The wrappers draw from NumPy's
//! global generator (`np.random.normal`, looked up on the module's `np`) with
//! the same arguments in the same order, so a seeded run gives the same
//! samples; the per-sample valuation kernels were already Rust and are called
//! as the original called them.

use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyTuple};

use crate::fr::{Gsv, builtin, coerce, fl, fr, int, none, none_or_empty, opt_number, or_else};
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

/// `key=lambda x: pd.to_datetime(x, errors="coerce")`
pub(crate) fn date_key<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, |args, _| -> PyResult<Py<PyAny>> {
        let py = args.py();
        Ok(py
            .import("pandas")?
            .getattr("to_datetime")?
            .call((args.get_item(0)?,), Some(&coerce(py)?))?
            .unbind())
    })
}

/// `sorted(xs, key=lambda x: pd.to_datetime(x, errors="coerce"))`
fn sorted_by_date<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let key = date_key(py)?;
    builtin(py, "sorted")?.call((xs,), Some(&kwargs(py, &[("key", key.as_any())])?))
}

/// `xs[-n:]`
fn last_n<'py>(xs: &Bound<'py, PyAny>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = xs.py();
    xs.get_item(builtin(py, "slice")?.call1((n.neg()?, none(py)))?)
}

fn is_value_or_type(py: Python<'_>, e: &PyErr) -> bool {
    e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py)
}

/// `try: v = float(x); if <finite test>: keep(v) except (ValueError, TypeError): pass`
fn finite_float<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    test: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<bool>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let r = builtin(py, "float")?
        .call1((x,))
        .and_then(|v| Ok(if test(&v)? { Some(v) } else { None }));
    match r {
        Ok(v) => Ok(v),
        Err(e) if is_value_or_type(py, &e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// `not np.isnan(v) and not np.isinf(v)`
fn not_nan_or_inf(np: &Bound<'_, PyModule>, v: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!np.call_method1("isnan", (v,))?.is_truthy()?
        && !np.call_method1("isinf", (v,))?.is_truthy()?)
}

#[pyfunction]
pub fn fr_estimate_growth_rate<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    ticker_info: &Bound<'py, PyAny>,
    item_name: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = years;
    let m = fr(py)?;
    let np = py.import("numpy")?;
    let default = || fl(py, 0.05);

    // 1. Analyst expected growth (priority)
    if ticker_info.is_truthy()? {
        let analyst_ee =
            ticker_info.call_method1("get", ("_earnings_estimate", PyDict::new(py)))?;
        if analyst_ee.is_truthy()? {
            let expected = PyList::empty(py);
            for p in ["0y", "+1y"] {
                let row = analyst_ee.call_method1("get", (p,))?;
                if row.is_truthy()?
                    && row.contains("growth")?
                    && !row.get_item("growth")?.is_none()
                    && let Some(v) =
                        finite_float(py, &row.get_item("growth")?, |v| not_nan_or_inf(&np, v))?
                {
                    expected.append(v)?;
                }
            }
            if expected.is_truthy()? {
                return builtin(py, "sum")?.call1((&expected,))?.div(expected.len());
            }
        }

        // Fallback to standard info fields
        let g = or_else(
            ticker_info.call_method1("get", ("earningsGrowth",))?,
            || ticker_info.call_method1("get", ("revenueGrowth",)),
        )?;
        if !g.is_none()
            && let Some(v) = finite_float(py, &g, |v| not_nan_or_inf(&np, v))?
        {
            return Ok(v);
        }
    }

    // 2. Historical CAGR
    if !none_or_empty(financials_df)? {
        let body = || -> PyResult<Option<Bound<'py, PyAny>>> {
            let pd = py.import("pandas")?;
            let gsv = m.getattr("_get_statement_value")?;
            let get = |col: &Bound<'py, PyAny>| gsv.call1((financials_df, item_name, col));
            let cols = sorted_by_date(py, &financials_df.getattr("columns")?)?;
            let mut recent: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
            for col in cols.try_iter()? {
                let col = col?;
                let val = get(&col)?;
                if !val.is_none() && val.gt(0)? {
                    recent.push((pd.getattr("to_datetime")?.call1((&col,))?, val));
                }
            }
            if recent.len() >= 2 {
                let (end_date, end_val) = recent.last().expect("two or more").clone();
                let mut start_idx = 0;
                for i in (0..recent.len() - 1).rev() {
                    let years_diff = end_date.sub(&recent[i].0)?.getattr("days")?.div(365.25)?;
                    if years_diff.ge(2.5)? {
                        start_idx = i;
                        break;
                    }
                }
                let (start_date, start_val) = &recent[start_idx];
                let n_years = end_date.sub(start_date)?.getattr("days")?.div(365.25)?;
                if n_years.gt(0.5)? {
                    let res = end_val
                        .div(start_val)?
                        .pow(int(py, 1)?.div(&n_years)?, py.None())?
                        .sub(1)?;
                    if np.call_method1("isnan", (&res,))?.is_truthy()?
                        || np.call_method1("isinf", (&res,))?.is_truthy()?
                    {
                        return Ok(Some(default()));
                    }
                    return Ok(Some(res));
                }
            }
            Ok(None)
        };
        match body() {
            Ok(Some(v)) => return Ok(v),
            Ok(None) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => {}
            Err(e) => return Err(e),
        }
    }

    // 3. Final default
    Ok(default())
}

#[pyfunction]
pub fn fr_historical_growth_regression<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    item_name: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if none_or_empty(financials_df)? {
        return Ok(none(py));
    }
    let m = fr(py)?;
    let body = || -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import("pandas")?;
        let np = py.import("numpy")?;
        let gsv = m.getattr("_get_statement_value")?;
        let mut dated = PyList::empty(py).into_any();
        for col in financials_df.getattr("columns")?.try_iter()? {
            let col = col?;
            let ts = pd
                .getattr("to_datetime")?
                .call((&col,), Some(&coerce(py)?))?;
            let val = gsv.call1((financials_df, item_name, &col))?;
            if pd.call_method1("notna", (&ts,))?.is_truthy()? && !val.is_none() && val.gt(0)? {
                dated.call_method1("append", (PyTuple::new(py, [ts, val])?,))?;
            }
        }
        let first = py.import("operator")?.getattr("itemgetter")?.call1((0,))?;
        dated.call_method("sort", (), Some(&kwargs(py, &[("key", &first)])?))?;
        dated = last_n(&dated, years)?;
        if dated.len()? < 3 {
            return Ok(none(py));
        }

        let t0 = dated.get_item(0)?.get_item(0)?;
        let xs_list = PyList::empty(py);
        let ys_list = PyList::empty(py);
        for p in dated.try_iter()? {
            let dv = unpack(&p?, 2)?;
            xs_list.append(dv[0].sub(&t0)?.getattr("days")?.div(365.25)?)?;
        }
        for p in dated.try_iter()? {
            ys_list.append(unpack(&p?, 2)?.swap_remove(1))?;
        }
        let float_t = builtin(py, "float")?;
        let array = |l: &Bound<'py, PyList>| -> PyResult<Bound<'py, PyAny>> {
            np.getattr("array")?
                .call((l,), Some(&kwargs(py, &[("dtype", &float_t)])?))
        };
        let xs = array(&xs_list)?;
        let ys = np.call_method1("log", (array(&ys_list)?,))?;
        if xs.get_item(-1)?.sub(xs.get_item(0)?)?.lt(1.0)? {
            return Ok(none(py));
        }
        let slope = float_t.call1((np.call_method1("polyfit", (&xs, &ys, 1))?.get_item(0)?,))?;
        let growth = float_t.call1((np.call_method1("expm1", (slope,))?,))?;
        if np.call_method1("isfinite", (&growth,))?.is_truthy()? {
            Ok(growth)
        } else {
            Ok(none(py))
        }
    };
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => Ok(none(py)),
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fr_blended_growth_estimate<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    ticker_info: &Bound<'py, PyAny>,
    item_name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let np = py.import("numpy")?;
    let o = Ops::new(py)?;
    let signals = PyDict::new(py);

    if ticker_info.is_truthy()? {
        let analyst_rates = PyList::empty(py);
        for period in ["0y", "+1y"] {
            let ee = or_else(
                ticker_info.call_method1("get", ("_earnings_estimate",))?,
                || Ok(PyDict::new(py).into_any()),
            )?;
            let row = ee.call_method1("get", (period,))?;
            if row.is_truthy()?
                && !row.call_method1("get", ("growth",))?.is_none()
                && let Some(v) = finite_float(py, &row.get_item("growth")?, |v| {
                    np.call_method1("isfinite", (v,))?.is_truthy()
                })?
            {
                analyst_rates.append(v)?;
            }
        }
        if analyst_rates.is_truthy()? {
            signals.set_item(
                "analyst",
                m.getattr("investa_core")?
                    .call_method1("np_mean", (analyst_rates,))?,
            )?;
        }
    }

    let regression = m.getattr("_historical_growth_regression")?.call(
        (financials_df,),
        Some(&kwargs(py, &[("item_name", item_name)])?),
    )?;
    if !regression.is_none() {
        signals.set_item("regression", regression)?;
    }

    let cagr = m.getattr("estimate_growth_rate")?.call(
        (financials_df,),
        Some(&kwargs(
            py,
            &[("ticker_info", &none(py)), ("item_name", item_name)],
        )?),
    )?;
    // 0.05 is the estimator's "I found nothing" default; don't treat it as evidence.
    if !cagr.is_none()
        && np.call_method1("isfinite", (&cagr,))?.is_truthy()?
        && o.abs(&cagr.sub(0.05)?)?.gt(1e-9)?
    {
        signals.set_item("cagr", builtin(py, "float")?.call1((&cagr,))?)?;
    }

    let out = PyDict::new(py);
    if !signals.is_truthy()? {
        out.set_item("growth", m.getattr("BASE_RATE_GROWTH")?)?;
        out.set_item("raw_growth", none(py))?;
        out.set_item("signals", PyDict::new(py))?;
        out.set_item("method", "base rate (no growth evidence)")?;
        return Ok(out);
    }

    let shrink = m.getattr("GROWTH_SHRINK_WEIGHT")?;
    let base = m.getattr("BASE_RATE_GROWTH")?;
    let args = PyTuple::new(
        py,
        [
            signals.as_any().call_method1("get", ("regression",))?,
            signals.as_any().call_method1("get", ("cagr",))?,
            signals.as_any().call_method1("get", ("analyst",))?,
            shrink.clone(),
            base.clone(),
            m.getattr("MIN_PROJECTED_GROWTH")?,
            m.getattr("MAX_PROJECTED_GROWTH")?,
        ],
    )?;
    let cr = unpack(
        &m.getattr("investa_core")?
            .getattr("blend_growth")?
            .call1(args)?,
        2,
    )?;
    let method = format!(
        "{} -> {}, shrunk {}/{} toward {} base rate",
        o.fmt(&builtin(py, "sorted")?.call1((&signals,))?)?,
        o.fmt_spec(&cr[1], ".1%")?,
        o.fmt_spec(&shrink, ".0%")?,
        o.fmt_spec(&int(py, 1)?.sub(&shrink)?, ".0%")?,
        o.fmt_spec(&base, ".0%")?
    );
    out.set_item("growth", &cr[0])?;
    out.set_item("raw_growth", &cr[1])?;
    out.set_item("signals", signals)?;
    out.set_item("method", method)?;
    Ok(out)
}

#[pyfunction]
pub fn fr_enrich_ticker_info<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let g = Gsv::new(&fr(py)?)?;
    let enriched = ticker_info.call_method0("copy")?;
    let get = |k: &str| enriched.call_method1("get", (k,));
    let present = |df: &Bound<'py, PyAny>| -> PyResult<bool> { Ok(!none_or_empty(df)?) };
    let latest = |df: &Bound<'py, PyAny>| df.getattr("columns")?.get_item(0);

    // Check 1: Shares Outstanding
    if !get("sharesOutstanding")?.is_truthy()? {
        if present(balance_sheet_df)? {
            let col = latest(balance_sheet_df)?;
            let mut shares = g.get(balance_sheet_df, "Ordinary Shares Number", &col)?;
            if !shares.is_truthy()? {
                shares = g.get(balance_sheet_df, "Share Issued", &col)?;
            }
            if shares.is_truthy()? {
                enriched.set_item("sharesOutstanding", shares)?;
            }
        }
        if !get("sharesOutstanding")?.is_truthy()? && present(financials_df)? {
            let col = latest(financials_df)?;
            let mut shares = g.get(financials_df, "Diluted Average Shares", &col)?;
            if !shares.is_truthy()? {
                shares = g.get(financials_df, "Basic Average Shares", &col)?;
            }
            if shares.is_truthy()? {
                enriched.set_item("sharesOutstanding", shares)?;
            }
        }
    }

    // Check 2: Total Revenue
    if !get("totalRevenue")?.is_truthy()? && present(financials_df)? {
        let rev = g.get(financials_df, "Total Revenue", &latest(financials_df)?)?;
        if rev.is_truthy()? {
            enriched.set_item("totalRevenue", rev)?;
        }
    }

    // Check 3: Total Cash / Debt
    if get("totalCash")?.is_none() && present(balance_sheet_df)? {
        let cash = g.get(
            balance_sheet_df,
            "Cash And Cash Equivalents",
            &latest(balance_sheet_df)?,
        )?;
        if cash.is_truthy()? {
            enriched.set_item("totalCash", cash)?;
        }
    }
    if get("totalDebt")?.is_none() && present(balance_sheet_df)? {
        let debt = g.get(balance_sheet_df, "Total Debt", &latest(balance_sheet_df)?)?;
        if debt.is_truthy()? {
            enriched.set_item("totalDebt", debt)?;
        }
    }

    // Check 4: EPS (Trailing)
    if !get("trailingEps")?.is_truthy()? && present(financials_df)? {
        let eps = g.get(financials_df, "Diluted EPS", &latest(financials_df)?)?;
        if eps.is_truthy()? {
            enriched.set_item("trailingEps", eps)?;
        }
    }

    // Check 5: Dividend Rate
    if !get("dividendRate")?.is_truthy()?
        && !get("trailingAnnualDividendRate")?.is_truthy()?
        && present(cashflow_df)?
    {
        let div_paid = g.get(cashflow_df, "Cash Dividends Paid", &latest(cashflow_df)?)?;
        let shares = get("sharesOutstanding")?;
        if div_paid.is_truthy()? && shares.is_truthy()? && shares.gt(0)? {
            let rate = builtin(py, "abs")?.call1((div_paid,))?.div(&shares)?;
            enriched.set_item("dividendRate", rate)?;
        }
    }
    Ok(enriched)
}

/// `sorted(set(a.columns).intersection(set(b.columns)), key=<date>)`
fn common_columns<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    as_list: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let set = builtin(py, "set")?;
    let mut common = set
        .call1((a.getattr("columns")?,))?
        .call_method1("intersection", (set.call1((b.getattr("columns")?,))?,))?;
    if as_list {
        common = builtin(py, "list")?.call1((common,))?;
    }
    sorted_by_date(py, &common)
}

/// Revenue and FCF per common period, as the margin estimators collect them.
fn margin_inputs<'py>(
    m: &Bound<'py, PyModule>,
    financials_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    cols: &Bound<'py, PyAny>,
    revenues: &Bound<'py, PyList>,
    fcfs: &Bound<'py, PyList>,
) -> PyResult<()> {
    let gsv = m.getattr("_get_statement_value")?;
    let fcf_of = m.getattr("_extract_fcf_from_statement")?;
    for col in cols.try_iter()? {
        let col = col?;
        let revenue = opt_number(&gsv.call1((financials_df, "Total Revenue", &col))?)?;
        let fcf = opt_number(&fcf_of.call1((cashflow_df, &col))?)?;
        revenues.append(revenue)?;
        fcfs.append(fcf)?;
    }
    Ok(())
}

#[pyfunction]
pub fn fr_estimate_fcf_margin<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if none_or_empty(financials_df)? || none_or_empty(cashflow_df)? {
        return Ok(fl(py, 0.05));
    }
    let m = fr(py)?;
    let body = || -> PyResult<Option<Bound<'py, PyAny>>> {
        let common = common_columns(py, financials_df, cashflow_df, true)?;
        let (revenues, fcfs) = (PyList::empty(py), PyList::empty(py));
        margin_inputs(
            &m,
            financials_df,
            cashflow_df,
            &last_n(&common, years)?,
            &revenues,
            &fcfs,
        )?;
        let core = m.getattr("investa_core")?;
        let margins = core.call_method1("plausible_margins", (revenues, fcfs))?;
        if margins.is_truthy()? {
            return Ok(Some(core.call_method1("np_median", (margins,))?));
        }
        Ok(None)
    };
    match body() {
        Ok(Some(v)) => return Ok(v),
        Ok(None) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!("Failed to estimate FCF margin: {}", es(py, &e)),
            )?;
        }
        Err(e) => return Err(e),
    }
    Ok(fl(py, 0.05))
}

#[pyfunction]
pub fn fr_through_cycle_fcf_margin<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let (revenues, fcfs) = (PyList::empty(py), PyList::empty(py));
    let body = || -> PyResult<()> {
        if !financials_df.is_none() && !cashflow_df.is_none() {
            let common = common_columns(py, financials_df, cashflow_df, false)?;
            margin_inputs(
                &m,
                financials_df,
                cashflow_df,
                &last_n(&common, years)?,
                &revenues,
                &fcfs,
            )?;
        }
        Ok(())
    };
    match body() {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("Through-cycle margin extraction failed: {}", es(py, &e)),
            )?;
        }
        Err(e) => return Err(e),
    }

    let core = m.getattr("investa_core")?;
    let margins = core.call_method1("plausible_margins", (&revenues, &fcfs))?;
    let n = builtin(py, "len")?.call1((&margins,))?;
    let out = PyDict::new(py);
    if n.lt(m.getattr("MIN_CYCLE_OBSERVATIONS")?)? {
        out.set_item("margin", none(py))?;
        out.set_item("observations", n)?;
        return Ok(out);
    }
    out.set_item("margin", core.call_method1("np_median", (&margins,))?)?;
    out.set_item("observations", builtin(py, "len")?.call1((&margins,))?)?;
    Ok(out)
}

fn fcf_result<'py>(
    py: Python<'py>,
    fcf: Bound<'py, PyAny>,
    method: Bound<'py, PyAny>,
    history: &Bound<'py, PyAny>,
    normalized: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("fcf", fcf)?;
    out.set_item("method", method)?;
    out.set_item("history", history)?;
    out.set_item("normalized", normalized)?;
    Ok(out)
}

#[pyfunction]
pub fn fr_normalized_base_fcf<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let o = Ops::new(py)?;
    let len = builtin(py, "len")?;
    let float = builtin(py, "float")?;
    let mut history = PyList::empty(py);
    let collect = || -> PyResult<()> {
        if !none_or_empty(cashflow_df)? {
            let cols = sorted_by_date(py, &cashflow_df.getattr("columns")?)?;
            let fcf_of = m.getattr("_extract_fcf_from_statement")?;
            for col in last_n(&cols, years)?.try_iter()? {
                let fcf = fcf_of.call1((cashflow_df, col?))?;
                if !fcf.is_none() {
                    history.append(fcf)?;
                }
            }
        }
        Ok(())
    };
    match collect() {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("FCF history extraction failed: {}", es(py, &e)),
            )?;
        }
        Err(e) => return Err(e),
    }

    let last_or_none = |h: &Bound<'py, PyList>| -> PyResult<Bound<'py, PyAny>> {
        if h.is_truthy()? {
            h.as_any().get_item(-1)
        } else {
            Ok(none(py))
        }
    };
    let mut latest_fcf = last_or_none(&history)?;
    if latest_fcf.is_none() {
        latest_fcf = ticker_info.call_method1("get", ("freeCashflow",))?;
    }
    let revenue = ticker_info.call_method1("get", ("totalRevenue",))?;
    let revenue_positive = revenue.is_truthy()? && revenue.gt(0)?;

    if revenue_positive {
        let plausible = PyList::empty(py);
        for f in history.iter() {
            if f.div(&revenue)?.le(0.6)? {
                plausible.append(f)?;
            }
        }
        if plausible.len() != history.len() {
            log(
                py,
                "debug",
                format!(
                    "Dropping {} implausible FCF year(s) for {}",
                    history.len() - plausible.len(),
                    o.fmt(&ticker_info.call_method1("get", ("symbol",))?)?
                ),
            )?;
            history = plausible;
            latest_fcf = last_or_none(&history)?;
        }
        if latest_fcf.is_truthy()? && latest_fcf.div(&revenue)?.gt(0.6)? {
            latest_fcf = none(py);
        }
    }

    // Preferred: the cycle.
    let cycle = m
        .getattr("through_cycle_fcf_margin")?
        .call1((financials_df, cashflow_df))?;
    let margin = cycle.get_item("margin")?;
    if margin.is_truthy()? && margin.gt(0)? && revenue.is_truthy()? && revenue.gt(0)? {
        let cycle_fcf = cycle.get_item("margin")?.mul(&revenue)?;
        if cycle_fcf.gt(0)? {
            let method = format!(
                "through-cycle {} FCF margin over {}y x current revenue",
                o.fmt_spec(&cycle.get_item("margin")?, ".1%")?,
                o.fmt(&cycle.get_item("observations")?)?
            );
            return fcf_result(
                py,
                float.call1((cycle_fcf,))?,
                pystr(py, &method),
                history.as_any(),
                true,
            );
        }
    }

    if history.len() >= 3 {
        let floats = PyList::empty(py);
        for f in history.iter() {
            floats.append(float.call1((f,))?)?;
        }
        let median_fcf = m
            .getattr("investa_core")?
            .call_method1("np_median", (floats,))?;
        if median_fcf.gt(0)? {
            let method = format!("median of {}y FCF", o.fmt(&len.call1((&history,))?)?);
            return fcf_result(py, median_fcf, pystr(py, &method), history.as_any(), true);
        }
    }

    let latest_or_zero = if latest_fcf.is_truthy()? {
        latest_fcf.clone()
    } else {
        int(py, 0)?
    };
    if latest_or_zero.gt(0)? {
        return fcf_result(
            py,
            float.call1((latest_fcf,))?,
            pystr(
                py,
                "latest reported FCF (insufficient history to normalize)",
            ),
            history.as_any(),
            false,
        );
    }
    fcf_result(
        py,
        none(py),
        pystr(py, "no positive normalized free cash flow"),
        history.as_any(),
        false,
    )
}

#[pyfunction]
pub fn fr_generate_mc_stats<'py>(
    py: Python<'py>,
    intrinsic_values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let body = || -> PyResult<Bound<'py, PyDict>> {
        let np = m.getattr("np")?;
        let contiguous = np.call_method(
            "ascontiguousarray",
            (intrinsic_values,),
            Some(&kwargs(py, &[("dtype", &np.getattr("float64")?)])?),
        )?;
        let stats = m
            .getattr("investa_core")?
            .call_method1("mc_stats", (contiguous,))?;
        let out = PyDict::new(py);
        if stats.is_none() {
            return Ok(out);
        }
        let s = unpack(&stats, 7)?;

        // 7-point Gaussian smoothing kernel
        let kernel = np.call_method1(
            "array",
            (PyList::new(py, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])?,),
        )?;
        let smoothed = np.call_method(
            "convolve",
            (&s[5], kernel),
            Some(&kwargs(py, &[("mode", &pystr(py, "same"))])?),
        )?;
        let float = builtin(py, "float")?;
        let histogram = PyList::empty(py);
        for pc in builtin(py, "zip")?.call1((&s[6], smoothed))?.try_iter()? {
            let pc = unpack(&pc?, 2)?;
            let bin = PyDict::new(py);
            bin.set_item("price", float.call1((&pc[0],))?)?;
            bin.set_item("count", float.call1((&pc[1],))?)?;
            histogram.append(bin)?;
        }
        for (k, v) in ["bear", "conservative", "base", "bull", "std_dev"]
            .iter()
            .zip(&s)
        {
            out.set_item(*k, v)?;
        }
        out.set_item("histogram", histogram)?;
        Ok(out)
    };
    match body() {
        Ok(d) => Ok(d),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!("Error computing MC stats: {}", es(py, &e)),
            )?;
            Ok(PyDict::new(py))
        }
        Err(e) => Err(e),
    }
}

// --- the Monte Carlo wrappers ---------------------------------------------

/// What every Monte Carlo wrapper reaches for.
struct Mc<'py> {
    py: Python<'py>,
    m: Bound<'py, PyModule>,
    normal: Bound<'py, PyAny>,
    max: Bound<'py, PyAny>,
    abs: Bound<'py, PyAny>,
    float: Bound<'py, PyAny>,
    int: Bound<'py, PyAny>,
}

impl<'py> Mc<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let m = fr(py)?;
        Ok(Mc {
            py,
            normal: m.getattr("np")?.getattr("random")?.getattr("normal")?,
            m,
            max: builtin(py, "max")?,
            abs: builtin(py, "abs")?,
            float: builtin(py, "float")?,
            int: builtin(py, "int")?,
        })
    }
    /// `max(floor, x * k)`
    fn sigma(&self, floor: f64, x: &Bound<'py, PyAny>, k: f64) -> PyResult<Bound<'py, PyAny>> {
        self.max.call1((floor, x.mul(k)?))
    }
    /// `max(floor, abs(x) * k)`
    fn sigma_abs(&self, floor: f64, x: &Bound<'py, PyAny>, k: f64) -> PyResult<Bound<'py, PyAny>> {
        self.sigma(floor, &self.abs.call1((x,))?, k)
    }
    /// `np.random.normal(mean, sigma, iterations)`
    fn draw(
        &self,
        mean: &Bound<'py, PyAny>,
        sigma: Bound<'py, PyAny>,
        n: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.normal.call1((mean, sigma, n))
    }
    fn c(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.m.getattr(name)
    }
    fn f(&self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.float.call1((x,))
    }
    fn kernel(&self, name: &str, args: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        self.m
            .getattr("investa_core")?
            .getattr(name)?
            .call1(PyTuple::new(self.py, args)?)
    }
    fn stats(&self, values: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.m.getattr("_generate_mc_stats")?.call1((values,))
    }
    /// `ticker_info.get(k) or 0`
    fn or_zero(&self, ti: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
        or_else(ti.call_method1("get", (k,))?, || int(self.py, 0))
    }
    /// `shares = ticker_info.get("sharesOutstanding")`, None when `not shares or shares <= 0`.
    fn shares(&self, ti: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let shares = ti.call_method1("get", ("sharesOutstanding",))?;
        if !shares.is_truthy()? || shares.le(0)? {
            return Ok(None);
        }
        Ok(Some(shares))
    }
}

/// `try: ... except Exception as e: logging.error(f"{label}: {e}"); return {}`
fn guarded<'py>(
    py: Python<'py>,
    label: &str,
    body: impl FnOnce(&Mc<'py>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let r = Mc::new(py).and_then(|mc| body(&mc));
    match r {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(py, "error", format!("{label}: {}", es(py, &e)))?;
            Ok(PyDict::new(py).into_any())
        }
        Err(e) => Err(e),
    }
}

fn empty<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
    PyDict::new(py).into_any()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_mc_dcf<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    base_fcf: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo DCF failed", |mc| {
        let Some(shares) = mc.shares(ticker_info)? else {
            return Ok(empty(py));
        };
        let growth = mc.draw(
            base_growth,
            mc.sigma_abs(0.06, base_growth, 0.25)?,
            iterations,
        )?;
        let discount = mc.draw(
            base_discount,
            mc.sigma_abs(0.015, base_discount, 0.15)?,
            iterations,
        )?;
        let cash = mc.or_zero(ticker_info, "totalCash")?;
        let debt = mc.or_zero(ticker_info, "totalDebt")?;
        let values = mc.kernel(
            "mc_dcf",
            vec![
                growth,
                discount,
                mc.c("MIN_PROJECTED_GROWTH")?,
                mc.c("MAX_PROJECTED_GROWTH")?,
                mc.c("MIN_DISCOUNT_RATE")?,
                mc.c("MAX_DISCOUNT_RATE")?,
                mc.int.call1((projection_years,))?,
                mc.f(terminal_growth)?,
                mc.f(base_fcf)?,
                mc.f(&cash)?,
                mc.f(&debt)?,
                mc.f(&shares)?,
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_mc_dcfo<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    base_cfo: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo D-CFO failed", |mc| {
        let Some(shares) = mc.shares(ticker_info)? else {
            return Ok(empty(py));
        };
        let cfo_per_share = base_cfo.div(&shares)?;
        let growth = mc.draw(
            base_growth,
            mc.sigma_abs(0.05, base_growth, 0.25)?,
            iterations,
        )?;
        let discount = mc.draw(
            base_discount,
            mc.sigma_abs(0.015, base_discount, 0.15)?,
            iterations,
        )?;
        let cash = mc.or_zero(ticker_info, "totalCash")?;
        let debt = mc.or_zero(ticker_info, "totalDebt")?;
        let net_cash_per_share = cash.sub(&debt)?.div(&shares)?;
        let values = mc.kernel(
            "mc_discounted",
            vec![
                growth,
                discount,
                fl(py, -0.05),
                fl(py, 0.25),
                mc.c("MIN_DISCOUNT_RATE")?,
                mc.c("MAX_DISCOUNT_RATE")?,
                mc.int.call1((projection_years,))?,
                mc.f(terminal_growth)?,
                mc.f(&cfo_per_share)?,
                mc.f(&net_cash_per_share)?,
            ],
        )?;
        mc.stats(values)
    })
}

/// D-NI and DDM: the same discounted kernel with no net cash.
#[allow(clippy::too_many_arguments)]
fn mc_discounted_per_share<'py>(
    py: Python<'py>,
    label: &str,
    growth_floor: f64,
    growth_cap: f64,
    base: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
    growth_sigma_floor: f64,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, label, |mc| {
        let growth = mc.draw(
            base_growth,
            mc.sigma_abs(growth_sigma_floor, base_growth, 0.25)?,
            iterations,
        )?;
        let discount = mc.draw(
            base_discount,
            mc.sigma_abs(0.015, base_discount, 0.15)?,
            iterations,
        )?;
        let values = mc.kernel(
            "mc_discounted",
            vec![
                growth,
                discount,
                fl(py, growth_floor),
                fl(py, growth_cap),
                mc.c("MIN_DISCOUNT_RATE")?,
                mc.c("MAX_DISCOUNT_RATE")?,
                mc.int.call1((projection_years,))?,
                mc.f(terminal_growth)?,
                mc.f(base)?,
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_dni<'py>(
    py: Python<'py>,
    base_eps: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    mc_discounted_per_share(
        py,
        "Monte Carlo D-NI failed",
        -0.05,
        0.25,
        base_eps,
        base_growth,
        base_discount,
        projection_years,
        terminal_growth,
        iterations,
        0.05,
    )
}

#[pyfunction]
pub fn fr_mc_ddm<'py>(
    py: Python<'py>,
    base_dividend: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    mc_discounted_per_share(
        py,
        "Monte Carlo DDM failed",
        -0.05,
        0.15,
        base_dividend,
        base_growth,
        base_discount,
        projection_years,
        terminal_growth,
        iterations,
        0.03,
    )
}

#[pyfunction]
pub fn fr_mc_mean_pe<'py>(
    py: Python<'py>,
    eps: &Bound<'py, PyAny>,
    applied_pe: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo Mean P/E failed", |mc| {
        let eps_s = mc.draw(eps, mc.sigma_abs(0.1, eps, 0.15)?, iterations)?;
        let pe_s = mc.draw(applied_pe, mc.sigma(1.5, applied_pe, 0.18)?, iterations)?;
        let values = mc.kernel(
            "mc_base_times_multiple",
            vec![
                eps_s,
                pe_s,
                fl(py, 0.01),
                mc.f(&eps.mul(2.5)?)?,
                fl(py, 5.0),
                fl(py, 45.0),
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_peg<'py>(
    py: Python<'py>,
    eps: &Bound<'py, PyAny>,
    applied_growth_pct: &Bound<'py, PyAny>,
    dividend_yield_pct: &Bound<'py, PyAny>,
    target_peg: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo PEG failed", |mc| {
        let eps_s = mc.draw(eps, mc.sigma_abs(0.1, eps, 0.15)?, iterations)?;
        let growth_s = mc.draw(
            applied_growth_pct,
            mc.sigma_abs(1.5, applied_growth_pct, 0.25)?,
            iterations,
        )?;
        let peg_s = mc.draw(target_peg, mc.sigma(0.1, target_peg, 0.15)?, iterations)?;
        let values = mc.kernel(
            "mc_peg",
            vec![
                eps_s,
                growth_s,
                peg_s,
                mc.f(&eps.mul(2.5)?)?,
                mc.f(dividend_yield_pct)?,
            ],
        )?;
        mc.stats(values)
    })
}

/// Mean P/B and mean P/S: sampled base x sampled multiple.
#[allow(clippy::too_many_arguments)]
fn mc_times_multiple<'py>(
    py: Python<'py>,
    label: &str,
    base: &Bound<'py, PyAny>,
    multiple: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
    base_k: f64,
    multiple_floor: f64,
    multiple_lo: f64,
    multiple_hi: f64,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, label, |mc| {
        let base_s = mc.draw(base, mc.sigma(0.1, base, base_k)?, iterations)?;
        let multiple_s = mc.draw(
            multiple,
            mc.sigma(multiple_floor, multiple, 0.18)?,
            iterations,
        )?;
        let values = mc.kernel(
            "mc_base_times_multiple",
            vec![
                base_s,
                multiple_s,
                fl(py, 0.01),
                mc.f(&base.mul(2.0)?)?,
                fl(py, multiple_lo),
                fl(py, multiple_hi),
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_mean_pb<'py>(
    py: Python<'py>,
    book_value_per_share: &Bound<'py, PyAny>,
    applied_pb: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    mc_times_multiple(
        py,
        "Monte Carlo Mean P/B failed",
        book_value_per_share,
        applied_pb,
        iterations,
        0.08,
        0.15,
        0.4,
        8.0,
    )
}

#[pyfunction]
pub fn fr_mc_mean_ps<'py>(
    py: Python<'py>,
    sales_per_share: &Bound<'py, PyAny>,
    applied_ps: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    mc_times_multiple(
        py,
        "Monte Carlo Mean P/S failed",
        sales_per_share,
        applied_ps,
        iterations,
        0.10,
        0.2,
        0.4,
        15.0,
    )
}

#[pyfunction]
pub fn fr_mc_psg<'py>(
    py: Python<'py>,
    sales_per_share: &Bound<'py, PyAny>,
    applied_growth_pct: &Bound<'py, PyAny>,
    gross_margin: &Bound<'py, PyAny>,
    target_psg: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo PSG failed", |mc| {
        let sps_s = mc.draw(
            sales_per_share,
            mc.sigma(0.1, sales_per_share, 0.10)?,
            iterations,
        )?;
        let growth_s = mc.draw(
            applied_growth_pct,
            mc.sigma(2.0, applied_growth_pct, 0.25)?,
            iterations,
        )?;
        let gm_s = mc.draw(
            gross_margin,
            mc.sigma(0.02, gross_margin, 0.08)?,
            iterations,
        )?;
        let psg_s = mc.draw(target_psg, mc.sigma(0.1, target_psg, 0.15)?, iterations)?;
        let values = mc.kernel(
            "mc_psg",
            vec![
                sps_s,
                growth_s,
                gm_s,
                psg_s,
                mc.f(&sales_per_share.mul(2.0)?)?,
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_graham<'py>(
    py: Python<'py>,
    eps: &Bound<'py, PyAny>,
    base_growth: &Bound<'py, PyAny>,
    base_bond_yield: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo Graham failed", |mc| {
        let growth_s = mc.draw(
            base_growth,
            mc.sigma_abs(1.0, base_growth, 0.2)?,
            iterations,
        )?;
        let yield_s = mc.draw(
            base_bond_yield,
            mc.sigma_abs(0.3, base_bond_yield, 0.1)?,
            iterations,
        )?;
        let values = mc.kernel("mc_graham", vec![growth_s, yield_s, mc.f(eps)?])?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_epv<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    normalized_ebit: &Bound<'py, PyAny>,
    tax_rate: &Bound<'py, PyAny>,
    base_discount: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo EPV failed", |mc| {
        let Some(shares) = mc.shares(ticker_info)? else {
            return Ok(empty(py));
        };
        let ebit_s = mc.draw(
            normalized_ebit,
            mc.sigma_abs(1.0, normalized_ebit, 0.15)?,
            iterations,
        )?;
        let discount_s = mc.draw(
            base_discount,
            mc.sigma_abs(0.01, base_discount, 0.12)?,
            iterations,
        )?;
        let cash = mc.or_zero(ticker_info, "totalCash")?;
        let debt = mc.or_zero(ticker_info, "totalDebt")?;
        let values = mc.kernel(
            "mc_epv",
            vec![
                ebit_s,
                discount_s,
                mc.f(&normalized_ebit.mul(2.5)?)?,
                mc.c("MIN_DISCOUNT_RATE")?,
                mc.c("MAX_DISCOUNT_RATE")?,
                mc.f(tax_rate)?,
                mc.f(&cash)?,
                mc.f(&debt)?,
                mc.f(&shares)?,
            ],
        )?;
        mc.stats(values)
    })
}

#[pyfunction]
pub fn fr_mc_lynch<'py>(
    py: Python<'py>,
    eps: &Bound<'py, PyAny>,
    growth_rate_pct: &Bound<'py, PyAny>,
    dividend_yield_pct: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "Monte Carlo Lynch failed", |mc| {
        let eps_s = mc.draw(eps, mc.sigma_abs(0.1, eps, 0.15)?, iterations)?;
        let growth_s = mc.draw(
            growth_rate_pct,
            mc.sigma_abs(1.5, growth_rate_pct, 0.25)?,
            iterations,
        )?;
        let values = mc.kernel(
            "mc_lynch",
            vec![
                eps_s,
                growth_s,
                mc.f(&eps.mul(2.5)?)?,
                mc.f(dividend_yield_pct)?,
            ],
        )?;
        mc.stats(values)
    })
}
