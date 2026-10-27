//! `portfolio_analyzer`'s standalone helpers: transaction type ids, the
//! correlation matrix, scenario impact, rebalancing trades, HHI, the health
//! score and projected cash interest.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`). The arithmetic is Python's own
//! number operations on the values it is handed (a NumPy scalar stays one),
//! with `sum`, `round`, `max`, `min` and `format` the builtins, so results
//! keep the original's types and rounding. Module constants and siblings are
//! read from `portfolio_analyzer` at call time.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};

use crate::pyhelp::{log, pystr, s};

fn pa(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("portfolio_analyzer")
}

fn builtin<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr(name)
}

fn f<'py>(py: Python<'py>, x: f64) -> Bound<'py, PyAny> {
    PyFloat::new(py, x).into_any()
}

/// `get_type_id`
#[pyfunction]
pub fn pa_get_type_id<'py>(py: Python<'py>, t: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let t: String = t
        .str()?
        .call_method0("lower")?
        .call_method0("strip")?
        .extract()?;
    let name = match t.as_str() {
        "buy" => "TYPE_BUY",
        "sell" => "TYPE_SELL",
        "deposit" => "TYPE_DEPOSIT",
        "withdrawal" => "TYPE_WITHDRAWAL",
        "dividend" => "TYPE_DIVIDEND",
        "fee" | "fees" => "TYPE_FEES",
        "split" | "stock split" => "TYPE_SPLIT",
        "transfer" => "TYPE_TRANSFER",
        "short sell" => "TYPE_SHORT_SELL",
        "buy to cover" => "TYPE_BUY_TO_COVER",
        "tax" | "withholding tax" => "TYPE_TAX",
        "interest" => "TYPE_INTEREST",
        "spin off" | "spin-off" | "spinoff" => "TYPE_SPIN_OFF",
        _ => "TYPE_UNKNOWN",
    };
    pa(py)?.getattr(name)
}

/// `calculate_correlation_matrix`
#[pyfunction]
pub fn pa_correlation_matrix<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    if df.getattr("empty")?.is_truthy()? {
        log(
            py,
            "warning",
            "Input historical_returns_df is empty. Cannot calculate correlation matrix.",
        )?;
        return pd.call_method0("DataFrame");
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "include",
        PyList::new(py, [py.import("numpy")?.getattr("number")?])?,
    )?;
    let numeric = df.call_method("select_dtypes", (), Some(&kwargs))?;
    if numeric.getattr("empty")?.is_truthy()? {
        log(
            py,
            "warning",
            "No numeric columns found in historical_returns_df. Cannot calculate correlation matrix.",
        )?;
        return pd.call_method0("DataFrame");
    }
    numeric.call_method0("corr")
}

/// `run_scenario_analysis`: sum of beta x shock, and its impact on the value.
#[pyfunction]
pub fn pa_run_scenario_analysis<'py>(
    py: Python<'py>,
    factor_betas: &Bound<'py, PyAny>,
    scenario_shocks: &Bound<'py, PyAny>,
    portfolio_value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut ret = f(py, 0.0);
    for item in scenario_shocks.call_method0("items")?.try_iter()? {
        let item = item?;
        let beta = factor_betas.call_method1("get", (item.get_item(0)?, 0.0))?;
        ret = ret.add(beta.mul(item.get_item(1)?)?)?;
    }
    let impact = portfolio_value.mul(&ret)?;
    let out = PyDict::new(py);
    out.set_item("estimated_portfolio_return", ret)?;
    out.set_item("estimated_portfolio_impact", impact)?;
    Ok(out)
}

/// `calculate_rebalancing_trades`: (trades frame, summary).
#[pyfunction]
pub fn pa_rebalancing_trades<'py>(
    py: Python<'py>,
    holdings_df: &Bound<'py, PyAny>,
    target_alloc_pct: &Bound<'py, PyAny>,
    new_cash: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let pd = py.import("pandas")?;
    if holdings_df.getattr("empty")?.is_truthy()? {
        return Ok((pd.call_method0("DataFrame")?, PyDict::new(py)));
    }
    let mv_col = format!("Market Value ({})", s(display_currency)?);
    let price_col = format!("Price ({})", s(display_currency)?);
    let cash = pa(py)?.getattr("CASH_SYMBOL_CSV")?;
    let symbols = holdings_df.get_item("Symbol")?;
    let cash_row = holdings_df.get_item(symbols.call_method1("__eq__", (&cash,))?)?;
    let cash_value = if cash_row.getattr("empty")?.is_truthy()? {
        f(py, 0.0)
    } else {
        cash_row
            .get_item(mv_col.as_str())?
            .getattr("iloc")?
            .get_item(0)?
    };
    let non_cash = || -> PyResult<Bound<'py, PyAny>> {
        holdings_df.get_item(
            holdings_df
                .get_item("Symbol")?
                .call_method1("__ne__", (&cash,))?,
        )
    };
    let invested = non_cash()?.get_item(mv_col.as_str())?.call_method0("sum")?;
    let total = invested.add(&cash_value)?.add(new_cash)?;
    let summary = PyDict::new(py);
    summary.set_item("Total Portfolio Value (After Rebalance)", &total)?;
    summary.set_item("Total Value to Sell", 0.0)?;
    summary.set_item("Total Value to Buy", 0.0)?;
    summary.set_item("Net Cash Change", 0.0)?;
    summary.set_item("Estimated Number of Trades", 0)?;
    let lookup = |col: &str| -> PyResult<Bound<'py, PyAny>> {
        non_cash()?
            .call_method1("set_index", ("Symbol",))?
            .get_item(col)?
            .call_method0("to_dict")
    };
    let values = lookup(&mv_col)?;
    let prices = lookup(&price_col)?;
    let accounts = lookup("Account")?;
    let trades = PyList::empty(py);
    let format = builtin(py, "format")?;
    for item in target_alloc_pct.call_method0("items")?.try_iter()? {
        let item = item?;
        let (symbol, target_pct) = (item.get_item(0)?, item.get_item(1)?);
        let target_value = total.mul(target_pct.div(100.0)?)?;
        if symbol.eq(&cash)? {
            summary.set_item("Net Cash Change", target_value.sub(&cash_value)?)?;
            continue;
        }
        let current = values.call_method1("get", (&symbol, 0.0))?;
        let trade_value = target_value.sub(&current)?;
        if !builtin(py, "abs")?.call1((&trade_value,))?.gt(0.01)? {
            continue;
        }
        let buy = trade_value.gt(0)?;
        let price = prices.call_method1("get", (&symbol, 0.0))?;
        let account = accounts.call_method1("get", (&symbol, "N/A"))?;
        let quantity = if price.gt(0)? {
            trade_value.div(&price)?
        } else {
            0i64.into_pyobject(py)?.into_any()
        };
        let t = PyDict::new(py);
        t.set_item("Action", if buy { "BUY" } else { "SELL" })?;
        t.set_item("Symbol", &symbol)?;
        t.set_item("Account", account)?;
        t.set_item("Quantity", quantity)?;
        t.set_item("Current Price", &price)?;
        t.set_item("Trade Value", &trade_value)?;
        t.set_item(
            "Note",
            format!("Rebalance to {}%", s(&format.call1((&target_pct, ".2f"))?)?),
        )?;
        trades.append(t)?;
        if buy {
            summary.set_item(
                "Total Value to Buy",
                summary
                    .get_item("Total Value to Buy")?
                    .unwrap()
                    .add(&trade_value)?,
            )?;
        } else {
            let sold = builtin(py, "abs")?.call1((&trade_value,))?;
            summary.set_item(
                "Total Value to Sell",
                summary
                    .get_item("Total Value to Sell")?
                    .unwrap()
                    .add(sold)?,
            )?;
        }
    }
    summary.set_item("Estimated Number of Trades", trades.len())?;
    Ok((pd.call_method1("DataFrame", (trades,))?, summary))
}

/// `calculate_hhi`: the sum of squared normalised weights.
#[pyfunction]
pub fn pa_calculate_hhi<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !weights.is_truthy()? {
        return Ok(f(py, 1.0));
    }
    let sum = builtin(py, "sum")?;
    let total = sum.call1((weights,))?;
    if total.eq(0)? {
        return Ok(f(py, 0.0));
    }
    let normalised = PyList::empty(py);
    for w in weights.try_iter()? {
        normalised.append(w?.div(&total)?)?;
    }
    let squares = PyList::empty(py);
    for w in normalised.iter() {
        squares.append(w.pow(2, py.None())?)?;
    }
    sum.call1((squares,))
}

/// `calculate_health_score`: diversification, efficiency and stability.
#[pyfunction]
pub fn pa_health_score<'py>(
    py: Python<'py>,
    summary_df: &Bound<'py, PyAny>,
    risk_metrics: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = risk_free_rate;
    let pd = py.import("pandas")?;
    let mut relevant = pd.call_method0("DataFrame")?;
    let mut mv_col: Option<Bound<'py, PyAny>> = None;
    for c in summary_df.getattr("columns")?.try_iter()? {
        let c = c?;
        if c.call_method1("startswith", ("Market Value (",))?
            .is_truthy()?
        {
            mv_col = Some(c);
            break;
        }
    }
    if let Some(mv) = &mv_col
        && !summary_df.getattr("empty")?.is_truthy()?
        && summary_df.getattr("columns")?.contains("Symbol")?
    {
        let is_total = if summary_df.getattr("columns")?.contains("is_total")? {
            summary_df
                .get_item("is_total")?
                .call_method1("fillna", (false,))?
        } else {
            let kwargs = PyDict::new(py);
            kwargs.set_item("index", summary_df.getattr("index")?)?;
            pd.call_method("Series", (false,), Some(&kwargs))?
        };
        let mask = summary_df
            .get_item("Symbol")?
            .call_method1("__ne__", ("Total",))?
            .call_method1(
                "__and__",
                (summary_df.get_item(mv)?.call_method1("__gt__", (0,))?,),
            )?
            .call_method1("__and__", (is_total.call_method0("__invert__")?,))?;
        relevant = summary_df.get_item(mask)?;
    }
    if relevant.getattr("empty")?.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("overall_score", 0)?;
        out.set_item("components", PyDict::new(py))?;
        out.set_item("rating", "N/A")?;
        return Ok(out);
    }
    let mv = mv_col.expect("relevant rows imply a column");
    let weights = relevant.get_item(&mv)?.call_method0("tolist")?;
    let hhi = pa(py)?.getattr("calculate_hhi")?.call1((weights,))?;
    let (max, min, round) = (
        builtin(py, "max")?,
        builtin(py, "min")?,
        builtin(py, "round")?,
    );
    let clamp = |x: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        max.call1((0, min.call1((100, x))?))
    };
    // 100 * (1.0 - hhi ** 0.6) + 10
    let div = f(py, 1.0)
        .sub(hhi.pow(0.6, py.None())?)?
        .rmul_py(py, 100)?
        .add(10)?;
    let div = clamp(div)?;
    let sharpe = risk_metrics.call_method1("get", ("Sharpe Ratio", 0.0))?;
    let eff = if sharpe.le(0)? {
        max.call1((0, sharpe.mul(10)?.radd_py(py, 30)?))?
    } else if sharpe.le(1.0)? {
        sharpe.mul(50)?.radd_py(py, 30)?
    } else {
        sharpe.sub(1.0)?.mul(40)?.radd_py(py, 80)?
    };
    let eff = clamp(eff)?;
    let vol = risk_metrics.call_method1("get", ("Volatility (Ann.)", 0.0))?;
    let vol_score = if vol.eq(0)? {
        f(py, 50.0)
    } else if vol.lt(0.05)? {
        f(py, 70.0)
    } else if vol.le(0.25)? {
        f(py, 100.0)
    } else {
        max.call1((0, vol.sub(0.25)?.mul(200)?.rsub_py(py, 100)?))?
    };
    let overall = div.mul(0.4)?.add(eff.mul(0.4)?)?.add(vol_score.mul(0.2)?)?;
    let rating = if overall.ge(80)? {
        "Excellent"
    } else if overall.ge(60)? {
        "Good"
    } else if overall.ge(40)? {
        "Fair"
    } else if overall.ge(20)? {
        "Poor"
    } else {
        "Critical"
    };
    let component = |score: &Bound<'py, PyAny>,
                     metric: Bound<'py, PyAny>,
                     label: &str|
     -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("score", round.call1((score, 1))?)?;
        d.set_item("metric", metric)?;
        d.set_item("label", label)?;
        Ok(d)
    };
    let components = PyDict::new(py);
    components.set_item(
        "diversification",
        component(&div, round.call1((&hhi, 3))?, "HHI (Concentration)")?,
    )?;
    components.set_item(
        "efficiency",
        component(&eff, round.call1((&sharpe, 2))?, "Sharpe Ratio")?,
    )?;
    let vol_text = format!("{}%", s(&round.call1((vol.mul(100)?, 1))?)?);
    components.set_item(
        "stability",
        component(&vol_score, pystr(py, &vol_text), "Volatility")?,
    )?;
    let out = PyDict::new(py);
    out.set_item("overall_score", round.call1((&overall, 1))?)?;
    out.set_item("rating", rating)?;
    out.set_item("components", components)?;
    Ok(out)
}

/// Reflected arithmetic (`k op x`) as Python evaluates it.
trait Reflected<'py> {
    fn rmul_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>>;
    fn radd_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>>;
    fn rsub_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>>;
}

impl<'py> Reflected<'py> for Bound<'py, PyAny> {
    fn rmul_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>> {
        k.into_pyobject(py)?.into_any().mul(self)
    }
    fn radd_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>> {
        k.into_pyobject(py)?.into_any().add(self)
    }
    fn rsub_py(&self, py: Python<'py>, k: i64) -> PyResult<Bound<'py, PyAny>> {
        k.into_pyobject(py)?.into_any().sub(self)
    }
}

/// `generate_cash_interest_events`: estimated monthly interest on each
/// account's cash, from the current month to `end_date`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_cash_interest_events<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    interest_rates: &Bound<'py, PyAny>,
    thresholds: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let _ = (start_date, display_currency);
    let events = PyList::empty(py);
    let balances = py
        .import("collections")?
        .getattr("defaultdict")?
        .call1((py.get_type::<PyFloat>(),))?;
    let keys = builtin(py, "list")?.call1((interest_rates.call_method0("keys")?,))?;
    log(
        py,
        "info",
        format!(
            "GenCashInterest: Processing {} summary rows. Interest Rates keys: {}",
            rows.len()?,
            s(&keys)?
        ),
    )?;
    let number = pyo3::types::PyTuple::new(py, [builtin(py, "int")?, builtin(py, "float")?])?;
    for row in rows.try_iter()? {
        let row = row?;
        let sym = row.call_method1("get", ("Symbol", ""))?;
        let is_cash = sym.eq("$CASH")?
            || sym.call_method1("startswith", ("Cash ",))?.is_truthy()?
            || row.call_method1("get", ("Name",))?.eq("Cash")?
            || sym.contains("Cash (")?;
        if !is_cash {
            continue;
        }
        let account = row.call_method1("get", ("Account",))?;
        if !account.is_truthy()? {
            continue;
        }
        let mut mv = f(py, 0.0);
        for item in row.call_method0("items")?.try_iter()? {
            let item = item?;
            let (k, v) = (item.get_item(0)?, item.get_item(1)?);
            if k.call_method1("startswith", ("Market Value (",))?
                .is_truthy()?
                && v.is_instance(&number)?
            {
                mv = v;
                break;
            }
        }
        balances.set_item(&account, balances.get_item(&account)?.add(&mv)?)?;
        log(
            py,
            "info",
            format!(
                "GenCashInterest: Found cash row for {}: {}, MV={}",
                s(&account)?,
                s(&sym)?,
                s(&mv)?
            ),
        )?;
    }
    log(
        py,
        "info",
        format!(
            "GenCashInterest: Aggregated Balances: {}",
            s(&builtin(py, "dict")?.call1((&balances,))?)?
        ),
    )?;
    // `date.today()`, as the original has it (not the market's day).
    let today = py
        .import("datetime")?
        .getattr("date")?
        .call_method0("today")?;
    let pd = py.import("pandas")?;
    let format = builtin(py, "format")?;
    for item in balances.call_method0("items")?.try_iter()? {
        let item = item?;
        let (account, balance) = (item.get_item(0)?, item.get_item(1)?);
        let rate = interest_rates.call_method1("get", (&account,))?;
        if !rate.is_truthy()? || rate.le(0)? {
            continue;
        }
        let threshold = thresholds.call_method1("get", (&account, 0.0))?;
        let effective = builtin(py, "max")?.call1((0, balance.sub(&threshold)?))?;
        if effective.le(0)? {
            continue;
        }
        let annual = effective.mul(rate.div(100.0)?)?;
        let monthly = annual.div(12.0)?;
        if !monthly.gt(0.01)? {
            continue;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("day", 1)?;
        let mut iter_date = today.call_method("replace", (), Some(&kwargs))?;
        log(
            py,
            "info",
            format!(
                "GenCashInterest: Projecting for {}: Rate={}%, Bal={}, Monthly={}, Start={}",
                s(&account)?,
                s(&rate)?,
                s(&balance)?,
                s(&format.call1((&monthly, ".2f"))?)?,
                s(&iter_date)?
            ),
        )?;
        let name = format!("Cash Interest ({})", s(&account)?);
        while iter_date.le(end_date)? {
            let e = PyDict::new(py);
            e.set_item("symbol", "$CASH")?;
            e.set_item("dividend_date", iter_date.str()?)?;
            e.set_item("ex_dividend_date", "")?;
            e.set_item("amount", &monthly)?;
            e.set_item("status", "estimated")?;
            e.set_item("account", &account)?;
            e.set_item("name", PyString::new(py, &name))?;
            events.append(e)?;
            let months = PyDict::new(py);
            months.set_item("months", 1)?;
            let next = pd
                .call_method1("Timestamp", (&iter_date,))?
                .add(pd.call_method("DateOffset", (), Some(&months))?)?;
            iter_date = next.call_method0("date")?;
        }
    }
    Ok(events)
}
