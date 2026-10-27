//! `portfolio_analyzer._build_summary_rows`: one summary row per stock holding
//! and per cash balance, in the display currency, then the cash-interest
//! estimate, each row's share of the total, the cross-account IRR and the
//! account filter.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`), `_HoldingRows` included. Every
//! figure is computed with Python's own operations on the holding's objects,
//! in the original's operand order, so a NumPy quantity keeps the row's values
//! NumPy scalars, `max(0, cost)` can still hand back the integer, and a
//! division by an exact zero raises where it raised. The intra-day flows,
//! which the original computed on Python floats, are native. The finutils
//! helpers and the cash-flow ledger are looked up on `portfolio_analyzer`.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyUnboundLocalError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PySet, PyString};

use crate::mdb::unpack2;
use crate::pyhelp::{es, log, log_exc, pystr};

/// The Python operations the rows are computed with.
pub(crate) struct Ops<'py> {
    pub(crate) py: Python<'py>,
    pub(crate) pd: Bound<'py, PyModule>,
    pub(crate) abs: Bound<'py, PyAny>,
    pub(crate) float: Bound<'py, PyAny>,
    pub(crate) format: Bound<'py, PyAny>,
    pub(crate) max: Bound<'py, PyAny>,
    pub(crate) sum: Bound<'py, PyAny>,
    pub(crate) str_t: Bound<'py, PyAny>,
    pub(crate) int_t: Bound<'py, PyAny>,
}

impl<'py> Ops<'py> {
    pub(crate) fn new(py: Python<'py>) -> PyResult<Self> {
        let b = py.import("builtins")?;
        Ok(Ops {
            py,
            pd: py.import("pandas")?,
            abs: b.getattr("abs")?,
            float: b.getattr("float")?,
            format: b.getattr("format")?,
            max: b.getattr("max")?,
            sum: b.getattr("sum")?,
            str_t: b.getattr("str")?,
            int_t: b.getattr("int")?,
        })
    }
    pub(crate) fn notna(&self, x: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.pd.call_method1("notna", (x,))?.is_truthy()
    }
    pub(crate) fn isna(&self, x: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.pd.call_method1("isna", (x,))?.is_truthy()
    }
    pub(crate) fn fl(&self, v: f64) -> Bound<'py, PyAny> {
        PyFloat::new(self.py, v).into_any()
    }
    pub(crate) fn nan(&self) -> Bound<'py, PyAny> {
        self.fl(f64::NAN)
    }
    pub(crate) fn abs(&self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.abs.call1((x,))
    }
    /// `abs(x) > bound`
    pub(crate) fn abs_gt(&self, x: &Bound<'py, PyAny>, bound: f64) -> PyResult<bool> {
        self.abs(x)?.gt(bound)
    }
    /// `float(x)`
    pub(crate) fn float(&self, x: &Bound<'py, PyAny>) -> PyResult<f64> {
        self.float.call1((x,))?.extract()
    }
    /// `{x}` in an f-string.
    pub(crate) fn fmt(&self, x: &Bound<'py, PyAny>) -> PyResult<String> {
        self.fmt_spec(x, "")
    }
    pub(crate) fn fmt_spec(&self, x: &Bound<'py, PyAny>, spec: &str) -> PyResult<String> {
        Ok(self
            .format
            .call1((x, spec))?
            .cast::<PyString>()?
            .to_str()?
            .to_owned())
    }
    pub(crate) fn is_number(&self, x: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(x.is_instance(&self.int_t)? || x.is_instance(&self.float)?)
    }
    /// `str(x).strip()`
    pub(crate) fn str_strip(&self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.str_t.call1((x,))?.call_method0("strip")
    }
    pub(crate) fn get(
        &self,
        d: &Bound<'py, PyAny>,
        key: &str,
        default: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        d.call_method1("get", (key, default))
    }
    pub(crate) fn get_f(
        &self,
        d: &Bound<'py, PyAny>,
        key: &str,
        default: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.get(d, key, self.fl(default))
    }
}

/// `x * fx_rate if pd.notna(fx_rate) else np.nan`
fn in_display<'py>(
    o: &Ops<'py>,
    x: &Bound<'py, PyAny>,
    fx: &Bound<'py, PyAny>,
    fx_ok: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if fx_ok { x.mul(fx) } else { Ok(o.nan()) }
}

/// `_HoldingRows`: the ledger rows every holding filters, selected once. A
/// failure is kept and raised again for each holding.
struct HoldingRows<'py> {
    df: Bound<'py, PyAny>,
    report_date: Bound<'py, PyAny>,
    days: Option<Result<Bound<'py, PyAny>, PyErr>>,
    on_report_date: Option<Result<Bound<'py, PyAny>, PyErr>>,
    priced: Option<Result<Bound<'py, PyAny>, PyErr>>,
}

/// `_HoldingRows._once`: an `Exception` is memoised and re-raised, anything
/// else propagates and is tried again next time.
pub(crate) fn once<'py>(
    py: Python<'py>,
    slot: &mut Option<Result<Bound<'py, PyAny>, PyErr>>,
    build: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if slot.is_none() {
        match build() {
            Ok(v) => *slot = Some(Ok(v)),
            Err(e) if e.is_instance_of::<PyException>(py) => *slot = Some(Err(e)),
            Err(e) => return Err(e),
        }
    }
    match slot.as_ref().expect("filled above") {
        Ok(v) => Ok(v.clone()),
        Err(e) => Err(e.clone_ref(py)),
    }
}

impl<'py> HoldingRows<'py> {
    fn days(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let df = &self.df;
        once(py, &mut self.days, || {
            df.get_item("Date")?.getattr("dt")?.getattr("date")
        })
    }

    /// `df[df["Date"].dt.date == report_date]`
    fn on_report_date(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(r) = &self.on_report_date {
            return match r {
                Ok(v) => Ok(v.clone()),
                Err(e) => Err(e.clone_ref(py)),
            };
        }
        let built = self
            .days(py)
            .and_then(|d| d.rich_compare(&self.report_date, CompareOp::Eq))
            .and_then(|mask| self.df.get_item(mask));
        let mut slot = None;
        let out = once(py, &mut slot, || built);
        self.on_report_date = slot;
        out
    }

    /// Rows with a positive Price/Share dated on or before the report date.
    fn priced_by_report_date(
        &mut self,
        py: Python<'py>,
        o: &Ops<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(r) = &self.priced {
            return match r {
                Ok(v) => Ok(v.clone()),
                Err(e) => Err(e.clone_ref(py)),
            };
        }
        let built = (|| {
            let df = self.df.clone();
            let price = df.get_item("Price/Share")?;
            let a = price.call_method0("notna")?;
            let kw = PyDict::new(py);
            kw.set_item("errors", "coerce")?;
            let b =
                o.pd.call_method("to_numeric", (df.get_item("Price/Share")?,), Some(&kw))?
                    .rich_compare(1e-9, CompareOp::Gt)?;
            let ab = a.bitand(b)?;
            let c = self
                .days(py)?
                .rich_compare(&self.report_date, CompareOp::Le)?;
            df.get_item(ab.bitand(c)?)
        })();
        let mut slot = None;
        let out = once(py, &mut slot, || built);
        self.priced = slot;
        out
    }
}

/// What the loops carry from one holding to the next.
struct State<'py> {
    has_errors: bool,
    has_warnings: bool,
    /// `cost_basis_display_local`, set by a long holding only: a short one
    /// reads whatever the last long holding left (see the port doc).
    cost_basis_display_local: Option<Bound<'py, PyAny>>,
}

/// `cf_dates` as `calculate_irr` wants them: text parsed, datetimes cut to
/// dates, in place.
fn normalise_dates<'py>(
    m: &Bound<'py, PyModule>,
    o: &Ops<'py>,
    cf_dates: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let datetime = m.getattr("datetime")?;
    for i in 0..cf_dates.len()? {
        let d = cf_dates.get_item(i)?;
        if d.is_instance(&o.str_t)? {
            match datetime
                .call_method1("strptime", (&d, "%Y-%m-%d"))
                .and_then(|v| v.call_method0("date"))
            {
                Ok(v) => cf_dates.set_item(i, v)?,
                Err(e) if e.is_instance_of::<PyValueError>(o.py) => {}
                Err(e) => return Err(e),
            }
        } else if d.is_instance(&datetime)? {
            cf_dates.set_item(i, d.call_method0("date")?)?;
        }
    }
    Ok(())
}

/// `(cf_dates[-1] - cf_dates[0]).days >= 365`
fn spans_a_year(cf_dates: &Bound<'_, PyAny>) -> PyResult<bool> {
    cf_dates
        .get_item(-1)?
        .sub(cf_dates.get_item(0)?)?
        .getattr("days")?
        .ge(365)
}

struct Ctx<'a, 'py> {
    m: &'a Bound<'py, PyModule>,
    o: &'a Ops<'py>,
    display_currency: &'a Bound<'py, PyAny>,
    dc: &'a str,
    default_currency: &'a Bound<'py, PyAny>,
    transactions_df: &'a Bound<'py, PyAny>,
    current_stock_data: &'a Bound<'py, PyAny>,
    report_date: &'a Bound<'py, PyAny>,
    current_fx_rates_vs_usd: &'a Bound<'py, PyAny>,
    current_fx_prev_close_vs_usd: &'a Bound<'py, PyAny>,
    user_excluded_symbols: &'a Bound<'py, PyAny>,
    manual_prices_dict: &'a Bound<'py, PyAny>,
    published_navs: &'a Bound<'py, PyAny>,
    cash_flow_ledger: &'a Bound<'py, PyAny>,
}

impl<'py> Ctx<'_, 'py> {
    fn key(&self, label: &str) -> String {
        format!("{label} ({})", self.dc)
    }
    fn conversion(
        &self,
        local: &Bound<'py, PyAny>,
        rates: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.m
            .getattr("get_conversion_rate")?
            .call1((local, self.display_currency, rates))
    }
}

/// The stock loop's body for one holding.
fn stock_row<'py>(
    c: &Ctx<'_, 'py>,
    st: &mut State<'py>,
    rows_cache: &mut HoldingRows<'py>,
    account_ccy: &Bound<'py, PyDict>,
    holding_key: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let (m, o, py) = (c.m, c.o, c.o.py);
    let (symbol, account) = unpack2(holding_key)?;
    let current_qty = o.get_f(data, "qty", 0.0)?;
    let realized_gain_local = o.get_f(data, "realized_gain_local", 0.0)?;
    let dividends_local = o.get_f(data, "dividends_local", 0.0)?;
    let commissions_local = o.get_f(data, "commissions_local", 0.0)?;
    let taxes_local = o.get_f(data, "taxes_local", 0.0)?;
    let local_currency = o.get(data, "local_currency", c.default_currency.clone())?;
    let current_total_cost_local = o.get_f(data, "total_cost_local", 0.0)?;
    let short_proceeds_local = o.get_f(data, "short_proceeds_local", 0.0)?;
    let total_cost_invested_local = o.get_f(data, "total_cost_invested_local", 0.0)?;
    let cumulative_investment_local = o.get_f(data, "cumulative_investment_local", 0.0)?;
    let total_buy_cost_local = o.get_f(data, "total_buy_cost_local", 0.0)?;
    let total_cost_display_historical_fx_val =
        o.get_f(data, "total_cost_display_historical_fx", f64::NAN)?;
    let realized_gain_display_from_holdings = o.get_f(data, "realized_gain_display", f64::NAN)?;
    let tags_list = o.get(data, "tags", PyList::empty(py).into_any())?;

    account_ccy.set_item(&account, &local_currency)?;
    let stock_data = c
        .current_stock_data
        .call_method1("get", (&symbol, PyDict::new(py)))?;

    // --- Price Determination ---
    // "Unknown" in the original, which every branch below overwrites.
    let mut price_source: String;
    let mut current_price_local = f64::NAN;
    let mut day_change_local = f64::NAN;
    let mut day_change_pct = o.nan();

    let (fs, fa) = (o.fmt(&symbol)?, o.fmt(&account)?);
    if m.getattr("is_cash_symbol")?
        .call1((&symbol,))?
        .is_truthy()?
    {
        price_source = "Internal (Cash)".into();
        current_price_local = 1.0;
        day_change_local = 0.0;
        day_change_pct = o.fl(0.0);
    } else {
        let is_excluded = c.user_excluded_symbols.contains(&symbol)?;
        let raw = stock_data.call_method1("get", ("price",))?;
        let valid = o.notna(&raw)? && o.is_number(&raw)? && raw.gt(1e-9)?;
        if !is_excluded && valid {
            price_source = "Yahoo API/Cache".into();
            current_price_local = o.float(&raw)?;
            let dcl_raw = stock_data.call_method1("get", ("change",))?;
            let dcp_raw = stock_data.call_method1("get", ("changesPercentage",))?;
            day_change_local = if o.notna(&dcl_raw)? {
                o.float(&dcl_raw)?
            } else {
                f64::NAN
            };
            day_change_pct = if o.notna(&dcp_raw)? {
                o.fl(o.float(&dcp_raw)?)
            } else {
                o.nan()
            };
            log(
                py,
                "debug",
                format!(
                    "Price OK ({fs}): Using Yahoo price {}",
                    o.fmt(&o.fl(current_price_local))?
                ),
            )?;
        } else if !is_excluded {
            price_source = "Yahoo Invalid".into();
            st.has_warnings = true;
        } else {
            log(
                py,
                "debug",
                format!("Info: Symbol {fs} is excluded. Skipping Yahoo fetch, trying fallbacks."),
            )?;
            price_source = "Excluded".into();
            st.has_warnings = true;
        }
    }

    // A published NAV sits between the market feed and the manual override.
    if current_price_local.is_nan() && c.published_navs.is_truthy()? {
        let nav_row = c.published_navs.call_method1("get", (&symbol,))?;
        if !nav_row.is_none() {
            let (nav_date, nav_value) = unpack2(&nav_row)?;
            current_price_local = o.float(&nav_value)?;
            price_source = format!("Published NAV ({})", o.fmt(&nav_date)?);
            day_change_local = f64::NAN;
            day_change_pct = o.nan();
        }
    }

    if current_price_local.is_nan() {
        let manual_price = c.manual_prices_dict.call_method1("get", (&symbol,))?;
        if !manual_price.is_none()
            && o.notna(&manual_price)?
            && o.is_number(&manual_price)?
            && manual_price.gt(0)?
        {
            current_price_local = o.float(&manual_price)?;
            price_source += " - Manual Fallback";
            log(
                py,
                "debug",
                format!(
                    "Info: Used MANUAL fallback price {} for {fs}/{fa}",
                    o.fmt(&o.fl(current_price_local))?
                ),
            )?;
            day_change_local = f64::NAN;
            day_change_pct = o.nan();
        } else if c.manual_prices_dict.contains(&symbol)? {
            log(
                py,
                "warning",
                format!(
                    "Warn: Manual price found for {fs} but was invalid/zero ({}). Trying Last TX.",
                    o.fmt(&manual_price)?
                ),
            )?;
        } else {
            log(
                py,
                "debug",
                format!("Info: Manual price not found for {fs}. Trying Last TX."),
            )?;
        }
    }

    if current_price_local.is_nan() {
        if o.abs(&current_qty)?.lt(1e-9)? {
            current_price_local = 0.0;
            price_source += " - Qty 0 (Skip)";
        } else {
            if !price_source.contains("Manual Fallback") {
                price_source += " - No Manual";
            }
            let attempt = (|| -> PyResult<()> {
                let priced = rows_cache.priced_by_report_date(py, o)?;
                let mut account_mask = priced
                    .get_item("Account")?
                    .rich_compare(&account, CompareOp::Eq)?;
                if c.transactions_df
                    .getattr("columns")?
                    .contains("To Account")?
                {
                    account_mask = account_mask.bitor(
                        priced
                            .get_item("To Account")?
                            .rich_compare(&account, CompareOp::Eq)?,
                    )?;
                }
                let mask = priced
                    .get_item("Symbol")?
                    .rich_compare(&symbol, CompareOp::Eq)?
                    .bitand(account_mask)?;
                let fallback_tx = priced.get_item(mask)?.call_method0("copy")?;
                if !fallback_tx.getattr("empty")?.is_truthy()? {
                    let kw = PyDict::new(py);
                    kw.set_item("by", PyList::new(py, ["Date", "original_index"])?)?;
                    kw.set_item("inplace", true)?;
                    kw.set_item("ascending", true)?;
                    fallback_tx.call_method("sort_values", (), Some(&kw))?;
                    let last_tx_row = fallback_tx.getattr("iloc")?.get_item(-1)?;
                    let kw = PyDict::new(py);
                    kw.set_item("errors", "coerce")?;
                    let last_tx_price = o.pd.call_method(
                        "to_numeric",
                        (last_tx_row.get_item("Price/Share")?,),
                        Some(&kw),
                    )?;
                    let last_tx_date = last_tx_row.get_item("Date")?.call_method0("date")?;
                    if o.notna(&last_tx_price)? && last_tx_price.gt(1e-9)? {
                        current_price_local = o.float(&last_tx_price)?;
                        price_source += &format!(
                            " - Last TX ({}@{})",
                            o.fmt_spec(&last_tx_price, ".2f")?,
                            o.fmt(&last_tx_date)?
                        );
                        log(
                            py,
                            "debug",
                            format!(
                                "Info: Used Last TX fallback price {} for {fs}/{fa}",
                                o.fmt(&o.fl(current_price_local))?
                            ),
                        )?;
                    } else {
                        log(
                            py,
                            "warning",
                            format!(
                                "Warn: Fallback TX found for {fs}/{fa} but price invalid ({}). Using 0.",
                                o.fmt(&last_tx_price)?
                            ),
                        )?;
                        price_source += " - Last TX Invalid/Zero";
                        current_price_local = 0.0;
                    }
                } else {
                    log(
                        py,
                        "warning",
                        format!(
                            "Warn: No valid prior TX found for fallback for {fs}/{fa}. Using 0."
                        ),
                    )?;
                    price_source += " - No Last TX";
                    current_price_local = 0.0;
                }
                Ok(())
            })();
            match attempt {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    log(
                        py,
                        "error",
                        format!(
                            "ERROR during last TX fallback for {fs}/{fa}: {}",
                            es(py, &e)
                        ),
                    )?;
                    price_source += " - Fallback Error";
                    current_price_local = 0.0;
                }
                Err(e) => return Err(e),
            }
        }
        day_change_local = f64::NAN;
        day_change_pct = o.nan();
    }

    if current_price_local.is_nan() {
        log(
            py,
            "error",
            format!("ERROR: All price sources failed for {fs}/{fa}. Forcing 0."),
        )?;
        current_price_local = 0.0;
        if !price_source.contains("Using 0") && !price_source.contains("Error") {
            price_source += " - Forced Zero";
        }
        st.has_warnings = true;
    }
    let price = o.fl(current_price_local);

    // --- Currency Conversion ---
    let mut fx_rate = c.conversion(&local_currency, c.current_fx_rates_vs_usd)?;
    if o.isna(&fx_rate)? {
        log(
            py,
            "error",
            format!(
                "CRITICAL ERROR: Failed FX rate {}->{} for {fs}/{fa}.",
                o.fmt(&local_currency)?,
                o.fmt(c.display_currency)?
            ),
        )?;
        st.has_errors = true;
        fx_rate = o.nan();
    }
    let fx_ok = o.notna(&fx_rate)?;

    // --- Unified dividend calculations (forward-looking) ---
    let indicated = m
        .getattr("calculate_indicated_dividend")?
        .call1((&stock_data,))?;
    let dividend_yield_direct = stock_data.call_method1("get", ("dividendYield",))?;
    let mut div_yield_on_cost = o.nan();
    let mut div_yield_on_current = o.nan();
    let mut est_annual_income_local = o.nan();
    if indicated.gt(0)? {
        if o.notna(&current_qty)? && current_qty.gt(1e-9)? {
            est_annual_income_local = indicated.mul(&current_qty)?;
        }
        if o.notna(&current_qty)?
            && current_qty.gt(1e-9)?
            && o.notna(&current_total_cost_local)?
            && current_total_cost_local.gt(1e-9)?
        {
            let avg_cost_price_local = current_total_cost_local.div(&current_qty)?;
            if avg_cost_price_local.gt(1e-9)? {
                div_yield_on_cost = indicated.div(&avg_cost_price_local)?.mul(100.0)?;
            }
        }
        if !current_price_local.is_nan() && current_price_local > 1e-9 {
            div_yield_on_current = indicated.div(&price)?.mul(100.0)?;
        } else if o.notna(&dividend_yield_direct)? && dividend_yield_direct.gt(0)? {
            div_yield_on_current = if dividend_yield_direct.lt(1.0)? {
                dividend_yield_direct.mul(100.0)?
            } else {
                dividend_yield_direct.clone()
            };
        }
    }
    let mut est_annual_income_display = o.nan();
    if o.notna(&est_annual_income_local)? && fx_ok {
        est_annual_income_display = est_annual_income_local.mul(&fx_rate)?;
    }
    log(
        py,
        "debug",
        format!(
            "Symbol: {fs}, Indicated Rate Local: {}, Est Income Local: {}, Est Income Display: {}",
            o.fmt(&indicated)?,
            o.fmt(&est_annual_income_local)?,
            o.fmt(&est_annual_income_display)?
        ),
    )?;
    log(
        py,
        "debug",
        format!(
            "  Yield Cost Local: {}, Yield Current Local: {}",
            o.fmt(&div_yield_on_cost)?,
            o.fmt(&div_yield_on_current)?
        ),
    )?;

    // --- Display-currency values ---
    let market_value_local = current_qty.mul(&price)?;
    let market_value_display = in_display(o, &market_value_local, &fx_rate, fx_ok)?;
    let mut day_change_value_display = o.nan();
    let price_move_known = !day_change_local.is_nan();

    if !current_price_local.is_nan()
        && fx_ok
        && (price_move_known || local_currency.ne(c.display_currency)?)
    {
        let mut qty_change_today = 0.0f64;
        let mut net_flow_today_local = 0.0f64;
        let flows = (|| -> PyResult<()> {
            let today = rows_cache.on_report_date(py)?;
            let tx_today = if today.getattr("empty")?.is_truthy()? {
                today
            } else if c
                .transactions_df
                .getattr("columns")?
                .contains("To Account")?
            {
                let sym = today
                    .get_item("Symbol")?
                    .rich_compare(&symbol, CompareOp::Eq)?;
                let acc = today
                    .get_item("Account")?
                    .rich_compare(&account, CompareOp::Eq)?;
                let to = today
                    .get_item("To Account")?
                    .rich_compare(&account, CompareOp::Eq)?;
                today.get_item(sym.bitand(acc.bitor(to)?)?)?
            } else {
                let sym = today
                    .get_item("Symbol")?
                    .rich_compare(&symbol, CompareOp::Eq)?;
                let acc = today
                    .get_item("Account")?
                    .rich_compare(&account, CompareOp::Eq)?;
                today.get_item(sym.bitand(acc)?)?
            };
            if !tx_today.getattr("empty")?.is_truthy()? {
                for item in crate::pyhelp::iterrows(&tx_today)?.try_iter()? {
                    let (_, tx) = unpack2(&item?)?;
                    let t_type = o
                        .str_t
                        .call1((tx.get_item("Type")?,))?
                        .call_method0("lower")?
                        .call_method0("strip")?;
                    let t_type = t_type.cast::<PyString>()?.to_string_lossy().into_owned();
                    let t_qty = o.float(&tx.get_item("Quantity")?)?.abs();
                    let t_price = o.float(&tx.get_item("Price/Share")?)?;
                    let t_comm = if o.notna(&tx.get_item("Commission")?)? {
                        o.float(&tx.get_item("Commission")?)?
                    } else {
                        0.0
                    };
                    let t_acc = o.str_strip(&o.get(&tx, "Account", pystr(py, ""))?)?;
                    let t_to_acc = o.str_strip(&o.get(&tx, "To Account", pystr(py, ""))?)?;
                    match t_type.as_str() {
                        "buy" | "deposit" | "short sell" => {
                            qty_change_today += if t_type != "short sell" {
                                t_qty
                            } else {
                                -t_qty
                            };
                            net_flow_today_local += (t_qty * t_price) + t_comm;
                        }
                        "sell" | "withdrawal" | "buy to cover" => {
                            qty_change_today -= if t_type != "buy to cover" {
                                t_qty
                            } else {
                                -t_qty
                            };
                            net_flow_today_local -= (t_qty * t_price) - t_comm;
                        }
                        "fees" => net_flow_today_local += t_comm,
                        "transfer" => {
                            if t_to_acc.eq(&account)? {
                                qty_change_today += t_qty;
                                net_flow_today_local += t_qty * t_price;
                            } else if t_acc.eq(&account)? {
                                qty_change_today -= t_qty;
                                net_flow_today_local -= t_qty * t_price;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(())
        })();
        match flows {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "warning",
                format!(
                    "Error calculating intra-day flow for {fs}/{fa}: {}",
                    es(py, &e)
                ),
            )?,
            Err(e) => return Err(e),
        }

        let fx_prev = if local_currency.eq(c.display_currency)? {
            o.fl(1.0)
        } else {
            c.conversion(&local_currency, c.current_fx_prev_close_vs_usd)?
        };
        let starting_qty = current_qty.sub(o.fl(qty_change_today))?;
        if o.notna(&fx_prev)? {
            let val_today = current_qty.mul(&price)?.mul(&fx_rate)?;
            let price_yesterday_local = if price_move_known {
                current_price_local - day_change_local
            } else {
                current_price_local
            };
            let val_yesterday = starting_qty
                .mul(o.fl(price_yesterday_local))?
                .mul(&fx_prev)?;
            let net_flow_today_display = o.fl(net_flow_today_local).mul(&fx_rate)?;
            day_change_value_display =
                val_today.sub(&val_yesterday)?.sub(net_flow_today_display)?;
            if o.abs_gt(&val_yesterday, 1e-9)? {
                day_change_pct = day_change_value_display.div(&val_yesterday)?.mul(100.0)?;
            }
        } else {
            day_change_value_display = starting_qty.mul(o.fl(day_change_local))?.mul(&fx_rate)?;
        }
    }
    let current_price_display = if !current_price_local.is_nan() && fx_ok {
        price.mul(&fx_rate)?
    } else {
        o.nan()
    };
    let mut cost_basis_display = o.nan();
    let mut avg_cost_price_display = o.nan();
    let mut unrealized_gain_display = o.nan();
    let mut unrealized_gain_pct = o.nan();

    let is_long = current_qty.gt(1e-9)?;
    let is_short = current_qty.lt(-1e-9)?;
    if is_long {
        let cbdl = o.max.call1((0, &current_total_cost_local))?;
        cost_basis_display = in_display(o, &cbdl, &fx_rate, fx_ok)?;
        st.cost_basis_display_local = Some(cbdl);
        if o.notna(&cost_basis_display)? {
            avg_cost_price_display = if o.abs_gt(&current_qty, 1e-9)? {
                cost_basis_display.div(&current_qty)?
            } else {
                o.nan()
            };
            if o.notna(&market_value_display)? {
                unrealized_gain_display = market_value_display.sub(&cost_basis_display)?;
                unrealized_gain_pct = if o.abs_gt(&cost_basis_display, 1e-9)? {
                    unrealized_gain_display
                        .div(&cost_basis_display)?
                        .mul(100.0)?
                } else if o.abs_gt(&market_value_display, 1e-9)? {
                    o.fl(f64::INFINITY)
                } else {
                    o.fl(0.0)
                };
            }
        }
    } else if is_short {
        avg_cost_price_display = o.nan();
        cost_basis_display = o.fl(0.0);
        let short_proceeds_display = in_display(o, &short_proceeds_local, &fx_rate, fx_ok)?;
        if o.notna(&market_value_display)? && o.notna(&short_proceeds_display)? {
            let cost_to_cover = o.abs(&market_value_display)?;
            unrealized_gain_display = short_proceeds_display.sub(&cost_to_cover)?;
            unrealized_gain_pct = if o.abs_gt(&short_proceeds_display, 1e-9)? {
                unrealized_gain_display
                    .div(&short_proceeds_display)?
                    .mul(100.0)?
            } else if o.abs_gt(&cost_to_cover, 1e-9)? {
                o.fl(f64::NEG_INFINITY)
            } else {
                o.fl(0.0)
            };
        }
    }

    let realized_gain_display = if o.notna(&realized_gain_display_from_holdings)? {
        realized_gain_display_from_holdings.clone()
    } else {
        in_display(o, &realized_gain_local, &fx_rate, fx_ok)?
    };
    let dividends_display = in_display(o, &dividends_local, &fx_rate, fx_ok)?;
    let commissions_display = in_display(o, &commissions_local, &fx_rate, fx_ok)?;
    let taxes_display = in_display(o, &taxes_local, &fx_rate, fx_ok)?;
    let total_cost_invested_display = in_display(o, &total_cost_invested_local, &fx_rate, fx_ok)?;
    let cumulative_investment_display =
        in_display(o, &cumulative_investment_local, &fx_rate, fx_ok)?;
    let total_buy_cost_display = in_display(o, &total_buy_cost_local, &fx_rate, fx_ok)?;

    let unrealized_gain_comp = if o.notna(&unrealized_gain_display)? {
        unrealized_gain_display.clone()
    } else {
        o.fl(0.0)
    };
    let taxes_comp = if o.notna(&taxes_display)? {
        taxes_display.clone()
    } else {
        o.fl(0.0)
    };
    let total_gain_display = if o.notna(&realized_gain_display)?
        && o.notna(&dividends_display)?
        && o.notna(&commissions_display)?
    {
        realized_gain_display
            .add(&unrealized_gain_comp)?
            .add(&dividends_display)?
            .sub(&commissions_display)?
            .sub(&taxes_comp)?
    } else {
        o.nan()
    };

    let mut total_return_pct = o.nan();
    if o.notna(&total_gain_display)? && o.notna(&total_buy_cost_display)? {
        if o.abs_gt(&total_buy_cost_display, 1e-9)? {
            total_return_pct = total_gain_display
                .div(&total_buy_cost_display)?
                .mul(100.0)?;
        } else if o.abs(&total_gain_display)?.le(1e-9)? {
            total_return_pct = o.fl(0.0);
        }
    }

    let irr = (|| -> PyResult<Bound<'py, PyAny>> {
        let mv_for_irr = if o.abs_gt(&current_qty, 1e-9)? && o.notna(&market_value_local)? {
            o.abs(&market_value_local)?
        } else {
            o.fl(0.0)
        };
        let kw = PyDict::new(py);
        kw.set_item("is_transfer_a_flow", true)?;
        kw.set_item("report_date", c.report_date)?;
        let flows = c.cash_flow_ledger.call_method(
            "flows_for",
            (&symbol, &account, mv_for_irr),
            Some(&kw),
        )?;
        let (cf_dates, cf_values) = unpack2(&flows)?;
        if cf_dates.is_truthy()? && cf_values.is_truthy()? {
            normalise_dates(m, o, &cf_dates)?;
            if spans_a_year(&cf_dates)? {
                return m.getattr("calculate_irr")?.call1((&cf_dates, &cf_values));
            }
            return Ok(o.nan());
        }
        Ok(o.nan())
    })();
    let stock_irr = match irr {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log_exc(
                py,
                "warning",
                format!(
                    "Warning: IRR calculation failed for {fs}/{fa}: {}",
                    es(py, &e)
                ),
                &e,
            )?;
            st.has_warnings = true;
            o.nan()
        }
        Err(e) => return Err(e),
    };

    // --- FX gain/loss ---
    let mut fx_gl = o.nan();
    let mut fx_gl_pct = o.nan();
    if local_currency.eq(c.display_currency)? {
        fx_gl = o.fl(0.0);
        fx_gl_pct = o.fl(0.0);
    } else if o.abs_gt(&current_total_cost_local, 1e-9)?
        && o.abs_gt(&current_total_cost_local, 1e-9)?
        && o.notna(&cost_basis_display)?
        && o.abs_gt(&cost_basis_display, 1e-9)?
        && o.notna(&total_cost_display_historical_fx_val)?
    {
        let hist = &total_cost_display_historical_fx_val;
        let r = (|| -> PyResult<()> {
            fx_gl = cost_basis_display.sub(hist)?;
            if o.notna(&fx_gl)? && o.abs_gt(hist, 1e-9)? {
                fx_gl_pct = fx_gl.div(hist)?.mul(100.0)?;
            } else if o.notna(&fx_gl)? && o.abs(&fx_gl)?.lt(1e-9)? {
                fx_gl_pct = o.fl(0.0);
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyZeroDivisionError>(py) => log(
                py,
                "warning",
                format!(
                    "FX G/L Calc: ZeroDivisionError for {fs}/{fa}. current_total_cost_local: {}",
                    o.fmt(&current_total_cost_local)?
                ),
            )?,
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "error",
                format!("Error calculating FX G/L for {fs}/{fa}: {}", es(py, &e)),
            )?,
            Err(e) => return Err(e),
        }
    }

    let irr_value_to_store = if o.notna(&stock_irr)? {
        stock_irr.mul(100.0)?
    } else {
        o.nan()
    };
    if o.isna(&irr_value_to_store)? && current_qty.ne(0)? {
        log(
            py,
            "debug",
            format!("Debug: IRR is NaN for non-zero holding {fs}/{fa}"),
        )?;
    }

    let row = PyDict::new(py);
    row.set_item("Account", &account)?;
    row.set_item("Symbol", &symbol)?;
    row.set_item("Quantity", &current_qty)?;
    row.set_item(c.key("Avg Cost"), &avg_cost_price_display)?;
    row.set_item(c.key("Price"), &current_price_display)?;
    row.set_item(c.key("Cost Basis"), &cost_basis_display)?;
    row.set_item(c.key("Market Value"), &market_value_display)?;
    row.set_item(c.key("Day Change"), &day_change_value_display)?;
    row.set_item("Day Change %", &day_change_pct)?;
    row.set_item(c.key("Unreal. Gain"), &unrealized_gain_display)?;
    row.set_item("Unreal. Gain %", &unrealized_gain_pct)?;
    row.set_item(c.key("Realized Gain"), &realized_gain_display)?;
    row.set_item(c.key("Dividends"), &dividends_display)?;
    row.set_item(c.key("Commissions"), &commissions_display)?;
    row.set_item(c.key("Taxes"), &taxes_display)?;
    row.set_item(c.key("Total Gain"), &total_gain_display)?;
    row.set_item(c.key("Total Cost Invested"), &total_cost_invested_display)?;
    row.set_item("Total Return %", &total_return_pct)?;
    row.set_item(
        c.key("Cumulative Investment"),
        &cumulative_investment_display,
    )?;
    row.set_item(c.key("Total Buy Cost"), &total_buy_cost_display)?;
    row.set_item("IRR (%)", &irr_value_to_store)?;
    row.set_item("Local Currency", &local_currency)?;
    row.set_item("Price Source", &price_source)?;
    row.set_item("Div. Yield (Cost) %", &div_yield_on_cost)?;
    row.set_item("Div. Yield (Current) %", &div_yield_on_current)?;
    row.set_item(c.key("Est. Ann. Income"), &est_annual_income_display)?;
    row.set_item(c.key("FX Gain/Loss"), &fx_gl)?;
    row.set_item("FX Gain/Loss %", &fx_gl_pct)?;
    let tax = if fx_ok {
        o.get_f(data, "taxes_local", 0.0)?.mul(&fx_rate)?
    } else {
        o.fl(0.0)
    };
    row.set_item(c.key("Tax"), tax)?;
    row.set_item("Name", o.get(&stock_data, "name", pystr(py, ""))?)?;
    row.set_item(
        "sparkline_7d",
        o.get(&stock_data, "sparkline_7d", PyList::empty(py).into_any())?,
    )?;
    row.set_item(
        "sparkline_1m",
        o.get(&stock_data, "sparkline_1m", PyList::empty(py).into_any())?,
    )?;
    for k in ["exchange", "fullExchangeName", "quoteType"] {
        row.set_item(k, stock_data.call_method1("get", (k,))?)?;
    }
    row.set_item("Tags", &tags_list)?;
    row.set_item("Market Value (Local)", &market_value_local)?;
    row.set_item("Price (Local)", &price)?;
    let avg_cost_local = if o.abs_gt(&current_qty, 1e-9)? {
        match &st.cost_basis_display_local {
            Some(cbdl) => cbdl.div(&current_qty)?,
            None => {
                return Err(PyUnboundLocalError::new_err(
                    "cannot access local variable 'cost_basis_display_local' where it is not associated with a value",
                ));
            }
        }
    } else {
        o.nan()
    };
    row.set_item("Avg Cost (Local)", avg_cost_local)?;
    row.set_item("fx_rate", &fx_rate)?;

    if symbol.eq("AAPL")? {
        log(
            py,
            "info",
            format!(
                "DEBUG_MARKET: AAPL Data -> Exchange: {}, Full: {}, MarketOverride: {}",
                o.fmt(&stock_data.call_method1("get", ("exchange",))?)?,
                o.fmt(&stock_data.call_method1("get", ("fullExchangeName",))?)?,
                o.fmt(&data.call_method1("get", ("Market",))?)?
            ),
        )?;
    }
    Ok(row)
}

/// The cash loop's body for one balance, or `None` for a zero balance.
fn cash_row<'py>(
    c: &Ctx<'_, 'py>,
    st: &mut State<'py>,
    currency: &Bound<'py, PyAny>,
    h: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let (m, o, py) = (c.m, c.o, c.o.py);
    let current_qty = o.get_f(h, "qty", 0.0)?;
    if o.abs(&current_qty)?
        .lt(m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?)?
    {
        return Ok(None);
    }
    let account = o.get(h, "account", pystr(py, "Unknown"))?;
    let realized_gain_local = o.get_f(h, "realized_gain_local", 0.0)?;
    let dividends_local = o.get_f(h, "dividends_local", 0.0)?;
    let commissions_local = o.get_f(h, "commissions_local", 0.0)?;
    let taxes_local = o.get_f(h, "taxes_local", 0.0)?;
    let _current_total_cost_local = o.get_f(h, "total_cost_local", 0.0)?;
    let total_cost_invested_local = o.get_f(h, "total_cost_invested_local", 0.0)?;
    let cumulative_investment_local = o.get_f(h, "cumulative_investment_local", 0.0)?;
    let total_buy_cost_local = o.get_f(h, "total_buy_cost_local", 0.0)?;
    let hist = o.get_f(h, "total_cost_display_historical_fx", f64::NAN)?;
    let local_currency = currency;
    let fa = o.fmt(&account)?;

    let symbol_for_display = m
        .getattr("get_currency_symbol_from_code")?
        .call1((local_currency,))?;
    let cash_display_symbol = format!("Cash ({})", o.fmt(&symbol_for_display)?);
    let price_local = o.fl(1.0);

    let mut fx_rate = c.conversion(local_currency, c.current_fx_rates_vs_usd)?;
    if o.isna(&fx_rate)? {
        log(
            py,
            "error",
            format!(
                "CRITICAL ERROR: Failed FX rate {}->{} for cash in {fa}.",
                o.fmt(local_currency)?,
                o.fmt(c.display_currency)?
            ),
        )?;
        st.has_errors = true;
        fx_rate = o.nan();
    }
    let fx_ok = o.notna(&fx_rate)?;

    // The row's yield is filled in by the interest pass; this only has to
    // fail where the original failed.
    let est_annual_income_display = o.fl(0.0);
    let mv_for_yield = current_qty.mul(if fx_ok { fx_rate.clone() } else { o.fl(0.0) })?;
    if mv_for_yield.gt(1e-9)? {
        est_annual_income_display.div(&mv_for_yield)?.mul(100.0)?;
    } else {
        est_annual_income_display.gt(0)?;
    }

    let market_value_local = current_qty.mul(&price_local)?;
    let market_value_display = in_display(o, &market_value_local, &fx_rate, fx_ok)?;

    let mut day_change_value_display = o.fl(0.0);
    let mut day_change_pct = o.fl(0.0);
    if local_currency.ne(c.display_currency)? && fx_ok && o.notna(&current_qty)? {
        let fx_prev = c.conversion(local_currency, c.current_fx_prev_close_vs_usd)?;
        if o.notna(&fx_prev)? {
            let val_today = current_qty.mul(1.0)?.mul(&fx_rate)?;
            let val_yesterday = current_qty.mul(1.0)?.mul(&fx_prev)?;
            day_change_value_display = val_today.sub(&val_yesterday)?;
            if o.abs_gt(&val_yesterday, 1e-9)? {
                day_change_pct = day_change_value_display.div(&val_yesterday)?.mul(100.0)?;
            }
        }
    }

    let current_price_display = in_display(o, &price_local, &fx_rate, fx_ok)?;
    let cost_basis_display = market_value_display.clone();
    let avg_cost_price_display = current_price_display.clone();
    let unrealized_gain_display = o.fl(0.0);
    let unrealized_gain_pct = o.fl(0.0);

    let realized_gain_display = in_display(o, &realized_gain_local, &fx_rate, fx_ok)?;
    let dividends_display = in_display(o, &dividends_local, &fx_rate, fx_ok)?;
    let commissions_display = in_display(o, &commissions_local, &fx_rate, fx_ok)?;
    let taxes_display = in_display(o, &taxes_local, &fx_rate, fx_ok)?;
    let total_cost_invested_display = in_display(o, &total_cost_invested_local, &fx_rate, fx_ok)?;
    let cumulative_investment_display =
        in_display(o, &cumulative_investment_local, &fx_rate, fx_ok)?;
    let total_buy_cost_display = in_display(o, &total_buy_cost_local, &fx_rate, fx_ok)?;

    let taxes_comp = if o.notna(&taxes_display)? {
        taxes_display.clone()
    } else {
        o.fl(0.0)
    };
    let mut total_gain_display = o.nan();
    if o.notna(&realized_gain_display)?
        && o.notna(&dividends_display)?
        && o.notna(&commissions_display)?
    {
        total_gain_display = realized_gain_display
            .add(&unrealized_gain_display)?
            .add(&dividends_display)?
            .sub(&commissions_display)?
            .sub(&taxes_comp)?;
    }

    let mut total_return_pct = o.nan();
    if o.notna(&total_gain_display)? && o.notna(&total_buy_cost_display)? {
        if o.abs_gt(&total_buy_cost_display, 1e-9)? {
            total_return_pct = total_gain_display
                .div(&total_buy_cost_display)?
                .mul(100.0)?;
        } else if o.abs(&total_gain_display)?.le(1e-9)? {
            total_return_pct = o.fl(0.0);
        }
    }

    // A cash balance has no IRR of its own (its account's MWR is the figure).
    let stock_irr = o.nan();

    let mut fx_gl = o.nan();
    let mut fx_gl_pct = o.nan();
    if local_currency.eq(c.display_currency)? {
        fx_gl = o.fl(0.0);
        fx_gl_pct = o.fl(0.0);
    } else if o.notna(&cost_basis_display)?
        && o.abs_gt(&cost_basis_display, 1e-9)?
        && o.notna(&hist)?
    {
        let r = (|| -> PyResult<()> {
            fx_gl = cost_basis_display.sub(&hist)?;
            if o.abs_gt(&hist, 1e-9)? {
                fx_gl_pct = fx_gl.div(&hist)?.mul(100.0)?;
            } else if o.abs(&fx_gl)?.lt(1e-9)? {
                fx_gl_pct = o.fl(0.0);
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "error",
                format!("Error calculating FX G/L for cash {fa}: {}", es(py, &e)),
            )?,
            Err(e) => return Err(e),
        }
    }

    let row = PyDict::new(py);
    row.set_item("Account", &account)?;
    row.set_item("Symbol", cash_display_symbol)?;
    row.set_item("Quantity", &current_qty)?;
    row.set_item(c.key("Avg Cost"), &avg_cost_price_display)?;
    row.set_item(c.key("Price"), &current_price_display)?;
    row.set_item(c.key("Cost Basis"), &cost_basis_display)?;
    row.set_item(c.key("Market Value"), &market_value_display)?;
    row.set_item(c.key("Day Change"), &day_change_value_display)?;
    row.set_item("Day Change %", &day_change_pct)?;
    row.set_item(c.key("Unreal. Gain"), &unrealized_gain_display)?;
    row.set_item("Unreal. Gain %", &unrealized_gain_pct)?;
    row.set_item(c.key("Realized Gain"), &realized_gain_display)?;
    row.set_item(c.key("Dividends"), &dividends_display)?;
    row.set_item(c.key("Commissions"), &commissions_display)?;
    row.set_item(c.key("Taxes"), &taxes_display)?;
    row.set_item(c.key("Total Gain"), &total_gain_display)?;
    row.set_item(c.key("Total Cost Invested"), &total_cost_invested_display)?;
    row.set_item("Total Return %", &total_return_pct)?;
    row.set_item(
        c.key("Cumulative Investment"),
        &cumulative_investment_display,
    )?;
    row.set_item(c.key("Total Buy Cost"), &total_buy_cost_display)?;
    row.set_item("IRR (%)", &stock_irr)?;
    row.set_item("Local Currency", local_currency)?;
    row.set_item("Price Source", "Internal (Cash)")?;
    row.set_item("Div. Yield (Current) %", o.nan())?;
    row.set_item(c.key("Est. Ann. Income"), o.fl(0.0))?;
    row.set_item(c.key("FX Gain/Loss"), &fx_gl)?;
    row.set_item("FX Gain/Loss %", &fx_gl_pct)?;
    row.set_item("Name", "Cash")?;
    row.set_item("Market Value (Local)", &market_value_local)?;
    row.set_item("fx_rate", &fx_rate)?;
    Ok(Some(row))
}

/// `defaultdict(list)[key].append(value)` on a plain dict.
fn push<'py>(
    groups: &Bound<'py, PyDict>,
    key: &Bound<'py, PyAny>,
    value: Bound<'py, PyAny>,
) -> PyResult<()> {
    match groups.get_item(key)? {
        Some(list) => list.call_method1("append", (value,))?,
        None => {
            groups.set_item(key, PyList::new(groups.py(), [value])?)?;
            return Ok(());
        }
    };
    Ok(())
}

/// The cash-interest pass: each account's expected interest, above its
/// interest-free threshold, spread over its cash rows by market value.
fn apply_cash_interest<'py>(
    c: &Ctx<'_, 'py>,
    rows: &Bound<'py, PyList>,
    account_interest_rates: &Bound<'py, PyAny>,
    interest_free_thresholds: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let (o, py) = (c.o, c.o.py);
    let mv_key = c.key("Market Value");
    let income_key = c.key("Est. Ann. Income");
    let account_cash_map = PyDict::new(py);
    for row in rows.iter() {
        let starts = o
            .get(&row, "Symbol", pystr(py, ""))?
            .call_method1("startswith", ("Cash",))?
            .is_truthy()?;
        if starts || row.call_method1("get", ("Symbol",))?.eq("$CASH")? {
            let acc = row.call_method1("get", ("Account",))?;
            if acc.is_truthy()? {
                push(&account_cash_map, &acc, row)?;
            }
        }
    }
    for (acc_name, acc_rows) in account_cash_map.iter() {
        let rate_pct = o.get_any_f(account_interest_rates, &acc_name, 0.0)?;
        if rate_pct.le(0)? {
            continue;
        }
        let thresh = o.get_any_f(interest_free_thresholds, &acc_name, 0.0)?;
        let mut fx_rate = o.fl(1.0);
        if acc_rows.is_truthy()? {
            let first = acc_rows.get_item(0)?;
            if first.contains("fx_rate")? && o.notna(&first.get_item("fx_rate")?)? {
                fx_rate = first.get_item("fx_rate")?;
            }
        }
        let thresh_display = thresh.mul(&fx_rate)?;
        let mvs = PyList::empty(py);
        for r in acc_rows.try_iter()? {
            mvs.append(o.get_f(&r?, &mv_key, 0.0)?)?;
        }
        let total_mv_display = o.sum.call1((mvs,))?;
        let effective = o.max.call1((0.0, total_mv_display.sub(&thresh_display)?))?;
        let total_income = effective.mul(rate_pct.div(100.0)?)?;
        if total_mv_display.gt(1e-9)? {
            for r in acc_rows.try_iter()? {
                let r = r?;
                let mv = o.get_f(&r, &mv_key, 0.0)?;
                let weight = mv.div(&total_mv_display)?;
                let allocated = total_income.mul(&weight)?;
                r.set_item(&income_key, &allocated)?;
                if mv.gt(1e-9)? {
                    r.set_item("Div. Yield (Current) %", allocated.div(&mv)?.mul(100.0)?)?;
                }
            }
        } else {
            total_income.gt(0)?;
        }
    }
    Ok(())
}

impl<'py> Ops<'py> {
    /// `mapping.get(key, default)` for a key that is a Python object.
    pub(crate) fn get_any_f(
        &self,
        d: &Bound<'py, PyAny>,
        key: &Bound<'py, PyAny>,
        default: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        d.call_method1("get", (key, self.fl(default)))
    }
}

/// The cross-account IRR per symbol, written over each row's own IRR.
fn apply_aggregate_irr<'py>(c: &Ctx<'_, 'py>, rows: &Bound<'py, PyList>) -> PyResult<()> {
    let (m, o, py) = (c.m, c.o, c.o.py);
    let agg_mvs = PyDict::new(py);
    for r in rows.iter() {
        let sym = r.call_method1("get", ("Symbol",))?;
        if sym.is_truthy()?
            && !o
                .str_t
                .call1((&sym,))?
                .call_method1("startswith", ("Cash",))?
                .is_truthy()?
        {
            let mv = o.float(&o.get_f(&r, "Market Value (Local)", 0.0)?)?.abs();
            let prev: f64 = match agg_mvs.get_item(&sym)? {
                Some(v) => v.extract()?,
                None => 0.0,
            };
            agg_mvs.set_item(&sym, prev + mv)?;
        }
    }

    let agg_irrs = PyDict::new(py);
    for (sym, total_mv_local) in agg_mvs.iter() {
        let r = (|| -> PyResult<()> {
            let kw = PyDict::new(py);
            kw.set_item("symbol", &sym)?;
            kw.set_item("account", py.None())?;
            kw.set_item("final_market_value", o.float(&total_mv_local)?)?;
            kw.set_item("is_transfer_a_flow", false)?;
            kw.set_item("report_date", c.report_date)?;
            let flows = c.cash_flow_ledger.call_method("flows_for", (), Some(&kw))?;
            let (cf_dates, cf_values) = unpack2(&flows)?;
            if cf_dates.is_truthy()? && cf_values.is_truthy()? {
                normalise_dates(m, o, &cf_dates)?;
                if spans_a_year(&cf_dates)? {
                    let agg_irr = m.getattr("calculate_irr")?.call1((&cf_dates, &cf_values))?;
                    if o.notna(&agg_irr)? {
                        agg_irrs.set_item(&sym, agg_irr.mul(100.0)?)?;
                    }
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
                    "Failed to calculate aggregate IRR for {}: {}",
                    o.fmt(&sym)?,
                    es(py, &e)
                ),
            )?,
            Err(e) => return Err(e),
        }
    }

    for r in rows.iter() {
        let sym = r.call_method1("get", ("Symbol",))?;
        if agg_irrs.contains(&sym)? {
            let v = agg_irrs.as_any().get_item(&sym)?;
            r.set_item("Aggregate IRR (%)", &v)?;
            r.set_item("IRR (%)", agg_irrs.as_any().get_item(&sym)?)?;
        }
    }
    Ok(())
}

type RowsOut<'py> = (Bound<'py, PyList>, Bound<'py, PyDict>, bool, bool);

/// `_build_summary_rows`: (rows, account -> local currency, has errors, has
/// warnings).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_build_summary_rows<'py>(
    py: Python<'py>,
    holdings: &Bound<'py, PyAny>,
    current_stock_data: &Bound<'py, PyAny>,
    current_fx_rates_vs_usd: &Bound<'py, PyAny>,
    current_fx_prev_close_vs_usd: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    report_date: &Bound<'py, PyAny>,
    shortable_symbols: &Bound<'py, PyAny>,
    user_excluded_symbols: &Bound<'py, PyAny>,
    user_symbol_map: &Bound<'py, PyAny>,
    manual_prices_dict: &Bound<'py, PyAny>,
    published_navs: &Bound<'py, PyAny>,
    account_interest_rates: &Bound<'py, PyAny>,
    interest_free_thresholds: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
) -> PyResult<RowsOut<'py>> {
    let _ = (shortable_symbols, user_symbol_map);
    let m = py.import("portfolio_analyzer")?;
    let o = Ops::new(py)?;
    let rows = PyList::empty(py);
    let account_ccy = PyDict::new(py);
    let mut st = State {
        has_errors: false,
        has_warnings: false,
        cost_basis_display_local: None,
    };

    let include_norm = if include_accounts.is_truthy()? {
        let s = PySet::empty(py)?;
        for a in include_accounts.try_iter()? {
            s.add(o.str_strip(&a?)?.call_method0("upper")?)?;
        }
        Some(s)
    } else {
        None
    };
    let empty = || PyDict::new(py).into_any();
    let account_interest_rates = if account_interest_rates.is_none() {
        empty()
    } else {
        account_interest_rates.clone()
    };
    let interest_free_thresholds = if interest_free_thresholds.is_none() {
        empty()
    } else {
        interest_free_thresholds.clone()
    };

    let dc = o.fmt(display_currency)?;
    log(
        py,
        "info",
        format!("Calculating final portfolio summary rows in {dc}..."),
    )?;

    // --- Separate cash from stock holdings ---
    let cash_by_currency = PyDict::new(py);
    let mut stock_holdings: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for item in holdings.call_method0("items")?.try_iter()? {
        let (holding_key, data) = unpack2(&item?)?;
        let (symbol, account) = unpack2(&holding_key)?;
        if m.getattr("is_cash_symbol")?
            .call1((&symbol,))?
            .is_truthy()?
        {
            let currency = o.get(&data, "local_currency", default_currency.clone())?;
            let entry = PyDict::new(py);
            entry.set_item("account", &account)?;
            entry.call_method1("update", (&data,))?;
            push(&cash_by_currency, &currency, entry.into_any())?;
        } else {
            stock_holdings.push((holding_key, data));
        }
    }

    // Every holding's IRR flows come from one read of the ledger.
    let cash_flow_ledger = m.getattr("CashFlowLedger")?.call1((transactions_df,))?;
    let mut rows_cache = HoldingRows {
        df: transactions_df.clone(),
        report_date: report_date.clone(),
        days: None,
        on_report_date: None,
        priced: None,
    };
    let c = Ctx {
        m: &m,
        o: &o,
        display_currency,
        dc: &dc,
        default_currency,
        transactions_df,
        current_stock_data,
        report_date,
        current_fx_rates_vs_usd,
        current_fx_prev_close_vs_usd,
        user_excluded_symbols,
        manual_prices_dict,
        published_navs,
        cash_flow_ledger: &cash_flow_ledger,
    };

    for (holding_key, data) in &stock_holdings {
        let row = stock_row(
            &c,
            &mut st,
            &mut rows_cache,
            &account_ccy,
            holding_key,
            data,
        )?;
        rows.append(row)?;
    }

    // --- Cash holdings, per account ---
    for (currency, holding_list) in cash_by_currency.iter() {
        if !holding_list.is_truthy()? {
            continue;
        }
        for h in holding_list.try_iter()? {
            if let Some(row) = cash_row(&c, &mut st, &currency, &h?)? {
                rows.append(row)?;
            }
        }
    }

    if account_interest_rates.is_truthy()? {
        apply_cash_interest(
            &c,
            &rows,
            &account_interest_rates,
            &interest_free_thresholds,
        )?;
    }

    // --- % of total ---
    let mv_key = c.key("Market Value");
    let mvs = PyList::empty(py);
    for row in rows.iter() {
        if o.notna(&row.call_method1("get", (&mv_key,))?)? {
            mvs.append(o.get_f(&row, &mv_key, 0.0)?)?;
        }
    }
    let total_mv_display = o.sum.call1((mvs,))?;
    for row in rows.iter() {
        let mv = o.get_f(&row, &mv_key, 0.0)?;
        if o.notna(&mv)? && o.abs_gt(&total_mv_display, 1e-9)? {
            row.set_item("pct_of_total", mv.div(&total_mv_display)?.mul(100.0)?)?;
        } else {
            row.set_item("pct_of_total", 0.0)?;
        }
    }

    if rows.is_truthy()? {
        apply_aggregate_irr(&c, &rows)?;
    }

    let rows = match include_norm {
        Some(norm) => {
            let kept = PyList::empty(py);
            for r in rows.iter() {
                let acc = r.call_method1("get", ("Account",))?;
                let acc = if acc.is_truthy()? { acc } else { pystr(py, "") };
                if norm.contains(o.str_strip(&acc)?.call_method0("upper")?)? {
                    kept.append(r)?;
                }
            }
            kept
        }
        None => rows,
    };
    Ok((rows, account_ccy.copy()?, st.has_errors, st.has_warnings))
}
