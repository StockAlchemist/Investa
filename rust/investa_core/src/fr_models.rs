//! The twelve valuation models of `financial_ratios`: DCF, Graham, DDM,
//! Lynch, earnings power, discounted CFO, discounted net income, mean P/E,
//! PEG, mean P/B, mean P/S and PSG — their input reading, eligibility checks,
//! the projection call, and the result and diagnostics dicts.
//!
//! Ported line for line from `src/financial_ratios.py` (frozen in
//! `tests/reference/financial_ratios_p15.py`). Figures are combined with
//! Python's own operations in the original's operand order, `np.clip` and the
//! finiteness tests are NumPy's own (through the module's `np`), and every
//! sibling — the statement lookup, the WACC, the growth blend, the
//! normalized FCF, the projection — is looked up on `financial_ratios` at
//! call time. Each model turns any exception into its `{"error": ...}`
//! exactly as the original's `except Exception` did.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::fr::{builtin, fl, fr, int, none, none_or_empty, or_else, pybool};
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, pystr, unpack};

/// What every model reaches for.
struct Md<'py> {
    py: Python<'py>,
    m: Bound<'py, PyModule>,
    np: Bound<'py, PyAny>,
    gsv: Bound<'py, PyAny>,
    float: Bound<'py, PyAny>,
    o: Ops<'py>,
}

impl<'py> Md<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let m = fr(py)?;
        Ok(Md {
            py,
            np: m.getattr("np")?,
            gsv: m.getattr("_get_statement_value")?,
            m,
            float: builtin(py, "float")?,
            o: Ops::new(py)?,
        })
    }
    fn c(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.m.getattr(name)
    }
    fn present(&self, df: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(!none_or_empty(df)?)
    }
    /// `_get_statement_value(df, item, df.columns[0])`
    fn latest(&self, df: &Bound<'py, PyAny>, item: &str) -> PyResult<Bound<'py, PyAny>> {
        self.gsv
            .call1((df, item, df.getattr("columns")?.get_item(0)?))
    }
    /// `float(np.clip(x, lo, hi))`
    fn clip(
        &self,
        x: &Bound<'py, PyAny>,
        lo: impl IntoPyObject<'py>,
        hi: impl IntoPyObject<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.float
            .call1((self.np.call_method1("clip", (x, lo, hi))?,))
    }
    /// `not np.isfinite(v) or v <= 0`
    fn not_finite_positive(&self, v: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(!self.np.call_method1("isfinite", (v,))?.is_truthy()? || v.le(0)?)
    }
    fn get(&self, ti: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
        ti.call_method1("get", (k,))
    }
    /// `ti.get(a) or ti.get(b)`
    fn get_or(&self, ti: &Bound<'py, PyAny>, a: &str, b: &str) -> PyResult<Bound<'py, PyAny>> {
        or_else(self.get(ti, a)?, || self.get(ti, b))
    }
    /// `shares = ti.get("sharesOutstanding")`, from the balance sheet when falsy.
    fn shares(
        &self,
        ti: &Bound<'py, PyAny>,
        bs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shares = self.get(ti, "sharesOutstanding")?;
        if !shares.is_truthy()? && self.present(bs)? {
            return self.latest(bs, "Ordinary Shares Number");
        }
        Ok(shares)
    }
    /// `x = ti.get(key)`, from the balance sheet when None, then `x or 0`.
    fn info_or_bs(
        &self,
        ti: &Bound<'py, PyAny>,
        bs: &Bound<'py, PyAny>,
        key: &str,
        item: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut x = self.get(ti, key)?;
        if x.is_none() && self.present(bs)? {
            x = self.latest(bs, item)?;
        }
        or_else(x, || int(self.py, 0))
    }
    /// The EPS chain: the argument, else `trailingEps`, else the statement.
    fn eps(
        &self,
        ti: &Bound<'py, PyAny>,
        fin: &Bound<'py, PyAny>,
        eps: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut eps = eps.clone();
        if eps.is_none() {
            eps = self.get(ti, "trailingEps")?;
        }
        if eps.is_none() && self.present(fin)? {
            eps = self.latest(fin, "Diluted EPS")?;
        }
        Ok(eps)
    }
    /// `blended_growth_estimate(df, ticker_info=ti, item_name=item)`
    fn blend(
        &self,
        df: &Bound<'py, PyAny>,
        ti: &Bound<'py, PyAny>,
        item: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.c("blended_growth_estimate")?.call(
            (df,),
            Some(&kwargs(
                self.py,
                &[("ticker_info", ti), ("item_name", &pystr(self.py, item))],
            )?),
        )
    }
    fn wacc(
        &self,
        ti: &Bound<'py, PyAny>,
        fin: &Bound<'py, PyAny>,
        bs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.c("calculate_wacc")?.call1((ti, fin, bs))
    }
    /// `wacc_res.get("cost_of_equity") or wacc_res.get("wacc") or 0.09`
    fn cost_of_equity(&self, wacc_res: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        or_else(self.get(wacc_res, "cost_of_equity")?, || {
            or_else(self.get(wacc_res, "wacc")?, || Ok(fl(self.py, 0.09)))
        })
    }
    /// `_scalar_projection(...)`, unpacked.
    fn project(
        &self,
        base: &Bound<'py, PyAny>,
        growth: &Bound<'py, PyAny>,
        discount: &Bound<'py, PyAny>,
        years: &Bound<'py, PyAny>,
        terminal: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let r = self
            .c("_scalar_projection")?
            .call1((base, growth, discount, years, terminal))?;
        let r = unpack(&r, 3)?;
        Ok((r[0].clone(), r[2].clone()))
    }
    fn f(&self, x: &Bound<'py, PyAny>, spec: &str) -> PyResult<String> {
        self.o.fmt_spec(x, spec)
    }
}

fn d<'py>(py: Python<'py>, pairs: Vec<(&str, Bound<'py, PyAny>)>) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    for (k, v) in pairs {
        out.set_item(k, v)?;
    }
    Ok(out.into_any())
}

fn s<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyAny> {
    PyString::new(py, text).into_any()
}

fn error<'py>(py: Python<'py>, msg: &str) -> PyResult<Bound<'py, PyAny>> {
    d(py, vec![("error", s(py, msg))])
}

fn error_with<'py>(
    py: Python<'py>,
    msg: &str,
    diagnostics: Vec<(&str, Bound<'py, PyAny>)>,
) -> PyResult<Bound<'py, PyAny>> {
    d(
        py,
        vec![("error", s(py, msg)), ("diagnostics", d(py, diagnostics)?)],
    )
}

/// `try: ... except Exception as e: return {"error": f"{label}: {str(e)}"}`
fn guarded<'py>(
    py: Python<'py>,
    label: &str,
    body: impl FnOnce(&Md<'py>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match Md::new(py).and_then(|md| body(&md)) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            error(py, &format!("{label}: {}", es(py, &e)))
        }
        Err(e) => Err(e),
    }
}

/// `x if x is not None else other`
fn or_if_none<'py>(x: &Bound<'py, PyAny>, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    if x.is_none() {
        other.clone()
    } else {
        x.clone()
    }
}

/// `(history or {}).get(k)`
fn hist_get<'py>(
    py: Python<'py>,
    history: &Bound<'py, PyAny>,
    k: &str,
) -> PyResult<Bound<'py, PyAny>> {
    or_else(history.clone(), || Ok(PyDict::new(py).into_any()))?.call_method1("get", (k,))
}

/// The four descriptive fields the newer models carry, the same text twice.
fn describe<'py>(
    py: Python<'py>,
    best: &str,
    caveat: &str,
) -> Vec<(&'static str, Bound<'py, PyAny>)> {
    vec![
        ("when_to_use", s(py, best)),
        ("best_suited_for", s(py, best)),
        ("key_limitation", s(py, caveat)),
        ("key_caveats", s(py, caveat)),
    ]
}

fn with_head<'py>(
    head: Vec<(&'static str, Bound<'py, PyAny>)>,
    extra: Vec<(&'static str, Bound<'py, PyAny>)>,
    tail: Vec<(&'static str, Bound<'py, PyAny>)>,
) -> Vec<(&'static str, Bound<'py, PyAny>)> {
    head.into_iter().chain(extra).chain(tail).collect()
}

/// `"; ".join(notes)` into `res["parameters"]["note"]`
fn set_note<'py>(res: &Bound<'py, PyAny>, notes: Vec<String>) -> PyResult<()> {
    res.get_item("parameters")?
        .set_item("note", notes.join("; "))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_iv_dcf<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth_rate: &Bound<'py, PyAny>,
    target_fcf_margin: &Bound<'py, PyAny>,
    fcf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs, cf) = (ticker_info, financials_df, balance_sheet_df, cashflow_df);
    guarded(py, "DCF calculation failed", |md| {
        let mut current_revenue = md.get(ti, "totalRevenue")?;
        if !current_revenue.is_truthy()? && md.present(fin)? {
            current_revenue = md.latest(fin, "Total Revenue")?;
        }
        let mut model_method = "DCF";

        // 1. Base FCF — a normalized figure, not one fiscal year.
        let mut fcf = fcf.clone();
        let mut fcf_method = s(py, "caller-supplied FCF");
        let mut fcf_history = PyList::empty(py).into_any();
        if fcf.is_none() {
            let base_res = md.c("normalized_base_fcf")?.call1((ti, fin, cf))?;
            fcf = base_res.get_item("fcf")?;
            fcf_method = base_res.get_item("method")?;
            fcf_history = base_res.get_item("history")?;
        }

        let mut used_fcf_margin = none(py);
        if (fcf.is_none() || fcf.le(0)?) && current_revenue.is_truthy()? && current_revenue.gt(0)? {
            let mut target_fcf_margin = target_fcf_margin.clone();
            if target_fcf_margin.is_none() {
                target_fcf_margin = md.c("estimate_fcf_margin")?.call1((fin, cf))?;
            }
            let mut positive_years = 0i64;
            for v in fcf_history.try_iter()? {
                if v?.gt(0)? {
                    positive_years += 1;
                }
            }
            let n = fcf_history.len()?;
            let has_track_record = n >= 3 && (positive_years as f64) >= (n as f64) / 2.0;

            if target_fcf_margin.gt(0)? && has_track_record {
                fcf = current_revenue.mul(&target_fcf_margin)?;
                model_method = "Revenue-based DCF";
                used_fcf_margin = target_fcf_margin.clone();
                fcf_method = s(
                    py,
                    &format!(
                        "revenue x normalized {} FCF margin ({}/{} cash-positive years)",
                        md.f(&target_fcf_margin, ".1%")?,
                        positive_years,
                        n
                    ),
                );
            } else {
                return error_with(
                    py,
                    "No sustained positive free cash flow to value",
                    vec![
                        ("fcf_years", int(py, n as i64)?),
                        ("positive_years", int(py, positive_years)?),
                        ("normalized_margin", target_fcf_margin),
                    ],
                );
            }
        }

        if fcf.is_none() || fcf.le(0)? {
            return error(py, "Negative or missing Free Cash Flow");
        }

        // 2. Discount Rate (WACC)
        let discount_rate = if discount_rate.is_none() {
            md.wacc(ti, fin, bs)?.get_item("wacc")?
        } else {
            builtin(py, "max")?.call1((0.075, discount_rate))?
        };

        // 3. Growth Rate
        let mut growth_rate = growth_rate.clone();
        let mut growth_method = s(py, "caller-supplied growth");
        let mut growth_signals = PyDict::new(py).into_any();
        if growth_rate.is_none() {
            let growth_res = md.blend(fin, ti, "Net Income")?;
            growth_rate = growth_res.get_item("growth")?;
            growth_method = growth_res.get_item("method")?;
            growth_signals = growth_res.get_item("signals")?;
        }
        let max_growth = md.c("MAX_PROJECTED_GROWTH")?;
        let applied_growth = md.clip(&growth_rate, md.c("MIN_PROJECTED_GROWTH")?, &max_growth)?;

        // 4-5. Linear growth fade with mid-year discounting, Gordon terminal value.
        let (sum_pv_fcf, pv_terminal_value) = md.project(
            &fcf,
            &applied_growth,
            &discount_rate,
            projection_years,
            terminal_growth_rate,
        )?;

        // 6. Enterprise Value to Equity Value
        let enterprise_value = sum_pv_fcf.add(&pv_terminal_value)?;
        let cash = md.info_or_bs(ti, bs, "totalCash", "Total Cash")?;
        let debt = md.info_or_bs(ti, bs, "totalDebt", "Total Debt")?;
        let equity_value = enterprise_value.add(&cash)?.sub(&debt)?;

        let shares_outstanding = md.shares(ti, bs)?;
        if !shares_outstanding.is_truthy()? {
            return error(py, "Missing shares outstanding");
        }

        let intrinsic_value = equity_value.div(&shares_outstanding)?;
        if md
            .np
            .call_method1("isnan", (&intrinsic_value,))?
            .is_truthy()?
            || md
                .np
                .call_method1("isinf", (&intrinsic_value,))?
                .is_truthy()?
        {
            return error(py, "DCF resulted in invalid NaN/Inf value");
        }

        let tv_share = if enterprise_value.is_truthy()? {
            pv_terminal_value.div(&enterprise_value)?
        } else {
            none(py)
        };

        if intrinsic_value.le(0)? {
            return error_with(
                py,
                "Net debt exceeds discounted cash flows (no residual equity value)",
                vec![
                    ("equity_value", equity_value),
                    ("enterprise_value", enterprise_value),
                ],
            );
        }

        let params = d(
            py,
            vec![
                ("discount_rate", discount_rate),
                ("growth_rate", growth_rate.clone()),
                ("applied_growth", applied_growth.clone()),
                ("terminal_growth_rate", terminal_growth_rate.clone()),
                ("projection_years", projection_years.clone()),
                ("base_fcf", fcf),
                ("total_cash", cash),
                ("total_debt", debt),
                ("shares_outstanding", shares_outstanding),
                ("fcf_margin", used_fcf_margin),
                ("fcf_method", fcf_method),
                ("growth_method", growth_method),
                ("growth_signals", growth_signals),
                ("terminal_value_share", tv_share.clone()),
                ("mid_year_discounting", pybool(py, true)),
            ],
        )?;
        let res = d(
            py,
            vec![
                ("intrinsic_value", intrinsic_value),
                ("model", s(py, model_method)),
                ("parameters", params),
            ],
        )?;
        let mut notes = vec![format!(
            "Linear growth fade over {}y with mid-year discounting",
            md.o.fmt(projection_years)?
        )];
        if growth_rate.gt(&max_growth)? {
            notes.push(format!(
                "growth clamped {} -> {}",
                md.f(&growth_rate, ".1%")?,
                md.f(&applied_growth, ".1%")?
            ));
        }
        if !tv_share.is_none() && tv_share.gt(0.75)? {
            notes.push(format!(
                "{} of value is terminal — low confidence",
                md.f(&tv_share, ".0%")?
            ));
        }
        set_note(&res, notes)?;
        Ok(res)
    })
}

#[pyfunction]
pub fn fr_iv_graham<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    eps: &Bound<'py, PyAny>,
    bond_yield: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs) = (ticker_info, financials_df, balance_sheet_df);
    guarded(py, "Graham calculation failed", |md| {
        let mut growth_rate = growth_rate.clone();
        let mut growth_method = s(py, "caller-supplied growth");
        if growth_rate.is_none() {
            let growth_res = md.blend(fin, ti, "Net Income")?;
            growth_rate = growth_res.get_item("growth")?.mul(100)?;
            growth_method = growth_res.get_item("method")?;
        }

        // Y is a corporate bond yield, floored because 4.4/Y is a divisor.
        let mut bond_yield = bond_yield.clone();
        if bond_yield.is_none() {
            bond_yield = md.c("RISK_FREE_RATE")?.mul(100)?;
        }
        let bond_yield = builtin(py, "max")?.call1((bond_yield, 3.5))?;

        let eps = md.eps(ti, fin, eps)?;

        let mut book_value = none(py);
        if md.present(bs)? {
            let period = bs.getattr("columns")?.get_item(0)?;
            let total_equity = or_else(
                md.gsv.call1((bs, "Total Stockholder Equity", &period))?,
                || {
                    md.gsv
                        .call1((bs, "Total Equity Gross Minority Interest", &period))
                },
            )?;
            let mut shares = md.get(ti, "sharesOutstanding")?;
            if !shares.is_truthy()? && md.present(bs)? {
                shares = md.gsv.call1((bs, "Ordinary Shares Number", &period))?;
            }
            if total_equity.is_truthy()? && shares.is_truthy()? && shares.gt(0)? {
                book_value = total_equity.div(&shares)?;
            }
        }

        // No book-value substitution.
        if eps.is_none() || eps.le(0)? {
            return error_with(
                py,
                "Negative or missing EPS",
                vec![("book_value_per_share", book_value)],
            );
        }

        let applied_growth = md.clip(&growth_rate, -5.0, 15.0)?;
        let intrinsic_value = eps
            .mul(fl(py, 8.5).add(int(py, 2)?.mul(&applied_growth)?)?)?
            .mul(4.4)?
            .div(&bond_yield)?;

        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Graham formula produced a non-positive value");
        }

        let graham_number =
            if book_value.is_truthy()? && book_value.gt(0)? && eps.is_truthy()? && eps.gt(0)? {
                md.float.call1((md
                    .np
                    .call_method1("sqrt", (fl(py, 22.5).mul(&eps)?.mul(&book_value)?,))?,))?
            } else {
                none(py)
            };

        let implied_pe = if eps.is_truthy()? {
            intrinsic_value.div(&eps)?
        } else {
            none(py)
        };
        let params = d(
            py,
            vec![
                ("eps", eps),
                ("growth_rate_pct", growth_rate.clone()),
                ("applied_growth_pct", applied_growth),
                ("bond_yield_proxy", bond_yield),
                ("growth_method", growth_method),
                ("book_value_per_share", book_value),
                ("graham_number", graham_number),
                ("implied_pe", implied_pe),
            ],
        )?;
        let res = d(
            py,
            vec![
                ("intrinsic_value", intrinsic_value),
                ("model", s(py, "Graham's Revised Formula")),
                ("parameters", params),
            ],
        )?;
        if growth_rate.gt(15.0)? {
            res.get_item("parameters")?.set_item(
                "note",
                "Growth capped at 15% — Graham's 'g' is a long-run rate",
            )?;
        }
        Ok(res)
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_iv_ddm<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth_rate: &Bound<'py, PyAny>,
    base_dividend: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, cf, bs) = (ticker_info, financials_df, cashflow_df, balance_sheet_df);
    guarded(py, "DDM calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        let current_price = md.get_or(ti, "currentPrice", "regularMarketPrice")?;

        // 1. Base Dividend per share
        let mut base_dividend = base_dividend.clone();
        if base_dividend.is_none() {
            base_dividend = md.get_or(ti, "dividendRate", "trailingAnnualDividendRate")?;
        }
        if (base_dividend.is_none() || base_dividend.le(0)?)
            && md.present(cf)?
            && shares.is_truthy()?
            && shares.gt(0)?
        {
            let div_paid = md.latest(cf, "Cash Dividends Paid")?;
            if div_paid.is_truthy()? && div_paid.ne(0)? {
                base_dividend = builtin(py, "abs")?.call1((div_paid,))?.div(&shares)?;
            }
        }
        if base_dividend.is_none() || base_dividend.le(0)? {
            return error(py, "No positive dividend rate reported");
        }

        // Dividend sustainability check
        let payout = md.get(ti, "payoutRatio")?;
        if !payout.is_none() && payout.gt(1.5)? {
            return error_with(
                py,
                &format!(
                    "Dividend payout ratio ({}) exceeds sustainable threshold (>150%)",
                    md.f(&payout, ".1%")?
                ),
                vec![("payout_ratio", payout), ("base_dividend", base_dividend)],
            );
        }

        // 2. Dividend Growth Rate
        let mut growth_method = s(py, "caller-supplied dividend growth");
        let applied_growth = if growth_rate.is_none() {
            let div_history = PyList::empty(py);
            if md.present(cf)? {
                let key = crate::fr_growth::date_key(py)?;
                let cols = builtin(py, "sorted")?.call(
                    (cf.getattr("columns")?,),
                    Some(&kwargs(py, &[("key", key.as_any())])?),
                )?;
                let last10 = cols.get_item(builtin(py, "slice")?.call1((-10, none(py)))?)?;
                for c in last10.try_iter()? {
                    let dv = md.gsv.call1((cf, "Cash Dividends Paid", c?))?;
                    if !dv.is_none() && dv.ne(0)? {
                        div_history.append(builtin(py, "abs")?.call1((dv,))?)?;
                    }
                }
            }

            let raw_growth;
            if div_history.len() >= 3 {
                let first_d = div_history.get_item(0)?;
                let last_d = div_history.as_any().get_item(-1)?;
                let n_yrs = div_history.len() as i64 - 1;
                if first_d.gt(0)? && last_d.gt(0)? && n_yrs > 0 {
                    let hist_cagr = last_d
                        .div(&first_d)?
                        .pow(fl(py, 1.0).div(n_yrs)?, py.None())?
                        .sub(1.0)?;
                    raw_growth = if md.np.call_method1("isfinite", (&hist_cagr,))?.is_truthy()? {
                        hist_cagr
                    } else {
                        fl(py, 0.035)
                    };
                    growth_method = s(
                        py,
                        &format!(
                            "{}y historical dividend CAGR ({})",
                            div_history.len(),
                            md.f(&raw_growth, ".1%")?
                        ),
                    );
                } else {
                    raw_growth = fl(py, 0.035);
                }
            } else {
                raw_growth = md.blend(fin, ti, "Net Income")?.get_item("growth")?;
                growth_method = s(
                    py,
                    &format!("Earnings growth blend ({})", md.f(&raw_growth, ".1%")?),
                );
            }

            // Shrink toward sustainable long-run dividend growth (3.5%)
            let shrunk = fl(py, 0.35).mul(&raw_growth)?.add(0.65 * 0.035)?;
            md.clip(&shrunk, -0.02, 0.12)?
        } else {
            md.clip(growth_rate, -0.05, 0.15)?
        };

        // 3. Cost of Equity (CAPM)
        let mut discount_rate = discount_rate.clone();
        if discount_rate.is_none() {
            discount_rate = md.cost_of_equity(&md.wacc(ti, fin, bs)?)?;
        }
        let discount_rate = md.clip(
            &discount_rate,
            md.c("MIN_DISCOUNT_RATE")?,
            md.c("MAX_DISCOUNT_RATE")?,
        )?;

        // 4-5. Projections with mid-year discounting, Gordon terminal value
        let (sum_pv_divs, pv_terminal_value) = md.project(
            &base_dividend,
            &applied_growth,
            &discount_rate,
            projection_years,
            terminal_growth_rate,
        )?;
        let intrinsic_value = sum_pv_divs.add(&pv_terminal_value)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "DDM produced non-positive value");
        }
        let tv_share = if intrinsic_value.is_truthy()? {
            pv_terminal_value.div(&intrinsic_value)?
        } else {
            none(py)
        };

        let dividend_yield_pct = if current_price.is_truthy()? && current_price.gt(0)? {
            base_dividend.div(&current_price)?.mul(100)?
        } else {
            none(py)
        };
        let params = d(
            py,
            vec![
                ("base_dividend", base_dividend),
                ("dividend_yield_pct", dividend_yield_pct),
                ("payout_ratio", payout),
                ("growth_rate", or_if_none(growth_rate, &applied_growth)),
                ("applied_growth", applied_growth.clone()),
                ("terminal_growth_rate", terminal_growth_rate.clone()),
                ("discount_rate", discount_rate),
                ("projection_years", projection_years.clone()),
                ("growth_method", growth_method),
                ("terminal_value_share", tv_share.clone()),
                ("mid_year_discounting", pybool(py, true)),
            ],
        )?;
        let res = d(
            py,
            vec![
                ("intrinsic_value", intrinsic_value),
                ("model", s(py, "Dividend Discount Model (Multi-Stage)")),
                ("parameters", params),
            ],
        )?;
        let mut notes = vec![format!(
            "Multi-stage DDM fading to {} terminal growth",
            md.f(terminal_growth_rate, ".1%")?
        )];
        if !tv_share.is_none() && tv_share.gt(0.75)? {
            notes.push(format!("{} of value is terminal", md.f(&tv_share, ".0%")?));
        }
        set_note(&res, notes)?;
        Ok(res)
    })
}

/// `(div_rate / price) * k` when both are there, else `dividendYield * k`,
/// else the default — the yield the Lynch and PEG models add to growth.
fn dividend_yield_pct<'py>(
    md: &Md<'py>,
    ti: &Bound<'py, PyAny>,
    k: Bound<'py, PyAny>,
    default: Option<Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let div_rate = md.get_or(ti, "dividendRate", "trailingAnnualDividendRate")?;
    let price = md.get_or(ti, "currentPrice", "regularMarketPrice")?;
    if div_rate.is_truthy()? && price.is_truthy()? && price.gt(0)? {
        return Ok(Some(div_rate.div(&price)?.mul(&k)?));
    }
    if md.get(ti, "dividendYield")?.is_truthy()? {
        return Ok(Some(ti.get_item("dividendYield")?.mul(&k)?));
    }
    Ok(default)
}

#[pyfunction]
pub fn fr_iv_lynch<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    eps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin) = (ticker_info, financials_df);
    guarded(py, "Lynch calculation failed", |md| {
        let eps = md.eps(ti, fin, eps)?;
        if eps.is_none() || eps.le(0)? {
            return error(py, "Negative or missing EPS");
        }
        let mut growth_rate = growth_rate.clone();
        if growth_rate.is_none() {
            growth_rate = md
                .blend(fin, ti, "Net Income")?
                .get_item("growth")?
                .mul(100)?;
        }
        let div_yield_pct =
            dividend_yield_pct(md, ti, int(py, 100)?, Some(fl(py, 0.0)))?.expect("defaulted");

        // Fair multiplier = growth + dividend yield, clamped to [5.0, 25.0]
        let fair_multiplier = md.clip(&growth_rate.add(&div_yield_pct)?, 5.0, 25.0)?;
        let intrinsic_value = eps.mul(&fair_multiplier)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Lynch formula produced non-positive value");
        }
        let params = d(
            py,
            vec![
                ("eps", eps),
                ("growth_rate_pct", growth_rate),
                ("dividend_yield_pct", div_yield_pct),
                ("fair_pe_multiplier", fair_multiplier),
                (
                    "note",
                    s(
                        py,
                        "Fair Value where PEG = 1.0 (adjusted for dividend yield)",
                    ),
                ),
            ],
        )?;
        d(
            py,
            vec![
                ("intrinsic_value", intrinsic_value),
                ("model", s(py, "Peter Lynch Fair Value")),
                ("parameters", params),
            ],
        )
    })
}

#[pyfunction]
pub fn fr_iv_epv<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = cashflow_df;
    let (ti, fin, bs) = (ticker_info, financials_df, balance_sheet_df);
    guarded(py, "EPV calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        if !shares.is_truthy()? || shares.le(0)? {
            return error(py, "Missing shares outstanding");
        }

        // Normalized EBIT across the cycle.
        let ebit_history = PyList::empty(py);
        if md.present(fin)? {
            let key = crate::fr_growth::date_key(py)?;
            let cols = builtin(py, "sorted")?.call(
                (fin.getattr("columns")?,),
                Some(&kwargs(py, &[("key", key.as_any())])?),
            )?;
            let last5 = cols.get_item(builtin(py, "slice")?.call1((-5, none(py)))?)?;
            for col in last5.try_iter()? {
                let ebit = md.gsv.call1((fin, "Operating Income", col?))?;
                if !ebit.is_none() {
                    ebit_history.append(ebit)?;
                }
            }
        }
        if !ebit_history.is_truthy()? {
            return error(py, "No operating income history for earnings power");
        }
        let floats = PyList::empty(py);
        for e in ebit_history.iter() {
            floats.append(md.float.call1((e,))?)?;
        }
        let normalized_ebit = md.c("investa_core")?.call_method1("np_median", (floats,))?;
        if normalized_ebit.le(0)? {
            return error_with(
                py,
                "No positive normalized operating income",
                vec![("normalized_ebit", normalized_ebit)],
            );
        }

        // Effective tax rate from the same statements, banded to sane values.
        let mut tax_rate = fl(py, 0.21);
        if md.present(fin)? {
            let latest = fin.getattr("columns")?.get_item(0)?;
            let tax = md.gsv.call1((fin, "Tax Provision", &latest))?;
            let pretax = md.gsv.call1((fin, "Pretax Income", &latest))?;
            if !tax.is_none() && pretax.is_truthy()? && pretax.gt(0)? {
                tax_rate = md.clip(&tax.div(&pretax)?, 0.0, 0.5)?;
            }
        }

        let mut discount_rate = discount_rate.clone();
        if discount_rate.is_none() {
            discount_rate = md.wacc(ti, fin, bs)?.get_item("wacc")?;
        }
        let discount_rate = md.clip(
            &discount_rate,
            md.c("MIN_DISCOUNT_RATE")?,
            md.c("MAX_DISCOUNT_RATE")?,
        )?;

        let nopat = normalized_ebit.mul(int(py, 1)?.sub(&tax_rate)?)?;
        let enterprise_value = nopat.div(&discount_rate)?;
        let cash = md.info_or_bs(ti, bs, "totalCash", "Total Cash")?;
        let debt = md.info_or_bs(ti, bs, "totalDebt", "Total Debt")?;

        let equity_value = enterprise_value.add(&cash)?.sub(&debt)?;
        if equity_value.le(0)? {
            return error_with(
                py,
                "Net debt exceeds earnings power value",
                vec![("enterprise_value", enterprise_value)],
            );
        }
        let intrinsic_value = equity_value.div(&shares)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "EPV produced an invalid value");
        }

        let params = d(
            py,
            vec![
                ("normalized_ebit", normalized_ebit),
                ("ebit_years", int(py, ebit_history.len() as i64)?),
                ("tax_rate", tax_rate),
                ("discount_rate", discount_rate),
                ("nopat", nopat),
                ("net_cash", cash.sub(&debt)?),
                ("shares", shares),
                (
                    "note",
                    s(
                        py,
                        "Assumes zero growth; value of current earning power only",
                    ),
                ),
            ],
        )?;
        let text = "Conservative valuation of normalized sustainable operating earnings in perpetuity assuming zero future growth.";
        let caveat = "Strictly a no-growth baseline floor; gives zero credit to value-accretive capital reinvestment or growth opportunities.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Earnings Power Value (no growth)")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_iv_dcfo<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth_rate: &Bound<'py, PyAny>,
    base_cfo: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs, cf) = (ticker_info, financials_df, balance_sheet_df, cashflow_df);
    guarded(py, "Discounted CFO calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        if !shares.is_truthy()? || shares.le(0)? {
            return error(py, "Missing shares outstanding");
        }

        let mut base_cfo = base_cfo.clone();
        if base_cfo.is_none() {
            if md.present(cf)? {
                base_cfo = md.latest(cf, "Operating Cash Flow")?;
            }
            if base_cfo.is_none() {
                base_cfo = md.get(ti, "operatingCashflow")?;
            }
        }
        if base_cfo.is_none() || base_cfo.le(0)? {
            return error(py, "Negative or missing Operating Cash Flow");
        }
        let cfo_per_share = base_cfo.div(&shares)?;

        let mut growth_method = s(py, "caller-supplied growth");
        let applied_growth = if growth_rate.is_none() {
            let source = if cf.is_none() { fin } else { cf };
            let growth_res = md.blend(source, ti, "Operating Cash Flow")?;
            let raw_growth = growth_res.get_item("growth")?;
            growth_method = growth_res.get_item("method")?;
            let shrunk = fl(py, 0.40).mul(&raw_growth)?.add(0.60 * 0.04)?;
            md.clip(&shrunk, -0.05, 0.20)?
        } else {
            md.clip(growth_rate, -0.05, 0.20)?
        };

        let mut discount_rate = discount_rate.clone();
        if discount_rate.is_none() {
            discount_rate = md.wacc(ti, fin, bs)?.call_method1("get", ("wacc", 0.09))?;
        }
        let discount_rate = md.clip(
            &discount_rate,
            md.c("MIN_DISCOUNT_RATE")?,
            md.c("MAX_DISCOUNT_RATE")?,
        )?;

        let (sum_pv_cfos, pv_terminal) = md.project(
            &cfo_per_share,
            &applied_growth,
            &discount_rate,
            projection_years,
            terminal_growth_rate,
        )?;
        let cash = md.info_or_bs(ti, bs, "totalCash", "Total Cash")?;
        let debt = md.info_or_bs(ti, bs, "totalDebt", "Total Debt")?;
        let net_cash_per_share = cash.sub(&debt)?.div(&shares)?;
        let intrinsic_value = sum_pv_cfos.add(&pv_terminal)?.add(&net_cash_per_share)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Discounted CFO produced non-positive value");
        }

        let params = d(
            py,
            vec![
                ("base_cfo", base_cfo),
                ("cfo_per_share", cfo_per_share),
                ("growth_rate", or_if_none(growth_rate, &applied_growth)),
                ("applied_growth", applied_growth.clone()),
                ("discount_rate", discount_rate),
                ("terminal_growth_rate", terminal_growth_rate.clone()),
                ("projection_years", projection_years.clone()),
                ("net_cash_per_share", net_cash_per_share),
                ("growth_method", growth_method),
                ("mid_year_discounting", pybool(py, true)),
            ],
        )?;
        let text = "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).";
        let caveat = "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Discounted Cash from Operations")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_iv_dni<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth_rate: &Bound<'py, PyAny>,
    base_eps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs) = (ticker_info, financials_df, balance_sheet_df);
    guarded(py, "Discounted Net Income calculation failed", |md| {
        let base_eps = md.eps(ti, fin, base_eps)?;
        if base_eps.is_none() || base_eps.le(0)? {
            return error(py, "Negative or missing Net Income / EPS");
        }

        let mut growth_method = s(py, "caller-supplied earnings growth");
        let applied_growth = if growth_rate.is_none() {
            let growth_res = md.blend(fin, ti, "Net Income")?;
            let raw_growth = growth_res.get_item("growth")?;
            growth_method = growth_res.get_item("method")?;
            let shrunk = fl(py, 0.40).mul(&raw_growth)?.add(0.60 * 0.04)?;
            md.clip(&shrunk, -0.05, 0.20)?
        } else {
            md.clip(growth_rate, -0.05, 0.20)?
        };

        let mut discount_rate = discount_rate.clone();
        if discount_rate.is_none() {
            discount_rate = md.cost_of_equity(&md.wacc(ti, fin, bs)?)?;
        }
        let discount_rate = md.clip(
            &discount_rate,
            md.c("MIN_DISCOUNT_RATE")?,
            md.c("MAX_DISCOUNT_RATE")?,
        )?;

        let (sum_pv_eps, pv_terminal) = md.project(
            &base_eps,
            &applied_growth,
            &discount_rate,
            projection_years,
            terminal_growth_rate,
        )?;
        let intrinsic_value = sum_pv_eps.add(&pv_terminal)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Discounted Net Income produced non-positive value");
        }

        let params = d(
            py,
            vec![
                ("base_eps", base_eps),
                ("growth_rate", or_if_none(growth_rate, &applied_growth)),
                ("applied_growth", applied_growth.clone()),
                ("discount_rate", discount_rate),
                ("terminal_growth_rate", terminal_growth_rate.clone()),
                ("projection_years", projection_years.clone()),
                ("growth_method", growth_method),
                ("mid_year_discounting", pybool(py, true)),
            ],
        )?;
        let text = "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.";
        let caveat = "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Discounted Net Income")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

/// The traded-multiple history a mean-multiple model applies: (multiple,
/// source) from `history`, or None when there is none to average.
fn traded_multiple<'py>(
    md: &Md<'py>,
    history: &Bound<'py, PyAny>,
    label: &str,
    spec: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    if !history.is_truthy()? || !md.get(history, "value")?.is_truthy()? {
        return Ok(None);
    }
    let value = md.float.call1((history.get_item("value")?,))?;
    let source = format!(
        "Median traded {label} {}x ({} years, {})",
        md.f(&value, spec)?,
        md.o.fmt(&history.get_item("n")?)?,
        md.o.fmt(&history.get_item("span")?)?
    );
    Ok(Some((value, s(md.py, &source))))
}

fn multiple_stats<'py>(
    py: Python<'py>,
    history: &Bound<'py, PyAny>,
) -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>> {
    Ok(vec![
        ("multiple_observations", hist_get(py, history, "n")?),
        ("multiple_span", hist_get(py, history, "span")?),
        ("multiple_p25", hist_get(py, history, "low")?),
        ("multiple_p75", hist_get(py, history, "high")?),
    ])
}

fn no_history_error(kind: &str) -> String {
    format!(
        "No traded {kind} history to average — a fair multiple must come from what this company has actually changed hands at"
    )
}

#[pyfunction]
pub fn fr_iv_mean_pe<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    mean_pe: &Bound<'py, PyAny>,
    eps: &Bound<'py, PyAny>,
    history: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin) = (ticker_info, financials_df);
    guarded(py, "Mean P/E calculation failed", |md| {
        let eps = md.eps(ti, fin, eps)?;
        if eps.is_none() || eps.le(0)? {
            return error(py, "Negative or missing EPS");
        }
        let mut mean_pe = mean_pe.clone();
        let mut pe_source = s(py, "caller-supplied");
        if mean_pe.is_none() {
            let Some((value, source)) = traded_multiple(md, history, "P/E", ".1f")? else {
                return error_with(py, &no_history_error("P/E"), vec![("eps", eps)]);
            };
            mean_pe = value;
            pe_source = source;
        }
        let applied_pe = md.clip(&mean_pe, 5.0, 45.0)?;
        let intrinsic_value = eps.mul(&applied_pe)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Mean P/E produced non-positive value");
        }
        let params = d(
            py,
            with_head(
                vec![
                    ("eps", eps),
                    ("mean_pe", mean_pe),
                    ("applied_pe", applied_pe),
                    ("pe_source", pe_source),
                ],
                multiple_stats(py, history)?,
                vec![],
            ),
        )?;
        let text = "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.";
        let caveat = "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Mean P/E Ratio")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

#[pyfunction]
pub fn fr_iv_peg<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    growth_rate: &Bound<'py, PyAny>,
    eps: &Bound<'py, PyAny>,
    target_peg: &Bound<'py, PyAny>,
    dividend_yield_pct: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin) = (ticker_info, financials_df);
    guarded(py, "PEG calculation failed", |md| {
        let eps = md.eps(ti, fin, eps)?;
        if eps.is_none() || eps.le(0)? {
            return error(py, "Negative or missing EPS");
        }
        let mut growth_rate = growth_rate.clone();
        let mut growth_method = s(py, "caller-supplied growth");
        if growth_rate.is_none() {
            let growth_res = md.blend(fin, ti, "Net Income")?;
            growth_rate = growth_res.get_item("growth")?.mul(100.0)?;
            growth_method = growth_res.get_item("method")?;
        }
        let dividend_yield_pct = if dividend_yield_pct.is_none() {
            dividend_yield_pct_or_zero(md, ti)?
        } else {
            dividend_yield_pct.clone()
        };

        let applied_growth = md.clip(&growth_rate, 3.0, 35.0)?;
        let target_peg = md.clip(target_peg, 0.5, 2.0)?;
        let fair_pe_multiplier = md.clip(
            &target_peg.mul(applied_growth.add(&dividend_yield_pct)?)?,
            5.0,
            35.0,
        )?;
        let intrinsic_value = eps.mul(&fair_pe_multiplier)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "PEG calculation produced non-positive value");
        }
        let params = d(
            py,
            vec![
                ("eps", eps),
                ("growth_rate_pct", growth_rate),
                ("applied_growth_pct", applied_growth),
                ("dividend_yield_pct", dividend_yield_pct),
                ("target_peg", target_peg),
                ("fair_pe_multiplier", fair_pe_multiplier),
                ("growth_method", growth_method),
            ],
        )?;
        let text = "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.";
        let caveat = "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "PEG Ratio Fair Value")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

/// PEG's yield: `* 100.0`, else 0.0.
fn dividend_yield_pct_or_zero<'py>(
    md: &Md<'py>,
    ti: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(dividend_yield_pct(md, ti, fl(md.py, 100.0), Some(fl(md.py, 0.0)))?.expect("defaulted"))
}

#[pyfunction]
pub fn fr_iv_mean_pb<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    book_value_per_share: &Bound<'py, PyAny>,
    mean_pb: &Bound<'py, PyAny>,
    history: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, bs) = (ticker_info, balance_sheet_df);
    guarded(py, "Mean P/B calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        let mut bvps = book_value_per_share.clone();
        if bvps.is_none() {
            bvps = md.get(ti, "bookValue")?;
            if (bvps.is_none() || bvps.le(0)?)
                && md.present(bs)?
                && shares.is_truthy()?
                && shares.gt(0)?
            {
                let total_equity = md.latest(bs, "Total Stockholder Equity")?;
                if total_equity.is_truthy()? && total_equity.gt(0)? {
                    bvps = total_equity.div(&shares)?;
                }
            }
        }
        if bvps.is_none() || bvps.le(0)? {
            return error(py, "Negative or missing Book Value per Share");
        }

        let sector = or_else(md.get(ti, "sector")?, || Ok(s(py, "")))?.call_method0("lower")?;
        let mut mean_pb = mean_pb.clone();
        let mut pb_source = s(py, "caller-supplied");
        if mean_pb.is_none() {
            if let Some((value, source)) = traded_multiple(md, history, "P/B", ".2f")? {
                mean_pb = value;
                pb_source = source;
            } else if sector.contains("bank")? || sector.contains("financial")? {
                mean_pb = fl(py, 1.30);
                pb_source = s(py, "Banking benchmark (1.30x) — no traded history");
            } else if sector.contains("real estate")? || sector.contains("reit")? {
                mean_pb = fl(py, 1.05);
                pb_source = s(py, "REIT NAV benchmark (1.05x) — no traded history");
            } else {
                return error_with(
                    py,
                    &no_history_error("P/B"),
                    vec![("book_value_per_share", bvps)],
                );
            }
        }

        let applied_pb = md.clip(&mean_pb, 0.4, 8.0)?;
        let intrinsic_value = bvps.mul(&applied_pb)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Mean P/B produced non-positive value");
        }
        let params = d(
            py,
            with_head(
                vec![
                    ("book_value_per_share", bvps),
                    ("mean_pb", mean_pb),
                    ("applied_pb", applied_pb),
                    ("pb_source", pb_source),
                    ("sector", md.get(ti, "sector")?),
                ],
                multiple_stats(py, history)?,
                vec![],
            ),
        )?;
        let text = "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.";
        let caveat = "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Mean P/B Ratio")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

/// Sales per share for P/S and PSG: the argument, else revenue / shares.
fn sales_per_share<'py>(
    md: &Md<'py>,
    ti: &Bound<'py, PyAny>,
    fin: &Bound<'py, PyAny>,
    shares: &Bound<'py, PyAny>,
    sps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !sps.is_none() {
        return Ok(sps.clone());
    }
    let mut revenue = md.get(ti, "totalRevenue")?;
    if revenue.is_none() && md.present(fin)? {
        revenue = md.latest(fin, "Total Revenue")?;
    }
    if revenue.is_truthy()? && shares.is_truthy()? && shares.gt(0)? {
        return revenue.div(shares);
    }
    Ok(sps.clone())
}

#[pyfunction]
pub fn fr_iv_mean_ps<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    sales_per_share_arg: &Bound<'py, PyAny>,
    mean_ps: &Bound<'py, PyAny>,
    history: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs) = (ticker_info, financials_df, balance_sheet_df);
    guarded(py, "Mean P/S calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        let sps = sales_per_share(md, ti, fin, &shares, sales_per_share_arg)?;
        if sps.is_none() || sps.le(0)? {
            return error(py, "Negative or missing Revenue / Sales per share");
        }
        let mut mean_ps = mean_ps.clone();
        let mut ps_source = s(py, "caller-supplied");
        if mean_ps.is_none() {
            let Some((value, source)) = traded_multiple(md, history, "P/S", ".2f")? else {
                return error_with(py, &no_history_error("P/S"), vec![("sales_per_share", sps)]);
            };
            mean_ps = value;
            ps_source = source;
        }
        let applied_ps = md.clip(&mean_ps, 0.4, 15.0)?;
        let intrinsic_value = sps.mul(&applied_ps)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "Mean P/S produced non-positive value");
        }
        let params = d(
            py,
            with_head(
                vec![
                    ("sales_per_share", sps),
                    ("mean_ps", mean_ps),
                    ("applied_ps", applied_ps),
                    ("ps_source", ps_source),
                ],
                multiple_stats(py, history)?,
                vec![],
            ),
        )?;
        let text = "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.";
        let caveat = "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Mean P/S Ratio")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}

#[pyfunction]
pub fn fr_iv_psg<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    sales_per_share_arg: &Bound<'py, PyAny>,
    revenue_growth_pct: &Bound<'py, PyAny>,
    target_psg: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (ti, fin, bs) = (ticker_info, financials_df, balance_sheet_df);
    guarded(py, "PSG calculation failed", |md| {
        let shares = md.shares(ti, bs)?;
        let sps = sales_per_share(md, ti, fin, &shares, sales_per_share_arg)?;
        if sps.is_none() || sps.le(0)? {
            return error(py, "Negative or missing Revenue / Sales per share");
        }

        let mut revenue_growth_pct = revenue_growth_pct.clone();
        let mut growth_method = s(py, "caller-supplied growth");
        if revenue_growth_pct.is_none() {
            let growth_res = md.blend(fin, ti, "Total Revenue")?;
            revenue_growth_pct = growth_res.get_item("growth")?.mul(100.0)?;
            growth_method = growth_res.get_item("method")?;
        }

        let mut gross_margin = md.get(ti, "grossMargins")?;
        if gross_margin.is_none() && md.present(fin)? {
            let rev = md.latest(fin, "Total Revenue")?;
            let gp = md.latest(fin, "Gross Profit")?;
            if rev.is_truthy()? && gp.is_truthy()? && rev.gt(0)? {
                gross_margin = gp.div(&rev)?;
            }
        }
        let gross_margin = md.clip(&or_if_none(&gross_margin, &fl(py, 0.50)), 0.20, 0.95)?;

        let applied_growth = md.clip(&revenue_growth_pct, 5.0, 45.0)?;
        let target_psg = md.clip(target_psg, 0.5, 2.0)?;

        // Fair P/S multiplier scales with revenue growth weighted by gross margin
        let raw = applied_growth
            .div(10.0)?
            .mul(&gross_margin)?
            .mul(&target_psg)?
            .mul(1.5)?;
        let fair_ps_multiplier = md.clip(&raw, 0.5, 12.0)?;
        let intrinsic_value = sps.mul(&fair_ps_multiplier)?;
        if md.not_finite_positive(&intrinsic_value)? {
            return error(py, "PSG calculation produced non-positive value");
        }
        let params = d(
            py,
            vec![
                ("sales_per_share", sps),
                ("revenue_growth_pct", revenue_growth_pct),
                ("applied_growth_pct", applied_growth),
                ("gross_margin_pct", gross_margin.mul(100.0)?),
                ("target_psg", target_psg),
                ("fair_ps_multiplier", fair_ps_multiplier),
                ("growth_method", growth_method),
            ],
        )?;
        let text = "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.";
        let caveat = "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.";
        d(
            py,
            with_head(
                vec![
                    ("intrinsic_value", intrinsic_value),
                    ("model", s(py, "Price-to-Sales Growth (PSG)")),
                ],
                describe(py, text, caveat),
                vec![("parameters", params)],
            ),
        )
    })
}
