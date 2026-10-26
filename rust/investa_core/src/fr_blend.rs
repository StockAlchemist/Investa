//! The top of `financial_ratios`: the best-fit method recommendation, the
//! per-company Monte Carlo seed and its context manager, the blend's
//! membership, reliability and dividend gate, the blend itself, the
//! comprehensive valuation that runs all twelve models, and the per-symbol
//! entry point that fetches what it needs.
//!
//! Ported line for line from `src/financial_ratios.py` (frozen in
//! `tests/reference/financial_ratios_p15.py`). Figures are combined with
//! Python's own operations; NumPy (`np.isfinite`, `np.clip`, the global
//! generator), `hashlib` and the blend kernel are called as the original
//! called them, and every sibling is looked up on `financial_ratios` at call
//! time. The fetches in `get_intrinsic_value_for_symbol` are the provider's
//! own methods.

use pyo3::exceptions::{PyAttributeError, PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};

use crate::fr::{builtin, fl, fr, int, none, or_else, pybool};
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

fn s<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyAny> {
    PyString::new(py, text).into_any()
}

fn d<'py>(py: Python<'py>, pairs: Vec<(&str, Bound<'py, PyAny>)>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in pairs {
        out.set_item(k, v)?;
    }
    Ok(out)
}

fn get<'py>(x: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    x.call_method1("get", (k,))
}

/// `x.get(k, {})`
fn get_dict<'py>(x: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    x.call_method1("get", (k, PyDict::new(x.py())))
}

/// `(ti.get(k) or "").lower()`
fn lowered<'py>(ti: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    or_else(get(ti, k)?, || Ok(s(ti.py(), "")))?.call_method0("lower")
}

/// `ti.get(k) or 0.0`
fn or_zero<'py>(ti: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    or_else(get(ti, k)?, || Ok(fl(ti.py(), 0.0)))
}

/// One recommendation: its fixed texts and the model's value.
fn method<'py>(
    py: Python<'py>,
    key: &str,
    name: Bound<'py, PyAny>,
    when: &str,
    limitation: &str,
    rationale: &str,
    iv: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(d(
        py,
        vec![
            ("method_key", s(py, key)),
            ("name", name),
            ("when_to_use", s(py, when)),
            ("best_suited_for", s(py, when)),
            ("key_limitation", s(py, limitation)),
            ("key_caveats", s(py, limitation)),
            ("rationale", s(py, rationale)),
            ("intrinsic_value", iv),
        ],
    )?
    .into_any())
}

/// `models.get(k, {}).get("intrinsic_value") is not None`
fn has_value(models: &Bound<'_, PyAny>, k: &str) -> PyResult<bool> {
    Ok(!get(&get_dict(models, k)?, "intrinsic_value")?.is_none())
}

/// `models[k]["intrinsic_value"]`
fn value_of<'py>(models: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    models.get_item(k)?.get_item("intrinsic_value")
}

const PB_WHEN: &str = "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.";
const PB_LIMIT: &str = "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.";
const DNI_WHEN: &str = "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.";
const DNI_LIMIT: &str = "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.";
const DDM_WHEN: &str = "Mature dividend payers and utilities with long track records of consistent dividend growth and sustainable payout ratios (<100%).";
const DDM_LIMIT: &str = "Only reflects value returned as direct dividends; entirely unsuited for non-dividend payers and ignores share repurchases or cash retained.";
const PSG_WHEN: &str = "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.";
const PSG_LIMIT: &str = "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.";
const PS_WHEN: &str = "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.";
const PS_LIMIT: &str = "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.";
const DCF_WHEN: &str = "Cash-generative companies with steady, predictable Free Cash Flow (Operating Cash Flow minus Capital Expenditures).";
const DCF_LIMIT: &str = "Highly sensitive to growth and discount rate (WACC) inputs; unsuitable for cyclical, negative-FCF, or lumpy CapEx businesses.";
const DCFO_WHEN: &str = "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).";
const DCFO_LIMIT: &str = "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.";
const PEG_WHEN: &str = "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.";
const PEG_LIMIT: &str = "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.";
const PE_WHEN: &str = "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.";
const PE_LIMIT: &str = "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.";

#[pyfunction]
#[pyo3(signature = (ticker_info, models, financials_df=None, cashflow_df=None, is_bank_or_financial=None))]
pub fn fr_recommend_best_valuation_method<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    models: &Bound<'py, PyAny>,
    financials_df: Option<&Bound<'py, PyAny>>,
    cashflow_df: Option<&Bound<'py, PyAny>>,
    is_bank_or_financial: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = (financials_df, cashflow_df);
    let ti = ticker_info;
    let sector = lowered(ti, "sector")?;
    let industry = lowered(ti, "industry")?;

    let mut is_financial = match is_bank_or_financial {
        Some(v) => v.is_truthy()?,
        None => false,
    };
    if sector.contains("bank")?
        || sector.contains("financial")?
        || industry.contains("bank")?
        || industry.contains("insurance")?
        || sector.contains("real estate")?
    {
        is_financial = true;
    }

    let eps = or_zero(ti, "trailingEps")?;
    let fcf = or_zero(ti, "freeCashflow")?;
    let cfo = or_zero(ti, "operatingCashflow")?;
    let rev_growth = or_zero(ti, "revenueGrowth")?;
    let div_yield = or_zero(ti, "dividendYield")?;

    // 1. Banks, REITs, Financials
    if is_financial {
        if (sector.contains("bank")? || sector.contains("real estate")?)
            && has_value(models, "mean_pb")?
        {
            return method(
                py,
                "mean_pb",
                s(py, "Mean P/B Ratio"),
                PB_WHEN,
                PB_LIMIT,
                "Financial institutions and property assets are primarily asset-backed and valued relative to book value and marked-to-market equity.",
                value_of(models, "mean_pb")?,
            );
        }
        if has_value(models, "dni")? {
            return method(
                py,
                "dni",
                s(py, "Discounted Net Income"),
                DNI_WHEN,
                DNI_LIMIT,
                "Financial firms do not have traditional industrial CapEx; discounting Net Income at Cost of Equity is the canonical approach.",
                value_of(models, "dni")?,
            );
        }
        if has_value(models, "ddm")? && div_yield.gt(0.02)? {
            return method(
                py,
                "ddm",
                s(py, "Dividend Discount Model"),
                DDM_WHEN,
                DDM_LIMIT,
                "High dividend payout yields direct cash return to shareholders.",
                value_of(models, "ddm")?,
            );
        }
    }

    // 2. Speculative / Unprofitable High Growth
    if eps.le(0)? && rev_growth.ge(0.12)? {
        if has_value(models, "psg")? {
            return method(
                py,
                "psg",
                s(py, "Price-to-Sales Growth (PSG)"),
                PSG_WHEN,
                PSG_LIMIT,
                "Unprofitable hyper-growth companies require growth-relative revenue multiples while unit economics scale.",
                value_of(models, "psg")?,
            );
        }
        if has_value(models, "mean_ps")? {
            return method(
                py,
                "mean_ps",
                s(py, "Mean P/S Ratio"),
                PS_WHEN,
                PS_LIMIT,
                "Historical sales multiple baseline for companies without consistent earnings.",
                value_of(models, "mean_ps")?,
            );
        }
    }

    // 3. Predictable Free Cash Flow Generator -> DCF (Primary Method)
    if has_value(models, "dcf")? && fcf.gt(0)? {
        return method(
            py,
            "dcf",
            s(py, "Discounted Free Cash Flow (Primary)"),
            DCF_WHEN,
            DCF_LIMIT,
            "Strong, predictable Free Cash Flow makes DCF the gold standard valuation model.",
            value_of(models, "dcf")?,
        );
    }

    // 4. Steady Operating Cash Flow but Lumpy CapEx -> D-CFO
    if has_value(models, "dcfo")? && cfo.gt(0)? {
        return method(
            py,
            "dcfo",
            s(py, "Discounted Cash from Operations"),
            DCFO_WHEN,
            DCFO_LIMIT,
            "Operating cash flow is durable while capital expenditure fluctuates through reinvestment cycles.",
            value_of(models, "dcfo")?,
        );
    }

    // 5. Steady Earnings -> PEG or Mean P/E
    if has_value(models, "peg")? && eps.gt(0)? {
        return method(
            py,
            "peg",
            s(py, "PEG Ratio Fair Value"),
            PEG_WHEN,
            PEG_LIMIT,
            "Growth-adjusted earnings multiple benchmark matching earnings expansion with valuation.",
            value_of(models, "peg")?,
        );
    }
    if has_value(models, "mean_pe")? && eps.gt(0)? {
        return method(
            py,
            "mean_pe",
            s(py, "Mean P/E Ratio"),
            PE_WHEN,
            PE_LIMIT,
            "Mature, profitable company with established earnings multiple baseline.",
            value_of(models, "mean_pe")?,
        );
    }

    // Fallback to Graham or whatever model succeeded
    for k in ["graham", "epv", "lynch", "mean_pb", "mean_ps"] {
        if has_value(models, k)? {
            let name = models
                .get_item(k)?
                .call_method1("get", ("model", k.to_uppercase()))?;
            return method(
                py,
                k,
                name,
                "Specialized fallback valuation model applicable to available reporting.",
                "May not account for all future growth phases or capital structure nuances.",
                "Alternative baseline model applicable to available reporting.",
                value_of(models, k)?,
            );
        }
    }

    method(
        py,
        "none",
        s(py, "No Defensible Model"),
        "N/A",
        "N/A",
        "Fundamental data is insufficient to compute a reliable intrinsic value.",
        none(py),
    )
}

#[pyfunction]
pub fn fr_stable_mc_seed<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ti = ticker_info;
    let key = or_else(get(ti, "symbol")?, || {
        or_else(get(ti, "shortName")?, || {
            or_else(get(ti, "longName")?, || Ok(s(py, "investa")))
        })
    })?
    .str()?;
    let digest = fr(py)?
        .getattr("hashlib")?
        .call_method(
            "blake2b",
            (key.call_method0("encode")?,),
            Some(&kwargs(py, &[("digest_size", &int(py, 4)?)])?),
        )?
        .call_method0("digest")?;
    builtin(py, "int")?.call_method1("from_bytes", (digest, "big"))
}

/// `_deterministic_draws(ticker_info)`: seed the global RNG for one
/// company's simulations, then put it back.
#[pyclass(module = "investa_core")]
pub struct DeterministicDraws {
    ticker_info: Py<PyAny>,
    state: Option<Py<PyAny>>,
}

#[pymethods]
impl DeterministicDraws {
    #[new]
    fn new(ticker_info: Py<PyAny>) -> Self {
        DeterministicDraws {
            ticker_info,
            state: None,
        }
    }

    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = fr(py)?;
        let random = m.getattr("np")?.getattr("random")?;
        let state = random.call_method0("get_state")?;
        let seed = m
            .getattr("_stable_mc_seed")?
            .call1((self.ticker_info.bind(py),))?;
        random.call_method1("seed", (seed,))?;
        self.state = Some(state.unbind());
        Ok(py.None())
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(state) = self.state.take() {
            fr(py)?
                .getattr("np")?
                .getattr("random")?
                .call_method1("set_state", (state,))?;
        }
        Ok(false)
    }
}

#[pyfunction]
pub fn fr_blend_profile<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let sector = lowered(ticker_info, "sector")?;
    let industry = lowered(ticker_info, "industry")?;
    if industry.contains("reit")? || sector.contains("reit")? {
        return Ok(s(py, "reit"));
    }
    let o = Ops::new(py)?;
    let both = s(py, &format!("{} {}", o.fmt(&sector)?, o.fmt(&industry)?));
    for k in ["financial", "bank", "insurance"] {
        if both.contains(k)? {
            return Ok(s(py, "financial"));
        }
    }
    Ok(s(py, "operating"))
}

#[pyfunction]
pub fn fr_model_reliability<'py>(
    py: Python<'py>,
    model_res: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mc = or_else(
        get(
            &or_else(model_res.clone(), || Ok(PyDict::new(py).into_any()))?,
            "mc",
        )?,
        || Ok(PyDict::new(py).into_any()),
    )?;
    let (bear, base, bull) = (get(&mc, "bear")?, get(&mc, "base")?, get(&mc, "bull")?);
    if bear.is_none() || base.is_none() || bull.is_none() {
        return Ok(fl(py, 1.0));
    }
    let float = builtin(py, "float")?;
    let floats = (|| -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        Ok((
            float.call1((&bear,))?,
            float.call1((&base,))?,
            float.call1((&bull,))?,
        ))
    })();
    let (bear, base, bull) = match floats {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            return Ok(fl(py, 1.0));
        }
        Err(e) => return Err(e),
    };
    let m = fr(py)?;
    m.getattr("investa_core")?
        .getattr("model_reliability")?
        .call1((
            bear,
            base,
            bull,
            m.getattr("RELIABILITY_CV_REF")?,
            m.getattr("RELIABILITY_FLOOR")?,
        ))
}

#[pyfunction]
pub fn fr_ddm_describes_the_company(
    py: Python<'_>,
    ticker_info: &Bound<'_, PyAny>,
    profile: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if profile.eq("reit")? {
        return Ok(true);
    }
    let payout = get(ticker_info, "payoutRatio")?;
    let payout = match builtin(py, "float")?.call1((payout,)) {
        Ok(p) => p,
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            return Ok(false);
        }
        Err(e) => return Err(e),
    };
    let m = fr(py)?;
    let finite = m.getattr("np")?.call_method1("isfinite", (&payout,))?;
    if !finite.is_truthy()? {
        return Ok(false);
    }
    payout.ge(m.getattr("DDM_MIN_PAYOUT_FOR_BLEND")?)
}

#[pyfunction]
pub fn fr_blend_intrinsic_values<'py>(
    py: Python<'py>,
    models: &Bound<'py, PyAny>,
    ticker_info: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let np = m.getattr("np")?;
    let o = Ops::new(py)?;
    let float = builtin(py, "float")?;
    let profile = m.getattr("_blend_profile")?.call1((ticker_info,))?;
    let table = d(
        py,
        vec![
            ("financial", m.getattr("MODEL_BLEND_PRIORS_FINANCIAL")?),
            ("reit", m.getattr("MODEL_BLEND_PRIORS_REIT")?),
        ],
    )?;
    let priors = table.as_any().call_method1(
        "get",
        (&profile, m.getattr("MODEL_BLEND_PRIORS_OPERATING")?),
    )?;

    let contributions = PyList::empty(py);
    let excluded = PyDict::new(py);
    let ddm_describes = m.getattr("_ddm_describes_the_company")?;
    let reliability_of = m.getattr("_model_reliability")?;

    for item in priors.call_method0("items")?.try_iter()? {
        let kp = unpack(&item?, 2)?;
        let (key, prior) = (&kp[0], &kp[1]);
        let model_res = or_else(models.call_method1("get", (key,))?, || {
            Ok(PyDict::new(py).into_any())
        })?;
        let iv = get(&model_res, "intrinsic_value")?;
        if iv.is_none() || !np.call_method1("isfinite", (&iv,))?.is_truthy()? || iv.le(0)? {
            continue;
        }
        if key.eq("ddm")? && !ddm_describes.call1((ticker_info, &profile))?.is_truthy()? {
            excluded.set_item(
                key,
                format!(
                    "Payout ratio below {} — the dividend is not the whole of distributable cash, so DDM values the dividend rather than the company",
                    o.fmt_spec(&m.getattr("DDM_MIN_PAYOUT_FOR_BLEND")?, ".0%")?
                ),
            )?;
            continue;
        }
        let reliability = reliability_of.call1((&model_res,))?;
        let weight = float.call1((prior,))?.mul(&reliability)?;
        contributions.append(d(
            py,
            vec![
                ("key", key.clone()),
                ("value", float.call1((&iv,))?),
                ("prior", float.call1((prior,))?),
                ("reliability", reliability),
                ("weight", weight),
            ],
        )?)?;
    }

    if !contributions.is_truthy()? {
        return d(
            py,
            vec![
                ("value", none(py)),
                ("profile", profile),
                ("weights", PyDict::new(py).into_any()),
                ("contributions", PyList::empty(py).into_any()),
                ("spread_pct", none(py)),
                ("confidence", fl(py, 0.0)),
                ("excluded", excluded.into_any()),
            ],
        );
    }

    let column = |k: &str| -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for c in contributions.iter() {
            out.append(c.get_item(k)?)?;
        }
        Ok(out)
    };
    let r = m.getattr("investa_core")?.getattr("blend")?.call1((
        column("value")?,
        column("prior")?,
        column("reliability")?,
    ))?;
    let r = unpack(&r, 4)?;
    let weights = PyDict::new(py);
    for pair in builtin(py, "zip")?
        .call1((&contributions, &r[1]))?
        .try_iter()?
    {
        let cw = unpack(&pair?, 2)?;
        weights.set_item(cw[0].get_item("key")?, &cw[1])?;
    }
    d(
        py,
        vec![
            ("value", float.call1((&r[0],))?),
            ("profile", profile),
            ("weights", weights.into_any()),
            ("contributions", contributions.into_any()),
            ("spread_pct", r[2].clone()),
            ("confidence", r[3].clone()),
            ("excluded", excluded.into_any()),
        ],
    )
}

/// `call(f, *args, **kw)` on a module sibling.
fn call_kw<'py>(
    m: &Bound<'py, PyModule>,
    name: &str,
    args: Vec<Bound<'py, PyAny>>,
    kw: &[(&str, &Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyAny>> {
    let py = m.py();
    m.getattr(name)?
        .call(PyTuple::new(py, args)?, Some(&kwargs(py, kw)?))
}

/// `"intrinsic_value" in res and "parameters" in res`
fn valued(res: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(res.contains("intrinsic_value")? && res.contains("parameters")?)
}

/// `params.get(a, params.get(b, default))`
fn get2<'py>(
    params: &Bound<'py, PyAny>,
    a: &str,
    b: &str,
    default: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inner = params.call_method1("get", (b, default))?;
    params.call_method1("get", (a, inner))
}

/// `((value - price) / price) * 100`, None when NaN or infinite.
fn margin_of_safety<'py>(
    np: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    price: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = np.py();
    let mos = value.sub(price)?.div(price)?.mul(100)?;
    if !mos.is_none()
        && (np.call_method1("isnan", (&mos,))?.is_truthy()?
            || np.call_method1("isinf", (&mos,))?.is_truthy()?)
    {
        return Ok(none(py));
    }
    Ok(mos)
}

/// `x is not None and np.isfinite(x)` (and `x > 0` when asked).
fn usable(np: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, positive: bool) -> PyResult<bool> {
    Ok(
        !x.is_none()
            && np.call_method1("isfinite", (x,))?.is_truthy()?
            && (!positive || x.gt(0)?),
    )
}

/// The twelve models' Monte Carlo runs, inside the seeded block.
#[allow(clippy::too_many_arguments)]
fn run_simulations<'py>(
    m: &Bound<'py, PyModule>,
    ti: &Bound<'py, PyAny>,
    res: &[Bound<'py, PyAny>; 12],
    iterations: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = m.py();
    let it = [("iterations", iterations)];
    let [
        dcf,
        dcfo,
        dni,
        mean_pe,
        peg,
        mean_pb,
        mean_ps,
        psg,
        graham,
        ddm,
        epv,
        lynch,
    ] = res;
    let p = |r: &Bound<'py, PyAny>| r.get_item("parameters");
    let k = |params: &Bound<'py, PyAny>, key: &str| params.get_item(key);

    if valued(dcf)? {
        let params = p(dcf)?;
        let growth = get2(&params, "applied_growth", "growth_rate", fl(py, 0.05))?;
        let mc = call_kw(
            m,
            "run_monte_carlo_dcf",
            vec![
                ti.clone(),
                k(&params, "base_fcf")?,
                growth,
                k(&params, "discount_rate")?,
                k(&params, "projection_years")?,
                k(&params, "terminal_growth_rate")?,
            ],
            &it,
        )?; // fmt: skip
        dcf.set_item("mc", mc)?;
    }
    if valued(dcfo)? {
        let params = p(dcfo)?;
        let growth = get2(&params, "applied_growth", "growth_rate", fl(py, 0.05))?;
        let mc = call_kw(
            m,
            "run_monte_carlo_dcfo",
            vec![
                ti.clone(),
                k(&params, "base_cfo")?,
                growth,
                k(&params, "discount_rate")?,
                k(&params, "projection_years")?,
                k(&params, "terminal_growth_rate")?,
            ],
            &it,
        )?; // fmt: skip
        dcfo.set_item("mc", mc)?;
    }
    if valued(dni)? {
        let params = p(dni)?;
        let growth = get2(&params, "applied_growth", "growth_rate", fl(py, 0.05))?;
        let mc = call_kw(
            m,
            "run_monte_carlo_dni",
            vec![
                k(&params, "base_eps")?,
                growth,
                k(&params, "discount_rate")?,
                k(&params, "projection_years")?,
                k(&params, "terminal_growth_rate")?,
            ],
            &it,
        )?; // fmt: skip
        dni.set_item("mc", mc)?;
    }
    if valued(mean_pe)? {
        let params = p(mean_pe)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_mean_pe",
            vec![k(&params, "eps")?, k(&params, "applied_pe")?],
            &it,
        )?;
        mean_pe.set_item("mc", mc)?;
    }
    if valued(peg)? {
        let params = p(peg)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_peg",
            vec![
                k(&params, "eps")?,
                get2(
                    &params,
                    "applied_growth_pct",
                    "growth_rate_pct",
                    fl(py, 10.0),
                )?,
                params.call_method1("get", ("dividend_yield_pct", 0.0))?,
                params.call_method1("get", ("target_peg", 1.0))?,
            ],
            &it,
        )?;
        peg.set_item("mc", mc)?;
    }
    if valued(mean_pb)? {
        let params = p(mean_pb)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_mean_pb",
            vec![
                k(&params, "book_value_per_share")?,
                k(&params, "applied_pb")?,
            ],
            &it,
        )?;
        mean_pb.set_item("mc", mc)?;
    }
    if valued(mean_ps)? {
        let params = p(mean_ps)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_mean_ps",
            vec![k(&params, "sales_per_share")?, k(&params, "applied_ps")?],
            &it,
        )?;
        mean_ps.set_item("mc", mc)?;
    }
    if valued(psg)? {
        let params = p(psg)?;
        let gm = if params.contains("gross_margin_pct")? {
            params
                .call_method1("get", ("gross_margin_pct", 50.0))?
                .div(100.0)?
        } else {
            fl(py, 0.50)
        };
        let mc = call_kw(
            m,
            "run_monte_carlo_psg",
            vec![
                k(&params, "sales_per_share")?,
                get2(
                    &params,
                    "applied_growth_pct",
                    "revenue_growth_pct",
                    fl(py, 15.0),
                )?,
                gm,
                params.call_method1("get", ("target_psg", 1.0))?,
            ],
            &it,
        )?;
        psg.set_item("mc", mc)?;
    }
    if valued(graham)? {
        let params = p(graham)?;
        if params.contains("eps")? {
            let growth = get2(
                &params,
                "applied_growth_pct",
                "growth_rate_pct",
                int(py, 0)?,
            )?;
            let mc = call_kw(
                m,
                "run_monte_carlo_graham",
                vec![k(&params, "eps")?, growth, k(&params, "bond_yield_proxy")?],
                &it,
            )?;
            graham.set_item("mc", mc)?;
        }
    }
    if valued(ddm)? {
        let params = p(ddm)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_ddm",
            vec![
                k(&params, "base_dividend")?,
                k(&params, "applied_growth")?,
                k(&params, "discount_rate")?,
                k(&params, "projection_years")?,
                k(&params, "terminal_growth_rate")?,
            ],
            &it,
        )?; // fmt: skip
        ddm.set_item("mc", mc)?;
    }
    if valued(epv)? {
        let params = p(epv)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_epv",
            vec![
                ti.clone(),
                k(&params, "normalized_ebit")?,
                k(&params, "tax_rate")?,
                k(&params, "discount_rate")?,
            ],
            &it,
        )?; // fmt: skip
        epv.set_item("mc", mc)?;
    }
    if valued(lynch)? {
        let params = p(lynch)?;
        let mc = call_kw(
            m,
            "run_monte_carlo_lynch",
            vec![
                k(&params, "eps")?,
                params.call_method1("get", ("growth_rate_pct", 10.0))?,
                params.call_method1("get", ("dividend_yield_pct", 0.0))?,
            ],
            &it,
        )?;
        lynch.set_item("mc", mc)?;
    }
    Ok(())
}

const MODEL_KEYS: [&str; 12] = [
    "dcf", "dcfo", "dni", "mean_pe", "peg", "mean_pb", "mean_ps", "psg", "graham", "ddm", "epv",
    "lynch",
];

/// The ETF/fund result: NAV in place of the models.
fn nav_result<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    quote_type: &Bound<'py, PyAny>,
    current_price: &Bound<'py, PyAny>,
    nav_price: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let o = Ops::new(py)?;
    let recommended = method(
        py,
        "nav",
        s(py, "Net Asset Value (NAV)"),
        "ETFs, CEFs, and Mutual Funds holding diversified portfolios of publicly-traded securities.",
        "Fund NAV equals total underlying portfolio assets minus fund liabilities, reflecting portfolio holdings rather than operating business cash flows.",
        "Funds are valued by their reported Net Asset Value per share.",
        nav_price.clone(),
    )?;
    let models = PyDict::new(py);
    for k in MODEL_KEYS {
        models.set_item(
            k,
            d(
                py,
                vec![
                    ("model", s(py, "N/A (ETF/Fund)")),
                    ("intrinsic_value", none(py)),
                ],
            )?,
        )?;
    }
    let results = d(
        py,
        vec![
            ("current_price", current_price.clone()),
            ("average_intrinsic_value", nav_price.clone()),
            (
                "valuation_note",
                s(
                    py,
                    &format!(
                        "Valuation based on Net Asset Value (NAV) for {}.",
                        o.fmt(quote_type)?
                    ),
                ),
            ),
            ("valuation_status", s(py, "nav")),
            ("valuation_confidence", fl(py, 1.0)),
            ("recommended_method", recommended),
            ("models", models.into_any()),
        ],
    )?;
    if current_price.is_truthy()? {
        results.set_item(
            "margin_of_safety_pct",
            margin_of_safety(np, nav_price, current_price)?,
        )?;
    }
    Ok(results.into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_comprehensive_intrinsic_value<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: &Bound<'py, PyAny>,
    overrides: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
    prices_df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fr(py)?;
    let np = m.getattr("np")?;
    let o = Ops::new(py)?;
    let (fin, bs, cf) = (financials_df, balance_sheet_df, cashflow_df);

    // 0. Enrich data with statement fallbacks to maximize coverage
    let ti = m
        .getattr("_enrich_ticker_info")?
        .call1((ticker_info, fin, bs, cf))?;
    let ov = or_else(overrides.clone(), || Ok(PyDict::new(py).into_any()))?;
    let traded = m
        .getattr("historical_mean_multiples")?
        .call1((fin, bs, cf, prices_df))?;

    let og = |k: &str| ov.call_method1("get", (k,));
    let og_or = |k: &str, default: &Bound<'py, PyAny>| ov.call_method1("get", (k, default));
    let int_t = builtin(py, "int")?;

    // DCF overrides
    let dcf_discount = og("dcf_discount_rate")?;
    let dcf_growth = og("dcf_growth_rate")?;
    let dcf_terminal = og_or("dcf_terminal_growth", &fl(py, 0.02))?;
    let dcf_projection = int_t.call1((og_or("dcf_projection_years", &int(py, 10)?)?,))?;
    let dcf_fcf = og("dcf_fcf")?;
    let target_fcf_margin = og("target_fcf_margin")?;
    // Graham overrides
    let graham_growth = og("graham_growth_rate")?;
    let graham_eps = og("graham_eps")?;
    let graham_bond_yield = og("graham_bond_yield")?;
    // DDM overrides
    let ddm_discount = og("ddm_discount_rate")?;
    let ddm_growth = og("ddm_growth_rate")?;
    let ddm_terminal = og_or("ddm_terminal_growth", &fl(py, 0.02))?;
    let ddm_projection = int_t.call1((og_or("ddm_projection_years", &int(py, 10)?)?,))?;
    let ddm_dividend = og("ddm_base_dividend")?;

    let dcf_res = call_kw(
        &m,
        "calculate_intrinsic_value_dcf",
        vec![ti.clone(), fin.clone(), bs.clone(), cf.clone()],
        &[
            ("discount_rate", &dcf_discount),
            ("growth_rate", &dcf_growth),
            ("projection_years", &dcf_projection),
            ("terminal_growth_rate", &dcf_terminal),
            ("fcf", &dcf_fcf),
            ("target_fcf_margin", &target_fcf_margin),
        ],
    )?;
    let dcfo_res = call_kw(
        &m,
        "calculate_intrinsic_value_dcfo",
        vec![ti.clone(), fin.clone(), bs.clone(), cf.clone()],
        &[
            (
                "discount_rate",
                &og_or("dcfo_discount_rate", &dcf_discount)?,
            ),
            ("growth_rate", &og_or("dcfo_growth_rate", &dcf_growth)?),
            ("projection_years", &dcf_projection),
            ("terminal_growth_rate", &dcf_terminal),
            ("base_cfo", &og("dcfo_base_cfo")?),
        ],
    )?;
    let dni_res = call_kw(
        &m,
        "calculate_intrinsic_value_dni",
        vec![ti.clone(), fin.clone(), bs.clone()],
        &[
            ("discount_rate", &og_or("dni_discount_rate", &ddm_discount)?),
            ("growth_rate", &og_or("dni_growth_rate", &graham_growth)?),
            ("projection_years", &dcf_projection),
            ("terminal_growth_rate", &dcf_terminal),
            ("base_eps", &og_or("dni_base_eps", &graham_eps)?),
        ],
    )?;
    let mean_pe_res = call_kw(
        &m,
        "calculate_intrinsic_value_mean_pe",
        vec![ti.clone(), fin.clone()],
        &[
            ("eps", &og_or("mean_pe_eps", &graham_eps)?),
            ("mean_pe", &og("mean_pe_multiple")?),
            ("history", &get(&traded, "pe")?),
        ],
    )?;
    let peg_res = call_kw(
        &m,
        "calculate_intrinsic_value_peg",
        vec![ti.clone(), fin.clone()],
        &[
            ("growth_rate", &og_or("peg_growth_rate", &graham_growth)?),
            ("eps", &og_or("peg_eps", &graham_eps)?),
            ("target_peg", &og_or("peg_target", &fl(py, 1.0))?),
        ],
    )?;
    let mean_pb_res = call_kw(
        &m,
        "calculate_intrinsic_value_mean_pb",
        vec![ti.clone(), bs.clone()],
        &[
            ("book_value_per_share", &og("mean_pb_bvps")?),
            ("mean_pb", &og("mean_pb_multiple")?),
            ("history", &get(&traded, "pb")?),
        ],
    )?;
    let mean_ps_res = call_kw(
        &m,
        "calculate_intrinsic_value_mean_ps",
        vec![ti.clone(), fin.clone(), bs.clone()],
        &[
            ("sales_per_share", &og("mean_ps_sps")?),
            ("mean_ps", &og("mean_ps_multiple")?),
            ("history", &get(&traded, "ps")?),
        ],
    )?;
    let psg_res = call_kw(
        &m,
        "calculate_intrinsic_value_psg",
        vec![ti.clone(), fin.clone(), bs.clone()],
        &[
            ("sales_per_share", &og("psg_sps")?),
            ("revenue_growth_pct", &og("psg_revenue_growth")?),
            ("target_psg", &og_or("psg_target", &fl(py, 1.0))?),
        ],
    )?;
    let graham_res = call_kw(
        &m,
        "calculate_intrinsic_value_graham",
        vec![ti.clone(), fin.clone(), bs.clone()],
        &[
            ("growth_rate", &graham_growth),
            ("eps", &graham_eps),
            ("bond_yield", &graham_bond_yield),
        ],
    )?;
    let ddm_res = call_kw(
        &m,
        "calculate_intrinsic_value_ddm",
        vec![ti.clone(), fin.clone(), cf.clone(), bs.clone()],
        &[
            ("discount_rate", &ddm_discount),
            ("growth_rate", &ddm_growth),
            ("projection_years", &ddm_projection),
            ("terminal_growth_rate", &ddm_terminal),
            ("base_dividend", &ddm_dividend),
        ],
    )?;
    let epv_res = call_kw(
        &m,
        "calculate_earnings_power_value",
        vec![ti.clone(), fin.clone(), bs.clone(), cf.clone()],
        &[("discount_rate", &og_or("epv_discount_rate", &dcf_discount)?)],
    )?;
    let lynch_res = call_kw(
        &m,
        "calculate_intrinsic_value_lynch",
        vec![ti.clone(), fin.clone()],
        &[
            ("growth_rate", &og_or("lynch_growth_rate", &graham_growth)?),
            ("eps", &og_or("lynch_eps", &graham_eps)?),
        ],
    )?;
    let all = [
        dcf_res,
        dcfo_res,
        dni_res,
        mean_pe_res,
        peg_res,
        mean_pb_res,
        mean_ps_res,
        psg_res,
        graham_res,
        ddm_res,
        epv_res,
        lynch_res,
    ];

    // Monte Carlo, seeded per company; the global RNG is restored on the way out.
    let draws = m.getattr("_deterministic_draws")?.call1((&ti,))?;
    crate::mdb::with1(py, &draws, |_| run_simulations(&m, &ti, &all, iterations))?;

    let current_price = or_else(get(&ti, "currentPrice")?, || get(&ti, "regularMarketPrice"))?;

    // --- ETF Valuation Logic ---
    let quote_type = ti
        .call_method1("get", ("quoteType", ""))?
        .call_method0("upper")?;
    if quote_type.eq("ETF")? || quote_type.eq("MUTUALFUND")? {
        let mut nav_price = get(&ti, "navPrice")?;
        if (nav_price.is_none() || nav_price.eq(0)?) && current_price.is_truthy()? {
            nav_price = current_price.clone();
        }
        if nav_price.is_truthy()? {
            return nav_result(py, &np, &quote_type, &current_price, &nav_price);
        }
    }

    let models_map = PyDict::new(py);
    for (k, r) in MODEL_KEYS.iter().zip(&all) {
        models_map.set_item(*k, r)?;
    }
    let [
        dcf_res,
        dcfo_res,
        dni_res,
        _,
        _,
        _,
        _,
        _,
        graham_res,
        ddm_res,
        epv_res,
        lynch_res,
    ] = &all;

    // Best-fit method recommendation
    let recommended =
        m.getattr("recommend_best_valuation_method")?
            .call1((&ti, &models_map, fin, cf))?;
    let results = d(
        py,
        vec![
            ("current_price", current_price.clone()),
            ("recommended_method", recommended),
            ("models", models_map.clone().into_any()),
        ],
    )?;

    // --- Eligibility ---------------------------------------------------------
    let market_cap = get(&ti, "marketCap")?;
    let min_price = m.getattr("MIN_PRICE_FOR_VALUATION")?;
    let ineligible = if !current_price.is_none() && current_price.lt(&min_price)? {
        Some(format!(
            "Price below ${} — per-share valuation is dominated by data noise",
            o.fmt_spec(&min_price, ".0f")?
        ))
    } else if !market_cap.is_none() && market_cap.lt(m.getattr("MIN_MARKET_CAP_FOR_VALUATION")?)? {
        Some("Market cap below $50M — fundamentals are too unreliable to value".to_owned())
    } else {
        None
    };
    if let Some(note) = ineligible {
        results.set_item("average_intrinsic_value", none(py))?;
        results.set_item("valuation_status", "ineligible")?;
        results.set_item("valuation_confidence", 0.0)?;
        results.set_item("valuation_note", note)?;
        return Ok(results.into_any());
    }

    // --- Business-aware, uncertainty-weighted blend --------------------------
    let blend = m
        .getattr("blend_intrinsic_values")?
        .call1((&models_map, &ti))?;
    let contributions = blend.get_item("contributions")?;

    // EPV travels alongside the estimate as the no-growth floor, not inside it.
    let float = builtin(py, "float")?;
    let epv_iv = get(epv_res, "intrinsic_value")?;
    if usable(&np, &epv_iv, true)? {
        results.set_item("earnings_power_floor", float.call1((epv_iv,))?)?;
    }
    let lynch_iv = get(lynch_res, "intrinsic_value")?;
    if usable(&np, &lynch_iv, true)? {
        results.set_item("lynch_fair_value", float.call1((lynch_iv,))?)?;
    }
    // A DDM held out by the payout gate is still worth showing.
    let ddm_iv = get(ddm_res, "intrinsic_value")?;
    if blend.get_item("excluded")?.contains("ddm")? && usable(&np, &ddm_iv, false)? {
        results.set_item("dividend_discount_floor", float.call1((ddm_iv,))?)?;
    }

    results.set_item("blend_profile", blend.get_item("profile")?)?;
    if blend.get_item("excluded")?.is_truthy()? {
        results.set_item("blend_exclusions", blend.get_item("excluded")?)?;
    }

    if !contributions.is_truthy()? {
        let by_profile = d(
            py,
            vec![
                (
                    "financial",
                    PyTuple::new(py, [dni_res, ddm_res])?.into_any(),
                ),
                ("reit", PyTuple::new(py, [dcfo_res, ddm_res])?.into_any()),
            ],
        )?;
        let eligible = by_profile.as_any().call_method1(
            "get",
            (
                blend.get_item("profile")?,
                PyTuple::new(py, [dcf_res, graham_res, ddm_res])?,
            ),
        )?;
        let reasons = PyList::empty(py);
        for r in eligible.try_iter()? {
            let r = r?;
            if get(&r, "error")?.is_truthy()? {
                reasons.append(get(&r, "error")?)?;
            }
        }
        for v in blend
            .get_item("excluded")?
            .call_method0("values")?
            .try_iter()?
        {
            reasons.append(v?)?;
        }
        results.set_item("average_intrinsic_value", none(py))?;
        results.set_item("valuation_status", "no_model")?;
        results.set_item("valuation_confidence", 0.0)?;
        let note = if reasons.is_truthy()? {
            let unique = builtin(py, "dict")?.call_method1("fromkeys", (&reasons,))?;
            s(py, "No model could value this company: ")
                .add(pystr(py, "; ").call_method1("join", (unique,))?)?
        } else {
            s(py, "No model could value this company.")
        };
        results.set_item("valuation_note", note)?;
        return Ok(results.into_any());
    }

    let mut avg_intrinsic = blend.get_item("value")?;
    results.set_item("model_weights", blend.get_item("weights")?)?;
    let reliability = PyDict::new(py);
    for c in contributions.try_iter()? {
        let c = c?;
        reliability.set_item(c.get_item("key")?, c.get_item("reliability")?)?;
    }
    results.set_item("model_reliability", reliability)?;
    let spread_pct = blend.get_item("spread_pct")?;
    let mut confidence = blend.get_item("confidence")?;

    // --- Output sanity band --------------------------------------------------
    let mut status = "ok";
    let mut notes: Vec<String> = Vec::new();
    if current_price.is_truthy()? && current_price.gt(0)? {
        let ratio = avg_intrinsic.div(&current_price)?;
        let (max_ratio, min_ratio) = (m.getattr("MAX_IV_TO_PRICE")?, m.getattr("MIN_IV_TO_PRICE")?);
        if ratio.gt(&max_ratio)? || ratio.lt(&min_ratio)? {
            let clamped = float.call1((np.call_method1(
                "clip",
                (
                    &avg_intrinsic,
                    min_ratio.mul(&current_price)?,
                    max_ratio.mul(&current_price)?,
                ),
            )?,))?;
            notes.push(format!(
                "Model output {}x price is outside the credible band; clamped to {}x. Treat as low confidence.",
                o.fmt_spec(&ratio, ".1f")?,
                o.fmt_spec(&clamped.div(&current_price)?, ".1f")?
            ));
            avg_intrinsic = clamped;
            status = "clamped";
        }
    }
    if status == "clamped" {
        // The published number is no longer the one the models produced.
        confidence = py
            .import("operator")?
            .call_method1("imul", (&confidence, 0.5))?;
    }

    if !spread_pct.is_none() && spread_pct.gt(100)? {
        let mut parts = Vec::new();
        for c in contributions.try_iter()? {
            let c = c?;
            parts.push(format!(
                "{}={}",
                o.fmt(&c.get_item("key")?)?,
                o.fmt_spec(&c.get_item("value")?, ".2f")?
            ));
        }
        notes.push(format!(
            "Models disagree by {}% of the blended value ({}).",
            o.fmt_spec(&spread_pct, ".0f")?,
            parts.join(", ")
        ));
    }

    if confidence.lt(m.getattr("LOW_CONFIDENCE_THRESHOLD")?)? && status == "ok" {
        status = "low_confidence";
        if notes.is_empty() {
            notes.push(format!(
                "Low confidence ({}): the contributing models are individually uncertain or few in number.",
                o.fmt_spec(&confidence, ".0%")?
            ));
        }
    }

    if !np
        .call_method1("isfinite", (&avg_intrinsic,))?
        .is_truthy()?
    {
        avg_intrinsic = none(py);
        status = "no_model";
        confidence = fl(py, 0.0);
    }

    results.set_item("average_intrinsic_value", &avg_intrinsic)?;
    results.set_item("valuation_status", status)?;
    results.set_item(
        "valuation_confidence",
        builtin(py, "round")?.call1((float.call1((&confidence,))?, 3))?,
    )?;
    results.set_item("model_spread_pct", &spread_pct)?;
    if !notes.is_empty() {
        results.set_item("valuation_note", notes.join(" "))?;
    }

    // Probabilistic range from the contributing models at their blend weights.
    let (mut weighted_bear, mut weighted_bull, mut band_weight) =
        (fl(py, 0.0), fl(py, 0.0), fl(py, 0.0));
    let weights = blend.get_item("weights")?;
    for c in contributions.try_iter()? {
        let c = c?;
        let key = c.get_item("key")?;
        let mc = or_else(
            get(
                &or_else(models_map.as_any().call_method1("get", (&key,))?, || {
                    Ok(PyDict::new(py).into_any())
                })?,
                "mc",
            )?,
            || Ok(PyDict::new(py).into_any()),
        )?;
        let (bear, bull) = (get(&mc, "bear")?, get(&mc, "bull")?);
        if bear.is_none() || bull.is_none() {
            continue;
        }
        if !(np.call_method1("isfinite", (&bear,))?.is_truthy()?
            && np.call_method1("isfinite", (&bull,))?.is_truthy()?)
        {
            continue;
        }
        let w = weights.call_method1("get", (&key, 0.0))?;
        weighted_bear = weighted_bear.add(bear.mul(&w)?)?;
        weighted_bull = weighted_bull.add(bull.mul(&w)?)?;
        band_weight = band_weight.add(&w)?;
    }
    if band_weight.gt(0)? {
        results.set_item(
            "range",
            d(
                py,
                vec![
                    ("bear", weighted_bear.div(&band_weight)?),
                    ("bull", weighted_bull.div(&band_weight)?),
                ],
            )?,
        )?;
    }

    if current_price.is_truthy()? && avg_intrinsic.is_truthy()? {
        results.set_item(
            "margin_of_safety_pct",
            margin_of_safety(&np, &avg_intrinsic, &current_price)?,
        )?;
    }
    Ok(results.into_any())
}

/// `getattr(obj, name, default)`
fn getattr_or<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    default: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match obj.getattr(name) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyAttributeError>(obj.py()) => Ok(default),
        Err(e) => Err(e),
    }
}

fn error_dict<'py>(py: Python<'py>, msg: String) -> PyResult<Bound<'py, PyAny>> {
    Ok(d(py, vec![("error", s(py, &msg))])?.into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_intrinsic_value_for_symbol<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    mdp: &Bound<'py, PyAny>,
    config_manager: &Bound<'py, PyAny>,
    iterations: &Bound<'py, PyAny>,
    force_refresh: &Bound<'py, PyAny>,
    prefetched_financials: &Bound<'py, PyAny>,
    prefetched_balance_sheet: &Bound<'py, PyAny>,
    prefetched_cashflow: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let map_to_yf_symbol = py.import("finutils")?.getattr("map_to_yf_symbol")?;
    let config = py.import("config")?;
    let o = Ops::new(py)?;
    let set = builtin(py, "set")?;

    // 1. Map symbol
    let (user_symbol_map, user_excluded_symbols) = if config_manager.is_truthy()? {
        let mo = config_manager.getattr("manual_overrides")?;
        (
            mo.call_method1("get", ("user_symbol_map", PyDict::new(py)))?,
            set.call1((mo.call_method1("get", ("user_excluded_symbols", PyList::empty(py)))?,))?,
        )
    } else {
        (
            getattr_or(
                &config,
                "SYMBOL_MAP_TO_YFINANCE",
                PyDict::new(py).into_any(),
            )?,
            set.call1((getattr_or(
                &config,
                "YFINANCE_EXCLUDED_SYMBOLS",
                PyList::empty(py).into_any(),
            )?,))?,
        )
    };

    let yf_symbol = map_to_yf_symbol.call1((symbol, &user_symbol_map, &user_excluded_symbols))?;
    if !yf_symbol.is_truthy()? {
        if user_excluded_symbols.contains(symbol.call_method0("upper")?)? {
            return error_dict(
                py,
                format!("Symbol {} is in the exclusion list.", o.fmt(symbol)?),
            );
        }
        return error_dict(
            py,
            format!("Could not map {} to Yahoo Finance symbol", o.fmt(symbol)?),
        );
    }

    // 2. Fetch COMPLETE data
    let body = || -> PyResult<Bound<'py, PyAny>> {
        let refresh = [("force_refresh", force_refresh)];
        let mut info = mdp.call_method(
            "get_fundamental_data",
            (&yf_symbol,),
            Some(&kwargs(py, &refresh)?),
        )?;

        // Detect "poisoned" or insufficient info and force a fresh fetch.
        let mut is_sparse = !info.is_truthy()? || info.len()? <= 8;
        if info.is_truthy()?
            && info
                .call_method1("get", ("quoteType", ""))?
                .call_method0("upper")?
                .eq("EQUITY")?
            && (!get(&info, "lastFiscalYearEnd")?.is_truthy()?
                || !get(&info, "mostRecentQuarter")?.is_truthy()?)
        {
            is_sparse = true;
        }
        if is_sparse {
            log(
                py,
                "warning",
                format!(
                    "Detected sparse/poisoned fundamental info for {} in get_intrinsic_value_for_symbol. Forcing refresh.",
                    o.fmt(&yf_symbol)?
                ),
            )?;
            info = mdp.call_method(
                "get_fundamental_data",
                (&yf_symbol,),
                Some(&kwargs(py, &[("force_refresh", &pybool(py, true))])?),
            )?;
        }
        if !info.is_truthy()? {
            return error_dict(
                py,
                format!("No fundamental data found for {}", o.fmt(&yf_symbol)?),
            );
        }

        // Patch with live price for accuracy (fundamentals cache can be up to 24h)
        let live = || -> PyResult<()> {
            let currencies =
                PySet::new(py, [getattr_or(&config, "DEFAULT_CURRENCY", s(py, "USD"))?])?;
            let quotes = mdp.call_method1(
                "get_current_quotes",
                (
                    PyList::new(py, [symbol])?,
                    currencies,
                    &user_symbol_map,
                    &user_excluded_symbols,
                ),
            )?;
            let q_res = unpack(&quotes, 5)?.swap_remove(0);
            if q_res.contains(symbol)? {
                let live_p = get(&q_res.get_item(symbol)?, "price")?;
                if live_p.is_truthy()? {
                    info.set_item("regularMarketPrice", &live_p)?;
                    info.set_item("currentPrice", &live_p)?;
                }
            }
            Ok(())
        };
        match live() {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "warning",
                    format!(
                        "Live price patch failed during intrinsic value calculation for {}: {}",
                        o.fmt(symbol)?,
                        es(py, &e)
                    ),
                )?;
            }
            Err(e) => return Err(e),
        }

        // Use Prefetched or Fetch New
        let statement = |pre: &Bound<'py, PyAny>, name: &str| -> PyResult<Bound<'py, PyAny>> {
            if !pre.is_none() {
                return Ok(pre.clone());
            }
            mdp.call_method(name, (&yf_symbol, "annual"), Some(&kwargs(py, &refresh)?))
        };
        let financials = statement(prefetched_financials, "get_financials")?;
        let balance_sheet = statement(prefetched_balance_sheet, "get_balance_sheet")?;
        let cashflow = statement(prefetched_cashflow, "get_cashflow")?;

        // 3. Handle Overrides
        let mut symbol_overrides = PyDict::new(py).into_any();
        if config_manager.is_truthy()? {
            let val_overrides = config_manager
                .getattr("manual_overrides")?
                .call_method1("get", ("valuation_overrides", PyDict::new(py)))?;
            symbol_overrides = val_overrides
                .call_method1("get", (yf_symbol.call_method0("upper")?, PyDict::new(py)))?;
            if !symbol_overrides.is_truthy()? {
                symbol_overrides = val_overrides
                    .call_method1("get", (symbol.call_method0("upper")?, PyDict::new(py)))?;
            }
        }

        // 4. Calculate
        let m = fr(py)?;
        let comprehensive = m.getattr("get_comprehensive_intrinsic_value")?;
        let prices = m
            .getattr("load_price_history_for_multiples")?
            .call1((mdp, &yf_symbol))?;
        comprehensive.call(
            (info, financials, balance_sheet, cashflow),
            Some(&kwargs(
                py,
                &[
                    ("overrides", &symbol_overrides),
                    ("iterations", iterations),
                    ("prices_df", &prices),
                ],
            )?),
        )
    };
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Error in get_intrinsic_value_for_symbol for {}: {}",
                    o.fmt(&yf_symbol)?,
                    es(py, &e)
                ),
            )?;
            error_dict(py, es(py, &e))
        }
        Err(e) => Err(e),
    }
}
