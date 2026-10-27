//! Buffett quality metrics: `buffett_metrics._compute_generic_metrics`,
//! `_compute_bank_metrics` and `_compute_reit_metrics`, ported line for line.
//!
//! Inputs keep Python's raw values: a concept series maps a period to `None`
//! (Python's None) or a float that may be NaN or infinite, because several
//! metrics read raw values without the `_finite` filter — the revenue deltas
//! (`current or 0`), loss-year counts (NaN counts as observed), share counts
//! (infinity passes `v > 0` and is filtered later by `_cagr`). Ratio values
//! arrive already passed through `_finite`: every use of them does so.
//!
//! The statistics are Python's `statistics` module: `median` is the middle
//! element or `(a + b) / 2`; `pstdev` is computed in exact rational
//! arithmetic and correctly rounded ([`py_pstdev`]).

// Nested `if`s mirror the Python branch for branch.
#![allow(clippy::collapsible_if)]

use std::collections::HashMap;

use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};

use crate::mc::libm_pow;

pub const MIN_PERIODS_FOR_CONSISTENCY: usize = 5;
pub const DURABILITY_WINDOW: usize = 10;

/// One concept: period -> raw value (`None` is Python's None).
pub type Series = HashMap<String, Option<f64>>;
/// Concept name -> series.
pub type Concepts = HashMap<String, Series>;
/// Period -> ratio label -> value already passed through `_finite`.
pub type Ratios = HashMap<String, HashMap<String, Option<f64>>>;

/// A metric value as Python stores it: None, a float, or an int (the counts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    None,
    Float(f64),
    Int(i64),
}

impl From<Option<f64>> for Metric {
    fn from(v: Option<f64>) -> Self {
        v.map_or(Metric::None, Metric::Float)
    }
}

/// Metrics in Python's insertion order.
pub type Metrics = Vec<(&'static str, Metric)>;

fn set(m: &mut Metrics, key: &'static str, value: impl Into<Metric>) {
    let value = value.into();
    if let Some(slot) = m.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        m.push((key, value));
    }
}

fn get_metric(m: &Metrics, key: &str) -> Option<f64> {
    m.iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| match v {
            Metric::Float(f) => Some(*f),
            Metric::Int(i) => Some(*i as f64),
            Metric::None => None,
        })
}

// --- Python helpers ----------------------------------------------------------

#[inline]
fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite())
}

/// `x or 0.0` / `x or 0` on a value that may be None.
#[inline]
fn or0(v: Option<f64>) -> f64 {
    match v {
        Some(x) if x != 0.0 => x,
        _ => 0.0,
    }
}

/// `_safe_divide`: None unless both are finite and the denominator nonzero.
fn safe_divide(num: Option<f64>, den: Option<f64>) -> Option<f64> {
    let (n, d) = (finite(num)?, finite(den)?);
    if d == 0.0 { None } else { Some(n / d) }
}

/// `statistics.median` of the finite values (`_median`).
pub fn py_median(values: &[Option<f64>]) -> Option<f64> {
    let mut clean: Vec<f64> = values.iter().filter_map(|&v| finite(v)).collect();
    if clean.is_empty() {
        return None;
    }
    // sorted() is stable, as is sort_by: 0.0 and -0.0 keep their order.
    clean.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = clean.len();
    Some(if n % 2 == 1 {
        clean[n / 2]
    } else {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    })
}

/// `_stdev`: `statistics.pstdev` of the finite values, needing two.
pub fn py_stdev(values: &[Option<f64>]) -> Option<f64> {
    let clean: Vec<f64> = values.iter().filter_map(|&v| finite(v)).collect();
    if clean.len() < 2 {
        None
    } else {
        Some(py_pstdev(&clean))
    }
}

/// `statistics.pstdev` for finite floats: the population variance in exact
/// rational arithmetic, then CPython's correctly rounded `_float_sqrt_of_frac`.
pub fn py_pstdev(data: &[f64]) -> f64 {
    let n = data.len();
    assert!(n >= 1, "pstdev requires at least one data point");
    // Every float is mantissa * 2^exp; put them all on the smallest exponent.
    let decoded: Vec<(BigInt, i32)> = data
        .iter()
        .map(|&x| {
            let (mant, exp, sign) = decode(x);
            (BigInt::from(mant) * i64::from(sign), exp)
        })
        .collect();
    let e_min = decoded.iter().map(|&(_, e)| e).min().unwrap();
    let ints: Vec<BigInt> = decoded
        .into_iter()
        .map(|(m, e)| m << ((e - e_min) as usize))
        .collect();
    let count = BigInt::from(n);
    let sx: BigInt = ints.iter().sum();
    let sxx: BigInt = ints.iter().map(|x| x * x).sum();
    // mss = (n*sxx - sx^2) / n^2, in units of 2^(2*e_min).
    let num = &count * &sxx - &sx * &sx;
    let mut numerator = num.to_biguint().expect("variance is non-negative");
    let mut denominator = (&count * &count).to_biguint().unwrap();
    let scale = 2 * i64::from(e_min);
    if scale >= 0 {
        numerator <<= scale as usize;
    } else {
        denominator <<= (-scale) as usize;
    }
    float_sqrt_of_frac(&numerator, &denominator)
}

/// `f64::integer_decode`: (mantissa, exponent, sign) with x = sign*mant*2^exp.
fn decode(x: f64) -> (u64, i32, i8) {
    let bits = x.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = if exponent == 0 {
        (bits & 0xf_ffff_ffff_ffff) << 1
    } else {
        (bits & 0xf_ffff_ffff_ffff) | 0x10_0000_0000_0000
    };
    (mantissa, exponent - (1023 + 52), sign)
}

/// `statistics._integer_sqrt_of_frac_rto`: isqrt(n // m), rounded to odd.
fn isqrt_rto(n: &BigUint, m: &BigUint) -> BigUint {
    let a = (n / m).sqrt();
    let exact = &a * &a * m == *n;
    if exact { a } else { a | BigUint::from(1u8) }
}

/// `statistics._float_sqrt_of_frac`: sqrt(n/m), correctly rounded.
fn float_sqrt_of_frac(n: &BigUint, m: &BigUint) -> f64 {
    const SQRT_BIT_WIDTH: i64 = 109; // 2 * 53 + 3
    if n.is_zero() {
        return 0.0;
    }
    let bits = n.bits() as i64 - m.bits() as i64 - SQRT_BIT_WIDTH;
    let q = bits.div_euclid(2); // Python's // floors
    let root = if q >= 0 {
        isqrt_rto(n, &(m << (2 * q) as usize))
    } else {
        isqrt_rto(&(n << (-2 * q) as usize), m)
    };
    // `numerator / denominator` with a power-of-two denominator: rounding the
    // (~55-bit) root to a double and scaling by 2^q is the correctly rounded
    // quotient Python's int division returns.
    let r = root.to_u128().expect("root fits 128 bits") as f64;
    scale_pow2(r, q)
}

/// x * 2^e, exactly while the result stays a normal double.
fn scale_pow2(mut x: f64, mut e: i64) -> f64 {
    while e > 1000 {
        x *= f64::from_bits(((1023 + 1000) as u64) << 52);
        e -= 1000;
    }
    while e < -1000 {
        x *= f64::from_bits(((1023 - 1000) as u64) << 52);
        e += 1000;
    }
    x * f64::from_bits(((1023 + e) as u64) << 52)
}

/// `_cagr`: (end / start)^(1 / years) - 1 over the finite values; None when
/// fewer than two or either end is not positive.
pub fn py_cagr(series_oldest_first: &[Option<f64>]) -> Option<f64> {
    let clean: Vec<f64> = series_oldest_first
        .iter()
        .filter_map(|&v| finite(v))
        .collect();
    if clean.len() < 2 {
        return None;
    }
    let (start, end) = (clean[0], clean[clean.len() - 1]);
    let years = (clean.len() - 1) as f64;
    if start <= 0.0 || end <= 0.0 {
        return None;
    }
    Some(libm_pow(end / start, 1.0 / years) - 1.0)
}

fn raw(c: &Concepts, name: &str, period: &str) -> Option<f64> {
    c.get(name).and_then(|s| s.get(period)).copied().flatten()
}

fn ratio(r: &Ratios, period: &str, label: &str) -> Option<f64> {
    r.get(period)
        .and_then(|row| row.get(label))
        .copied()
        .flatten()
}

fn keys(c: &Concepts, name: &str) -> Vec<String> {
    c.get(name)
        .map(|s| s.keys().cloned().collect())
        .unwrap_or_default()
}

/// `sorted(set(a) | set(b) ...)` of the named concepts' periods.
fn sorted_periods(c: &Concepts, names: &[&str]) -> Vec<String> {
    let mut all: Vec<String> = names.iter().flat_map(|n| keys(c, n)).collect();
    all.sort();
    all.dedup();
    all
}

fn window_of(periods: &[String]) -> &[String] {
    &periods[periods.len().saturating_sub(DURABILITY_WINDOW)..]
}

/// `_aligned`: one concept's raw values on the period axis.
fn aligned(c: &Concepts, name: &str, periods: &[String]) -> Vec<Option<f64>> {
    periods.iter().map(|p| raw(c, name, p)).collect()
}

/// `_total_debt`: short + long-term debt, None when neither is reported.
pub fn total_debt(c: &Concepts, period: &str) -> Option<f64> {
    let short_term = finite(raw(c, "short_term_debt", period));
    let long_term = finite(raw(c, "long_term_debt", period));
    if short_term.is_none() && long_term.is_none() {
        return None;
    }
    Some(or0(short_term) + or0(long_term))
}

/// `_owner_earnings`: operating cash flow less capex, None without OCF.
fn owner_earnings(c: &Concepts, periods: &[String]) -> Vec<Option<f64>> {
    periods
        .iter()
        .map(|p| {
            let operating = finite(raw(c, "operating_cash_flow", p))?;
            let spend = finite(raw(c, "capex", p));
            Some(operating - or0(spend))
        })
        .collect()
}

fn ratio_series(r: &Ratios, window: &[String], label: &str) -> Vec<Option<f64>> {
    window.iter().map(|p| ratio(r, p, label)).collect()
}

fn present(v: &[Option<f64>]) -> Vec<Option<f64>> {
    v.iter().copied().filter(|x| x.is_some()).collect()
}

fn positive_raw(v: &[Option<f64>]) -> Vec<Option<f64>> {
    v.iter()
        .copied()
        .filter(|x| matches!(x, Some(s) if *s > 0.0))
        .collect()
}

fn per_share(values: &[Option<f64>], shares: &[Option<f64>]) -> Vec<Option<f64>> {
    values
        .iter()
        .zip(shares)
        .filter_map(|(&v, &s)| safe_divide(v, s).filter(|x| *x > 0.0).map(Some))
        .collect()
}

// --- generic model -----------------------------------------------------------

/// `_compute_generic_metrics`.
pub fn generic_metrics(c: &Concepts, r: &Ratios) -> Metrics {
    let mut m: Metrics = Vec::new();
    let all_periods = sorted_periods(c, &["net_income", "revenue"]);
    let window = window_of(&all_periods);

    let roe_series = ratio_series(r, window, "Return on Equity (ROE) (%)");
    let roa_series = ratio_series(r, window, "Return on Assets (ROA) (%)");
    let gross_series = ratio_series(r, window, "Gross Profit Margin (%)");
    let net_margin_series = ratio_series(r, window, "Net Profit Margin (%)");
    set(&mut m, "roe_median", py_median(&roe_series));
    set(&mut m, "roa_median", py_median(&roa_series));
    set(&mut m, "gross_margin_median", py_median(&gross_series));
    set(&mut m, "net_margin_median", py_median(&net_margin_series));

    let clean_roe: Vec<f64> = roe_series.iter().filter_map(|&v| finite(v)).collect();
    let above_15 = clean_roe.iter().filter(|&&v| v >= 15.0).count();
    let has_roe = !clean_roe.is_empty();
    set(
        &mut m,
        "roe_years_above_15",
        has_roe.then(|| above_15 as f64 / clean_roe.len() as f64),
    );
    set(
        &mut m,
        "roe_years_above_15_count",
        has_roe.then_some(above_15 as f64),
    );
    set(
        &mut m,
        "roe_observation_years",
        has_roe.then_some(clean_roe.len() as f64),
    );

    let mut roic_values = Vec::new();
    for p in window {
        let ebit = finite(raw(c, "pretax_income", p));
        let interest = or0(finite(raw(c, "interest_expense", p)));
        let tax = finite(raw(c, "tax_provision", p));
        let pretax = ebit;
        let Some(ebit) = ebit else { continue };
        let ebit = ebit + interest;
        let mut tax_rate = match pretax {
            Some(pt) if pt != 0.0 && pt > 0.0 => safe_divide(tax, pretax),
            _ => None,
        };
        if !matches!(tax_rate, Some(t) if (0.0..=0.6).contains(&t)) {
            tax_rate = Some(0.21);
        }
        let nopat = ebit * (1.0 - tax_rate.unwrap());
        let equity = finite(raw(c, "equity", p));
        let debt =
            or0(finite(raw(c, "short_term_debt", p))) + or0(finite(raw(c, "long_term_debt", p)));
        let cash = or0(finite(raw(c, "cash", p)));
        let Some(equity) = equity else { continue };
        let invested = equity + debt - cash;
        if let Some(v) = safe_divide(Some(nopat), Some(invested)) {
            if -2.0 < v && v < 3.0 {
                roic_values.push(Some(v * 100.0));
            }
        }
    }
    set(&mut m, "roic_median", py_median(&roic_values));

    if let Some(latest) = all_periods.last().filter(|s| !s.is_empty()) {
        set(
            &mut m,
            "debt_to_equity",
            ratio(r, latest, "Debt-to-Equity Ratio"),
        );
        set(&mut m, "current_ratio", ratio(r, latest, "Current Ratio"));
        let cash = or0(finite(raw(c, "cash", latest)));
        let td = total_debt(c, latest);
        let interest = finite(raw(c, "interest_expense", latest));
        let pretax = finite(raw(c, "pretax_income", latest));
        set(&mut m, "total_debt", td);
        set(&mut m, "net_debt", td.map(|t| t - cash));
        let interest_truthy = matches!(interest, Some(i) if i != 0.0);
        // `if interest and interest > 0 and pretax is not None`
        if let (Some(i), Some(pt)) = (interest.filter(|&i| i != 0.0 && i > 0.0), pretax) {
            set(&mut m, "interest_coverage", Some((pt + i) / i));
        } else if td == Some(0.0) || (td.is_none() && !interest_truthy) {
            set(&mut m, "interest_coverage", None::<f64>);
            set(&mut m, "debt_free", Some(1.0));
        } else {
            set(&mut m, "interest_coverage", None::<f64>);
        }
    }

    let oe = owner_earnings(c, window);
    let latest_oe = oe.iter().rev().find_map(|&v| v);
    set(&mut m, "owner_earnings_latest", latest_oe);
    let nd_to_oe = match latest_oe {
        Some(l) if l != 0.0 && l > 0.0 => safe_divide(get_metric(&m, "net_debt"), Some(l)),
        _ => None,
    };
    set(&mut m, "net_debt_to_owner_earnings", nd_to_oe);

    set(&mut m, "roe_stdev", py_stdev(&roe_series));
    let observed_oe: Vec<f64> = oe.iter().filter_map(|&v| v).collect();
    set(
        &mut m,
        "negative_owner_earnings_years",
        Metric::Int(observed_oe.iter().filter(|&&v| v < 0.0).count() as i64),
    );
    set(
        &mut m,
        "owner_earnings_years",
        (!observed_oe.is_empty()).then_some(observed_oe.len() as f64),
    );

    let revenue_series = aligned(c, "revenue", window);
    let mut growth_rates = Vec::new();
    for pair in revenue_series.windows(2) {
        let (previous, current) = (pair[0], pair[1]);
        if let Some(rate) = safe_divide(Some(or0(current) - or0(previous)), previous) {
            if rate.abs() < 5.0 {
                growth_rates.push(Some(rate));
            }
        }
    }
    set(&mut m, "revenue_growth_stdev", py_stdev(&growth_rates));

    let mut fcf_margins = Vec::new();
    for (p, &owner) in window.iter().zip(&oe) {
        let revenue = finite(raw(c, "revenue", p));
        if let Some(margin) = safe_divide(owner, revenue) {
            if -2.0 < margin && margin < 2.0 {
                fcf_margins.push(Some(margin));
            }
        }
    }
    set(&mut m, "fcf_margin_median", py_median(&fcf_margins));
    set(&mut m, "fcf_margin_stdev", py_stdev(&fcf_margins));

    set(&mut m, "revenue_cagr", py_cagr(&present(&revenue_series)));
    set(&mut m, "owner_earnings_cagr", py_cagr(&present(&oe)));

    let equity_series = aligned(c, "equity", window);
    let shares_series = aligned(c, "shares_diluted", window);
    set(
        &mut m,
        "book_value_per_share_cagr",
        py_cagr(&per_share(&equity_series, &shares_series)),
    );
    set(
        &mut m,
        "share_count_cagr",
        py_cagr(&positive_raw(&shares_series)),
    );

    let dividends = aligned(c, "dividends_paid", window);
    let mut payout_ratios = Vec::new();
    for (&dividend, &owner) in dividends.iter().zip(&oe) {
        let (Some(dividend), Some(owner)) = (dividend, owner) else {
            continue;
        };
        if owner <= 0.0 {
            continue;
        }
        let ratio = dividend.abs() / owner;
        if (0.0..3.0).contains(&ratio) {
            payout_ratios.push(Some(ratio));
        }
    }
    set(&mut m, "payout_ratio_median", py_median(&payout_ratios));

    if window.len() >= MIN_PERIODS_FOR_CONSISTENCY {
        let (first, last) = (&window[0], &window[window.len() - 1]);
        let ebit_start = finite(raw(c, "pretax_income", first));
        let ebit_end = finite(raw(c, "pretax_income", last));
        let equity_start = finite(raw(c, "equity", first));
        let equity_end = finite(raw(c, "equity", last));
        if let (Some(es), Some(ee), Some(qs), Some(qe)) =
            (ebit_start, ebit_end, equity_start, equity_end)
        {
            let delta_capital = qe - qs;
            if delta_capital > 0.0 {
                let incremental = (ee - es) / delta_capital;
                if -2.0 < incremental && incremental < 5.0 {
                    set(&mut m, "incremental_roic", Some(incremental * 100.0));
                }
            }
        }
    }
    m
}

// --- bank model --------------------------------------------------------------

/// `_compute_bank_metrics`.
pub fn bank_metrics(c: &Concepts, sector: &Concepts, r: &Ratios) -> Metrics {
    let mut m: Metrics = Vec::new();
    let all_periods = sorted_periods(c, &["net_income"]);
    let window = window_of(&all_periods);

    let roe_series = ratio_series(r, window, "Return on Equity (ROE) (%)");
    let roa_series = ratio_series(r, window, "Return on Assets (ROA) (%)");
    set(&mut m, "roe_median", py_median(&roe_series));
    set(&mut m, "roa_median", py_median(&roa_series));
    set(&mut m, "roe_stdev", py_stdev(&roe_series));

    let clean_roe: Vec<f64> = roe_series.iter().filter_map(|&v| finite(v)).collect();
    let above_12 = clean_roe.iter().filter(|&&v| v >= 12.0).count();
    let has_roe = !clean_roe.is_empty();
    set(
        &mut m,
        "roe_years_above_12",
        has_roe.then(|| above_12 as f64 / clean_roe.len() as f64),
    );
    set(
        &mut m,
        "roe_years_above_12_count",
        has_roe.then_some(above_12 as f64),
    );
    set(
        &mut m,
        "roe_observation_years",
        has_roe.then_some(clean_roe.len() as f64),
    );

    let mut equity_to_assets = Vec::new();
    for p in window {
        if let Some(v) = safe_divide(raw(c, "equity", p), raw(c, "total_assets", p)) {
            if 0.0 < v && v < 1.0 {
                equity_to_assets.push(Some(v * 100.0));
            }
        }
    }
    set(
        &mut m,
        "equity_to_assets_median",
        py_median(&equity_to_assets),
    );
    set(
        &mut m,
        "equity_to_assets_latest",
        equity_to_assets.last().copied().flatten(),
    );

    let mut provision_rates = Vec::new();
    for p in window {
        if let Some(v) = safe_divide(
            raw(sector, "provision_for_credit_losses", p),
            raw(sector, "loans", p),
        ) {
            if -0.1 < v && v < 0.5 {
                provision_rates.push(Some(v * 100.0));
            }
        }
    }
    set(&mut m, "provision_rate_median", py_median(&provision_rates));
    set(&mut m, "provision_rate_stdev", py_stdev(&provision_rates));

    let mut nim_values = Vec::new();
    for p in window {
        if let Some(v) = safe_divide(
            raw(sector, "net_interest_income", p),
            raw(c, "total_assets", p),
        ) {
            if 0.0 < v && v < 0.25 {
                nim_values.push(Some(v * 100.0));
            }
        }
    }
    set(&mut m, "net_interest_margin_median", py_median(&nim_values));
    set(&mut m, "net_interest_margin_stdev", py_stdev(&nim_values));

    let mut efficiency = Vec::new();
    for p in window {
        let net_interest = finite(raw(sector, "net_interest_income", p));
        let noninterest_income = finite(raw(sector, "noninterest_income", p));
        let noninterest_expense = finite(raw(sector, "noninterest_expense", p));
        let (Some(ni), Some(_)) = (net_interest, noninterest_expense) else {
            continue;
        };
        let revenue = ni + or0(noninterest_income);
        if let Some(v) = safe_divide(noninterest_expense, Some(revenue)) {
            if 0.0 < v && v < 2.0 {
                efficiency.push(Some(v * 100.0));
            }
        }
    }
    set(&mut m, "efficiency_ratio_median", py_median(&efficiency));

    let net_income_series = aligned(c, "net_income", window);
    let observed: Vec<f64> = net_income_series.iter().filter_map(|&v| v).collect();
    set(
        &mut m,
        "loss_years",
        Metric::Int(observed.iter().filter(|&&v| v < 0.0).count() as i64),
    );
    set(
        &mut m,
        "net_income_years",
        (!observed.is_empty()).then_some(observed.len() as f64),
    );

    let equity_series = aligned(c, "equity", window);
    let shares_series = aligned(c, "shares_diluted", window);
    set(
        &mut m,
        "book_value_per_share_cagr",
        py_cagr(&per_share(&equity_series, &shares_series)),
    );

    let banking_revenue: Vec<Option<f64>> = window
        .iter()
        .map(|p| match finite(raw(sector, "net_interest_income", p)) {
            Some(ni) => Some(ni + or0(finite(raw(sector, "noninterest_income", p)))),
            None => finite(raw(c, "revenue", p)),
        })
        .collect();
    set(&mut m, "revenue_cagr", py_cagr(&present(&banking_revenue)));
    set(
        &mut m,
        "share_count_cagr",
        py_cagr(&positive_raw(&shares_series)),
    );
    m
}

// --- REIT model --------------------------------------------------------------

/// `_compute_reit_metrics`.
pub fn reit_metrics(c: &Concepts, sector: &Concepts, _r: &Ratios) -> Metrics {
    let mut m: Metrics = Vec::new();
    let all_periods = sorted_periods(c, &["net_income"]);
    let window = window_of(&all_periods);

    let ffo_series: Vec<Option<f64>> = window
        .iter()
        .map(|p| {
            let net_income = finite(raw(c, "net_income", p))?;
            let depreciation = or0(finite(raw(c, "depreciation_amortization", p)));
            let gains = or0(finite(raw(sector, "gain_on_sale_real_estate", p)));
            Some(net_income + depreciation - gains)
        })
        .collect();
    let observed: Vec<f64> = ffo_series.iter().filter_map(|&v| v).collect();
    set(
        &mut m,
        "ffo_latest",
        ffo_series.iter().rev().find_map(|&v| v),
    );
    set(
        &mut m,
        "ffo_cagr",
        py_cagr(&observed.iter().map(|&v| Some(v)).collect::<Vec<_>>()),
    );
    set(
        &mut m,
        "negative_ffo_years",
        Metric::Int(observed.iter().filter(|&&v| v < 0.0).count() as i64),
    );
    set(
        &mut m,
        "ffo_years",
        (!observed.is_empty()).then_some(observed.len() as f64),
    );

    let mut ffo_margins = Vec::new();
    let mut ffo_on_equity = Vec::new();
    for (p, &ffo) in window.iter().zip(&ffo_series) {
        let Some(ffo) = ffo else { continue };
        let revenue = finite(raw(c, "revenue", p));
        if let Some(v) = safe_divide(Some(ffo), revenue) {
            if -2.0 < v && v < 2.0 {
                ffo_margins.push(Some(v * 100.0));
            }
        }
        if let Some(v) = safe_divide(Some(ffo), raw(c, "equity", p)) {
            if -1.0 < v && v < 2.0 {
                ffo_on_equity.push(Some(v * 100.0));
            }
        }
    }
    set(&mut m, "ffo_margin_median", py_median(&ffo_margins));
    set(&mut m, "ffo_margin_stdev", py_stdev(&ffo_margins));
    set(&mut m, "ffo_on_equity_median", py_median(&ffo_on_equity));

    if let Some(latest) = window.last().filter(|s| !s.is_empty()) {
        let td = total_debt(c, latest);
        set(&mut m, "total_debt", td);
        if td.is_some() {
            let ffo_latest = get_metric(&m, "ffo_latest");
            let to_ffo = match ffo_latest {
                Some(f) if f != 0.0 && f > 0.0 => safe_divide(td, Some(f)),
                _ => None,
            };
            set(&mut m, "debt_to_ffo", to_ffo);
            set(
                &mut m,
                "debt_to_real_estate",
                safe_divide(td, raw(sector, "real_estate_net", latest)),
            );
        }
    }

    let shares_series = aligned(c, "shares_diluted", window);
    set(
        &mut m,
        "share_count_cagr",
        py_cagr(&positive_raw(&shares_series)),
    );
    set(
        &mut m,
        "ffo_per_share_cagr",
        py_cagr(&per_share(&ffo_series, &shares_series)),
    );
    set(
        &mut m,
        "real_estate_cagr",
        py_cagr(&present(&aligned(sector, "real_estate_net", window))),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pstdev_matches_cpython_examples() {
        // statistics.pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75]) == 0.986893273527251
        assert_eq!(
            py_pstdev(&[1.5, 2.5, 2.5, 2.75, 3.25, 4.75]),
            0.986893273527251
        );
        assert_eq!(py_pstdev(&[2.0, 2.0]), 0.0);
        assert_eq!(py_pstdev(&[1.0, 3.0]), 1.0);
    }

    #[test]
    fn median_like_statistics() {
        assert_eq!(
            py_median(&[Some(3.0), None, Some(1.0), Some(f64::NAN)]),
            Some(2.0)
        );
        assert_eq!(py_median(&[None]), None);
    }

    #[test]
    fn cagr_refuses_non_positive_ends() {
        assert_eq!(py_cagr(&[Some(-1.0), Some(2.0)]), None);
        assert_eq!(
            py_cagr(&[Some(100.0), Some(121.0), Some(f64::INFINITY)]).map(|g| (g * 1e6).round()),
            Some(210_000.0)
        );
    }
}
