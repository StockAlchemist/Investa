//! The deterministic side of `financial_ratios`: the scalar faded projection
//! the DCF/DDM/D-CFO/D-NI models share, the cost of capital, the growth
//! blend, the medians the normalizers take, and the model blend.
//!
//! These replace Python *scalar* arithmetic, so they reproduce Python's rules
//! rather than NumPy's array rules:
//! * `float ** x` and `np.float64 ** x` are the C library's `pow`, fast paths
//!   or not ([`libm_pow`]) — measured: NumPy scalars never take the
//!   array fast paths.
//! * `sum(list)` is CPython's compensated sum while the items are exact
//!   floats, and plain addition once a NumPy scalar appears ([`py_sum`]); a
//!   projection's terms are NumPy scalars exactly when one of its inputs was.
//! * `max(a, b)` keeps `a` unless `b > a`.
//! * Truthiness: `None` and `0` are false, NaN is true ([`truthy`]).
//!
//! What stays in Python: reading the statements and the company profile,
//! choosing between the methods, the result dictionaries and their notes, and
//! one-expression formulas (Graham's `eps * (8.5 + 2g) * 4.4 / y`, Lynch's
//! `eps * multiplier`), which would gain a call and lose nothing by moving.

use crate::mc::{libm_pow, np_clip};
use crate::stats::{np_mean, np_sum};
use crate::util::{PyNum, py_sum};

/// Python's `max(a, b)`: `a` unless `b > a`.
#[inline]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python truthiness of an optional number: `None` and zero are false.
#[inline]
pub fn truthy(v: Option<f64>) -> bool {
    matches!(v, Some(x) if x != 0.0)
}

/// `np.median(values)`: NaN when any value is NaN or there are none; else the
/// mean (NumPy's, from +0.0) of the middle one or two order statistics.
pub fn np_median(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = sorted.len();
    let middle = if n % 2 == 1 {
        &sorted[n / 2..n / 2 + 1]
    } else {
        &sorted[n / 2 - 1..n / 2 + 1]
    };
    np_sum(middle) / middle.len() as f64
}

/// The scalar faded projection inside `calculate_intrinsic_value_dcf` and its
/// DDM/D-CFO/D-NI siblings.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    /// Each year's discounted flow.
    pub pv: Vec<f64>,
    /// `sum(pv)`, as Python adds it (0.0 when there are no years).
    pub sum_pv: f64,
    /// The last projected (undiscounted) flow: the base when there are no years.
    pub last: f64,
    pub pv_terminal: f64,
}

/// `for y in range(1, projection_years + 1): ...` then the Gordon terminal
/// value, exactly as the models loop. `numpy_terms` says whether the terms are
/// NumPy scalars (one of the inputs was), which decides how `sum` adds them.
pub fn scalar_projection(
    base: f64,
    applied_growth: f64,
    discount_rate: f64,
    projection_years: i64,
    terminal_growth_rate: f64,
    numpy_terms: bool,
) -> Projection {
    let mut pv = Vec::new();
    let mut current = base;
    let mut y = 1_i64;
    while y < projection_years + 1 {
        let fade = if projection_years > 1 {
            (y - 1) as f64 / (projection_years - 1) as f64
        } else {
            0.0
        };
        let yearly = applied_growth - (applied_growth - terminal_growth_rate) * fade;
        let next = current * (1.0 + yearly);
        pv.push(next / libm_pow(1.0 + discount_rate, y as f64 - 0.5));
        current = next;
        y += 1;
    }
    let safe_discount = py_max(discount_rate, terminal_growth_rate + 0.01);
    let terminal =
        (current * (1.0 + terminal_growth_rate)) / (safe_discount - terminal_growth_rate);
    let pv_terminal = terminal / libm_pow(1.0 + discount_rate, projection_years as f64);
    let kind = if numpy_terms {
        PyNum::Other
    } else {
        PyNum::Float
    };
    let items: Vec<(f64, PyNum)> = pv.iter().map(|&v| (v, kind)).collect();
    // sum([]) is the int 0, and 0 + x is 0.0 + x.
    let sum_pv = py_sum(&items).unwrap_or(0.0);
    Projection {
        pv,
        sum_pv,
        last: current,
        pv_terminal,
    }
}

/// `calculate_wacc`'s inputs once read out of the profile and statements.
/// `None` is Python's `None`; statements absent means no statement values.
#[derive(Debug, Clone, Copy)]
pub struct WaccInputs {
    pub raw_beta: Option<f64>,
    pub risk_free_rate: f64,
    pub market_return: f64,
    pub default_tax_rate: f64,
    pub market_cap: Option<f64>,
    pub total_debt: Option<f64>,
    pub have_statements: bool,
    pub interest_expense: Option<f64>,
    pub income_tax_expense: Option<f64>,
    pub pretax_income: Option<f64>,
    pub operating_income: Option<f64>,
    pub max_cost_of_debt: f64,
    pub min_discount: f64,
    pub max_discount: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wacc {
    pub wacc: f64,
    pub cost_of_equity: f64,
    pub cost_of_debt: f64,
    pub tax_rate: f64,
    pub beta: f64,
    pub size_premium: f64,
    /// None when there was no market cap: the WACC is then the clipped cost
    /// of equity, and the debt side is not reported.
    pub weights: Option<(f64, f64)>,
}

/// `calculate_wacc`: CAPM with a Blume-adjusted beta and a size premium, cost
/// of debt from interest/debt or a coverage-based synthetic spread, weights
/// at market value. `size_bands` are `SIZE_PREMIUM_BANDS`.
pub fn wacc(inp: &WaccInputs, size_bands: &[(f64, f64)]) -> Wacc {
    let rf = inp.risk_free_rate;
    let beta = match inp.raw_beta {
        None => 1.0,
        Some(b) if b.is_nan() => 1.0,
        Some(b) => np_clip(0.67 * b + 0.33 * 1.0, 0.3, 3.0),
    };
    let mut cost_of_equity = rf + beta * (inp.market_return - rf);

    let mut size_premium = 0.0;
    if truthy(inp.market_cap) {
        let mc = inp.market_cap.unwrap();
        for &(threshold, premium) in size_bands {
            if mc < threshold {
                size_premium = premium;
                break;
            }
        }
    }
    cost_of_equity += size_premium;

    let mut cost_of_debt = rf + 0.015;
    let mut tax_rate = inp.default_tax_rate;
    let total_debt = inp.total_debt;
    if inp.have_statements {
        let ie = inp.interest_expense;
        if truthy(ie) && truthy(total_debt) && total_debt.unwrap() > 0.0 {
            let implied = ie.unwrap().abs() / total_debt.unwrap();
            cost_of_debt = np_clip(implied, rf, inp.max_cost_of_debt);
        } else if truthy(inp.operating_income) && truthy(ie) && ie.unwrap().abs() > 0.0 {
            let icr = inp.operating_income.unwrap() / ie.unwrap().abs();
            let spread = if icr > 8.5 {
                0.010
            } else if icr > 6.5 {
                0.0125
            } else if icr > 4.0 {
                0.0175
            } else if icr > 3.0 {
                0.025
            } else if icr > 1.5 {
                0.035
            } else {
                0.055
            };
            cost_of_debt = rf + spread;
        }
        if truthy(inp.income_tax_expense)
            && truthy(inp.pretax_income)
            && inp.pretax_income.unwrap() > 0.0
        {
            tax_rate = np_clip(
                inp.income_tax_expense.unwrap() / inp.pretax_income.unwrap(),
                0.0,
                0.5,
            );
        }
    }

    if !truthy(inp.market_cap) {
        return Wacc {
            wacc: np_clip(cost_of_equity, inp.min_discount, inp.max_discount),
            cost_of_equity,
            cost_of_debt,
            tax_rate,
            beta,
            size_premium,
            weights: None,
        };
    }
    let market_cap = inp.market_cap.unwrap();
    // `total_debt or 0`
    let debt = if truthy(total_debt) {
        total_debt.unwrap()
    } else {
        0.0
    };
    let total_value = market_cap + debt;
    let weight_equity = market_cap / total_value;
    let weight_debt = debt / total_value;
    let w = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1.0 - tax_rate));
    Wacc {
        wacc: np_clip(w, inp.min_discount, inp.max_discount),
        cost_of_equity,
        cost_of_debt,
        tax_rate,
        beta,
        size_premium,
        weights: Some((weight_equity, weight_debt)),
    }
}

/// `blended_growth_estimate`'s arithmetic: history collapsed to one view (the
/// median of regression and CAGR), averaged with the analysts', shrunk toward
/// the base rate and clamped. Returns (clamped, raw), or None with no signal.
pub fn blend_growth(
    regression: Option<f64>,
    cagr: Option<f64>,
    analyst: Option<f64>,
    shrink_weight: f64,
    base_rate: f64,
    min_growth: f64,
    max_growth: f64,
) -> Option<(f64, f64)> {
    let historical: Vec<f64> = [regression, cagr].into_iter().flatten().collect();
    let mut views = Vec::new();
    if !historical.is_empty() {
        views.push(np_median(&historical));
    }
    if let Some(a) = analyst {
        views.push(a);
    }
    if views.is_empty() {
        return None;
    }
    let raw = np_mean(&views);
    let shrunk = shrink_weight * raw + (1.0 - shrink_weight) * base_rate;
    Some((np_clip(shrunk, min_growth, max_growth), raw))
}

/// The FCF margins `estimate_fcf_margin` / `through_cycle_fcf_margin` keep:
/// periods with a positive revenue and a reported FCF, inside the (-100%,
/// 60%) plausibility band. `revenue[i]`/`fcf[i]` None mean "not reported".
pub fn plausible_margins(revenue: &[Option<f64>], fcf: &[Option<f64>]) -> Vec<f64> {
    revenue
        .iter()
        .zip(fcf)
        .filter_map(|(&rev, &f)| {
            let (r, f) = (rev?, f?);
            if !truthy(Some(r)) || r <= 0.0 {
                return None;
            }
            let margin = f / r;
            (-1.0 < margin && margin < 0.6).then_some(margin)
        })
        .collect()
}

/// `_model_reliability` on a model's Monte Carlo bear/base/bull: 1 / (1 +
/// (cv / cv_ref)^2), floored, where cv = (bull - bear) / (2 base). 1.0 when the
/// figures cannot support a dispersion.
pub fn model_reliability(bear: f64, base: f64, bull: f64, cv_ref: f64, floor: f64) -> f64 {
    if !base.is_finite() || base <= 0.0 || !(bull - bear).is_finite() {
        return 1.0;
    }
    let cv = (bull - bear) / (2.0 * base);
    if !cv.is_finite() || cv < 0.0 {
        return 1.0;
    }
    np_clip(1.0 / (1.0 + libm_pow(cv / cv_ref, 2.0)), floor, 1.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Blend {
    pub value: f64,
    pub weights: Vec<f64>,
    pub spread_pct: Option<f64>,
    pub confidence: f64,
}

/// `blend_intrinsic_values`' arithmetic over the contributing models (in the
/// prior table's order): weight = prior x reliability, a weighted arithmetic
/// mean, the max-min spread, and confidence from reliability, dispersion and
/// breadth. Non-empty input.
pub fn blend(values: &[f64], priors: &[f64], reliabilities: &[f64]) -> Blend {
    let weights: Vec<f64> = priors
        .iter()
        .zip(reliabilities)
        .map(|(&p, &r)| p * r)
        .collect();
    let as_floats = |v: &[f64]| v.iter().map(|&x| (x, PyNum::Float)).collect::<Vec<_>>();
    let total_weight = py_sum(&as_floats(&weights)).unwrap_or(0.0);
    let weighted: Vec<f64> = values.iter().zip(&weights).map(|(&v, &w)| v * w).collect();
    let value = py_sum(&as_floats(&weighted)).unwrap_or(0.0) / total_weight;
    let norm_weights = weights.iter().map(|&w| w / total_weight).collect();

    let spread_pct = if values.len() > 1 && value > 0.0 {
        let max = values.iter().copied().fold(values[0], py_max);
        let min = values.iter().copied().fold(values[0], crate::util::py_min);
        Some((max - min) / value * 100.0)
    } else {
        None
    };

    let mean_reliability = np_mean(reliabilities);
    // `(spread_pct or 0.0) / 100.0`
    let spread_frac = match spread_pct {
        Some(s) if s != 0.0 => s,
        _ => 0.0,
    } / 100.0;
    let dispersion_factor = 1.0 / (1.0 + libm_pow(spread_frac / 0.75, 2.0));
    let breadth_factor = match values.len() {
        1 => 0.75,
        2 => 0.90,
        _ => 1.0,
    };
    Blend {
        value,
        weights: norm_weights,
        spread_pct,
        confidence: np_clip(
            mean_reliability * dispersion_factor * breadth_factor,
            0.0,
            1.0,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_like_numpy() {
        assert_eq!(np_median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(np_median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(np_median(&[1.0, f64::NAN]).is_nan());
        assert!(np_median(&[]).is_nan());
    }

    #[test]
    fn no_years_means_only_the_terminal_value() {
        let p = scalar_projection(100.0, 0.05, 0.1, 0, 0.02, false);
        assert!(p.pv.is_empty());
        assert_eq!(p.sum_pv, 0.0);
        assert_eq!(p.last, 100.0);
    }

    #[test]
    fn truthiness() {
        assert!(!truthy(None));
        assert!(!truthy(Some(0.0)));
        assert!(truthy(Some(f64::NAN)));
        assert!(truthy(Some(-1.0)));
    }
}
