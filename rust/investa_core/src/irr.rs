//! Money-weighted return: `finutils.calculate_npv` and `finutils.calculate_irr`,
//! ported line for line together with the two SciPy root finders the IRR
//! leans on, so the solver visits the same rates and returns the same bits.
//!
//! * [`newton_secant`] — `scipy.optimize.newton` with no derivative (SciPy
//!   1.16, `_zeros_py.py`): the secant method.
//! * [`brentq`] — `scipy.optimize.brentq` (SciPy 1.16,
//!   `optimize/Zeros/brentq.c`), including the wrapper's rule that a NaN
//!   function value aborts the search.
//!
//! Every place SciPy or `calculate_irr` raises becomes `None` here; the IRR
//! loop treats it the way the Python `except: continue` / `pass` did.

// Nested `if`s mirror the Python branch for branch, so the two can be read
// side by side; collapsing them would hide that.
#![allow(clippy::collapsible_if)]

use crate::irr_grid::SCAN_GRID_BITS;
use crate::util::{PyNum, py_sum};

/// Microseconds per day: dates arrive as microsecond ticks, so a datetime's
/// time of day survives into `(d - start).days`, which floors.
pub const DAY_TICKS: i64 = 86_400_000_000;

/// `numpy.isclose(a, b, rtol, atol)` for scalars, `equal_nan=False`.
#[inline]
pub fn np_isclose(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_finite() && b.is_finite() {
        (a - b).abs() <= atol + rtol * b.abs()
    } else {
        a == b
    }
}

/// `calculate_npv`'s result. `negative_delta_at` is the first flow dated
/// before the first one — Python logged a warning there and returned NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Npv {
    pub value: f64,
    pub negative_delta_at: Option<usize>,
}

/// `calculate_npv(rate, dates, cash_flows)`, with `days[i]` the whole days from
/// the first date (`(dates[i] - dates[0]).days`). Lengths must match.
pub fn npv(rate: f64, days: &[i64], flows: &[f64]) -> Npv {
    let nan = Npv {
        value: f64::NAN,
        negative_delta_at: None,
    };
    if !rate.is_finite() {
        return nan;
    }
    if days.is_empty() {
        return Npv {
            value: 0.0,
            negative_delta_at: None,
        };
    }
    let base = 1.0 + rate;
    if base <= 1e-18 {
        return nan;
    }
    let mut acc = 0.0_f64;
    for i in 0..flows.len() {
        let time_delta_years = days[i] as f64 / 365.0;
        if time_delta_years < 0.0 {
            return Npv {
                value: f64::NAN,
                negative_delta_at: Some(i),
            };
        }
        let cf = flows[i];
        if !cf.is_finite() {
            continue;
        }
        // base > 1e-18 here, so the logarithm and the power are defined, and
        // the +/-700 guard keeps the power finite.
        let log_val = base.ln() * time_delta_years;
        let denominator = if log_val > 700.0 {
            f64::INFINITY
        } else if log_val < -700.0 {
            0.0
        } else {
            base.powf(time_delta_years)
        };
        if denominator == 0.0 {
            if cf > 0.0 {
                acc += 1e150;
            } else if cf < 0.0 {
                acc -= 1e150;
            }
            continue;
        }
        if !denominator.is_finite() {
            if time_delta_years == 0.0 {
                acc += cf;
            }
            continue;
        }
        acc += cf / denominator;
    }
    Npv {
        value: if acc.is_finite() { acc } else { f64::NAN },
        negative_delta_at: None,
    }
}

/// `scipy.optimize.newton(func, x0, tol=tol, maxiter=maxiter)` with no
/// derivative: the secant method. `None` where SciPy raises (no convergence,
/// or a flat secant between two distinct points).
pub fn newton_secant(
    mut func: impl FnMut(f64) -> f64,
    x0: f64,
    tol: f64,
    maxiter: usize,
) -> Option<f64> {
    let rtol = 0.0;
    let mut p0 = x0 * 1.0;
    let eps = 1e-4;
    let mut p1 = x0 * (1.0 + eps);
    p1 += if p1 >= 0.0 { eps } else { -eps };
    let mut q0 = func(p0);
    let mut q1 = func(p1);
    if q1.abs() < q0.abs() {
        std::mem::swap(&mut p0, &mut p1);
        std::mem::swap(&mut q0, &mut q1);
    }
    for _ in 0..maxiter {
        if q1 == q0 {
            if p1 != p0 {
                return None; // "Tolerance of ... reached", raised under disp=True
            }
            return Some((p1 + p0) / 2.0);
        }
        let p = if q1.abs() > q0.abs() {
            (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1)
        } else {
            (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0)
        };
        if np_isclose(p, p1, rtol, tol) {
            return Some(p);
        }
        p0 = p1;
        q0 = q1;
        p1 = p;
        q1 = func(p1);
    }
    None // "Failed to converge", raised under disp=True
}

/// `a * b + c` the way SciPy's compiled `brentq` evaluates it at the three
/// sites it fuses. SciPy's arm64 build (clang, `-ffp-contract=on`) emits
/// `fmadd`/`fnmsub` there — one rounding instead of two — so the port fuses
/// too, and the root matches to the last bit. x86-64 wheels are built for
/// baseline x86-64, which has no FMA instruction, so nothing is fused there.
/// The three sites, from the disassembly of `_zeros.cpython-312-darwin.so`:
/// `delta`, the extrapolation numerator, and the step bound `3|sbis| - delta`.
#[inline]
fn brent_fma(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `scipy.optimize.brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=iter)`.
/// `None` where SciPy raises: a NaN function value, endpoints of the same
/// sign, or no convergence.
pub fn brentq(
    mut f: impl FnMut(f64) -> f64,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    iter: usize,
) -> Option<f64> {
    // _wrap_nan_raise: a NaN value raises ValueError.
    let mut eval = |x: f64| -> Option<f64> {
        let fx = f(x);
        if fx.is_nan() { None } else { Some(fx) }
    };
    let mut xpre = xa;
    let mut xcur = xb;
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut fpre = eval(xpre)?;
    let mut fcur = eval(xcur)?;
    if fpre == 0.0 {
        return Some(xpre);
    }
    if fcur == 0.0 {
        return Some(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return None; // "f(a) and f(b) must have different signs"
    }
    for _ in 0..iter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.is_sign_negative() != fcur.is_sign_negative()) {
            xblk = xpre;
            fblk = fpre;
            scur = xcur - xpre;
            spre = scur;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = brent_fma(rtol, xcur.abs(), xtol) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Some(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * brent_fma(fblk, dblk, -(fpre * dpre)) / (dblk * dpre * (fblk - fpre))
            };
            let lim_a = spre.abs();
            let lim_b = brent_fma(sbis.abs(), 3.0, -delta);
            // C: MIN(a, b) is ((a) < (b) ? (a) : (b))
            if 2.0 * stry.abs() < (if lim_a < lim_b { lim_a } else { lim_b }) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = eval(xcur)?;
    }
    None // CONVERR, raised under disp=True
}

/// Why `calculate_irr` returned what it did. Only `SolverFailed` was logged
/// at warning level in Python; the rest were debug lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IrrStatus {
    Ok = 0,
    /// Too few flows, all on one date, or a non-finite flow.
    InvalidInput = 1,
    /// Nothing left after stripping zero flows, all zero, first flow not an
    /// investment, or no return at all.
    InvalidPattern = 2,
    /// Returns negligible against capital: effectively -100%, no root.
    TotalLoss = 3,
    /// NPV never changes sign on the scan grid: no IRR exists.
    NoRoot = 4,
    /// A sign change was found but no solver produced a usable root.
    SolverFailed = 5,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrrOutcome {
    pub rate: f64,
    pub status: IrrStatus,
    /// The flows the solver used were `flows[first..=last]`, divided by the
    /// largest magnitude when `scaled` — for the failure log.
    pub first: usize,
    pub last: usize,
    pub scaled: bool,
    /// Some NPV evaluation met a flow dated before the first (unsorted input).
    pub saw_negative_delta: bool,
}

/// `calculate_irr(dates, cash_flows)`. `ticks[i]` is `dates[i]` in microseconds
/// on any common origin; `kinds[i]` says how Python's `sum()` would add
/// `cash_flows[i]`. The `isinstance(cf, (int, float))` check stays in Python.
pub fn calculate_irr(ticks: &[i64], flows: &[f64], kinds: &[PyNum]) -> IrrOutcome {
    let outcome = |status, first, last, scaled, neg| IrrOutcome {
        rate: f64::NAN,
        status,
        first,
        last,
        scaled,
        saw_negative_delta: neg,
    };
    let n = flows.len();
    // 1. Basic input validation
    if ticks.len() < 2 || n < 2 || ticks.len() != n || kinds.len() != n {
        return outcome(IrrStatus::InvalidInput, 0, 0, false, false);
    }
    if ticks.iter().all(|&t| t == ticks[0]) {
        return outcome(IrrStatus::InvalidInput, 0, 0, false, false);
    }
    if flows.iter().any(|cf| !cf.is_finite()) {
        return outcome(IrrStatus::InvalidInput, 0, 0, false, false);
    }

    // 2. Strip leading and trailing zeros
    let mut first = 0;
    while first < n && flows[first].abs() < 1e-9 {
        first += 1;
    }
    let mut last = n - 1;
    while last > first && flows[last].abs() < 1e-9 {
        last -= 1;
    }
    if (last as i64) - (first as i64) < 1 {
        return outcome(IrrStatus::InvalidPattern, 0, 0, false, false);
    }
    let cash = &flows[first..=last];
    let kinds = &kinds[first..=last];
    let t0 = ticks[first];
    // (d - start).days floors, like timedelta.
    let days: Vec<i64> = ticks[first..=last]
        .iter()
        .map(|&t| (t - t0).div_euclid(DAY_TICKS))
        .collect();

    // 3. Cash flow pattern validation
    let mut first_non_zero: Option<f64> = None;
    let mut non_zero: Vec<(f64, PyNum)> = Vec::new();
    let mut max_abs_flow = 0.0_f64;
    for (i, &cf) in cash.iter().enumerate() {
        let abs_cf = cf.abs();
        if abs_cf > 1e-9 {
            non_zero.push((cf, kinds[i]));
            if abs_cf > max_abs_flow {
                max_abs_flow = abs_cf;
            }
            if first_non_zero.is_none() {
                first_non_zero = Some(cf);
            }
        }
    }
    let Some(first_non_zero) = first_non_zero else {
        return outcome(IrrStatus::InvalidPattern, first, last, false, false);
    };
    if first_non_zero >= -1e-9 {
        return outcome(IrrStatus::InvalidPattern, first, last, false, false);
    }
    if !non_zero.iter().any(|&(cf, _)| cf > 1e-9) {
        return outcome(IrrStatus::InvalidPattern, first, last, false, false);
    }
    let positives: Vec<_> = non_zero
        .iter()
        .copied()
        .filter(|&(cf, _)| cf > 0.0)
        .collect();
    let negatives: Vec<_> = non_zero
        .iter()
        .copied()
        .filter(|&(cf, _)| cf < 0.0)
        .collect();
    let sum_pos = py_sum(&positives).unwrap_or(0.0);
    let sum_neg = -py_sum(&negatives).unwrap_or(0.0);
    if sum_neg > 0.0 && sum_pos / sum_neg < 1e-6 {
        return outcome(IrrStatus::TotalLoss, first, last, false, false);
    }

    // Normalize: scaling the flows does not change the IRR, and keeps the
    // solver away from huge magnitudes (JPY, VND).
    let scaled = max_abs_flow > 1e6;
    let solver_flows: Vec<f64> = if scaled {
        cash.iter().map(|&cf| cf / max_abs_flow).collect()
    } else {
        cash.to_vec()
    };

    let mut saw_negative_delta = false;
    let mut f = |rate: f64| -> f64 {
        let r = npv(rate, &days, &solver_flows);
        if r.negative_delta_at.is_some() {
            saw_negative_delta = true;
        }
        r.value
    };

    // 4a. Newton (secant) from several starting points
    let mut irr_result = f64::NAN;
    for x0 in [0.1, 0.0, -0.2, 0.5, -0.5] {
        if let Some(res) = newton_secant(&mut f, x0, 1e-6, 50) {
            if res.is_finite() && res > -1.0 && res < 1000.0 {
                let npv_check = f(res);
                if np_isclose(npv_check, 0.0, 1e-5, 1e-2) {
                    irr_result = res;
                    break;
                }
            }
        }
    }

    // 4b. Scan the NPV curve to bracket a root, then Brent
    let mut found_bracket = false;
    if irr_result.is_nan() {
        let mut prev: Option<(f64, f64)> = None;
        for bits in SCAN_GRID_BITS {
            let r_scan = f64::from_bits(bits);
            let npv_val = f(r_scan);
            if let Some((prev_r, prev_npv)) = prev {
                if !npv_val.is_nan() && !prev_npv.is_nan() && npv_val * prev_npv < 0.0 {
                    found_bracket = true;
                    if let Some(res) = brentq(&mut f, prev_r, r_scan, 1e-6, 1e-6, 100) {
                        if res.is_finite() && res > -1.0 {
                            irr_result = res;
                            break;
                        }
                    }
                }
            }
            if !npv_val.is_nan() {
                prev = Some((r_scan, npv_val));
            }
        }
        if irr_result.is_nan() && !found_bracket {
            return outcome(IrrStatus::NoRoot, first, last, scaled, saw_negative_delta);
        }
    }

    // 5. Final validation
    if !(irr_result.is_finite() && irr_result > -1.0) {
        return outcome(
            IrrStatus::SolverFailed,
            first,
            last,
            scaled,
            saw_negative_delta,
        );
    }
    IrrOutcome {
        rate: irr_result,
        status: IrrStatus::Ok,
        first,
        last,
        scaled,
        saw_negative_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_annual_doubling_is_one_hundred_percent() {
        // -100 today, +200 in 365 days.
        let ticks = [0, 365 * DAY_TICKS];
        let out = calculate_irr(&ticks, &[-100.0, 200.0], &[PyNum::Float; 2]);
        assert_eq!(out.status, IrrStatus::Ok);
        assert!((out.rate - 1.0).abs() < 1e-6, "{}", out.rate);
    }

    #[test]
    fn a_return_first_is_not_an_investment() {
        let ticks = [0, 365 * DAY_TICKS];
        let out = calculate_irr(&ticks, &[100.0, -50.0], &[PyNum::Float; 2]);
        assert_eq!(out.status, IrrStatus::InvalidPattern);
        assert!(out.rate.is_nan());
    }

    #[test]
    fn npv_floors_partial_days_like_timedelta() {
        // 1.5 days apart counts as 1 day; -0.5 days as -1 (a negative delta).
        let ticks = [0, DAY_TICKS + DAY_TICKS / 2];
        let days: Vec<i64> = ticks.iter().map(|t| t.div_euclid(DAY_TICKS)).collect();
        assert_eq!(days, vec![0, 1]);
        assert_eq!((-DAY_TICKS / 2).div_euclid(DAY_TICKS), -1);
    }
}
