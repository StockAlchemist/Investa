//! NumPy/pandas reductions reproduced exactly, and `risk_metrics` built on them.
//!
//! Exactness rests on two facts measured against the installed stack
//! (NumPy 2.0 and 2.5, pandas 2.3, no bottleneck/numexpr):
//!
//! * `np.add.reduce` over a contiguous float64 array starts from +0.0 and adds
//!   NumPy's pairwise sum of each buffer in turn ([`np_sum`]): 8192 elements in
//!   NumPy 2.0, the whole array in 2.5, measured at import;
//!   pandas' `mean`/`var`/`std` call it on a copy with NaN replaced by 0, then
//!   divide by the non-NaN count.
//! * `np.minimum` returns -0.0 for a {0.0, -0.0} pair in either order.
//!
//! `np.cov`'s BLAS dot product cannot be reproduced (Accelerate on macOS,
//! OpenBLAS on Linux: the summation order is theirs), so `calculate_beta`
//! keeps pandas' covariance and only its variance comes from here.

use std::sync::atomic::{AtomicUsize, Ordering};

/// How many elements `np.add.reduce` hands its inner loop at a time over a
/// contiguous float64 array. NumPy 2.0 buffers 8192; NumPy 2.5 does not buffer
/// a contiguous array at all (`usize::MAX`: one pairwise sum over the whole of
/// it). [`calibrate_reduce_buffer`] measures the installed NumPy at import.
static REDUCE_BUFFER: AtomicUsize = AtomicUsize::new(8192);

/// `np.add.reduce` over a contiguous float64 array — what `ndarray.sum()`,
/// `ndarray.mean()` and pandas' reductions add with. The reduction starts
/// from the identity +0.0 (so negative zeros sum to +0.0) and adds the
/// [`pairwise_sum`] of each buffer in turn.
pub fn np_sum(a: &[f64]) -> f64 {
    np_sum_buffered(a, REDUCE_BUFFER.load(Ordering::Relaxed))
}

fn np_sum_buffered(a: &[f64], buffer: usize) -> f64 {
    let mut acc = 0.0_f64;
    for block in a.chunks(buffer) {
        acc += pairwise_sum(block);
    }
    acc
}

/// Matches [`np_sum`] to the installed NumPy: sums a fixed 20,000-element
/// probe (past one 8192 buffer, values spread over seven decades so the
/// grouping shows in the last bits) with `np.add.reduce`, and keeps whichever
/// buffering reproduces it. Returns the buffer chosen. Should neither match (a
/// NumPy that reduces some third way), the 8192 default stays and the parity
/// tests will say so.
pub fn calibrate_reduce_buffer(py: pyo3::Python<'_>) -> pyo3::PyResult<usize> {
    use pyo3::prelude::*;
    let probe: Vec<f64> = (0..20_000_i32)
        .map(|i| ((i as f64) * 0.618_033_988_749_895).sin() * 10f64.powi(i % 7))
        .collect();
    let np = py.import("numpy")?;
    let array = np.call_method1("array", (probe.clone(),))?;
    let numpy_sum: f64 = np
        .getattr("add")?
        .call_method1("reduce", (array,))?
        .extract()?;
    for buffer in [usize::MAX, 8192] {
        if np_sum_buffered(&probe, buffer).to_bits() == numpy_sum.to_bits() {
            REDUCE_BUFFER.store(buffer, Ordering::Relaxed);
            return Ok(buffer);
        }
    }
    Ok(REDUCE_BUFFER.load(Ordering::Relaxed))
}

/// NumPy's `pairwise_sum` inner loop for one buffer of float64.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.mean` of a non-empty array: `np_sum / n`. NaN for an empty one.
pub fn np_mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    np_sum(a) / a.len() as f64
}

/// `np.minimum` on two floats: NaN propagates, and -0.0 beats 0.0.
#[inline]
pub fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { a } else { b }
    } else if a < b {
        a
    } else {
        b
    }
}

/// `np.maximum` on two floats: NaN propagates, and 0.0 beats -0.0.
#[inline]
pub fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() { b } else { a }
    } else if a > b {
        a
    } else {
        b
    }
}

/// The values with NaN replaced by 0, and how many were not NaN — pandas'
/// `_get_values(..., fill_value=0)` and `_get_counts`.
fn filled(a: &[f64]) -> (Vec<f64>, usize) {
    let mut count = 0;
    let v = a
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                count += 1;
                x
            }
        })
        .collect();
    (v, count)
}

/// `Series.mean()` (pandas `nanmean`, skipna): NaN when nothing is left.
pub fn pd_mean(a: &[f64]) -> f64 {
    let (v, count) = filled(a);
    if count == 0 {
        return f64::NAN;
    }
    np_sum(&v) / count as f64
}

/// `Series.var(ddof=ddof)` (pandas `nanvar`): the two-pass algorithm on the
/// NaN-filled copy. NaN when the count is at most `ddof`.
pub fn pd_var(a: &[f64], ddof: usize) -> f64 {
    let (v, count) = filled(a);
    if count <= ddof {
        return f64::NAN;
    }
    let count_f = count as f64;
    let avg = np_sum(&v) / count_f;
    let sqr: Vec<f64> = a
        .iter()
        .zip(&v)
        .map(|(&orig, &x)| {
            if orig.is_nan() {
                0.0
            } else {
                let d = avg - x;
                d * d // `** 2` takes NumPy's square fast path
            }
        })
        .collect();
    np_sum(&sqr) / (count_f - ddof as f64)
}

/// `Series.std(ddof=ddof)`: the square root of [`pd_var`].
pub fn pd_std(a: &[f64], ddof: usize) -> f64 {
    pd_var(a, ddof).sqrt()
}

/// `Series.min()` (pandas `nanmin`): NaN when every value is NaN.
pub fn pd_min(a: &[f64]) -> f64 {
    let mut acc: Option<f64> = None;
    for &x in a {
        let x = if x.is_nan() { f64::INFINITY } else { x };
        acc = Some(match acc {
            None => x,
            Some(m) => np_minimum(m, x),
        });
    }
    match acc {
        Some(m) if a.iter().any(|x| !x.is_nan()) => m,
        _ => f64::NAN,
    }
}

/// `calculate_drawdown_series`: `(s - s.cummax()) / s.cummax()`, with pandas'
/// skipna cummax (NaN counts as -inf in the running max and stays NaN out).
pub fn drawdown_series(values: &[f64]) -> Vec<f64> {
    let mut running = f64::NEG_INFINITY;
    let mut started = false;
    values
        .iter()
        .map(|&x| {
            if x.is_nan() {
                return f64::NAN;
            }
            running = if started { np_maximum(running, x) } else { x };
            started = true;
            (x - running) / running
        })
        .collect()
}

/// `calculate_max_drawdown`: the drawdown minimum, 0.0 for an empty or
/// all-NaN series.
pub fn max_drawdown(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mdd = pd_min(&drawdown_series(values));
    if mdd.is_nan() { 0.0 } else { mdd }
}

/// `calculate_volatility`: annualized sample standard deviation.
pub fn volatility(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    pd_std(returns, 1) * periods_per_year.sqrt()
}

/// `calculate_sharpe_ratio`.
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let rf_periodic = risk_free_rate / periods_per_year;
    let excess: Vec<f64> = returns.iter().map(|&r| r - rf_periodic).collect();
    let mean_excess_return = pd_mean(&excess);
    let std_dev_returns = pd_std(returns, 1);
    if std_dev_returns == 0.0 {
        return 0.0;
    }
    let sharpe_daily = mean_excess_return / std_dev_returns;
    sharpe_daily * periods_per_year.sqrt()
}

/// `calculate_sortino_ratio`. Infinite when nothing ever fell short of the
/// target and the mean excess return is positive.
pub fn sortino_ratio(
    returns: &[f64],
    risk_free_rate: f64,
    periods_per_year: f64,
    target_return: f64,
) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let rf_periodic = risk_free_rate / periods_per_year;
    let excess: Vec<f64> = returns.iter().map(|&r| r - rf_periodic).collect();
    let mean_excess_return = pd_mean(&excess);

    let no_downside = !returns.iter().any(|&r| r < target_return);
    let fallback = if mean_excess_return > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    if no_downside {
        return fallback;
    }
    // underperformance[underperformance > 0] = 0, then ** 2
    let squared: Vec<f64> = returns
        .iter()
        .map(|&r| {
            let u = r - target_return;
            let u = if u > 0.0 { 0.0 } else { u };
            u * u
        })
        .collect();
    let downside_deviation = pd_mean(&squared).sqrt();
    if downside_deviation == 0.0 {
        return fallback;
    }
    let sortino_daily = mean_excess_return / downside_deviation;
    sortino_daily * periods_per_year.sqrt()
}

/// `pct_change(fill_method=None).replace([inf, -inf], nan)`: NaN first, and
/// wherever the ratio is undefined or infinite.
pub fn pct_change(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let r = if i == 0 {
            f64::NAN
        } else {
            values[i] / values[i - 1] - 1.0
        };
        out.push(if r.is_infinite() { f64::NAN } else { r });
    }
    out
}

/// `Series.clip(lower, upper)` on NaN-free values.
#[inline]
pub fn clip(x: f64, lower: f64, upper: f64) -> f64 {
    if x < lower {
        lower
    } else if x > upper {
        upper
    } else {
        x
    }
}

/// One benchmark's row of `calculate_benchmark_scoreboard`, from aligned
/// (port, bench) returns: (alpha, beta, r2, tracking_error, information_ratio),
/// population moments, annualized with `ppy`.
pub fn scoreboard_stats(rp: &[f64], rb: &[f64], ppy: f64) -> (f64, f64, f64, f64, f64) {
    let sqrt_ppy = ppy.sqrt();
    let (mp, mb) = (np_mean(rp), np_mean(rb));
    let cov = np_mean(
        &rp.iter()
            .zip(rb)
            .map(|(&p, &b)| (p - mp) * (b - mb))
            .collect::<Vec<_>>(),
    );
    let var_b = np_mean(&rb.iter().map(|&b| (b - mb) * (b - mb)).collect::<Vec<_>>());
    let var_p = np_mean(&rp.iter().map(|&p| (p - mp) * (p - mp)).collect::<Vec<_>>());
    let beta = if var_b > 0.0 { cov / var_b } else { 0.0 };
    let alpha = (mp - beta * mb) * ppy * 100.0;
    let corr = if var_p > 0.0 && var_b > 0.0 {
        cov / (var_p * var_b).sqrt()
    } else {
        0.0
    };
    let diffs: Vec<f64> = rp.iter().zip(rb).map(|(&p, &b)| p - b).collect();
    let m_diff = np_mean(&diffs);
    let te_daily = np_mean(
        &diffs
            .iter()
            .map(|&d| (d - m_diff) * (d - m_diff))
            .collect::<Vec<_>>(),
    )
    .sqrt();
    let te = te_daily * sqrt_ppy * 100.0;
    let ir = if te_daily > 0.0 {
        (m_diff * ppy) / (te_daily * sqrt_ppy)
    } else {
        0.0
    };
    (alpha, beta, corr * corr, te, ir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_zeros_sum_to_positive_zero() {
        assert!(!np_sum(&[-0.0; 9]).is_sign_negative());
        assert!(!np_sum(&[-0.0]).is_sign_negative());
    }

    #[test]
    fn pairwise_sum_small_and_large() {
        assert_eq!(np_sum(&[]), 0.0);
        assert_eq!(np_sum(&[1.0, 2.0, 3.0]), 6.0);
        let v: Vec<f64> = (1..=1000).map(f64::from).collect();
        assert_eq!(np_sum(&v), 500_500.0);
    }

    #[test]
    fn signed_zero_minimum() {
        assert!(np_minimum(0.0, -0.0).is_sign_negative());
        assert!(np_minimum(-0.0, 0.0).is_sign_negative());
        assert!(np_minimum(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn variance_needs_more_than_ddof_values() {
        assert!(pd_var(&[1.0], 1).is_nan());
        assert!(pd_var(&[f64::NAN, 2.0], 1).is_nan());
        assert_eq!(pd_var(&[1.0, 3.0], 1), 2.0);
    }
}
