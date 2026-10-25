//! Portfolio-value projections: the arithmetic of `projections.compute_projection`
//! (a lognormal cone with drift shrinkage and parameter uncertainty), the whole
//! of `walk_forward_errors`, and `summarize_errors`' statistics.
//!
//! The walk-forward is the one place this module changes the cost: the
//! original refits the projection on a growing window at every monthly anchor,
//! re-deriving the window's returns through pandas each time, which is
//! quadratic in the history. Here the window's log returns are a prefix of one
//! precomputed array; the moments are still recomputed per window from scratch
//! (pandas' pairwise sums), so every number is the original's.
//!
//! Semantics measured on the installed stack: `np.log1p`, `math.log1p` and
//! `math.erf` are the C library's; Python's `round` ties to even, so ten years
//! is `round(3652.5)` = 3652 days.

use crate::stats::{np_mean, np_sum, pd_mean, pd_std};
use crate::valuation::np_median;

/// `_BANDS`: standard-normal quantiles, in the original's key order.
pub const BANDS: [(&str, f64); 4] = [
    ("p10", -1.2815515594),
    ("p25", -0.6744897502),
    ("p75", 0.6744897502),
    ("p90", 1.2815515594),
];
const DRIFT_SHRINK_K: f64 = 10.0;
const DAYS_PER_YEAR: f64 = 365.25;
const DAY_NS: i64 = 86_400_000_000_000;

unsafe extern "C" {
    /// The C library's erf, which `math.erf` calls.
    fn erf(x: f64) -> f64;
}

/// One horizon of the cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub years: f64,
    pub median_value: f64,
    pub median_return_pct: f64,
    pub expected_value: f64,
    /// p10, p25, p75, p90
    pub bands: [f64; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    pub annual_return_pct: f64,
    pub annual_volatility_pct: f64,
    pub points: Vec<Point>,
}

/// `compute_projection` once the returns are prepared (clipped, NaN-free, at
/// least 20 of them): fit the annual log drift and volatility, shrink the
/// drift toward the benchmark, and draw the cone. None where the original
/// returns `{"available": False}` for a non-finite fit.
pub fn project(
    returns: &[f64],
    periods_per_year: f64,
    current_value: f64,
    horizons: &[f64],
    benchmark_log_return: Option<f64>,
) -> Option<Projection> {
    let log_ret: Vec<f64> = returns.iter().map(|r| r.ln_1p()).collect();
    project_log(
        &log_ret,
        periods_per_year,
        current_value,
        horizons,
        benchmark_log_return,
    )
}

fn project_log(
    log_ret: &[f64],
    periods_per_year: f64,
    current_value: f64,
    horizons: &[f64],
    benchmark_log_return: Option<f64>,
) -> Option<Projection> {
    let n_years = log_ret.len() as f64 / periods_per_year;
    let mu_raw = pd_mean(log_ret) * periods_per_year;
    let sigma_log = pd_std(log_ret, 1) * periods_per_year.sqrt();
    if !mu_raw.is_finite() || !sigma_log.is_finite() {
        return None;
    }
    let mut mu_log = mu_raw;
    if let Some(b) = benchmark_log_return.filter(|b| b.is_finite()) {
        let w_data = n_years / (n_years + DRIFT_SHRINK_K);
        mu_log = w_data * mu_raw + (1.0 - w_data) * b;
    }
    let points = horizons
        .iter()
        .map(|&t| {
            let drift = mu_log * t;
            let spread = if n_years > 0.0 {
                sigma_log * (t * (1.0 + t / n_years)).sqrt()
            } else {
                sigma_log * t.sqrt()
            };
            let mut bands = [0.0; 4];
            for (slot, &(_, z)) in bands.iter_mut().zip(BANDS.iter()) {
                *slot = current_value * (drift + z * spread).exp();
            }
            Point {
                years: t,
                median_value: current_value * drift.exp(),
                median_return_pct: (drift.exp() - 1.0) * 100.0,
                expected_value: current_value * (drift + 0.5 * sigma_log * sigma_log * t).exp(),
                bands,
            }
        })
        .collect();
    Some(Projection {
        annual_return_pct: (mu_log.exp() - 1.0) * 100.0,
        annual_volatility_pct: sigma_log * 100.0,
        points,
    })
}

/// `infer_periods_per_year(index)` for a DatetimeIndex given as ns ticks:
/// observations over `(last - first).days / 365.25`, else the default.
pub fn infer_periods_per_year(ticks: &[i64], default: f64) -> f64 {
    if ticks.len() > 1 {
        let days = (ticks[ticks.len() - 1] - ticks[0]).div_euclid(DAY_NS);
        let span_years = days as f64 / DAYS_PER_YEAR;
        if span_years > 0.0 {
            return ticks.len() as f64 / span_years;
        }
    }
    default
}

/// `_future_position`: the observation `years` after `index[pos]`, tolerating
/// a gap of up to ten days, or the positional equivalent without dates.
fn future_position(
    ticks: Option<&[i64]>,
    len: usize,
    pos: usize,
    years: f64,
    ppy: f64,
) -> Option<usize> {
    match ticks {
        Some(t) => {
            let days = (DAYS_PER_YEAR * years).round_ties_even() as i64;
            let target = t[pos] + days * DAY_NS;
            let nxt = t.partition_point(|&x| x < target);
            if nxt >= len {
                return None;
            }
            if (t[nxt] - target).div_euclid(DAY_NS) > 10 {
                return None;
            }
            Some(nxt)
        }
        None => {
            let nxt = pos + (years * ppy).round_ties_even() as usize;
            (nxt < len).then_some(nxt)
        }
    }
}

/// `_benchmark_prior` over the first `t` benchmark values (NaN = missing).
fn benchmark_prior(bench: &[f64], t: usize, ppy: f64) -> Option<f64> {
    let b: Vec<f64> = bench[..t].iter().copied().filter(|v| !v.is_nan()).collect();
    if (b.len() as f64) < ppy || ppy <= 0.0 {
        return None;
    }
    let (first, last) = (b[0], b[b.len() - 1]);
    if first <= 0.0 || last <= 0.0 {
        return None;
    }
    Some((last / first).ln() / (b.len() as f64 / ppy))
}

/// One `walk_forward_errors` row: (years, anchor position, z, actual, median).
pub type ErrorRow = (f64, usize, f64, f64, f64);

/// `walk_forward_errors` on a positive, NaN-free wealth index. `ticks` are
/// its dates (ns) when it has a DatetimeIndex; `bench` is the benchmark
/// already reindexed and forward-filled onto it (NaN where missing).
pub fn walk_forward_errors(
    wealth: &[f64],
    ticks: Option<&[i64]>,
    bench: Option<&[f64]>,
    horizons_sorted: &[f64],
    min_history_years: f64,
    step: Option<usize>,
) -> Vec<ErrorRow> {
    let n = wealth.len();
    let mut rows = Vec::new();
    if n < 30 || horizons_sorted.is_empty() {
        return rows;
    }
    let ppy = match ticks {
        Some(t) => infer_periods_per_year(t, 252.0),
        None => 252.0,
    };
    let step = step.unwrap_or_else(|| ((ppy / 12.0).round_ties_even() as usize).max(1));
    let start = (ppy * min_history_years).round_ties_even() as usize;
    if start >= n {
        return rows;
    }
    // Every window's returns are a prefix of these: r_i = w_i / w_{i-1} - 1,
    // clipped like compute_projection's, then log1p.
    let log_ret: Vec<f64> = wealth
        .windows(2)
        .map(|w| {
            let r = w[1] / w[0] - 1.0;
            // pandas clip(-0.90, 1.0); r is never NaN for a positive series
            let r = r.clamp(-0.90, 1.0);
            r.ln_1p()
        })
        .collect();

    let mut t = start;
    while t < n {
        let anchor = t - 1;
        if future_position(ticks, n, anchor, horizons_sorted[0], ppy).is_none() {
            break;
        }
        let v0 = wealth[t - 1];
        let prior = bench.and_then(|b| benchmark_prior(b, t, ppy));
        // compute_projection(window = wealth[:t]) needs 30 values, and so has
        // the 20 returns it also requires.
        if t >= 30 && v0 > 0.0 {
            let window_ppy = match ticks {
                Some(tk) => infer_periods_per_year(&tk[1..t], 252.0),
                None => 252.0,
            };
            if let Some(proj) =
                project_log(&log_ret[..t - 1], window_ppy, v0, horizons_sorted, prior)
            {
                for p in &proj.points {
                    let Some(fut) = future_position(ticks, n, anchor, p.years, ppy) else {
                        continue;
                    };
                    let mu = (p.median_value / v0).ln();
                    let sd = ((p.bands[3] / v0).ln() - mu) / BANDS[3].1;
                    if !mu.is_finite() || !sd.is_finite() || sd <= 0.0 {
                        continue;
                    }
                    let actual = (wealth[fut] / v0).ln();
                    rows.push((p.years, anchor, (actual - mu) / sd, actual, mu));
                }
            }
        }
        t += step;
    }
    rows
}

/// `summarize_errors`' statistics for one horizon's errors (at least 3):
/// (std_z, in_band_pct, below_p10_pct, above_p90_pct, mean_u,
/// median_actual_return_pct, median_projected_return_pct).
pub fn summarize(z: &[f64], actual_log: &[f64], median_log: &[f64]) -> [f64; 7] {
    let u: Vec<f64> = z
        .iter()
        .map(|&v| 0.5 * (1.0 + unsafe { erf(v / 2.0_f64.sqrt()) }))
        .collect();
    let n = z.len() as f64;
    let mean_z = np_sum(z) / n;
    let sq: Vec<f64> = z.iter().map(|&v| (v - mean_z) * (v - mean_z)).collect();
    let std_z = (np_sum(&sq) / (n - 1.0)).sqrt();
    let share = |pred: &dyn Fn(f64) -> bool| -> f64 {
        let flags: Vec<f64> = u.iter().map(|&x| if pred(x) { 1.0 } else { 0.0 }).collect();
        np_sum(&flags) / n * 100.0
    };
    [
        std_z,
        share(&|x| x > 0.1 && x < 0.9),
        share(&|x| x < 0.1),
        share(&|x| x > 0.9),
        np_mean(&u),
        (np_median(actual_log).exp() - 1.0) * 100.0,
        (np_median(median_log).exp() - 1.0) * 100.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_years_round_to_even() {
        assert_eq!((DAYS_PER_YEAR * 10.0).round_ties_even(), 3652.0);
        assert_eq!((DAYS_PER_YEAR * 3.0).round_ties_even(), 1096.0);
    }

    #[test]
    fn erf_is_the_c_library() {
        assert_eq!(unsafe { erf(0.0) }, 0.0);
        assert!((unsafe { erf(1.0) } - 0.842_700_792_949_714_9).abs() < 1e-15);
    }

    #[test]
    fn flat_history_has_no_spread() {
        let r = vec![0.001; 40];
        let p = project(&r, 252.0, 100.0, &[1.0], None).unwrap();
        assert!(p.annual_volatility_pct.abs() < 1e-9);
        assert!((p.points[0].median_value - p.points[0].bands[0]).abs() < 1e-6);
    }
}
