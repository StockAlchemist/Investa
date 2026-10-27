//! Monte Carlo intrinsic-value models: the vectorized NumPy in
//! `financial_ratios.run_monte_carlo_*`, evaluated per sample, plus the
//! summary statistics of `_generate_mc_stats`.
//!
//! The random draws stay NumPy's (the legacy global generator, seeded per
//! company): the Python shims draw exactly as before, in the same order, and
//! hand the samples over. Everything after the draw happens here.
//!
//! NumPy semantics that decide the bits, each measured against the installed
//! NumPy 2.0:
//! * `np.clip(x, lo, hi)` is `MIN(MAX(x, lo), hi)` with NaN passing through,
//!   and `hi` winning when `lo > hi` ([`np_clip`]).
//! * `array ** scalar` takes fast paths by value — 2 squares, -1 takes the
//!   reciprocal, 0.5 the square root — and Apple's `pow` is not correctly
//!   rounded for them, so they differ ([`np_pow_scalar`]). Any other
//!   exponent, and `array ** array`, is the C library's `pow`.
//! * Row sums (`np.sum(axis=1)`) are pairwise, as for 1-D arrays.
//!
//! `np.convolve` (the histogram smoothing) goes through BLAS `ddot` and stays
//! in NumPy; this module returns the raw counts.

use crate::stats::{np_maximum, np_sum};

/// The C library's `pow`, which Python's `float ** float` and NumPy's
/// non-fast-path powers call. Never inlined and the exponent hidden from the
/// optimizer, so LLVM cannot fold a constant `pow(x, 2.0)` into `x * x`.
#[inline(never)]
pub fn libm_pow(x: f64, y: f64) -> f64 {
    x.powf(std::hint::black_box(y))
}

/// `array ** e` for a scalar exponent, with NumPy's value-based fast paths.
#[inline]
pub fn np_pow_scalar(x: f64, e: f64) -> f64 {
    if e == 2.0 {
        x * x
    } else if e == -1.0 {
        1.0 / x
    } else if e == 0.5 {
        x.sqrt()
    } else {
        libm_pow(x, e)
    }
}

/// `np.clip(x, lo, hi)` for floats: `_NPY_MIN(_NPY_MAX(x, lo), hi)`.
#[inline]
pub fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    let m = if x.is_nan() || x > lo { x } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

/// Clip bounds for one sampled input.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

/// The faded-growth projection with mid-year discounting and a Gordon
/// terminal value, shared by DCF, D-CFO, D-NI and DDM. Returns
/// (sum of discounted flows, discounted terminal value) for one sample.
///
/// With `projection_years <= 1` NumPy's fade factor is the one-element
/// `[0]`, so the growth row has a single column; with `projection_years <= 0`
/// there are no years to discount and the flow sum is empty.
pub fn faded_projection(
    base: f64,
    growth: f64,
    discount: f64,
    projection_years: i64,
    terminal_growth: f64,
) -> (f64, f64) {
    let p = projection_years;
    let ncols = if p > 1 { p as usize } else { 1 };
    let mut pv = Vec::with_capacity(ncols);
    let mut cum = 1.0_f64;
    let mut last = 0.0_f64;
    let one_plus_d = 1.0 + discount;
    for k in 0..ncols {
        let fade = if p > 1 {
            (k as f64) / ((p - 1) as f64)
        } else {
            0.0
        };
        let yearly = growth - (growth - terminal_growth) * fade;
        cum = if k == 0 {
            1.0 + yearly
        } else {
            cum * (1.0 + yearly)
        };
        let flow = base * cum;
        last = flow;
        if p >= 1 {
            let year = (k as i64 + 1) as f64;
            pv.push(flow / libm_pow(one_plus_d, year - 0.5));
        }
    }
    let sum_pv = np_sum(&pv);
    let safe_discount = np_maximum(discount, terminal_growth + 0.01);
    let terminal_value = (last * (1.0 + terminal_growth)) / (safe_discount - terminal_growth);
    let pv_terminal = terminal_value / np_pow_scalar(one_plus_d, p as f64);
    (sum_pv, pv_terminal)
}

/// Parameters of a discounted model: the clip on sampled growth, on sampled
/// discount rates, and the projection.
#[derive(Debug, Clone, Copy)]
pub struct Discounted {
    pub growth: Bounds,
    pub discount: Bounds,
    pub projection_years: i64,
    pub terminal_growth: f64,
}

fn discounted(
    base: f64,
    growth: &[f64],
    discount: &[f64],
    m: &Discounted,
    finish: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    growth
        .iter()
        .zip(discount)
        .map(|(&g, &d)| {
            let g = np_clip(g, m.growth.lo, m.growth.hi);
            let d = np_clip(d, m.discount.lo, m.discount.hi);
            let (sum_pv, pv_tv) =
                faded_projection(base, g, d, m.projection_years, m.terminal_growth);
            finish(sum_pv, pv_tv)
        })
        .collect()
}

/// `run_monte_carlo_dcf`: equity per share from projected free cash flow.
pub fn dcf(
    growth: &[f64],
    discount: &[f64],
    m: &Discounted,
    base_fcf: f64,
    cash: f64,
    debt: f64,
    shares: f64,
) -> Vec<f64> {
    discounted(base_fcf, growth, discount, m, |sum_pv, pv_tv| {
        let enterprise = sum_pv + pv_tv;
        (enterprise + cash - debt) / shares
    })
}

/// `run_monte_carlo_dcfo` (per share already) and `_dni`/`_ddm`
/// (`net_cash_per_share` 0): flows plus terminal value plus net cash.
pub fn discounted_per_share(
    growth: &[f64],
    discount: &[f64],
    m: &Discounted,
    base_per_share: f64,
    net_cash_per_share: Option<f64>,
) -> Vec<f64> {
    discounted(
        base_per_share,
        growth,
        discount,
        m,
        |sum_pv, pv_tv| match net_cash_per_share {
            Some(n) => sum_pv + pv_tv + n,
            None => sum_pv + pv_tv,
        },
    )
}

/// `run_monte_carlo_mean_pe` / `_mean_pb` / `_mean_ps`: a clipped base times
/// a clipped multiple.
pub fn base_times_multiple(
    base: &[f64],
    multiple: &[f64],
    base_bounds: Bounds,
    multiple_bounds: Bounds,
) -> Vec<f64> {
    base.iter()
        .zip(multiple)
        .map(|(&b, &x)| {
            np_clip(b, base_bounds.lo, base_bounds.hi)
                * np_clip(x, multiple_bounds.lo, multiple_bounds.hi)
        })
        .collect()
}

/// `run_monte_carlo_peg`: EPS x clip(PEG x (growth + yield), 5, 35).
pub fn peg(
    eps: &[f64],
    growth: &[f64],
    peg: &[f64],
    eps_hi: f64,
    dividend_yield_pct: f64,
) -> Vec<f64> {
    (0..eps.len())
        .map(|i| {
            let e = np_clip(eps[i], 0.01, eps_hi);
            let g = np_clip(growth[i], 2.0, 40.0);
            let p = np_clip(peg[i], 0.5, 2.5);
            e * np_clip(p * (g + dividend_yield_pct), 5.0, 35.0)
        })
        .collect()
}

/// `run_monte_carlo_psg`: sales per share x clip(growth/10 x margin x PSG x 1.5).
pub fn psg(sps: &[f64], growth: &[f64], margin: &[f64], target: &[f64], sps_hi: f64) -> Vec<f64> {
    (0..sps.len())
        .map(|i| {
            let s = np_clip(sps[i], 0.01, sps_hi);
            let g = np_clip(growth[i], 5.0, 50.0);
            let gm = np_clip(margin[i], 0.20, 0.95);
            let t = np_clip(target[i], 0.5, 2.0);
            s * np_clip((g / 10.0) * gm * t * 1.5, 0.5, 15.0)
        })
        .collect()
}

/// `run_monte_carlo_graham`: EPS x (8.5 + 2g) x 4.4 / bond yield.
pub fn graham(growth: &[f64], bond_yield: &[f64], eps: f64) -> Vec<f64> {
    growth
        .iter()
        .zip(bond_yield)
        .map(|(&g, &y)| {
            let g = np_clip(g, 0.0, 25.0);
            let y = np_clip(y, 0.5, 10.0);
            (eps * (8.5 + 2.0 * g) * 4.4) / y
        })
        .collect()
}

/// `run_monte_carlo_epv`: max(0.01, (NOPAT / discount + cash - debt) / shares).
#[allow(clippy::too_many_arguments)]
pub fn epv(
    ebit: &[f64],
    discount: &[f64],
    ebit_hi: f64,
    discount_bounds: Bounds,
    tax_rate: f64,
    cash: f64,
    debt: f64,
    shares: f64,
) -> Vec<f64> {
    let keep = 1.0 - tax_rate;
    ebit.iter()
        .zip(discount)
        .map(|(&e, &d)| {
            let e = np_clip(e, 0.01, ebit_hi);
            let d = np_clip(d, discount_bounds.lo, discount_bounds.hi);
            let nopat = e * keep;
            let enterprise = nopat / d;
            np_maximum(0.01, (enterprise + cash - debt) / shares)
        })
        .collect()
}

/// `run_monte_carlo_lynch`: EPS x clip(growth + yield, 5, 25).
pub fn lynch(eps: &[f64], growth: &[f64], eps_hi: f64, dividend_yield_pct: f64) -> Vec<f64> {
    eps.iter()
        .zip(growth)
        .map(|(&e, &g)| {
            let e = np_clip(e, 0.01, eps_hi);
            let g = np_clip(g, 2.0, 35.0);
            e * np_clip(g + dividend_yield_pct, 5.0, 25.0)
        })
        .collect()
}

/// What `_generate_mc_stats` reports, before the histogram is smoothed.
#[derive(Debug, Clone, PartialEq)]
pub struct McStats {
    pub bear: f64,
    pub conservative: f64,
    pub base: f64,
    pub bull: f64,
    pub std_dev: f64,
    /// 40 bins over the valid samples.
    pub counts: Vec<i64>,
    pub midpoints: Vec<f64>,
}

/// `np.percentile(sorted_values, q)` with the default linear method, on an
/// ascending, NaN-free, non-empty slice.
pub fn np_percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let quantile = q / 100.0;
    let virtual_index = (n as i64 - 1) as f64 * quantile;
    let (prev, next) = if virtual_index >= (n as f64) - 1.0 {
        (n - 1, n - 1)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let p = virtual_index.floor() as usize;
        (p, p + 1)
    };
    let gamma = virtual_index - prev as f64;
    let (a, b) = (sorted[prev], sorted[next]);
    // numpy._lerp: two-sided, so the endpoints are exact
    let diff_b_a = b - a;
    if gamma >= 0.5 {
        b - diff_b_a * (1.0 - gamma)
    } else {
        a + diff_b_a * gamma
    }
}

/// `np.histogram(values, bins)` for finite values: counts and bin edges.
pub fn np_histogram(values: &[f64], bins: usize) -> (Vec<i64>, Vec<f64>) {
    let mut first = values
        .iter()
        .copied()
        .fold(f64::INFINITY, crate::stats::np_minimum);
    let mut last = values.iter().copied().fold(f64::NEG_INFINITY, np_maximum);
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    // np.linspace(first, last, bins + 1): i * step + first, the last pinned.
    let step = (last - first) / bins as f64;
    let mut edges: Vec<f64> = (0..=bins).map(|i| (i as f64) * step + first).collect();
    edges[bins] = last;

    let norm_denom = last - first;
    let mut counts = vec![0_i64; bins];
    for &x in values {
        let f = ((x - first) / norm_denom) * bins as f64;
        let mut idx = f as i64;
        if idx == bins as i64 {
            idx -= 1;
        }
        if x < edges[idx as usize] {
            idx -= 1;
        }
        if x >= edges[idx as usize + 1] && idx != bins as i64 - 1 {
            idx += 1;
        }
        counts[idx as usize] += 1;
    }
    (counts, edges)
}

/// `_generate_mc_stats` without the smoothing: None when fewer than 50
/// samples are finite and positive.
pub fn mc_stats(intrinsic_values: &[f64]) -> Option<McStats> {
    let valid: Vec<f64> = intrinsic_values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if valid.len() < 50 {
        return None;
    }
    let (counts, edges) = np_histogram(&valid, 40);
    let midpoints = (0..40).map(|i| (edges[i] + edges[i + 1]) / 2.0).collect();

    let mut sorted = valid.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = valid.len() as f64;
    let mean = np_sum(&valid) / n;
    let sq: Vec<f64> = valid
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    Some(McStats {
        bear: np_percentile_sorted(&sorted, 10.0),
        conservative: np_percentile_sorted(&sorted, 25.0),
        base: np_percentile_sorted(&sorted, 50.0),
        bull: np_percentile_sorted(&sorted, 90.0),
        std_dev: (np_sum(&sq) / n).sqrt(),
        counts,
        midpoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_matches_numpy_rules() {
        assert_eq!(np_clip(5.0, 0.0, 1.0), 1.0);
        assert_eq!(np_clip(-5.0, 0.0, 1.0), 0.0);
        assert!(np_clip(f64::NAN, 0.0, 1.0).is_nan());
        // lo > hi: hi wins, as in NumPy
        assert_eq!(np_clip(0.5, 2.0, 1.0), 1.0);
    }

    #[test]
    fn percentile_endpoints_are_exact() {
        let v: Vec<f64> = (1..=101).map(f64::from).collect();
        assert_eq!(np_percentile_sorted(&v, 50.0), 51.0);
        assert_eq!(np_percentile_sorted(&v, 10.0), 11.0);
        assert_eq!(np_percentile_sorted(&v, 100.0), 101.0);
    }

    #[test]
    fn histogram_puts_the_maximum_in_the_last_bin() {
        let v: Vec<f64> = (0..100).map(f64::from).collect();
        let (counts, edges) = np_histogram(&v, 40);
        assert_eq!(counts.iter().sum::<i64>(), 100);
        assert_eq!(edges[40], 99.0);
        assert!(counts[39] >= 1);
    }
}
