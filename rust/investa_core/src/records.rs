//! The aggregations behind three reader-facing views:
//! * `valuation_history.bands`: a company's multiples against its own record
//!   (median, quartiles, range, and where today sits).
//! * `strategies.evaluate_trend_signal`: the moving-average market signal's
//!   averages, provisional average, flip price and history.
//! * `track_record.stress_response`: peak-to-trough falls and recoveries in
//!   each downturn window.
//!
//! Loading the filings and prices, and building the result dictionaries, stay
//! in Python. Python rules reproduced:
//! * `max(d, key=...)` / `min` keep the first of equal values.
//! * `sum(list)` follows the items' types ([`py_sum`]). Iterating a pandas
//!   Series yields Python floats, so `list(series)` sums compensated.
//! * `Series.iloc[-0:]` is the whole series, which is what a one-month
//!   average's `iloc[-(n - 1):]` becomes.

use crate::mc::np_percentile_sorted;
use crate::stats::{np_maximum, np_minimum, pd_mean};
use crate::util::{PyNum, py_sum};
use crate::valuation::np_median;

/// `bands`' statistics over a positive, finite population and today's value:
/// (median, p25, p75, low, high, percentile of today).
pub fn band_stats(population: &[f64], now: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut sorted = population.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let low = population.iter().copied().fold(f64::INFINITY, np_minimum);
    let high = population
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, np_maximum);
    let at_or_below = population.iter().filter(|&&v| v <= now).count();
    (
        np_median(population),
        np_percentile_sorted(&sorted, 25.0),
        np_percentile_sorted(&sorted, 75.0),
        low,
        high,
        100.0 * at_or_below as f64 / population.len() as f64,
    )
}

/// `Series.iloc[-k:]` on a slice, where k = 0 means the whole series.
fn last_k(values: &[f64], k: usize) -> &[f64] {
    if k == 0 || k >= values.len() {
        values
    } else {
        &values[values.len() - k..]
    }
}

/// The trend signal's arithmetic over the completed month-end closes (at
/// least `sma_months` of them): (sma, provisional_sma, flip_close, the moving
/// average at each completed month, None before the window fills).
pub fn trend_sma(
    completed: &[f64],
    sma_months: usize,
    latest_close: f64,
) -> (f64, f64, f64, Vec<Option<f64>>) {
    let sma = pd_mean(last_k(completed, sma_months));

    // list(completed.iloc[-(n - 1):]) holds Python floats (iterating a
    // Series unboxes its values), and so does the latest close appended to it:
    // sum() adds them with CPython's compensated summation.
    let previous = last_k(completed, sma_months - 1);
    let mut provisional: Vec<(f64, PyNum)> = previous.iter().map(|&v| (v, PyNum::Float)).collect();
    provisional.push((latest_close, PyNum::Float));
    let provisional_sma = py_sum(&provisional).unwrap_or(0.0) / provisional.len() as f64;

    let flip_close = if previous.is_empty() {
        sma
    } else {
        let items: Vec<(f64, PyNum)> = previous.iter().map(|&v| (v, PyNum::Float)).collect();
        py_sum(&items).unwrap_or(0.0) / previous.len() as f64
    };

    let history = (0..completed.len())
        .map(|i| {
            (i + 1 >= sma_months).then(|| {
                let start = (i + 1).saturating_sub(sma_months);
                pd_mean(&completed[start..=i])
            })
        })
        .collect();
    (sma, provisional_sma, flip_close, history)
}

/// One metric's fall through one downturn window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stress {
    pub peak_year: i64,
    pub trough_year: i64,
    pub peak: f64,
    pub trough: f64,
    pub change_pct: f64,
    pub recovered_year: Option<i64>,
}

/// `stress_response`'s inner search for one metric and window. `annual` is
/// {fiscal year: value}; None where the original skips the metric (no peak,
/// a non-positive peak, or nothing after it).
pub fn stress_window(
    annual: &[(i64, f64)],
    peak_years: &[i64],
    trough_years: &[i64],
) -> Option<Stress> {
    let value = |y: i64| annual.iter().find(|(yy, _)| *yy == y).map(|&(_, v)| v);

    // max(peaks, key=peaks.get): the first of equal values; replaced only by a
    // strictly greater one.
    let mut best: Option<(i64, f64)> = None;
    for &y in peak_years {
        if let Some(v) = value(y) {
            match best {
                None => best = Some((y, v)),
                Some((_, b)) if v > b => best = Some((y, v)),
                _ => {}
            }
        }
    }
    let (peak_year, peak) = best?;
    // `if not peak or peak <= 0` — NaN is truthy and NaN <= 0 is false.
    if peak == 0.0 || peak <= 0.0 {
        return None;
    }

    let mut worst: Option<(i64, f64)> = None;
    for &y in trough_years {
        if y <= peak_year {
            continue;
        }
        if let Some(v) = value(y) {
            match worst {
                None => worst = Some((y, v)),
                Some((_, w)) if v < w => worst = Some((y, v)),
                _ => {}
            }
        }
    }
    let (trough_year, trough) = worst?;
    let change_pct = (trough - peak) / peak.abs() * 100.0;

    let mut recovered_year = None;
    if trough < peak {
        let mut later: Vec<i64> = annual
            .iter()
            .map(|&(y, _)| y)
            .filter(|&y| y > trough_year)
            .collect();
        later.sort_unstable();
        later.dedup();
        recovered_year = later
            .into_iter()
            .find(|&y| value(y).is_some_and(|v| v >= peak));
    }
    Some(Stress {
        peak_year,
        trough_year,
        peak,
        trough,
        change_pct,
        recovered_year,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_month_average_takes_the_whole_history() {
        let (_, provisional, flip, _) = trend_sma(&[1.0, 2.0, 3.0], 1, 10.0);
        assert_eq!(flip, 2.0); // iloc[-0:] is everything
        assert_eq!(provisional, (1.0 + 2.0 + 3.0 + 10.0) / 4.0);
    }

    #[test]
    fn a_fall_from_a_loss_is_not_a_fall() {
        assert!(stress_window(&[(2007, -5.0), (2009, -10.0)], &[2007], &[2009]).is_none());
    }

    #[test]
    fn recovery_is_the_first_year_back_at_the_peak() {
        let annual = [
            (2007, 100.0),
            (2009, 60.0),
            (2011, 90.0),
            (2012, 101.0),
            (2013, 120.0),
        ];
        let s = stress_window(&annual, &[2007, 2008], &[2008, 2009, 2010]).unwrap();
        assert_eq!(s.recovered_year, Some(2012));
        assert_eq!(s.change_pct, -40.0);
    }
}
