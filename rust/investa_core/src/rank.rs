//! Cross-sectional scoring for the Buffett ranking: `buffett_rank`'s
//! winsorised percentile, pillar means and weight-renormalised blends (also
//! used by `buffett_value.score_value`).
//!
//! Inputs are float arrays with NaN for missing, aligned to one group of
//! companies; pandas keeps the grouping, the index and the final sort (its
//! default quicksort orders ties in its own way, so the order stays pandas').
//! The pandas semantics reproduced:
//! * `Series.quantile(q)` is NumPy's linear percentile of the non-NaN values
//!   at `q * 100.0`, so an infinite value can make a bound infinite or NaN.
//! * `Series.clip(lower, upper)` ignores a NaN bound and swaps reversed ones.
//! * `Series.rank(pct=True, na_option="keep")` gives a tie group the integer
//!   sum of its 1-based positions over its size, divided by the non-NaN count.
//! * Row means skip NaN and add left to right (NumPy's pairwise sum, which is
//!   sequential for the handful of columns a pillar has).

use crate::mc::np_percentile_sorted;
use crate::stats::np_sum;
use crate::valuation::py_max;

/// `_winsorised_percentile`: a 0-100 percentile of the values clipped at
/// their 1st/99th centile; all NaN when fewer than two values are present.
pub fn winsorised_percentile(values: &[f64], higher_is_better: bool) -> Vec<f64> {
    let present: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if present.len() < 2 {
        return vec![f64::NAN; values.len()];
    }
    let mut sorted = present.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let mut lower = Some(np_percentile_sorted(&sorted, 0.01 * 100.0)).filter(|v| !v.is_nan());
    let mut upper = Some(np_percentile_sorted(&sorted, 0.99 * 100.0)).filter(|v| !v.is_nan());
    if let (Some(lo), Some(hi)) = (lower, upper) {
        // `min(lower, upper), max(lower, upper)`
        lower = Some(if hi < lo { hi } else { lo });
        upper = Some(py_max(lo, hi));
    }
    let clipped: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                return v;
            }
            // pandas: `where(self >= lower, lower)`, then `where(self <= upper,
            // upper)` against the original value. Neither side is NaN here.
            let mut r = v;
            if lower.is_some_and(|lo| v < lo) {
                r = lower.unwrap();
            }
            if upper.is_some_and(|hi| v > hi) {
                r = upper.unwrap();
            }
            r
        })
        .collect();
    let ranked = rank_pct_average(&clipped);
    ranked
        .into_iter()
        .map(|r| {
            let r = if higher_is_better { r } else { 1.0 - r };
            r * 100.0
        })
        .collect()
}

/// `Series.rank(method="average", pct=True, na_option="keep")`.
pub fn rank_pct_average(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).filter(|&i| !values[i].is_nan()).collect();
    // A stable sort, like pandas' mergesort; ties are averaged anyway.
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).expect("no NaN"));
    let count = order.len() as f64;
    let mut out = vec![f64::NAN; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let dups = (j - i + 1) as i64;
        let sum_ranks: i64 = ((i + 1) as i64..=(j + 1) as i64).sum();
        let rank = sum_ranks as f64 / dups as f64;
        for &k in &order[i..=j] {
            out[k] = rank / count;
        }
        i = j + 1;
    }
    out
}

/// `pd.concat(columns, axis=1).mean(axis=1, skipna=True)`: per row, the mean
/// of the non-NaN entries; NaN where a row has none.
pub fn row_mean(columns: &[Vec<f64>]) -> Vec<f64> {
    let n = columns.first().map_or(0, Vec::len);
    (0..n)
        .map(|row| {
            let mut count = 0;
            let filled: Vec<f64> = columns
                .iter()
                .map(|c| {
                    let v = c[row];
                    if v.is_nan() {
                        0.0
                    } else {
                        count += 1;
                        v
                    }
                })
                .collect();
            if count == 0 {
                f64::NAN
            } else {
                np_sum(&filled) / count as f64
            }
        })
        .collect()
}

/// `_quality_score` / `score_value`'s blend: per row, the weighted sum of the
/// present columns over the sum of their weights; NaN where none is present.
/// `total.add(v.fillna(0) * w * present)` and `weight_sum.add(w * present)`,
/// in column order.
pub fn weighted_renormalised(columns: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let n = columns.first().map_or(0, Vec::len);
    (0..n)
        .map(|row| {
            let mut total = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for (c, &w) in columns.iter().zip(weights) {
                let v = c[row];
                let present = if v.is_nan() { 0.0 } else { 1.0 };
                let filled = if v.is_nan() { 0.0 } else { v };
                total += filled * w * present;
                weight_sum += w * present;
            }
            if weight_sum == 0.0 {
                f64::NAN
            } else {
                total / weight_sum
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_average_ties_and_keep_nan() {
        let r = rank_pct_average(&[3.0, 1.0, 3.0, f64::NAN, 2.0]);
        assert_eq!(r[1], 0.25);
        assert_eq!(r[4], 0.5);
        assert_eq!(r[0], 3.5 / 4.0);
        assert_eq!(r[2], 3.5 / 4.0);
        assert!(r[3].is_nan());
    }

    #[test]
    fn too_few_values_are_all_nan() {
        assert!(
            winsorised_percentile(&[1.0, f64::NAN], true)
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn renormalised_over_present_weights() {
        let out =
            weighted_renormalised(&[vec![50.0, f64::NAN], vec![100.0, f64::NAN]], &[0.3, 0.1]);
        assert_eq!(out[0], (50.0 * 0.3 + 100.0 * 0.1) / (0.3 + 0.1));
        assert!(out[1].is_nan());
    }
}
