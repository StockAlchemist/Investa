//! The daily portfolio history's arithmetic.
//!
//! [`value_grid`] is the valuation step of
//! `portfolio_history._value_daily_holdings_vectorized`. Python still aligns
//! the prices and FX rates onto the day grid with pandas (reindex, forward
//! fill, interpolation); what runs here is the broadcast that followed: every
//! (day, symbol, account) cell valued at qty × price × FX, with the last
//! transaction price where the market price is missing, zero for empty and
//! excluded cells, and each day summed. The original built four temporary
//! arrays the size of the whole grid to do it (9k days × 90 symbols × 18
//! accounts is 14M cells, 115 MB each); this makes one pass per day.
//!
//! The day sums are NumPy's: `np.sum(V, axis=(1, 2))` reduces each day's
//! contiguous (symbol, account) block as one run ([`np_sum`]), and
//! `np.sum(V_cash, axis=1)` each day's accounts.
//!
//! The rest is the daily scans of `_calculate_accumulated_gains_and_resample`:
//! the time-weighted gain (with its bridge over -100% days), each benchmark's
//! accumulated gain, the money-weighted gain and ROI, and the drawdown. The
//! date filtering, resampling and baseline normalisation stay pandas. pandas'
//! `cumsum`/`cumprod`/`cummax` on float64 are NumPy's accumulations, which
//! start from the first element (so a leading -0.0 survives), with NaN
//! skipped; `cummax` keeps 0.0 over -0.0 ([`np_maximum`]).

use crate::stats::{np_maximum, np_sum};

/// Dimensions of the (days, symbols, accounts) grid.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub days: usize,
    pub symbols: usize,
    pub accounts: usize,
}

/// Each day's total value: stocks plus cash. Arrays are C-ordered:
/// `qty` and `last_prices` are (days, symbols, accounts), `prices` is
/// (days, symbols), `fx` and `cash` are (days, accounts).
pub fn value_grid(
    g: Grid,
    qty: &[f64],
    last_prices: Option<&[f64]>,
    prices: &[f64],
    fx: &[f64],
    cash: &[f64],
    included: &[bool],
) -> Vec<f64> {
    let (s_n, a_n) = (g.symbols, g.accounts);
    let mut stocks = vec![0.0_f64; s_n * a_n];
    let mut cash_values = vec![0.0_f64; a_n];
    let mut out = Vec::with_capacity(g.days);
    for d in 0..g.days {
        let fx_day = &fx[d * a_n..(d + 1) * a_n];
        for s in 0..s_n {
            let market = prices[d * s_n + s];
            let base = (d * s_n + s) * a_n;
            for a in 0..a_n {
                let q = qty[base + a];
                // V[qty == 0] = 0.0 and V[:, :, ~included] = 0.0 overwrite
                // whatever the product was, NaN included.
                stocks[s * a_n + a] = if q == 0.0 || !included[a] {
                    0.0
                } else {
                    let price = match last_prices {
                        Some(last) if market.is_nan() => last[base + a],
                        _ => market,
                    };
                    q * price * fx_day[a]
                };
            }
        }
        for a in 0..a_n {
            let c = cash[d * a_n + a];
            cash_values[a] = if c == 0.0 || !included[a] {
                0.0
            } else {
                c * fx_day[a]
            };
        }
        out.push(np_sum(&stocks) + np_sum(&cash_values));
    }
    out
}

/// `np.cumprod` / `np.cumsum`: a running fold that starts from the first value.
fn accumulate(values: &[f64], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = f64::NAN;
    for (i, &v) in values.iter().enumerate() {
        acc = if i == 0 { v } else { f(acc, v) };
        out.push(acc);
    }
    out
}

/// The portfolio's time-weighted accumulation: (gain factors, accumulated
/// gain, drawdown in percent).
pub fn portfolio_twr(daily_return: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // 1 + r.fillna(0.0), with factors below 1e-6 (a -100% day) bridged at 1.0
    // so the product does not stay at zero forever.
    let factors: Vec<f64> = daily_return
        .iter()
        .map(|&r| {
            let g = 1.0 + if r.is_nan() { 0.0 } else { r };
            if g < 1e-6 { 1.0 } else { g }
        })
        .collect();
    let mut accumulated = accumulate(&factors, |a, b| a * b);
    if daily_return.first().is_some_and(|r| r.is_nan()) {
        accumulated[0] = f64::NAN;
    }
    // (acc / acc.cummax() - 1) * 100, NaN → 0. cummax skips NaN: a NaN
    // stays NaN and the running maximum carries past it.
    let mut running = f64::NEG_INFINITY;
    let drawdown = accumulated
        .iter()
        .map(|&a| {
            if a.is_nan() {
                return 0.0;
            }
            running = np_maximum(running, a);
            let d = ((a / running) - 1.0) * 100.0;
            if d.is_nan() { 0.0 } else { d }
        })
        .collect();
    (factors, accumulated, drawdown)
}

/// A benchmark's accumulated gain from its price column (NaN where there is
/// no price): the return between consecutive prices, carried forward over
/// the gaps, 0 before the first, accumulated, with the first day NaN. None
/// when there are no prices at all.
pub fn benchmark_accumulated(prices: &[f64]) -> Option<Vec<f64>> {
    let mut previous: Option<f64> = None;
    let mut carried = f64::NAN;
    let mut factors = Vec::with_capacity(prices.len());
    for &p in prices {
        if !p.is_nan() {
            // pct_change(fill_method=None) on the NaN-free prices
            carried = match previous {
                Some(q) => p / q - 1.0,
                None => f64::NAN,
            };
            previous = Some(p);
        }
        // .reindex(...).ffill().fillna(0.0), then 1 + r
        factors.push(1.0 + if carried.is_nan() { 0.0 } else { carried });
    }
    previous?;
    let mut accumulated = accumulate(&factors, |a, b| a * b);
    if let Some(first) = accumulated.first_mut() {
        *first = f64::NAN;
    }
    Some(accumulated)
}

/// The money-weighted view: (cumulative net flow, absolute gain, ROI in
/// percent, NaN where the cumulative flow is not positive).
pub fn money_weighted(value: &[f64], net_flow: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let flows: Vec<f64> = net_flow
        .iter()
        .map(|&f| if f.is_nan() { 0.0 } else { f })
        .collect();
    let cumulative = accumulate(&flows, |a, b| a + b);
    let gain: Vec<f64> = value
        .iter()
        .zip(&cumulative)
        .map(|(&v, &c)| v - c)
        .collect();
    let roi = gain
        .iter()
        .zip(&cumulative)
        .map(|(&g, &c)| if c <= 0.0 { f64::NAN } else { (g / c) * 100.0 })
        .collect();
    (cumulative, gain, roi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wiped_out_day_is_bridged() {
        let (g, acc, dd) = portfolio_twr(&[f64::NAN, 0.1, -1.0, 0.1]);
        assert_eq!(g, vec![1.0, 1.1, 1.0, 1.1]);
        assert!(acc[0].is_nan());
        assert_eq!(acc[3], 1.1 * 1.0 * 1.1);
        assert_eq!(dd, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_benchmark_carries_its_last_return_over_gaps() {
        let acc = benchmark_accumulated(&[f64::NAN, 100.0, f64::NAN, 110.0, f64::NAN]).unwrap();
        assert!(acc[0].is_nan());
        let r = 110.0 / 100.0 - 1.0;
        assert_eq!(acc[3], 1.0 * 1.0 * (1.0 + r));
        assert_eq!(acc[4], acc[3] * (1.0 + r));
        assert!(benchmark_accumulated(&[f64::NAN]).is_none());
    }

    #[test]
    fn an_empty_cell_is_worth_nothing_even_without_a_price() {
        let g = Grid {
            days: 1,
            symbols: 2,
            accounts: 1,
        };
        let v = value_grid(
            g,
            &[0.0, 2.0],
            None,
            &[f64::NAN, 10.0],
            &[1.5],
            &[4.0],
            &[true],
        );
        assert_eq!(v, vec![2.0 * 10.0 * 1.5 + 4.0 * 1.5]);
    }

    #[test]
    fn a_missing_market_price_falls_back_to_the_last_trade() {
        let g = Grid {
            days: 1,
            symbols: 1,
            accounts: 2,
        };
        let v = value_grid(
            g,
            &[1.0, 3.0],
            Some(&[7.0, 9.0]),
            &[f64::NAN],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[true, false],
        );
        assert_eq!(v, vec![7.0]);
    }
}
