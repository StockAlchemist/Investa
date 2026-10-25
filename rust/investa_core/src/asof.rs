//! `finutils.get_historical_price` over a price frame: pandas' `DataFrame.asof`
//! plus the original's backfill, answered by binary search instead of a pandas
//! call per lookup (a summary makes about 1,300 of them).
//!
//! `DataFrame.asof(t)` takes the last row at or before `t` with no NaN in *any*
//! column, and NaN when there is none. Before the first row it returns NaN at
//! once, and the original then backfills with the first row's price even if
//! another column of that row is NaN. Python builds the table only for frames
//! whose semantics this covers (a sorted UTC or naive DatetimeIndex) and keeps
//! the original path for everything else.

/// The valid rows of one frame, as UTC nanosecond ticks.
#[derive(Debug, Clone)]
pub struct PriceTable {
    /// Ticks of the rows with no NaN in any column, ascending.
    ticks: Vec<i64>,
    /// The price of each of those rows.
    values: Vec<f64>,
    /// The frame's first tick, valid or not.
    first_tick: i64,
    /// The first row's price, NaN or not: the backfill.
    first_value: f64,
}

impl PriceTable {
    pub fn new(ticks: Vec<i64>, values: Vec<f64>, first_tick: i64, first_value: f64) -> Self {
        PriceTable {
            ticks,
            values,
            first_tick,
            first_value,
        }
    }

    /// The price on `t`, or NaN.
    pub fn asof(&self, t: i64) -> f64 {
        if t < self.first_tick {
            return self.first_value;
        }
        match self.ticks.partition_point(|&x| x <= t) {
            0 => f64::NAN,
            k => self.values[k - 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asof_skips_rows_with_a_gap_and_backfills_before_the_start() {
        // Rows at 10, 20 (has a NaN column, so not valid), 30.
        let t = PriceTable::new(vec![10, 30], vec![1.0, 3.0], 10, 1.0);
        assert_eq!(t.asof(5), 1.0); // before the frame: the first row's price
        assert_eq!(t.asof(10), 1.0);
        assert_eq!(t.asof(25), 1.0); // row 20 is skipped
        assert_eq!(t.asof(99), 3.0);
    }

    #[test]
    fn no_valid_row_yet_is_nan_not_a_backfill() {
        // The first row (tick 10) has a NaN column; the first valid row is 30.
        let t = PriceTable::new(vec![30], vec![3.0], 10, 2.0);
        assert!(t.asof(15).is_nan());
        assert_eq!(t.asof(9), 2.0);
    }
}
