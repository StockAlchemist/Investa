//! The per-row cash-flow rules behind the IRR and MWR:
//! `finutils.get_cash_flows_for_symbol_account` (one holding, in its local
//! currency) and `get_cash_flows_for_mwr` (an account or the portfolio).
//!
//! Python selects and sorts the rows, reads them with the originals' parsing
//! (`pd.to_numeric`, NaN commission as 0), decides each transfer's direction
//! (it compares account names and notes), converts MWR flows to the target
//! currency and assembles the dated series. What runs here is each row's flow,
//! branch for branch, NaN where the original's arithmetic gives NaN.

/// A row's type, lowercased and stripped as the originals compare it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Other = 0,
    Buy = 1,
    Deposit = 2,
    Sell = 3,
    Withdrawal = 4,
    Transfer = 5,
    ShortSell = 6,
    BuyToCover = 7,
    Dividend = 8,
    Fees = 9,
    Split = 10,
    Interest = 11,
}

impl Kind {
    pub fn from_code(c: u8) -> Option<Kind> {
        use Kind::*;
        Some(match c {
            0 => Other,
            1 => Buy,
            2 => Deposit,
            3 => Sell,
            4 => Withdrawal,
            5 => Transfer,
            6 => ShortSell,
            7 => BuyToCover,
            8 => Dividend,
            9 => Fees,
            10 => Split,
            11 => Interest,
            _ => return None,
        })
    }
}

/// One parsed row. `commission` is already 0.0 where it was NaN.
/// `direction` is a transfer's sign: -1 into the holding or scope, +1 out of
/// it, 0 neither.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub kind: Kind,
    pub qty: f64,
    pub price: f64,
    pub commission: f64,
    pub total: f64,
    pub direction: i8,
}

/// `abs(total) if > 1e-9 else abs(qty)`, NaN read as 0: a cash row's amount.
fn cash_amount(r: &Row) -> f64 {
    let total_abs = if r.total.is_nan() { 0.0 } else { r.total.abs() };
    let qty_abs = if r.qty.is_nan() { 0.0 } else { r.qty.abs() };
    if total_abs > 1e-9 { total_abs } else { qty_abs }
}

/// The dividend branch both originals share: the total when known, else
/// qty × price (or the price alone), less commission. With neither, the
/// amount is 0.0 and the flow is -commission.
fn dividend(r: &Row, qty_abs: f64) -> f64 {
    let amount = if !r.total.is_nan() {
        r.total
    } else if !r.price.is_nan() {
        if !r.qty.is_nan() && qty_abs > 0.0 {
            qty_abs * r.price
        } else {
            r.price
        }
    } else {
        0.0
    };
    if amount.is_nan() {
        0.0
    } else {
        amount - r.commission
    }
}

/// `get_cash_flows_for_symbol_account`'s flow for each row, in the holding's
/// order (sorted by date, then id): buys and fees out, sales and dividends in,
/// transfers by direction when they count as flows. 0.0 where a row has none.
pub fn holding_flows(
    rows: &[Row],
    is_cash: bool,
    shortable: bool,
    transfer_is_flow: bool,
) -> Vec<f64> {
    use Kind::*;
    let mut last_seen_price: Option<f64> = None;
    rows.iter()
        .map(|r| {
            if !r.price.is_nan() && r.price > 1e-9 {
                last_seen_price = Some(r.price);
            } else if !r.total.is_nan() && !r.qty.is_nan() && r.qty.abs() > 1e-9 {
                last_seen_price = Some((r.total / r.qty).abs());
            }
            let qty_abs = if r.qty.is_nan() { 0.0 } else { r.qty.abs() };
            let cash_amt = cash_amount(r);
            let mut flow = 0.0_f64;
            if is_cash {
                match r.kind {
                    Buy | Deposit => flow = -(cash_amt + r.commission),
                    Sell | Withdrawal => flow = cash_amt - r.commission,
                    _ => {}
                }
            } else {
                match r.kind {
                    Buy | Deposit if !r.qty.is_nan() && r.qty > 0.0 && !r.price.is_nan() => {
                        flow = -((qty_abs * r.price) + r.commission);
                    }
                    Sell | Withdrawal if !r.price.is_nan() && !r.qty.is_nan() && qty_abs > 0.0 => {
                        flow = (qty_abs * r.price) - r.commission;
                    }
                    _ => {}
                }
            }
            match r.kind {
                Transfer if transfer_is_flow => {
                    let amount = if is_cash {
                        cash_amt
                    } else {
                        let mut price = r.price;
                        if (price.is_nan() || price <= 1e-9)
                            && let Some(p) = last_seen_price
                        {
                            price = p;
                        }
                        if price.is_nan() { 0.0 } else { r.qty * price }
                    };
                    flow = if r.direction < 0 { -amount } else { amount };
                }
                ShortSell | BuyToCover if shortable => {
                    if !r.price.is_nan() && !r.qty.is_nan() && qty_abs > 0.0 {
                        flow = if r.kind == ShortSell {
                            (qty_abs * r.price) - r.commission
                        } else {
                            -((qty_abs * r.price) + r.commission)
                        };
                    }
                }
                Dividend => flow = dividend(r, qty_abs),
                Fees => flow = -(r.commission.abs()),
                Split => {
                    flow = if r.commission != 0.0 {
                        -r.commission.abs()
                    } else {
                        0.0
                    }
                }
                _ => {}
            }
            flow
        })
        .collect()
}

/// Which of a row's inputs a flow was computed from. Python uses it to give
/// the flow the type the original's arithmetic would have: a NumPy scalar
/// when a text cell was parsed into one.
pub const USES_QTY: u8 = 1;
pub const USES_PRICE: u8 = 2;
pub const USES_TOTAL: u8 = 4;

/// `cash_amount`, and which input it came from.
fn cash_amount_uses(r: &Row) -> (f64, u8) {
    let total_abs = if r.total.is_nan() { 0.0 } else { r.total.abs() };
    if total_abs > 1e-9 {
        (total_abs, USES_TOTAL)
    } else if r.qty.is_nan() {
        (0.0, 0)
    } else {
        (r.qty.abs(), USES_QTY)
    }
}

/// `dividend`, and which inputs it came from.
fn dividend_uses(r: &Row, qty_abs: f64) -> (f64, u8) {
    let flow = dividend(r, qty_abs);
    let uses = if !r.total.is_nan() {
        USES_TOTAL
    } else if !r.price.is_nan() {
        if !r.qty.is_nan() && qty_abs > 0.0 {
            USES_QTY | USES_PRICE
        } else {
            USES_PRICE
        }
    } else {
        0
    };
    (flow, if flow.is_nan() && uses == 0 { 0 } else { uses })
}

/// `get_cash_flows_for_mwr`'s local-currency flow for each row, and the
/// inputs it used. `is_cash[i]` says whether row i is `$CASH`. Under the
/// portfolio basis only external flows count: `$CASH` deposits and
/// withdrawals, and transfers that cross the scope (`direction`); per trade,
/// every trade is a flow.
pub fn mwr_flows(rows: &[Row], is_cash: &[bool], portfolio_basis: bool) -> Vec<(f64, u8)> {
    use Kind::*;
    const QP: u8 = USES_QTY | USES_PRICE;
    rows.iter()
        .zip(is_cash)
        .map(|(r, &cash)| {
            let qty_abs = if r.qty.is_nan() { 0.0 } else { r.qty.abs() };
            let priced = !r.qty.is_nan() && !r.price.is_nan();
            // A transfer that leaves the scope is money out of it (+), one
            // that enters is money in (-).
            let stock_transfer = |r: &Row| match r.direction {
                d if d > 0 && priced => ((r.qty.abs() * r.price) - r.commission.abs(), QP),
                d if d < 0 && priced => (-(r.qty.abs() * r.price), QP),
                _ => (0.0, 0),
            };
            if !cash {
                if portfolio_basis {
                    return if r.kind == Transfer {
                        stock_transfer(r)
                    } else {
                        (0.0, 0)
                    };
                }
                let traded = !r.price.is_nan() && !r.qty.is_nan() && qty_abs > 0.0;
                return match r.kind {
                    Buy | Deposit if priced => (-((qty_abs * r.price) + r.commission), QP),
                    Sell | Withdrawal if priced => ((qty_abs * r.price) - r.commission, QP),
                    Dividend => dividend_uses(r, qty_abs),
                    ShortSell if traded => ((qty_abs * r.price) - r.commission, QP),
                    BuyToCover if traded => (-((qty_abs * r.price) + r.commission), QP),
                    Fees => (-r.commission.abs(), 0),
                    Transfer => stock_transfer(r),
                    _ => (0.0, 0),
                };
            }
            let (cash_amt, uses) = cash_amount_uses(r);
            if portfolio_basis {
                return match r.kind {
                    Deposit => (-(cash_amt + r.commission), uses),
                    Withdrawal => (cash_amt - r.commission, uses),
                    _ => (0.0, 0),
                };
            }
            match r.kind {
                Deposit | Buy => (-(cash_amt + r.commission), uses),
                Withdrawal | Sell => (cash_amt - r.commission, uses),
                Transfer if r.direction > 0 => (cash_amt - r.commission, uses),
                Transfer if r.direction < 0 => (-(cash_amt + r.commission), uses),
                _ => (0.0, 0),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: Kind, qty: f64, price: f64, commission: f64, total: f64) -> Row {
        Row {
            kind,
            qty,
            price,
            commission,
            total,
            direction: 0,
        }
    }

    #[test]
    fn a_transfer_without_a_price_takes_the_last_one_seen() {
        let mut t = row(Kind::Transfer, 10.0, f64::NAN, 0.0, f64::NAN);
        t.direction = -1;
        let rows = [row(Kind::Buy, 5.0, 20.0, 1.0, f64::NAN), t];
        assert_eq!(
            holding_flows(&rows, false, false, true),
            vec![-101.0, -200.0]
        );
    }

    #[test]
    fn a_dividend_with_no_amount_still_costs_its_commission() {
        let rows = [row(Kind::Dividend, f64::NAN, f64::NAN, 2.5, f64::NAN)];
        assert_eq!(holding_flows(&rows, false, false, false), vec![-2.5]);
    }

    #[test]
    fn portfolio_mwr_sees_only_external_cash() {
        let rows = [
            row(Kind::Deposit, 0.0, 1.0, 0.0, 100.0),
            row(Kind::Buy, 0.0, 1.0, 0.0, 100.0),
            row(Kind::Buy, 3.0, 10.0, 1.0, f64::NAN),
        ];
        assert_eq!(
            mwr_flows(&rows, &[true, true, false], true)
                .into_iter()
                .map(|(f, _)| f)
                .collect::<Vec<_>>(),
            vec![-100.0, 0.0, 0.0]
        );
        assert_eq!(
            mwr_flows(&rows, &[true, true, false], false)
                .into_iter()
                .map(|(f, _)| f)
                .collect::<Vec<_>>(),
            vec![-100.0, -100.0, -31.0]
        );
    }
}
