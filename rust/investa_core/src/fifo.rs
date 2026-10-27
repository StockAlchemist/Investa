//! FIFO tax lots and realized gains: the lot bookkeeping inside
//! `portfolio_analyzer.calculate_fifo_lots_and_gains`, ported line for line.
//!
//! Python still reads each transaction row (with its parsing quirks), looks up
//! the FX rates, decides each row's event, de-duplicates splits, and assembles
//! the gains frame and lot dicts. What runs here is everything that depends on
//! the lots themselves: transfers that move them, global splits, the two
//! spin-off legs, buys that open them, and FIFO sells that consume them.
//!
//! `holdings_long` is a `defaultdict`, so *reading* a key creates it, and the
//! returned dict keeps that creation order — including holdings left empty.
//! [`Book::touch`] reproduces exactly the accesses that create keys.
//!
//! Lot quantities and per-share costs come back as floats; everything else in
//! a lot dict (the purchase date, its FX rate, the transaction id) is never
//! changed after the lot opens, so Python reattaches the original objects.

use crate::util::py_min;

/// One open lot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lot {
    pub qty: f64,
    pub cost_per_share_local_net: f64,
    /// `date.toordinal()` of the purchase.
    pub purchase_date: i64,
    pub purchase_fx_to_display: f64,
    /// The rank of the lot's `original_tx_id` among all ids, which orders
    /// lots of one purchase date exactly as comparing the ids does.
    pub id_rank: i64,
    /// The event that opened the lot, so Python can hand back the original
    /// date, FX and id objects untouched.
    pub source: usize,
}

/// One row's effect, already parsed and priced in Python. Holdings are ids
/// for (symbol, account) pairs; `symbol_of[h]` names a holding's symbol.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    Transfer {
        from: usize,
        to: usize,
        qty: f64,
    },
    /// A split not seen before (Python de-duplicates by symbol, date, ratio).
    Split {
        symbol: usize,
        ratio: f64,
    },
    /// Opens a lot: a buy, or a spin-off's child receipt.
    Open {
        holding: usize,
        lot: Lot,
    },
    /// A spin-off's parent leg: the basis falls by `total_amt`, clamped.
    ReduceBasis {
        holding: usize,
        total_amt: f64,
    },
    Sell {
        holding: usize,
        qty: f64,
        price_local: f64,
        commission_local: f64,
        fx: f64,
    },
}

/// A sale's numbers, before the display-currency fallback Python applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sale {
    pub proceeds_local: f64,
    pub cost_basis_local: f64,
    pub realized_gain_local: f64,
    pub proceeds_display: f64,
    pub cost_basis_display: f64,
    pub realized_gain_display: f64,
}

/// The lots by holding, plus the order in which `holdings_long` created keys.
pub struct Book {
    lots: Vec<Vec<Lot>>,
    created: Vec<bool>,
    order: Vec<usize>,
}

impl Book {
    fn new(n: usize) -> Self {
        Book {
            lots: vec![Vec::new(); n],
            created: vec![false; n],
            order: Vec::new(),
        }
    }

    /// `holdings_long[key]` on a defaultdict: the access creates the key.
    fn touch(&mut self, h: usize) {
        if !self.created[h] {
            self.created[h] = true;
            self.order.push(h);
        }
    }

    /// The open lots in `dict(holdings_long)` order.
    pub fn into_ordered(self) -> Vec<(usize, Vec<Lot>)> {
        let mut lots = self.lots;
        self.order
            .into_iter()
            .map(|h| (h, std::mem::take(&mut lots[h])))
            .collect()
    }
}

/// Run the events. Returns the sales in event order and the final book.
pub fn run(events: &[Event], symbol_of: &[usize], tol: f64) -> (Vec<Sale>, Book) {
    let mut book = Book::new(symbol_of.len());
    let mut sales = Vec::new();

    for &event in events {
        match event {
            Event::Transfer { from, to, qty } => {
                book.touch(from);
                if book.lots[from].is_empty() {
                    continue; // "no lots tracked (maybe short or data gap)"
                }
                let mut remaining = qty;
                let mut kept = Vec::new();
                let mut moved = Vec::new();
                for mut lot in std::mem::take(&mut book.lots[from]) {
                    if remaining <= tol {
                        kept.push(lot);
                        continue;
                    }
                    let taken = py_min(remaining, lot.qty);
                    moved.push(Lot { qty: taken, ..lot });
                    lot.qty -= taken;
                    remaining -= taken;
                    if lot.qty > tol {
                        kept.push(lot);
                    }
                }
                book.lots[from] = kept;
                book.touch(to);
                book.lots[to].extend(moved);
                // Keep FIFO at the destination: a stable sort by purchase date,
                // then the original transaction id.
                book.lots[to].sort_by_key(|l| (l.purchase_date, l.id_rank));
            }
            Event::Split { symbol, ratio } => {
                for &h in &book.order {
                    if symbol_of[h] != symbol {
                        continue;
                    }
                    for lot in &mut book.lots[h] {
                        lot.qty *= ratio;
                        if lot.qty > 0.0 {
                            lot.cost_per_share_local_net /= ratio;
                        } else {
                            lot.cost_per_share_local_net = 0.0;
                        }
                    }
                }
            }
            Event::Open { holding, lot } => {
                book.touch(holding);
                book.lots[holding].push(lot);
            }
            Event::ReduceBasis { holding, total_amt } => {
                // `holdings_long.get(key, [])`: no key is created.
                let lots = &mut book.lots[holding];
                let mut current_basis = 0.0_f64;
                for lot in lots.iter() {
                    current_basis += lot.qty * lot.cost_per_share_local_net;
                }
                if current_basis > 1e-9 {
                    let reduce_amt = if total_amt < current_basis {
                        total_amt
                    } else {
                        current_basis
                    };
                    let factor = 1.0 - (reduce_amt / current_basis);
                    for lot in lots.iter_mut() {
                        lot.cost_per_share_local_net *= factor;
                    }
                }
            }
            Event::Sell {
                holding,
                qty,
                price_local,
                commission_local,
                fx,
            } => {
                book.touch(holding);
                let mut remaining = qty;
                let proceeds_local = (qty * price_local) - commission_local;
                let mut cost_local = 0.0_f64;
                let mut cost_display = 0.0_f64;
                let mut kept = Vec::new();
                for mut lot in std::mem::take(&mut book.lots[holding]) {
                    if remaining <= tol {
                        kept.push(lot);
                        continue;
                    }
                    let sold = py_min(remaining, lot.qty);
                    cost_local += sold * lot.cost_per_share_local_net;
                    let mut part = f64::NAN;
                    if !lot.purchase_fx_to_display.is_nan()
                        && !lot.cost_per_share_local_net.is_nan()
                    {
                        part = sold * lot.cost_per_share_local_net * lot.purchase_fx_to_display;
                    }
                    if !part.is_nan() {
                        // A NaN running total restarts from 0.0, as the original's
                        // `(total if notna(total) else 0.0) + part` does.
                        cost_display = (if cost_display.is_nan() {
                            0.0
                        } else {
                            cost_display
                        }) + part;
                    } else {
                        cost_display = f64::NAN;
                    }
                    lot.qty -= sold;
                    remaining -= sold;
                    if lot.qty >= tol {
                        kept.push(lot);
                    }
                }
                book.lots[holding] = kept;

                let realized_local = proceeds_local - cost_local;
                let mut proceeds_display = f64::NAN;
                if !fx.is_nan() && !proceeds_local.is_nan() {
                    proceeds_display = proceeds_local * fx;
                }
                let mut gain_display = f64::NAN;
                if !proceeds_display.is_nan() && !cost_display.is_nan() {
                    gain_display = proceeds_display - cost_display;
                }
                sales.push(Sale {
                    proceeds_local,
                    cost_basis_local: cost_local,
                    realized_gain_local: realized_local,
                    proceeds_display,
                    cost_basis_display: cost_display,
                    realized_gain_display: gain_display,
                });
            }
        }
    }
    (sales, book)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lot(qty: f64, cps: f64, date: i64, id: i64) -> Lot {
        Lot {
            qty,
            cost_per_share_local_net: cps,
            purchase_date: date,
            purchase_fx_to_display: 1.0,
            id_rank: id,
            source: id as usize,
        }
    }

    #[test]
    fn sells_consume_the_oldest_lot_first() {
        let events = [
            Event::Open {
                holding: 0,
                lot: lot(10.0, 100.0, 1, 1),
            },
            Event::Open {
                holding: 0,
                lot: lot(10.0, 200.0, 2, 2),
            },
            Event::Sell {
                holding: 0,
                qty: 15.0,
                price_local: 300.0,
                commission_local: 0.0,
                fx: 1.0,
            },
        ];
        let (sales, book) = run(&events, &[0], 1e-6);
        assert_eq!(sales[0].cost_basis_local, 10.0 * 100.0 + 5.0 * 200.0);
        let lots = book.into_ordered();
        assert_eq!(lots[0].1.len(), 1);
        assert_eq!(lots[0].1[0].qty, 5.0);
    }

    #[test]
    fn a_transfer_from_an_empty_holding_creates_only_the_source_key() {
        let (_, book) = run(
            &[Event::Transfer {
                from: 0,
                to: 1,
                qty: 3.0,
            }],
            &[0, 0],
            1e-6,
        );
        let ordered = book.into_ordered();
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].0, 0);
    }
}
