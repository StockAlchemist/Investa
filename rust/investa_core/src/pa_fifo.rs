//! `portfolio_analyzer.calculate_fifo_lots_and_gains`: the ledger read row by
//! row into lot events, replayed by the FIFO book (`fifo::run`), and the open
//! lots and realized-gains frame built from what it returns.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`). Each row is parsed with the same
//! Python operations (`pd.isna`, `pd.to_numeric`, `str().upper().strip()`)
//! and its arithmetic stays on the Python objects, so a quantity keeps its
//! NumPy type in the gains records; the event stream is built natively and
//! handed straight to the book. The FX bridge and the conversion fallback are
//! looked up on `portfolio_analyzer` at call time.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PySet, PyString, PyTuple};

use crate::fifo::{Event, Lot};
use crate::pyhelp::{es, log, pystr};

const OUTPUT_COLUMNS: [&str; 15] = [
    "Date",
    "Symbol",
    "Account",
    "Type",
    "Quantity",
    "Avg Sale Price (Local)",
    "Total Proceeds (Local)",
    "Total Cost Basis (Local)",
    "Realized Gain (Local)",
    "Sale/Cover FX Rate",
    "Total Proceeds (Display)",
    "Total Cost Basis (Display)",
    "Realized Gain (Display)",
    "LocalCurrency",
    "original_tx_id",
];

/// What the original kept per event: the lot's (purchase date, FX, id) for an
/// open, the partial gains record for a sell.
enum Ctx<'py> {
    None,
    Open(Bound<'py, PyTuple>),
    Sell(Bound<'py, PyDict>),
}

struct Parsed<'py> {
    tx_date: Bound<'py, PyAny>,
    symbol: Bound<'py, PyAny>,
    account: Bound<'py, PyAny>,
    tx_type: String,
    local_curr: Bound<'py, PyAny>,
    qty: Bound<'py, PyAny>,
    price_local: Bound<'py, PyAny>,
    commission_local: f64,
    split_ratio: Bound<'py, PyAny>,
}

/// (symbol, account) -> holding id; symbols get ids of their own for splits.
struct Holdings<'py> {
    ids: Bound<'py, PyDict>,
    keys: Vec<Bound<'py, PyTuple>>,
    symbol_ids: Bound<'py, PyDict>,
    symbol_of: Vec<usize>,
}

impl<'py> Holdings<'py> {
    /// `symbol_ids.setdefault(symbol, len(symbol_ids))`
    fn symbol(&self, symbol: &Bound<'py, PyAny>) -> PyResult<usize> {
        let next = self.symbol_ids.len();
        self.symbol_ids
            .call_method1("setdefault", (symbol, next))?
            .extract()
    }

    fn holding(&mut self, key: Bound<'py, PyTuple>) -> PyResult<usize> {
        if let Some(h) = self.ids.get_item(&key)? {
            return h.extract();
        }
        let h = self.keys.len();
        self.ids.set_item(&key, h)?;
        let s = self.symbol(&key.get_item(0)?)?;
        self.keys.push(key);
        self.symbol_of.push(s);
        Ok(h)
    }
}

struct Env<'py> {
    pd: Bound<'py, PyModule>,
    float: Bound<'py, PyAny>,
    str_t: Bound<'py, PyAny>,
    coerce: Bound<'py, PyDict>,
}

impl<'py> Env<'py> {
    fn isna(&self, v: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.pd.call_method1("isna", (v,))?.is_truthy()
    }
    fn notna(&self, v: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.pd.call_method1("notna", (v,))?.is_truthy()
    }
    /// `pd.to_numeric(v, errors="coerce")`
    fn numeric(&self, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.pd.call_method("to_numeric", (v,), Some(&self.coerce))
    }
    /// `float(v)`
    fn f(&self, v: &Bound<'py, PyAny>) -> PyResult<f64> {
        self.float.call1((v,))?.extract()
    }
    /// `float(v)` as the Python object.
    fn fobj(&self, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.float.call1((v,))
    }
    /// `str(v).upper().strip()`
    fn upper_strip(&self, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.str_t
            .call1((v,))?
            .call_method0("upper")?
            .call_method0("strip")
    }
    /// `v > bound` (or `<=`) with Python's comparison and truthiness.
    fn gt(&self, v: &Bound<'py, PyAny>, bound: f64) -> PyResult<bool> {
        v.gt(bound)
    }
    fn le(&self, v: &Bound<'py, PyAny>, bound: f64) -> PyResult<bool> {
        v.le(bound)
    }
    /// `np.nan if value is None else float(value)` (`_fifo_float`).
    fn fifo_float(&self, v: &Bound<'py, PyAny>) -> PyResult<f64> {
        if v.is_none() { Ok(f64::NAN) } else { self.f(v) }
    }
    /// `abs(float(v))`
    fn abs_float(&self, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.fobj(v)?.call_method0("__abs__")
    }
}

/// The `try:` block that reads one row; an exception is the caller's to
/// classify.
#[allow(clippy::too_many_arguments)]
fn parse<'py>(
    env: &Env<'py>,
    default_currency: &Bound<'py, PyAny>,
    tx_date_dt: &Bound<'py, PyAny>,
    symbol_raw: &Bound<'py, PyAny>,
    account_raw: &Bound<'py, PyAny>,
    type_raw: &Bound<'py, PyAny>,
    local_curr_raw: &Bound<'py, PyAny>,
    qty_raw: &Bound<'py, PyAny>,
    price_local_raw: &Bound<'py, PyAny>,
    commission_local_raw: &Bound<'py, PyAny>,
    split_ratio_raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Parsed<'py>>> {
    if env.isna(tx_date_dt)? {
        return Ok(None);
    }
    let tx_date = tx_date_dt.call_method0("date")?;

    let symbol = env.upper_strip(symbol_raw)?;
    let account = env.upper_strip(account_raw)?;
    let tx_type = env
        .str_t
        .call1((type_raw,))?
        .call_method0("lower")?
        .call_method0("strip")?;
    let use_raw = env.notna(local_curr_raw)?
        && env
            .str_t
            .call1((local_curr_raw,))?
            .call_method0("strip")?
            .is_truthy()?;
    let mut local_curr = env
        .str_t
        .call1((if use_raw {
            local_curr_raw
        } else {
            default_currency
        },))?
        .call_method0("upper")?;
    if !local_curr.is_truthy()? || local_curr.len()? != 3 {
        local_curr = default_currency.call_method0("upper")?;
    }

    let qty = env.numeric(qty_raw)?;
    let price_local = env.numeric(price_local_raw)?;
    let commission_local = if env.isna(commission_local_raw)? {
        0.0
    } else {
        env.f(&env.numeric(commission_local_raw)?)?
    };
    let split_ratio = env.numeric(split_ratio_raw)?;

    // The type is only ever compared with ASCII words, which a lossy copy
    // matches exactly as the str would.
    let tx_type = tx_type.cast::<PyString>()?.to_string_lossy().into_owned();
    Ok(Some(Parsed {
        tx_date,
        symbol,
        account,
        tx_type,
        local_curr,
        qty,
        price_local,
        commission_local,
        split_ratio,
    }))
}

/// `calculate_fifo_lots_and_gains`: (realized gains frame, open lots).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_fifo_lots_and_gains<'py>(
    py: Python<'py>,
    transactions_df: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    shortable_symbols: &Bound<'py, PyAny>,
    stock_quantity_close_tolerance: &Bound<'py, PyAny>,
    current_fx_rates_vs_usd: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let _ = shortable_symbols;
    let m = py.import("portfolio_analyzer")?;
    let pd = py.import("pandas")?;
    let builtins = py.import("builtins")?;
    let coerce = PyDict::new(py);
    coerce.set_item("errors", "coerce")?;
    let env = Env {
        pd: pd.clone(),
        float: builtins.getattr("float")?,
        str_t: builtins.getattr("str")?,
        coerce,
    };
    let output_columns = || PyList::new(py, OUTPUT_COLUMNS);
    let empty_frame = || -> PyResult<Bound<'py, PyAny>> {
        let kw = PyDict::new(py);
        kw.set_item("columns", output_columns()?)?;
        pd.getattr("DataFrame")?.call((), Some(&kw))
    };

    let mut hs = Holdings {
        ids: PyDict::new(py),
        keys: Vec::new(),
        symbol_ids: PyDict::new(py),
        symbol_of: Vec::new(),
    };

    let mut events: Vec<Event> = Vec::new();
    let mut context: Vec<Ctx<'py>> = Vec::new();

    // The historical rate depends only on the currency and the day here, and
    // a ledger repeats both, so each is asked once per call.
    let fx_memo = PyDict::new(py);
    let fx_on = |local_curr: &Bound<'py, PyAny>,
                 tx_date: &Bound<'py, PyAny>|
     -> PyResult<Bound<'py, PyAny>> {
        let key = PyTuple::new(py, [local_curr, tx_date])?;
        if !fx_memo.contains(&key)? {
            let rate = m.getattr("get_historical_rate_via_usd_bridge")?.call1((
                local_curr,
                display_currency,
                tx_date,
                historical_fx_yf,
            ))?;
            fx_memo.set_item(&key, rate)?;
        }
        fx_memo.as_any().get_item(&key)
    };

    // Track splits already applied to avoid double-processing.
    let processed_splits = PySet::empty(py)?;

    let n_rows = transactions_df.len()?;
    let columns = transactions_df.getattr("columns")?;
    let column = |name: &str, missing: Bound<'py, PyAny>| -> PyResult<Vec<Bound<'py, PyAny>>> {
        if columns.contains(name)? {
            transactions_df
                .get_item(name)?
                .call_method0("tolist")?
                .try_iter()?
                .collect()
        } else {
            Ok(vec![missing; n_rows])
        }
    };

    let required = ["Date", "Symbol", "Account", "Type"];
    for c in required {
        if !columns.contains(c)? {
            log(
                py,
                "warning",
                format!(
                    "Error parsing basic data for FIFO row: {}",
                    pystr(py, c).repr()?
                ),
            )?;
            return Ok((empty_frame()?, PyDict::new(py)));
        }
    }

    let none = || py.None().into_bound(py);
    let cols = [
        column("Date", none())?,
        column("Symbol", none())?,
        column("Account", none())?,
        column("Type", none())?,
        column("Local Currency", none())?,
        column("Quantity", none())?,
        column("Price/Share", none())?,
        column("Commission", none())?,
        column("Split Ratio", none())?,
        column("original_index", none())?,
        column("To Account", pystr(py, ""))?,
        column("Total Amount", none())?,
    ];
    let rows = cols.iter().map(Vec::len).min().unwrap_or(0);
    let cash = m.getattr("CASH_SYMBOL_CSV")?;

    let open_lot = |hs: &mut Holdings<'py>,
                    events: &mut Vec<Event>,
                    context: &mut Vec<Ctx<'py>>,
                    key: Bound<'py, PyTuple>,
                    qty: &Bound<'py, PyAny>,
                    cps: &Bound<'py, PyAny>,
                    tx_date: &Bound<'py, PyAny>,
                    fx: Bound<'py, PyAny>,
                    tx_id: &Bound<'py, PyAny>|
     -> PyResult<()> {
        let h = hs.holding(key)?;
        let fx_f = env.fifo_float(&fx)?;
        let day: i64 = tx_date.call_method0("toordinal")?.extract()?;
        let lot = Lot {
            qty: env.f(qty)?,
            cost_per_share_local_net: env.f(cps)?,
            purchase_date: day,
            purchase_fx_to_display: fx_f,
            id_rank: 0,
            source: events.len(),
        };
        events.push(Event::Open { holding: h, lot });
        context.push(Ctx::Open(PyTuple::new(
            py,
            [tx_date.clone(), fx, tx_id.clone()],
        )?));
        Ok(())
    };

    #[allow(clippy::needless_range_loop)]
    for i in 0..rows {
        let r = |c: usize| &cols[c][i];
        let parsed = match parse(
            &env,
            default_currency,
            r(0),
            r(1),
            r(2),
            r(3),
            r(4),
            r(5),
            r(6),
            r(7),
            r(8),
        ) {
            Ok(Some(p)) => p,
            Ok(None) => continue,
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "warning",
                    format!("Error parsing basic data for FIFO row: {}", es(py, &e)),
                )?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Parsed {
            tx_date,
            symbol,
            account,
            tx_type,
            local_curr,
            qty,
            mut price_local,
            commission_local,
            split_ratio,
        } = parsed;
        let tx_id = r(9);
        let to_account_raw = r(10);
        let total_amount_raw = r(11);

        let holding_key = PyTuple::new(py, [&symbol, &account])?;

        if symbol.eq(&cash)?
            || matches!(
                tx_type.as_str(),
                "deposit" | "withdrawal" | "fees" | "dividend"
            )
        {
            continue;
        }

        // --- Handle Transfers ---
        if tx_type == "transfer" {
            let to_account = env.upper_strip(to_account_raw)?;
            if !to_account.is_truthy()? {
                continue;
            }
            let from = hs.holding(holding_key)?;
            let to = hs.holding(PyTuple::new(py, [&symbol, &to_account])?)?;
            let qty = env.f(&qty)?;
            events.push(Event::Transfer { from, to, qty });
            context.push(Ctx::None);
            continue;
        }

        if tx_type == "split" || tx_type == "stock split" {
            if env.notna(&split_ratio)? && env.gt(&split_ratio, 0.0)? {
                let split_event = PyTuple::new(py, [&symbol, &tx_date, &env.fobj(&split_ratio)?])?;
                if !processed_splits.contains(&split_event)? {
                    // Global split: apply to all accounts holding this symbol
                    let s = hs.symbol(&symbol)?;
                    let ratio = env.f(&split_ratio)?;
                    events.push(Event::Split { symbol: s, ratio });
                    context.push(Ctx::None);
                    processed_splits.add(split_event)?;
                }
            }
            continue;
        }

        // --- Spin-off (cash-neutral basis reallocation, two legs by qty) ---
        if matches!(tx_type.as_str(), "spin off" | "spin-off" | "spinoff") {
            let total_amt_raw = env.numeric(total_amount_raw)?;
            let total_amt = if env.notna(&total_amt_raw)? {
                env.abs_float(&total_amt_raw)?
            } else {
                PyFloat::new(py, 0.0).into_any()
            };
            if env.notna(&qty)? && env.gt(&qty, 1e-9)? {
                // Child receipt: open a lot at the allocated per-share basis.
                let cps = if env.gt(&qty, 1e-9)? {
                    total_amt.div(&qty)?
                } else {
                    PyFloat::new(py, 0.0).into_any()
                };
                let fx_child = fx_on(&local_curr, &tx_date)?;
                open_lot(
                    &mut hs,
                    &mut events,
                    &mut context,
                    holding_key,
                    &qty,
                    &cps,
                    &tx_date,
                    fx_child,
                    tx_id,
                )?;
            } else {
                // Parent basis reduction: the remaining lots' total basis
                // drops by `total_amt` (clamped >= 0).
                let h = hs.holding(holding_key)?;
                let total_amt = env.f(&total_amt)?;
                events.push(Event::ReduceBasis {
                    holding: h,
                    total_amt,
                });
                context.push(Ctx::None);
            }
            continue;
        }

        if env.isna(&qty)? || env.le(&qty, 1e-9)? {
            continue;
        }

        let fx_rate = fx_on(&local_curr, &tx_date)?;

        if tx_type == "buy" {
            if env.isna(&price_local)? || env.le(&price_local, 1e-9)? {
                let total_amt_raw = env.numeric(total_amount_raw)?;
                if env.notna(&total_amt_raw)?
                    && env.gt(&env.abs_float(&total_amt_raw)?, 1e-9)?
                    && env.gt(&qty, 1e-9)?
                {
                    price_local = env.abs_float(&total_amt_raw)?.div(&qty)?;
                } else {
                    continue;
                }
            }
            let cps = qty
                .mul(&price_local)?
                .add(PyFloat::new(py, commission_local))?
                .div(&qty)?;
            open_lot(
                &mut hs,
                &mut events,
                &mut context,
                holding_key,
                &qty,
                &cps,
                &tx_date,
                fx_rate,
                tx_id,
            )?;
        } else if tx_type == "sell" {
            if env.isna(&price_local)? {
                let total_amt_raw = env.numeric(total_amount_raw)?;
                price_local = if env.notna(&total_amt_raw)? && env.gt(&qty, 1e-9)? {
                    env.abs_float(&total_amt_raw)?.div(&qty)?
                } else {
                    PyFloat::new(py, 0.0).into_any()
                };
            }
            // max(0.0, float(price_local))
            let p = env.f(&price_local)?;
            let p = if p > 0.0 { p } else { 0.0 };
            let h = hs.holding(holding_key)?;
            let fx = env.fifo_float(&fx_rate)?;
            let record = PyDict::new(py);
            record.set_item("Date", &tx_date)?;
            record.set_item("Symbol", &symbol)?;
            record.set_item("Account", &account)?;
            record.set_item("Type", "Sale Long")?;
            record.set_item("Quantity", &qty)?;
            record.set_item("Avg Sale Price (Local)", p)?;
            record.set_item("Sale/Cover FX Rate", &fx_rate)?;
            record.set_item("LocalCurrency", &local_curr)?;
            record.set_item("original_tx_id", tx_id)?;
            events.push(Event::Sell {
                holding: h,
                qty: env.f(&qty)?,
                price_local: p,
                commission_local,
                fx,
            });
            context.push(Ctx::Sell(record));
        }
    }

    // Lots of one purchase date stay in transaction-id order after a
    // transfer; the book compares ids by rank.
    let ids = PySet::empty(py)?;
    for c in &context {
        if let Ctx::Open(t) = c {
            ids.add(t.get_item(2)?)?;
        }
    }
    let id_rank = PyDict::new(py);
    for (r, v) in builtins
        .getattr("sorted")?
        .call1((ids,))?
        .try_iter()?
        .enumerate()
    {
        id_rank.set_item(v?, r)?;
    }
    for (e, c) in events.iter_mut().zip(&context) {
        if let (Event::Open { lot, .. }, Ctx::Open(t)) = (e, c) {
            lot.id_rank = id_rank.as_any().get_item(t.get_item(2)?)?.extract()?;
        }
    }

    let tol: f64 = env.f(stock_quantity_close_tolerance)?;
    let (sales, book) = py.detach(|| crate::fifo::run(&events, &hs.symbol_of, tol));

    let open_lots = PyDict::new(py);
    for (h, lots) in book.into_ordered() {
        let out = PyList::empty(py);
        for l in lots {
            let Ctx::Open(t) = &context[l.source] else {
                unreachable!("a lot opens on an open event")
            };
            let d = PyDict::new(py);
            d.set_item("qty", l.qty)?;
            d.set_item("cost_per_share_local_net", l.cost_per_share_local_net)?;
            d.set_item("purchase_date", t.get_item(0)?)?;
            d.set_item("purchase_fx_to_display", t.get_item(1)?)?;
            d.set_item("original_tx_id", t.get_item(2)?)?;
            out.append(d)?;
        }
        open_lots.set_item(&hs.keys[h], out)?;
    }

    let records = PyList::empty(py);
    let sells = context.iter().filter_map(|c| match c {
        Ctx::Sell(d) => Some(d),
        _ => None,
    });
    for (record, s) in sells.zip(&sales) {
        let local_curr = record.as_any().get_item("LocalCurrency")?;
        let gain_local = PyFloat::new(py, s.realized_gain_local).into_any();
        let mut gain_display = PyFloat::new(py, s.realized_gain_display).into_any();
        // Fallback logic for display gain
        if s.realized_gain_display.is_nan() && !s.realized_gain_local.is_nan() {
            if current_fx_rates_vs_usd.is_truthy()? {
                let attempt = || -> PyResult<Option<Bound<'py, PyAny>>> {
                    let rate = m.getattr("get_conversion_rate")?.call1((
                        &local_curr,
                        display_currency,
                        current_fx_rates_vs_usd,
                    ))?;
                    if env.notna(&rate)? {
                        return Ok(Some(gain_local.mul(&rate)?));
                    }
                    Ok(None)
                };
                match attempt() {
                    Ok(Some(g)) => gain_display = g,
                    Ok(None) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => {}
                    Err(e) => return Err(e),
                }
            }
            if env.isna(&gain_display)? && local_curr.eq(display_currency)? {
                gain_display = gain_local.clone();
            }
        }
        record.set_item("Total Proceeds (Local)", s.proceeds_local)?;
        record.set_item("Total Cost Basis (Local)", s.cost_basis_local)?;
        record.set_item("Realized Gain (Local)", &gain_local)?;
        record.set_item("Total Proceeds (Display)", s.proceeds_display)?;
        record.set_item("Total Cost Basis (Display)", s.cost_basis_display)?;
        record.set_item("Realized Gain (Display)", gain_display)?;
        records.append(record)?;
    }

    if records.is_empty() {
        return Ok((empty_frame()?, open_lots));
    }

    let kw = PyDict::new(py);
    kw.set_item("columns", output_columns()?)?;
    let df = pd.getattr("DataFrame")?.call((records,), Some(&kw))?;
    let kw = PyDict::new(py);
    kw.set_item("by", PyList::new(py, ["Date", "Symbol", "Account"])?)?;
    kw.set_item("inplace", true)?;
    df.call_method("sort_values", (), Some(&kw))?;
    Ok((df, open_lots))
}
