//! Python bindings. Each function keeps the argument order of the Numba kernel
//! it replaces, so the Python call sites stay as they were.

use std::borrow::Cow;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::analyzer::{self, Ledger};
use crate::holdings::{self, Book, Transactions, TypeIds};
use crate::util::KernelError;

impl From<KernelError> for PyErr {
    fn from(e: KernelError) -> Self {
        match e {
            KernelError::Index(_) => PyIndexError::new_err(e.to_string()),
            KernelError::Length { .. } => PyValueError::new_err(e.to_string()),
        }
    }
}

/// Borrow the array's buffer when it is contiguous, copy it when it is not.
fn view<'a, T: Element + Clone>(a: &'a PyReadonlyArray1<'_, T>) -> Cow<'a, [T]> {
    match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().to_vec()),
    }
}

fn usize_arg(what: &str, v: i64) -> PyResult<usize> {
    usize::try_from(v).map_err(|_| PyValueError::new_err(format!("{what} must be >= 0, got {v}")))
}

type HoldingsOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
);

/// `_calculate_holdings_numba`: returns (qty, cost, currency, cash,
/// cash_currency, last_prices).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn calculate_holdings<'py>(
    py: Python<'py>,
    target_date_ordinal: i64,
    tx_dates_ordinal: PyReadonlyArray1<'py, i64>,
    tx_symbols: PyReadonlyArray1<'py, i64>,
    tx_accounts: PyReadonlyArray1<'py, i64>,
    tx_to_accounts: PyReadonlyArray1<'py, i64>,
    tx_types: PyReadonlyArray1<'py, i64>,
    tx_quantities: PyReadonlyArray1<'py, f64>,
    tx_prices: PyReadonlyArray1<'py, f64>,
    tx_totals: PyReadonlyArray1<'py, f64>,
    tx_commissions: PyReadonlyArray1<'py, f64>,
    tx_split_ratios: PyReadonlyArray1<'py, f64>,
    tx_local_currencies: PyReadonlyArray1<'py, i64>,
    num_symbols: i64,
    num_accounts: i64,
    _num_currencies: i64,
    split_type_id: i64,
    stock_split_type_id: i64,
    buy_type_id: i64,
    deposit_type_id: i64,
    sell_type_id: i64,
    withdrawal_type_id: i64,
    short_sell_type_id: i64,
    buy_to_cover_type_id: i64,
    transfer_type_id: i64,
    spin_off_type_id: i64,
    fees_type_id: i64,
    dividend_type_id: i64,
    interest_type_id: i64,
    tax_type_id: i64,
    cash_symbol_id: i64,
    stock_qty_close_tolerance: f64,
    shortable_symbol_ids: PyReadonlyArray1<'py, i64>,
    acc_cash_modes: PyReadonlyArray1<'py, i64>,
) -> PyResult<HoldingsOut<'py>> {
    let ns = usize_arg("num_symbols", num_symbols)?;
    let na = usize_arg("num_accounts", num_accounts)?;
    let (dates, syms, accs, to_accs, types) = (
        view(&tx_dates_ordinal),
        view(&tx_symbols),
        view(&tx_accounts),
        view(&tx_to_accounts),
        view(&tx_types),
    );
    let (qtys, prices, totals, comms, splits, ccys) = (
        view(&tx_quantities),
        view(&tx_prices),
        view(&tx_totals),
        view(&tx_commissions),
        view(&tx_split_ratios),
        view(&tx_local_currencies),
    );
    let (shortable, modes) = (view(&shortable_symbol_ids), view(&acc_cash_modes));

    let tx = Transactions {
        dates: &dates,
        symbols: &syms,
        accounts: &accs,
        to_accounts: &to_accs,
        types: &types,
        quantities: &qtys,
        prices: &prices,
        totals: &totals,
        commissions: &comms,
        split_ratios: &splits,
    };
    let book = Book {
        num_symbols: ns,
        num_accounts: na,
        types: TypeIds {
            split: split_type_id,
            stock_split: stock_split_type_id,
            buy: buy_type_id,
            deposit: deposit_type_id,
            sell: sell_type_id,
            withdrawal: withdrawal_type_id,
            short_sell: short_sell_type_id,
            buy_to_cover: buy_to_cover_type_id,
            transfer: transfer_type_id,
            spin_off: spin_off_type_id,
            fees: fees_type_id,
            dividend: dividend_type_id,
            interest: interest_type_id,
            tax: tax_type_id,
        },
        cash_symbol_id,
        stock_qty_close_tolerance,
        shortable_symbol_ids: &shortable,
        acc_cash_modes: &modes,
    };

    let out = py.detach(|| holdings::calculate_holdings(target_date_ordinal, &tx, &ccys, &book))?;
    let m2 = |v: Vec<f64>| Array2::from_shape_vec((ns, na), v).expect("kernel shape");
    Ok((
        m2(out.qty).into_pyarray(py),
        m2(out.cost).into_pyarray(py),
        Array2::from_shape_vec((ns, na), out.currency)
            .expect("kernel shape")
            .into_pyarray(py),
        Array1::from_vec(out.cash).into_pyarray(py),
        Array1::from_vec(out.cash_currency).into_pyarray(py),
        m2(out.last_prices).into_pyarray(py),
    ))
}

type DailyOut<'py> = (
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
);

/// `_calculate_daily_holdings_chronological_numba`: returns (daily_qty,
/// daily_cash, daily_last_prices). Note the argument order: destination
/// accounts come before source accounts, as in the original.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn calculate_daily_holdings_chronological<'py>(
    py: Python<'py>,
    date_ordinals: PyReadonlyArray1<'py, i64>,
    tx_dates_ordinal: PyReadonlyArray1<'py, i64>,
    tx_symbols: PyReadonlyArray1<'py, i64>,
    tx_to_accounts: PyReadonlyArray1<'py, i64>,
    tx_accounts: PyReadonlyArray1<'py, i64>,
    tx_types: PyReadonlyArray1<'py, i64>,
    tx_quantities: PyReadonlyArray1<'py, f64>,
    tx_commissions: PyReadonlyArray1<'py, f64>,
    tx_split_ratios: PyReadonlyArray1<'py, f64>,
    tx_prices: PyReadonlyArray1<'py, f64>,
    tx_totals: PyReadonlyArray1<'py, f64>,
    num_symbols: i64,
    num_accounts: i64,
    split_type_id: i64,
    stock_split_type_id: i64,
    buy_type_id: i64,
    deposit_type_id: i64,
    sell_type_id: i64,
    withdrawal_type_id: i64,
    short_sell_type_id: i64,
    buy_to_cover_type_id: i64,
    transfer_type_id: i64,
    spin_off_type_id: i64,
    dividend_type_id: i64,
    interest_type_id: i64,
    fees_type_id: i64,
    tax_type_id: i64,
    cash_symbol_id: i64,
    stock_qty_close_tolerance: f64,
    shortable_symbol_ids: PyReadonlyArray1<'py, i64>,
    acc_cash_modes: PyReadonlyArray1<'py, i64>,
) -> PyResult<DailyOut<'py>> {
    let ns = usize_arg("num_symbols", num_symbols)?;
    let na = usize_arg("num_accounts", num_accounts)?;
    let days = view(&date_ordinals);
    let (dates, syms, accs, to_accs, types) = (
        view(&tx_dates_ordinal),
        view(&tx_symbols),
        view(&tx_accounts),
        view(&tx_to_accounts),
        view(&tx_types),
    );
    let (qtys, prices, totals, comms, splits) = (
        view(&tx_quantities),
        view(&tx_prices),
        view(&tx_totals),
        view(&tx_commissions),
        view(&tx_split_ratios),
    );
    let (shortable, modes) = (view(&shortable_symbol_ids), view(&acc_cash_modes));

    let tx = Transactions {
        dates: &dates,
        symbols: &syms,
        accounts: &accs,
        to_accounts: &to_accs,
        types: &types,
        quantities: &qtys,
        prices: &prices,
        totals: &totals,
        commissions: &comms,
        split_ratios: &splits,
    };
    let book = Book {
        num_symbols: ns,
        num_accounts: na,
        types: TypeIds {
            split: split_type_id,
            stock_split: stock_split_type_id,
            buy: buy_type_id,
            deposit: deposit_type_id,
            sell: sell_type_id,
            withdrawal: withdrawal_type_id,
            short_sell: short_sell_type_id,
            buy_to_cover: buy_to_cover_type_id,
            transfer: transfer_type_id,
            spin_off: spin_off_type_id,
            fees: fees_type_id,
            dividend: dividend_type_id,
            interest: interest_type_id,
            tax: tax_type_id,
        },
        cash_symbol_id,
        stock_qty_close_tolerance,
        shortable_symbol_ids: &shortable,
        acc_cash_modes: &modes,
    };

    let nd = days.len();
    let out = py.detach(|| holdings::calculate_daily_holdings_chronological(&days, &tx, &book))?;
    let m3 = |v: Vec<f64>| Array3::from_shape_vec((nd, ns, na), v).expect("kernel shape");
    Ok((
        m3(out.qty).into_pyarray(py),
        Array2::from_shape_vec((nd, na), out.cash)
            .expect("kernel shape")
            .into_pyarray(py),
        m3(out.last_prices).into_pyarray(py),
    ))
}

type CoreOut<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<f64>>);

/// `_process_numba_core`: returns (state[num_syms, num_accs, 14],
/// transfer_costs[num_tx]).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn process_core<'py>(
    py: Python<'py>,
    sym_ids: PyReadonlyArray1<'py, i64>,
    acc_ids: PyReadonlyArray1<'py, i64>,
    type_ids: PyReadonlyArray1<'py, i64>,
    qtys: PyReadonlyArray1<'py, f64>,
    prices: PyReadonlyArray1<'py, f64>,
    comms: PyReadonlyArray1<'py, f64>,
    totals: PyReadonlyArray1<'py, f64>,
    split_ratios: PyReadonlyArray1<'py, f64>,
    to_acc_ids: PyReadonlyArray1<'py, i64>,
    local_curr_ids: PyReadonlyArray1<'py, i64>,
    fx_rates_hist: PyReadonlyArray1<'py, f64>,
    shortable_sym_ids: PyReadonlyArray1<'py, i64>,
    stock_qty_tol: f64,
    cash_sym_id: i64,
    num_tx: i64,
    num_syms: i64,
    num_accs: i64,
    acc_cash_modes: PyReadonlyArray1<'py, i64>,
) -> PyResult<CoreOut<'py>> {
    let ntx = usize_arg("num_tx", num_tx)?;
    let ns = usize_arg("num_syms", num_syms)?;
    let na = usize_arg("num_accs", num_accs)?;
    let (s, a, t, q, p, c, tot, sp, to, lc, fx) = (
        view(&sym_ids),
        view(&acc_ids),
        view(&type_ids),
        view(&qtys),
        view(&prices),
        view(&comms),
        view(&totals),
        view(&split_ratios),
        view(&to_acc_ids),
        view(&local_curr_ids),
        view(&fx_rates_hist),
    );
    // Numba iterated range(num_tx) over arrays that may be longer; only the
    // first num_tx rows count.
    let head = |v: &[i64]| v.get(..ntx).map(<[i64]>::to_vec);
    let headf = |v: &[f64]| v.get(..ntx).map(<[f64]>::to_vec);
    let short =
        || PyValueError::new_err(format!("transaction arrays are shorter than num_tx={ntx}"));
    let ledger_vecs = (
        head(&s).ok_or_else(short)?,
        head(&a).ok_or_else(short)?,
        head(&t).ok_or_else(short)?,
        headf(&q).ok_or_else(short)?,
        headf(&p).ok_or_else(short)?,
        headf(&c).ok_or_else(short)?,
        headf(&tot).ok_or_else(short)?,
        headf(&sp).ok_or_else(short)?,
        head(&to).ok_or_else(short)?,
        head(&lc).ok_or_else(short)?,
        headf(&fx).ok_or_else(short)?,
    );
    let ledger = Ledger {
        sym_ids: &ledger_vecs.0,
        acc_ids: &ledger_vecs.1,
        type_ids: &ledger_vecs.2,
        qtys: &ledger_vecs.3,
        prices: &ledger_vecs.4,
        comms: &ledger_vecs.5,
        totals: &ledger_vecs.6,
        split_ratios: &ledger_vecs.7,
        to_acc_ids: &ledger_vecs.8,
        local_curr_ids: &ledger_vecs.9,
        fx_rates_hist: &ledger_vecs.10,
    };
    let (shortable, modes) = (view(&shortable_sym_ids), view(&acc_cash_modes));

    let out = py.detach(|| {
        analyzer::process_core(
            &ledger,
            &shortable,
            stock_qty_tol,
            cash_sym_id,
            ntx,
            ns,
            na,
            &modes,
        )
    })?;
    Ok((
        Array3::from_shape_vec((ns, na, analyzer::FIELDS), out.state)
            .expect("kernel shape")
            .into_pyarray(py),
        Array1::from_vec(out.transfer_costs).into_pyarray(py),
    ))
}

/// `_dividend_amount`, exposed for parity tests.
#[pyfunction]
pub fn dividend_amount(qty: f64, price: f64, total_amount: f64) -> f64 {
    holdings::dividend_amount(qty, price, total_amount)
}
