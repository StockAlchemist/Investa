//! Python bindings. Each function keeps the argument order of the Numba kernel
//! it replaces, so the Python call sites stay as they were.

use std::borrow::Cow;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyUntypedArrayMethods,
};
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

/// `calculate_npv`: returns (npv, index of the first flow dated before the
/// first one, or -1). `days[i]` is `(dates[i] - dates[0]).days`.
#[pyfunction]
pub fn npv(
    rate: f64,
    days: PyReadonlyArray1<'_, i64>,
    flows: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, i64)> {
    let (d, f) = (view(&days), view(&flows));
    if d.len() != f.len() {
        return Err(PyValueError::new_err(
            "The lengths of dates and cash_flows do not match.",
        ));
    }
    let out = crate::irr::npv(rate, &d, &f);
    Ok((out.value, out.negative_delta_at.map_or(-1, |i| i as i64)))
}

/// `calculate_irr`: returns (rate, status, first, last, scaled,
/// saw_negative_delta). `ticks` are the dates in microseconds; `kinds` code
/// each flow's Python type for `sum()` (0 float, 1 int, 2 other).
#[pyfunction]
pub fn calculate_irr(
    py: Python<'_>,
    ticks: PyReadonlyArray1<'_, i64>,
    flows: PyReadonlyArray1<'_, f64>,
    kinds: PyReadonlyArray1<'_, u8>,
) -> PyResult<(f64, u8, usize, usize, bool, bool)> {
    let (t, f, k) = (view(&ticks), view(&flows), view(&kinds));
    let kinds: Vec<_> = k
        .iter()
        .map(|&c| crate::util::PyNum::from_code(c))
        .collect();
    let out = py.detach(|| crate::irr::calculate_irr(&t, &f, &kinds));
    Ok((
        out.rate,
        out.status as u8,
        out.first,
        out.last,
        out.scaled,
        out.saw_negative_delta,
    ))
}

/// The IRR scan grid as floats, for the test that recomputes it with NumPy.
#[pyfunction]
pub fn irr_scan_grid() -> Vec<f64> {
    crate::irr_grid::SCAN_GRID_BITS
        .iter()
        .map(|&b| f64::from_bits(b))
        .collect()
}

/// CPython 3.12 `sum()` over (value, kind) pairs, exposed for its parity test.
#[pyfunction]
pub fn py_sum(values: Vec<f64>, kinds: Vec<u8>) -> PyResult<Option<f64>> {
    if values.len() != kinds.len() {
        return Err(PyValueError::new_err("values and kinds differ in length"));
    }
    let items: Vec<_> = values
        .into_iter()
        .zip(kinds)
        .map(|(v, k)| (v, crate::util::PyNum::from_code(k)))
        .collect();
    Ok(crate::util::py_sum(&items))
}

/// Test hook: Rust `brentq` on `npv(rate, days, flows)`; `None` where SciPy
/// raises.
#[pyfunction]
pub fn brentq_npv(
    days: Vec<i64>,
    flows: Vec<f64>,
    a: f64,
    b: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> Option<f64> {
    crate::irr::brentq(
        |r| crate::irr::npv(r, &days, &flows).value,
        a,
        b,
        xtol,
        rtol,
        maxiter,
    )
}

/// Test hook: Rust `newton_secant` on `npv(rate, days, flows)`; `None` where
/// SciPy raises.
#[pyfunction]
pub fn newton_npv(
    days: Vec<i64>,
    flows: Vec<f64>,
    x0: f64,
    tol: f64,
    maxiter: usize,
) -> Option<f64> {
    crate::irr::newton_secant(
        |r| crate::irr::npv(r, &days, &flows).value,
        x0,
        tol,
        maxiter,
    )
}

// --- stats.rs: NumPy/pandas reductions and risk metrics -------------------

macro_rules! array_to_f64 {
    ($name:ident, $doc:literal, $f:path) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(values: PyReadonlyArray1<'_, f64>) -> f64 {
            $f(&view(&values))
        }
    };
}

array_to_f64!(
    np_sum,
    "NumPy's pairwise sum (`np.add.reduce`).",
    crate::stats::np_sum
);
array_to_f64!(pd_mean, "`Series.mean()` (skipna).", crate::stats::pd_mean);
array_to_f64!(pd_min, "`Series.min()` (skipna).", crate::stats::pd_min);
array_to_f64!(
    max_drawdown,
    "`risk_metrics.calculate_max_drawdown` on the values.",
    crate::stats::max_drawdown
);

/// `Series.var(ddof=ddof)` (skipna).
#[pyfunction]
pub fn pd_var(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> f64 {
    crate::stats::pd_var(&view(&values), ddof)
}

/// `Series.std(ddof=ddof)` (skipna).
#[pyfunction]
pub fn pd_std(values: PyReadonlyArray1<'_, f64>, ddof: usize) -> f64 {
    crate::stats::pd_std(&view(&values), ddof)
}

/// `risk_metrics.calculate_drawdown_series` on the values.
#[pyfunction]
pub fn drawdown_series<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(crate::stats::drawdown_series(&view(&values))).into_pyarray(py)
}

/// `pct_change(fill_method=None).replace([inf, -inf], nan)` on the values.
#[pyfunction]
pub fn pct_change<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(crate::stats::pct_change(&view(&values))).into_pyarray(py)
}

/// `risk_metrics.calculate_volatility`.
#[pyfunction]
pub fn volatility(returns: PyReadonlyArray1<'_, f64>, periods_per_year: f64) -> f64 {
    crate::stats::volatility(&view(&returns), periods_per_year)
}

/// `risk_metrics.calculate_sharpe_ratio`.
#[pyfunction]
pub fn sharpe_ratio(
    returns: PyReadonlyArray1<'_, f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> f64 {
    crate::stats::sharpe_ratio(&view(&returns), risk_free_rate, periods_per_year)
}

/// `risk_metrics.calculate_sortino_ratio`.
#[pyfunction]
pub fn sortino_ratio(
    returns: PyReadonlyArray1<'_, f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
    target_return: f64,
) -> f64 {
    crate::stats::sortino_ratio(
        &view(&returns),
        risk_free_rate,
        periods_per_year,
        target_return,
    )
}

/// One `calculate_benchmark_scoreboard` row from aligned returns: (alpha,
/// beta, r2, tracking_error, information_ratio).
#[pyfunction]
pub fn scoreboard_stats(
    rp: PyReadonlyArray1<'_, f64>,
    rb: PyReadonlyArray1<'_, f64>,
    ppy: f64,
) -> (f64, f64, f64, f64, f64) {
    crate::stats::scoreboard_stats(&view(&rp), &view(&rb), ppy)
}

// --- mc.rs: Monte Carlo valuation models ---------------------------------

fn to_py_vec<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray(py)
}

fn same_len(what: &str, a: usize, b: usize) -> PyResult<()> {
    if a == b {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{what}: sample arrays differ in length ({a} vs {b})"
        )))
    }
}

fn discounted_params(
    g_lo: f64,
    g_hi: f64,
    d_lo: f64,
    d_hi: f64,
    projection_years: i64,
    terminal_growth: f64,
) -> crate::mc::Discounted {
    crate::mc::Discounted {
        growth: crate::mc::Bounds { lo: g_lo, hi: g_hi },
        discount: crate::mc::Bounds { lo: d_lo, hi: d_hi },
        projection_years,
        terminal_growth,
    }
}

/// `run_monte_carlo_dcf` after the draws: equity value per share per sample.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mc_dcf<'py>(
    py: Python<'py>,
    growth: PyReadonlyArray1<'py, f64>,
    discount: PyReadonlyArray1<'py, f64>,
    g_lo: f64,
    g_hi: f64,
    d_lo: f64,
    d_hi: f64,
    projection_years: i64,
    terminal_growth: f64,
    base_fcf: f64,
    cash: f64,
    debt: f64,
    shares: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (g, d) = (view(&growth), view(&discount));
    same_len("mc_dcf", g.len(), d.len())?;
    let m = discounted_params(g_lo, g_hi, d_lo, d_hi, projection_years, terminal_growth);
    Ok(to_py_vec(
        py,
        crate::mc::dcf(&g, &d, &m, base_fcf, cash, debt, shares),
    ))
}

/// `run_monte_carlo_dcfo` / `_dni` / `_ddm` after the draws.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (growth, discount, g_lo, g_hi, d_lo, d_hi, projection_years, terminal_growth, base_per_share, net_cash_per_share=None))]
pub fn mc_discounted<'py>(
    py: Python<'py>,
    growth: PyReadonlyArray1<'py, f64>,
    discount: PyReadonlyArray1<'py, f64>,
    g_lo: f64,
    g_hi: f64,
    d_lo: f64,
    d_hi: f64,
    projection_years: i64,
    terminal_growth: f64,
    base_per_share: f64,
    net_cash_per_share: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (g, d) = (view(&growth), view(&discount));
    same_len("mc_discounted", g.len(), d.len())?;
    let m = discounted_params(g_lo, g_hi, d_lo, d_hi, projection_years, terminal_growth);
    Ok(to_py_vec(
        py,
        crate::mc::discounted_per_share(&g, &d, &m, base_per_share, net_cash_per_share),
    ))
}

/// `run_monte_carlo_mean_pe` / `_mean_pb` / `_mean_ps` after the draws.
#[pyfunction]
pub fn mc_base_times_multiple<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, f64>,
    multiple: PyReadonlyArray1<'py, f64>,
    base_lo: f64,
    base_hi: f64,
    multiple_lo: f64,
    multiple_hi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (b, m) = (view(&base), view(&multiple));
    same_len("mc_base_times_multiple", b.len(), m.len())?;
    Ok(to_py_vec(
        py,
        crate::mc::base_times_multiple(
            &b,
            &m,
            crate::mc::Bounds {
                lo: base_lo,
                hi: base_hi,
            },
            crate::mc::Bounds {
                lo: multiple_lo,
                hi: multiple_hi,
            },
        ),
    ))
}

/// `run_monte_carlo_peg` after the draws.
#[pyfunction]
pub fn mc_peg<'py>(
    py: Python<'py>,
    eps: PyReadonlyArray1<'py, f64>,
    growth: PyReadonlyArray1<'py, f64>,
    peg: PyReadonlyArray1<'py, f64>,
    eps_hi: f64,
    dividend_yield_pct: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (e, g, p) = (view(&eps), view(&growth), view(&peg));
    same_len("mc_peg", e.len(), g.len())?;
    same_len("mc_peg", e.len(), p.len())?;
    Ok(to_py_vec(
        py,
        crate::mc::peg(&e, &g, &p, eps_hi, dividend_yield_pct),
    ))
}

/// `run_monte_carlo_psg` after the draws.
#[pyfunction]
pub fn mc_psg<'py>(
    py: Python<'py>,
    sps: PyReadonlyArray1<'py, f64>,
    growth: PyReadonlyArray1<'py, f64>,
    margin: PyReadonlyArray1<'py, f64>,
    target: PyReadonlyArray1<'py, f64>,
    sps_hi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (s, g, m, t) = (view(&sps), view(&growth), view(&margin), view(&target));
    for len in [g.len(), m.len(), t.len()] {
        same_len("mc_psg", s.len(), len)?;
    }
    Ok(to_py_vec(py, crate::mc::psg(&s, &g, &m, &t, sps_hi)))
}

/// `run_monte_carlo_graham` after the draws.
#[pyfunction]
pub fn mc_graham<'py>(
    py: Python<'py>,
    growth: PyReadonlyArray1<'py, f64>,
    bond_yield: PyReadonlyArray1<'py, f64>,
    eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (g, y) = (view(&growth), view(&bond_yield));
    same_len("mc_graham", g.len(), y.len())?;
    Ok(to_py_vec(py, crate::mc::graham(&g, &y, eps)))
}

/// `run_monte_carlo_epv` after the draws.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mc_epv<'py>(
    py: Python<'py>,
    ebit: PyReadonlyArray1<'py, f64>,
    discount: PyReadonlyArray1<'py, f64>,
    ebit_hi: f64,
    d_lo: f64,
    d_hi: f64,
    tax_rate: f64,
    cash: f64,
    debt: f64,
    shares: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (e, d) = (view(&ebit), view(&discount));
    same_len("mc_epv", e.len(), d.len())?;
    Ok(to_py_vec(
        py,
        crate::mc::epv(
            &e,
            &d,
            ebit_hi,
            crate::mc::Bounds { lo: d_lo, hi: d_hi },
            tax_rate,
            cash,
            debt,
            shares,
        ),
    ))
}

/// `run_monte_carlo_lynch` after the draws.
#[pyfunction]
pub fn mc_lynch<'py>(
    py: Python<'py>,
    eps: PyReadonlyArray1<'py, f64>,
    growth: PyReadonlyArray1<'py, f64>,
    eps_hi: f64,
    dividend_yield_pct: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (e, g) = (view(&eps), view(&growth));
    same_len("mc_lynch", e.len(), g.len())?;
    Ok(to_py_vec(
        py,
        crate::mc::lynch(&e, &g, eps_hi, dividend_yield_pct),
    ))
}

type McStatsOut<'py> = (
    f64,
    f64,
    f64,
    f64,
    f64,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// `_generate_mc_stats` before smoothing: (bear, conservative, base, bull,
/// std_dev, counts[40], midpoints[40]), or None under 50 valid samples.
#[pyfunction]
pub fn mc_stats<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> Option<McStatsOut<'py>> {
    crate::mc::mc_stats(&view(&values)).map(|s| {
        (
            s.bear,
            s.conservative,
            s.base,
            s.bull,
            s.std_dev,
            Array1::from_vec(s.counts).into_pyarray(py),
            Array1::from_vec(s.midpoints).into_pyarray(py),
        )
    })
}

// --- valuation.rs: deterministic valuation arithmetic ----------------------

/// `np.median(values)`.
#[pyfunction]
pub fn np_median(values: Vec<f64>) -> f64 {
    crate::valuation::np_median(&values)
}

/// `np.mean(values)` of a list.
#[pyfunction]
pub fn np_mean(values: Vec<f64>) -> f64 {
    crate::stats::np_mean(&values)
}

/// The models' scalar faded projection: (pv list, sum(pv), last flow,
/// discounted terminal value).
#[pyfunction]
pub fn scalar_projection(
    base: f64,
    applied_growth: f64,
    discount_rate: f64,
    projection_years: i64,
    terminal_growth_rate: f64,
    numpy_terms: bool,
) -> (Vec<f64>, f64, f64, f64) {
    let p = crate::valuation::scalar_projection(
        base,
        applied_growth,
        discount_rate,
        projection_years,
        terminal_growth_rate,
        numpy_terms,
    );
    (p.pv, p.sum_pv, p.last, p.pv_terminal)
}

/// (wacc, cost_of_equity, cost_of_debt, tax_rate, beta, size_premium, weights).
type WaccOut = (f64, f64, f64, f64, f64, f64, Option<(f64, f64)>);

/// `calculate_wacc` once its inputs are read: (wacc, cost_of_equity,
/// cost_of_debt, tax_rate, beta, size_premium, weights or None).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn wacc(
    raw_beta: Option<f64>,
    risk_free_rate: f64,
    market_return: f64,
    default_tax_rate: f64,
    market_cap: Option<f64>,
    total_debt: Option<f64>,
    have_statements: bool,
    interest_expense: Option<f64>,
    income_tax_expense: Option<f64>,
    pretax_income: Option<f64>,
    operating_income: Option<f64>,
    max_cost_of_debt: f64,
    min_discount: f64,
    max_discount: f64,
    size_bands: Vec<(f64, f64)>,
) -> WaccOut {
    let w = crate::valuation::wacc(
        &crate::valuation::WaccInputs {
            raw_beta,
            risk_free_rate,
            market_return,
            default_tax_rate,
            market_cap,
            total_debt,
            have_statements,
            interest_expense,
            income_tax_expense,
            pretax_income,
            operating_income,
            max_cost_of_debt,
            min_discount,
            max_discount,
        },
        &size_bands,
    );
    (
        w.wacc,
        w.cost_of_equity,
        w.cost_of_debt,
        w.tax_rate,
        w.beta,
        w.size_premium,
        w.weights,
    )
}

/// `blended_growth_estimate`'s arithmetic: (clamped, raw), or None.
#[pyfunction]
pub fn blend_growth(
    regression: Option<f64>,
    cagr: Option<f64>,
    analyst: Option<f64>,
    shrink_weight: f64,
    base_rate: f64,
    min_growth: f64,
    max_growth: f64,
) -> Option<(f64, f64)> {
    crate::valuation::blend_growth(
        regression,
        cagr,
        analyst,
        shrink_weight,
        base_rate,
        min_growth,
        max_growth,
    )
}

/// The plausible FCF margins over (revenue, fcf) periods.
#[pyfunction]
pub fn plausible_margins(revenue: Vec<Option<f64>>, fcf: Vec<Option<f64>>) -> PyResult<Vec<f64>> {
    if revenue.len() != fcf.len() {
        return Err(PyValueError::new_err("revenue and fcf differ in length"));
    }
    Ok(crate::valuation::plausible_margins(&revenue, &fcf))
}

/// `_model_reliability` on bear/base/bull.
#[pyfunction]
pub fn model_reliability(bear: f64, base: f64, bull: f64, cv_ref: f64, floor: f64) -> f64 {
    crate::valuation::model_reliability(bear, base, bull, cv_ref, floor)
}

/// `blend_intrinsic_values`' arithmetic: (value, weights, spread_pct,
/// confidence).
#[pyfunction]
pub fn blend(
    values: Vec<f64>,
    priors: Vec<f64>,
    reliabilities: Vec<f64>,
) -> PyResult<(f64, Vec<f64>, Option<f64>, f64)> {
    if values.is_empty() || values.len() != priors.len() || values.len() != reliabilities.len() {
        return Err(PyValueError::new_err("blend needs equal, non-empty inputs"));
    }
    let b = crate::valuation::blend(&values, &priors, &reliabilities);
    Ok((b.value, b.weights, b.spread_pct, b.confidence))
}

/// `np.percentile(values, q)` (linear) of NaN-free values.
#[pyfunction]
pub fn np_percentile(mut values: Vec<f64>, q: f64) -> PyResult<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return Err(PyValueError::new_err(
            "np_percentile needs non-empty, NaN-free values",
        ));
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    Ok(crate::mc::np_percentile_sorted(&values, q))
}

// --- buffett.rs: Buffett quality metrics ---------------------------------

/// (key, value, is_int) in Python's insertion order.
type MetricsOut = Vec<(&'static str, Option<f64>, bool)>;

fn metrics_out(m: crate::buffett::Metrics) -> MetricsOut {
    use crate::buffett::Metric;
    m.into_iter()
        .map(|(k, v)| match v {
            Metric::None => (k, None, false),
            Metric::Float(f) => (k, Some(f), false),
            Metric::Int(i) => (k, Some(i as f64), true),
        })
        .collect()
}

/// `buffett_metrics._compute_generic_metrics`.
#[pyfunction]
pub fn buffett_generic_metrics(
    concepts: crate::buffett::Concepts,
    ratios: crate::buffett::Ratios,
) -> MetricsOut {
    metrics_out(crate::buffett::generic_metrics(&concepts, &ratios))
}

/// `buffett_metrics._compute_bank_metrics`.
#[pyfunction]
pub fn buffett_bank_metrics(
    concepts: crate::buffett::Concepts,
    sector: crate::buffett::Concepts,
    ratios: crate::buffett::Ratios,
) -> MetricsOut {
    metrics_out(crate::buffett::bank_metrics(&concepts, &sector, &ratios))
}

/// `buffett_metrics._compute_reit_metrics`.
#[pyfunction]
pub fn buffett_reit_metrics(
    concepts: crate::buffett::Concepts,
    sector: crate::buffett::Concepts,
    ratios: crate::buffett::Ratios,
) -> MetricsOut {
    metrics_out(crate::buffett::reit_metrics(&concepts, &sector, &ratios))
}

/// `statistics.pstdev` of finite floats, exposed for its parity test.
#[pyfunction]
pub fn py_pstdev(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("py_pstdev needs finite values"));
    }
    Ok(crate::buffett::py_pstdev(&values))
}

/// `buffett_metrics._total_debt`: short + long-term debt at one period.
#[pyfunction]
pub fn buffett_total_debt(concepts: crate::buffett::Concepts, period: &str) -> Option<f64> {
    crate::buffett::total_debt(&concepts, period)
}

/// `buffett_metrics._cagr` over raw values, oldest first.
#[pyfunction]
pub fn buffett_cagr(values: Vec<Option<f64>>) -> Option<f64> {
    crate::buffett::py_cagr(&values)
}

// --- rank.rs: cross-sectional scoring --------------------------------------

/// `buffett_rank._winsorised_percentile` on float values (NaN = missing).
#[pyfunction]
pub fn winsorised_percentile<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    higher_is_better: bool,
) -> Bound<'py, PyArray1<f64>> {
    to_py_vec(
        py,
        crate::rank::winsorised_percentile(&view(&values), higher_is_better),
    )
}

/// Row-wise NaN-skipping mean of equal-length columns.
#[pyfunction]
pub fn row_mean<'py>(
    py: Python<'py>,
    columns: Vec<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cols: Vec<Vec<f64>> = columns.iter().map(|c| view(c).into_owned()).collect();
    if cols.windows(2).any(|w| w[0].len() != w[1].len()) {
        return Err(PyValueError::new_err("row_mean: columns differ in length"));
    }
    Ok(to_py_vec(py, crate::rank::row_mean(&cols)))
}

/// Weight-renormalised blend of equal-length columns.
#[pyfunction]
pub fn weighted_renormalised<'py>(
    py: Python<'py>,
    columns: Vec<PyReadonlyArray1<'py, f64>>,
    weights: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cols: Vec<Vec<f64>> = columns.iter().map(|c| view(c).into_owned()).collect();
    if cols.len() != weights.len() || cols.windows(2).any(|w| w[0].len() != w[1].len()) {
        return Err(PyValueError::new_err(
            "weighted_renormalised: mismatched inputs",
        ));
    }
    Ok(to_py_vec(
        py,
        crate::rank::weighted_renormalised(&cols, &weights),
    ))
}

// --- projection.rs: portfolio projections ----------------------------------

/// (annual_return_pct, annual_volatility_pct, per-horizon (median, median
/// return %, expected, p10, p25, p75, p90)).
type ProjectionOut = (f64, f64, Vec<(f64, f64, f64, f64, f64, f64, f64)>);

/// `compute_projection` on prepared returns; None where it is unavailable.
#[pyfunction]
#[pyo3(signature = (returns, periods_per_year, current_value, horizons, benchmark_log_return=None))]
pub fn projection(
    returns: PyReadonlyArray1<'_, f64>,
    periods_per_year: f64,
    current_value: f64,
    horizons: Vec<f64>,
    benchmark_log_return: Option<f64>,
) -> Option<ProjectionOut> {
    let p = crate::projection::project(
        &view(&returns),
        periods_per_year,
        current_value,
        &horizons,
        benchmark_log_return,
    )?;
    let rows = p
        .points
        .iter()
        .map(|pt| {
            let b = pt.bands;
            (
                pt.median_value,
                pt.median_return_pct,
                pt.expected_value,
                b[0],
                b[1],
                b[2],
                b[3],
            )
        })
        .collect();
    Some((p.annual_return_pct, p.annual_volatility_pct, rows))
}

/// `walk_forward_errors`: rows of (years, anchor position, z, actual_log,
/// median_log).
#[pyfunction]
#[pyo3(signature = (wealth, ticks, bench, horizons, min_history_years, step=None))]
pub fn walk_forward(
    py: Python<'_>,
    wealth: PyReadonlyArray1<'_, f64>,
    ticks: Option<PyReadonlyArray1<'_, i64>>,
    bench: Option<PyReadonlyArray1<'_, f64>>,
    horizons: Vec<f64>,
    min_history_years: f64,
    step: Option<usize>,
) -> PyResult<Vec<crate::projection::ErrorRow>> {
    let w = view(&wealth);
    let t = ticks.as_ref().map(view);
    let b = bench.as_ref().map(view);
    for len in [t.as_ref().map(|x| x.len()), b.as_ref().map(|x| x.len())]
        .into_iter()
        .flatten()
    {
        if len != w.len() {
            return Err(PyValueError::new_err(
                "walk_forward: ticks/bench must match wealth",
            ));
        }
    }
    Ok(py.detach(|| {
        crate::projection::walk_forward_errors(
            &w,
            t.as_deref(),
            b.as_deref(),
            &horizons,
            min_history_years,
            step,
        )
    }))
}

/// `summarize_errors`' per-horizon statistics.
#[pyfunction]
pub fn summarize_errors(
    z: Vec<f64>,
    actual_log: Vec<f64>,
    median_log: Vec<f64>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    if z.len() < 3 || z.len() != actual_log.len() || z.len() != median_log.len() {
        return Err(PyValueError::new_err(
            "summarize_errors: needs 3+ equal-length rows",
        ));
    }
    let s = crate::projection::summarize(&z, &actual_log, &median_log);
    Ok((s[0], s[1], s[2], s[3], s[4], s[5], s[6]))
}

// --- records.rs: valuation bands, trend signal, stress response -----------

/// `valuation_history.bands`' statistics: (median, p25, p75, low, high,
/// percentile of `now`).
#[pyfunction]
pub fn band_stats(population: Vec<f64>, now: f64) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    if population.is_empty() || population.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "band_stats needs a finite, non-empty population",
        ));
    }
    Ok(crate::records::band_stats(&population, now))
}

/// `strategies.evaluate_trend_signal`'s averages: (sma, provisional_sma,
/// flip_close, per-month moving average or None).
#[pyfunction]
pub fn trend_sma(
    completed: Vec<f64>,
    sma_months: usize,
    latest_close: f64,
) -> PyResult<(f64, f64, f64, Vec<Option<f64>>)> {
    if sma_months == 0 || completed.len() < sma_months {
        return Err(PyValueError::new_err(
            "trend_sma needs at least sma_months >= 1 closes",
        ));
    }
    Ok(crate::records::trend_sma(
        &completed,
        sma_months,
        latest_close,
    ))
}

/// `track_record.stress_response` for one metric and window: (peak_year,
/// trough_year, peak, trough, change_pct, recovered_year), or None.
#[pyfunction]
pub fn stress_window(
    annual: Vec<(i64, f64)>,
    peak_years: Vec<i64>,
    trough_years: Vec<i64>,
) -> Option<(i64, i64, f64, f64, f64, Option<i64>)> {
    crate::records::stress_window(&annual, &peak_years, &trough_years).map(|s| {
        (
            s.peak_year,
            s.trough_year,
            s.peak,
            s.trough,
            s.change_pct,
            s.recovered_year,
        )
    })
}

// --- fifo.rs: FIFO tax lots and realized gains -----------------------------

/// The open lots per holding: (holding ids in `holdings_long` order, offsets
/// into the lot arrays, qty, cost per share, opening event).
type FifoLots = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<usize>);

/// `calculate_fifo_lots_and_gains`' lot bookkeeping over one event per row.
/// Columns by kind (0 transfer, 1 split, 2 open, 3 reduce basis, 4 sell):
/// `a` holding / source / symbol, `b` destination, `x` qty / ratio / total,
/// `y` cost per share / price, `z` FX at purchase / commission, `w` sale FX,
/// `day` purchase ordinal, `rank` id rank. Returns the sales (proceeds, cost
/// and gain, local then display) and the open lots.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fifo_lots(
    py: Python<'_>,
    kind: Vec<u8>,
    a: Vec<usize>,
    b: Vec<usize>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    day: Vec<i64>,
    rank: Vec<i64>,
    symbol_of: Vec<usize>,
    tol: f64,
) -> PyResult<(Vec<[f64; 6]>, FifoLots)> {
    use crate::fifo::{Event, Lot};
    let n = kind.len();
    for (name, len) in [
        ("a", a.len()),
        ("b", b.len()),
        ("x", x.len()),
        ("y", y.len()),
        ("z", z.len()),
        ("w", w.len()),
        ("day", day.len()),
        ("rank", rank.len()),
    ] {
        if len != n {
            return Err(PyValueError::new_err(format!(
                "fifo_lots: {name} has {len} entries, kind has {n}"
            )));
        }
    }
    let holdings = symbol_of.len();
    let check = |h: usize| -> PyResult<usize> {
        if h < holdings {
            Ok(h)
        } else {
            Err(PyIndexError::new_err(format!(
                "fifo_lots: holding {h} out of {holdings}"
            )))
        }
    };
    let mut events = Vec::with_capacity(n);
    for i in 0..n {
        events.push(match kind[i] {
            0 => Event::Transfer {
                from: check(a[i])?,
                to: check(b[i])?,
                qty: x[i],
            },
            // A symbol with no holding yet matches no lots.
            1 => Event::Split {
                symbol: a[i],
                ratio: x[i],
            },
            2 => Event::Open {
                holding: check(a[i])?,
                lot: Lot {
                    qty: x[i],
                    cost_per_share_local_net: y[i],
                    purchase_date: day[i],
                    purchase_fx_to_display: z[i],
                    id_rank: rank[i],
                    source: i,
                },
            },
            3 => Event::ReduceBasis {
                holding: check(a[i])?,
                total_amt: x[i],
            },
            4 => Event::Sell {
                holding: check(a[i])?,
                qty: x[i],
                price_local: y[i],
                commission_local: z[i],
                fx: w[i],
            },
            k => {
                return Err(PyValueError::new_err(format!(
                    "fifo_lots: unknown kind {k}"
                )));
            }
        });
    }
    let (sales, book) = py.detach(|| crate::fifo::run(&events, &symbol_of, tol));
    let sales = sales
        .into_iter()
        .map(|s| {
            [
                s.proceeds_local,
                s.cost_basis_local,
                s.realized_gain_local,
                s.proceeds_display,
                s.cost_basis_display,
                s.realized_gain_display,
            ]
        })
        .collect();
    let (mut keys, mut offsets) = (Vec::new(), vec![0]);
    let (mut qty, mut cps, mut source) = (Vec::new(), Vec::new(), Vec::new());
    for (h, lots) in book.into_ordered() {
        keys.push(h);
        for l in lots {
            qty.push(l.qty);
            cps.push(l.cost_per_share_local_net);
            source.push(l.source);
        }
        offsets.push(qty.len());
    }
    Ok((sales, (keys, offsets, qty, cps, source)))
}

// --- history.rs: the daily portfolio history ------------------------------

/// A C-ordered buffer of `a`: borrowed when it already is one.
fn standard<'a, D: numpy::ndarray::Dimension>(
    a: &'a numpy::PyReadonlyArray<'_, f64, D>,
) -> Cow<'a, [f64]> {
    match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
    }
}

/// `_value_daily_holdings_vectorized`'s valuation: each day's stocks
/// (qty × price × FX, last trade price where the market's is NaN) plus cash,
/// with empty and excluded cells at zero.
#[pyfunction]
#[pyo3(signature = (qty, last_prices, prices, fx, cash, included))]
pub fn value_holdings_grid<'py>(
    py: Python<'py>,
    qty: numpy::PyReadonlyArray3<'py, f64>,
    last_prices: Option<numpy::PyReadonlyArray3<'py, f64>>,
    prices: numpy::PyReadonlyArray2<'py, f64>,
    fx: numpy::PyReadonlyArray2<'py, f64>,
    cash: numpy::PyReadonlyArray2<'py, f64>,
    included: Vec<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = qty.shape();
    let g = crate::history::Grid {
        days: shape[0],
        symbols: shape[1],
        accounts: shape[2],
    };
    let bad = |what: &str, got: &[usize], want: &[usize]| {
        PyValueError::new_err(format!(
            "value_holdings_grid: {what} has shape {got:?}, expected {want:?}"
        ))
    };
    if let Some(l) = &last_prices
        && l.shape() != shape
    {
        return Err(bad("last_prices", l.shape(), shape));
    }
    if prices.shape() != [g.days, g.symbols] {
        return Err(bad("prices", prices.shape(), &[g.days, g.symbols]));
    }
    for (what, arr) in [("fx", &fx), ("cash", &cash)] {
        if arr.shape() != [g.days, g.accounts] {
            return Err(bad(what, arr.shape(), &[g.days, g.accounts]));
        }
    }
    if included.len() != g.accounts {
        return Err(PyValueError::new_err(
            "value_holdings_grid: included needs one flag per account",
        ));
    }
    let (q, p, f, c) = (
        standard(&qty),
        standard(&prices),
        standard(&fx),
        standard(&cash),
    );
    let l = last_prices.as_ref().map(standard);
    let out = py.detach(|| crate::history::value_grid(g, &q, l.as_deref(), &p, &f, &c, &included));
    Ok(Array1::from_vec(out).into_pyarray(py))
}

/// `_calculate_accumulated_gains_and_resample`'s portfolio scan: (gain
/// factors, accumulated gain, drawdown %).
#[pyfunction]
pub fn portfolio_twr<'py>(
    py: Python<'py>,
    daily_return: PyReadonlyArray1<'py, f64>,
) -> TripleOut<'py> {
    let (g, a, d) = crate::history::portfolio_twr(&view(&daily_return));
    triple(py, g, a, d)
}

/// A benchmark's accumulated gain from its price column, or None when it has
/// no prices.
#[pyfunction]
pub fn benchmark_accumulated<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> Option<Bound<'py, PyArray1<f64>>> {
    crate::history::benchmark_accumulated(&view(&prices))
        .map(|v| Array1::from_vec(v).into_pyarray(py))
}

/// The money-weighted columns: (cumulative net flow, absolute gain, ROI %).
#[pyfunction]
pub fn money_weighted<'py>(
    py: Python<'py>,
    value: PyReadonlyArray1<'py, f64>,
    net_flow: PyReadonlyArray1<'py, f64>,
) -> PyResult<TripleOut<'py>> {
    let (v, f) = (view(&value), view(&net_flow));
    if v.len() != f.len() {
        return Err(PyValueError::new_err(
            "money_weighted: value and net_flow differ in length",
        ));
    }
    let (c, g, r) = crate::history::money_weighted(&v, &f);
    Ok(triple(py, c, g, r))
}

type TripleOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

fn triple(py: Python<'_>, a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> TripleOut<'_> {
    (
        Array1::from_vec(a).into_pyarray(py),
        Array1::from_vec(b).into_pyarray(py),
        Array1::from_vec(c).into_pyarray(py),
    )
}

// --- asof.rs: historical price lookups -------------------------------------

/// `get_historical_price` for one frame: built once, then asked per date.
#[pyclass(name = "PriceTable", module = "investa_core", frozen)]
pub struct PyPriceTable(crate::asof::PriceTable);

#[pymethods]
impl PyPriceTable {
    /// `ticks`/`values`: the rows with no NaN in any column (ascending UTC ns
    /// and their prices); `first_tick`/`first_value`: the frame's first row.
    #[new]
    fn new(
        ticks: PyReadonlyArray1<'_, i64>,
        values: PyReadonlyArray1<'_, f64>,
        first_tick: i64,
        first_value: f64,
    ) -> PyResult<Self> {
        let (t, v) = (view(&ticks).into_owned(), view(&values).into_owned());
        if t.len() != v.len() {
            return Err(PyValueError::new_err(
                "PriceTable: ticks and values differ in length",
            ));
        }
        if t.windows(2).any(|w| w[1] < w[0]) {
            return Err(PyValueError::new_err("PriceTable: ticks must be ascending"));
        }
        Ok(PyPriceTable(crate::asof::PriceTable::new(
            t,
            v,
            first_tick,
            first_value,
        )))
    }

    /// The price on UTC tick `t`, or NaN.
    fn asof(&self, t: i64) -> f64 {
        self.0.asof(t)
    }
}

// --- cashflows.rs: IRR and MWR cash flows ----------------------------------

fn flow_rows(
    kind: &[u8],
    qty: &[f64],
    price: &[f64],
    commission: &[f64],
    total: &[f64],
    direction: &[i8],
) -> PyResult<Vec<crate::cashflows::Row>> {
    let n = kind.len();
    if [
        qty.len(),
        price.len(),
        commission.len(),
        total.len(),
        direction.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(PyValueError::new_err(
            "cash flows: columns differ in length",
        ));
    }
    (0..n)
        .map(|i| {
            let k = crate::cashflows::Kind::from_code(kind[i]).ok_or_else(|| {
                PyValueError::new_err(format!("cash flows: unknown kind {}", kind[i]))
            })?;
            Ok(crate::cashflows::Row {
                kind: k,
                qty: qty[i],
                price: price[i],
                commission: commission[i],
                total: total[i],
                direction: direction[i],
            })
        })
        .collect()
}

/// `get_cash_flows_for_symbol_account`'s flow per row (sorted by date, id).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn holding_flows(
    kind: Vec<u8>,
    qty: Vec<f64>,
    price: Vec<f64>,
    commission: Vec<f64>,
    total: Vec<f64>,
    direction: Vec<i8>,
    is_cash: bool,
    shortable: bool,
    transfer_is_flow: bool,
) -> PyResult<Vec<f64>> {
    let rows = flow_rows(&kind, &qty, &price, &commission, &total, &direction)?;
    Ok(crate::cashflows::holding_flows(
        &rows,
        is_cash,
        shortable,
        transfer_is_flow,
    ))
}

/// `get_cash_flows_for_mwr`'s local-currency flow per row, and which inputs
/// it used (bits: 1 qty, 2 price, 4 total).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mwr_flows(
    kind: Vec<u8>,
    qty: Vec<f64>,
    price: Vec<f64>,
    commission: Vec<f64>,
    total: Vec<f64>,
    direction: Vec<i8>,
    is_cash: Vec<bool>,
    portfolio_basis: bool,
) -> PyResult<Vec<(f64, u8)>> {
    let rows = flow_rows(&kind, &qty, &price, &commission, &total, &direction)?;
    if is_cash.len() != rows.len() {
        return Err(PyValueError::new_err(
            "mwr_flows: is_cash differs in length",
        ));
    }
    Ok(crate::cashflows::mwr_flows(
        &rows,
        &is_cash,
        portfolio_basis,
    ))
}

// --- ratios.rs: the ratio history ----------------------------------------

/// `calculate_key_ratios_timeseries`' arithmetic. `inputs` holds 48 lookups
/// per period (NaN where `_get_statement_value` returned None); `flags` holds
/// (year-earlier balance sheet, year-earlier income statement, cash flow) per
/// period. Returns each period's 28 fields, None where the original stores None.
#[pyfunction]
pub fn key_ratios(
    inputs: Vec<Vec<f64>>,
    flags: Vec<(bool, bool, bool)>,
) -> PyResult<Vec<Vec<Option<f64>>>> {
    use crate::ratios::{Flags, period_ratios, slot};
    if inputs.len() != flags.len() {
        return Err(PyValueError::new_err(
            "key_ratios: one flag triple per period",
        ));
    }
    inputs
        .iter()
        .zip(flags)
        .map(|(row, (prev_balance, prev_income, cashflow))| {
            if row.len() != slot::COUNT {
                return Err(PyValueError::new_err(format!(
                    "key_ratios: {} lookups per period, expected {}",
                    row.len(),
                    slot::COUNT
                )));
            }
            let mut a = [None; slot::COUNT];
            for (k, &v) in row.iter().enumerate() {
                a[k] = if v.is_nan() { None } else { Some(v) };
            }
            let f = Flags {
                prev_balance,
                prev_income,
                cashflow,
            };
            Ok(period_ratios(&a, f).to_vec())
        })
        .collect()
}

/// The output field names, in order.
#[pyfunction]
pub fn key_ratio_fields() -> Vec<&'static str> {
    crate::ratios::FIELDS.to_vec()
}

// --- the daily price archive -------------------------------------------------

/// Resolve SQLite through Python's `_sqlite3` extension (its `__file__`).
/// False when that image is not loaded or does not export the C API; the
/// reader then stays off and `market_db` reads through pandas.
#[pyfunction]
pub fn sqlite_bind(module_path: &str) -> bool {
    crate::sqlite_read::bind(module_path)
}

type OhlcvOut<'py> = (
    Vec<String>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Vec<Bound<'py, PyAny>>,
);

/// The daily rows for `symbols`, as (symbols, offsets, dates in ns, [open,
/// high, low, close, adj_close, volume]); each column is int64 when every
/// cell was an INTEGER and float64 otherwise. None when the rows hold values
/// only the pandas reader handles; an OSError when the database cannot be read.
#[pyfunction]
pub fn read_daily_ohlcv<'py>(
    py: Python<'py>,
    db_path: &str,
    symbols: Vec<String>,
    interval: &str,
    start: &str,
    end: &str,
) -> PyResult<Option<OhlcvOut<'py>>> {
    use crate::sqlite_read::{Column, Outcome};
    let outcome = py
        .detach(|| crate::sqlite_read::read_daily_ohlcv(db_path, &symbols, interval, start, end))
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.0))?;
    let batch = match outcome {
        Outcome::Batch(b) => b,
        Outcome::Unsupported(_) => return Ok(None),
    };
    let columns = batch
        .columns
        .into_iter()
        .map(|c| match c {
            Column::Int(v) => Array1::from_vec(v).into_pyarray(py).into_any(),
            Column::Real(v) => Array1::from_vec(v).into_pyarray(py).into_any(),
        })
        .collect();
    Ok(Some((
        batch.symbols,
        Array1::from_vec(batch.offsets).into_pyarray(py),
        Array1::from_vec(batch.dates_ns).into_pyarray(py),
        columns,
    )))
}
