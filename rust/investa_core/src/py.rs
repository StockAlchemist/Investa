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
