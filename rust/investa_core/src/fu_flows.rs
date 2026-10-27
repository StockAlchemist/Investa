//! The cash-flow builders of `finutils`: `CashFlowLedger` (one transactions
//! frame read once for any number of per-holding IRR flows), the per-holding
//! builder on top of it, and the account/portfolio MWR builder.
//!
//! Ported line for line from `src/finutils.py` (frozen in
//! `tests/reference/finutils_p16.py`). The ledger keeps the original's lazy
//! caches — symbol groups, per-account masks built with the original's own
//! pandas expressions, the parsed rows, the transfer test (whose failure is
//! raised again for every account, as before) — and the per-row flow rules
//! were already Rust (`cashflows.rs`). Flows are summed per day with Python's
//! own arithmetic, so a NumPy market value keeps the total a NumPy scalar.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyString, PyTuple};

use crate::fr::{builtin, fl, none, pybool};
use crate::fu::fu;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{log, setitem, unpack};

/// `_cell_text(values, i)`: `str(values[i])`, "" for a missing column.
fn cell_text<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    i: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if values.is_none() {
        return Ok(PyString::new(py, "").into_any());
    }
    Ok(values.get_item(i)?.str()?.into_any())
}

fn empty_result<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, [PyList::empty(py), PyList::empty(py)])?.into_any())
}

/// `df[name].tolist() if name in df.columns else None`
fn texts<'py>(df: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    if df.getattr("columns")?.contains(name)? {
        df.get_item(name)?.call_method0("tolist")
    } else {
        Ok(none(df.py()))
    }
}

/// The tail both builders share: the market value as the final flow, then
/// the refusals (fewer than two dates, all of one sign).
fn finish<'py>(
    py: Python<'py>,
    final_dates: Bound<'py, PyList>,
    final_flows: Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyAny>> {
    if final_dates.len() < 2 {
        return empty_result(py);
    }
    let abs = builtin(py, "abs")?;
    let mut non_zero = Vec::new();
    for cf in final_flows.iter() {
        if abs.call1((&cf,))?.gt(1e-9)? {
            non_zero.push(cf);
        }
    }
    let mut all_nonneg = true;
    for cf in &non_zero {
        if !cf.ge(-1e-9)? {
            all_nonneg = false;
            break;
        }
    }
    let mut all_nonpos = false;
    if !all_nonneg {
        all_nonpos = true;
        for cf in &non_zero {
            if !cf.le(1e-9)? {
                all_nonpos = false;
                break;
            }
        }
    }
    if non_zero.is_empty() || all_nonneg || all_nonpos {
        return empty_result(py);
    }
    Ok(PyTuple::new(py, [final_dates, final_flows])?.into_any())
}

/// `final_flows[-1] += x`
fn add_to_last<'py>(flows: &Bound<'py, PyList>, x: &Bound<'py, PyAny>) -> PyResult<()> {
    let last = flows.len() - 1;
    let v = flows.get_item(last)?.add(x)?;
    flows.set_item(last, v)
}

#[pyclass(module = "investa_core")]
pub struct CashFlowLedger {
    #[pyo3(get)]
    df: Py<PyAny>,
    by_symbol: Option<Py<PyDict>>,
    account_masks: Py<PyDict>,
    parsed: Option<Py<PyDict>>,
    is_transfer: Option<(bool, Py<PyAny>)>,
    ids: Option<Py<PyAny>>,
}

impl CashFlowLedger {
    fn frame<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.df.bind(py).clone()
    }
}

#[pymethods]
impl CashFlowLedger {
    #[new]
    fn new(py: Python<'_>, transactions_df: Py<PyAny>) -> Self {
        CashFlowLedger {
            df: transactions_df,
            by_symbol: None,
            account_masks: PyDict::new(py).unbind(),
            parsed: None,
            is_transfer: None,
            ids: None,
        }
    }

    fn _symbol_positions<'py>(
        &mut self,
        py: Python<'py>,
        symbol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let df = self.frame(py);
        if !symbol.is_instance_of::<PyString>() {
            let eq = df.get_item("Symbol")?.rich_compare(symbol, CompareOp::Eq)?;
            return fu(py)?
                .getattr("np")?
                .call_method1("flatnonzero", (eq.call_method0("to_numpy")?,))?
                .call_method0("tolist");
        }
        if self.by_symbol.is_none() {
            let groups = PyDict::new(py);
            for (i, s) in df
                .get_item("Symbol")?
                .call_method0("tolist")?
                .try_iter()?
                .enumerate()
            {
                let s = s?;
                if s.is_instance_of::<PyString>() {
                    match groups.get_item(&s)? {
                        Some(l) => l.call_method1("append", (i,))?,
                        None => {
                            groups.set_item(&s, PyList::new(py, [i])?)?;
                            continue;
                        }
                    };
                }
            }
            self.by_symbol = Some(groups.unbind());
        }
        self.by_symbol
            .as_ref()
            .expect("grouped")
            .bind(py)
            .as_any()
            .call_method1("get", (symbol, PyList::empty(py)))
    }

    fn _account_mask<'py>(
        &mut self,
        py: Python<'py>,
        account: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let masks = self.account_masks.bind(py).clone();
        if masks.contains(account)? {
            return masks.as_any().get_item(account);
        }
        let df = self.frame(py);
        let columns = df.getattr("columns")?;
        let mut mask = df
            .get_item("Account")?
            .rich_compare(account, CompareOp::Eq)?;
        if columns.contains("To Account")? {
            mask = mask.bitor(
                df.get_item("To Account")?
                    .rich_compare(account, CompareOp::Eq)?,
            )?;
        }
        // Legacy transfers name the account in the Note, read as a regex.
        if columns.contains("Note")? {
            let note = df.get_item("Note")?;
            let contains_kw = kwargs(
                py,
                &[("case", &pybool(py, false)), ("na", &pybool(py, false))],
            )?;
            let rhs = match self._transfer_notes_naming(py, &df, &note, account, &contains_kw)? {
                Some(rhs) => rhs,
                None => {
                    let named = note.getattr("str")?.call_method(
                        "contains",
                        (account,),
                        Some(&contains_kw),
                    )?;
                    named.bitand(self._transfer_rows(py)?)?
                }
            };
            mask = py.import("operator")?.call_method1("ior", (mask, rhs))?;
        }
        let mask = mask.call_method(
            "to_numpy",
            (),
            Some(&kwargs(py, &[("dtype", &builtin(py, "bool")?)])?),
        )?;
        masks.set_item(account, &mask)?;
        Ok(mask)
    }

    /// `df["Note"].str.contains(account, case=False, na=False) & transfer_rows`
    /// with the regex run on the transfer rows alone (a few dozen of
    /// thousands): every other row is False after the `&` whatever its note
    /// says. Taken only where it cannot differ from the full call: an account
    /// that compiles, both ways, to a pattern with no groups (so the full call
    /// neither raises nor warns before the transfer rows are read), an object
    /// Note column of strings and missing values (so the subset passes the
    /// `.str` accessor's check exactly when the column does), and a plain bool
    /// transfer mask. `None` sends the caller to the full call.
    fn _transfer_notes_naming<'py>(
        &mut self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        note: &Bound<'py, PyAny>,
        account: &Bound<'py, PyAny>,
        contains_kw: &Bound<'py, PyDict>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if !account.is_exact_instance_of::<pyo3::types::PyString>() {
            return Ok(None);
        }
        let re = py.import("re")?;
        for flags in [0, re.getattr("IGNORECASE")?.extract::<i64>()?] {
            match re.call_method1("compile", (account, flags)) {
                Ok(p) if p.getattr("groups")?.extract::<i64>()? == 0 => {}
                _ => return Ok(None),
            }
        }
        let np = py.import("numpy")?;
        let object = np.call_method1("dtype", ("O",))?;
        if !note.getattr("dtype")?.eq(&object)? {
            return Ok(None);
        }
        let inferred = py.import("pandas.api.types")?.call_method(
            "infer_dtype",
            (note,),
            Some(&kwargs(py, &[("skipna", &pybool(py, true))])?),
        )?;
        if !matches!(inferred.extract::<&str>()?, "string" | "empty") {
            return Ok(None);
        }
        let transfer = self._transfer_rows(py)?;
        if !transfer
            .getattr("dtype")?
            .eq(np.call_method1("dtype", ("bool",))?)?
        {
            return Ok(None);
        }
        let positions = np.call_method1("flatnonzero", (transfer.call_method0("to_numpy")?,))?;
        let named = note
            .getattr("iloc")?
            .get_item(&positions)?
            .getattr("str")?
            .call_method("contains", (account,), Some(contains_kw))?;
        let hits = np.call_method1("zeros", (df.len()?, builtin(py, "bool")?))?;
        hits.set_item(&positions, named.call_method0("to_numpy")?)?;
        let series = py.import("pandas")?.getattr("Series")?;
        Ok(Some(series.call(
            (hits,),
            Some(&kwargs(py, &[("index", &df.getattr("index")?)])?),
        )?))
    }

    fn _transfer_rows<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.is_transfer.is_none() {
            let df = self.frame(py);
            let r = df
                .get_item("Type")
                .and_then(|t| t.getattr("str")?.call_method0("lower"))
                .and_then(|l| l.rich_compare("transfer", CompareOp::Eq));
            self.is_transfer = Some(match r {
                Ok(v) => (true, v.unbind()),
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    (false, e.into_value(py).into_any())
                }
                Err(e) => return Err(e),
            });
        }
        let (ok, value) = self.is_transfer.as_ref().expect("set");
        if !ok {
            return Err(PyErr::from_value(value.bind(py).clone()));
        }
        Ok(value.bind(py).clone())
    }

    fn _rows<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.parsed.is_none() {
            let df = self.frame(py);
            let m = fu(py)?;
            let types = texts(&df, "Type")?;
            let flow_kinds = m.getattr("_FLOW_KINDS")?;
            let n = df.len()?;
            let kinds = PyList::empty(py);
            for i in 0..n {
                let key = cell_text(py, &types, i)?
                    .call_method0("lower")?
                    .call_method0("strip")?;
                kinds.append(flow_kinds.call_method1("get", (key, 0))?)?;
            }
            let np = m.getattr("np")?;
            let parsed = PyDict::new(py);
            parsed.set_item(
                "kind",
                np.call_method(
                    "array",
                    (kinds,),
                    Some(&kwargs(py, &[("dtype", &np.getattr("uint8")?)])?),
                )?,
            )?;
            let numeric = m.getattr("_numeric_column")?;
            parsed.set_item("qty", numeric.call1((&df, "Quantity"))?)?;
            parsed.set_item("price", numeric.call1((&df, "Price/Share"))?)?;
            parsed.set_item(
                "commission",
                m.getattr("_commission_column")?.call1((&df,))?,
            )?;
            parsed.set_item("total", numeric.call1((&df, "Total Amount"))?)?;
            parsed.set_item("date", df.get_item("Date")?.call_method0("tolist")?)?;
            parsed.set_item("account", texts(&df, "Account")?)?;
            parsed.set_item("to_account", texts(&df, "To Account")?)?;
            parsed.set_item("note", texts(&df, "Note")?)?;
            self.parsed = Some(parsed.unbind());
        }
        Ok(self.parsed.as_ref().expect("parsed").bind(py).clone())
    }

    fn _sorted<'py>(
        &mut self,
        py: Python<'py>,
        positions: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let df = self.frame(py);
        if !df.getattr("columns")?.contains("original_index")? {
            return Err(PyKeyError::new_err("original_index"));
        }
        let dates = self._rows(py)?.as_any().get_item("date")?;
        if self.ids.is_none() {
            self.ids = Some(
                df.get_item("original_index")?
                    .call_method0("tolist")?
                    .unbind(),
            );
        }
        let ids = self.ids.as_ref().expect("ids").bind(py).clone();
        let pd = py.import("pandas")?;
        let timestamp = pd.getattr("Timestamp")?;
        let nat = pd.getattr("NaT")?;
        let int_t = builtin(py, "int")?;
        let np_integer = fu(py)?.getattr("np")?.getattr("integer")?;
        let mut clean = true;
        for p in positions.try_iter()? {
            let d = dates.get_item(p?)?;
            if !d.is_instance(&timestamp)? || d.is(&nat) {
                clean = false;
                break;
            }
        }
        if clean {
            for p in positions.try_iter()? {
                let v = ids.get_item(p?)?;
                if !(v.is_instance(&int_t)? || v.is_instance(&np_integer)?) {
                    clean = false;
                    break;
                }
            }
        }
        if clean {
            let (d, i) = (dates.clone().unbind(), ids.clone().unbind());
            let key =
                PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<Py<PyAny>> {
                    let py = args.py();
                    let p = args.get_item(0)?;
                    Ok(
                        PyTuple::new(py, [d.bind(py).get_item(&p)?, i.bind(py).get_item(&p)?])?
                            .into_any()
                            .unbind(),
                    )
                })?;
            return builtin(py, "sorted")?
                .call((positions,), Some(&kwargs(py, &[("key", key.as_any())])?));
        }
        let keys = df
            .getattr("iloc")?
            .get_item(positions)?
            .get_item(PyList::new(py, ["Date", "original_index"])?)?
            .call_method0("copy")?;
        setitem(&keys, "_position", positions)?;
        keys.call_method(
            "sort_values",
            (),
            Some(&kwargs(
                py,
                &[(
                    "by",
                    &PyList::new(py, ["Date", "original_index"])?.into_any(),
                )],
            )?),
        )?
        .get_item("_position")?
        .call_method0("tolist")
    }

    #[pyo3(signature = (symbol, account, final_market_value, is_transfer_a_flow, report_date=None))]
    fn flows_for<'py>(
        &mut self,
        py: Python<'py>,
        symbol: &Bound<'py, PyAny>,
        account: &Bound<'py, PyAny>,
        final_market_value: &Bound<'py, PyAny>,
        is_transfer_a_flow: &Bound<'py, PyAny>,
        report_date: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let nothing = none(py);
        let report_date = report_date.unwrap_or(&nothing);
        let m = fu(py)?;
        let mut positions = self._symbol_positions(py, symbol)?;
        if !account.is_none() {
            let mask = self._account_mask(py, account)?;
            let kept = PyList::empty(py);
            for p in positions.try_iter()? {
                let p = p?;
                if mask.get_item(&p)?.is_truthy()? {
                    kept.append(p)?;
                }
            }
            positions = kept.into_any();
        }
        if !positions.is_truthy()? {
            return empty_result(py);
        }
        let order = self._sorted(py, &positions)?;
        let rows = self._rows(py)?;
        let rows = rows.as_any();
        let np = m.getattr("np")?;
        let take = np.call_method(
            "asarray",
            (&order,),
            Some(&kwargs(py, &[("dtype", &np.getattr("intp")?)])?),
        )?;
        let kind = rows
            .get_item("kind")?
            .get_item(&take)?
            .call_method0("tolist")?;
        let kinds: Vec<u8> = kind.extract()?;
        let order_list: Vec<Bound<'py, PyAny>> = order.try_iter()?.collect::<PyResult<_>>()?;
        let row_dates = rows.get_item("date")?;
        let tx_dates: Vec<Bound<'py, PyAny>> = order_list
            .iter()
            .map(|p| row_dates.get_item(p)?.call_method0("date"))
            .collect::<PyResult<_>>()?;

        // A transfer's direction, when it counts as a flow.
        let transfer: u8 = m.getattr("_TRANSFER")?.extract()?;
        let mut direction = vec![0i8; order_list.len()];
        if is_transfer_a_flow.is_truthy()? {
            let o = Ops::new(py)?;
            for (j, p) in order_list.iter().enumerate() {
                if kinds[j] != transfer {
                    continue;
                }
                let p: usize = p.extract()?;
                let to_account =
                    cell_text(py, &rows.get_item("to_account")?, p)?.call_method0("strip")?;
                let acct_col =
                    cell_text(py, &rows.get_item("account")?, p)?.call_method0("strip")?;
                let note = cell_text(py, &rows.get_item("note")?, p)?.call_method0("lower")?;
                let lower = o.fmt(&account.call_method0("lower")?)?;
                let is_from_note = note.contains(format!("from {lower}"))?;
                let is_to_note = note.contains(format!("to {lower}"))?;
                let is_incoming = to_account.eq(account)? || is_to_note;
                let is_outgoing = acct_col.eq(account)? || is_from_note;
                direction[j] = if is_incoming && !is_outgoing {
                    -1
                } else if is_outgoing && !is_incoming {
                    1
                } else if to_account.eq(account)? {
                    -1
                } else {
                    1
                };
            }
        }

        let column = |name: &str| -> PyResult<Bound<'py, PyAny>> {
            rows.get_item(name)?.get_item(&take)?.call_method0("tolist")
        };
        let args = PyTuple::new(
            py,
            [
                kind.clone(),
                column("qty")?,
                column("price")?,
                column("commission")?,
                column("total")?,
                PyList::new(py, &direction)?.into_any(),
                pybool(py, symbol.eq(m.getattr("CASH_SYMBOL_CSV")?)?),
                pybool(py, m.getattr("SHORTABLE_SYMBOLS")?.contains(symbol)?),
                pybool(py, is_transfer_a_flow.is_truthy()?),
            ],
        )?;
        let flows: Vec<f64> = m
            .getattr("investa_core")?
            .getattr("holding_flows")?
            .call1(args)?
            .extract()?;

        let dates_flows = PyDict::new(py);
        for (tx_date, cf) in tx_dates.iter().zip(&flows) {
            if !cf.is_nan() && cf.abs() > 1e-9 {
                let held: f64 = match dates_flows.get_item(tx_date)? {
                    Some(v) => v.extract()?,
                    None => 0.0,
                };
                dates_flows.set_item(tx_date, held + cf)?;
            }
        }

        let sorted_dates = builtin(py, "sorted")?.call1((dates_flows.keys(),))?;
        let abs = builtin(py, "abs")?;
        if !sorted_dates.is_truthy()? && abs.call1((final_market_value,))?.lt(1e-9)? {
            return empty_result(py);
        }
        let final_dates = PyList::empty(py);
        let final_flows = PyList::empty(py);
        let float = builtin(py, "float")?;
        for d in sorted_dates.try_iter()? {
            let d = d?;
            final_flows.append(float.call1((dates_flows.as_any().get_item(&d)?,))?)?;
            final_dates.append(d)?;
        }
        let pd = py.import("pandas")?;
        let fmv_abs = if pd
            .call_method1("notna", (final_market_value,))?
            .is_truthy()?
        {
            abs.call1((final_market_value,))?
        } else {
            fl(py, 0.0)
        };
        let date_t = py.import("datetime")?.getattr("date")?;
        if fmv_abs.gt(1e-9)? && report_date.is_instance(&date_t)? {
            if final_dates.is_empty() {
                let first = self
                    .frame(py)
                    .get_item("Date")?
                    .getattr("iloc")?
                    .get_item(&order)?
                    .call_method0("min")?
                    .call_method0("date")?;
                if report_date.ge(first)? {
                    final_dates.append(report_date)?;
                    final_flows.append(&fmv_abs)?;
                } else {
                    return empty_result(py);
                }
            } else {
                let last = final_dates.get_item(final_dates.len() - 1)?;
                if report_date.ge(&last)? {
                    if last.eq(report_date)? {
                        add_to_last(&final_flows, &fmv_abs)?;
                    } else {
                        final_dates.append(report_date)?;
                        final_flows.append(&fmv_abs)?;
                    }
                }
            }
        }
        finish(py, final_dates, final_flows)
    }
}

#[pyfunction]
#[pyo3(signature = (symbol, account, transactions_df, final_market_value, is_transfer_a_flow, report_date=None))]
pub fn fu_cash_flows_for_symbol_account<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    account: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    final_market_value: &Bound<'py, PyAny>,
    is_transfer_a_flow: &Bound<'py, PyAny>,
    report_date: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let nothing = none(py);
    fu(py)?
        .getattr("CashFlowLedger")?
        .call1((transactions_df,))?
        .call_method1(
            "flows_for",
            (
                symbol,
                account,
                final_market_value,
                is_transfer_a_flow,
                report_date.unwrap_or(&nothing),
            ),
        )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fu_cash_flows_for_mwr<'py>(
    py: Python<'py>,
    account_transactions: &Bound<'py, PyAny>,
    final_account_market_value: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    target_currency: &Bound<'py, PyAny>,
    fx_rates: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    historical_fx_rates: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    flow_basis: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = display_currency;
    if account_transactions.getattr("empty")?.is_truthy()? {
        return empty_result(py);
    }
    let m = fu(py)?;
    let pd = py.import("pandas")?;
    let np = m.getattr("np")?;
    let o = Ops::new(py)?;
    let acc_tx = account_transactions.call_method(
        "sort_values",
        (),
        Some(&kwargs(
            py,
            &[
                (
                    "by",
                    &PyList::new(py, ["Date", "original_index"])?.into_any(),
                ),
                ("ascending", &pybool(py, true)),
            ],
        )?),
    )?;

    // The accounts in scope; transfers within it are internal.
    let norm = |a: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        a.str()?.call_method0("strip")?.call_method0("upper")
    };
    let included = pyo3::types::PySet::empty(py)?;
    let source = if include_accounts.is_truthy()? {
        include_accounts.clone()
    } else {
        let set = builtin(py, "set")?;
        let all_accs = set.call1((account_transactions
            .get_item("Account")?
            .call_method0("dropna")?
            .call_method0("unique")?,))?;
        if account_transactions
            .getattr("columns")?
            .contains("To Account")?
        {
            all_accs.call_method1(
                "update",
                (account_transactions
                    .get_item("To Account")?
                    .call_method0("dropna")?
                    .call_method0("unique")?,),
            )?;
        }
        all_accs
    };
    for a in source.try_iter()? {
        let a = a?;
        if a.is_truthy()? {
            included.add(norm(a)?)?;
        }
    }

    let types: Vec<Bound<'py, PyAny>> = acc_tx
        .get_item("Type")?
        .call_method0("tolist")?
        .try_iter()?
        .map(|t| t?.str()?.call_method0("lower")?.call_method0("strip"))
        .collect::<PyResult<_>>()?;
    let symbols = acc_tx.get_item("Symbol")?.call_method0("tolist")?;
    acc_tx.get_item("Quantity")?;
    acc_tx.get_item("Price/Share")?; // read strictly, as the original did
    let numeric = m.getattr("_numeric_column")?;
    let with_types = [("with_types", &pybool(py, true))];
    let qv = unpack(
        &numeric.call((&acc_tx, "Quantity"), Some(&kwargs(py, &with_types)?))?,
        2,
    )?;
    let pv = unpack(
        &numeric.call((&acc_tx, "Price/Share"), Some(&kwargs(py, &with_types)?))?,
        2,
    )?;
    let commission = m.getattr("_commission_column")?.call1((&acc_tx,))?;
    let tv = unpack(
        &numeric.call((&acc_tx, "Total Amount"), Some(&kwargs(py, &with_types)?))?,
        2,
    )?;
    let tx_dates: Vec<Bound<'py, PyAny>> = acc_tx
        .get_item("Date")?
        .call_method0("tolist")?
        .try_iter()?
        .map(|d| d?.call_method0("date"))
        .collect::<PyResult<_>>()?;

    // Skip any transactions after the evaluated end date.
    let mut kept = Vec::new();
    for (i, d) in tx_dates.iter().enumerate() {
        if !d.gt(end_date)? {
            kept.push(i);
        }
    }
    let local_currencies = if kept.is_empty() {
        PyList::empty(py).into_any()
    } else {
        acc_tx.get_item("Local Currency")?.call_method0("tolist")?
    };
    let accounts = texts(&acc_tx, "Account")?;
    let to_accounts = texts(&acc_tx, "To Account")?;
    let flow_kinds = m.getattr("_FLOW_KINDS")?;
    let mut kinds = Vec::with_capacity(kept.len());
    for &i in &kept {
        kinds.push(
            flow_kinds
                .call_method1("get", (&types[i], 0))?
                .extract::<u8>()?,
        );
    }
    // A transfer crossing the scope: leaving is money out (+), entering in (-).
    let transfer: u8 = m.getattr("_TRANSFER")?.extract()?;
    let mut direction = vec![0i8; kept.len()];
    for (j, &i) in kept.iter().enumerate() {
        if kinds[j] != transfer {
            continue;
        }
        let acct = norm(cell_text(py, &accounts, i)?)?;
        let to_acct = norm(cell_text(py, &to_accounts, i)?)?;
        let is_outbound = included.contains(&acct)?;
        let is_inbound = to_acct.is_truthy()? && included.contains(&to_acct)?;
        if is_outbound && !is_inbound {
            direction[j] = 1;
        } else if !is_outbound && is_inbound {
            direction[j] = -1;
        }
    }

    let take = np.call_method(
        "asarray",
        (PyList::new(py, &kept)?,),
        Some(&kwargs(py, &[("dtype", &np.getattr("intp")?)])?),
    )?;
    let picked = |a: &Bound<'py, PyAny>| a.get_item(&take)?.call_method0("tolist");
    let cash = m.getattr("CASH_SYMBOL_CSV")?;
    let mut is_cash = Vec::with_capacity(kept.len());
    for &i in &kept {
        is_cash.push(symbols.get_item(i)?.eq(&cash)?);
    }
    let args = PyTuple::new(
        py,
        [
            PyList::new(py, &kinds)?.into_any(),
            picked(&qv[0])?,
            picked(&pv[0])?,
            picked(&commission)?,
            picked(&tv[0])?,
            PyList::new(py, &direction)?.into_any(),
            PyList::new(py, &is_cash)?.into_any(),
            pybool(py, flow_basis.eq("portfolio")?),
        ],
    )?;
    let flows = m
        .getattr("investa_core")?
        .getattr("mwr_flows")?
        .call1(args)?;

    let typed =
        |a: &Bound<'py, PyAny>| -> PyResult<Vec<bool>> { a.call_method0("tolist")?.extract() };
    let (qty_np, price_np, total_np) = (typed(&qv[1])?, typed(&pv[1])?, typed(&tv[1])?);
    let float64 = np.getattr("float64")?;
    let nan = np.getattr("nan")?;
    let abs = builtin(py, "abs")?;
    let conversion = m.getattr("get_conversion_rate")?;
    let dates_flows = PyDict::new(py);
    for (&i, item) in kept.iter().zip(flows.try_iter()?) {
        let fu_ = unpack(&item?, 2)?;
        let uses: i64 = fu_[1].extract()?;
        let mut cash_flow_local = fu_[0].clone();
        if uses != 0
            && ((uses & 1 != 0 && qty_np[i])
                || (uses & 2 != 0 && price_np[i])
                || (uses & 4 != 0 && total_np[i]))
        {
            cash_flow_local = float64.call1((cash_flow_local,))?;
        }
        let tx_date = &tx_dates[i];
        let local_currency = local_currencies.get_item(i)?;
        let mut cash_flow_target = cash_flow_local.clone();
        if pd.call_method1("notna", (&cash_flow_local,))?.is_truthy()?
            && abs.call1((&cash_flow_local,))?.gt(1e-9)?
        {
            if local_currency.ne(target_currency)? {
                let mut rate = nan.clone();
                if historical_fx_rates.is_truthy()? {
                    let key = PyTuple::new(py, [tx_date.clone(), local_currency.clone()])?;
                    if historical_fx_rates.contains(&key)? {
                        rate = historical_fx_rates.get_item(&key)?;
                    }
                }
                if pd.call_method1("isna", (&rate,))?.is_truthy()? {
                    rate = conversion.call1((&local_currency, target_currency, fx_rates))?;
                }
                if pd.call_method1("isna", (&rate,))?.is_truthy()? {
                    log(
                        py,
                        "warning",
                        format!(
                            "MWR calc conversion failed for {} on {}. Skipping.",
                            o.fmt(&local_currency)?,
                            o.fmt(tx_date)?
                        ),
                    )?;
                    cash_flow_target = fl(py, 0.0);
                } else {
                    cash_flow_target = cash_flow_local.mul(&rate)?;
                }
            }
            if pd
                .call_method1("notna", (&cash_flow_target,))?
                .is_truthy()?
                && abs.call1((&cash_flow_target,))?.gt(1e-9)?
            {
                let held = match dates_flows.get_item(tx_date)? {
                    Some(v) => v,
                    None => fl(py, 0.0),
                };
                dates_flows.set_item(tx_date, held.add(&cash_flow_target)?)?;
            }
        }
    }

    let sorted_dates = builtin(py, "sorted")?.call1((dates_flows.keys(),))?;
    if !sorted_dates.is_truthy()? && abs.call1((final_account_market_value,))?.lt(1e-9)? {
        return empty_result(py);
    }
    let final_dates = PyList::empty(py);
    let final_flows = PyList::empty(py);
    for d in sorted_dates.try_iter()? {
        let d = d?;
        final_flows.append(dates_flows.as_any().get_item(&d)?)?;
        final_dates.append(d)?;
    }
    let fmv = if pd
        .call_method1("notna", (final_account_market_value,))?
        .is_truthy()?
    {
        builtin(py, "float")?.call1((final_account_market_value,))?
    } else {
        fl(py, 0.0)
    };
    let fmv_abs = abs.call1((fmv,))?;
    let date_t = py.import("datetime")?.getattr("date")?;
    if fmv_abs.gt(1e-9)? && end_date.is_instance(&date_t)? {
        if final_dates.is_empty() {
            return empty_result(py);
        }
        let last = final_dates.get_item(final_dates.len() - 1)?;
        if end_date.ge(&last)? {
            if last.eq(end_date)? {
                add_to_last(&final_flows, &fmv_abs)?;
            } else {
                final_dates.append(end_date)?;
                final_flows.append(&fmv_abs)?;
            }
        }
    }
    finish(py, final_dates, final_flows)
}
