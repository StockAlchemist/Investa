//! `corporate_actions`: the deferred corporate-action arithmetic, IBKR
//! spin-off parsing and row building, and split deduplication.
//!
//! Ported line for line from `src/corporate_actions.py` (frozen in
//! `tests/reference/corporate_actions.py`). The arithmetic helpers use
//! Python's number operations on whatever they are given, as the originals
//! did, so integers in still give integers out; the spin-off parser uses the
//! module's own compiled patterns (Python `re` semantics); the split
//! deduplication is the same pandas operations in the original's order.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};

use crate::pyhelp::{pystr, setitem};

fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("corporate_actions")
}

fn le<'py>(a: &Bound<'py, PyAny>, b: f64) -> PyResult<bool> {
    a.le(b)
}

/// `apply_return_of_capital`: (qty, cost reduced by the cash, excess gain).
#[pyfunction]
pub fn ca_apply_return_of_capital<'py>(
    py: Python<'py>,
    current_qty: &Bound<'py, PyAny>,
    current_cost: &Bound<'py, PyAny>,
    cash_distributed: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    if le(current_qty, 1e-9)? {
        return PyTuple::new(py, [current_qty, current_cost, cash_distributed]);
    }
    if cash_distributed.le(current_cost)? {
        return PyTuple::new(
            py,
            [
                current_qty.clone(),
                current_cost.sub(cash_distributed)?,
                PyFloat::new(py, 0.0).into_any(),
            ],
        );
    }
    let excess = cash_distributed.sub(current_cost)?;
    PyTuple::new(
        py,
        [
            current_qty.clone(),
            PyFloat::new(py, 0.0).into_any(),
            excess,
        ],
    )
}

/// `apply_stock_dividend`: (qty plus the shares received, cost unchanged).
#[pyfunction]
pub fn ca_apply_stock_dividend<'py>(
    py: Python<'py>,
    current_qty: &Bound<'py, PyAny>,
    current_cost: &Bound<'py, PyAny>,
    shares_received: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    if le(current_qty, 1e-9)? || le(shares_received, 1e-9)? {
        return PyTuple::new(py, [current_qty, current_cost]);
    }
    PyTuple::new(
        py,
        [current_qty.add(shares_received)?, current_cost.clone()],
    )
}

/// `apply_spin_off`: the allocated basis, clamped to [0, parent cost], moved
/// from the parent to the child.
#[pyfunction]
pub fn ca_apply_spin_off<'py>(
    py: Python<'py>,
    parent_qty: &Bound<'py, PyAny>,
    parent_cost: &Bound<'py, PyAny>,
    child_qty: &Bound<'py, PyAny>,
    allocated_basis: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut moved = allocated_basis.clone();
    if moved.lt(0.0)? {
        moved = PyFloat::new(py, 0.0).into_any();
    }
    if moved.gt(parent_cost)? {
        moved = parent_cost.clone();
    }
    PyTuple::new(
        py,
        [
            parent_qty.clone(),
            parent_cost.sub(&moved)?,
            child_qty.clone(),
            moved,
        ],
    )
}

/// `parse_spinoff_description`: (parent, child, ratio) from an IBKR spin-off
/// description, or None.
#[pyfunction]
pub fn ca_parse_spinoff_description<'py>(
    py: Python<'py>,
    desc: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !desc.is_truthy()? {
        return Ok(py.None());
    }
    let low = desc
        .call_method0("lower")?
        .call_method1("replace", ("-", " "))?;
    if !low.contains("spin")? || !low.contains("off")? {
        return Ok(py.None());
    }
    let m = module(py)?;
    let parent_match = m
        .getattr("_SPINOFF_PARENT_RE")?
        .call_method1("match", (desc,))?;
    if parent_match.is_none() {
        return Ok(py.None());
    }
    let parent = parent_match.call_method1("group", (1,))?;
    let re = py.import("re")?;
    let mut child = py.None().into_bound(py);
    let groups = m
        .getattr("_SPINOFF_PAREN_GROUP_RE")?
        .call_method1("findall", (desc,))?;
    let reversed = py
        .import("builtins")?
        .getattr("reversed")?
        .call1((groups,))?;
    for group in reversed.try_iter()? {
        let first = group?
            .call_method1("split", (",",))?
            .get_item(0)?
            .call_method0("strip")?;
        if !re
            .call_method1("fullmatch", (r"[A-Z0-9.]{1,6}", &first))?
            .is_none()
        {
            child = first;
            break;
        }
    }
    if !child.is_truthy()? || child.eq(&parent)? {
        return Ok(py.None());
    }
    let ratio_match = m
        .getattr("_SPINOFF_RATIO_RE")?
        .call_method1("search", (desc,))?;
    let ratio = if ratio_match.is_truthy()? {
        ratio_match.call_method1("group", (1,))?
    } else {
        pystr(py, "spin-off")
    };
    Ok(PyTuple::new(py, [parent, child, ratio])?
        .into_any()
        .unbind())
}

/// `build_spinoff_legs`: the child receipt, and the parent's basis cut when
/// the allocation is known.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ca_build_spinoff_legs<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyAny>,
    child: &Bound<'py, PyAny>,
    child_qty: &Bound<'py, PyAny>,
    date_str: &Bound<'py, PyAny>,
    account: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
    allocated_basis: &Bound<'py, PyAny>,
    ratio: &Bound<'py, PyAny>,
    currency: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let legs = PyList::empty(py);
    if child_qty.le(1e-9)? {
        return Ok(legs);
    }
    let float = py.get_type::<PyFloat>();
    let basis = if allocated_basis.is_truthy()? {
        allocated_basis.clone()
    } else {
        PyFloat::new(py, 0.0).into_any()
    };
    let mut allocated = float.call1((basis,))?;
    if allocated.lt(0.0)? {
        allocated = PyFloat::new(py, 0.0).into_any();
    }
    let per_share = if child_qty.gt(1e-9)? {
        allocated.div(child_qty)?
    } else {
        PyFloat::new(py, 0.0).into_any()
    };
    let cut = |text: String| -> PyResult<Bound<'py, PyAny>> {
        PyString::new(py, &text).get_item(
            py.import("builtins")?
                .getattr("slice")?
                .call1((py.None(), 100))?,
        )
    };
    let leg = |symbol: &Bound<'py, PyAny>,
               qty: Bound<'py, PyAny>,
               price: Bound<'py, PyAny>,
               note: Bound<'py, PyAny>|
     -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("Date", date_str)?;
        d.set_item("Type", "Spin-off")?;
        d.set_item("Symbol", symbol)?;
        d.set_item("Quantity", qty)?;
        d.set_item("Price/Share", price)?;
        d.set_item("Total Amount", &allocated)?;
        d.set_item("Commission", 0.0)?;
        d.set_item("Account", account)?;
        d.set_item("Note", note)?;
        d.set_item("Local Currency", currency)?;
        d.set_item("user_id", user_id)?;
        Ok(d)
    };
    legs.append(leg(
        child,
        child_qty.clone(),
        per_share,
        cut(format!("Spinoff {} from {}", ratio.str()?, parent.str()?))?,
    )?)?;
    if allocated.gt(1e-9)? {
        legs.append(leg(
            parent,
            PyFloat::new(py, 0.0).into_any(),
            PyFloat::new(py, 0.0).into_any(),
            cut(format!("Spinoff basis reallocation to {}", child.str()?))?,
        )?)?;
    }
    Ok(legs)
}

fn kw<'py>(py: Python<'py>, pairs: &[(&str, Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

/// `deduplicate_split_transactions`: one row per (symbol, month, ratio)
/// split event, an "All Accounts" row preferred.
#[pyfunction]
pub fn ca_deduplicate_split_transactions<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if df.is_none() || df.getattr("empty")?.is_truthy()? {
        return Ok(df.clone());
    }
    if !df.getattr("columns")?.contains("Type")? {
        return Ok(df.clone());
    }
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let is_split = df
        .get_item("Type")?
        .getattr("str")?
        .call_method0("lower")?
        .getattr("str")?
        .call_method0("strip")?
        .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?;
    if !is_split.call_method0("any")?.is_truthy()? {
        return Ok(df.clone());
    }
    // De-fragment before slicing (upstream concatenations can leave the block
    // manager in a state where ~is_split or .copy() fails).
    let rebuilt = (|| -> PyResult<Bound<'py, PyAny>> {
        let cols = PyDict::new(py);
        for c in df.getattr("columns")?.try_iter()? {
            let c = c?;
            cols.set_item(&c, df.get_item(&c)?.call_method0("to_numpy")?)?;
        }
        pd.call_method(
            "DataFrame",
            (cols,),
            Some(&kw(py, &[("index", df.getattr("index")?)])?),
        )
    })();
    let df = match rebuilt {
        Ok(d) => d,
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let original_index = df.getattr("index")?;
            let d = pd.call_method1("DataFrame", (df.call_method1("to_dict", ("records",))?,))?;
            d.setattr("index", original_index)?;
            d
        }
        Err(e) => return Err(e),
    };
    let other = df.get_item(is_split.call_method0("__invert__")?)?;
    let splits = df.get_item(&is_split)?.call_method0("copy")?;
    let builtins_str = py.import("builtins")?.getattr("str")?;
    if splits.getattr("columns")?.contains("Account")? {
        let all_accounts = splits
            .get_item("Account")?
            .call_method1("astype", (&builtins_str,))?
            .getattr("str")?
            .call_method0("lower")?
            .getattr("str")?
            .call_method0("strip")?
            .call_method1("__eq__", ("all accounts",))?;
        setitem(
            &splits,
            "__split_priority",
            np.call_method1("where", (all_accounts, 0, 1))?,
        )?;
    } else {
        setitem(&splits, "__split_priority", 1)?;
    }
    let months = pd
        .call_method1("to_datetime", (splits.get_item("Date")?,))?
        .getattr("dt")?
        .call_method1("to_period", ("M",))?;
    setitem(&splits, "__split_ym", months)?;
    if splits.getattr("columns")?.contains("Split Ratio")? {
        let ratios = pd
            .call_method(
                "to_numeric",
                (splits.get_item("Split Ratio")?,),
                Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
            )?
            .call_method1("fillna", (1.0,))?
            .call_method1("round", (6,))?;
        setitem(&splits, "Split Ratio", ratios)?;
    } else {
        setitem(&splits, "Split Ratio", 1.0)?;
    }
    let sort_cols = PyList::new(py, ["Symbol", "__split_ym", "__split_priority"])?;
    if splits.getattr("columns")?.contains("original_index")? {
        sort_cols.append("original_index")?;
    }
    let splits = splits.call_method(
        "sort_values",
        (),
        Some(&kw(py, &[("by", sort_cols.into_any())])?),
    )?;
    let deduped = splits.call_method(
        "drop_duplicates",
        (),
        Some(&kw(
            py,
            &[
                (
                    "subset",
                    PyList::new(py, ["Symbol", "__split_ym", "Split Ratio"])?.into_any(),
                ),
                ("keep", pystr(py, "first")),
            ],
        )?),
    )?;
    let columns = PyList::new(
        py,
        df.getattr("columns")?
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let deduped = deduped.get_item(columns)?.call_method0("copy")?;
    let other = other.call_method0("copy")?;
    let frames = PyList::empty(py);
    for f in [other, deduped] {
        if !f.getattr("empty")?.is_truthy()? {
            frames.append(f)?;
        }
    }
    if frames.is_empty() {
        return df
            .getattr("iloc")?
            .get_item(
                py.import("builtins")?
                    .getattr("slice")?
                    .call1((py.None(), 0))?,
            )?
            .call_method0("copy");
    }
    let result = if frames.len() == 1 {
        frames.get_item(0)?.call_method0("copy")?
    } else {
        pd.call_method1("concat", (frames,))?
    };
    if result.getattr("columns")?.contains("original_index")? {
        result.call_method(
            "sort_values",
            (),
            Some(&kw(py, &[("by", pystr(py, "original_index"))])?),
        )
    } else {
        result.call_method0("sort_index")
    }
}
