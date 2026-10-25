//! `data_loader`, `csv_utils` and `io_handlers`: loading and cleaning the
//! transaction ledger from the database or a CSV, standardising a CSV's
//! headers, and the export writers.
//!
//! Ported line for line from those modules (frozen in `tests/reference/`).
//! The CSV parse itself stays `pd.read_csv` — its float parsing and type
//! inference are pandas' own — and the cleaning is the same pandas column
//! operations, driven from here in the original's order; every column store
//! goes through `pyhelp::setitem`, so pandas sees the reference count a
//! Python statement gives it. Module-level names (the column maps, the DB
//! helpers `data_loader` imported, `CASH_SYMBOL_CSV`) are read from the
//! module at call time, so a patched attribute is still the one used.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};

use crate::pyhelp::{es, log, pystr, s, setitem};

fn dl(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("data_loader")
}

fn kw<'py>(py: Python<'py>, pairs: &[(&str, Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

fn truth(py: Python<'_>, b: bool) -> Bound<'_, PyAny> {
    pyo3::types::PyBool::new(py, b).to_owned().into_any()
}

/// `_normalize_header`: the header stripped, "" for a non-string.
#[pyfunction]
pub fn dl_normalize_header<'py>(
    py: Python<'py>,
    header: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !header.is_instance_of::<PyString>() {
        return Ok(pystr(py, ""));
    }
    header.call_method0("strip")
}

/// `_map_headers_and_drop_unwanted`: rename by the mapping, keep the expected
/// columns; returns (frame, {original header: cleaned name}).
#[pyfunction]
pub fn dl_map_headers<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    col_mapping: &Bound<'py, PyAny>,
    expected_cols: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let normalize = dl(py)?.getattr("_normalize_header")?;
    let applied = PyDict::new(py);
    let rename = PyDict::new(py);
    let normalized = PyDict::new(py);
    for col in df.getattr("columns")?.try_iter()? {
        let col = col?;
        normalized.set_item(&col, normalize.call1((&col,))?)?;
    }
    for (raw, norm) in normalized.iter() {
        if col_mapping.contains(&norm)? {
            let cleaned = col_mapping.get_item(&norm)?;
            rename.set_item(&raw, &cleaned)?;
            applied.set_item(&raw, &cleaned)?;
        } else if expected_cols.contains(&norm)? && !rename.contains(&raw)? {
            applied.set_item(&raw, &norm)?;
        }
    }
    let renamed = df.call_method(
        "rename",
        (),
        Some(&kw(py, &[("columns", rename.into_any())])?),
    )?;
    let keep = PyList::empty(py);
    let renamed_cols = renamed.getattr("columns")?;
    for col in expected_cols.try_iter()? {
        let col = col?;
        if renamed_cols.contains(&col)? {
            keep.append(col)?;
        }
    }
    let out = renamed.get_item(keep)?.call_method0("copy")?;
    Ok((out, applied))
}

const COMMON_FORMATS: [&str; 10] = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%b %d, %Y",
    "%d-%b-%Y",
    "%B %d, %Y",
    "%Y%m%d",
];

/// `_parse_date_robustly`: pandas' own parse, then the listed formats for the
/// dates it could not read (ordinal suffixes removed first).
#[pyfunction]
pub fn dl_parse_date_robustly<'py>(
    py: Python<'py>,
    date_series: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let parsed = pd.call_method(
        "to_datetime",
        (date_series,),
        Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
    )?;
    let failed = parsed
        .get_item(parsed.call_method0("isna")?)?
        .getattr("index")?;
    if failed.getattr("empty")?.is_truthy()? {
        return Ok(parsed);
    }
    log(
        py,
        "debug",
        format!(
            "Robust date parsing: {} dates initially failed direct parse. Trying specific formats.",
            failed.len()?
        ),
    )?;
    let re = py.import("re")?;
    let strptime = py
        .import("datetime")?
        .getattr("datetime")?
        .getattr("strptime")?;
    let ignorecase = re.getattr("IGNORECASE")?;
    for idx in failed.try_iter()? {
        let idx = idx?;
        let original = date_series.getattr("loc")?.get_item(&idx)?;
        if pd.call_method1("isna", (&original,))?.is_truthy()?
            || !original.is_instance_of::<PyString>()
            || !original.call_method0("strip")?.is_truthy()?
        {
            continue;
        }
        let cleaned = re.call_method(
            "sub",
            (
                r"(\d+)(st|nd|rd|th)",
                r"\1",
                original.call_method0("strip")?,
            ),
            Some(&kw(py, &[("flags", ignorecase.clone())])?),
        )?;
        for fmt in COMMON_FORMATS {
            match strptime.call1((&cleaned, fmt)) {
                Ok(when) => {
                    parsed.getattr("loc")?.set_item(&idx, when)?;
                    break;
                }
                Err(e)
                    if e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) =>
                {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(parsed)
}

/// The seven-element result of `load_and_clean_transactions`.
type Loaded<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    bool,
    bool,
    Bound<'py, PyAny>,
);

fn empty_frame(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("pandas")?.call_method0("DataFrame")
}

fn frame_with_columns<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("pandas")?.call_method(
        "DataFrame",
        (),
        Some(&kw(py, &[("columns", columns.clone())])?),
    )
}

/// `load_and_clean_transactions`: (cleaned frame, raw frame for the ignored
/// rows, ignored indices, reasons, has_errors, has_warnings, header map).
#[pyfunction]
pub fn dl_load_and_clean<'py>(
    py: Python<'py>,
    source_path: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    is_db_source: &Bound<'py, PyAny>,
) -> PyResult<Loaded<'py>> {
    let m = dl(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let is_db = is_db_source.is_truthy()?;
    log(
        py,
        "info",
        format!(
            "Data Loader: Starting load from {}: {}",
            if is_db { "DB" } else { "CSV" },
            s(source_path)?
        ),
    )?;
    let mut has_errors = false;
    let mut has_warnings = false;
    let ignored_indices = PySet::empty(py)?;
    let ignored_reasons = PyDict::new(py);
    let mut header_map = PyDict::new(py).into_any();
    let mut raw = py.None().into_bound(py);
    let mut loaded_from_db = false;
    let mut df = py.None().into_bound(py);
    let expected = m.getattr("EXPECTED_CLEANED_COLUMNS")?;

    if is_db && m.getattr("DB_UTILS_AVAILABLE")?.is_truthy()? {
        let db_conn = m.getattr("initialize_database")?.call1((source_path,))?;
        if db_conn.is_truthy()? {
            let got = m.getattr("load_all_transactions_from_db")?.call1((
                &db_conn,
                account_currency_map,
                default_currency,
            ))?;
            let (frame, success) = (got.get_item(0)?, got.get_item(1)?.is_truthy()?);
            if success && !frame.is_none() && !frame.getattr("empty")?.is_truthy()? {
                raw = frame.call_method0("copy")?;
                loaded_from_db = true;
                log(
                    py,
                    "info",
                    format!(
                        "Successfully loaded {} transactions from database.",
                        frame.len()?
                    ),
                )?;
                df = frame;
            } else if success && (frame.is_none() || frame.getattr("empty")?.is_truthy()?) {
                log(
                    py,
                    "info",
                    "Database is empty or load_all_transactions_from_db returned no data. Will check for CSV.",
                )?;
                raw = empty_frame(py)?;
            } else {
                log(
                    py,
                    "error",
                    "Failed to load transactions from database. Will check for CSV.",
                )?;
                has_errors = true;
                raw = empty_frame(py)?;
            }
        } else {
            log(
                py,
                "error",
                format!(
                    "Failed to connect to database at {}. Will check for CSV.",
                    s(source_path)?
                ),
            )?;
            has_errors = true;
            raw = empty_frame(py)?;
        }
    }

    if !loaded_from_db {
        if is_db {
            log(
                py,
                "info",
                format!(
                    "DB source specified but no data loaded. Checking for CSV at '{}' (assuming it might be a CSV for migration).",
                    s(source_path)?
                ),
            )?;
            if !has_errors {
                log(
                    py,
                    "info",
                    "Database is empty. No CSV migration attempted by data_loader directly.",
                )?;
            }
        }
        if !is_db {
            log(py, "info", format!("Loading from CSV: {}", s(source_path)?))?;
            if !py
                .import("os")?
                .getattr("path")?
                .call_method1("exists", (source_path,))?
                .is_truthy()?
            {
                log(
                    py,
                    "error",
                    format!("Transactions CSV file not found: {}", s(source_path)?),
                )?;
                return Ok((
                    py.None().into_bound(py),
                    py.None().into_bound(py),
                    ignored_indices.into_any(),
                    ignored_reasons.into_any(),
                    true,
                    has_warnings,
                    header_map,
                ));
            }
            let read = (|| -> PyResult<Option<Loaded<'py>>> {
                let r = pd.call_method(
                    "read_csv",
                    (source_path,),
                    Some(&kw(
                        py,
                        &[
                            ("dtype", py.import("builtins")?.getattr("str")?),
                            ("keep_default_na", truth(py, false)),
                            ("skipinitialspace", truth(py, true)),
                        ],
                    )?),
                )?;
                if r.getattr("empty")?.is_truthy()? && r.getattr("columns")?.len()? == 0 {
                    log(
                        py,
                        "warning",
                        format!("CSV file '{}' is empty or has no headers.", s(source_path)?),
                    )?;
                    return Ok(Some((
                        frame_with_columns(py, &expected)?,
                        empty_frame(py)?,
                        PySet::empty(py)?.into_any(),
                        PyDict::new(py).into_any(),
                        false,
                        true,
                        PyDict::new(py).into_any(),
                    )));
                }
                setitem(&r, "original_index", r.getattr("index")?)?;
                raw = r.clone();
                let processing = r.call_method0("copy")?;
                let mapped = m.getattr("_map_headers_and_drop_unwanted")?.call1((
                    processing,
                    m.getattr("COLUMN_MAPPING_CSV_TO_INTERNAL")?,
                    &expected,
                ))?;
                let (processed, applied) = (mapped.get_item(0)?, mapped.get_item(1)?);
                header_map = applied;
                if processed.getattr("empty")?.is_truthy()? && !r.getattr("empty")?.is_truthy()? {
                    log(
                        py,
                        "warning",
                        format!(
                            "No columns mapped to expected standard names from CSV: {}. Check headers.",
                            s(source_path)?
                        ),
                    )?;
                    for idx in r.getattr("index")?.try_iter()? {
                        let idx = idx?;
                        ignored_indices.add(&idx)?;
                        ignored_reasons.set_item(&idx, "CSV Header Mapping Failed")?;
                    }
                    has_warnings = true;
                    return Ok(Some((
                        r.clone(),
                        r.clone(),
                        ignored_indices.clone().into_any(),
                        ignored_reasons.clone().into_any(),
                        has_errors,
                        has_warnings,
                        header_map.clone(),
                    )));
                }
                log(
                    py,
                    "info",
                    format!(
                        "Loaded {} rows from CSV '{}'. Mapped headers.",
                        processed.len()?,
                        s(source_path)?
                    ),
                )?;
                df = processed;
                Ok(None)
            })();
            match read {
                Ok(Some(early)) => return Ok(early),
                Ok(None) => {}
                Err(e)
                    if e.value(py)
                        .is_instance(&pd.getattr("errors")?.getattr("EmptyDataError")?)? =>
                {
                    log(
                        py,
                        "warning",
                        format!("Transactions CSV file is empty: {}", s(source_path)?),
                    )?;
                    return Ok((
                        frame_with_columns(py, &expected)?,
                        empty_frame(py)?,
                        ignored_indices.into_any(),
                        ignored_reasons.into_any(),
                        has_errors,
                        true,
                        header_map,
                    ));
                }
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                    log(
                        py,
                        "error",
                        format!(
                            "Error reading or initially processing CSV file {}: {}",
                            s(source_path)?,
                            es(py, &e)
                        ),
                    )?;
                    return Ok((
                        py.None().into_bound(py),
                        py.None().into_bound(py),
                        ignored_indices.into_any(),
                        ignored_reasons.into_any(),
                        true,
                        has_warnings,
                        header_map,
                    ));
                }
                Err(e) => return Err(e),
            }
        } else {
            df = empty_frame(py)?;
            if raw.is_none() {
                raw = empty_frame(py)?;
            }
        }
    }

    // --- Common Cleaning Steps ---
    if !df.is_instance(&pd.getattr("DataFrame")?)? {
        df = empty_frame(py)?;
        if !has_errors {
            log(
                py,
                "error",
                "DataFrame `df` is not a pandas DataFrame after load attempts.",
            )?;
            has_errors = true;
        }
    }
    if df.getattr("empty")?.is_truthy()? {
        log(
            py,
            "info",
            "No transactions to process after loading stage.",
        )?;
        return Ok((
            df,
            raw,
            ignored_indices.into_any(),
            ignored_reasons.into_any(),
            has_errors,
            has_warnings,
            header_map,
        ));
    }

    if !df.getattr("columns")?.contains("original_index")? {
        if !df.getattr("empty")?.is_truthy()? {
            setitem(&df, "original_index", df.getattr("index")?)?;
            log(
                py,
                "warning",
                "Added 'original_index' from DataFrame index as it was missing.",
            )?;
            has_warnings = true;
        } else if !raw.is_none() && raw.getattr("columns")?.contains("original_index")? {
        } else if !raw.is_none() && !raw.getattr("empty")?.is_truthy()? {
            setitem(&raw, "original_index", raw.getattr("index")?)?;
        }
    }

    for col in expected.try_iter()? {
        let col = col?;
        if !df.getattr("columns")?.contains(&col)? {
            setitem(&df, &col, pd.getattr("NA")?)?;
            log(
                py,
                "debug",
                format!("Added missing expected column '{}' to DataFrame.", s(&col)?),
            )?;
        }
    }

    // Date parsing (critical)
    if df.getattr("columns")?.contains("Date")? {
        if !pd
            .getattr("api")?
            .getattr("types")?
            .call_method1("is_datetime64_any_dtype", (df.get_item("Date")?,))?
            .is_truthy()?
        {
            let dates = m
                .getattr("_parse_date_robustly")?
                .call1((df.get_item("Date")?,))?;
            setitem(&df, "Date", dates)?;
        }
        let failed = df.get_item("Date")?.call_method0("isna")?;
        if failed.call_method0("any")?.is_truthy()? {
            for idx in df.get_item(&failed)?.getattr("index")?.try_iter()? {
                let idx = idx?;
                let original = df.getattr("loc")?.get_item((idx, "original_index"))?;
                ignored_indices.add(&original)?;
                ignored_reasons.set_item(&original, "Invalid or Unparseable Date")?;
            }
            df = df
                .get_item(failed.call_method0("__invert__")?)?
                .call_method0("copy")?;
            log(
                py,
                "warning",
                format!(
                    "Removed {} rows due to unparseable dates.",
                    s(&failed.call_method0("sum")?)?
                ),
            )?;
            has_warnings = true;
        }
    } else {
        log(
            py,
            "error",
            "CRITICAL: 'Date' column is missing. Cannot proceed with processing.",
        )?;
        if df.getattr("columns")?.contains("original_index")? {
            for v in df
                .get_item("original_index")?
                .call_method0("tolist")?
                .try_iter()?
            {
                let v = v?;
                ignored_indices.add(&v)?;
                ignored_reasons.set_item(&v, "Missing Date Column")?;
            }
        }
        let empty = df
            .getattr("iloc")?
            .get_item(py.import("builtins")?.getattr("slice")?.call1((0, 0))?)?;
        return Ok((
            empty,
            raw,
            ignored_indices.into_any(),
            ignored_reasons.into_any(),
            true,
            has_warnings,
            header_map,
        ));
    }

    // Numeric conversions
    for col in [
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Split Ratio",
    ] {
        if !df.getattr("columns")?.contains(col)? {
            continue;
        }
        let original = df.get_item(col)?.call_method0("copy")?;
        let numeric = pd.call_method(
            "to_numeric",
            (df.get_item(col)?,),
            Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
        )?;
        setitem(&df, col, numeric)?;
        let failed = df
            .get_item(col)?
            .call_method0("isna")?
            .call_method1("__and__", (original.call_method0("notna")?,))?
            .call_method1("__and__", (original.call_method1("__ne__", ("",))?,))?;
        if failed.call_method0("any")?.is_truthy()? {
            for idx in df.get_item(&failed)?.getattr("index")?.try_iter()? {
                let idx = idx?;
                let original_idx = df.getattr("loc")?.get_item((&idx, "original_index"))?;
                log(
                    py,
                    "warning",
                    format!(
                        "Row {}: Could not convert '{col}' value '{}' to numeric. Set to NaN.",
                        s(&original_idx)?,
                        s(&original.getattr("loc")?.get_item(&idx)?)?
                    ),
                )?;
            }
            has_warnings = true;
        }
    }

    let builtins_str = py.import("builtins")?.getattr("str")?;
    for col in ["Symbol", "Account", "Note", "Local Currency"] {
        if df.getattr("columns")?.contains(col)? {
            let cleaned = df
                .get_item(col)?
                .call_method1("astype", (&builtins_str,))?
                .getattr("str")?
                .call_method0("strip")?
                .call_method(
                    "replace",
                    ("None", ""),
                    Some(&kw(py, &[("regex", truth(py, false))])?),
                )?;
            setitem(&df, col, cleaned)?;
        }
    }

    if df.getattr("columns")?.contains("Type")? {
        let t = df
            .get_item("Type")?
            .call_method1("astype", (&builtins_str,))?
            .getattr("str")?
            .call_method0("strip")?
            .getattr("str")?
            .call_method0("lower")?;
        setitem(&df, "Type", t)?;
        let spellings = PyDict::new(py);
        spellings.set_item("spin-off", "spin off")?;
        spellings.set_item("spinoff", "spin off")?;
        let t = df.get_item("Type")?.call_method1("replace", (spellings,))?;
        setitem(&df, "Type", t)?;
        match py
            .import("corporate_actions")
            .and_then(|ca| ca.getattr("RESERVED_CORPORATE_ACTION_TYPES"))
        {
            Ok(reserved) => {
                let mask = df.get_item("Type")?.call_method1("isin", (reserved,))?;
                if mask.call_method0("any")?.is_truthy()? {
                    let counts = df
                        .getattr("loc")?
                        .get_item((&mask, "Type"))?
                        .call_method0("value_counts")?
                        .call_method0("to_dict")?;
                    log(
                        py,
                        "warning",
                        format!(
                            "Found {} transaction(s) of reserved corporate-action types not yet applied by the engine: {}. Math is defined in corporate_actions.py; integration into portfolio_logic.py JIT dispatchers is pending.",
                            s(&mask.call_method0("sum")?)?,
                            s(&counts)?
                        ),
                    )?;
                }
            }
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyImportError>(py) => {}
            Err(e) => return Err(e),
        }
    }

    if df.getattr("columns")?.contains("Tags")? {
        let tags = df
            .get_item("Tags")?
            .call_method1("astype", (&builtins_str,))?
            .getattr("str")?
            .call_method0("strip")?;
        setitem(&df, "Tags", tags)?;
        let blank = df
            .get_item("Tags")?
            .getattr("str")?
            .call_method0("lower")?
            .call_method1("isin", (PyList::new(py, ["nan", "none", "<na>"])?,))?;
        df.getattr("loc")?.set_item((blank, "Tags"), "")?;
    }

    if df.getattr("columns")?.contains("Symbol")? {
        let cash = m.getattr("CASH_SYMBOL_CSV")?;
        let upper = df
            .get_item("Symbol")?
            .getattr("str")?
            .call_method0("upper")?
            .call_method1("isin", (PyList::new(py, ["CASH", "USD"])?,))?;
        df.getattr("loc")?.set_item((upper, "Symbol"), &cash)?;
        if df.getattr("columns")?.contains("Type")? {
            let flows = df
                .get_item("Type")?
                .getattr("str")?
                .call_method0("lower")?
                .call_method1("isin", (PyList::new(py, ["deposit", "withdrawal"])?,))?;
            let blank = df
                .get_item("Symbol")?
                .getattr("str")?
                .call_method0("lower")?
                .call_method1("isin", (PyList::new(py, ["", "nan", "none", "<na>"])?,))?;
            df.getattr("loc")?
                .set_item((flows.call_method1("__and__", (blank,))?, "Symbol"), &cash)?;
        }
    }

    let empties = PyList::new(
        py,
        [
            pystr(py, ""),
            py.None().into_bound(py),
            np.getattr("nan")?,
            pd.getattr("NA")?,
        ],
    )?;
    if !df.getattr("columns")?.contains("Local Currency")?
        || df
            .get_item("Local Currency")?
            .call_method1("isin", (&empties,))?
            .call_method0("all")?
            .is_truthy()?
    {
        log(
            py,
            "info",
            "'Local Currency' column missing or empty. Assigning based on account_currency_map.",
        )?;
        let filled = df
            .get_item("Account")?
            .call_method1("map", (account_currency_map,))?
            .call_method1("fillna", (default_currency,))?;
        setitem(&df, "Local Currency", filled)?;
        has_warnings = true;
    } else {
        let normalized = df
            .get_item("Local Currency")?
            .call_method1("astype", (&builtins_str,))?
            .getattr("str")?
            .call_method0("strip")?
            .getattr("str")?
            .call_method0("upper")?;
        let invalid = normalized
            .call_method1(
                "isin",
                (PyList::new(py, ["", "<NA>", "NAN", "NONE", "N/A"])?,),
            )?
            .call_method1(
                "__or__",
                (normalized
                    .getattr("str")?
                    .call_method0("len")?
                    .call_method1("__ne__", (3,))?,),
            )?
            .call_method1(
                "__or__",
                (df.get_item("Local Currency")?.call_method0("isna")?,),
            )?;
        if invalid.call_method0("any")?.is_truthy()? {
            let loc = df.getattr("loc")?;
            let fill = loc
                .get_item((&invalid, "Account"))?
                .call_method1("map", (account_currency_map,))?
                .call_method1("fillna", (default_currency,))?;
            loc.set_item((&invalid, "Local Currency"), fill)?;
            log(
                py,
                "info",
                format!(
                    "Filled or corrected {} missing/invalid Local Currency values.",
                    s(&invalid.call_method0("sum")?)?
                ),
            )?;
            has_warnings = true;
        }
    }
    let upper = df
        .get_item("Local Currency")?
        .getattr("str")?
        .call_method0("upper")?;
    setitem(&df, "Local Currency", upper)?;

    let original_df = if raw.is_none() { empty_frame(py)? } else { raw };
    log(
        py,
        "info",
        format!(
            "Data loading and cleaning finished. Errors: {}, Warnings: {}, Ignored: {}",
            if has_errors { "True" } else { "False" },
            if has_warnings { "True" } else { "False" },
            ignored_indices.len()
        ),
    )?;
    Ok((
        df,
        original_df,
        ignored_indices.into_any(),
        ignored_reasons.into_any(),
        has_errors,
        has_warnings,
        header_map,
    ))
}

/// `csv_utils.convert_csv_headers_to_cleaned_format`: (success, message).
#[pyfunction]
pub fn dl_convert_csv_headers<'py>(
    py: Python<'py>,
    input_csv_path: &Bound<'py, PyAny>,
    output_csv_path: &Bound<'py, PyAny>,
) -> PyResult<(bool, String)> {
    let cu = py.import("csv_utils")?;
    let pd = py.import("pandas")?;
    log(
        py,
        "info",
        format!(
            "Attempting to standardize CSV headers for '{}' -> '{}'",
            s(input_csv_path)?,
            s(output_csv_path)?
        ),
    )?;
    let quote_minimal = py.import("csv")?.getattr("QUOTE_MINIMAL")?;
    let to_csv_kw = || -> PyResult<Bound<'py, PyDict>> {
        kw(
            py,
            &[
                ("index", truth(py, false)),
                ("encoding", pystr(py, "utf-8")),
                ("quoting", quote_minimal.clone()),
            ],
        )
    };
    let desired = cu.getattr("DESIRED_CLEANED_COLUMN_ORDER")?;
    let run = (|| -> PyResult<(bool, String)> {
        let df = pd.call_method(
            "read_csv",
            (input_csv_path,),
            Some(&kw(
                py,
                &[
                    ("dtype", py.import("builtins")?.getattr("str")?),
                    ("keep_default_na", truth(py, false)),
                    ("skipinitialspace", truth(py, true)),
                ],
            )?),
        )?;
        if df.getattr("empty")?.is_truthy()? && df.getattr("columns")?.len()? == 0 {
            log(
                py,
                "warning",
                format!(
                    "Input CSV '{}' is empty or has no headers. No conversion performed.",
                    s(input_csv_path)?
                ),
            )?;
            let headers = desired.get_item(
                py.import("builtins")?
                    .getattr("slice")?
                    .call1((py.None(),))?,
            )?;
            if !headers.contains("Local Currency")? {
                headers.call_method1("append", ("Local Currency",))?;
            }
            frame_with_columns(py, &headers)?.call_method(
                "to_csv",
                (output_csv_path,),
                Some(&to_csv_kw()?),
            )?;
            return Ok((
                true,
                "Input CSV was empty; output CSV (if different) created with standard headers."
                    .into(),
            ));
        }
        let standard = cu.getattr("STANDARD_ORIGINAL_TO_CLEANED_MAP")?;
        let normalized = PyDict::new(py);
        for col in df.getattr("columns")?.try_iter()? {
            let col = col?;
            normalized.set_item(col.call_method0("strip")?, &col)?;
        }
        let rename = PyDict::new(py);
        for (norm, raw) in normalized.iter() {
            if standard.contains(&norm)? {
                rename.set_item(&raw, standard.get_item(&norm)?)?;
            } else if desired.contains(&norm)? {
            } else {
                log(
                    py,
                    "debug",
                    format!(
                        "Column '{}' (normalized: '{}') not in standard map or desired order. Will be kept if not explicitly dropped.",
                        s(&raw)?,
                        s(&norm)?
                    ),
                )?;
            }
        }
        df.call_method(
            "rename",
            (),
            Some(&kw(
                py,
                &[
                    ("columns", rename.clone().into_any()),
                    ("inplace", truth(py, true)),
                ],
            )?),
        )?;
        log(
            py,
            "info",
            format!(
                "Applied column renames based on standard map: {}",
                s(&rename)?
            ),
        )?;
        let order = PyList::empty(py);
        let present = PySet::new(
            py,
            df.getattr("columns")?
                .try_iter()?
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        for col in desired.try_iter()? {
            let col = col?;
            if present.contains(&col)? {
                order.append(col)?;
            }
        }
        for col in df.getattr("columns")?.try_iter()? {
            let col = col?;
            if !order.contains(&col)? {
                order.append(col)?;
            }
        }
        df.get_item(order)?
            .call_method("to_csv", (output_csv_path,), Some(&to_csv_kw()?))?;
        let msg = format!(
            "Successfully standardized CSV headers and saved to '{}'",
            s(output_csv_path)?
        );
        log(py, "info", &msg)?;
        Ok((true, msg))
    })();
    match run {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py) => {
            let msg = format!(
                "Error: Input CSV file not found at '{}'",
                s(input_csv_path)?
            );
            log(py, "error", &msg)?;
            Ok((false, msg))
        }
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let msg = format!(
                "An unexpected error occurred during CSV header conversion: {}",
                es(py, &e)
            );
            crate::pyhelp::log_exc(py, "exception", &msg, &e)?;
            Ok((false, msg))
        }
        Err(e) => Err(e),
    }
}

/// `io_handlers.write_dataframe_to_csv`
#[pyfunction]
pub fn dl_write_csv<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    file_path: &Bound<'py, PyAny>,
    encoding: &Bound<'py, PyAny>,
    quoting: &Bound<'py, PyAny>,
) -> PyResult<()> {
    df.call_method(
        "to_csv",
        (file_path,),
        Some(&kw(
            py,
            &[
                ("index", truth(py, false)),
                ("encoding", encoding.clone()),
                ("quoting", quoting.clone()),
            ],
        )?),
    )?;
    Ok(())
}

/// `io_handlers.write_dataframe_to_excel`
#[pyfunction]
pub fn dl_write_excel<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    file_path: &Bound<'py, PyAny>,
    engine: &Bound<'py, PyAny>,
) -> PyResult<()> {
    df.call_method(
        "to_excel",
        (file_path,),
        Some(&kw(
            py,
            &[("index", truth(py, false)), ("engine", engine.clone())],
        )?),
    )?;
    Ok(())
}

#[allow(dead_code)]
fn unused(_: &Bound<'_, PyTuple>) {}
