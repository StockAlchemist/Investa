//! The glue of `edgar_provider`: the `as_of` block, the shared store, the
//! companyfacts parsers, the quarterly derivation (canonical period ends,
//! per-filing split bases, the year-to-date ladder), the bulk and per-company
//! ingest, concept resolution, the split-consistent series, revisions, and
//! the yfinance-shaped statement frames with their derived rows.
//!
//! Ported line for line from `src/edgar_provider.py` (frozen in
//! `tests/reference/edgar_provider_p15.py`). The fact store itself was
//! already Rust (`ef_store.rs`). Values are combined with Python's own
//! operations in the original's operand order; module state, constants and
//! siblings are looked up on `edgar_provider` at call time. The private
//! per-element helpers (`_span_days`, `_is_annual_duration`, `_median`, the
//! ISO-day match in `_period_stamps`) run here directly, since they are pure;
//! the dates are still parsed by `datetime.fromisoformat`. The network
//! fetchers (`sec_get_json`, `download_bulk_archive`) stay Python.

use pyo3::exceptions::{
    PyKeyError, PyOSError, PyStopIteration, PyTypeError, PyUnboundLocalError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyFloat, PyList, PySet, PySlice, PyString, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

fn ep(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("edgar_provider")
}

fn is_value_or_type(py: Python<'_>, err: &PyErr) -> bool {
    err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py)
}

fn empty_dict(py: Python<'_>) -> Bound<'_, PyAny> {
    PyDict::new(py).into_any()
}

fn empty_list(py: Python<'_>) -> Bound<'_, PyAny> {
    PyList::empty(py).into_any()
}

fn tup<'py>(py: Python<'py>, items: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, items)?.into_any())
}

fn int<'py>(py: Python<'py>, v: i64) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyobject(py)?.into_any())
}

/// `x.get(key, default)`
fn get_or<'py>(
    x: &Bound<'py, PyAny>,
    key: impl IntoPyObject<'py>,
    default: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    x.call_method1(intern!(x.py(), "get"), (key, default))
}

fn get<'py>(x: &Bound<'py, PyAny>, key: impl IntoPyObject<'py>) -> PyResult<Bound<'py, PyAny>> {
    x.call_method1(intern!(x.py(), "get"), (key,))
}

fn items<'py>(x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    x.call_method0(intern!(x.py(), "items"))
}

/// `str(x).zfill(10)`
fn zfill10<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    x.str()?.call_method1(intern!(py, "zfill"), (10,))
}

fn builtin<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr(name)
}

/// `a <= x <= b` (Python's chained comparison).
fn between(x: &Bound<'_, PyAny>, lo: &Bound<'_, PyAny>, hi: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(lo.le(x)? && x.le(hi)?)
}

// --- as_of and the shared store ------------------------------------------

/// `as_of(date)`: read the store as it stood on `date` for this block.
#[pyclass(module = "investa_core")]
pub struct AsOf {
    date: Py<PyAny>,
    token: Option<Py<PyAny>>,
}

#[pymethods]
impl AsOf {
    #[new]
    fn new(date: Py<PyAny>) -> Self {
        AsOf { date, token: None }
    }

    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let token = ep(py)?
            .getattr("_as_of_var")?
            .call_method1("set", (self.date.bind(py),))?;
        self.token = Some(token.unbind());
        Ok(py.None())
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(token) = self.token.take() {
            ep(py)?
                .getattr("_as_of_var")?
                .call_method1("reset", (token,))?;
        }
        Ok(false)
    }
}

#[pyfunction]
pub fn ep_db_path<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let config = m.getattr("config")?;
    let path = py.import("os")?.getattr("path")?;
    let directory = path.call_method1(
        "join",
        (
            config.call_method0("get_app_data_dir")?,
            config.getattr("DB_DIR")?,
        ),
    )?;
    let exist_ok = true.into_pyobject(py)?.to_owned().into_any();
    py.import("os")?.call_method(
        "makedirs",
        (&directory,),
        Some(&kwargs(py, &[("exist_ok", &exist_ok)])?),
    )?;
    path.call_method1("join", (&directory, m.getattr("_DB_FILENAME")?))
}

#[pyfunction]
pub fn ep_get_store<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    if m.getattr("_shared_store")?.is_none() {
        m.setattr("_shared_store", m.getattr("EdgarFactStore")?.call0()?)?;
    }
    m.getattr("_shared_store")
}

// --- parsing --------------------------------------------------------------

/// `_span_days`: the length of a duration fact in days, or None.
fn span_days<'py>(
    fromiso: &Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = fromiso.py();
    if !start.is_truthy()? {
        return Ok(None);
    }
    let days = fromiso
        .call1((end,))
        .and_then(|e| e.sub(fromiso.call1((start,))?))
        .and_then(|d| d.getattr(intern!(py, "days")));
    match days {
        Ok(d) => Ok(Some(d)),
        Err(e) if is_value_or_type(py, &e) => Ok(None),
        Err(e) => Err(e),
    }
}

fn fromiso<'py>(m: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("datetime")?.getattr("fromisoformat")
}

#[pyfunction]
pub fn ep_span_days<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let f = fromiso(&ep(py)?)?;
    Ok(span_days(&f, start, end)?.unwrap_or_else(|| py.None().into_bound(py)))
}

/// `_is_annual_duration`: instants, and durations of roughly a year.
fn is_annual_duration<'py>(
    fromiso: &Bound<'py, PyAny>,
    lo: &Bound<'py, PyAny>,
    hi: &Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    if !start.is_truthy()? {
        return Ok(true); // instant (balance-sheet) fact
    }
    match span_days(fromiso, start, end)? {
        Some(span) => between(&span, lo, hi),
        None => Ok(false),
    }
}

#[pyfunction]
pub fn ep_is_annual_duration<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let m = ep(py)?;
    is_annual_duration(
        &fromiso(&m)?,
        &m.getattr("_MIN_ANNUAL_DAYS")?,
        &m.getattr("_MAX_ANNUAL_DAYS")?,
        start,
        end,
    )
}

/// (wanted tags, zero-padded CIK, the `us-gaap` block)
type ParseHead<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);

/// The head shared by both parsers: the default tag filter, the zero-padded
/// CIK and the `us-gaap` block, or None where the original returns `[]`.
fn parse_head<'py>(
    m: &Bound<'py, PyModule>,
    payload: &Bound<'py, PyAny>,
    wanted_tags: &Bound<'py, PyAny>,
) -> PyResult<Option<ParseHead<'py>>> {
    let py = m.py();
    let wanted = if wanted_tags.is_none() {
        builtin(py, "set")?.call1((m.getattr("all_tags")?.call0()?,))?
    } else {
        wanted_tags.clone()
    };
    let cik = match payload.get_item("cik").and_then(|c| zfill10(py, &c)) {
        Ok(c) => c,
        Err(e)
            if e.is_instance_of::<PyKeyError>(py)
                || e.is_instance_of::<PyTypeError>(py)
                || e.is_instance_of::<PyValueError>(py) =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let us_gaap = get_or(
        &get_or(payload, "facts", empty_dict(py))?,
        "us-gaap",
        empty_dict(py),
    )?;
    if !us_gaap.is_truthy()? {
        return Ok(None);
    }
    Ok(Some((wanted, cik, us_gaap)))
}

/// Every `(tag, unit, entry)` of the wanted tags, in document order.
fn for_each_entry<'py>(
    wanted: &Bound<'py, PyAny>,
    us_gaap: &Bound<'py, PyAny>,
    mut f: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>, &Bound<'py, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let py = wanted.py();
    for item in items(us_gaap)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        if !wanted.contains(&kv[0])? {
            continue;
        }
        for unit_item in items(&get_or(&kv[1], "units", empty_dict(py))?)?.try_iter()? {
            let ue = unpack(&unit_item?, 2)?;
            for entry in ue[1].try_iter()? {
                f(&kv[0], &ue[0], &entry?)?;
            }
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (payload, wanted_tags=None))]
pub fn ep_parse_company_facts<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
    wanted_tags: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let m = ep(py)?;
    let rows = PyList::empty(py);
    let none = py.None().into_bound(py);
    let Some((wanted, cik, us_gaap)) = parse_head(&m, payload, wanted_tags.unwrap_or(&none))?
    else {
        return Ok(rows);
    };
    let forms = m.getattr("_ANNUAL_FORMS")?;
    let (lo, hi) = (
        m.getattr("_MIN_ANNUAL_DAYS")?,
        m.getattr("_MAX_ANNUAL_DAYS")?,
    );
    let fi = fromiso(&m)?;
    for_each_entry(&wanted, &us_gaap, |tag, unit, entry| {
        if !forms.contains(get(entry, "form")?)? {
            return Ok(());
        }
        let end = get(entry, "end")?;
        if !end.is_truthy()? {
            return Ok(());
        }
        let start = get(entry, "start")?;
        if !is_annual_duration(&fi, &lo, &hi, &start, &end)? {
            return Ok(());
        }
        rows.append(tup(
            py,
            vec![
                cik.clone(),
                tag.clone(),
                end,
                start,
                get(entry, "val")?,
                unit.clone(),
                get(entry, "form")?,
                get_or(entry, "accn", pystr(py, ""))?,
                get(entry, "filed")?,
            ],
        )?)
    })?;
    Ok(rows)
}

#[pyfunction]
#[pyo3(signature = (payload, wanted_tags=None))]
pub fn ep_parse_company_quarterly_facts<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
    wanted_tags: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let m = ep(py)?;
    let rows = PyList::empty(py);
    let none = py.None().into_bound(py);
    let Some((wanted, cik, us_gaap)) = parse_head(&m, payload, wanted_tags.unwrap_or(&none))?
    else {
        return Ok(rows);
    };
    let forms = m.getattr("_QUARTERLY_FORMS")?;
    let max_ytd = m.getattr("_MAX_YTD_DAYS")?;
    let zero = int(py, 0)?;
    let fi = fromiso(&m)?;
    for_each_entry(&wanted, &us_gaap, |tag, unit, entry| {
        if !forms.contains(get(entry, "form")?)? {
            return Ok(());
        }
        let end = get(entry, "end")?;
        if !end.is_truthy()? {
            return Ok(());
        }
        let start = get(entry, "start")?;
        // Instants are balance-sheet points and are kept as filed;
        // durations are kept up to a year (the year-to-date ladder).
        if let Some(span) = span_days(&fi, &start, &end)?
            && !(zero.lt(&span)? && span.le(&max_ytd)?)
        {
            return Ok(());
        }
        let start_or_empty = if start.is_truthy()? {
            start
        } else {
            pystr(py, "")
        };
        rows.append(tup(
            py,
            vec![
                cik.clone(),
                tag.clone(),
                start_or_empty,
                end,
                get(entry, "val")?,
                unit.clone(),
                get(entry, "form")?,
                get_or(entry, "accn", pystr(py, ""))?,
                get(entry, "filed")?,
            ],
        )?)
    })?;
    Ok(rows)
}

/// `_canonical_period_ends`: {end as filed: the end it should be read as}.
#[pyfunction]
pub fn ep_canonical_period_ends<'py>(
    py: Python<'py>,
    spans_by_tag: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let fi = fromiso(&m)?;
    let tolerance = m.getattr("_PERIOD_END_TOLERANCE_DAYS")?;
    let counts = PyDict::new(py);
    let zero = int(py, 0)?;
    for spans in spans_by_tag.call_method0("values")?.try_iter()? {
        for key in spans?.try_iter()? {
            let se = unpack(&key?, 2)?;
            let n = get_or(counts.as_any(), &se[1], zero.clone())?.add(1)?;
            counts.set_item(&se[1], n)?;
        }
    }

    let canonical = PyDict::new(py);
    let max = builtin(py, "max")?;
    let close_cluster = |cluster: &[Bound<'py, PyAny>]| -> PyResult<()> {
        if cluster.is_empty() {
            return Ok(());
        }
        // Most-reported wins; the later date breaks a tie.
        let keyed = PyList::empty(py);
        for end in cluster {
            keyed.append(tup(py, vec![counts.as_any().get_item(end)?, end.clone()])?)?;
        }
        let best = max.call1((keyed,))?.get_item(1)?;
        for end in cluster {
            canonical.set_item(end, &best)?;
        }
        Ok(())
    };

    let mut cluster: Vec<Bound<'py, PyAny>> = Vec::new();
    for end in builtin(py, "sorted")?.call1((&counts,))?.try_iter()? {
        let end = end?;
        if let Some(first) = cluster.first() {
            let gap = match span_days(&fi, first, &end)? {
                Some(d) if d.is_truthy()? => d,
                _ => zero.clone(),
            };
            if gap.le(&tolerance)? {
                cluster.push(end);
                continue;
            }
        }
        close_cluster(&cluster)?;
        cluster = vec![end];
    }
    close_cluster(&cluster)?;
    Ok(canonical)
}

/// `_apply_canonical_ends`: one tag's spans on the canonical period ends.
#[pyfunction]
pub fn ep_apply_canonical_ends<'py>(
    py: Python<'py>,
    spans: &Bound<'py, PyAny>,
    canonical: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let rewritten = PyDict::new(py);
    for item in items(spans)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let se = unpack(&kv[0], 2)?;
        let target = get_or(canonical, &se[1], se[1].clone())?;
        let key = tup(py, vec![se[0].clone(), target.clone()])?;
        if target.eq(&se[1])? || !rewritten.contains(&key)? {
            rewritten.set_item(key, &kv[1])?;
        }
    }
    Ok(rewritten)
}

/// `_median`
fn median<'py>(py: Python<'py>, values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ordered = builtin(py, "sorted")?.call1((values,))?;
    let n = ordered.len()? as i64;
    let middle = n / 2;
    if n % 2 != 0 {
        return ordered.get_item(middle);
    }
    ordered
        .get_item(middle - 1)?
        .add(ordered.get_item(middle)?)?
        .div(2)
}

#[pyfunction]
pub fn ep_median<'py>(py: Python<'py>, values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    median(py, values)
}

/// `_filing_basis_factors`: {filing: the factor putting it on the newest basis}.
#[pyfunction]
pub fn ep_filing_basis_factors<'py>(
    py: Python<'py>,
    filings: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let placed = PyDict::new(py);
    let factors = PyDict::new(py);
    for item in filings.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (key, spans) = (&kv[0], &kv[1]);
        let ratios = PyList::empty(py);
        for span_item in items(spans)?.try_iter()? {
            let sv = unpack(&span_item?, 2)?;
            let value = unpack(&sv[1], 2)?.swap_remove(0);
            if value.is_truthy()? && placed.contains(&sv[0])? {
                ratios.append(placed.as_any().get_item(&sv[0])?.div(&value)?)?;
            }
        }
        let factor = if ratios.is_truthy()? {
            median(py, ratios.as_any())?
        } else {
            PyFloat::new(py, 1.0).into_any()
        };
        factors.set_item(key, factor)?;
        for span_item in items(spans)?.try_iter()? {
            let sv = unpack(&span_item?, 2)?;
            let value = unpack(&sv[1], 2)?.swap_remove(0);
            let scaled = value.mul(factors.as_any().get_item(key)?)?;
            if !placed.contains(&sv[0])? {
                placed.set_item(&sv[0], scaled)?;
            }
        }
    }
    Ok(factors)
}

/// `_derive_quarterly_series`: one tag's three-month series, by period end.
#[pyfunction]
#[pyo3(signature = (spans, additive=None))]
pub fn ep_derive_quarterly_series<'py>(
    py: Python<'py>,
    spans: &Bound<'py, PyAny>,
    additive: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let fi = fromiso(&m)?;
    let (q_lo, q_hi) = (
        m.getattr("_MIN_QUARTER_DAYS")?,
        m.getattr("_MAX_QUARTER_DAYS")?,
    );
    let instants = PyDict::new(py);
    let direct = PyDict::new(py);
    let shortest = PyDict::new(py);
    let ladders = PyDict::new(py);

    for item in items(spans)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let se = unpack(&kv[0], 2)?;
        let vu = unpack(&kv[1], 2)?;
        let (start, end, value, unit) = (&se[0], &se[1], &vu[0], &vu[1]);
        if !start.is_truthy()? {
            instants.set_item(end, tup(py, vec![value.clone(), unit.clone()])?)?;
            continue;
        }
        let Some(span) = span_days(&fi, start, end)? else {
            continue;
        };
        if between(&span, &q_lo, &q_hi)? {
            direct.set_item(end, tup(py, vec![value.clone(), unit.clone()])?)?;
        }
        let held = get(shortest.as_any(), end)?;
        if held.is_none() || span.lt(held.get_item(0)?)? {
            shortest.set_item(
                end,
                tup(py, vec![span.clone(), value.clone(), unit.clone()])?,
            )?;
        }
        let rungs = match ladders.get_item(start)? {
            Some(r) => r,
            None => {
                let r = PyList::empty(py).into_any();
                ladders.set_item(start, &r)?;
                r
            }
        };
        rungs.call_method1(
            "append",
            (tup(py, vec![end.clone(), value.clone(), unit.clone()])?,),
        )?;
    }

    let additive = match additive {
        Some(a) => a.is_truthy()?,
        None => true,
    };
    let out = instants.copy()?;
    if !additive {
        let shortest_values = PyDict::new(py);
        for item in shortest.iter() {
            let t = unpack(&item.1, 3)?;
            shortest_values.set_item(item.0, tup(py, vec![t[1].clone(), t[2].clone()])?)?;
        }
        out.update(shortest_values.as_mapping())?;
        return Ok(out);
    }

    let (g_lo, g_hi) = (
        m.getattr("_MIN_QUARTER_GAP_DAYS")?,
        m.getattr("_MAX_QUARTER_GAP_DAYS")?,
    );
    let by_end = py.import("operator")?.getattr("itemgetter")?.call1((0,))?;
    let derived = PyDict::new(py);
    for (_start, rungs) in ladders.iter() {
        rungs.call_method("sort", (), Some(&kwargs(py, &[("key", &by_end)])?))?;
        let n = rungs.len()?;
        for i in 1..n {
            let cur = unpack(&rungs.get_item(i)?, 3)?;
            let prev = unpack(&rungs.get_item(i - 1)?, 3)?;
            let ok = match span_days(&fi, &prev[0], &cur[0])? {
                Some(gap) => between(&gap, &g_lo, &g_hi)?,
                None => false,
            };
            if !ok {
                continue;
            }
            derived.set_item(
                &cur[0],
                tup(py, vec![cur[1].sub(&prev[1])?, cur[2].clone()])?,
            )?;
        }
    }

    // Instants are already per-period; the quarter-length durations are the
    // authority for everything they cover.
    out.update(derived.as_mapping())?;
    out.update(direct.as_mapping())?;
    Ok(out)
}

// --- ingest ---------------------------------------------------------------

/// `iter_bulk_documents`: the archive's companyfacts documents, one at a
/// time. Like the generator, it opens the archive on the first `next()` and
/// closes it when the members run out or an uncaught error ends it.
#[pyclass(module = "investa_core")]
pub struct BulkDocuments {
    zip_path: Py<PyAny>,
    ciks: Py<PyAny>,
    archive: Option<Py<PyAny>>,
    members: Option<Py<PyAny>>,
    done: bool,
}

impl BulkDocuments {
    fn finish(&mut self, py: Python<'_>, err: Option<&PyErr>) -> PyResult<()> {
        self.done = true;
        self.members = None;
        if let Some(archive) = self.archive.take() {
            let archive = archive.bind(py);
            let exit = archive.get_type().getattr("__exit__")?;
            match err {
                None => exit.call1((archive, py.None(), py.None(), py.None()))?,
                Some(e) => exit.call1((archive, e.get_type(py), e.value(py), e.traceback(py)))?,
            };
        }
        Ok(())
    }

    fn advance(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.archive.is_none() {
            let archive = py
                .import("zipfile")?
                .getattr("ZipFile")?
                .call1((self.zip_path.bind(py),))?;
            let entered = archive
                .get_type()
                .getattr("__enter__")?
                .call1((&archive,))?;
            self.archive = Some(entered.unbind());
            let members = self
                .archive
                .as_ref()
                .expect("just set")
                .bind(py)
                .call_method0("infolist")?
                .try_iter()?;
            self.members = Some(members.into_any().unbind());
        }
        let archive = self.archive.as_ref().expect("open").bind(py).clone();
        let members = self.members.as_ref().expect("open").bind(py).clone();
        let ciks = self.ciks.bind(py);
        let decode_error = py.import("json")?.getattr("JSONDecodeError")?;
        loop {
            let Some(info) = members.call_method0("__next__").map(Some).or_else(|e| {
                if e.is_instance_of::<PyStopIteration>(py) {
                    Ok(None)
                } else {
                    Err(e)
                }
            })?
            else {
                return Ok(None);
            };
            if !info
                .getattr("filename")?
                .call_method1("startswith", ("CIK",))?
                .is_truthy()?
                || !info
                    .getattr("filename")?
                    .call_method1("endswith", (".json",))?
                    .is_truthy()?
            {
                continue;
            }
            if !ciks.is_none() {
                // Filenames are CIK##########.json
                let digits = info
                    .getattr("filename")?
                    .get_item(PySlice::new(py, 3, -5, 1))?;
                if !ciks.contains(zfill10(py, &digits)?)? {
                    continue;
                }
            }
            let loaded = archive.call_method1("open", (&info,)).and_then(|handle| {
                with1(py, &handle, |h| {
                    let text = py.import("io")?.getattr("TextIOWrapper")?.call(
                        (h,),
                        Some(&kwargs(py, &[("encoding", &pystr(py, "utf-8"))])?),
                    )?;
                    py.import("json")?.call_method1("load", (text,))
                })
            });
            match loaded {
                Ok(Some(doc)) => return Ok(Some(doc.unbind())),
                Ok(None) => continue,
                Err(e)
                    if e.value(py).is_instance(&decode_error)?
                        || e.is_instance_of::<PyKeyError>(py)
                        || e.is_instance_of::<PyOSError>(py) =>
                {
                    let o = Ops::new(py)?;
                    log(
                        py,
                        "warning",
                        format!(
                            "EDGAR: skipping {}: {}",
                            o.fmt(&info.getattr("filename")?)?,
                            es(py, &e)
                        ),
                    )?;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

#[pymethods]
impl BulkDocuments {
    #[new]
    #[pyo3(signature = (zip_path, ciks=None))]
    fn new(py: Python<'_>, zip_path: Py<PyAny>, ciks: Option<Py<PyAny>>) -> Self {
        BulkDocuments {
            zip_path,
            ciks: ciks.unwrap_or_else(|| py.None()),
            archive: None,
            members: None,
            done: false,
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.done {
            return Err(PyStopIteration::new_err(()));
        }
        match self.advance(py) {
            Ok(Some(doc)) => Ok(doc),
            Ok(None) => {
                self.finish(py, None)?;
                Err(PyStopIteration::new_err(()))
            }
            Err(e) => {
                self.finish(py, Some(&e))?;
                Err(e)
            }
        }
    }

    /// `generator.close()`: release the archive.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.done {
            self.finish(py, None)?;
        }
        Ok(())
    }
}

/// `d[key] += n` on the stats dict.
fn bump<'py>(d: &Bound<'py, PyDict>, key: &str, n: &Bound<'py, PyAny>) -> PyResult<()> {
    let v = d.as_any().get_item(key)?.add(n)?;
    d.set_item(key, v)
}

#[pyfunction]
#[pyo3(signature = (zip_path, ciks=None, limit=None, progress_every=None))]
pub fn ep_ingest_bulk<'py>(
    py: Python<'py>,
    zip_path: &Bound<'py, PyAny>,
    ciks: Option<&Bound<'py, PyAny>>,
    limit: Option<&Bound<'py, PyAny>>,
    progress_every: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let o = Ops::new(py)?;
    let none = py.None().into_bound(py);
    let ciks = ciks.unwrap_or(&none);
    let limit = limit.unwrap_or(&none);
    let default_every = int(py, 250)?;
    let progress_every = progress_every.unwrap_or(&default_every);
    let store = m.getattr("get_store")?.call0()?;
    let stats = PyDict::new(py);
    for k in ["companies", "facts", "empty"] {
        stats.set_item(k, 0)?;
    }
    let mut batch = PyList::empty(py);
    let wanted = builtin(py, "set")?.call1((m.getattr("all_tags")?.call0()?,))?;
    let one = int(py, 1)?;
    let len = builtin(py, "len")?;

    let documents = m
        .getattr("iter_bulk_documents")?
        .call((zip_path,), Some(&kwargs(py, &[("ciks", ciks)])?))?
        .try_iter()?;
    let parse = m.getattr("parse_company_facts")?;
    for document in documents {
        let document = document?;
        let rows = parse.call(
            (&document,),
            Some(&kwargs(py, &[("wanted_tags", &wanted)])?),
        )?;
        let cik = zfill10(py, &get_or(&document, "cik", pystr(py, ""))?)?;

        if rows.is_truthy()? {
            batch.call_method1("extend", (&rows,))?;
            bump(&stats, "facts", &len.call1((&rows,))?)?;
        } else {
            bump(&stats, "empty", &one)?;
        }

        store.call_method1("mark_ingested", (&cik, len.call1((&rows,))?))?;
        bump(&stats, "companies", &one)?;

        if batch.len() >= 50_000 {
            store.call_method1("upsert_facts", (&batch,))?;
            batch = PyList::empty(py);
        }

        let companies = stats.as_any().get_item("companies")?;
        if companies.rem(progress_every)?.eq(0)? {
            log(
                py,
                "info",
                format!(
                    "EDGAR: ingested {} companies, {} facts",
                    o.fmt(&companies)?,
                    o.fmt(&stats.as_any().get_item("facts")?)?
                ),
            )?;
        }
        if limit.is_truthy()? && stats.as_any().get_item("companies")?.ge(limit)? {
            break;
        }
    }
    // The loop's iterator is released here, as the `for` statement releases it.

    if batch.is_truthy()? {
        store.call_method1("upsert_facts", (&batch,))?;
    }

    log(
        py,
        "info",
        format!(
            "EDGAR: bulk ingest complete — {} companies, {} facts, {} with no usable data",
            o.fmt(&stats.as_any().get_item("companies")?)?,
            o.fmt(&stats.as_any().get_item("facts")?)?,
            o.fmt(&stats.as_any().get_item("empty")?)?
        ),
    )?;
    Ok(stats)
}

#[pyfunction]
pub fn ep_ingest_company<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let cik = zfill10(py, cik)?;
    let url = m.getattr("_COMPANY_FACTS_URL")?.call_method(
        "format",
        (),
        Some(&kwargs(py, &[("cik", &cik)])?),
    )?;
    let payload = m.getattr("sec_get_json")?.call1((url,))?;
    if !payload.is_truthy()? {
        return int(py, 0);
    }
    let rows = m.getattr("parse_company_facts")?.call1((payload,))?;
    let store = m.getattr("get_store")?.call0()?;
    store.call_method1("upsert_facts", (&rows,))?;
    let len = builtin(py, "len")?;
    store.call_method1("mark_ingested", (&cik, len.call1((&rows,))?))?;
    len.call1((&rows,))
}

#[pyfunction]
pub fn ep_ingest_company_quarterly<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    max_age_days: &Bound<'py, PyAny>,
    empty_retry_hours: &Bound<'py, PyAny>,
    timeout: &Bound<'py, PyAny>,
    retries: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let cik = zfill10(py, cik)?;
    let store = m.getattr("get_store")?.call0()?;
    let failed = m.getattr("_QUARTERLY_FETCH_FAILED")?;

    let state = store.call_method1("quarterly_ingest_state", (&cik,))?;
    if !state.is_none() {
        let lc = unpack(&state, 2)?;
        let age = m
            .getattr("datetime")?
            .call_method0("now")?
            .sub(&lc[0])?
            .call_method0("total_seconds")?;
        let cooldown = if lc[1].eq(&failed)? {
            empty_retry_hours.mul(3600)?
        } else {
            max_age_days.mul(86400)?
        };
        if age.lt(cooldown)? {
            return int(py, 0);
        }
    }

    let url = m.getattr("_COMPANY_FACTS_URL")?.call_method(
        "format",
        (),
        Some(&kwargs(py, &[("cik", &cik)])?),
    )?;
    let payload = m.getattr("sec_get_json")?.call(
        (url,),
        Some(&kwargs(py, &[("timeout", timeout), ("retries", retries)])?),
    )?;
    if !payload.is_truthy()? {
        store.call_method1("mark_quarterly_ingested", (&cik, &failed))?;
        return int(py, 0);
    }

    let rows = m
        .getattr("parse_company_quarterly_facts")?
        .call1((payload,))?;
    store.call_method1("upsert_quarterly_facts", (&rows,))?;
    let len = builtin(py, "len")?;
    store.call_method1("mark_quarterly_ingested", (&cik, len.call1((&rows,))?))?;
    let o = Ops::new(py)?;
    log(
        py,
        "info",
        format!(
            "EDGAR: {} quarterly facts for CIK {}",
            o.fmt(&len.call1((&rows,))?)?,
            o.fmt(&cik)?
        ),
    )?;
    len.call1((&rows,))
}

// --- concept resolution ---------------------------------------------------

/// `resolve_concept`: walk a fallback chain, period by period.
#[pyfunction]
pub fn ep_resolve_concept<'py>(
    py: Python<'py>,
    tag_series: &Bound<'py, PyAny>,
    chain: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let values = PyDict::new(py);
    let provenance = PyDict::new(py);
    for tag in chain.try_iter()? {
        let tag = tag?;
        let series = get(tag_series, &tag)?;
        if !series.is_truthy()? {
            continue;
        }
        for item in items(&series)?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let value = unpack(&kv[1], 2)?.swap_remove(0);
            if !values.contains(&kv[0])? {
                values.set_item(&kv[0], value)?;
                provenance.set_item(&kv[0], &tag)?;
            }
        }
    }
    PyTuple::new(py, [values.into_any(), provenance.into_any()])
}

/// `concepts or list(chains)` and the tags their chains need.
fn wanted_and_tags<'py>(
    py: Python<'py>,
    chains: &Bound<'py, PyAny>,
    concepts: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PySet>)> {
    let wanted = if concepts.is_truthy()? {
        concepts.clone()
    } else {
        builtin(py, "list")?.call1((chains,))?
    };
    let tags_needed = PySet::empty(py)?;
    for name in wanted.try_iter()? {
        for tag in get_or(chains, name?, empty_list(py))?.try_iter()? {
            tags_needed.add(tag?)?;
        }
    }
    Ok((wanted, tags_needed))
}

/// The resolved values (`which` 0) or provenance (1) of every wanted concept
/// that has any.
fn resolve_all<'py>(
    m: &Bound<'py, PyModule>,
    chains: &Bound<'py, PyAny>,
    wanted: &Bound<'py, PyAny>,
    tag_series: &Bound<'py, PyAny>,
    which: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let py = m.py();
    let resolve = m.getattr("resolve_concept")?;
    let resolved = PyDict::new(py);
    for name in wanted.try_iter()? {
        let name = name?;
        let pair = unpack(
            &resolve.call1((tag_series, get_or(chains, &name, empty_list(py))?))?,
            2,
        )?;
        if pair[which].is_truthy()? {
            resolved.set_item(name, &pair[which])?;
        }
    }
    Ok(resolved)
}

#[pyfunction]
#[pyo3(signature = (cik, concepts=None))]
pub fn ep_get_concept_values<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concepts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let none = py.None().into_bound(py);
    let chains = m.getattr("all_concepts")?.call0()?;
    let (wanted, tags_needed) = wanted_and_tags(py, &chains, concepts.unwrap_or(&none))?;
    let tag_series = m
        .getattr("get_store")?
        .call0()?
        .call_method1("get_many_tag_series", (cik, tags_needed))?;
    resolve_all(&m, &chains, &wanted, &tag_series, 0)
}

#[pyfunction]
#[pyo3(signature = (cik, concepts=None))]
pub fn ep_get_concept_provenance<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concepts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let none = py.None().into_bound(py);
    let chains = m.getattr("all_concepts")?.call0()?;
    let (wanted, tags_needed) = wanted_and_tags(py, &chains, concepts.unwrap_or(&none))?;
    let tag_series = m
        .getattr("get_store")?
        .call0()?
        .call_method1("get_many_tag_series", (cik, tags_needed))?;
    resolve_all(&m, &chains, &wanted, &tag_series, 1)
}

/// `_values_by_filing`: one dict per (tag, filing), most trusted first.
#[pyfunction]
pub fn ep_values_by_filing<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concept: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = ep(py)?;
    let ordered = PyList::empty(py);
    let tags = get_or(
        &m.getattr("all_concepts")?.call0()?,
        concept,
        empty_list(py),
    )?;
    if !tags.is_truthy()? {
        return Ok(ordered);
    }
    let by_filing = m
        .getattr("get_store")?
        .call0()?
        .call_method1("get_tag_series_by_filing", (cik, &tags))?;
    let dict = builtin(py, "dict")?;
    for tag in tags.try_iter()? {
        let filings = get(&by_filing, tag?)?;
        let filings = if filings.is_truthy()? {
            filings
        } else {
            empty_list(py)
        };
        for filing in filings.try_iter()? {
            ordered.append(dict.call1((filing?,))?)?;
        }
    }
    Ok(ordered)
}

/// `_filed_ratio`: later over earlier, as the most authoritative filing has it.
#[pyfunction]
pub fn ep_filed_ratio<'py>(
    py: Python<'py>,
    by_filing: &Bound<'py, PyAny>,
    earlier: &Bound<'py, PyAny>,
    later: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for values in by_filing.try_iter()? {
        let values = values?;
        let first = get(&values, earlier)?;
        let second = get(&values, later)?;
        if first.is_truthy()? && second.is_truthy()? {
            return second.div(first);
        }
    }
    Ok(py.None().into_bound(py))
}

#[pyfunction]
pub fn ep_split_consistent_series<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concept: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let assembled = get_or(
        &m.getattr("get_concept_values")?
            .call1((cik, PyList::new(py, [concept])?))?,
        concept,
        empty_dict(py),
    )?;
    let ordered = builtin(py, "sorted")?.call1((&assembled,))?;
    let n = ordered.len()?;
    if n < 2 {
        return builtin(py, "dict")?.call1((&assembled,));
    }

    let by_filing = m.getattr("_values_by_filing")?.call1((cik, concept))?;
    let filed_ratio = m.getattr("_filed_ratio")?;

    let corrected = PyDict::new(py);
    let newest = ordered.get_item(-1)?;
    corrected.set_item(&newest, assembled.get_item(&newest)?)?;
    for i in (0..n - 1).rev() {
        let earlier = ordered.get_item(i)?;
        let later = ordered.get_item(i + 1)?;
        let mut ratio = filed_ratio.call1((&by_filing, &earlier, &later))?;
        if !ratio.is_truthy()? {
            // No filing reports both years: absence of evidence is not a step.
            let first = get(&assembled, &earlier)?;
            let second = get(&assembled, &later)?;
            ratio = if first.is_truthy()? && second.is_truthy()? {
                second.div(first)?
            } else {
                py.None().into_bound(py)
            };
        }
        let anchor = get(corrected.as_any(), &later)?;
        if !ratio.is_truthy()? || anchor.is_none() {
            corrected.set_item(&earlier, assembled.get_item(&earlier)?)?;
            continue;
        }
        corrected.set_item(&earlier, anchor.div(ratio)?)?;
    }
    Ok(corrected.into_any())
}

#[pyfunction]
pub fn ep_revisions<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concepts: &Bound<'py, PyAny>,
    min_change: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = ep(py)?;
    let o = Ops::new(py)?;
    let chains = m.getattr("all_concepts")?.call0()?;
    let rescaled = m.getattr("_SPLIT_RESCALED_CONCEPTS")?;
    let source = if concepts.is_truthy()? {
        concepts.clone()
    } else {
        builtin(py, "list")?.call1((&chains,))?
    };
    let wanted = PyList::empty(py);
    for name in source.try_iter()? {
        let name = name?;
        if !rescaled.contains(&name)? {
            wanted.append(name)?;
        }
    }
    let tag_to_concept = PyDict::new(py);
    for name in wanted.iter() {
        for tag in get_or(&chains, &name, empty_list(py))?.try_iter()? {
            tag_to_concept.set_item(tag?, &name)?;
        }
    }
    let found = PyList::empty(py);
    if !tag_to_concept.is_truthy()? {
        return Ok(found);
    }

    let history = m.getattr("get_store")?.call0()?.call_method1(
        "get_tag_revisions",
        (cik, builtin(py, "list")?.call1((&tag_to_concept,))?),
    )?;
    let provenance = m.getattr("get_concept_provenance")?.call1((cik, &wanted))?;

    for item in items(&history)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let tp = unpack(&kv[0], 2)?;
        let (tag, period_end, entries) = (&tp[0], &tp[1], &kv[1]);
        if entries.len()? < 2 {
            continue;
        }
        let concept = tag_to_concept.as_any().get_item(tag)?;
        if get(&get_or(&provenance, &concept, empty_dict(py))?, period_end)?.ne(tag)? {
            continue;
        }
        let first = unpack(&entries.get_item(0)?, 3)?;
        let last = unpack(&entries.get_item(-1)?, 3)?;
        let (original, current) = (&first[1], &last[1]);
        if original.is_none() || current.is_none() || !original.is_truthy()? {
            continue;
        }
        let change = current.sub(original)?.div(o.abs(original)?)?;
        if o.abs(&change)?.lt(min_change)? {
            continue;
        }
        let row = PyDict::new(py);
        row.set_item("concept", &concept)?;
        row.set_item("tag", tag)?;
        row.set_item("period_end", period_end)?;
        row.set_item("original", original)?;
        row.set_item("current", current)?;
        row.set_item("change_pct", change.mul(100.0)?)?;
        row.set_item("first_filed", &first[0])?;
        row.set_item("restated_filed", &last[0])?;
        row.set_item("first_form", &first[2])?;
        row.set_item("restated_form", &last[2])?;
        row.set_item("revision_count", builtin(py, "len")?.call1((entries,))?)?;
        found.append(row)?;
    }

    // Largest revisions first.
    let key = PyCFunction::new_closure(py, None, None, |args, _| -> PyResult<Py<PyAny>> {
        let py = args.py();
        let pct = args.get_item(0)?.get_item("change_pct")?;
        Ok(builtin(py, "abs")?.call1((pct,))?.unbind())
    })?;
    let reverse = true.into_pyobject(py)?.to_owned().into_any();
    found.call_method(
        "sort",
        (),
        Some(&kwargs(
            py,
            &[("key", key.as_any()), ("reverse", &reverse)],
        )?),
    )?;
    Ok(found)
}

// --- statement assembly ---------------------------------------------------

#[pyfunction]
pub fn ep_all_chain_tags<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    let chains = ep(py)?.getattr("all_concepts")?.call0()?;
    for chain in chains.call_method0("values")?.try_iter()? {
        for tag in chain?.try_iter()? {
            out.append(tag?)?;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn ep_tags_for<'py>(
    py: Python<'py>,
    concepts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let chains = ep(py)?.getattr("all_concepts")?.call0()?;
    let out = PyList::empty(py);
    for concept in concepts.try_iter()? {
        for tag in get_or(&chains, concept?, empty_list(py))?.try_iter()? {
            out.append(tag?)?;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn ep_quarterly_basis_factors<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let factors = PyDict::new(py);
    let ordered_tags = m
        .getattr("_tags_for")?
        .call1((m.getattr("_SHARE_SCALED")?,))?;
    if !ordered_tags.is_truthy()? {
        return Ok(factors);
    }
    let by_filing = m
        .getattr("get_store")?
        .call0()?
        .call_method1("get_many_tag_spans_by_filing", (cik, &ordered_tags))?;
    let basis = m.getattr("_filing_basis_factors")?;
    // Chain order, so the preferred share tag speaks for a filing it appears in.
    for tag in ordered_tags.try_iter()? {
        let filings = get(&by_filing, tag?)?;
        if !filings.is_truthy()? {
            continue;
        }
        for item in items(&basis.call1((filings,))?)?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            if !factors.contains(&kv[0])? {
                factors.set_item(&kv[0], &kv[1])?;
            }
        }
    }
    Ok(factors)
}

#[pyfunction]
#[pyo3(signature = (filings, factors, inverse=None))]
pub fn ep_spans_on_latest_basis<'py>(
    py: Python<'py>,
    filings: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
    inverse: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let no = false.into_pyobject(py)?.to_owned().into_any();
    let inverse = inverse.unwrap_or(&no);
    let corrected = PyDict::new(py);
    for item in filings.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let factor = get(factors, &kv[0])?;
        let factor = if factor.is_truthy()? {
            factor
        } else {
            PyFloat::new(py, 1.0).into_any()
        };
        for span_item in items(&kv[1])?.try_iter()? {
            let sv = unpack(&span_item?, 2)?;
            let vu = unpack(&sv[1], 2)?;
            if !corrected.contains(&sv[0])? {
                let v = if inverse.is_truthy()? {
                    vu[0].div(&factor)?
                } else {
                    vu[0].mul(&factor)?
                };
                corrected.set_item(&sv[0], tup(py, vec![v, vu[1].clone()])?)?;
            }
        }
    }
    Ok(corrected)
}

#[pyfunction]
pub fn ep_split_adjustment_factors<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let factors = PyDict::new(py);
    let concept_values = m.getattr("get_concept_values")?;
    let consistent = m.getattr("split_consistent_series")?;
    for concept in ["shares_diluted", "shares_basic", "shares_outstanding"] {
        let assembled = get_or(
            &concept_values.call1((cik, PyList::new(py, [concept])?))?,
            concept,
            empty_dict(py),
        )?;
        if !assembled.is_truthy()? {
            continue;
        }
        let corrected = consistent.call1((cik, concept))?;
        for period in assembled.try_iter()? {
            let period = period?;
            if !factors.contains(&period)?
                && get(&assembled, &period)?.is_truthy()?
                && get(&corrected, &period)?.is_truthy()?
            {
                let f = corrected
                    .get_item(&period)?
                    .div(assembled.get_item(&period)?)?;
                factors.set_item(&period, f)?;
            }
        }
    }
    Ok(factors)
}

#[pyfunction]
pub fn ep_apply_split_adjustment<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if !factors.is_truthy()? {
        return Ok(());
    }
    let m = ep(py)?;
    let operator = py.import("operator")?;
    for (group, op) in [("_SHARE_SCALED", "imul"), ("_PER_SHARE_SCALED", "itruediv")] {
        let op = operator.getattr(op)?;
        for concept in m.getattr(group)?.try_iter()? {
            let series = get(values, concept?)?;
            if !series.is_truthy()? {
                continue;
            }
            for item in items(factors)?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                if series.contains(&kv[0])? && kv[1].is_truthy()? {
                    let v = op.call1((series.get_item(&kv[0])?, &kv[1]))?;
                    series.set_item(&kv[0], v)?;
                }
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn ep_concept_labels<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let out = PyDict::new(py);
    for name in [
        "_INCOME_LABELS",
        "_BALANCE_LABELS",
        "_CASHFLOW_LABELS",
        "_NEGATE_ON_EMIT",
    ] {
        out.call_method1("update", (m.getattr(name)?,))?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (values, labels, negate=None))]
pub fn ep_frame_from_concepts<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    labels: &Bound<'py, PyAny>,
    negate: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let columns = PyDict::new(py);
    let add = |concept: &Bound<'py, PyAny>, label: &Bound<'py, PyAny>, sign: i64| -> PyResult<()> {
        for item in items(&get_or(values, concept, empty_dict(py))?)?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let v = kv[1].mul(sign)?;
            columns
                .call_method1("setdefault", (&kv[0], PyDict::new(py)))?
                .set_item(label, v)?;
        }
        Ok(())
    };
    for item in items(labels)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        add(&kv[0], &kv[1], 1)?;
    }
    let negate = match negate {
        Some(n) if n.is_truthy()? => n.clone(),
        _ => empty_dict(py),
    };
    for item in items(&negate)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        add(&kv[0], &kv[1], -1)?;
    }

    if !columns.is_truthy()? {
        return pd.getattr("DataFrame")?.call0();
    }

    let mut frame = pd.getattr("DataFrame")?.call1((&columns,))?;
    let coerced = pd.getattr("to_datetime")?.call(
        (frame.getattr("columns")?,),
        Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
    )?;
    frame.setattr("columns", coerced)?;
    let mask = frame.getattr("columns")?.call_method0("notna")?;
    frame = frame
        .getattr("loc")?
        .get_item(PyTuple::new(py, [PySlice::full(py).into_any(), mask])?)?;
    // yfinance orders newest-first, and `financial_ratios` reads columns[0] as
    // the latest period.
    let one = int(py, 1)?;
    let no = false.into_pyobject(py)?.to_owned().into_any();
    frame.call_method(
        "sort_index",
        (),
        Some(&kwargs(py, &[("axis", &one), ("ascending", &no)])?),
    )
}

/// `list(A) + list(B) + list(C)` of the three statement concept tables.
fn statement_concepts<'py>(m: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyAny>> {
    let list = builtin(m.py(), "list")?;
    list.call1((m.getattr("INCOME_CONCEPTS")?,))?
        .add(list.call1((m.getattr("BALANCE_CONCEPTS")?,))?)?
        .add(list.call1((m.getattr("CASHFLOW_CONCEPTS")?,))?)
}

/// The three frames and their derived rows, as both statement readers build them.
fn statement_frames<'py>(
    m: &Bound<'py, PyModule>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = m.py();
    let frame = m.getattr("_frame_from_concepts")?;
    let income = frame.call1((values, m.getattr("_INCOME_LABELS")?))?;
    let balance = frame.call1((values, m.getattr("_BALANCE_LABELS")?))?;
    let cashflow = frame.call(
        (values, m.getattr("_CASHFLOW_LABELS")?),
        Some(&kwargs(py, &[("negate", &m.getattr("_NEGATE_ON_EMIT")?)])?),
    )?;

    m.getattr("_add_derived_rows")?
        .call1((&income, &balance, values, &cashflow))?;

    let out = PyDict::new(py);
    out.set_item("financials", income)?;
    out.set_item("balance_sheet", balance)?;
    out.set_item("cashflow", cashflow)?;
    Ok(out)
}

#[pyfunction]
pub fn ep_get_statements<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let concept_names = statement_concepts(&m)?;
    let session = m.getattr("company_session")?.call1((cik,))?;
    let values = with1(py, &session, |_| {
        let values = m
            .getattr("get_concept_values")?
            .call1((cik, &concept_names))?;
        let list = builtin(py, "list")?;
        let scaled = list
            .call1((m.getattr("_SHARE_SCALED")?,))?
            .add(list.call1((m.getattr("_PER_SHARE_SCALED")?,))?)?;
        for concept in scaled.try_iter()? {
            let concept = concept?;
            if values.contains(&concept)? {
                let series = m
                    .getattr("split_consistent_series")?
                    .call1((cik, &concept))?;
                values.set_item(&concept, series)?;
            }
        }
        Ok(values)
    })?
    .ok_or_else(|| {
        PyUnboundLocalError::new_err(
            "cannot access local variable 'values' where it is not associated with a value",
        )
    })?;
    statement_frames(&m, &values)
}

#[pyfunction]
#[pyo3(signature = (cik, concepts=None))]
pub fn ep_get_quarterly_concept_values<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    concepts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let none = py.None().into_bound(py);
    let chains = m.getattr("all_concepts")?.call0()?;
    let (wanted, tags_needed) = wanted_and_tags(py, &chains, concepts.unwrap_or(&none))?;

    let store = m.getattr("get_store")?.call0()?;
    let spans = store.call_method1("get_many_tag_spans", (cik, &tags_needed))?;

    // Split-sensitive tags are put on one basis first.
    let set = builtin(py, "set")?;
    let tags_for = m.getattr("_tags_for")?;
    let share_tags = set.call1((tags_for.call1((m.getattr("_SHARE_SCALED")?,))?,))?;
    let per_share_tags = set.call1((tags_for.call1((m.getattr("_PER_SHARE_SCALED")?,))?,))?;
    let sensitive = tags_needed.bitand(share_tags.bitor(&per_share_tags)?)?;
    if sensitive.is_truthy()? {
        let factors = m.getattr("_quarterly_basis_factors")?.call1((cik,))?;
        if factors.is_truthy()? {
            let by_filing =
                store.call_method1("get_many_tag_spans_by_filing", (cik, &sensitive))?;
            let latest = m.getattr("_spans_on_latest_basis")?;
            for item in items(&by_filing)?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                let inverse = per_share_tags
                    .contains(&kv[0])?
                    .into_pyobject(py)?
                    .to_owned()
                    .into_any();
                let rebased = latest.call(
                    (&kv[1], &factors),
                    Some(&kwargs(py, &[("inverse", &inverse)])?),
                )?;
                spans.set_item(&kv[0], rebased)?;
            }
        }
    }

    // Before deriving anything: a period end mistyped in one filing is one
    // period, not two.
    let canonical = m.getattr("_canonical_period_ends")?.call1((&spans,))?;
    let derive = m.getattr("_derive_quarterly_series")?;
    let apply = m.getattr("_apply_canonical_ends")?;
    let non_additive = m.getattr("_NON_ADDITIVE_TAGS")?;
    let tag_series = PyDict::new(py);
    for item in items(&spans)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let rows = apply.call1((&kv[1], &canonical))?;
        let additive = (!non_additive.contains(&kv[0])?)
            .into_pyobject(py)?
            .to_owned()
            .into_any();
        let series = derive.call((rows,), Some(&kwargs(py, &[("additive", &additive)])?))?;
        tag_series.set_item(&kv[0], series)?;
    }

    resolve_all(&m, &chains, &wanted, tag_series.as_any(), 0)
}

#[pyfunction]
pub fn ep_get_quarterly_statements<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ep(py)?;
    let concept_names = statement_concepts(&m)?;
    let values = m
        .getattr("get_quarterly_concept_values")?
        .call1((cik, concept_names))?;
    if !values.is_truthy()? {
        return Ok(PyDict::new(py));
    }
    statement_frames(&m, &values)
}

/// `cells(frame, items)` inside `_add_derived_rows`: (column, value) for the
/// periods the frame has, in the given order. Each item's value is computed
/// before its period is stamped, as the generator feeding the original does.
fn cells<'py>(
    pd: &Bound<'py, PyModule>,
    stamps: &Bound<'py, PyAny>,
    frame: &Bound<'py, PyAny>,
    items: impl Iterator<Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>>,
) -> PyResult<Bound<'py, PyList>> {
    let py = pd.py();
    let out = PyList::empty(py);
    let to_datetime = pd.getattr("to_datetime")?;
    let isna = pd.getattr("isna")?;
    for item in items {
        let (period_end, value) = item?;
        if !stamps.contains(&period_end)? {
            let stamp = to_datetime.call(
                (&period_end,),
                Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
            )?;
            stamps.set_item(&period_end, stamp)?;
        }
        let stamp = stamps.get_item(&period_end)?;
        if isna.call1((&stamp,))?.is_truthy()? || !frame.getattr("columns")?.contains(&stamp)? {
            continue;
        }
        out.append(tup(py, vec![stamp, value])?)?;
    }
    Ok(out)
}

/// `((p, f(p, v)) for p, v in d.items() if keep(p))`, lazily.
fn pairs<'py, 'a>(
    d: &'a Bound<'py, PyAny>,
    keep: impl Fn(&Bound<'py, PyAny>) -> PyResult<bool> + 'a,
    f: impl Fn(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> + 'a,
) -> PyResult<impl Iterator<Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> + 'a> {
    let it = items(d)?.try_iter()?;
    Ok(it.filter_map(move |item| {
        let step = || -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
            let kv = unpack(&item?, 2)?;
            if !keep(&kv[0])? {
                return Ok(None);
            }
            let v = f(&kv[0], &kv[1])?;
            Ok(Some((kv[0].clone(), v)))
        };
        step().transpose()
    }))
}

#[pyfunction]
#[pyo3(signature = (income, balance, values, cashflow=None))]
pub fn ep_add_derived_rows<'py>(
    py: Python<'py>,
    income: &Bound<'py, PyAny>,
    balance: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    cashflow: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    let m = ep(py)?;
    let pd = py.import("pandas")?;
    let stamps = m.getattr("_period_stamps")?.call1((values,))?;
    let zero = || PyFloat::new(py, 0.0).into_any();
    let empty =
        |frame: &Bound<'py, PyAny>| -> PyResult<bool> { frame.getattr("empty")?.is_truthy() };

    // Total Debt is rarely tagged as such; it is short-term plus long-term.
    let short_term = get_or(values, "short_term_debt", empty_dict(py))?;
    let long_term = get_or(values, "long_term_debt", empty_dict(py))?;
    let set = builtin(py, "set")?;
    let debt_periods = set
        .call1((&short_term,))?
        .bitor(set.call1((&long_term,))?)?;
    if debt_periods.is_truthy()? && !empty(balance)? {
        let it = debt_periods.try_iter()?.map(|p| {
            let p = p?;
            let v = get_or(&short_term, &p, zero())?.add(get_or(&long_term, &p, zero())?)?;
            Ok((p, v))
        });
        let c = cells(&pd, &stamps, balance, it)?;
        m.getattr("_write_row")?.call1((balance, "Total Debt", c))?;
    }

    // EBIT: pretax income plus interest expense.
    let pretax = get_or(values, "pretax_income", empty_dict(py))?;
    let interest = get_or(values, "interest_expense", empty_dict(py))?;
    if pretax.is_truthy()? && !empty(income)? {
        let it = pairs(
            &pretax,
            |_| Ok(true),
            |p, v| v.add(get_or(&interest, p, zero())?),
        )?;
        let c = cells(&pd, &stamps, income, it)?;
        m.getattr("_write_row")?.call1((income, "Ebit", c))?;
    }

    // Free cash flow is not a filed concept; `capex` is filed as a payment.
    let operating = get_or(values, "operating_cash_flow", empty_dict(py))?;
    let capex = get_or(values, "capex", empty_dict(py))?;
    if let Some(cashflow) = cashflow.filter(|c| !c.is_none())
        && !empty(cashflow)?
    {
        let it = pairs(
            &operating,
            |p| capex.contains(p),
            |p, ocf| ocf.sub(capex.get_item(p)?),
        )?;
        let c = cells(&pd, &stamps, cashflow, it)?;
        m.getattr("_write_row")?
            .call1((cashflow, "Free Cash Flow", c))?;
    }

    // Some filers omit Liabilities entirely; assets minus equity recovers it.
    let assets = get_or(values, "total_assets", empty_dict(py))?;
    let equity = get_or(values, "equity", empty_dict(py))?;
    let label = "Total Liabilities Net Minority Interest";
    if !empty(balance)? && !balance.getattr("index")?.contains(label)? {
        let it = pairs(
            &assets,
            |p| equity.contains(p),
            |p, v| v.sub(equity.get_item(p)?),
        )?;
        let c = cells(&pd, &stamps, balance, it)?;
        m.getattr("_write_row")?.call1((balance, label, c))?;
    }
    Ok(())
}

/// `_ISO_DAY.fullmatch(p)` for an ASCII string: `\d{4}-\d{2}-\d{2}`.
fn ascii_iso_day(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
}

#[pyfunction]
pub fn ep_period_stamps<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let iso_day = m.getattr("_ISO_DAY")?;
    let str_type = py.get_type::<PyString>();
    let ends = PySet::empty(py)?;
    for series in values.call_method0("values")?.try_iter()? {
        for p in series?.try_iter()? {
            let p = p?;
            if !p.get_type().is(&str_type) {
                continue;
            }
            let text = p.cast::<PyString>()?.to_str()?;
            // `\d` is any Unicode digit; only an ASCII string can be judged here.
            let matched = if text.is_ascii() {
                ascii_iso_day(text)
            } else {
                iso_day.call_method1("fullmatch", (&p,))?.is_truthy()?
            };
            if matched {
                ends.add(p)?;
            }
        }
    }
    let ends = builtin(py, "sorted")?.call1((ends,))?;
    if !ends.is_truthy()? {
        return Ok(empty_dict(py));
    }
    let pd = py.import("pandas")?;
    let parsed = pd.getattr("to_datetime")?.call(
        (pd.getattr("Index")?.call1((&ends,))?,),
        Some(&kwargs(
            py,
            &[
                ("errors", &pystr(py, "coerce")),
                ("format", &pystr(py, "%Y-%m-%d")),
            ],
        )?),
    )?;
    builtin(py, "dict")?.call1((builtin(py, "zip")?.call1((ends, parsed))?,))
}

#[pyfunction]
pub fn ep_write_row<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
    label: &Bound<'py, PyAny>,
    cells: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if !cells.is_truthy()? {
        return Ok(());
    }
    let np = py.import("numpy")?;
    let simple = (|| -> PyResult<bool> {
        if !frame
            .getattr("columns")?
            .getattr("is_unique")?
            .is_truthy()?
        {
            return Ok(false);
        }
        let dtype_t = np.getattr("dtype")?;
        let f64_t = np.getattr("float64")?;
        for d in frame.getattr("dtypes")?.try_iter()? {
            let d = d?;
            if !(d.is_instance(&dtype_t)? && d.eq(&f64_t)?) {
                return Ok(false);
            }
        }
        let float_t = py.get_type::<PyFloat>();
        for cell in cells.try_iter()? {
            let cv = unpack(&cell?, 2)?;
            if !cv[1].get_type().is(&float_t) {
                return Ok(false);
            }
        }
        let index = frame.getattr("index")?;
        Ok(!index.contains(label)?
            || index
                .call_method1("get_indexer_for", (PyList::new(py, [label])?,))?
                .getattr("size")?
                .eq(1)?)
    })()?;
    if !simple {
        for cell in cells.try_iter()? {
            let cv = unpack(&cell?, 2)?;
            frame
                .getattr("loc")?
                .set_item(PyTuple::new(py, [label.clone(), cv[0].clone()])?, &cv[1])?;
        }
        return Ok(());
    }
    let by_column = builtin(py, "dict")?.call1((cells,))?;
    if frame.getattr("index")?.contains(label)? {
        let list = builtin(py, "list")?;
        let new_values = list.call1((by_column.call_method0("values")?,))?;
        frame.getattr("loc")?.set_item(
            PyTuple::new(py, [label.clone(), list.call1((&by_column,))?])?,
            new_values,
        )?;
    } else {
        let columns = frame.getattr("columns")?;
        let row = np.call_method1("full", (columns.len()?, f64::NAN))?;
        let position = PyDict::new(py);
        for (j, c) in columns.try_iter()?.enumerate() {
            position.set_item(c?, j)?;
        }
        for item in items(&by_column)?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            row.set_item(position.as_any().get_item(&kv[0])?, &kv[1])?;
        }
        frame.getattr("loc")?.set_item(label, row)?;
    }
    Ok(())
}

#[pyfunction]
pub fn ep_get_sector_concepts<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    sector_model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ep(py)?;
    let list = builtin(py, "list")?;
    for (model, table) in [("bank", "BANK_CONCEPTS"), ("reit", "REIT_CONCEPTS")] {
        if sector_model.eq(model)? {
            return m
                .getattr("get_concept_values")?
                .call1((cik, list.call1((m.getattr(table)?,))?));
        }
    }
    Ok(empty_dict(py))
}
