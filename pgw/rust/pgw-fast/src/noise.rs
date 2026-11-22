//! Outbound-noise monitor, every-call part (`pgw.observability.noise.OutboundNoiseMonitor.guard_packed`):
//! the duplicate-row (mask reuse) fingerprints of a packed wire payload in one call (`noise_dup_keys`),
//! replacing about twenty small numpy operations per trip. Bit-identical to `_packed_columns` +
//! `_dup_fold`'s key computation (wrapping uint64 arithmetic, the same splitmix64 finalizer); the
//! duplicate bookkeeping (seen set, history) stays in Python, or with the native decode driver in
//! [`DupHistory`] (shared by both).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(noise_dup_keys, m)?)?;
    m.add_class::<DupHistory>()?;
    Ok(())
}

/// The duplicate-row history of the outbound-noise monitor (`OutboundNoiseMonitor._dup_record`) in
/// native code, shared by the monitor's Python path and the native decode driver (`decode.rs`), which
/// records a payload's fingerprints before it is sent, with the GIL released. Same bookkeeping as the
/// Python set + deque: a row of which any key was seen is a duplicate (and adds nothing); otherwise
/// every key of the row is added (group order); then the oldest keys beyond `limit` are dropped.
#[pyclass(module = "pgw_fast", frozen)]
pub struct DupHistory {
    inner: std::sync::Mutex<DupState>,
}

#[derive(Default)]
struct DupState {
    seen: std::collections::HashSet<u64>,
    order: std::collections::VecDeque<u64>,
}

impl DupHistory {
    /// Record `rows` rows of `groups` keys (row-major); returns the duplicate rows.
    pub(crate) fn record_raw(&self, keys: &[u64], rows: usize, groups: usize, limit: usize) -> usize {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *g;
        let mut dups = 0usize;
        for t in 0..rows {
            let row = &keys[t * groups..(t + 1) * groups];
            if row.iter().any(|k| st.seen.contains(k)) {
                dups += 1;
                continue;
            }
            for &k in row {
                st.seen.insert(k);
                st.order.push_back(k);
            }
        }
        while st.order.len() > limit {
            if let Some(k) = st.order.pop_front() {
                st.seen.remove(&k);
            }
        }
        dups
    }
}

#[pymethods]
impl DupHistory {
    #[new]
    fn new() -> Self {
        DupHistory { inner: std::sync::Mutex::new(DupState::default()) }
    }

    /// ``record(keys, limit) -> duplicate rows``: ``keys`` uint64 ``(rows, groups)``.
    fn record(&self, py: Python<'_>, keys: &Bound<'_, PyArray2<u64>>, limit: usize) -> PyResult<usize> {
        let ro = keys.try_readonly().map_err(|e| PyValueError::new_err(format!("keys: {e}")))?;
        let (rows, groups) = (ro.shape()[0], ro.shape()[1]);
        let ks = ro.as_slice().map_err(|_| PyValueError::new_err("keys must be C-contiguous"))?;
        Ok(py.allow_threads(|| self.record_raw(ks, rows, groups, limit)))
    }

    /// Replace the history with ``order`` (oldest first).
    fn load(&self, order: Vec<u64>) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.seen.clear();
        g.order.clear();
        for k in order {
            g.seen.insert(k);
            g.order.push_back(k);
        }
    }

    /// Forget everything.
    fn clear(&self) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.seen.clear();
        g.order.clear();
    }

    /// Keys held (with repeats, as the Python deque).
    fn __len__(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).order.len()
    }
}

#[inline(always)]
fn mix64(mut h: u64) -> u64 {
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h ^ (h >> 31)
}

/// Fingerprints `(rows, groups)` of a packed payload (see the module docs).
#[allow(clippy::too_many_arguments)]
pub fn dup_keys(
    payload: &[u8],
    rows: usize,
    n: usize,
    nb: usize,
    cols: &[i64],
    shift: u32,
    mult: &[u64],
    groups: usize,
    per: usize,
    out: &mut [u64],
) -> Result<(), String> {
    if payload.len() != rows * n * nb || !(1..=8).contains(&nb) {
        return Err(format!("packed payload has {} bytes, expected {rows}x{n}x{nb}", payload.len()));
    }
    if groups == 0 || per == 0 || cols.len() < groups * per || mult.len() < groups * per || out.len() != rows * groups
        || shift >= 64
    {
        return Err("bad fingerprint parameters".into());
    }
    if cols.iter().take(groups * per).any(|&c| c < 0 || c as usize >= n) {
        return Err("column index out of range".into());
    }
    let nu = n as u64;
    for t in 0..rows {
        for g in 0..groups {
            let mut acc = 0u64;
            for p in 0..per {
                let idx = g * per + p;
                let off = (t * n + cols[idx] as usize) * nb;
                let mut word = [0u8; 8];
                word[..nb].copy_from_slice(&payload[off..off + nb]);
                let top = u64::from_le_bytes(word) >> shift;
                acc ^= mix64(top.wrapping_mul(mult[idx]).wrapping_add(nu));
            }
            let salt = nu.wrapping_mul(0x9E3779B1).wrapping_add(((g as u64) + 1).wrapping_mul(0x85EBCA77));
            out[t * groups + g] = mix64(acc ^ salt);
        }
    }
    Ok(())
}

/// ``noise_dup_keys(payload, rows, n, nb, cols, shift, mult, groups, per) -> uint64 (rows, groups)``:
/// the noise monitor's duplicate-row fingerprints of a packed payload (``rows * n`` little-endian
/// ``nb``-byte ring elements): per row and group, the XOR over the group's ``per`` columns of
/// ``mix64((v >> shift) * mult[i] + n)``, then ``mix64(fold ^ salt_g)`` with
/// ``salt_g = n * 0x9E3779B1 + (g + 1) * 0x85EBCA77`` (all mod 2^64).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn noise_dup_keys<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
    rows: usize,
    n: usize,
    nb: usize,
    cols: &Bound<'py, PyArray1<i64>>,
    shift: u32,
    mult: &Bound<'py, PyArray1<u64>>,
    groups: usize,
    per: usize,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let ca = cols.try_readonly().map_err(|e| PyValueError::new_err(format!("cols: {e}")))?;
    let ma = mult.try_readonly().map_err(|e| PyValueError::new_err(format!("mult: {e}")))?;
    let cs = ca.as_slice().map_err(|_| PyValueError::new_err("cols must be contiguous"))?;
    let ms = ma.as_slice().map_err(|_| PyValueError::new_err("mult must be contiguous"))?;
    let mut keys = vec![0u64; rows * groups];
    let res = if let Ok(b) = payload.downcast::<PyBytes>() {
        dup_keys(b.as_bytes(), rows, n, nb, cs, shift, ms, groups, per, &mut keys)
    } else if let Ok(a) = payload.downcast::<numpy::PyArrayDyn<u8>>() {
        let ro = a.try_readonly().map_err(|e| PyValueError::new_err(format!("payload: {e}")))?;
        let s = ro.as_slice().map_err(|_| PyValueError::new_err("payload must be C-contiguous"))?;
        dup_keys(s, rows, n, nb, cs, shift, ms, groups, per, &mut keys)
    } else {
        let b: Vec<u8> = payload.extract().map_err(|_| PyValueError::new_err("payload must be bytes-like"))?;
        dup_keys(&b, rows, n, nb, cs, shift, ms, groups, per, &mut keys)
    };
    res.map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, groups), keys)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let _ = py;
    Ok(PyArray2::from_owned_array(py, arr))
}
