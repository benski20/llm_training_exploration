//! Outbound-noise monitor, every-call part (`pgw.observability.noise.OutboundNoiseMonitor.guard_packed`):
//! the duplicate-row (mask reuse) fingerprints of a packed wire payload in one call (`noise_dup_keys`),
//! replacing about twenty small numpy operations per trip. Bit-identical to `_packed_columns` +
//! `_dup_fold`'s key computation (wrapping uint64 arithmetic, the same splitmix64 finalizer); the
//! duplicate bookkeeping (seen set, history) stays in Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(noise_dup_keys, m)?)?;
    Ok(())
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
