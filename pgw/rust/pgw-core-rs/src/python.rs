//! Optional Python bindings (feature `python`), module `pgw_core_rs`.
//!
//! Thin wrappers over the safe Rust API using flat lists, intended for cross-checking the Python
//! core and for auditors. Production hot paths stay in `pgw.core` (vectorized NumPy/torch).
//! Build with `maturin develop -m rust/pgw-core-rs/Cargo.toml --features python`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::{ErrorCode, PgwError};
use crate::prng::Label;
use crate::ring::RingParams;

fn to_py(e: PgwError) -> PyErr {
    let msg = format!("{}: {}", e.code().as_str(), e.message());
    match e.code() {
        ErrorCode::Config | ErrorCode::BadShape | ErrorCode::RingOverflow | ErrorCode::Protocol => {
            PyValueError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

fn ring(name: &str) -> PyResult<RingParams> {
    RingParams::from_name(name).map_err(to_py)
}

fn labels(objs: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Label>> {
    objs.into_iter()
        .map(|o| {
            if o.is_instance_of::<pyo3::types::PyBool>() {
                Err(PyValueError::new_err("config: bool is not a valid PRNG label"))
            } else if let Ok(i) = o.extract::<i64>() {
                Ok(Label::Int(i))
            } else if let Ok(s) = o.extract::<String>() {
                Ok(Label::Str(s))
            } else if let Ok(b) = o.downcast::<PyBytes>() {
                // Only real `bytes` (as in pgw.core.prng); a list of ints is not a bytes label.
                Ok(Label::Bytes(b.as_bytes().to_vec()))
            } else {
                Err(PyValueError::new_err("config: unsupported PRNG label type"))
            }
        })
        .collect()
}

fn expander(seed: &[u8]) -> PyResult<crate::prng::SeedExpander> {
    crate::prng::SeedExpander::new(seed).map_err(to_py)
}

/// `random_bytes(seed, n, labels) -> bytes`
#[pyfunction]
fn random_bytes<'py>(
    py: Python<'py>,
    seed: &[u8],
    n: usize,
    labels_: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = expander(seed)?.random_bytes(n, &labels(labels_)?).map_err(to_py)?;
    Ok(PyBytes::new(py, &out))
}

/// `ring_uniform_rows(seed, cols, k, row_start, row_count, labels) -> list[int]`
#[pyfunction]
fn ring_uniform_rows(
    seed: &[u8],
    cols: usize,
    k: u32,
    row_start: u64,
    row_count: usize,
    labels_: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<u64>> {
    expander(seed)?.ring_uniform_rows(cols, k, row_start, row_count, &labels(labels_)?).map_err(to_py)
}

/// `mask_activations(xq, r, ring) -> list[int]`
#[pyfunction]
fn mask_activations(xq: Vec<i64>, r: Vec<u64>, ring_name: &str) -> PyResult<Vec<u64>> {
    crate::masking::mask_activations(&xq, &r, &ring(ring_name)?).map_err(to_py)
}

/// `unmask_outputs(y_hat, wr, ring, in_features=None) -> list[int]`
#[pyfunction]
#[pyo3(signature = (y_hat, wr, ring_name, in_features=None))]
fn unmask_outputs(y_hat: Vec<u64>, wr: Vec<u64>, ring_name: &str, in_features: Option<u64>) -> PyResult<Vec<i64>> {
    crate::masking::unmask_outputs(&y_hat, &wr, &ring(ring_name)?, in_features).map_err(to_py)
}

/// `ringmm(x, rows, n, w8, m, ring) -> list[int]` (int8-limb algorithm)
#[pyfunction]
fn ringmm(x: Vec<u64>, rows: usize, n: usize, w8: Vec<i8>, m: usize, ring_name: &str) -> PyResult<Vec<u64>> {
    crate::ringmm::ringmm_limbs(&x, rows, n, &w8, m, &ring(ring_name)?).map_err(to_py)
}

/// `freivalds_check(x_hat, y_hat, rows, n, m, u, uw, c, ring) -> list[bool]`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn freivalds_check(
    x_hat: Vec<u64>,
    y_hat: Vec<u64>,
    rows: usize,
    n: usize,
    m: usize,
    u: Vec<u64>,
    uw: Vec<u64>,
    c: usize,
    ring_name: &str,
) -> PyResult<Vec<bool>> {
    crate::freivalds::freivalds_check(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &ring(ring_name)?).map_err(to_py)
}

/// `pack_ring(values, k) -> bytes`
#[pyfunction]
fn pack_ring<'py>(py: Python<'py>, values: Vec<u64>, k: u32) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(py, &crate::packing::pack_ring(&values, k).map_err(to_py)?))
}

/// The `pgw_core_rs` extension module.
#[pymodule]
fn pgw_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PROTOCOL_VERSION", crate::PROTOCOL_VERSION)?;
    m.add_function(wrap_pyfunction!(random_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ring_uniform_rows, m)?)?;
    m.add_function(wrap_pyfunction!(mask_activations, m)?)?;
    m.add_function(wrap_pyfunction!(unmask_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(ringmm, m)?)?;
    m.add_function(wrap_pyfunction!(freivalds_check, m)?)?;
    m.add_function(wrap_pyfunction!(pack_ring, m)?)?;
    Ok(())
}
