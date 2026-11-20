//! # pgw-fast
//!
//! Native hot path for PGW (Python module `pgw_fast`). Each submodule owns one part of the per-call
//! critical path and registers its functions in [`pgw_fast`]:
//!
//! | module | Python counterpart it replaces |
//! |---|---|
//! | [`finish`] | `MaskedBackend._finish`: unpack `y_hat`, Freivalds check, unmask, range check, dequantize (`finish_group`: several sessions' chunks of one linear in one pass over the shared check vectors) |
//! | [`prepare`] | `MaskedBackend._prepare`: quantize (capped), mask, pack |
//! | [`store`] | `MaskReservation.take`: decrypt `W r` chunks, derive `r` |
//! | [`lmhead`] | `h @ lm_head^T` (float32, several rows per pass over the head, fixed summation order; same bits as `pgw.engine.lm_head.lm_head_reference`) |
//!
//! Every function is bit-identical to the Python path (tests in `pgw/tests/native/`), takes numpy
//! arrays / bytes without copying, and releases the GIL while it computes.

use pyo3::prelude::*;

pub mod finish;
pub mod hemask;
pub mod lmhead;
pub mod prepare;
pub mod store;

/// Build/version info (lets Python check it loaded a compatible extension).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `pgw_fast` Python module.
#[pymodule]
fn pgw_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    finish::register(m)?;
    hemask::register(m)?;
    lmhead::register(m)?;
    prepare::register(m)?;
    store::register(m)?;
    Ok(())
}
