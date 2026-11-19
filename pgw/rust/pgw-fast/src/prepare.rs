//! See the crate docs. Functions are registered in [`register`].

use pyo3::prelude::*;

/// Add this module's functions to `pgw_fast`.
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
