//! Native trip kernels for the gateway's TripEngine (chat-speed plan WP-A4a; stub created by WP-A0,
//! owned by WP-A4a).
//!
//! Will hold `prepare_into` / `finish_into` over reused buffers, a single-pass constant-time
//! activation bit-width selection (same choice as the descending search in `prepare`), and the split
//! Freivalds API (input half at prepare time, output half on a helper thread). Every function must be
//! bit-identical to `finish_group` / `prepare_many` and free of data-dependent loop counts.
//!
//! Until WP-A4a lands this module registers **no** Python functions; the gateway keeps using
//! [`crate::prepare`] and [`crate::finish`].

use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast` (none yet).
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
