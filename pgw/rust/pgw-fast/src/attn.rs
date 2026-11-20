//! Row-deterministic attention over the KV cache (chat-speed plan WP-B1, phase B; stub created by
//! WP-A0, owned by WP-B1).
//!
//! Will hold attention whose output row depends only on that row's query and the cache prefix it
//! sees (invariant under chunk size, thread count and the number of rows in the pass), for the
//! opt-in `float_ops = "native"` reference.
//!
//! Until WP-B1 lands this module registers **no** Python functions; the gateway keeps using torch SDPA.

use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast` (none yet).
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
