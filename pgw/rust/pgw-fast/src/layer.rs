//! Native deterministic float path of one decoder layer (chat-speed plan WP-B1, phase B; stub created
//! by WP-A0, owned by WP-B1).
//!
//! Will hold the opt-in "native-float" reference: RMSNorm, RoPE, residuals and the SiLU gate in a
//! fixed order without FMA (documented ISA pin). It is a **new** reference (its bits differ from
//! today's torch path), selected only by the gateway model setting `float_ops = "native"`.
//!
//! Until WP-B1 lands this module registers **no** Python functions; the gateway keeps using torch.

use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast` (none yet).
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
