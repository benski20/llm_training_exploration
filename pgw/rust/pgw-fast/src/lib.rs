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
//! | [`bitpack`] | `pgw.hemask.wire.pack_bits` / `unpack_bits` around maskgen requests (fused modulus switch, reply shift) |
//! | [`writer`] | `SegmentWriter.write_linear` / `RawSegmentWriter.write_chunk`: pack, AES-256-GCM, `pwrite` (Unix only) |
//! | [`local`] | `MaskedBackend._local_fast`: exact local linear (quantize, int8 GEMV, range check, rescale) for `local_linears` |
//! | [`noise`] | `OutboundNoiseMonitor.guard_packed`: duplicate-row fingerprints of a packed payload |
//! | [`lmhead`] | `h @ lm_head^T` (float32, several rows per pass over the head, fixed summation order; same bits as `pgw.engine.lm_head.lm_head_reference`) |
//! | [`lane`] | stub (chat-speed WP-A3): native trip lane client; registers nothing yet |
//! | [`lane_service`] | chat-speed WP-A1: Python bindings of the service trip lane (`rust/pgw-lane`): `LaneService`, `LaneDeviceLock` |
//! | [`trip`] | stub (chat-speed WP-A4a): `prepare_into` / `finish_into` for the TripEngine; registers nothing yet |
//! | [`tripnative`] | opt-in `PGW_TRIP_NATIVE`: `TripLease` / `TripTicket`, one native call per trip on each side of the lane call (take rows of a reservation lease, derive `r`, prepare; finish), `r` of the next row derived ahead on a helper thread |
//! | [`layer`] | opt-in pinned float formulation (`float_ops = "native"`): RMSNorm, residual add + RMSNorm, SiLU gate (fixed order, no FMA, no libm) |
//! | [`attn`] | opt-in row-deterministic attention over the KV cache, fused with RoPE and the cache write (`float_ops = "native"`) |
//!
//! Every function is bit-identical to the Python path (tests in `pgw/tests/native/`), takes numpy
//! arrays / bytes without copying, and releases the GIL while it computes.

use pyo3::prelude::*;

pub mod attn;
pub mod bitpack;
pub mod finish;
pub mod hemask;
pub mod lane;
pub mod lane_service;
pub mod layer;
pub mod lmhead;
pub mod local;
pub mod localgpu;
pub mod noise;
pub mod prepare;
pub mod store;
pub mod trip;
pub mod tripnative;
#[cfg(unix)]
pub mod writer;

/// Build/version info (lets Python check it loaded a compatible extension).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `pgw_fast` Python module.
#[pymodule]
fn pgw_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    attn::register(m)?;
    bitpack::register(m)?;
    finish::register(m)?;
    hemask::register(m)?;
    lane::register(m)?;
    lane_service::register(m)?;
    layer::register(m)?;
    lmhead::register(m)?;
    local::register(m)?;
    localgpu::register(m)?;
    noise::register(m)?;
    prepare::register(m)?;
    store::register(m)?;
    trip::register(m)?;
    tripnative::register(m)?;
    #[cfg(unix)]
    writer::register(m)?;
    Ok(())
}
