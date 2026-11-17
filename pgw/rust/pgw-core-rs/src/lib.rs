//! # pgw-core-rs
//!
//! Auditable Rust reimplementation of the security-critical core of PGW (Private Gateway),
//! bit-compatible with the Python package `pgw.core` (SPEC.md sections 5, 7.3 and 7.5):
//!
//! | module | contents | Python counterpart |
//! |---|---|---|
//! | [`ring`] | `Z_{2^K}` parameters (K32A8/K40A12/K48A20), centering, balanced digits | `pgw.core.ring` |
//! | [`quant`] | per-row symmetric quantization, rescaling | `pgw.core.ring` |
//! | [`packing`] | `K/8`-byte little-endian ring encoding | `pgw.core.packing` |
//! | [`ringmm`] | exact `x @ W8^T mod 2^K`: reference and int8-limb/int32 algorithm | `pgw.core.ringmm` |
//! | [`masking`] | `x_hat = xq + r`, `yq = center(y_hat - W r)`, range check | `pgw.core.masking` |
//! | [`freivalds`] | check-vector precompute, per-row verification, soundness bound | `pgw.core.freivalds` |
//! | [`prng`] | HKDF-SHA256 + ChaCha20 seed expansion, seekable mask rows | `pgw.core.prng` |
//! | [`reservation`] | single-use slot ledger, rollback checks, per-linear row cursors | `pgw.maskstore` |
//!
//! Cross-language test vectors live in `pgw/tests/vectors/` and are checked by both
//! `cargo test` (`tests/vectors.rs`) and `pytest tests/vectors`.
//!
//! Matrices are row-major slices with explicit dimensions; canonical ring values are `u64`.
//! Nothing in this crate logs or formats secret material.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod freivalds;
pub mod masking;
pub mod packing;
pub mod prng;
pub mod quant;
pub mod reservation;
pub mod ring;
pub mod ringmm;

#[cfg(feature = "python")]
mod python;

pub use error::{ErrorCode, PgwError, Result};
pub use freivalds::{freivalds_check, freivalds_precompute, freivalds_verify, soundness_bits};
pub use masking::{check_output_range, mask_activations, unmask_outputs};
pub use packing::{pack_ring, packed_size, unpack_ring};
pub use prng::{encode_labels, Label, SeedExpander, SEED_BYTES};
pub use quant::{dequantize_outputs, quantize_activations, quantize_weight_rows};
pub use reservation::{
    Ledger, Reservation, ReservationCursor, SegmentState, SegmentStatus, SlotExtent, TakePlan, WitnessStatus,
};
pub use ring::{balanced_digits, int8_limbs, recompose_digits, RingParams};
pub use ringmm::{ringmm_limbs, ringmm_reference};

/// Protocol version implemented by this crate.
pub const PROTOCOL_VERSION: u32 = 1;
