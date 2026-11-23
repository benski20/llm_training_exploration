//! # pgw-pks: kernels of the compact-hint encrypted mask scheme `pks-v3`
//!
//! See `pgw/research/plans/he-compact-hint-pks.md` (scheme, parameters, noise) and the numpy reference
//! `pgw.hemask.pks`. [`kernel`] holds the customer kernels (upload encryption, reply key, decryption)
//! and the provider's key switch (also the customer's one-time compact-hint build); [`arith`] the
//! Montgomery arithmetic and NTTs. The Python module `pgw_pks` (feature `python`) wraps them.

pub mod arith;
pub mod kernel;
#[cfg(target_arch = "x86_64")]
pub mod simd;
#[cfg(target_arch = "x86_64")]
pub mod vntt;
#[cfg(target_arch = "x86_64")]
pub mod simd2;
#[cfg(target_arch = "x86_64")]
pub mod ifma;
#[cfg(target_arch = "x86_64")]
pub mod ifma_pad;
#[cfg(feature = "python")]
mod py;

pub use arith::{Isa, L};
pub use kernel::{ctx, D, NR, NU, PACK, RANK};
