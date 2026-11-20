//! # pgw-ntt
//!
//! Exact ring products for the customer side of the encrypted (MLWE) mask factory
//! (`pgw.hemask`): `U @ S mod 2^64` where `U` is a big operand (the public matrix `A`, 64-bit words,
//! for encryption; the hint `H = W8 A`, 56-bit byte planes, for decryption) and `S` the
//! block-negacyclic matrix of a small ternary secret (`pgw.hemask.lwe.secret_matrix`). The dense
//! path multiplies by `S` as a `dim x 256` matrix (`dim * 256` int8 MACs per row and byte plane);
//! here each row costs `rank` pointwise products per coefficient and prime plus one inverse NTT.
//!
//! The integer product is computed exactly with NTTs over three primes `~2^29` and CRT (bound
//! proof in [`arith`]), then reduced mod `2^64`, so the result equals the dense path bit for bit.
//!
//! Python module `pgw_ntt` (feature `python`, built by `rust/tools/build_ntt.py`); the Python
//! wrappers are `pgw.hemask.ntt`.

pub mod arith;
#[cfg(target_arch = "x86_64")]
mod avx512;
pub mod kernel;
#[cfg(feature = "python")]
mod py;
