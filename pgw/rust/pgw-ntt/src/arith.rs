//! Primes, Montgomery constants and twiddle tables (computed once at first use).
//!
//! # Why three 29-bit primes are enough (the exactness bound)
//!
//! Every product this crate computes is `Z = sum_l a_l * s_l` in `Z[X]/(X^256 + 1)` with `a_l` the
//! polynomials of one row of a *big* operand (unsigned integers below `2^b`, `b <= 64`) and `s_l` the
//! polynomials of a *small* secret (signed integers, ternary in the protocol). One output coefficient
//! is a signed sum of products `a_{l,t} * (+-s_{l,u})`, one per pair `(l, t)`, so
//!
//! ```text
//! |Z_j| <= (2^b - 1) * L1(s),        L1(s) = sum_l sum_t |s_{l,t}|
//! ```
//!
//! For the protocol (ternary `s`, rank `k`) `L1(s) <= 256 k`; at the default rank 16 that is
//! `2^12`, so `|Z_j| < 2^76` for the 64-bit public matrix (encrypt) and `< 2^68` for the 56-bit hint
//! (decrypt).
//!
//! The NTTs give `Z mod p_i` for three primes `p_0 > p_1 > p_2`, all in `(2^28, 2^29)` and
//! `= 1 mod 512` (so a primitive 512-th root of unity exists and the length-256 negacyclic NTT
//! works). Garner's CRT with a *centred* last digit returns the unique integer congruent to the
//! residues in
//!
//! ```text
//! [-p_0 p_1 (p_2 - 1)/2,  p_0 p_1 (p_2 - 1)/2 + p_0 p_1 - 1]      (P = p_0 p_1 p_2 consecutive integers)
//! ```
//!
//! so `Z` is recovered exactly whenever `(2^b - 1) * L1(s) <= HALF := p_0 p_1 (p_2 - 1) / 2`. The
//! primes are 536870401, 536865281, 536856577, so `HALF = 77368301288923437828400128 ~ 2^85.9999`,
//! i.e. `L1(s) <= 4194144` for a 64-bit big operand: ternary rank up to 16383 (the protocol uses
//! 8-16), a margin of 10 bits over `2^76` at rank 16 (18 bits for the 56-bit hint). [`bound_ok`]
//! checks the exact condition with 128-bit integers on every call, and the tests check it at the
//! boundary (all-max coefficients). The exact `Z` is then reduced mod `2^64` (wrapping arithmetic is
//! a ring homomorphism), which is what the dense path returns.
//!
//! # Lazy Montgomery arithmetic bounds (`R = 2^32`)
//!
//! * `redc(t) = (t + m p) / R` with `m = -t p^-1 mod R` is `t R^-1 mod p`; it needs `t + m p < 2^64`
//!   and returns a value `< t / R + p`. For `t = x * w` with `x < R` and `w < p` that is `< 2p`.
//! * NTT butterflies keep values in `[0, 2p)`; sums are `< 4p < 2^31` and are brought back with one
//!   `min(v, v - 2p)` (wrapping subtraction: the minimum is `v - 2p` exactly when `v >= 2p`).
//! * Pointwise multiply-accumulate: `a < p`, `s < p`, so each product is `< 2^58`; up to
//!   [`MAC_CHUNK`] `= 48` products plus `m p < 2^61` stay below `2^64`, and `redc` of the sum is
//!   `< 48 p / 8 + p = 7 p`; two `min` steps (`4p`, `2p`) give `[0, 2p)`.

use std::sync::OnceLock;

/// Ring degree (`X^256 + 1`).
pub const N: usize = 256;
/// Number of CRT primes.
pub const NP: usize = 3;
/// Products accumulated in 64 bits before one Montgomery reduction (see the module docs).
pub const MAC_CHUNK: usize = 48;

/// One NTT prime with its Montgomery constants and tables (all residues `< p`).
#[derive(Clone, Debug)]
pub struct Prime {
    pub p: u32,
    /// `-p^-1 mod 2^32`.
    pub pn: u32,
    /// `zetas[k] = psi^brv8(k) * R mod p` (forward NTT, `k = 1..255`).
    pub zetas: [u32; N],
    /// `izetas[k] = -psi^brv8(k) * R mod p` (inverse NTT).
    pub izetas: [u32; N],
    /// `R^2 / 256 mod p`: `mont_mul(x, sscale) = x * R / 256`, applied to the secret's NTT so that the
    /// accumulate's `redc` and the unscaled inverse NTT give the exact product.
    pub sscale: u32,
    /// `2^(8d) * R mod p` for byte `d` (residues of byte-plane operands).
    pub c_byte: [u32; 8],
    /// `R mod p` and `2^32 * R mod p` (residues of 64-bit words from their 32-bit halves).
    pub c_lo: u32,
    pub c_hi: u32,
}

/// CRT constants for the three primes.
#[derive(Clone, Debug)]
pub struct Crt {
    /// `p_0^-1 mod p_1`, Montgomery form mod `p_1`.
    pub inv01: u32,
    /// `p_0 mod p_2`, Montgomery form mod `p_2`.
    pub p0_m2: u32,
    /// `(p_0 p_1)^-1 mod p_2`, Montgomery form mod `p_2`.
    pub inv012: u32,
    /// `p_0 p_1` (below `2^58`).
    pub p01: u64,
    /// `HALF = p_0 p_1 (p_2 - 1) / 2`: products with `|Z| <= HALF` are recovered exactly.
    pub half: u128,
}

/// All constants.
#[derive(Clone, Debug)]
pub struct Params {
    pub primes: [Prime; NP],
    pub crt: Crt,
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

pub fn pow_mod(mut b: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    b %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * b as u128 % p as u128) as u64;
        }
        b = (b as u128 * b as u128 % p as u128) as u64;
        e >>= 1;
    }
    r
}

fn inv_mod(a: u64, p: u64) -> u64 {
    pow_mod(a, p - 2, p)
}

fn brv8(k: usize) -> usize {
    (k as u8).reverse_bits() as usize
}

/// The three largest primes below `2^29` that are `1 mod 512`, in decreasing order.
pub fn find_primes() -> [u32; NP] {
    let mut out = [0u32; NP];
    let mut c = (1u64 << 29) - 511; // 2^29 - 511 = 1 mod 512
    let mut i = 0;
    while i < NP {
        if is_prime(c) {
            out[i] = c as u32;
            i += 1;
        }
        c -= 512;
    }
    out
}

fn make_prime(p: u32) -> Prime {
    let pp = p as u64;
    let r = (1u128 << 32) % pp as u128;
    let r = r as u64;
    let to_mont = |x: u64| ((x as u128 * r as u128) % pp as u128) as u64 as u32;
    // -p^-1 mod 2^32 by Newton iteration
    let mut inv: u32 = 1;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(p.wrapping_mul(inv)));
    }
    assert_eq!(p.wrapping_mul(inv), 1);
    let pn = inv.wrapping_neg();
    // primitive 512-th root of unity: x^((p-1)/512) with psi^256 = -1
    let mut psi = 0u64;
    for x in 2..pp {
        let c = pow_mod(x, (pp - 1) / 512, pp);
        if pow_mod(c, 256, pp) == pp - 1 {
            psi = c;
            break;
        }
    }
    assert!(psi != 0);
    let mut zetas = [0u32; N];
    let mut izetas = [0u32; N];
    for k in 0..N {
        let z = pow_mod(psi, brv8(k) as u64, pp);
        zetas[k] = to_mont(z);
        izetas[k] = to_mont((pp - z) % pp);
    }
    let r2 = (r as u128 * r as u128 % pp as u128) as u64;
    let sscale = (r2 as u128 * inv_mod(N as u64, pp) as u128 % pp as u128) as u32;
    let mut c_byte = [0u32; 8];
    for (d, c) in c_byte.iter_mut().enumerate() {
        *c = to_mont(pow_mod(2, 8 * d as u64, pp));
    }
    Prime { p, pn, zetas, izetas, sscale, c_byte, c_lo: to_mont(1), c_hi: to_mont(pow_mod(2, 32, pp)) }
}

/// The constants (computed on first use).
pub fn params() -> &'static Params {
    static P: OnceLock<Params> = OnceLock::new();
    P.get_or_init(|| {
        let ps = find_primes();
        let primes = [make_prime(ps[0]), make_prime(ps[1]), make_prime(ps[2])];
        let (p0, p1, p2) = (ps[0] as u64, ps[1] as u64, ps[2] as u64);
        let m1 = |x: u64| (((x % p1) as u128 * ((1u128 << 32) % p1 as u128)) % p1 as u128) as u32;
        let m2 = |x: u64| (((x % p2) as u128 * ((1u128 << 32) % p2 as u128)) % p2 as u128) as u32;
        let p01 = p0 * p1;
        let crt = Crt {
            inv01: m1(inv_mod(p0 % p1, p1)),
            p0_m2: m2(p0 % p2),
            inv012: m2(inv_mod(p01 % p2, p2)),
            p01,
            half: p01 as u128 * (p2 as u128 - 1) / 2,
        };
        Params { primes, crt }
    })
}

/// `true` when products of a big operand below `2^big_bits` with a secret of L1 norm `l1` are
/// recovered exactly (see the module docs).
pub fn bound_ok(big_bits: u32, l1: u64) -> bool {
    if big_bits == 0 || big_bits > 64 {
        return false;
    }
    let big_max: u128 = (1u128 << big_bits) - 1;
    match big_max.checked_mul(l1 as u128) {
        Some(b) => b <= params().crt.half,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_are_ntt_friendly_and_bound_holds() {
        let ps = find_primes();
        for w in ps.windows(2) {
            assert!(w[0] > w[1]);
        }
        for &p in &ps {
            assert!(is_prime(p as u64));
            assert_eq!(p % 512, 1);
            assert!(p > 1 << 28 && p < 1 << 29);
        }
        let h = params().crt.half;
        // > 2^85.9 (see the module docs)
        assert!(h > (1u128 << 85) + (1u128 << 84) + (1u128 << 83));
        // rank 16 ternary, 64-bit big: L1 <= 4096
        assert!(bound_ok(64, 4096));
        assert!(bound_ok(56, 4096));
        // the exact limit
        let lmax = (h / ((1u128 << 64) - 1)) as u64;
        assert!(bound_ok(64, lmax));
        assert!(!bound_ok(64, lmax + 1));
        assert!(lmax > 14_000 * 256);
    }

    #[test]
    fn roots_are_primitive() {
        for pr in &params().primes {
            let p = pr.p as u64;
            // zetas[128] = psi^brv8(128) = psi^1 (Montgomery form): check psi^256 = -1 via the table
            let r_inv = inv_mod((1u64 << 32) % p, p);
            let psi = (pr.zetas[128] as u64 * r_inv) % p;
            assert_eq!(pow_mod(psi, 256, p), p - 1);
            assert_eq!(pow_mod(psi, 512, p), 1);
        }
    }
}
