//! Freivalds check over the prime field `F_p`, `p = 2^31 - 1` (opt-in segment format `"p31"`).
//!
//! The ring check (`crate::finish`, [`crate::trip::freivalds_output`]) compares `u . y_hat` with
//! `(u W) . x_hat` modulo `2^K` on the masked values and needs `c = 7` vectors at K32A12C27 for 21 bits.
//! The prime check compares, on the *unmasked* integers the gateway already has,
//!
//! ```text
//!     u . yq  ==  (u W mod p) . xq   (mod p)        yq = centered(y_hat - W r),  xq = quantized input
//! ```
//!
//! with one secret `u` uniform in `F_p^m`: an accepted wrong row differs from the genuine `W xq` by a
//! non-zero integer vector `e` with `|e_j| <= 2B < p` (range check, cap), so `e mod p != 0` and the
//! check misses it with probability `1/p` (31 bits). Soundness argument: SPEC 5.1 and
//! `research/notes/prime-field-freivalds.md`.
//!
//! # Exact arithmetic
//!
//! Every value is an `i32`-range integer `v` (`|xq| < 2^31`, `|yq| <= 2^31` for `K <= 32`) and every
//! check word `u < 2^32`. The kernels compute `sum u_j v_j mod p` exactly:
//!
//! * `v' = v + 2^31` is in `[0, 2^32)` (one xor of the sign bit), so `u_j v'_j < 2^64` (one
//!   `vpmuludq` lane); `sum u v = sum u v' - 2^31 sum u`, and `2^31 = 1 (mod p)`, so the result is
//!   `(sum u v' - sum u) mod p`.
//! * Each product is folded once, `x -> (x & p) + (x >> 31)` (`2^31 = 1 mod p`, so the fold keeps the
//!   residue), giving `< 2^31 + 2^33`; sums of up to `2^30` folded terms fit in `u64`, and `n, m <= 2^30`
//!   is checked. The final reduction folds twice and subtracts `p` once with a mask.
//! * The loops are plain `u32 x u32 -> u64` products that LLVM vectorizes to `vpmuludq` (AVX2: 4 lanes,
//!   AVX-512: 8 lanes); the AVX-512 and AVX2 builds are selected at run time and give the same integers
//!   (wrapping-free integer arithmetic: identical on every ISA).
//!
//! # Constant time
//!
//! No branch or address depends on activations, masks or check vectors: loops run over the public
//! `rows`, `c`, `m`, `n`; the verdict compares every check vector of every row (no early exit); the
//! final `- p` is a mask select. Scratch buffers holding `yq` are wiped.

use crate::finish::{decode, wipe, Src, BLOCK, STATUS_FREIVALDS, STATUS_OK, STATUS_RANGE, STATUS_U_ZERO};

/// The field modulus `2^31 - 1` of the `"p31"` check format.
pub const P31: u64 = (1 << 31) - 1;
/// Largest vector length the lazy `u64` accumulation supports (each folded term is `< 2^34`).
pub const MAX_LEN: usize = 1 << 30;

#[inline(always)]
fn fold(x: u64) -> u64 {
    (x & P31) + (x >> 31)
}

/// `x mod p` for any `u64` (two folds, then one masked subtraction).
#[inline(always)]
pub fn reduce(x: u64) -> u64 {
    let y = fold(fold(x)); // < 2^31 + 8 < 2p
    let ge = 0u64.wrapping_sub(u64::from(y >= P31));
    y - (P31 & ge)
}

/// `(sum u_j * v_j) mod p` for `u32` words and signed `i32`-range values (exact).
#[inline(always)]
fn dot_i32(u: &[u32], v: &[i32]) -> (u64, u64) {
    let mut acc = 0u64;
    let mut acc_u = 0u64;
    for (a, b) in u.iter().zip(v) {
        let vb = u64::from((*b as u32) ^ 0x8000_0000); // v + 2^31 in [0, 2^32)
        let prod = u64::from(*a) * vb;
        acc += (prod & P31) + (prod >> 31);
        acc_u += u64::from(*a);
    }
    (acc, acc_u)
}

/// Combine the lazy sums of [`dot_i32`] into `sum u v mod p`.
#[inline(always)]
fn finish_dot(acc: u64, acc_u: u64) -> u64 {
    reduce(reduce(acc) + (P31 - reduce(acc_u)))
}

#[inline(always)]
fn dot_body(u: &[u32], v: &[i32]) -> (u64, u64) {
    dot_i32(u, v)
}

/// Horizontal sum of `u64` lanes (wrapping adds of values far below 2^64 / lanes: exact).
#[inline(always)]
fn hsum(l: &[u64]) -> u64 {
    l.iter().fold(0u64, |a, &b| a + b)
}

/// AVX-512F: 8 elements per step. `vpmovzxdq` widens `u` and `v ^ 2^31` to 64-bit lanes, `vpmuludq`
/// multiplies the low 32 bits exactly (`< 2^64`), then the fold and the lazy sums as in [`dot_i32`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(u: &[u32], v: &[i32]) -> (u64, u64) {
    use std::arch::x86_64::*;
    let len = u.len().min(v.len());
    let main = len - len % 16;
    let pm = _mm512_set1_epi64(P31 as i64);
    let sign = _mm512_set1_epi64(0x8000_0000);
    let (mut a0, mut a1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let (mut s0, mut s1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
    let mut i = 0;
    while i < main {
        // SAFETY: i + 16 <= len for both slices (unaligned loads).
        let ua = _mm512_cvtepu32_epi64(_mm256_loadu_si256(u.as_ptr().add(i).cast()));
        let ub = _mm512_cvtepu32_epi64(_mm256_loadu_si256(u.as_ptr().add(i + 8).cast()));
        let va = _mm512_xor_si512(_mm512_cvtepu32_epi64(_mm256_loadu_si256(v.as_ptr().add(i).cast())), sign);
        let vb = _mm512_xor_si512(_mm512_cvtepu32_epi64(_mm256_loadu_si256(v.as_ptr().add(i + 8).cast())), sign);
        let pa = _mm512_mul_epu32(ua, va);
        let pb = _mm512_mul_epu32(ub, vb);
        a0 = _mm512_add_epi64(a0, _mm512_add_epi64(_mm512_and_si512(pa, pm), _mm512_srli_epi64(pa, 31)));
        a1 = _mm512_add_epi64(a1, _mm512_add_epi64(_mm512_and_si512(pb, pm), _mm512_srli_epi64(pb, 31)));
        s0 = _mm512_add_epi64(s0, ua);
        s1 = _mm512_add_epi64(s1, ub);
        i += 16;
    }
    let mut la = [0u64; 8];
    let mut ls = [0u64; 8];
    _mm512_storeu_si512(la.as_mut_ptr().cast(), _mm512_add_epi64(a0, a1));
    _mm512_storeu_si512(ls.as_mut_ptr().cast(), _mm512_add_epi64(s0, s1));
    let (ta, ts) = dot_i32(&u[main..len], &v[main..len]);
    let out = (hsum(&la) + ta, hsum(&ls) + ts);
    // the lanes held products of secret-derived values: clear the stack copies
    crate::finish::wipe(&mut la);
    crate::finish::wipe(&mut ls);
    out
}

/// AVX2: 4 elements per `vpmuludq` (two accumulators), as [`dot_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(u: &[u32], v: &[i32]) -> (u64, u64) {
    use std::arch::x86_64::*;
    let len = u.len().min(v.len());
    let main = len - len % 8;
    let pm = _mm256_set1_epi64x(P31 as i64);
    let sign = _mm256_set1_epi64x(0x8000_0000);
    let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let (mut s0, mut s1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i < main {
        // SAFETY: i + 8 <= len for both slices (unaligned loads).
        let ua = _mm256_cvtepu32_epi64(_mm_loadu_si128(u.as_ptr().add(i).cast()));
        let ub = _mm256_cvtepu32_epi64(_mm_loadu_si128(u.as_ptr().add(i + 4).cast()));
        let va = _mm256_xor_si256(_mm256_cvtepu32_epi64(_mm_loadu_si128(v.as_ptr().add(i).cast())), sign);
        let vb = _mm256_xor_si256(_mm256_cvtepu32_epi64(_mm_loadu_si128(v.as_ptr().add(i + 4).cast())), sign);
        let pa = _mm256_mul_epu32(ua, va);
        let pb = _mm256_mul_epu32(ub, vb);
        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(_mm256_and_si256(pa, pm), _mm256_srli_epi64(pa, 31)));
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(_mm256_and_si256(pb, pm), _mm256_srli_epi64(pb, 31)));
        s0 = _mm256_add_epi64(s0, ua);
        s1 = _mm256_add_epi64(s1, ub);
        i += 8;
    }
    let mut la = [0u64; 4];
    let mut ls = [0u64; 4];
    _mm256_storeu_si256(la.as_mut_ptr().cast(), _mm256_add_epi64(a0, a1));
    _mm256_storeu_si256(ls.as_mut_ptr().cast(), _mm256_add_epi64(s0, s1));
    let (ta, ts) = dot_i32(&u[main..len], &v[main..len]);
    let out = (hsum(&la) + ta, hsum(&ls) + ts);
    crate::finish::wipe(&mut la);
    crate::finish::wipe(&mut ls);
    out
}

/// Which build of the dot product runs (decided once).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    /// AVX-512F (8 `u64` lanes).
    Avx512,
    /// AVX2 (4 lanes).
    Avx2,
    /// Portable.
    Scalar,
}

/// The ISA the kernels use on this CPU (`PGW_PRIME_ISA=avx2|scalar` forces a narrower one, for tests).
pub fn isa() -> Isa {
    static ISA: std::sync::OnceLock<Isa> = std::sync::OnceLock::new();
    *ISA.get_or_init(|| {
        let force = std::env::var("PGW_PRIME_ISA").unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        {
            if force != "avx2" && force != "scalar" && std::is_x86_feature_detected!("avx512f") {
                return Isa::Avx512;
            }
            if force != "scalar" && std::is_x86_feature_detected!("avx2") {
                return Isa::Avx2;
            }
        }
        let _ = force;
        Isa::Scalar
    })
}

/// Lazy sums of `u . v` (see [`finish_dot`]); `u.len() == v.len()`.
#[inline]
pub fn dot_lazy(u: &[u32], v: &[i32]) -> (u64, u64) {
    debug_assert_eq!(u.len(), v.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX-512F was detected at run time.
        Isa::Avx512 => unsafe { dot_avx512(u, v) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected at run time.
        Isa::Avx2 => unsafe { dot_avx2(u, v) },
        _ => dot_body(u, v),
    }
}

/// `sum u_j v_j mod p` (exact) for `u32` words and `i32` values.
pub fn dot_mod_p(u: &[u32], v: &[i32]) -> u64 {
    assert!(u.len() == v.len() && u.len() <= MAX_LEN, "bad lengths");
    let mut acc = 0u64;
    let mut acc_u = 0u64;
    // blocks keep each lazy sum below 2^64 whatever the length (<= MAX_LEN terms of < 2^34 each)
    for (ub, vb) in u.chunks(1 << 20).zip(v.chunks(1 << 20)) {
        let (a, au) = dot_lazy(ub, vb);
        acc = reduce(acc) + reduce(a);
        acc_u = reduce(acc_u) + reduce(au);
    }
    finish_dot(acc, acc_u)
}

/// The prime input half of one row: `rhs[i] = uw_i . xq mod p` for `uw (c, n)` (row-major `u32`)
/// and the row's quantized input `xq (n,)` (`|xq| < 2^31`). `rhs.len() == c`.
pub fn input_half_row(uw: &[u32], xq: &[i32], rhs: &mut [u64]) {
    let n = xq.len();
    assert!(uw.len() == rhs.len() * n, "uw must be (c, n)");
    for (i, o) in rhs.iter_mut().enumerate() {
        *o = dot_mod_p(&uw[i * n..(i + 1) * n], xq);
    }
}

/// True if every word of `u` is zero (reads every word: no early exit).
fn all_zero(u: &[u32]) -> bool {
    u.iter().fold(0u32, |a, &v| a | v) == 0
}

/// `W r` words `[a, b)` of a `(rows, m)` buffer as `u32` (the low 32 bits: `K <= 32`).
#[inline(always)]
fn wr_fill(wr: Src<'_>, a: usize, b: usize, dst: &mut [u32]) {
    match wr {
        Src::U32(s) => dst.copy_from_slice(&s[a..b]),
        Src::U64(s) => {
            for (d, v) in dst.iter_mut().zip(&s[a..b]) {
                *d = *v as u32;
            }
        }
        Src::Bytes(_) => unreachable!("validated: W r is a word array"),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn output_half_k<const NB: usize>(
    y_hat: &[u8],
    wr: Src<'_>,
    u: &[u32],
    rhs: &[u64],
    c: usize,
    m: usize,
    k: u32,
    bound: u64,
) -> (u8, usize) {
    let rows = rhs.len() / c;
    let shift = 32 - k;
    let mut ybuf = vec![0u32; BLOCK];
    let mut wbuf = vec![0u32; BLOCK];
    let mut qbuf = vec![0i32; BLOCK];
    let mut acc = vec![0u64; rows * c];
    let mut acc_u = vec![0u64; rows * c];
    let bound = bound.min(i64::MAX as u64) as i64;
    let mut out_of_range = 0u64;
    let mut b0 = 0;
    while b0 < m {
        let b1 = (b0 + BLOCK).min(m);
        let len = b1 - b0;
        for t in 0..rows {
            let yb = &mut ybuf[..len];
            decode::<u32, NB>(&y_hat[(t * m + b0) * NB..(t * m + b1) * NB], yb);
            let wb = &mut wbuf[..len];
            wr_fill(wr, t * m + b0, t * m + b1, wb);
            let qb = &mut qbuf[..len];
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for j in 0..len {
                // centered K-bit difference: exact in i32 (|yq| <= 2^(K-1) <= 2^31)
                let yq = ((yb[j].wrapping_sub(wb[j]) << shift) as i32) >> shift;
                lo = lo.min(yq);
                hi = hi.max(yq);
                qb[j] = yq;
            }
            out_of_range |= u64::from(i64::from(hi) > bound) | u64::from(i64::from(lo) < -bound);
            for i in 0..c {
                let (a, au) = dot_lazy(&u[i * m + b0..i * m + b1], qb);
                let x = t * c + i;
                acc[x] = reduce(acc[x]) + reduce(a);
                acc_u[x] = reduce(acc_u[x]) + reduce(au);
            }
        }
        b0 = b1;
    }
    let mut bad = 0usize;
    for t in 0..rows {
        let row_bad = (0..c).fold(false, |a, i| {
            let x = t * c + i;
            a | (finish_dot(acc[x], acc_u[x]) != reduce(rhs[x]))
        });
        bad += usize::from(row_bad);
    }
    wipe(&mut ybuf);
    wipe(&mut wbuf);
    wipe(&mut qbuf);
    wipe(&mut acc);
    wipe(&mut acc_u);
    if out_of_range != 0 {
        return (STATUS_RANGE, 0);
    }
    if bad > 0 {
        (STATUS_FREIVALDS, bad)
    } else {
        (STATUS_OK, 0)
    }
}

/// The prime output half of a chunk: for every row `t`, `yq_t = centered(y_hat_t - W r_t)` (K bits),
/// the range check `|yq_t| <= bound`, and `u_i . yq_t == rhs[t, i] (mod p)` for every check vector.
///
/// `y_hat`: wire bytes `rows * m * K/8`; `wr`: `(rows, m)` words (read only, not wiped); `u`: `(c, m)`
/// `u32` words; `rhs`: `(rows, c)` from the prime input half. `K` in `{8, 16, 24, 32}`.
///
/// Returns `(status, bad_rows)`: [`STATUS_U_ZERO`] if `u` is all zero (verifies nothing),
/// [`STATUS_RANGE`] if any `|yq| > bound` (no check verdict then), [`STATUS_FREIVALDS`] with the number of
/// failing rows, else [`STATUS_OK`]. Panics on inconsistent shapes (validate first).
#[allow(clippy::too_many_arguments)]
pub fn output_half(
    y_hat: &[u8],
    wr: Src<'_>,
    u: &[u32],
    rhs: &[u64],
    c: usize,
    m: usize,
    k: u32,
    bound: u64,
) -> (u8, usize) {
    assert!(c >= 1 && m <= MAX_LEN && u.len() == c * m && rhs.len() % c == 0, "bad check shapes");
    let rows = rhs.len() / c;
    let nb = (k / 8) as usize;
    assert!(y_hat.len() == rows * m * nb && wr.elems(nb) == rows * m, "bad chunk shapes");
    assert!(!matches!(wr, Src::Bytes(_)), "W r must be words");
    if all_zero(u) {
        return (STATUS_U_ZERO, 0);
    }
    #[cfg(target_arch = "x86_64")]
    if isa() != Isa::Scalar {
        // SAFETY: AVX2 is available whenever isa() is not Scalar (AVX-512F CPUs have AVX2).
        return unsafe { output_half_avx2(y_hat, wr, u, rhs, c, m, k, bound) };
    }
    output_half_any(y_hat, wr, u, rhs, c, m, k, bound)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn output_half_any(
    y_hat: &[u8],
    wr: Src<'_>,
    u: &[u32],
    rhs: &[u64],
    c: usize,
    m: usize,
    k: u32,
    bound: u64,
) -> (u8, usize) {
    match k {
        8 => output_half_k::<1>(y_hat, wr, u, rhs, c, m, k, bound),
        16 => output_half_k::<2>(y_hat, wr, u, rhs, c, m, k, bound),
        24 => output_half_k::<3>(y_hat, wr, u, rhs, c, m, k, bound),
        32 => output_half_k::<4>(y_hat, wr, u, rhs, c, m, k, bound),
        k => panic!("the prime check needs K in {{8, 16, 24, 32}}, got {k}"),
    }
}

/// [`output_half_any`] compiled for AVX2 (the decode / centering loop; the dot dispatches on its own).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn output_half_avx2(
    y_hat: &[u8],
    wr: Src<'_>,
    u: &[u32],
    rhs: &[u64],
    c: usize,
    m: usize,
    k: u32,
    bound: u64,
) -> (u8, usize) {
    output_half_any(y_hat, wr, u, rhs, c, m, k, bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_dot(u: &[u32], v: &[i32]) -> u64 {
        let p = P31 as i128;
        let s: i128 = u.iter().zip(v).map(|(a, b)| i128::from(*a) * i128::from(*b)).sum();
        (((s % p) + p) % p) as u64
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn dot_exact_extremes() {
        let mut s = 7u64;
        for len in [0usize, 1, 3, 17, 1000, 4099] {
            let u: Vec<u32> = (0..len).map(|_| lcg(&mut s) as u32).collect();
            for kind in 0..4 {
                let v: Vec<i32> = (0..len)
                    .map(|_| match kind {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        2 => (lcg(&mut s) as i32) >> 4,
                        _ => lcg(&mut s) as i32,
                    })
                    .collect();
                assert_eq!(dot_mod_p(&u, &v), ref_dot(&u, &v), "len {len} kind {kind}");
                let um: Vec<u32> = u.iter().map(|_| u32::MAX).collect();
                assert_eq!(dot_mod_p(&um, &v), ref_dot(&um, &v));
            }
        }
    }

    #[test]
    fn reduce_all_ranges() {
        for x in [0u64, 1, P31 - 1, P31, P31 + 1, 2 * P31, u64::MAX, u64::MAX - 1, 1 << 62, (1 << 33) + 5] {
            assert_eq!(reduce(x), x % P31, "{x}");
        }
    }

    #[test]
    fn output_half_catches_single_entry() {
        let (m, c, rows) = (300usize, 1usize, 2usize);
        let mut s = 3u64;
        let u: Vec<u32> = (0..m).map(|_| (lcg(&mut s) % P31) as u32).collect();
        let yq: Vec<i32> = (0..rows * m).map(|_| ((lcg(&mut s) % (1 << 20)) as i32) - (1 << 19)).collect();
        let wr: Vec<u32> = (0..rows * m).map(|_| lcg(&mut s) as u32).collect();
        let mut y = vec![0u8; rows * m * 4];
        for i in 0..rows * m {
            y[i * 4..i * 4 + 4].copy_from_slice(&(yq[i] as u32).wrapping_add(wr[i]).to_le_bytes());
        }
        let rhs: Vec<u64> = (0..rows).map(|t| dot_mod_p(&u, &yq[t * m..(t + 1) * m])).collect();
        assert_eq!(output_half(&y, Src::U32(&wr), &u, &rhs, c, m, 32, 1 << 27), (STATUS_OK, 0));
        let mut bad = y.clone();
        bad[4 * 7] ^= 1;
        assert_eq!(output_half(&bad, Src::U32(&wr), &u, &rhs, c, m, 32, 1 << 27), (STATUS_FREIVALDS, 1));
        assert_eq!(output_half(&y, Src::U32(&wr), &u, &rhs, c, m, 32, 1 << 10).0, STATUS_RANGE);
        let z = vec![0u32; m];
        assert_eq!(output_half(&y, Src::U32(&wr), &z, &rhs, c, m, 32, 1 << 27).0, STATUS_U_ZERO);
    }
}
