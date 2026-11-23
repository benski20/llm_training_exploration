//! The interleaved (`L = 16` lanes) NTTs written once over a 16-lane u32 vector type, instantiated for
//! AVX-512 (one `zmm`) and AVX2 (two `ymm`): same butterflies, same order, same integers as
//! [`crate::arith::Prime::inv_l_mul`] / [`crate::arith::Prime::fwd_l`] (tested bit for bit). Branch-free in
//! the data (Montgomery products by `vpmuludq`, reductions by `vpminud`).
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use crate::arith::{Prime, L};
use std::arch::x86_64::*;

/// 16 u32 lanes.
pub trait V16: Copy {
    unsafe fn ld(p: *const u32) -> Self;
    unsafe fn st(p: *mut u32, v: Self);
    unsafe fn splat(x: u32) -> Self;
    unsafe fn add(a: Self, b: Self) -> Self;
    unsafe fn sub(a: Self, b: Self) -> Self;
    unsafe fn min(a: Self, b: Self) -> Self;
    /// [`crate::arith::mont`] per lane, `b` a broadcast 32-bit value.
    unsafe fn mont_bc(a: Self, b: Self, p: Self, pn: Self) -> Self;
}

#[derive(Clone, Copy)]
pub struct Z(pub __m512i);

impl V16 for Z {
    #[inline(always)]
    unsafe fn ld(p: *const u32) -> Self {
        Z(_mm512_loadu_si512(p as *const _))
    }
    #[inline(always)]
    unsafe fn st(p: *mut u32, v: Self) {
        _mm512_storeu_si512(p as *mut _, v.0)
    }
    #[inline(always)]
    unsafe fn splat(x: u32) -> Self {
        Z(_mm512_set1_epi32(x as i32))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Z(_mm512_add_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Z(_mm512_sub_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn min(a: Self, b: Self) -> Self {
        Z(_mm512_min_epu32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn mont_bc(a: Self, b: Self, p: Self, pn: Self) -> Self {
        let t0 = _mm512_mul_epu32(a.0, b.0);
        let t1 = _mm512_mul_epu32(_mm512_srli_epi64(a.0, 32), b.0);
        let s0 = _mm512_add_epi64(t0, _mm512_mul_epu32(_mm512_mul_epu32(t0, pn.0), p.0));
        let s1 = _mm512_add_epi64(t1, _mm512_mul_epu32(_mm512_mul_epu32(t1, pn.0), p.0));
        Z(_mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(s0, 32), s1))
    }
}

#[derive(Clone, Copy)]
pub struct Y(pub __m256i, pub __m256i);

#[inline(always)]
unsafe fn mont8(a: __m256i, b: __m256i, p: __m256i, pn: __m256i) -> __m256i {
    let t0 = _mm256_mul_epu32(a, b);
    let t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), b);
    let s0 = _mm256_add_epi64(t0, _mm256_mul_epu32(_mm256_mul_epu32(t0, pn), p));
    let s1 = _mm256_add_epi64(t1, _mm256_mul_epu32(_mm256_mul_epu32(t1, pn), p));
    _mm256_blend_epi32(_mm256_srli_epi64(s0, 32), s1, 0xAA)
}

impl V16 for Y {
    #[inline(always)]
    unsafe fn ld(p: *const u32) -> Self {
        Y(_mm256_loadu_si256(p as *const _), _mm256_loadu_si256(p.add(8) as *const _))
    }
    #[inline(always)]
    unsafe fn st(p: *mut u32, v: Self) {
        _mm256_storeu_si256(p as *mut _, v.0);
        _mm256_storeu_si256(p.add(8) as *mut _, v.1);
    }
    #[inline(always)]
    unsafe fn splat(x: u32) -> Self {
        let v = _mm256_set1_epi32(x as i32);
        Y(v, v)
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Y(_mm256_add_epi32(a.0, b.0), _mm256_add_epi32(a.1, b.1))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Y(_mm256_sub_epi32(a.0, b.0), _mm256_sub_epi32(a.1, b.1))
    }
    #[inline(always)]
    unsafe fn min(a: Self, b: Self) -> Self {
        Y(_mm256_min_epu32(a.0, b.0), _mm256_min_epu32(a.1, b.1))
    }
    #[inline(always)]
    unsafe fn mont_bc(a: Self, b: Self, p: Self, pn: Self) -> Self {
        Y(mont8(a.0, b.0, p.0, pn.0), mont8(a.1, b.1, p.1, pn.1))
    }
}

#[derive(Clone, Copy)]
struct C<V> {
    p: V,
    p2: V,
    pn: V,
}

#[inline(always)]
unsafe fn red<V: V16>(v: V, q: V) -> V {
    V::min(v, V::sub(v, q))
}

#[inline(always)]
unsafe fn gs<V: V16>(x: *mut u32, y: *mut u32, z: V, c: C<V>) {
    let (u, w) = (V::ld(x), V::ld(y));
    V::st(x, red(V::add(u, w), c.p2));
    V::st(y, V::mont_bc(V::sub(V::add(u, c.p2), w), z, c.p, c.pn));
}

#[inline(always)]
unsafe fn ct<V: V16>(x: *mut u32, y: *mut u32, z: V, c: C<V>) {
    let (u, w) = (V::ld(x), V::ld(y));
    let t = V::mont_bc(w, z, c.p, c.pn);
    V::st(x, red(V::add(u, t), c.p2));
    V::st(y, red(V::sub(V::add(u, c.p2), t), c.p2));
}

const SUB: usize = 256;
const MAXPTS: usize = 32;

/// `a = INTT(h * t)` without `N^-1` (same values as [`Prime::inv_l_mul`]): pointwise product and the stages
/// `len < 256` per 256-coefficient sub-block (in L1), then the stages `len >= 256` column by column through
/// an L1 buffer (one pass over the array instead of one per stage).
#[inline(always)]
pub unsafe fn inv_l_mul<V: V16>(pr: &Prime, a: &mut [u32], h: &[u32], t: &[u32]) {
    let n = pr.n;
    assert!(a.len() >= n * L && h.len() >= n * L && t.len() >= n && n >= 2);
    let c = C { p: V::splat(pr.p), p2: V::splat(2 * pr.p), pn: V::splat(pr.pn) };
    let ap = a.as_mut_ptr();
    let hp = h.as_ptr();
    let sub = SUB.min(n);
    for sb in 0..n / sub {
        let base = sb * sub;
        for k in base..base + sub {
            V::st(ap.add(k * L), V::mont_bc(V::ld(hp.add(k * L)), V::splat(t[k]), c.p, c.pn));
        }
        let mut len = 1usize;
        while len < sub {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..sub / (2 * len) {
                let z = V::splat(pr.izm[nb + b0 + bb]);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    gs(x.add(i * L), x.add((i + len) * L), z, c);
                }
            }
            len *= 2;
        }
    }
    if sub < n {
        let pts = n / sub;
        assert!(pts <= MAXPTS);
        let mut buf = [0u32; MAXPTS * L];
        let bp = buf.as_mut_ptr();
        for c0 in 0..sub {
            for k in 0..pts {
                V::st(bp.add(k * L), V::ld(ap.add((c0 + sub * k) * L)));
            }
            let mut lk = 1usize;
            while lk < pts {
                let nb = pts / (2 * lk);
                for bk in 0..nb {
                    let z = V::splat(pr.izm[nb + bk]);
                    let x = bp.add(2 * bk * lk * L);
                    for i in 0..lk {
                        gs(x.add(i * L), x.add((i + lk) * L), z, c);
                    }
                }
                lk *= 2;
            }
            for k in 0..pts {
                V::st(ap.add((c0 + sub * k) * L), V::ld(bp.add(k * L)));
            }
        }
        for k in 0..MAXPTS {
            V::st(bp.add(k * L), V::splat(0));
        }
    }
}

/// Forward NTT (same values as [`Prime::fwd_l`]): stages `len >= 256` column by column, then the rest per
/// sub-block.
#[inline(always)]
pub unsafe fn fwd_l<V: V16>(pr: &Prime, a: &mut [u32]) {
    let n = pr.n;
    assert!(a.len() >= n * L && n >= 2);
    let c = C { p: V::splat(pr.p), p2: V::splat(2 * pr.p), pn: V::splat(pr.pn) };
    let ap = a.as_mut_ptr();
    let to = SUB.min(n / 2).max(n / MAXPTS);
    let pts = n / to;
    let mut buf = [0u32; MAXPTS * L];
    let bp = buf.as_mut_ptr();
    for c0 in 0..to {
        for k in 0..pts {
            V::st(bp.add(k * L), V::ld(ap.add((c0 + to * k) * L)));
        }
        let mut lk = pts / 2;
        while lk >= 1 {
            let nb = pts / (2 * lk);
            for bk in 0..nb {
                let z = V::splat(pr.zm[nb + bk]);
                let x = bp.add(2 * bk * lk * L);
                for i in 0..lk {
                    ct(x.add(i * L), x.add((i + lk) * L), z, c);
                }
            }
            lk /= 2;
        }
        for k in 0..pts {
            V::st(ap.add((c0 + to * k) * L), V::ld(bp.add(k * L)));
        }
    }
    for k in 0..MAXPTS {
        V::st(bp.add(k * L), V::splat(0));
    }
    for sb in 0..pts {
        let base = sb * to;
        let mut len = to / 2;
        while len >= 1 {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..to / (2 * len) {
                let z = V::splat(pr.zm[nb + b0 + bb]);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    ct(x.add(i * L), x.add((i + len) * L), z, c);
                }
            }
            len /= 2;
        }
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn inv_l_mul_512(pr: &Prime, a: &mut [u32], h: &[u32], t: &[u32]) {
    inv_l_mul::<Z>(pr, a, h, t)
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn fwd_l_512(pr: &Prime, a: &mut [u32]) {
    fwd_l::<Z>(pr, a)
}

#[target_feature(enable = "avx2")]
pub unsafe fn inv_l_mul_256(pr: &Prime, a: &mut [u32], h: &[u32], t: &[u32]) {
    inv_l_mul::<Y>(pr, a, h, t)
}

#[target_feature(enable = "avx2")]
pub unsafe fn fwd_l_256(pr: &Prime, a: &mut [u32]) {
    fwd_l::<Y>(pr, a)
}
