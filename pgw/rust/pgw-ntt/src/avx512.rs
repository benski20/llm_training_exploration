//! AVX-512 versions of the hot kernels (same arithmetic as the portable lane loops in
//! [`crate::kernel`], hence the same bits).
//!
//! A `[u32; 16]` lane vector is one `__m512i`. 32 x 32 -> 64-bit products use `vpmuludq` on the
//! even lanes and on the odd lanes shifted down, and the two Montgomery results are blended back
//! into 16 lanes (`mont`): 6 multiplies, no widening or narrowing shuffles. The multiply-accumulate
//! keeps 64-bit even/odd accumulators for `GB` secrets in registers.

use crate::arith::{Prime, MAC_CHUNK, N};
use crate::kernel::{Kern, Lanes, GB, JB};
use core::arch::x86_64::*;

pub(crate) struct Avx512;

impl Kern for Avx512 {
    #[inline(always)]
    fn ntt_fwd(a: &mut [Lanes], pr: &Prime) {
        // SAFETY: only reached from `blocks_avx512` / the profiler, after the CPU check.
        unsafe { ntt_fwd(a, pr) }
    }
    #[inline(always)]
    fn ntt_inv(a: &mut [Lanes], pr: &Prime) {
        // SAFETY: as above.
        unsafe { ntt_inv(a, pr) }
    }
    #[inline(always)]
    fn mac(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, pr: &Prime, out: &mut [Lanes]) {
        // SAFETY: as above.
        unsafe { mac(a, sh, rank, gp, pr, out) }
    }
    #[inline(always)]
    fn load_residues(
        big: &crate::kernel::Big,
        row0: usize,
        dim: usize,
        cs: usize,
        ce: usize,
        out: &mut [Lanes],
        _tmp: &mut [u32],
    ) {
        // SAFETY: as above.
        unsafe { load_residues(big, row0, dim, cs, ce, out) }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn ld(p: *const Lanes) -> __m512i {
    _mm512_loadu_si512(p as *const __m512i)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn st(p: *mut Lanes, v: __m512i) {
    _mm512_storeu_si512(p as *mut __m512i, v)
}

/// `min(x, x - m)`: `x mod m` for `x < 2m`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn red(x: __m512i, m: __m512i) -> __m512i {
    _mm512_min_epu32(x, _mm512_sub_epi32(x, m))
}

/// Montgomery reduction of even/odd 64-bit lanes (`< 2^64` after adding `m p`) into 16 u32 lanes.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn redc2(te: __m512i, to: __m512i, p: __m512i, pn: __m512i) -> __m512i {
    let me = _mm512_mul_epu32(te, pn);
    let mo = _mm512_mul_epu32(to, pn);
    let ue = _mm512_add_epi64(te, _mm512_mul_epu32(me, p));
    let uo = _mm512_add_epi64(to, _mm512_mul_epu32(mo, p));
    _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(ue, 32), uo)
}

/// `x * w * R^-1 mod p` in `[0, 2p)` for `x < 2^32`, `w < p` (`w` broadcast).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mont(x: __m512i, w: __m512i, p: __m512i, pn: __m512i) -> __m512i {
    let te = _mm512_mul_epu32(x, w);
    let to = _mm512_mul_epu32(_mm512_srli_epi64(x, 32), w);
    redc2(te, to, p, pn)
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn ntt_fwd(a: &mut [Lanes], pr: &Prime) {
    assert!(a.len() >= N);
    let a = a.as_mut_ptr();
    let p = _mm512_set1_epi32(pr.p as i32);
    let pn = _mm512_set1_epi32(pr.pn as i32);
    let p2 = _mm512_set1_epi32(2 * pr.p as i32);
    // Harvey's lazy butterflies: values in [0, 4p), one reduction of x per butterfly
    let mut k = 0usize;
    let mut len = N / 2;
    while len > 1 {
        let mut start = 0;
        while start < N {
            k += 1;
            let z = _mm512_set1_epi32(pr.zetas[k] as i32);
            for j in start..start + len {
                let x = red(ld(a.add(j)), p2);
                let t = mont(ld(a.add(j + len)), z, p, pn);
                st(a.add(j + len), _mm512_sub_epi32(_mm512_add_epi32(x, p2), t));
                st(a.add(j), _mm512_add_epi32(x, t));
            }
            start += 2 * len;
        }
        len >>= 1;
    }
    // last stage (len 1) with the final reduction to [0, p)
    for j in (0..N).step_by(2) {
        k += 1;
        let z = _mm512_set1_epi32(pr.zetas[k] as i32);
        let x = red(ld(a.add(j)), p2);
        let t = mont(ld(a.add(j + 1)), z, p, pn);
        st(a.add(j + 1), red(red(_mm512_sub_epi32(_mm512_add_epi32(x, p2), t), p2), p));
        st(a.add(j), red(red(_mm512_add_epi32(x, t), p2), p));
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn ntt_inv(a: &mut [Lanes], pr: &Prime) {
    assert!(a.len() >= N);
    let a = a.as_mut_ptr();
    let p = _mm512_set1_epi32(pr.p as i32);
    let pn = _mm512_set1_epi32(pr.pn as i32);
    let p2 = _mm512_set1_epi32(2 * pr.p as i32);
    let mut k = N;
    let mut len = 1;
    while len < N {
        let mut start = 0;
        while start < N {
            k -= 1;
            let z = _mm512_set1_epi32(pr.izetas[k] as i32);
            for j in start..start + len {
                let x = ld(a.add(j));
                let y = ld(a.add(j + len));
                st(a.add(j), red(_mm512_add_epi32(x, y), p2));
                st(a.add(j + len), mont(_mm512_sub_epi32(_mm512_add_epi32(x, p2), y), z, p, pn));
            }
            start += 2 * len;
        }
        len <<= 1;
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn mac(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, pr: &Prime, out: &mut [Lanes]) {
    assert!(a.len() >= rank * N && out.len() >= gp * N && gp.is_multiple_of(GB) && sh.len() >= N * rank * gp);
    for jb in (0..N).step_by(JB) {
        for g0 in (0..gp).step_by(GB) {
            mac_part(a, sh, rank, gp, g0, jb, pr, out);
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn mac_part(
    a: &[Lanes],
    sh: &[u32],
    rank: usize,
    gp: usize,
    g0: usize,
    jb: usize,
    pr: &Prime,
    out: &mut [Lanes],
) {
    let ap = a.as_ptr();
    let sp = sh.as_ptr();
    let op = out.as_mut_ptr();
    let p = _mm512_set1_epi32(pr.p as i32);
    let pn = _mm512_set1_epi32(pr.pn as i32);
    let p2 = _mm512_set1_epi32(2 * pr.p as i32);
    let p4 = _mm512_set1_epi32(4 * pr.p as i32);
    for j in jb..jb + JB {
        let mut part = [_mm512_setzero_si512(); GB];
        let mut l0 = 0;
        while l0 < rank {
            let l1 = (l0 + MAC_CHUNK).min(rank);
            let mut ae = [_mm512_setzero_si512(); GB];
            let mut ao = [_mm512_setzero_si512(); GB];
            for l in l0..l1 {
                let av = ld(ap.add(l * N + j));
                let avo = _mm512_srli_epi64(av, 32);
                let s = sp.add((j * rank + l) * gp + g0);
                for g in 0..GB {
                    let sv = _mm512_set1_epi32(*s.add(g) as i32);
                    ae[g] = _mm512_add_epi64(ae[g], _mm512_mul_epu32(av, sv));
                    ao[g] = _mm512_add_epi64(ao[g], _mm512_mul_epu32(avo, sv));
                }
            }
            for g in 0..GB {
                let u = red(red(redc2(ae[g], ao[g], p, pn), p4), p2);
                part[g] = if l0 == 0 { u } else { red(_mm512_add_epi32(part[g], u), p2) };
            }
            l0 = l1;
        }
        for (g, v) in part.iter().enumerate() {
            st(op.add((g0 + g) * N + j), *v);
        }
    }
}

/// In-register transpose of a 16 x 16 u32 tile: four rounds of `vpermt2d`, round `k` swapping bit
/// `k` of the row index with bit `k` of the column index.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn transpose16(r: &mut [__m512i; 16]) {
    for k in 0..4 {
        let bit = 1i32 << k;
        let mut ia = [0i32; 16];
        let mut ib = [0i32; 16];
        for c in 0..16i32 {
            if c & bit == 0 {
                ia[c as usize] = c;
                ib[c as usize] = c + bit;
            } else {
                ia[c as usize] = 16 + c - bit;
                ib[c as usize] = 16 + c;
            }
        }
        let ia = _mm512_loadu_si512(ia.as_ptr() as *const __m512i);
        let ib = _mm512_loadu_si512(ib.as_ptr() as *const __m512i);
        for r0 in 0..16usize {
            if r0 & (1 << k) != 0 {
                continue;
            }
            let (a, b) = (r[r0], r[r0 | (1 << k)]);
            r[r0] = _mm512_permutex2var_epi32(a, ia, b);
            r[r0 | (1 << k)] = _mm512_permutex2var_epi32(a, ib, b);
        }
    }
}

/// [`crate::kernel`]'s `load_residues` with AVX-512: per 16 columns, the 32-bit halves of the 16
/// rows' words, then per prime one Montgomery reduction of `lo * (R mod p) + hi * (2^32 R mod p)`
/// (`< 2^33 p`, so the result is `< 3p`, then `[0, 2p)`) and a 16 x 16 transpose into lanes.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub(crate) unsafe fn load_residues(
    big: &crate::kernel::Big,
    row0: usize,
    dim: usize,
    cs: usize,
    ce: usize,
    out: &mut [Lanes],
) {
    use crate::arith::{params, NP};
    use crate::kernel::Big;
    assert!(out.len() >= NP * dim && cs.is_multiple_of(16) && ce.is_multiple_of(16) && cs <= ce && ce <= dim);
    let ps = &params().primes;
    let nrows = big.rows().saturating_sub(row0).min(16);
    let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    let x80 = _mm512_set1_epi32(0x80);
    let zero = _mm512_setzero_si512();
    let op = out.as_mut_ptr();
    for c0 in (cs..ce).step_by(16) {
        let mut lo = [zero; 16];
        let mut hi = [zero; 16];
        for lane in 0..nrows {
            let r = row0 + lane;
            match *big {
                Big::U64 { data, .. } => {
                    assert!(data.len() >= (r + 1) * dim);
                    let p = data.as_ptr().add(r * dim + c0) as *const __m512i;
                    let v0 = _mm512_loadu_si512(p);
                    let v1 = _mm512_loadu_si512(p.add(1));
                    lo[lane] = _mm512_permutex2var_epi32(v0, even, v1);
                    hi[lane] = _mm512_permutex2var_epi32(v0, odd, v1);
                }
                Big::Planes { data, npl, first, .. } => {
                    assert!(data.len() >= (r + 1) * npl * dim);
                    let (mut l, mut h) = (zero, zero);
                    for d in 0..npl {
                        let p = data.as_ptr().add((r * npl + d) * dim + c0) as *const __m128i;
                        let b = _mm512_xor_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128(p)), x80);
                        let pos = first + d;
                        if pos < 4 {
                            l = _mm512_or_si512(l, _mm512_sll_epi32(b, _mm_cvtsi32_si128(8 * pos as i32)));
                        } else {
                            h = _mm512_or_si512(h, _mm512_sll_epi32(b, _mm_cvtsi32_si128(8 * (pos - 4) as i32)));
                        }
                    }
                    lo[lane] = l;
                    hi[lane] = h;
                }
                Big::Pre { .. } => unreachable!(),
            }
        }
        for (pi, pr) in ps.iter().enumerate() {
            let p = _mm512_set1_epi32(pr.p as i32);
            let pn = _mm512_set1_epi32(pr.pn as i32);
            let p2 = _mm512_set1_epi32(2 * pr.p as i32);
            let cl = _mm512_set1_epi32(pr.c_lo as i32);
            let ch = _mm512_set1_epi32(pr.c_hi as i32);
            let mut t = [zero; 16];
            for lane in 0..nrows {
                let (l, h) = (lo[lane], hi[lane]);
                let xe = _mm512_add_epi64(_mm512_mul_epu32(l, cl), _mm512_mul_epu32(h, ch));
                let xo = _mm512_add_epi64(
                    _mm512_mul_epu32(_mm512_srli_epi64(l, 32), cl),
                    _mm512_mul_epu32(_mm512_srli_epi64(h, 32), ch),
                );
                t[lane] = red(redc2(xe, xo, p, pn), p2);
            }
            transpose16(&mut t);
            for (c, v) in t.iter().enumerate() {
                st(op.add(pi * dim + c0 + c), *v);
            }
        }
    }
}
