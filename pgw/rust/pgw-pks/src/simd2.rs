//! Hand-written AVX2 copies (x86-64, used when the ISA is `avx2`): an 8-way ChaCha20 keystream (same bytes
//! as the `chacha20` crate, tested). The NTTs shared with AVX-512 are in [`crate::vntt`].
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::needless_range_loop)]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn rotl<const N: i32, const M: i32>(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_slli_epi32::<N>(x), _mm256_srli_epi32::<M>(x))
}

/// In-register transpose of an 8 x 8 u32 block.
#[inline(always)]
unsafe fn transpose8x8(r: &mut [__m256i; 8]) {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256::<0x20>(u0, u4);
    r[1] = _mm256_permute2x128_si256::<0x20>(u1, u5);
    r[2] = _mm256_permute2x128_si256::<0x20>(u2, u6);
    r[3] = _mm256_permute2x128_si256::<0x20>(u3, u7);
    r[4] = _mm256_permute2x128_si256::<0x31>(u0, u4);
    r[5] = _mm256_permute2x128_si256::<0x31>(u1, u5);
    r[6] = _mm256_permute2x128_si256::<0x31>(u2, u6);
    r[7] = _mm256_permute2x128_si256::<0x31>(u3, u7);
}

#[inline(always)]
unsafe fn red8(v: __m256i, q: __m256i) -> __m256i {
    _mm256_min_epu32(v, _mm256_sub_epi32(v, q))
}

#[inline(always)]
unsafe fn mont8(a: __m256i, b: __m256i, p: __m256i, pn: __m256i) -> __m256i {
    let t0 = _mm256_mul_epu32(a, b);
    let t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), b);
    let s0 = _mm256_add_epi64(t0, _mm256_mul_epu32(_mm256_mul_epu32(t0, pn), p));
    let s1 = _mm256_add_epi64(t1, _mm256_mul_epu32(_mm256_mul_epu32(t1, pn), p));
    _mm256_blend_epi32(_mm256_srli_epi64(s0, 32), s1, 0xAA)
}

/// Exact `u64 -> f64` of 4 lanes below 2^32 (`x | 2^52` as a double, minus 2^52).
#[inline(always)]
unsafe fn u32x4_to_f64(x: __m256i) -> __m256d {
    let magic = _mm256_set1_epi64x(0x4330_0000_0000_0000);
    _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(x, magic)), _mm256_set1_pd(4503599627370496.0))
}

/// The AVX2 copy of [`crate::simd::dec_combine`]: same integers, and the same IEEE-754 operations in the same
/// order (`u64 -> f64` conversions are exact or, for reply words above 2^53, one correctly rounded addition
/// as in `as f64`; the final truncation by `vroundpd` toward zero).
#[target_feature(enable = "avx2")]
pub unsafe fn dec_combine(k: &crate::simd::DecConsts, s0: *const u32, s1: *const u32, lanes: usize,
                          c0: crate::simd::C0p, idx0: &[usize; 16], outp: &[*mut u32; 16]) {
    let (p0v, p1v) = (_mm256_set1_epi32(k.p0 as i32), _mm256_set1_epi32(k.p1 as i32));
    let (p02, p12) = (_mm256_set1_epi32((2 * k.p0) as i32), _mm256_set1_epi32((2 * k.p1) as i32));
    let pn1 = _mm256_set1_epi32(k.pn1 as i32);
    let inv = _mm256_set1_epi32(k.inv01_m as i32);
    // [lane half][j block][j in block]
    let mut r0 = [[[_mm256_setzero_si256(); 8]; 4]; 2];
    let mut kk = [[[_mm256_setzero_si256(); 8]; 4]; 2];
    for j in 0..32 {
        for h in 0..2 {
            let x0 = _mm256_loadu_si256(s0.add(16 * j + 8 * h) as *const _);
            let x1 = _mm256_loadu_si256(s1.add(16 * j + 8 * h) as *const _);
            let a = red8(red8(x0, p02), p0v);
            let b = red8(red8(x1, p12), p1v);
            let am = red8(a, p1v);
            let kv = red8(mont8(_mm256_sub_epi32(_mm256_add_epi32(b, p1v), am), inv, p1v, pn1), p1v);
            r0[h][j / 8][j % 8] = a;
            kk[h][j / 8][j % 8] = kv;
        }
    }
    for h in 0..2 {
        for jb in 0..4 {
            transpose8x8(&mut r0[h][jb]); // r0[h][jb][i] = lane 8 h + i, rows 8 jb .. 8 jb + 7
            transpose8x8(&mut kk[h][jb]);
        }
    }
    let p0f = _mm256_set1_pd(k.p0 as f64);
    let wsc = _mm256_set1_pd(k.wscale);
    let csc = _mm256_set1_pd(k.c0scale);
    let half = _mm256_set1_pd(0.5);
    let two32 = _mm256_set1_pd(4294967296.0);
    let magic = _mm256_set1_pd(4503599627370496.0);
    let m32 = _mm256_set1_epi64x(0xFFFF_FFFF);
    let pick = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let nsh = _mm_set_epi32(12, 8, 4, 0);
    for l in 0..lanes {
        let (h, i) = (l / 8, l % 8);
        for jb in 0..4 {
            let (rv, kv) = (r0[h][jb][i], kk[h][jb][i]);
            let mut res = [_mm_setzero_si128(); 2];
            for (q, rq) in res.iter_mut().enumerate() {
                let (r4, k4) = if q == 0 {
                    (_mm256_castsi256_si128(rv), _mm256_castsi256_si128(kv))
                } else {
                    (_mm256_extracti128_si256(rv, 1), _mm256_extracti128_si256(kv, 1))
                };
                let wf = _mm256_mul_pd(_mm256_add_pd(_mm256_cvtepi32_pd(r4), _mm256_mul_pd(p0f, _mm256_cvtepi32_pd(k4))), wsc);
                let idx = idx0[l] + 8 * jb + 4 * q;
                let vf = match c0 {
                    crate::simd::C0p::Words(w) => {
                        let v = _mm256_loadu_si256(w.add(idx) as *const _);
                        let hi = u32x4_to_f64(_mm256_srli_epi64(v, 32));
                        let lo = u32x4_to_f64(_mm256_and_si256(v, m32));
                        _mm256_add_pd(_mm256_mul_pd(hi, two32), lo)
                    }
                    crate::simd::C0p::Packed36 { lo, hi } => {
                        let l4 = _mm256_cvtepu32_epi64(_mm_loadu_si128(lo.add(4 * idx) as *const _));
                        let wd = (hi.add(idx / 2) as *const u16).read_unaligned() as i32;
                        let nib = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(wd), nsh), _mm_set1_epi32(15));
                        _mm256_add_pd(_mm256_mul_pd(_mm256_cvtepi32_pd(nib), two32), u32x4_to_f64(l4))
                    }
                };
                let x = _mm256_add_pd(_mm256_mul_pd(vf, csc), wf);
                let t = _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_add_pd(x, half));
                let bits = _mm256_castpd_si256(_mm256_add_pd(t, magic));
                *rq = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(bits, pick));
            }
            _mm256_storeu_si256(outp[l].add(8 * jb) as *mut _, _mm256_set_m128i(res[1], res[0]));
        }
    }
    r0 = [[[_mm256_setzero_si256(); 8]; 4]; 2];
    kk = [[[_mm256_setzero_si256(); 8]; 4]; 2];
    std::hint::black_box((&r0, &kk));
}

/// The AVX2 copy of [`crate::simd::enc_out`] (same integers: canonical CRT digits, the same centred lift).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn enc_out(k: &crate::simd::EncConsts, s: *const u32, nu: usize, lanes: usize, c0: usize, cols: usize,
                      n: usize, r: *const i64, rows_valid: usize, noise: *const i32, b: *mut u64) {
    let ev: [__m256i; 3] = std::array::from_fn(|i| _mm256_set1_epi32(k.e[i] as i32));
    let e2x: [__m256i; 3] = std::array::from_fn(|i| _mm256_set1_epi32((2 * k.e[i]) as i32));
    let (pn1, pn2) = (_mm256_set1_epi32(k.pn[1] as i32), _mm256_set1_epi32(k.pn[2] as i32));
    let inv01 = _mm256_set1_epi32(k.inv01_m as i32);
    let e0m2 = _mm256_set1_epi32(k.e0_m2 as i32);
    let inv012 = _mm256_set1_epi32(k.inv012_m as i32);
    let e0q = _mm256_set1_epi64x(k.e[0] as i64);
    let (pl, ph) = (_mm256_set1_epi64x((k.p01 & 0xFFFF_FFFF) as i64), _mm256_set1_epi64x((k.p01 >> 32) as i64));
    let emod = _mm256_set1_epi64x(k.emod as i64);
    let half2 = _mm256_set1_epi64x((k.e[2] / 2) as i64);
    let m32 = _mm256_set1_epi64x(0xFFFF_FFFF);
    let m56 = _mm256_set1_epi64x(((1u64 << 56) - 1) as i64);
    let gidx = _mm256_set_epi64x(3 * n as i64, 2 * n as i64, n as i64, 0);
    let quad = |v: __m256i, q: usize| -> __m256i {
        _mm256_cvtepu32_epi64(if q == 0 { _mm256_castsi256_si128(v) } else { _mm256_extracti128_si256(v, 1) })
    };
    // xs[lane quad][u in tile] (4 u64 lanes each)
    let mut xs = [[_mm256_setzero_si256(); 4]; 4];
    for u0 in (0..256usize).step_by(4) {
        let valid = rows_valid.saturating_sub(u0).min(4) as i64;
        let gmask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(valid), _mm256_set_epi64x(3, 2, 1, 0));
        for kk in 0..16 {
            for uu in 0..4 {
                let c = (16 * (u0 + uu) + kk) * 16;
                for h in 0..2 {
                    let o = c + 8 * h;
                    let ld = |p: usize| _mm256_loadu_si256(s.add(p * nu * 16 + o) as *const _);
                    let r0 = red8(red8(ld(0), e2x[0]), ev[0]);
                    let r1 = red8(red8(ld(1), e2x[1]), ev[1]);
                    let r2 = red8(ld(2), ev[2]);
                    let k1 = red8(mont8(_mm256_sub_epi32(_mm256_add_epi32(r1, ev[1]), red8(r0, ev[1])), inv01, ev[1], pn1),
                                  ev[1]);
                    let a = red8(mont8(k1, e0m2, ev[2], pn2), ev[2]);
                    let z = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_add_epi32(r2, e2x[2]), red8(r0, ev[2])), a);
                    let k2 = red8(mont8(z, inv012, ev[2], pn2), ev[2]);
                    for q in 0..2 {
                        let x01 = _mm256_add_epi64(quad(r0, q), _mm256_mul_epu32(quad(k1, q), e0q));
                        let kw = quad(k2, q);
                        let pk = _mm256_add_epi64(_mm256_mul_epu32(kw, pl), _mm256_slli_epi64(_mm256_mul_epu32(kw, ph), 32));
                        let x = _mm256_add_epi64(x01, pk);
                        let neg = _mm256_cmpgt_epi64(kw, half2);
                        xs[2 * h + q][uu] = _mm256_sub_epi64(x, _mm256_and_si256(emod, neg));
                    }
                }
            }
            for qd in xs.iter_mut() {
                let t0 = _mm256_unpacklo_epi64(qd[0], qd[1]);
                let t1 = _mm256_unpackhi_epi64(qd[0], qd[1]);
                let t2 = _mm256_unpacklo_epi64(qd[2], qd[3]);
                let t3 = _mm256_unpackhi_epi64(qd[2], qd[3]);
                qd[0] = _mm256_permute2x128_si256::<0x20>(t0, t2);
                qd[1] = _mm256_permute2x128_si256::<0x20>(t1, t3);
                qd[2] = _mm256_permute2x128_si256::<0x31>(t0, t2);
                qd[3] = _mm256_permute2x128_si256::<0x31>(t1, t3);
            }
            for l in 0..lanes {
                let f = (c0 + l) * 16 + kk;
                let x = xs[l / 4][l % 4];
                let e = _mm256_cvtepi32_epi64(_mm_loadu_si128(noise.add((l * 16 + kk) * 256 + u0) as *const _));
                let msg = if valid <= 0 {
                    _mm256_setzero_si256()
                } else {
                    _mm256_mask_i64gather_epi64::<8>(_mm256_setzero_si256(), r.add(u0 * n + f), gidx, gmask)
                };
                let msg = _mm256_slli_epi64(_mm256_and_si256(msg, m32), 24);
                let v = _mm256_and_si256(_mm256_add_epi64(_mm256_add_epi64(x, e), msg), m56);
                _mm256_storeu_si256(b.wrapping_add(f * cols + u0) as *mut _, v);
            }
        }
    }
    xs = [[_mm256_setzero_si256(); 4]; 4];
    std::hint::black_box(&xs);
}

/// The AVX2 copy of [`crate::simd::load_hint`]: 16 polynomials of `n` 24-bit residues (`src[l]`) into
/// `h[c 16 + l]` (lanes `>= lanes` zero).
#[target_feature(enable = "avx2")]
pub unsafe fn load_hint(src: &[*const u8; 16], lanes: usize, n: usize, h: *mut u32) {
    let mask6 = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1); // 24 bytes = 6 dwords, no read past them
    let perm = _mm256_set_epi32(0, 5, 4, 3, 0, 2, 1, 0);
    let shuf = _mm256_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4,
                               3, -1, 2, 1, 0);
    for c in (0..n).step_by(8) {
        let mut v = [[_mm256_setzero_si256(); 8]; 2];
        for l in 0..lanes {
            let raw = _mm256_maskload_epi32(src[l].add(3 * c) as *const i32, mask6);
            v[l / 8][l % 8] = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(raw, perm), shuf);
        }
        transpose8x8(&mut v[0]); // v[0][j] = coefficient c + j, lanes 0..8
        transpose8x8(&mut v[1]);
        for j in 0..8 {
            _mm256_storeu_si256(h.add((c + j) * 16) as *mut _, v[0][j]);
            _mm256_storeu_si256(h.add((c + j) * 16 + 8) as *mut _, v[1][j]);
        }
    }
}

/// ChaCha20 keystream (RFC 8439, zero nonce, 32-bit block counter from `first`) into `out` (a multiple of
/// 64 bytes), 8 blocks per iteration (one per lane).
#[target_feature(enable = "avx2")]
pub unsafe fn chacha20_blocks(key: &[u8; 32], first: u32, out: &mut [u8]) {
    assert!(out.len().is_multiple_of(64));
    let nblk = out.len() / 64;
    assert!(first as u64 + nblk as u64 <= 1u64 << 32, "chacha20: block counter overflow");
    let kw: [u32; 8] = std::array::from_fn(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()));
    let cst = [0x6170_7865u32, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
    let init: [__m256i; 16] = std::array::from_fn(|i| match i {
        0..=3 => _mm256_set1_epi32(cst[i] as i32),
        4..=11 => _mm256_set1_epi32(kw[i - 4] as i32),
        _ => _mm256_setzero_si256(),
    });
    let rot16 = _mm256_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5,
                                4, 7, 6, 1, 0, 3, 2);
    let rot8 = _mm256_set_epi8(14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5,
                               4, 7, 2, 1, 0, 3);
    let lane = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let mut tmp = [0u8; 512];
    let mut b = 0usize;
    while b < nblk {
        let mut x = init;
        x[12] = _mm256_add_epi32(_mm256_set1_epi32(first.wrapping_add(b as u32) as i32), lane);
        let ctr = x[12];
        macro_rules! qr {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                x[$a] = _mm256_add_epi32(x[$a], x[$b]);
                x[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(x[$d], x[$a]), rot16);
                x[$c] = _mm256_add_epi32(x[$c], x[$d]);
                x[$b] = rotl::<12, 20>(_mm256_xor_si256(x[$b], x[$c]));
                x[$a] = _mm256_add_epi32(x[$a], x[$b]);
                x[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(x[$d], x[$a]), rot8);
                x[$c] = _mm256_add_epi32(x[$c], x[$d]);
                x[$b] = rotl::<7, 25>(_mm256_xor_si256(x[$b], x[$c]));
            };
        }
        for _ in 0..10 {
            qr!(0, 4, 8, 12);
            qr!(1, 5, 9, 13);
            qr!(2, 6, 10, 14);
            qr!(3, 7, 11, 15);
            qr!(0, 5, 10, 15);
            qr!(1, 6, 11, 12);
            qr!(2, 7, 8, 13);
            qr!(3, 4, 9, 14);
        }
        for i in 0..16 {
            x[i] = _mm256_add_epi32(x[i], if i == 12 { ctr } else { init[i] });
        }
        let mut lo: [__m256i; 8] = std::array::from_fn(|i| x[i]);
        let mut hi: [__m256i; 8] = std::array::from_fn(|i| x[8 + i]);
        transpose8x8(&mut lo); // lo[l] = words 0..7 of block l
        transpose8x8(&mut hi);
        let take = (nblk - b).min(8);
        let dst = if take == 8 { out.as_mut_ptr().add(64 * b) } else { tmp.as_mut_ptr() };
        for l in 0..8 {
            _mm256_storeu_si256(dst.add(64 * l) as *mut _, lo[l]);
            _mm256_storeu_si256(dst.add(64 * l + 32) as *mut _, hi[l]);
        }
        if take < 8 {
            out[64 * b..64 * (b + take)].copy_from_slice(&tmp[..64 * take]);
        }
        b += take;
    }
    tmp.fill(0);
    std::hint::black_box(&tmp);
}
