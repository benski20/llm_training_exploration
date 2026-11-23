//! Hand-written AVX-512 copies of the hot loops (x86-64 only, used when the ISA is `avx512`).
//!
//! Every function computes exactly the integers (and, in the decryption's final rounding, exactly the
//! IEEE-754 double operations, in the same order, without FMA contraction) of the portable code in
//! [`crate::arith`] / [`crate::kernel`], so the outputs are bit-identical for every ISA (tested). All
//! arithmetic is branch-free in the data: Montgomery products by `vpmuludq`, reductions by `vpminud`.
//!
//! Why not IFMA (`vpmadd52*`): it works on 8 lanes of 52 bits; for these primes (< 2^30) a 16-lane u32
//! Montgomery product needs the same number of 512-bit multiplier uops per lane (6 per 16 lanes against
//! 3 per 8), while the additions and reductions would run on half as many lanes per instruction.
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use crate::arith::{Prime, L};
use std::arch::x86_64::*;

/// Montgomery constants of one prime, broadcast.
#[derive(Clone, Copy)]
pub struct M {
    p: __m512i,
    p2: __m512i,
    pn: __m512i,
}

#[inline(always)]
pub unsafe fn consts(p: u32, pn: u32) -> M {
    M { p: _mm512_set1_epi32(p as i32), p2: _mm512_set1_epi32((2 * p) as i32), pn: _mm512_set1_epi32(pn as i32) }
}

#[inline(always)]
unsafe fn ld(p: *const u32) -> __m512i {
    _mm512_loadu_si512(p as *const _)
}

#[inline(always)]
unsafe fn st(p: *mut u32, v: __m512i) {
    _mm512_storeu_si512(p as *mut _, v)
}

/// `min(v, v - q)` per u32 lane (`[0, 2q) -> [0, q)`).
#[inline(always)]
pub unsafe fn red(v: __m512i, q: __m512i) -> __m512i {
    _mm512_min_epu32(v, _mm512_sub_epi32(v, q))
}

/// [`crate::arith::mont`] on 16 lanes: `(a b + m p) / 2^32`, `m = (a b mod 2^32) pn mod 2^32`.
#[inline(always)]
pub unsafe fn mont(a: __m512i, b: __m512i, m: M) -> __m512i {
    let t0 = _mm512_mul_epu32(a, b);
    let t1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
    let s0 = _mm512_add_epi64(t0, _mm512_mul_epu32(_mm512_mul_epu32(t0, m.pn), m.p));
    let s1 = _mm512_add_epi64(t1, _mm512_mul_epu32(_mm512_mul_epu32(t1, m.pn), m.p));
    _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(s0, 32), s1)
}

/// [`mont`] with `b` a 32-bit value broadcast to every lane (`b >> 32` per 64-bit lane is `b` itself).
#[inline(always)]
pub unsafe fn mont_bc(a: __m512i, b: __m512i, m: M) -> __m512i {
    let t0 = _mm512_mul_epu32(a, b);
    let t1 = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), b);
    let s0 = _mm512_add_epi64(t0, _mm512_mul_epu32(_mm512_mul_epu32(t0, m.pn), m.p));
    let s1 = _mm512_add_epi64(t1, _mm512_mul_epu32(_mm512_mul_epu32(t1, m.pn), m.p));
    _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(s0, 32), s1)
}

/// Gentleman-Sande butterfly on the vectors at `x`, `y` (twiddle `z` broadcast).
#[inline(always)]
unsafe fn gs(x: *mut u32, y: *mut u32, z: __m512i, m: M) {
    let (u, w) = (ld(x), ld(y));
    st(x, red(_mm512_add_epi32(u, w), m.p2));
    st(y, mont_bc(_mm512_sub_epi32(_mm512_add_epi32(u, m.p2), w), z, m));
}

/// Cooley-Tukey butterfly.
#[inline(always)]
unsafe fn ct(x: *mut u32, y: *mut u32, z: __m512i, m: M) {
    let (u, w) = (ld(x), ld(y));
    let t = mont_bc(w, z, m);
    st(x, red(_mm512_add_epi32(u, t), m.p2));
    st(y, red(_mm512_sub_epi32(_mm512_add_epi32(u, m.p2), t), m.p2));
}

const SUB: usize = 256;
const MAXPTS: usize = 32;

/// [`Prime::inv_l_mul`] (same values): pointwise product and the stages `len < 256` per 256-coefficient
/// sub-block, then the stages `len >= 256` column-wise through an L1 buffer.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn inv_l_mul(pr: &Prime, a: &mut [u32], h: &[u32], t: &[u32]) {
    let n = pr.n;
    assert!(a.len() >= n * L && h.len() >= n * L && t.len() >= n && n >= 2);
    let m = consts(pr.p, pr.pn);
    let ap = a.as_mut_ptr();
    let hp = h.as_ptr();
    let sub = SUB.min(n);
    for sb in 0..n / sub {
        let base = sb * sub;
        for c in base..base + sub {
            st(ap.add(c * L), mont_bc(ld(hp.add(c * L)), _mm512_set1_epi32(t[c] as i32), m));
        }
        let mut len = 1usize;
        while len < sub {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..sub / (2 * len) {
                let z = _mm512_set1_epi32(pr.izm[nb + b0 + bb] as i32);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    gs(x.add(i * L), x.add((i + len) * L), z, m);
                }
            }
            len *= 2;
        }
    }
    if sub < n {
        inv_cols(pr, a, sub, m);
    }
}

#[inline(always)]
unsafe fn inv_cols(pr: &Prime, a: &mut [u32], from: usize, m: M) {
    let n = pr.n;
    let pts = n / from;
    assert!(pts <= MAXPTS);
    let ap = a.as_mut_ptr();
    let mut buf = [0u32; MAXPTS * L];
    let bp = buf.as_mut_ptr();
    for c0 in 0..from {
        for k in 0..pts {
            st(bp.add(k * L), ld(ap.add((c0 + from * k) * L)));
        }
        let mut lk = 1usize;
        while lk < pts {
            let nb = pts / (2 * lk);
            for bk in 0..nb {
                let z = _mm512_set1_epi32(pr.izm[nb + bk] as i32);
                let x = bp.add(2 * bk * lk * L);
                for i in 0..lk {
                    gs(x.add(i * L), x.add((i + lk) * L), z, m);
                }
            }
            lk *= 2;
        }
        for k in 0..pts {
            st(ap.add((c0 + from * k) * L), ld(bp.add(k * L)));
        }
    }
    for k in 0..MAXPTS {
        st(bp.add(k * L), _mm512_setzero_si512());
    }
}

/// [`Prime::fwd_l`] (same values): stages `len >= 256` column-wise, then the rest per sub-block.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn fwd_l(pr: &Prime, a: &mut [u32]) {
    let n = pr.n;
    assert!(a.len() >= n * L && n >= 2);
    let m = consts(pr.p, pr.pn);
    let ap = a.as_mut_ptr();
    let to = SUB.min(n / 2).max(n / MAXPTS);
    let pts = n / to;
    let mut buf = [0u32; MAXPTS * L];
    let bp = buf.as_mut_ptr();
    for c0 in 0..to {
        for k in 0..pts {
            st(bp.add(k * L), ld(ap.add((c0 + to * k) * L)));
        }
        let mut lk = pts / 2;
        while lk >= 1 {
            let nb = pts / (2 * lk);
            for bk in 0..nb {
                let z = _mm512_set1_epi32(pr.zm[nb + bk] as i32);
                let x = bp.add(2 * bk * lk * L);
                for i in 0..lk {
                    ct(x.add(i * L), x.add((i + lk) * L), z, m);
                }
            }
            lk /= 2;
        }
        for k in 0..pts {
            st(ap.add((c0 + to * k) * L), ld(bp.add(k * L)));
        }
    }
    for k in 0..MAXPTS {
        st(bp.add(k * L), _mm512_setzero_si512());
    }
    for sb in 0..pts {
        let base = sb * to;
        let mut len = to / 2;
        while len >= 1 {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..to / (2 * len) {
                let z = _mm512_set1_epi32(pr.zm[nb + b0 + bb] as i32);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    ct(x.add(i * L), x.add((i + len) * L), z, m);
                }
            }
            len /= 2;
        }
    }
}

/// In-register transpose of a 16 x 16 u32 block (`r[i]` lane `c` -> `r[c]` lane `i`).
#[inline(always)]
pub unsafe fn transpose16(r: &mut [__m512i; 16]) {
    let mut s = 8usize;
    while s >= 1 {
        let mut lo = [0i32; 16];
        let mut hi = [0i32; 16];
        for c in 0..16 {
            if c & s == 0 {
                lo[c] = c as i32;
                hi[c] = (c + s) as i32;
            } else {
                lo[c] = (16 + c - s) as i32;
                hi[c] = (16 + c) as i32;
            }
        }
        let (il, ih) = (_mm512_loadu_si512(lo.as_ptr() as *const _), _mm512_loadu_si512(hi.as_ptr() as *const _));
        for i in 0..16 {
            if i & s == 0 {
                let (a, b) = (r[i], r[i + s]);
                r[i] = _mm512_permutex2var_epi32(a, il, b);
                r[i + s] = _mm512_permutex2var_epi32(a, ih, b);
            }
        }
        s /= 2;
    }
}

/// ChaCha20 keystream (RFC 8439 block function, 32-byte key, all-zero 96-bit nonce, 32-bit block counter
/// starting at block `first`) written to `out` (a multiple of 64 bytes): 16 blocks per iteration, one block
/// per lane, so the bytes are exactly those of the `chacha20` crate's `ChaCha20` seeked to `64 first`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn chacha20_blocks(key: &[u8; 32], first: u32, out: &mut [u8]) {
    assert!(out.len() % 64 == 0);
    let nblk = out.len() / 64;
    assert!(first as u64 + nblk as u64 <= 1u64 << 32, "chacha20: block counter overflow");
    let kw: [u32; 8] = std::array::from_fn(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()));
    let init: [__m512i; 16] = [
        _mm512_set1_epi32(0x6170_7865),
        _mm512_set1_epi32(0x3320_646e),
        _mm512_set1_epi32(0x7962_2d32),
        _mm512_set1_epi32(0x6b20_6574),
        _mm512_set1_epi32(kw[0] as i32),
        _mm512_set1_epi32(kw[1] as i32),
        _mm512_set1_epi32(kw[2] as i32),
        _mm512_set1_epi32(kw[3] as i32),
        _mm512_set1_epi32(kw[4] as i32),
        _mm512_set1_epi32(kw[5] as i32),
        _mm512_set1_epi32(kw[6] as i32),
        _mm512_set1_epi32(kw[7] as i32),
        _mm512_setzero_si512(), // counter: set per iteration
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
    ];
    let lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let mut tmp = [0u8; 1024];
    let mut b = 0usize;
    while b < nblk {
        let mut x = init;
        x[12] = _mm512_add_epi32(_mm512_set1_epi32(first.wrapping_add(b as u32) as i32), lane);
        let ctr = x[12];
        macro_rules! qr {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                x[$a] = _mm512_add_epi32(x[$a], x[$b]);
                x[$d] = _mm512_rol_epi32(_mm512_xor_si512(x[$d], x[$a]), 16);
                x[$c] = _mm512_add_epi32(x[$c], x[$d]);
                x[$b] = _mm512_rol_epi32(_mm512_xor_si512(x[$b], x[$c]), 12);
                x[$a] = _mm512_add_epi32(x[$a], x[$b]);
                x[$d] = _mm512_rol_epi32(_mm512_xor_si512(x[$d], x[$a]), 8);
                x[$c] = _mm512_add_epi32(x[$c], x[$d]);
                x[$b] = _mm512_rol_epi32(_mm512_xor_si512(x[$b], x[$c]), 7);
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
            x[i] = _mm512_add_epi32(x[i], if i == 12 { ctr } else { init[i] });
        }
        // x[i] lane l = word i of block l: transpose to block-major
        transpose16(&mut x);
        let take = (nblk - b).min(16);
        let dst = if take == 16 { out.as_mut_ptr().add(64 * b) } else { tmp.as_mut_ptr() };
        for (l, v) in x.iter().enumerate() {
            _mm512_storeu_si512(dst.add(64 * l) as *mut _, *v);
        }
        if take < 16 {
            out[64 * b..64 * (b + take)].copy_from_slice(&tmp[..64 * take]);
        }
        b += take;
    }
    tmp.fill(0);
    std::hint::black_box(&tmp);
}

/// In-register transpose of an 8 x 8 u64 block.
#[inline(always)]
pub unsafe fn transpose8(r: &mut [__m512i; 8]) {
    let mut s = 4usize;
    while s >= 1 {
        let mut lo = [0i64; 8];
        let mut hi = [0i64; 8];
        for c in 0..8 {
            if c & s == 0 {
                lo[c] = c as i64;
                hi[c] = (c + s) as i64;
            } else {
                lo[c] = (8 + c - s) as i64;
                hi[c] = (8 + c) as i64;
            }
        }
        let (il, ih) = (_mm512_loadu_si512(lo.as_ptr() as *const _), _mm512_loadu_si512(hi.as_ptr() as *const _));
        for i in 0..8 {
            if i & s == 0 {
                let (a, b) = (r[i], r[i + s]);
                r[i] = _mm512_permutex2var_epi64(a, il, b);
                r[i + s] = _mm512_permutex2var_epi64(a, ih, b);
            }
        }
        s /= 2;
    }
}

/// Constants of the upload's CRT (pad primes `e0 > e1 > e2`, all `< 2^30`).
pub struct EncConsts {
    pub e: [u32; 3],
    pub pn: [u32; 3],
    /// `e0^-1 R mod e1`.
    pub inv01_m: u32,
    /// `e0 R mod e2`.
    pub e0_m2: u32,
    /// `(e0 e1)^-1 R mod e2`.
    pub inv012_m: u32,
    /// `e0 e1` and `e0 e1 e2 mod 2^64`.
    pub p01: u64,
    pub emod: u64,
}

/// Upload output of one group and 16-block chunk: for every slot `u < 256`, feature `f = 16 (c0 + l) + k`
/// (`l < lanes`): `b[f cols + u] = (crt(s) + e + 2^24 (r[u, f] mod 2^32)) mod 2^56`, the centred CRT of the
/// pad from the three inverse NTTs `s` (`3 x NU x L`), the noise `noise[(16 l + k) 256 + u]` and the masks
/// `r` (row `u` at `r + u n`, rows `>= rows_valid` are zero). Same values as the portable loop of
/// `kernel::enc_chunk` (the CRT digits `k1`, `k2` are the unique canonical residues either way).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn enc_out(k: &EncConsts, s: *const u32, nu: usize, lanes: usize, c0: usize, cols: usize, n: usize,
                      r: *const i64, rows_valid: usize, noise: *const i32, b: *mut u64) {
    let m1 = consts(k.e[1], k.pn[1]);
    let m2 = consts(k.e[2], k.pn[2]);
    let ev: [__m512i; 3] = std::array::from_fn(|i| _mm512_set1_epi32(k.e[i] as i32));
    let e2x: [__m512i; 3] = std::array::from_fn(|i| _mm512_set1_epi32((2 * k.e[i]) as i32));
    let inv01 = _mm512_set1_epi32(k.inv01_m as i32);
    let e0m2 = _mm512_set1_epi32(k.e0_m2 as i32);
    let inv012 = _mm512_set1_epi32(k.inv012_m as i32);
    let e0q = _mm512_set1_epi64(k.e[0] as i64);
    let (pl, ph) = (_mm512_set1_epi64((k.p01 & 0xFFFF_FFFF) as i64), _mm512_set1_epi64((k.p01 >> 32) as i64));
    let emod = _mm512_set1_epi64(k.emod as i64);
    let half2 = _mm512_set1_epi64((k.e[2] / 2) as i64);
    let m32 = _mm512_set1_epi64(0xFFFF_FFFF);
    let m56 = _mm512_set1_epi64(((1u64 << 56) - 1) as i64);
    let gidx = _mm512_set_epi64(7 * n as i64, 6 * n as i64, 5 * n as i64, 4 * n as i64, 3 * n as i64, 2 * n as i64,
                                n as i64, 0);
    let ext = |v: __m512i, h: usize| -> __m512i {
        if h == 0 {
            _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v))
        } else {
            _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(v, 1))
        }
    };
    let mut xs = [[_mm512_setzero_si512(); 8]; 2];
    for u0 in (0..256usize).step_by(8) {
        let gmask: __mmask8 = if u0 + 8 <= rows_valid {
            0xFF
        } else if u0 >= rows_valid {
            0
        } else {
            ((1u32 << (rows_valid - u0)) - 1) as u8
        };
        for kk in 0..16 {
            for uu in 0..8 {
                let c = (16 * (u0 + uu) + kk) * L;
                let r0 = red(red(ld(s.add(c)), e2x[0]), ev[0]);
                let r1 = red(red(ld(s.add(nu * L + c)), e2x[1]), ev[1]);
                let r2 = red(ld(s.add(2 * nu * L + c)), ev[2]);
                let k1 = red(mont_bc(_mm512_sub_epi32(_mm512_add_epi32(r1, ev[1]), red(r0, ev[1])), inv01, m1), ev[1]);
                let a = red(mont_bc(k1, e0m2, m2), ev[2]);
                let z = _mm512_sub_epi32(_mm512_sub_epi32(_mm512_add_epi32(r2, e2x[2]), red(r0, ev[2])), a);
                let k2 = red(mont_bc(z, inv012, m2), ev[2]);
                for h in 0..2 {
                    let x01 = _mm512_add_epi64(ext(r0, h), _mm512_mul_epu32(ext(k1, h), e0q));
                    let kw = ext(k2, h);
                    let pk = _mm512_add_epi64(_mm512_mul_epu32(kw, pl), _mm512_slli_epi64(_mm512_mul_epu32(kw, ph), 32));
                    let x = _mm512_add_epi64(x01, pk);
                    let neg = _mm512_cmpgt_epu64_mask(kw, half2);
                    xs[h][uu] = _mm512_mask_sub_epi64(x, neg, x, emod);
                }
            }
            transpose8(&mut xs[0]);
            transpose8(&mut xs[1]);
            for l in 0..lanes {
                let f = (c0 + l) * 16 + kk;
                let x = xs[l / 8][l % 8];
                let e = _mm512_cvtepi32_epi64(_mm256_loadu_si256(noise.add((l * 16 + kk) * 256 + u0) as *const _));
                let msg = if gmask == 0 {
                    _mm512_setzero_si512()
                } else {
                    _mm512_mask_i64gather_epi64::<8>(_mm512_setzero_si512(), gmask, gidx, r.add(u0 * n + f))
                };
                let msg = _mm512_slli_epi64(_mm512_and_si512(msg, m32), 24);
                let v = _mm512_and_si512(_mm512_add_epi64(_mm512_add_epi64(x, e), msg), m56);
                _mm512_storeu_si512(b.add(f * cols + u0) as *mut _, v);
            }
        }
    }
    xs = [[_mm512_setzero_si512(); 8]; 2];
    std::hint::black_box(&xs);
}

/// Interleave 16 polynomials of `n` 24-bit little-endian residues (`src[l]`, `3 n` bytes each; lanes
/// `>= lanes` are zero) into `h[c L + l]` (the compact hint's load step of the decryption).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn load_hint(src: &[*const u8; L], lanes: usize, n: usize, h: *mut u32) {
    let perm = _mm512_set_epi32(12, 11, 10, 9, 9, 8, 7, 6, 6, 5, 4, 3, 3, 2, 1, 0);
    let shuf = _mm512_broadcast_i32x4(_mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0));
    let mut v = [_mm512_setzero_si512(); 16];
    for c in (0..n).step_by(16) {
        for (l, vl) in v.iter_mut().enumerate() {
            *vl = if l < lanes {
                let raw = _mm512_maskz_loadu_epi8(0x0000_FFFF_FFFF_FFFF, src[l].add(3 * c) as *const i8);
                _mm512_shuffle_epi8(_mm512_permutexvar_epi32(perm, raw), shuf)
            } else {
                _mm512_setzero_si512()
            };
        }
        transpose16(&mut v);
        for (i, vi) in v.iter().enumerate() {
            _mm512_storeu_si512(h.add((c + i) * L) as *mut _, *vi);
        }
    }
}

/// Reply-key output for one prime: `beta[l][c] = red(red(mont(k[l][c], th[c]) + m16[c L + l], 2p), p)`
/// (`k` in Montgomery form, `th` the NTT of `t'`, `m16` the interleaved NTT of `e + P iota(t)`). Same
/// values as the portable loop of `kernel::ksk_inner`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn ksk_out(p: u32, pn: u32, n: usize, km: *const u32, th: *const u32, m16: *const u32, beta: *mut u32) {
    let m = consts(p, pn);
    let mut v = [_mm512_setzero_si512(); 16];
    for c in (0..n).step_by(16) {
        for (i, vi) in v.iter_mut().enumerate() {
            *vi = ld(m16.add((c + i) * L));
        }
        transpose16(&mut v); // v[l] = m16[c .. c + 16][l]
        let t = ld(th.add(c));
        for (l, vl) in v.iter().enumerate() {
            let kt = mont(ld(km.add(l * n + c)), t, m);
            st(beta.add(l * n + c), red(red(_mm512_add_epi32(kt, *vl), m.p2), m.p));
        }
    }
    v = [_mm512_setzero_si512(); 16];
    std::hint::black_box(&v);
}

/// Constants of the decryption's final combine.
pub struct DecConsts {
    pub p0: u32,
    pub p1: u32,
    pub pn1: u32,
    pub inv01_m: u32,
    pub c0scale: f64,
    pub wscale: f64,
}

/// Reply value source of one block row: `Words` base pointer or the packed-36 layout.
#[derive(Clone, Copy)]
pub enum C0p {
    Words(*const u64),
    /// `lo` (u32 words) and `hi` (nibble stream) base pointers.
    Packed36 { lo: *const u8, hi: *const u8 },
}

/// The decryption's final combine for one slot `u` of a 16-block chunk: CRT of `c1 t'` from the two
/// inverse NTTs (`s0`, `s1`: the 32 vectors of coefficients `32 u .. 32 u + 31`), plus the reply value,
/// rounded: `out[l][j] = (c0 * c0scale + w * wscale + 0.5) as u64 as u32` exactly as the portable code
/// (`kernel::dec_chunk`). `c0` index of lane `l`, row `j`: `idx0[l] + j`; output pointer `outp[l]`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
pub unsafe fn dec_combine(k: &DecConsts, s0: *const u32, s1: *const u32, lanes: usize, c0: C0p, idx0: &[usize; L],
                          outp: &[*mut u32; L]) {
    let m1 = consts(k.p1, k.pn1);
    let (p0v, p1v) = (_mm512_set1_epi32(k.p0 as i32), _mm512_set1_epi32(k.p1 as i32));
    let (p02, p12) = (_mm512_set1_epi32((2 * k.p0) as i32), _mm512_set1_epi32((2 * k.p1) as i32));
    let inv = _mm512_set1_epi32(k.inv01_m as i32);
    let mut r0 = [[_mm512_setzero_si512(); 16]; 2];
    let mut kk = [[_mm512_setzero_si512(); 16]; 2];
    for j in 0..32 {
        let x0 = ld(s0.add(j * L));
        let x1 = ld(s1.add(j * L));
        let a = red(red(x0, p02), p0v);
        let b = red(red(x1, p12), p1v);
        let am = red(a, p1v);
        let kv = red(mont_bc(_mm512_sub_epi32(_mm512_add_epi32(b, p1v), am), inv, m1), p1v);
        r0[j / 16][j % 16] = a;
        kk[j / 16][j % 16] = kv;
    }
    for h in 0..2 {
        transpose16(&mut r0[h]);
        transpose16(&mut kk[h]);
    }
    let p0f = _mm512_set1_pd(k.p0 as f64);
    let wsc = _mm512_set1_pd(k.wscale);
    let csc = _mm512_set1_pd(k.c0scale);
    let half = _mm512_set1_pd(0.5);
    let sh = _mm512_set_epi64(28, 24, 20, 16, 12, 8, 4, 0);
    let m4 = _mm512_set1_epi64(15);
    for l in 0..lanes {
        for q in 0..4 {
            let j = 8 * q;
            let (rv, kv) = (r0[q / 2][l], kk[q / 2][l]);
            let (r8, k8) = if q % 2 == 0 {
                (_mm512_castsi512_si256(rv), _mm512_castsi512_si256(kv))
            } else {
                (_mm512_extracti64x4_epi64(rv, 1), _mm512_extracti64x4_epi64(kv, 1))
            };
            // w = r0 + p0 k exactly (< 2^48), then the one rounding of w * wscale
            let wf = _mm512_mul_pd(_mm512_add_pd(_mm512_cvtepu32_pd(r8), _mm512_mul_pd(p0f, _mm512_cvtepu32_pd(k8))), wsc);
            let idx = idx0[l] + j;
            let v = match c0 {
                C0p::Words(w) => _mm512_loadu_si512(w.add(idx) as *const _),
                C0p::Packed36 { lo, hi } => {
                    let lo8 = _mm256_loadu_si256(lo.add(4 * idx) as *const _);
                    let wd = (hi.add(idx / 2) as *const u32).read_unaligned();
                    let nib = _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64(wd as i64), sh), m4);
                    _mm512_or_si512(_mm512_cvtepu32_epi64(lo8), _mm512_slli_epi64(nib, 32))
                }
            };
            let x = _mm512_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), csc), wf);
            let r = _mm512_cvttpd_epu64(_mm512_add_pd(x, half));
            _mm256_storeu_si256(outp[l].add(j) as *mut _, _mm512_cvtepi64_epi32(r));
        }
    }
    // wipe the secret-dependent registers' spill copies
    r0 = [[_mm512_setzero_si512(); 16]; 2];
    kk = [[_mm512_setzero_si512(); 16]; 2];
    std::hint::black_box((&r0, &kk));
}
