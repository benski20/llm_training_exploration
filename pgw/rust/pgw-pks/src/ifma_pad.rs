//! AVX-512 IFMA upload pad (`Isa::Avx512Ifma`): the exact product `a_C t` over `Z[Y]/(Y^4096 + 1)` (needs
//! a CRT range above `2^69`) with TWO ~2^50 NTT primes on 52-bit lanes instead of three ~2^30 primes on
//! 32-bit lanes. The pad is an exact integer, so the bodies `b` are bit-identical to the three-prime path
//! and the numpy reference whatever primes are used (tested); the primes are an internal choice.
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::needless_range_loop)]

use crate::arith::L;
use crate::ifma::{inv_mul_tab, ld, mulq, red, shoup_ct, shoup_exact, st, K};
use crate::simd::transpose8;
use std::arch::x86_64::*;
use std::sync::OnceLock;

fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    (a as u128 * b as u128 % p as u128) as u64
}

fn powmod64(mut b: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    b %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, b, p);
        }
        b = mulmod(b, b, p);
        e >>= 1;
    }
    r
}

fn is_prime64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// One prime `p < 2^50`, `p = 1 mod 2n`, with forward (`psi^brv(k)`) and inverse (`psi^-brv(k)`) twiddles
/// and their Shoup companions.
pub struct Tab {
    pub p: u64,
    pub n: usize,
    z: Vec<u64>,
    zs: Vec<u64>,
    iz: Vec<u64>,
    izs: Vec<u64>,
}

impl Tab {
    fn new(p: u64, n: usize) -> Tab {
        let lg = n.trailing_zeros();
        let e = (p - 1) / (2 * n as u64);
        // psi of order exactly 2n: psi^n = -1
        let psi = (2u64..).map(|g| powmod64(g, e, p)).find(|&s| powmod64(s, n as u64, p) == p - 1).unwrap();
        let ipsi = powmod64(psi, p - 2, p);
        let brv = |k: usize| ((k as u32).reverse_bits() >> (32 - lg)) as u64;
        let z: Vec<u64> = (0..n).map(|k| powmod64(psi, brv(k), p)).collect();
        let iz: Vec<u64> = (0..n).map(|k| powmod64(ipsi, brv(k), p)).collect();
        let zs = z.iter().map(|&w| shoup_exact(w, p)).collect();
        let izs = iz.iter().map(|&w| shoup_exact(w, p)).collect();
        Tab { p, n, z, zs, iz, izs }
    }
}

/// The pad context: primes `f0 > f1` (the largest below `2^50` that are `1 mod 8192`), CRT constants.
pub struct Pad {
    pub t: [Tab; 2],
    /// `2^32 mod f_i` and `N^-1 mod f_i` with Shoup companions.
    r32: [(u64, u64); 2],
    ninv: [(u64, u64); 2],
    /// `f0^-1 mod f1` with its Shoup companion; `f0 f1 mod 2^64`.
    inv01: (u64, u64),
    fmod: u64,
}

pub fn pad_ctx(n: usize) -> &'static Pad {
    static C: OnceLock<Pad> = OnceLock::new();
    let c = C.get_or_init(|| {
        let step = 2 * n as u64;
        let mut c = ((1u64 << 50) - 1) / step * step + 1;
        let mut ps = Vec::new();
        while ps.len() < 2 {
            if is_prime64(c) {
                ps.push(c);
            }
            c -= step;
        }
        let t = [Tab::new(ps[0], n), Tab::new(ps[1], n)];
        let pair = |w: u64, p: u64| (w, shoup_exact(w, p));
        let r32 = [pair((1u64 << 32) % ps[0], ps[0]), pair((1u64 << 32) % ps[1], ps[1])];
        let ninv = [pair(powmod64(n as u64, ps[0] - 2, ps[0]), ps[0]), pair(powmod64(n as u64, ps[1] - 2, ps[1]), ps[1])];
        let inv01 = pair(powmod64(ps[0] % ps[1], ps[1] - 2, ps[1]), ps[1]);
        Pad { t, r32, ninv, inv01, fmod: ps[0].wrapping_mul(ps[1]) }
    });
    assert_eq!(c.t[0].n, n);
    c
}

#[inline(always)]
fn kk(p: u64) -> (u64, u64) {
    (p, 2 * p)
}

#[inline(always)]
unsafe fn kv(p: u64) -> K {
    let (p, p2) = kk(p);
    K { q: _mm512_set1_epi64(p as i64), q2: _mm512_set1_epi64(p2 as i64), m52: _mm512_set1_epi64((1 << 52) - 1) }
}

/// Scalar Shoup product (`a < 2^52`), in `[0, 2p)`; constant time.
#[inline(always)]
fn mulq_s(a: u64, w: u64, ws: u64, p: u64) -> u64 {
    let qh = ((a as u128 * ws as u128) >> 52) as u64;
    a.wrapping_mul(w).wrapping_sub(qh.wrapping_mul(p))
}

#[inline(always)]
fn red_s(v: u64, q: u64) -> u64 {
    v.min(v.wrapping_sub(q))
}

/// NTT of one group's ternary upload secret modulo both pad primes (`out[i]`, `< f_i`) and the Shoup
/// companions (`outs[i]`): scalar, branch-free (secret).
pub fn secret_hat(pc: &Pad, t: &[i8], out: &mut [Vec<u64>; 2], outs: &mut [Vec<u64>; 2]) {
    for i in 0..2 {
        let tb = &pc.t[i];
        let (p, p2, n) = (tb.p, 2 * tb.p, tb.n);
        let a = &mut out[i];
        a.clear();
        a.extend(t.iter().map(|&x| {
            let m = ((x as i64) >> 63) as u64; // all ones when negative
            (x as i64 as u64).wrapping_add(p & m)
        }));
        let (mut len, mut k) = (n / 2, 1usize);
        while len >= 1 {
            let nb = n / (2 * len);
            for b in 0..nb {
                let (z, zs) = (tb.z[k + b], tb.zs[k + b]);
                for j in 2 * b * len..2 * b * len + len {
                    let tt = mulq_s(a[j + len], z, zs, p);
                    let u = a[j];
                    a[j] = red_s(u + tt, p2);
                    a[j + len] = red_s(u + p2 - tt, p2);
                }
            }
            k += nb;
            len /= 2;
        }
        let o = &mut outs[i];
        o.clear();
        for x in a.iter_mut() {
            *x = red_s(*x, p);
            o.push(shoup_ct(*x, p));
        }
    }
}

/// Cooley-Tukey butterfly on 16 lanes (two 8-lane vectors per coefficient), values in `[0, 2p)`.
#[inline(always)]
unsafe fn ct(x: *mut u64, y: *mut u64, z: __m512i, zs: __m512i, k: K) {
    for h in 0..2 {
        let (u, w) = (ld(x.add(8 * h)), ld(y.add(8 * h)));
        let t = mulq(w, z, zs, k);
        st(x.add(8 * h), red(_mm512_add_epi64(u, t), k.q2));
        st(y.add(8 * h), red(_mm512_sub_epi64(_mm512_add_epi64(u, k.q2), t), k.q2));
    }
}

const SUB: usize = 256;
const MAXPTS: usize = 32;

/// Forward NTT on 16 interleaved lanes modulo `tb.p`, then times `N^-1` and reduced to `[0, p)`.
#[inline(always)]
unsafe fn fwd_tab(tb: &Tab, ninv: (u64, u64), a: &mut [u64]) {
    let n = tb.n;
    assert!(a.len() >= n * L);
    let k = kv(tb.p);
    let ap = a.as_mut_ptr();
    let to = SUB.min(n / 2).max(n / MAXPTS);
    let pts = n / to;
    let tw = |i: usize| (_mm512_set1_epi64(tb.z[i] as i64), _mm512_set1_epi64(tb.zs[i] as i64));
    let mut buf = [0u64; MAXPTS * L];
    let bp = buf.as_mut_ptr();
    for c0 in 0..to {
        for kk2 in 0..pts {
            for h in 0..2 {
                st(bp.add(kk2 * L + 8 * h), ld(ap.add((c0 + to * kk2) * L + 8 * h)));
            }
        }
        let mut lk = pts / 2;
        while lk >= 1 {
            let nb = pts / (2 * lk);
            for bk in 0..nb {
                let (z, zs) = tw(nb + bk);
                let x = bp.add(2 * bk * lk * L);
                for i in 0..lk {
                    ct(x.add(i * L), x.add((i + lk) * L), z, zs, k);
                }
            }
            lk /= 2;
        }
        for kk2 in 0..pts {
            for h in 0..2 {
                st(ap.add((c0 + to * kk2) * L + 8 * h), ld(bp.add(kk2 * L + 8 * h)));
            }
        }
    }
    for sb in 0..pts {
        let base = sb * to;
        let mut len = to / 2;
        while len >= 1 {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..to / (2 * len) {
                let (z, zs) = tw(nb + b0 + bb);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    ct(x.add(i * L), x.add((i + len) * L), z, zs, k);
                }
            }
            len /= 2;
        }
    }
    let (w, ws) = (_mm512_set1_epi64(ninv.0 as i64), _mm512_set1_epi64(ninv.1 as i64));
    for i in 0..n * L / 8 {
        st(ap.add(8 * i), red(mulq(ld(ap.add(8 * i)), w, ws, k), k.q));
    }
}

/// The public ring elements of one 16-block chunk, in the NTT domain modulo both pad primes, times `N^-1`:
/// `a[i][c][l]` (`2 x n x 16` u64). `raw[l]`: block `l`'s keystream (`8 n` bytes, low 56 bits per word);
/// lanes `>= lanes` are zero.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub unsafe fn pad_a(pc: &Pad, raw: &[u8], lanes: usize, a: &mut [u64]) {
    let n = pc.t[0].n;
    assert!(raw.len() >= L * n * 8 && a.len() >= 2 * n * L);
    let m56 = _mm512_set1_epi64(((1u64 << 56) - 1) as i64);
    let ks = [kv(pc.t[0].p), kv(pc.t[1].p)];
    let r32 = [
        (_mm512_set1_epi64(pc.r32[0].0 as i64), _mm512_set1_epi64(pc.r32[0].1 as i64)),
        (_mm512_set1_epi64(pc.r32[1].0 as i64), _mm512_set1_epi64(pc.r32[1].1 as i64)),
    ];
    let m32 = _mm512_set1_epi64(0xFFFF_FFFF);
    let rp = raw.as_ptr() as *const u64;
    let ap = a.as_mut_ptr();
    for c in (0..n).step_by(8) {
        // v[i][l] = residues of coefficients c .. c + 8 of block l modulo f_i
        let mut v = [[_mm512_setzero_si512(); 16]; 2];
        for l in 0..lanes {
            let w = _mm512_and_si512(_mm512_loadu_si512(rp.add(l * n + c) as *const _), m56);
            let (hi, lo) = (_mm512_srli_epi64(w, 32), _mm512_and_si512(w, m32));
            for i in 0..2 {
                // a = hi 2^32 + lo: Shoup(hi, 2^32 mod f) + lo < 2 f + 2^32 < 4 f, then into [0, 2 f)
                let x = _mm512_add_epi64(mulq(hi, r32[i].0, r32[i].1, ks[i]), lo);
                v[i][l] = red(x, ks[i].q2);
            }
        }
        for i in 0..2 {
            let (lo8, hi8) = v[i].split_at_mut(8);
            let lo8: &mut [__m512i; 8] = lo8.try_into().unwrap();
            let hi8: &mut [__m512i; 8] = hi8.try_into().unwrap();
            transpose8(lo8); // lo8[j] = coefficient c + j, lanes 0..8
            transpose8(hi8);
            for j in 0..8 {
                st(ap.add(i * n * L + (c + j) * L), lo8[j]);
                st(ap.add(i * n * L + (c + j) * L + 8), hi8[j]);
            }
        }
    }
    for i in 0..2 {
        fwd_tab(&pc.t[i], pc.ninv[i], &mut a[i * n * L..(i + 1) * n * L]);
    }
}

/// `s[i] = INTT_i(a[i] * t_hat_i)` for both pad primes (`s`, `a`: `2 x n x 16`), values in `[0, 2 f_i)`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub unsafe fn pad_mul(pc: &Pad, a: &[u64], th: &[Vec<u64>; 2], ths: &[Vec<u64>; 2], s: &mut [u64]) {
    let n = pc.t[0].n;
    for i in 0..2 {
        let tb = &pc.t[i];
        inv_mul_tab(tb.p, &tb.iz, &tb.izs, &mut s[i * n * L..(i + 1) * n * L], &a[i * n * L..(i + 1) * n * L],
                    &th[i], &ths[i]);
    }
}

/// The upload output of one group (as [`crate::simd::enc_out`]) from the two-prime pad `s` (`2 x n x 16`):
/// `x = r0 + f0 ((r1 - r0) f0^-1 mod f1)`, centred (`k > f1 / 2` -> minus `f0 f1`), mod `2^64`, then noise,
/// message and `mod 2^56` exactly as before.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn pad_out(pc: &Pad, s: &[u64], lanes: usize, c0: usize, cols: usize, n_in: usize, r: *const i64,
                      rows_valid: usize, noise: *const i32, b: *mut u64) {
    let n = pc.t[0].n;
    let (k0, k1) = (kv(pc.t[0].p), kv(pc.t[1].p));
    let (f0, f1) = (_mm512_set1_epi64(pc.t[0].p as i64), _mm512_set1_epi64(pc.t[1].p as i64));
    let (iv, ivs) = (_mm512_set1_epi64(pc.inv01.0 as i64), _mm512_set1_epi64(pc.inv01.1 as i64));
    let fmod = _mm512_set1_epi64(pc.fmod as i64);
    let half1 = _mm512_set1_epi64((pc.t[1].p / 2) as i64);
    let m32 = _mm512_set1_epi64(0xFFFF_FFFF);
    let m56 = _mm512_set1_epi64(((1u64 << 56) - 1) as i64);
    let z = _mm512_setzero_si512();
    let gidx = _mm512_set_epi64(7 * n_in as i64, 6 * n_in as i64, 5 * n_in as i64, 4 * n_in as i64,
                                3 * n_in as i64, 2 * n_in as i64, n_in as i64, 0);
    let sp = s.as_ptr();
    let mut xs = [[_mm512_setzero_si512(); 8]; 2];
    for u0 in (0..256usize).step_by(8) {
        let gmask: __mmask8 = if u0 + 8 <= rows_valid {
            0xFF
        } else if u0 >= rows_valid {
            0
        } else {
            ((1u32 << (rows_valid - u0)) - 1) as u8
        };
        for kk2 in 0..16 {
            for uu in 0..8 {
                let c = (16 * (u0 + uu) + kk2) * L;
                for h in 0..2 {
                    let r0 = red(ld(sp.add(c + 8 * h)), k0.q);
                    let r1 = red(ld(sp.add(n * L + c + 8 * h)), k1.q);
                    let r0m1 = red(r0, k1.q); // r0 < f0 < 2 f1
                    let d = _mm512_sub_epi64(_mm512_add_epi64(r1, f1), r0m1); // < 2 f1
                    let kq = red(mulq(d, iv, ivs, k1), k1.q); // canonical (r1 - r0) f0^-1 mod f1
                    // f0 kq mod 2^64 from the 104-bit product: low 52 bits + (high 52 bits << 52)
                    let lo = _mm512_madd52lo_epu64(z, kq, f0);
                    let hi = _mm512_madd52hi_epu64(z, kq, f0);
                    let x = _mm512_add_epi64(r0, _mm512_add_epi64(lo, _mm512_slli_epi64(hi, 52)));
                    let neg = _mm512_cmpgt_epu64_mask(kq, half1);
                    xs[h][uu] = _mm512_mask_sub_epi64(x, neg, x, fmod);
                }
            }
            transpose8(&mut xs[0]);
            transpose8(&mut xs[1]);
            for l in 0..lanes {
                let f = (c0 + l) * 16 + kk2;
                let x = xs[l / 8][l % 8];
                let e = _mm512_cvtepi32_epi64(_mm256_loadu_si256(noise.add((l * 16 + kk2) * 256 + u0) as *const _));
                let msg = if gmask == 0 {
                    _mm512_setzero_si512()
                } else {
                    _mm512_mask_i64gather_epi64::<8>(_mm512_setzero_si512(), gmask, gidx, r.add(u0 * n_in + f))
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
