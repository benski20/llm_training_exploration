//! AVX-512 IFMA decryption (`Isa::Avx512Ifma`): ONE composite negacyclic NTT modulo `q1 = p0 p1`
//! (< 2^48) on 52-bit lanes instead of one 24-bit NTT per prime.
//!
//! Since both primes are `1 mod 16384`, `psi = CRT(psi_0, psi_1)` (the per-prime roots of
//! [`crate::arith::Prime`]) is a primitive 16384-th root of unity modulo `q1`, and the NTT modulo `q1` with
//! the CRT of the per-prime twiddles reduces modulo `p_i` to the per-prime NTT, index by index. So the
//! product `c1 t' mod q1` computed here is the same canonical integer that the two-prime path reconstructs
//! with its CRT, and the rest of the combine is the same f64 sequence: bit-identical outputs (tested).
//!
//! Products are Shoup products with 52-bit IFMA (`vpmadd52{lo,hi}uq`): `a w mod q` for `a < 2^52` and a
//! fixed `w < q` with `w' = floor(w 2^52 / q)` gives a value in `[0, 2q)`, branch-free. Per 16-lane GS
//! butterfly: 2 x (3 IFMA + 7 simple ops) on 8-lane vectors, against 2 x 17 ops for two 24-bit primes.
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use crate::arith::{mont, powmod, red as red32, Prime, L};
use crate::simd::{transpose8, C0p, DecConsts};
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Twiddles and CRT constants modulo `q1`.
pub struct Q1 {
    pub q: u64,
    /// `psi^-brv(k) mod q1` and its Shoup companion (same indexing as `Prime::izm`).
    izq: Vec<u64>,
    izs: Vec<u64>,
    /// Hint residues -> `q1`: `H = h0 a0 + h1 a1 mod q1` (`a_i = CRT(R^-1 mod p_i, 0)` resp. `(0, ..)`).
    a: [u64; 2],
    as_: [u64; 2],
}

fn shoup_exact(w: u64, q: u64) -> u64 {
    (((w as u128) << 52) / q as u128) as u64
}

fn crt(p0: u64, p1: u64, a0: u64, a1: u64) -> u64 {
    let inv01 = powmod(p0 % p1, p1 - 2, p1);
    let k = (a1 % p1 + p1 - a0 % p1) % p1 * inv01 % p1;
    a0 + p0 * k
}

/// The `q1` context for the two `Q1` primes (degree `NR`).
pub fn q1ctx(pr0: &Prime, pr1: &Prime) -> &'static Q1 {
    static C: OnceLock<Q1> = OnceLock::new();
    C.get_or_init(|| {
        let (p0, p1) = (pr0.p as u64, pr1.p as u64);
        let q = p0 * p1;
        let rinv = |p: u64| powmod((1u64 << 32) % p, p - 2, p);
        let (ri0, ri1) = (rinv(p0), rinv(p1));
        let izq: Vec<u64> = (0..pr0.n)
            .map(|k| crt(p0, p1, pr0.izm[k] as u64 * ri0 % p0, pr1.izm[k] as u64 * ri1 % p1))
            .collect();
        let izs = izq.iter().map(|&w| shoup_exact(w, q)).collect();
        let a = [crt(p0, p1, ri0, 0), crt(p0, p1, 0, ri1)];
        let as_ = [shoup_exact(a[0], q), shoup_exact(a[1], q)];
        Q1 { q, izq, izs, a, as_ }
    })
}

/// `T = CRT(t0, t1) mod q1` and its Shoup companion `floor(T 2^52 / q1)` for one group's `t'` NTT (secret:
/// constant time, no division: an f64 estimate within 2 of the quotient and four masked corrections).
pub fn secret_operand(qc: &Q1, pr1: &Prime, inv01_m: u32, t0: &[u32], t1: &[u32], out_t: &mut [u64], out_s: &mut [u64]) {
    let q = qc.q;
    let (p0, p1) = ((q / pr1.p as u64) as u32, pr1.p);
    let c = 4503599627370496.0f64 / q as f64; // 2^52 / q
    for i in 0..t0.len() {
        // Garner (Montgomery, branch-free): T = t0 + p0 ((t1 - t0) p0^-1 mod p1), t_i < p_i
        let r0m1 = red32(t0[i], p1);
        let k = red32(mont(t1[i].wrapping_add(p1).wrapping_sub(r0m1), inv01_m, p1, pr1.pn), p1);
        let v = t0[i] as u64 + p0 as u64 * k as u64;
        let mut est = (v as f64 * c) as i64 as i128;
        let mut rem = ((v as i128) << 52) - est * q as i128;
        for _ in 0..2 {
            let neg = (rem < 0) as i128;
            est -= neg;
            rem += neg * q as i128;
        }
        for _ in 0..2 {
            let big = (rem >= q as i128) as i128;
            est += big;
            rem -= big * q as i128;
        }
        out_t[i] = v;
        out_s[i] = est as u64;
    }
}

#[derive(Clone, Copy)]
struct K {
    q: __m512i,
    q2: __m512i,
    m52: __m512i,
}

#[inline(always)]
unsafe fn ld(p: *const u64) -> __m512i {
    _mm512_loadu_si512(p as *const _)
}

#[inline(always)]
unsafe fn st(p: *mut u64, v: __m512i) {
    _mm512_storeu_si512(p as *mut _, v)
}

#[inline(always)]
unsafe fn red(v: __m512i, q: __m512i) -> __m512i {
    _mm512_min_epu64(v, _mm512_sub_epi64(v, q))
}

/// Shoup product `a w mod q` in `[0, 2q)` (`a < 2^52`, `w < q`, `ws = floor(w 2^52 / q)`).
#[inline(always)]
unsafe fn mulq(a: __m512i, w: __m512i, ws: __m512i, k: K) -> __m512i {
    let z = _mm512_setzero_si512();
    let qh = _mm512_madd52hi_epu64(z, a, ws);
    let lo = _mm512_madd52lo_epu64(z, a, w);
    let lq = _mm512_madd52lo_epu64(z, qh, k.q);
    _mm512_and_si512(_mm512_sub_epi64(lo, lq), k.m52)
}

/// GS butterfly on 16 lanes (two 8-lane vectors per coefficient), values in `[0, 2q)`.
#[inline(always)]
unsafe fn gs(x: *mut u64, y: *mut u64, z: __m512i, zs: __m512i, k: K) {
    for h in 0..2 {
        let (u, w) = (ld(x.add(8 * h)), ld(y.add(8 * h)));
        st(x.add(8 * h), red(_mm512_add_epi64(u, w), k.q2));
        st(y.add(8 * h), mulq(_mm512_sub_epi64(_mm512_add_epi64(u, k.q2), w), z, zs, k));
    }
}

/// Hint load step: `h16` (two primes, `[i][c][l]` u32 residues `< p_i`) -> `H` (`[c][l]` u64 `< 2q`).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub unsafe fn hint_to_q1(qc: &Q1, h16: &[u32], n: usize, out: &mut [u64]) {
    assert!(h16.len() >= 2 * n * L && out.len() >= n * L);
    let k = K { q: _mm512_set1_epi64(qc.q as i64), q2: _mm512_set1_epi64(2 * qc.q as i64), m52: _mm512_set1_epi64((1 << 52) - 1) };
    let (a0, a1) = (_mm512_set1_epi64(qc.a[0] as i64), _mm512_set1_epi64(qc.a[1] as i64));
    let (s0, s1) = (_mm512_set1_epi64(qc.as_[0] as i64), _mm512_set1_epi64(qc.as_[1] as i64));
    let (hp, op) = (h16.as_ptr(), out.as_mut_ptr());
    for i in 0..n * L / 8 {
        let h0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256(hp.add(8 * i) as *const _));
        let h1 = _mm512_cvtepu32_epi64(_mm256_loadu_si256(hp.add(n * L + 8 * i) as *const _));
        let v = _mm512_add_epi64(mulq(h0, a0, s0, k), mulq(h1, a1, s1, k));
        st(op.add(8 * i), red(v, k.q2));
    }
}

const SUB: usize = 256;
const MAXPTS: usize = 32;

/// `a = INTT_q1(h * t)` without `N^-1` on 16 interleaved lanes (`a`, `h`: `n x 16` u64, `h < 2^52`; `t`,
/// `ts`: the secret operand and its Shoup companions). Output in `[0, 2q)`. Same stage order and L1
/// blocking as [`crate::vntt::inv_l_mul`].
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub unsafe fn inv_mul_q1(qc: &Q1, a: &mut [u64], h: &[u64], t: &[u64], ts: &[u64]) {
    let n = qc.izq.len();
    assert!(a.len() >= n * L && h.len() >= n * L && t.len() >= n && ts.len() >= n);
    let k = K { q: _mm512_set1_epi64(qc.q as i64), q2: _mm512_set1_epi64(2 * qc.q as i64), m52: _mm512_set1_epi64((1 << 52) - 1) };
    let (ap, hp) = (a.as_mut_ptr(), h.as_ptr());
    let sub = SUB.min(n);
    let tw = |i: usize| (_mm512_set1_epi64(qc.izq[i] as i64), _mm512_set1_epi64(qc.izs[i] as i64));
    for sb in 0..n / sub {
        let base = sb * sub;
        for c in base..base + sub {
            let (tv, tsv) = (_mm512_set1_epi64(t[c] as i64), _mm512_set1_epi64(ts[c] as i64));
            for hh in 0..2 {
                st(ap.add(c * L + 8 * hh), mulq(ld(hp.add(c * L + 8 * hh)), tv, tsv, k));
            }
        }
        let mut len = 1usize;
        while len < sub {
            let nb = n / (2 * len);
            let b0 = base / (2 * len);
            for bb in 0..sub / (2 * len) {
                let (z, zs) = tw(nb + b0 + bb);
                let x = ap.add((base + 2 * bb * len) * L);
                for i in 0..len {
                    gs(x.add(i * L), x.add((i + len) * L), z, zs, k);
                }
            }
            len *= 2;
        }
    }
    if sub < n {
        let pts = n / sub;
        assert!(pts <= MAXPTS);
        let mut buf = [0u64; MAXPTS * L];
        let bp = buf.as_mut_ptr();
        for c0 in 0..sub {
            for kk in 0..pts {
                for hh in 0..2 {
                    st(bp.add(kk * L + 8 * hh), ld(ap.add((c0 + sub * kk) * L + 8 * hh)));
                }
            }
            let mut lk = 1usize;
            while lk < pts {
                let nb = pts / (2 * lk);
                for bk in 0..nb {
                    let (z, zs) = tw(nb + bk);
                    let x = bp.add(2 * bk * lk * L);
                    for i in 0..lk {
                        gs(x.add(i * L), x.add((i + lk) * L), z, zs, k);
                    }
                }
                lk *= 2;
            }
            for kk in 0..pts {
                for hh in 0..2 {
                    st(ap.add((c0 + sub * kk) * L + 8 * hh), ld(bp.add(kk * L + 8 * hh)));
                }
            }
        }
        for kk in 0..MAXPTS * L / 8 {
            st(bp.add(8 * kk), _mm512_setzero_si512());
        }
    }
}

/// The decryption combine for one slot `u` from the `q1` NTT output (`s`: the 32 coefficients
/// `32 u .. 32 u + 31`, 16 lanes each, `< 2q`): `w` reduced to `[0, q1)` is exactly the two-prime path's CRT
/// value, then the same f64 operations as [`crate::simd::dec_combine`].
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
pub unsafe fn dec_combine_q1(qc: &Q1, kc: &DecConsts, s: *const u64, lanes: usize, c0: C0p, idx0: &[usize; L],
                             outp: &[*mut u32; L]) {
    let q = _mm512_set1_epi64(qc.q as i64);
    let mut w = [[[_mm512_setzero_si512(); 8]; 4]; 2]; // [lane half][j block][j in block]
    for j in 0..32 {
        for h in 0..2 {
            w[h][j / 8][j % 8] = red(ld(s.add(j * L + 8 * h)), q);
        }
    }
    for wh in w.iter_mut() {
        for blk in wh.iter_mut() {
            transpose8(blk); // blk[i] = lane 8 h + i, rows 8 jb .. 8 jb + 7
        }
    }
    let wsc = _mm512_set1_pd(kc.wscale);
    let csc = _mm512_set1_pd(kc.c0scale);
    let half = _mm512_set1_pd(0.5);
    let sh = _mm512_set_epi64(28, 24, 20, 16, 12, 8, 4, 0);
    let m4 = _mm512_set1_epi64(15);
    for l in 0..lanes {
        for jb in 0..4 {
            let wf = _mm512_mul_pd(_mm512_cvtepu64_pd(w[l / 8][jb][l % 8]), wsc);
            let idx = idx0[l] + 8 * jb;
            let v = match c0 {
                C0p::Words(p) => _mm512_loadu_si512(p.add(idx) as *const _),
                C0p::Packed36 { lo, hi } => {
                    let lo8 = _mm256_loadu_si256(lo.add(4 * idx) as *const _);
                    let wd = (hi.add(idx / 2) as *const u32).read_unaligned();
                    let nib = _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64(wd as i64), sh), m4);
                    _mm512_or_si512(_mm512_cvtepu32_epi64(lo8), _mm512_slli_epi64(nib, 32))
                }
            };
            let x = _mm512_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), csc), wf);
            let r = _mm512_cvttpd_epu64(_mm512_add_pd(x, half));
            _mm256_storeu_si256(outp[l].add(8 * jb) as *mut _, _mm512_cvtepi64_epi32(r));
        }
    }
    w = [[[_mm512_setzero_si512(); 8]; 4]; 2];
    std::hint::black_box(&w);
}
