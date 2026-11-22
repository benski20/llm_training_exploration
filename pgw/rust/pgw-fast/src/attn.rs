//! Row-deterministic attention over the KV cache (`float_ops = "native"`, opt-in; see [`crate::layer`]).
//!
//! Each output row depends only on that row's query and the cache prefix it sees: the same bits
//! whatever the thread count, the instruction set (baseline, AVX2, AVX-512) or the number of rows in
//! the pass (so a speculative verify row equals plain decoding by construction). Per query head `h`
//! (KV head `h / groups`) and row at position `p`, positions `j = 0..=p` are split into **blocks**
//! `[b B, (b + 1) B)` of a fixed size `B` that depends on nothing but the formulation:
//!
//! * `s_j = dot(q, k_j) * scale`: the dot in float32 (16 lane sums over `d mod 16`, then a fixed tree),
//!   widened to float64 and scaled;
//! * `m = max_j s_j` (exact: the order does not matter), `e_j = exp(s_j - m)` in float64
//!   ([`crate::layer::exp_f64`]) rounded to float32;
//! * per block: `den_b = sum_{j in b} e_j` in float64 and `acc_bd = sum_{j in b} e_j * v_jd` in float32,
//!   each from 0 in `j` order;
//! * merge in block order: `den = (((den_0 + den_1) + den_2) + ...)` (float64), `acc_d = (((acc_0d +
//!   acc_1d) + ...)` (float32), `o_d = (acc_d / den) as f32` (the quotient in float64).
//!
//! Two pinned formulations (the gateway and its plaintext reference must select the same one):
//!
//! * **v1** (`block = 0`, the original): one block of any length (`B = infinity`), i.e. plain `j`-order
//!   sums over the whole prefix. Parallel over rows and KV heads (each head's math is unchanged).
//! * **v2 "blocked"** (`block = B > 0`, `B = 256` in `pgw`): parallel over rows, KV heads and blocks.
//!   A row whose prefix fits in one block (`p < B`) has exactly the v1 bits.
//!
//! The KV cache may be stored as float32 or as bfloat16 (`*_bf16` functions; `uint16` bit patterns,
//! rounded to nearest-even from float32 like torch's `.to(torch.bfloat16)`): a bf16 element is widened to
//! float32 exactly, then the same math runs (a separate pinned reference: the cache rounding changes K/V).
//!
//! [`float_rope_attend`] also applies RoPE (torch's `apply_rope` bits: `x * cos + cat(x2, x1) *
//! cat(-sin1, sin2)`, two float32 products and a sum) to q and k of a `qkv` result and writes k and v
//! into the cache at the pass's positions before attending: one call per layer between the `qkv` and
//! `o` linears.

use crate::layer::{exp_f64, verr};
use crate::prepare::pool::Pool;
use numpy::{PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Register this module's functions in `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(float_rope_attend, m)?)?;
    m.add_function(wrap_pyfunction!(float_rope_attend_bf16, m)?)?;
    m.add_function(wrap_pyfunction!(float_attend, m)?)?;
    m.add_function(wrap_pyfunction!(attn_isa_level, m)?)?;
    m.add_function(wrap_pyfunction!(attn_set_isa_cap, m)?)?;
    m.add_function(wrap_pyfunction!(bf16_round, m)?)?;
    Ok(())
}

/// Block size of the v2 ("blocked") formulation used by `pgw` (positions per block).
pub const ATTN_BLOCK: usize = 256;

/// A KV cache element: float32, or bfloat16 as its `u16` bit pattern.
pub trait Kv: Copy + Send + Sync + 'static {
    /// The element is bfloat16 (`u16` bits).
    const BF16: bool;
    fn f(self) -> f32;
    fn from_f32(x: f32) -> Self;
}

impl Kv for f32 {
    const BF16: bool = false;
    #[inline(always)]
    fn f(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(x: f32) -> f32 {
        x
    }
}

impl Kv for u16 {
    const BF16: bool = true;
    #[inline(always)]
    fn f(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
    #[inline(always)]
    fn from_f32(x: f32) -> u16 {
        bf16_rne(x)
    }
}

/// float32 -> bfloat16 bits, round to nearest even (torch's `round_to_nearest_even`; NaN -> 0x7FC0).
#[inline(always)]
pub fn bf16_rne(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

// ---- instruction-set dispatch (same IEEE operations, same order: only the vector width changes) ----

/// Highest instruction-set level allowed (tests lower it): 2 = AVX-512, 1 = AVX2, 0 = baseline.
static ISA_CAP: AtomicU8 = AtomicU8::new(2);

fn isa_detected() -> u8 {
    static L: OnceLock<u8> = OnceLock::new();
    *L.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512dq") && std::is_x86_feature_detected!("avx512vl")
                && std::is_x86_feature_detected!("avx2")
            {
                return 2;
            }
            if std::is_x86_feature_detected!("avx2") {
                return 1;
            }
        }
        0
    })
}

#[inline]
fn isa_level() -> u8 {
    isa_detected().min(ISA_CAP.load(Ordering::Relaxed))
}

/// `dot(a, b)` in float32: 16 lane sums (`d mod 16`, in order), then a fixed pairwise tree.
#[inline(always)]
fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut l = [0f32; 16];
    let n = a.len().min(b.len());
    let n16 = n / 16 * 16;
    let (a16, b16) = (&a[..n16], &b[..n16]);
    for (ca, cb) in a16.chunks_exact(16).zip(b16.chunks_exact(16)) {
        for j in 0..16 {
            l[j] += ca[j] * cb[j];
        }
    }
    for (j, idx) in (n16..n).enumerate() {
        l[j] += a[idx] * b[idx];
    }
    let mut w = 16;
    while w > 1 {
        w /= 2;
        for j in 0..w {
            l[j] += l[j + w];
        }
    }
    l[0]
}

/// Shapes of one attention problem.
#[derive(Clone, Copy)]
pub struct Dims {
    pub nh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub scale: f64,
}

/// The original single-row v1 kernel, kept as the reference the blocked engine is tested against.
///
/// Per query head: `s_j = (dot32(q, k_j) as f64) * scale`; `m = max_j s_j`; `e_j = exp(s_j - m)` in
/// float64, rounded to float32; `den = sum_j e_j` in float64 (`j` order); `acc_d = sum_j e_j * v_jd` in
/// float32 (`j` order); `o_d = (acc_d / den) as f32` in float64.
pub fn attend_row_ref(q: &[f32], k: &[f32], v: &[f32], len: usize, dm: Dims, out: &mut [f32]) {
    let Dims { nh, kvh, hd, scale } = dm;
    let groups = nh / kvh;
    let stride = kvh * hd;
    let mut s = vec![0f64; len];
    for g in 0..kvh {
        for h in 0..groups {
            let hq = g * groups + h;
            let qh = &q[hq * hd..(hq + 1) * hd];
            for j in 0..len {
                s[j] = dot32(qh, &k[j * stride + g * hd..j * stride + (g + 1) * hd]) as f64 * scale;
            }
            let mut m = f64::NEG_INFINITY;
            for &x in s.iter() {
                m = if x > m { x } else { m };
            }
            let e: Vec<f32> = s.iter().map(|&x| exp_f64(x - m) as f32).collect();
            let mut d = 0f64;
            for &x in e.iter() {
                d += x as f64;
            }
            let mut acc = vec![0f32; hd];
            for j in 0..len {
                let vj = &v[j * stride + g * hd..j * stride + (g + 1) * hd];
                for (a, &vd) in acc.iter_mut().zip(vj) {
                    *a += e[j] * vd;
                }
            }
            for dd in 0..hd {
                out[hq * hd + dd] = (acc[dd] as f64 / d) as f32;
            }
        }
    }
}

/// A raw pointer shared by the units of one call; each unit writes a disjoint range.
#[derive(Clone, Copy)]
struct P<T>(*mut T);
// SAFETY: units write disjoint ranges, and the submitting call outlives every unit (Pool::run).
unsafe impl<T> Send for P<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for P<T> {}

impl<T> P<T> {
    /// `self + n` (a method, so closures capture the whole `P`).
    ///
    /// # Safety
    /// `n` within the allocation.
    #[inline(always)]
    unsafe fn at(self, n: usize) -> P<T> {
        // SAFETY: the caller's.
        P(unsafe { self.0.add(n) })
    }
}

thread_local! {
    /// Per-thread scratch: a K or V row widened to float32, and `e` of one unit.
    static UNIT: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    /// Per-calling-thread scratch of a call: scores, block maxima, row maxima, block dens, block accs.
    static CALL: RefCell<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

/// Phase A of a unit (row, KV head `g`, block `[j0, j1)`): `s` of the group's heads and their block
/// maxima. `s_t`: the row's scores `(nh, lmax)`; `bm_t`: the row's block maxima `(nh, nbmax)`.
#[inline(always)]
fn scores_body<T: Kv>(q_t: &[f32], k: &[T], j0: usize, j1: usize, g: usize, dm: Dims, s_t: P<f64>, lmax: usize,
                      bm_t: P<f64>, b: usize, nbmax: usize) {
    let Dims { nh, kvh, hd, scale } = dm;
    let groups = nh / kvh;
    let stride = kvh * hd;
    UNIT.with(|u| {
        let mut u = u.borrow_mut();
        let kr = &mut u.0;
        kr.resize(hd, 0.0);
        for j in j0..j1 {
            for (dst, &x) in kr.iter_mut().zip(&k[j * stride + g * hd..j * stride + (g + 1) * hd]) {
                *dst = x.f();
            }
            for h in 0..groups {
                let hq = g * groups + h;
                let sv = dot32(&q_t[hq * hd..(hq + 1) * hd], kr) as f64 * scale;
                // SAFETY: (hq, j) of this row is written by this unit only; in bounds (j < lmax).
                unsafe { *s_t.0.add(hq * lmax + j) = sv };
            }
        }
    });
    for h in 0..groups {
        let hq = g * groups + h;
        // SAFETY: the range was written above by this unit.
        let sl = unsafe { std::slice::from_raw_parts(s_t.0.add(hq * lmax + j0), j1 - j0) };
        let mut m = f64::NEG_INFINITY;
        for &x in sl {
            m = if x > m { x } else { m };
        }
        // SAFETY: (hq, b) is this unit's.
        unsafe { *bm_t.0.add(hq * nbmax + b) = m };
    }
}
/// Phase B of a unit: `e_j`, the block's `den_b` (float64) and `acc_b` (float32) of the group's heads.
/// `m_t`: the row's maxima `(nh,)`; `den_t (nh, nbmax)`, `acc_t (nh, nbmax, hd)`: the row's partials.
#[inline(always)]
fn values_body<T: Kv>(v: &[T], s_t: &[f64], m_t: &[f64], j0: usize, j1: usize, g: usize, dm: Dims, lmax: usize,
                      den_t: P<f64>, acc_t: P<f32>, b: usize, nbmax: usize) {
    let Dims { nh, kvh, hd, .. } = dm;
    let groups = nh / kvh;
    let stride = kvh * hd;
    let n = j1 - j0;
    UNIT.with(|u| {
        let mut u = u.borrow_mut();
        let (vr, e) = &mut *u;
        vr.resize(hd, 0.0);
        e.resize(groups * n, 0.0);
        for h in 0..groups {
            let hq = g * groups + h;
            let mh = m_t[hq];
            for (ej, &sj) in e[h * n..(h + 1) * n].iter_mut().zip(&s_t[hq * lmax + j0..hq * lmax + j1]) {
                *ej = exp_f64(sj - mh) as f32;
            }
        }
        // the group's dens interleaved over heads (independent chains), each in j order
        let mut den = [0f64; 16];
        let mut h0 = 0;
        while h0 < groups {
            let hn = (groups - h0).min(16);
            den[..hn].iter_mut().for_each(|x| *x = 0.0);
            for jj in 0..n {
                for h in 0..hn {
                    den[h] += e[(h0 + h) * n + jj] as f64;
                }
            }
            for h in 0..hn {
                // SAFETY: (hq, b) is this unit's.
                unsafe { *den_t.0.add((g * groups + h0 + h) * nbmax + b) = den[h] };
            }
            h0 += hn;
        }
        for h in 0..groups {
            let hq = g * groups + h;
            // SAFETY: (hq, b, :) is this unit's.
            let a = unsafe { std::slice::from_raw_parts_mut(acc_t.0.add((hq * nbmax + b) * hd), hd) };
            a.iter_mut().for_each(|x| *x = 0.0);
        }
        for jj in 0..n {
            let j = j0 + jj;
            for (dst, &x) in vr.iter_mut().zip(&v[j * stride + g * hd..j * stride + (g + 1) * hd]) {
                *dst = x.f();
            }
            for h in 0..groups {
                let hq = g * groups + h;
                let ej = e[h * n + jj];
                // SAFETY: as above.
                let a = unsafe { std::slice::from_raw_parts_mut(acc_t.0.add((hq * nbmax + b) * hd), hd) };
                for (ad, &vd) in a.iter_mut().zip(vr.iter()) {
                    *ad += ej * vd;
                }
            }
        }
    });
}
/// Phase A of a unit at the best allowed instruction set (see [`scores_body`]).
#[allow(clippy::too_many_arguments)]
fn scores_unit<T: Kv>(q_t: &[f32], k: &[T], j0: usize, j1: usize, g: usize, dm: Dims, s_t: P<f64>, lmax: usize,
                      bm_t: P<f64>, b: usize, nbmax: usize) {
    #[cfg(target_arch = "x86_64")]
    if dm.hd % 16 == 0 {
        match isa_level() {
            // SAFETY: the CPU has the features (isa_detected); the unit's ranges are in bounds.
            2 => return unsafe { simd::scores_512::<T>(q_t, k, j0, j1, g, dm, s_t, lmax, bm_t, b, nbmax) },
            // SAFETY: as above.
            1 => return unsafe { simd::scores_256::<T>(q_t, k, j0, j1, g, dm, s_t, lmax, bm_t, b, nbmax) },
            _ => {}
        }
    }
    scores_body::<T>(q_t, k, j0, j1, g, dm, s_t, lmax, bm_t, b, nbmax)
}

/// Phase B of a unit at the best allowed instruction set (see [`values_body`]).
#[allow(clippy::too_many_arguments)]
fn values_unit<T: Kv>(v: &[T], s_t: &[f64], m_t: &[f64], j0: usize, j1: usize, g: usize, dm: Dims, lmax: usize,
                      den_t: P<f64>, acc_t: P<f32>, b: usize, nbmax: usize) {
    #[cfg(target_arch = "x86_64")]
    if dm.hd % 16 == 0 {
        match isa_level() {
            // SAFETY: as in scores_unit.
            2 => return unsafe { simd::values_512::<T>(v, s_t, m_t, j0, j1, g, dm, lmax, den_t, acc_t, b, nbmax) },
            // SAFETY: as above.
            1 => return unsafe { simd::values_256::<T>(v, s_t, m_t, j0, j1, g, dm, lmax, den_t, acc_t, b, nbmax) },
            _ => {}
        }
    }
    values_body::<T>(v, s_t, m_t, j0, j1, g, dm, lmax, den_t, acc_t, b, nbmax)
}

/// AVX-512 and AVX2 kernels of the two phases: the scalar bodies' IEEE operations in the same order,
/// lane for lane (a 16-lane dot accumulator is one zmm or two ymm registers; the tree is the same
/// pairwise reduction; `exp` is [`exp_f64`] on 8 or 4 float64 lanes; mul and add stay separate, no FMA).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
mod simd {
    use super::{exp_f64, Dims, Kv, P};
    use crate::layer::{LN2_HI, LN2_LO, LOG2E};
    use std::arch::x86_64::*;

    const MAGIC: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52, as in exp_f64
    /// The Horner coefficients of exp_f64, highest first (the same constant expressions).
    const C: [f64; 13] = [1.0 / 479_001_600.0, 1.0 / 39_916_800.0, 1.0 / 3_628_800.0, 1.0 / 362_880.0,
                          1.0 / 40_320.0, 1.0 / 5_040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5,
                          1.0, 1.0];
    /// Positions per tile of phase A (K rows of a tile stay in L1 across the group's heads).
    const TILE: usize = 32;

    // ---------------- AVX-512 ----------------

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    unsafe fn ld16<T: Kv>(p: *const T) -> __m512 {
        if T::BF16 {
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i))))
        } else {
            _mm512_loadu_ps(p as *const f32)
        }
    }

    /// The fixed tree of 16 lanes after its first level (`v8[j] = l[j] + l[j + 8]`): w = 4, 2, 1.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn tree8(v8: __m256) -> f32 {
        let v4 = _mm_add_ps(_mm256_castps256_ps128(v8), _mm256_extractf128_ps::<1>(v8));
        let v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
        let v1 = _mm_add_ss(v2, _mm_shuffle_ps::<1>(v2, v2));
        _mm_cvtss_f32(v1)
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    unsafe fn exp8(x: __m512d) -> __m512d {
        let lo = _mm512_set1_pd(-745.0);
        let hi = _mm512_set1_pd(709.0);
        let xc = _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(x, lo), x, lo);
        let xc = _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(xc, hi), xc, hi);
        let mg = _mm512_set1_pd(MAGIC);
        let t = _mm512_add_pd(_mm512_mul_pd(xc, _mm512_set1_pd(LOG2E)), mg);
        let k = _mm512_sub_pd(t, mg);
        let r = _mm512_sub_pd(_mm512_sub_pd(xc, _mm512_mul_pd(k, _mm512_set1_pd(LN2_HI))),
                              _mm512_mul_pd(k, _mm512_set1_pd(LN2_LO)));
        let mut p = _mm512_set1_pd(C[0]);
        for &c in &C[1..] {
            p = _mm512_add_pd(_mm512_mul_pd(p, r), _mm512_set1_pd(c));
        }
        let k1 = _mm512_sub_pd(_mm512_add_pd(_mm512_mul_pd(k, _mm512_set1_pd(0.5)), mg), mg);
        let k2 = _mm512_sub_pd(k, k1);
        let bias = _mm512_set1_pd(MAGIC + 1023.0);
        let f1 = _mm512_castsi512_pd(_mm512_slli_epi64::<52>(_mm512_castpd_si512(_mm512_add_pd(k1, bias))));
        let f2 = _mm512_castsi512_pd(_mm512_slli_epi64::<52>(_mm512_castpd_si512(_mm512_add_pd(k2, bias))));
        _mm512_mul_pd(_mm512_mul_pd(p, f1), f2)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    pub(super) unsafe fn scores_512<T: Kv>(q_t: &[f32], k: &[T], j0: usize, j1: usize, g: usize, dm: Dims,
                                           s_t: P<f64>, lmax: usize, bm_t: P<f64>, b: usize, nbmax: usize) {
        let Dims { nh, kvh, hd, scale } = dm;
        let groups = nh / kvh;
        let stride = kvh * hd;
        let nc = hd / 16;
        assert!(j1 * stride <= k.len() && (g + 1) * hd <= stride && q_t.len() >= nh * hd);
        let (qp, kp) = (q_t.as_ptr(), k.as_ptr());
        let mut tj = j0;
        while tj < j1 {
            let te = (tj + TILE).min(j1);
            for h in 0..groups {
                let hq = g * groups + h;
                let qh = qp.add(hq * hd);
                let srow = s_t.0.add(hq * lmax);
                for j in tj..te {
                    let kr = kp.add(j * stride + g * hd);
                    let mut acc = _mm512_setzero_ps();
                    for c in 0..nc {
                        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(qh.add(16 * c)), ld16(kr.add(16 * c))));
                    }
                    let v8 = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps::<1>(acc));
                    *srow.add(j) = tree8(v8) as f64 * scale;
                }
            }
            tj = te;
        }
        for h in 0..groups {
            let hq = g * groups + h;
            let srow = s_t.0.add(hq * lmax);
            let mut m = f64::NEG_INFINITY;
            for j in j0..j1 {
                let x = *srow.add(j);
                m = if x > m { x } else { m };
            }
            *bm_t.0.add(hq * nbmax + b) = m;
        }
    }

    /// `acc[0..N] = sum_j e_j * v_j[chunks]` (each from 0, `j` order); with `den`, also `sum_j e_j` (float64).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    unsafe fn pv_512<T: Kv, const N: usize>(e: *const f32, n: usize, vp: *const T, stride: usize, out: *mut f32,
                                           den: Option<&mut f64>) {
        let mut acc = [_mm512_setzero_ps(); N];
        let mut d = 0f64;
        let want = den.is_some();
        for jj in 0..n {
            let ej = *e.add(jj);
            if want {
                d += ej as f64;
            }
            let ev = _mm512_set1_ps(ej);
            let vr = vp.add(jj * stride);
            for (i, a) in acc.iter_mut().enumerate() {
                *a = _mm512_add_ps(*a, _mm512_mul_ps(ev, ld16(vr.add(16 * i))));
            }
        }
        for (i, a) in acc.iter().enumerate() {
            _mm512_storeu_ps(out.add(16 * i), *a);
        }
        if let Some(x) = den {
            *x = d;
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    pub(super) unsafe fn values_512<T: Kv>(v: &[T], s_t: &[f64], m_t: &[f64], j0: usize, j1: usize, g: usize,
                                           dm: Dims, lmax: usize, den_t: P<f64>, acc_t: P<f32>, b: usize,
                                           nbmax: usize) {
        let Dims { nh, kvh, hd, .. } = dm;
        let groups = nh / kvh;
        let stride = kvh * hd;
        let n = j1 - j0;
        assert!(j1 * stride <= v.len() && (g + 1) * hd <= stride && s_t.len() >= nh * lmax && j1 <= lmax);
        super::UNIT.with(|u| {
            let mut u = u.borrow_mut();
            let e = &mut u.1;
            e.resize(n, 0.0);
            let ep = e.as_mut_ptr();
            for h in 0..groups {
                let hq = g * groups + h;
                let sp = s_t.as_ptr().add(hq * lmax + j0);
                let mv = _mm512_set1_pd(m_t[hq]);
                let n8 = n / 8 * 8;
                let mut jj = 0;
                while jj < n8 {
                    let x = _mm512_sub_pd(_mm512_loadu_pd(sp.add(jj)), mv);
                    _mm256_storeu_ps(ep.add(jj), _mm512_cvtpd_ps(exp8(x)));
                    jj += 8;
                }
                for jj in n8..n {
                    *ep.add(jj) = exp_f64(*sp.add(jj) - m_t[hq]) as f32;
                }
                let vp = v.as_ptr().add(j0 * stride + g * hd);
                let out = acc_t.0.add((hq * nbmax + b) * hd);
                let mut den = 0f64;
                let nc = hd / 16;
                let mut c = 0;
                while c < nc {
                    let first = if c == 0 { Some(&mut den) } else { None };
                    let (vo, oo) = (vp.add(16 * c), out.add(16 * c));
                    let step = match nc - c {
                        r if r >= 8 => { pv_512::<T, 8>(ep, n, vo, stride, oo, first); 8 }
                        r if r >= 4 => { pv_512::<T, 4>(ep, n, vo, stride, oo, first); 4 }
                        r if r >= 2 => { pv_512::<T, 2>(ep, n, vo, stride, oo, first); 2 }
                        _ => { pv_512::<T, 1>(ep, n, vo, stride, oo, first); 1 }
                    };
                    c += step;
                }
                *den_t.0.add(hq * nbmax + b) = den;
            }
        });
    }

    // ---------------- AVX2 ----------------

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn ld8<T: Kv>(p: *const T) -> __m256 {
        if T::BF16 {
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i))))
        } else {
            _mm256_loadu_ps(p as *const f32)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn exp4(x: __m256d) -> __m256d {
        let lo = _mm256_set1_pd(-745.0);
        let hi = _mm256_set1_pd(709.0);
        let xc = _mm256_blendv_pd(x, lo, _mm256_cmp_pd::<_CMP_LT_OQ>(x, lo));
        let xc = _mm256_blendv_pd(xc, hi, _mm256_cmp_pd::<_CMP_GT_OQ>(xc, hi));
        let mg = _mm256_set1_pd(MAGIC);
        let t = _mm256_add_pd(_mm256_mul_pd(xc, _mm256_set1_pd(LOG2E)), mg);
        let k = _mm256_sub_pd(t, mg);
        let r = _mm256_sub_pd(_mm256_sub_pd(xc, _mm256_mul_pd(k, _mm256_set1_pd(LN2_HI))),
                              _mm256_mul_pd(k, _mm256_set1_pd(LN2_LO)));
        let mut p = _mm256_set1_pd(C[0]);
        for &c in &C[1..] {
            p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(c));
        }
        let k1 = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(k, _mm256_set1_pd(0.5)), mg), mg);
        let k2 = _mm256_sub_pd(k, k1);
        let bias = _mm256_set1_pd(MAGIC + 1023.0);
        let f1 = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_castpd_si256(_mm256_add_pd(k1, bias))));
        let f2 = _mm256_castsi256_pd(_mm256_slli_epi64::<52>(_mm256_castpd_si256(_mm256_add_pd(k2, bias))));
        _mm256_mul_pd(_mm256_mul_pd(p, f1), f2)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scores_256<T: Kv>(q_t: &[f32], k: &[T], j0: usize, j1: usize, g: usize, dm: Dims,
                                           s_t: P<f64>, lmax: usize, bm_t: P<f64>, b: usize, nbmax: usize) {
        let Dims { nh, kvh, hd, scale } = dm;
        let groups = nh / kvh;
        let stride = kvh * hd;
        let nc = hd / 16;
        assert!(j1 * stride <= k.len() && (g + 1) * hd <= stride && q_t.len() >= nh * hd);
        let (qp, kp) = (q_t.as_ptr(), k.as_ptr());
        let mut tj = j0;
        while tj < j1 {
            let te = (tj + TILE).min(j1);
            for h in 0..groups {
                let hq = g * groups + h;
                let qh = qp.add(hq * hd);
                let srow = s_t.0.add(hq * lmax);
                for j in tj..te {
                    let kr = kp.add(j * stride + g * hd);
                    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                    for c in 0..nc {
                        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(qh.add(16 * c)), ld8(kr.add(16 * c))));
                        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(qh.add(16 * c + 8)),
                                                             ld8(kr.add(16 * c + 8))));
                    }
                    *srow.add(j) = tree8(_mm256_add_ps(a0, a1)) as f64 * scale;
                }
            }
            tj = te;
        }
        for h in 0..groups {
            let hq = g * groups + h;
            let srow = s_t.0.add(hq * lmax);
            let mut m = f64::NEG_INFINITY;
            for j in j0..j1 {
                let x = *srow.add(j);
                m = if x > m { x } else { m };
            }
            *bm_t.0.add(hq * nbmax + b) = m;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pv_256<T: Kv, const N: usize>(e: *const f32, n: usize, vp: *const T, stride: usize, out: *mut f32,
                                           den: Option<&mut f64>) {
        let mut acc = [_mm256_setzero_ps(); N];
        let mut d = 0f64;
        let want = den.is_some();
        for jj in 0..n {
            let ej = *e.add(jj);
            if want {
                d += ej as f64;
            }
            let ev = _mm256_set1_ps(ej);
            let vr = vp.add(jj * stride);
            for (i, a) in acc.iter_mut().enumerate() {
                *a = _mm256_add_ps(*a, _mm256_mul_ps(ev, ld8(vr.add(8 * i))));
            }
        }
        for (i, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(out.add(8 * i), *a);
        }
        if let Some(x) = den {
            *x = d;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn values_256<T: Kv>(v: &[T], s_t: &[f64], m_t: &[f64], j0: usize, j1: usize, g: usize,
                                           dm: Dims, lmax: usize, den_t: P<f64>, acc_t: P<f32>, b: usize,
                                           nbmax: usize) {
        let Dims { nh, kvh, hd, .. } = dm;
        let groups = nh / kvh;
        let stride = kvh * hd;
        let n = j1 - j0;
        assert!(j1 * stride <= v.len() && (g + 1) * hd <= stride && s_t.len() >= nh * lmax && j1 <= lmax);
        super::UNIT.with(|u| {
            let mut u = u.borrow_mut();
            let e = &mut u.1;
            e.resize(n, 0.0);
            let ep = e.as_mut_ptr();
            for h in 0..groups {
                let hq = g * groups + h;
                let sp = s_t.as_ptr().add(hq * lmax + j0);
                let mv = _mm256_set1_pd(m_t[hq]);
                let n4 = n / 4 * 4;
                let mut jj = 0;
                while jj < n4 {
                    let x = _mm256_sub_pd(_mm256_loadu_pd(sp.add(jj)), mv);
                    _mm_storeu_ps(ep.add(jj), _mm256_cvtpd_ps(exp4(x)));
                    jj += 4;
                }
                for jj in n4..n {
                    *ep.add(jj) = exp_f64(*sp.add(jj) - m_t[hq]) as f32;
                }
                let vp = v.as_ptr().add(j0 * stride + g * hd);
                let out = acc_t.0.add((hq * nbmax + b) * hd);
                let mut den = 0f64;
                let nc = hd / 8;
                let mut c = 0;
                while c < nc {
                    let first = if c == 0 { Some(&mut den) } else { None };
                    let (vo, oo) = (vp.add(8 * c), out.add(8 * c));
                    let step = match nc - c {
                        r if r >= 8 => { pv_256::<T, 8>(ep, n, vo, stride, oo, first); 8 }
                        r if r >= 4 => { pv_256::<T, 4>(ep, n, vo, stride, oo, first); 4 }
                        r if r >= 2 => { pv_256::<T, 2>(ep, n, vo, stride, oo, first); 2 }
                        _ => { pv_256::<T, 1>(ep, n, vo, stride, oo, first); 1 }
                    };
                    c += step;
                }
                *den_t.0.add(hq * nbmax + b) = den;
            }
        });
    }
}

/// The persistent pool for `threads` threads (the caller plus `threads - 1` workers), started on first use.
fn pool_for(threads: usize) -> Option<&'static Pool> {
    if threads <= 1 {
        return None;
    }
    static POOLS: OnceLock<Mutex<Vec<(usize, &'static Pool)>>> = OnceLock::new();
    let mut ps = POOLS.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, p)) = ps.iter().find(|(n, _)| *n == threads) {
        return Some(p);
    }
    let p: &'static Pool = Box::leak(Box::new(Pool::new(threads - 1)));
    ps.push((threads, p));
    Some(p)
}

fn run_units(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize) + Sync)) {
    match pool {
        Some(p) if n > 1 => {
            assert!(p.run(n, f), "attention unit panicked");
        }
        _ => (0..n).for_each(f),
    }
}

/// Rows `t` of `q (rows, nh * hd)` at positions `start + t` attend to `k`, `v` positions `[0, start + t]`
/// (`(len, kvh, hd)`, row stride `kvh * hd`), into `out (rows, nh * hd)`. `block`: 0 = v1, else the v2
/// block size. Units (row, KV head, block) run on `threads` threads (0 or 1: the calling thread); every
/// row's bits are the same for any `threads` and any `rows`.
#[allow(clippy::too_many_arguments)]
pub fn attend_rows<T: Kv>(q: &[f32], k: &[T], v: &[T], rows: usize, start: usize, dm: Dims, out: &mut [f32],
                          threads: usize, block: usize) {
    if rows == 0 {
        return;
    }
    let Dims { nh, kvh, hd, .. } = dm;
    let groups = nh / kvh;
    let w = nh * hd;
    let lmax = start + rows;
    let bs = if block == 0 { lmax } else { block };
    let nb = |t: usize| (start + t + 1).div_ceil(bs);
    let nbmax = nb(rows - 1);
    let pool = pool_for(threads);
    CALL.with(|c| {
        let mut c = c.borrow_mut();
        let (s, bm, mx, pden, pacc) = &mut *c;
        s.resize(rows * nh * lmax, 0.0);
        bm.resize(rows * nh * nbmax, 0.0);
        mx.resize(rows * nh, 0.0);
        pden.resize(rows * nh * nbmax, 0.0);
        pacc.resize(rows * nh * nbmax * hd, 0.0);
        let (sp, bmp) = (P(s.as_mut_ptr()), P(bm.as_mut_ptr()));
        let units = rows * kvh * nbmax;
        let unit = |i: usize| {
            let (t, g, b) = (i / (kvh * nbmax), (i / nbmax) % kvh, i % nbmax);
            let len = start + t + 1;
            let j0 = b * bs;
            if j0 >= len {
                return;
            }
            let j1 = (j0 + bs).min(len);
            // SAFETY: row t's regions (in bounds: t < rows).
            let (s_t, bm_t) = unsafe { (sp.at(t * nh * lmax), bmp.at(t * nh * nbmax)) };
            scores_unit::<T>(&q[t * w..(t + 1) * w], k, j0, j1, g, dm, s_t, lmax, bm_t, b, nbmax);
        };
        run_units(pool, units, &unit);
        for t in 0..rows {
            for hq in 0..nh {
                let mut m = f64::NEG_INFINITY;
                for &x in &bm[(t * nh + hq) * nbmax..(t * nh + hq) * nbmax + nb(t)] {
                    m = if x > m { x } else { m };
                }
                mx[t * nh + hq] = m;
            }
        }
        let (sr, mr): (&[f64], &[f64]) = (s, mx);
        let (dp, ap) = (P(pden.as_mut_ptr()), P(pacc.as_mut_ptr()));
        let unit = |i: usize| {
            let (t, g, b) = (i / (kvh * nbmax), (i / nbmax) % kvh, i % nbmax);
            let len = start + t + 1;
            let j0 = b * bs;
            if j0 >= len {
                return;
            }
            let j1 = (j0 + bs).min(len);
            // SAFETY: row t's regions.
            let (den_t, acc_t) = unsafe { (dp.at(t * nh * nbmax), ap.at(t * nh * nbmax * hd)) };
            values_unit::<T>(v, &sr[t * nh * lmax..(t + 1) * nh * lmax], &mr[t * nh..(t + 1) * nh], j0, j1, g, dm,
                             lmax, den_t, acc_t, b, nbmax);
        };
        run_units(pool, units, &unit);
        // merge in block order
        for t in 0..rows {
            let nbt = nb(t);
            for hq in 0..nh {
                let base = (t * nh + hq) * nbmax;
                let mut den = pden[base];
                for b in 1..nbt {
                    den += pden[base + b];
                }
                let o = &mut out[t * w + hq * hd..t * w + (hq + 1) * hd];
                o.copy_from_slice(&pacc[base * hd..(base + 1) * hd]);
                for b in 1..nbt {
                    for (od, &a) in o.iter_mut().zip(&pacc[(base + b) * hd..(base + b + 1) * hd]) {
                        *od += a;
                    }
                }
                for od in o.iter_mut() {
                    *od = (*od as f64 / den) as f32;
                }
            }
        }
        let _ = groups;
    });
}

/// Threads for a pass: one for decode-sized work, more for long prompts (bits do not depend on it).
fn auto_threads(rows: usize, start: usize, threads: usize) -> usize {
    if threads > 0 {
        return threads;
    }
    let work = rows * (start + rows);
    if rows < 4 || work < 1 << 14 {
        1
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8)
    }
}

/// RoPE on one head vector into `out`: `x * cos + cat(x2, x1) * cat(-sin1, sin2)` (float32).
#[inline]
fn rope_head(x: &[f32], cos: &[f32], sin: &[f32], out: &mut [f32]) {
    let h = x.len() / 2;
    for d in 0..h {
        out[d] = x[d] * cos[d] + x[d + h] * -sin[d];
    }
    for d in h..2 * h {
        out[d] = x[d] * cos[d] + x[d - h] * sin[d];
    }
}

/// The body of [`float_rope_attend`] for a cache of `T` (see there).
#[allow(clippy::too_many_arguments)]
pub fn rope_attend<T: Kv>(ys: &[f32], cs: &[f32], ss: &[f32], kl: &mut [T], vl: &mut [T], rows: usize, start: usize,
                          dm: Dims, os: &mut [f32], threads: usize, block: usize) {
    let Dims { nh, kvh, hd, .. } = dm;
    let ycols = (nh + 2 * kvh) * hd;
    let stride = kvh * hd;
    let mut q = vec![0f32; rows * nh * hd];
    let mut kt = vec![0f32; hd];
    for t in 0..rows {
        let yt = &ys[t * ycols..(t + 1) * ycols];
        let (c, s) = (&cs[t * hd..(t + 1) * hd], &ss[t * hd..(t + 1) * hd]);
        for h in 0..nh {
            rope_head(&yt[h * hd..(h + 1) * hd], c, s, &mut q[(t * nh + h) * hd..(t * nh + h + 1) * hd]);
        }
        let p = start + t;
        for g in 0..kvh {
            rope_head(&yt[(nh + g) * hd..(nh + g + 1) * hd], c, s, &mut kt);
            for (dst, &x) in kl[p * stride + g * hd..p * stride + (g + 1) * hd].iter_mut().zip(&kt) {
                *dst = T::from_f32(x);
            }
            let vs = &yt[(nh + kvh + g) * hd..(nh + kvh + g + 1) * hd];
            for (dst, &x) in vl[p * stride + g * hd..p * stride + (g + 1) * hd].iter_mut().zip(vs) {
                *dst = T::from_f32(x);
            }
        }
    }
    let len = start + rows;
    attend_rows(&q, &kl[..len * stride], &vl[..len * stride], rows, start, dm, os, threads, block);
}

#[allow(clippy::too_many_arguments)]
fn rope_attend_py<'py, T: Kv + numpy::Element>(
    py: Python<'py>, name: &str, y: &Bound<'py, PyArray2<f32>>, cos: &Bound<'py, PyArray2<f32>>,
    sin: &Bound<'py, PyArray2<f32>>, kc: &Bound<'py, PyArray4<T>>, vc: &Bound<'py, PyArray4<T>>, layer: usize,
    start: usize, nh: usize, scale: f64, out: &Bound<'py, PyArray2<f32>>, threads: usize, block: usize,
) -> PyResult<()> {
    let ks = kc.shape().to_vec();
    let (layers, cap, kvh, hd) = (ks[0], ks[1], ks[2], ks[3]);
    let rows = y.shape()[0];
    let ycols = y.shape()[1];
    if vc.shape() != ks.as_slice() || layer >= layers || kvh == 0 || nh % kvh != 0 || hd % 2 != 0
        || ycols != (nh + 2 * kvh) * hd || cos.shape() != [rows, hd] || sin.shape() != [rows, hd]
        || out.shape() != [rows, nh * hd] || start + rows > cap
    {
        return Err(verr(format!("{name}: shapes")));
    }
    for (a, n) in [(y.is_c_contiguous(), "y"), (cos.is_c_contiguous(), "cos"), (sin.is_c_contiguous(), "sin"),
                   (kc.is_c_contiguous(), "kc"), (vc.is_c_contiguous(), "vc"), (out.is_c_contiguous(), "out")] {
        if !a {
            return Err(verr(format!("{name}: {n} must be C-contiguous")));
        }
    }
    let yr = y.try_readonly().map_err(|e| verr(format!("y: {e}")))?;
    let cr = cos.try_readonly().map_err(|e| verr(format!("cos: {e}")))?;
    let sr = sin.try_readonly().map_err(|e| verr(format!("sin: {e}")))?;
    let mut kw = kc.try_readwrite().map_err(|e| verr(format!("kc: {e}")))?;
    let mut vw = vc.try_readwrite().map_err(|e| verr(format!("vc: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (ys, cs, ss) = (yr.as_slice()?, cr.as_slice()?, sr.as_slice()?);
    let (kall, vall, os) = (kw.as_slice_mut()?, vw.as_slice_mut()?, ow.as_slice_mut()?);
    let dm = Dims { nh, kvh, hd, scale };
    let threads = auto_threads(rows, start, threads);
    py.allow_threads(|| {
        let lsz = cap * kvh * hd;
        let kl = &mut kall[layer * lsz..(layer + 1) * lsz];
        let vl = &mut vall[layer * lsz..(layer + 1) * lsz];
        rope_attend(ys, cs, ss, kl, vl, rows, start, dm, os, threads, block);
    });
    Ok(())
}

/// One layer's attention block of a pass: RoPE on q and k of `y (rows, (nh + 2 kvh) hd)` (columns
/// `[q | k | v]`), k and v written into `kc`, `vc` (`(layers, cap, kvh, hd)` float32) at
/// `[layer, start + t]`, then row-deterministic causal attention into `out (rows, nh * hd)`.
/// `threads`: 0 = automatic; `block`: 0 = the v1 formulation, `B > 0` = v2 with blocks of `B` positions.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (y, cos, sin, kc, vc, layer, start, nh, scale, out, threads = 0, block = 0))]
fn float_rope_attend<'py>(
    py: Python<'py>, y: &Bound<'py, PyArray2<f32>>, cos: &Bound<'py, PyArray2<f32>>, sin: &Bound<'py, PyArray2<f32>>,
    kc: &Bound<'py, PyArray4<f32>>, vc: &Bound<'py, PyArray4<f32>>, layer: usize, start: usize, nh: usize,
    scale: f64, out: &Bound<'py, PyArray2<f32>>, threads: usize, block: usize,
) -> PyResult<()> {
    rope_attend_py(py, "float_rope_attend", y, cos, sin, kc, vc, layer, start, nh, scale, out, threads, block)
}

/// [`float_rope_attend`] with a bfloat16 cache: `kc`, `vc` are `uint16` bit patterns; the new k and v are
/// rounded to nearest-even (torch's `.to(torch.bfloat16)` bits) when written.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (y, cos, sin, kc, vc, layer, start, nh, scale, out, threads = 0, block = 0))]
fn float_rope_attend_bf16<'py>(
    py: Python<'py>, y: &Bound<'py, PyArray2<f32>>, cos: &Bound<'py, PyArray2<f32>>, sin: &Bound<'py, PyArray2<f32>>,
    kc: &Bound<'py, PyArray4<u16>>, vc: &Bound<'py, PyArray4<u16>>, layer: usize, start: usize, nh: usize,
    scale: f64, out: &Bound<'py, PyArray2<f32>>, threads: usize, block: usize,
) -> PyResult<()> {
    rope_attend_py(py, "float_rope_attend_bf16", y, cos, sin, kc, vc, layer, start, nh, scale, out, threads, block)
}

/// Row-deterministic causal attention of `q (rows, nh, hd)` over `k`, `v` `(len, kvh, hd)`; row `t` sees
/// positions `[0, start + t]`; `out (rows, nh * hd)`. `threads`, `block`: as [`float_rope_attend`].
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (q, k, v, start, scale, out, threads = 0, block = 0))]
fn float_attend<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyArray3<f32>>,
    k: &Bound<'py, PyArray3<f32>>,
    v: &Bound<'py, PyArray3<f32>>,
    start: usize,
    scale: f64,
    out: &Bound<'py, PyArray2<f32>>,
    threads: usize,
    block: usize,
) -> PyResult<()> {
    let (rows, nh, hd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (len, kvh) = (k.shape()[0], k.shape()[1]);
    if v.shape() != k.shape() || k.shape()[2] != hd || kvh == 0 || nh % kvh != 0 || start + rows > len
        || out.shape() != [rows, nh * hd]
    {
        return Err(verr("float_attend: shapes"));
    }
    if !(q.is_c_contiguous() && k.is_c_contiguous() && v.is_c_contiguous() && out.is_c_contiguous()) {
        return Err(verr("float_attend: arrays must be C-contiguous float32"));
    }
    let qr = q.try_readonly().map_err(|e| verr(format!("q: {e}")))?;
    let kr = k.try_readonly().map_err(|e| verr(format!("k: {e}")))?;
    let vr = v.try_readonly().map_err(|e| verr(format!("v: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (qs, ks, vs) = (qr.as_slice()?, kr.as_slice()?, vr.as_slice()?);
    let os = ow.as_slice_mut()?;
    let dm = Dims { nh, kvh, hd, scale };
    let threads = auto_threads(rows, start, threads);
    let st = kvh * hd;
    py.allow_threads(|| attend_rows(qs, &ks[..(start + rows) * st], &vs[..(start + rows) * st], rows, start, dm, os,
                                    threads, block));
    Ok(())
}

/// Instruction-set level the attention kernels run at: 2 = AVX-512, 1 = AVX2, 0 = baseline.
#[pyfunction]
fn attn_isa_level() -> u8 {
    isa_level()
}

/// Cap the attention kernels' instruction-set level (tests: the bits do not depend on it).
#[pyfunction]
fn attn_set_isa_cap(level: u8) {
    ISA_CAP.store(level.min(2), Ordering::Relaxed);
}

/// float32 -> bfloat16 bit patterns (round to nearest even), as the bf16 cache stores them.
#[pyfunction]
fn bf16_round<'py>(x: &Bound<'py, numpy::PyArray1<f32>>, out: &Bound<'py, numpy::PyArray1<u16>>) -> PyResult<()> {
    let xr = x.try_readonly().map_err(|e| verr(format!("x: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (xs, os) = (xr.as_slice()?, ow.as_slice_mut()?);
    if xs.len() != os.len() {
        return Err(verr("bf16_round: shapes"));
    }
    for (o, &v) in os.iter_mut().zip(xs) {
        *o = bf16_rne(v);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
            })
            .collect()
    }

    /// The blocked reference written out directly (one row): the formulation in the module docs.
    fn blocked_row_ref(q: &[f32], k: &[f32], v: &[f32], len: usize, dm: Dims, bs: usize, out: &mut [f32]) {
        let Dims { nh, kvh, hd, scale } = dm;
        let groups = nh / kvh;
        let stride = kvh * hd;
        for hq in 0..nh {
            let g = hq / groups;
            let s: Vec<f64> = (0..len)
                .map(|j| dot32(&q[hq * hd..(hq + 1) * hd], &k[j * stride + g * hd..j * stride + (g + 1) * hd]) as f64
                    * scale)
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f32> = s.iter().map(|&x| exp_f64(x - m) as f32).collect();
            let (mut den, mut acc) = (0f64, vec![0f32; hd]);
            for (b, j0) in (0..len).step_by(bs).enumerate() {
                let j1 = (j0 + bs).min(len);
                let mut db = 0f64;
                let mut ab = vec![0f32; hd];
                for j in j0..j1 {
                    db += e[j] as f64;
                    for d in 0..hd {
                        ab[d] += e[j] * v[j * stride + g * hd + d];
                    }
                }
                if b == 0 {
                    den = db;
                    acc = ab;
                } else {
                    den += db;
                    for d in 0..hd {
                        acc[d] += ab[d];
                    }
                }
            }
            for d in 0..hd {
                out[hq * hd + d] = (acc[d] as f64 / den) as f32;
            }
        }
    }

    #[test]
    fn v1_matches_original_kernel_any_threads_rows_isa() {
        let dm = Dims { nh: 14, kvh: 2, hd: 64, scale: 0.125 };
        let (len, w, st) = (37usize, 14 * 64, 2 * 64);
        let q = data(len * w, 1);
        let k = data(len * st, 2);
        let v = data(len * st, 3);
        let mut want = vec![0f32; len * w];
        for t in 0..len {
            attend_row_ref(&q[t * w..(t + 1) * w], &k, &v, t + 1, dm, &mut want[t * w..(t + 1) * w]);
        }
        for cap in [2u8, 1, 0] {
            ISA_CAP.store(cap, Ordering::Relaxed);
            for threads in [1usize, 2, 3, 5] {
                let mut a = vec![0f32; len * w];
                attend_rows(&q, &k, &v, len, 0, dm, &mut a, threads, 0);
                assert_eq!(a, want, "threads {threads} isa {cap}");
            }
        }
        ISA_CAP.store(2, Ordering::Relaxed);
        for t in [0usize, 9, 36] {
            let mut o = vec![0f32; w];
            attend_rows(&q[t * w..(t + 1) * w], &k[..(t + 1) * st], &v[..(t + 1) * st], 1, t, dm, &mut o, 1, 0);
            assert_eq!(&o[..], &want[t * w..(t + 1) * w], "row {t}");
        }
    }

    #[test]
    fn v2_blocked_any_threads_rows_isa_and_matches_reference() {
        let dm = Dims { nh: 16, kvh: 2, hd: 128, scale: 0.088 };
        let (len, w, st) = (300usize, 16 * 128, 2 * 128);
        let q = data(len * w, 4);
        let k = data(len * st, 5);
        let v = data(len * st, 6);
        let bs = 32;
        // the last 40 rows as one pass (a long verify), positions 260..300
        let (rows, start) = (40usize, 260usize);
        let qs = &q[start * w..];
        let mut want = vec![0f32; rows * w];
        for t in 0..rows {
            blocked_row_ref(&qs[t * w..(t + 1) * w], &k, &v, start + t + 1, dm, bs, &mut want[t * w..(t + 1) * w]);
        }
        for cap in [2u8, 1, 0] {
            ISA_CAP.store(cap, Ordering::Relaxed);
            for threads in [1usize, 2, 4, 7] {
                let mut a = vec![0f32; rows * w];
                attend_rows(qs, &k, &v, rows, start, dm, &mut a, threads, bs);
                assert_eq!(a, want, "threads {threads} isa {cap}");
            }
        }
        ISA_CAP.store(2, Ordering::Relaxed);
        for t in [0usize, 3, 39] {
            let p = start + t;
            let mut o = vec![0f32; w];
            attend_rows(&qs[t * w..(t + 1) * w], &k[..(p + 1) * st], &v[..(p + 1) * st], 1, p, dm, &mut o, 3, bs);
            assert_eq!(&o[..], &want[t * w..(t + 1) * w], "row {t}");
        }
        // a prefix within one block is v1 bit for bit
        let mut a = vec![0f32; w];
        let mut b = vec![0f32; w];
        attend_rows(&q[20 * w..21 * w], &k[..21 * st], &v[..21 * st], 1, 20, dm, &mut a, 2, bs);
        attend_row_ref(&q[20 * w..21 * w], &k, &v, 21, dm, &mut b);
        assert_eq!(a, b);
        // close to a naive float64 softmax
        let want_rows = want;
        for t in [0usize, 39] {
            let p = start + t;
            for h in [0usize, 9] {
                let g = h / 8;
                let sc: Vec<f64> = (0..=p)
                    .map(|j| (0..128).map(|d| qs[t * w + h * 128 + d] as f64 * k[j * st + g * 128 + d] as f64)
                        .sum::<f64>() * 0.088)
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let den: f64 = e.iter().sum();
                for d in 0..128 {
                    let want: f64 = (0..=p).map(|j| e[j] * v[j * st + g * 128 + d] as f64).sum::<f64>() / den;
                    let got = want_rows[t * w + h * 128 + d] as f64;
                    assert!((got - want).abs() <= 1e-5 * want.abs().max(1e-1), "t={t} h={h} d={d} {got} {want}");
                }
            }
        }
    }

    #[test]
    fn bf16_cache_equals_widened_f32_cache() {
        let dm = Dims { nh: 8, kvh: 2, hd: 64, scale: 0.125 };
        let (len, w, st) = (90usize, 8 * 64, 2 * 64);
        let q = data(len * w, 7);
        let k: Vec<u16> = data(len * st, 8).into_iter().map(bf16_rne).collect();
        let v: Vec<u16> = data(len * st, 9).into_iter().map(bf16_rne).collect();
        let kf: Vec<f32> = k.iter().map(|&x| x.f()).collect();
        let vf: Vec<f32> = v.iter().map(|&x| x.f()).collect();
        let mut b = vec![0f32; len * w];
        ISA_CAP.store(0, Ordering::Relaxed);
        attend_rows(&q, &kf, &vf, len, 0, dm, &mut b, 1, 16); // scalar float32 reference
        for cap in [2u8, 1, 0] {
            ISA_CAP.store(cap, Ordering::Relaxed);
            for threads in [1usize, 3] {
                let mut a = vec![0f32; len * w];
                attend_rows(&q, &k, &v, len, 0, dm, &mut a, threads, 16);
                assert_eq!(a, b, "isa {cap} threads {threads}");
            }
        }
        ISA_CAP.store(2, Ordering::Relaxed);
        assert_eq!(bf16_rne(1.0), 0x3f80);
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80); // tie to even
        assert_eq!(bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn rope_matches_reference_form() {
        let x = data(64, 9);
        let c = data(64, 10);
        let s = data(64, 11);
        let mut o = vec![0f32; 64];
        rope_head(&x, &c, &s, &mut o);
        for d in 0..64 {
            let rot = if d < 32 { -x[d + 32] } else { x[d - 32] };
            assert_eq!(o[d], x[d] * c[d] + rot * s[d]);
        }
    }

    /// `cargo test --release -p pgw-fast --lib attn_time -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_time_72b() {
        let dm = Dims { nh: 64, kvh: 8, hd: 128, scale: 0.088 };
        let len: usize = std::env::var("ATTN_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let st = 8 * 128;
        let q = data(64 * 128, 1);
        let k = data(len * st, 2);
        let v = data(len * st, 3);
        let mut o = vec![0f32; 64 * 128];
        fn cpu() -> f64 {
            let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            // SAFETY: a valid out pointer.
            unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
            ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
        }
        for cap in [2u8, 1, 0] {
            ISA_CAP.store(cap, Ordering::Relaxed);
            for threads in [1usize, 4] {
                for block in [0usize, 256] {
                    attend_rows(&q, &k, &v, 1, len - 1, dm, &mut o, threads, block);
                    let (mut best, mut c) = (f64::MAX, f64::MAX);
                    for _ in 0..20 {
                        let (t0, c0) = (std::time::Instant::now(), cpu());
                        attend_rows(&q, &k, &v, 1, len - 1, dm, &mut o, threads, block);
                        best = best.min(t0.elapsed().as_secs_f64());
                        c = c.min(cpu() - c0);
                    }
                    println!("isa {cap} threads {threads} block {block}: wall min {:.0} us, caller cpu min {:.0} us",
                             best * 1e6, c * 1e6);
                }
            }
        }
        ISA_CAP.store(2, Ordering::Relaxed);
    }
}
