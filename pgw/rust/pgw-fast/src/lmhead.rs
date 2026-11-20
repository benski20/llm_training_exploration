//! Deterministic float32 LM head: `logits = h @ W^T` for several rows in one pass over `W`.
//!
//! The gateway computes `rms_norm(h) @ lm_head^T` in float32 (`TransformerModel`); the head is
//! `vocab x hidden` (2.2 GB for Qwen2.5-7B), so the product is memory bound and a pass that reads
//! each weight row once for all `R` rows costs about as much as one row.
//!
//! **Fixed summation order.** Every output is computed with the same IEEE operations in the same
//! order, whatever `R`, the thread count, the blocking or the CPU features detected at run time:
//!
//! 1. `LANES = 16` partial sums `acc[l]`, all starting at `+0.0`;
//! 2. for each full chunk `c` of 16 columns, in increasing order: `acc[l] = acc[l] + h[16c+l] * w[16c+l]`
//!    (product rounded to f32, then the sum rounded to f32: no FMA);
//! 3. the last `D mod 16` columns: `acc[l] = acc[l] + h[16C+l] * w[16C+l]` for `l < D mod 16`;
//! 4. a fixed tree: `acc[l] + acc[l+8]` (8 sums), then `+4`, `+2`, `+1`.
//!
//! The lanes live in vector registers (SSE2 `4 x __m128`, AVX `2 x __m256`, or a plain array):
//! lane `l` is the same element of the same accumulator on every path, and only separate multiply
//! and add instructions are used (Rust never contracts `a + b * c` into an FMA nor reassociates
//! float operations), so every path gives identical bits; the tests compare them with a scalar
//! rendering of the order. `pgw.engine.lm_head.lm_head_reference` implements the same order in torch
//! (`tests/native/test_lm_head.py` checks the bits).
//!
//! Work split: worker threads take dynamic chunks of weight rows; each worker reads a few weight
//! rows (`NV`, kept in L1/L2) and computes their dot products with every row of `h` (a block of
//! rows small enough to stay in L2), so the head is streamed from memory once per call for up to
//! `H_BLOCK_BYTES / (4 D)` rows (36 at `D = 3584`).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Partial sums per dot product (the fixed summation order depends on it).
pub const LANES: usize = 16;
/// Bytes of `h` rows kept hot in cache while one block of weight rows is reused across them.
const H_BLOCK_BYTES: usize = 512 * 1024;
/// Roughly this many multiply-adds per worker thread before going parallel.
const WORK_PER_THREAD: usize = 1 << 22;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lm_head, m)?)?;
    m.add("LM_HEAD_LANES", LANES)?;
    Ok(())
}

/// Combine the 16 partial sums in the fixed tree order.
#[inline(always)]
fn tree(acc: &[f32; LANES]) -> f32 {
    let mut s8 = [0f32; 8];
    for l in 0..8 {
        s8[l] = acc[l] + acc[l + 8];
    }
    let mut s4 = [0f32; 4];
    for l in 0..4 {
        s4[l] = s8[l] + s8[l + 4];
    }
    let s2 = [s4[0] + s4[2], s4[1] + s4[3]];
    s2[0] + s2[1]
}

/// Raw output pointer shared by workers that write disjoint elements.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers write pairwise disjoint index sets; the scope joins them before the Vec is read.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// 16 float32 lanes: the partial sums of one dot product. Implementations only differ in how the
/// lanes are laid out in registers; every lane sees the same IEEE operations (`a + b * c` with two
/// roundings, never an FMA).
trait Lanes: Copy {
    /// All lanes `+0.0`.
    unsafe fn zero() -> Self;
    /// Load 16 consecutive values (unaligned).
    unsafe fn load(p: *const f32) -> Self;
    /// `self + a * b` lane-wise (product rounded, then the sum).
    unsafe fn mul_add(self, a: Self, b: Self) -> Self;
    /// The lanes as an array.
    unsafe fn lanes(self) -> [f32; LANES];
}

/// Portable lanes (any target; LLVM may vectorise it, the result is the same).
#[derive(Clone, Copy)]
struct Scalar([f32; LANES]);

impl Lanes for Scalar {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Scalar([0.0; LANES])
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        Scalar(std::ptr::read_unaligned(p as *const [f32; LANES]))
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self, b: Self) -> Self {
        let mut o = self.0;
        for ((o, x), y) in o.iter_mut().zip(a.0).zip(b.0) {
            *o += x * y;
        }
        Scalar(o)
    }
    #[inline(always)]
    unsafe fn lanes(self) -> [f32; LANES] {
        self.0
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Lanes, LANES};
    use std::arch::x86_64::*;

    /// SSE2 (x86_64 baseline): four `__m128`.
    #[derive(Clone, Copy)]
    pub struct Sse([__m128; 4]);

    impl Lanes for Sse {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Sse([_mm_setzero_ps(); 4])
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Sse([_mm_loadu_ps(p), _mm_loadu_ps(p.add(4)), _mm_loadu_ps(p.add(8)), _mm_loadu_ps(p.add(12))])
        }
        #[inline(always)]
        unsafe fn mul_add(self, a: Self, b: Self) -> Self {
            Sse(std::array::from_fn(|i| _mm_add_ps(self.0[i], _mm_mul_ps(a.0[i], b.0[i]))))
        }
        #[inline(always)]
        unsafe fn lanes(self) -> [f32; LANES] {
            let mut o = [0f32; LANES];
            for i in 0..4 {
                _mm_storeu_ps(o.as_mut_ptr().add(4 * i), self.0[i]);
            }
            o
        }
    }

    /// AVX: two `__m256` (separate multiply and add instructions: no FMA).
    #[derive(Clone, Copy)]
    pub struct Avx([__m256; 2]);

    impl Lanes for Avx {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Avx([_mm256_setzero_ps(); 2])
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Avx([_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))])
        }
        #[inline(always)]
        unsafe fn mul_add(self, a: Self, b: Self) -> Self {
            Avx([
                _mm256_add_ps(self.0[0], _mm256_mul_ps(a.0[0], b.0[0])),
                _mm256_add_ps(self.0[1], _mm256_mul_ps(a.0[1], b.0[1])),
            ])
        }
        #[inline(always)]
        unsafe fn lanes(self) -> [f32; LANES] {
            let mut o = [0f32; LANES];
            _mm256_storeu_ps(o.as_mut_ptr(), self.0[0]);
            _mm256_storeu_ps(o.as_mut_ptr().add(8), self.0[1]);
            o
        }
    }
}

/// Software prefetch distance along a weight row (floats); the head is streamed from memory.
const PREFETCH: usize = 512;

/// Hint that `p`'s cache line is needed soon (no effect on results; may point past the data).
#[inline(always)]
fn prefetch(p: *const f32) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// `NV` dot products of `h` (`d` values) with the weight rows starting at `w`, `w + d`, ...
///
/// # Safety
/// `h` must hold `d` values and `w` `NV * d` values; `L`'s instructions must be available.
#[inline(always)]
unsafe fn dots<L: Lanes, const NV: usize>(h: *const f32, w: *const f32, d: usize) -> [f32; NV] {
    let full = d / LANES * LANES;
    let mut acc = [L::zero(); NV];
    let mut k = 0;
    while k < full {
        let hv = L::load(h.add(k));
        for (j, a) in acc.iter_mut().enumerate() {
            prefetch(w.wrapping_add(j * d + k + PREFETCH));
            *a = a.mul_add(hv, L::load(w.add(j * d + k)));
        }
        k += LANES;
    }
    let mut out = [0f32; NV];
    for (j, a) in acc.iter().enumerate() {
        let mut s = a.lanes();
        for (l, sl) in s.iter_mut().enumerate().take(d - full) {
            *sl += *h.add(full + l) * *w.add(j * d + full + l);
        }
        out[j] = tree(&s);
    }
    out
}

/// Outputs `out[r, v]` for all `r` and `v` in `[va, vb)`, `NV` weight rows at a time.
///
/// # Safety
/// Shapes as in [`lm_head_run`]; `L`'s instructions must be available; no other thread writes
/// these outputs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn work_with<L: Lanes, const NV: usize>(
    h: &[f32],
    w: &[f32],
    r: usize,
    d: usize,
    v: usize,
    va: usize,
    vb: usize,
    out: OutPtr,
) {
    let (hp, wp) = (h.as_ptr(), w.as_ptr());
    let rb = (H_BLOCK_BYTES / (4 * d.max(1))).clamp(1, 64);
    let mut r0 = 0;
    while r0 < r {
        let r1 = (r0 + rb).min(r);
        let mut j = va;
        while j + NV <= vb {
            for ri in r0..r1 {
                let o = dots::<L, NV>(hp.add(ri * d), wp.add(j * d), d);
                for (q, val) in o.iter().enumerate() {
                    *out.0.add(ri * v + j + q) = *val;
                }
            }
            j += NV;
        }
        while j < vb {
            for ri in r0..r1 {
                *out.0.add(ri * v + j) = dots::<L, 1>(hp.add(ri * d), wp.add(j * d), d)[0];
            }
            j += 1;
        }
        r0 = r1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn work_avx(h: &[f32], w: &[f32], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    work_with::<x86::Avx, 4>(h, w, r, d, v, va, vb, out)
}

/// Instruction set used by [`lm_head_run`] (tests force each one to compare the bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Best available at run time.
    Auto,
    /// Portable code.
    Scalar,
    /// SSE2 (x86_64 baseline).
    Sse,
    /// AVX (256-bit; multiply and add, no FMA).
    Avx,
}

#[allow(clippy::too_many_arguments)]
fn work(isa: Isa, h: &[f32], w: &[f32], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    // SAFETY (all arms): shapes were checked by lm_head_run, each worker owns its output columns,
    // and an instruction set is only used when the CPU has it.
    #[cfg(target_arch = "x86_64")]
    {
        let avx = std::is_x86_feature_detected!("avx");
        match isa {
            Isa::Auto | Isa::Avx if avx => return unsafe { work_avx(h, w, r, d, v, va, vb, out) },
            Isa::Auto | Isa::Avx | Isa::Sse => {
                return unsafe { work_with::<x86::Sse, 2>(h, w, r, d, v, va, vb, out) };
            }
            Isa::Scalar => {}
        }
    }
    let _ = isa;
    unsafe { work_with::<Scalar, 2>(h, w, r, d, v, va, vb, out) }
}

/// Automatic worker count: the available parallelism (queried once).
fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |v| v.get()).min(16))
}

/// `h (r, d) @ w (v, d)^T -> (r, v)` row-major, in the fixed summation order (`threads = 0`:
/// automatic). The result does not depend on `threads`, `isa` or `r` (row `i` of the output only
/// depends on row `i` of `h`).
pub fn lm_head_run(h: &[f32], w: &[f32], r: usize, d: usize, v: usize, threads: usize, isa: Isa) -> Vec<f32> {
    assert_eq!(h.len(), r * d, "h has the wrong length");
    assert_eq!(w.len(), v * d, "w has the wrong length");
    let mut out: Vec<f32> = Vec::with_capacity(r * v);
    if r * v == 0 {
        return out;
    }
    let ptr = OutPtr(out.as_mut_ptr());
    let max_t = if threads == 0 { default_threads() } else { threads };
    let t = (r * v * d.max(1) / WORK_PER_THREAD).clamp(1, max_t.max(1)).min(v.div_ceil(64));
    if t == 1 {
        work(isa, h, w, r, d, v, 0, v, ptr);
    } else {
        // Dynamic chunks of weight rows (a slow worker does not hold up the others); the chunks
        // partition `[0, v)` and which worker computes a chunk does not change its bits.
        let chunk = (v.div_ceil(t * 16)).clamp(64, 4096).div_ceil(64) * 64;
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..t {
                let next = &next;
                s.spawn(move || loop {
                    let va = next.fetch_add(chunk, Ordering::Relaxed);
                    if va >= v {
                        break;
                    }
                    work(isa, h, w, r, d, v, va, (va + chunk).min(v), ptr);
                });
            }
        });
    }
    // SAFETY: the workers' ranges partition `[0, v)` and each writes every row of its range.
    unsafe { out.set_len(r * v) };
    out
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Deterministic float32 LM head ``h @ w.T`` (see the module docs for the fixed summation order).
///
/// Args:
///     h: float32 ``(R, D)``, C-contiguous.
///     w: float32 ``(V, D)``, C-contiguous (read once for all ``R`` rows).
///     threads: worker threads (0 = automatic); the result does not depend on it.
///
/// Returns float32 ``(R, V)``; row ``i`` only depends on row ``i`` of ``h`` (bit for bit).
#[pyfunction]
#[pyo3(signature = (h, w, threads = 0))]
fn lm_head<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ha: PyReadonlyArray2<'py, f32> = h.extract().map_err(|_| verr("h must be a 2-D float32 array"))?;
    let wa: PyReadonlyArray2<'py, f32> = w.extract().map_err(|_| verr("w must be a 2-D float32 array"))?;
    let (r, d) = (ha.shape()[0], ha.shape()[1]);
    let (v, d2) = (wa.shape()[0], wa.shape()[1]);
    if d != d2 {
        return Err(verr(format!("h has {d} columns but w has {d2}")));
    }
    // `as_slice` also accepts Fortran order, which would be read transposed.
    if !ha.is_c_contiguous() || !wa.is_c_contiguous() {
        return Err(verr("h and w must be C-contiguous"));
    }
    let hs = ha.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
    let ws = wa.as_slice().map_err(|_| verr("w must be C-contiguous"))?;
    let out = py.allow_threads(|| lm_head_run(hs, ws, r, d, v, threads, Isa::Auto));
    let a = Array2::from_shape_vec((r, v), out).map_err(|e| verr(e.to_string()))?;
    Ok(a.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn randf(state: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| ((rng(state) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale).collect()
    }

    /// The documented order, written out with scalar code.
    fn reference(h: &[f32], w: &[f32], r: usize, d: usize, v: usize) -> Vec<f32> {
        let mut out = vec![0f32; r * v];
        for i in 0..r {
            for j in 0..v {
                let mut acc = [0f32; LANES];
                for k in 0..d {
                    acc[k % LANES] += h[i * d + k] * w[j * d + k];
                }
                let mut width = LANES;
                while width > 1 {
                    width /= 2;
                    for l in 0..width {
                        acc[l] += acc[l + width];
                    }
                }
                out[i * v + j] = acc[0];
            }
        }
        out
    }

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn matches_scalar_order_for_every_isa_thread_count_and_row_count() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for &(d, v) in &[(1usize, 3usize), (15, 9), (16, 4), (17, 13), (64, 37), (100, 70), (515, 33)] {
            let w = randf(&mut s, v * d, 0.1);
            let h = randf(&mut s, 9 * d, 4.0);
            let want = reference(&h, &w, 9, d, v);
            for isa in [Isa::Auto, Isa::Scalar, Isa::Sse, Isa::Avx] {
                for threads in [1usize, 2, 3, 7] {
                    for r in 1..=9 {
                        let got = lm_head_run(&h[..r * d], &w, r, d, v, threads, isa);
                        assert_eq!(bits(&got), bits(&want[..r * v]), "d={d} v={v} r={r} t={threads} {isa:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn large_shape_parallel_matches_single_thread() {
        let mut s = 7u64;
        let (r, d, v) = (5usize, 896usize, 4099usize);
        let w = randf(&mut s, v * d, 0.05);
        let h = randf(&mut s, r * d, 2.0);
        let one = lm_head_run(&h, &w, r, d, v, 1, Isa::Scalar);
        assert_eq!(bits(&one), bits(&reference(&h, &w, r, d, v)));
        for t in [0usize, 4, 16] {
            assert_eq!(bits(&lm_head_run(&h, &w, r, d, v, t, Isa::Auto)), bits(&one));
        }
        for i in 0..r {
            assert_eq!(
                bits(&lm_head_run(&h[i * d..(i + 1) * d], &w, 1, d, v, 0, Isa::Auto)),
                bits(&one[i * v..(i + 1) * v])
            );
        }
    }

    /// Timing per instruction set (`cargo test --release -p pgw-fast bench_isa -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_isa() {
        let (d, v) = (3584usize, std::env::var("BV").map_or(152064 / 4, |s| s.parse().unwrap()));
        let mut s = 3u64;
        let w = randf(&mut s, v * d, 0.05);
        let h = randf(&mut s, 8 * d, 2.0);
        for isa in [Isa::Scalar, Isa::Sse, Isa::Avx] {
            for r in [1usize, 4, 8] {
                let reps = (152064 / v).max(1);
                let _ = lm_head_run(&h[..r * d], &w, r, d, v, 1, isa);
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(lm_head_run(&h[..r * d], &w, r, d, v, 1, isa));
                }
                eprintln!("{isa:?} r={r}: {:?} per 152064 x {d} head", t.elapsed() * (152064 / v / reps) as u32);
            }
        }
    }

    #[test]
    fn empty_shapes() {
        assert!(lm_head_run(&[], &[1.0, 2.0], 0, 2, 1, 0, Isa::Auto).is_empty());
        assert_eq!(lm_head_run(&[], &[], 2, 0, 3, 0, Isa::Auto), vec![0.0; 6]);
    }
}
