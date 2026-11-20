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
//! The lanes live in vector registers (SSE2 `4 x __m128`, AVX/AVX2 `2 x __m256`, AVX-512F one
//! `__m512`, or a plain array; the best one the CPU has is picked at run time, `Isa`):
//! lane `l` is the same element of the same accumulator on every path, and only separate multiply
//! and add instructions are used (Rust never contracts `a + b * c` into an FMA nor reassociates
//! float operations), so every path gives identical bits; the tests compare them with a scalar
//! rendering of the order. `pgw.engine.lm_head.lm_head_reference` implements the same order in torch
//! (`tests/native/test_lm_head.py` checks the bits).
//!
//! Work split: worker threads take dynamic chunks of weight rows; each worker reads a few weight
//! rows (`NV`, kept in L1/L2) and computes their dot products with every row of `h` (a block of
//! rows small enough to stay in L2), so the head is streamed from memory once per call for up to
//! `H_BLOCK_BYTES / (4 D)` rows (36 at `D = 3584`). Inside that, outputs are register-blocked
//! (`RB` rows of `h` by `NV` weight rows, see [`block`]): each weight chunk is loaded and widened
//! once for `RB` rows, which matters for multi-session steps (one row per stream). Blocking only
//! chooses which outputs are computed side by side, never the order of an output's own operations.

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
    m.add_function(wrap_pyfunction!(lm_head_bf16, m)?)?;
    m.add_function(wrap_pyfunction!(lm_head_isa, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_strict, m)?)?;
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
pub trait Lanes: Copy {
    /// All lanes `+0.0`.
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn zero() -> Self;
    /// Load 16 consecutive values (unaligned).
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn load(p: *const f32) -> Self;
    /// Load 16 consecutive bfloat16 values widened to f32 (exact).
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn load_bf16(p: *const u16) -> Self;
    /// `self + a * b` lane-wise (product rounded, then the sum).
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn mul_add(self, a: Self, b: Self) -> Self;
    /// The lanes as an array.
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn lanes(self) -> [f32; LANES];
}

/// Element type of the weight matrix. `f32` weights are used as stored; `Bf16` weights (bfloat16
/// bits, as Qwen and most open checkpoints ship them) are widened to f32 on load, which is exact
/// (`f32::from_bits((b as u32) << 16)`), so a bfloat16 head gives exactly the bits of its float32
/// copy while reading half the memory.
pub trait WElem: Copy + Send + Sync {
    /// Load 16 consecutive weights as f32 lanes.
    ///
    /// # Safety
    /// Pointers must cover 16 readable values and the CPU must support the implementation's instructions.
    unsafe fn load<L: Lanes>(p: *const Self) -> L;
    /// One weight as f32 (exact).
    fn f32(self) -> f32;
}

impl WElem for f32 {
    #[inline(always)]
    unsafe fn load<L: Lanes>(p: *const Self) -> L {
        L::load(p)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        self
    }
}

/// bfloat16 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Bf16(pub u16);

impl WElem for Bf16 {
    #[inline(always)]
    unsafe fn load<L: Lanes>(p: *const Self) -> L {
        L::load_bf16(p as *const u16)
    }
    #[inline(always)]
    fn f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
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
    unsafe fn load_bf16(p: *const u16) -> Self {
        let b = std::ptr::read_unaligned(p as *const [u16; LANES]);
        Scalar(b.map(|x| f32::from_bits((x as u32) << 16)))
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

    /// 16 bfloat16 values widened to f32 in four `__m128` (lanes 0-3, 4-7, 8-11, 12-15): each u16
    /// interleaved below a zero u16 is exactly its f32 bit pattern (SSE2 only).
    #[inline(always)]
    unsafe fn bf16x16(p: *const u16) -> [__m128; 4] {
        let z = _mm_setzero_si128();
        let a = _mm_loadu_si128(p as *const __m128i);
        let b = _mm_loadu_si128(p.add(8) as *const __m128i);
        [
            _mm_castsi128_ps(_mm_unpacklo_epi16(z, a)),
            _mm_castsi128_ps(_mm_unpackhi_epi16(z, a)),
            _mm_castsi128_ps(_mm_unpacklo_epi16(z, b)),
            _mm_castsi128_ps(_mm_unpackhi_epi16(z, b)),
        ]
    }

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
        unsafe fn load_bf16(p: *const u16) -> Self {
            Sse(bf16x16(p))
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
        unsafe fn load_bf16(p: *const u16) -> Self {
            let q = bf16x16(p);
            Avx([_mm256_set_m128(q[1], q[0]), _mm256_set_m128(q[3], q[2])])
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

    /// AVX2: two `__m256` like [`Avx`], with a cheaper bfloat16 widening (zero-extend each u16 to
    /// u32 and shift it into the high half: exactly its f32 bit pattern).
    #[derive(Clone, Copy)]
    pub struct Avx2([__m256; 2]);

    impl Lanes for Avx2 {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Avx2([_mm256_setzero_ps(); 2])
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Avx2([_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))])
        }
        #[inline(always)]
        unsafe fn load_bf16(p: *const u16) -> Self {
            let a = _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i));
            let b = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(8) as *const __m128i));
            Avx2([_mm256_castsi256_ps(_mm256_slli_epi32::<16>(a)), _mm256_castsi256_ps(_mm256_slli_epi32::<16>(b))])
        }
        #[inline(always)]
        unsafe fn mul_add(self, a: Self, b: Self) -> Self {
            Avx2([
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

    /// AVX-512F: all 16 lanes in one `__m512` (lane `l` is element `l`, as on every other path;
    /// separate multiply and add instructions, never an FMA).
    #[derive(Clone, Copy)]
    pub struct Avx512(__m512);

    impl Lanes for Avx512 {
        #[inline(always)]
        unsafe fn zero() -> Self {
            Avx512(_mm512_setzero_ps())
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Avx512(_mm512_loadu_ps(p))
        }
        #[inline(always)]
        unsafe fn load_bf16(p: *const u16) -> Self {
            let a = _mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i));
            Avx512(_mm512_castsi512_ps(_mm512_slli_epi32::<16>(a)))
        }
        #[inline(always)]
        unsafe fn mul_add(self, a: Self, b: Self) -> Self {
            Avx512(_mm512_add_ps(self.0, _mm512_mul_ps(a.0, b.0)))
        }
        #[inline(always)]
        unsafe fn lanes(self) -> [f32; LANES] {
            let mut o = [0f32; LANES];
            _mm512_storeu_ps(o.as_mut_ptr(), self.0);
            o
        }
    }
}

/// Software prefetch distance along a weight row (floats); the head is streamed from memory.
const PREFETCH: usize = 512;

/// Hint that `p`'s cache line is needed soon (no effect on results; may point past the data).
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// The `RB x NV` dot products of `RB` rows of `h` (row `i` at `h + i * d`) with `NV` weight rows
/// (row `j` at `w + j * d`), written to `out[i * v + j]` (row and column offsets already applied).
///
/// Register blocking only decides which outputs are computed side by side: every output still sees
/// its own 16 lanes updated over the columns in increasing order, then its tail and the tree, so
/// the bits do not depend on `RB` or `NV`. Each weight chunk is loaded (and widened) once for the
/// `RB` rows.
///
/// # Safety
/// `h` must hold `RB * d` values, `w` `NV * d` values, `out` must be writable at the offsets above;
/// `L`'s instructions must be available.
#[inline(always)]
unsafe fn block<L: Lanes, W: WElem, const RB: usize, const NV: usize>(
    h: *const f32,
    w: *const W,
    d: usize,
    out: *mut f32,
    v: usize,
) {
    let full = d / LANES * LANES;
    let mut acc = [[L::zero(); NV]; RB];
    let mut k = 0;
    while k < full {
        let mut wv = [L::zero(); NV];
        for (j, x) in wv.iter_mut().enumerate() {
            prefetch(w.wrapping_add(j * d + k + PREFETCH));
            *x = W::load::<L>(w.add(j * d + k));
        }
        for (i, row) in acc.iter_mut().enumerate() {
            let hv = L::load(h.add(i * d + k));
            for (a, x) in row.iter_mut().zip(wv.iter()) {
                *a = a.mul_add(hv, *x);
            }
        }
        k += LANES;
    }
    for (i, row) in acc.iter().enumerate() {
        for (j, a) in row.iter().enumerate() {
            let mut s = a.lanes();
            for (l, sl) in s.iter_mut().enumerate().take(d - full) {
                *sl += *h.add(i * d + full + l) * (*w.add(j * d + full + l)).f32();
            }
            *out.add(i * v + j) = tree(&s);
        }
    }
}

/// Outputs `out[r, v]` for all `r` and `v` in `[va, vb)`: blocks of `RB` rows by `NV` weight rows,
/// then the remainders one row / one weight row at a time.
///
/// # Safety
/// Shapes as in [`lm_head_run`]; `L`'s instructions must be available; no other thread writes
/// these outputs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn work_with<L: Lanes, W: WElem, const RB: usize, const NV: usize>(
    h: &[f32],
    w: &[W],
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
            let mut ri = r0;
            while ri + RB <= r1 {
                block::<L, W, RB, NV>(hp.add(ri * d), wp.add(j * d), d, out.0.add(ri * v + j), v);
                ri += RB;
            }
            while ri < r1 {
                block::<L, W, 1, NV>(hp.add(ri * d), wp.add(j * d), d, out.0.add(ri * v + j), v);
                ri += 1;
            }
            j += NV;
        }
        while j < vb {
            for ri in r0..r1 {
                block::<L, W, 1, 1>(hp.add(ri * d), wp.add(j * d), d, out.0.add(ri * v + j), v);
            }
            j += 1;
        }
        r0 = r1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn work_avx<W: WElem>(h: &[f32], w: &[W], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    if r == 1 {
        work_with::<x86::Avx, W, 1, 4>(h, w, r, d, v, va, vb, out)
    } else {
        work_with::<x86::Avx, W, 2, 2>(h, w, r, d, v, va, vb, out)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn work_avx2<W: WElem>(h: &[f32], w: &[W], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    if r == 1 {
        work_with::<x86::Avx2, W, 1, 4>(h, w, r, d, v, va, vb, out)
    } else {
        work_with::<x86::Avx2, W, 2, 2>(h, w, r, d, v, va, vb, out)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn work_avx512<W: WElem>(h: &[f32], w: &[W], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    if r == 1 {
        work_with::<x86::Avx512, W, 1, 8>(h, w, r, d, v, va, vb, out)
    } else {
        work_with::<x86::Avx512, W, 4, 4>(h, w, r, d, v, va, vb, out)
    }
}

/// Instruction set used by [`lm_head_run`] (tests force each one to compare the bits). Every
/// choice gives the same bits; only the speed differs.
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
    /// AVX2 (as AVX, with integer bfloat16 widening).
    Avx2,
    /// AVX-512F (one 512-bit register per dot product).
    Avx512,
}

impl Isa {
    /// Parse a name (`auto`, `scalar`, `sse`, `avx`, `avx2`, `avx512`).
    pub fn parse(name: &str) -> Option<Isa> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Isa::Auto,
            "scalar" => Isa::Scalar,
            "sse" | "sse2" => Isa::Sse,
            "avx" => Isa::Avx,
            "avx2" => Isa::Avx2,
            "avx512" | "avx512f" => Isa::Avx512,
            _ => return None,
        })
    }

    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Isa::Auto => "auto",
            Isa::Scalar => "scalar",
            Isa::Sse => "sse",
            Isa::Avx => "avx",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }

    /// The instruction set actually used for a request: the request if this CPU has it, else the
    /// best available one below it (`Auto`: the best available).
    pub fn resolve(self) -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            let mut want = self;
            if want == Isa::Auto {
                want = Isa::Avx512;
            }
            if want == Isa::Avx512 && !std::is_x86_feature_detected!("avx512f") {
                want = Isa::Avx2;
            }
            if want == Isa::Avx2 && !std::is_x86_feature_detected!("avx2") {
                want = Isa::Avx;
            }
            if want == Isa::Avx && !std::is_x86_feature_detected!("avx") {
                want = Isa::Sse;
            }
            want
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Isa::Scalar
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn work<W: WElem>(isa: Isa, h: &[f32], w: &[W], r: usize, d: usize, v: usize, va: usize, vb: usize, out: OutPtr) {
    // SAFETY (all arms): shapes were checked by lm_head_run, each worker owns its output columns,
    // and `resolve` only returns an instruction set the CPU has.
    #[cfg(target_arch = "x86_64")]
    {
        match isa.resolve() {
            Isa::Avx512 => return unsafe { work_avx512(h, w, r, d, v, va, vb, out) },
            Isa::Avx2 => return unsafe { work_avx2(h, w, r, d, v, va, vb, out) },
            Isa::Avx => return unsafe { work_avx(h, w, r, d, v, va, vb, out) },
            Isa::Sse => return unsafe { work_with::<x86::Sse, W, 1, 2>(h, w, r, d, v, va, vb, out) },
            Isa::Scalar | Isa::Auto => {}
        }
    }
    let _ = isa;
    unsafe { work_with::<Scalar, W, 1, 2>(h, w, r, d, v, va, vb, out) }
}

/// Automatic worker count: the available parallelism (queried once).
fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |v| v.get()).min(16))
}

/// `h (r, d) @ w (v, d)^T -> (r, v)` row-major, in the fixed summation order (`threads = 0`:
/// automatic). The result does not depend on `threads`, `isa` or `r` (row `i` of the output only
/// depends on row `i` of `h`).
pub fn lm_head_run<W: WElem>(h: &[f32], w: &[W], r: usize, d: usize, v: usize, threads: usize, isa: Isa) -> Vec<f32> {
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

/// The instruction set :func:`lm_head` uses for ``isa`` on this CPU (e.g. ``"avx512"`` for ``"auto"``).
#[pyfunction]
#[pyo3(signature = (isa = "auto"))]
fn lm_head_isa(isa: &str) -> PyResult<&'static str> {
    Ok(Isa::parse(isa).ok_or_else(|| verr(format!("unknown isa {isa:?}")))?.resolve().name())
}

/// The `k` largest values of `a` with their indices, in descending order, **only when that answer
/// is unique**: no NaN anywhere, and the `k + 1` largest values strictly decreasing (so no tie
/// inside the top `k` nor across its boundary; `-0.0` and `+0.0` count as a tie). Then every
/// correct top-k (e.g. `torch.topk(a, k)`, whose order among equal values is
/// implementation-defined) returns exactly these pairs. `None` otherwise (the caller falls back to
/// its reference) and for `k > a.len()`. Comparisons only: no arithmetic on the values.
pub fn top_k_strict_run(a: &[f32], k: usize) -> Option<Vec<(usize, f32)>> {
    if k > a.len() {
        return None;
    }
    if k == 0 {
        return if a.iter().any(|x| x.is_nan()) { None } else { Some(Vec::new()) };
    }
    let m = (k + 1).min(a.len());
    // the m largest so far, descending; equal values in index order
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(m + 1);
    const CHUNK: usize = 32;
    for (c, chunk) in a.chunks(CHUNK).enumerate() {
        if best.len() == m {
            // Skip a chunk with no NaN and nothing above the current m-th value (a branch-free max
            // that the compiler vectorizes; `x != x` is the NaN test).
            let thr = best[m - 1].0;
            let (mut mx, mut nan) = (f32::NEG_INFINITY, false);
            for &x in chunk {
                nan |= x != x;
                mx = if x > mx { x } else { mx };
            }
            if !nan && mx <= thr {
                continue;
            }
        }
        for (o, &x) in chunk.iter().enumerate() {
            if x.is_nan() {
                return None;
            }
            if best.len() == m && x <= best[m - 1].0 {
                continue;
            }
            let pos = best.iter().position(|b| x > b.0).unwrap_or(best.len());
            best.insert(pos, (x, c * CHUNK + o));
            best.truncate(m);
        }
    }
    if best.windows(2).any(|p| p[0].0 <= p[1].0) {
        return None;
    }
    Some(best.into_iter().take(k).map(|(x, i)| (i, x)).collect())
}

/// :func:`top_k_strict_run` for a 1-D float32 array: ``(indices, values)`` lists, or ``None`` when
/// the top ``k`` is not unique (ties at or above the ``k``-th value, or a NaN).
#[pyfunction]
fn top_k_strict<'py>(
    py: Python<'py>,
    a: numpy::PyReadonlyArray1<'py, f32>,
    k: usize,
) -> PyResult<Option<(Vec<usize>, Vec<f32>)>> {
    let got = if let Ok(s) = a.as_slice() {
        py.allow_threads(|| top_k_strict_run(s, k))
    } else {
        let v: Vec<f32> = a.as_array().iter().copied().collect();
        py.allow_threads(|| top_k_strict_run(&v, k))
    };
    Ok(got.map(|pairs| pairs.into_iter().unzip()))
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
///     isa: instruction set (``auto``, ``avx512``, ``avx2``, ``avx``, ``sse``, ``scalar``; one this
///         CPU lacks falls back to the best available below it); the result does not depend on it.
///
/// Returns float32 ``(R, V)``; row ``i`` only depends on row ``i`` of ``h`` (bit for bit).
#[pyfunction]
#[pyo3(signature = (h, w, threads = 0, isa = "auto"))]
fn lm_head<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let isa = Isa::parse(isa).ok_or_else(|| verr(format!("unknown isa {isa:?}")))?;
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
    let out = py.allow_threads(|| lm_head_run(hs, ws, r, d, v, threads, isa));
    let a = Array2::from_shape_vec((r, v), out).map_err(|e| verr(e.to_string()))?;
    Ok(a.into_pyarray(py))
}

/// [`lm_head`] with a bfloat16 head given as its raw bits (``uint16``/``int16`` ``(V, D)``): each
/// weight is widened to float32 exactly, so the result equals ``lm_head(h, w.astype(float32))`` bit
/// for bit while reading half the memory.
#[pyfunction]
#[pyo3(signature = (h, w, threads = 0, isa = "auto"))]
fn lm_head_bf16<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    w: &Bound<'py, PyAny>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let isa = Isa::parse(isa).ok_or_else(|| verr(format!("unknown isa {isa:?}")))?;
    let ha: PyReadonlyArray2<'py, f32> = h.extract().map_err(|_| verr("h must be a 2-D float32 array"))?;
    let wa: PyReadonlyArray2<'py, u16> =
        w.extract().map_err(|_| verr("w must be a 2-D uint16 array (bfloat16 bits)"))?;
    let (r, d) = (ha.shape()[0], ha.shape()[1]);
    let (v, d2) = (wa.shape()[0], wa.shape()[1]);
    if d != d2 {
        return Err(verr(format!("h has {d} columns but w has {d2}")));
    }
    if !ha.is_c_contiguous() || !wa.is_c_contiguous() {
        return Err(verr("h and w must be C-contiguous"));
    }
    let hs = ha.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
    let ws = wa.as_slice().map_err(|_| verr("w must be C-contiguous"))?;
    // SAFETY: `Bf16` is a `repr(transparent)` u16.
    let wb: &[Bf16] = unsafe { std::slice::from_raw_parts(ws.as_ptr() as *const Bf16, ws.len()) };
    let out = py.allow_threads(|| lm_head_run(hs, wb, r, d, v, threads, isa));
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

    const ALL_ISAS: [Isa; 6] = [Isa::Auto, Isa::Scalar, Isa::Sse, Isa::Avx, Isa::Avx2, Isa::Avx512];

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
            for isa in ALL_ISAS {
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
        for isa in [Isa::Scalar, Isa::Sse, Isa::Avx, Isa::Avx2, Isa::Avx512] {
            for r in [1usize, 4, 8] {
                let reps = (152064 / v).max(1);
                let _ = lm_head_run(&h[..r * d], &w, r, d, v, 1, isa);
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(lm_head_run(&h[..r * d], &w, r, d, v, 1, isa));
                }
                eprintln!("{isa:?} r={r}: {:?} per 152064 x {d} head", t.elapsed() * (152064 / v / reps) as u32);
                let wb: Vec<Bf16> = w.iter().map(|x| Bf16((x.to_bits() >> 16) as u16)).collect();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(lm_head_run(&h[..r * d], &wb, r, d, v, 1, isa));
                }
                eprintln!("{isa:?} r={r} bf16: {:?}", t.elapsed() * (152064 / v / reps) as u32);
            }
        }
    }

    #[test]
    fn bf16_head_equals_its_f32_copy_for_every_isa() {
        let mut s = 0x51u64;
        for &(d, v) in &[(1usize, 3usize), (16, 4), (17, 13), (100, 70), (515, 33)] {
            let wf: Vec<f32> =
                randf(&mut s, v * d, 0.1).iter().map(|x| f32::from_bits(x.to_bits() & 0xFFFF_0000)).collect();
            let wb: Vec<Bf16> = wf.iter().map(|x| Bf16((x.to_bits() >> 16) as u16)).collect();
            let h = randf(&mut s, 9 * d, 4.0);
            let want = reference(&h, &wf, 9, d, v);
            for isa in ALL_ISAS {
                for threads in [1usize, 3] {
                    for r in [1usize, 4, 9] {
                        let got = lm_head_run(&h[..r * d], &wb, r, d, v, threads, isa);
                        assert_eq!(bits(&got), bits(&want[..r * v]), "d={d} v={v} r={r} t={threads} {isa:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn top_k_strict_matches_a_full_sort_or_declines() {
        let mut s = 0x7031u64;
        for n in [1usize, 2, 3, 5, 17, 100, 1000] {
            for trial in 0..40 {
                // few distinct values in some trials: ties are common there
                let levels = if trial % 2 == 0 { 7.0 } else { 1e6 };
                let mut a: Vec<f32> = randf(&mut s, n, 2.0).iter().map(|x| (x * levels).round() / levels).collect();
                if trial % 5 == 0 {
                    a[(rng(&mut s) as usize) % n] = f32::NEG_INFINITY;
                }
                if trial % 7 == 0 {
                    a[(rng(&mut s) as usize) % n] = -0.0;
                }
                let mut sorted: Vec<(f32, usize)> = a.iter().copied().zip(0..).collect();
                sorted.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.cmp(&y.1)));
                for k in [0usize, 1, 2, 3, 5, 20] {
                    let got = top_k_strict_run(&a, k);
                    if k > n {
                        assert!(got.is_none());
                        continue;
                    }
                    let m = (k + 1).min(n);
                    let unique = sorted[..m].windows(2).all(|p| p[0].0 > p[1].0);
                    match got {
                        None => assert!(!unique, "declined a unique top-{k} (n={n})"),
                        Some(g) => {
                            assert!(unique, "answered a tied top-{k} (n={n})");
                            let want: Vec<(usize, u32)> = sorted[..k].iter().map(|x| (x.1, x.0.to_bits())).collect();
                            let got: Vec<(usize, u32)> = g.iter().map(|x| (x.0, x.1.to_bits())).collect();
                            assert_eq!(got, want);
                        }
                    }
                }
            }
        }
        assert!(top_k_strict_run(&[1.0, f32::NAN, 0.5], 1).is_none());
        // a NaN far behind the top values (in a chunk the pre-filter would otherwise skip)
        let mut far: Vec<f32> = (0..1000).map(|i| -(i as f32)).collect();
        assert_eq!(top_k_strict_run(&far, 2), Some(vec![(0, 0.0), (1, -1.0)]));
        far[900] = f32::NAN;
        assert!(top_k_strict_run(&far, 2).is_none());
        assert!(top_k_strict_run(&[0.0, -0.0, -1.0], 1).is_none());
        assert_eq!(top_k_strict_run(&[0.0, -1.0, -1.0], 1), Some(vec![(0, 0.0)]));
    }

    #[test]
    fn empty_shapes() {
        assert!(lm_head_run(&[], &[1.0f32, 2.0], 0, 2, 1, 0, Isa::Auto).is_empty());
        assert_eq!(lm_head_run::<f32>(&[], &[], 2, 0, 3, 0, Isa::Auto), vec![0.0; 6]);
    }
}
