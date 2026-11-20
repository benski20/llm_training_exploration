//! Fused per-chunk finish of a masked linear call (replaces the loop body of
//! `MaskedBackend._finish`).
//!
//! For one chunk of `rows` tokens of a linear with `m` outputs and `n` inputs, [`finish_chunk`]
//! does in one pass over the service result:
//!
//! 1. decode `y_hat` from the wire (`K/8` little-endian bytes per element, row-major),
//! 2. Freivalds check `(y_hat @ u^T) & mask == (x_hat @ uw^T) & mask` for every row (if check
//!    vectors are given; an all-zero `u` means the reservation was closed),
//! 3. unmask `yq = centered((y_hat - wr) mod 2^K)` and the range check `|yq| <= bound`,
//! 4. dequantize `f32(((f64(yq) * sx[t]) * f64(scale[j])) + f64(bias[j]))`, the same IEEE
//!    operations in the same order as `pgw.core.ring.dequantize_outputs` (bit-identical).
//!
//! Verdict order matches Python: all-zero `u` (status 3), Freivalds (status 1), range (status 2).
//!
//! [`finish_group`] runs the same pass over several chunks of one linear whose rows were masked
//! from extents of the same segment (typically one chunk per concurrent session): the Freivalds
//! vectors depend only on the segment and the linear, so they are read from memory once for all
//! rows, while every chunk keeps its own `W r`. It also accepts the vectors and `W r` as 32-bit
//! words (`K <= 32`, half the memory traffic), `x_hat` as the request's wire bytes, and zeroes each
//! `W r` in place as soon as it has been used.
//!
//! Arithmetic is done in a machine word `W` (u32 for `K <= 32`, u64 above): every quantity only
//! matters mod `2^K` and `2^K` divides `2^32` / `2^64`, so wrapping arithmetic in `W` followed by a
//! final reduction is exact. Centering is a sign extension of the low `K` bits.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::mem::MaybeUninit;
use std::sync::OnceLock;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(finish_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(finish_group, m)?)?;
    Ok(())
}

/// Status codes returned by [`finish_chunk`].
pub const STATUS_OK: u8 = 0;
/// Freivalds check failed on at least one row.
pub const STATUS_FREIVALDS: u8 = 1;
/// A decoded output lies outside `[-bound, bound]`.
pub const STATUS_RANGE: u8 = 2;
/// The check vectors are all zero (reservation closed while the call was in flight).
pub const STATUS_U_ZERO: u8 = 3;

// ---- machine word ----------------------------------------------------------------------------

/// Wrapping ring word (u32 or u64).
pub trait Word: Copy + Send + Sync + Default + Eq + 'static {
    /// Bit width of the word.
    const BITS: u32;
    /// Truncate a u64 to the word.
    fn trunc(v: u64) -> Self;
    /// Wrapping multiply.
    fn wmul(self, o: Self) -> Self;
    /// Wrapping add.
    fn wadd(self, o: Self) -> Self;
    /// Wrapping subtract.
    fn wsub(self, o: Self) -> Self;
    /// Bitwise and.
    fn and(self, o: Self) -> Self;
    /// Sign-extend the low `BITS - shift` bits to i64 (`shift = BITS - K`).
    fn sext(self, shift: u32) -> i64;
    /// All-ones mask of the low `k` bits.
    fn mask(k: u32) -> Self;
    /// Zero-extend to u64.
    fn widen(self) -> u64;
}

impl Word for u32 {
    const BITS: u32 = 32;
    #[inline(always)]
    fn trunc(v: u64) -> Self {
        v as u32
    }
    #[inline(always)]
    fn wmul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn wadd(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn wsub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        self & o
    }
    #[inline(always)]
    fn sext(self, shift: u32) -> i64 {
        (((self << shift) as i32) >> shift) as i64
    }
    #[inline(always)]
    fn mask(k: u32) -> Self {
        if k >= 32 {
            u32::MAX
        } else {
            (1u32 << k) - 1
        }
    }
    #[inline(always)]
    fn widen(self) -> u64 {
        u64::from(self)
    }
}

impl Word for u64 {
    const BITS: u32 = 64;
    #[inline(always)]
    fn trunc(v: u64) -> Self {
        v
    }
    #[inline(always)]
    fn wmul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline(always)]
    fn wadd(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline(always)]
    fn wsub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        self & o
    }
    #[inline(always)]
    fn sext(self, shift: u32) -> i64 {
        ((self << shift) as i64) >> shift
    }
    #[inline(always)]
    fn mask(k: u32) -> Self {
        if k >= 64 {
            u64::MAX
        } else {
            (1u64 << k) - 1
        }
    }
    #[inline(always)]
    fn widen(self) -> u64 {
        self
    }
}

// ---- kernel ----------------------------------------------------------------------------------

/// Borrowed inputs of one chunk (all row-major, already validated).
#[derive(Clone, Copy)]
pub struct Chunk<'a> {
    /// Wire bytes, `rows * m * K/8`.
    pub y_hat: &'a [u8],
    /// `W r mod 2^K`, `(rows, m)` (any representative; only the low `K` bits are used).
    pub wr: &'a [u64],
    /// Masked inputs `(rows, n)` and check data `u (c, m)`, `uw (c, n)`, or `None`.
    pub checks: Option<Checks<'a>>,
    /// Per-row activation scale `(rows,)`.
    pub sx: &'a [f64],
    /// Per-output weight scale `(m,)`.
    pub scale: &'a [f32],
    /// Optional bias `(m,)`.
    pub bias: Option<&'a [f32]>,
    /// Ring width (multiple of 8 in `[8, 56]`).
    pub k: u32,
    /// Output bound (inclusive).
    pub bound: u64,
    /// Rows.
    pub rows: usize,
    /// Output features.
    pub m: usize,
}

/// Freivalds inputs.
#[derive(Clone, Copy)]
pub struct Checks<'a> {
    /// `(rows, n)`.
    pub x_hat: &'a [u64],
    /// `(c, m)`.
    pub u: &'a [u64],
    /// `(c, n)`.
    pub uw: &'a [u64],
    /// Number of check vectors.
    pub c: usize,
    /// Input features.
    pub n: usize,
}

/// Result of [`finish`]: `(out, status, bad_rows)`; `out` is `Some` only for [`STATUS_OK`].
pub type Finished = (Option<Vec<f32>>, u8, usize);

/// Row-major ring elements: 64-bit words, 32-bit words (`K <= 32`), or wire bytes (`K/8` LE each).
#[derive(Clone, Copy)]
pub enum Src<'a> {
    /// Words (only the low `K` bits are used).
    U64(&'a [u64]),
    /// 32-bit words (`K <= 32`).
    U32(&'a [u32]),
    /// Packed wire bytes, `K/8` little-endian bytes per element.
    Bytes(&'a [u8]),
}

impl<'a> Src<'a> {
    /// Number of elements (`nb` = bytes per packed element).
    pub fn elems(&self, nb: usize) -> usize {
        match self {
            Src::U64(s) => s.len(),
            Src::U32(s) => s.len(),
            Src::Bytes(s) => s.len() / nb,
        }
    }

    /// Elements `[a, b)` as words into `dst` (`dst.len() == b - a`).
    #[inline(always)]
    pub(crate) fn fill<W: Word, const NB: usize>(&self, a: usize, b: usize, dst: &mut [W]) {
        match *self {
            Src::U64(s) => {
                for (d, v) in dst.iter_mut().zip(&s[a..b]) {
                    *d = W::trunc(*v);
                }
            }
            Src::U32(s) => {
                for (d, v) in dst.iter_mut().zip(&s[a..b]) {
                    *d = W::trunc(u64::from(*v));
                }
            }
            Src::Bytes(s) => decode::<W, NB>(&s[a * NB..b * NB], dst),
        }
    }

    /// Elements `[a, b)` as words without a copy, when they are stored at the word width.
    #[inline(always)]
    pub(crate) fn words<W: Word>(&self, a: usize, b: usize) -> Option<&'a [W]> {
        let (ptr, ok) = match *self {
            Src::U32(s) if W::BITS == 32 => (s[a..b].as_ptr().cast::<W>(), true),
            Src::U64(s) if W::BITS == 64 => (s[a..b].as_ptr().cast::<W>(), true),
            _ => (std::ptr::null(), false),
        };
        if !ok || std::mem::size_of::<W>() * 8 != W::BITS as usize || std::mem::align_of::<W>() > W::BITS as usize / 8 {
            return None;
        }
        // SAFETY: `W` is a plain unsigned word of the same size (u32 / u64; every bit pattern is a
        // valid value) and no stricter alignment; the range was bounds-checked above.
        Some(unsafe { std::slice::from_raw_parts(ptr, b - a) })
    }

    /// True if every element is zero in its low `k` bits. Reads every element (no early exit, so the
    /// time does not depend on where the first nonzero word of the secret vectors lies).
    pub(crate) fn all_zero(&self, k: u32) -> bool {
        let mask = u64::mask(k);
        let acc = match *self {
            Src::U64(s) => s.iter().fold(0u64, |a, &v| a | (v & mask)),
            Src::U32(s) => s.iter().fold(0u64, |a, &v| a | (u64::from(v) & mask)),
            Src::Bytes(s) => s.iter().fold(0u64, |a, &v| a | u64::from(v)),
        };
        acc == 0
    }
}

/// `W r` of one part, read through a raw pointer so the kernel can also zero it (one-time-pad
/// derived) right after each block is used, while it is still in cache.
#[derive(Clone, Copy)]
pub struct WrBuf {
    ptr: *mut u8,
    len: usize,
    wide: bool,
    wipe: bool,
}

// SAFETY: workers read (and, with `wipe`, zero) pairwise disjoint column ranges of each row; the
// caller keeps the buffer alive and unaliased (no other references) until the kernel returns.
unsafe impl Send for WrBuf {}
unsafe impl Sync for WrBuf {}

impl WrBuf {
    /// A `(rows, m)` buffer of u64 words (`wide`) or u32 words, `len` elements.
    ///
    /// # Safety
    /// `ptr` must point to `len` valid, aligned elements that nothing else reads or writes until
    /// the kernel using this buffer returns.
    pub unsafe fn new(ptr: *mut u8, len: usize, wide: bool, wipe: bool) -> Self {
        WrBuf { ptr, len, wide, wipe }
    }

    /// Elements `[a, b)` into `dst`, then zero them if `wipe`.
    #[inline(always)]
    fn take<W: Word>(&self, a: usize, b: usize, dst: &mut [W]) {
        debug_assert!(a <= b && b <= self.len);
        if self.wide {
            // SAFETY: in bounds (validated shapes), aligned, exclusively owned by this worker's columns.
            let p = unsafe { self.ptr.cast::<u64>().add(a) };
            {
                let s = unsafe { std::slice::from_raw_parts(p, b - a) };
                for (d, v) in dst.iter_mut().zip(s) {
                    *d = W::trunc(*v);
                }
            }
            if self.wipe {
                for i in 0..b - a {
                    // SAFETY: as above; no reference into the range is alive.
                    unsafe { std::ptr::write_volatile(p.add(i), 0) };
                }
            }
        } else {
            let p = unsafe { self.ptr.cast::<u32>().add(a) };
            {
                let s = unsafe { std::slice::from_raw_parts(p, b - a) };
                for (d, v) in dst.iter_mut().zip(s) {
                    *d = W::trunc(u64::from(*v));
                }
            }
            if self.wipe {
                for i in 0..b - a {
                    // SAFETY: as above.
                    unsafe { std::ptr::write_volatile(p.add(i), 0) };
                }
            }
        }
    }

    /// Zero the whole buffer (paths that return before the workers run).
    pub(crate) fn wipe_all(&self) {
        if !self.wipe {
            return;
        }
        let bytes = self.len * if self.wide { 8 } else { 4 };
        for i in 0..bytes {
            // SAFETY: `bytes` bytes starting at `ptr` are valid and unaliased.
            unsafe { std::ptr::write_volatile(self.ptr.add(i), 0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

/// One part of a [`Group`]: consecutive rows of one call (one session's masks).
#[derive(Clone, Copy)]
pub struct Part<'a> {
    /// Wire bytes `(rows, m)`.
    pub y_hat: &'a [u8],
    /// `W r` `(rows, m)`.
    pub wr: WrBuf,
    /// Masked inputs `(rows, n)` (required when the group has check vectors).
    pub x_hat: Option<Src<'a>>,
    /// Per-row activation scale.
    pub sx: &'a [f64],
}

/// Freivalds vectors of a group: `u (c, m)` and `uw (c, n)`.
#[derive(Clone, Copy)]
pub struct GroupChecks<'a> {
    /// `(c, m)`.
    pub u: Src<'a>,
    /// `(c, n)`.
    pub uw: Src<'a>,
    /// Number of check vectors.
    pub c: usize,
    /// Input features.
    pub n: usize,
}

/// Several parts of the same linear whose rows were masked from extents with the same check
/// vectors (or that are not checked). Each part keeps its own `W r` (its own one-time pads); only
/// `u`/`uw`, which depend on the segment and the linear, are shared, so they are read once.
pub struct Group<'a> {
    /// Parts in output order.
    pub parts: Vec<Part<'a>>,
    /// Shared check vectors, or `None` (no check).
    pub checks: Option<GroupChecks<'a>>,
    /// Per-output weight scale `(m,)`.
    pub scale: &'a [f32],
    /// Optional bias `(m,)`.
    pub bias: Option<&'a [f32]>,
    /// Ring width.
    pub k: u32,
    /// Output bound (inclusive).
    pub bound: u64,
    /// Output features.
    pub m: usize,
}

/// Column block size (elements) for the cache-blocked passes.
pub(crate) const BLOCK: usize = 2048;
/// Roughly this many element-operations per worker thread before going parallel.
const WORK_PER_THREAD: usize = 1 << 19;

/// Overwrite a buffer of secret-derived words with zeros (not elided by the optimiser).
pub(crate) fn wipe<T: Copy + Default>(buf: &mut [T]) {
    for v in buf.iter_mut() {
        // SAFETY: `v` is a valid, aligned, exclusive reference.
        unsafe { std::ptr::write_volatile(v, T::default()) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Raw output pointer shared by workers that write disjoint elements.
#[derive(Clone, Copy)]
pub(crate) struct OutPtr(pub(crate) *mut f32);
// SAFETY: workers write pairwise disjoint index sets; the scope joins them before the Vec is read.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// What one worker returns: partial Freivalds sums `(rows*c)` for both sides, and min/max of `yq`.
pub(crate) struct Partial<W> {
    lhs: Vec<W>,
    rhs: Vec<W>,
    ymin: i64,
    ymax: i64,
}

#[inline(always)]
pub(crate) fn dot<W: Word>(a: &[W], b: &[W]) -> W {
    let mut s = W::default();
    for (x, y) in a.iter().zip(b) {
        s = s.wadd(x.wmul(*y));
    }
    s
}

#[inline(always)]
pub(crate) fn decode<W: Word, const NB: usize>(src: &[u8], dst: &mut [W]) {
    for (d, ch) in dst.iter_mut().zip(src.chunks_exact(NB)) {
        let mut a = [0u8; 8];
        a[..NB].copy_from_slice(ch);
        *d = W::trunc(u64::from_le_bytes(a));
    }
}

/// Worker body: output columns `[ja, jb)` of every row, input columns `[na, nb)` of the RHS check.
#[inline(always)]
fn work<W: Word, const NB: usize>(
    g: &Group<'_>,
    rows: &[(usize, usize)],
    out: OutPtr,
    ja: usize,
    jb: usize,
    na: usize,
    nb: usize,
) -> Partial<W> {
    let m = g.m;
    let nrows = rows.len();
    let shift = W::BITS - g.k;
    let c = g.checks.map_or(0, |ck| ck.c);
    let mut lhs = vec![W::default(); nrows * c];
    let mut rhs = vec![W::default(); nrows * c];
    let mut ybuf = vec![W::default(); BLOCK];
    let mut wbuf = vec![W::default(); BLOCK];
    let mut ubuf = vec![W::default(); BLOCK * c];
    let (mut ymin, mut ymax) = (i64::MAX, i64::MIN);

    let mut b0 = ja;
    while b0 < jb {
        let b1 = (b0 + BLOCK).min(jb);
        let len = b1 - b0;
        let mut u_direct = false;
        if let Some(ck) = g.checks {
            u_direct = ck.u.words::<W>(0, 0).is_some();
            if !u_direct {
                for i in 0..c {
                    ck.u.fill::<W, NB>(i * m + b0, i * m + b1, &mut ubuf[i * BLOCK..i * BLOCK + len]);
                }
            }
        }
        let scale = &g.scale[b0..b1];
        for (gt, &(pi, t)) in rows.iter().enumerate() {
            let p = &g.parts[pi];
            let yb = &mut ybuf[..len];
            decode::<W, NB>(&p.y_hat[(t * m + b0) * NB..(t * m + b1) * NB], yb);
            let wb = &mut wbuf[..len];
            p.wr.take::<W>(t * m + b0, t * m + b1, wb);
            let sxt = p.sx[t];
            // SAFETY: row gt, columns [b0, b1) lie in the allocation and are written by this worker only.
            let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(gt * m + b0).cast::<MaybeUninit<f32>>(), len) };
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            match g.bias {
                Some(bias) => {
                    let bias = &bias[b0..b1];
                    for j in 0..len {
                        let yq = yb[j].wsub(wb[j]).sext(shift);
                        lo = lo.min(yq);
                        hi = hi.max(yq);
                        o[j].write(((yq as f64 * sxt) * scale[j] as f64 + bias[j] as f64) as f32);
                    }
                }
                None => {
                    for j in 0..len {
                        let yq = yb[j].wsub(wb[j]).sext(shift);
                        lo = lo.min(yq);
                        hi = hi.max(yq);
                        o[j].write(((yq as f64 * sxt) * scale[j] as f64) as f32);
                    }
                }
            }
            ymin = ymin.min(lo);
            ymax = ymax.max(hi);
            for i in 0..c {
                let us = match (u_direct, g.checks) {
                    (true, Some(ck)) => ck.u.words::<W>(i * m + b0, i * m + b1).unwrap_or(&[]),
                    _ => &ubuf[i * BLOCK..i * BLOCK + len],
                };
                let s = dot(yb, us);
                lhs[gt * c + i] = lhs[gt * c + i].wadd(s);
            }
        }
        b0 = b1;
    }

    if let Some(ck) = g.checks {
        let n = ck.n;
        let mut b0 = na;
        while b0 < nb {
            let b1 = (b0 + BLOCK).min(nb);
            let len = b1 - b0;
            let uw_direct = ck.uw.words::<W>(0, 0).is_some();
            if !uw_direct {
                for i in 0..c {
                    ck.uw.fill::<W, NB>(i * n + b0, i * n + b1, &mut ubuf[i * BLOCK..i * BLOCK + len]);
                }
            }
            for (gt, &(pi, t)) in rows.iter().enumerate() {
                let xb = &mut ybuf[..len];
                // validated: every part has x_hat when the group has checks
                if let Some(x) = g.parts[pi].x_hat {
                    x.fill::<W, NB>(t * n + b0, t * n + b1, xb);
                }
                for i in 0..c {
                    let us = if uw_direct {
                        ck.uw.words::<W>(i * n + b0, i * n + b1).unwrap_or(&[])
                    } else {
                        &ubuf[i * BLOCK..i * BLOCK + len]
                    };
                    let s = dot(xb, us);
                    rhs[gt * c + i] = rhs[gt * c + i].wadd(s);
                }
            }
            b0 = b1;
        }
    }
    wipe(&mut ybuf);
    wipe(&mut wbuf);
    wipe(&mut ubuf);
    Partial { lhs, rhs, ymin, ymax }
}

/// Same as [`work`], compiled for AVX2 (vpmulld for the 32-bit dot products, wider f64 lanes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn work_avx2<W: Word, const NB: usize>(
    g: &Group<'_>,
    rows: &[(usize, usize)],
    out: OutPtr,
    ja: usize,
    jb: usize,
    na: usize,
    nb: usize,
) -> Partial<W> {
    // `fma` only widens the instruction set: Rust never contracts `a * b + c` into an FMA.
    work::<W, NB>(g, rows, out, ja, jb, na, nb)
}

fn work_dispatch<W: Word, const NB: usize>(
    g: &Group<'_>,
    rows: &[(usize, usize)],
    out: OutPtr,
    ja: usize,
    jb: usize,
    na: usize,
    nb: usize,
) -> Partial<W> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports the enabled features.
            return unsafe { work_avx2::<W, NB>(g, rows, out, ja, jb, na, nb) };
        }
    }
    work::<W, NB>(g, rows, out, ja, jb, na, nb)
}

/// Automatic worker count: the available parallelism (queried once; it can read cgroup files).
fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |v| v.get()).min(16))
}

/// Split `[0, len)` into `parts` ranges aligned to 64 elements.
fn split(len: usize, parts: usize, i: usize) -> (usize, usize) {
    let per = len.div_ceil(parts).div_ceil(64) * 64;
    ((i * per).min(len), ((i + 1) * per).min(len))
}

/// How the column blocks of a finish are spread over threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exec {
    /// Scoped threads spawned for this call (`0`: automatic count): the [`finish_group`] behaviour.
    Scoped(usize),
    /// The persistent prepare pool ([`crate::prepare::pool::global`]): `1` keeps the call on the
    /// calling thread, `0` uses as many pool workers as the work warrants.
    Pool(usize),
}

fn run_group<W: Word, const NB: usize>(g: &Group<'_>, threads: usize) -> Finished {
    let nrows: usize = g.parts.iter().map(|p| p.sx.len()).sum();
    // Uninitialised: workers write every element (their column ranges partition `[0, m)`), and a
    // plain malloc can reuse warm memory instead of faulting in fresh zeroed pages each call.
    let mut out: Vec<f32> = Vec::with_capacity(nrows * g.m);
    let (status, bad_rows) = run_group_core::<W, NB>(g, OutPtr(out.as_mut_ptr()), Exec::Scoped(threads), None);
    if status == STATUS_U_ZERO {
        return (None, status, bad_rows);
    }
    // SAFETY: every element of `out` was written by exactly one worker (joined in the core), and
    // after a failed verdict the core zeroed all of them again.
    unsafe { out.set_len(nrows * g.m) };
    if status != STATUS_OK {
        return (None, status, bad_rows);
    }
    (Some(out), STATUS_OK, 0)
}

/// The fused finish of a validated group into `out` (`sum rows x m` float32, valid for writes).
///
/// Returns `(status, bad_rows)`. Every output element is written unless the status is
/// [`STATUS_U_ZERO`] (nothing is written then); after any other failed verdict the outputs are
/// zeroed again before returning. With `rhs_pre` (`rows x c` words: the Freivalds input half
/// `uw . x_hat`, computed at prepare time) the check compares `u . y_hat` against it and the parts
/// need no `x_hat`.
pub(crate) fn run_group_core<W: Word, const NB: usize>(
    g: &Group<'_>,
    out: OutPtr,
    exec: Exec,
    rhs_pre: Option<&[u64]>,
) -> (u8, usize) {
    let m = g.m;
    let rows: Vec<(usize, usize)> =
        g.parts.iter().enumerate().flat_map(|(pi, p)| (0..p.sx.len()).map(move |t| (pi, t))).collect();
    let nrows = rows.len();
    let c = g.checks.map_or(0, |ck| ck.c);
    // with a precomputed input half the workers skip the `uw . x_hat` pass
    let n = if rhs_pre.is_some() { 0 } else { g.checks.map_or(0, |ck| ck.n) };
    if let Some(ck) = g.checks {
        if ck.u.all_zero(g.k) {
            for p in &g.parts {
                p.wr.wipe_all();
            }
            return (STATUS_U_ZERO, 0);
        }
    }
    let ptr = out;
    let work_units = nrows * (m * (c + 2) + n * c);
    let cap_cols = m.div_ceil(64).max(1);

    let mut parts: Vec<Partial<W>> = match exec {
        Exec::Scoped(threads) => {
            let max_t = if threads == 0 { default_threads() } else { threads };
            let t = (work_units / WORK_PER_THREAD).clamp(1, max_t.max(1)).min(cap_cols);
            if t == 1 {
                vec![work_dispatch::<W, NB>(g, &rows, ptr, 0, m, 0, n)]
            } else {
                let rows = &rows;
                std::thread::scope(|s| {
                    let hs: Vec<_> = (0..t)
                        .map(|i| {
                            let (ja, jb) = split(m, t, i);
                            let (na, nb) = split(n, t, i);
                            s.spawn(move || work_dispatch::<W, NB>(g, rows, ptr, ja, jb, na, nb))
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().expect("finish worker panicked")).collect()
                })
            }
        }
        Exec::Pool(threads) => {
            let pool = crate::prepare::pool::global();
            let max_t = if threads == 0 { pool.workers() + 1 } else { threads };
            let t = (work_units / WORK_PER_THREAD).clamp(1, max_t.max(1)).min(cap_cols);
            if t == 1 {
                vec![work_dispatch::<W, NB>(g, &rows, ptr, 0, m, 0, n)]
            } else {
                let slots: Vec<std::sync::Mutex<Option<Partial<W>>>> =
                    (0..t).map(|_| std::sync::Mutex::new(None)).collect();
                let rows = &rows;
                let ok = pool.run(t, &|i| {
                    let (ja, jb) = split(m, t, i);
                    let (na, nb) = split(n, t, i);
                    let pt = work_dispatch::<W, NB>(g, rows, ptr, ja, jb, na, nb);
                    *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(pt);
                });
                let got: Vec<Partial<W>> =
                    slots.into_iter().filter_map(|s| s.into_inner().unwrap_or_else(|e| e.into_inner())).collect();
                if !ok || got.len() != t {
                    // SAFETY: the pool returned after every task; `nrows * m` elements are valid.
                    unsafe { zero_out(ptr, nrows * m) };
                    panic!("finish worker panicked");
                }
                got
            }
        }
    };
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    let (mut ymin, mut ymax) = (i64::MAX, i64::MIN);
    let mut status = STATUS_OK;
    let mut bad_rows = 0usize;
    if c > 0 {
        let mask = W::mask(g.k);
        let mut lhs = vec![W::default(); nrows * c];
        let mut rhs = vec![W::default(); nrows * c];
        for pt in &parts {
            for (a, b) in lhs.iter_mut().zip(&pt.lhs) {
                *a = a.wadd(*b);
            }
            for (a, b) in rhs.iter_mut().zip(&pt.rhs) {
                *a = a.wadd(*b);
            }
        }
        if let Some(pre) = rhs_pre {
            for (a, b) in rhs.iter_mut().zip(pre) {
                *a = a.wadd(W::trunc(*b));
            }
        }
        for tr in 0..nrows {
            // every check vector is compared (no early exit on the first mismatch)
            let row_bad = (0..c).fold(false, |acc, i| acc | (lhs[tr * c + i].and(mask) != rhs[tr * c + i].and(mask)));
            bad_rows += usize::from(row_bad);
        }
        wipe(&mut lhs);
        wipe(&mut rhs);
        if bad_rows > 0 {
            status = STATUS_FREIVALDS;
        }
    }
    for pt in parts.iter_mut() {
        ymin = ymin.min(pt.ymin);
        ymax = ymax.max(pt.ymax);
        wipe(&mut pt.lhs);
        wipe(&mut pt.rhs);
    }
    let bound = g.bound.min(i64::MAX as u64) as i64;
    if status == STATUS_OK && nrows * m > 0 && (ymax > bound || ymin < -bound) {
        status = STATUS_RANGE;
    }
    if status != STATUS_OK {
        // SAFETY: every element was written by a worker; zero them again (fail closed).
        unsafe { zero_out(ptr, nrows * m) };
        return (status, bad_rows);
    }
    (STATUS_OK, 0)
}

/// Zero `len` floats at `p` (volatile: they are unmasked outputs of a rejected call).
///
/// # Safety
/// `p` must be valid for `len` writes and not aliased by a live reference.
unsafe fn zero_out(p: OutPtr, len: usize) {
    for i in 0..len {
        std::ptr::write_volatile(p.0.add(i), 0.0);
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// [`run_group_core`] for a validated group of any supported `k` (panics on another width).
///
/// # Safety
/// `out` must be valid for `sum rows x m` float32 writes and not aliased until this returns.
pub(crate) unsafe fn finish_group_into_raw(
    g: &Group<'_>,
    out: *mut f32,
    exec: Exec,
    rhs_pre: Option<&[u64]>,
) -> (u8, usize) {
    let o = OutPtr(out);
    match g.k {
        8 => run_group_core::<u32, 1>(g, o, exec, rhs_pre),
        16 => run_group_core::<u32, 2>(g, o, exec, rhs_pre),
        24 => run_group_core::<u32, 3>(g, o, exec, rhs_pre),
        32 => run_group_core::<u32, 4>(g, o, exec, rhs_pre),
        40 => run_group_core::<u64, 5>(g, o, exec, rhs_pre),
        48 => run_group_core::<u64, 6>(g, o, exec, rhs_pre),
        56 => run_group_core::<u64, 7>(g, o, exec, rhs_pre),
        k => panic!("unsupported ring width {k}"),
    }
}

/// Run the fused finish of a validated group (`threads = 0`: automatic). Output rows are the
/// parts' rows in order.
pub fn finish_group_run(g: &Group<'_>, threads: usize) -> Finished {
    match g.k {
        8 => run_group::<u32, 1>(g, threads),
        16 => run_group::<u32, 2>(g, threads),
        24 => run_group::<u32, 3>(g, threads),
        32 => run_group::<u32, 4>(g, threads),
        40 => run_group::<u64, 5>(g, threads),
        48 => run_group::<u64, 6>(g, threads),
        56 => run_group::<u64, 7>(g, threads),
        k => panic!("unsupported ring width {k}"),
    }
}

/// Run the fused finish on validated inputs (`threads = 0`: automatic).
pub fn finish(p: &Chunk<'_>, threads: usize) -> Finished {
    // SAFETY: read-only use (`wipe = false`): the kernel never writes through the pointer.
    let wr = unsafe { WrBuf::new(p.wr.as_ptr() as *mut u8, p.wr.len(), true, false) };
    let g = Group {
        parts: vec![Part { y_hat: p.y_hat, wr, x_hat: p.checks.map(|ck| Src::U64(ck.x_hat)), sx: &p.sx[..p.rows] }],
        checks: p.checks.map(|ck| GroupChecks { u: Src::U64(ck.u), uw: Src::U64(ck.uw), c: ck.c, n: ck.n }),
        scale: p.scale,
        bias: p.bias,
        k: p.k,
        bound: p.bound,
        m: p.m,
    };
    finish_group_run(&g, threads)
}

// ---- Python binding --------------------------------------------------------------------------

pub(crate) fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Reinterpret an int64 slice as uint64 (same size and alignment).
pub(crate) fn i64_as_u64(s: &[i64]) -> &[u64] {
    // SAFETY: i64 and u64 have identical size, alignment and no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u64>(), s.len()) }
}

/// A read-only 2-D int64 or uint64 array (zero-copy).
enum Ring2<'py> {
    I(numpy::PyReadonlyArray2<'py, i64>),
    U(numpy::PyReadonlyArray2<'py, u64>),
}

impl<'py> Ring2<'py> {
    fn extract(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Self> {
        if let Ok(a) = obj.downcast::<PyArray2<i64>>() {
            return Ok(Ring2::I(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        if let Ok(a) = obj.downcast::<PyArray2<u64>>() {
            return Ok(Ring2::U(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        Err(verr(format!("{name} must be a 2-D int64 or uint64 numpy array")))
    }
    fn shape(&self) -> (usize, usize) {
        let s = match self {
            Ring2::I(a) => a.shape(),
            Ring2::U(a) => a.shape(),
        };
        (s[0], s[1])
    }
    fn slice(&self, name: &str) -> PyResult<&[u64]> {
        let c_contig = match self {
            Ring2::I(a) => a.is_c_contiguous(),
            Ring2::U(a) => a.is_c_contiguous(),
        };
        if !c_contig {
            return Err(verr(format!("{name} must be C-contiguous")));
        }
        match self {
            Ring2::I(a) => a.as_slice().map(i64_as_u64),
            Ring2::U(a) => a.as_slice(),
        }
        .map_err(|_| verr(format!("{name} must be C-contiguous")))
    }
}

pub(crate) fn arr1<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    ty: &str,
) -> PyResult<numpy::PyReadonlyArray1<'py, T>> {
    let a = obj.downcast::<PyArray1<T>>().map_err(|_| verr(format!("{name} must be a 1-D {ty} numpy array")))?;
    a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))
}

pub(crate) fn slice1<'a, T: numpy::Element>(a: &'a numpy::PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| verr(format!("{name} must be C-contiguous")))
}

/// Python return value of [`finish_chunk`]: `(out, status, bad_rows)`.
type PyFinished<'py> = (Option<Bound<'py, PyArray2<f32>>>, u8, usize);

/// Fused per-chunk finish of a masked linear call.
///
/// Args:
///     y_hat: service result wire bytes (``rows * m * k/8``).
///     wr: ``W r mod 2^K``, int64 (or uint64) ``(rows, m)``.
///     x_hat: masked inputs int64 (or uint64) ``(rows, n)``, or None (required with check vectors).
///     u, uw: Freivalds check vectors uint64 ``(c, m)`` and ``(c, n)``, or both None (no check).
///     sx: float64 ``(rows,)``; scale: float32 ``(m,)``; bias: float32 ``(m,)`` or None.
///     k: ring width (multiple of 8 in [8, 56]); bound: inclusive output bound.
///     threads: worker threads (0 = automatic).
///
/// Returns ``(out, status, bad_rows)``: ``out`` float32 ``(rows, m)`` when ``status == 0``, else
/// None; status 1 = Freivalds failed (``bad_rows`` rows), 2 = output range, 3 = ``u`` all zero.
#[pyfunction]
#[pyo3(signature = (y_hat, wr, x_hat, u, uw, sx, scale, bias, k, bound, threads = 0))]
#[allow(clippy::too_many_arguments)]
fn finish_chunk<'py>(
    py: Python<'py>,
    y_hat: &Bound<'py, PyAny>,
    wr: &Bound<'py, PyAny>,
    x_hat: Option<&Bound<'py, PyAny>>,
    u: Option<&Bound<'py, PyAny>>,
    uw: Option<&Bound<'py, PyAny>>,
    sx: &Bound<'py, PyAny>,
    scale: &Bound<'py, PyAny>,
    bias: Option<&Bound<'py, PyAny>>,
    k: u32,
    bound: u64,
    threads: usize,
) -> PyResult<PyFinished<'py>> {
    if k % 8 != 0 || !(8..=56).contains(&k) {
        return Err(verr(format!("ring width k={k} must be a multiple of 8 in [8, 56]")));
    }
    let y_bytes = y_hat.downcast::<PyBytes>().map_err(|_| verr("y_hat must be bytes"))?;
    let y_hat: &[u8] = y_bytes.as_bytes();
    let wr_a = Ring2::extract(wr, "wr")?;
    let (rows, m) = wr_a.shape();
    let wr_s = wr_a.slice("wr")?;
    let expect = rows * m * (k as usize / 8);
    if y_hat.len() != expect {
        return Err(verr(format!(
            "ring payload has {} bytes, expected {expect} for shape ({rows}, {m}) K={k}",
            y_hat.len()
        )));
    }
    let sx_a = arr1::<f64>(sx, "sx", "float64")?;
    let sx_s = slice1(&sx_a, "sx")?;
    if sx_s.len() != rows {
        return Err(verr(format!("sx has {} entries, expected {rows}", sx_s.len())));
    }
    let sc_a = arr1::<f32>(scale, "scale", "float32")?;
    let sc_s = slice1(&sc_a, "scale")?;
    if sc_s.len() != m {
        return Err(verr(format!("scale has {} entries, expected {m}", sc_s.len())));
    }
    let bias_a = bias.map(|b| arr1::<f32>(b, "bias", "float32")).transpose()?;
    let bias_s = bias_a.as_ref().map(|b| slice1(b, "bias")).transpose()?;
    if let Some(b) = bias_s {
        if b.len() != m {
            return Err(verr(format!("bias has {} entries, expected {m}", b.len())));
        }
    }
    let (xa, ua, uwa) = match (u, uw) {
        (None, None) => (None, None, None),
        (Some(u), Some(uw)) => {
            let x = x_hat.ok_or_else(|| verr("x_hat is required with check vectors"))?;
            (Some(Ring2::extract(x, "x_hat")?), Some(Ring2::extract(u, "u")?), Some(Ring2::extract(uw, "uw")?))
        }
        _ => return Err(verr("u and uw must both be given or both be None")),
    };
    let checks = match (&xa, &ua, &uwa) {
        (Some(xa), Some(ua), Some(uwa)) => {
            let (xr, n) = xa.shape();
            let (c, um) = ua.shape();
            let (c2, n2) = uwa.shape();
            if xr != rows || um != m || c2 != c || n2 != n {
                return Err(verr(format!(
                    "check shapes do not match: x_hat ({xr}, {n}), u ({c}, {um}), uw ({c2}, {n2}), rows={rows} m={m}"
                )));
            }
            if c < 1 {
                return Err(verr("Freivalds check needs at least one check vector (c=0)"));
            }
            Some(Checks { x_hat: xa.slice("x_hat")?, u: ua.slice("u")?, uw: uwa.slice("uw")?, c, n })
        }
        _ => None,
    };
    let chunk = Chunk { y_hat, wr: wr_s, checks, sx: sx_s, scale: sc_s, bias: bias_s, k, bound, rows, m };
    let (out, status, bad) = py.allow_threads(|| finish(&chunk, threads));
    let out = match out {
        Some(v) => {
            let a = Array2::from_shape_vec((rows, m), v).map_err(|e| verr(e.to_string()))?;
            Some(a.into_pyarray(py))
        }
        None => None,
    };
    Ok((out, status, bad))
}

/// A read-only ring matrix argument of [`finish_group`]: int64/uint64/uint32 array or wire bytes.
pub(crate) enum RoRing<'py> {
    I(numpy::PyReadonlyArray2<'py, i64>),
    U(numpy::PyReadonlyArray2<'py, u64>),
    U32(numpy::PyReadonlyArray2<'py, u32>),
    B(Bound<'py, PyBytes>),
}

impl<'py> RoRing<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>, name: &str, k: u32) -> PyResult<Self> {
        if let Ok(b) = obj.downcast::<PyBytes>() {
            return Ok(RoRing::B(b.clone()));
        }
        if let Ok(a) = obj.downcast::<PyArray2<i64>>() {
            return Ok(RoRing::I(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        if let Ok(a) = obj.downcast::<PyArray2<u64>>() {
            return Ok(RoRing::U(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        if let Ok(a) = obj.downcast::<PyArray2<u32>>() {
            if k > 32 {
                return Err(verr(format!("{name}: uint32 arrays need K <= 32")));
            }
            return Ok(RoRing::U32(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        Err(verr(format!("{name} must be bytes or a 2-D int64/uint64/uint32 numpy array")))
    }

    /// Elements and, for arrays, the 2-D shape.
    pub(crate) fn src(&self, name: &str) -> PyResult<(Src<'_>, Option<(usize, usize)>)> {
        let cerr = || verr(format!("{name} must be C-contiguous"));
        let c_order = match self {
            RoRing::I(a) => a.is_c_contiguous(),
            RoRing::U(a) => a.is_c_contiguous(),
            RoRing::U32(a) => a.is_c_contiguous(),
            RoRing::B(_) => true,
        };
        if !c_order {
            // `as_slice` also accepts Fortran order, which would be read transposed
            return Err(cerr());
        }
        Ok(match self {
            RoRing::I(a) => {
                (Src::U64(i64_as_u64(a.as_slice().map_err(|_| cerr())?)), Some((a.shape()[0], a.shape()[1])))
            }
            RoRing::U(a) => (Src::U64(a.as_slice().map_err(|_| cerr())?), Some((a.shape()[0], a.shape()[1]))),
            RoRing::U32(a) => (Src::U32(a.as_slice().map_err(|_| cerr())?), Some((a.shape()[0], a.shape()[1]))),
            RoRing::B(b) => (Src::Bytes(b.as_bytes()), None),
        })
    }
}

/// The `W r` argument of [`finish_group`]: writable when the kernel wipes it.
pub(crate) enum WrArg<'py> {
    RwI(numpy::PyReadwriteArray2<'py, i64>),
    RwU(numpy::PyReadwriteArray2<'py, u64>),
    RwU32(numpy::PyReadwriteArray2<'py, u32>),
    Ro(RoRing<'py>),
}

impl<'py> WrArg<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>, k: u32, wipe: bool) -> PyResult<Self> {
        if !wipe {
            let r = RoRing::extract(obj, "wr", k)?;
            if matches!(r, RoRing::B(_)) {
                return Err(verr("wr must be a 2-D int64/uint64/uint32 numpy array"));
            }
            return Ok(WrArg::Ro(r));
        }
        let rw = |e: numpy::BorrowError| verr(format!("wr must be writable to be wiped: {e}"));
        if let Ok(a) = obj.downcast::<PyArray2<i64>>() {
            return Ok(WrArg::RwI(a.try_readwrite().map_err(rw)?));
        }
        if let Ok(a) = obj.downcast::<PyArray2<u64>>() {
            return Ok(WrArg::RwU(a.try_readwrite().map_err(rw)?));
        }
        if let Ok(a) = obj.downcast::<PyArray2<u32>>() {
            if k > 32 {
                return Err(verr("wr: uint32 arrays need K <= 32"));
            }
            return Ok(WrArg::RwU32(a.try_readwrite().map_err(rw)?));
        }
        Err(verr("wr must be a 2-D int64/uint64/uint32 numpy array"))
    }

    /// `(buffer, (rows, m))`.
    pub(crate) fn buf(&mut self, wipe: bool) -> PyResult<(WrBuf, (usize, usize))> {
        let cerr = || verr("wr must be C-contiguous");
        let c_order = match self {
            WrArg::RwI(a) => a.is_c_contiguous(),
            WrArg::RwU(a) => a.is_c_contiguous(),
            WrArg::RwU32(a) => a.is_c_contiguous(),
            WrArg::Ro(_) => true, // checked in `RoRing::src`
        };
        if !c_order {
            return Err(cerr());
        }
        // SAFETY (all arms): the borrow guard held in `self` keeps the array alive and excludes other
        // numpy borrows until the kernel returns; the pointer covers exactly the slice's elements.
        Ok(match self {
            WrArg::RwI(a) => {
                let sh = (a.shape()[0], a.shape()[1]);
                let s = a.as_slice_mut().map_err(|_| cerr())?;
                (unsafe { WrBuf::new(s.as_mut_ptr().cast(), s.len(), true, wipe) }, sh)
            }
            WrArg::RwU(a) => {
                let sh = (a.shape()[0], a.shape()[1]);
                let s = a.as_slice_mut().map_err(|_| cerr())?;
                (unsafe { WrBuf::new(s.as_mut_ptr().cast(), s.len(), true, wipe) }, sh)
            }
            WrArg::RwU32(a) => {
                let sh = (a.shape()[0], a.shape()[1]);
                let s = a.as_slice_mut().map_err(|_| cerr())?;
                (unsafe { WrBuf::new(s.as_mut_ptr().cast(), s.len(), false, wipe) }, sh)
            }
            WrArg::Ro(r) => {
                let (src, sh) = r.src("wr")?;
                let sh = sh.ok_or_else(|| verr("wr must be an array"))?;
                // `wipe` is false here: the kernel only reads through the pointer.
                match src {
                    Src::U64(s) => (unsafe { WrBuf::new(s.as_ptr() as *mut u8, s.len(), true, false) }, sh),
                    Src::U32(s) => (unsafe { WrBuf::new(s.as_ptr() as *mut u8, s.len(), false, false) }, sh),
                    Src::Bytes(_) => return Err(verr("wr must be an array")),
                }
            }
        })
    }
}

/// Fused finish of several chunks of one linear that share Freivalds vectors (one session each).
///
/// Args:
///     parts: sequence of ``(y_hat, wr, x_hat, sx)``: service wire bytes ``(rows_i, m)``; ``W r`` as a
///         2-D int64/uint64/uint32 array ``(rows_i, m)``; masked inputs ``(rows_i, n)`` as wire bytes
///         (the request payload) or a 2-D array, or None when unchecked; float64 ``sx (rows_i,)``.
///     u, uw: shared check vectors ``(c, m)`` / ``(c, n)`` (uint64, int64 or, for K <= 32, uint32),
///         or both None (no check).
///     scale, bias, k, bound: as in :func:`finish_chunk`.
///     wipe_wr: zero every ``wr`` in place (one-time-pad derived) as soon as it is used; the arrays
///         must then be writable. On a raised error nothing is wiped; on any returned status all are.
///     threads: worker threads (0 = automatic).
///
/// Returns ``(out, status, bad_rows)``: ``out`` float32 ``(sum rows_i, m)`` (parts' rows in order)
/// when ``status == 0``; statuses as :func:`finish_chunk`.
#[pyfunction]
#[pyo3(signature = (parts, u, uw, scale, bias, k, bound, wipe_wr = true, threads = 0))]
#[allow(clippy::too_many_arguments)]
fn finish_group<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyAny>,
    u: Option<&Bound<'py, PyAny>>,
    uw: Option<&Bound<'py, PyAny>>,
    scale: &Bound<'py, PyAny>,
    bias: Option<&Bound<'py, PyAny>>,
    k: u32,
    bound: u64,
    wipe_wr: bool,
    threads: usize,
) -> PyResult<PyFinished<'py>> {
    if k % 8 != 0 || !(8..=56).contains(&k) {
        return Err(verr(format!("ring width k={k} must be a multiple of 8 in [8, 56]")));
    }
    let nbytes = (k / 8) as usize;
    let sc_a = arr1::<f32>(scale, "scale", "float32")?;
    let sc_s = slice1(&sc_a, "scale")?;
    let m = sc_s.len();
    let bias_a = bias.map(|b| arr1::<f32>(b, "bias", "float32")).transpose()?;
    let bias_s = bias_a.as_ref().map(|b| slice1(b, "bias")).transpose()?;
    if let Some(b) = bias_s {
        if b.len() != m {
            return Err(verr(format!("bias has {} entries, expected {m}", b.len())));
        }
    }
    let (ua, uwa) = match (u, uw) {
        (None, None) => (None, None),
        (Some(u), Some(uw)) => (Some(RoRing::extract(u, "u", k)?), Some(RoRing::extract(uw, "uw", k)?)),
        _ => return Err(verr("u and uw must both be given or both be None")),
    };
    let checks = match (&ua, &uwa) {
        (Some(ua), Some(uwa)) => {
            let (us, ush) = ua.src("u")?;
            let (uws, uwsh) = uwa.src("uw")?;
            let ((c, um), (c2, n)) = match (ush, uwsh) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(verr("u and uw must be 2-D arrays")),
            };
            if um != m || c2 != c {
                return Err(verr(format!("check shapes do not match: u ({c}, {um}), uw ({c2}, {n}), m={m}")));
            }
            if c < 1 {
                return Err(verr("Freivalds check needs at least one check vector (c=0)"));
            }
            Some(GroupChecks { u: us, uw: uws, c, n })
        }
        _ => None,
    };

    // Borrow every part (guards live until the kernel returns), then build the kernel view.
    let mut ys: Vec<Bound<'py, PyBytes>> = Vec::new();
    let mut wrs: Vec<WrArg<'py>> = Vec::new();
    let mut xs: Vec<Option<RoRing<'py>>> = Vec::new();
    let mut sxs: Vec<numpy::PyReadonlyArray1<'py, f64>> = Vec::new();
    for item in parts.try_iter()? {
        let item = item?;
        let (y, wr, x, sx): (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>) =
            item.extract().map_err(|_| verr("each part must be a (y_hat, wr, x_hat, sx) tuple"))?;
        ys.push(y.downcast::<PyBytes>().map_err(|_| verr("y_hat must be bytes"))?.clone());
        wrs.push(WrArg::extract(&wr, k, wipe_wr)?);
        xs.push(if x.is_none() { None } else { Some(RoRing::extract(&x, "x_hat", k)?) });
        sxs.push(arr1::<f64>(&sx, "sx", "float64")?);
    }
    let mut kparts: Vec<Part<'_>> = Vec::with_capacity(ys.len());
    for (i, ((y, wr), (x, sx))) in ys.iter().zip(wrs.iter_mut()).zip(xs.iter().zip(sxs.iter())).enumerate() {
        let y_hat = y.as_bytes();
        let sx_s = slice1(sx, "sx")?;
        let rows = sx_s.len();
        let (wrb, (wrows, wm)) = wr.buf(wipe_wr)?;
        if wrows != rows || wm != m {
            return Err(verr(format!("part {i}: wr shape ({wrows}, {wm}) != ({rows}, {m})")));
        }
        if y_hat.len() != rows * m * nbytes {
            return Err(verr(format!(
                "part {i}: ring payload has {} bytes, expected {} for shape ({rows}, {m}) K={k}",
                y_hat.len(),
                rows * m * nbytes
            )));
        }
        let x_hat = match (x, &checks) {
            (Some(x), Some(ck)) => {
                let (s, sh) = x.src("x_hat")?;
                if let Some((xr, xn)) = sh {
                    if xr != rows || xn != ck.n {
                        return Err(verr(format!("part {i}: x_hat shape ({xr}, {xn}) != ({rows}, {})", ck.n)));
                    }
                }
                if s.elems(nbytes) != rows * ck.n || matches!(s, Src::Bytes(b) if b.len() != rows * ck.n * nbytes) {
                    return Err(verr(format!("part {i}: x_hat has the wrong size for ({rows}, {})", ck.n)));
                }
                Some(s)
            }
            (None, Some(_)) => return Err(verr(format!("part {i}: x_hat is required with check vectors"))),
            _ => None,
        };
        kparts.push(Part { y_hat, wr: wrb, x_hat, sx: sx_s });
    }
    let total: usize = kparts.iter().map(|p| p.sx.len()).sum();
    let g = Group { parts: kparts, checks, scale: sc_s, bias: bias_s, k, bound, m };
    let (out, status, bad) = py.allow_threads(|| finish_group_run(&g, threads));
    drop(g);
    drop(wrs);
    let out = match out {
        Some(v) => {
            let a = Array2::from_shape_vec((total, m), v).map_err(|e| verr(e.to_string()))?;
            Some(a.into_pyarray(py))
        }
        None => None,
    };
    Ok((out, status, bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic generator (splitmix64).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[derive(Clone)]
    struct Case {
        y: Vec<u8>,
        wr: Vec<u64>,
        x: Vec<u64>,
        u: Vec<u64>,
        uw: Vec<u64>,
        sx: Vec<f64>,
        sc: Vec<f32>,
        bias: Vec<f32>,
    }

    /// Consistent case: y_hat = W x_hat with a random small W, so Freivalds passes.
    fn make(k: u32, rows: usize, m: usize, n: usize, c: usize, seed: u64) -> (Case, Vec<i64>) {
        let mut g = Rng(seed);
        let mask = u64::mask(k);
        let w: Vec<i64> = (0..m * n).map(|_| (g.next() % 255) as i64 - 127).collect();
        let xq: Vec<i64> = (0..rows * n).map(|_| (g.next() % 255) as i64 - 127).collect();
        let r: Vec<u64> = (0..rows * n).map(|_| g.next() & mask).collect();
        let x: Vec<u64> = (0..rows * n).map(|i| (xq[i] as u64).wrapping_add(r[i]) & mask).collect();
        let mut y = Vec::new();
        let mut wr = Vec::new();
        let mut yq = Vec::new();
        for t in 0..rows {
            for j in 0..m {
                let (mut a, mut b, mut q) = (0u64, 0u64, 0i64);
                for i in 0..n {
                    let wv = w[j * n + i] as u64;
                    a = a.wrapping_add(wv.wrapping_mul(x[t * n + i]));
                    b = b.wrapping_add(wv.wrapping_mul(r[t * n + i]));
                    q += w[j * n + i] * xq[t * n + i];
                }
                y.extend_from_slice(&(a & mask).to_le_bytes()[..k as usize / 8]);
                wr.push(b & mask);
                yq.push(((q as u64) & mask).sext(64 - k));
            }
        }
        let u: Vec<u64> = (0..c * m).map(|_| g.next() & mask).collect();
        let uw: Vec<u64> = (0..c * n)
            .map(|idx| {
                let (i, col) = (idx / n, idx % n);
                (0..m).fold(0u64, |s, j| s.wrapping_add(u[i * m + j].wrapping_mul(w[j * n + col] as u64))) & mask
            })
            .collect();
        let sx = (0..rows).map(|t| 0.01 + t as f64 * 1e-3).collect();
        let sc = (0..m).map(|j| 1e-3 * (j as f32 + 1.0)).collect();
        let bias = (0..m).map(|j| j as f32 * 0.5 - 1.0).collect();
        (Case { y, wr, x, u, uw, sx, sc, bias }, yq)
    }

    fn chunk<'a>(cs: &'a Case, k: u32, rows: usize, m: usize, n: usize, c: usize, bound: u64) -> Chunk<'a> {
        Chunk {
            y_hat: &cs.y,
            wr: &cs.wr,
            checks: Some(Checks { x_hat: &cs.x, u: &cs.u, uw: &cs.uw, c, n }),
            sx: &cs.sx,
            scale: &cs.sc,
            bias: Some(&cs.bias),
            k,
            bound,
            rows,
            m,
        }
    }

    #[test]
    fn exact_and_verdicts() {
        for &k in &[16u32, 24, 32, 40, 48, 56] {
            for &(rows, m, n, c) in &[(1usize, 7usize, 5usize, 1usize), (3, 130, 70, 3), (5, 4500, 9, 2)] {
                let (cs, yq) = make(k, rows, m, n, c, 7 + k as u64);
                for threads in [1usize, 3] {
                    let p = chunk(&cs, k, rows, m, n, c, 127 * 127 * n as u64);
                    let (out, st, bad) = finish(&p, threads);
                    assert_eq!((st, bad), (STATUS_OK, 0), "k={k} rows={rows} m={m}");
                    let out = out.unwrap();
                    for t in 0..rows {
                        for j in 0..m {
                            let e = ((yq[t * m + j] as f64 * cs.sx[t]) * cs.sc[j] as f64 + cs.bias[j] as f64) as f32;
                            assert_eq!(out[t * m + j].to_bits(), e.to_bits());
                        }
                    }
                    // range check
                    let maxabs = yq.iter().map(|v| v.unsigned_abs()).max().unwrap();
                    if maxabs > 0 {
                        let p = chunk(&cs, k, rows, m, n, c, maxabs - 1);
                        assert_eq!(finish(&p, threads).1, STATUS_RANGE);
                    }
                    // tamper one element of the last row
                    let mut bad_cs = cs.clone();
                    bad_cs.y[(rows * m - 1) * (k as usize / 8)] ^= 1;
                    let p = chunk(&bad_cs, k, rows, m, n, c, u64::MAX);
                    let (o, st, bad) = finish(&p, threads);
                    assert!(o.is_none());
                    // u . e with e = ±1 at one column: misses only if u[i][col] == 0 for all i.
                    assert_eq!((st, bad), (STATUS_FREIVALDS, 1));
                    // all-zero u
                    let mut z = cs.clone();
                    z.u.fill(0);
                    assert_eq!(finish(&chunk(&z, k, rows, m, n, c, u64::MAX), threads).1, STATUS_U_ZERO);
                }
            }
        }
    }

    #[test]
    fn sign_extension() {
        assert_eq!(0xFFFF_FFFFu32.sext(0), -1);
        assert_eq!(0x80u32.sext(24), -128);
        assert_eq!(0x7Fu32.sext(24), 127);
        assert_eq!((1u64 << 39).sext(24), -(1i64 << 39));
        assert_eq!(((1u64 << 39) - 1).sext(24), (1i64 << 39) - 1);
    }
}
