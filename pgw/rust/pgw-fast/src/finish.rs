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

/// Column block size (elements) for the cache-blocked passes.
const BLOCK: usize = 2048;
/// Roughly this many element-operations per worker thread before going parallel.
const WORK_PER_THREAD: usize = 1 << 19;

/// Overwrite a buffer of secret-derived words with zeros (not elided by the optimiser).
fn wipe<T: Copy + Default>(buf: &mut [T]) {
    for v in buf.iter_mut() {
        // SAFETY: `v` is a valid, aligned, exclusive reference.
        unsafe { std::ptr::write_volatile(v, T::default()) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Raw output pointer shared by workers that write disjoint elements.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers write pairwise disjoint index sets; the scope joins them before the Vec is read.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// What one worker returns: partial Freivalds sums `(rows*c)` for both sides, and min/max of `yq`.
struct Partial<W> {
    lhs: Vec<W>,
    rhs: Vec<W>,
    ymin: i64,
    ymax: i64,
}

#[inline(always)]
fn dot<W: Word>(a: &[W], b: &[W]) -> W {
    let mut s = W::default();
    for (x, y) in a.iter().zip(b) {
        s = s.wadd(x.wmul(*y));
    }
    s
}

#[inline(always)]
fn decode<W: Word, const NB: usize>(src: &[u8], dst: &mut [W]) {
    for (d, ch) in dst.iter_mut().zip(src.chunks_exact(NB)) {
        let mut a = [0u8; 8];
        a[..NB].copy_from_slice(ch);
        *d = W::trunc(u64::from_le_bytes(a));
    }
}

/// Worker body: output columns `[ja, jb)` of every row, input columns `[na, nb)` of the RHS check.
#[inline(always)]
fn work<W: Word, const NB: usize>(
    p: &Chunk<'_>,
    out: OutPtr,
    ja: usize,
    jb: usize,
    na: usize,
    nb: usize,
) -> Partial<W> {
    let (rows, m) = (p.rows, p.m);
    let shift = W::BITS - p.k;
    let c = p.checks.map_or(0, |ck| ck.c);
    let mut lhs = vec![W::default(); rows * c];
    let mut rhs = vec![W::default(); rows * c];
    let mut ybuf = vec![W::default(); BLOCK];
    let mut ubuf = vec![W::default(); BLOCK * c];
    let (mut ymin, mut ymax) = (i64::MAX, i64::MIN);

    let mut b0 = ja;
    while b0 < jb {
        let b1 = (b0 + BLOCK).min(jb);
        let len = b1 - b0;
        if let Some(ck) = p.checks {
            for i in 0..c {
                let src = &ck.u[i * m + b0..i * m + b1];
                for (d, s) in ubuf[i * BLOCK..i * BLOCK + len].iter_mut().zip(src) {
                    *d = W::trunc(*s);
                }
            }
        }
        let scale = &p.scale[b0..b1];
        for t in 0..rows {
            let yb = &mut ybuf[..len];
            decode::<W, NB>(&p.y_hat[(t * m + b0) * NB..(t * m + b1) * NB], yb);
            let wr = &p.wr[t * m + b0..t * m + b1];
            let sxt = p.sx[t];
            // SAFETY: row t, columns [b0, b1) lie in the allocation and are written by this worker only.
            let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(t * m + b0).cast::<MaybeUninit<f32>>(), len) };
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            match p.bias {
                Some(bias) => {
                    let bias = &bias[b0..b1];
                    for j in 0..len {
                        let yq = yb[j].wsub(W::trunc(wr[j])).sext(shift);
                        lo = lo.min(yq);
                        hi = hi.max(yq);
                        o[j].write(((yq as f64 * sxt) * scale[j] as f64 + bias[j] as f64) as f32);
                    }
                }
                None => {
                    for j in 0..len {
                        let yq = yb[j].wsub(W::trunc(wr[j])).sext(shift);
                        lo = lo.min(yq);
                        hi = hi.max(yq);
                        o[j].write(((yq as f64 * sxt) * scale[j] as f64) as f32);
                    }
                }
            }
            ymin = ymin.min(lo);
            ymax = ymax.max(hi);
            for i in 0..c {
                let s = dot(yb, &ubuf[i * BLOCK..i * BLOCK + len]);
                lhs[t * c + i] = lhs[t * c + i].wadd(s);
            }
        }
        b0 = b1;
    }

    if let Some(ck) = p.checks {
        let n = ck.n;
        let mut b0 = na;
        while b0 < nb {
            let b1 = (b0 + BLOCK).min(nb);
            let len = b1 - b0;
            for i in 0..c {
                let src = &ck.uw[i * n + b0..i * n + b1];
                for (d, s) in ubuf[i * BLOCK..i * BLOCK + len].iter_mut().zip(src) {
                    *d = W::trunc(*s);
                }
            }
            for t in 0..rows {
                let xb = &mut ybuf[..len];
                for (d, s) in xb.iter_mut().zip(&ck.x_hat[t * n + b0..t * n + b1]) {
                    *d = W::trunc(*s);
                }
                for i in 0..c {
                    let s = dot(xb, &ubuf[i * BLOCK..i * BLOCK + len]);
                    rhs[t * c + i] = rhs[t * c + i].wadd(s);
                }
            }
            b0 = b1;
        }
    }
    wipe(&mut ybuf);
    wipe(&mut ubuf);
    Partial { lhs, rhs, ymin, ymax }
}

/// Same as [`work`], compiled for AVX2 (vpmulld for the 32-bit dot products, wider f64 lanes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn work_avx2<W: Word, const NB: usize>(
    p: &Chunk<'_>,
    out: OutPtr,
    ja: usize,
    jb: usize,
    na: usize,
    nb: usize,
) -> Partial<W> {
    // `fma` only widens the instruction set: Rust never contracts `a * b + c` into an FMA.
    work::<W, NB>(p, out, ja, jb, na, nb)
}

fn work_dispatch<W: Word, const NB: usize>(
    p: &Chunk<'_>,
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
            return unsafe { work_avx2::<W, NB>(p, out, ja, jb, na, nb) };
        }
    }
    work::<W, NB>(p, out, ja, jb, na, nb)
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

fn run<W: Word, const NB: usize>(p: &Chunk<'_>, threads: usize) -> Finished {
    let (rows, m) = (p.rows, p.m);
    let c = p.checks.map_or(0, |ck| ck.c);
    let n = p.checks.map_or(0, |ck| ck.n);
    if let Some(ck) = p.checks {
        if ck.u.iter().all(|&v| v & u64::mask(p.k) == 0) {
            return (None, STATUS_U_ZERO, 0);
        }
    }
    // Uninitialised: workers write every element (their column ranges partition `[0, m)`), and a
    // plain malloc can reuse warm memory instead of faulting in fresh zeroed pages each call.
    let mut out: Vec<f32> = Vec::with_capacity(rows * m);
    let ptr = OutPtr(out.as_mut_ptr());
    let work_units = rows * (m * (c + 2) + n * c);
    let max_t = if threads == 0 { default_threads() } else { threads };
    let t = (work_units / WORK_PER_THREAD).clamp(1, max_t.max(1)).min(m.div_ceil(64).max(1));

    let mut parts: Vec<Partial<W>> = if t == 1 {
        vec![work_dispatch::<W, NB>(p, ptr, 0, m, 0, n)]
    } else {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..t)
                .map(|i| {
                    let (ja, jb) = split(m, t, i);
                    let (na, nb) = split(n, t, i);
                    s.spawn(move || work_dispatch::<W, NB>(p, ptr, ja, jb, na, nb))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("finish_chunk worker panicked")).collect()
        })
    };

    // SAFETY: every element of `out` was written by exactly one worker (joined above).
    unsafe { out.set_len(rows * m) };

    let (mut ymin, mut ymax) = (i64::MAX, i64::MIN);
    let mut status = STATUS_OK;
    let mut bad_rows = 0usize;
    if c > 0 {
        let mask = W::mask(p.k);
        let mut lhs = vec![W::default(); rows * c];
        let mut rhs = vec![W::default(); rows * c];
        for pt in &parts {
            for (a, b) in lhs.iter_mut().zip(&pt.lhs) {
                *a = a.wadd(*b);
            }
            for (a, b) in rhs.iter_mut().zip(&pt.rhs) {
                *a = a.wadd(*b);
            }
        }
        for tr in 0..rows {
            let row_ok = (0..c).all(|i| lhs[tr * c + i].and(mask) == rhs[tr * c + i].and(mask));
            bad_rows += usize::from(!row_ok);
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
    let bound = p.bound.min(i64::MAX as u64) as i64;
    if status == STATUS_OK && rows * m > 0 && (ymax > bound || ymin < -bound) {
        status = STATUS_RANGE;
    }
    if status != STATUS_OK {
        wipe(&mut out);
        return (None, status, bad_rows);
    }
    (Some(out), STATUS_OK, 0)
}

/// Run the fused finish on validated inputs (`threads = 0`: automatic).
pub fn finish(p: &Chunk<'_>, threads: usize) -> Finished {
    match p.k {
        8 => run::<u32, 1>(p, threads),
        16 => run::<u32, 2>(p, threads),
        24 => run::<u32, 3>(p, threads),
        32 => run::<u32, 4>(p, threads),
        40 => run::<u64, 5>(p, threads),
        48 => run::<u64, 6>(p, threads),
        56 => run::<u64, 7>(p, threads),
        k => panic!("unsupported ring width {k}"),
    }
}

// ---- Python binding --------------------------------------------------------------------------

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Reinterpret an int64 slice as uint64 (same size and alignment).
fn i64_as_u64(s: &[i64]) -> &[u64] {
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

fn arr1<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    ty: &str,
) -> PyResult<numpy::PyReadonlyArray1<'py, T>> {
    let a = obj.downcast::<PyArray1<T>>().map_err(|_| verr(format!("{name} must be a 1-D {ty} numpy array")))?;
    a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))
}

fn slice1<'a, T: numpy::Element>(a: &'a numpy::PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
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
