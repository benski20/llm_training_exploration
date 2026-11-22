//! Exact local linear on the gateway (R1 / E2 `local_linears`): quantize + integer GEMV + range check +
//! dequantize in one call (`local_linear_into`), replacing `MaskedBackend._local_fast`'s chain of
//! numpy temporaries and a two-digit `torch._int_mm`.
//!
//! * Quantization is the TripEngine's constant-time prepare ([`crate::trip::prepare_into`]) with a zero
//!   mask, so `x_hat = xq mod 2^K`: the same `xq` and `sx` bits as every masked trip.
//! * The product `y = xq @ W8^T` is exact integer arithmetic: `xq` as int16 (`|xq| < 2^(A-1)`, `A <= 15`),
//!   int8 weights, int32 partial sums flushed into int64 every [`flush_rows`] input features (so no
//!   partial sum can overflow), AVX2 `vpmaddwd` on two input features at a time when available. Any
//!   summation order gives the same integers, so the column split over threads changes no bit.
//! * The range check is the masked path's (`|y| <= bound`, status 2), evaluated over every output before
//!   the verdict (no early exit), and the rescale is the finish kernel's expression
//!   `((y * sx_t) * scale_j + bias_j) as f32` in float64, i.e. `pgw.core.ring.dequantize_outputs`.
//! * Constant time with respect to the data: every loop runs over public sizes (`rows`, `n`, `m`); the
//!   only data-dependent branch is the error return of the quantizer (which the reference raises too).
//!   The work is local (nothing is sent), but its duration precedes the next trip, so it must not depend
//!   on the activations.
//! * Scratch buffers (quantized activations, partial sums) are thread-local, reused and zeroed after use.

use crate::finish::wipe;
use crate::prepare::{pool, PrepareError, PrepareParams};
use crate::trip::{prepare_into, PrepOut};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(local_linear_into, m)?)?;
    Ok(())
}

/// Largest rows (tokens) per call.
pub const MAX_ROWS: usize = 64;
/// `n * m` from which the columns are spread over the persistent pool (when `threads != 1`; the caller
/// decides above this floor: a decode step reads each weight once, from DRAM, so even 1 MB weights gain).
const PAR_MIN_MACS: usize = 1 << 18;
/// Columns per pool task unit: one 64-byte cache line of every int8 weight row.
const PAR_UNIT: usize = 64;

/// Input features summed in int32 before a flush to int64: `R * 2^(A-1) * 2^7 < 2^31`.
pub fn flush_rows(a: u32) -> usize {
    1usize << (24 - a.min(15)) // A = 12: 4096; A = 15: 512 (always even)
}

/// Feature rows per register pass: the accumulators of a 64-column block are loaded and stored once per
/// `IB` features, and the `IB` weight rows are read as `IB` sequential streams.
const IB: usize = 16;

/// Exact `y (rows, m) = xq (rows, n) @ w8t (n, m)` into `y` (int64, overwritten), columns `[j0, j1)`.
#[allow(clippy::too_many_arguments)]
fn gemv_cols(xq: &[i16], rows: usize, n: usize, w8t: &[i8], m: usize, j0: usize, j1: usize, a: u32, y: &mut [i64]) {
    for t in 0..rows {
        y[t * m + j0..t * m + j1].fill(0);
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available (checked above); every access stays inside the slices (see below).
        unsafe { gemv_avx2(xq, rows, n, w8t, m, j0, j1, a, y) };
        return;
    }
    add_scalar(xq, rows, n, w8t, m, j0, j1, 0, n, y);
}

/// `y[t, j] += sum_{i in [i0, i1)} xq[t, i] * w8t[i, j]` for `j in [ja, jb)` (int64 throughout).
#[allow(clippy::too_many_arguments)]
fn add_scalar(
    xq: &[i16],
    rows: usize,
    n: usize,
    w8t: &[i8],
    m: usize,
    ja: usize,
    jb: usize,
    i0: usize,
    i1: usize,
    y: &mut [i64],
) {
    if ja >= jb {
        return;
    }
    for i in i0..i1 {
        let wr = &w8t[i * m + ja..i * m + jb];
        for t in 0..rows {
            let xi = i64::from(xq[t * n + i]);
            for (yv, &wv) in y[t * m + ja..t * m + jb].iter_mut().zip(wr) {
                *yv += xi * i64::from(wv);
            }
        }
    }
}

/// AVX2 product: int32 accumulators in `vpunpck` order (per 16 columns: lo = cols 0-3, 8-11; hi = 4-7,
/// 12-15) kept in `lanes`, 64 columns at a time in registers over `IB` features, flushed into `y` every
/// [`flush_rows`] features. The odd last feature of a flush block and a column tail (< 16) go through
/// [`add_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_avx2(xq: &[i16], rows: usize, n: usize, w8t: &[i8], m: usize, j0: usize, j1: usize, a: u32, y: &mut [i64]) {
    let w = j1 - j0;
    let chunks = w / 16;
    let vec_w = chunks * 16;
    let fr = flush_rows(a);
    let mut lanes = vec![0i32; rows * vec_w];
    let mut f0 = 0;
    while f0 < n {
        let f1 = (f0 + fr).min(n);
        let pairs_end = f0 + (f1 - f0) / 2 * 2;
        if vec_w > 0 {
            lanes.fill(0);
            let mut ib = f0;
            while ib < pairs_end {
                let ie = (ib + IB).min(pairs_end);
                for t in 0..rows {
                    let xr = &xq[t * n..(t + 1) * n];
                    let mut c = 0;
                    while c < chunks {
                        let take = (chunks - c).min(4);
                        let acc = lanes.as_mut_ptr().add(t * vec_w + c * 16);
                        let base = w8t.as_ptr().add(j0 + c * 16);
                        match take {
                            4 => block_avx2::<4>(xr, base, m, ib, ie, acc),
                            3 => block_avx2::<3>(xr, base, m, ib, ie, acc),
                            2 => block_avx2::<2>(xr, base, m, ib, ie, acc),
                            _ => block_avx2::<1>(xr, base, m, ib, ie, acc),
                        }
                        c += take;
                    }
                }
                ib = ie;
            }
            // un-permute into y: lo lane k -> col k (k < 4) or k + 4; hi lane k -> col k + 4 (k < 4) or k + 8
            for t in 0..rows {
                for c in 0..chunks {
                    let src = &lanes[t * vec_w + c * 16..t * vec_w + c * 16 + 16];
                    let dst = &mut y[t * m + j0 + c * 16..t * m + j0 + c * 16 + 16];
                    for k in 0..8 {
                        let cl = if k < 4 { k } else { k + 4 };
                        let ch = if k < 4 { k + 4 } else { k + 8 };
                        dst[cl] += i64::from(src[k]);
                        dst[ch] += i64::from(src[8 + k]);
                    }
                }
            }
            add_scalar(xq, rows, n, w8t, m, j0, j0 + vec_w, pairs_end, f1, y); // odd last feature
        }
        f0 = f1;
    }
    wipe(&mut lanes);
    add_scalar(xq, rows, n, w8t, m, j0 + vec_w, j1, 0, n, y); // column tail
}

/// `C` chunks of 16 columns of one row: load the accumulators at `acc[0 .. 16 C]` (`vpunpck` order), add
/// the feature pairs `[i0, i1)` (`i1 - i0` even) of the weight columns at `base`, store them back.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_avx2<const C: usize>(xr: &[i16], base: *const i8, m: usize, i0: usize, i1: usize, acc: *mut i32) {
    use std::arch::x86_64::*;
    let mut r = [_mm256_setzero_si256(); 8];
    for c in 0..C {
        r[2 * c] = _mm256_loadu_si256(acc.add(c * 16).cast());
        r[2 * c + 1] = _mm256_loadu_si256(acc.add(c * 16 + 8).cast());
    }
    let mut i = i0;
    while i < i1 {
        let xp = _mm256_set1_epi32(((xr[i] as u16 as u32) | ((xr[i + 1] as u16 as u32) << 16)) as i32);
        let r0 = base.add(i * m);
        let r1 = r0.add(m);
        for c in 0..C {
            let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(r0.add(c * 16).cast()));
            let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(r1.add(c * 16).cast()));
            r[2 * c] = _mm256_add_epi32(r[2 * c], _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, a1), xp));
            r[2 * c + 1] = _mm256_add_epi32(r[2 * c + 1], _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, a1), xp));
        }
        i += 2;
    }
    for c in 0..C {
        _mm256_storeu_si256(acc.add(c * 16).cast(), r[2 * c]);
        _mm256_storeu_si256(acc.add(c * 16 + 8).cast(), r[2 * c + 1]);
    }
    for v in r.iter_mut() {
        *v = _mm256_setzero_si256();
    }
}

/// Exact product over every column, on the pool when large (column blocks of whole 64-column units;
/// each output is the same integer sum whichever task computes it, so the split changes no bit).
fn gemv(xq: &[i16], rows: usize, n: usize, w8t: &[i8], m: usize, a: u32, y: &mut [i64], threads: usize) -> bool {
    let p = pool::global();
    let max_t = if threads == 0 { p.workers() + 1 } else { threads };
    if max_t <= 1 || n * m < PAR_MIN_MACS || p.workers() == 0 || m < 2 * PAR_UNIT {
        gemv_cols(xq, rows, n, w8t, m, 0, m, a, y);
        return true;
    }
    let units = m.div_ceil(PAR_UNIT);
    let parts = (max_t * 2).min(units);
    let per = units.div_ceil(parts);
    let yp = SendPtr(y.as_mut_ptr());
    let ylen = y.len();
    p.run(parts, &|pi| {
        let j0 = (pi * per * PAR_UNIT).min(m);
        let j1 = ((pi + 1) * per * PAR_UNIT).min(m);
        if j0 >= j1 {
            return;
        }
        // SAFETY: tasks write disjoint column ranges [j0, j1) of every row; `y` outlives `run`.
        let ys = unsafe { std::slice::from_raw_parts_mut(yp.get(), ylen) };
        gemv_cols(xq, rows, n, w8t, m, j0, j1, a, ys);
    })
}

#[derive(Clone, Copy)]
struct SendPtr(*mut i64);
// SAFETY: used only for disjoint column writes inside `pool.run`, which joins before returning.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut i64 {
        self.0
    }
}

struct Scratch {
    zeros: Vec<i64>,
    payload: Vec<u8>,
    xhat: Vec<i64>,
    xq: Vec<i16>,
    y: Vec<i64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> =
        const { RefCell::new(Scratch { zeros: Vec::new(), payload: Vec::new(), xhat: Vec::new(), xq: Vec::new(), y: Vec::new() }) };
}

/// Outcome of [`local_linear`].
#[derive(Debug, PartialEq, Eq)]
pub enum LocalStatus {
    /// Outputs written.
    Ok,
    /// Some `|y| > bound` (the masked path's status 2); `out` is zeroed.
    Range,
}

/// Inputs of [`local_linear`].
pub struct LocalArgs<'a> {
    /// float32 `(rows, n)`.
    pub x: &'a [f32],
    /// int8 `(n, m)` (the transposed weight, row-major).
    pub w8t: &'a [i8],
    /// float32 `(m,)`.
    pub scale: &'a [f32],
    /// float32 `(m,)` or none.
    pub bias: Option<&'a [f32]>,
    /// Ring and linear parameters (`a <= 15`).
    pub prm: PrepareParams,
    /// Inclusive output bound (`RingParams.output_bound(n)`).
    pub bound: u64,
    /// Input features.
    pub n: usize,
}

/// Quantize + exact product + range check + rescale (see the module docs). `sx (rows,)` receives the
/// activation scales, `out (rows, m)` the outputs.
pub fn local_linear(f: &LocalArgs<'_>, sx: &mut [f64], out: &mut [f32], threads: usize) -> Result<LocalStatus, PrepareError> {
    let prm = f.prm;
    prm.validate()?;
    let (n, m, rows) = (f.n, f.scale.len(), sx.len());
    if prm.a > 15 {
        return Err(PrepareError::Invalid(format!("local linear needs A <= 15, got {}", prm.a)));
    }
    if n == 0 || rows == 0 || rows > MAX_ROWS {
        return Err(PrepareError::Invalid(format!("rows must be in [1, {MAX_ROWS}] and n >= 1")));
    }
    if f.x.len() != rows * n || f.w8t.len() != n * m || out.len() != rows * m || f.bias.is_some_and(|b| b.len() != m) {
        return Err(PrepareError::Invalid("buffer sizes do not match the shape".into()));
    }
    let nb = (prm.k / 8) as usize;
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        let s = &mut *s;
        if s.zeros.len() < rows * n {
            s.zeros.resize(rows * n, 0);
        }
        s.payload.resize(rows * n * nb, 0);
        s.xhat.resize(rows * n, 0);
        s.xq.resize(rows * n, 0);
        s.y.resize(rows * m, 0);
        let res = prepare_into(
            f.x,
            &s.zeros[..rows * n],
            n,
            prm,
            None,
            PrepOut { payload: &mut s.payload[..], sx, xhat: &mut s.xhat[..], rhs: &mut [] },
            1,
        );
        wipe(&mut s.payload);
        if let Err(e) = res {
            wipe(&mut s.xhat);
            wipe(out);
            return Err(e);
        }
        // centered representative of x_hat = xq mod 2^K (|xq| < 2^(A-1) <= 2^14)
        let sh = 64 - prm.k;
        for (q, &v) in s.xq.iter_mut().zip(s.xhat.iter()) {
            *q = ((v << sh) >> sh) as i16;
        }
        wipe(&mut s.xhat);
        let ok = gemv(&s.xq, rows, n, f.w8t, m, prm.a, &mut s.y, threads);
        wipe(&mut s.xq);
        if !ok {
            wipe(&mut s.y);
            wipe(out);
            return Err(PrepareError::Invalid("internal error in a local linear task".into()));
        }
        // range check over every output (no early exit), then the rescale
        let mut amax: u64 = 0;
        for &v in s.y.iter() {
            amax = amax.max(v.unsigned_abs());
        }
        for t in 0..rows {
            let sxt = sx[t];
            let yr = &s.y[t * m..(t + 1) * m];
            let o = &mut out[t * m..(t + 1) * m];
            match f.bias {
                Some(b) => {
                    for j in 0..m {
                        o[j] = ((yr[j] as f64 * sxt) * f.scale[j] as f64 + b[j] as f64) as f32;
                    }
                }
                None => {
                    for j in 0..m {
                        o[j] = ((yr[j] as f64 * sxt) * f.scale[j] as f64) as f32;
                    }
                }
            }
        }
        wipe(&mut s.y);
        if amax > f.bound {
            wipe(out);
            return Ok(LocalStatus::Range);
        }
        Ok(LocalStatus::Ok)
    })
}

/// Exact local linear (quantize, int8 product, range check, rescale) into caller buffers.
///
/// Args:
///     x: float32 ``(rows, n)``, C-contiguous (``1 <= rows <= 64``).
///     w8t: int8 ``(n, m)``, C-contiguous (the transposed weight).
///     scale: float32 ``(m,)``; bias: float32 ``(m,)`` or None.
///     k, a, cap, wnorm: ring and linear parameters as in ``trip_prepare_into`` (``a <= 15``).
///     bound: inclusive output bound (``RingParams.output_bound(n)``).
///     sx: writable float64 ``(rows,)`` (activation scales); out: writable float32 ``(rows, m)``.
///     threads: 1 = calling thread; 0 = the persistent pool for large weights.
///
/// Returns the status: 0 ok, 2 an output is out of range (``out`` zeroed). Quantizer errors raise
/// ``ValueError`` (``"ring_overflow: ..."`` as ``trip_prepare_into``). The GIL is released.
#[pyfunction]
#[pyo3(signature = (x, w8t, scale, bias, k, a, cap, wnorm, bound, sx, out, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn local_linear_into<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f32>>,
    w8t: &Bound<'py, PyArray2<i8>>,
    scale: &Bound<'py, PyArray1<f32>>,
    bias: Option<&Bound<'py, PyArray1<f32>>>,
    k: u32,
    a: u32,
    cap: u32,
    wnorm: f64,
    bound: u64,
    sx: &Bound<'py, PyArray1<f64>>,
    out: &Bound<'py, PyArray2<f32>>,
    threads: usize,
) -> PyResult<u8> {
    let verr = |s: &str| PyValueError::new_err(s.to_string());
    let xa = x.try_readonly().map_err(|e| verr(&format!("x: {e}")))?;
    let wa = w8t.try_readonly().map_err(|e| verr(&format!("w8t: {e}")))?;
    let sa = scale.try_readonly().map_err(|e| verr(&format!("scale: {e}")))?;
    let ba = match bias {
        Some(b) => Some(b.try_readonly().map_err(|e| verr(&format!("bias: {e}")))?),
        None => None,
    };
    let (rows, n) = (xa.shape()[0], xa.shape()[1]);
    if wa.shape()[0] != n {
        return Err(verr("w8t must be (n, m) with n = x.shape[1]"));
    }
    let m = wa.shape()[1];
    let xs = xa.as_slice().map_err(|_| verr("x must be C-contiguous"))?;
    let ws = wa.as_slice().map_err(|_| verr("w8t must be C-contiguous"))?;
    let ss = sa.as_slice().map_err(|_| verr("scale must be contiguous"))?;
    if ss.len() != m {
        return Err(verr("scale must have m entries"));
    }
    let bs = match &ba {
        Some(b) => Some(b.as_slice().map_err(|_| verr("bias must be contiguous"))?),
        None => None,
    };
    let mut sxa = sx.try_readwrite().map_err(|e| verr(&format!("sx: {e}")))?;
    let mut oa = out.try_readwrite().map_err(|e| verr(&format!("out: {e}")))?;
    if oa.shape() != [rows, m] {
        return Err(verr("out must be (rows, m)"));
    }
    let sxs = sxa.as_slice_mut().map_err(|_| verr("sx must be contiguous"))?;
    if sxs.len() != rows {
        return Err(verr("sx must have rows entries"));
    }
    let os = oa.as_slice_mut().map_err(|_| verr("out must be C-contiguous"))?;
    let args = LocalArgs { x: xs, w8t: ws, scale: ss, bias: bs, prm: PrepareParams { k, a, cap, wnorm }, bound, n };
    match py.allow_threads(|| local_linear(&args, sxs, os, threads)) {
        Ok(LocalStatus::Ok) => Ok(0),
        Ok(LocalStatus::Range) => Ok(2),
        Err(e) => Err(e.into_py()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn gemv_matches_naive() {
        let mut seed = 7u64;
        for &(rows, n, m, a) in &[(1, 5, 3, 12), (2, 37, 300, 12), (3, 1031, 517, 15), (1, 4100, 33, 15), (8, 64, 16, 12)] {
            let lim = (1i64 << (a - 1)) - 1;
            let xq: Vec<i16> = (0..rows * n).map(|_| ((lcg(&mut seed) as i64 % (2 * lim + 1)) - lim) as i16).collect();
            let w: Vec<i8> = (0..n * m).map(|_| (lcg(&mut seed) % 256) as u8 as i8).collect();
            let mut y = vec![0i64; rows * m];
            gemv_cols(&xq, rows, n, &w, m, 0, m, a, &mut y);
            for t in 0..rows {
                for j in 0..m {
                    let want: i64 = (0..n).map(|i| i64::from(xq[t * n + i]) * i64::from(w[i * m + j])).sum();
                    assert_eq!(y[t * m + j], want, "rows={rows} n={n} m={m} t={t} j={j}");
                }
            }
            // extreme values: no int32 overflow within a flush block
            let xq: Vec<i16> = vec![-(lim as i16); rows * n];
            let w: Vec<i8> = vec![-128; n * m];
            gemv_cols(&xq, rows, n, &w, m, 0, m, a, &mut y);
            assert!(y.iter().all(|&v| v == lim * 128 * n as i64));
        }
    }

    #[test]
    fn pooled_gemv_matches_one_thread() {
        // column splits in 64-column units over the pool (m not a multiple of 64 or 16 included)
        let mut seed = 11u64;
        for &(rows, n, m) in &[(1, 896, 1152), (1, 4864, 896), (3, 600, 1000), (2, 2048, 130), (1, 1024, 4100)] {
            let lim = (1i64 << 11) - 1;
            let xq: Vec<i16> = (0..rows * n).map(|_| ((lcg(&mut seed) as i64 % (2 * lim + 1)) - lim) as i16).collect();
            let w: Vec<i8> = (0..n * m).map(|_| (lcg(&mut seed) % 256) as u8 as i8).collect();
            let mut one = vec![0i64; rows * m];
            gemv_cols(&xq, rows, n, &w, m, 0, m, 12, &mut one);
            for threads in [0usize, 2, 3] {
                let mut par = vec![-1i64; rows * m];
                assert!(gemv(&xq, rows, n, &w, m, 12, &mut par, threads));
                assert_eq!(par, one, "rows={rows} n={n} m={m} threads={threads}");
            }
        }
    }
}
