//! Gateway prepare step: quantize float activations per token, mask with the one-time pad `r`,
//! and pack the masked values to wire bytes, in one pass (`MaskedBackend._prepare`).
//!
//! Bit-identical to the Python composition
//! `quantize_activations(x, ring, wnorm)` -> `mask_activations(xq, r, ring)` -> `pack_ring(x_hat, ring)`:
//!
//! * uncapped rings: `sx = amax / qmax` (1.0 for all-zero rows), `xq = clip(rint(x / sx), -qmax, qmax)`
//!   in float64 (`x` is float32, widened exactly);
//! * capped rings (`cap > 0`): for bits `a, a-1, ..., 2` the first (most) bits whose
//!   `sqrt(sum(q*q)) * wnorm * (1 + 1e-9) <= 2^cap` win (per row). The float64 sum of squares of
//!   these integers is exact in any summation order while `n * qmax^2 < 2^53` (any width below
//!   5e8 for A=12; A=20 below 3.3e4), so it equals numpy's pairwise sum there;
//! * `x_hat = (xq + r) mod 2^K`, packed as `K/8` little-endian bytes per element.
//!
//! Functions are registered in [`register`].

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;
use zeroize::Zeroize;

/// Relative slack on the float64 norm product (`pgw.core.ring._NORM_SLACK`).
pub(crate) const NORM_SLACK: f64 = 1.0 + 1e-9;

/// Kernel errors (mapped to Python `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareError {
    /// Bad arguments (shape, ring parameters).
    Invalid(String),
    /// Non-finite input or no bit width fits the cap (Python `RingOverflowError`).
    Overflow(String),
}

impl PrepareError {
    pub(crate) fn into_py(self) -> PyErr {
        match self {
            PrepareError::Invalid(m) => PyValueError::new_err(m),
            PrepareError::Overflow(m) => PyValueError::new_err(format!("ring_overflow: {m}")),
        }
    }
}

/// Ring / quantization parameters of one call.
#[derive(Debug, Clone, Copy)]
pub struct PrepareParams {
    /// Ring bits (multiple of 8 in [8, 56]).
    pub k: u32,
    /// Signed activation bits (`qmax = 2^(a-1) - 1`), in [2, min(k-1, 48)].
    pub a: u32,
    /// Output cap bits (0 = uncapped), in [1, 62].
    pub cap: u32,
    /// Largest Euclidean norm of the linear's int8 weight rows (capped rings only).
    pub wnorm: f64,
}

impl PrepareParams {
    pub(crate) fn validate(&self) -> Result<(), PrepareError> {
        if self.k % 8 != 0 || !(8..=56).contains(&self.k) {
            return Err(PrepareError::Invalid(format!("ring K must be a multiple of 8 in [8, 56], got {}", self.k)));
        }
        if !(2..=self.k.saturating_sub(1).min(48)).contains(&self.a) {
            return Err(PrepareError::Invalid(format!(
                "activation bits A must be in [2, min(K-1, 48)], got A={} K={}",
                self.a, self.k
            )));
        }
        if self.cap > 62 {
            return Err(PrepareError::Invalid(format!("output cap must be 0 or in [1, 62], got {}", self.cap)));
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn qmax_of(bits: u32) -> f64 {
    ((1u64 << (bits - 1)) - 1) as f64
}

/// `2^52 + 2^51`: adding and subtracting it rounds any `|v| < 2^51` to an integer, ties to even
/// (IEEE round-to-nearest-even), without SSE4.1 `roundsd` / a libm `roundeven` call.
const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

/// `ROUND_MAGIC + q` for an integer `|q| < 2^51` has the bit pattern `MAGIC_BITS + q`.
const MAGIC_BITS: i64 = 0x4338_0000_0000_0000;

/// `ROUND_MAGIC + clip(rint(v / s), -qmax, qmax)`, with `rint` exactly as numpy computes it.
///
/// Clipping to the integer bounds before rounding gives the same value (rounding is monotone and
/// fixes integers), and keeps `|v| <= qmax < 2^51` for the magic-constant rounding. The integer is
/// [`biased_to_i64`]; the float is `b - ROUND_MAGIC` (possibly `-0.0` where numpy has `0.0`, which
/// is the same integer and the same square).
#[inline(always)]
pub(crate) fn quant_biased(v: f32, s: f64, qmax: f64) -> f64 {
    let c = f64::from(v) / s;
    let c = if c < -qmax {
        -qmax
    } else if c > qmax {
        qmax
    } else {
        c
    };
    c + ROUND_MAGIC
}

/// The integer held by a [`quant_biased`] value (pure integer op, vectorizes without AVX-512).
#[inline(always)]
pub(crate) fn biased_to_i64(b: f64) -> i64 {
    (b.to_bits() as i64).wrapping_sub(MAGIC_BITS)
}

/// Relative tolerance of the reciprocal-multiply fast path: `x * fl(1/s)` differs from the correctly
/// rounded `fl(x / s)` by at most about `|x/s| * 2^-51` (three roundings), far below `2^-40`.
const FAST_TOL: f64 = 1.0 / (1u64 << 40) as f64;

/// Quantize one row into `q` with true division (numpy's exact operation sequence).
#[inline(always)]
pub(crate) fn quant_row_exact(xr: &[f32], s: f64, qmax: f64, q: &mut [i64]) {
    for (qv, v) in q.iter_mut().zip(xr) {
        *qv = biased_to_i64(quant_biased(*v, s, qmax));
    }
}

/// Quantize one row into `q`, bit-identical to [`quant_row_exact`].
///
/// Fast path: `p = x * (1/s)` instead of `x / s` (vectorizes ~2x better). `rint` and the clip to
/// integer bounds only depend on which side of a half-integer the quotient lies, so wherever `p`
/// is farther than `|p| * 2^-40` from every half-integer it yields the same integer as the
/// correctly rounded quotient. If any element of the row is closer (exact ties included), the
/// row is recomputed with true division.
#[inline(always)]
fn quant_row(xr: &[f32], s: f64, qmax: f64, q: &mut [i64]) {
    let inv = 1.0 / s;
    let mut near = 0u64;
    for (qv, v) in q.iter_mut().zip(xr) {
        let pv = f64::from(*v) * inv;
        let c = if pv < -qmax {
            -qmax
        } else if pv > qmax {
            qmax
        } else {
            pv
        };
        let b = c + ROUND_MAGIC;
        // |p - rint(clip(p))| is computed exactly (Sterbenz); a (near-)tie has it close to 0.5
        let d = ((pv - (b - ROUND_MAGIC)).abs() - 0.5).abs();
        near |= u64::from(d <= pv.abs() * FAST_TOL);
        *qv = biased_to_i64(b);
    }
    if near != 0 {
        quant_row_exact(xr, s, qmax, q);
    }
}

/// `sum(q * q)` as float64; exact (so equal to numpy's float64 sum in any order) while the true
/// sum is below `2^53`.
#[inline(always)]
pub(crate) fn sum_squares(q: &[i64], qmax: f64) -> f64 {
    // integer path: |q| < 2^31 and the total cannot overflow i64
    if qmax < 2_147_483_648.0 && (q.len() as f64) * qmax * qmax < 9.0e18 {
        let mut acc = 0i64;
        for v in q {
            let qi = i64::from(*v as i32);
            acc = acc.wrapping_add(qi.wrapping_mul(qi));
        }
        acc as f64
    } else {
        q.iter().map(|v| (*v as f64) * (*v as f64)).sum()
    }
}

/// Bit pattern of the largest `|x|` in a row. For non-negative floats the integer order of the
/// bit patterns is the float order, and any NaN/inf gives a pattern `>= INF_BITS`.
#[inline(always)]
pub(crate) fn row_amax_bits(row: &[f32]) -> u32 {
    let mut acc = [0u32; 8];
    let mut c = row.chunks_exact(8);
    for xs in &mut c {
        for i in 0..8 {
            acc[i] = acc[i].max(xs[i].to_bits() & 0x7fff_ffff);
        }
    }
    let tail = c.remainder().iter().fold(0u32, |m, v| m.max(v.to_bits() & 0x7fff_ffff));
    acc.iter().fold(tail, |m, v| m.max(*v))
}

/// `f32::INFINITY.to_bits()`.
pub(crate) const INF_BITS: u32 = 0x7f80_0000;

/// Per-row amax into `amax` (as f64); `false` if any value is NaN/inf.
#[inline(always)]
fn amax_rows_generic(x: &[f32], n: usize, amax: &mut [f64]) -> bool {
    let mut finite = true;
    for (row, a) in x.chunks_exact(n).zip(amax.iter_mut()) {
        let m = row_amax_bits(row);
        finite &= m < INF_BITS;
        *a = f64::from(f32::from_bits(m));
    }
    finite
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn amax_rows_avx2(x: &[f32], n: usize, amax: &mut [f64]) -> bool {
    amax_rows_generic(x, n, amax)
}

/// [`amax_rows_generic`], compiled for AVX2 when the CPU has it (same IEEE results).
fn amax_rows(x: &[f32], n: usize, amax: &mut [f64]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked above).
        return unsafe { amax_rows_avx2(x, n, amax) };
    }
    amax_rows_generic(x, n, amax)
}

/// Mask and pack one row: `x_hat = (xq + r) & mask`, `K/8` LE bytes each.
#[inline(always)]
pub(crate) fn mask_pack_row(xq: &[i64], r: &[i64], k: u32, xhat: &mut [i64], out: &mut [u8]) {
    let mask = ((1u64 << k) - 1) as i64;
    for ((q, rv), xh) in xq.iter().zip(r).zip(xhat.iter_mut()) {
        *xh = q.wrapping_add(*rv) & mask;
    }
    let nb = (k / 8) as usize;
    match nb {
        4 => {
            for (v, o) in xhat.iter().zip(out.chunks_exact_mut(4)) {
                o.copy_from_slice(&(*v as u32).to_le_bytes());
            }
        }
        _ => {
            for (v, o) in xhat.iter().zip(out.chunks_exact_mut(nb)) {
                o.copy_from_slice(&(*v as u64).to_le_bytes()[..nb]);
            }
        }
    }
}

/// Fused quantize + mask + pack over `rows x n` (row-major) inputs.
///
/// Writes `x_hat` (`rows * n`), `sx` (`rows`) and the wire payload (`rows * n * K/8` bytes).
/// Temporaries holding `xq` are zeroized before returning.
pub fn quantize_mask_pack_into(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    xhat: &mut [i64],
    sx: &mut [f64],
    payload: &mut [u8],
) -> Result<(), PrepareError> {
    p.validate()?;
    let nb = (p.k / 8) as usize;
    let rows = sx.len();
    if x.len() != rows * n || r.len() != rows * n || xhat.len() != rows * n || payload.len() != rows * n * nb {
        return Err(PrepareError::Invalid("buffer sizes do not match the shape".into()));
    }
    if rows == 0 {
        return Ok(());
    }
    if n == 0 {
        return Err(PrepareError::Invalid("activations must have at least one feature".into()));
    }
    // Python checks the whole tensor before quantizing any row. `sx` holds the row amax until
    // `prepare_rows` replaces it with the scale.
    if !amax_rows(x, n, sx) {
        return Err(PrepareError::Overflow("non-finite activation values cannot be quantized".into()));
    }
    // Single-threaded: per-call work is ~1-2 ns/element; spawning workers cost more than it saved
    // in measurements (decode calls are 1 row; prefill parallelism belongs to the caller).
    let res = prepare_rows(x, r, n, p, xhat, sx, payload);
    if res.is_err() {
        xhat.zeroize();
        payload.zeroize();
    }
    res
}

/// [`prepare_rows_generic`], compiled for AVX2 when the CPU has it (same IEEE results: no FMA
/// contraction, divisions and roundings are correctly rounded either way).
fn prepare_rows(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    xhat: &mut [i64],
    sx: &mut [f64],
    payload: &mut [u8],
) -> Result<(), PrepareError> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked above).
        return unsafe { prepare_rows_avx2(x, r, n, p, xhat, sx, payload) };
    }
    prepare_rows_generic(x, r, n, p, xhat, sx, payload)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prepare_rows_avx2(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    xhat: &mut [i64],
    sx: &mut [f64],
    payload: &mut [u8],
) -> Result<(), PrepareError> {
    prepare_rows_generic(x, r, n, p, xhat, sx, payload)
}

/// Quantize, mask and pack whole rows (`sx.len()` rows of width `n`); `sx` holds each row's amax
/// on entry and its scale on exit.
#[inline(always)]
fn prepare_rows_generic(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    xhat: &mut [i64],
    sx: &mut [f64],
    payload: &mut [u8],
) -> Result<(), PrepareError> {
    let nb = (p.k / 8) as usize;
    let limit = (1u64 << p.cap) as f64;
    let mut q = vec![0i64; n];
    let mut res = Ok(());
    for (t, ((xr, rr), (xh, out))) in x
        .chunks_exact(n)
        .zip(r.chunks_exact(n))
        .zip(xhat.chunks_exact_mut(n).zip(payload.chunks_exact_mut(n * nb)))
        .enumerate()
    {
        let amax = sx[t];
        if p.cap == 0 {
            let qmax = qmax_of(p.a);
            let s = if amax > 0.0 { amax / qmax } else { 1.0 };
            quant_row(xr, s, qmax, &mut q);
            sx[t] = s;
        } else {
            let mut found = false;
            for bits in (2..=p.a).rev() {
                let qmax = qmax_of(bits);
                let s = if amax > 0.0 { amax / qmax } else { 1.0 };
                quant_row(xr, s, qmax, &mut q);
                let ss = sum_squares(&q, qmax);
                if ss.sqrt() * p.wnorm * NORM_SLACK <= limit {
                    sx[t] = s;
                    found = true;
                    break;
                }
            }
            if !found {
                res = Err(PrepareError::Overflow(format!(
                    "K{}A{}C{}: activations cannot be bounded by 2^{} for this linear",
                    p.k, p.a, p.cap, p.cap
                )));
                break;
            }
        }
        mask_pack_row(&q, rr, p.k, xh, out);
    }
    q.zeroize();
    res
}

/// Read a 2-D C-contiguous slice, or a (zeroized-on-drop by the caller) owned copy.
enum Data<'a, T> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<T> Data<'_, T> {
    fn slice(&self) -> &[T] {
        match self {
            Data::Borrowed(s) => s,
            Data::Owned(v) => v,
        }
    }
}

/// `(payload, x_hat, sx)` returned to Python (`x_hat` is None with `want_xhat=False`).
type Prepared<'py> = (Bound<'py, PyBytes>, Option<Bound<'py, PyArray2<i64>>>, Bound<'py, numpy::PyArray1<f64>>);

/// [`quantize_mask_pack_into`] without an `x_hat` output (one row-sized scratch buffer instead):
/// the same checks, in the same order (every row's finiteness first), and the same payload.
fn quantize_mask_pack_payload(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    sx: &mut [f64],
    payload: &mut [u8],
) -> Result<(), PrepareError> {
    p.validate()?;
    let nb = (p.k / 8) as usize;
    let rows = sx.len();
    if x.len() != rows * n || r.len() != rows * n || payload.len() != rows * n * nb {
        return Err(PrepareError::Invalid("buffer sizes do not match the shape".into()));
    }
    if rows == 0 {
        return Ok(());
    }
    if n == 0 {
        return Err(PrepareError::Invalid("activations must have at least one feature".into()));
    }
    let mut amax = vec![0.0f64; rows];
    if !amax_rows(x, n, &mut amax) {
        return Err(PrepareError::Overflow("non-finite activation values cannot be quantized".into()));
    }
    let mut scratch = vec![0i64; n];
    let mut res = Ok(());
    for t in 0..rows {
        let (a, b) = (t * n, (t + 1) * n);
        let row = quantize_mask_pack_into(
            &x[a..b],
            &r[a..b],
            n,
            p,
            &mut scratch,
            &mut sx[t..t + 1],
            &mut payload[a * nb..b * nb],
        );
        if row.is_err() {
            res = row;
            break;
        }
    }
    scratch.zeroize();
    if res.is_err() {
        payload.zeroize();
    }
    res
}

/// `quantize_mask_pack(x, r, k, a, cap, wnorm, want_xhat=True) -> (payload, x_hat, sx)`.
///
/// `x`: float32 `(rows, n)`; `r`: int64 `(rows, n)` one-time mask in `[0, 2^K)`; `cap = 0` means an
/// uncapped ring. Returns the wire bytes (`K/8` LE per element), `x_hat` int64 `(rows, n)` in
/// `[0, 2^K)` and `sx` float64 `(rows,)`. Raises `ValueError("ring_overflow: ...")` for non-finite
/// input or when no bit width fits the cap, `ValueError` for bad shapes/dtypes/parameters. With
/// `want_xhat=False`, `x_hat` is not materialized (None; the payload holds the same values).
#[pyfunction]
#[pyo3(signature = (x, r, k, a, cap, wnorm, want_xhat = true))]
#[allow(clippy::too_many_arguments)]
fn quantize_mask_pack<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
    k: u32,
    a: u32,
    cap: u32,
    wnorm: f64,
    want_xhat: bool,
) -> PyResult<Prepared<'py>> {
    let p = PrepareParams { k, a, cap, wnorm };
    p.validate().map_err(PrepareError::into_py)?;
    let xa = x
        .downcast::<PyArray2<f32>>()
        .map_err(|_| PyValueError::new_err("x must be a 2-D float32 numpy array"))?
        .readonly();
    let ra = r
        .downcast::<PyArray2<i64>>()
        .map_err(|_| PyValueError::new_err("r must be a 2-D int64 numpy array"))?
        .readonly();
    let shape = xa.shape();
    if shape != ra.shape() {
        return Err(PyValueError::new_err(format!("activation shape {:?} != mask shape {:?}", xa.shape(), ra.shape())));
    }
    let (rows, n) = (shape[0], shape[1]);
    let nb = (k / 8) as usize;
    // C order only: `as_slice` also accepts Fortran order, whose elements it would read transposed
    let xs = match xa.as_slice().ok().filter(|_| xa.is_c_contiguous()) {
        Some(s) => Data::Borrowed(s),
        None => Data::Owned(xa.as_array().iter().copied().collect()),
    };
    let mut rs = match ra.as_slice().ok().filter(|_| ra.is_c_contiguous()) {
        Some(s) => Data::Borrowed(s),
        None => Data::Owned(ra.as_array().iter().copied().collect()),
    };
    let mut xhat = if want_xhat { vec![0i64; rows * n] } else { Vec::new() };
    let mut sx = vec![1.0f64; rows];
    let payload = PyBytes::new_with(py, rows * n * nb, |buf| {
        let (xsl, rsl) = (xs.slice(), rs.slice());
        py.allow_threads(|| {
            if want_xhat {
                quantize_mask_pack_into(xsl, rsl, n, p, &mut xhat, &mut sx, buf)
            } else {
                quantize_mask_pack_payload(xsl, rsl, n, p, &mut sx, buf)
            }
        })
        .map_err(PrepareError::into_py)
    });
    if let Data::Owned(v) = &mut rs {
        v.zeroize();
    }
    let payload = payload?;
    let xhat = if want_xhat {
        Some(
            Array2::from_shape_vec((rows, n), xhat).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
        )
    } else {
        None
    };
    Ok((payload, xhat, sx.into_pyarray(py)))
}

// ---- prepare_many: several calls (sessions) in one native call, over a persistent pool ----------

/// A persistent worker pool for [`prepare_many_into`] (spawning threads per call costs about
/// 20-50 us each, as much as a whole decode-sized prepare).
///
/// One job at a time: a caller that finds the pool busy (another Python thread released the GIL
/// into it) runs its tasks on its own thread instead, which gives the same results.
pub mod pool {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock, TryLockError};
    use std::time::{Duration, Instant};

    /// Upper bound on worker threads (the gateway also runs the event loop and the wire client).
    const MAX_WORKERS: usize = 7;
    /// Upper bound with `PGW_POOL_WORKERS` set explicitly (a 32-48 core gateway).
    const MAX_WORKERS_OPT_IN: usize = 47;
    /// Workers poll for the next job this long before sleeping on the condvar (a condvar wake-up
    /// costs 10-50 us natively, more in a gVisor sandbox; jobs of one step come back to back).
    /// `PGW_POOL_SPIN_US` (read once) overrides it, e.g. to keep the workers awake across a decode
    /// step's float work and round trips.
    const SPIN_DEFAULT_US: u64 = 50;

    fn spin() -> Duration {
        static SPIN: OnceLock<Duration> = OnceLock::new();
        *SPIN.get_or_init(|| {
            let us = std::env::var("PGW_POOL_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok());
            Duration::from_micros(us.unwrap_or(SPIN_DEFAULT_US).min(1_000_000))
        })
    }

    /// Opt-in "warm while active" spin (`PGW_POOL_ACTIVE_SPIN_US`, default 0 = off): while at least one
    /// [`Warm`] guard is alive (the native decode drivers hold one for each call, i.e. for a decode pass's
    /// trips, float work and lane waits), idle workers keep polling for the next job up to this long
    /// after their last job instead of sleeping on the condvar after [`spin`]. Outside a pass they fall
    /// back to the ordinary short spin, so an idle gateway does not burn its cores (and does not take
    /// them from a co-located mask factory). Timing only: which thread runs a task never changes a bit.
    fn active_spin() -> Duration {
        static SPIN: OnceLock<Duration> = OnceLock::new();
        *SPIN.get_or_init(|| {
            let us = std::env::var("PGW_POOL_ACTIVE_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok());
            Duration::from_micros(us.unwrap_or(0).min(1_000_000))
        })
    }

    /// Live [`Warm`] guards (process-wide).
    static ACTIVE: AtomicUsize = AtomicUsize::new(0);

    /// Keeps the pool's workers polling while alive (see [`active_spin`]; a no-op when it is off).
    pub struct Warm(bool);

    impl Warm {
        /// Start a warm window (wakes sleeping workers so they are polling when the first job comes).
        pub fn new() -> Warm {
            if active_spin().is_zero() {
                return Warm(false);
            }
            if ACTIVE.fetch_add(1, Ordering::AcqRel) == 0 {
                if let Some(p) = GLOBAL.get() {
                    let _slot = p.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
                    p.shared.cv.notify_all();
                }
            }
            Warm(true)
        }
    }

    impl Default for Warm {
        fn default() -> Self {
            Warm::new()
        }
    }

    impl Drop for Warm {
        fn drop(&mut self) {
            if self.0 {
                ACTIVE.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    /// `PGW_POOL_CPUS` (read once; default unset = no pinning): a CPU list such as `2-9,12` the workers
    /// are pinned to (worker `i` on the `i`-th CPU of the list, cyclically), so the trip split's workers
    /// own their cores and do not share them with a co-located factory or the attention pool
    /// (give those the other CPUs). Linux only; a failed pin leaves the worker unpinned.
    fn pin_cpus() -> &'static [usize] {
        static CPUS: OnceLock<Vec<usize>> = OnceLock::new();
        CPUS.get_or_init(|| std::env::var("PGW_POOL_CPUS").ok().map(|v| parse_cpu_list(&v)).unwrap_or_default())
    }

    /// `"2-5,8"` -> `[2, 3, 4, 5, 8]` (invalid parts are skipped).
    pub fn parse_cpu_list(v: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in v.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => {
                    if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                        if a <= b && b < 4096 {
                            out.extend(a..=b);
                        }
                    }
                }
                None => {
                    if let Ok(c) = part.parse::<usize>() {
                        if c < 4096 {
                            out.push(c);
                        }
                    }
                }
            }
        }
        out
    }

    #[cfg(target_os = "linux")]
    fn pin_to(cpu: usize) {
        // SAFETY: a zeroed cpu_set_t is a valid empty set; the index is checked against its size;
        // sched_setaffinity(0, ...) only changes the calling thread's mask.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if cpu < 8 * std::mem::size_of::<libc::cpu_set_t>() {
                libc::CPU_SET(cpu, &mut set);
                let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn pin_to(_cpu: usize) {}

    /// Lifetime-erased task function. Only called for task indices `< n`, while the submitting
    /// caller waits in [`Pool::run`], so the borrow it points to is alive at every call.
    struct FnPtr(*const (dyn Fn(usize) + Sync));
    // SAFETY: the pointee is `Sync` and outlives every call (see `Pool::run`).
    unsafe impl Send for FnPtr {}
    // SAFETY: as above.
    unsafe impl Sync for FnPtr {}

    struct Job {
        f: FnPtr,
        n: usize,
        next: AtomicUsize,
        done: AtomicUsize,
        panicked: AtomicBool,
    }

    impl Job {
        /// Claim and run tasks until none is left.
        fn work(&self) {
            loop {
                let i = self.next.fetch_add(1, Ordering::AcqRel);
                if i >= self.n {
                    return;
                }
                // SAFETY: `i < n`, so the submitter is still waiting for this task (FnPtr docs).
                let f = unsafe { &*self.f.0 };
                if catch_unwind(AssertUnwindSafe(|| f(i))).is_err() {
                    self.panicked.store(true, Ordering::Release);
                }
                self.done.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    struct Shared {
        slot: Mutex<Option<Arc<Job>>>,
        cv: Condvar,
        generation: AtomicU64,
    }

    /// The pool (see the module docs).
    pub struct Pool {
        shared: Arc<Shared>,
        busy: Mutex<()>,
        workers: usize,
    }

    fn worker(shared: Arc<Shared>, cpu: Option<usize>) {
        if let Some(c) = cpu {
            pin_to(c);
        }
        let mut seen = 0u64;
        let spin = spin();
        let warm = active_spin();
        loop {
            let t0 = Instant::now();
            loop {
                if shared.generation.load(Ordering::Acquire) != seen {
                    break;
                }
                let el = t0.elapsed();
                if el >= spin && (warm.is_zero() || el >= warm || ACTIVE.load(Ordering::Acquire) == 0) {
                    break;
                }
                std::hint::spin_loop();
            }
            let job = {
                let mut slot = shared.slot.lock().unwrap_or_else(|e| e.into_inner());
                let mut rewarm = false;
                while shared.generation.load(Ordering::Acquire) == seen {
                    if !warm.is_zero() && ACTIVE.load(Ordering::Acquire) > 0 {
                        rewarm = true; // a warm window is open (or just opened): poll again
                        break;
                    }
                    slot = shared.cv.wait(slot).unwrap_or_else(|e| e.into_inner());
                }
                if rewarm {
                    drop(slot);
                    continue;
                }
                seen = shared.generation.load(Ordering::Acquire);
                slot.clone()
            };
            if let Some(job) = job {
                job.work();
            }
        }
    }

    impl Pool {
        /// A pool with `workers` threads (fewer if spawning fails).
        pub fn new(workers: usize) -> Pool {
            let shared = Arc::new(Shared { slot: Mutex::new(None), cv: Condvar::new(), generation: AtomicU64::new(0) });
            let mut started = 0;
            let cpus = pin_cpus();
            for i in 0..workers {
                let s = Arc::clone(&shared);
                let cpu = if cpus.is_empty() { None } else { Some(cpus[i % cpus.len()]) };
                let ok =
                    std::thread::Builder::new().name(format!("pgw-prep-{i}")).spawn(move || worker(s, cpu)).is_ok();
                started += usize::from(ok);
            }
            Pool { shared, busy: Mutex::new(()), workers: started }
        }

        /// Worker threads (the caller adds one more).
        pub fn workers(&self) -> usize {
            self.workers
        }

        /// Run `f(i)` for every `i in 0..n`, on the workers and the calling thread; returns when
        /// all calls returned. `false` if a call panicked (every other call still ran).
        pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
            if n == 0 {
                return true;
            }
            let guard = match self.busy.try_lock() {
                Ok(g) => Some(g),
                Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if guard.is_none() || self.workers == 0 || n == 1 {
                let mut ok = true;
                for i in 0..n {
                    ok &= catch_unwind(AssertUnwindSafe(|| f(i))).is_ok();
                }
                return ok;
            }
            let fp: *const (dyn Fn(usize) + Sync + '_) = f;
            // SAFETY (lifetime erasure only): `f` is called only for indices < n, and this function
            // does not return before all n calls completed (`done == n` below).
            let fp: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(fp) };
            let job = Arc::new(Job {
                f: FnPtr(fp),
                n,
                next: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                panicked: AtomicBool::new(false),
            });
            {
                let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
                *slot = Some(Arc::clone(&job));
                self.shared.generation.fetch_add(1, Ordering::AcqRel);
            }
            self.shared.cv.notify_all();
            job.work();
            while job.done.load(Ordering::Acquire) < n {
                std::hint::spin_loop();
                std::thread::yield_now();
            }
            {
                let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
                *slot = None; // late workers find nothing (and would only touch the Job's counters)
            }
            drop(guard);
            !job.panicked.load(Ordering::Acquire)
        }
    }

    static GLOBAL: OnceLock<Pool> = OnceLock::new();

    /// The process-wide pool: `min(available_parallelism, 8) - 1` workers (`available_parallelism`
    /// honours the cgroup CPU quota), started on first use. `PGW_POOL_WORKERS` (read once) sets the
    /// worker count: lower (e.g. to leave cores to a co-located mask factory) or, opt-in, higher than
    /// the default 7 for a many-core gateway splitting 32B/72B trips (at most `available_parallelism - 1`
    /// and `MAX_WORKERS_OPT_IN`).
    pub fn global() -> &'static Pool {
        GLOBAL.get_or_init(|| {
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
            let mut workers = cpus.min(MAX_WORKERS + 1).saturating_sub(1);
            if let Some(w) = std::env::var("PGW_POOL_WORKERS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
                workers = w.min(cpus.saturating_sub(1)).min(MAX_WORKERS_OPT_IN);
            }
            Pool::new(workers)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cpu_list_parsing() {
            assert_eq!(parse_cpu_list("2-5,8"), vec![2, 3, 4, 5, 8]);
            assert_eq!(parse_cpu_list(" 1 , 3-3 ,x, 9-7"), vec![1, 3]);
            assert!(parse_cpu_list("").is_empty());
        }

        #[test]
        fn warm_guard_keeps_results() {
            let _w = Warm::new();
            let p = Pool::new(2);
            let sum = AtomicUsize::new(0);
            assert!(p.run(64, &|i| {
                sum.fetch_add(i, Ordering::Relaxed);
            }));
            assert_eq!(sum.load(Ordering::Relaxed), 64 * 63 / 2);
        }

        #[test]
        fn runs_every_index_once() {
            let p = Pool::new(3);
            for n in [0usize, 1, 2, 7, 100] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                assert!(p.run(n, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                }));
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
            }
        }

        #[test]
        fn concurrent_callers_and_panics() {
            let p = Arc::new(Pool::new(2));
            let hs: Vec<_> = (0..4)
                .map(|_| {
                    let p = Arc::clone(&p);
                    std::thread::spawn(move || {
                        for _ in 0..200 {
                            let sum = AtomicUsize::new(0);
                            assert!(p.run(16, &|i| {
                                sum.fetch_add(i, Ordering::Relaxed);
                            }));
                            assert_eq!(sum.load(Ordering::Relaxed), 120);
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let count = AtomicUsize::new(0);
            let ok = p.run(8, &|i| {
                count.fetch_add(1, Ordering::Relaxed);
                assert!(i != 3, "task 3 fails");
            });
            assert!(!ok);
            assert_eq!(count.load(Ordering::Relaxed), 8);
        }
    }
}

/// Below this many activation elements in total `quantize_mask_pack_many` stays on the calling
/// thread (a decode step of a few sessions: waking workers would cost about what it saves).
pub const MANY_PAR_MIN_ELEMS: usize = 48 * 1024;
/// Rows are grouped into tasks of at least this many elements.
const MANY_TASK_ELEMS: usize = 16 * 1024;

/// One task of [`prepare_many_into`]: whole rows of one item.
pub struct ManyTask<'a> {
    /// Index of the item the rows belong to.
    pub item: usize,
    /// Activations `rows x n`.
    pub x: &'a [f32],
    /// Masks `rows x n`.
    pub r: &'a [i64],
    /// `x_hat` output (`rows x n`), or empty for a payload-only item.
    pub xhat: &'a mut [i64],
    /// Scales (`rows`).
    pub sx: &'a mut [f64],
    /// Wire bytes (`rows x n x K/8`).
    pub payload: &'a mut [u8],
}

/// Non-finite input outranks a cap overflow within one item (the single-item kernel checks every
/// row's finiteness before quantizing any row).
fn is_nonfinite(e: &PrepareError) -> bool {
    matches!(e, PrepareError::Overflow(m) if m.starts_with("non-finite"))
}

/// Run `tasks` (each: [`quantize_mask_pack_into`], or the payload-only variant when `xhat` is
/// empty) with the parameters `params[item]` and width `widths[item]` of its item, in parallel on
/// the global pool when `parallel`. Returns the error of the lowest failing item (the one the
/// per-item calls in order would have raised), with every output zeroed after a failure.
pub fn prepare_many_into(
    tasks: Vec<ManyTask<'_>>,
    params: &[PrepareParams],
    widths: &[usize],
    parallel: bool,
) -> Result<(), PrepareError> {
    let cells: Vec<Mutex<ManyTask<'_>>> = tasks.into_iter().map(Mutex::new).collect();
    let errs: Vec<Mutex<Option<PrepareError>>> = (0..params.len()).map(|_| Mutex::new(None)).collect();
    let run_one = |i: usize| {
        let mut guard = cells[i].lock().unwrap_or_else(|e| e.into_inner());
        let t = &mut *guard;
        let (p, n) = (params[t.item], widths[t.item]);
        let res = if t.xhat.is_empty() {
            quantize_mask_pack_payload(t.x, t.r, n, p, t.sx, t.payload)
        } else {
            quantize_mask_pack_into(t.x, t.r, n, p, t.xhat, t.sx, t.payload)
        };
        if let Err(e) = res {
            let mut slot = errs[t.item].lock().unwrap_or_else(|e| e.into_inner());
            let replace = match &*slot {
                None => true,
                Some(old) => is_nonfinite(&e) && !is_nonfinite(old),
            };
            if replace {
                *slot = Some(e);
            }
        }
    };
    let ok = if parallel {
        pool::global().run(cells.len(), &run_one)
    } else {
        let mut ok = true;
        for i in 0..cells.len() {
            ok &= std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(i))).is_ok();
        }
        ok
    };
    let mut first = if ok { None } else { Some(PrepareError::Invalid("internal error in a prepare task".into())) };
    if first.is_none() {
        first = errs.iter().find_map(|e| e.lock().unwrap_or_else(|e| e.into_inner()).take());
    }
    match first {
        None => Ok(()),
        Some(err) => {
            for c in &cells {
                let mut t = c.lock().unwrap_or_else(|e| e.into_inner());
                t.xhat.zeroize();
                t.payload.zeroize();
            }
            Err(err)
        }
    }
}

/// Rows per task for an item of `rows x n` (whole rows, at least [`MANY_TASK_ELEMS`] elements).
fn rows_per_task(rows: usize, n: usize, parallel: bool) -> usize {
    if !parallel {
        return rows.max(1);
    }
    MANY_TASK_ELEMS.div_ceil(n.max(1)).clamp(1, rows.max(1))
}

/// `quantize_mask_pack_many(xs, rs, k, a, cap, wnorms, want_xhat) -> [(payload, x_hat | None, sx)]`.
///
/// [`quantize_mask_pack`] for several items (the wire items of one round trip: every session's
/// call of a linear, row blocks of a prefill) in one call. Item `i` uses `wnorms[i]` and
/// `want_xhat[i]`; results are bit-identical to calling `quantize_mask_pack` per item in order,
/// including which error is raised (the lowest failing item's). Large batches are spread over a
/// persistent worker pool (rows are independent, so the split changes no byte); small ones (decode
/// of a few sessions) run on the calling thread. The GIL is released while computing.
#[pyfunction]
#[pyo3(signature = (xs, rs, k, a, cap, wnorms, want_xhat))]
#[allow(clippy::too_many_arguments)]
fn quantize_mask_pack_many<'py>(
    py: Python<'py>,
    xs: Vec<Bound<'py, PyAny>>,
    rs: Vec<Bound<'py, PyAny>>,
    k: u32,
    a: u32,
    cap: u32,
    wnorms: Vec<f64>,
    want_xhat: Vec<bool>,
) -> PyResult<Vec<Prepared<'py>>> {
    let m = xs.len();
    if rs.len() != m || wnorms.len() != m || want_xhat.len() != m {
        return Err(PyValueError::new_err("xs, rs, wnorms and want_xhat must have the same length"));
    }
    PrepareParams { k, a, cap, wnorm: 0.0 }.validate().map_err(PrepareError::into_py)?;
    let params: Vec<PrepareParams> = wnorms.iter().map(|&wnorm| PrepareParams { k, a, cap, wnorm }).collect();
    let nb = (k / 8) as usize;
    let mut xro = Vec::with_capacity(m);
    let mut rro = Vec::with_capacity(m);
    for (x, r) in xs.iter().zip(&rs) {
        let xa = x
            .downcast::<PyArray2<f32>>()
            .map_err(|_| PyValueError::new_err("x must be a 2-D float32 numpy array"))?
            .readonly();
        let ra = r
            .downcast::<PyArray2<i64>>()
            .map_err(|_| PyValueError::new_err("r must be a 2-D int64 numpy array"))?
            .readonly();
        if xa.shape() != ra.shape() {
            return Err(PyValueError::new_err(format!(
                "activation shape {:?} != mask shape {:?}",
                xa.shape(),
                ra.shape()
            )));
        }
        xro.push(xa);
        rro.push(ra);
    }
    let shapes: Vec<(usize, usize)> = xro.iter().map(|a| (a.shape()[0], a.shape()[1])).collect();
    let widths: Vec<usize> = shapes.iter().map(|s| s.1).collect();
    if shapes.iter().any(|&(rows, n)| rows > 0 && n == 0) {
        return Err(PyValueError::new_err("activations must have at least one feature"));
    }
    let xd: Vec<Data<'_, f32>> = xro
        .iter()
        .map(|a| match a.as_slice().ok().filter(|_| a.is_c_contiguous()) {
            Some(s) => Data::Borrowed(s), // C order only: `as_slice` also accepts Fortran order
            None => Data::Owned(a.as_array().iter().copied().collect()),
        })
        .collect();
    let mut rd: Vec<Data<'_, i64>> = rro
        .iter()
        .map(|a| match a.as_slice().ok().filter(|_| a.is_c_contiguous()) {
            Some(s) => Data::Borrowed(s), // C order only: `as_slice` also accepts Fortran order
            None => Data::Owned(a.as_array().iter().copied().collect()),
        })
        .collect();
    // Outputs are allocated with the GIL held and filled without it.
    let mut payloads: Vec<Bound<'py, PyBytes>> = Vec::with_capacity(m);
    let mut xhats: Vec<Option<Bound<'py, PyArray2<i64>>>> = Vec::with_capacity(m);
    let mut sxs: Vec<Bound<'py, numpy::PyArray1<f64>>> = Vec::with_capacity(m);
    for (i, &(rows, n)) in shapes.iter().enumerate() {
        let len = rows
            .checked_mul(n)
            .and_then(|e| e.checked_mul(nb))
            .ok_or_else(|| PyValueError::new_err("activation shape too large"))?;
        payloads.push(PyBytes::new_with(py, len, |_| Ok(()))?);
        xhats.push(if want_xhat[i] { Some(PyArray2::<i64>::zeros(py, [rows, n], false)) } else { None });
        sxs.push(numpy::PyArray1::<f64>::from_vec(py, vec![1.0f64; rows]));
    }
    let total: usize = shapes.iter().map(|s| s.0 * s.1).sum();
    let parallel = total >= MANY_PAR_MIN_ELEMS && pool::global().workers() > 0;
    let res = {
        let mut tasks: Vec<ManyTask<'_>> = Vec::new();
        for i in 0..m {
            let (rows, n) = shapes[i];
            if rows == 0 {
                continue;
            }
            // SAFETY: a fresh bytes object of `rows * n * nb` bytes that no Python code has seen
            // yet; it is written only here (by the tasks below), before it is returned.
            let pl: &mut [u8] = unsafe {
                let ptr = pyo3::ffi::PyBytes_AsString(payloads[i].as_ptr()).cast::<u8>();
                std::slice::from_raw_parts_mut(ptr, rows * n * nb)
            };
            let xh: &mut [i64] = match &xhats[i] {
                // SAFETY: fresh C-contiguous array, not shared with Python code yet.
                Some(arr) => unsafe { arr.as_slice_mut() }.map_err(|e| PyValueError::new_err(e.to_string()))?,
                None => &mut [],
            };
            // SAFETY: as above.
            let sx: &mut [f64] = unsafe { sxs[i].as_slice_mut() }.map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (xs_i, rs_i) = (xd[i].slice(), rd[i].slice());
            let per = rows_per_task(rows, n, parallel);
            let (mut xh_rest, mut sx_rest, mut pl_rest) = (xh, sx, pl);
            let mut row = 0;
            while row < rows {
                let take = per.min(rows - row);
                let (sx_t, sx_r) = sx_rest.split_at_mut(take);
                let (pl_t, pl_r) = pl_rest.split_at_mut(take * n * nb);
                let xh_t: &mut [i64] = if xh_rest.is_empty() {
                    &mut []
                } else {
                    let (a_, b_) = xh_rest.split_at_mut(take * n);
                    xh_rest = b_;
                    a_
                };
                tasks.push(ManyTask {
                    item: i,
                    x: &xs_i[row * n..(row + take) * n],
                    r: &rs_i[row * n..(row + take) * n],
                    xhat: xh_t,
                    sx: sx_t,
                    payload: pl_t,
                });
                sx_rest = sx_r;
                pl_rest = pl_r;
                row += take;
            }
        }
        py.allow_threads(|| prepare_many_into(tasks, &params, &widths, parallel))
    };
    for d in &mut rd {
        if let Data::Owned(v) = d {
            v.zeroize();
        }
    }
    res.map_err(PrepareError::into_py)?;
    Ok(payloads.into_iter().zip(xhats).zip(sxs).map(|((p, x), s)| (p, x, s)).collect())
}

/// Worker threads of the persistent prepare pool (0: every call runs on the calling thread).
#[pyfunction]
fn prepare_pool_workers() -> usize {
    pool::global().workers()
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_mask_pack, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_mask_pack_many, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_pool_workers, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type Out = (Vec<i64>, Vec<f64>, Vec<u8>);

    fn run(x: &[f32], r: &[i64], n: usize, p: PrepareParams) -> Result<Out, PrepareError> {
        let rows = x.len() / n;
        let mut xh = vec![0; rows * n];
        let mut sx = vec![0.0; rows];
        let mut pl = vec![0; rows * n * (p.k as usize / 8)];
        quantize_mask_pack_into(x, r, n, p, &mut xh, &mut sx, &mut pl)?;
        Ok((xh, sx, pl))
    }

    const K32A12: PrepareParams = PrepareParams { k: 32, a: 12, cap: 0, wnorm: 0.0 };

    #[test]
    fn uncapped_rounds_half_even_and_masks() {
        // amax 2047 -> sx = 1; 0.5 -> 0, 1.5 -> 2, -2.5 -> -2
        let x = [2047.0f32, 0.5, 1.5, -2.5];
        let r = [0i64, 1, (1 << 32) - 1, 5];
        let (xh, sx, pl) = run(&x, &r, 4, K32A12).unwrap();
        assert_eq!(sx, vec![1.0]);
        assert_eq!(xh, vec![2047, 1, 1, 3]);
        assert_eq!(&pl[..4], &2047u32.to_le_bytes());
        assert_eq!(&pl[12..], &3u32.to_le_bytes());
    }

    #[test]
    fn zero_row_uses_unit_scale() {
        let (xh, sx, _) = run(&[0.0, -0.0], &[7, 9], 2, K32A12).unwrap();
        assert_eq!(sx, vec![1.0]);
        assert_eq!(xh, vec![7, 9]);
    }

    #[test]
    fn packs_k40() {
        let p = PrepareParams { k: 40, a: 12, cap: 0, wnorm: 0.0 };
        let (xh, _, pl) = run(&[-2047.0], &[0], 1, p).unwrap();
        assert_eq!(xh, vec![(1i64 << 40) - 2047]);
        assert_eq!(pl, ((1u64 << 40) - 2047).to_le_bytes()[..5].to_vec());
    }

    #[test]
    fn capped_drops_bits() {
        // q at 12 bits is 2047 -> 2047 * wnorm > 2^20 for wnorm 1024; at 11 bits 1023*1024 <= 2^20
        let p = PrepareParams { k: 32, a: 12, cap: 20, wnorm: 1024.0 };
        let (xh, sx, _) = run(&[1.0], &[0], 1, p).unwrap();
        assert_eq!(xh, vec![1023]);
        assert_eq!(sx, vec![1.0 / 1023.0]);
    }

    #[test]
    fn errors() {
        assert!(matches!(run(&[f32::NAN], &[0], 1, K32A12), Err(PrepareError::Overflow(_))));
        assert!(matches!(run(&[f32::INFINITY], &[0], 1, K32A12), Err(PrepareError::Overflow(_))));
        let p = PrepareParams { k: 32, a: 12, cap: 20, wnorm: 1e30 };
        assert!(matches!(run(&[1.0], &[0], 1, p), Err(PrepareError::Overflow(_))));
        let bad = PrepareParams { k: 33, a: 12, cap: 0, wnorm: 0.0 };
        assert!(matches!(run(&[1.0], &[0], 1, bad), Err(PrepareError::Invalid(_))));
    }

    /// xorshift64* (deterministic, no extra dependency)
    fn rng(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[test]
    fn fast_quant_matches_division_near_ties() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let n = 64;
        let mut q_fast = vec![0i64; n];
        let mut q_exact = vec![0i64; n];
        for trial in 0..20_000 {
            let qmax = qmax_of(2 + (rng(&mut st) % 11) as u32);
            let amax = f32::from_bits(0x0080_0000 + (rng(&mut st) % 0x7e00_0000) as u32).abs();
            let s = f64::from(amax) / qmax;
            let xr: Vec<f32> = (0..n)
                .map(|_| {
                    let u = rng(&mut st);
                    let v = if trial % 2 == 0 {
                        // near a half-integer multiple of s, nudged by a few f32 ulps
                        let h = ((u % (2 * qmax as u64 + 1)) as f64 - qmax) + 0.5;
                        let base = (h * s) as f32;
                        f32::from_bits((base.to_bits() as i64 + (u >> 60) as i64 - 8) as u32)
                    } else {
                        amax * ((u >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
                    };
                    if v.is_finite() && v.abs() <= amax {
                        v
                    } else {
                        0.0
                    }
                })
                .collect();
            quant_row(&xr, s, qmax, &mut q_fast);
            quant_row_exact(&xr, s, qmax, &mut q_exact);
            assert_eq!(q_fast, q_exact, "trial {trial}");
            let f: f64 = q_exact.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            assert_eq!(sum_squares(&q_fast, qmax), f);
        }
    }

    #[test]
    fn exact_quant_is_numpy_rint() {
        // round half to even on exact ties, both signs; clip at qmax
        let q = &mut [0i64; 8];
        quant_row_exact(&[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 10.0, -10.0], 1.0, 3.0, q);
        assert_eq!(q, &[0, 2, 2, 0, -2, -2, 3, -3]);
        let q2 = &mut [0i64; 8];
        quant_row(&[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 10.0, -10.0], 1.0, 3.0, q2);
        assert_eq!(q, q2);
    }

    /// Items of mixed widths split into row tasks (on the pool or not): the same bytes as one
    /// `quantize_mask_pack_into` per item.
    #[test]
    fn many_matches_per_item() {
        let mut st = 0x1234_5678_9abc_def1u64;
        let shapes = [(40usize, 896usize), (1, 3584), (3, 7), (64, 1024), (0, 16)];
        let xs: Vec<Vec<f32>> = shapes
            .iter()
            .map(|&(r, n)| (0..r * n).map(|_| ((rng(&mut st) >> 40) as f32 / 1e5) - 80.0).collect())
            .collect();
        let rs: Vec<Vec<i64>> =
            shapes.iter().map(|&(r, n)| (0..r * n).map(|_| (rng(&mut st) >> 32) as i64).collect()).collect();
        let params: Vec<PrepareParams> =
            (0..shapes.len()).map(|i| PrepareParams { k: 32, a: 12, cap: 27, wnorm: 50.0 + i as f64 }).collect();
        let widths: Vec<usize> = shapes.iter().map(|s| s.1).collect();
        for parallel in [false, true] {
            for with_xhat in [false, true] {
                let mut outs: Vec<Out> = shapes
                    .iter()
                    .map(|&(r, n)| (vec![0; if with_xhat { r * n } else { 0 }], vec![1.0; r], vec![0u8; r * n * 4]))
                    .collect();
                let mut tasks = Vec::new();
                for (i, ((xh, sx, pl), &(rows, n))) in outs.iter_mut().zip(&shapes).enumerate() {
                    let per = rows_per_task(rows, n, parallel).min(7);
                    let (mut xr, mut sr, mut pr) = (xh.as_mut_slice(), sx.as_mut_slice(), pl.as_mut_slice());
                    let mut row = 0;
                    while row < rows {
                        let t = per.min(rows - row);
                        let (sa, sb) = sr.split_at_mut(t);
                        let (pa, pb) = pr.split_at_mut(t * n * 4);
                        let xa: &mut [i64] = if xr.is_empty() {
                            &mut []
                        } else {
                            let (a, b) = xr.split_at_mut(t * n);
                            xr = b;
                            a
                        };
                        tasks.push(ManyTask {
                            item: i,
                            x: &xs[i][row * n..(row + t) * n],
                            r: &rs[i][row * n..(row + t) * n],
                            xhat: xa,
                            sx: sa,
                            payload: pa,
                        });
                        sr = sb;
                        pr = pb;
                        row += t;
                    }
                }
                prepare_many_into(tasks, &params, &widths, parallel).unwrap();
                for (i, (xh, sx, pl)) in outs.iter().enumerate() {
                    let n = shapes[i].1;
                    let (wx, wsx, wpl) = run(&xs[i], &rs[i], n, params[i]).unwrap();
                    assert_eq!(pl, &wpl, "item {i}");
                    let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(sx), bits(&wsx), "item {i}");
                    if with_xhat {
                        assert_eq!(xh, &wx, "item {i}");
                    }
                }
            }
        }
    }

    /// The lowest failing item's error; inside an item a non-finite value outranks a cap overflow
    /// (as in the single-item kernel), and outputs are zeroed.
    #[test]
    fn many_error_order() {
        let bad_cap = PrepareParams { k: 32, a: 12, cap: 20, wnorm: 1e30 };
        let ok = PrepareParams { k: 32, a: 12, cap: 0, wnorm: 0.0 };
        let x = [1.0f32, 1.0, 1.0, f32::NAN];
        let r = [0i64; 4];
        for parallel in [false, true] {
            let (mut s, mut p) = (vec![1.0; 2], vec![7u8; 16]);
            let (sa, sb) = s.split_at_mut(1);
            let (pa, pb) = p.split_at_mut(8);
            let tasks = vec![
                ManyTask { item: 0, x: &x[..2], r: &r[..2], xhat: &mut [], sx: sa, payload: pa },
                ManyTask { item: 0, x: &x[2..], r: &r[2..], xhat: &mut [], sx: sb, payload: pb },
            ];
            let e = prepare_many_into(tasks, &[bad_cap], &[2], parallel).unwrap_err();
            assert!(is_nonfinite(&e), "{e:?}");
            assert!(p.iter().all(|b| *b == 0));
            // item 0 fine, item 1 cap overflow, item 2 non-finite: item 1's error
            let (mut s3, mut p3) = (vec![1.0; 3], vec![0u8; 12]);
            let (s0, rest) = s3.split_at_mut(1);
            let (s1, s2) = rest.split_at_mut(1);
            let (p0, rest) = p3.split_at_mut(4);
            let (p1, p2) = rest.split_at_mut(4);
            let tasks = vec![
                ManyTask { item: 0, x: &x[..1], r: &r[..1], xhat: &mut [], sx: s0, payload: p0 },
                ManyTask { item: 1, x: &x[1..2], r: &r[1..2], xhat: &mut [], sx: s1, payload: p1 },
                ManyTask { item: 2, x: &x[3..], r: &r[3..], xhat: &mut [], sx: s2, payload: p2 },
            ];
            let e = prepare_many_into(tasks, &[ok, bad_cap, ok], &[1, 1, 1], parallel).unwrap_err();
            assert!(matches!(e, PrepareError::Overflow(ref m) if m.starts_with("K32A12C20")), "{e:?}");
        }
    }
}
