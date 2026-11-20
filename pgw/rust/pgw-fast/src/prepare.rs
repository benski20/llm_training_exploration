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
use zeroize::Zeroize;

/// Relative slack on the float64 norm product (`pgw.core.ring._NORM_SLACK`).
const NORM_SLACK: f64 = 1.0 + 1e-9;

/// Kernel errors (mapped to Python `ValueError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareError {
    /// Bad arguments (shape, ring parameters).
    Invalid(String),
    /// Non-finite input or no bit width fits the cap (Python `RingOverflowError`).
    Overflow(String),
}

impl PrepareError {
    fn into_py(self) -> PyErr {
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
    fn validate(&self) -> Result<(), PrepareError> {
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
fn qmax_of(bits: u32) -> f64 {
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
fn quant_biased(v: f32, s: f64, qmax: f64) -> f64 {
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
fn biased_to_i64(b: f64) -> i64 {
    (b.to_bits() as i64).wrapping_sub(MAGIC_BITS)
}

/// Relative tolerance of the reciprocal-multiply fast path: `x * fl(1/s)` differs from the correctly
/// rounded `fl(x / s)` by at most about `|x/s| * 2^-51` (three roundings), far below `2^-40`.
const FAST_TOL: f64 = 1.0 / (1u64 << 40) as f64;

/// Quantize one row into `q` with true division (numpy's exact operation sequence).
#[inline(always)]
fn quant_row_exact(xr: &[f32], s: f64, qmax: f64, q: &mut [i64]) {
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
fn sum_squares(q: &[i64], qmax: f64) -> f64 {
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
fn row_amax_bits(row: &[f32]) -> u32 {
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
const INF_BITS: u32 = 0x7f80_0000;

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
fn mask_pack_row(xq: &[i64], r: &[i64], k: u32, xhat: &mut [i64], out: &mut [u8]) {
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
    let xs = match xa.as_slice() {
        Ok(s) => Data::Borrowed(s),
        Err(_) => Data::Owned(xa.as_array().iter().copied().collect()),
    };
    let mut rs = match ra.as_slice() {
        Ok(s) => Data::Borrowed(s),
        Err(_) => Data::Owned(ra.as_array().iter().copied().collect()),
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

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_mask_pack, m)?)?;
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
}
