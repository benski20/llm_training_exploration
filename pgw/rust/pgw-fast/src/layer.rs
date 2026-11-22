//! Native deterministic float path of one decoder layer (`float_ops = "native"`, opt-in).
//!
//! A **pinned formulation** of the gateway's plaintext float ops: RMSNorm, the residual add and the
//! SiLU gate, each in a fixed order of IEEE-754 operations (no FMA: Rust never contracts `a * b + c`,
//! and nothing here uses `mul_add`; no libm: `exp` is [`exp_f64`], a fixed polynomial), so the bits do
//! not depend on the CPU's instruction set, the thread count or how many rows share a call. Each
//! output row depends only on that row's inputs.
//!
//! It is a **new reference**: RMSNorm and SiLU differ from torch's in the last bits (torch's reduction
//! order and vectorized `exp` depend on the ISA), so the gateway and the plaintext reference must both
//! select it (`TransformerModel.float_ops = NativeFloatOps`). The residual add is torch's bit for bit.
//!
//! * RMSNorm: `s = sum x_i^2` in float64 (each square is exact in float64; 8 lane sums over `i mod 8`,
//!   then a fixed tree), `r = (1 / sqrt(s / n + eps)) as f32`, `y_i = w_i * (x_i * r)` in float32 (the
//!   HF form's two float32 products).
//! * SiLU gate: `(g / (1 + exp(-g))) as f32 * u` (the quotient in float64, one rounding to float32,
//!   then torch's float32 product).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(float_rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(float_add_rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(float_add, m)?)?;
    m.add_function(wrap_pyfunction!(float_silu_mul, m)?)?;
    Ok(())
}

pub(crate) fn verr(s: impl Into<String>) -> PyErr {
    PyValueError::new_err(s.into())
}

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01; // upper bits of ln 2 (k * LN2_HI exact for |k| < 2^20)
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const LOG2E: f64 = std::f64::consts::LOG2_E;

/// `e^x` in float64 from IEEE operations only (no libm, no FMA, no branches, so it vectorizes):
/// `x = k ln2 + r`, `|r| <= ln2 / 2` (`k` rounded to nearest-even by adding `1.5 * 2^52`), a degree-12
/// Taylor polynomial in Horner form (truncation below 1e-16 relative), times `2^k` as two exact
/// power-of-two products. Relative error a few ulp of float64, far below float32 resolution. `x` is
/// clamped to `[-745, 709]` first (so `e^x` saturates at about `8.2e307` and `0`); NaN gives NaN.
#[inline(always)]
pub fn exp_f64(x: f64) -> f64 {
    const MAGIC: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    let xc = if x < -745.0 { -745.0 } else if x > 709.0 { 709.0 } else { x };
    let t = xc * LOG2E + MAGIC;
    let k = t - MAGIC;
    let r = (xc - k * LN2_HI) - k * LN2_LO;
    let mut p = 1.0 / 479_001_600.0; // 1/12!
    p = p * r + 1.0 / 39_916_800.0;
    p = p * r + 1.0 / 3_628_800.0;
    p = p * r + 1.0 / 362_880.0;
    p = p * r + 1.0 / 40_320.0;
    p = p * r + 1.0 / 5_040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // 2^k as 2^k1 * 2^k2, each a normal float64 (k in [-1075, 1023]; garbage only for NaN, which p
    // carries): k1 = k / 2 rounded, and the exponent field from the low bits of `k_i + 1023 + 1.5 * 2^52`
    // (floating-point adds of integers, exact; then one 64-bit shift: all vectorizable)
    let k1 = (k * 0.5 + MAGIC) - MAGIC;
    let k2 = k - k1;
    let f1 = f64::from_bits((k1 + (MAGIC + 1023.0)).to_bits() << 52);
    let f2 = f64::from_bits((k2 + (MAGIC + 1023.0)).to_bits() << 52);
    (p * f1) * f2
}

/// Defines `$name(args)`: `$body(args)` compiled once with AVX2 enabled (used when the CPU has it) and
/// once for the baseline. Wider vectors only: FMA stays off and Rust never contracts `a * b + c`, so
/// both run the same IEEE operations in the same order (the same bits).
macro_rules! isa_dispatch {
    ($vis:vis fn $name:ident => $body:ident ( $($a:ident : $t:ty),* $(,)? )) => {
        $vis fn $name($($a: $t),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($a: $t),*) {
                    $body($($a),*)
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 is available (checked above).
                    return unsafe { avx2($($a),*) };
                }
            }
            $body($($a),*)
        }
    };
}
pub(crate) use isa_dispatch;

/// `sum x_i^2` in float64: 8 lane sums (`i mod 8`, in order), then `((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7))`.
#[inline(always)]
pub fn sum_sq(x: &[f32]) -> f64 {
    let mut a = [0f64; 8];
    let mut ch = x.chunks_exact(8);
    for c in &mut ch {
        for i in 0..8 {
            let v = c[i] as f64;
            a[i] += v * v;
        }
    }
    for (i, &v) in ch.remainder().iter().enumerate() {
        let v = v as f64;
        a[i] += v * v;
    }
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// One row of RMSNorm (see the module docs).
#[inline(always)]
pub fn rms_norm_row(x: &[f32], w: &[f32], eps: f64, out: &mut [f32]) {
    let n = x.len();
    let r = (1.0 / (sum_sq(x) / n as f64 + eps).sqrt()) as f32;
    for i in 0..n {
        out[i] = w[i] * (x[i] * r);
    }
}

/// One row of the SiLU gate (see the module docs).
#[inline(always)]
pub fn silu_mul_row(g: &[f32], u: &[f32], out: &mut [f32]) {
    for i in 0..g.len() {
        let gv = g[i] as f64;
        let s = (gv / (1.0 + exp_f64(-gv))) as f32;
        out[i] = s * u[i];
    }
}

#[inline(always)]
fn rms_rows_body(xs: &[f32], ws: &[f32], eps: f64, os: &mut [f32], n: usize) {
    for (x, o) in xs.chunks_exact(n).zip(os.chunks_exact_mut(n)) {
        rms_norm_row(x, ws, eps, o);
    }
}
isa_dispatch!(fn rms_rows => rms_rows_body(xs: &[f32], ws: &[f32], eps: f64, os: &mut [f32], n: usize));

#[inline(always)]
fn add_body(hs: &[f32], ys: &[f32], os: &mut [f32]) {
    for ((o, &h), &y) in os.iter_mut().zip(hs).zip(ys) {
        *o = h + y;
    }
}
isa_dispatch!(fn add_rows => add_body(hs: &[f32], ys: &[f32], os: &mut [f32]));

#[inline(always)]
fn silu_rows_body(gs: &[f32], os: &mut [f32], inter: usize) {
    for (row, o) in gs.chunks_exact(2 * inter).zip(os.chunks_exact_mut(inter)) {
        silu_mul_row(&row[..inter], &row[inter..], o);
    }
}
isa_dispatch!(fn silu_rows => silu_rows_body(gs: &[f32], os: &mut [f32], inter: usize));

fn shape2(a: &Bound<'_, PyArray2<f32>>, name: &str) -> PyResult<(usize, usize)> {
    if !a.is_c_contiguous() {
        return Err(verr(format!("{name} must be C-contiguous float32")));
    }
    Ok((a.shape()[0], a.shape()[1]))
}

/// RMSNorm of each row: `out = w * (x * r)` (module docs). `x`, `out`: `(rows, n)`; `w`: `(n,)`.
#[pyfunction]
fn float_rms_norm<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f32>>,
    w: &Bound<'py, PyArray1<f32>>,
    eps: f64,
    out: &Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let (rows, n) = shape2(x, "x")?;
    if shape2(out, "out")? != (rows, n) || w.len() != n || !w.is_c_contiguous() {
        return Err(verr("float_rms_norm: shapes"));
    }
    let xr = x.try_readonly().map_err(|e| verr(format!("x: {e}")))?;
    let wr = w.try_readonly().map_err(|e| verr(format!("w: {e}")))?;
    let mut orw = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (xs, ws) = (xr.as_slice()?, wr.as_slice()?);
    let os = orw.as_slice_mut()?;
    let _ = rows;
    py.allow_threads(|| rms_rows(xs, ws, eps, os, n));
    Ok(())
}

/// Residual add then RMSNorm: `h_out = h + y` (float32, torch's bits), `out = rms_norm(h_out, w)`.
/// All `(rows, n)` except `w (n,)`; `h_out` must not be `h` or `y` (captured hidden states keep `h`).
#[pyfunction]
fn float_add_rms_norm<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyArray2<f32>>,
    y: &Bound<'py, PyArray2<f32>>,
    w: &Bound<'py, PyArray1<f32>>,
    eps: f64,
    h_out: &Bound<'py, PyArray2<f32>>,
    out: &Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let (rows, n) = shape2(h, "h")?;
    if shape2(y, "y")? != (rows, n) || shape2(h_out, "h_out")? != (rows, n) || shape2(out, "out")? != (rows, n)
        || w.len() != n || !w.is_c_contiguous()
    {
        return Err(verr("float_add_rms_norm: shapes"));
    }
    let hr = h.try_readonly().map_err(|e| verr(format!("h: {e}")))?;
    let yr = y.try_readonly().map_err(|e| verr(format!("y: {e}")))?;
    let wr = w.try_readonly().map_err(|e| verr(format!("w: {e}")))?;
    let mut hw = h_out.try_readwrite().map_err(|e| verr(format!("h_out: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (hs, ys, ws) = (hr.as_slice()?, yr.as_slice()?, wr.as_slice()?);
    let (ho, os) = (hw.as_slice_mut()?, ow.as_slice_mut()?);
    py.allow_threads(|| {
        add_rows(hs, ys, ho);
        rms_rows(ho, ws, eps, os, n);
    });
    Ok(())
}

/// Residual add `out = h + y` (float32; torch's bits).
#[pyfunction]
fn float_add<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyArray2<f32>>,
    y: &Bound<'py, PyArray2<f32>>,
    out: &Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let s = shape2(h, "h")?;
    if shape2(y, "y")? != s || shape2(out, "out")? != s {
        return Err(verr("float_add: shapes"));
    }
    let hr = h.try_readonly().map_err(|e| verr(format!("h: {e}")))?;
    let yr = y.try_readonly().map_err(|e| verr(format!("y: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (hs, ys) = (hr.as_slice()?, yr.as_slice()?);
    let os = ow.as_slice_mut()?;
    py.allow_threads(|| add_rows(hs, ys, os));
    Ok(())
}

/// SiLU gate of a `gate_up` result: `gu (rows, 2 * inter)` = `[gate | up]` per row, `out (rows, inter)`.
#[pyfunction]
fn float_silu_mul<'py>(
    py: Python<'py>,
    gu: &Bound<'py, PyArray2<f32>>,
    out: &Bound<'py, PyArray2<f32>>,
) -> PyResult<()> {
    let (rows, n2) = shape2(gu, "gu")?;
    let inter = n2 / 2;
    if n2 % 2 != 0 || shape2(out, "out")? != (rows, inter) {
        return Err(verr("float_silu_mul: shapes"));
    }
    let gr = gu.try_readonly().map_err(|e| verr(format!("gu: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let gs = gr.as_slice()?;
    let os = ow.as_slice_mut()?;
    let _ = rows;
    py.allow_threads(|| silu_rows(gs, os, inter));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_libm_closely() {
        let mut x = -745.0f64;
        while x < 709.0 {
            let (a, b) = (exp_f64(x), x.exp());
            if b > 1e-300 && x < 709.0 {
                assert!(((a - b) / b).abs() < 1e-14, "x={x} {a} {b}");
            }
            x += 0.0137;
        }
        assert_eq!(exp_f64(0.0), 1.0);
        assert_eq!(exp_f64(1000.0), exp_f64(709.0));
        assert!(exp_f64(709.0) > 8e307 && exp_f64(709.0).is_finite());
        assert_eq!(exp_f64(-1000.0), exp_f64(-745.0));
        assert!(exp_f64(-745.0) < 1e-323);
        assert!(exp_f64(f64::NAN).is_nan());
        // float32 results: the rounding of exp_f64 agrees with libm's in all but rare ties
        let mut bad = 0;
        for i in 0..200_000 {
            let x = -20.0 + i as f64 * 2e-4;
            if (exp_f64(x) as f32) != (x.exp() as f32) {
                bad += 1;
            }
        }
        assert!(bad < 5, "{bad}");
    }

    #[test]
    fn norm_and_silu_close_to_exact() {
        let x: Vec<f32> = (0..896).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let w: Vec<f32> = (0..896).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let mut out = vec![0f32; 896];
        rms_norm_row(&x, &w, 1e-6, &mut out);
        let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 896.0;
        let r = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..896 {
            let want = w[i] as f64 * x[i] as f64 * r;
            assert!((out[i] as f64 - want).abs() <= 1e-6 * want.abs().max(1e-3));
        }
        let g = [-100.0f32, -3.0, -0.5, 0.0, 0.5, 3.0, 100.0];
        let u = [1.0f32; 7];
        let mut o = [0f32; 7];
        silu_mul_row(&g, &u, &mut o);
        for i in 0..7 {
            let gv = g[i] as f64;
            assert!((o[i] as f64 - gv / (1.0 + (-gv).exp())).abs() <= 1e-6 * gv.abs().max(1.0));
        }
    }
}
