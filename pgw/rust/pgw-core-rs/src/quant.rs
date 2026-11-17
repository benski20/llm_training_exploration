//! Symmetric quantization (SPEC.md section 5; `pgw.core.ring.quantize_*`).
//!
//! Matches the Python core bit for bit: float64 arithmetic, round-half-to-even (as `torch.round`),
//! then clamp. Matrices are row-major slices with explicit dimensions.

use crate::error::{PgwError, Result};
use crate::ring::{RingParams, WEIGHT_QMAX};

fn check_len(len: usize, rows: usize, cols: usize, what: &str) -> Result<()> {
    match rows.checked_mul(cols) {
        Some(n) if n == len => Ok(()),
        _ => Err(PgwError::shape(format!("{what} has {len} elements, expected {rows}x{cols}"))),
    }
}

/// Quantize activations `(rows, n)` per row to signed `A`-bit integers.
///
/// Returns `(xq, sx)` with `|xq| <= qmax` and `x ~= xq * sx[row]`. Non-finite input is rejected
/// with `ring_overflow`, as in Python.
pub fn quantize_activations(x: &[f64], rows: usize, n: usize, params: &RingParams) -> Result<(Vec<i64>, Vec<f64>)> {
    check_len(x.len(), rows, n, "activations")?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PgwError::overflow("non-finite activation values cannot be quantized"));
    }
    let qmax = params.qmax() as f64;
    let mut xq = Vec::with_capacity(x.len());
    let mut sx = Vec::with_capacity(rows);
    for row in x.chunks_exact(n.max(1)).take(rows) {
        let amax = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let s = if amax > 0.0 { amax / qmax } else { 1.0 };
        sx.push(s);
        xq.extend(row.iter().map(|v| (v / s).round_ties_even().clamp(-qmax, qmax) as i64));
    }
    if n == 0 {
        sx.resize(rows, 1.0);
    }
    Ok((xq, sx))
}

/// Quantize a float weight `(m, n)` to int8 per output row: `sw[j] = max|W[j,:]| / 127`.
///
/// Returns `(w8, sw)` with `w8` in `[-127, 127]` and `sw` as `f32` (the stored pack format).
/// Non-finite weights are rejected with `ring_overflow` (stricter than the Python core, which
/// would silently produce a meaningless row).
pub fn quantize_weight_rows(w: &[f64], m: usize, n: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    check_len(w.len(), m, n, "weight")?;
    if w.iter().any(|v| !v.is_finite()) {
        return Err(PgwError::overflow("non-finite weight values cannot be quantized"));
    }
    let qmax = WEIGHT_QMAX as f64;
    let mut w8 = Vec::with_capacity(w.len());
    let mut sw = Vec::with_capacity(m);
    for j in 0..m {
        let row = &w[j * n..(j + 1) * n];
        let amax = row.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        let s = if amax > 0.0 { amax / qmax } else { 1.0 };
        sw.push(s as f32);
        w8.extend(row.iter().map(|v| (v / s).round_ties_even().clamp(-qmax, qmax) as i8));
    }
    Ok((w8, sw))
}

/// Rescale exact integer outputs `(rows, m)`: `y = yq * sx[row] * sw[col] (+ bias[col])` in float64.
pub fn dequantize_outputs(
    yq: &[i64],
    rows: usize,
    m: usize,
    sx: &[f64],
    sw: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f64>> {
    check_len(yq.len(), rows, m, "outputs")?;
    if sx.len() != rows || sw.len() != m || bias.is_some_and(|b| b.len() != m) {
        return Err(PgwError::shape("scale or bias length does not match the output shape"));
    }
    let mut out = Vec::with_capacity(yq.len());
    for t in 0..rows {
        for j in 0..m {
            let mut y = yq[t * m + j] as f64 * sx[t] * sw[j] as f64;
            if let Some(b) = bias {
                y += b[j] as f64;
            }
            out.push(y);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_basic() {
        let p = RingParams::K32A8; // qmax 127
        let x = [1.0, -0.5, 0.25, 0.0, 0.0, 0.0];
        let (xq, sx) = quantize_activations(&x, 2, 3, &p).unwrap();
        assert_eq!(xq, vec![127, -64, 32, 0, 0, 0]); // -63.5 -> -64 (ties to even), 31.75 -> 32
        assert_eq!(sx, vec![1.0 / 127.0, 1.0]);
        assert!(quantize_activations(&[f64::NAN], 1, 1, &p).is_err());
        assert!(quantize_activations(&[1.0, 2.0], 1, 3, &p).is_err());
        let (xq, sx) = quantize_activations(&[], 2, 0, &p).unwrap();
        assert!(xq.is_empty());
        assert_eq!(sx, vec![1.0, 1.0]);
    }

    #[test]
    fn ties_to_even() {
        // 0.5 * qmax / amax scaling: choose values that land exactly on .5
        let p = RingParams::new(16, 3).unwrap(); // qmax 3
        let x = [3.0, 1.5, 0.5, -2.5];
        let (xq, _) = quantize_activations(&x, 1, 4, &p).unwrap();
        assert_eq!(xq, vec![3, 2, 0, -2]);
    }

    #[test]
    fn weights_and_dequant() {
        let w = [0.5, -1.0, 0.0, 0.0, 2.0, 1.0];
        let (w8, sw) = quantize_weight_rows(&w, 3, 2).unwrap();
        assert_eq!(w8, vec![64, -127, 0, 0, 127, 64]);
        assert_eq!(sw, vec![(1.0f64 / 127.0) as f32, 1.0, (2.0f64 / 127.0) as f32]);
        let y = dequantize_outputs(&[2, -3], 1, 2, &[0.5], &[1.0, 2.0], Some(&[0.25, 0.0])).unwrap();
        assert_eq!(y, vec![1.25, -3.0]);
        assert!(dequantize_outputs(&[2, -3], 1, 2, &[0.5], &[1.0], None).is_err());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(quantize_weight_rows(&[1.0, bad], 1, 2).unwrap_err().code().as_str(), "ring_overflow");
        }
        assert!(quantize_weight_rows(&[1.0], 1, 2).is_err());
    }
}
