//! fused2 integrity check over `F_p`, `p = 2^61 - 1` (opt-in fused2 schedule; `pgw.engine.fused2`).
//!
//! `p61_check(y, offset, u, x, uw, bound)`: for every row `t`, the range check `|y[t, j]| <= bound` and, for every
//! check vector `c`, `sum_j u[c, j] (y[t, j] - offset[t, j]) == sum_i uw[c, i] x[t, i] (mod p)`. This is both
//! stages of the fused2 check (stage 1 with the residual as `offset` and `uw = (u * hmul) W1 mod p`; stage 2 with
//! no offset and `uw = u W2' mod p`), the same predicate as the Python `Fused2MaskedBackend._verify` (numpy limb
//! arithmetic), bit for bit: it only answers equal / not equal.
//!
//! Arithmetic: every operand is reduced to `[0, p)` first (Mersenne fold of the magnitude, then negated; exact for any `i64` difference),
//! products of two values `< 2^61` are `< 2^122`, so up to 32 of them are summed in a `u128` before a Mersenne
//! fold (`x = hi * 2^61 + lo == hi + lo`). No floating point; the result does not depend on the order.

use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The Mersenne prime `2^61 - 1`.
pub const P: u64 = (1u64 << 61) - 1;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(p61_check, m)?)?;
    Ok(())
}

/// `x mod p` for any `u128`.
#[inline]
pub fn fold128(x: u128) -> u64 {
    let p = P as u128;
    let mut v = (x & p) + (x >> 61); // < 2^61 + 2^67
    v = (v & p) + (v >> 61); // < 2^61 + 2^6
    let mut r = v as u64;
    if r >= P {
        r -= P;
    }
    r
}

/// `v mod p` in `[0, p)` for any `i128`.
#[inline]
pub fn red(v: i128) -> u64 {
    // |v| folded (no division), then negated in the field: -r == p - r
    let r = fold128(v.unsigned_abs());
    if v < 0 && r != 0 {
        P - r
    } else {
        r
    }
}

/// `sum_j a[j] * b(j) mod p` with `a` in `[0, p)` and `b(j)` already reduced (the tests' second formulation).
#[cfg(test)]
fn dot<F: Fn(usize) -> u64>(a: &[u64], b: F) -> u64 {
    let mut acc: u64 = 0;
    let mut j = 0;
    while j < a.len() {
        let end = (j + 32).min(a.len());
        let mut s: u128 = 0;
        for (k, &av) in a[j..end].iter().enumerate() {
            s += av as u128 * b(j + k) as u128;
        }
        acc = fold128(acc as u128 + fold128(s) as u128);
        j = end;
    }
    acc
}

/// `sum_j a[j] * b(j) mod p` with `a` in `[0, p)` and `|b(j)| < 2^63`: eight signed 64x64 -> 128-bit products
/// (each `< 2^124`) summed in an `i128`, then reduced (no per-element reduction).
#[inline]
fn dot_signed<F: Fn(usize) -> i64>(a: &[u64], b: F) -> u64 {
    let mut acc: u64 = 0;
    let mut j = 0;
    while j < a.len() {
        let end = (j + 8).min(a.len());
        let mut s: i128 = 0;
        for (k, &av) in a[j..end].iter().enumerate() {
            s += (av as i64 as i128) * (b(j + k) as i128); // av < 2^61: the i64 cast is exact
        }
        acc = fold128(acc as u128 + red(s) as u128);
        j = end;
    }
    acc
}

/// The check on slices (row-major `y (t, m)`, `offset (t, m)` or none, `u (c, m)`, `x (t, n)`, `uw (c, n)`).
/// Returns 0 (pass), 1 (Freivalds mismatch) or 2 (range).
#[allow(clippy::too_many_arguments)]
pub fn check(y: &[i64], offset: Option<&[i64]>, u: &[u64], x: &[i64], uw: &[u64], m: usize, n: usize, bound: i64) -> u8 {
    let rows = if m == 0 { 0 } else { y.len() / m };
    let c = if m == 0 { 0 } else { u.len() / m };
    if y.iter().any(|&v| v.unsigned_abs() > bound as u64) {
        return 2;
    }
    // the caller's own operands (residual, inputs) are bounded by the same cap (the Python binding refuses
    // anything wider), so the differences and products below cannot overflow
    debug_assert!(bound < 1 << 61);
    for t in 0..rows {
        let yr = &y[t * m..(t + 1) * m];
        let orow = offset.map(|o| &o[t * m..(t + 1) * m]);
        let xr = &x[t * n..(t + 1) * n];
        for ci in 0..c {
            let lhs = match orow {
                Some(o) => dot_signed(&u[ci * m..(ci + 1) * m], |j| yr[j] - o[j]), // both <= 2^61: no overflow
                None => dot_signed(&u[ci * m..(ci + 1) * m], |j| yr[j]),
            };
            let rhs = dot_signed(&uw[ci * n..(ci + 1) * n], |i| xr[i]);
            if lhs != rhs {
                return 1;
            }
        }
    }
    0
}

/// Python: `p61_check(y, offset, u, x, uw, bound) -> int` (0 pass, 1 Freivalds mismatch, 2 range). int64 C-contiguous
/// arrays; `u` / `uw` entries in `[0, p)` (refused otherwise). Releases the GIL.
#[pyfunction]
#[pyo3(signature = (y, offset, u, x, uw, bound))]
fn p61_check<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyArray2<i64>>,
    offset: Option<&Bound<'py, PyArray2<i64>>>,
    u: &Bound<'py, PyArray2<i64>>,
    x: &Bound<'py, PyArray2<i64>>,
    uw: &Bound<'py, PyArray2<i64>>,
    bound: i64,
) -> PyResult<u8> {
    let verr = |s: &str| PyValueError::new_err(s.to_string());
    let ya = y.try_readonly().map_err(|e| verr(&format!("y: {e}")))?;
    let ua = u.try_readonly().map_err(|e| verr(&format!("u: {e}")))?;
    let xa = x.try_readonly().map_err(|e| verr(&format!("x: {e}")))?;
    let wa = uw.try_readonly().map_err(|e| verr(&format!("uw: {e}")))?;
    let oa = match offset {
        Some(o) => Some(o.try_readonly().map_err(|e| verr(&format!("offset: {e}")))?),
        None => None,
    };
    let (t, m) = (ya.shape()[0], ya.shape()[1]);
    let n = xa.shape()[1];
    if xa.shape()[0] != t || ua.shape()[1] != m || wa.shape()[1] != n || wa.shape()[0] != ua.shape()[0] {
        return Err(verr("p61_check: shapes must be y (t, m), u (c, m), x (t, n), uw (c, n)"));
    }
    if let Some(o) = &oa {
        if o.shape() != ya.shape() {
            return Err(verr("p61_check: offset must have y's shape"));
        }
    }
    if !(0..1i64 << 61).contains(&bound) {
        return Err(verr("p61_check: bound must be in [0, 2^61)"));
    }
    let ys = ya.as_slice().map_err(|_| verr("y must be C-contiguous"))?;
    let xs = xa.as_slice().map_err(|_| verr("x must be C-contiguous"))?;
    let us = ua.as_slice().map_err(|_| verr("u must be C-contiguous"))?;
    let ws = wa.as_slice().map_err(|_| verr("uw must be C-contiguous"))?;
    let os = match &oa {
        Some(o) => Some(o.as_slice().map_err(|_| verr("offset must be C-contiguous"))?),
        None => None,
    };
    if us.iter().chain(ws.iter()).any(|&v| v < 0 || v as u64 >= P) {
        return Err(verr("p61_check: check vectors must be in [0, p)"));
    }
    let wide = |v: &i64| v.unsigned_abs() > 1 << 61;
    if os.is_some_and(|o| o.iter().any(wide)) || xs.iter().any(wide) {
        return Err(verr("p61_check: offset and x must be within 2^61"));
    }
    // SAFETY: i64 and u64 have the same size and alignment and every bit pattern is valid; the values were checked
    // to be in [0, p), so the reinterpretation is the identity on them. The borrows stay alive for the call.
    let uu: &[u64] = unsafe { std::slice::from_raw_parts(us.as_ptr().cast::<u64>(), us.len()) };
    let wu: &[u64] = unsafe { std::slice::from_raw_parts(ws.as_ptr().cast::<u64>(), ws.len()) };
    Ok(py.allow_threads(|| check(ys, os, uu, xs, wu, m, n, bound)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    fn ref_dot(a: &[u64], b: &[i128]) -> u64 {
        let mut acc: i128 = 0;
        for (x, y) in a.iter().zip(b) {
            acc = (acc + (*x as i128) * (y.rem_euclid(P as i128))).rem_euclid(P as i128);
        }
        acc as u64
    }

    #[test]
    fn fold_and_dot_exact_at_extremes() {
        assert_eq!(fold128(u128::MAX), (u128::MAX % P as u128) as u64);
        assert_eq!(fold128(P as u128), 0);
        let mut s = 7u64;
        for len in [1usize, 31, 32, 33, 1000, 4097] {
            let a: Vec<u64> = (0..len).map(|i| if i % 3 == 0 { P - 1 } else { lcg(&mut s) % P }).collect();
            let b: Vec<i128> = (0..len)
                .map(|i| match i % 4 {
                    0 => (1i128 << 62) - 1,
                    1 => -(1i128 << 62) + 1,
                    _ => (lcg(&mut s) as i64) as i128 >> 1,
                })
                .collect();
            assert_eq!(dot(&a, |j| red(b[j])), ref_dot(&a, &b));
            let b62: Vec<i64> = b.iter().map(|&v| (v >> 1) as i64).collect();
            let b62w: Vec<i128> = b62.iter().map(|&v| v as i128).collect();
            assert_eq!(dot_signed(&a, |j| b62[j]), ref_dot(&a, &b62w));
        }
    }

    #[test]
    fn check_catches_any_single_error() {
        let (m, n) = (37usize, 53usize);
        let mut s = 11u64;
        let w: Vec<i64> = (0..m * n).map(|_| (lcg(&mut s) % 255) as i64 - 127).collect();
        let x: Vec<i64> = (0..n).map(|_| (lcg(&mut s) % 4001) as i64 - 2000).collect();
        let off: Vec<i64> = (0..m).map(|_| (lcg(&mut s) >> 20) as i64 - (1 << 43)).collect();
        let mut y: Vec<i64> = (0..m).map(|j| off[j] + (0..n).map(|i| w[j * n + i] * x[i]).sum::<i64>()).collect();
        let u: Vec<u64> = (0..m).map(|_| lcg(&mut s) % P).collect();
        let uw: Vec<u64> = (0..n)
            .map(|i| {
                let col: Vec<i128> = (0..m).map(|j| w[j * n + i] as i128).collect();
                ref_dot(&u, &col)
            })
            .collect();
        assert_eq!(check(&y, Some(&off), &u, &x, &uw, m, n, 1 << 53), 0);
        y[5] += 1;
        assert_eq!(check(&y, Some(&off), &u, &x, &uw, m, n, 1 << 53), 1);
        y[5] -= 1;
        y[0] = (1 << 53) + 1;
        assert_eq!(check(&y, Some(&off), &u, &x, &uw, m, n, 1 << 53), 2);
    }
}
