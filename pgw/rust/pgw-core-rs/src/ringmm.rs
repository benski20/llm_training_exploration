//! Exact ring matrix products `y = x @ W8^T mod 2^K` (SPEC.md 5, `pgw.core.ringmm`).
//!
//! Two independent implementations are provided so each can audit the other:
//!
//! * [`ringmm_reference`]: direct wrapping `u64` multiply-accumulate (exact because `2^K | 2^64`).
//! * [`ringmm_limbs`]: the server algorithm. `x` is split into `K/8` balanced base-256 digits
//!   (`int8`), each digit plane is multiplied with the int8 weight using **int32** accumulators over
//!   inner chunks of at most [`MAX_INT32_CHUNK`] elements (no int32 overflow is possible since
//!   `|d * w| <= 128 * 128`), and the planes are recombined as `sum_l acc_l << 8l mod 2^K`.
//!
//! The compute service only ever runs the second algorithm on uniformly random `x_hat`.

use crate::error::{PgwError, Result};
use crate::ring::{int8_limbs, RingParams};

/// Largest inner-dimension chunk for an int32 accumulator: `131064 * 128 * 128 < 2^31`.
pub const MAX_INT32_CHUNK: usize = 131_064;

fn check_shapes(x_len: usize, rows: usize, n: usize, w_len: usize, m: usize) -> Result<()> {
    if rows.checked_mul(n) != Some(x_len) {
        return Err(PgwError::shape(format!("x has {x_len} elements, expected {rows}x{n}")));
    }
    if m.checked_mul(n) != Some(w_len) {
        return Err(PgwError::shape(format!("weight has {w_len} elements, expected {m}x{n}")));
    }
    if rows.checked_mul(m).is_none() {
        return Err(PgwError::shape(format!("output size {rows}x{m} overflows")));
    }
    Ok(())
}

/// Reference product: `y[t, j] = sum_i x[t, i] * w8[j, i] mod 2^K`, shape `(rows, m)`.
pub fn ringmm_reference(
    x: &[u64],
    rows: usize,
    n: usize,
    w8: &[i8],
    m: usize,
    params: &RingParams,
) -> Result<Vec<u64>> {
    check_shapes(x.len(), rows, n, w8.len(), m)?;
    let mut y = vec![0u64; rows * m];
    for t in 0..rows {
        let xr = &x[t * n..(t + 1) * n];
        for j in 0..m {
            let wr = &w8[j * n..(j + 1) * n];
            let acc = xr.iter().zip(wr).fold(0u64, |a, (&xv, &wv)| a.wrapping_add(xv.wrapping_mul(wv as i64 as u64)));
            y[t * m + j] = acc & params.mask();
        }
    }
    Ok(y)
}

/// Limb product with int8 digits and int32 accumulation (the `cuda_int8` algorithm).
///
/// Weights may take any `i8` value including -128.
pub fn ringmm_limbs(x: &[u64], rows: usize, n: usize, w8: &[i8], m: usize, params: &RingParams) -> Result<Vec<u64>> {
    check_shapes(x.len(), rows, n, w8.len(), m)?;
    let limbs = params.limbs() as usize;
    // digit planes: planes[l][t * n + i]
    let mut planes = vec![vec![0i8; rows * n]; limbs];
    for (idx, &v) in x.iter().enumerate() {
        for (l, d) in int8_limbs(v, params).into_iter().enumerate() {
            planes[l][idx] = d;
        }
    }
    let mut y = vec![0u64; rows * m];
    for (l, plane) in planes.iter().enumerate() {
        let shift = 8 * l as u32;
        for t in 0..rows {
            let xr = &plane[t * n..(t + 1) * n];
            for j in 0..m {
                let wr = &w8[j * n..(j + 1) * n];
                let mut total: i64 = 0;
                for (xc, wc) in xr.chunks(MAX_INT32_CHUNK).zip(wr.chunks(MAX_INT32_CHUNK)) {
                    let mut acc: i32 = 0;
                    for (&d, &w) in xc.iter().zip(wc) {
                        acc += d as i32 * w as i32; // cannot overflow: chunk bound above
                    }
                    total += acc as i64;
                }
                let cell = &mut y[t * m + j];
                *cell = cell.wrapping_add((total as u64).wrapping_shl(shift)) & params.mask();
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state ^ (*state >> 29)
    }

    #[test]
    fn limbs_match_reference() {
        let mut s = 7u64;
        for p in [RingParams::K32A8, RingParams::K40A12, RingParams::K48A20, RingParams::new(56, 30).unwrap()] {
            let (rows, n, m) = (3, 37, 5);
            let x: Vec<u64> = (0..rows * n).map(|_| lcg(&mut s) & p.mask()).collect();
            let mut w: Vec<i8> = (0..m * n).map(|_| lcg(&mut s) as i8).collect();
            w[0] = -128;
            w[1] = 127;
            let a = ringmm_reference(&x, rows, n, &w, m, &p).unwrap();
            let b = ringmm_limbs(&x, rows, n, &w, m, &p).unwrap();
            assert_eq!(a, b, "{p}");
        }
    }

    #[test]
    fn small_known_product() {
        let p = RingParams::K32A8;
        let x = [p.reduce(-1), 2, 3]; // one row
        let w = [1i8, 1, 1, -1, 0, 2]; // 2 x 3
        let y = ringmm_reference(&x, 1, 3, &w, 2, &p).unwrap();
        assert_eq!(y, vec![4, p.reduce(7)]);
        assert_eq!(ringmm_limbs(&x, 1, 3, &w, 2, &p).unwrap(), y);
    }

    #[test]
    fn shape_errors_and_empty() {
        let p = RingParams::K40A12;
        assert!(ringmm_reference(&[1, 2], 1, 3, &[0; 3], 1, &p).is_err());
        assert!(ringmm_limbs(&[1, 2, 3], 1, 3, &[0; 4], 1, &p).is_err());
        assert!(ringmm_limbs(&[], 0, 3, &[0; 6], 2, &p).unwrap().is_empty());
        // n == 0 makes every shape check trivially true; the output size must still be checked
        assert!(ringmm_reference(&[], usize::MAX, 0, &[], usize::MAX, &p).is_err());
        assert!(ringmm_limbs(&[], usize::MAX, 0, &[], 2, &p).is_err());
        assert_eq!(ringmm_limbs(&[], 2, 0, &[], 3, &p).unwrap(), vec![0; 6]);
    }
}
