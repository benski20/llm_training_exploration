//! Freivalds integrity checks over `Z_{2^K}` (`pgw.core.freivalds`).
//!
//! The factory draws `c` secret check vectors `u_j in Z_{2^K}^m` per linear and stores
//! `u_j^T W8 mod 2^K`. The gateway accepts a service result row `y_hat[t]` for input `x_hat[t]`
//! only if `u_j . y_hat[t] == (u_j^T W8) . x_hat[t] (mod 2^K)` for every `j`.
//!
//! Soundness: an error `e != 0` with 2-adic valuation `v` escapes one check with probability
//! `2^-(K-v)`; errors with large `v` are caught deterministically by the range check, which gives
//! the per-call bound `2^(-c*h)` with `h = K - ceil(log2(4 * bound))` ([`soundness_bits`]).

use crate::error::{ErrorCode, PgwError, Result};
use crate::ring::{ceil_log2, RingParams};

/// `uw = u @ W8 mod 2^K`, shape `(c, n)`, for `u` of shape `(c, m)` and `w8` of shape `(m, n)`.
pub fn freivalds_precompute(
    u: &[u64],
    c: usize,
    w8: &[i8],
    m: usize,
    n: usize,
    params: &RingParams,
) -> Result<Vec<u64>> {
    if c.checked_mul(m) != Some(u.len()) {
        return Err(PgwError::shape(format!("check vectors have {} elements, expected {c}x{m}", u.len())));
    }
    if m.checked_mul(n) != Some(w8.len()) {
        return Err(PgwError::shape(format!("weight has {} elements, expected {m}x{n}", w8.len())));
    }
    let size = c.checked_mul(n).ok_or_else(|| PgwError::shape("check matrix size overflows"))?;
    let mut uw = vec![0u64; size];
    for j in 0..c {
        let out = &mut uw[j * n..(j + 1) * n];
        for (row, &uv) in u[j * m..(j + 1) * m].iter().enumerate() {
            let wr = &w8[row * n..(row + 1) * n];
            for (o, &w) in out.iter_mut().zip(wr) {
                *o = o.wrapping_add(uv.wrapping_mul(w as i64 as u64));
            }
        }
        out.iter_mut().for_each(|o| *o &= params.mask());
    }
    Ok(uw)
}

fn dot(a: &[u64], b: &[u64], mask: u64) -> u64 {
    a.iter().zip(b).fold(0u64, |acc, (&x, &y)| acc.wrapping_add((x & mask).wrapping_mul(y & mask))) & mask
}

/// Per-row verdicts: `true` where every check vector agrees.
///
/// Shapes: `x_hat (rows, n)`, `y_hat (rows, m)`, `u (c, m)`, `uw (c, n)`. At least one check vector
/// is required (`c >= 1`): with `c == 0` every row would vacuously "pass", so it is a `config`
/// error (stricter than the Python core).
#[allow(clippy::too_many_arguments)]
pub fn freivalds_check(
    x_hat: &[u64],
    y_hat: &[u64],
    rows: usize,
    n: usize,
    m: usize,
    u: &[u64],
    uw: &[u64],
    c: usize,
    params: &RingParams,
) -> Result<Vec<bool>> {
    if rows.checked_mul(n) != Some(x_hat.len()) || rows.checked_mul(m) != Some(y_hat.len()) {
        return Err(PgwError::shape(format!(
            "bad shapes: x_hat {} elements, y_hat {} elements for rows={rows} n={n} m={m}",
            x_hat.len(),
            y_hat.len()
        )));
    }
    if c.checked_mul(m) != Some(u.len()) || c.checked_mul(n) != Some(uw.len()) {
        return Err(PgwError::shape("check vector shapes do not match the linear"));
    }
    if c == 0 {
        return Err(PgwError::config("Freivalds check needs at least one check vector (c >= 1)"));
    }
    let mask = params.mask();
    Ok((0..rows)
        .map(|t| {
            let xr = &x_hat[t * n..(t + 1) * n];
            let yr = &y_hat[t * m..(t + 1) * m];
            (0..c).all(|j| dot(yr, &u[j * m..(j + 1) * m], mask) == dot(xr, &uw[j * n..(j + 1) * n], mask))
        })
        .collect())
}

/// Error with `integrity_check_failed` unless every row passes [`freivalds_check`].
#[allow(clippy::too_many_arguments)]
pub fn freivalds_verify(
    x_hat: &[u64],
    y_hat: &[u64],
    rows: usize,
    n: usize,
    m: usize,
    u: &[u64],
    uw: &[u64],
    c: usize,
    params: &RingParams,
    linear_id: &str,
) -> Result<()> {
    let ok = freivalds_check(x_hat, y_hat, rows, n, m, u, uw, c, params)?;
    let bad = ok.iter().filter(|&&v| !v).count();
    if bad > 0 {
        return Err(PgwError::new(
            ErrorCode::Integrity,
            format!("compute service result failed the Freivalds check; linear={linear_id} bad_rows={bad} rows={rows}"),
        ));
    }
    Ok(())
}

/// Lower bound on `-log2(false accept probability)` per call with range check plus `checks` vectors.
pub fn soundness_bits(params: &RingParams, in_features: u64, checks: u32) -> u32 {
    let bound = params.output_bound(in_features).max(1);
    let h = params.k() as i64 - ceil_log2(4 * bound) as i64;
    (h.max(0) as u32).saturating_mul(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ringmm::ringmm_reference;

    #[test]
    fn accepts_genuine_rejects_tampered() {
        let p = RingParams::K40A12;
        let (rows, n, m, c) = (3usize, 6usize, 4usize, 2usize);
        let w8: Vec<i8> = (0..m * n).map(|i| ((i * 37) % 255) as i8).collect();
        let x_hat: Vec<u64> =
            (0..rows * n).map(|i| (i as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D) & p.mask()).collect();
        let u: Vec<u64> = (0..c * m).map(|i| (i as u64 + 3).wrapping_mul(0x9E37_79B9_7F4A_7C15) & p.mask()).collect();
        let uw = freivalds_precompute(&u, c, &w8, m, n, &p).unwrap();
        let mut y_hat = ringmm_reference(&x_hat, rows, n, &w8, m, &p).unwrap();
        assert_eq!(freivalds_check(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &p).unwrap(), vec![true; 3]);
        assert!(freivalds_verify(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &p, "0.qkv").is_ok());
        y_hat[m + 1] = p.add(y_hat[m + 1], 1); // odd error: always caught when some u is odd
        let ok = freivalds_check(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &p).unwrap();
        assert_eq!(ok, vec![true, false, true]);
        let e = freivalds_verify(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &p, "0.qkv").unwrap_err();
        assert_eq!(e.code(), ErrorCode::Integrity);
        assert!(freivalds_check(&x_hat, &y_hat, rows, n, m, &u, &uw[1..], c, &p).is_err());
        assert!(freivalds_precompute(&u, c, &w8, m, n + 1, &p).is_err());
    }

    #[test]
    fn soundness_matches_spec_example() {
        // SPEC 3.2: K40A12, n = 4096 -> h = 8, 16 bits with c = 2.
        assert_eq!(soundness_bits(&RingParams::K40A12, 4096, 2), 16);
        assert_eq!(soundness_bits(&RingParams::K40A12, 4096, 1), 8);
        assert_eq!(soundness_bits(&RingParams::K32A8, 1 << 20, 2), 0);
        assert_eq!(soundness_bits(&RingParams::K40A12, 4096, u32::MAX), u32::MAX);
        assert_eq!(soundness_bits(&RingParams::K40A12, 4096, 0), 0);
    }

    #[test]
    fn zero_check_vectors_never_accept() {
        let p = RingParams::K32A8;
        let (x_hat, y_hat) = (vec![1u64, 2], vec![12345u64]); // rows=1, n=2, m=1; nonsense result
        assert_eq!(freivalds_check(&x_hat, &y_hat, 1, 2, 1, &[], &[], 0, &p).unwrap_err().code(), ErrorCode::Config);
        assert!(freivalds_verify(&x_hat, &y_hat, 1, 2, 1, &[], &[], 0, &p, "0.o").is_err());
    }
}
