//! One-time-mask math for one masked linear call (`pgw.core.masking`).
//!
//! ```text
//! x_hat = (xq + r) mod 2^K                  mask_activations  -> sent to the compute service
//! y_hat = W8 x_hat mod 2^K                  computed by the service
//! yq    = centered((y_hat - W8 r) mod 2^K)  unmask_outputs    -> exact xq @ W8^T
//! ```
//!
//! `r` is uniform in `Z_{2^K}` and used exactly once, so `x_hat` is uniform and independent of `xq`.

use crate::error::{ErrorCode, PgwError, Result};
use crate::ring::RingParams;

fn check_canonical(v: &[u64], params: &RingParams, what: &str) -> Result<()> {
    if let Some(pos) = v.iter().position(|&e| !params.is_canonical(e)) {
        return Err(PgwError::overflow(format!("{what}[{pos}] is not a canonical {} ring element", params.name())));
    }
    Ok(())
}

/// `x_hat = (xq + r) mod 2^K`, elementwise.
///
/// `r` must be canonical (a non-canonical mask indicates a corrupt mask source and is rejected).
/// `xq` must satisfy `|xq| <= qmax`: a larger value would void the headroom guarantee
/// (`qmax * 127 * n < 2^(K-1)`), so the decoded output could wrap silently or be flagged as
/// tampered. Both checks are stricter than the Python core and raise `ring_overflow`.
pub fn mask_activations(xq: &[i64], r: &[u64], params: &RingParams) -> Result<Vec<u64>> {
    if xq.len() != r.len() {
        return Err(PgwError::shape(format!("activation length {} != mask length {}", xq.len(), r.len())));
    }
    check_canonical(r, params, "mask")?;
    let qmax = params.qmax();
    if let Some(pos) = xq.iter().position(|&v| v.unsigned_abs() > qmax) {
        return Err(PgwError::overflow(format!(
            "activation[{pos}] is outside the signed {}-bit range [-{qmax}, {qmax}]",
            params.a()
        )));
    }
    Ok(xq.iter().zip(r).map(|(&x, &m)| params.add(x as u64, m)).collect())
}

/// Error with `integrity_check_failed` if any `|yq| > qmax * 127 * n`.
pub fn check_output_range(yq: &[i64], params: &RingParams, in_features: u64) -> Result<()> {
    let bound = params.output_bound(in_features);
    if yq.iter().any(|&v| v.unsigned_abs() as u128 > bound) {
        return Err(PgwError::new(
            ErrorCode::Integrity,
            format!(
                "decoded linear output outside the representable range (corrupt or tampered result); bound={bound} ring={}",
                params.name()
            ),
        ));
    }
    Ok(())
}

/// `yq = centered((y_hat - wr) mod 2^K)`; with `in_features` also runs [`check_output_range`].
pub fn unmask_outputs(y_hat: &[u64], wr: &[u64], params: &RingParams, in_features: Option<u64>) -> Result<Vec<i64>> {
    if y_hat.len() != wr.len() {
        return Err(PgwError::shape(format!("result length {} != W r length {}", y_hat.len(), wr.len())));
    }
    let yq: Vec<i64> = y_hat.iter().zip(wr).map(|(&y, &w)| params.center(params.sub(y, w))).collect();
    if let Some(n) = in_features {
        check_output_range(&yq, params, n)?;
    }
    Ok(yq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ringmm::ringmm_reference;

    #[test]
    fn mask_unmask_exact() {
        let p = RingParams::K40A12;
        let (rows, n, m) = (2usize, 4usize, 3usize);
        let xq: Vec<i64> = vec![2047, -2047, 0, 5, -1, 1, 100, -300];
        let w8: Vec<i8> = vec![127, -127, 3, 0, 1, 1, 1, 1, -5, 7, 0, 2];
        let r: Vec<u64> =
            (0..rows * n).map(|i| (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1)) & p.mask()).collect();
        let x_hat = mask_activations(&xq, &r, &p).unwrap();
        let y_hat = ringmm_reference(&x_hat, rows, n, &w8, m, &p).unwrap();
        let wr = ringmm_reference(&r, rows, n, &w8, m, &p).unwrap();
        let yq = unmask_outputs(&y_hat, &wr, &p, Some(n as u64)).unwrap();
        let xc: Vec<u64> = xq.iter().map(|&v| p.reduce(v)).collect();
        let plain: Vec<i64> =
            ringmm_reference(&xc, rows, n, &w8, m, &p).unwrap().into_iter().map(|v| p.center(v)).collect();
        assert_eq!(yq, plain);
        assert_eq!(yq[0], 2047 * 127 + 2047 * 127);
    }

    #[test]
    fn errors() {
        let p = RingParams::K32A8;
        assert_eq!(mask_activations(&[1], &[1, 2], &p).unwrap_err().code(), ErrorCode::BadShape);
        assert_eq!(mask_activations(&[1], &[1 << 32], &p).unwrap_err().code(), ErrorCode::RingOverflow);
        // |xq| <= qmax (127 for A=8) is enforced
        assert!(mask_activations(&[127, -127], &[0, 0], &p).is_ok());
        assert_eq!(mask_activations(&[128], &[0], &p).unwrap_err().code(), ErrorCode::RingOverflow);
        assert_eq!(mask_activations(&[i64::MIN], &[0], &p).unwrap_err().code(), ErrorCode::RingOverflow);
        assert_eq!(unmask_outputs(&[1], &[], &p, None).unwrap_err().code(), ErrorCode::BadShape);
        // tampered: decodes far outside the bound
        let e = unmask_outputs(&[1 << 30], &[0], &p, Some(16)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::Integrity);
        let bound = p.output_bound(16) as i64;
        assert!(check_output_range(&[bound, -bound], &p, 16).is_ok());
        assert!(check_output_range(&[bound + 1], &p, 16).is_err());
        assert!(check_output_range(&[i64::MIN], &p, 16).is_err());
    }
}
