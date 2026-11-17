//! Compact ring tensor encoding (SPEC.md 6.3, `pgw.core.packing`).
//!
//! Each element is `K/8` little-endian bytes, row-major, no header.

use crate::error::{ErrorCode, PgwError, Result};

fn bytes_per(k: u32) -> Result<usize> {
    if k % 8 != 0 || !(8..=64).contains(&k) {
        return Err(PgwError::shape(format!("bad ring width {k}")));
    }
    Ok((k / 8) as usize)
}

/// Byte length of `count` packed elements.
pub fn packed_size(count: usize, k: u32) -> Result<usize> {
    count.checked_mul(bytes_per(k)?).ok_or_else(|| PgwError::shape("packed size overflows usize"))
}

/// Pack ring values (each reduced mod `2^K` first) into `K/8` bytes each.
pub fn pack_ring(values: &[u64], k: u32) -> Result<Vec<u8>> {
    let nb = bytes_per(k)?;
    let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut out = Vec::with_capacity(values.len() * nb);
    for &v in values {
        out.extend_from_slice(&(v & mask).to_le_bytes()[..nb]);
    }
    Ok(out)
}

/// Unpack exactly `count` elements. A length mismatch is a `protocol` error.
pub fn unpack_ring(data: &[u8], count: usize, k: u32) -> Result<Vec<u64>> {
    let nb = bytes_per(k)?;
    let expected = packed_size(count, k)?;
    if data.len() != expected {
        return Err(PgwError::new(
            ErrorCode::Protocol,
            format!("ring payload has {} bytes, expected {expected} for {count} elements K={k}", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(nb)
        .map(|c| {
            let mut b = [0u8; 8];
            b[..nb].copy_from_slice(c);
            u64::from_le_bytes(b)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for k in [16u32, 32, 40, 48, 56, 64] {
            let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
            let vals: Vec<u64> = vec![0, 1, mask, 0x0102_0304_0506_0708 & mask, mask >> 1];
            let p = pack_ring(&vals, k).unwrap();
            assert_eq!(p.len(), vals.len() * (k as usize / 8));
            assert_eq!(unpack_ring(&p, vals.len(), k).unwrap(), vals);
        }
    }

    #[test]
    fn layout_and_errors() {
        assert_eq!(pack_ring(&[0x01_0203_0405], 40).unwrap(), vec![5, 4, 3, 2, 1]);
        // values are reduced before packing
        assert_eq!(pack_ring(&[u64::MAX], 16).unwrap(), vec![0xff, 0xff]);
        let e = unpack_ring(&[0; 9], 2, 40).unwrap_err();
        assert_eq!(e.code(), ErrorCode::Protocol);
        assert_eq!(pack_ring(&[1], 12).unwrap_err().code(), ErrorCode::BadShape);
        assert!(unpack_ring(&[], 0, 40).unwrap().is_empty());
    }
}
