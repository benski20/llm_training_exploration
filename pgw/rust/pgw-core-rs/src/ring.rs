//! Ring parameters and arithmetic over `Z_{2^K}` (SPEC.md section 5, `pgw.core.ring`).
//!
//! Canonical ring values are `u64` in `[0, 2^K)`. Centered values are `i64` in
//! `[-2^(K-1), 2^(K-1))`. All arithmetic uses wrapping `u64` operations followed by a mask, which is
//! exact because `2^K` divides `2^64`.

use crate::error::{PgwError, Result};

/// Smallest supported ring width.
pub const MIN_K: u32 = 16;
/// Largest supported ring width (kept below 64 so `a + b` of canonical values never wraps `i64`).
pub const MAX_K: u32 = 56;
/// Largest int8 weight magnitude produced by quantization.
pub const WEIGHT_QMAX: u64 = 127;

/// Ring width `K` and signed activation width `A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RingParams {
    k: u32,
    a: u32,
}

impl RingParams {
    /// `K32A8`: cheapest preset (measured lossy).
    pub const K32A8: RingParams = RingParams { k: 32, a: 8 };
    /// `K40A12`: default preset.
    pub const K40A12: RingParams = RingParams { k: 40, a: 12 };
    /// `K48A20`: maximum-quality preset.
    pub const K48A20: RingParams = RingParams { k: 48, a: 20 };
    /// The default ring (`K40A12`).
    pub const DEFAULT: RingParams = Self::K40A12;

    /// Validate and build. `K` must be a multiple of 8 in `[16, 56]` and `A` in `[2, K-8]`.
    pub fn new(k: u32, a: u32) -> Result<Self> {
        if !(MIN_K..=MAX_K).contains(&k) || k % 8 != 0 {
            return Err(PgwError::config(format!("ring K must be a multiple of 8 in [{MIN_K}, {MAX_K}], got {k}")));
        }
        if a < 2 || a > k - 8 {
            return Err(PgwError::config(format!("activation bits A must be in [2, K-8], got A={a} K={k}")));
        }
        Ok(Self { k, a })
    }

    /// Parse a canonical name such as `"K40A12"` (case-insensitive, surrounding whitespace ignored).
    pub fn from_name(name: &str) -> Result<Self> {
        let s = name.trim().to_ascii_uppercase();
        let bad = || PgwError::config(format!("bad ring name {name:?}; expected e.g. 'K40A12'"));
        let rest = s.strip_prefix('K').ok_or_else(bad)?;
        let (k_str, a_str) = rest.split_once('A').ok_or_else(bad)?;
        let k: u32 = k_str.parse().map_err(|_| bad())?;
        let a: u32 = a_str.parse().map_err(|_| bad())?;
        Self::new(k, a)
    }

    /// Ring bit width `K`.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// Signed activation bit width `A`.
    pub const fn a(&self) -> u32 {
        self.a
    }

    /// Canonical name, e.g. `"K40A12"`.
    pub fn name(&self) -> String {
        format!("K{}A{}", self.k, self.a)
    }

    /// `2^K`.
    pub const fn modulus(&self) -> u64 {
        1u64 << self.k
    }

    /// `2^K - 1`.
    pub const fn mask(&self) -> u64 {
        (1u64 << self.k) - 1
    }

    /// `2^(K-1)`; centered values lie in `[-half, half)`.
    pub const fn half(&self) -> u64 {
        1u64 << (self.k - 1)
    }

    /// Largest quantized activation magnitude `2^(A-1) - 1`.
    pub const fn qmax(&self) -> u64 {
        (1u64 << (self.a - 1)) - 1
    }

    /// Number of int8 limb GEMMs per ring matmul (`K / 8`).
    pub const fn limbs(&self) -> u32 {
        self.k / 8
    }

    /// Bytes per packed ring element (`K / 8`).
    pub const fn bytes_per_element(&self) -> usize {
        (self.k / 8) as usize
    }

    /// Largest possible `|yq|` of a genuine product with inner dimension `n`: `qmax * 127 * n`.
    pub fn output_bound(&self, in_features: u64) -> u128 {
        self.qmax() as u128 * WEIGHT_QMAX as u128 * in_features as u128
    }

    /// Largest inner dimension whose products decode unambiguously (`output_bound < 2^(K-1)`).
    pub fn max_in_features(&self) -> u64 {
        (self.half() - 1) / (self.qmax() * WEIGHT_QMAX)
    }

    /// Error with `ring_overflow` if a linear with this inner dimension could wrap.
    pub fn check_in_features(&self, in_features: u64) -> Result<()> {
        if in_features > self.max_in_features() {
            return Err(PgwError::overflow(format!(
                "{}: inner dimension {in_features} exceeds the decodable maximum {}; use a larger ring",
                self.name(),
                self.max_in_features()
            )));
        }
        Ok(())
    }

    /// Reduce a signed value to canonical form `[0, 2^K)`.
    #[inline]
    pub const fn reduce(&self, v: i64) -> u64 {
        (v as u64) & self.mask()
    }

    /// Map a ring value to its centered representative in `[-2^(K-1), 2^(K-1))`.
    #[inline]
    pub const fn center(&self, v: u64) -> i64 {
        let v = v & self.mask();
        if v >= self.half() {
            v as i64 - self.modulus() as i64
        } else {
            v as i64
        }
    }

    /// `(a + b) mod 2^K`.
    #[inline]
    pub const fn add(&self, a: u64, b: u64) -> u64 {
        a.wrapping_add(b) & self.mask()
    }

    /// `(a - b) mod 2^K`.
    #[inline]
    pub const fn sub(&self, a: u64, b: u64) -> u64 {
        a.wrapping_sub(b) & self.mask()
    }

    /// `(a * b) mod 2^K`.
    #[inline]
    pub const fn mul(&self, a: u64, b: u64) -> u64 {
        a.wrapping_mul(b) & self.mask()
    }

    /// `true` if `v` is canonical (`v < 2^K`).
    #[inline]
    pub const fn is_canonical(&self, v: u64) -> bool {
        v <= self.mask()
    }
}

impl Default for RingParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl std::fmt::Display for RingParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "K{}A{}", self.k, self.a)
    }
}

impl std::str::FromStr for RingParams {
    type Err = PgwError;
    fn from_str(s: &str) -> Result<Self> {
        Self::from_name(s)
    }
}

/// Ceiling of `log2(x)` for `x >= 1`, computed exactly on integers.
pub(crate) fn ceil_log2(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        128 - (x - 1).leading_zeros()
    }
}

/// Balanced base-`2^bits` digits of a canonical ring value (`pgw.core.ring.balanced_digits`).
///
/// Returns `ceil(k / bits)` digits `d_i` with `-2^(bits-1) <= d_i < 2^(bits-1)` and
/// `sum_i d_i * 2^(bits*i) == v (mod 2^k)`. With `bits == 8` every digit fits in an `i8`.
pub fn balanced_digits(v: u64, k: u32, bits: u32) -> Result<Vec<i64>> {
    if !(2..=62).contains(&bits) {
        return Err(PgwError::config(format!("digit width must be in [2, 62], got {bits}")));
    }
    if !(1..=64).contains(&k) {
        return Err(PgwError::config(format!("ring width must be in [1, 64], got {k}")));
    }
    let n = k.div_ceil(bits);
    let base_mask: i64 = (1i64 << bits) - 1;
    let half: i64 = 1i64 << (bits - 1);
    let full: i64 = 1i64 << bits;
    let mut cur = v as i64;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut d = cur & base_mask;
        if d >= half {
            d -= full;
        }
        out.push(d);
        // Arithmetic shift, as in torch int64 `>>`; `cur - d` is a multiple of 2^bits.
        cur = cur.wrapping_sub(d) >> bits;
    }
    Ok(out)
}

/// Balanced base-256 digits as `i8` limbs (the `cuda_int8` layout, `K/8` limbs).
pub fn int8_limbs(v: u64, params: &RingParams) -> Vec<i8> {
    balanced_digits(v & params.mask(), params.k(), 8)
        .expect("bits=8 and K in [16,56] are always valid")
        .into_iter()
        .map(|d| d as i8)
        .collect()
}

/// Recompose `sum_i d_i * 2^(bits*i) mod 2^k` (inverse of [`balanced_digits`]).
pub fn recompose_digits(digits: &[i64], k: u32, bits: u32) -> u64 {
    let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut acc: u64 = 0;
    for (i, &d) in digits.iter().enumerate() {
        let shift = bits as u64 * i as u64;
        if shift >= 64 {
            break; // contributes 0 mod 2^64
        }
        acc = acc.wrapping_add((d as u64).wrapping_shl(shift as u32));
    }
    acc & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_names() {
        assert_eq!(RingParams::DEFAULT, RingParams::K40A12);
        for (name, p) in
            [("K32A8", RingParams::K32A8), ("k40a12", RingParams::K40A12), (" K48A20 ", RingParams::K48A20)]
        {
            assert_eq!(RingParams::from_name(name).unwrap(), p);
            assert_eq!(p.name().parse::<RingParams>().unwrap(), p);
        }
        assert_eq!(RingParams::K40A12.to_string(), "K40A12");
        for bad in ["", "40A12", "K40", "KxA1", "K41A12", "K64A12", "K8A2", "K40A33", "K40A1"] {
            assert!(RingParams::from_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn derived_constants() {
        let p = RingParams::K40A12;
        assert_eq!(p.modulus(), 1 << 40);
        assert_eq!(p.mask(), (1 << 40) - 1);
        assert_eq!(p.half(), 1 << 39);
        assert_eq!(p.qmax(), 2047);
        assert_eq!(p.limbs(), 5);
        assert_eq!(p.bytes_per_element(), 5);
        assert_eq!(p.output_bound(4096), 2047 * 127 * 4096);
        let mx = p.max_in_features();
        assert!(p.output_bound(mx) < p.half() as u128);
        assert!(p.output_bound(mx + 1) >= p.half() as u128);
        assert!(p.check_in_features(mx).is_ok());
        let e = p.check_in_features(mx + 1).unwrap_err();
        assert_eq!(e.code().as_str(), "ring_overflow");
    }

    #[test]
    fn arithmetic() {
        let p = RingParams::K32A8;
        assert_eq!(p.reduce(-1), (1u64 << 32) - 1);
        assert_eq!(p.center((1u64 << 32) - 1), -1);
        assert_eq!(p.center(1u64 << 31), -(1i64 << 31));
        assert_eq!(p.center((1u64 << 31) - 1), (1i64 << 31) - 1);
        assert_eq!(p.add(p.mask(), 2), 1);
        assert_eq!(p.sub(1, 2), p.mask());
        assert_eq!(p.mul(p.mask(), p.mask()), 1);
        for v in [-5i64, 0, 7, -(1 << 31), (1 << 31) - 1] {
            assert_eq!(p.center(p.reduce(v)), v);
        }
    }

    #[test]
    fn digits_roundtrip_and_range() {
        for k in [16u32, 32, 40, 48, 56] {
            let mask = (1u64 << k) - 1;
            let vals = [0, 1, 127, 128, 255, 256, mask, mask - 127, mask / 2, (mask / 2) + 1, 0x1234_5678_9abc & mask];
            for &v in &vals {
                for bits in [8u32, 13, 26] {
                    let d = balanced_digits(v, k, bits).unwrap();
                    assert_eq!(d.len() as u32, k.div_ceil(bits));
                    let h = 1i64 << (bits - 1);
                    assert!(d.iter().all(|&x| -h <= x && x < h));
                    assert_eq!(recompose_digits(&d, k, bits), v);
                }
            }
        }
        assert!(balanced_digits(1, 40, 1).is_err());
        assert!(balanced_digits(1, 40, 63).is_err());
        let limbs = int8_limbs(RingParams::K40A12.reduce(-1), &RingParams::K40A12);
        assert_eq!(limbs, vec![-1, 0, 0, 0, 0]);
    }

    #[test]
    fn ceil_log2_exact() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1 << 40), 40);
        assert_eq!(ceil_log2((1 << 40) + 1), 41);
    }
}
