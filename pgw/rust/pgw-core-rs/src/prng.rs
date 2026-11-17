//! Deterministic CSPRNG seed expansion (SPEC.md 7.3, `pgw.core.prng`). Normative and frozen.
//!
//! ```text
//! info   = "pgw/v1/prng" || for each label: u32_be(len(enc(label))) || enc(label)
//! key    = HKDF-Expand(SHA-256, prk = seed, info, L = 32)
//! stream = ChaCha20(key, counter = 0, nonce = 0^12)       (RFC 8439, 32-bit block counter)
//! enc(str) = "s" || utf8, enc(int) = "i" || i64_be, enc(bytes) = "b" || raw
//! ring element = u64_le(8 stream bytes) mod 2^K           (exactly uniform since 2^K | 2^64)
//! ```
//!
//! The mask for slot `s` of linear `L` with `n` inputs is row `s` of the `(slots, n)` matrix drawn
//! from the stream labelled `("r", L)`; [`SeedExpander::ring_uniform_rows`] seeks directly to it.
//!
//! Seeds are created by the mask factory (`os.urandom(32)`, unique per segment); this crate only
//! consumes them, so it deliberately offers no seed generator.

use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use chacha20::ChaCha20;
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

use crate::error::{PgwError, Result};

/// Seed length in bytes.
pub const SEED_BYTES: usize = 32;
const DOMAIN: &[u8] = b"pgw/v1/prng";
/// ChaCha20 (RFC 8439) has a 32-bit block counter: a stream is at most `2^32` 64-byte blocks.
pub const MAX_STREAM_BYTES: u64 = 64 << 32;

/// A typed stream label. Typed prefixes keep `Str("1")` and `Int(1)` distinct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Label {
    /// UTF-8 string, encoded `"s" || utf8`.
    Str(String),
    /// Signed 64-bit integer, encoded `"i" || i64_be`.
    Int(i64),
    /// Raw bytes, encoded `"b" || bytes`.
    Bytes(Vec<u8>),
}

impl Label {
    fn encode(&self) -> Vec<u8> {
        match self {
            Label::Str(s) => [b"s".as_slice(), s.as_bytes()].concat(),
            Label::Int(i) => [b"i".as_slice(), &i.to_be_bytes()].concat(),
            Label::Bytes(b) => [b"b".as_slice(), b].concat(),
        }
    }
}

impl From<&str> for Label {
    fn from(s: &str) -> Self {
        Label::Str(s.to_owned())
    }
}

impl From<String> for Label {
    fn from(s: String) -> Self {
        Label::Str(s)
    }
}

impl From<i64> for Label {
    fn from(i: i64) -> Self {
        Label::Int(i)
    }
}

impl From<&[u8]> for Label {
    fn from(b: &[u8]) -> Self {
        Label::Bytes(b.to_vec())
    }
}

/// Injective encoding of a label sequence (the HKDF `info`).
pub fn encode_labels(labels: &[Label]) -> Result<Vec<u8>> {
    let mut out = DOMAIN.to_vec();
    for lab in labels {
        let e = lab.encode();
        let len = u32::try_from(e.len()).map_err(|_| PgwError::config("PRNG label longer than 2^32 bytes"))?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&e);
    }
    Ok(out)
}

/// Expands a 32-byte seed into labelled, independent pseudorandom streams.
///
/// The seed is zeroized on drop and never printed by `Debug`.
pub struct SeedExpander {
    seed: Zeroizing<[u8; SEED_BYTES]>,
}

impl std::fmt::Debug for SeedExpander {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SeedExpander(<redacted>)")
    }
}

impl SeedExpander {
    /// Build from exactly [`SEED_BYTES`] bytes.
    pub fn new(seed: &[u8]) -> Result<Self> {
        let arr: [u8; SEED_BYTES] =
            seed.try_into().map_err(|_| PgwError::config(format!("seed must be exactly {SEED_BYTES} bytes")))?;
        Ok(Self { seed: Zeroizing::new(arr) })
    }

    /// Stream key for `labels`: `HKDF-Expand(SHA-256, seed, info(labels), 32)`.
    pub fn key(&self, labels: &[Label]) -> Result<Zeroizing<[u8; 32]>> {
        let info = encode_labels(labels)?;
        let hk =
            Hkdf::<Sha256>::from_prk(self.seed.as_slice()).map_err(|_| PgwError::config("invalid HKDF PRK length"))?;
        let mut okm = Zeroizing::new([0u8; 32]);
        hk.expand(&info, okm.as_mut_slice()).map_err(|_| PgwError::config("HKDF expand failed"))?;
        Ok(okm)
    }

    /// Derive a 32-byte sub-seed: `key(("child", *labels))`.
    pub fn derive_seed(&self, labels: &[Label]) -> Result<Zeroizing<[u8; 32]>> {
        let mut full = Vec::with_capacity(labels.len() + 1);
        full.push(Label::from("child"));
        full.extend_from_slice(labels);
        self.key(&full)
    }

    /// Independent sub-expander seeded with [`SeedExpander::derive_seed`].
    pub fn child(&self, labels: &[Label]) -> Result<SeedExpander> {
        let s = self.derive_seed(labels)?;
        SeedExpander::new(s.as_slice())
    }

    /// `len` keystream bytes starting at byte `offset` of the stream for `labels`.
    pub fn stream_bytes(&self, offset: u64, len: usize, labels: &[Label]) -> Result<Vec<u8>> {
        if offset.checked_add(len as u64).map_or(true, |end| end > MAX_STREAM_BYTES) {
            return Err(PgwError::config("stream offset beyond the 2^32-block ChaCha20 counter range"));
        }
        let key = self.key(labels)?;
        let mut cipher = ChaCha20::new(key.as_ref().into(), &[0u8; 12].into());
        cipher
            .try_seek(offset)
            .map_err(|_| PgwError::config("stream offset beyond the 2^32-block ChaCha20 counter range"))?;
        let mut buf = vec![0u8; len];
        cipher
            .try_apply_keystream(&mut buf)
            .map_err(|_| PgwError::config("stream end beyond the 2^32-block ChaCha20 counter range"))?;
        Ok(buf)
    }

    /// The first `n` bytes of the stream for `labels`.
    pub fn random_bytes(&self, n: usize, labels: &[Label]) -> Result<Vec<u8>> {
        self.stream_bytes(0, n, labels)
    }

    /// `count` uniform ring elements in `[0, 2^k)` (`k` in `1..=64`), i.e. the flattened
    /// `ring_uniform(shape, k, *labels)` of the Python core.
    pub fn ring_uniform(&self, count: usize, k: u32, labels: &[Label]) -> Result<Vec<u64>> {
        self.ring_elements(0, count, k, labels)
    }

    /// Rows `[row_start, row_start + row_count)` of the `(rows, cols)` matrix that
    /// [`SeedExpander::ring_uniform`] would produce, flattened row-major, without generating
    /// earlier rows. This is how the gateway regenerates `r` for a mask slot.
    pub fn ring_uniform_rows(
        &self,
        cols: usize,
        k: u32,
        row_start: u64,
        row_count: usize,
        labels: &[Label],
    ) -> Result<Vec<u64>> {
        if cols == 0 {
            return Err(PgwError::config("cols must be > 0 and row_start/row_count >= 0"));
        }
        let start = (cols as u64).checked_mul(row_start).ok_or_else(|| PgwError::config("row offset overflows"))?;
        let count = cols.checked_mul(row_count).ok_or_else(|| PgwError::config("row count overflows"))?;
        self.ring_elements(start, count, k, labels)
    }

    fn ring_elements(&self, elem_offset: u64, count: usize, k: u32, labels: &[Label]) -> Result<Vec<u64>> {
        if !(1..=64).contains(&k) {
            return Err(PgwError::config(format!("ring width must be in [1, 64], got {k}")));
        }
        let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
        let byte_off = elem_offset.checked_mul(8).ok_or_else(|| PgwError::config("stream offset overflows"))?;
        let nbytes = count.checked_mul(8).ok_or_else(|| PgwError::config("stream length overflows"))?;
        let mut raw = self.stream_bytes(byte_off, nbytes, labels)?;
        let out = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8-byte chunk")) & mask).collect();
        raw.zeroize();
        Ok(out)
    }

    /// Integers in `[low, high)` by modulo reduction of `u64` words (bias < 2^-32 for ranges < 2^32).
    pub fn uniform_int(&self, count: usize, low: i64, high: i64, labels: &[Label]) -> Result<Vec<i64>> {
        let span = (high as i128) - (low as i128);
        if span <= 0 || span > (1i128 << 32) {
            return Err(PgwError::config("need low < high and high - low <= 2^32"));
        }
        let raw = self.ring_uniform(count, 64, labels)?;
        Ok(raw.into_iter().map(|v| (v % span as u64) as i64 + low).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kat_seed() -> Vec<u8> {
        (0u8..32).collect()
    }

    fn l(labels: &[&str]) -> Vec<Label> {
        labels.iter().map(|s| Label::from(*s)).collect()
    }

    /// Frozen known answers shared with `tests/core/test_prng.py::test_known_answers`.
    #[test]
    fn known_answers() {
        let s = SeedExpander::new(&kat_seed()).unwrap();
        let labels = vec![Label::from("mask"), Label::from("0.qkv"), Label::Int(0)];
        let b = s.random_bytes(16, &labels).unwrap();
        assert_eq!(hex(&b), "a2b5b70fe6fc44d9296ef4cb4aa4a77b");
        assert_eq!(
            s.ring_uniform(4, 40, &labels).unwrap(),
            vec![988106175906, 321249373737, 254846079836, 563174071857]
        );
        let d = s.derive_seed(&[Label::from("segment"), Label::Int(1)]).unwrap();
        assert_eq!(hex(d.as_slice()), "00bc5de3b96a347f37eb7113018cc757222c86488560b4a81e1d4df28c199af7");
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn label_encoding_is_injective() {
        let a = encode_labels(&l(&["a", "bc"])).unwrap();
        let b = encode_labels(&l(&["ab", "c"])).unwrap();
        assert_ne!(a, b);
        assert_ne!(encode_labels(&[Label::from("1")]).unwrap(), encode_labels(&[Label::Int(1)]).unwrap());
        assert_ne!(encode_labels(&[Label::from("1")]).unwrap(), encode_labels(&[Label::Bytes(b"1".to_vec())]).unwrap());
        let e = encode_labels(&[Label::Int(-2)]).unwrap();
        assert_eq!(&e[..11], b"pgw/v1/prng");
        assert_eq!(&e[11..], &[0, 0, 0, 9, b'i', 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe]);
    }

    #[test]
    fn determinism_prefix_and_children() {
        let s = SeedExpander::new(&kat_seed()).unwrap();
        let q = l(&["q"]);
        assert_eq!(s.random_bytes(10, &q).unwrap(), s.random_bytes(20, &q).unwrap()[..10]);
        assert_ne!(s.random_bytes(8, &l(&["x"])).unwrap(), s.random_bytes(8, &l(&["y"])).unwrap());
        let c = s.child(&l(&["c"])).unwrap();
        let c2 = SeedExpander::new(s.derive_seed(&l(&["c"])).unwrap().as_slice()).unwrap();
        assert_eq!(c.random_bytes(8, &l(&["z"])).unwrap(), c2.random_bytes(8, &l(&["z"])).unwrap());
        assert!(format!("{s:?}").contains("redacted"));
    }

    #[test]
    fn rows_match_full_matrix() {
        let s = SeedExpander::new(&kat_seed()).unwrap();
        let lab = l(&["r", "0.qkv"]);
        for (cols, start, count) in [(1usize, 0u64, 5usize), (3, 7, 4), (13, 5, 0), (896, 2, 3), (5, 0, 40)] {
            let full = s.ring_uniform(48 * cols, 40, &lab).unwrap();
            let part = s.ring_uniform_rows(cols, 40, start, count, &lab).unwrap();
            let a = start as usize * cols;
            assert_eq!(part, full[a..a + count * cols]);
        }
        assert!(s.ring_uniform_rows(0, 40, 0, 1, &lab).is_err());
    }

    #[test]
    fn ranges_and_errors() {
        assert!(SeedExpander::new(b"short").is_err());
        let s = SeedExpander::new(&kat_seed()).unwrap();
        for k in [32u32, 40, 48] {
            let v = s.ring_uniform(4096, k, &l(&["u"])).unwrap();
            assert!(v.iter().all(|&x| x < 1 << k));
            let ones = v.iter().filter(|&&x| x >> (k - 1) == 1).count();
            assert!((1800..2300).contains(&ones), "top bit balance {ones}");
        }
        let v = s.uniform_int(1000, -127, 128, &l(&["w"])).unwrap();
        assert!(v.iter().all(|&x| (-127..=127).contains(&x)));
        assert!(s.uniform_int(3, 5, 5, &[]).is_err());
        assert!(s.uniform_int(3, 0, (1 << 32) + 1, &[]).is_err());
        assert!(s.ring_uniform(1, 0, &[]).is_err());
        // counter range: the last block is reachable, one byte past it is not
        assert_eq!(s.stream_bytes(MAX_STREAM_BYTES - 8, 8, &[]).unwrap().len(), 8);
        assert!(s.stream_bytes(MAX_STREAM_BYTES - 8, 9, &[]).is_err());
        assert!(s.ring_uniform_rows(1, 40, 1 << 61, 1, &[]).is_err());
    }
}
