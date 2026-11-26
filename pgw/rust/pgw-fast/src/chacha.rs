//! AVX-512 ChaCha20 for the mask ring stream ([`crate::store::fill_ring`]): 16 blocks per iteration, one
//! block per 32-bit lane, the keystream words masked to the ring width and stored straight into the output
//! (no XOR pass, no second masking pass). The block function is RFC 8439 with the all-zero 96-bit nonce and
//! the 32-bit block counter, i.e. exactly the bytes of the `chacha20` crate's `ChaCha20` seeked to
//! `64 first` (tests below and `store::tests`). The kernel is the one of `pgw-pks` (`simd::chacha20_blocks`,
//! the pks-v3 noise and upload streams) with the ring mask fused into the store.
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::needless_range_loop)]

use std::arch::x86_64::*;

/// Index vectors of one 16 x 16 u32 transpose round (swap the elements whose row and column differ in bit `s`).
const fn tr_idx(s: usize, hi: bool) -> [i32; 16] {
    let mut a = [0i32; 16];
    let mut c = 0;
    while c < 16 {
        a[c] = match (c & s == 0, hi) {
            (true, false) => c as i32,
            (true, true) => (c + s) as i32,
            (false, false) => (16 + c - s) as i32,
            (false, true) => (16 + c) as i32,
        };
        c += 1;
    }
    a
}

static TR16: [[[i32; 16]; 2]; 4] = [
    [tr_idx(8, false), tr_idx(8, true)],
    [tr_idx(4, false), tr_idx(4, true)],
    [tr_idx(2, false), tr_idx(2, true)],
    [tr_idx(1, false), tr_idx(1, true)],
];

#[inline(always)]
unsafe fn round16<const S: usize>(r: &mut [__m512i; 16]) {
    let k = S.trailing_zeros() as usize;
    let t = &TR16[3 - k];
    let (il, ih) = (_mm512_loadu_si512(t[0].as_ptr() as *const _), _mm512_loadu_si512(t[1].as_ptr() as *const _));
    for i in 0..16 {
        if i & S == 0 {
            let (a, b) = (r[i], r[i + S]);
            r[i] = _mm512_permutex2var_epi32(a, il, b);
            r[i + S] = _mm512_permutex2var_epi32(a, ih, b);
        }
    }
}

/// `r[i]` lane `c` -> `r[c]` lane `i`.
#[inline(always)]
unsafe fn transpose16(r: &mut [__m512i; 16]) {
    round16::<8>(r);
    round16::<4>(r);
    round16::<2>(r);
    round16::<1>(r);
}

/// True when the AVX-512 kernel can run on this CPU (`PGW_FAST_CHACHA512=0` turns it off: the `chacha20` crate).
pub fn available() -> bool {
    use std::sync::OnceLock;
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| {
        let off = std::env::var("PGW_FAST_CHACHA512")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
            .unwrap_or(false);
        !off && std::is_x86_feature_detected!("avx512f")
    })
}

/// Keystream blocks `first .. first + out.len() / 8` of the stream `key` (zero nonce) as little-endian u64
/// words, each ANDed with `mask`, written over `out` (`out.len()` a multiple of 8: 8 words per block).
/// The caller checks `first + blocks <= 2^32` (no counter wrap).
#[target_feature(enable = "avx512f")]
pub unsafe fn ring_blocks(key: &[u8; 32], first: u32, out: &mut [u64], mask: u64) {
    assert!(out.len().is_multiple_of(8));
    let nblk = out.len() / 8;
    assert!(first as u64 + nblk as u64 <= 1u64 << 32, "chacha20: block counter overflow");
    let kw: [u32; 8] = std::array::from_fn(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()));
    let init: [__m512i; 16] = [
        _mm512_set1_epi32(0x6170_7865),
        _mm512_set1_epi32(0x3320_646e),
        _mm512_set1_epi32(0x7962_2d32),
        _mm512_set1_epi32(0x6b20_6574),
        _mm512_set1_epi32(kw[0] as i32),
        _mm512_set1_epi32(kw[1] as i32),
        _mm512_set1_epi32(kw[2] as i32),
        _mm512_set1_epi32(kw[3] as i32),
        _mm512_set1_epi32(kw[4] as i32),
        _mm512_set1_epi32(kw[5] as i32),
        _mm512_set1_epi32(kw[6] as i32),
        _mm512_set1_epi32(kw[7] as i32),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
        _mm512_setzero_si512(),
    ];
    let lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let mv = _mm512_set1_epi64(mask as i64);
    let op = out.as_mut_ptr();
    let mut b = 0usize;
    while b < nblk {
        let mut x = init;
        x[12] = _mm512_add_epi32(_mm512_set1_epi32(first.wrapping_add(b as u32) as i32), lane);
        let ctr = x[12];
        macro_rules! qr {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                x[$a] = _mm512_add_epi32(x[$a], x[$b]);
                x[$d] = _mm512_rol_epi32(_mm512_xor_si512(x[$d], x[$a]), 16);
                x[$c] = _mm512_add_epi32(x[$c], x[$d]);
                x[$b] = _mm512_rol_epi32(_mm512_xor_si512(x[$b], x[$c]), 12);
                x[$a] = _mm512_add_epi32(x[$a], x[$b]);
                x[$d] = _mm512_rol_epi32(_mm512_xor_si512(x[$d], x[$a]), 8);
                x[$c] = _mm512_add_epi32(x[$c], x[$d]);
                x[$b] = _mm512_rol_epi32(_mm512_xor_si512(x[$b], x[$c]), 7);
            };
        }
        for _ in 0..10 {
            qr!(0, 4, 8, 12);
            qr!(1, 5, 9, 13);
            qr!(2, 6, 10, 14);
            qr!(3, 7, 11, 15);
            qr!(0, 5, 10, 15);
            qr!(1, 6, 11, 12);
            qr!(2, 7, 8, 13);
            qr!(3, 4, 9, 14);
        }
        for i in 0..16 {
            x[i] = _mm512_add_epi32(x[i], if i == 12 { ctr } else { init[i] });
        }
        transpose16(&mut x); // x[l] = the 64 bytes of block b + l
        let take = (nblk - b).min(16);
        for l in 0..take {
            _mm512_storeu_si512(op.add(8 * (b + l)) as *mut _, _mm512_and_si512(x[l], mv));
        }
        b += take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
    use chacha20::ChaCha20;

    #[test]
    fn ring_blocks_match_the_crate() {
        if !available() {
            return;
        }
        let key: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(5));
        for (first, nblk, mask) in [(0u32, 1usize, u64::MAX), (3, 15, 0xFFFF_FFFF), (17, 16, (1 << 40) - 1),
                                    (1000, 33, 0xFF), (u32::MAX - 20, 21, u64::MAX)] {
            let mut want = vec![0u8; 64 * nblk];
            let mut c = ChaCha20::new((&key).into(), &[0u8; 12].into());
            c.seek(64 * first as u64);
            if first as u64 + nblk as u64 <= (1u64 << 32) - 1 {
                c.apply_keystream(&mut want);
            } else {
                // the crate refuses the last block: compare all blocks but the last
                let n = 64 * (nblk - 1);
                c.apply_keystream(&mut want[..n]);
            }
            let mut got = vec![0u64; 8 * nblk];
            unsafe { ring_blocks(&key, first, &mut got, mask) };
            let cmp = if first as u64 + nblk as u64 <= (1u64 << 32) - 1 { nblk } else { nblk - 1 };
            for i in 0..8 * cmp {
                let w = u64::from_le_bytes(want[8 * i..8 * i + 8].try_into().unwrap()) & mask;
                assert_eq!(got[i], w, "first={first} i={i}");
            }
        }
    }
}
