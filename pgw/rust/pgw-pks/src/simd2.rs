//! Hand-written AVX2 copies (x86-64, used when the ISA is `avx2`): an 8-way ChaCha20 keystream (same bytes
//! as the `chacha20` crate, tested). The NTTs shared with AVX-512 are in [`crate::vntt`].
#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use std::arch::x86_64::*;

#[inline(always)]
unsafe fn rotl<const N: i32, const M: i32>(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_slli_epi32::<N>(x), _mm256_srli_epi32::<M>(x))
}

/// In-register transpose of an 8 x 8 u32 block.
#[inline(always)]
unsafe fn transpose8x8(r: &mut [__m256i; 8]) {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256::<0x20>(u0, u4);
    r[1] = _mm256_permute2x128_si256::<0x20>(u1, u5);
    r[2] = _mm256_permute2x128_si256::<0x20>(u2, u6);
    r[3] = _mm256_permute2x128_si256::<0x20>(u3, u7);
    r[4] = _mm256_permute2x128_si256::<0x31>(u0, u4);
    r[5] = _mm256_permute2x128_si256::<0x31>(u1, u5);
    r[6] = _mm256_permute2x128_si256::<0x31>(u2, u6);
    r[7] = _mm256_permute2x128_si256::<0x31>(u3, u7);
}

/// ChaCha20 keystream (RFC 8439, zero nonce, 32-bit block counter from `first`) into `out` (a multiple of
/// 64 bytes), 8 blocks per iteration (one per lane).
#[target_feature(enable = "avx2")]
pub unsafe fn chacha20_blocks(key: &[u8; 32], first: u32, out: &mut [u8]) {
    assert!(out.len() % 64 == 0);
    let nblk = out.len() / 64;
    assert!(first as u64 + nblk as u64 <= 1u64 << 32, "chacha20: block counter overflow");
    let kw: [u32; 8] = std::array::from_fn(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()));
    let cst = [0x6170_7865u32, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
    let init: [__m256i; 16] = std::array::from_fn(|i| match i {
        0..=3 => _mm256_set1_epi32(cst[i] as i32),
        4..=11 => _mm256_set1_epi32(kw[i - 4] as i32),
        _ => _mm256_setzero_si256(),
    });
    let rot16 = _mm256_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5,
                                4, 7, 6, 1, 0, 3, 2);
    let rot8 = _mm256_set_epi8(14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5,
                               4, 7, 2, 1, 0, 3);
    let lane = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let mut tmp = [0u8; 512];
    let mut b = 0usize;
    while b < nblk {
        let mut x = init;
        x[12] = _mm256_add_epi32(_mm256_set1_epi32(first.wrapping_add(b as u32) as i32), lane);
        let ctr = x[12];
        macro_rules! qr {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                x[$a] = _mm256_add_epi32(x[$a], x[$b]);
                x[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(x[$d], x[$a]), rot16);
                x[$c] = _mm256_add_epi32(x[$c], x[$d]);
                x[$b] = rotl::<12, 20>(_mm256_xor_si256(x[$b], x[$c]));
                x[$a] = _mm256_add_epi32(x[$a], x[$b]);
                x[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(x[$d], x[$a]), rot8);
                x[$c] = _mm256_add_epi32(x[$c], x[$d]);
                x[$b] = rotl::<7, 25>(_mm256_xor_si256(x[$b], x[$c]));
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
            x[i] = _mm256_add_epi32(x[i], if i == 12 { ctr } else { init[i] });
        }
        let mut lo: [__m256i; 8] = std::array::from_fn(|i| x[i]);
        let mut hi: [__m256i; 8] = std::array::from_fn(|i| x[8 + i]);
        transpose8x8(&mut lo); // lo[l] = words 0..7 of block l
        transpose8x8(&mut hi);
        let take = (nblk - b).min(8);
        let dst = if take == 8 { out.as_mut_ptr().add(64 * b) } else { tmp.as_mut_ptr() };
        for l in 0..8 {
            _mm256_storeu_si256(dst.add(64 * l) as *mut _, lo[l]);
            _mm256_storeu_si256(dst.add(64 * l + 32) as *mut _, hi[l]);
        }
        if take < 8 {
            out[64 * b..64 * (b + take)].copy_from_slice(&tmp[..64 * take]);
        }
        b += take;
    }
    tmp.fill(0);
    std::hint::black_box(&tmp);
}
