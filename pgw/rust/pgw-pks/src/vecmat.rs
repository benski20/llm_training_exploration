//! Check-vector products of the mask factory: `x @ W8` for a few row vectors `x` against an int8 weight
//! (`m x n`, row-major), exactly.
//!
//! * [`vecmat_wrap32`]: `sum_i x[c][i] * W8[i][j] mod 2^32` (wrapping 32-bit arithmetic). The ring check
//!   vectors `u W8 mod 2^32` of K32 segments and the factory's private self-check sets `v W8 mod 2^32`
//!   (`pgw.factory.kernels.ring_vecmat`).
//! * [`vecmat_p31`]: `sum_i u[c][i] * W8[i][j] mod (2^31 - 1)` for `u` in `[0, p)`: the prime-field check
//!   vectors (`pgw.core.freivalds.prime_precompute`). `u` is split into two 16-bit limbs; each limb's
//!   products (`|x w| <= 65535 * 128 < 2^23`) are summed in 32-bit lanes for at most [`FLUSH_ROWS`] rows
//!   (`< 2^31`), then in 64-bit integers (`|sum| < m 2^23 < 2^63`), and reduced once at the end.
//!
//! Both are plain integer sums, so the result does not depend on the order of the additions: the tiles,
//! the threads and the ISA (AVX-512 / AVX2 / portable copies of one generic body) give the same bits as
//! the numpy / torch reference (`tests/factory/test_vecmat_native.py`). The weight is read once per tile of
//! `T` columns with the `C` vectors' accumulators held side by side; up to 16 vectors per pass.

/// Rows summed in 32-bit lanes before the exact path adds them to its 64-bit sums.
pub const FLUSH_ROWS: usize = 255;
/// The Mersenne prime of the prime-field check.
pub const P31: u64 = (1 << 31) - 1;

#[inline(always)]
fn load_row<const T: usize>(w: &[i8], off: usize, width: usize) -> [i32; T] {
    let mut wv = [0i32; T];
    if width == T {
        let row = &w[off..off + T];
        for j in 0..T {
            wv[j] = i32::from(row[j]);
        }
    } else {
        for j in 0..width {
            wv[j] = i32::from(w[off + j]);
        }
    }
    wv
}

/// Rows per block: every tile of a thread's columns is advanced by one block of rows before the next block,
/// so `W8` is read row-major (a few rows at a time, each row's column range contiguous) and the tiles'
/// accumulators are loaded and stored once per block.
const RB: usize = 16;
/// Blocks summed in 32-bit lanes by the exact path before its 64-bit sums (`15 * 16 = 240 <= 255` rows).
const FLUSH_BLOCKS: usize = FLUSH_ROWS / RB;

/// Columns `[j0, j1)`: `out[c][j] = sum_i x[i*C + c] * w[i][j] mod 2^32` (wrapping).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_wrap<const C: usize, const T: usize>(x: &[u32], m: usize, w: &[i8], n: usize, j0: usize, j1: usize,
                                              out: &mut [u32], out_stride: usize) {
    let tiles = (j1 - j0).div_ceil(T);
    let mut acc = vec![[[0u32; T]; C]; tiles];
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + RB).min(m);
        for (t, at) in acc.iter_mut().enumerate() {
            let jt = j0 + t * T;
            let width = T.min(j1 - jt);
            let mut a = *at;
            for i in i0..i1 {
                let wv = load_row::<T>(w, i * n + jt, width);
                let xi = &x[i * C..i * C + C];
                for c in 0..C {
                    let xv = xi[c];
                    for j in 0..T {
                        a[c][j] = a[c][j].wrapping_add(xv.wrapping_mul(wv[j] as u32));
                    }
                }
            }
            *at = a;
        }
        i0 = i1;
    }
    for (t, at) in acc.iter().enumerate() {
        let jt = j0 + t * T;
        let width = T.min(j1 - jt);
        for c in 0..C {
            out[c * out_stride + jt..c * out_stride + jt + width].copy_from_slice(&at[c][..width]);
        }
    }
}

/// Columns `[j0, j1)`: exact sums of `x` (`|x| < 2^16`) against `w`, 32-bit partial sums over at most
/// [`FLUSH_ROWS`] rows (`|x w| <= 65535 * 128 < 2^23`: no overflow), 64-bit totals (`|sum| < m 2^23`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cols_exact<const C: usize, const T: usize>(x: &[i32], m: usize, w: &[i8], n: usize, j0: usize, j1: usize,
                                               out: &mut [i64], out_stride: usize) {
    let tiles = (j1 - j0).div_ceil(T);
    let mut acc = vec![[[0i32; T]; C]; tiles];
    let mut tot = vec![[[0i64; T]; C]; tiles];
    let mut i0 = 0;
    let mut blocks = 0;
    while i0 < m {
        let i1 = (i0 + RB).min(m);
        for (t, at) in acc.iter_mut().enumerate() {
            let jt = j0 + t * T;
            let width = T.min(j1 - jt);
            let mut a = *at;
            for i in i0..i1 {
                let wv = load_row::<T>(w, i * n + jt, width);
                let xi = &x[i * C..i * C + C];
                for c in 0..C {
                    let xv = xi[c];
                    for j in 0..T {
                        a[c][j] = a[c][j].wrapping_add(xv.wrapping_mul(wv[j]));
                    }
                }
            }
            *at = a;
        }
        i0 = i1;
        blocks += 1;
        if blocks == FLUSH_BLOCKS || i0 >= m {
            for (at, tt) in acc.iter_mut().zip(tot.iter_mut()) {
                for c in 0..C {
                    for j in 0..T {
                        tt[c][j] = tt[c][j].wrapping_add(i64::from(at[c][j]));
                        at[c][j] = 0;
                    }
                }
            }
            blocks = 0;
        }
    }
    for (t, tt) in tot.iter().enumerate() {
        let jt = j0 + t * T;
        let width = T.min(j1 - jt);
        for c in 0..C {
            out[c * out_stride + jt..c * out_stride + jt + width].copy_from_slice(&tt[c][..width]);
        }
    }
}

/// AVX-512 copies of [`cols_wrap`] / [`cols_exact`] for whole 16-column tiles (the same sums: integer
/// additions in another order). Columns `[j0, j0 + 16 * tiles)`; the caller runs the rest portably.
#[cfg(target_arch = "x86_64")]
mod x512 {
    use super::{FLUSH_BLOCKS, RB};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn wrap<const C: usize>(x: &[u32], m: usize, w: &[i8], n: usize, j0: usize, tiles: usize,
                                       out: &mut [u32], out_stride: usize) {
        assert!(x.len() >= m * C && w.len() >= m * n && j0 + 16 * tiles <= n);
        assert!(C == 0 || out.len() >= (C - 1) * out_stride + j0 + 16 * tiles);
        let mut acc: Vec<[__m512i; C]> = vec![[_mm512_setzero_si512(); C]; tiles];
        let (xp, wp) = (x.as_ptr(), w.as_ptr());
        let mut i0 = 0;
        while i0 < m {
            let i1 = (i0 + RB).min(m);
            for (t, at) in acc.iter_mut().enumerate() {
                let jt = j0 + 16 * t;
                let mut a = *at;
                for i in i0..i1 {
                    // SAFETY: i < m, jt + 16 <= n (asserted): inside `w`; i * C + c < m * C: inside `x`.
                    let wv = _mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(i * n + jt) as *const __m128i));
                    for (c, ac) in a.iter_mut().enumerate() {
                        let xv = _mm512_set1_epi32(*xp.add(i * C + c) as i32);
                        *ac = _mm512_add_epi32(*ac, _mm512_mullo_epi32(xv, wv));
                    }
                }
                *at = a;
            }
            i0 = i1;
        }
        let op = out.as_mut_ptr();
        for (t, at) in acc.iter().enumerate() {
            for (c, ac) in at.iter().enumerate() {
                // SAFETY: c * out_stride + j0 + 16 t + 16 <= out.len() (asserted).
                _mm512_storeu_si512(op.add(c * out_stride + j0 + 16 * t) as *mut __m512i, *ac);
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn exact<const C: usize>(x: &[i32], m: usize, w: &[i8], n: usize, j0: usize, tiles: usize,
                                        out: &mut [i64], out_stride: usize) {
        assert!(x.len() >= m * C && w.len() >= m * n && j0 + 16 * tiles <= n);
        assert!(C == 0 || out.len() >= (C - 1) * out_stride + j0 + 16 * tiles);
        let z = _mm512_setzero_si512();
        let mut acc: Vec<[__m512i; C]> = vec![[z; C]; tiles];
        let mut lo: Vec<[__m512i; C]> = vec![[z; C]; tiles];
        let mut hi: Vec<[__m512i; C]> = vec![[z; C]; tiles];
        let (xp, wp) = (x.as_ptr(), w.as_ptr());
        let mut i0 = 0;
        let mut blocks = 0;
        while i0 < m {
            let i1 = (i0 + RB).min(m);
            for (t, at) in acc.iter_mut().enumerate() {
                let jt = j0 + 16 * t;
                let mut a = *at;
                for i in i0..i1 {
                    // SAFETY: as in `wrap`. No lane overflows: |x w| < 2^23 over at most 240 rows.
                    let wv = _mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(i * n + jt) as *const __m128i));
                    for (c, ac) in a.iter_mut().enumerate() {
                        let xv = _mm512_set1_epi32(*xp.add(i * C + c));
                        *ac = _mm512_add_epi32(*ac, _mm512_mullo_epi32(xv, wv));
                    }
                }
                *at = a;
            }
            i0 = i1;
            blocks += 1;
            if blocks == FLUSH_BLOCKS || i0 >= m {
                for t in 0..tiles {
                    for c in 0..C {
                        let a = acc[t][c];
                        lo[t][c] = _mm512_add_epi64(lo[t][c], _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a)));
                        hi[t][c] = _mm512_add_epi64(hi[t][c], _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(a)));
                        acc[t][c] = z;
                    }
                }
                blocks = 0;
            }
        }
        let op = out.as_mut_ptr();
        for t in 0..tiles {
            for c in 0..C {
                let p = op.add(c * out_stride + j0 + 16 * t);
                // SAFETY: as in `wrap` (16 int64 per tile and vector).
                _mm512_storeu_si512(p as *mut __m512i, lo[t][c]);
                _mm512_storeu_si512(p.add(8) as *mut __m512i, hi[t][c]);
            }
        }
    }
}

macro_rules! isa_copies {
    ($name:ident, $generic:ident, $simd:ident, $xt:ty, $ot:ty) => {
        /// Columns `[j0, j1)` (output already offset so that column `j0` is at `out[j0]`), dispatched on the ISA.
        #[allow(clippy::too_many_arguments)]
        fn $name<const C: usize, const T: usize>(x: &[$xt], m: usize, w: &[i8], n: usize, j0: usize, j1: usize,
                                                  out: &mut [$ot], out_stride: usize) {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    let tiles = (j1 - j0) / 16;
                    // SAFETY: the CPU supports AVX-512F (checked above); bounds are asserted inside.
                    unsafe { x512::$simd::<C>(x, m, w, n, j0, tiles, out, out_stride) };
                    if j0 + 16 * tiles < j1 {
                        $generic::<C, T>(x, m, w, n, j0 + 16 * tiles, j1, out, out_stride);
                    }
                    return;
                }
                #[target_feature(enable = "avx2")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn v256<const C: usize, const T: usize>(x: &[$xt], m: usize, w: &[i8], n: usize, j0: usize,
                                                                j1: usize, out: &mut [$ot], out_stride: usize) {
                    $generic::<C, T>(x, m, w, n, j0, j1, out, out_stride)
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: as above.
                    unsafe { v256::<C, T>(x, m, w, n, j0, j1, out, out_stride) };
                    return;
                }
            }
            $generic::<C, T>(x, m, w, n, j0, j1, out, out_stride)
        }
    };
}

isa_copies!(isa_wrap, cols_wrap, wrap, u32, u32);
isa_copies!(isa_exact, cols_exact, exact, i32, i64);

/// Run `f(j0, j1, out_cols)` over column ranges (multiples of `align`) in `threads` scoped threads. Each
/// call gets the whole `c x n` output (disjoint column ranges are written).
fn par_cols<O: Send>(n: usize, align: usize, threads: usize, out: &mut [O],
                     f: impl Fn(usize, usize, &mut [O]) + Sync) {
    let tiles = n.div_ceil(align);
    let threads = threads.max(1).min(tiles.max(1));
    if threads <= 1 {
        f(0, n, out);
        return;
    }
    let per = tiles.div_ceil(threads) * align;
    struct Ptr<O>(*mut O, usize);
    // SAFETY: the threads write disjoint column ranges of `out` (each `[j0, j1)` of every row).
    unsafe impl<O> Send for Ptr<O> {}
    unsafe impl<O> Sync for Ptr<O> {}
    let p = Ptr(out.as_mut_ptr(), out.len());
    std::thread::scope(|s| {
        let (f, p) = (&f, &p);
        for j0 in (0..n).step_by(per) {
            let j1 = (j0 + per).min(n);
            s.spawn(move || {
                // SAFETY: see `Ptr`; the slice covers the whole output, each thread touches its columns only.
                let o = unsafe { std::slice::from_raw_parts_mut(p.0, p.1) };
                f(j0, j1, o)
            });
        }
    });
}

macro_rules! dispatch {
    ($c:expr, $isa:ident, $($args:expr),*) => {
        match $c {
            1 => $isa::<1, 256>($($args),*),
            2 => $isa::<2, 128>($($args),*),
            3 => $isa::<3, 128>($($args),*),
            4 => $isa::<4, 64>($($args),*),
            5 => $isa::<5, 64>($($args),*),
            6 => $isa::<6, 64>($($args),*),
            7 => $isa::<7, 32>($($args),*),
            8 => $isa::<8, 32>($($args),*),
            9 => $isa::<9, 32>($($args),*),
            10 => $isa::<10, 32>($($args),*),
            11 => $isa::<11, 32>($($args),*),
            12 => $isa::<12, 32>($($args),*),
            13 => $isa::<13, 32>($($args),*),
            14 => $isa::<14, 32>($($args),*),
            15 => $isa::<15, 32>($($args),*),
            16 => $isa::<16, 32>($($args),*),
            _ => unreachable!(),
        }
    };
}

/// Interleave rows `c0..c0+cc` of `x` (`c x m`) as `(m, cc)`.
fn interleave<X: Copy + Default>(x: &[X], m: usize, c0: usize, cc: usize) -> Vec<X> {
    let mut t = vec![X::default(); m * cc];
    for k in 0..cc {
        let row = &x[(c0 + k) * m..(c0 + k + 1) * m];
        for i in 0..m {
            t[i * cc + k] = row[i];
        }
    }
    t
}

/// Errors of the vecmat kernels (shape mismatches).
#[derive(Debug)]
pub struct ShapeError(pub String);

fn check(xlen: usize, c: usize, m: usize, wlen: usize, n: usize, olen: usize) -> Result<(), ShapeError> {
    if xlen != c * m || wlen != m * n || olen != c * n {
        return Err(ShapeError(format!("vecmat shapes: x {xlen} != {c} x {m}, w {wlen} != {m} x {n} or out {olen} != {c} x {n}")));
    }
    Ok(())
}

/// `out[c][j] = sum_i x[c][i] * W8[i][j] mod 2^32` for `x` `(c, m)` uint32 and `w` `(m, n)` int8 (row-major).
pub fn vecmat_wrap32(x: &[u32], c: usize, w: &[i8], m: usize, n: usize, out: &mut [u32], threads: usize)
                     -> Result<(), ShapeError> {
    check(x.len(), c, m, w.len(), n, out.len())?;
    let mut c0 = 0;
    while c0 < c {
        let cc = (c - c0).min(16);
        let xt = interleave(x, m, c0, cc);
        let sub = &mut out[c0 * n..(c0 + cc) * n];
        par_cols(n, 256, threads, sub, |j0, j1, o| dispatch!(cc, isa_wrap, &xt, m, w, n, j0, j1, o, n));
        c0 += cc;
    }
    Ok(())
}

/// `out[c][j] = sum_i u[c][i] * W8[i][j] mod (2^31 - 1)` for `u` `(c, m)` with entries in `[0, 2^31 - 1)`.
pub fn vecmat_p31(u: &[u64], c: usize, w: &[i8], m: usize, n: usize, out: &mut [u64], threads: usize)
                  -> Result<(), ShapeError> {
    check(u.len(), c, m, w.len(), n, out.len())?;
    if u.iter().any(|&v| v >= P31) {
        return Err(ShapeError("p31 check vectors must be in [0, 2^31 - 1)".into()));
    }
    if m >= 1 << 29 {
        return Err(ShapeError("out_features too large for the exact limb product".into()));
    }
    // limbs: row 2k = low 16 bits of u[k], row 2k + 1 = the high 15 bits
    let mut limbs = vec![0i32; 2 * c * m];
    for k in 0..c {
        for i in 0..m {
            let v = u[k * m + i];
            limbs[(2 * k) * m + i] = (v & 0xFFFF) as i32;
            limbs[(2 * k + 1) * m + i] = (v >> 16) as i32;
        }
    }
    let cl = 2 * c;
    let mut sums = vec![0i64; cl * n];
    let mut c0 = 0;
    while c0 < cl {
        let cc = (cl - c0).min(16);
        let xt = interleave(&limbs, m, c0, cc);
        let sub = &mut sums[c0 * n..(c0 + cc) * n];
        par_cols(n, 256, threads, sub, |j0, j1, o| dispatch!(cc, isa_exact, &xt, m, w, n, j0, j1, o, n));
        c0 += cc;
    }
    let p = P31 as i64;
    for k in 0..c {
        for j in 0..n {
            let lo = sums[(2 * k) * n + j].rem_euclid(p);
            let hi = sums[(2 * k + 1) * n + j].rem_euclid(p);
            out[k * n + j] = ((lo + (hi << 16)) % p) as u64;
        }
    }
    Ok(())
}

/// Rows per block of [`check_rows32`]: the vectors' row of `cu` / `cw` (one 16-lane register) is loaded once per
/// block and multiplied into every row's accumulator.
const CHECK_RB: usize = 16;

/// `W r` elements of the self-check: int64 ring elements, or their low 32 bits (`K <= 32`, the factory's compact
/// form); only the value mod 2^32 enters the check.
pub trait Lo32: Copy + Sync {
    /// The value mod 2^32.
    fn lo32(self) -> u32;
}

impl Lo32 for i64 {
    #[inline(always)]
    fn lo32(self) -> u32 {
        self as u32
    }
}

impl Lo32 for u32 {
    #[inline(always)]
    fn lo32(self) -> u32 {
        self
    }
}

#[inline(always)]
fn check_block_generic<W: Lo32, const RB: usize>(r: &[i64], n: usize, wr: &[W], m: usize, cu: &[u32], cw: &[u32],
                                         mask: u32, t0: usize, bad: &mut [u8]) {
    let mut l = [[0u32; 16]; RB];
    let mut rr = [[0u32; 16]; RB];
    for i in 0..m {
        let cv = &cu[i * 16..i * 16 + 16];
        for t in 0..RB {
            let x = wr[(t0 + t) * m + i].lo32();
            for k in 0..16 {
                l[t][k] = l[t][k].wrapping_add(x.wrapping_mul(cv[k]));
            }
        }
    }
    for i in 0..n {
        let cv = &cw[i * 16..i * 16 + 16];
        for t in 0..RB {
            let x = r[(t0 + t) * n + i] as u32;
            for k in 0..16 {
                rr[t][k] = rr[t][k].wrapping_add(x.wrapping_mul(cv[k]));
            }
        }
    }
    for t in 0..RB {
        bad[t] = u8::from((0..16).any(|k| (l[t][k] & mask) != (rr[t][k] & mask)));
    }
    l.iter_mut().for_each(|a| a.fill(0));
    rr.iter_mut().for_each(|a| a.fill(0));
}

#[cfg(target_arch = "x86_64")]
mod c512 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block<W: super::Lo32, const RB: usize>(r: &[i64], n: usize, wr: &[W], m: usize, cu: &[u32], cw: &[u32],
                                         mask: u32, t0: usize, bad: &mut [u8]) {
        assert!(wr.len() >= (t0 + RB) * m && r.len() >= (t0 + RB) * n && cu.len() >= 16 * m && cw.len() >= 16 * n);
        assert!(bad.len() >= RB);
        let z = _mm512_setzero_si512();
        let mut l = [z; RB];
        let mut rr = [z; RB];
        let (wp, rp, up, vp) = (wr.as_ptr(), r.as_ptr(), cu.as_ptr(), cw.as_ptr());
        for i in 0..m {
            // SAFETY: i < m: 16 lanes of cu's row i; (t0 + t) * m + i < wr.len() (asserted).
            let cv = _mm512_loadu_si512(up.add(16 * i) as *const __m512i);
            for (t, a) in l.iter_mut().enumerate() {
                let x = _mm512_set1_epi32((*wp.add((t0 + t) * m + i)).lo32() as i32);
                *a = _mm512_add_epi32(*a, _mm512_mullo_epi32(x, cv));
            }
        }
        for i in 0..n {
            // SAFETY: as above for cw and r.
            let cv = _mm512_loadu_si512(vp.add(16 * i) as *const __m512i);
            for (t, a) in rr.iter_mut().enumerate() {
                let x = _mm512_set1_epi32(*rp.add((t0 + t) * n + i) as i32);
                *a = _mm512_add_epi32(*a, _mm512_mullo_epi32(x, cv));
            }
        }
        let mv = _mm512_set1_epi32(mask as i32);
        for t in 0..RB {
            let ne = _mm512_cmpneq_epi32_mask(_mm512_and_si512(l[t], mv), _mm512_and_si512(rr[t], mv));
            bad[t] = u8::from(ne != 0);
            l[t] = z;
            rr[t] = z;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_block<W: Lo32, const RB: usize>(r: &[i64], n: usize, wr: &[W], m: usize, cu: &[u32], cw: &[u32], mask: u32,
                                t0: usize, bad: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU supports AVX-512F (checked above); bounds are asserted inside.
            unsafe { c512::block::<W, RB>(r, n, wr, m, cu, cw, mask, t0, bad) };
            return;
        }
        #[target_feature(enable = "avx2")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn v256<W: Lo32, const RB: usize>(r: &[i64], n: usize, wr: &[W], m: usize, cu: &[u32], cw: &[u32],
                                        mask: u32, t0: usize, bad: &mut [u8]) {
            check_block_generic::<W, RB>(r, n, wr, m, cu, cw, mask, t0, bad)
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: as above.
            unsafe { v256::<W, RB>(r, n, wr, m, cu, cw, mask, t0, bad) };
            return;
        }
    }
    check_block_generic::<W, RB>(r, n, wr, m, cu, cw, mask, t0, bad)
}

/// Pad `(rows, c)` vectors to 16 lanes (zeros: they add nothing to either side of the comparison).
fn pad16(v: &[u32], rows: usize, c: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * 16];
    for i in 0..rows {
        out[i * 16..i * 16 + c].copy_from_slice(&v[i * c..i * c + c]);
    }
    out
}

/// The factory's every-row Freivalds self-check for `K <= 32` (`mask < 2^32`): `bad[t] = 1` where
/// `(wr[t] @ cu) & mask != (r[t] @ cw) & mask` for some of the `c <= 16` vectors. Every product and sum is taken
/// mod 2^32, which `2^K` divides: the same flags as the uint64 reference (`pgw.hemask.backend.verify_rows`).
/// `r` `(rows, n)` and `wr` `(rows, m)` int64 ring elements, `cu` `(m, c)` and `cw` `(n, c)` uint32.
#[allow(clippy::too_many_arguments)]
pub fn check_rows32<W: Lo32>(r: &[i64], n: usize, wr: &[W], m: usize, cu: &[u32], cw: &[u32], c: usize, mask: u32,
                    bad: &mut [u8], threads: usize) -> Result<(), ShapeError> {
    let rows = bad.len();
    if !(1..=16).contains(&c) || r.len() != rows * n || wr.len() != rows * m || cu.len() != m * c || cw.len() != n * c {
        return Err(ShapeError(format!("check_rows32: r {} / {rows} x {n}, wr {} / {rows} x {m}, cu {} / {m} x {c}, cw {} / {n} x {c}",
                                      r.len(), wr.len(), cu.len(), cw.len())));
    }
    let mut cu16 = pad16(cu, m, c);
    let mut cw16 = pad16(cw, n, c);
    {
        let (cu16, cw16) = (&cu16[..], &cw16[..]);
        let one = |t0: usize, part: &mut [u8]| {
            let full = part.len() / CHECK_RB * CHECK_RB;
            for b in (0..full).step_by(CHECK_RB) {
                check_block::<W, CHECK_RB>(r, n, wr, m, cu16, cw16, mask, t0 + b, &mut part[b..b + CHECK_RB]);
            }
            for t in full..part.len() {
                check_block::<W, 1>(r, n, wr, m, cu16, cw16, mask, t0 + t, &mut part[t..=t]);
            }
        };
        let blocks = rows.div_ceil(CHECK_RB).max(1);
        let threads = threads.max(1).min(blocks);
        if threads <= 1 {
            one(0, bad);
        } else {
            let per = blocks.div_ceil(threads) * CHECK_RB;
            std::thread::scope(|s| {
                let one = &one;
                for (k, part) in bad.chunks_mut(per).enumerate() {
                    s.spawn(move || one(k * per, part));
                }
            });
        }
    }
    use zeroize::Zeroize;
    cu16.zeroize();
    cw16.zeroize();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn check_rows32_matches_u64_reference() {
        let mut s = 11u64;
        for &(rows, m, n, c, k) in &[(1usize, 5usize, 3usize, 1usize, 32u32), (37, 40, 17, 14, 32), (16, 9, 8, 16, 27),
                                    (70, 33, 65, 2, 32)] {
            let mask = if k == 32 { u32::MAX } else { (1u32 << k) - 1 };
            let r: Vec<i64> = (0..rows * n).map(|_| (lcg(&mut s) as u32 & mask) as i64).collect();
            let cu: Vec<u32> = (0..m * c).map(|_| lcg(&mut s) as u32 & mask).collect();
            let cw: Vec<u32> = (0..n * c).map(|_| lcg(&mut s) as u32).collect();
            let mut wr: Vec<i64> = (0..rows * m).map(|_| (lcg(&mut s) as u32 & mask) as i64).collect();
            // make some rows consistent: wr such that row sums match is hard; compare against the reference flags
            for t in (0..rows).step_by(3) {
                wr[t * m] = (wr[t * m] + (1 << 31)) & i64::from(mask); // a high-valuation error
            }
            let mut bad = vec![0u8; rows];
            check_rows32(&r, n, &wr, m, &cu, &cw, c, mask, &mut bad, 3).unwrap();
            for t in 0..rows {
                let mut want = false;
                for j in 0..c {
                    let mut l = 0u64;
                    let mut rr = 0u64;
                    for i in 0..m {
                        l = l.wrapping_add((wr[t * m + i] as u64).wrapping_mul(u64::from(cu[i * c + j])));
                    }
                    for i in 0..n {
                        rr = rr.wrapping_add((r[t * n + i] as u64).wrapping_mul(u64::from(cw[i * c + j])));
                    }
                    want |= (l & u64::from(mask)) != (rr & u64::from(mask));
                }
                assert_eq!(bad[t] != 0, want, "row {t}");
            }
        }
        // consistent rows pass: wr = W r for a random W, cu = v, cw = v W (mod 2^32)
        let (rows, m, n, c) = (40usize, 24usize, 19usize, 14usize);
        let w: Vec<i64> = (0..m * n).map(|_| (lcg(&mut s) as u8 as i8) as i64).collect();
        let r: Vec<i64> = (0..rows * n).map(|_| lcg(&mut s) as u32 as i64).collect();
        let wr: Vec<i64> = (0..rows * m).map(|x| {
            let (t, i) = (x / m, x % m);
            (0..n).fold(0u32, |a, j| a.wrapping_add((r[t * n + j] as u32).wrapping_mul(w[i * n + j] as u32))) as i64
        }).collect();
        let v: Vec<u32> = (0..m * c).map(|_| lcg(&mut s) as u32).collect();
        let vw: Vec<u32> = (0..n * c).map(|x| {
            let (j, k) = (x / c, x % c);
            (0..m).fold(0u32, |a, i| a.wrapping_add(v[i * c + k].wrapping_mul(w[i * n + j] as u32)))
        }).collect();
        let mut bad = vec![1u8; rows];
        check_rows32(&r, n, &wr, m, &v, &vw, c, u32::MAX, &mut bad, 2).unwrap();
        assert!(bad.iter().all(|&b| b == 0));
    }

    #[test]
    fn wrap32_and_p31_match_scalar() {
        let mut s = 7u64;
        for &(c, m, n) in &[(1usize, 1usize, 1usize), (2, 37, 300), (14, 70, 513), (17, 9, 33), (3, 600, 257)] {
            let w: Vec<i8> = (0..m * n).map(|i| if i % 7 == 0 { -128 } else if i % 11 == 0 { 127 } else { lcg(&mut s) as i8 }).collect();
            let x: Vec<u32> = (0..c * m).map(|i| if i % 5 == 0 { u32::MAX } else { lcg(&mut s) as u32 }).collect();
            let mut o = vec![0u32; c * n];
            vecmat_wrap32(&x, c, &w, m, n, &mut o, 3).unwrap();
            let u: Vec<u64> = (0..c * m).map(|i| if i % 5 == 0 { P31 - 1 } else { lcg(&mut s) % P31 }).collect();
            let mut op = vec![0u64; c * n];
            vecmat_p31(&u, c, &w, m, n, &mut op, 2).unwrap();
            for k in 0..c {
                for j in 0..n {
                    let mut a = 0u32;
                    let mut b: i128 = 0;
                    for i in 0..m {
                        a = a.wrapping_add(x[k * m + i].wrapping_mul(w[i * n + j] as i32 as u32));
                        b += i128::from(u[k * m + i]) * i128::from(w[i * n + j]);
                    }
                    assert_eq!(o[k * n + j], a);
                    assert_eq!(op[k * n + j] as i128, b.rem_euclid(P31 as i128));
                }
            }
        }
    }
}
