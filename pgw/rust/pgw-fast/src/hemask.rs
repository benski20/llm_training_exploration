//! Customer side of the encrypted (MLWE) mask factory (`pgw.hemask`): noise and secret sampling,
//! the integer combines around the int8 byte-plane GEMMs, and the every-row self-check.
//!
//! Every function is bit-identical to the Python code it replaces (`pgw/tests/hemask/
//! test_native_sampling.py` compares them on the same seeds), computes with the GIL released, and
//! splits large work over threads. The GEMMs themselves stay in `torch._int_mm` (AMX/VNNI); only the
//! surrounding uint64 work moves here, fused into one pass.
//!
//! | function | Python reference (`pgw.hemask.lwe` / `backend`) |
//! |---|---|
//! | [`he_ternary`] | `_ternary(SeedExpander(seed), (count,), "s")` |
//! | [`he_secret_matrix`] | `secret_matrix(_ternary(SeedExpander(seed), (rank, 256), "s"))` |
//! | [`he_cbd`] | `_cbd(SeedExpander(seed), (rows, cols), eta, "e")` |
//! | [`he_encrypt_combine`] | `(pad + e + (msg << delta)) & q_mask`, `pad = _combine(part, offset)` |
//! | [`he_decrypt_combine`] | `((y - corr + half) >> delta) & (2^K - 1)`, transposed into the output |
//! | [`he_check_rows`] | `(wr @ cu) & mask != (r @ cw) & mask` per row (`MaskGenerator._self_check`) |
//!
//! Sampling uses the audited key schedule of [`pgw_core_rs::prng::SeedExpander`] (HKDF-SHA256) and
//! ChaCha20 (RFC 8439, zero nonce): the fresh per-(linear, group) seed comes from `os.urandom` in
//! Python and is never stored. Secret buffers are zeroized before they are freed.

use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use chacha20::ChaCha20;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pgw_core_rs::prng::{Label, SeedExpander, MAX_STREAM_BYTES};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use zeroize::{Zeroize, Zeroizing};

/// Polynomial degree of the MLWE ring (`pgw.hemask.lwe.DEGREE`).
pub const DEGREE: usize = 256;
/// Below this many elements a kernel stays on the calling thread.
const PAR_MIN_ELEMS: usize = 1 << 16;
/// Target elements per worker thread above the threshold.
const PAR_ELEMS_PER_THREAD: usize = 1 << 15;

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn threads_for(elems: usize) -> usize {
    if elems < PAR_MIN_ELEMS {
        return 1;
    }
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    cpus.min(elems / PAR_ELEMS_PER_THREAD).max(1)
}

/// Run `f(first_unit, part)` over disjoint runs of whole units (`unit` elements each) of `out`.
fn par_units<T: Send>(out: &mut [T], unit: usize, work_per_unit: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let units = out.len().checked_div(unit).unwrap_or(0);
    let threads = threads_for(units.saturating_mul(work_per_unit)).min(units.max(1));
    if threads <= 1 {
        f(0, out);
        return;
    }
    let per = units.div_ceil(threads);
    std::thread::scope(|s| {
        let mut parts = out.chunks_mut(per * unit).enumerate();
        let first = parts.next();
        let f = &f;
        for (i, part) in parts {
            s.spawn(move || f(i * per, part));
        }
        if let Some((_, part)) = first {
            f(0, part);
        }
    });
}

/// Bytes of a `bytes` / `bytearray` / buffer object (zeroized on drop).
fn secret_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Zeroizing<Vec<u8>>> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Zeroizing::new(b.as_bytes().to_vec()));
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        return Ok(Zeroizing::new(b.to_vec()));
    }
    let buf = PyBuffer::<u8>::get(obj).map_err(|_| PyTypeError::new_err("seed must be bytes-like"))?;
    Ok(Zeroizing::new(buf.to_vec(obj.py())?))
}

fn expander(seed: &[u8]) -> Result<SeedExpander, String> {
    SeedExpander::new(seed).map_err(|e| e.to_string())
}

/// ChaCha20 keystream bytes `[offset, offset + out.len())` of the stream `key` (written over `out`).
fn keystream(key: &[u8; 32], offset: u64, out: &mut [u8]) -> Result<(), String> {
    let end = offset.checked_add(out.len() as u64).ok_or("stream offset overflows")?;
    // The last 64-byte block (counter 2^32 - 1) is refused by the `cipher` wrapper; HE streams are
    // at most a few hundred MB, far below that.
    if end > MAX_STREAM_BYTES - 64 {
        return Err("stream offset beyond the ChaCha20 counter range".into());
    }
    out.fill(0);
    let mut c = ChaCha20::new(key.into(), &[0u8; 12].into());
    c.seek(offset);
    c.apply_keystream(out);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

/// `count` uniform ternary values (as -1/0/1) exactly as `lwe._ternary(sx, (count,), "s")`: round
/// `ctr` draws `2 * missing + 64` bytes of stream `("s", ctr)`, keeps bytes `< 252` and maps them to
/// `b % 3 - 1`, until `count` values are collected.
pub fn ternary(seed: &[u8], count: usize) -> Result<Vec<i8>, String> {
    ternary_rounds(seed, count, |need| 2 * need + 64)
}

/// [`ternary`] with round `ctr` drawing `draw(missing)` bytes (`>= 1`). Stops at exactly `count`
/// values, so the output buffer never reallocates (a reallocation would free an unwiped copy of the
/// secret); Python keeps only the first `count` values, so the result is the same.
fn ternary_rounds(seed: &[u8], count: usize, draw: impl Fn(usize) -> usize) -> Result<Vec<i8>, String> {
    let sx = expander(seed)?;
    let mut out: Vec<i8> = Vec::with_capacity(count);
    let mut ctr: i64 = 0;
    while out.len() < count {
        let need = count - out.len();
        let key = sx.key(&[Label::from("s"), Label::Int(ctr)]).map_err(|e| e.to_string())?;
        ctr += 1;
        let mut raw = Zeroizing::new(vec![0u8; draw(need).max(1)]);
        keystream(&key, 0, &mut raw)?;
        for &b in raw.iter() {
            if b < 252 {
                out.push((b % 3) as i8 - 1);
                if out.len() == count {
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// The block matrix `S` (`rank*256 x 256`, int8, row-major) of `secret_matrix(s)`:
/// `S[l*256 + t, j] = s[l][j - t]` for `j >= t`, `-s[l][j - t + 256]` otherwise.
pub fn secret_matrix_into(s: &[i8], rank: usize, out: &mut [i8]) {
    debug_assert_eq!(s.len(), rank * DEGREE);
    debug_assert_eq!(out.len(), rank * DEGREE * DEGREE);
    for l in 0..rank {
        let sl = &s[l * DEGREE..(l + 1) * DEGREE];
        for t in 0..DEGREE {
            let row = &mut out[(l * DEGREE + t) * DEGREE..(l * DEGREE + t + 1) * DEGREE];
            for (j, o) in row.iter_mut().enumerate() {
                *o = if j >= t { sl[j - t] } else { -sl[j + DEGREE - t] };
            }
        }
    }
}

/// Centred binomial value of one 8-byte little-endian word: `popcount(w & m) - popcount((w >> eta) & m)`.
#[inline(always)]
fn cbd1(w: u64, eta: u32, m: u64) -> i64 {
    i64::from((w & m).count_ones()) - i64::from(((w >> eta) & m).count_ones())
}

fn eta_mask(eta: u32) -> Result<u64, String> {
    if !(1..=32).contains(&eta) {
        return Err(format!("eta must be in [1, 32], got {eta}"));
    }
    Ok((1u64 << eta) - 1)
}

/// CBD(eta) noise elements `[elem0, elem0 + out.len())` of the stream `key` (8 bytes per element).
fn cbd_fill(key: &[u8; 32], elem0: usize, eta: u32, out: &mut [i64]) -> Result<(), String> {
    let m = eta_mask(eta)?;
    // SAFETY: an i64 slice is a valid, aligned u8 slice of 8x the length; no other reference.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 8) };
    keystream(key, 8 * elem0 as u64, bytes)?;
    for v in out.iter_mut() {
        *v = cbd1(u64::from_le(*v as u64), eta, m);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Combines
// ---------------------------------------------------------------------------------------------

/// `sum_p (part[p] + offset) << 8 (first + p)` mod 2^64 (one output element; test reference).
#[cfg(test)]
fn combine1(part: &[i32], stride: usize, j: usize, off: i32, first: u32) -> u64 {
    let mut acc = 0u64;
    for (p, sh) in (first..).take(part.len() / stride).enumerate() {
        let v = (i64::from(part[p * stride + j]) as u64).wrapping_add(i64::from(off) as u64);
        acc = acc.wrapping_add(v.wrapping_shl(8 * sh));
    }
    acc
}

/// Accumulate one row of planes: `acc[j] += (part[p, j] + offset[j]) << 8 (first + p)` for all `p`.
#[inline(always)]
fn combine_row(part_row: &[i32], planes: usize, cols: usize, offset: &[i32], first: u32, j0: usize, acc: &mut [u64]) {
    let w = acc.len();
    for p in 0..planes {
        let sh = 8 * (first + p as u32);
        let src = &part_row[p * cols + j0..p * cols + j0 + w];
        for ((a, &x), &o) in acc.iter_mut().zip(src).zip(&offset[j0..j0 + w]) {
            // i32 + i32 never overflows i64; wrapping ops keep the loop free of overflow-check branches.
            let v = (i64::from(x) as u64).wrapping_add(i64::from(o) as u64);
            *a = a.wrapping_add(v.wrapping_shl(sh));
        }
    }
}

struct EncArgs<'a> {
    part: &'a [i32], // (nb, planes, 256)
    planes: usize,
    offset: &'a [i32], // (256,)
    first: u32,
    r: &'a [i64], // (rows, n): masks, column c of the group is feature c
    rows: usize,
    n: usize,
    row0: usize,
    delta: u32,
    qmask: u64,
    eta: u32,
}

/// Rows `[row0 + c0, row0 + c0 + out.len()/256)` of `b = (pad + e + (msg << delta)) & qmask`.
fn encrypt_rows(a: &EncArgs<'_>, key: &[u8; 32], c0: usize, out: &mut [u64]) -> Result<(), String> {
    let nrows = out.len() / DEGREE;
    let mut e = Zeroizing::new(vec![0i64; DEGREE]);
    let mut msg = Zeroizing::new(vec![0u64; DEGREE]);
    let mut acc = vec![0u64; DEGREE];
    for ci in 0..nrows {
        let lc = c0 + ci; // row within this block
        let c = a.row0 + lc; // feature (global row of b)
        cbd_fill(key, c * DEGREE, a.eta, &mut e)?;
        for (j, mv) in msg.iter_mut().enumerate().take(a.rows) {
            *mv = a.r[j * a.n + c] as u64;
        }
        acc.fill(0);
        combine_row(
            &a.part[lc * a.planes * DEGREE..(lc + 1) * a.planes * DEGREE],
            a.planes,
            DEGREE,
            a.offset,
            a.first,
            0,
            &mut acc,
        );
        let o = &mut out[ci * DEGREE..(ci + 1) * DEGREE];
        for j in 0..DEGREE {
            let m = if j < a.rows { msg[j].wrapping_shl(a.delta) } else { 0 };
            o[j] = acc[j].wrapping_add(e[j] as u64).wrapping_add(m) & a.qmask;
        }
    }
    acc.zeroize();
    Ok(())
}

struct DecArgs<'a> {
    part: &'a [i32], // (nb, planes, cols)
    planes: usize,
    cols: usize,
    offset: &'a [i32], // (cols,)
    first: u32,
    pc0: usize,   // first part column handled (y column 0)
    y: &'a [u64], // (m, yc)
    yc: usize,
    row0: usize,
    slots: &'a [i64], // (yc,) output row of each y column, or < 0 to skip
    delta: u32,
    kmask: u64,
    m: usize,
    nb: usize,
}

/// For y columns `[j0, j1)` and all `nb` hint rows, write the rounded plaintexts. Rows are walked in
/// order so `part` and `y` stream sequentially; the `j1 - j0` output rows being filled (one element
/// per hint row) stay in cache.
fn decrypt_cols(a: &DecArgs<'_>, j0: usize, j1: usize, out: *mut i64) {
    let w = j1 - j0;
    let mut acc = vec![0u64; w];
    // validated: 0 <= s < out rows, so s * m + row0 + nb <= out.len()
    let base: Vec<usize> =
        a.slots[j0..j1].iter().map(|&s| if s < 0 { usize::MAX } else { s as usize * a.m + a.row0 }).collect();
    let half = 1u64 << (a.delta - 1);
    for i in 0..a.nb {
        acc.fill(0);
        combine_row(
            &a.part[i * a.planes * a.cols..(i + 1) * a.planes * a.cols],
            a.planes,
            a.cols,
            a.offset,
            a.first,
            a.pc0 + j0,
            &mut acc,
        );
        let yrow = &a.y[(a.row0 + i) * a.yc + j0..(a.row0 + i) * a.yc + j1];
        for (c, &yv) in acc.iter_mut().zip(yrow) {
            *c = (yv.wrapping_sub(*c).wrapping_add(half) >> a.delta) & a.kmask;
        }
        for (&b, &v) in base.iter().zip(acc.iter()) {
            if b != usize::MAX {
                // SAFETY: `b + i` is inside the output (validated shapes); column runs of
                // different threads have distinct slots (checked), so writes never alias.
                unsafe { *out.add(b.wrapping_add(i)) = v as i64 };
            }
        }
    }
    acc.zeroize();
}

struct SendPtr(*mut i64);
// SAFETY: threads write disjoint elements (distinct output rows per thread, see `decrypt_cols`).
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// ---------------------------------------------------------------------------------------------
// Row check
// ---------------------------------------------------------------------------------------------

/// Rows per block of the 32-bit check kernel (each check-vector row is loaded once per block).
const CHECK_RB: usize = 4;

/// Check vectors as zero-padded `u32` rows of `CP` lanes (`v` is `(len, c)` uint64; padding lanes are 0
/// on both sides of the comparison, so they never differ).
fn pad_u32<const CP: usize>(v: &[u64], len: usize, c: usize) -> Zeroizing<Vec<u32>> {
    let mut out = Zeroizing::new(vec![0u32; len * CP]);
    for (dst, src) in out.chunks_exact_mut(CP).zip(v.chunks_exact(c)) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s as u32;
        }
    }
    out
}

/// `acc[b] += x[t + b] * v` over a block of `RB` rows (`x` row-major `(rows, len)`), mod 2^32.
#[inline(always)]
fn dot_block<const CP: usize, const RB: usize>(x: &[i64], len: usize, t: usize, v: &[u32], acc: &mut [[u32; CP]; RB]) {
    let xs = &x[t * len..(t + RB) * len];
    for (i, vrow) in v.chunks_exact(CP).enumerate().take(len) {
        let vrow: &[u32; CP] = vrow.try_into().expect("CP lanes");
        for (b, a) in acc.iter_mut().enumerate() {
            let xv = xs[b * len + i] as u32;
            for (aj, &vj) in a.iter_mut().zip(vrow) {
                *aj = aj.wrapping_add(xv.wrapping_mul(vj));
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn check_u32_part<const CP: usize, const RB: usize>(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u32],
    cw: &[u32],
    mask: u32,
    t: usize,
    flags: &mut [u8],
) {
    let mut l = [[0u32; CP]; RB];
    let mut rr = [[0u32; CP]; RB];
    dot_block::<CP, RB>(wr, m, t, cu, &mut l);
    dot_block::<CP, RB>(r, n, t, cw, &mut rr);
    for (b, f) in flags.iter_mut().enumerate().take(RB) {
        *f = u8::from(l[b].iter().zip(&rr[b]).any(|(&x, &y)| (x & mask) != (y & mask)));
    }
    for a in l.iter_mut().chain(rr.iter_mut()) {
        a.zeroize();
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn check_u32_rows<const CP: usize>(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u32],
    cw: &[u32],
    mask: u32,
    t0: usize,
    part: &mut [u8],
) {
    let full = part.len() / CHECK_RB * CHECK_RB;
    for (bi, fl) in part[..full].chunks_exact_mut(CHECK_RB).enumerate() {
        check_u32_part::<CP, CHECK_RB>(r, n, wr, m, cu, cw, mask, t0 + bi * CHECK_RB, fl);
    }
    for ti in full..part.len() {
        check_u32_part::<CP, 1>(r, n, wr, m, cu, cw, mask, t0 + ti, &mut part[ti..=ti]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn check_u32_avx512<const CP: usize>(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u32],
    cw: &[u32],
    mask: u32,
    t0: usize,
    part: &mut [u8],
) {
    check_u32_rows::<CP>(r, n, wr, m, cu, cw, mask, t0, part);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn check_u32_avx2<const CP: usize>(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u32],
    cw: &[u32],
    mask: u32,
    t0: usize,
    part: &mut [u8],
) {
    check_u32_rows::<CP>(r, n, wr, m, cu, cw, mask, t0, part);
}

/// The check for `K <= 32` (`mask < 2^32`): the same flags as the uint64 path (every product and sum is
/// taken mod 2^32, which `2^K` divides), on 32-bit SIMD lanes with the `c` vectors side by side (`CP` lanes)
/// and blocks of [`CHECK_RB`] rows. This keeps the 40-bit self-check of encrypted factories (14 vectors at
/// capped K32 widths) cheap.
#[allow(clippy::too_many_arguments)]
fn check_rows_u32<const CP: usize>(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u64],
    cw: &[u64],
    c: usize,
    mask: u32,
    bad: &mut [u8],
) {
    let rows = bad.len();
    let cu32 = pad_u32::<CP>(cu, m, c);
    let cw32 = pad_u32::<CP>(cw, n, c);
    let (cu32, cw32) = (&cu32[..], &cw32[..]);
    let one = |t0: usize, part: &mut [u8]| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2") {
                // SAFETY: the CPU supports the enabled features (checked above).
                unsafe { check_u32_avx512::<CP>(r, n, wr, m, cu32, cw32, mask, t0, part) };
                return;
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: as above.
                unsafe { check_u32_avx2::<CP>(r, n, wr, m, cu32, cw32, mask, t0, part) };
                return;
            }
        }
        check_u32_rows::<CP>(r, n, wr, m, cu32, cw32, mask, t0, part);
    };
    let threads = threads_for(rows.saturating_mul(n + m) / 16).min(rows.div_ceil(CHECK_RB).max(1));
    if threads <= 1 {
        one(0, bad);
        return;
    }
    let per = rows.div_ceil(threads).div_ceil(CHECK_RB) * CHECK_RB;
    std::thread::scope(|s| {
        let one = &one;
        for (i, part) in bad.chunks_mut(per).enumerate() {
            s.spawn(move || one(i * per, part));
        }
    });
}

/// `true` where `(wr[t] @ cu) & mask != (r[t] @ cw) & mask` (all arithmetic mod 2^64; for `mask < 2^32`
/// and at most 16 vectors on 32-bit SIMD lanes, [`check_rows_u32`], with the same flags).
#[allow(clippy::too_many_arguments)]
pub fn check_rows(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u64],
    cw: &[u64],
    c: usize,
    mask: u64,
    bad: &mut [u8],
) {
    if mask <= u64::from(u32::MAX) && (1..=16).contains(&c) {
        if c <= 8 {
            check_rows_u32::<8>(r, n, wr, m, cu, cw, c, mask as u32, bad);
        } else {
            check_rows_u32::<16>(r, n, wr, m, cu, cw, c, mask as u32, bad);
        }
        return;
    }
    check_rows_u64(r, n, wr, m, cu, cw, c, mask, bad);
}

/// The uint64 path of [`check_rows`] (any `K` up to 63, any number of vectors).
#[allow(clippy::too_many_arguments)]
pub fn check_rows_u64(
    r: &[i64],
    n: usize,
    wr: &[i64],
    m: usize,
    cu: &[u64],
    cw: &[u64],
    c: usize,
    mask: u64,
    bad: &mut [u8],
) {
    let rows = bad.len();
    let threads = threads_for(rows.saturating_mul(n + m) / 16).min(rows.max(1));
    let one = |t0: usize, part: &mut [u8]| {
        let mut l = vec![0u64; c];
        let mut rr = vec![0u64; c];
        for (ti, flag) in part.iter_mut().enumerate() {
            let t = t0 + ti;
            l.fill(0);
            rr.fill(0);
            for (i, &x) in wr[t * m..(t + 1) * m].iter().enumerate() {
                let x = x as u64;
                for (a, &u) in l.iter_mut().zip(&cu[i * c..(i + 1) * c]) {
                    *a = a.wrapping_add(x.wrapping_mul(u));
                }
            }
            for (i, &x) in r[t * n..(t + 1) * n].iter().enumerate() {
                let x = x as u64;
                for (a, &u) in rr.iter_mut().zip(&cw[i * c..(i + 1) * c]) {
                    *a = a.wrapping_add(x.wrapping_mul(u));
                }
            }
            *flag = u8::from(l.iter().zip(&rr).any(|(&x, &y)| (x & mask) != (y & mask)));
        }
        l.zeroize();
        rr.zeroize();
    };
    if threads <= 1 {
        one(0, bad);
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        let one = &one;
        for (i, part) in bad.chunks_mut(per).enumerate() {
            s.spawn(move || one(i * per, part));
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------------------------

/// Uniform ternary samples (int8 ``-1/0/1``) of ``pgw.hemask.lwe._ternary(SeedExpander(seed), (count,), "s")``.
#[pyfunction]
#[pyo3(signature = (seed, count))]
fn he_ternary<'py>(py: Python<'py>, seed: &Bound<'py, PyAny>, count: usize) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let seed = secret_bytes(seed)?;
    let mut v = py.allow_threads(|| ternary(&seed, count)).map_err(verr)?;
    let arr = PyArray1::<i8>::from_slice(py, &v);
    v.zeroize();
    Ok(arr)
}

/// Test hook: [`he_ternary`] with every rejection round drawing a fixed ``draw`` bytes, so the
/// later rounds (stream labels ``("s", 1)``, ``("s", 2)``, ...) are exercised; the production draw
/// (``2 * missing + 64``) almost never needs a second round.
#[pyfunction]
#[pyo3(name = "_he_ternary_fixed_draw", signature = (seed, count, draw))]
fn he_ternary_fixed_draw<'py>(
    py: Python<'py>,
    seed: &Bound<'py, PyAny>,
    count: usize,
    draw: usize,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let seed = secret_bytes(seed)?;
    let mut v = py.allow_threads(|| ternary_rounds(&seed, count, |_| draw)).map_err(verr)?;
    let arr = PyArray1::<i8>::from_slice(py, &v);
    v.zeroize();
    Ok(arr)
}

/// Secret block matrix ``S`` (``rank*256 x 256`` int8) of a fresh ternary secret drawn from ``seed``:
/// ``secret_matrix(_ternary(SeedExpander(seed), (rank, 256), "s"))``.
#[pyfunction]
#[pyo3(signature = (seed, rank))]
fn he_secret_matrix<'py>(py: Python<'py>, seed: &Bound<'py, PyAny>, rank: usize) -> PyResult<Bound<'py, PyArray2<i8>>> {
    if rank == 0 || rank > 64 {
        return Err(verr(format!("rank must be in [1, 64], got {rank}")));
    }
    let seed = secret_bytes(seed)?;
    let arr = PyArray2::<i8>::zeros(py, [rank * DEGREE, DEGREE], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let out = unsafe { arr.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
    py.allow_threads(|| -> Result<(), String> {
        let mut s = ternary(&seed, rank * DEGREE)?;
        secret_matrix_into(&s, rank, out);
        s.zeroize();
        Ok(())
    })
    .map_err(verr)?;
    Ok(arr)
}

/// CBD(eta) noise ``(rows, cols)`` int64 of ``_cbd(SeedExpander(seed), (rows, cols), eta, "e")``.
#[pyfunction]
#[pyo3(signature = (seed, rows, cols, eta))]
fn he_cbd<'py>(
    py: Python<'py>,
    seed: &Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    eta: u32,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let seed = secret_bytes(seed)?;
    let key = expander(&seed).and_then(|sx| sx.key(&[Label::from("e")]).map_err(|e| e.to_string())).map_err(verr)?;
    rows.checked_mul(cols).and_then(|n| n.checked_mul(8)).ok_or_else(|| verr("size overflows"))?;
    let arr = PyArray2::<i64>::zeros(py, [rows, cols], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let out = unsafe { arr.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
    py.allow_threads(|| -> Result<(), String> {
        let mut err = Ok(());
        let errs = std::sync::Mutex::new(&mut err);
        par_units(out, 1, 1, |e0, part| {
            if let Err(e) = cbd_fill(&key, e0, eta, part) {
                **errs.lock().unwrap() = Err(e);
            }
        });
        err
    })
    .map_err(verr)?;
    Ok(arr)
}

/// Fused tail of ``encrypt_masks`` for ciphertext rows ``[row0, row0 + nb)``:
/// ``out[row0 + c] = (pad[c] + e[row0 + c] + (msg[row0 + c] << delta_bits)) & (2^q_bits - 1)`` with
/// ``pad[c, j] = sum_p (part[c, p, j] + offset[j]) << 8 (first_byte + p)`` (mod 2^64),
/// ``e = _cbd(SeedExpander(seed), (n, 256), eta, "e")`` and ``msg[c, j] = r[j, c]`` (0 for ``j >= rows``).
///
/// Args:
///     part: int32 ``(nb, planes, 256)``: the int8 GEMM of the public-matrix byte planes with ``S``.
///     offset: int32 ``(256,)``: ``128 * colsum(S)`` (the -128 plane bias).
///     r: int64 ``(rows, n)`` masks of the group (``rows <= 256``).
///     out: uint64 ``(n, 256)`` ciphertext, written in rows ``[row0, row0 + nb)``.
#[pyfunction]
#[pyo3(signature = (part, offset, first_byte, seed, eta, r, row0, delta_bits, q_bits, out))]
#[allow(clippy::too_many_arguments)]
fn he_encrypt_combine<'py>(
    py: Python<'py>,
    part: PyReadonlyArray3<'py, i32>,
    offset: PyReadonlyArray1<'py, i32>,
    first_byte: u32,
    seed: &Bound<'py, PyAny>,
    eta: u32,
    r: PyReadonlyArray2<'py, i64>,
    row0: usize,
    delta_bits: u32,
    q_bits: u32,
    mut out: PyReadwriteArray2<'py, u64>,
) -> PyResult<()> {
    let (nb, planes, deg) = (part.shape()[0], part.shape()[1], part.shape()[2]);
    let (rows, n) = (r.shape()[0], r.shape()[1]);
    let (on, odeg) = (out.shape()[0], out.shape()[1]);
    if deg != DEGREE || odeg != DEGREE || offset.shape()[0] != DEGREE {
        return Err(verr("part, offset and out must have 256 columns"));
    }
    if rows > DEGREE || on != n || row0.checked_add(nb).is_none_or(|e| e > n) {
        return Err(verr(format!(
            "shape mismatch: part {:?}, r {:?}, out {:?}, row0 {row0}",
            part.shape(),
            r.shape(),
            out.shape()
        )));
    }
    if planes == 0
        || first_byte as usize + planes > 8
        || !(1..=64).contains(&q_bits)
        || delta_bits == 0
        || delta_bits >= q_bits
    {
        return Err(verr("bad planes/first_byte/q_bits/delta_bits"));
    }
    eta_mask(eta).map_err(verr)?;
    let seed = secret_bytes(seed)?;
    let key = expander(&seed).and_then(|sx| sx.key(&[Label::from("e")]).map_err(|e| e.to_string())).map_err(verr)?;
    let cerr = |w: &str| verr(format!("{w} must be C-contiguous"));
    let part_s = part.as_slice().map_err(|_| cerr("part"))?;
    let off_s = offset.as_slice().map_err(|_| cerr("offset"))?;
    let r_s = r.as_slice().map_err(|_| cerr("r"))?;
    let out_s = out.as_slice_mut().map_err(|_| cerr("out"))?;
    let out_blk = &mut out_s[row0 * DEGREE..(row0 + nb) * DEGREE];
    let a = EncArgs {
        part: part_s,
        planes,
        offset: off_s,
        first: first_byte,
        r: r_s,
        rows,
        n,
        row0,
        delta: delta_bits,
        qmask: if q_bits == 64 { u64::MAX } else { (1u64 << q_bits) - 1 },
        eta,
    };
    py.allow_threads(|| -> Result<(), String> {
        let mut err = Ok(());
        let errs = std::sync::Mutex::new(&mut err);
        par_units(out_blk, DEGREE, DEGREE * (planes + 4), |c0, blk| {
            if let Err(e) = encrypt_rows(&a, &key, c0, blk) {
                **errs.lock().unwrap() = Err(e);
            }
        });
        err
    })
    .map_err(verr)
}

/// Fused tail of the decryption of hint rows ``[row0, row0 + nb)`` for the y columns of one reply:
/// ``out[slots[j], row0 + i] = ((y[row0 + i, j] - corr[i, pc0 + j] + 2^(delta-1)) >> delta) & (2^k - 1)``
/// with ``corr[i, c] = sum_p (part[i, p, c] + offset[c]) << 8 (first_byte + p)`` (mod 2^64); columns
/// with ``slots[j] < 0`` (padding slots of a partial group) are skipped.
///
/// Args:
///     part: int32 ``(nb, planes, C)``: the int8 GEMM of hint byte planes with ``[S_1 | S_2 | ...]``.
///     offset: int32 ``(C,)``.
///     pc0: first column of ``part`` that belongs to ``y`` (``y`` covers ``[pc0, pc0 + yc)``).
///     y: uint64 ``(m, yc)`` provider reply.
///     slots: int64 ``(yc,)`` distinct output rows (or negative).
///     out: int64 ``(slots_total, m)``.
#[pyfunction]
#[pyo3(signature = (part, offset, first_byte, pc0, y, row0, slots, delta_bits, k, out))]
#[allow(clippy::too_many_arguments)]
fn he_decrypt_combine<'py>(
    py: Python<'py>,
    part: PyReadonlyArray3<'py, i32>,
    offset: PyReadonlyArray1<'py, i32>,
    first_byte: u32,
    pc0: usize,
    y: PyReadonlyArray2<'py, u64>,
    row0: usize,
    slots: PyReadonlyArray1<'py, i64>,
    delta_bits: u32,
    k: u32,
    mut out: PyReadwriteArray2<'py, i64>,
) -> PyResult<()> {
    let (nb, planes, cols) = (part.shape()[0], part.shape()[1], part.shape()[2]);
    let (m, yc) = (y.shape()[0], y.shape()[1]);
    let (orows, om) = (out.shape()[0], out.shape()[1]);
    if offset.shape()[0] != cols
        || slots.shape()[0] != yc
        || pc0.checked_add(yc).is_none_or(|e| e > cols)
        || om != m
        || row0.checked_add(nb).is_none_or(|e| e > m)
    {
        return Err(verr(format!(
            "shape mismatch: part {:?}, offset {:?}, y {:?}, slots {:?}, out {:?}, pc0 {pc0}, \
                                 row0 {row0}",
            part.shape(),
            offset.shape(),
            y.shape(),
            slots.shape(),
            out.shape()
        )));
    }
    if planes == 0 || first_byte as usize + planes > 8 || delta_bits == 0 || delta_bits >= 64 || !(1..=63).contains(&k)
    {
        return Err(verr("bad planes/first_byte/delta_bits/k"));
    }
    let cerr = |w: &str| verr(format!("{w} must be C-contiguous"));
    let slots_s = slots.as_slice().map_err(|_| cerr("slots"))?;
    let mut seen = vec![false; orows];
    for &s in slots_s {
        if s >= 0 {
            let s = s as usize;
            if s >= orows || std::mem::replace(&mut seen[s], true) {
                return Err(verr(format!("slot {s} out of range or repeated")));
            }
        }
    }
    let a = DecArgs {
        part: part.as_slice().map_err(|_| cerr("part"))?,
        planes,
        cols,
        offset: offset.as_slice().map_err(|_| cerr("offset"))?,
        first: first_byte,
        pc0,
        y: y.as_slice().map_err(|_| cerr("y"))?,
        yc,
        row0,
        slots: slots_s,
        delta: delta_bits,
        kmask: (1u64 << k) - 1,
        m,
        nb,
    };
    let out_s = out.as_slice_mut().map_err(|_| cerr("out"))?;
    let ptr = SendPtr(out_s.as_mut_ptr());
    py.allow_threads(|| {
        let threads = threads_for(nb.saturating_mul(yc).saturating_mul(planes) / 4).min(yc.div_ceil(64).max(1));
        let per = yc.div_ceil(threads.max(1)).div_ceil(8) * 8;
        let p = &ptr;
        let a = &a;
        std::thread::scope(|s| {
            for j0 in (0..yc).step_by(per.max(1)).skip(1) {
                s.spawn(move || decrypt_cols(a, j0, (j0 + per).min(yc), p.0));
            }
            decrypt_cols(a, 0, per.min(yc), p.0);
        });
    });
    Ok(())
}

/// Every-row Freivalds self-check of the factory: ``bad[t] = any((wr[t] @ cu) & mask != (r[t] @ cw) & mask)``
/// (uint64 arithmetic mod 2^64, as ``MaskGenerator._self_check``; for ``mask < 2^32`` and at most 16 vectors
/// the same flags on 32-bit SIMD lanes, unless ``wide`` asks for the uint64 path). Returns uint8 ``(rows,)``.
///
/// Args:
///     r: int64 ``(rows, n)``; wr: int64 ``(rows, m)``; cu: uint64 ``(m, c)``; cw: uint64 ``(n, c)``.
#[pyfunction]
#[pyo3(signature = (r, wr, cu, cw, mask, wide=false))]
#[allow(clippy::too_many_arguments)]
fn he_check_rows<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, i64>,
    wr: PyReadonlyArray2<'py, i64>,
    cu: PyReadonlyArray2<'py, u64>,
    cw: PyReadonlyArray2<'py, u64>,
    mask: u64,
    wide: bool,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (rows, n) = (r.shape()[0], r.shape()[1]);
    let m = wr.shape()[1];
    let c = cu.shape()[1];
    if wr.shape()[0] != rows || cu.shape()[0] != m || cw.shape()[0] != n || cw.shape()[1] != c {
        return Err(verr(format!(
            "shape mismatch: r {:?}, wr {:?}, cu {:?}, cw {:?}",
            r.shape(),
            wr.shape(),
            cu.shape(),
            cw.shape()
        )));
    }
    let cerr = |w: &str| verr(format!("{w} must be C-contiguous"));
    let (rs, ws, us, vs) = (
        r.as_slice().map_err(|_| cerr("r"))?,
        wr.as_slice().map_err(|_| cerr("wr"))?,
        cu.as_slice().map_err(|_| cerr("cu"))?,
        cw.as_slice().map_err(|_| cerr("cw"))?,
    );
    let arr = PyArray1::<u8>::zeros(py, [rows], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let bad = unsafe { arr.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
    py.allow_threads(|| {
        if wide {
            check_rows_u64(rs, n, ws, m, us, vs, c, mask, bad)
        } else {
            check_rows(rs, n, ws, m, us, vs, c, mask, bad)
        }
    });
    Ok(arr)
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(he_ternary, m)?)?;
    m.add_function(wrap_pyfunction!(he_ternary_fixed_draw, m)?)?;
    m.add_function(wrap_pyfunction!(he_secret_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(he_cbd, m)?)?;
    m.add_function(wrap_pyfunction!(he_encrypt_combine, m)?)?;
    m.add_function(wrap_pyfunction!(he_decrypt_combine, m)?)?;
    m.add_function(wrap_pyfunction!(he_check_rows, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine1_matches_combine_row() {
        let part: Vec<i32> = (0..7 * 4).map(|x| x * 7919 - 50_000).collect();
        let off = [3, -5, 1 << 20, -(1 << 21)];
        let mut acc = vec![0u64; 4];
        combine_row(&part, 7, 4, &off, 0, 0, &mut acc);
        for j in 0..4 {
            assert_eq!(acc[j], combine1(&part, 4, j, off[j], 0));
        }
    }

    #[test]
    fn ternary_is_balanced_and_deterministic() {
        let seed: Vec<u8> = (0u8..32).collect();
        let a = ternary(&seed, 30_000).unwrap();
        assert_eq!(a, ternary(&seed, 30_000).unwrap());
        let cnt = |v: i8| a.iter().filter(|&&x| x == v).count() as f64 / a.len() as f64;
        for v in [-1, 0, 1] {
            assert!((cnt(v) - 1.0 / 3.0).abs() < 0.02);
        }
    }

    #[test]
    fn ternary_never_reallocates_and_short_draws_continue() {
        let seed: Vec<u8> = (0u8..32).collect();
        let a = ternary(&seed, 5000).unwrap();
        assert_eq!((a.len(), a.capacity()), (5000, 5000));
        // Round 0 of a short draw is a prefix of round 0 of the full draw (same stream).
        let short = ternary_rounds(&seed, 5000, |_| 100).unwrap();
        let first = short.iter().zip(&a).take_while(|(x, y)| x == y).count();
        assert!((90..=100).contains(&first), "{first}");
        assert_eq!(short.len(), 5000);
        assert_ne!(short, a);
    }

    #[test]
    fn cbd_offset_matches_prefix() {
        let sx = SeedExpander::new(&[7u8; 32]).unwrap();
        let key = sx.key(&[Label::from("e")]).unwrap();
        let mut whole = vec![0i64; 1000];
        cbd_fill(&key, 0, 21, &mut whole).unwrap();
        let mut tail = vec![0i64; 300];
        cbd_fill(&key, 700, 21, &mut tail).unwrap();
        assert_eq!(&whole[700..], &tail[..]);
        assert!(whole.iter().all(|&x| x.abs() <= 21));
    }
}
