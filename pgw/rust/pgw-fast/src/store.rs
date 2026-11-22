//! Mask fetch of `MaskReservation.take` (`pgw.maskstore.store`): regenerate the one-time pad `r`
//! and decode the decrypted `W r` rows. Functions are registered in [`register`].
//!
//! * [`mask_rows`] = `SeedExpander(seed).ring_uniform_rows(cols, k, row_start, rows, *labels)`.
//!   The stream key comes from the audited [`pgw_core_rs::prng::SeedExpander::key`] (HKDF-SHA256);
//!   the ChaCha20 keystream (RFC 8439, zero nonce, seek to byte `8 * cols * row_start`) is written
//!   directly into the numpy output, split over threads for large outputs (seeking is free).
//! * [`mask_key`] / [`mask_rows_with_key`]: the same with the HKDF step hoisted out.
//! * [`unpack_rows`] = `unpack_ring(buf[a:b], (rows, cols), k)` with `a = row_offset*cols*k/8`.
//!
//! All of them return a fresh C-contiguous `int64` array of shape `(rows, cols)` and compute
//! with the GIL released. Errors are `ValueError` (bad sizes/ranges) or `TypeError` (bad label).

use chacha20::cipher::consts::U10;
use chacha20::cipher::{Block, KeyIvInit, StreamCipher, StreamCipherCore, StreamCipherSeek, StreamCipherSeekCore};
use chacha20::{ChaCha20, ChaChaCore};
use numpy::{PyArray2, PyArrayMethods};
use pgw_core_rs::prng::{Label, SeedExpander, MAX_STREAM_BYTES};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyInt, PyString};
use zeroize::{Zeroize, Zeroizing};

/// Stream byte offset of the last ChaCha20 block (counter `2^32 - 1`).
const LAST_BLOCK_START: u64 = MAX_STREAM_BYTES - 64;

/// Below this many output bytes `mask_rows` stays on the calling thread (a thread spawn costs
/// ~20-50 us; ChaCha20 runs at ~1.5-2 GB/s per core).
const PAR_MIN_BYTES: usize = 512 << 10;
/// `unpack_rows` is memory-bound; threads only pay off for very large copies.
const UNPACK_PAR_MIN_BYTES: usize = 2 << 20;
/// Target work per thread above the threshold.
const PAR_BYTES_PER_THREAD: usize = 256 << 10;

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Run `f(first_elem_index, part)` over disjoint parts of `out`, in parallel when large.
fn for_each_part<T: Send>(out: &mut [T], bytes_per_elem: usize, min_bytes: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let total = out.len().saturating_mul(bytes_per_elem);
    let threads = if total < min_bytes {
        1
    } else {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        cpus.min(total / PAR_BYTES_PER_THREAD).max(1)
    };
    if threads <= 1 {
        f(0, out);
        return;
    }
    let per = out.len().div_ceil(threads);
    std::thread::scope(|s| {
        let mut parts = out.chunks_mut(per).enumerate();
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

/// Ring width mask for `k` in `1..=64`.
fn ring_mask(k: u32) -> Result<u64, String> {
    if !(1..=64).contains(&k) {
        return Err(format!("ring width must be in [1, 64], got {k}"));
    }
    Ok(if k == 64 { u64::MAX } else { (1u64 << k) - 1 })
}

/// Elements `[elem_offset, elem_offset + out.len())` of the ring stream keyed by `key`:
/// `u64_le(8 keystream bytes) & (2^k - 1)`, stored as `i64` (wrapping for `k = 64`, like numpy's
/// `astype(np.int64)`). `out` must be zero on entry (the keystream is XORed in).
pub(crate) fn fill_ring(key: &[u8; 32], elem_offset: u64, out: &mut [i64], k: u32) -> Result<(), String> {
    let mask = ring_mask(k)?;
    let beyond = "stream offset beyond the 2^32-block ChaCha20 counter range";
    let byte_off = elem_offset.checked_mul(8).ok_or(beyond)?;
    let end = (out.len() as u64).checked_mul(8).and_then(|n| byte_off.checked_add(n)).ok_or(beyond)?;
    if end > MAX_STREAM_BYTES {
        return Err(beyond.into());
    }
    for_each_part(out, 8, PAR_MIN_BYTES, |first, part| {
        let pos = byte_off + 8 * first as u64;
        // SAFETY: an i64 slice is a valid, aligned u8 slice of 8x the length; no other reference.
        let bytes = unsafe { std::slice::from_raw_parts_mut(part.as_mut_ptr().cast::<u8>(), part.len() * 8) };
        // The `cipher` wrapper refuses to emit the whole last block (counter 2^32 - 1), which
        // Python/OpenSSL produce normally; that block is generated with the raw core instead.
        let split = LAST_BLOCK_START.saturating_sub(pos).min(bytes.len() as u64) as usize;
        let (head, tail) = bytes.split_at_mut(split);
        if !head.is_empty() {
            let mut cipher = ChaCha20::new(key.into(), &[0u8; 12].into());
            cipher.seek(pos); // pos < LAST_BLOCK_START: in range
            cipher.apply_keystream(head);
        }
        if !tail.is_empty() {
            let mut core = ChaChaCore::<U10>::new(key.into(), &[0u8; 12].into());
            core.set_block_pos(u32::MAX);
            let mut blk = Block::<ChaChaCore<U10>>::default();
            core.write_keystream_block(&mut blk);
            let o = (pos + split as u64 - LAST_BLOCK_START) as usize; // tail ends at <= MAX_STREAM_BYTES
            for (b, s) in tail.iter_mut().zip(&blk[o..]) {
                *b ^= s;
            }
            blk.as_mut_slice().zeroize();
        }
        for v in part.iter_mut() {
            *v = (u64::from_le(*v as u64) & mask) as i64;
        }
    });
    Ok(())
}

/// Zero-extend `N`-byte little-endian elements of `src` into `out`.
fn decode_n<const N: usize>(src: &[u8], out: &mut [i64]) {
    for (o, c) in out.iter_mut().zip(src.chunks_exact(N)) {
        let mut w = [0u8; 8];
        w[..N].copy_from_slice(c);
        *o = u64::from_le_bytes(w) as i64;
    }
}

fn decode(src: &[u8], out: &mut [i64], nb: usize) {
    match nb {
        1 => decode_n::<1>(src, out),
        2 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = i64::from(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        3 => decode_n::<3>(src, out),
        4 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = i64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        5 => decode_n::<5>(src, out),
        6 => decode_n::<6>(src, out),
        7 => decode_n::<7>(src, out),
        8 => decode_n::<8>(src, out),
        _ => unreachable!("nb validated to 1..=8"),
    }
}

/// Bytes of a `bytes` / `bytearray` / buffer object (zeroized on drop: seeds and keys).
fn secret_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Zeroizing<Vec<u8>>> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Zeroizing::new(b.as_bytes().to_vec()));
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        return Ok(Zeroizing::new(b.to_vec()));
    }
    let buf = PyBuffer::<u8>::get(obj).map_err(|_| PyTypeError::new_err("expected bytes-like"))?;
    Ok(Zeroizing::new(buf.to_vec(obj.py())?))
}

/// Python label (`str`, `int` fitting i64, bytes-like; not `bool`) -> [`Label`].
fn to_label(obj: &Bound<'_, PyAny>) -> PyResult<Label> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("bool is not a valid PRNG label"));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Label::Int(obj.extract::<i64>()?));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Label::Str(s.to_str()?.to_owned()));
    }
    match secret_bytes(obj) {
        Ok(b) => Ok(Label::Bytes(b.to_vec())),
        Err(_) => Err(PyTypeError::new_err(format!(
            "unsupported PRNG label type {}",
            obj.get_type().name().map_or_else(|_| "?".to_owned(), |n| n.to_string())
        ))),
    }
}

fn to_labels(labels: &Bound<'_, PyAny>) -> PyResult<Vec<Label>> {
    if labels.is_instance_of::<PyString>() || labels.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err("labels must be a list/tuple of labels, not a single str/bytes"));
    }
    labels.try_iter()?.map(|o| to_label(&o?)).collect()
}

fn derive_key(seed: &[u8], labels: &[Label]) -> PyResult<Zeroizing<[u8; 32]>> {
    let sx = SeedExpander::new(seed).map_err(|e| value_err(e.to_string()))?;
    sx.key(labels).map_err(|e| value_err(e.to_string()))
}

/// Validated `(rows, cols, elem_offset)` of a mask row range.
fn row_range(cols: i64, row_start: i64, rows: i64) -> PyResult<(usize, usize, u64)> {
    if cols <= 0 || row_start < 0 || rows < 0 {
        return Err(value_err("cols must be > 0 and row_start/row_count >= 0"));
    }
    let off = (cols as u64).checked_mul(row_start as u64).ok_or_else(|| value_err("row offset overflows"))?;
    let (rows, cols) = (rows as usize, cols as usize);
    rows.checked_mul(cols).and_then(|n| n.checked_mul(8)).ok_or_else(|| value_err("row count overflows"))?;
    Ok((rows, cols, off))
}

fn rows_from_key<'py>(
    py: Python<'py>,
    key: &[u8; 32],
    cols: i64,
    k: u32,
    row_start: i64,
    rows: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (rows, cols, off) = row_range(cols, row_start, rows)?;
    ring_mask(k).map_err(value_err)?;
    let arr = PyArray2::<i64>::zeros(py, [rows, cols], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let out = unsafe { arr.as_slice_mut() }.map_err(|e| value_err(e.to_string()))?;
    py.allow_threads(|| fill_ring(key, off, out, k)).map_err(value_err)?;
    Ok(arr)
}

/// Rows `[row_start, row_start + rows)` of the `(*, cols)` ring matrix of the stream `labels` of
/// `seed`, as int64 values in `[0, 2^k)`: bit-identical to
/// `SeedExpander(seed).ring_uniform_rows(cols, k, row_start, rows, *labels)`.
#[pyfunction]
#[pyo3(signature = (seed, labels, cols, k, row_start, rows))]
fn mask_rows<'py>(
    py: Python<'py>,
    seed: &Bound<'py, PyAny>,
    labels: &Bound<'py, PyAny>,
    cols: i64,
    k: u32,
    row_start: i64,
    rows: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let seed = secret_bytes(seed)?;
    let labels = to_labels(labels)?;
    let key = derive_key(&seed, &labels)?;
    rows_from_key(py, &key, cols, k, row_start, rows)
}

/// The 32-byte ChaCha20 stream key `HKDF-Expand(seed, info(labels))` (`SeedExpander._key`).
/// Secret: as sensitive as the seed for this stream.
#[pyfunction]
#[pyo3(signature = (seed, labels))]
fn mask_key<'py>(
    py: Python<'py>,
    seed: &Bound<'py, PyAny>,
    labels: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let seed = secret_bytes(seed)?;
    let labels = to_labels(labels)?;
    let key = derive_key(&seed, &labels)?;
    Ok(PyBytes::new(py, key.as_slice()))
}

/// [`mask_rows`] with the stream key from [`mask_key`].
#[pyfunction]
#[pyo3(signature = (key, cols, k, row_start, rows))]
fn mask_rows_with_key<'py>(
    py: Python<'py>,
    key: &Bound<'py, PyAny>,
    cols: i64,
    k: u32,
    row_start: i64,
    rows: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let raw = secret_bytes(key)?;
    let key: Zeroizing<[u8; 32]> =
        Zeroizing::new(raw.as_slice().try_into().map_err(|_| value_err("key must be exactly 32 bytes"))?);
    rows_from_key(py, &key, cols, k, row_start, rows)
}

/// `rows` rows of `cols` packed `k/8`-byte little-endian ring elements starting at row
/// `row_offset` of `buf`, as an int64 `(rows, cols)` array: equal to
/// `unpack_ring(buf[a:b], (rows, cols), k)` with `a = row_offset * cols * k/8`, `b = a + rows * cols * k/8`.
#[pyfunction]
#[pyo3(signature = (buf, row_offset, rows, cols, k))]
fn unpack_rows<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyAny>,
    row_offset: i64,
    rows: i64,
    cols: i64,
    k: u32,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    if k % 8 != 0 || !(8..=64).contains(&k) {
        return Err(value_err(format!("bad ring width {k}")));
    }
    if row_offset < 0 || rows < 0 || cols < 0 {
        return Err(value_err("row_offset, rows and cols must be >= 0"));
    }
    let nb = (k / 8) as usize;
    let (rows, cols) = (rows as usize, cols as usize);
    let pb = PyBuffer::<u8>::get(buf)?;
    if !pb.is_c_contiguous() {
        return Err(value_err("buffer must be C-contiguous"));
    }
    let len = pb.len_bytes();
    let row_bytes = cols.checked_mul(nb).ok_or_else(|| value_err("row size overflows"))?;
    let a = (row_offset as usize).checked_mul(row_bytes);
    let b = rows.checked_mul(row_bytes).and_then(|n| a.and_then(|a| a.checked_add(n)));
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) if b <= len => (a, b),
        _ => {
            return Err(value_err(format!(
                "rows [{row_offset}, {row_offset}+{rows}) of {cols}x{nb}-byte elements exceed the {len}-byte buffer"
            )))
        }
    };
    let arr = PyArray2::<i64>::zeros(py, [rows, cols], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let out = unsafe { arr.as_slice_mut() }.map_err(|e| value_err(e.to_string()))?;
    // SAFETY: `pb` holds a buffer export (the object cannot be resized or freed while it lives) of
    // `len` contiguous bytes; `a..b` is within it.
    let src: &[u8] = unsafe { std::slice::from_raw_parts(pb.buf_ptr().cast::<u8>(), len) };
    let src = &src[a..b];
    py.allow_threads(|| {
        for_each_part(out, 8, UNPACK_PAR_MIN_BYTES, |first, part| {
            decode(&src[first * nb..(first + part.len()) * nb], part, nb)
        });
    });
    drop(pb);
    Ok(arr)
}

/// [`unpack_rows`] as uint32 (`K <= 32`): half the memory of the int64 form, same values. The
/// gateway's fused finish (`finish_group`) reads `W r` in this form.
#[pyfunction]
#[pyo3(signature = (buf, row_offset, rows, cols, k))]
fn unpack_rows_u32<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyAny>,
    row_offset: i64,
    rows: i64,
    cols: i64,
    k: u32,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    if k % 8 != 0 || !(8..=32).contains(&k) {
        return Err(value_err(format!("bad ring width {k} for uint32 rows")));
    }
    if row_offset < 0 || rows < 0 || cols < 0 {
        return Err(value_err("row_offset, rows and cols must be >= 0"));
    }
    let nb = (k / 8) as usize;
    let (rows, cols) = (rows as usize, cols as usize);
    let pb = PyBuffer::<u8>::get(buf)?;
    if !pb.is_c_contiguous() {
        return Err(value_err("buffer must be C-contiguous"));
    }
    let len = pb.len_bytes();
    let row_bytes = cols.checked_mul(nb).ok_or_else(|| value_err("row size overflows"))?;
    let a = (row_offset as usize).checked_mul(row_bytes);
    let b = rows.checked_mul(row_bytes).and_then(|n| a.and_then(|a| a.checked_add(n)));
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) if b <= len => (a, b),
        _ => {
            return Err(value_err(format!(
                "rows [{row_offset}, {row_offset}+{rows}) of {cols}x{nb}-byte elements exceed the {len}-byte buffer"
            )))
        }
    };
    let arr = PyArray2::<u32>::zeros(py, [rows, cols], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let out = unsafe { arr.as_slice_mut() }.map_err(|e| value_err(e.to_string()))?;
    // SAFETY: `pb` holds a buffer export of `len` contiguous bytes; `a..b` is within it.
    let src: &[u8] = unsafe { std::slice::from_raw_parts(pb.buf_ptr().cast::<u8>(), len) };
    let src = &src[a..b];
    py.allow_threads(|| match nb {
        4 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        _ => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(nb)) {
                let mut w = [0u8; 4];
                w[..nb].copy_from_slice(c);
                *o = u32::from_le_bytes(w);
            }
        }
    });
    drop(pb);
    Ok(arr)
}

/// One job of [`claim_rows_many`], validated, with raw pointers into Python-owned memory.
struct ClaimJob {
    key: Zeroizing<[u8; 32]>,
    off: u64,
    r: *mut i64,
    r_len: usize,
    src: *const u8,
    src_len: usize,
    wr: *mut u32,
    wr_len: usize,
}

// SAFETY: the pointers refer to distinct freshly allocated output arrays (written by exactly one
// job) and to read-only buffer exports held by the caller for the duration of the parallel section.
unsafe impl Send for ClaimJob {}
unsafe impl Sync for ClaimJob {}

fn run_claim(job: &ClaimJob, k: u32, nb: usize) -> Result<(), String> {
    // SAFETY: see `ClaimJob`; each job's outputs are only touched by the thread running it.
    let r = unsafe { std::slice::from_raw_parts_mut(job.r, job.r_len) };
    fill_ring(&job.key, job.off, r, k)?;
    let src = unsafe { std::slice::from_raw_parts(job.src, job.src_len) };
    let wr = unsafe { std::slice::from_raw_parts_mut(job.wr, job.wr_len) };
    if nb == 4 {
        for (o, c) in wr.iter_mut().zip(src.chunks_exact(4)) {
            *o = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    } else {
        for (o, c) in wr.iter_mut().zip(src.chunks_exact(nb)) {
            let mut w = [0u8; 4];
            w[..nb].copy_from_slice(c);
            *o = u32::from_le_bytes(w);
        }
    }
    Ok(())
}

/// Batched mask fetch for many sessions/linears in one call (`pgw.maskstore.store.take_checked_many`).
///
/// `jobs`: `(key, cols, row_start, buf, row_offset, rows, out_cols)` tuples. Job `i` yields
/// `(r, wr)`: `r = mask_rows_with_key(key, cols, k, row_start, rows)` (int64) and
/// `wr = unpack_rows_u32(buf, row_offset, rows, out_cols, k)` (uint32, `k <= 32`), bit-identical to
/// those functions. Jobs run on up to `threads` threads (0: automatic) with the GIL released; the
/// caller must keep every `buf` unchanged until the call returns.
#[pyfunction]
#[pyo3(signature = (jobs, k, threads=0))]
fn claim_rows_many<'py>(
    py: Python<'py>,
    jobs: &Bound<'py, PyAny>,
    k: u32,
    threads: usize,
) -> PyResult<Vec<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<u32>>)>> {
    if k % 8 != 0 || !(8..=32).contains(&k) {
        return Err(value_err(format!("bad ring width {k} for uint32 rows")));
    }
    let nb = (k / 8) as usize;
    let mut outs = Vec::new();
    let mut work: Vec<ClaimJob> = Vec::new();
    let mut exports: Vec<PyBuffer<u8>> = Vec::new();
    let mut total = 0usize;
    for item in jobs.try_iter()? {
        let item = item?;
        let (key_obj, cols, row_start, buf, row_offset, rows, out_cols): (
            Bound<'py, PyAny>,
            i64,
            i64,
            Bound<'py, PyAny>,
            i64,
            i64,
            i64,
        ) = item.extract()?;
        let raw = secret_bytes(&key_obj)?;
        let key: Zeroizing<[u8; 32]> =
            Zeroizing::new(raw.as_slice().try_into().map_err(|_| value_err("key must be exactly 32 bytes"))?);
        let (rows_u, cols_u, off) = row_range(cols, row_start, rows)?;
        if row_offset < 0 || out_cols < 0 {
            return Err(value_err("row_offset and out_cols must be >= 0"));
        }
        let out_cols_u = out_cols as usize;
        let pb = PyBuffer::<u8>::get(&buf)?;
        if !pb.is_c_contiguous() {
            return Err(value_err("buffer must be C-contiguous"));
        }
        let len = pb.len_bytes();
        let row_bytes = out_cols_u.checked_mul(nb).ok_or_else(|| value_err("row size overflows"))?;
        let a = (row_offset as usize).checked_mul(row_bytes);
        let b = rows_u.checked_mul(row_bytes).and_then(|n| a.and_then(|a| a.checked_add(n)));
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) if b <= len => (a, b),
            _ => return Err(value_err(format!("rows [{row_offset}, {row_offset}+{rows}) exceed the {len}-byte buffer"))),
        };
        let r_arr = PyArray2::<i64>::zeros(py, [rows_u, cols_u], false);
        let wr_arr = PyArray2::<u32>::zeros(py, [rows_u, out_cols_u], false);
        // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
        let r_sl = unsafe { r_arr.as_slice_mut() }.map_err(|e| value_err(e.to_string()))?;
        let wr_sl = unsafe { wr_arr.as_slice_mut() }.map_err(|e| value_err(e.to_string()))?;
        total += r_sl.len() * 8 + wr_sl.len() * 4;
        work.push(ClaimJob {
            key,
            off,
            r: r_sl.as_mut_ptr(),
            r_len: r_sl.len(),
            // SAFETY: `pb` (kept in `exports`) holds the buffer export; `a..b` is within it.
            src: unsafe { pb.buf_ptr().cast::<u8>().add(a) },
            src_len: b - a,
            wr: wr_sl.as_mut_ptr(),
            wr_len: wr_sl.len(),
        });
        exports.push(pb);
        outs.push((r_arr, wr_arr));
    }
    let n = work.len();
    let cpus = std::thread::available_parallelism().map_or(1, |c| c.get());
    let want = if threads == 0 { cpus } else { threads };
    let t = want.min(n).min(total / PAR_BYTES_PER_THREAD + 1).max(1);
    let res: Result<(), String> = py.allow_threads(|| {
        if t <= 1 {
            for j in &work {
                run_claim(j, k, nb)?;
            }
            return Ok(());
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        let body = || loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= n {
                break;
            }
            if let Err(e) = run_claim(&work[i], k, nb) {
                *err.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
            }
        };
        std::thread::scope(|s| {
            for _ in 1..t {
                s.spawn(body);
            }
            body();
        });
        match err.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    });
    drop(exports);
    res.map_err(value_err)?;
    Ok(outs)
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(claim_rows_many, m)?)?;
    m.add_function(wrap_pyfunction!(mask_rows, m)?)?;
    m.add_function(wrap_pyfunction!(mask_key, m)?)?;
    m.add_function(wrap_pyfunction!(mask_rows_with_key, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_rows, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_rows_u32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Vec<u8> {
        (0u8..32).collect()
    }

    #[test]
    fn fill_ring_matches_core() {
        let sx = SeedExpander::new(&seed()).unwrap();
        let labels = vec![Label::from("r"), Label::from("3.qkv")];
        let key = sx.key(&labels).unwrap();
        // includes a parallel case (> PAR_MIN_BYTES) and offsets past 4 GB
        for (cols, start, rows, k) in [
            (1usize, 0u64, 5usize, 40u32),
            (7, 3, 9, 24),
            (3584, 1, 1, 32),
            (18944, 12345, 40, 56),
            (3584, (5u64 << 32) / (8 * 3584), 3, 16),
            (5, 1, 0, 64),
        ] {
            let want = sx.ring_uniform_rows(cols, k, start, rows, &labels).unwrap();
            let mut got = vec![0i64; cols * rows];
            fill_ring(&key, cols as u64 * start, &mut got, k).unwrap();
            assert!(want.iter().zip(&got).all(|(&w, &g)| w as i64 == g), "cols={cols} start={start}");
        }
    }

    #[test]
    fn fill_ring_last_block() {
        let key = [9u8; 32];
        let last = LAST_BLOCK_START / 8; // first element of the last block
        let mut a = vec![0i64; 16]; // the last two blocks: head + raw-core tail
        fill_ring(&key, last - 8, &mut a, 64).unwrap();
        let mut b = vec![0i64; 8]; // the last block alone
        fill_ring(&key, last, &mut b, 64).unwrap();
        assert_eq!(a[8..], b[..]);
        // the wrapper does emit bytes 8..64 of the last block (seek lands mid-block)
        let mut c = ChaCha20::new((&key).into(), &[0u8; 12].into());
        c.seek(LAST_BLOCK_START + 8);
        let mut tail = [0u8; 56];
        c.apply_keystream(&mut tail);
        let want: Vec<i64> = tail.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap()) as i64).collect();
        assert_eq!(b[1..], want[..]);
        // and the block before it agrees with the wrapper
        let mut prev = [0u8; 64];
        let mut c = ChaCha20::new((&key).into(), &[0u8; 12].into());
        c.seek(LAST_BLOCK_START - 64);
        c.apply_keystream(&mut prev);
        let want: Vec<i64> = prev.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap()) as i64).collect();
        assert_eq!(a[..8], want[..]);
    }

    #[test]
    fn fill_ring_range_errors() {
        let key = [7u8; 32];
        let mut one = [0i64; 1];
        assert!(fill_ring(&key, MAX_STREAM_BYTES / 8 - 1, &mut one, 40).is_ok());
        assert!(fill_ring(&key, MAX_STREAM_BYTES / 8, &mut one, 40).is_err());
        assert!(fill_ring(&key, u64::MAX, &mut one, 40).is_err());
        assert!(fill_ring(&key, 0, &mut one, 0).is_err());
        assert!(fill_ring(&key, 0, &mut one, 65).is_err());
    }

    #[test]
    fn decode_zero_extends() {
        let src: Vec<u8> = (1u8..=40).collect();
        for nb in 1..=8usize {
            let n = src.len() / nb;
            let mut out = vec![0i64; n];
            decode(&src[..n * nb], &mut out, nb);
            for (i, &v) in out.iter().enumerate() {
                let mut w = [0u8; 8];
                w[..nb].copy_from_slice(&src[i * nb..(i + 1) * nb]);
                assert_eq!(v, u64::from_le_bytes(w) as i64);
            }
        }
    }

    #[test]
    fn parts_cover_everything() {
        let mut v = vec![0usize; 1 << 18];
        for_each_part(&mut v, 8, PAR_MIN_BYTES, |first, part| {
            for (i, x) in part.iter_mut().enumerate() {
                *x = first + i;
            }
        });
        assert!(v.iter().enumerate().all(|(i, &x)| i == x));
    }
}
