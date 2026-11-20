//! Segment writer of `pgw.maskstore` (factory side): pack ring rows, AES-256-GCM, `pwrite`.
//! Functions are registered in [`register`] (Unix only: the Python side falls back elsewhere).
//!
//! * [`seal_chunks`] = `RawSegmentWriter.write_chunk` for already packed plaintext: for every job
//!   `(index, offset, start, len)` it encrypts `data[start:start+len]` and writes
//!   `ciphertext || tag` at file offset `offset`.
//! * [`seal_ring_chunks`] = `pack_ring(values[a:b], k)` followed by the same, for row ranges of a
//!   2-D ring matrix (`SegmentWriter.write_linear`): no packed copy, no Python AES, no `bytearray`
//!   bookkeeping. It first checks that *every* element of `values` lies in `[0, 2^k)` and writes
//!   nothing (returns `False`) if one does not, like `SegmentWriter._check_tensor`.
//!
//! The chunk format is SPEC 7.x / `pgw.maskstore.format`, byte for byte:
//! AES-256-GCM with the segment DEK, nonce `chunk_nonce(prefix, i) = prefix(4) || u64be(i)`,
//! AAD `chunk_aad(digest, i) = SHA-256(header JSON) || u32be(i)`, 16-byte tag after the ciphertext.
//! Plaintext only ever lives in per-thread scratch buffers that are zeroized on drop; the key is
//! held in zeroizing memory and the cipher state (`aes-gcm` with `zeroize`) is wiped on drop.
//! Work is spread over threads (one chunk at a time per thread) with the GIL released.

use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use aes_gcm::aead::AeadInOut;
use aes_gcm::{Aes256Gcm, Key, KeyInit, Nonce};
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use zeroize::Zeroizing;

/// GCM tag bytes (`format.GCM_TAG_BYTES`).
pub const TAG_BYTES: usize = 16;
/// `NONCE_PREFIX_BYTES`.
pub const NONCE_PREFIX_BYTES: usize = 4;
/// SHA-256 digest bytes.
pub const DIGEST_BYTES: usize = 32;
/// Below this much plaintext per thread an extra thread costs more than it saves.
const PAR_MIN_BYTES_PER_THREAD: usize = 1 << 20;
/// Upper bound of the automatic thread count.
const MAX_AUTO_THREADS: usize = 8;

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Error inside the GIL-free part.
#[derive(Debug)]
pub enum WriteError {
    Io(io::Error),
    Crypto,
}

impl From<WriteError> for PyErr {
    fn from(e: WriteError) -> Self {
        match e {
            WriteError::Io(e) => e.into(),
            WriteError::Crypto => value_err("AES-GCM encryption failed (chunk too large)"),
        }
    }
}

/// One chunk: index (nonce/AAD), file offset of the ciphertext, plaintext source range (bytes for
/// [`seal_chunks`], rows for [`seal_ring_chunks`]).
#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub index: u64,
    pub offset: u64,
    pub start: usize,
    pub len: usize,
}

/// AES-256-GCM with the segment's nonce and AAD scheme.
pub struct Sealer {
    aead: Aes256Gcm,
    prefix: [u8; NONCE_PREFIX_BYTES],
    digest: [u8; DIGEST_BYTES],
}

impl Sealer {
    pub fn new(key: &[u8; 32], prefix: [u8; NONCE_PREFIX_BYTES], digest: [u8; DIGEST_BYTES]) -> Self {
        // Borrow the key as `&Key` (a reference cast, no copy) so no unzeroized stack copy of the
        // DEK is left behind; the caller holds it in `Zeroizing`.
        let key: &Key<Aes256Gcm> = key.into();
        Self { aead: Aes256Gcm::new(key), prefix, digest }
    }

    /// `chunk_nonce(prefix, index)`.
    pub fn nonce(&self, index: u64) -> [u8; 12] {
        let mut n = [0u8; 12];
        n[..4].copy_from_slice(&self.prefix);
        n[4..].copy_from_slice(&index.to_be_bytes());
        n
    }

    /// `chunk_aad(digest, index)`; `index` must fit u32 (checked by the callers).
    pub fn aad(&self, index: u32) -> [u8; DIGEST_BYTES + 4] {
        let mut a = [0u8; DIGEST_BYTES + 4];
        a[..DIGEST_BYTES].copy_from_slice(&self.digest);
        a[DIGEST_BYTES..].copy_from_slice(&index.to_be_bytes());
        a
    }

    /// Encrypt `buf[..len - 16]` in place and put the tag into the last 16 bytes.
    pub fn seal_in_place(&self, index: u32, buf: &mut [u8]) -> Result<(), WriteError> {
        let n = buf.len() - TAG_BYTES;
        let (plain, tag_out) = buf.split_at_mut(n);
        let tag = self
            .aead
            .encrypt_inout_detached(&Nonce::from(self.nonce(u64::from(index))), &self.aad(index), plain.into())
            .map_err(|_| WriteError::Crypto)?;
        tag_out.copy_from_slice(&tag);
        Ok(())
    }
}

/// `pwrite` all of `buf` at `off` on a borrowed descriptor (not closed here).
fn write_all_at(fd: i32, buf: &[u8], off: u64) -> io::Result<()> {
    // SAFETY: `fd` is a valid open descriptor owned by the caller for the whole call (checked >= 0
    // at the boundary); `ManuallyDrop` keeps it open.
    let f = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
    f.write_all_at(buf, off)
}

/// Thread count for `jobs` with `total` plaintext bytes (`requested == 0`: automatic).
fn thread_count(requested: usize, jobs: usize, total: usize) -> usize {
    let t = if requested == 0 {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        cpus.min(MAX_AUTO_THREADS).min(total / PAR_MIN_BYTES_PER_THREAD).max(1)
    } else {
        requested
    };
    t.min(jobs).max(1)
}

/// Run `fill(job, plaintext_buf)` then seal and write every job, over `threads` threads; each
/// thread owns one zeroizing scratch buffer of `scratch_len` bytes. Stops at the first error.
pub fn run_jobs(
    fd: i32,
    sealer: &Sealer,
    jobs: &[Job],
    plain_len: impl Fn(&Job) -> usize + Sync,
    fill: impl Fn(&Job, &mut [u8]) + Sync,
    threads: usize,
) -> Result<(), WriteError> {
    let total: usize = jobs.iter().map(&plain_len).sum();
    let scratch_len = jobs.iter().map(&plain_len).max().unwrap_or(0) + TAG_BYTES;
    let threads = thread_count(threads, jobs.len(), total);
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let first_err: Mutex<Option<WriteError>> = Mutex::new(None);
    let worker = || {
        let mut scratch = Zeroizing::new(vec![0u8; scratch_len]);
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = jobs.get(i) else { return };
            let n = plain_len(job) + TAG_BYTES;
            let buf = &mut scratch[..n];
            fill(job, &mut buf[..n - TAG_BYTES]);
            let r = sealer
                .seal_in_place(job.index as u32, buf)
                .and_then(|()| write_all_at(fd, buf, job.offset).map_err(WriteError::Io));
            if let Err(e) = r {
                stop.store(true, Ordering::Relaxed);
                let mut g = first_err.lock().unwrap_or_else(|p| p.into_inner());
                g.get_or_insert(e);
                return;
            }
        }
    };
    if threads <= 1 {
        worker();
    } else {
        std::thread::scope(|s| {
            let w = &worker;
            for _ in 1..threads {
                s.spawn(w);
            }
            worker();
        });
    }
    match first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ------------------------------------------------------------------------------------------------
// ring rows: range check and packing
// ------------------------------------------------------------------------------------------------

/// A ring element type accepted by [`seal_ring_chunks`].
pub trait RingElem: Copy + Send + Sync {
    /// Bits that must be zero for a canonical value under `mask` (sign bit included for signed).
    fn bad_bits(self, mask: u64) -> u64;
    /// The value as u64 (only called on checked values).
    fn bits(self) -> u64;
}

impl RingElem for i64 {
    #[inline(always)]
    fn bad_bits(self, mask: u64) -> u64 {
        (self as u64) & (!mask | (1u64 << 63))
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        self as u64
    }
}

impl RingElem for u64 {
    #[inline(always)]
    fn bad_bits(self, mask: u64) -> u64 {
        self & !mask
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        self
    }
}

/// `int32` holds the low 32 bits of the element (compact GPU form, `K <= 32`), like `pack_ring`.
impl RingElem for i32 {
    #[inline(always)]
    fn bad_bits(self, mask: u64) -> u64 {
        u64::from(self as u32) & !mask
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        u64::from(self as u32)
    }
}

impl RingElem for u32 {
    #[inline(always)]
    fn bad_bits(self, mask: u64) -> u64 {
        u64::from(self) & !mask
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        u64::from(self)
    }
}

/// Ring mask for `k` (a multiple of 8 in `8..=64`).
pub fn ring_mask(k: u32) -> u64 {
    if k >= 64 {
        u64::MAX
    } else {
        (1u64 << k) - 1
    }
}

/// True if every element lies in `[0, 2^k)` (parallel for large inputs).
pub fn all_in_range<T: RingElem>(vals: &[T], k: u32, threads: usize) -> bool {
    let mask = ring_mask(k);
    let check = |s: &[T]| s.iter().fold(0u64, |a, &v| a | v.bad_bits(mask)) == 0;
    let bytes = std::mem::size_of_val(vals);
    let t = thread_count(threads, vals.len().max(1), bytes / 4); // memory-bound: 4 MB per thread
    if t <= 1 {
        return check(vals);
    }
    let per = vals.len().div_ceil(t);
    std::thread::scope(|s| {
        let hs: Vec<_> = vals.chunks(per).skip(1).map(|c| s.spawn(move || check(c))).collect();
        let first = check(&vals[..per.min(vals.len())]);
        hs.into_iter().fold(first, |a, h| h.join().unwrap_or(false) && a)
    })
}

#[inline(always)]
fn pack_n<T: RingElem, const N: usize>(src: &[T], out: &mut [u8]) {
    for (o, &v) in out.chunks_exact_mut(N).zip(src) {
        o.copy_from_slice(&v.bits().to_le_bytes()[..N]);
    }
}

/// `pack_ring(src, k)` for values already known to be in range: `k/8` little-endian bytes each.
pub fn pack<T: RingElem>(src: &[T], out: &mut [u8], nb: usize) {
    debug_assert_eq!(out.len(), src.len() * nb);
    match nb {
        1 => pack_n::<T, 1>(src, out),
        2 => pack_n::<T, 2>(src, out),
        3 => pack_n::<T, 3>(src, out),
        4 => {
            for (o, &v) in out.chunks_exact_mut(4).zip(src) {
                o.copy_from_slice(&(v.bits() as u32).to_le_bytes());
            }
        }
        5 => pack_n::<T, 5>(src, out),
        6 => pack_n::<T, 6>(src, out),
        7 => pack_n::<T, 7>(src, out),
        8 => pack_n::<T, 8>(src, out),
        _ => unreachable!("nb validated to 1..=8"),
    }
}

/// Range check of all of `vals`, then pack + seal + write the row-range jobs. `Ok(false)`: out of
/// range, nothing written.
pub fn seal_rows<T: RingElem>(
    fd: i32,
    sealer: &Sealer,
    vals: &[T],
    cols: usize,
    k: u32,
    jobs: &[Job],
    threads: usize,
) -> Result<bool, WriteError> {
    if !all_in_range(vals, k, threads) {
        return Ok(false);
    }
    let nb = (k / 8) as usize;
    run_jobs(
        fd,
        sealer,
        jobs,
        |j| j.len * cols * nb,
        |j, out| pack(&vals[j.start * cols..(j.start + j.len) * cols], out, nb),
        threads,
    )?;
    Ok(true)
}

// ------------------------------------------------------------------------------------------------
// Python boundary
// ------------------------------------------------------------------------------------------------

fn key32(obj: &Bound<'_, PyAny>) -> PyResult<Zeroizing<[u8; 32]>> {
    let mut k = Zeroizing::new([0u8; 32]);
    let wrong = || value_err("key must be exactly 32 bytes");
    if let Ok(b) = obj.downcast::<PyBytes>() {
        k.copy_from_slice(b.as_bytes().get(..32).filter(|_| b.as_bytes().len() == 32).ok_or_else(wrong)?);
        return Ok(k);
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        // SAFETY: copied immediately while holding the GIL; no Python code runs in between.
        let s = unsafe { b.as_bytes() };
        k.copy_from_slice(s.get(..32).filter(|_| s.len() == 32).ok_or_else(wrong)?);
        return Ok(k);
    }
    let buf = PyBuffer::<u8>::get(obj).map_err(|_| PyTypeError::new_err("key must be bytes-like"))?;
    let v = Zeroizing::new(buf.to_vec(obj.py())?);
    k.copy_from_slice(v.get(..32).filter(|_| v.len() == 32).ok_or_else(wrong)?);
    Ok(k)
}

fn sealer(key: &Bound<'_, PyAny>, nonce_prefix: &[u8], header_digest: &[u8]) -> PyResult<Sealer> {
    let key = key32(key)?;
    let prefix: [u8; NONCE_PREFIX_BYTES] =
        nonce_prefix.try_into().map_err(|_| value_err("nonce_prefix must be exactly 4 bytes"))?;
    let digest: [u8; DIGEST_BYTES] =
        header_digest.try_into().map_err(|_| value_err("header_digest must be exactly 32 bytes"))?;
    Ok(Sealer::new(&key, prefix, digest))
}

/// Validated jobs: chunk index fits the u32 AAD field, source ranges within `limit` units.
fn to_jobs(raw: Vec<(u64, u64, u64, u64)>, limit: usize, unit: &str) -> PyResult<Vec<Job>> {
    raw.into_iter()
        .map(|(index, offset, start, len)| {
            if index > u64::from(u32::MAX) {
                return Err(value_err(format!("chunk index {index} does not fit the 32-bit AAD field")));
            }
            if offset > i64::MAX as u64 {
                return Err(value_err("file offset out of range"));
            }
            match start.checked_add(len) {
                Some(end) if end <= limit as u64 => {}
                _ => return Err(value_err(format!("chunk {index}: {unit} [{start}, {start}+{len}) exceed {limit}"))),
            }
            Ok(Job { index, offset, start: start as usize, len: len as usize })
        })
        .collect()
}

fn check_fd(fd: i32) -> PyResult<()> {
    if fd < 0 {
        return Err(value_err("fd must be an open file descriptor"));
    }
    Ok(())
}

/// Encrypt and write already packed chunks: for every `(index, offset, start, len)` in `jobs`,
/// `pwrite(fd, AESGCM(key).encrypt(chunk_nonce(nonce_prefix, index), data[start:start+len],
/// chunk_aad(header_digest, index)), offset)`. Byte-identical to `RawSegmentWriter.write_chunk`.
#[pyfunction]
#[pyo3(signature = (fd, key, nonce_prefix, header_digest, data, jobs, threads=0))]
#[allow(clippy::too_many_arguments)]
fn seal_chunks(
    py: Python<'_>,
    fd: i32,
    key: &Bound<'_, PyAny>,
    nonce_prefix: &[u8],
    header_digest: &[u8],
    data: &Bound<'_, PyAny>,
    jobs: Vec<(u64, u64, u64, u64)>,
    threads: usize,
) -> PyResult<()> {
    check_fd(fd)?;
    let sealer = sealer(key, nonce_prefix, header_digest)?;
    let pb =
        PyBuffer::<u8>::get(data).map_err(|_| PyTypeError::new_err("data must be a bytes-like object of bytes"))?;
    if !pb.is_c_contiguous() {
        return Err(value_err("data must be C-contiguous"));
    }
    let len = pb.len_bytes();
    let jobs = to_jobs(jobs, len, "bytes")?;
    // SAFETY: `pb` holds a buffer export (the object cannot be resized or freed while it lives) of
    // `len` contiguous bytes.
    let src: &[u8] = unsafe { std::slice::from_raw_parts(pb.buf_ptr().cast::<u8>(), len) };
    py.allow_threads(|| {
        run_jobs(fd, &sealer, &jobs, |j| j.len, |j, out| out.copy_from_slice(&src[j.start..j.start + j.len]), threads)
    })?;
    drop(pb);
    Ok(())
}

/// Range-check, pack, encrypt and write row ranges of a ring matrix.
///
/// `values` is a C-contiguous 2-D int64 / uint64 array (or int32 / uint32 holding the low 32 bits,
/// `k <= 32`) of shape `(rows, cols)`. Every element must lie in `[0, 2^k)`; otherwise nothing is
/// written and the result is `False`. Then for every `(index, offset, row_start, row_count)` in
/// `jobs`, the chunk `pack_ring(values[row_start:row_start+row_count], k)` is sealed and written
/// exactly like [`seal_chunks`]. Returns `True`. `jobs` may be empty (range check only).
#[pyfunction]
#[pyo3(signature = (fd, key, nonce_prefix, header_digest, values, k, jobs, threads=0))]
#[allow(clippy::too_many_arguments)]
fn seal_ring_chunks(
    py: Python<'_>,
    fd: i32,
    key: &Bound<'_, PyAny>,
    nonce_prefix: &[u8],
    header_digest: &[u8],
    values: &Bound<'_, PyAny>,
    k: u32,
    jobs: Vec<(u64, u64, u64, u64)>,
    threads: usize,
) -> PyResult<bool> {
    check_fd(fd)?;
    if !k.is_multiple_of(8) || !(8..=64).contains(&k) {
        return Err(value_err(format!("bad ring width {k}")));
    }
    let sealer = sealer(key, nonce_prefix, header_digest)?;
    let cerr = || value_err("values must be C-contiguous");
    let narrow = |what: &str| -> PyResult<()> {
        if k > 32 {
            return Err(value_err(format!("{what} values need K <= 32")));
        }
        Ok(())
    };
    macro_rules! go {
        ($a:expr) => {{
            let ro = $a.try_readonly().map_err(|e| value_err(format!("values: {e}")))?;
            let (rows, cols) = (ro.shape()[0], ro.shape()[1]);
            let jobs = to_jobs(jobs, rows, "rows")?;
            let vals = ro.as_slice().map_err(|_| cerr())?;
            let ok = py.allow_threads(|| seal_rows(fd, &sealer, vals, cols, k, &jobs, threads))?;
            drop(ro);
            return Ok(ok);
        }};
    }
    if let Ok(a) = values.downcast::<PyArray2<i64>>() {
        go!(a)
    }
    if let Ok(a) = values.downcast::<PyArray2<u64>>() {
        go!(a)
    }
    if let Ok(a) = values.downcast::<PyArray2<i32>>() {
        narrow("int32")?;
        go!(a)
    }
    if let Ok(a) = values.downcast::<PyArray2<u32>>() {
        narrow("uint32")?;
        go!(a)
    }
    Err(PyTypeError::new_err("values must be a 2-D int64/uint64/int32/uint32 numpy array"))
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seal_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(seal_ring_chunks, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::aead::Aead;
    use aes_gcm::aead::Payload;

    fn sealer() -> Sealer {
        let key: [u8; 32] = core::array::from_fn(|i| i as u8);
        Sealer::new(&key, [1, 2, 3, 4], [7u8; 32])
    }

    #[test]
    fn seal_matches_aead_api() {
        let s = sealer();
        let key: [u8; 32] = core::array::from_fn(|i| i as u8);
        let plain: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let mut buf = plain.clone();
        buf.extend_from_slice(&[0u8; TAG_BYTES]);
        s.seal_in_place(5, &mut buf).unwrap();
        let want = Aes256Gcm::new(&Key::<Aes256Gcm>::from(key))
            .encrypt(&Nonce::from(s.nonce(5)), Payload { msg: &plain, aad: &s.aad(5) })
            .unwrap();
        assert_eq!(buf, want);
        assert_eq!(&s.nonce(5)[..], &[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(&s.aad(0x0102_0304)[32..], &[1, 2, 3, 4]);
    }

    #[test]
    fn range_check() {
        assert!(all_in_range(&[0i64, 5, (1 << 32) - 1], 32, 1));
        assert!(!all_in_range(&[0i64, 1 << 32], 32, 1));
        assert!(!all_in_range(&[0i64, -1], 32, 1));
        assert!(!all_in_range(&[-1i64], 64, 1));
        assert!(all_in_range(&[i64::MAX], 64, 1));
        assert!(all_in_range(&[u64::MAX], 64, 1));
        assert!(all_in_range(&[-1i32], 32, 1)); // low 32 bits: 2^32 - 1
        assert!(!all_in_range(&[256u32], 8, 1));
        let mut big = vec![3i64; 3 << 20];
        assert!(all_in_range(&big, 8, 0));
        *big.last_mut().unwrap() = 256;
        assert!(!all_in_range(&big, 8, 4));
    }

    #[test]
    fn pack_little_endian() {
        let v = [0x0102_0304_0506_0708i64, 0x0a0b_0c0d_0e0f_1011];
        for nb in 1..=8usize {
            let mut out = vec![0u8; 2 * nb];
            pack(&v, &mut out, nb);
            for (i, &x) in v.iter().enumerate() {
                assert_eq!(&out[i * nb..(i + 1) * nb], &x.to_le_bytes()[..nb]);
            }
        }
    }
}
