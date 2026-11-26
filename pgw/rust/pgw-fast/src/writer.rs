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
//! held in zeroizing memory and the cipher state is wiped on drop (below).
//! Work is spread over threads (one chunk at a time per thread) with the GIL released.
//!
//! # AES-256-GCM implementation: AWS-LC (`aws-lc-rs`)
//!
//! Since 2026-11-26 the cipher is AWS-LC's `EVP_aead_aes_256_gcm` through `aws_lc_rs::aead::LessSafeKey`
//! (before: RustCrypto `aes-gcm` 0.11). AES-256-GCM with a given key, nonce, AAD and plaintext has exactly
//! one output, so the at-rest bytes do not change: the tests below seal and open across both
//! implementations, `tests/native/test_writer.py` compares with `cryptography`'s AESGCM and
//! `tests/native/test_aead_compat.py` with a file written by the RustCrypto build. AWS-LC picks at run time its stitched VAES + VPCLMULQDQ
//! (AVX-512 / AVX2), AES-NI + PCLMULQDQ or ARMv8 AES + PMULL code; one core here: 13 GB/s against 2.8.
//!
//! Wipe on drop (verified in aws-lc-sys 0.47.0 / aws-lc-rs 1.19.0 sources):
//! * The round keys and the GHASH key table (`struct aead_aes_gcm_ctx`: `ks`, `gcm_key`) live inside the
//!   `EVP_AEAD_CTX` (`state`), which aws-lc-rs allocates with `OPENSSL_malloc` (`aead/aead_ctx.rs`,
//!   `build_context`) and owns as `LcPtr<EVP_AEAD_CTX>`, whose `Drop` calls `EVP_AEAD_CTX_free`
//!   (`ptr.rs`). That calls the AEAD's `cleanup` and `OPENSSL_free(ctx)` (`crypto/fipsmodule/cipher/
//!   aead.c`), and `OPENSSL_free` runs `OPENSSL_cleanse(ptr, size + OPENSSL_MALLOC_PREFIX)` over the
//!   whole allocation before `free` (`crypto/mem.c`; this crate installs no AWS-LC memory hooks, the only
//!   path that skips it). Tests `aws_lc_free_cleanses_the_key_schedule` and
//!   `dropped_key_leaves_no_key_schedule` read the freed context back.
//! * The DEK passed in is borrowed (no copy here; callers hold it in `Zeroizing`).
//! * Per call AWS-LC copies the GHASH key table and the per-message `E_K(J0)` into a stack
//!   `GCM128_CONTEXT` that it does not clear (RustCrypto's per-message GHASH state was likewise a stack
//!   value). The round keys themselves are only read through a pointer. Every thread that ran the cipher
//!   overwrites the stack below its frame with zeros when it is done ([`scrub_stack`]), so no such copy
//!   outlives the `seal_*` / `chunk_open_into` call that made it.
//!
//! Constant time: AES and GHASH run on the AES-NI / VAES / ARMv8 AES and carry-less multiply instructions
//! (without them, on AWS-LC's constant-time vpaes / bitsliced AES and `gcm_nohw`; no table lookups), the
//! tag is compared with `CRYPTO_memcmp`, and a failed open zeroes the output here.

use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use aws_lc_rs::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
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

/// Stack bytes [`scrub_stack`] overwrites: far more than the deepest AWS-LC seal / open frame (its
/// `GCM128_CONTEXT` is under 1 KiB, the assembly keeps its state in registers).
const SCRUB_STACK_BYTES: usize = 16 << 10;

/// Overwrite `SCRUB_STACK_BYTES` of the stack below the caller's frame with zeros: the region the
/// cipher calls just made used, which may still hold copies of the GHASH key table and `E_K(J0)`
/// (module docs). Call it from the frame that called the cipher, after the last call.
#[inline(never)]
pub fn scrub_stack() {
    let mut buf = [0u8; SCRUB_STACK_BYTES];
    zeroize::Zeroize::zeroize(&mut buf[..]); // volatile writes: not optimised away
    std::hint::black_box(&buf);
}

/// The AES-256-GCM key schedule (AWS-LC context; wiped on drop, module docs).
pub fn aead_key(key: &[u8; 32]) -> LessSafeKey {
    // `UnboundKey::new` only fails for a key of the wrong length (32 is checked by the type). The key
    // is borrowed: no copy of the DEK is made here.
    LessSafeKey::new(UnboundKey::new(&AES_256_GCM, key).expect("AES-256 key is 32 bytes"))
}

/// A chunk failed authentication (corrupt or tampered ciphertext, tag, nonce, AAD or key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthFailed;

/// Authenticate and decrypt `buf` (ciphertext without the tag) in place. On failure `buf` is zeroed
/// (no unauthenticated plaintext is left) and [`AuthFailed`] returned.
pub fn open_in_place(
    aead: &LessSafeKey,
    nonce: [u8; 12],
    aad: &[u8],
    tag: &[u8; TAG_BYTES],
    buf: &mut [u8],
) -> Result<(), AuthFailed> {
    let ok = aead.open_in_place_separate_tag(Nonce::assume_unique_for_key(nonce), Aad::from(aad), tag, buf).is_ok();
    if !ok {
        // zero whatever is in the buffer (ciphertext or a partial decryption)
        zeroize::Zeroize::zeroize(buf);
        return Err(AuthFailed);
    }
    Ok(())
}

/// AES-256-GCM with the segment's nonce and AAD scheme.
pub struct Sealer {
    aead: LessSafeKey,
    prefix: [u8; NONCE_PREFIX_BYTES],
    digest: [u8; DIGEST_BYTES],
}

impl Sealer {
    pub fn new(key: &[u8; 32], prefix: [u8; NONCE_PREFIX_BYTES], digest: [u8; DIGEST_BYTES]) -> Self {
        // The key is borrowed (no unzeroized copy of the DEK is left behind); the caller holds it
        // in `Zeroizing`.
        Self { aead: aead_key(key), prefix, digest }
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
            .seal_in_place_separate_tag(
                Nonce::assume_unique_for_key(self.nonce(u64::from(index))),
                Aad::from(&self.aad(index)),
                plain,
            )
            .map_err(|_| WriteError::Crypto)?;
        tag_out.copy_from_slice(tag.as_ref());
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
    let work = || {
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
    // Each thread clears the stack its cipher calls used once it is done (module docs).
    let worker = || {
        work();
        scrub_stack();
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
    /// From this `k` on every value of the type is canonical (`bad_bits` is always 0), so the range
    /// check has nothing to find: `u32` / `i32` (low 32 bits) at `k >= 32`, `u64` at 64; `i64` never
    /// (its negative values are out of range at every `k`).
    const ALWAYS_IN_RANGE_FROM_K: u32;
}

impl RingElem for i64 {
    const ALWAYS_IN_RANGE_FROM_K: u32 = u32::MAX;
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
    const ALWAYS_IN_RANGE_FROM_K: u32 = 64;
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
    const ALWAYS_IN_RANGE_FROM_K: u32 = 32;
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
    const ALWAYS_IN_RANGE_FROM_K: u32 = 32;
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
    if k >= T::ALWAYS_IN_RANGE_FROM_K {
        // e.g. the compact uint32 / int32 `W r` at K = 32: no value can fail, skip the pass over memory
        return true;
    }
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
/// ``chunk_open_into(key, nonce, aad, fd, offset, cipher_len, out)``: read one encrypted segment chunk
/// (``cipher_len`` bytes at ``offset`` of the open descriptor ``fd``: ciphertext then the 16-byte tag),
/// authenticate and decrypt it with AES-256-GCM straight into the ``bytearray`` ``out`` (exactly
/// ``cipher_len - 16`` bytes). The inverse of the writer's sealing (``SegmentFile.decrypt_into``'s
/// bytes); the GIL is released for the read and the cipher, so a helper thread can decrypt ahead
/// while the gateway's main thread runs.
///
/// ``out`` must not be touched by any other thread during the call (the caller's fresh buffer). On any
/// failure (short read, authentication) ``out`` is zeroed and ``ValueError("auth: ...")`` /
/// ``OSError`` is raised: no unauthenticated plaintext is ever left behind.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chunk_open_into(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
    nonce: &[u8],
    aad: &[u8],
    fd: i32,
    offset: u64,
    cipher_len: usize,
    out: &Bound<'_, pyo3::types::PyByteArray>,
) -> PyResult<()> {
    use pyo3::exceptions::{PyOSError, PyValueError};
    let mut kb = zeroize::Zeroizing::new([0u8; 32]);
    {
        let kbytes: zeroize::Zeroizing<Vec<u8>> = if let Ok(b) = key.downcast::<pyo3::types::PyBytes>() {
            zeroize::Zeroizing::new(b.as_bytes().to_vec())
        } else if let Ok(b) = key.downcast::<pyo3::types::PyByteArray>() {
            zeroize::Zeroizing::new(b.to_vec())
        } else {
            return Err(PyValueError::new_err("key must be bytes or bytearray"));
        };
        if kbytes.len() != 32 {
            return Err(PyValueError::new_err("key must be exactly 32 bytes"));
        }
        kb.copy_from_slice(&kbytes);
    }
    if nonce.len() != 12 || fd < 0 || cipher_len < TAG_BYTES || out.len() != cipher_len - TAG_BYTES {
        return Err(PyValueError::new_err(format!(
            "chunk_open_into: bad arguments (nonce {} bytes, fd {fd}, cipher_len {cipher_len}, out {} bytes)",
            nonce.len(),
            out.len()
        )));
    }
    let nonce: [u8; 12] = nonce.try_into().expect("checked length");
    let aad = aad.to_vec();
    let n = cipher_len - TAG_BYTES;
    let ptr = out.data() as usize;
    let res: Result<(), Result<io::Error, AuthFailed>> = py.allow_threads(|| {
        // SAFETY: `out` is a bytearray of exactly `n` bytes that nobody else touches during the call
        // (documented contract); it is not resized while the GIL is released by us.
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, n) };
        // SAFETY: `fd` is a valid open descriptor owned by the caller for the whole call.
        let f = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
        let mut tag = [0u8; TAG_BYTES];
        let r = f.read_exact_at(buf, offset).and_then(|_| f.read_exact_at(&mut tag, offset + n as u64));
        if let Err(e) = r {
            buf.fill(0);
            return Err(Ok(e));
        }
        // The key schedule lives in `aead` (wiped when it drops at the end of the block); the stack the
        // cipher used is cleared after it (module docs).
        let r = {
            let aead = aead_key(&kb);
            open_in_place(&aead, nonce, &aad, &tag, buf).map_err(Err)
        };
        scrub_stack();
        r
    });
    match res {
        Ok(()) => Ok(()),
        Err(Ok(e)) => Err(PyOSError::new_err(format!("chunk read failed: {e}"))),
        Err(Err(AuthFailed)) => Err(PyValueError::new_err("auth: chunk failed authentication (corrupt or tampered)")),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_open_into, m)?)?;
    m.add_function(wrap_pyfunction!(seal_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(seal_ring_chunks, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // RustCrypto AES-GCM: the implementation of this module before 2026-11-26 ("old"), as reference.
    use aes_gcm::aead::{Aead, AeadInOut, Payload};
    use aes_gcm::{Aes256Gcm, Key, KeyInit};

    const KEY: [u8; 32] = [
        0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81, 0x1f, 0x35,
        0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4,
    ];
    /// Plaintext lengths: empty, partial / whole / several blocks, the 4- and 8-block strides of the
    /// assembly, and around 1 MiB (the store's chunk size).
    const LENS: [usize; 17] = [0, 1, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256, 4097, 65_536, (1 << 20) + 5];

    fn sealer() -> Sealer {
        Sealer::new(&KEY, [1, 2, 3, 4], [7u8; 32])
    }

    fn old() -> Aes256Gcm {
        Aes256Gcm::new(&Key::<Aes256Gcm>::from(KEY))
    }

    fn plain(n: usize, seed: u32) -> Vec<u8> {
        (0..n as u32).map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8).collect()
    }

    /// `ciphertext || tag` of `p` under chunk `index`, new implementation.
    fn seal_new(s: &Sealer, index: u32, p: &[u8]) -> Vec<u8> {
        let mut buf = p.to_vec();
        buf.extend_from_slice(&[0u8; TAG_BYTES]);
        s.seal_in_place(index, &mut buf).unwrap();
        buf
    }

    fn open_new(s: &Sealer, index: u32, ct: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        let (c, t) = ct.split_at(ct.len() - TAG_BYTES);
        let mut buf = c.to_vec();
        let tag: [u8; TAG_BYTES] = t.try_into().unwrap();
        match open_in_place(&s.aead, s.nonce(u64::from(index)), &s.aad(index), &tag, &mut buf) {
            Ok(()) => Ok(buf),
            Err(AuthFailed) => Err(buf),
        }
    }

    #[test]
    fn seal_matches_old_implementation() {
        let s = sealer();
        for (i, &n) in LENS.iter().enumerate() {
            let index = [0u32, 1, 5, 0x0102_0304, u32::MAX][i % 5];
            let p = plain(n, i as u32);
            let want = old()
                .encrypt(&aes_gcm::Nonce::from(s.nonce(u64::from(index))), Payload { msg: &p, aad: &s.aad(index) })
                .unwrap();
            assert_eq!(seal_new(&s, index, &p), want, "len {n}");
        }
        assert_eq!(&s.nonce(5)[..], &[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(&s.aad(0x0102_0304)[32..], &[1, 2, 3, 4]);
    }

    /// Test case 14 of the GCM specification (McGrew & Viega): K = 0^256, IV = 0^96, P = 0^128, no AAD.
    #[test]
    fn known_answer() {
        let aead = aead_key(&[0u8; 32]);
        let mut buf = [0u8; 16];
        let tag =
            aead.seal_in_place_separate_tag(Nonce::assume_unique_for_key([0u8; 12]), Aad::empty(), &mut buf).unwrap();
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(hex(&buf), "cea7403d4d606b6e074ec5d3baf39d18");
        assert_eq!(hex(tag.as_ref()), "d0d1c8a799996bf0265b98b5d48ab919");
    }

    #[test]
    fn old_and_new_open_each_others_chunks() {
        let s = sealer();
        for (i, &n) in LENS.iter().enumerate() {
            let index = i as u32 * 977;
            let p = plain(n, 99 + i as u32);
            let nonce = aes_gcm::Nonce::from(s.nonce(u64::from(index)));
            // old -> new
            let ct_old = old().encrypt(&nonce, Payload { msg: &p, aad: &s.aad(index) }).unwrap();
            assert_eq!(open_new(&s, index, &ct_old).unwrap(), p, "old -> new, len {n}");
            // new -> old (detached in-place API, as the old chunk_open_into)
            let ct_new = seal_new(&s, index, &p);
            let (c, t) = ct_new.split_at(n);
            let mut buf = c.to_vec();
            let tag = aes_gcm::Tag::from(<[u8; TAG_BYTES]>::try_from(t).unwrap());
            old().decrypt_inout_detached(&nonce, &s.aad(index), (&mut buf[..]).into(), &tag).unwrap();
            assert_eq!(buf, p, "new -> old, len {n}");
        }
    }

    #[test]
    fn tampering_fails_and_zeroes_the_output() {
        let s = sealer();
        let index = 7u32;
        for &n in &[0usize, 1, 16, 1000, 65_537] {
            let p = plain(n, 3);
            let ct = seal_new(&s, index, &p);
            assert_eq!(open_new(&s, index, &ct).unwrap(), p);
            let mut cases: Vec<(String, Vec<u8>, u32, Sealer)> = Vec::new();
            for &pos in &[0usize, n / 2, n.saturating_sub(1), n, n + 15] {
                if pos < ct.len() {
                    let mut c = ct.clone();
                    c[pos] ^= 0x01;
                    cases.push((format!("bit flip at {pos}"), c, index, sealer()));
                }
            }
            cases.push(("other chunk index (nonce + AAD)".into(), ct.clone(), index + 1, sealer()));
            cases.push(("other nonce prefix".into(), ct.clone(), index, Sealer::new(&KEY, [1, 2, 3, 5], [7u8; 32])));
            cases.push((
                "other header digest (AAD)".into(),
                ct.clone(),
                index,
                Sealer::new(&KEY, [1, 2, 3, 4], [8u8; 32]),
            ));
            let mut k2 = KEY;
            k2[31] ^= 0x80;
            cases.push(("other key".into(), ct.clone(), index, Sealer::new(&k2, [1, 2, 3, 4], [7u8; 32])));
            if n > 0 {
                // drop the last ciphertext byte (tag kept)
                let mut c = ct[..n - 1].to_vec();
                c.extend_from_slice(&ct[n..]);
                cases.push(("truncated".into(), c, index, sealer()));
            }
            for (what, c, idx, s2) in cases {
                let out = open_new(&s2, idx, &c).expect_err(&what);
                assert!(out.iter().all(|&b| b == 0), "{what}: output not zeroed");
            }
        }
    }

    /// AWS-LC's free path wipes an AES-256-GCM context: the same allocation and release as aws-lc-rs'
    /// `LcPtr<EVP_AEAD_CTX>` (`OPENSSL_malloc` + `EVP_AEAD_CTX_init`, `EVP_AEAD_CTX_free` on drop), with
    /// the freed memory read back. AES-256 round keys 0 and 1 are the key itself, so the live context holds
    /// the key bytes; after the free, nothing past the allocator's own list pointers (the first 16 bytes of
    /// the chunk, i.e. the 8-byte size prefix and the context's first 8 bytes) may be non-zero.
    #[test]
    fn aws_lc_free_cleanses_the_key_schedule() {
        use aws_lc_sys::{
            EVP_AEAD_CTX_free, EVP_AEAD_CTX_init, EVP_AEAD_CTX_zero, EVP_aead_aes_256_gcm, OPENSSL_malloc, EVP_AEAD_CTX,
        };
        // A fresh thread: its malloc cache is its own, so no other test reuses the freed chunk meanwhile.
        std::thread::spawn(|| {
            let size = std::mem::size_of::<EVP_AEAD_CTX>();
            let find = |bytes: &[u8]| bytes.windows(32).any(|w| w == KEY);
            // SAFETY: the calls of aws-lc-rs `AeadCtx::build_context`; the context is read while live
            // (size bytes from its start), then, once freed, through a volatile read of memory that stays
            // mapped (a small heap chunk in this thread's cache; nothing allocates in between).
            unsafe {
                let ctx: *mut EVP_AEAD_CTX = OPENSSL_malloc(size).cast();
                assert!(!ctx.is_null());
                EVP_AEAD_CTX_zero(ctx);
                assert_eq!(
                    EVP_AEAD_CTX_init(ctx, EVP_aead_aes_256_gcm(), KEY.as_ptr(), 32, 16, std::ptr::null_mut()),
                    1
                );
                let p = ctx.cast::<u8>();
                let live: Vec<u8> = (0..size).map(|i| p.add(i).read_volatile()).collect();
                assert!(find(&live), "the scan does not find the key schedule in the live context");
                EVP_AEAD_CTX_free(ctx);
                let freed: Vec<u8> = (0..size).map(|i| p.add(i).read_volatile()).collect();
                assert!(!find(&freed), "key schedule left in freed memory");
                assert!(freed[8..].iter().all(|&b| b == 0), "freed context not cleansed");
            }
        })
        .join()
        .unwrap();
    }

    /// The key this module builds (`aead_key`, a `LessSafeKey`) leaves no key schedule behind when it is
    /// dropped, after sealing and opening. Its freed context is recovered by allocating a block of the
    /// same size right after the drop: glibc hands the thread's most recently freed chunk of that size
    /// back. A control run first frees a context *without* the cleanse (plain `free`) and must find the
    /// key in the recovered block, which shows the method sees what a drop leaves.
    #[test]
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn dropped_key_leaves_no_key_schedule() {
        use aws_lc_sys::{
            EVP_AEAD_CTX_init, EVP_AEAD_CTX_zero, EVP_aead_aes_256_gcm, OPENSSL_free, OPENSSL_malloc, EVP_AEAD_CTX,
        };
        std::thread::spawn(|| {
            let size = std::mem::size_of::<EVP_AEAD_CTX>();
            let find = |bytes: &[u8]| bytes.windows(32).any(|w| w == KEY);
            // SAFETY: a fresh allocation of `size` bytes is read (uninitialised bytes read as u8 through
            // volatile reads of mapped heap memory) and released with OPENSSL_free.
            let recover = || unsafe {
                let p: *mut u8 = OPENSSL_malloc(size).cast();
                assert!(!p.is_null());
                let v: Vec<u8> = (0..size).map(|i| p.add(i).read_volatile()).collect();
                OPENSSL_free(p.cast());
                (p as usize, v)
            };
            // warm up aws-lc-rs (one-time initialisation allocations)
            drop(aead_key(&[1u8; 32]));
            // control: a context freed without the cleanse is found again
            // SAFETY: as aws-lc-rs' `AeadCtx::build_context`, then a raw `free` of the malloc'd block
            // (OPENSSL_malloc's 8-byte size prefix precedes the context).
            unsafe {
                let ctx: *mut EVP_AEAD_CTX = OPENSSL_malloc(size).cast();
                EVP_AEAD_CTX_zero(ctx);
                assert_eq!(
                    EVP_AEAD_CTX_init(ctx, EVP_aead_aes_256_gcm(), KEY.as_ptr(), 32, 16, std::ptr::null_mut()),
                    1
                );
                libc::free(ctx.cast::<u8>().sub(8).cast());
            }
            assert!(find(&recover().1), "control: the recovered block does not show an uncleansed context");
            // the real thing: seal and open with the key, drop it
            let mut s = sealer();
            let ct = seal_new(&s, 3, &plain(4096, 1));
            open_new(&s, 3, &ct).unwrap();
            // the words of the live key: one of them is its context pointer
            // SAFETY: reading the bytes of a live value.
            let words: Vec<usize> = unsafe {
                let n = std::mem::size_of::<LessSafeKey>() / 8;
                (0..n).map(|i| (&s.aead as *const LessSafeKey).cast::<usize>().add(i).read_unaligned()).collect()
            };
            s = Sealer::new(&[2u8; 32], [0; 4], [0; 32]); // drops the first key
            let (p, freed) = recover();
            assert!(words.contains(&p), "recovered block is not the dropped key's context");
            assert!(!find(&freed), "key schedule left after drop");
            assert!(freed[8..].iter().all(|&b| b == 0), "dropped context not cleansed");
            drop(s);
        })
        .join()
        .unwrap();
    }

    /// `aead_key` holds its context as aws-lc-rs' `LcPtr<EVP_AEAD_CTX>`: dropping it must go through
    /// `EVP_AEAD_CTX_free` (the path tested above). The key and the context are owned through one
    /// pointer: a `LessSafeKey` is a pointer plus the algorithm reference, so it holds no key bytes inline.
    #[test]
    fn less_safe_key_holds_no_key_bytes_inline() {
        let k = aead_key(&KEY);
        let n = std::mem::size_of::<LessSafeKey>();
        // SAFETY: reading the bytes of a live value.
        let raw = unsafe { std::slice::from_raw_parts((&k as *const LessSafeKey).cast::<u8>(), n) };
        assert!(n <= 64, "LessSafeKey grew to {n} bytes");
        assert!(!raw.windows(16).any(|w| w == &KEY[..16] || w == &KEY[16..]));
    }

    #[test]
    fn run_jobs_threads_write_the_same_file() {
        let s = sealer();
        let data = plain(3 << 20, 11);
        let jobs: Vec<Job> = (0..7u64)
            .map(|i| {
                let start = (i as usize) * 400_000;
                Job { index: i, offset: i * (400_000 + TAG_BYTES as u64), start, len: 400_000 }
            })
            .collect();
        let mut files = Vec::new();
        for threads in [1usize, 3] {
            let f = tempfile_in_target(threads);
            run_jobs(
                std::os::fd::AsRawFd::as_raw_fd(&f),
                &s,
                &jobs,
                |j| j.len,
                |j, out| out.copy_from_slice(&data[j.start..j.start + j.len]),
                threads,
            )
            .unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut &f, &mut bytes).unwrap();
            for j in &jobs {
                let at = j.offset as usize;
                let ct = &bytes[at..at + j.len + TAG_BYTES];
                assert_eq!(open_new(&s, j.index as u32, ct).unwrap(), &data[j.start..j.start + j.len]);
            }
            files.push(bytes);
        }
        assert_eq!(files[0], files[1]);
    }

    fn tempfile_in_target(tag: usize) -> File {
        let p = std::env::temp_dir().join(format!("pgw-writer-test-{}-{tag}", std::process::id()));
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        f
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
        assert!(!all_in_range(&[i64::MIN], 64, 1));
        assert!(!all_in_range(&[256u32], 8, 1));
        let mut big = vec![3i64; 3 << 20];
        assert!(all_in_range(&big, 8, 0));
        *big.last_mut().unwrap() = 256;
        assert!(!all_in_range(&big, 8, 4));
    }

    /// The range-check shortcut (`ALWAYS_IN_RANGE_FROM_K`) only skips checks that cannot fail.
    #[test]
    fn range_shortcut_is_exact() {
        fn full<T: RingElem>(vals: &[T], k: u32) -> bool {
            vals.iter().all(|v| v.bad_bits(ring_mask(k)) == 0)
        }
        fn check<T: RingElem>(extremes: &[T]) {
            for k in (8..=64).step_by(8) {
                if k >= T::ALWAYS_IN_RANGE_FROM_K {
                    assert!(full(extremes, k), "k {k}");
                }
                assert_eq!(all_in_range(extremes, k, 1), full(extremes, k), "k {k}");
            }
        }
        check(&[0u32, 1, 0x8000_0000, u32::MAX]);
        check(&[0i32, 1, -1, i32::MIN, i32::MAX]);
        check(&[0u64, 1, 1 << 32, 1 << 63, u64::MAX]);
        check(&[0i64, 1, -1, i64::MIN, i64::MAX, 1 << 32]);
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
