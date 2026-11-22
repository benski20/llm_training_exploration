//! One native call per trip on each side of the lane call (`PGW_TRIP_NATIVE`, opt-in).
//!
//! The Python reservation layers ([`pgw.maskstore.MaskReservation`], `WindowedReservation`) hand a
//! *lease* to this module: the next rows of one linear that live in one decrypted `W r` chunk of one
//! segment extent, the extent's `r` stream key and its Freivalds vectors. Per trip the gateway then
//! makes one call before the lane call ([`TripLease::prepare`]: take the rows, derive `r`, quantize,
//! mask and pack into the wire payload, Freivalds input half) and one after it
//! ([`TripTicket::finish`]: unmask with `W r`, range check, Freivalds, rescale, wipe).
//!
//! # Single use and accounting
//!
//! * A lease never hands out a row twice: [`TripLease::prepare`] advances the lease position (under
//!   its lock, with the GIL held) *before* any secret is derived, and nothing ever moves it back. A
//!   failed prepare, a failed or abandoned trip and a fallback leave the row consumed.
//! * The reservation's cursor of the linear is `start_row + pos` while the lease is attached
//!   ([`TripLease::cursor`]); the Python side detaches the lease ([`TripLease::detach`]) before any
//!   other path takes rows of that linear, and closing the reservation closes its leases
//!   ([`TripLease::close`]): burn-on-close counts every row past the cursor as before.
//! * Rows of the lease that were never handed out are zeroed (with the whole chunk) when the lease
//!   and the last ticket reading it are gone; they were never sent, so a later path may derive them
//!   again (exactly what a dropped prefetch stage does).
//!
//! # Secrets
//!
//! The `r` stream key is zeroized on detach/close/drop; `r` lives in a per-thread scratch buffer that
//! is zero between calls; `W r` rows are read straight from the decrypted chunk and zeroed by the
//! finish kernel as they are used (a ticket dropped without finishing zeroes its rows); `sx` and the
//! Freivalds input half are wiped when the ticket finishes or is dropped.
//!
//! # Freivalds format
//!
//! Every use of the check format goes through [`check_iface`] (input half at prepare, output half at
//! finish, deferred submission to the helper thread), so a change of the check format touches only
//! that module.

use crate::finish::{wipe, RoRing, Src, WrBuf};
use crate::prepare::{PrepareError, PrepareParams};
use crate::store::fill_ring;
use crate::trip::{finish_into, prepare_into_field, CheckMode, FinishArgs, PrepOut, TripCheck};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard};
use zeroize::{Zeroize, Zeroizing};

/// Add this module's classes to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TripLease>()?;
    m.add_class::<TripTicket>()?;
    Ok(())
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

// ---- the narrow interface to the Freivalds check format -----------------------------------------

/// The only code of this module that knows the Freivalds check format (`u (c, m)`, `uw (c, n)`, the
/// input half `rhs (rows, c) = uw . x_hat mod 2^K` and the output half `u . y_hat`). It calls the
/// existing kernels of [`crate::trip`]; a format change replaces these four functions.
mod check_iface {
    use super::*;
    use crate::trip::InputHalf;

    /// `c` (number of check vectors) of `u (c, m)` / `uw (c, n)`, validating the shapes.
    pub(super) fn vectors(u: &RoRing<'_>, uw: &RoRing<'_>, m: usize, n: usize) -> PyResult<usize> {
        let (_, ush) = u.src("u")?;
        let (_, uwsh) = uw.src("uw")?;
        match (ush, uwsh) {
            (Some((c, um)), Some((c2, un))) if c >= 1 && c == c2 && um == m && un == n => Ok(c),
            _ => Err(verr(format!("check vectors do not match the linear: u {ush:?}, uw {uwsh:?}, m={m} n={n}"))),
        }
    }

    /// Words of `rhs` per row (the prepare-time input half's size).
    pub(super) fn rhs_len(rows: usize, c: usize) -> usize {
        rows * c
    }

    /// The input half for [`prepare_into`] (computed in the same pass as the payload).
    pub(super) fn input_half<'a>(uw: Src<'a>, c: usize) -> InputHalf<'a> {
        InputHalf { uw, c }
    }

    /// The finish kernel's check mode for a synchronous check against the prepare-time `rhs`.
    pub(super) fn split<'a>(u: Src<'a>, rhs: &'a [u64], c: usize) -> CheckMode<'a> {
        CheckMode::Split { u, rhs, c }
    }

    /// The check field of a lease: `0` (the ring check over `Z_{2^K}`) or [`crate::prime::P31`] (the
    /// `"p31"` segment format: one vector over `F_p` on the unmasked `xq` / `yq`, `u32` words, `K` in
    /// {8, 16, 24, 32}). Validates the vectors' word type for the prime format.
    pub(super) fn field(field: u64, u: &RoRing<'_>, uw: &RoRing<'_>, k: u32) -> PyResult<u64> {
        match field {
            0 => Ok(0),
            f if f == crate::prime::P31 => {
                if !matches!(k, 8 | 16 | 24 | 32) {
                    return Err(verr(format!("the prime check needs K in {{8, 16, 24, 32}}, got {k}")));
                }
                if !matches!(u, RoRing::U32(_)) || !matches!(uw, RoRing::U32(_)) {
                    return Err(verr("prime-field check vectors must be uint32 arrays"));
                }
                Ok(f)
            }
            f => Err(verr(format!("unknown check field {f} (0: ring, 2^31 - 1: p31)"))),
        }
    }

    /// Whether the output half runs synchronously inside finish (the prime format never defers: the
    /// result is used only after it was verified).
    pub(super) fn sync_only(field: u64) -> bool {
        field != 0
    }

    /// The prime output half (range check + `u . yq == rhs mod p`) on the trip's `W r` (read only, before
    /// the finish kernel unmasks and wipes it). Returns the finish statuses.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prime_output(
        y_hat: &[u8],
        wr: Src<'_>,
        u: Src<'_>,
        rhs: &[u64],
        c: usize,
        m: usize,
        k: u32,
        bound: u64,
        split: usize,
    ) -> (u8, usize) {
        match u {
            Src::U32(us) => crate::prime::output_half_split(y_hat, wr, us, rhs, c, m, k, bound, split),
            // validated at lease creation (uint32 vectors): unreachable; fail closed
            _ => (crate::finish::STATUS_FREIVALDS, rhs.len() / c.max(1)),
        }
    }

    /// The output half on the native helper thread (`trip_freivalds_submit`); `rhs` is wiped here.
    pub(super) fn submit(
        py: Python<'_>,
        y_hat: &Bound<'_, PyAny>,
        u: &Bound<'_, PyAny>,
        rhs: &mut [u64],
        rows: usize,
        c: usize,
        k: u32,
    ) -> PyResult<TripCheck> {
        let arr = PyArray2::<u64>::zeros(py, [rows, c], false);
        {
            // SAFETY: freshly allocated, C-contiguous, not shared with Python code yet.
            let s = unsafe { arr.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
            s.copy_from_slice(rhs);
        }
        wipe(rhs);
        let res = crate::trip::trip_freivalds_submit(y_hat, u, arr.as_any(), k);
        // the submission copied rhs (or failed): zero our temporary either way
        if let Ok(mut rw) = arr.try_readwrite() {
            if let Ok(s) = rw.as_slice_mut() {
                wipe(s);
            }
        }
        res
    }
}

// ---- chunk and lease state -----------------------------------------------------------------------

/// A decrypted `W r` chunk (a Python `bytearray` the reservation handed over and no longer touches).
/// Zeroed when the last lease or ticket reading it is gone.
struct ChunkHold {
    obj: Option<Py<PyAny>>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the bytearray's buffer is owned by `obj` (kept alive until after the zeroing in Drop) and
// never resized; only this module reads or writes it after the hand-over, and tickets touch
// pairwise disjoint row ranges.
unsafe impl Send for ChunkHold {}
unsafe impl Sync for ChunkHold {}

/// Zero `len` bytes at `ptr` with one `memset` that the optimiser cannot drop: the memory stays
/// reachable afterwards (it belongs to a Python object released only later, through an opaque decref)
/// and its address escapes through `black_box`.
///
/// # Safety
/// `len` bytes at `ptr` must be valid for writes and not accessed concurrently.
unsafe fn zero_bytes(ptr: *mut u8, len: usize) {
    std::ptr::write_bytes(ptr, 0, len);
    std::hint::black_box(ptr);
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

impl Drop for ChunkHold {
    fn drop(&mut self) {
        // SAFETY: `len` bytes at `ptr` stay valid while `obj` is held (dropped below); no lease or
        // ticket references this chunk any more.
        unsafe { zero_bytes(self.ptr, self.len) };
        // pyo3 defers the decref when the GIL is not held
        drop(self.obj.take());
    }
}

struct LeaseState {
    /// The reservation closed: no more rows, in-flight tickets fail closed.
    closed: bool,
    /// Another path took over this linear's cursor (or the lease ran out): no more rows.
    detached: bool,
    /// Rows handed out.
    pos: usize,
    key: Zeroizing<[u8; 32]>,
    /// The lease's rows, chunk by chunk, in order (released on detach/close).
    pieces: Vec<Piece>,
    /// `r` of row `.0` (the next row to hand out), derived ahead on the helper thread; used by the
    /// trip that takes that row only, zeroed otherwise.
    ahead: Option<(usize, Vec<i64>)>,
}

impl LeaseState {
    fn drop_ahead(&mut self) {
        if let Some((_, mut v)) = self.ahead.take() {
            wipe(&mut v);
        }
    }
}

type RJob = Box<dyn FnOnce() + Send + 'static>;

/// The `r` helper threads (persistent; `PGW_TRIP_R_HELPERS`, read once, default 1, at most 8; jobs
/// taken in submission order): derive the `r` of a lease's next trip ahead.
fn r_helper() -> &'static Mutex<std::sync::mpsc::Sender<RJob>> {
    static TX: std::sync::OnceLock<Mutex<std::sync::mpsc::Sender<RJob>>> = std::sync::OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<RJob>();
        let rx = Arc::new(Mutex::new(rx));
        let helpers = std::env::var("PGW_TRIP_R_HELPERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        for i in 0..helpers {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("pgw-trip-r{i}"))
                .spawn(move || loop {
                    let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                })
                .expect("cannot start a trip r thread");
        }
        Mutex::new(tx)
    })
}

/// Derive `r` of lease row `row` (segment slot `slot`) on the helper thread; it is stored only if
/// that row is still the lease's next one when it is ready (else zeroed).
fn schedule_ahead(core: &Arc<LeaseCore>, key: &[u8; 32], row: usize, slot: u64, n: usize, t: usize, k: u32) {
    let core = Arc::clone(core);
    let key = Zeroizing::new(*key);
    let job: RJob = Box::new(move || {
        let mut v = vec![0i64; n * t];
        let ok = fill_ring(&key, slot * n as u64, &mut v, k).is_ok();
        drop(key);
        let mut g = core.lock();
        if ok && !g.closed && !g.detached && g.pos == row && g.ahead.is_none() {
            g.ahead = Some((row, v));
        } else {
            drop(g);
            wipe(&mut v);
        }
    });
    let sent = r_helper().lock().unwrap_or_else(|e| e.into_inner()).send(job);
    if let Err(e) = sent {
        drop(e.0); // helper gone: the row is derived inline (the closure's buffers are untouched)
    }
}

/// Lease rows `[start, start + rows)` live in chunk rows `[row_off, row_off + rows)` of `hold`.
struct Piece {
    hold: Arc<ChunkHold>,
    row_off: usize,
    start: usize,
    rows: usize,
}

/// `rows` rows of `W r` at element offset `elem_off` of a chunk (one ticket's share of a piece).
struct Seg {
    hold: Arc<ChunkHold>,
    elem_off: usize,
    rows: usize,
}

struct LeaseCore {
    st: Mutex<LeaseState>,
}

impl LeaseCore {
    fn lock(&self) -> MutexGuard<'_, LeaseState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn closed(&self) -> bool {
        self.lock().closed
    }
    fn end(&self, close: bool) -> usize {
        let mut g = self.lock();
        g.detached = true;
        g.closed |= close;
        g.key.zeroize();
        g.pieces.clear();
        g.drop_ahead();
        g.pos
    }
}

thread_local! {
    /// `r` scratch of the calling thread: zero between calls (`fill_ring` XORs the keystream in).
    static R_SCRATCH: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    /// `W r` widened from a packed chunk (K != 32): zero between calls.
    static WR_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// The next rows of one linear in one decrypted `W r` chunk of one segment extent (see the module
/// docs). Created by the Python reservation; used by the gateway's native per-trip path.
#[pyclass(module = "pgw_fast")]
pub struct TripLease {
    core: Arc<LeaseCore>,
    n: usize,
    m: usize,
    prm: PrepareParams,
    /// Segment slot of lease row 0 (`r` stream offset `slot * n`).
    slot0: u64,
    /// Lease rows.
    rows: usize,
    /// Reservation-global row index of lease row 0.
    start_row: usize,
    u: Py<PyAny>,
    uw: Py<PyAny>,
    c: usize,
    /// `W r` rows read in place as u32 words (K = 32 and an aligned buffer); else widened per trip.
    direct: bool,
    /// Derive the next row's `r` on the helper thread after each 1-row trip.
    r_ahead: bool,
    /// Check field ([`check_iface::field`]): 0 ring, `2^31 - 1` prime (`"p31"`).
    field: u64,
}

#[pymethods]
impl TripLease {
    /// ``TripLease(key, n, m, k, a, cap, wnorm, slot0, chunks, start_row, u, uw)``.
    ///
    /// ``key``: the 32-byte ``r`` stream key of (extent, linear); ``chunks``: ``[(chunk, row_off,
    /// rows), ...]``, the decrypted ``W r`` chunks holding the lease's consecutive rows (each a
    /// ``bytearray`` the caller hands over: it must never touch it again; zeroed here), starting at
    /// chunk row ``row_off``; the lease's first row has segment slot ``slot0`` and reservation row
    /// ``start_row``; ``u``/``uw``: the extent's Freivalds vectors (read-only arrays, shared per
    /// segment).
    #[new]
    #[pyo3(signature = (key, n, m, k, a, cap, wnorm, slot0, chunks, start_row, u, uw, r_ahead = false, field = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        key: &Bound<'_, PyAny>,
        n: usize,
        m: usize,
        k: u32,
        a: u32,
        cap: u32,
        wnorm: f64,
        slot0: u64,
        chunks: Vec<(Bound<'_, PyByteArray>, usize, usize)>,
        start_row: usize,
        u: &Bound<'_, PyAny>,
        uw: &Bound<'_, PyAny>,
        r_ahead: bool,
        field: u64,
    ) -> PyResult<Self> {
        let prm = PrepareParams { k, a, cap, wnorm };
        prm.validate().map_err(PrepareError::into_py)?;
        let nb = (k / 8) as usize;
        let mut kb = Zeroizing::new([0u8; 32]);
        {
            let kbytes: Zeroizing<Vec<u8>> = if let Ok(b) = key.downcast::<PyBytes>() {
                Zeroizing::new(b.as_bytes().to_vec())
            } else if let Ok(b) = key.downcast::<PyByteArray>() {
                Zeroizing::new(b.to_vec())
            } else {
                return Err(verr("key must be bytes or bytearray"));
            };
            if kbytes.len() != 32 {
                return Err(verr("key must be exactly 32 bytes"));
            }
            kb.copy_from_slice(&kbytes);
        }
        if n == 0 || m == 0 || chunks.is_empty() {
            return Err(verr("n and m must be positive and chunks non-empty"));
        }
        let mut direct = nb == 4 && cfg!(target_endian = "little");
        let mut pieces = Vec::with_capacity(chunks.len());
        let mut rows = 0usize;
        for (chunk, row_off, prows) in &chunks {
            let (row_off, prows) = (*row_off, *prows);
            let len = chunk.len();
            if prows == 0 || (row_off + prows) * m * nb > len {
                return Err(verr(format!(
                    "chunk has {len} bytes, lease needs rows [{row_off}, {}) of {m}x{nb}",
                    row_off + prows
                )));
            }
            let ptr = chunk.data();
            direct &= (ptr as usize) % 4 == 0;
            let hold = ChunkHold { obj: Some(chunk.clone().into_any().unbind()), ptr, len };
            pieces.push(Piece { hold: Arc::new(hold), row_off, start: rows, rows: prows });
            rows += prows;
        }
        let (c, field) = {
            let ua = RoRing::extract(u, "u", k)?;
            let uwa = RoRing::extract(uw, "uw", k)?;
            if matches!(ua, RoRing::B(_)) || matches!(uwa, RoRing::B(_)) {
                return Err(verr("u and uw must be 2-D numpy arrays"));
            }
            (check_iface::vectors(&ua, &uwa, m, n)?, check_iface::field(field, &ua, &uwa, k)?)
        };
        Ok(TripLease {
            core: Arc::new(LeaseCore {
                st: Mutex::new(LeaseState { closed: false, detached: false, pos: 0, key: kb, pieces, ahead: None }),
            }),
            n,
            m,
            prm,
            slot0,
            rows,
            start_row,
            u: u.clone().unbind(),
            uw: uw.clone().unbind(),
            c,
            direct,
            r_ahead,
            field,
        })
    }

    /// The check field (0: ring check, ``2**31 - 1``: the prime-field ``"p31"`` format).
    #[getter]
    fn field(&self) -> u64 {
        self.field
    }

    /// True if the next ``t`` rows can be handed out (attached, not closed, enough rows left).
    fn ready(&self, t: usize) -> bool {
        let g = self.core.lock();
        !g.closed && !g.detached && g.pos + t <= self.rows
    }

    /// Rows handed out.
    #[getter]
    fn pos(&self) -> usize {
        self.core.lock().pos
    }

    /// Rows not yet handed out (0 once detached or closed).
    #[getter]
    fn left(&self) -> usize {
        let g = self.core.lock();
        if g.closed || g.detached {
            0
        } else {
            self.rows - g.pos
        }
    }

    /// Reservation-global index of the lease's first row.
    #[getter]
    fn start_row(&self) -> usize {
        self.start_row
    }

    /// Rows in the lease.
    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    /// The linear's cursor in the reservation: ``start_row + pos``.
    #[getter]
    fn cursor(&self) -> usize {
        self.start_row + self.core.lock().pos
    }

    /// True after :meth:`close`.
    #[getter]
    fn closed(&self) -> bool {
        self.core.closed()
    }

    /// Stop handing out rows (another path takes over the cursor); returns the cursor. The key is
    /// zeroized and the chunk released (zeroed once no ticket reads it). Idempotent.
    fn detach(&self) -> usize {
        self.start_row + self.core.end(false)
    }

    /// :meth:`detach`, and in-flight tickets fail closed (the reservation closed). Returns the cursor.
    fn close(&self) -> usize {
        self.start_row + self.core.end(true)
    }

    /// The per-trip call before the lane call: take the next ``rows(x)`` rows, derive ``r``,
    /// quantize + mask + pack (constant-time width selection, as ``trip_prepare_into``) and, with
    /// ``check``, the Freivalds input half.
    ///
    /// Returns ``None`` (no row taken) if the lease cannot serve the call, else ``(payload, ticket,
    /// xh, xq)``: the wire bytes, the :class:`TripTicket` that finishes the trip, and with ``sampled``
    /// the masked values ``x_hat`` and the plaintext ``xq = center(x_hat - r)`` as int64 ``(rows, n)``
    /// arrays for the noise monitor's statistics (secret: zero ``xq`` after use).
    ///
    /// Raises ``ValueError("nonfinite: ...")`` before any row is taken for non-finite activations,
    /// and ``ValueError("ring_overflow: ...")`` (row consumed, outputs zeroed) when no width fits.
    ///
    /// ``dup``: the outbound-noise monitor's duplicate-row fingerprint parameters ``(cols, shift, mult,
    /// groups, per)`` (``noise_dup_keys``): the fingerprints of the payload are then computed in this call
    /// and returned as a fifth element (uint64 ``(rows, groups)``, else ``None``); the caller records them
    /// (the monitor's every-call check) before the payload is sent.
    #[pyo3(signature = (x, check, sampled = false, threads = 1, dup = None, split = 0))]
    #[allow(clippy::type_complexity)]
    fn prepare<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        check: bool,
        sampled: bool,
        threads: usize,
        dup: Option<(Bound<'py, PyArray1<i64>>, u32, Bound<'py, PyArray1<u64>>, usize, usize)>,
        split: usize,
    ) -> PyResult<
        Option<(
            Bound<'py, PyBytes>,
            TripTicket,
            Option<Bound<'py, PyArray2<i64>>>,
            Option<Bound<'py, PyArray2<i64>>>,
            Option<Bound<'py, PyArray2<u64>>>,
        )>,
    > {
        let n = self.n;
        let xa = x
            .downcast::<PyArray2<f32>>()
            .map_err(|_| verr("x must be a 2-D float32 numpy array"))?
            .try_readonly()
            .map_err(|e| verr(format!("x: {e}")))?;
        if !xa.is_c_contiguous() || xa.shape()[1] != n {
            return Err(verr(format!("x must be C-contiguous (t, {n}), got {:?}", xa.shape())));
        }
        let t = xa.shape()[0];
        if t == 0 {
            return Err(verr("x has no rows"));
        }
        let xs = xa.as_slice().map_err(|e| verr(e.to_string()))?;
        if !all_finite(xs) {
            return Err(verr("nonfinite: non-finite activation values cannot be quantized"));
        }
        let uwa = if check { Some(RoRing::extract(self.uw.bind(py), "uw", self.prm.k)?) } else { None };
        // the fingerprint parameters are validated before any row is taken
        let dup_ro = match &dup {
            Some((cols, shift, mult, groups, per)) => {
                let ca = cols.try_readonly().map_err(|e| verr(format!("cols: {e}")))?;
                let ma = mult.try_readonly().map_err(|e| verr(format!("mult: {e}")))?;
                {
                    let cs = ca.as_slice().map_err(|_| verr("cols must be contiguous"))?;
                    let ms = ma.as_slice().map_err(|_| verr("mult must be contiguous"))?;
                    if !DupIn::valid(cs, *shift, ms, *groups, *per, n) {
                        return Err(verr("bad fingerprint parameters"));
                    }
                }
                Some((ca, *shift, ma, *groups, *per))
            }
            None => None,
        };
        if !self.ready(t) {
            return Ok(None);
        }
        let nb = (self.prm.k / 8) as usize;
        let plen = t * n * nb;
        // SAFETY: a fresh, unshared bytes object of `plen` bytes; filled below before anyone sees it.
        let payload: Bound<'py, PyBytes> = unsafe {
            let p = pyo3::ffi::PyBytes_FromStringAndSize(std::ptr::null(), plen as pyo3::ffi::Py_ssize_t);
            Bound::from_owned_ptr_or_err(py, p)?.downcast_into_unchecked()
        };
        // SAFETY: exclusive (just created, not returned yet).
        let pl: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(payload.as_bytes().as_ptr() as *mut u8, plen) };
        let (xh_arr, xq_arr) = if sampled {
            (Some(PyArray2::<i64>::zeros(py, [t, n], false)), Some(PyArray2::<i64>::zeros(py, [t, n], false)))
        } else {
            (None, None)
        };
        // SAFETY: fresh arrays, not shared with Python code until returned.
        let xh: &mut [i64] = match &xh_arr {
            Some(a) => unsafe { a.as_slice_mut() }.map_err(|e| verr(e.to_string()))?,
            None => &mut [],
        };
        let xq: &mut [i64] = match &xq_arr {
            Some(a) => unsafe { a.as_slice_mut() }.map_err(|e| verr(e.to_string()))?,
            None => &mut [],
        };
        let keys_arr = dup_ro.as_ref().map(|(_, _, _, groups, _)| PyArray2::<u64>::zeros(py, [t, *groups], false));
        // SAFETY: fresh array, not shared with Python code yet.
        let keys: &mut [u64] = match &keys_arr {
            Some(a) => unsafe { a.as_slice_mut() }.map_err(|e| verr(e.to_string()))?,
            None => &mut [],
        };
        let uw_src = match &uwa {
            Some(a) => Some(a.src("uw")?.0),
            None => None,
        };
        let dup_in = match &dup_ro {
            Some((ca, shift, ma, groups, per)) => Some(DupIn {
                cols: ca.as_slice().map_err(|e| verr(e.to_string()))?,
                shift: *shift,
                mult: ma.as_slice().map_err(|e| verr(e.to_string()))?,
                groups: *groups,
                per: *per,
            }),
            None => None,
        };
        let res = py.allow_threads(|| {
            self.prepare_core(xs, t, uw_src, threads, split, PrepIo { payload: pl, xh, xq, keys }, dup_in.as_ref())
        });
        match res {
            Ok(None) => Ok(None),
            Ok(Some(mut ticket)) => {
                if check {
                    ticket.u = Some(self.u.clone_ref(py));
                }
                Ok(Some((payload, ticket, xh_arr, xq_arr, keys_arr)))
            }
            Err(e) => Err(e.into_py()),
        }
    }
}

/// Why [`TripLease::prepare_core`] failed.
pub(crate) enum CoreErr {
    /// Non-finite activations (no row taken).
    NonFinite,
    /// The prepare kernel refused (row consumed, outputs zeroed).
    Prep(PrepareError),
    /// Anything else (row consumed when it happened after the take, outputs zeroed).
    Msg(String),
}

impl CoreErr {
    pub(crate) fn into_py(self) -> PyErr {
        match self {
            CoreErr::NonFinite => verr("nonfinite: non-finite activation values cannot be quantized"),
            CoreErr::Prep(e) => e.into_py(),
            CoreErr::Msg(m) => verr(m),
        }
    }
}

/// The outbound-noise monitor's duplicate-row fingerprint parameters (`noise_dup_keys`).
pub(crate) struct DupIn<'a> {
    pub(crate) cols: &'a [i64],
    pub(crate) shift: u32,
    pub(crate) mult: &'a [u64],
    pub(crate) groups: usize,
    pub(crate) per: usize,
}

impl DupIn<'_> {
    pub(crate) fn valid(cols: &[i64], shift: u32, mult: &[u64], groups: usize, per: usize, n: usize) -> bool {
        groups > 0
            && per > 0
            && cols.len() >= groups * per
            && mult.len() >= groups * per
            && shift < 64
            && cols.iter().take(groups * per).all(|&c| c >= 0 && (c as usize) < n)
    }
}

/// Output buffers of [`TripLease::prepare_core`]: the payload (`rows * n * K/8` bytes), with sampling
/// `x_hat` and `xq` (`rows * n`, else empty), with fingerprints the keys (`rows * groups`, else empty).
pub(crate) struct PrepIo<'a> {
    pub(crate) payload: &'a mut [u8],
    pub(crate) xh: &'a mut [i64],
    pub(crate) xq: &'a mut [i64],
    pub(crate) keys: &'a mut [u64],
}

pub(crate) fn all_finite(xs: &[f32]) -> bool {
    xs.iter().fold(true, |ok, v| ok & v.is_finite())
}

impl TripLease {
    pub(crate) fn n_in(&self) -> usize {
        self.n
    }
    pub(crate) fn m_out(&self) -> usize {
        self.m
    }
    pub(crate) fn k_bits(&self) -> u32 {
        self.prm.k
    }
    pub(crate) fn uw_obj(&self) -> &Py<PyAny> {
        &self.uw
    }
    pub(crate) fn u_obj(&self) -> &Py<PyAny> {
        &self.u
    }

    /// The GIL-free core of `prepare` (the Python method and the decode driver, `decode.rs`):
    /// finiteness (no row taken on failure), take `t` rows (consumed from then on), derive `r`,
    /// quantize + mask + pack, the input half when `uw` is given, `x_hat`/`xq` when `io.xh` is
    /// non-empty, the fingerprints when `dup` is given. `Ok(None)`: the lease cannot serve (nothing
    /// taken). The returned ticket has no check vectors attached (`u`): the caller attaches them or
    /// passes them to [`TripTicket::finish_core`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_core(
        &self,
        xs: &[f32],
        t: usize,
        uw: Option<Src<'_>>,
        threads: usize,
        split: usize,
        io: PrepIo<'_>,
        dup: Option<&DupIn<'_>>,
    ) -> Result<Option<TripTicket>, CoreErr> {
        let n = self.n;
        let nb = (self.prm.k / 8) as usize;
        let PrepIo { payload: pl, xh, xq, keys } = io;
        if t == 0 || xs.len() != t * n || pl.len() != t * n * nb {
            return Err(CoreErr::Msg(format!("prepare_core: shapes (t={t}, n={n})")));
        }
        if (!xh.is_empty() && (xh.len() != t * n || xq.len() != t * n)) || (xh.is_empty() && !xq.is_empty()) {
            return Err(CoreErr::Msg("prepare_core: x_hat/xq buffers".into()));
        }
        if let Some(d) = dup {
            if !DupIn::valid(d.cols, d.shift, d.mult, d.groups, d.per, n) || keys.len() != t * d.groups {
                return Err(CoreErr::Msg("bad fingerprint parameters".into()));
            }
        }
        if !all_finite(xs) {
            return Err(CoreErr::NonFinite);
        }
        let check = uw.is_some();
        // ---- take the rows (consumed from here on, whatever happens below) ----
        let (row0, segs, mut key, pre) = {
            let mut g = self.core.lock();
            if g.closed || g.detached || g.pos + t > self.rows || g.pieces.is_empty() {
                return Ok(None);
            }
            let row0 = g.pos;
            let (a, b) = (row0, row0 + t);
            let segs: Vec<Seg> = g
                .pieces
                .iter()
                .filter(|p| p.start < b && a < p.start + p.rows)
                .map(|p| {
                    let lo = a.max(p.start);
                    let hi = b.min(p.start + p.rows);
                    Seg { hold: Arc::clone(&p.hold), elem_off: (p.row_off + lo - p.start) * self.m, rows: hi - lo }
                })
                .collect();
            g.pos += t;
            let pre = match g.ahead.take() {
                Some((row, v)) if row == row0 && v.len() == t * n => Some(Zeroizing::new(v)),
                Some((_, mut v)) => {
                    wipe(&mut v);
                    None
                }
                None => None,
            };
            // the next trip of this lease (assumed to take as many rows) gets its r derived ahead
            if self.r_ahead && g.pos + t <= self.rows {
                schedule_ahead(&self.core, &g.key, g.pos, self.slot0 + g.pos as u64, n, t, self.prm.k);
            }
            (row0, segs, Zeroizing::new(*g.key), pre)
        };
        let c = self.c;
        let mut ticket = TripTicket {
            core: Arc::clone(&self.core),
            segs,
            rows: t,
            m: self.m,
            k: self.prm.k,
            direct: self.direct,
            sx: vec![0f64; t],
            rhs: if check { vec![0u64; check_iface::rhs_len(t, c)] } else { Vec::new() },
            c: if check { c } else { 0 },
            u: None,
            done: false,
            field: self.field,
        };
        let half = uw.map(|s| check_iface::input_half(s, c));
        let prm = self.prm;
        let slot = self.slot0 + row0 as u64;
        let field = if check { self.field } else { 0 };
        let mut pre = pre;
        let res: Result<(), CoreErr> = {
            let sx: &mut [f64] = &mut ticket.sx;
            let rhs: &mut [u64] = &mut ticket.rhs;
            R_SCRATCH.with(|cell| {
                let mut r = cell.borrow_mut();
                let have = pre.is_some();
                let rs: &mut [i64] = match pre.as_mut() {
                    Some(v) => &mut v[..],
                    None => {
                        if r.len() < t * n {
                            r.resize(t * n, 0); // zero-extended: the scratch stays all zero between calls
                        }
                        &mut r[..t * n]
                    }
                };
                let out = (|| -> Result<(), CoreErr> {
                    if !have {
                        if split >= 2 {
                            crate::trip::split::fill_ring_split(&key, slot * n as u64, rs, prm.k, split)
                                .map_err(|e| CoreErr::Msg(e.to_string()))?;
                        } else {
                            fill_ring(&key, slot * n as u64, rs, prm.k).map_err(|e| CoreErr::Msg(e.to_string()))?;
                        }
                    }
                    let po = PrepOut { payload: &mut *pl, sx, xhat: &mut *xh, rhs };
                    if split >= 2 {
                        crate::trip::split::prepare_split(xs, rs, n, prm, half, field, po, split)
                            .map_err(CoreErr::Prep)?;
                    } else {
                        prepare_into_field(xs, rs, n, prm, half, field, po, threads).map_err(CoreErr::Prep)?;
                    }
                    if !xq.is_empty() {
                        let mask: u64 = if prm.k == 64 { u64::MAX } else { (1u64 << prm.k) - 1 };
                        let half_mod: u64 = 1u64 << (prm.k - 1);
                        for ((q, &h), &rv) in xq.iter_mut().zip(xh.iter()).zip(rs.iter()) {
                            let v = (h as u64).wrapping_sub(rv as u64) & mask;
                            *q = if v >= half_mod { v as i64 - (mask as i64) - 1 } else { v as i64 };
                        }
                    }
                    Ok(())
                })();
                wipe(rs); // one-time pad: never kept
                out
            })
        };
        key.zeroize();
        let res = res.and_then(|()| match dup {
            Some(d) => crate::noise::dup_keys(pl, t, n, nb, d.cols, d.shift, d.mult, d.groups, d.per, keys)
                .map_err(CoreErr::Msg),
            None => Ok(()),
        });
        if let Err(e) = res {
            wipe(pl);
            wipe(xh);
            wipe(xq);
            ticket.wipe_all(); // the rows stay consumed; their W r is zeroed now
            return Err(e);
        }
        Ok(Some(ticket))
    }
}

impl Drop for TripLease {
    fn drop(&mut self) {
        let mut g = self.core.lock();
        g.detached = true;
        g.key.zeroize();
        g.pieces.clear();
        g.drop_ahead();
    }
}

/// One trip's gateway-side state between the two native calls: which `W r` rows it unmasks with,
/// the activation scales and (when the check was drawn) the Freivalds input half. Single use;
/// dropping it unfinished zeroes its `W r` rows and every secret it holds.
#[pyclass(module = "pgw_fast")]
pub struct TripTicket {
    core: Arc<LeaseCore>,
    /// The trip's `W r` rows, in order (one segment unless the rows span two chunks).
    segs: Vec<Seg>,
    rows: usize,
    m: usize,
    k: u32,
    direct: bool,
    sx: Vec<f64>,
    rhs: Vec<u64>,
    c: usize,
    u: Option<Py<PyAny>>,
    done: bool,
    /// Check field of the lease (prime: the output half runs synchronously inside `finish`).
    field: u64,
}

impl TripTicket {
    /// Zero this trip's `W r` rows in the chunk, `sx` and `rhs`; release the chunk.
    pub(crate) fn wipe_all(&mut self) {
        let nb = (self.k / 8) as usize;
        for sg in self.segs.drain(..) {
            let a = sg.elem_off * nb;
            let b = (a + sg.rows * self.m * nb).min(sg.hold.len);
            if b > a {
                // SAFETY: in bounds of the held chunk; only this ticket touches these bytes.
                unsafe { zero_bytes(sg.hold.ptr.add(a), b - a) };
            }
        }
        wipe(&mut self.sx);
        wipe(&mut self.rhs);
        self.done = true;
    }
}

impl Drop for TripTicket {
    fn drop(&mut self) {
        self.wipe_all();
    }
}

#[pymethods]
impl TripTicket {
    /// Rows of the trip.
    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    /// True if the Freivalds check was drawn for this trip.
    #[getter]
    fn checked(&self) -> bool {
        self.c > 0
    }

    /// The lane call and the finish of this trip in one call (``PGW_TRIP_DRIVER``, opt-in): send
    /// ``body`` (this trip's payload) with the A1 header on ``channel`` (a ``pgw_fast.TripChannel`` the
    /// caller acquired), read the reply straight into a fresh bytes object (no zero-fill, no copy) and run
    /// :meth:`finish` on it, the GIL released for both.
    ///
    /// Returns ``(lane_status, compute_us, error, result)``. ``result`` is ``finish``'s tuple when the lane
    /// served the trip (``lane_status == 0``); otherwise it is ``None``, the ticket is untouched and the
    /// caller sends the same payload through its fallback client and finishes with :meth:`finish`
    /// (``lane_status``: the lane's refusal status, or 255 with ``error`` naming a channel error, after
    /// which the channel is closed).
    #[cfg(unix)]
    #[pyo3(signature = (channel, ring, linear, request_id, deadline_ms, body, timeout_s, scale, bias, bound,
                        defer = false, threads = 1, split = 0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn lane_finish<'py>(
        &mut self,
        py: Python<'py>,
        channel: &Bound<'py, crate::lane::TripChannel>,
        ring: u8,
        linear: u16,
        request_id: &[u8],
        deadline_ms: u32,
        body: &Bound<'py, PyBytes>,
        timeout_s: f64,
        scale: &Bound<'py, PyAny>,
        bias: Option<&Bound<'py, PyAny>>,
        bound: u64,
        defer: bool,
        threads: usize,
        split: usize,
    ) -> PyResult<(u8, u32, Option<&'static str>, Option<(u8, usize, Bound<'py, PyArray2<f32>>, Option<TripCheck>)>)>
    {
        if self.done {
            return Err(verr("trip ticket already finished or wiped (single use)"));
        }
        let rid: [u8; 16] = request_id.try_into().map_err(|_| verr("request_id must be 16 bytes"))?;
        if !(timeout_s.is_finite() && timeout_s > 0.0 && timeout_s < 1e7) {
            return Err(verr("timeout_s must be a positive number of seconds"));
        }
        let rows = u16::try_from(self.rows).map_err(|_| verr("too many rows for the lane"))?;
        let src = body.as_bytes();
        let body_len = u32::try_from(src.len()).map_err(|_| verr("body too large"))?;
        let olen = self.rows * self.m * (self.k / 8) as usize;
        // SAFETY: a fresh, unshared bytes object; written by the channel below before anyone sees it (on a
        // failed trip it is dropped unseen).
        let reply: Bound<'py, PyBytes> = unsafe {
            let p = pyo3::ffi::PyBytes_FromStringAndSize(std::ptr::null(), olen as pyo3::ffi::Py_ssize_t);
            Bound::from_owned_ptr_or_err(py, p)?.downcast_into_unchecked()
        };
        // SAFETY: exclusive (just created, not returned yet); `src` is another object (an immutable payload).
        let dst: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(reply.as_bytes().as_ptr() as *mut u8, olen) };
        let h = crate::lane::RequestHeader { ring, linear, rows, deadline_ms, body_len, request_id: rid };
        let to = std::time::Duration::from_secs_f64(timeout_s);
        let ch = channel.get();
        let res = py.allow_threads(|| ch.trip_raw(&h, src, dst, to));
        match res {
            Ok((0, cus)) => {
                let fin = self.finish(py, reply.as_any(), scale, bias, bound, defer, threads, split)?;
                Ok((0, cus, None, Some(fin)))
            }
            Ok((st, cus)) => Ok((st, cus, None, None)),
            Err(e) => {
                let name = match e {
                    crate::lane::LaneError::Io(_) => "OSError",
                    crate::lane::LaneError::Protocol(_) => "ProtocolError",
                    crate::lane::LaneError::Refused(_) => "Refused",
                    crate::lane::LaneError::Closed => "Closed",
                };
                Ok((255, 0, Some(name), None))
            }
        }
    }

    /// Zero everything this ticket holds (a trip that will not be finished). Idempotent.
    fn wipe(&mut self) {
        self.wipe_all();
    }

    /// The per-trip call after the lane call: unmask ``y_hat`` with this trip's ``W r`` rows, range
    /// check, Freivalds (synchronously, or with ``defer`` on the native helper thread), rescale and
    /// add the bias; ``W r``, ``sx`` and the input half are zeroed.
    ///
    /// Returns ``(status, bad_rows, out, pending)`` with the statuses of ``trip_finish_into`` (3:
    /// the reservation closed during the trip); ``out`` is float32 ``(rows, m)`` (zeros unless the
    /// status is 0) and ``pending`` the deferred check's ``TripCheck`` (or ``None``).
    #[pyo3(signature = (y_hat, scale, bias, bound, defer = false, threads = 1, split = 0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        y_hat: &Bound<'py, PyAny>,
        scale: &Bound<'py, PyAny>,
        bias: Option<&Bound<'py, PyAny>>,
        bound: u64,
        defer: bool,
        threads: usize,
        split: usize,
    ) -> PyResult<(u8, usize, Bound<'py, PyArray2<f32>>, Option<TripCheck>)> {
        if self.done {
            return Err(verr("trip ticket already finished or wiped (single use)"));
        }
        let (rows, m, k) = (self.rows, self.m, self.k);
        let nb = (k / 8) as usize;
        if self.core.closed() {
            self.wipe_all();
            return Ok((3, 0, PyArray2::<f32>::zeros(py, [rows, m], false), None));
        }
        // SAFETY: every element is written by the finish kernel (outputs, or zeros on any failure).
        let out = unsafe { PyArray2::<f32>::new(py, [rows, m], false) };
        let yb: Bound<'py, PyAny> = y_hat.clone();
        let ys: &[u8] = if let Ok(b) = yb.downcast::<PyBytes>() {
            b.as_bytes()
        } else if let Ok(a) = yb.downcast::<PyArrayDyn<u8>>() {
            if !a.is_c_contiguous() {
                self.wipe_all();
                return Err(verr("y_hat must be C-contiguous"));
            }
            // SAFETY: read-only use while the array (kept alive by `yb`) is not written by us.
            unsafe { a.as_slice() }.map_err(|e| verr(e.to_string()))?
        } else {
            self.wipe_all();
            return Err(verr("y_hat must be bytes or a uint8 numpy array"));
        };
        if ys.len() != rows * m * nb {
            self.wipe_all();
            return Err(verr(format!("ring payload has {} bytes, expected {}", ys.len(), rows * m * nb)));
        }
        let sc_a = scale
            .downcast::<PyArray1<f32>>()
            .map_err(|_| verr("scale must be a 1-D float32 array"))?
            .try_readonly()
            .map_err(|e| verr(format!("scale: {e}")))?;
        let sc = sc_a.as_slice().map_err(|e| verr(e.to_string()))?;
        let bias_a = match bias {
            Some(b) => Some(
                b.downcast::<PyArray1<f32>>()
                    .map_err(|_| verr("bias must be a 1-D float32 array"))?
                    .try_readonly()
                    .map_err(|e| verr(format!("bias: {e}")))?,
            ),
            None => None,
        };
        let bs = match &bias_a {
            Some(b) => Some(b.as_slice().map_err(|e| verr(e.to_string()))?),
            None => None,
        };
        if sc.len() != m || bs.is_some_and(|b| b.len() != m) {
            self.wipe_all();
            return Err(verr(format!("scale/bias must have {m} entries")));
        }
        let mut pending: Option<TripCheck> = None;
        let c = self.c;
        let ua = match &self.u {
            Some(u) => Some(RoRing::extract(u.bind(py), "u", k)?),
            None => None,
        };
        let prime = c > 0 && check_iface::sync_only(self.field);
        let defer = defer && !prime;
        let mut mode = FinishMode::Sync;
        if c > 0 && defer {
            let u = self.u.as_ref().map(|u| u.bind(py).clone());
            if let Some(u) = u {
                let mut rhs = std::mem::take(&mut self.rhs);
                match check_iface::submit(py, &yb, &u, &mut rhs, rows, c, k) {
                    Ok(p) => {
                        pending = Some(p);
                        mode = FinishMode::Submitted;
                    }
                    Err(e) => {
                        wipe(&mut rhs);
                        self.wipe_all();
                        return Err(e);
                    }
                }
            }
        }
        let u_src = match &ua {
            Some(a) => Some(a.src("u")?.0),
            None => None,
        };
        // SAFETY: a fresh array, not shared with Python until returned.
        let os: &mut [f32] = unsafe { out.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
        let res = py.allow_threads(|| self.finish_core(ys, u_src, sc, bs, bound, mode, threads, split, os));
        self.u = None;
        match res {
            // a failed status with a pending check: the caller waits for it before raising
            Ok((status, bad, _none)) => Ok((status, bad, out, pending)),
            Err(FinishErr::Msg(e)) => {
                drop(pending);
                Err(verr(e))
            }
            Err(FinishErr::Internal) => {
                drop(pending);
                Err(PyRuntimeError::new_err("TripTicket.finish: internal error in a finish worker (outputs zeroed)"))
            }
        }
    }
}

/// How [`TripTicket::finish_core`] runs the Freivalds output half of a checked trip.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinishMode {
    /// Synchronously (ring: fused in the finish kernel; prime: before the unmask).
    Sync,
    /// Ring checks on the native helper thread ([`crate::trip::freivalds_submit_raw`]); prime: sync.
    NativeDefer,
    /// The caller already submitted the output half (the Python `TripCheck`): no check here.
    Submitted,
}

/// Why [`TripTicket::finish_core`] failed (the ticket is wiped, the outputs zeroed).
pub(crate) enum FinishErr {
    Msg(String),
    Internal,
}

impl TripTicket {
    pub(crate) fn is_checked(&self) -> bool {
        self.c > 0
    }
    pub(crate) fn attach_u(&mut self, u: Py<PyAny>) {
        if self.c > 0 {
            self.u = Some(u);
        }
    }

    /// The GIL-free core of `finish`: unmask `ys` (`rows * m * K/8` bytes) with this trip's `W r`,
    /// range check, Freivalds per `mode` (`u`: the lease's check vectors, required when checked),
    /// rescale + bias into `os` (`rows * m`, zeros unless the status is 0); `W r`, `sx`, `rhs` zeroed.
    /// Returns `(status, bad_rows, pending)` (`pending`: [`FinishMode::NativeDefer`] on a checked ring
    /// trip; it reads `ys` and `u`, which must stay valid until it was waited for or dropped).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_core(
        &mut self,
        ys: &[u8],
        u: Option<Src<'_>>,
        sc: &[f32],
        bs: Option<&[f32]>,
        bound: u64,
        mode: FinishMode,
        threads: usize,
        split: usize,
        os: &mut [f32],
    ) -> Result<(u8, usize, Option<crate::trip::RawPending>), FinishErr> {
        let threads = if split >= 2 { split } else { threads };
        if self.done {
            wipe(os);
            return Err(FinishErr::Msg("trip ticket already finished or wiped (single use)".into()));
        }
        let (rows, m, k) = (self.rows, self.m, self.k);
        let nb = (k / 8) as usize;
        if self.core.closed() {
            self.wipe_all();
            wipe(os);
            return Ok((3, 0, None));
        }
        if ys.len() != rows * m * nb || os.len() != rows * m {
            self.wipe_all();
            wipe(os);
            return Err(FinishErr::Msg(format!("ring payload has {} bytes, expected {}", ys.len(), rows * m * nb)));
        }
        if sc.len() != m || bs.is_some_and(|b| b.len() != m) {
            self.wipe_all();
            wipe(os);
            return Err(FinishErr::Msg(format!("scale/bias must have {m} entries")));
        }
        if self.segs.iter().map(|s| s.rows).sum::<usize>() != rows {
            self.wipe_all();
            wipe(os);
            return Err(FinishErr::Msg("trip ticket has no W r rows".into()));
        }
        let c = self.c;
        let prime = c > 0 && check_iface::sync_only(self.field);
        if c > 0 && mode != FinishMode::Submitted && u.is_none() {
            self.wipe_all();
            wipe(os);
            return Err(FinishErr::Msg("trip ticket lost its check vectors".into()));
        }
        let segs = std::mem::take(&mut self.segs);
        let mut rhs = std::mem::take(&mut self.rhs);
        let mut pending = None;
        if c > 0 && mode == FinishMode::NativeDefer && !prime {
            if let Some(us) = u {
                // SAFETY: the caller keeps `ys` and `u` alive and unchanged until the handle is done.
                pending = Some(unsafe { crate::trip::freivalds_submit_raw(ys, us, std::mem::take(&mut rhs), c, m, k) });
            }
        }
        let check = match (c > 0 && mode == FinishMode::Sync && !prime, u) {
            (true, Some(us)) => check_iface::split(us, &rhs, c),
            _ => CheckMode::Off,
        };
        // prime format: the output half reads W r before the finish kernel unmasks (and wipes) it
        let prime_u = if prime && mode != FinishMode::Submitted { u } else { None };
        let sx = std::mem::take(&mut self.sx);
        let direct = self.direct && segs.len() == 1;
        let res = WR_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let n_el = rows * m;
            let wrb = if direct {
                // SAFETY: the ticket's rows of the held chunk, 4-byte aligned u32 words (K = 32),
                // read and zeroed only by this call.
                unsafe { WrBuf::new(segs[0].hold.ptr.add(segs[0].elem_off * 4), n_el, false, true) }
            } else {
                if scratch.len() < n_el {
                    scratch.resize(n_el, 0);
                }
                let mut d0 = 0usize;
                for sg in &segs {
                    let cnt = sg.rows * m;
                    let base = sg.elem_off * nb;
                    for (i, d) in scratch[d0..d0 + cnt].iter_mut().enumerate() {
                        let mut w = [0u8; 8];
                        for (j, wb) in w.iter_mut().enumerate().take(nb) {
                            // SAFETY: in bounds of the held chunk (validated at lease creation);
                            // only this ticket touches these bytes.
                            let p = unsafe { sg.hold.ptr.add(base + i * nb + j) };
                            *wb = unsafe { std::ptr::read(p) };
                            unsafe { std::ptr::write_volatile(p, 0) };
                        }
                        *d = u64::from_le_bytes(w);
                    }
                    d0 += cnt;
                }
                // SAFETY: the scratch is exclusively ours for this call; zeroed by the kernel.
                unsafe { WrBuf::new(scratch.as_mut_ptr().cast::<u8>(), n_el, true, true) }
            };
            let args = FinishArgs { y_hat: ys, wr: wrb, sx: &sx, scale: sc, bias: bs, k, bound, check };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Some(us) = prime_u {
                    let wsrc = if direct {
                        // SAFETY: the same n_el aligned u32 words WrBuf reads below; nothing writes
                        // them until this read-only view is gone.
                        Src::U32(unsafe {
                            std::slice::from_raw_parts(segs[0].hold.ptr.add(segs[0].elem_off * 4).cast::<u32>(), n_el)
                        })
                    } else {
                        // SAFETY: the scratch words filled above (n_el of them), read only here.
                        Src::U64(unsafe { std::slice::from_raw_parts(scratch.as_ptr(), n_el) })
                    };
                    let (st, bad) = check_iface::prime_output(ys, wsrc, us, &rhs, c, m, k, bound, split);
                    if st != crate::finish::STATUS_OK {
                        // never unmasked: zero W r and the outputs (fail closed)
                        wrb.wipe_all();
                        wipe(&mut *os);
                        return (st, bad);
                    }
                }
                finish_into(&args, &mut *os, threads)
            }));
            if r.is_err() {
                wipe(&mut *os);
                wrb.wipe_all();
            }
            if !direct {
                wipe(&mut scratch[..n_el]);
            }
            r
        });
        let mut sx = sx;
        wipe(&mut sx);
        wipe(&mut rhs);
        drop(segs); // our rows were zeroed by the kernel (or copied out and zeroed above)
        self.done = true;
        match res {
            Ok((status, bad)) => Ok((status, bad, pending)),
            Err(_) => {
                drop(pending);
                Err(FinishErr::Internal)
            }
        }
    }
}
