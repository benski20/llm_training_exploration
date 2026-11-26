//! Native decode driver (`PGW_DECODE_NATIVE`, opt-in): the decoder layers of one decode pass of one
//! sequence (a few rows) run in one native call with the GIL released, Python called once per call.
//!
//! Between the masked trips the pinned native float formulation (`float_ops = "native"`, `layer.rs`,
//! `attn.rs`: the same kernels and the same bits as `TransformerModel._layer_native`) runs; each trip
//! is the native per-trip path of `tripnative.rs` unchanged: [`TripLease::prepare_core`] (rows taken
//! once, `r`, quantize + mask + pack, Freivalds input half, the noise monitor's fingerprints), the
//! duplicate-row check on the shared [`DupHistory`] *before* the payload leaves, the lane trip on the
//! native [`TripChannel`] (A1 header, request id in the uuid4 format), and [`TripTicket::finish_core`]
//! (range check, Freivalds ring synchronously / on the helper thread, p31 synchronously before the
//! unmask; unmask, rescale, `W r` wiped).
//!
//! Anything the driver does not handle itself goes back to Python at a trip boundary with nothing
//! lost: a trip the lane does not serve (no native channel, a refusal, a broken connection) is handed
//! back prepared (payload, ticket, request id) for the ordinary fallback, after which Python injects
//! the trip's output and the driver continues; failures return a code at the failing trip (outputs
//! zeroed, tickets wiped). Deferred ring checks are joined before the next payload is built (as
//! `MaskedBackend._prepare` joins them) and always before the call returns. Sampled noise-monitor
//! values (`x_hat`, `xq`) are handed back per call for the statistics.
//!
//! The pass state (residual stream, the next trip's input) lives here between calls and is zeroed when
//! the pass ends ([`DecodeDriver::end`]) or a new one begins.

use crate::attn::{auto_threads, rope_attend, Dims};
use crate::finish::{wipe, RoRing, Src};
use crate::lane::{LaneError, RequestHeader, TripChannel};
use crate::layer::{add_rows, rms_rows, silu_rows};
use crate::noise::DupHistory;
use crate::tripnative::{CoreErr, DupIn, FinishErr, FinishMode, PrepIo, TripLease, TripTicket};
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::time::{Duration, Instant};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodeDriver>()?;
    m.add_function(pyo3::wrap_pyfunction!(decode_phase_ns, m)?)?;
    // DecodeDriver.run takes flag bit 3 (hold a prepared trip; code 8) and resumes it on the next run
    m.add("DECODE_HOLD", 1u32)?;
    Ok(())
}

/// Cumulative ns of the driver's local phases per linear slot of a layer (qkv, o, gate_up, down):
/// `[prepare x4, finish x4, float ops x4, trips x4]` (process-wide counters for benchmarks; timing only).
static PHASE_NS: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

fn phase_add(i: usize, v: u64) {
    PHASE_NS[i].fetch_add(v, std::sync::atomic::Ordering::Relaxed);
}

/// The driver's phase counters (see [`PHASE_NS`]); `reset` zeroes them after reading.
#[pyfunction]
#[pyo3(signature = (reset = false))]
fn decode_phase_ns(reset: bool) -> Vec<u64> {
    PHASE_NS
        .iter()
        .map(|a| if reset { a.swap(0, std::sync::atomic::Ordering::Relaxed) } else { a.load(std::sync::atomic::Ordering::Relaxed) })
        .collect()
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

const F_CHECK: u8 = 1;
const F_SAMPLED: u8 = 2;
const F_DEFER: u8 = 4;
/// Hold the trip after its prepare (with its sampled values): Python runs the noise statistics, then a
/// later :meth:`DecodeDriver.run` from this trip sends the held payload (see [`Held`]).
const F_HOLD: u8 = 8;

/// Return codes of [`DecodeDriver::run`].
const DONE: u8 = 0;
const HANDBACK: u8 = 1;
const FAILED_STATUS: u8 = 2;
const PREPARE_ERROR: u8 = 3;
const DUPLICATE: u8 = 4;
const DEFERRED_FAILED: u8 = 5;
const NOT_READY: u8 = 6;
const FINISH_ERROR: u8 = 7;
const HELD: u8 = 8;

struct DupCfg {
    cols: Vec<i64>,
    shift: u32,
    mult: Vec<u64>,
    groups: usize,
    per: usize,
    limit: usize,
}

struct LinCfg {
    n: usize,
    m: usize,
    scale: Vec<f32>,
    bias: Option<Vec<f32>>,
    bound: u64,
    threads_p: usize,
    threads_f: usize,
    split_p: usize,
    split_f: usize,
    dup: Option<DupCfg>,
}

#[derive(Default)]
struct Pass {
    t: usize,
    start: usize,
    /// Trip index (`layer * 4 + step`) of the next trip.
    pos: usize,
    active: bool,
    h: Vec<f32>,
    h2: Vec<f32>,
    /// The next trip's input.
    x: Vec<f32>,
    /// A trip's output.
    y: Vec<f32>,
    /// A trip's reply (`y_hat`).
    reply: Vec<u8>,
    payload: Vec<u8>,
    keys: Vec<u64>,
    /// A trip prepared and held (``F_HOLD``): its payload is `payload`.
    held: Option<Held>,
}

/// A prepared trip kept in the driver between two runs (inline noise statistics): the ticket (rows taken,
/// pads derived), its request id; the payload stays in [`Pass::payload`]. The next run from `pos` sends
/// exactly these bytes with this id (no second prepare: one draw of rows, one request).
struct Held {
    pos: usize,
    ticket: TripTicket,
    rid: [u8; 16],
}

impl Pass {
    fn wipe(&mut self) {
        if let Some(mut h) = self.held.take() {
            h.ticket.wipe_all();
            h.rid.fill(0);
        }
        wipe(&mut self.h);
        wipe(&mut self.h2);
        wipe(&mut self.x);
        wipe(&mut self.y);
        wipe(&mut self.reply);
        wipe(&mut self.payload);
        self.active = false;
    }
}

/// Counters of one [`DecodeDriver::run`] call (handed to Python's statistics).
#[derive(Default, Clone, Copy)]
struct Stats {
    trips: u64,
    rows: u64,
    bytes_up: u64,
    bytes_down: u64,
    checked: u64,
    skipped: u64,
    deferred: u64,
    deferred_ok: u64,
    remote_ns: u64,
    local_ns: u64,
    lane_trips: u64,
    lane_sent: u64,
    lane_recv: u64,
    last_rtt_ns: u64,
    dup_rows: u64,
    dups: u64,
}

/// What a run returns besides the stats (built into Python objects with the GIL).
enum Outcome {
    Done,
    Handback { ticket: TripTicket, payload: Vec<u8>, rid: [u8; 16], lane_status: i32, lane_err: Option<&'static str> },
    Status(u8, usize),
    Prepare(&'static str, String),
    Duplicate(usize, usize),
    Deferred(u8, usize),
    NotReady,
    FinishErr(String),
    Held,
}

/// The native decode driver of one model (see the module docs). Not thread-safe across passes: one
/// pass at a time (the engine's plaintext thread).
#[pyclass(module = "pgw_fast")]
pub struct DecodeDriver {
    hidden: usize,
    nh: usize,
    kvh: usize,
    hd: usize,
    inter: usize,
    attn_scale: f64,
    eps: f64,
    attn_threads: usize,
    attn_block: usize,
    kv_bf16: bool,
    layers: usize,
    in_norm: Vec<Vec<f32>>,
    post_norm: Vec<Vec<f32>>,
    lin: Vec<Option<LinCfg>>,
    pass: Pass,
}

fn f32_vec(a: &Bound<'_, PyArray1<f32>>, name: &str) -> PyResult<Vec<f32>> {
    let ro = a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?;
    Ok(ro.as_slice().map_err(|_| verr(format!("{name} must be contiguous")))?.to_vec())
}

impl DecodeDriver {
    /// Input width of trip step `s`.
    fn step_n(&self, s: usize) -> usize {
        match s {
            0 | 2 => self.hidden,
            1 => self.nh * self.hd,
            _ => self.inter,
        }
    }
    /// Output width of trip step `s`.
    fn step_m(&self, s: usize) -> usize {
        match s {
            0 => (self.nh + 2 * self.kvh) * self.hd,
            1 | 3 => self.hidden,
            _ => 2 * self.inter,
        }
    }

    /// The float operations after trip `pos` (its output in `y`).
    #[allow(clippy::too_many_arguments)]
    fn after_trip<T: crate::attn::Kv>(
        cfg: &DriverView<'_>,
        p: &mut Pass,
        pos: usize,
        kv: &mut Option<(&mut [T], &mut [T])>,
        cos: &[f32],
        sin: &[f32],
    ) {
        let (l, s) = (pos / 4, pos % 4);
        let t = p.t;
        let hidden = cfg.hidden;
        match s {
            0 => {
                let (nh, kvh, hd) = (cfg.nh, cfg.kvh, cfg.hd);
                let ycols = (nh + 2 * kvh) * hd;
                if let Some((kc, vc)) = kv.as_mut() {
                    let lsz = cfg.cap * kvh * hd;
                    let kl = &mut kc[l * lsz..(l + 1) * lsz];
                    let vl = &mut vc[l * lsz..(l + 1) * lsz];
                    let dm = Dims { nh, kvh, hd, scale: cfg.attn_scale };
                    let threads = auto_threads(t, p.start, cfg.attn_threads);
                    rope_attend(&p.y[..t * ycols], cos, sin, kl, vl, t, p.start, dm, &mut p.x[..t * nh * hd], threads,
                                cfg.attn_block);
                }
            }
            1 => {
                add_rows(&p.h[..t * hidden], &p.y[..t * hidden], &mut p.h2[..t * hidden]);
                rms_rows(&p.h2[..t * hidden], cfg.post_norm[l], cfg.eps, &mut p.x[..t * hidden], hidden);
            }
            2 => {
                let inter = cfg.inter;
                silu_rows(&p.y[..t * 2 * inter], &mut p.x[..t * inter], inter);
            }
            _ => {
                add_rows(&p.h2[..t * hidden], &p.y[..t * hidden], &mut p.h[..t * hidden]);
                if l + 1 < cfg.layers {
                    rms_rows(&p.h[..t * hidden], cfg.in_norm[l + 1], cfg.eps, &mut p.x[..t * hidden], hidden);
                }
            }
        }
    }
}

/// The read-only configuration a run needs, borrowed apart from the pass state.
struct DriverView<'a> {
    hidden: usize,
    nh: usize,
    kvh: usize,
    hd: usize,
    inter: usize,
    attn_scale: f64,
    eps: f64,
    attn_threads: usize,
    attn_block: usize,
    layers: usize,
    cap: usize,
    in_norm: Vec<&'a [f32]>,
    post_norm: Vec<&'a [f32]>,
}

/// Per-trip inputs of a run, borrowed from Python objects held by the caller.
struct TripIn<'a> {
    lease: &'a TripLease,
    lin: &'a LinCfg,
    uw: Option<Src<'a>>,
    u: Option<Src<'a>>,
    flags: u8,
    rid: [u8; 16],
    lin_ix: u16,
}

struct LaneIn<'a> {
    ch: &'a TripChannel,
    ring: u8,
    deadline_ms: u32,
    timeout: Duration,
}

fn src_prefetch(s: Option<&Src<'_>>) {
    match s {
        Some(Src::U64(v)) => crate::lane::wait_prefetch::add_slice(v),
        Some(Src::U32(v)) => crate::lane::wait_prefetch::add_slice(v),
        Some(Src::Bytes(v)) => crate::lane::wait_prefetch::add_slice(v),
        None => {}
    }
}

fn ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// The trip loop (GIL released). `trips[i]` is trip `p.pos` at entry + i.
#[allow(clippy::too_many_arguments)]
fn run_loop<T: crate::attn::Kv>(
    cfg: &DriverView<'_>,
    p: &mut Pass,
    trips: &[TripIn<'_>],
    end: usize,
    lane: Option<&LaneIn<'_>>,
    dup_hist: Option<&DupHistory>,
    mut kv: Option<(&mut [T], &mut [T])>,
    cos: &[f32],
    sin: &[f32],
    sampled: &mut Vec<(usize, Vec<i64>, Vec<i64>)>,
    st: &mut Stats,
) -> Outcome {
    let t = p.t;
    let base = p.pos;
    let mut pending: Option<(crate::trip::RawPending, usize)> = None;
    // join a deferred check; Some((status, bad, pos)) when it failed
    let join = |pend: &mut Option<(crate::trip::RawPending, usize)>, st: &mut Stats| -> Option<(u8, usize, usize)> {
        let (h, at) = pend.take()?;
        let (s, bad) = h.wait();
        drop(h);
        if s == 0 {
            st.deferred_ok += 1;
            None
        } else {
            Some((s, bad, at))
        }
    };
    while p.pos < end {
        let pos = p.pos;
        let tr = &trips[pos - base];
        let lin = tr.lin;
        let (n, m) = (lin.n, lin.m);
        // the previous trip's deferred check: joined before any row is taken or payload built
        if let Some((s, bad, at)) = join(&mut pending, st) {
            p.pos = at;
            return Outcome::Deferred(s, bad);
        }
        let t0 = Instant::now();
        let nb = (tr.lease.k_bits() / 8) as usize;
        let resumed = match p.held.take() {
            Some(h) if h.pos == pos && p.payload.len() == t * n * nb => Some(h),
            Some(mut h) => {
                // never reached through run() (it checks the position first): fail closed
                h.ticket.wipe_all();
                wipe(&mut p.payload);
                return Outcome::Prepare("invalid", "held trip out of step".into());
            }
            None => None,
        };
        let (mut ticket, rid) = if let Some(h) = resumed {
            (h.ticket, h.rid)
        } else {
            p.payload.resize(t * n * nb, 0);
            let sampled_now = tr.flags & F_SAMPLED != 0;
            let (mut xh, mut xq) = if sampled_now { (vec![0i64; t * n], vec![0i64; t * n]) } else { (Vec::new(), Vec::new()) };
            let groups = lin.dup.as_ref().map_or(0, |d| d.groups);
            p.keys.resize(t * groups, 0);
            let dup_in = lin.dup.as_ref().map(|d| DupIn { cols: &d.cols, shift: d.shift, mult: &d.mult, groups: d.groups, per: d.per });
            let uw = if tr.flags & F_CHECK != 0 { tr.uw } else { None };
            let split_p = if lin.split_p >= 2 { lin.split_p } else { 0 };
            let got = tr.lease.prepare_core(
                &p.x[..t * n],
                t,
                uw,
                lin.threads_p,
                split_p,
                PrepIo { payload: &mut p.payload, xh: &mut xh, xq: &mut xq, keys: &mut p.keys },
                dup_in.as_ref(),
            );
            let mut ticket = match got {
                Ok(Some(tk)) => tk,
                Ok(None) => return Outcome::NotReady,
                Err(CoreErr::NonFinite) => {
                    return Outcome::Prepare("nonfinite", "non-finite activation values cannot be quantized".into())
                }
                Err(CoreErr::Prep(e)) => {
                    return match e {
                        crate::prepare::PrepareError::Overflow(msg) => Outcome::Prepare("overflow", msg),
                        crate::prepare::PrepareError::Invalid(msg) => Outcome::Prepare("invalid", msg),
                    }
                }
                Err(CoreErr::Msg(msg)) => return Outcome::Prepare("invalid", msg),
            };
            // the monitor's every-call check, before the payload leaves
            if let (Some(d), Some(hist)) = (lin.dup.as_ref(), dup_hist) {
                let dups = hist.record_raw(&p.keys[..t * d.groups], t, d.groups, d.limit);
                st.dup_rows += t as u64;
                if dups > 0 {
                    st.dups += dups as u64;
                    ticket.wipe_all();
                    wipe(&mut xq);
                    wipe(&mut xh);
                    return Outcome::Duplicate(dups, t);
                }
            }
            if sampled_now {
                sampled.push((pos, xh, xq));
            }
            let mut rid = tr.rid;
            rid[6] = (rid[6] & 0x0F) | 0x40;
            rid[8] = (rid[8] & 0x3F) | 0x80;
            let t1 = Instant::now();
            st.local_ns += ns(t1 - t0);
            phase_add(pos % 4, ns(t1 - t0));
            phase_add(12 + pos % 4, 1);
            if tr.flags & F_HOLD != 0 {
                // the payload stays in `p.payload`; nothing has left
                p.held = Some(Held { pos, ticket, rid });
                return Outcome::Held;
            }
            (ticket, rid)
        };
        let t1 = Instant::now();
        let lane = match lane {
            Some(l) => l,
            None => {
                return Outcome::Handback {
                    ticket,
                    payload: std::mem::take(&mut p.payload),
                    rid,
                    lane_status: -1,
                    lane_err: None,
                }
            }
        };
        let olen = t * m * nb;
        p.reply.resize(olen, 0);
        let hdr = RequestHeader {
            ring: lane.ring,
            linear: tr.lin_ix,
            rows: t as u16,
            deadline_ms: lane.deadline_ms,
            body_len: p.payload.len() as u32,
            request_id: rid,
        };
        if crate::lane::wait_prefetch::enabled() {
            // read right after the reply: this trip's W r and output check vector, the next trip's
            // input-half vector (prefetched while the provider computes; timing only)
            ticket.prefetch_wr();
            src_prefetch(tr.u.as_ref());
            if let Some(nx) = trips.get(pos + 1 - base) {
                src_prefetch(nx.uw.as_ref());
            }
        }
        let res = lane.ch.trip_raw(&hdr, &p.payload, &mut p.reply, lane.timeout);
        crate::lane::wait_prefetch::clear();
        let t2 = Instant::now();
        match res {
            Ok((0, _cus)) => {}
            Ok((status, _)) => {
                st.last_rtt_ns = ns(t2 - t1);
                return Outcome::Handback {
                    ticket,
                    payload: std::mem::take(&mut p.payload),
                    rid,
                    lane_status: i32::from(status),
                    lane_err: None,
                };
            }
            Err(e) => {
                let name = match e {
                    LaneError::Io(_) => "OSError",
                    LaneError::Protocol(_) => "ProtocolError",
                    LaneError::Refused(_) => "Refused",
                    LaneError::Closed => "Closed",
                };
                return Outcome::Handback {
                    ticket,
                    payload: std::mem::take(&mut p.payload),
                    rid,
                    lane_status: 255,
                    lane_err: Some(name),
                };
            }
        }
        st.lane_trips += 1;
        st.lane_sent += (crate::lane::REQUEST_LEN + p.payload.len()) as u64;
        st.lane_recv += (crate::lane::REPLY_LEN + olen) as u64;
        st.last_rtt_ns = ns(t2 - t1);
        st.remote_ns += ns(t2 - t1);
        st.bytes_up += p.payload.len() as u64;
        st.bytes_down += olen as u64;
        // finish
        let checked = ticket.is_checked();
        let mode = if tr.flags & F_DEFER != 0 { FinishMode::NativeDefer } else { FinishMode::Sync };
        let split_f = if lin.split_f >= 2 { lin.split_f } else { 0 };
        p.y.resize(p.y.len().max(t * m), 0.0);
        let fin = ticket.finish_core(&p.reply, tr.u, &lin.scale, lin.bias.as_deref(), lin.bound, mode, lin.threads_f,
                                     split_f, &mut p.y[..t * m]);
        drop(ticket);
        let tf = Instant::now();
        phase_add(4 + pos % 4, ns(tf - t2));
        st.trips += 1;
        st.rows += t as u64;
        match fin {
            Ok((0, _, pend)) => {
                match pend {
                    Some(h) => {
                        st.deferred += 1;
                        pending = Some((h, pos));
                    }
                    None => {
                        if checked {
                            st.checked += 1;
                        }
                    }
                }
                if !checked {
                    st.skipped += 1;
                }
            }
            Ok((status, bad, pend)) => {
                if let Some(h) = pend {
                    let _ = h.wait(); // its verdict no longer matters (it reads the reply)
                }
                p.pos = pos;
                return Outcome::Status(status, bad);
            }
            Err(FinishErr::Msg(msg)) => {
                p.pos = pos;
                return Outcome::FinishErr(msg);
            }
            Err(FinishErr::Internal) => {
                p.pos = pos;
                return Outcome::FinishErr("internal error in a finish worker (outputs zeroed)".into());
            }
        }
        DecodeDriver::after_trip(cfg, p, pos, &mut kv, cos, sin);
        p.pos = pos + 1;
        let te = Instant::now();
        phase_add(8 + pos % 4, ns(te - tf));
        st.local_ns += ns(te - t2);
    }
    if let Some((s, bad, at)) = join(&mut pending, st) {
        p.pos = at;
        return Outcome::Deferred(s, bad);
    }
    Outcome::Done
}

#[pymethods]
impl DecodeDriver {
    /// ``DecodeDriver(hidden, nh, kvh, hd, inter, attn_scale, eps, attn_threads, attn_block, kv_bf16, layers)``.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden: usize,
        nh: usize,
        kvh: usize,
        hd: usize,
        inter: usize,
        attn_scale: f64,
        eps: f64,
        attn_threads: usize,
        attn_block: usize,
        kv_bf16: bool,
        layers: usize,
    ) -> PyResult<Self> {
        if hidden == 0 || nh == 0 || kvh == 0 || nh % kvh != 0 || hd == 0 || hd % 2 != 0 || inter == 0 || layers == 0 {
            return Err(verr("DecodeDriver: bad dimensions"));
        }
        Ok(DecodeDriver {
            hidden,
            nh,
            kvh,
            hd,
            inter,
            attn_scale,
            eps,
            attn_threads,
            attn_block,
            kv_bf16,
            layers,
            in_norm: vec![Vec::new(); layers],
            post_norm: vec![Vec::new(); layers],
            lin: (0..4 * layers).map(|_| None).collect(),
            pass: Pass::default(),
        })
    }

    /// The layer's input and post-attention RMSNorm weights (float32 ``(hidden,)``).
    fn set_layer(&mut self, layer: usize, in_norm: &Bound<'_, PyArray1<f32>>, post_norm: &Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        if layer >= self.layers {
            return Err(verr("layer out of range"));
        }
        let (a, b) = (f32_vec(in_norm, "in_norm")?, f32_vec(post_norm, "post_norm")?);
        if a.len() != self.hidden || b.len() != self.hidden {
            return Err(verr("norm weights must have hidden entries"));
        }
        self.in_norm[layer] = a;
        self.post_norm[layer] = b;
        Ok(())
    }

    /// Constants of trip ``idx = layer * 4 + step`` (qkv, o, gate_up, down): the linear's widths, the
    /// finish's ``scale``/``bias``/``bound``, the kernels' thread counts and split, and the noise
    /// monitor's fingerprint parameters ``(cols, shift, mult, groups, per, limit)`` (or ``None``).
    #[pyo3(signature = (idx, n, m, scale, bias, bound, threads_p, threads_f, split_p, split_f, dup = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn set_linear(
        &mut self,
        idx: usize,
        n: usize,
        m: usize,
        scale: &Bound<'_, PyArray1<f32>>,
        bias: Option<&Bound<'_, PyArray1<f32>>>,
        bound: u64,
        threads_p: usize,
        threads_f: usize,
        split_p: usize,
        split_f: usize,
        dup: Option<(Bound<'_, PyArray1<i64>>, u32, Bound<'_, PyArray1<u64>>, usize, usize, usize)>,
    ) -> PyResult<()> {
        if idx >= self.lin.len() {
            return Err(verr("trip index out of range"));
        }
        let s = idx % 4;
        if n != self.step_n(s) || m != self.step_m(s) {
            return Err(verr(format!("trip {idx}: linear shape ({m}, {n}) does not match the layer")));
        }
        let scale = f32_vec(scale, "scale")?;
        let bias = match bias {
            Some(b) => Some(f32_vec(b, "bias")?),
            None => None,
        };
        if scale.len() != m || bias.as_ref().is_some_and(|b| b.len() != m) {
            return Err(verr("scale/bias must have m entries"));
        }
        let dup = match dup {
            Some((cols, shift, mult, groups, per, limit)) => {
                let cols = cols.try_readonly().map_err(|e| verr(format!("cols: {e}")))?.as_slice()?.to_vec();
                let mult = mult.try_readonly().map_err(|e| verr(format!("mult: {e}")))?.as_slice()?.to_vec();
                if !DupIn::valid(&cols, shift, &mult, groups, per, n) {
                    return Err(verr("bad fingerprint parameters"));
                }
                Some(DupCfg { cols, shift, mult, groups, per, limit })
            }
            None => None,
        };
        self.lin[idx] = Some(LinCfg { n, m, scale, bias, bound, threads_p, threads_f, split_p, split_f, dup });
        Ok(())
    }

    /// Start a pass at ``layer`` with the residual stream ``h (t, hidden)`` and that layer's normed input
    /// (``normed``, else computed here with its input norm); ``start``: the cache position of row 0.
    #[pyo3(signature = (h, layer, start, normed = None))]
    fn begin(
        &mut self,
        py: Python<'_>,
        h: &Bound<'_, PyArray2<f32>>,
        layer: usize,
        start: usize,
        normed: Option<&Bound<'_, PyArray2<f32>>>,
    ) -> PyResult<()> {
        self.pass.wipe();
        let hidden = self.hidden;
        let hr = h.try_readonly().map_err(|e| verr(format!("h: {e}")))?;
        if hr.shape()[1] != hidden || hr.shape()[0] == 0 || hr.shape()[0] > u16::MAX as usize {
            return Err(verr("h must be (t, hidden)"));
        }
        if layer >= self.layers || self.in_norm[layer].is_empty() {
            return Err(verr("layer out of range or not configured"));
        }
        let t = hr.shape()[0];
        let hs = hr.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
        let width = t * self.hidden.max(self.nh * self.hd).max(self.inter);
        let ymax = t * self.hidden.max((self.nh + 2 * self.kvh) * self.hd).max(2 * self.inter);
        let p = &mut self.pass;
        p.t = t;
        p.start = start;
        p.pos = layer * 4;
        p.h.clear();
        p.h.extend_from_slice(hs);
        p.h2.resize(t * hidden, 0.0);
        p.x.resize(width, 0.0);
        p.y.resize(ymax, 0.0);
        match normed {
            Some(a) => {
                let ar = a.try_readonly().map_err(|e| verr(format!("normed: {e}")))?;
                let s = ar.as_slice().map_err(|_| verr("normed must be C-contiguous"))?;
                if s.len() != t * hidden {
                    return Err(verr("normed must be (t, hidden)"));
                }
                p.x[..t * hidden].copy_from_slice(s);
            }
            None => {
                let w = &self.in_norm[layer];
                let eps = self.eps;
                let (hh, xx) = (&p.h, &mut p.x);
                py.allow_threads(|| rms_rows(&hh[..t * hidden], w, eps, &mut xx[..t * hidden], hidden));
            }
        }
        p.active = true;
        Ok(())
    }

    /// The next trip's index (``layer * 4 + step``).
    #[getter]
    fn pos(&self) -> usize {
        self.pass.pos
    }

    /// Run trips from :attr:`pos` up to (not including) trip ``end`` (a layer boundary). See the module
    /// docs. ``leases``/``flags``/``rids``: one per trip from :attr:`pos` (``flags`` bit 0 check, 1 noise
    /// sample, 2 defer; ``rids`` 16 random bytes each, the uuid4 bits are set here); ``lane``: ``(channel,
    /// ring index, linear indices (uint16, per trip index), deadline_ms, timeout_s)`` or ``None`` (every
    /// trip is handed back after its prepare); ``inject``: the output ``(t, m)`` of trip :attr:`pos`,
    /// which Python finished after a hand-back (applied first).
    ///
    /// Returns ``(code, info, stats, sampled)``; ``sampled``: ``[(pos, x_hat, xq)]`` (``xq`` secret:
    /// zero it after use).
    #[pyo3(signature = (end, leases, flags, rids, kc, vc, cos, sin, lane = None, dup_hist = None, inject = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        end: usize,
        leases: &Bound<'py, PyList>,
        flags: &[u8],
        rids: &[u8],
        kc: &Bound<'py, PyAny>,
        vc: &Bound<'py, PyAny>,
        cos: &Bound<'py, PyArray2<f32>>,
        sin: &Bound<'py, PyArray2<f32>>,
        lane: Option<(Bound<'py, TripChannel>, u8, Bound<'py, PyArray1<u16>>, u32, f64)>,
        dup_hist: Option<Bound<'py, DupHistory>>,
        inject: Option<Bound<'py, PyArray2<f32>>>,
    ) -> PyResult<(u8, PyObject, PyObject, PyObject)> {
        if !self.pass.active {
            return Err(verr("no pass in progress (begin first)"));
        }
        let t = self.pass.t;
        let hd = self.hd;
        let cr = cos.try_readonly().map_err(|e| verr(format!("cos: {e}")))?;
        let sr = sin.try_readonly().map_err(|e| verr(format!("sin: {e}")))?;
        if cr.shape() != [t, hd] || sr.shape() != [t, hd] {
            return Err(verr("cos/sin must be (t, head_dim)"));
        }
        let cs = cr.as_slice().map_err(|_| verr("cos must be C-contiguous"))?;
        let ss = sr.as_slice().map_err(|_| verr("sin must be C-contiguous"))?;
        // the KV cache (only the attention step writes it)
        let mut kf = None;
        let mut kb = None;
        let cap;
        {
            let shape: Vec<usize>;
            if self.kv_bf16 {
                let k = kc.downcast::<PyArray4<u16>>().map_err(|_| verr("kc must be a uint16 (bf16) 4-D array"))?;
                let v = vc.downcast::<PyArray4<u16>>().map_err(|_| verr("vc must be a uint16 (bf16) 4-D array"))?;
                shape = k.shape().to_vec();
                if v.shape() != shape.as_slice() || !k.is_c_contiguous() || !v.is_c_contiguous() {
                    return Err(verr("kc/vc shapes"));
                }
                kb = Some((k.try_readwrite().map_err(|e| verr(format!("kc: {e}")))?,
                           v.try_readwrite().map_err(|e| verr(format!("vc: {e}")))?));
            } else {
                let k = kc.downcast::<PyArray4<f32>>().map_err(|_| verr("kc must be a float32 4-D array"))?;
                let v = vc.downcast::<PyArray4<f32>>().map_err(|_| verr("vc must be a float32 4-D array"))?;
                shape = k.shape().to_vec();
                if v.shape() != shape.as_slice() || !k.is_c_contiguous() || !v.is_c_contiguous() {
                    return Err(verr("kc/vc shapes"));
                }
                kf = Some((k.try_readwrite().map_err(|e| verr(format!("kc: {e}")))?,
                           v.try_readwrite().map_err(|e| verr(format!("vc: {e}")))?));
            }
            if shape[0] != self.layers || shape[2] != self.kvh || shape[3] != hd || self.pass.start + t > shape[1] {
                return Err(verr("kc/vc must be (layers, capacity, kvh, head_dim) with room for the pass"));
            }
            cap = shape[1];
        }
        let start_pos = self.pass.pos;
        if end > 4 * self.layers || end < start_pos + usize::from(inject.is_some()) {
            return Err(verr("end must be after pos (and the injected trip) and within the pass"));
        }
        let first = start_pos + usize::from(inject.is_some());
        if self.pass.held.as_ref().is_some_and(|h| h.pos != first || inject.is_some()) {
            self.pass.wipe();
            return Err(verr("a held trip must be the first trip of the next run"));
        }
        let count = end.saturating_sub(first);
        if leases.len() < count || flags.len() < count || rids.len() < 16 * count {
            return Err(verr("leases/flags/rids must cover every trip to end"));
        }
        // per-trip inputs (borrowed under the GIL, used without it)
        let lease_refs: Vec<PyRef<'py, TripLease>> = leases
            .iter()
            .take(count)
            .map(|o| o.downcast_into::<TripLease>().map_err(|_| verr("leases must be TripLease")).map(|b| b.borrow()))
            .collect::<PyResult<_>>()?;
        let mut rings: Vec<(Option<RoRing<'py>>, Option<RoRing<'py>>)> = Vec::with_capacity(count);
        for (i, lr) in lease_refs.iter().enumerate() {
            let idx = first + i;
            let lin = self.lin[idx].as_ref().ok_or_else(|| verr(format!("trip {idx} not configured")))?;
            if lr.n_in() != lin.n || lr.m_out() != lin.m {
                return Err(verr(format!("trip {idx}: lease shape does not match")));
            }
            if flags[i] & F_CHECK != 0 {
                let k = lr.k_bits();
                rings.push((Some(RoRing::extract(lr.uw_obj().bind(py), "uw", k)?),
                            Some(RoRing::extract(lr.u_obj().bind(py), "u", k)?)));
            } else {
                rings.push((None, None));
            }
        }
        let lane_ix_ro = match &lane {
            Some((_, _, ix, _, _)) => Some(ix.try_readonly().map_err(|e| verr(format!("linear indices: {e}")))?),
            None => None,
        };
        let lane_ix: Option<&[u16]> = match &lane_ix_ro {
            Some(r) => Some(r.as_slice().map_err(|_| verr("linear indices must be contiguous"))?),
            None => None,
        };
        if lane_ix.is_some_and(|ix| ix.len() < 4 * self.layers) {
            return Err(verr("linear indices must cover every trip"));
        }
        let mut trips: Vec<TripIn<'_>> = Vec::with_capacity(count);
        for (i, lr) in lease_refs.iter().enumerate() {
            let idx = first + i;
            let (uw, u) = &rings[i];
            let uws = match uw {
                Some(a) => Some(a.src("uw")?.0),
                None => None,
            };
            let us = match u {
                Some(a) => Some(a.src("u")?.0),
                None => None,
            };
            let mut rid = [0u8; 16];
            rid.copy_from_slice(&rids[16 * i..16 * i + 16]);
            trips.push(TripIn {
                lease: lr,
                lin: self.lin[idx].as_ref().expect("checked above"),
                uw: uws,
                u: us,
                flags: flags[i],
                rid,
                lin_ix: lane_ix.map_or(0, |ix| ix[idx]),
            });
        }
        let chan = lane.as_ref().map(|(ch, ring, _, dl, to)| (ch.get(), *ring, *dl, *to));
        let lane_in = match chan {
            Some((ch, ring, deadline_ms, to)) => {
                if !(to.is_finite() && to > 0.0 && to < 1e7) {
                    return Err(verr("timeout_s must be a positive number of seconds"));
                }
                if t > usize::from(ch.max_rows_raw()) {
                    return Err(verr("too many rows for the lane channel"));
                }
                Some(LaneIn { ch, ring, deadline_ms, timeout: Duration::from_secs_f64(to) })
            }
            None => None,
        };
        let hist = dup_hist.as_ref().map(|h| h.get());
        let inj_ro = match &inject {
            Some(a) => Some(a.try_readonly().map_err(|e| verr(format!("inject: {e}")))?),
            None => None,
        };
        let inj: Option<&[f32]> = match &inj_ro {
            Some(r) => {
                let m = self.step_m(start_pos % 4);
                if r.shape() != [t, m] {
                    return Err(verr(format!("inject must be ({t}, {m})")));
                }
                Some(r.as_slice().map_err(|_| verr("inject must be C-contiguous"))?)
            }
            None => None,
        };
        let view = DriverView {
            hidden: self.hidden,
            nh: self.nh,
            kvh: self.kvh,
            hd: self.hd,
            inter: self.inter,
            attn_scale: self.attn_scale,
            eps: self.eps,
            attn_threads: self.attn_threads,
            attn_block: self.attn_block,
            layers: self.layers,
            cap,
            in_norm: self.in_norm.iter().map(|v| v.as_slice()).collect(),
            post_norm: self.post_norm.iter().map(|v| v.as_slice()).collect(),
        };
        let pass = &mut self.pass;
        let mut sampled: Vec<(usize, Vec<i64>, Vec<i64>)> = Vec::new();
        let mut stats = Stats::default();
        let outcome = {
            let mut kf_s = match kf.as_mut() {
                Some((k, v)) => Some((k.as_slice_mut()?, v.as_slice_mut()?)),
                None => None,
            };
            let mut kb_s = match kb.as_mut() {
                Some((k, v)) => Some((k.as_slice_mut()?, v.as_slice_mut()?)),
                None => None,
            };
            py.allow_threads(|| {
                // the trip split's pool workers stay warm for this call (PGW_POOL_ACTIVE_SPIN_US; timing only)
                let _warm = crate::prepare::pool::Warm::new();
                if let Some(y) = inj {
                    let m = y.len();
                    pass.y[..m].copy_from_slice(y);
                    let at = pass.pos;
                    if let Some(kv) = kf_s.as_mut() {
                        let mut s = Some((&mut *kv.0, &mut *kv.1));
                        DecodeDriver::after_trip::<f32>(&view, pass, at, &mut s, cs, ss);
                    } else if let Some(kv) = kb_s.as_mut() {
                        let mut s = Some((&mut *kv.0, &mut *kv.1));
                        DecodeDriver::after_trip::<u16>(&view, pass, at, &mut s, cs, ss);
                    }
                    pass.pos = at + 1;
                }
                let lane_ref = lane_in.as_ref();
                if let Some(kv) = kf_s.take() {
                    run_loop::<f32>(&view, pass, &trips, end, lane_ref, hist, Some(kv), cs, ss, &mut sampled, &mut stats)
                } else if let Some(kv) = kb_s.take() {
                    run_loop::<u16>(&view, pass, &trips, end, lane_ref, hist, Some(kv), cs, ss, &mut sampled, &mut stats)
                } else {
                    Outcome::NotReady
                }
            })
        };
        drop(trips);
        drop(rings);
        let st = &stats;
        let stats_t = PyTuple::new(
            py,
            [st.trips, st.rows, st.bytes_up, st.bytes_down, st.checked, st.skipped, st.deferred, st.deferred_ok,
             st.remote_ns, st.local_ns, st.lane_trips, st.lane_sent, st.lane_recv, st.last_rtt_ns, st.dup_rows, st.dups],
        )?;
        let n_of = |pos: usize| self.lin[pos].as_ref().map_or(0, |l| l.n);
        let samp = PyList::empty(py);
        for (pos, xh, xq) in sampled {
            let n = n_of(pos);
            let a = PyArray1::from_vec(py, xh).reshape([t, n])?;
            let b = PyArray1::from_vec(py, xq).reshape([t, n])?;
            samp.append((pos, a, b))?;
        }
        let (code, info): (u8, PyObject) = match outcome {
            Outcome::Done => (DONE, py.None()),
            Outcome::Handback { mut ticket, payload, rid, lane_status, lane_err } => {
                let idx = self.pass.pos - first;
                ticket.attach_u(lease_refs[idx].u_obj().clone_ref(py));
                let tk = Py::new(py, ticket)?;
                let pl = PyBytes::new(py, &payload);
                let mut payload = payload;
                wipe(&mut payload);
                (HANDBACK, (pl, tk, PyBytes::new(py, &rid), lane_status, lane_err).into_pyobject(py)?.into_any().unbind())
            }
            Outcome::Status(s, bad) => (FAILED_STATUS, (s, bad).into_pyobject(py)?.into_any().unbind()),
            Outcome::Prepare(kind, msg) => (PREPARE_ERROR, (kind, msg).into_pyobject(py)?.into_any().unbind()),
            Outcome::Duplicate(d, rows) => (DUPLICATE, (d, rows).into_pyobject(py)?.into_any().unbind()),
            Outcome::Deferred(s, bad) => (DEFERRED_FAILED, (s, bad).into_pyobject(py)?.into_any().unbind()),
            Outcome::NotReady => (NOT_READY, py.None()),
            Outcome::FinishErr(msg) => (FINISH_ERROR, msg.into_pyobject(py)?.into_any().unbind()),
            Outcome::Held => (HELD, py.None()),
        };
        if code != DONE && code != HANDBACK && code != HELD {
            self.pass.wipe(); // the pass is over (Python raises)
        }
        Ok((code, info, stats_t.into_any().unbind(), samp.into_any().unbind()))
    }

    /// End the pass: the residual stream into ``h_out (t, hidden)`` and, when the next layer's normed
    /// input was computed (the pass stopped before the last layer), into ``normed_out``; every pass
    /// buffer is zeroed.
    #[pyo3(signature = (h_out, normed_out = None))]
    fn end(&mut self, h_out: &Bound<'_, PyArray2<f32>>, normed_out: Option<&Bound<'_, PyArray2<f32>>>) -> PyResult<()> {
        let p = &mut self.pass;
        if !p.active || p.pos % 4 != 0 {
            p.wipe();
            return Err(verr("no finished pass to end"));
        }
        let n = p.t * self.hidden;
        let res = (|| -> PyResult<()> {
            let mut hw = h_out.try_readwrite().map_err(|e| verr(format!("h_out: {e}")))?;
            let hs = hw.as_slice_mut().map_err(|_| verr("h_out must be C-contiguous"))?;
            if hs.len() != n {
                return Err(verr("h_out must be (t, hidden)"));
            }
            hs.copy_from_slice(&p.h[..n]);
            if let Some(o) = normed_out {
                let mut ow = o.try_readwrite().map_err(|e| verr(format!("normed_out: {e}")))?;
                let os = ow.as_slice_mut().map_err(|_| verr("normed_out must be C-contiguous"))?;
                if os.len() != n {
                    return Err(verr("normed_out must be (t, hidden)"));
                }
                os.copy_from_slice(&p.x[..n]);
            }
            Ok(())
        })();
        p.wipe();
        res
    }

    /// Abandon the pass (zero its buffers). Idempotent.
    fn abort(&mut self) {
        self.pass.wipe();
    }

    /// A copy of trip :attr:`pos`'s input ``(t, n)`` (float32), for a trip Python runs on another path
    /// (then :meth:`run` with its output as ``inject``).
    fn current_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let p = &self.pass;
        if !p.active || p.pos >= 4 * self.layers {
            return Err(verr("no trip pending"));
        }
        let n = self.step_n(p.pos % 4);
        PyArray1::from_slice(py, &p.x[..p.t * n]).reshape([p.t, n])
    }
}
