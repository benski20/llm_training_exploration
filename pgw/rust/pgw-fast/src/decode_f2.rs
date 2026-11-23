//! Native fused2 decode driver (`PGW_FUSED2_DECODE_NATIVE`, opt-in; default off): the decoder layers of one
//! fused2 pass of one sequence (a few rows) — `2L + 1` masked provider trips (layer 0's qkv, then per layer the
//! pairs `o + gate_up` and `down + next qkv`, the last layer's `down` single) and the float work between them —
//! in one native call per run with the GIL released. Python (`pgw.engine.fused2_native`) runs once per call.
//!
//! It is the native twin of `pgw.engine.fused2.Fused2Schedule` on the native float ops (`float_ops = "native"`)
//! with `Fused2MaskedBackend` over the lane (`LaneFused2Provider`): the same integers, the same float64
//! roundings in the same order (numpy's pairwise sums are reproduced exactly by [`pairwise_sum`]), the same
//! attention / SiLU kernels (`attn.rs`, `layer.rs`), the same payload bytes and checks, so the outputs are bit
//! for bit those of the Python path (`tests/engine/test_fused2_native.py`).
//!
//! Per trip: quantization with the exact a-priori bounds (most activation bits `<= A` such that every genuine
//! output fits `2^C`), masking with the pass's mask rows (`x_hat = x + r_x`, `R_hat = R + v - hmul * W1 r_x`, all
//! mod `2^K`; taken by Python from the mask source, one slot per token, single use), the payload packed
//! `K/8`-byte little-endian, the duplicate-row fingerprints recorded on the shared [`DupHistory`] *before* the
//! payload leaves (optional), the lane trip on the native [`TripChannel`], native unpack + unmask + centering, the
//! range and `F_p` (`p = 2^61 - 1`) Freivalds checks of both stages (`u1 . (H - R) == (u1 hmul W1) . x`,
//! `u2 . Z == (u2 W2') . H`) before any output is used, then the float work. Sampled trips hand back
//! `(pos, x_hat, xq)` for the noise statistics; Python ends a call at a sampled trip, so its statistics run
//! before the next payload is built. Any failure returns a code at the failing trip and wipes the pass.

#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

use crate::attn::{auto_threads, rope_attend, Dims, Kv};
use crate::finish::wipe;
use crate::lane::{LaneError, RequestHeader, TripChannel};
use crate::layer::silu_rows;
use crate::noise::DupHistory;
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::time::{Duration, Instant};

/// Cumulative ns of the fused2 driver's local work per trip kind (benchmarks; timing only):
/// `[before the lane x3, after the lane x3, trips x3, unmask x3, checks x3, float x3]`, kinds = (first single,
/// pair A: o + gate_up, pair B: down + next qkv; the last single counts as B).
static F2_PHASE_NS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

fn f2_phase_add(i: usize, v: u64) {
    F2_PHASE_NS[i].fetch_add(v, std::sync::atomic::Ordering::Relaxed);
}

/// The fused2 driver's phase counters (see [`F2_PHASE_NS`]); `reset` zeroes them after reading.
#[pyfunction]
#[pyo3(signature = (reset = false))]
fn f2_phase_ns(reset: bool) -> Vec<u64> {
    F2_PHASE_NS
        .iter()
        .map(|a| if reset { a.swap(0, std::sync::atomic::Ordering::Relaxed) } else { a.load(std::sync::atomic::Ordering::Relaxed) })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Fused2Driver>()?;
    m.add_function(wrap_pyfunction!(f2_pairwise_sum, m)?)?;
    m.add_function(wrap_pyfunction!(f2_quantize, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(f2_phase_ns, m)?)?;
    Ok(())
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// `fused2._SLACK`.
const SLACK: f64 = 1.0 + 1e-9;
const P61: u64 = (1 << 61) - 1;

const F_SAMPLED: u8 = 2;
/// Test hook: hand back the trip's payload (the provider-visible body).
const F_RECORD: u8 = 4;

/// Return codes of [`Fused2Driver::run`].
const DONE: u8 = 0;
const LANE_STATUS: u8 = 2;
const OVERFLOW: u8 = 3;
const DUPLICATE: u8 = 4;
const CHECK_FAILED: u8 = 5;
const LANE_ERROR: u8 = 6;
const NONFINITE: u8 = 7;

/// numpy's pairwise summation of a contiguous float64 reduction (`DOUBLE_pairwise_sum`, `PW_BLOCKSIZE = 128`),
/// started from `-0.0` as numpy does: `np.add.reduce` / `sum(axis=-1)` give exactly this value (and `mean` this
/// value divided by `n`).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = -0.0f64;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum(x * x)` of one row (numpy: the product array, then the pairwise reduction).
fn sum_sq(x: &[f64], tmp: &mut Vec<f64>) -> f64 {
    tmp.clear();
    tmp.extend(x.iter().map(|v| v * v));
    pairwise_sum(tmp)
}

/// One row of `fused2._quant_rows`: `(q, s)` with `s = amax / qmax` (1 for a zero row), `q = clip(rint(x / s))`.
fn quant_row(x: &[f64], bits: u32, q: &mut [f64]) -> f64 {
    let qmax = ((1i64 << (bits - 1)) - 1) as f64;
    let amax = x.iter().fold(0f64, |m, v| m.max(v.abs()));
    let s = if amax > 0.0 { amax / qmax } else { 1.0 };
    for (o, &v) in q.iter_mut().zip(x) {
        o.clone_from(&(v / s).round_ties_even().clamp(-qmax, qmax));
    }
    s
}

/// Public constants of one fused2 linear (`F2Linear`).
#[derive(Clone)]
struct Lin {
    n: usize,
    m: usize,
    /// Row scales (float64; `normed_output`).
    scale: Vec<f64>,
    bias: Option<Vec<f64>>,
    /// Residual grid `scale / hmul` (a trip's first linear).
    grid: Vec<f64>,
    hmul: Option<Vec<u64>>,
    wnorm: f64,
    sigma: f64,
}

struct Checks {
    u: Vec<u64>,
    uw: Vec<u64>,
}

impl Drop for Checks {
    fn drop(&mut self) {
        wipe(&mut self.u);
        wipe(&mut self.uw);
    }
}

struct DupCfg {
    cols: Vec<i64>,
    shift: u32,
    mult: Vec<u64>,
    groups: usize,
    per: usize,
    limit: usize,
}

#[derive(Default)]
struct TripCfg {
    l1: Option<Lin>,
    l2: Option<Lin>,
    c1: Option<Checks>,
    c2: Option<Checks>,
    item: u16,
    dup: Option<DupCfg>,
}

/// The pass state between calls (secret: wiped when the pass ends or fails).
#[derive(Default)]
struct Pass {
    t: usize,
    start: usize,
    pos: usize,
    active: bool,
    /// The fixed-point residual stream as float64 (`ex["h"]`), `(t, hidden)`.
    h: Vec<f64>,
    /// The current layer's qkv outputs (float32), `(t, qkv)`.
    qkv: Vec<f32>,
    /// The next trip's activations (attention output or SiLU), float32.
    act: Vec<f32>,
    /// Scratch.
    gu: Vec<f32>,
    xq: Vec<i64>,
    rq: Vec<i64>,
    y: Vec<i64>,
    z: Vec<i64>,
    payload: Vec<u8>,
    reply: Vec<u8>,
    keys: Vec<u64>,
    f64a: Vec<f64>,
    f64b: Vec<f64>,
}

impl Pass {
    fn wipe(&mut self) {
        wipe(&mut self.h);
        wipe(&mut self.qkv);
        wipe(&mut self.act);
        wipe(&mut self.gu);
        wipe(&mut self.xq);
        wipe(&mut self.rq);
        wipe(&mut self.y);
        wipe(&mut self.z);
        wipe(&mut self.payload);
        wipe(&mut self.reply);
        wipe(&mut self.f64a);
        wipe(&mut self.f64b);
        self.active = false;
    }
}

#[derive(Default, Clone, Copy)]
struct Stats {
    trips: u64,
    rows: u64,
    bytes_up: u64,
    bytes_down: u64,
    checks: u64,
    remote_ns: u64,
    local_ns: u64,
    compute_us: u64,
    dup_rows: u64,
    dups: u64,
}

enum Outcome {
    Done,
    Status(u8),
    Overflow(String),
    NonFinite,
    Duplicate(usize),
    Check { stage: u8, code: u8 },
    Lane(&'static str),
}

/// One trip's masks (borrowed from Python's int64 arrays, read without the GIL).
struct MaskIn<'a> {
    r_x: &'a [i64],
    wr_x: &'a [i64],
    v: Option<&'a [i64]>,
    wv: Option<&'a [i64]>,
}

struct LaneIn<'a> {
    ch: &'a TripChannel,
    ring: u8,
    deadline_ms: u32,
    timeout: Duration,
}

struct View<'a> {
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
    k: u32,
    a_bits: u32,
    c_bits: u32,
    trips: &'a [TripCfg],
}

fn ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// `fused2.quantize_single` of one row: most bits `<= A` with `wnorm ||q|| <= 2^C`.
fn quantize_single_row(x: &[f64], wnorm: f64, a_bits: u32, c_bits: u32, q: &mut [f64], tmp: &mut Vec<f64>) -> Option<f64> {
    let limit = (1u64 << c_bits) as f64;
    for bits in (2..=a_bits).rev() {
        let s = quant_row(x, bits, q);
        let na = sum_sq(q, tmp).sqrt();
        if na * wnorm * SLACK <= limit {
            return Some(s);
        }
    }
    None
}

/// `fused2.quantize_pair` of one row: `(s, bits)` and `q`, `rr` (the residual on `l1`'s grid).
#[allow(clippy::too_many_arguments)]
fn quantize_pair_row(a: &[f64], h: &[f64], l1: &Lin, l2: Option<&Lin>, a_bits: u32, c_bits: u32, q: &mut [f64],
                     rr: &mut [f64], tmp: &mut Vec<f64>) -> Option<f64> {
    let limit = (1u64 << c_bits) as f64;
    for bits in (2..=a_bits).rev() {
        let sb = quant_row(a, bits, q);
        for ((o, &hv), &g) in rr.iter_mut().zip(h).zip(&l1.grid) {
            *o = (hv / (g * sb)).round_ties_even();
        }
        let na = sum_sq(q, tmp).sqrt();
        let rmax = rr.iter().fold(0f64, |m, v| m.max(v.abs()));
        let b_h = rmax + l1.wnorm * na;
        let mut fits = b_h * SLACK <= limit;
        if let Some(l2) = l2 {
            let b_z = l2.wnorm * (sum_sq(rr, tmp).sqrt() + l1.sigma * na);
            fits &= b_z * SLACK <= limit;
        }
        if fits {
            return Some(sb);
        }
    }
    None
}

/// `1 / sqrt(mean(h^2) + eps)` (`fused2.rinv_rows`).
fn rinv_row(h: &[f64], eps: f64, tmp: &mut Vec<f64>) -> f64 {
    1.0 / (sum_sq(h, tmp) / h.len() as f64 + eps).sqrt()
}

/// `fused2.normed_output` of one row (no gateway columns): `((Z * scale) * (s * rinv) + bias) as f32`.
fn normed_row(z: &[i64], lin: &Lin, s: f64, rinv: f64, out: &mut [f32]) {
    let f = s * rinv;
    for (j, (o, &zv)) in out.iter_mut().zip(z).enumerate() {
        let mut v = (zv as f64 * lin.scale[j]) * f;
        if let Some(b) = &lin.bias {
            v += b[j];
        }
        *o = v as f32;
    }
}

/// `fused2.residual_from` of one row: `(H * grid) * s` (float64).
fn residual_row(hq: &[i64], lin: &Lin, s: f64, out: &mut [f64]) {
    for ((o, &v), &g) in out.iter_mut().zip(hq).zip(&lin.grid) {
        *o = (v as f64 * g) * s;
    }
}

/// `PGW_FUSED2_SPLIT` (read once; default 0 = off): split the wide element-wise stages of a one-row pair trip
/// (unmask of `Z`, the stage-2 check's dot products, the normed gate/up and SiLU) into this many fixed element
/// ranges on the persistent pool. Element-wise results are the same bits; the check's sums are exact mod p.
fn split_cfg() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PGW_FUSED2_SPLIT").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0).min(64)
    })
}

/// `PGW_FUSED2_SPLIT_MIN_ELEMS` (read once; default 8192): stages with fewer elements stay serial.
fn split_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PGW_FUSED2_SPLIT_MIN_ELEMS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(8192).max(2)
    })
}

/// Ranges for `len` elements (0 = serial: off, or fewer than `split_min()` elements).
fn split_parts(len: usize) -> usize {
    let c = split_cfg();
    if c < 2 || len < split_min() {
        0
    } else {
        c.min(len.div_ceil(64)).max(2)
    }
}

/// `[a, b)` of range `i` of `parts` over `len` (64-element aligned boundaries).
fn range_of(len: usize, parts: usize, i: usize) -> (usize, usize) {
    let per = len.div_ceil(parts).div_ceil(64) * 64;
    ((i * per).min(len), ((i + 1) * per).min(len))
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
// SAFETY: used only for disjoint element ranges written by one task each while the owner waits.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

fn pack_into(vals: impl Iterator<Item = u64>, nb: usize, out: &mut [u8]) {
    // the same bytes as `chunk.copy_from_slice(&v.to_le_bytes()[..nb])` per element, with one 8-byte store per
    // element while 8 bytes fit (the spill into the next element is overwritten by it: elements go in order)
    let cnt = out.len() / nb;
    for (i, v) in vals.take(cnt).enumerate() {
        let o = i * nb;
        if o + 8 <= out.len() {
            out[o..o + 8].copy_from_slice(&v.to_le_bytes());
        } else {
            out[o..o + nb].copy_from_slice(&v.to_le_bytes()[..nb]);
        }
    }
}

#[inline(always)]
fn unpack_at(b: &[u8], nb: usize, i: usize) -> u64 {
    let o = i * nb;
    if o + 8 <= b.len() {
        // one unaligned 8-byte load, the element's nb low bytes kept (same value as the byte copy below)
        let w = u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
        return if nb >= 8 { w } else { w & ((1u64 << (8 * nb)) - 1) };
    }
    let mut w = [0u8; 8];
    w[..nb].copy_from_slice(&b[o..o + nb]);
    u64::from_le_bytes(w)
}

fn center(v: u64, k: u32) -> i64 {
    let mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let v = v & mask;
    if k < 64 && v >= 1u64 << (k - 1) {
        (v as i64) - (1i64 << (k - 1)) - (1i64 << (k - 1))
    } else {
        v as i64
    }
}

/// Run trips `p.pos .. end` (GIL released).
#[allow(clippy::too_many_arguments)]
fn run_loop<T: Kv>(
    cfg: &View<'_>,
    p: &mut Pass,
    masks: &[MaskIn<'_>],
    flags: &[u8],
    rids: &[[u8; 16]],
    end: usize,
    lane: &LaneIn<'_>,
    dup_hist: Option<&DupHistory>,
    kc: &mut [T],
    vc: &mut [T],
    cos: &[f32],
    sin: &[f32],
    sampled: &mut Vec<(usize, Vec<i64>, Vec<i64>)>,
    recorded: &mut Vec<(usize, Vec<u8>)>,
    st: &mut Stats,
) -> Outcome {
    let t = p.t;
    let base = p.pos;
    let k = cfg.k;
    let nb = (k / 8) as usize;
    let kmask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let bound = 1i64 << cfg.c_bits;
    let hidden = cfg.hidden;
    let total = 2 * cfg.layers + 1;
    let mut tmp: Vec<f64> = Vec::new();
    let mut svals = vec![0f64; t];
    while p.pos < end {
        let pos = p.pos;
        let i = pos - base;
        let tc = &cfg.trips[pos];
        let l1 = tc.l1.as_ref().expect("configured");
        let l2 = tc.l2.as_ref();
        let (n1, m1) = (l1.n, l1.m);
        let m2 = l2.map_or(0, |l| l.m);
        let last = pos + 1 == total;
        let t0 = Instant::now();
        // ---- quantize (the gateway's roundings) ----
        p.xq.resize(t * n1, 0);
        p.f64a.resize(t * n1, 0.0);
        let mut qrow = vec![0f64; n1];
        if pos == 0 {
            if !p.h[..t * hidden].iter().all(|v| v.is_finite()) {
                return Outcome::NonFinite;
            }
            for r in 0..t {
                let x = &p.h[r * hidden..(r + 1) * hidden];
                match quantize_single_row(x, l1.wnorm, cfg.a_bits, cfg.c_bits, &mut qrow, &mut tmp) {
                    Some(s) => svals[r] = s,
                    None => return Outcome::Overflow(format!("fused2: trip {pos} input cannot be bounded by 2^{}", cfg.c_bits)),
                }
                for (o, &v) in p.xq[r * n1..(r + 1) * n1].iter_mut().zip(&qrow) {
                    *o = v as i64;
                }
            }
            p.rq.clear();
        } else {
            if !p.act[..t * n1].iter().all(|v| v.is_finite()) || !p.h[..t * hidden].iter().all(|v| v.is_finite()) {
                return Outcome::NonFinite;
            }
            p.rq.resize(t * m1, 0);
            let mut rrow = vec![0f64; m1];
            let mut arow = vec![0f64; n1];
            for r in 0..t {
                for (o, &v) in arow.iter_mut().zip(&p.act[r * n1..(r + 1) * n1]) {
                    *o = f64::from(v);
                }
                let h = &p.h[r * hidden..(r + 1) * hidden];
                let l2q = if last { None } else { l2 };
                match quantize_pair_row(&arow, h, l1, l2q, cfg.a_bits, cfg.c_bits, &mut qrow, &mut rrow, &mut tmp) {
                    Some(s) => svals[r] = s,
                    None => return Outcome::Overflow(format!("fused2: trip {pos} cannot be bounded by 2^{}", cfg.c_bits)),
                }
                for (o, &v) in p.xq[r * n1..(r + 1) * n1].iter_mut().zip(&qrow) {
                    *o = v as i64;
                }
                for (o, &v) in p.rq[r * m1..(r + 1) * m1].iter_mut().zip(&rrow) {
                    *o = v as i64;
                }
            }
            wipe(&mut rrow);
            wipe(&mut arow);
        }
        wipe(&mut qrow);
        if p.xq[..t * n1].iter().any(|v| v.unsigned_abs() > bound as u64) {
            return Outcome::Overflow("fused2 input out of range".into());
        }
        // ---- mask + pack ----
        let mk = &masks[i];
        let pair = l2.is_some() && !last;
        let body_len = t * (n1 + if pair { m1 } else { 0 }) * nb;
        p.payload.resize(body_len, 0);
        {
            let xs = &p.xq[..t * n1];
            pack_into(xs.iter().zip(mk.r_x).map(|(&x, &r)| (x as u64).wrapping_add(r as u64) & kmask), nb,
                      &mut p.payload[..t * n1 * nb]);
            if pair {
                let (v, hm) = (mk.v.expect("pair masks"), l1.hmul.as_deref());
                let rq = &p.rq[..t * m1];
                let it = (0..t * m1).map(|e| {
                    let wr = mk.wr_x[e] as u64;
                    let wr = match hm {
                        Some(h) => wr.wrapping_mul(h[e % m1]) & kmask,
                        None => wr,
                    };
                    let r_r = (v[e] as u64).wrapping_sub(wr) & kmask; // r_R = v - hmul * W1 r_x
                    (rq[e] as u64).wrapping_add(r_r) & kmask
                });
                pack_into(it, nb, &mut p.payload[t * n1 * nb..]);
            }
        }
        // the noise monitor's every-call check, before the payload leaves
        if let (Some(d), Some(hist)) = (tc.dup.as_ref(), dup_hist) {
            p.keys.resize(t * d.groups, 0);
            if let Err(e) = crate::noise::dup_keys(&p.payload[..t * n1 * nb], t, n1, nb, &d.cols, d.shift, &d.mult,
                                                   d.groups, d.per, &mut p.keys) {
                return Outcome::Overflow(e);
            }
            let dups = hist.record_raw(&p.keys[..t * d.groups], t, d.groups, d.limit);
            st.dup_rows += t as u64;
            if dups > 0 {
                st.dups += dups as u64;
                return Outcome::Duplicate(dups);
            }
        }
        if flags[i] & F_SAMPLED != 0 {
            let xh: Vec<i64> = (0..t * n1).map(|e| unpack_at(&p.payload, nb, e) as i64).collect();
            sampled.push((pos, xh, p.xq[..t * n1].to_vec()));
        }
        if flags[i] & F_RECORD != 0 {
            recorded.push((pos, p.payload.clone()));
        }
        // ---- the trip ----
        let mut rid = rids[i];
        rid[6] = (rid[6] & 0x0F) | 0x40;
        rid[8] = (rid[8] & 0x3F) | 0x80;
        let out_m = if pair { m1 + m2 } else { m1 };
        let olen = t * out_m * nb;
        p.reply.resize(olen, 0);
        let hdr = RequestHeader {
            ring: lane.ring,
            linear: tc.item,
            rows: t as u16,
            deadline_ms: lane.deadline_ms,
            body_len: body_len as u32,
            request_id: rid,
        };
        let t1 = Instant::now();
        st.local_ns += ns(t1 - t0);
        let kind = if pos == 0 { 0 } else if (pos - 1) % 2 == 0 { 1 } else { 2 };
        f2_phase_add(kind, ns(t1 - t0));
        f2_phase_add(6 + kind, 1);
        if crate::lane::wait_prefetch::enabled() {
            // read right after the reply (timing only): this trip's unmask masks and check vectors, the
            // next trip's input masks
            use crate::lane::wait_prefetch::add_slice;
            add_slice(mk.wr_x);
            if let Some(v) = mk.v {
                add_slice(v);
            }
            if let Some(w) = mk.wv {
                add_slice(w);
            }
            for c in [tc.c1.as_ref(), tc.c2.as_ref()].into_iter().flatten() {
                add_slice(&c.u);
                add_slice(&c.uw);
            }
            if let Some(nx) = masks.get(i + 1) {
                add_slice(nx.r_x);
            }
        }
        let res = lane.ch.trip_raw(&hdr, &p.payload, &mut p.reply, lane.timeout);
        crate::lane::wait_prefetch::clear();
        let t2 = Instant::now();
        st.remote_ns += ns(t2 - t1);
        match res {
            Ok((0, cus)) => st.compute_us += u64::from(cus),
            Ok((status, _)) => return Outcome::Status(status),
            Err(e) => {
                return Outcome::Lane(match e {
                    LaneError::Io(_) => "OSError",
                    LaneError::Protocol(_) => "ProtocolError",
                    LaneError::Refused(_) => "Refused",
                    LaneError::Closed => "Closed",
                })
            }
        }
        st.trips += 1;
        st.rows += t as u64;
        st.bytes_up += body_len as u64;
        st.bytes_down += olen as u64;
        let tu = Instant::now();
        // ---- unmask + checks (before any output is used) ----
        p.y.resize(t * m1, 0);
        let rb = &p.reply;
        if pair {
            let v = mk.v.expect("pair masks");
            let wv = mk.wv.expect("pair masks");
            for r in 0..t {
                for j in 0..m1 {
                    let e = r * m1 + j;
                    p.y[e] = center(unpack_at(rb, nb, e).wrapping_sub(v[e] as u64), k); // reply: pack(H) || pack(Z)
                }
            }
            p.z.resize(t * m2, 0);
            let parts = split_parts(t * m2);
            if parts >= 2 {
                // the same element-wise unmask in fixed element ranges on the pool (PGW_FUSED2_SPLIT)
                let zp = SendPtr(p.z.as_mut_ptr());
                let off = t * m1;
                let ok = crate::prepare::pool::global().run(parts, &|i| {
                    let (a, b) = range_of(t * m2, parts, i);
                    for e in a..b {
                        let v = center(unpack_at(rb, nb, off + e).wrapping_sub(wv[e] as u64), k);
                        // SAFETY: element e is written by this task only; p.z outlives the run.
                        unsafe { *zp.get().add(e) = v };
                    }
                });
                if !ok {
                    return Outcome::Overflow("fused2: internal error in an unmask task".into());
                }
            } else {
                for r in 0..t {
                    for j in 0..m2 {
                        let e = r * m2 + j;
                        p.z[e] = center(unpack_at(rb, nb, t * m1 + e).wrapping_sub(wv[e] as u64), k);
                    }
                }
            }
        } else {
            for e in 0..t * m1 {
                p.y[e] = center(unpack_at(rb, nb, e).wrapping_sub(mk.wr_x[e] as u64), k);
            }
        }
        let tc0 = Instant::now();
        f2_phase_add(9 + kind, ns(tc0 - tu));
        let c1 = tc.c1.as_ref().expect("configured");
        // stage 1 (pair: H - R == hmul * W1 x; single: Y == W1 x)
        let off = if pair { Some(&p.rq[..t * m1]) } else { None };
        let s1 = crate::p61::check(&p.y[..t * m1], off, &c1.u, &p.xq[..t * n1], &c1.uw, m1, n1, bound);
        st.checks += 1;
        if s1 != 0 {
            return Outcome::Check { stage: 1, code: s1 };
        }
        if pair {
            let c2 = tc.c2.as_ref().expect("configured");
            let s2 = if t == 1 && split_parts(m2) >= 2 {
                crate::p61::check_split(&p.z[..m2], &c2.u, &p.y[..m1], &c2.uw, m2, m1, bound, split_parts(m2))
            } else {
                crate::p61::check(&p.z[..t * m2], None, &c2.u, &p.y[..t * m1], &c2.uw, m2, m1, bound)
            };
            st.checks += 1;
            if s2 != 0 {
                return Outcome::Check { stage: 2, code: s2 };
            }
        }
        let tfl = Instant::now();
        f2_phase_add(12 + kind, ns(tfl - tc0));
        // ---- the float work after the trip ----
        if pos == 0 {
            let qc = m1;
            p.qkv.resize(t * qc, 0.0);
            for r in 0..t {
                let rinv = rinv_row(&p.h[r * hidden..(r + 1) * hidden], cfg.eps, &mut tmp);
                normed_row(&p.y[r * m1..(r + 1) * m1], l1, svals[r], rinv, &mut p.qkv[r * qc..(r + 1) * qc]);
            }
            attend(cfg, p, 0, kc, vc, cos, sin);
        } else if last {
            for r in 0..t {
                let hq: Vec<i64> = (0..m1).map(|j| p.y[r * m1 + j] + p.rq[r * m1 + j]).collect();
                residual_row(&hq, l1, svals[r], &mut p.h[r * hidden..(r + 1) * hidden]);
            }
        } else {
            let l2 = l2.expect("pair");
            for r in 0..t {
                residual_row(&p.y[r * m1..(r + 1) * m1], l1, svals[r], &mut p.h[r * hidden..(r + 1) * hidden]);
            }
            let layer = (pos - 1) / 2;
            if (pos - 1) % 2 == 0 {
                // trip A: gate_up (normed), then SiLU
                p.gu.resize(t * m2, 0.0);
                let inter = cfg.inter;
                p.act.resize(t * inter.max(cfg.nh * cfg.hd), 0.0);
                let parts = split_parts(t * m2);
                if t == 1 && parts >= 2 && m2 == 2 * inter {
                    // the same element-wise float work in fixed ranges on the pool (PGW_FUSED2_SPLIT)
                    let rinv = rinv_row(&p.h[..hidden], cfg.eps, &mut tmp);
                    let f = svals[0] * rinv;
                    let (zs, gp, ap) = (&p.z[..m2], SendPtr(p.gu.as_mut_ptr()), SendPtr(p.act.as_mut_ptr()));
                    let ok = crate::prepare::pool::global().run(parts, &|i| {
                        let (a, b) = range_of(m2, parts, i);
                        for j in a..b {
                            let mut v = (zs[j] as f64 * l2.scale[j]) * f;
                            if let Some(bb) = &l2.bias {
                                v += bb[j];
                            }
                            // SAFETY: element j is written by this task only; p.gu outlives the run.
                            unsafe { *gp.get().add(j) = v as f32 };
                        }
                    });
                    let ok2 = ok && crate::prepare::pool::global().run(parts, &|i| {
                        let (a, b) = range_of(inter, parts, i);
                        if a < b {
                            // SAFETY: the gate/up halves were fully written above (joined); this task
                            // writes act[a..b] only.
                            let gu = unsafe { std::slice::from_raw_parts(gp.get().cast_const(), m2) };
                            let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add(a), b - a) };
                            crate::layer::silu_mul_row(&gu[a..b], &gu[inter + a..inter + b], out);
                        }
                    });
                    if !ok2 {
                        return Outcome::Overflow("fused2: internal error in a float task".into());
                    }
                } else {
                    for r in 0..t {
                        let rinv = rinv_row(&p.h[r * hidden..(r + 1) * hidden], cfg.eps, &mut tmp);
                        normed_row(&p.z[r * m2..(r + 1) * m2], l2, svals[r], rinv, &mut p.gu[r * m2..(r + 1) * m2]);
                    }
                    silu_rows(&p.gu[..t * 2 * inter], &mut p.act[..t * inter], inter);
                }
            } else {
                // trip B: the next layer's qkv (normed), then its attention
                p.qkv.resize(t * m2, 0.0);
                for r in 0..t {
                    let rinv = rinv_row(&p.h[r * hidden..(r + 1) * hidden], cfg.eps, &mut tmp);
                    normed_row(&p.z[r * m2..(r + 1) * m2], l2, svals[r], rinv, &mut p.qkv[r * m2..(r + 1) * m2]);
                }
                attend(cfg, p, layer + 1, kc, vc, cos, sin);
            }
        }
        p.pos = pos + 1;
        let te = Instant::now();
        f2_phase_add(15 + kind, ns(te - tfl));
        f2_phase_add(3 + kind, ns(te - t2));
        st.local_ns += ns(te - t2);
    }
    wipe(&mut tmp);
    Outcome::Done
}

/// RoPE + KV-cache write + attention of `layer` from `p.qkv` into `p.act` (the native float ops' bits).
fn attend<T: Kv>(cfg: &View<'_>, p: &mut Pass, layer: usize, kc: &mut [T], vc: &mut [T], cos: &[f32], sin: &[f32]) {
    let t = p.t;
    let (nh, kvh, hd) = (cfg.nh, cfg.kvh, cfg.hd);
    let lsz = cfg.cap * kvh * hd;
    let kl = &mut kc[layer * lsz..(layer + 1) * lsz];
    let vl = &mut vc[layer * lsz..(layer + 1) * lsz];
    let dm = Dims { nh, kvh, hd, scale: cfg.attn_scale };
    let threads = auto_threads(t, p.start, cfg.attn_threads);
    p.act.resize(t * (nh * hd).max(cfg.inter), 0.0);
    let ycols = (nh + 2 * kvh) * hd;
    rope_attend(&p.qkv[..t * ycols], cos, sin, kl, vl, t, p.start, dm, &mut p.act[..t * nh * hd], threads,
                cfg.attn_block);
}

fn f64_vec(a: &Bound<'_, PyArray1<f64>>, name: &str, len: usize) -> PyResult<Vec<f64>> {
    let ro = a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?;
    let v = ro.as_slice().map_err(|_| verr(format!("{name} must be contiguous")))?.to_vec();
    if v.len() != len {
        return Err(verr(format!("{name} must have {len} entries")));
    }
    Ok(v)
}

fn p61_vec(a: &Bound<'_, PyArray2<i64>>, name: &str, cols: usize) -> PyResult<Vec<u64>> {
    let ro = a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?;
    if ro.shape()[1] != cols || ro.shape()[0] == 0 {
        return Err(verr(format!("{name} must be (c, {cols})")));
    }
    let s = ro.as_slice().map_err(|_| verr(format!("{name} must be C-contiguous")))?;
    if s.iter().any(|&v| v < 0 || v as u64 >= P61) {
        return Err(verr(format!("{name}: check vectors must be in [0, p)")));
    }
    Ok(s.iter().map(|&v| v as u64).collect())
}

/// The native fused2 decode driver of one model (see the module docs). One pass at a time.
#[pyclass(module = "pgw_fast")]
pub struct Fused2Driver {
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
    k: u32,
    a_bits: u32,
    c_bits: u32,
    trips: Vec<TripCfg>,
    pass: Pass,
}

impl Fused2Driver {
    /// `(n1, m1, m2)` of trip `idx` (`m2 = 0`: single).
    fn widths(&self, idx: usize) -> (usize, usize, usize) {
        let (h, q, i2) = (self.hidden, (self.nh + 2 * self.kvh) * self.hd, 2 * self.inter);
        if idx == 0 {
            (h, q, 0)
        } else if (idx - 1) % 2 == 0 {
            (self.nh * self.hd, h, i2)
        } else if idx + 1 == 2 * self.layers + 1 {
            (self.inter, h, 0)
        } else {
            (self.inter, h, q)
        }
    }
}

#[pymethods]
impl Fused2Driver {
    /// ``Fused2Driver(hidden, nh, kvh, hd, inter, attn_scale, eps, attn_threads, attn_block, kv_bf16, layers, k, a, c)``.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(hidden: usize, nh: usize, kvh: usize, hd: usize, inter: usize, attn_scale: f64, eps: f64,
           attn_threads: usize, attn_block: usize, kv_bf16: bool, layers: usize, k: u32, a: u32, c: u32) -> PyResult<Self> {
        if hidden == 0 || nh == 0 || kvh == 0 || nh % kvh != 0 || hd == 0 || hd % 2 != 0 || inter == 0 || layers == 0 {
            return Err(verr("Fused2Driver: bad dimensions"));
        }
        if k % 8 != 0 || !(8..=64).contains(&k) || !(2..=24).contains(&a) || c == 0 || c + 1 >= 61 || c >= k {
            return Err(verr("Fused2Driver: ring must have K a multiple of 8 <= 64, 2 <= A <= 24, C < K, 2^(C+1) < p"));
        }
        Ok(Fused2Driver {
            hidden, nh, kvh, hd, inter, attn_scale, eps, attn_threads, attn_block, kv_bf16, layers, k,
            a_bits: a, c_bits: c,
            trips: (0..2 * layers + 1).map(|_| TripCfg::default()).collect(),
            pass: Pass::default(),
        })
    }

    /// Trips per pass (``2L + 1``).
    #[getter]
    fn trips(&self) -> usize {
        2 * self.layers + 1
    }

    /// The public constants of trip ``idx``'s linear ``stage`` (1: the first, 2: the folded second of a pair):
    /// ``scale`` (float64 ``(m,)``), ``bias`` or None, ``grid`` (``scale / hmul``; stage 1) or None, ``hmul``
    /// (int64) or None, ``wnorm``, ``sigma``; and the trip's lane item index.
    #[pyo3(signature = (idx, stage, n, m, scale, bias, grid, hmul, wnorm, sigma, item))]
    #[allow(clippy::too_many_arguments)]
    fn set_linear(&mut self, idx: usize, stage: u8, n: usize, m: usize, scale: &Bound<'_, PyArray1<f64>>,
                  bias: Option<&Bound<'_, PyArray1<f64>>>, grid: Option<&Bound<'_, PyArray1<f64>>>,
                  hmul: Option<&Bound<'_, PyArray1<i64>>>, wnorm: f64, sigma: f64, item: u16) -> PyResult<()> {
        if idx >= self.trips.len() {
            return Err(verr("trip index out of range"));
        }
        let (n1, m1, m2) = self.widths(idx);
        let ok = match stage {
            1 => n == n1 && m == m1,
            2 => m2 > 0 && n == m1 && m == m2,
            _ => false,
        };
        if !ok {
            return Err(verr(format!("trip {idx} stage {stage}: shape ({m}, {n}) does not match the layer")));
        }
        let scale = f64_vec(scale, "scale", m)?;
        let bias = bias.map(|b| f64_vec(b, "bias", m)).transpose()?;
        let grid = match grid {
            Some(g) => f64_vec(g, "grid", m)?,
            None => scale.clone(),
        };
        let hmul = match hmul {
            Some(h) => {
                let ro = h.try_readonly().map_err(|e| verr(format!("hmul: {e}")))?;
                let s = ro.as_slice()?;
                if s.len() != m || s.iter().any(|&v| v < 1) {
                    return Err(verr("hmul must have m entries >= 1"));
                }
                Some(s.iter().map(|&v| v as u64).collect())
            }
            None => None,
        };
        if !(wnorm.is_finite() && wnorm >= 0.0 && sigma.is_finite() && sigma >= 0.0) {
            return Err(verr("wnorm / sigma must be finite and >= 0"));
        }
        let lin = Lin { n, m, scale, bias, grid, hmul, wnorm, sigma };
        let tc = &mut self.trips[idx];
        tc.item = item;
        if stage == 1 {
            tc.l1 = Some(lin);
        } else {
            tc.l2 = Some(lin);
        }
        Ok(())
    }

    /// The secret check vectors of trip ``idx``'s ``stage``: ``u (c, m)`` and ``uw = u hmul W mod p (c, n)``.
    fn set_checks(&mut self, idx: usize, stage: u8, u: &Bound<'_, PyArray2<i64>>, uw: &Bound<'_, PyArray2<i64>>) -> PyResult<()> {
        if idx >= self.trips.len() {
            return Err(verr("trip index out of range"));
        }
        let (n1, m1, m2) = self.widths(idx);
        let (n, m) = match stage {
            1 => (n1, m1),
            2 if m2 > 0 => (m1, m2),
            _ => return Err(verr("bad stage")),
        };
        let (u, uw) = (p61_vec(u, "u", m)?, p61_vec(uw, "uw", n)?);
        if u.len() / m != uw.len() / n {
            return Err(verr("u and uw must have the same number of check vectors"));
        }
        let c = Checks { u, uw };
        if stage == 1 {
            self.trips[idx].c1 = Some(c);
        } else {
            self.trips[idx].c2 = Some(c);
        }
        Ok(())
    }

    /// The outbound-noise monitor's duplicate-row fingerprint parameters of trip ``idx`` (or None).
    #[pyo3(signature = (idx, dup = None))]
    #[allow(clippy::type_complexity)]
    fn set_dup(&mut self, idx: usize,
               dup: Option<(Bound<'_, PyArray1<i64>>, u32, Bound<'_, PyArray1<u64>>, usize, usize, usize)>) -> PyResult<()> {
        if idx >= self.trips.len() {
            return Err(verr("trip index out of range"));
        }
        let n = self.widths(idx).0;
        self.trips[idx].dup = match dup {
            Some((cols, shift, mult, groups, per, limit)) => {
                let cols = cols.try_readonly().map_err(|e| verr(format!("cols: {e}")))?.as_slice()?.to_vec();
                let mult = mult.try_readonly().map_err(|e| verr(format!("mult: {e}")))?.as_slice()?.to_vec();
                if !crate::tripnative::DupIn::valid(&cols, shift, &mult, groups, per, n) {
                    return Err(verr("bad fingerprint parameters"));
                }
                Some(DupCfg { cols, shift, mult, groups, per, limit })
            }
            None => None,
        };
        Ok(())
    }

    /// Start a pass with the embedded rows ``h (t, hidden)`` (float32) at cache position ``start``.
    fn begin(&mut self, h: &Bound<'_, PyArray2<f32>>, start: usize) -> PyResult<()> {
        self.pass.wipe();
        for (i, tc) in self.trips.iter().enumerate() {
            let (_, _, m2) = self.widths(i);
            let last = i + 1 == self.trips.len();
            let need2 = m2 > 0 && !last;
            if tc.l1.is_none() || tc.c1.is_none() || (need2 && (tc.l2.is_none() || tc.c2.is_none())) {
                return Err(verr(format!("trip {i} not configured")));
            }
        }
        let hr = h.try_readonly().map_err(|e| verr(format!("h: {e}")))?;
        if hr.shape()[1] != self.hidden || hr.shape()[0] == 0 || hr.shape()[0] > u16::MAX as usize {
            return Err(verr("h must be (t, hidden)"));
        }
        let t = hr.shape()[0];
        let hs = hr.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
        let p = &mut self.pass;
        p.t = t;
        p.start = start;
        p.pos = 0;
        p.h.clear();
        p.h.extend(hs.iter().map(|&v| f64::from(v)));
        p.active = true;
        Ok(())
    }

    /// The next trip's index.
    #[getter]
    fn pos(&self) -> usize {
        self.pass.pos
    }

    /// Run trips from :attr:`pos` up to (not including) ``end``. ``masks``: per trip from :attr:`pos` a tuple
    /// ``(r_x, wr_x, v, wv)`` of int64 ``(t, .)`` arrays (``v``/``wv`` None for a single trip); ``flags`` bit 1:
    /// hand back the trip's ``(x_hat, xq)`` (noise sample); ``rids`` 16 bytes per trip; ``lane``: ``(channel,
    /// ring index, deadline_ms, timeout_s)``. Returns ``(code, info, stats, sampled, recorded)`` (``recorded``: the
    /// payloads of trips with flag bit 2, a test hook).
    #[pyo3(signature = (end, masks, flags, rids, kc, vc, cos, sin, lane, dup_hist = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        end: usize,
        masks: &Bound<'py, PyList>,
        flags: &[u8],
        rids: &[u8],
        kc: &Bound<'py, PyAny>,
        vc: &Bound<'py, PyAny>,
        cos: &Bound<'py, PyArray2<f32>>,
        sin: &Bound<'py, PyArray2<f32>>,
        lane: (Bound<'py, TripChannel>, u8, u32, f64),
        dup_hist: Option<Bound<'py, DupHistory>>,
    ) -> PyResult<(u8, PyObject, PyObject, PyObject, PyObject)> {
        if !self.pass.active {
            return Err(verr("no pass in progress (begin first)"));
        }
        let t = self.pass.t;
        let hd = self.hd;
        let first = self.pass.pos;
        if end > self.trips.len() || end <= first {
            return Err(verr("end must be after pos and within the pass"));
        }
        let count = end - first;
        if masks.len() < count || flags.len() < count || rids.len() < 16 * count {
            return Err(verr("masks/flags/rids must cover every trip to end"));
        }
        let cr = cos.try_readonly().map_err(|e| verr(format!("cos: {e}")))?;
        let sr = sin.try_readonly().map_err(|e| verr(format!("sin: {e}")))?;
        if cr.shape() != [t, hd] || sr.shape() != [t, hd] {
            return Err(verr("cos/sin must be (t, head_dim)"));
        }
        let cs = cr.as_slice().map_err(|_| verr("cos must be C-contiguous"))?;
        let ss = sr.as_slice().map_err(|_| verr("sin must be C-contiguous"))?;
        // masks (borrowed read-only; Python wipes them after the call)
        type Ro<'a> = numpy::PyReadonlyArray2<'a, i64>;
        let mut held: Vec<(Ro<'py>, Ro<'py>, Option<Ro<'py>>, Option<Ro<'py>>)> = Vec::with_capacity(count);
        for i in 0..count {
            let idx = first + i;
            let (n1, m1, m2) = self.widths(idx);
            let pair = m2 > 0 && idx + 1 != self.trips.len();
            let tup = masks.get_item(i)?;
            let get = |j: usize, cols: usize, name: &str| -> PyResult<Option<Ro<'py>>> {
                let o = tup.get_item(j)?;
                if o.is_none() {
                    return Ok(None);
                }
                let a = o.downcast_into::<PyArray2<i64>>().map_err(|_| verr(format!("{name} must be int64 2-D")))?;
                let ro = a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?;
                if ro.shape() != [t, cols] || !ro.is_c_contiguous() {
                    return Err(verr(format!("trip {idx}: {name} must be ({t}, {cols}) C-contiguous")));
                }
                Ok(Some(ro))
            };
            let r_x = get(0, n1, "r_x")?.ok_or_else(|| verr("r_x missing"))?;
            let wr_x = get(1, m1, "wr_x")?.ok_or_else(|| verr("wr_x missing"))?;
            let (v, wv) = if pair {
                (Some(get(2, m1, "v")?.ok_or_else(|| verr("v missing"))?),
                 Some(get(3, m2, "wv")?.ok_or_else(|| verr("wv missing"))?))
            } else {
                (None, None)
            };
            held.push((r_x, wr_x, v, wv));
        }
        let mask_in: Vec<MaskIn<'_>> = held
            .iter()
            .map(|(a, b, c, d)| -> PyResult<MaskIn<'_>> {
                Ok(MaskIn {
                    r_x: a.as_slice()?,
                    wr_x: b.as_slice()?,
                    v: c.as_ref().map(|x| x.as_slice()).transpose()?,
                    wv: d.as_ref().map(|x| x.as_slice()).transpose()?,
                })
            })
            .collect::<PyResult<_>>()?;
        let rid_v: Vec<[u8; 16]> = (0..count).map(|i| rids[16 * i..16 * i + 16].try_into().expect("16")).collect();
        let (ch_b, ring, deadline_ms, to) = &lane;
        if !(to.is_finite() && *to > 0.0 && *to < 1e7) {
            return Err(verr("timeout_s must be a positive number of seconds"));
        }
        let ch = ch_b.get();
        if t > usize::from(ch.max_rows_raw()) {
            return Err(verr("too many rows for the lane channel"));
        }
        let lane_in = LaneIn { ch, ring: *ring, deadline_ms: *deadline_ms, timeout: Duration::from_secs_f64(*to) };
        let hist = dup_hist.as_ref().map(|h| h.get());
        // the KV cache
        let mut kf = None;
        let mut kb = None;
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
        let view = View {
            hidden: self.hidden, nh: self.nh, kvh: self.kvh, hd, inter: self.inter, attn_scale: self.attn_scale,
            eps: self.eps, attn_threads: self.attn_threads, attn_block: self.attn_block, layers: self.layers,
            cap: shape[1], k: self.k, a_bits: self.a_bits, c_bits: self.c_bits, trips: &self.trips,
        };
        let pass = &mut self.pass;
        let mut sampled: Vec<(usize, Vec<i64>, Vec<i64>)> = Vec::new();
        let mut recorded: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut stats = Stats::default();
        // the pool's workers stay warm for this call (PGW_POOL_ACTIVE_SPIN_US; timing only)
        let _warm = crate::prepare::pool::Warm::new();
        let outcome = if let Some((k, v)) = kf.as_mut() {
            let (ks, vs) = (k.as_slice_mut()?, v.as_slice_mut()?);
            py.allow_threads(|| run_loop::<f32>(&view, pass, &mask_in, flags, &rid_v, end, &lane_in, hist, ks, vs, cs, ss,
                                               &mut sampled, &mut recorded, &mut stats))
        } else if let Some((k, v)) = kb.as_mut() {
            let (ks, vs) = (k.as_slice_mut()?, v.as_slice_mut()?);
            py.allow_threads(|| run_loop::<u16>(&view, pass, &mask_in, flags, &rid_v, end, &lane_in, hist, ks, vs, cs, ss,
                                               &mut sampled, &mut recorded, &mut stats))
        } else {
            unreachable!()
        };
        drop(mask_in);
        drop(held);
        let st = &stats;
        let stats_t = PyTuple::new(py, [st.trips, st.rows, st.bytes_up, st.bytes_down, st.checks, st.remote_ns,
                                        st.local_ns, st.compute_us, st.dup_rows, st.dups])?;
        let samp = PyList::empty(py);
        for (pos, xh, xq) in sampled {
            let n = self.widths(pos).0;
            let a = PyArray1::from_vec(py, xh).reshape([t, n])?;
            let b = PyArray1::from_vec(py, xq).reshape([t, n])?;
            samp.append((pos, a, b))?;
        }
        let rec = PyList::empty(py);
        for (pos, pl) in recorded {
            rec.append((pos, pyo3::types::PyBytes::new(py, &pl)))?;
        }
        let at = self.pass.pos;
        let (code, info): (u8, PyObject) = match outcome {
            Outcome::Done => (DONE, py.None()),
            Outcome::Status(s) => (LANE_STATUS, (at, s).into_pyobject(py)?.into_any().unbind()),
            Outcome::Overflow(msg) => (OVERFLOW, (at, msg).into_pyobject(py)?.into_any().unbind()),
            Outcome::NonFinite => (NONFINITE, at.into_pyobject(py)?.into_any().unbind()),
            Outcome::Duplicate(d) => (DUPLICATE, (at, d).into_pyobject(py)?.into_any().unbind()),
            Outcome::Check { stage, code } => (CHECK_FAILED, (at, stage, code).into_pyobject(py)?.into_any().unbind()),
            Outcome::Lane(name) => (LANE_ERROR, (at, name).into_pyobject(py)?.into_any().unbind()),
        };
        if code != DONE {
            self.pass.wipe();
        }
        Ok((code, info, stats_t.into_any().unbind(), samp.into_any().unbind(), rec.into_any().unbind()))
    }

    /// End the pass: the residual stream as float32 into ``h_out (t, hidden)`` (``st.hs``); every pass buffer is
    /// zeroed.
    fn end(&mut self, h_out: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
        let p = &mut self.pass;
        if !p.active || p.pos != 2 * self.layers + 1 {
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
            for (o, &v) in hs.iter_mut().zip(&p.h[..n]) {
                *o = v as f32;
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
}

/// Test hook: numpy's pairwise sum of a float64 vector ([`pairwise_sum`]).
#[pyfunction]
fn f2_pairwise_sum(a: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let ro = a.try_readonly().map_err(|e| verr(format!("a: {e}")))?;
    Ok(pairwise_sum(ro.as_slice()?))
}

/// Test hook: one row of ``quantize_pair`` (``h``, ``grid`` given) or ``quantize_single`` (``h`` None):
/// ``(s, q, rr)`` or None when no bit count fits.
#[pyfunction]
#[pyo3(signature = (a, h, grid, wnorm1, sigma1, wnorm2, a_bits, c_bits))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn f2_quantize<'py>(py: Python<'py>, a: &Bound<'py, PyArray1<f64>>, h: Option<&Bound<'py, PyArray1<f64>>>,
                    grid: Option<&Bound<'py, PyArray1<f64>>>, wnorm1: f64, sigma1: f64, wnorm2: Option<f64>,
                    a_bits: u32, c_bits: u32)
                    -> PyResult<Option<(f64, Bound<'py, PyArray1<f64>>, Option<Bound<'py, PyArray1<f64>>>)>> {
    let a = a.try_readonly()?.as_slice()?.to_vec();
    let mut q = vec![0f64; a.len()];
    let mut tmp = Vec::new();
    match (h, grid) {
        (Some(h), Some(g)) => {
            let h = h.try_readonly()?.as_slice()?.to_vec();
            let g = g.try_readonly()?.as_slice()?.to_vec();
            let m = h.len();
            let l1 = Lin { n: a.len(), m, scale: g.clone(), bias: None, grid: g, hmul: None, wnorm: wnorm1, sigma: sigma1 };
            let l2 = wnorm2.map(|w| Lin { n: m, m: 0, scale: vec![], bias: None, grid: vec![], hmul: None, wnorm: w, sigma: 0.0 });
            let mut rr = vec![0f64; m];
            Ok(quantize_pair_row(&a, &h, &l1, l2.as_ref(), a_bits, c_bits, &mut q, &mut rr, &mut tmp)
                .map(|s| (s, PyArray1::from_vec(py, q), Some(PyArray1::from_vec(py, rr)))))
        }
        _ => Ok(quantize_single_row(&a, wnorm1, a_bits, c_bits, &mut q, &mut tmp).map(|s| (s, PyArray1::from_vec(py, q), None))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_definition_small() {
        let a: Vec<f64> = (0..300).map(|i| (i as f64) * 0.1 + 1e-3).collect();
        let s = pairwise_sum(&a);
        let exact: f64 = a.iter().sum();
        assert!((s - exact).abs() < 1e-9);
        assert_eq!(pairwise_sum(&a[..5]), (((-0.0 + a[0]) + a[1]) + a[2] + a[3]) + a[4]);
    }

    #[test]
    fn center_and_pack() {
        assert_eq!(center((1u64 << 56) - 1, 56), -1);
        assert_eq!(center(5, 56), 5);
        assert_eq!(center(1u64 << 55, 56), -(1i64 << 55));
        let mut out = [0u8; 14];
        pack_into([0x01020304050607u64, (1u64 << 56) - 1].into_iter(), 7, &mut out);
        assert_eq!(unpack_at(&out, 7, 0), 0x01020304050607);
        assert_eq!(unpack_at(&out, 7, 1), (1u64 << 56) - 1);
        // the 8-byte fast paths give the byte-copy reference's bytes and values (high bits dropped)
        for nb in [5usize, 6, 7, 8] {
            let vals: Vec<u64> = (0..37u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i << 60)).collect();
            let mut fast = vec![0u8; vals.len() * nb];
            pack_into(vals.iter().copied(), nb, &mut fast);
            let mut slow = vec![0u8; vals.len() * nb];
            for (c, v) in slow.chunks_exact_mut(nb).zip(&vals) {
                c.copy_from_slice(&v.to_le_bytes()[..nb]);
            }
            assert_eq!(fast, slow);
            let keep = if nb == 8 { u64::MAX } else { (1u64 << (8 * nb)) - 1 };
            for (i, v) in vals.iter().enumerate() {
                assert_eq!(unpack_at(&fast, nb, i), v & keep);
            }
        }
    }
}
