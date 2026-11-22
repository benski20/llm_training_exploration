//! Native trip kernels for the gateway's TripEngine (chat-speed plan WP-A4a).
//!
//! The per-trip prepare and finish of a masked linear, writing into caller-owned buffers that the
//! engine reuses from trip to trip, with the data-dependent choices made in constant time:
//!
//! | function | replaces | notes |
//! |---|---|---|
//! | [`prepare_into`] (`trip_prepare_into`) | `quantize_mask_pack` ([`crate::prepare`]) | same payload, `sx` and `x_hat` bits; constant-time width selection; optional fused Freivalds input half `uw . x_hat` |
//! | [`finish_into`] (`trip_finish_into`) | `finish_group` ([`crate::finish`]) | same output bits and statuses; output into a reused buffer; checks fused, split (input half precomputed) or off |
//! | [`freivalds_output`] (`trip_freivalds_output`, `trip_freivalds_submit` -> [`TripCheck`]) | the check part of `finish_group` | the output half `u . y_hat` against the precomputed input half, synchronously or on a native helper thread |
//!
//! Row blocks of a prefill are spread over the persistent prepare pool ([`crate::prepare::pool`]),
//! in prepare as well as in finish; rows (and output columns) are independent, so the split changes
//! no byte.
//!
//! # Constant-time width selection (capped rings)
//!
//! The reference (`pgw.core.ring._quantize_capped`, [`crate::prepare`]) tries `bits = A, A-1, ..., 2`
//! and keeps the first width whose quantized row satisfies `sqrt(sum q^2) * wnorm * (1 + 1e-9) <=
//! 2^cap`. The number of widths it evaluates depends on the activations, so the time of a trip
//! would reveal how large the row's norm is relative to its maximum. Here every row evaluates the
//! same number `W` of widths, where `W` depends only on public values (`n`, `A`, `cap`, `wnorm`):
//!
//! * With `Q_b = 2^(b-1) - 1` and `tau = ||x||_2 / max|x|`, the quantized row satisfies
//!   `| ||q_b||_2 - Q_b tau | <= E_b = sqrt(n) (1/2 + Q_b 2^-48)` (rounding to integers, plus the
//!   float64 division error), so with `T = 2^cap / (wnorm (1 + 1e-9))` width `b` *possibly fits*
//!   iff `tau <= P_b` and *certainly fits* iff `tau <= D_b`, for thresholds `P_b`, `D_b` that carry
//!   a relative margin of `1e-6` (much larger than any float64 rounding involved, for `n <= 2^31`).
//! * The reference picks the largest fitting width, so for a row it is among `[lo, hi]`, where `hi`
//!   is the largest possibly-fitting width and `lo` the largest certainly-fitting one (or 2). The
//!   widths above `hi` provably fail the reference test and `lo` provably passes it.
//! * `W` is the largest `hi - lo + 1` over all `tau`; `[lo, hi]` only changes at the thresholds, so
//!   evaluating every threshold gives the exact maximum. For the 7B linears it is 2.
//! * Every row evaluates the window `[start, start + W)` with `start = min(lo, A + 1 - W)`, which
//!   contains `[lo, hi]`. Each width is quantized exactly as the reference does, its sum of squares
//!   is tested with the reference's float64 expression, and the largest passing width is selected
//!   with masks, not branches. The result is therefore bit-identical to the descending search.
//! * If the parameters are degenerate (`wnorm` not finite and positive, `n > 2^31`), `W = A - 1`
//!   (every width, still constant time). Uncapped rings have one width (`W = 1`).
//! * The reference's sum of squares is the exact integer sum for widths with `Q_b < 2^31` and
//!   `n Q_b^2 < 9e18`, and a sequential float64 sum (a serial dependency chain, much slower) above.
//!   Which one a width takes depends on the width, hence on the data. The plan therefore fixes the
//!   work from public values only ([`SumPath`]): if every width in `[2, A]` takes the same kind,
//!   only that kind runs; otherwise (`Both`: capped rings with a large `A`, e.g. K56A48C53 at
//!   n = 18944) every evaluated width computes both sums in one pass and selects the reference's
//!   one with a mask (`b <= b_int`). The default K32A12C27 is `Int` at every 7B width.
//!
//! # Constant-time review notes
//!
//! * No loop count depends on activation values, masks or check vectors: row loops run over `n`,
//!   width loops over `W` or `A - 1`, check loops over `c` and `m`. Branches on data exist only on
//!   error paths that abort the call (non-finite input, no width fits, which the reference raises
//!   too) and in an unreachable guard (no width of the window fits although `lo` provably does: it
//!   falls back to the full descending search and is covered by `debug_assert`).
//! * Quantization always uses the true float64 division (no reciprocal fast path with a near-tie
//!   re-run, whose trigger would depend on the data). `x` is float32 widened to float64, so neither
//!   operand nor quotient is subnormal (no microcode assists on x86).
//! * The window start, chosen width and its scale are selected with integer masks; per element the
//!   selected `q` is an and/or over all `W` candidates (no secret-dependent addresses).
//! * The all-zero test of `u` reads every word; the Freivalds verdict compares every check vector.
//! * Residual: the latency of `divpd` / `sqrtsd` on normal operands varies by at most a cycle or two
//!   on some x86 cores (Skylake: 13-14 cycles); the quantization loop is throughput-bound, so this is
//!   nanoseconds per call against microseconds of network jitter. Callers must still not branch on
//!   the chosen widths (`sx`) outside these kernels.
//! * Measured here (4 shared vCPUs, K32A12C27, one row): prepare n=3584 11.3 us (descending search
//!   9.5 us), n=18944 54 us (102 us); the fused input half adds about 1 us at n=3584. Handing the
//!   output half to the helper thread costs the submitting thread about 11 us (thread wake-up);
//!   the output half itself is 6.6 us (m=4608) to 36 us (m=37888).
//! * Large-A capped rings cost more, publicly: K56A48C53 at n=18944, wnorm=2^25 runs `Both` sums
//!   (W=2): 112.7 us for a Gaussian row (width 24) and 112.7 us for a spike row (width 28),
//!   interleaved medians (before the fix, 79 vs 105 us). K16A8C16 with realistic `wnorm` has
//!   `d_b < 0` for every width, so W = A - 1 = 7 at n=18944 (4 at n=3584): about 3.5x the 7B W=2 cost.
//! * `threads > 1` only sets the task size of prefill blocks; the persistent pool decides how many
//!   workers pick them up.

use crate::finish::{
    self, decode, dot, wipe, Exec, Group, GroupChecks, Part, RoRing, Src, Word, WrBuf, BLOCK, STATUS_FREIVALDS,
    STATUS_OK, STATUS_U_ZERO,
};
use crate::prepare::{
    biased_to_i64, pool, qmax_of, quant_biased, row_amax_bits, sum_squares, PrepareError, PrepareParams, INF_BITS,
    MANY_PAR_MIN_ELEMS, NORM_SLACK,
};
use numpy::{PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Status of [`TripCheck::wait`] when the helper thread failed internally (fail closed).
pub const STATUS_INTERNAL: u8 = 4;

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trip_width_window, m)?)?;
    m.add_function(wrap_pyfunction!(trip_prepare_into, m)?)?;
    m.add_function(wrap_pyfunction!(trip_finish_into, m)?)?;
    m.add_function(wrap_pyfunction!(trip_freivalds_output, m)?)?;
    m.add_function(wrap_pyfunction!(trip_freivalds_submit, m)?)?;
    m.add_class::<TripCheck>()?;
    Ok(())
}

// ---- width plan ----------------------------------------------------------------------------------

/// Relative margin of the width bounds (see the module docs).
const MU: f64 = 1e-6;
/// Bound on the float64 division error per element, in units of `Q_b` (`2^-52` would do).
const EPS_Q: f64 = 1.0 / (1u64 << 48) as f64;
/// Widths are indexed by bit count (`<= 48`).
const NB_IDX: usize = 49;

/// How the sum of squares of a candidate width is formed (public, per plan; see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumPath {
    /// Uncapped ring: no sum of squares.
    Off,
    /// Every width the plan can evaluate takes [`crate::prepare::sum_squares`]' exact integer sum.
    Int,
    /// Every such width takes its sequential float64 sum.
    Float,
    /// Both kinds occur among the evaluable widths: every width computes both sums and selects the
    /// reference's one with a mask (`b <= b_int`), so the work does not depend on the chosen width.
    Both,
}

/// Whether [`crate::prepare::sum_squares`] sums width `qmax` rows of `n` elements as integers (its
/// exact condition; monotone: true up to some width, false above).
#[inline(always)]
fn int_sum_ok(n: usize, qmax: f64) -> bool {
    qmax < 2_147_483_648.0 && (n as f64) * qmax * qmax < 9.0e18
}

/// Public per-linear plan of the constant-time width selection.
#[derive(Clone, Debug)]
pub struct WidthPlan {
    a: u32,
    /// Widths evaluated per row.
    w: u32,
    /// `P_b` / `D_b` apply (capped ring with sane parameters).
    bounded: bool,
    p: [f64; NB_IDX],
    d: [f64; NB_IDX],
    /// The summation path(s) every evaluated width runs.
    sum: SumPath,
    /// Largest width whose reference sum is the integer one (1 if none).
    b_int: u32,
}

impl WidthPlan {
    /// The plan for rows of width `n` (parameters must be valid).
    pub fn new(n: usize, prm: PrepareParams) -> WidthPlan {
        let a = prm.a;
        let mut plan =
            WidthPlan { a, w: 1, bounded: false, p: [0.0; NB_IDX], d: [0.0; NB_IDX], sum: SumPath::Off, b_int: 1 };
        if prm.cap == 0 {
            return plan; // uncapped: the single width A
        }
        // Which reference summation each width takes depends on the width, which is secret, so a
        // plan whose evaluable widths (any of [2, A]) mix both kinds runs both for every width.
        plan.b_int = (2..=a).filter(|&b| int_sum_ok(n, qmax_of(b))).max().unwrap_or(1);
        plan.sum = if plan.b_int >= a {
            SumPath::Int
        } else if plan.b_int < 2 {
            SumPath::Float
        } else {
            SumPath::Both
        };
        let limit = (1u64 << prm.cap) as f64;
        let t = limit / (prm.wnorm * NORM_SLACK);
        let sane = prm.wnorm.is_finite() && prm.wnorm > 0.0 && t.is_finite() && t > 0.0 && n > 0 && n <= (1 << 31);
        if !sane {
            plan.w = a - 1; // every width (window [2, A])
            return plan;
        }
        plan.bounded = true;
        let sn = (n as f64).sqrt();
        for b in 2..=a {
            let q = qmax_of(b);
            let e = sn * (0.5 + q * EPS_Q) * (1.0 + MU);
            plan.p[b as usize] = (t * (1.0 + 2.0 * MU) + e) / (q * (1.0 - MU));
            plan.d[b as usize] = (t * (1.0 - 2.0 * MU) - e) / (q * (1.0 + MU));
        }
        // [lo, hi] is constant between consecutive thresholds (and equals its value at the upper
        // one), so the maximum over all tau is the maximum over the thresholds and 0.
        let mut w = 1u32;
        let mut eval = |tau: f64| {
            let (lo, hi) = plan.lo_hi(tau);
            if hi >= lo {
                w = w.max(hi - lo + 1);
            }
        };
        eval(0.0);
        for b in 2..=a as usize {
            eval(plan.p[b]);
            eval(plan.d[b]);
        }
        plan.w = w.min(a - 1);
        plan
    }

    /// Widths evaluated per row.
    pub fn window(&self) -> u32 {
        self.w
    }

    /// The summation path every evaluated width runs (public).
    pub fn sum_path(&self) -> SumPath {
        self.sum
    }

    /// `(lo, hi)`: largest certainly-fitting width (2 if none) and largest possibly-fitting width
    /// (1 if none). Scans every width (no early exit).
    #[inline(always)]
    fn lo_hi(&self, tau: f64) -> (u32, u32) {
        let (mut lo, mut hi) = (2u32, 1u32);
        for b in 2..=self.a {
            hi = hi.max(b * u32::from(tau <= self.p[b as usize]));
            lo = lo.max(b * u32::from(tau <= self.d[b as usize]));
        }
        (lo, hi)
    }

    /// First width of the row's window `[start, start + W)`.
    #[inline(always)]
    pub fn start(&self, tau: f64) -> u32 {
        let top = self.a + 1 - self.w;
        if !self.bounded {
            return top; // uncapped: A; degenerate: 2
        }
        self.lo_hi(tau).0.min(top)
    }
}

/// Cached plans (a plan costs up to ~10 us to build at A = 48; there is one per linear).
fn plan_for(n: usize, prm: PrepareParams) -> Arc<WidthPlan> {
    type Key = (usize, u32, u32, u32, u64);
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<WidthPlan>>>> = OnceLock::new();
    let key = (n, prm.k, prm.a, prm.cap, if prm.cap == 0 { 0 } else { prm.wnorm.to_bits() });
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Arc::clone(p);
    }
    let plan = Arc::new(WidthPlan::new(n, prm));
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    if c.len() >= 4096 {
        c.clear();
    }
    c.insert(key, Arc::clone(&plan));
    plan
}

// ---- prepare -------------------------------------------------------------------------------------

/// Sum of `f64(x)^2` (only used for the width bounds, which allow for its rounding error).
#[inline(always)]
fn sumsq(x: &[f32]) -> f64 {
    let mut acc = [0.0f64; 8];
    let mut c = x.chunks_exact(8);
    for xs in &mut c {
        for i in 0..8 {
            let v = f64::from(xs[i]);
            acc[i] += v * v;
        }
    }
    let tail: f64 = c.remainder().iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    acc.iter().sum::<f64>() + tail
}

/// Quantize one row with the true float64 division (the reference operation sequence).
#[inline(always)]
fn quant_exact(xr: &[f32], s: f64, qmax: f64, q: &mut [i64]) {
    for (qv, v) in q.iter_mut().zip(xr) {
        *qv = biased_to_i64(quant_biased(*v, s, qmax));
    }
}

/// Rows that took the (provably unreachable) full-search fallback; tests assert it stays 0.
static WINDOW_MISSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`quant_exact`] fused with the sums of squares the plan needs: the exact integer sum (when
/// `INT`; wrapping integer addition is associative, so the lane order does not matter) and the
/// sequential float64 sum (when `FLT`; the same additions in the same order as
/// [`crate::prepare::sum_squares`]).
#[inline(always)]
fn quant_ss<const INT: bool, const FLT: bool>(xr: &[f32], s: f64, qmax: f64, q: &mut [i64]) -> (i64, f64) {
    let mut acc = [0i64; 4];
    let mut fs = 0.0f64;
    let mut xc = xr.chunks_exact(4);
    let mut qc = q.chunks_exact_mut(4);
    for (xs, qs) in (&mut xc).zip(&mut qc) {
        for i in 0..4 {
            let v = biased_to_i64(quant_biased(xs[i], s, qmax));
            qs[i] = v;
            if INT {
                let qi = i64::from(v as i32);
                acc[i] = acc[i].wrapping_add(qi.wrapping_mul(qi));
            }
            if FLT {
                let vf = v as f64;
                fs += vf * vf;
            }
        }
    }
    let mut tail = 0i64;
    for (x, qv) in xc.remainder().iter().zip(qc.into_remainder()) {
        let v = biased_to_i64(quant_biased(*x, s, qmax));
        *qv = v;
        if INT {
            let qi = i64::from(v as i32);
            tail = tail.wrapping_add(qi.wrapping_mul(qi));
        }
        if FLT {
            let vf = v as f64;
            fs += vf * vf;
        }
    }
    (acc.iter().fold(tail, |a, b| a.wrapping_add(*b)), fs)
}

#[cfg(test)]
thread_local! {
    /// Summation paths run, one entry per evaluated width (tests check rows run the same sequence).
    static PATH_LOG: RefCell<Vec<SumPath>> = const { RefCell::new(Vec::new()) };
}

/// [`quant_exact`] and the sum of squares with the same value as [`crate::prepare::sum_squares`].
/// The path is the plan's (public); in [`SumPath::Both`] both sums are computed and `int_mask`
/// (all ones iff this width's reference sum is the integer one) selects without a branch. The float
/// sum starts from `+0.0` where `Iterator::sum` starts from `-0.0`: identical for `n >= 1` terms
/// `>= +0`.
#[inline(always)]
fn quant_exact_ss(xr: &[f32], s: f64, qmax: f64, q: &mut [i64], path: SumPath, int_mask: u64) -> f64 {
    #[cfg(test)]
    PATH_LOG.with(|l| {
        let mut l = l.borrow_mut();
        if l.len() < 1 << 16 {
            l.push(path);
        }
    });
    match path {
        SumPath::Off => {
            quant_exact(xr, s, qmax, q);
            0.0
        }
        SumPath::Int => quant_ss::<true, false>(xr, s, qmax, q).0 as f64,
        SumPath::Float => quant_ss::<false, true>(xr, s, qmax, q).1,
        SumPath::Both => {
            let (i, f) = quant_ss::<true, true>(xr, s, qmax, q);
            f64::from_bits(((i as f64).to_bits() & int_mask) | (f.to_bits() & !int_mask))
        }
    }
}

/// A row's failure (the caller turns it into the reference's error).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowErr {
    NonFinite,
    Overflow,
}

/// Per-thread scratch reused across calls: `q` candidates (`W x n`), `x_hat` words, `uw` blocks.
#[derive(Default)]
struct Scratch {
    q: Vec<i64>,
    xw: Vec<u64>,
    ubuf: Vec<u64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// `len` words of type `W` at the start of `buf` (grown, never shrunk).
fn words_mut<W: Word>(buf: &mut Vec<u64>, len: usize) -> &mut [W] {
    let need = (len * std::mem::size_of::<W>()).div_ceil(8);
    if buf.len() < need {
        buf.resize(need, 0);
    }
    // SAFETY: W is u32 or u64 (Word impls): plain words, alignment <= 8, every bit pattern valid;
    // `need` u64s cover `len` W's.
    unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<W>(), len) }
}

/// Freivalds input-half arguments: `uw (c, n)`.
#[derive(Clone, Copy)]
pub struct InputHalf<'a> {
    /// `(c, n)` check vectors `u W`.
    pub uw: Src<'a>,
    /// Number of check vectors.
    pub c: usize,
}

/// Outputs of one block of rows.
pub struct PrepOut<'a> {
    /// Wire bytes `(rows, n, K/8)`.
    pub payload: &'a mut [u8],
    /// Scales `(rows,)`.
    pub sx: &'a mut [f64],
    /// `x_hat (rows, n)`, or empty.
    pub xhat: &'a mut [i64],
    /// Freivalds input half `(rows, c)` (`uw . x_hat mod 2^K`), or empty.
    pub rhs: &'a mut [u64],
}

/// Prepare whole rows (`out.sx.len()` rows of width `n`) of the block starting at row `row0`;
/// returns the first failing row and why.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_block<W: Word, const NB: usize>(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    out: PrepOut<'_>,
    row0: usize,
    sc: &mut Scratch,
) -> Option<(usize, RowErr)> {
    {
        let w = plan.w as usize;
        if sc.q.len() < w * n {
            sc.q.resize(w * n, 0);
        }
        let mut err = None;
        let rows = out.sx.len();
        for t in 0..rows {
            let xr = &x[t * n..(t + 1) * n];
            let rr = &r[t * n..(t + 1) * n];
            let xh = if out.xhat.is_empty() { None } else { Some(&mut out.xhat[t * n..(t + 1) * n]) };
            let c = half.map_or(0, |h| h.c);
            let rh = if c == 0 { None } else { Some(&mut out.rhs[t * c..(t + 1) * c]) };
            let res = prepare_row::<W, NB>(
                xr,
                rr,
                prm,
                plan,
                half,
                sc,
                &mut out.payload[t * n * NB..(t + 1) * n * NB],
                xh,
                rh,
            );
            match res {
                Ok(s) => out.sx[t] = s,
                Err(e) => {
                    // a non-finite row outranks an earlier overflow (the reference checks finiteness
                    // of the whole tensor first); otherwise the first failing row
                    let replace = match err {
                        None => true,
                        Some((_, old)) => e == RowErr::NonFinite && old != RowErr::NonFinite,
                    };
                    if replace {
                        err = Some((row0 + t, e));
                    }
                }
            }
        }
        // the used parts only (the buffers keep the size of the largest call so far)
        wipe(&mut sc.q[..w * n]);
        let xw_used = (n * std::mem::size_of::<W>()).div_ceil(8).min(sc.xw.len());
        wipe(&mut sc.xw[..xw_used]);
        if half.is_some() {
            let ub_used = (BLOCK * std::mem::size_of::<W>()).div_ceil(8).min(sc.ubuf.len());
            wipe(&mut sc.ubuf[..ub_used]);
        }
        err
    }
}

/// One row: select the width in constant time, then mask, pack and (optionally) the input half.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_row<W: Word, const NB: usize>(
    xr: &[f32],
    rr: &[i64],
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    sc: &mut Scratch,
    payload: &mut [u8],
    xhat: Option<&mut [i64]>,
    rhs: Option<&mut [u64]>,
) -> Result<f64, RowErr> {
    let n = xr.len();
    let bits = row_amax_bits(xr);
    if bits >= INF_BITS {
        return Err(RowErr::NonFinite);
    }
    let amax = f64::from(f32::from_bits(bits));
    let tau = sumsq(xr).sqrt() / if amax > 0.0 { amax } else { 1.0 };
    let start = plan.start(tau);
    let mut w = plan.w as usize;
    let limit = (1u64 << prm.cap) as f64;
    let (mut chosen, mut s_sel, mut any) = (0usize, 0u64, false);
    for wi in 0..w {
        let b = start + wi as u32;
        let qm = qmax_of(b);
        let s = if amax > 0.0 { amax / qm } else { 1.0 };
        let qw = &mut sc.q[wi * n..(wi + 1) * n];
        let int_mask = 0u64.wrapping_sub(u64::from(b <= plan.b_int));
        let ss = quant_exact_ss(xr, s, qm, qw, plan.sum, int_mask);
        let fit = prm.cap == 0 || ss.sqrt() * prm.wnorm * NORM_SLACK <= limit;
        // larger widths come later and override (mask select, no branch)
        let mk = 0u64.wrapping_sub(u64::from(fit));
        chosen = (chosen & !(mk as usize)) | (wi & mk as usize);
        s_sel = (s_sel & !mk) | (s.to_bits() & mk);
        any |= fit;
    }
    if !any {
        if start <= 2 {
            return Err(RowErr::Overflow);
        }
        // Unreachable (`lo >= start` certainly fits): the full descending search, not constant time.
        WINDOW_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug_assert!(false, "width window missed the descending-search choice");
        let mut found = false;
        for b in (2..=prm.a).rev() {
            let qm = qmax_of(b);
            let s = if amax > 0.0 { amax / qm } else { 1.0 };
            let qw = &mut sc.q[..n];
            quant_exact(xr, s, qm, qw);
            if sum_squares(qw, qm).sqrt() * prm.wnorm * NORM_SLACK <= limit {
                s_sel = s.to_bits();
                found = true;
                break;
            }
        }
        if !found {
            return Err(RowErr::Overflow);
        }
        chosen = 0;
        w = 1;
    }

    // select q, mask, keep x_hat as words
    let kmask = ((1u64 << prm.k) - 1) as i64;
    let xw: &mut [W] = words_mut::<W>(&mut sc.xw, n);
    let q = &sc.q;
    // x_hat = (q + r) mod 2^K; written as words, packed K/8 little-endian bytes each
    let put = |j: usize, qv: i64, xw: &mut [W], payload: &mut [u8]| {
        let v = (qv.wrapping_add(rr[j]) & kmask) as u64;
        xw[j] = W::trunc(v);
        payload[j * NB..(j + 1) * NB].copy_from_slice(&v.to_le_bytes()[..NB]);
    };
    match w {
        1 => {
            for j in 0..n {
                put(j, q[j], xw, payload);
            }
        }
        2 => {
            let m1 = 0i64.wrapping_sub(chosen as i64); // all ones when the second width is chosen
            let (q0, q1) = q[..2 * n].split_at(n);
            for j in 0..n {
                put(j, (q0[j] & !m1) | (q1[j] & m1), xw, payload);
            }
        }
        _ => {
            for j in 0..n {
                let mut qv = 0i64;
                for wi in 0..w {
                    let m = 0i64.wrapping_sub(i64::from(wi == chosen));
                    qv |= q[wi * n + j] & m;
                }
                put(j, qv, xw, payload);
            }
        }
    }
    if let Some(xh) = xhat {
        for (o, v) in xh.iter_mut().zip(xw.iter()) {
            *o = v.widen() as i64; // < 2^K
        }
    }
    // Freivalds input half: rhs_i = uw_i . x_hat (mod 2^K)
    if let (Some(h), Some(rh)) = (half, rhs) {
        let c = h.c;
        if h.uw.words::<W>(0, 0).is_some() {
            for (i, o) in rh.iter_mut().enumerate() {
                let us = h.uw.words::<W>(i * n, (i + 1) * n).unwrap_or(&[]);
                *o = dot(xw, us).widen() & (kmask as u64);
            }
        } else {
            let ub: &mut [W] = words_mut::<W>(&mut sc.ubuf, BLOCK);
            let mut acc = vec![W::default(); c];
            let mut b0 = 0;
            while b0 < n {
                let b1 = (b0 + BLOCK).min(n);
                for (i, a) in acc.iter_mut().enumerate() {
                    h.uw.fill::<W, NB>(i * n + b0, i * n + b1, &mut ub[..b1 - b0]);
                    *a = a.wadd(dot(&xw[b0..b1], &ub[..b1 - b0]));
                }
                b0 = b1;
            }
            for (o, a) in rh.iter_mut().zip(&acc) {
                *o = a.widen() & (kmask as u64);
            }
            wipe(&mut acc);
        }
    }
    Ok(f64::from_bits(s_sel))
}

/// [`prepare_block`] compiled for AVX2 (same IEEE results: no FMA contraction in Rust, correctly
/// rounded divisions either way).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn prepare_block_avx2<W: Word, const NB: usize>(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    out: PrepOut<'_>,
    row0: usize,
    sc: &mut Scratch,
) -> Option<(usize, RowErr)> {
    prepare_block::<W, NB>(x, r, n, prm, plan, half, out, row0, sc)
}

#[allow(clippy::too_many_arguments)]
fn prepare_block_dispatch<W: Word, const NB: usize>(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    out: PrepOut<'_>,
    row0: usize,
) -> Option<(usize, RowErr)> {
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let sc = &mut *guard;
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked above).
            return unsafe { prepare_block_avx2::<W, NB>(x, r, n, prm, plan, half, out, row0, sc) };
        }
        prepare_block::<W, NB>(x, r, n, prm, plan, half, out, row0, sc)
    })
}

/// Rows of at least this many elements in total go to the pool in blocks (prefill).
const PAR_MIN_ELEMS: usize = MANY_PAR_MIN_ELEMS;
/// Elements per pool task (whole rows).
const TASK_ELEMS: usize = 16 * 1024;

/// Fused constant-time quantize + mask + pack (+ Freivalds input half) into caller buffers.
///
/// `x` and `r` are `rows x n` (row-major), `out.sx.len() == rows`, `out.payload` is
/// `rows * n * K/8` bytes, `out.xhat` is `rows * n` or empty, `out.rhs` is `rows * c` (with `half`)
/// or empty. Bit-identical to [`crate::prepare::quantize_mask_pack_into`] (payload, `sx`, `x_hat`)
/// and to the descending width search, including which error is returned. `threads`: `1` = the
/// calling thread; otherwise prefill-sized calls (at least [`MANY_PAR_MIN_ELEMS`] elements, two or
/// more rows) are split into whole-row blocks on the persistent pool. On error every output
/// (`payload`, `sx`, `xhat`, `rhs`) is zeroed.
pub fn prepare_into(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    half: Option<InputHalf<'_>>,
    out: PrepOut<'_>,
    threads: usize,
) -> Result<(), PrepareError> {
    prm.validate()?;
    let nb = (prm.k / 8) as usize;
    let rows = out.sx.len();
    let c = half.map_or(0, |h| h.c);
    if x.len() != rows * n
        || r.len() != rows * n
        || out.payload.len() != rows * n * nb
        || !(out.xhat.is_empty() || out.xhat.len() == rows * n)
        || out.rhs.len() != rows * c
    {
        return Err(PrepareError::Invalid("buffer sizes do not match the shape".into()));
    }
    if let Some(h) = half {
        if h.c == 0 || h.uw.elems(nb) != h.c * n || matches!(h.uw, Src::Bytes(_)) {
            return Err(PrepareError::Invalid("uw must be a (c, n) word array with c >= 1".into()));
        }
    }
    if rows == 0 {
        return Ok(());
    }
    if n == 0 {
        return Err(PrepareError::Invalid("activations must have at least one feature".into()));
    }
    let plan = plan_for(n, prm);
    match prm.k {
        8 => prepare_rows::<u32, 1>(x, r, n, prm, &plan, half, out, threads),
        16 => prepare_rows::<u32, 2>(x, r, n, prm, &plan, half, out, threads),
        24 => prepare_rows::<u32, 3>(x, r, n, prm, &plan, half, out, threads),
        32 => prepare_rows::<u32, 4>(x, r, n, prm, &plan, half, out, threads),
        40 => prepare_rows::<u64, 5>(x, r, n, prm, &plan, half, out, threads),
        48 => prepare_rows::<u64, 6>(x, r, n, prm, &plan, half, out, threads),
        _ => prepare_rows::<u64, 7>(x, r, n, prm, &plan, half, out, threads),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_rows<W: Word, const NB: usize>(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    out: PrepOut<'_>,
    threads: usize,
) -> Result<(), PrepareError> {
    let rows = out.sx.len();
    let c = half.map_or(0, |h| h.c);
    let pool = pool::global();
    let max_t = match threads {
        0 => pool.workers() + 1,
        t => t,
    };
    let parallel = max_t > 1 && rows > 1 && rows * n >= PAR_MIN_ELEMS && pool.workers() > 0;
    let per = if parallel { TASK_ELEMS.div_ceil(n).clamp(1, rows).max(rows.div_ceil(max_t * 4)) } else { rows };
    let PrepOut { payload, sx, xhat, rhs } = out;
    let err = if !parallel {
        let mut o = PrepOut { payload, sx, xhat, rhs };
        let po = PrepOut { payload: &mut *o.payload, sx: &mut *o.sx, xhat: &mut *o.xhat, rhs: &mut *o.rhs };
        let e = prepare_block_dispatch::<W, NB>(x, r, n, prm, plan, half, po, 0);
        if e.is_some() {
            zero_prep(&mut o);
        }
        e
    } else {
        // split the outputs into whole-row tasks
        let mut tasks: Vec<Mutex<(usize, PrepOut<'_>)>> = Vec::new();
        let (mut pl, mut sr, mut xr, mut rr) = (payload, sx, xhat, rhs);
        let mut row = 0;
        while row < rows {
            let take = per.min(rows - row);
            let (p0, p1) = std::mem::take(&mut pl).split_at_mut(take * n * NB);
            let (s0, s1) = std::mem::take(&mut sr).split_at_mut(take);
            let (x0, x1) =
                if xr.is_empty() { (&mut [][..], &mut [][..]) } else { std::mem::take(&mut xr).split_at_mut(take * n) };
            let (r0, r1) = std::mem::take(&mut rr).split_at_mut(take * c);
            tasks.push(Mutex::new((row, PrepOut { payload: p0, sx: s0, xhat: x0, rhs: r0 })));
            pl = p1;
            sr = s1;
            xr = x1;
            rr = r1;
            row += take;
        }
        let errs: Vec<Mutex<Option<(usize, RowErr)>>> = (0..tasks.len()).map(|_| Mutex::new(None)).collect();
        let ok = pool.run(tasks.len(), &|i| {
            let mut g = tasks[i].lock().unwrap_or_else(|e| e.into_inner());
            let (row0, ref mut o) = *g;
            let take = o.sx.len();
            let po = PrepOut { payload: &mut *o.payload, sx: &mut *o.sx, xhat: &mut *o.xhat, rhs: &mut *o.rhs };
            let e = prepare_block_dispatch::<W, NB>(
                &x[row0 * n..(row0 + take) * n],
                &r[row0 * n..(row0 + take) * n],
                n,
                prm,
                plan,
                half,
                po,
                row0,
            );
            *errs[i].lock().unwrap_or_else(|e| e.into_inner()) = e;
        });
        let mut first: Option<(usize, RowErr)> = None;
        for e in errs.iter().filter_map(|e| *e.lock().unwrap_or_else(|e| e.into_inner())) {
            first = match first {
                None => Some(e),
                Some(old) => {
                    let better =
                        (e.1 == RowErr::NonFinite && old.1 != RowErr::NonFinite) || (e.1 == old.1 && e.0 < old.0);
                    Some(if better { e } else { old })
                }
            };
        }
        if !ok {
            for t in &tasks {
                let mut g = t.lock().unwrap_or_else(|e| e.into_inner());
                zero_prep(&mut g.1);
            }
            return Err(PrepareError::Invalid("internal error in a prepare task".into()));
        }
        if first.is_some() {
            for t in &tasks {
                let mut g = t.lock().unwrap_or_else(|e| e.into_inner());
                zero_prep(&mut g.1);
            }
        }
        first
    };
    match err {
        None => Ok(()),
        Some((_, RowErr::NonFinite)) => {
            Err(PrepareError::Overflow("non-finite activation values cannot be quantized".into()))
        }
        Some((_, RowErr::Overflow)) => Err(PrepareError::Overflow(format!(
            "K{}A{}C{}: activations cannot be bounded by 2^{} for this linear",
            prm.k, prm.a, prm.cap, prm.cap
        ))),
    }
}

fn zero_prep(o: &mut PrepOut<'_>) {
    wipe(o.payload);
    wipe(o.sx);
    wipe(o.xhat);
    wipe(o.rhs);
}

// ---- finish --------------------------------------------------------------------------------------

/// The Freivalds mode of [`finish_into`].
#[derive(Clone, Copy)]
pub enum CheckMode<'a> {
    /// No check.
    Off,
    /// Both halves here: `u (c, m)`, `uw (c, n)` and the masked inputs `x_hat (rows, n)`.
    Full {
        /// `(c, m)`.
        u: Src<'a>,
        /// `(c, n)`.
        uw: Src<'a>,
        /// `(rows, n)`: words or the request's wire bytes.
        x_hat: Src<'a>,
        /// Check vectors.
        c: usize,
        /// Input features.
        n: usize,
    },
    /// Output half here, input half precomputed by [`prepare_into`]: `u (c, m)`, `rhs (rows, c)`.
    Split {
        /// `(c, m)`.
        u: Src<'a>,
        /// `(rows, c)`, `uw . x_hat mod 2^K`.
        rhs: &'a [u64],
        /// Check vectors.
        c: usize,
    },
}

/// Inputs of [`finish_into`] (one chunk of `rows` tokens of a linear with `m` outputs).
pub struct FinishArgs<'a> {
    /// Wire bytes `rows * m * K/8`.
    pub y_hat: &'a [u8],
    /// `W r (rows, m)` (zeroed in place when the buffer says so).
    pub wr: WrBuf,
    /// Scales `(rows,)`.
    pub sx: &'a [f64],
    /// `(m,)`.
    pub scale: &'a [f32],
    /// `(m,)`.
    pub bias: Option<&'a [f32]>,
    /// Ring width.
    pub k: u32,
    /// Inclusive output bound.
    pub bound: u64,
    /// Freivalds mode.
    pub check: CheckMode<'a>,
}

/// Fused finish into `out` (`rows x m` float32): bit-identical outputs and statuses to
/// [`crate::finish::finish_group_run`] on the same chunk (and, in [`CheckMode::Split`], to the full
/// check with the input half computed at prepare time). Returns `(status, bad_rows)`; `out` is
/// zeroed unless the status is OK (also for [`STATUS_U_ZERO`], so a reused buffer never keeps the
/// previous trip's outputs). `threads` as in [`prepare_into`]. Shapes must have been validated
/// (panics otherwise; a failed pool task also panics, after zeroing `out`).
pub fn finish_into(f: &FinishArgs<'_>, out: &mut [f32], threads: usize) -> (u8, usize) {
    let rows = f.sx.len();
    let m = f.scale.len();
    assert_eq!(out.len(), rows * m, "out has the wrong size");
    let (x_hat, checks, rhs_pre) = match f.check {
        CheckMode::Off => (None, None, None),
        CheckMode::Full { u, uw, x_hat, c, n } => (Some(x_hat), Some(GroupChecks { u, uw, c, n }), None),
        CheckMode::Split { u, rhs, c } => {
            assert_eq!(rhs.len(), rows * c, "rhs has the wrong size");
            (None, Some(GroupChecks { u, uw: Src::U64(&[]), c, n: 0 }), Some(rhs))
        }
    };
    let g = Group {
        parts: vec![Part { y_hat: f.y_hat, wr: f.wr, x_hat, sx: f.sx }],
        checks,
        scale: f.scale,
        bias: f.bias,
        k: f.k,
        bound: f.bound,
        m,
    };
    let exec = Exec::Pool(threads);
    // SAFETY: `out` is an exclusive slice of exactly `rows * m` floats.
    let res = unsafe { finish::finish_group_into_raw(&g, out.as_mut_ptr(), exec, rhs_pre) };
    if res.0 == STATUS_U_ZERO {
        wipe(out); // the core writes nothing then: fail closed on a reused buffer
    }
    res
}

/// The Freivalds output half: `(u . y_hat_t) mod 2^K == rhs_t` for every row `t` and check vector.
/// Returns `(status, bad_rows)` with the statuses of `finish_group` ([`STATUS_U_ZERO`] first).
pub fn freivalds_output(y_hat: &[u8], u: Src<'_>, rhs: &[u64], c: usize, m: usize, k: u32) -> (u8, usize) {
    match k {
        8 => fo_run::<u32, 1>(y_hat, u, rhs, c, m, k),
        16 => fo_run::<u32, 2>(y_hat, u, rhs, c, m, k),
        24 => fo_run::<u32, 3>(y_hat, u, rhs, c, m, k),
        32 => fo_run::<u32, 4>(y_hat, u, rhs, c, m, k),
        40 => fo_run::<u64, 5>(y_hat, u, rhs, c, m, k),
        48 => fo_run::<u64, 6>(y_hat, u, rhs, c, m, k),
        56 => fo_run::<u64, 7>(y_hat, u, rhs, c, m, k),
        k => panic!("unsupported ring width {k}"),
    }
}

/// [`fo_body`] compiled for AVX2 when the CPU has it (wrapping integer arithmetic: identical
/// results on every ISA; the u32 dot products of K <= 32 become `vpmulld`/`vpaddd`).
fn fo_run<W: Word, const NB: usize>(y_hat: &[u8], u: Src<'_>, rhs: &[u64], c: usize, m: usize, k: u32) -> (u8, usize) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked above).
        return unsafe { fo_body_avx2::<W, NB>(y_hat, u, rhs, c, m, k) };
    }
    fo_body::<W, NB>(y_hat, u, rhs, c, m, k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fo_body_avx2<W: Word, const NB: usize>(
    y_hat: &[u8],
    u: Src<'_>,
    rhs: &[u64],
    c: usize,
    m: usize,
    k: u32,
) -> (u8, usize) {
    fo_body::<W, NB>(y_hat, u, rhs, c, m, k)
}

#[inline(always)]
fn fo_body<W: Word, const NB: usize>(y_hat: &[u8], u: Src<'_>, rhs: &[u64], c: usize, m: usize, k: u32) -> (u8, usize) {
    if c == 0 {
        return (STATUS_OK, 0);
    }
    let rows = rhs.len() / c;
    assert!(rhs.len() == rows * c && y_hat.len() == rows * m * NB && u.elems(NB) == c * m, "bad shapes");
    if u.all_zero(k) {
        return (STATUS_U_ZERO, 0);
    }
    let mut lhs = vec![W::default(); rows * c];
    let mut ybuf = vec![W::default(); BLOCK];
    let mut ubuf = vec![W::default(); BLOCK * c];
    let direct = u.words::<W>(0, 0).is_some();
    let mut b0 = 0;
    while b0 < m {
        let b1 = (b0 + BLOCK).min(m);
        let len = b1 - b0;
        if !direct {
            for i in 0..c {
                u.fill::<W, NB>(i * m + b0, i * m + b1, &mut ubuf[i * BLOCK..i * BLOCK + len]);
            }
        }
        for t in 0..rows {
            let yb = &mut ybuf[..len];
            decode::<W, NB>(&y_hat[(t * m + b0) * NB..(t * m + b1) * NB], yb);
            for i in 0..c {
                let us = if direct {
                    u.words::<W>(i * m + b0, i * m + b1).unwrap_or(&[])
                } else {
                    &ubuf[i * BLOCK..i * BLOCK + len]
                };
                lhs[t * c + i] = lhs[t * c + i].wadd(dot(yb, us));
            }
        }
        b0 = b1;
    }
    let mask = W::mask(k);
    let mut bad = 0usize;
    for t in 0..rows {
        let row_bad =
            (0..c).fold(false, |acc, i| acc | (lhs[t * c + i].and(mask) != W::trunc(rhs[t * c + i]).and(mask)));
        bad += usize::from(row_bad);
    }
    wipe(&mut lhs);
    wipe(&mut ybuf);
    wipe(&mut ubuf);
    if bad > 0 {
        (STATUS_FREIVALDS, bad)
    } else {
        (STATUS_OK, 0)
    }
}

// ---- helper thread for the deferred output half --------------------------------------------------

type Job = Box<dyn FnOnce() + Send + 'static>;

/// The helper thread's queue (one persistent thread, jobs in submission order).
fn helper() -> &'static Mutex<std::sync::mpsc::Sender<Job>> {
    static TX: OnceLock<Mutex<std::sync::mpsc::Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("pgw-trip-check".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("cannot start the trip check thread");
        Mutex::new(tx)
    })
}

struct CheckState {
    res: Mutex<Option<(u8, usize)>>,
    cv: Condvar,
}

impl CheckState {
    fn set(&self, v: (u8, usize)) {
        *self.res.lock().unwrap_or_else(|e| e.into_inner()) = Some(v);
        self.cv.notify_all();
    }
    fn wait(&self) -> (u8, usize) {
        let mut g = self.res.lock().unwrap_or_else(|e| e.into_inner());
        while g.is_none() {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.unwrap_or((STATUS_INTERNAL, 0))
    }
}

/// Raw view of a buffer kept alive by the [`TripCheck`] that owns the job.
#[derive(Clone, Copy)]
enum RawSrc {
    U64(*const u64, usize),
    U32(*const u32, usize),
}

/// The pointers of one check job (alive until the job has run, see [`TripCheck`]).
struct RawJob {
    y: (*const u8, usize),
    u: RawSrc,
}
// SAFETY: the pointees are kept alive by the TripCheck (which waits for the job in `Drop`) and are
// only read by the job.
unsafe impl Send for RawJob {}

/// Read-only borrows of a pending job's `y_hat` and `u` (rust-numpy borrow guards, which also hold
/// strong references): while they are held, a read-write borrow of either array by `pgw_fast` (or
/// any rust-numpy extension, e.g. `trip_prepare_into` writing a reused buffer) fails with
/// `ValueError` instead of racing with the job. Plain numpy writes are not tracked by rust-numpy;
/// the caller contract (unchanged until `wait()`) still applies to those.
struct Held {
    _y: ByteIn<'static>,
    _u: RoRing<'static>,
}

// SAFETY: the guards are never used, only dropped, and every drop happens with the GIL held: in
// `TripCheck::wait` / `done` (which hold the GIL) or in `Drop`, which pyo3 runs from the type's
// `tp_dealloc` with the GIL held. Releasing a rust-numpy borrow needs nothing but the GIL.
unsafe impl Send for Held {}

/// A Freivalds output half running on the native helper thread (`trip_freivalds_submit`).
///
/// The `y_hat` buffer and `u` must stay unchanged until :meth:`wait` returns. The handle holds
/// read-only rust-numpy borrows of both until then (so a read-write borrow through `pgw_fast` raises
/// `ValueError`), and dropping the handle waits for the job (with the GIL held: call `wait()`).
#[pyclass(module = "pgw_fast")]
pub struct TripCheck {
    state: Arc<CheckState>,
    held: Mutex<Option<Held>>,
}

impl TripCheck {
    /// Drop the borrows (GIL held by the caller; only once the job has run).
    fn release(&self) {
        let h = self.held.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(h);
    }
}

#[pymethods]
impl TripCheck {
    /// Block (GIL released) until the check ran; returns ``(status, bad_rows)`` as
    /// ``trip_freivalds_output`` (status 4: the helper failed internally). Releases the borrows of
    /// ``y_hat`` and ``u``.
    fn wait(&self, py: Python<'_>) -> (u8, usize) {
        let st = Arc::clone(&self.state);
        let res = py.allow_threads(move || st.wait());
        self.release();
        res
    }

    /// True once the result is available (the borrows are released then).
    fn done(&self, _py: Python<'_>) -> bool {
        let d = self.state.res.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        if d {
            self.release();
        }
        d
    }
}

impl Drop for TripCheck {
    fn drop(&mut self) {
        // the job reads the buffers that `held` keeps alive: never release them before it ran
        let _ = self.state.wait();
        self.release();
    }
}

// ---- Python bindings -----------------------------------------------------------------------------

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// The elements of a C-contiguous array (`None` for any other layout, Fortran order included).
fn c_ro<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
) -> Option<&'a [T]> {
    if a.is_c_contiguous() {
        a.as_slice().ok()
    } else {
        None
    }
}

/// [`c_ro`] for a writable array.
fn c_rw<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a mut numpy::PyReadwriteArray<'_, T, D>,
) -> Option<&'a mut [T]> {
    if a.is_c_contiguous() {
        a.as_slice_mut().ok()
    } else {
        None
    }
}

/// A C-contiguous writable numpy array as a mutable slice (borrow guard returned alongside).
macro_rules! rw_slice {
    ($obj:expr, $ty:ty, $name:expr) => {{
        let arr = $obj
            .downcast::<PyArrayDyn<$ty>>()
            .map_err(|_| verr(format!("{} must be a {} numpy array", $name, stringify!($ty))))?;
        arr.try_readwrite().map_err(|e| verr(format!("{}: {e} (must be writable and not borrowed)", $name)))?
    }};
}

/// Bytes-like input: `bytes` or a 1-D C-contiguous uint8 numpy array.
enum ByteIn<'py> {
    B(Bound<'py, PyBytes>),
    A(numpy::PyReadonlyArrayDyn<'py, u8>),
}

impl<'py> ByteIn<'py> {
    fn extract(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Self> {
        if let Ok(b) = obj.downcast::<PyBytes>() {
            return Ok(ByteIn::B(b.clone()));
        }
        if let Ok(a) = obj.downcast::<PyArrayDyn<u8>>() {
            return Ok(ByteIn::A(a.try_readonly().map_err(|e| verr(format!("{name}: {e}")))?));
        }
        Err(verr(format!("{name} must be bytes or a uint8 numpy array")))
    }
    fn slice(&self, name: &str) -> PyResult<&[u8]> {
        match self {
            ByteIn::B(b) => Ok(b.as_bytes()),
            ByteIn::A(a) => c_ro(a).ok_or_else(|| verr(format!("{name} must be C-contiguous"))),
        }
    }
}

/// `trip_width_window(n, a, cap, wnorm) -> int`: widths each row evaluates (public; 1 uncapped).
#[pyfunction]
fn trip_width_window(n: usize, a: u32, cap: u32, wnorm: f64) -> PyResult<u32> {
    let prm = PrepareParams { k: 56, a, cap, wnorm };
    prm.validate().map_err(PrepareError::into_py)?;
    Ok(plan_for(n, prm).window())
}

/// Constant-time quantize + mask + pack into caller-owned buffers (the TripEngine's prepare).
///
/// Args:
///     x: float32 ``(rows, n)``, C-contiguous; r: int64 ``(rows, n)`` one-time masks in ``[0, 2^K)``.
///     k, a, cap, wnorm: ring and linear parameters as in ``quantize_mask_pack``.
///     payload: writable uint8 array of ``rows * n * K/8`` bytes (the wire item, reused per trip).
///     sx: writable float64 ``(rows,)``.
///     xhat: optional writable int64 ``(rows, n)``.
///     uw, rhs: optional Freivalds input half: ``uw (c, n)`` (uint64/int64, or uint32 for K <= 32)
///         and a writable uint64 ``rhs (rows, c)`` that receives ``uw . x_hat mod 2^K`` per row
///         (secret: zero it after the check).
///     threads: 1 = calling thread; 0 = the persistent pool for prefill-sized calls.
///
/// Bit-identical to ``quantize_mask_pack`` (same payload, ``sx`` and ``x_hat``; same
/// ``ValueError("ring_overflow: ...")``), but every row evaluates the same public number of widths.
/// On error every output buffer is zeroed. The GIL is released while computing.
#[pyfunction]
#[pyo3(signature = (x, r, k, a, cap, wnorm, payload, sx, xhat = None, uw = None, rhs = None, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn trip_prepare_into<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
    k: u32,
    a: u32,
    cap: u32,
    wnorm: f64,
    payload: &Bound<'py, PyAny>,
    sx: &Bound<'py, PyAny>,
    xhat: Option<&Bound<'py, PyAny>>,
    uw: Option<&Bound<'py, PyAny>>,
    rhs: Option<&Bound<'py, PyAny>>,
    threads: usize,
) -> PyResult<()> {
    let prm = PrepareParams { k, a, cap, wnorm };
    prm.validate().map_err(PrepareError::into_py)?;
    let nb = (k / 8) as usize;
    let xa = x
        .downcast::<PyArray2<f32>>()
        .map_err(|_| verr("x must be a 2-D float32 numpy array"))?
        .try_readonly()
        .map_err(|e| verr(format!("x: {e}")))?;
    let ra = r
        .downcast::<PyArray2<i64>>()
        .map_err(|_| verr("r must be a 2-D int64 numpy array"))?
        .try_readonly()
        .map_err(|e| verr(format!("r: {e}")))?;
    if xa.shape() != ra.shape() {
        return Err(verr(format!("activation shape {:?} != mask shape {:?}", xa.shape(), ra.shape())));
    }
    let (rows, n) = (xa.shape()[0], xa.shape()[1]);
    let xs = c_ro(&xa).ok_or_else(|| verr("x must be C-contiguous"))?;
    let rs = c_ro(&ra).ok_or_else(|| verr("r must be C-contiguous"))?;
    let mut pl = rw_slice!(payload, u8, "payload");
    let mut sxa = rw_slice!(sx, f64, "sx");
    let mut xha = match xhat {
        Some(o) => Some(rw_slice!(o, i64, "xhat")),
        None => None,
    };
    let (uwa, mut rha) = match (uw, rhs) {
        (None, None) => (None, None),
        (Some(u), Some(h)) => (Some(RoRing::extract(u, "uw", k)?), Some(rw_slice!(h, u64, "rhs"))),
        _ => return Err(verr("uw and rhs must both be given or both be None")),
    };
    let half = match &uwa {
        Some(ua) => {
            let (src, sh) = ua.src("uw")?;
            let (c, un) = sh.ok_or_else(|| verr("uw must be a 2-D array"))?;
            if un != n || c == 0 {
                return Err(verr(format!("uw shape ({c}, {un}) does not match n={n} (c >= 1)")));
            }
            Some(InputHalf { uw: src, c })
        }
        None => None,
    };
    let c = half.map_or(0, |h| h.c);
    let pls = c_rw(&mut pl).ok_or_else(|| verr("payload must be C-contiguous"))?;
    if pls.len() != rows * n * nb {
        return Err(verr(format!("payload has {} bytes, expected {}", pls.len(), rows * n * nb)));
    }
    let sxs = c_rw(&mut sxa).ok_or_else(|| verr("sx must be C-contiguous"))?;
    if sxs.len() != rows {
        return Err(verr(format!("sx has {} entries, expected {rows}", sxs.len())));
    }
    let xhs: &mut [i64] = match xha.as_mut() {
        Some(a) => {
            let s = c_rw(a).ok_or_else(|| verr("xhat must be C-contiguous"))?;
            if s.len() != rows * n {
                return Err(verr(format!("xhat has {} entries, expected {}", s.len(), rows * n)));
            }
            s
        }
        None => &mut [],
    };
    let rhs_s: &mut [u64] = match rha.as_mut() {
        Some(a) => {
            let s = c_rw(a).ok_or_else(|| verr("rhs must be C-contiguous"))?;
            if s.len() != rows * c {
                return Err(verr(format!("rhs has {} entries, expected {}", s.len(), rows * c)));
            }
            s
        }
        None => &mut [],
    };
    let out = PrepOut { payload: pls, sx: sxs, xhat: xhs, rhs: rhs_s };
    py.allow_threads(|| prepare_into(xs, rs, n, prm, half, out, threads)).map_err(PrepareError::into_py)
}

/// Fused finish of one chunk into a caller-owned output (the TripEngine's finish).
///
/// Args:
///     y_hat: service result (``bytes`` or uint8 array, ``rows * m * K/8`` bytes).
///     wr: ``W r`` 2-D int64/uint64/uint32 ``(rows, m)``; zeroed in place with ``wipe_wr``.
///     sx: float64 ``(rows,)``; scale: float32 ``(m,)``; bias: float32 ``(m,)`` or None.
///     k, bound: ring width and inclusive output bound.
///     out: writable float32 ``(rows, m)`` (reused per trip).
///     x_hat, u, uw: full Freivalds check (as ``finish_group``: ``x_hat`` as wire bytes or array).
///     u, rhs: split check: the output half here against ``rhs (rows, c)`` from
///         ``trip_prepare_into``. Give neither ``u`` nor ``uw``/``rhs`` for no check.
///     threads: 1 = calling thread; 0 = the persistent pool when the chunk is large (prefill).
///
/// Returns ``(status, bad_rows)`` with the statuses of ``finish_group``; ``out`` holds the outputs
/// only for status 0 and is zeroed for every other status. An internal failure of a pool worker
/// raises ``RuntimeError`` (an ``Exception``, not pyo3's ``PanicException``) with ``out`` zeroed
/// and ``wr`` wiped.
#[pyfunction]
#[pyo3(signature = (y_hat, wr, sx, scale, bias, k, bound, out, x_hat = None, u = None, uw = None, rhs = None, wipe_wr = true, threads = 1))]
#[allow(clippy::too_many_arguments)]
fn trip_finish_into<'py>(
    py: Python<'py>,
    y_hat: &Bound<'py, PyAny>,
    wr: &Bound<'py, PyAny>,
    sx: &Bound<'py, PyAny>,
    scale: &Bound<'py, PyAny>,
    bias: Option<&Bound<'py, PyAny>>,
    k: u32,
    bound: u64,
    out: &Bound<'py, PyAny>,
    x_hat: Option<&Bound<'py, PyAny>>,
    u: Option<&Bound<'py, PyAny>>,
    uw: Option<&Bound<'py, PyAny>>,
    rhs: Option<&Bound<'py, PyAny>>,
    wipe_wr: bool,
    threads: usize,
) -> PyResult<(u8, usize)> {
    if k % 8 != 0 || !(8..=56).contains(&k) {
        return Err(verr(format!("ring width k={k} must be a multiple of 8 in [8, 56]")));
    }
    let nb = (k / 8) as usize;
    let yb = ByteIn::extract(y_hat, "y_hat")?;
    let ys = yb.slice("y_hat")?;
    let sx_a = finish::arr1::<f64>(sx, "sx", "float64")?;
    let sx_s = finish::slice1(&sx_a, "sx")?;
    let rows = sx_s.len();
    let sc_a = finish::arr1::<f32>(scale, "scale", "float32")?;
    let sc_s = finish::slice1(&sc_a, "scale")?;
    let m = sc_s.len();
    let bias_a = bias.map(|b| finish::arr1::<f32>(b, "bias", "float32")).transpose()?;
    let bias_s = bias_a.as_ref().map(|b| finish::slice1(b, "bias")).transpose()?;
    if bias_s.is_some_and(|b| b.len() != m) {
        return Err(verr(format!("bias has the wrong size, expected {m}")));
    }
    if ys.len() != rows * m * nb {
        return Err(verr(format!(
            "ring payload has {} bytes, expected {} for shape ({rows}, {m}) K={k}",
            ys.len(),
            rows * m * nb
        )));
    }
    let ua = u.map(|u| RoRing::extract(u, "u", k)).transpose()?;
    let uwa = uw.map(|v| RoRing::extract(v, "uw", k)).transpose()?;
    let xa = x_hat.map(|v| RoRing::extract(v, "x_hat", k)).transpose()?;
    let rha = rhs
        .map(|h| {
            h.downcast::<PyArray2<u64>>()
                .map_err(|_| verr("rhs must be a 2-D uint64 numpy array"))?
                .try_readonly()
                .map_err(|e| verr(format!("rhs: {e}")))
        })
        .transpose()?;
    let check = match (&ua, &uwa, &rha) {
        (None, None, None) => CheckMode::Off,
        (Some(ua), Some(uwa), None) => {
            let (us, ush) = ua.src("u")?;
            let (uws, uwsh) = uwa.src("uw")?;
            let ((c, um), (c2, n)) = match (ush, uwsh) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(verr("u and uw must be 2-D arrays")),
            };
            if um != m || c2 != c || c == 0 {
                return Err(verr(format!("check shapes do not match: u ({c}, {um}), uw ({c2}, {n}), m={m}")));
            }
            let xa = xa.as_ref().ok_or_else(|| verr("x_hat is required with check vectors"))?;
            let (xs, xsh) = xa.src("x_hat")?;
            if let Some((xr, xn)) = xsh {
                if xr != rows || xn != n {
                    return Err(verr(format!("x_hat shape ({xr}, {xn}) != ({rows}, {n})")));
                }
            }
            if xs.elems(nb) != rows * n || matches!(xs, Src::Bytes(b) if b.len() != rows * n * nb) {
                return Err(verr(format!("x_hat has the wrong size for ({rows}, {n})")));
            }
            CheckMode::Full { u: us, uw: uws, x_hat: xs, c, n }
        }
        (Some(ua), None, Some(rh)) => {
            let (us, ush) = ua.src("u")?;
            let (c, um) = ush.ok_or_else(|| verr("u must be a 2-D array"))?;
            let rsh = rh.shape();
            if um != m || c == 0 || rsh[0] != rows || rsh[1] != c {
                return Err(verr(format!(
                    "check shapes do not match: u ({c}, {um}), rhs {:?}, rows={rows} m={m}",
                    rsh
                )));
            }
            let rs = c_ro(rh).ok_or_else(|| verr("rhs must be C-contiguous"))?;
            CheckMode::Split { u: us, rhs: rs, c }
        }
        _ => return Err(verr("give u with uw (and x_hat), or u with rhs, or no check arguments")),
    };
    let mut wra = finish::WrArg::extract(wr, k, wipe_wr)?;
    let (wrb, (wrows, wm)) = wra.buf(wipe_wr)?;
    if wrows != rows || wm != m {
        return Err(verr(format!("wr shape ({wrows}, {wm}) != ({rows}, {m})")));
    }
    let mut oa = out
        .downcast::<PyArray2<f32>>()
        .map_err(|_| verr("out must be a 2-D float32 numpy array"))?
        .try_readwrite()
        .map_err(|e| verr(format!("out: {e} (must be writable and not borrowed)")))?;
    if oa.shape() != [rows, m] {
        return Err(verr(format!("out shape {:?} != ({rows}, {m})", oa.shape())));
    }
    let os = c_rw(&mut oa).ok_or_else(|| verr("out must be C-contiguous"))?;
    let args = FinishArgs { y_hat: ys, wr: wrb, sx: sx_s, scale: sc_s, bias: bias_s, k, bound, check };
    let res = py.allow_threads(|| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finish_into(&args, os, threads)));
        if r.is_err() {
            // fail closed: no output and no `W r` survive a failed worker
            wipe(os);
            args.wr.wipe_all();
        }
        r
    });
    let _ = args; // borrows of `wra` end here
    drop(wra);
    res.map_err(|_| PyRuntimeError::new_err("trip_finish_into: internal error in a finish worker (outputs zeroed)"))
}

/// Validated `(y_hat, u, rhs)` of the output half: `(rows, c, m)`.
fn fo_shapes(
    ys: &[u8],
    u: &RoRing<'_>,
    rhs: &numpy::PyReadonlyArray2<'_, u64>,
    k: u32,
) -> PyResult<(usize, usize, usize)> {
    let (_, ush) = u.src("u")?;
    let (c, m) = ush.ok_or_else(|| verr("u must be a 2-D array"))?;
    let rsh = rhs.shape();
    let rows = rsh[0];
    if c == 0 || rsh[1] != c {
        return Err(verr(format!("rhs shape {:?} does not match u ({c}, {m})", rsh)));
    }
    let nb = (k / 8) as usize;
    if ys.len() != rows * m * nb {
        return Err(verr(format!("y_hat has {} bytes, expected {} for ({rows}, {m}) K={k}", ys.len(), rows * m * nb)));
    }
    Ok((rows, c, m))
}

fn check_k(k: u32) -> PyResult<()> {
    if k % 8 != 0 || !(8..=56).contains(&k) {
        return Err(verr(format!("ring width k={k} must be a multiple of 8 in [8, 56]")));
    }
    Ok(())
}

/// ``trip_freivalds_output(y_hat, u, rhs, k) -> (status, bad_rows)``: the Freivalds output half
/// ``u . y_hat`` against ``rhs (rows, c)`` from ``trip_prepare_into`` (GIL released). Statuses as
/// ``finish_group``: 0 ok, 1 failed rows, 3 ``u`` all zero.
#[pyfunction]
fn trip_freivalds_output<'py>(
    py: Python<'py>,
    y_hat: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    k: u32,
) -> PyResult<(u8, usize)> {
    check_k(k)?;
    let yb = ByteIn::extract(y_hat, "y_hat")?;
    let ys = yb.slice("y_hat")?;
    let ua = RoRing::extract(u, "u", k)?;
    let ra = rhs
        .downcast::<PyArray2<u64>>()
        .map_err(|_| verr("rhs must be a 2-D uint64 numpy array"))?
        .try_readonly()
        .map_err(|e| verr(format!("rhs: {e}")))?;
    let (_, c, m) = fo_shapes(ys, &ua, &ra, k)?;
    let (us, _) = ua.src("u")?;
    let rs = c_ro(&ra).ok_or_else(|| verr("rhs must be C-contiguous"))?;
    Ok(py.allow_threads(|| freivalds_output(ys, us, rs, c, m, k)))
}

/// ``trip_freivalds_submit(y_hat, u, rhs, k) -> TripCheck``: run the output half on the native
/// helper thread and return at once. ``y_hat`` and ``u`` must not change until ``wait()`` returns:
/// the handle keeps both alive and read-borrowed, so a ``pgw_fast`` call that would write either
/// raises ``ValueError`` meanwhile (``rhs`` is copied).
#[pyfunction]
pub(crate) fn trip_freivalds_submit<'py>(
    y_hat: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    k: u32,
) -> PyResult<TripCheck> {
    check_k(k)?;
    let yb = ByteIn::extract(y_hat, "y_hat")?;
    let ua = RoRing::extract(u, "u", k)?;
    if matches!(ua, RoRing::B(_)) {
        return Err(verr("u must be a 2-D numpy array"));
    }
    let (job, rhs_v, c, m) = {
        let ys = yb.slice("y_hat")?;
        let ra = rhs
            .downcast::<PyArray2<u64>>()
            .map_err(|_| verr("rhs must be a 2-D uint64 numpy array"))?
            .try_readonly()
            .map_err(|e| verr(format!("rhs: {e}")))?;
        let (_, c, m) = fo_shapes(ys, &ua, &ra, k)?;
        let rs = c_ro(&ra).ok_or_else(|| verr("rhs must be C-contiguous"))?.to_vec();
        let (us, _) = ua.src("u")?;
        let raw_u = match us {
            Src::U64(s) => RawSrc::U64(s.as_ptr(), s.len()),
            Src::U32(s) => RawSrc::U32(s.as_ptr(), s.len()),
            Src::Bytes(_) => return Err(verr("u must be a 2-D numpy array")),
        };
        (RawJob { y: (ys.as_ptr(), ys.len()), u: raw_u }, rs, c, m)
    };
    // SAFETY: only the lifetime changes. The guards are kept by the returned TripCheck and dropped
    // with the GIL held (see `Held`); their strong references keep the data the pointers above
    // address alive and in place (numpy data buffers and bytes objects never move).
    let held = unsafe {
        Held {
            _y: std::mem::transmute::<ByteIn<'py>, ByteIn<'static>>(yb),
            _u: std::mem::transmute::<RoRing<'py>, RoRing<'static>>(ua),
        }
    };
    let state = Arc::new(CheckState { res: Mutex::new(None), cv: Condvar::new() });
    let st = Arc::clone(&state);
    let run: Job = Box::new(move || {
        let job = job;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the TripCheck holding `keep` waits for this job before releasing them.
            let ys = unsafe { std::slice::from_raw_parts(job.y.0, job.y.1) };
            let us = match job.u {
                RawSrc::U64(p, l) => Src::U64(unsafe { std::slice::from_raw_parts(p, l) }),
                RawSrc::U32(p, l) => Src::U32(unsafe { std::slice::from_raw_parts(p, l) }),
            };
            freivalds_output(ys, us, &rhs_v, c, m, k)
        }));
        let mut rhs_v = rhs_v;
        wipe(&mut rhs_v);
        st.set(res.unwrap_or((STATUS_INTERNAL, 0)));
    });
    let sent = helper().lock().unwrap_or_else(|e| e.into_inner()).send(run);
    if let Err(e) = sent {
        // the helper thread is gone: run here (same result)
        (e.0)();
    }
    Ok(TripCheck { state, held: Mutex::new(Some(held)) })
}

#[cfg(test)]
mod tests;
