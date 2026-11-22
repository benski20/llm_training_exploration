//! Column-split prepare of decode-sized trips (`PGW_TRIP_SPLIT`, opt-in): the rows of one trip are cut
//! into a **fixed** partition of [`SPLIT_BLOCK`]-element column blocks (it depends on `n` only), and the
//! blocks are spread over `cores` tasks of the persistent pool. Every per-row reduction is a per-block
//! partial merged on the calling thread in block order, so the bytes do not depend on the thread count:
//!
//! | reduction | partial | merge |
//! |---|---|---|
//! | `max |x|` (width scale) | per block | `max` (exact) |
//! | `sum x^2` (the window start `tau` only) | float64 per block | added in block order (fixed) |
//! | `sum q_b^2` per candidate width | exact integer, wrapping | wrapping add (associative) |
//! | ring input half `uw . x_hat mod 2^K` | wrapping word sums | wrapping add |
//! | prime input half `uw . xq mod p` | lazy sums, reduced | sum, then reduced mod `p` (exact) |
//!
//! Output bytes (payload, `sx`, `x_hat`, `rhs`) are **identical to the one-thread kernel**
//! ([`super::prepare_into_field`] with `threads = 1`): the chosen width is the reference's (the
//! width window provably contains it for any `tau` within the plan's margin, and the float64 `sum x^2`
//! in another order differs by far less than that margin; the per-width test uses the exact integer
//! sum), every element is quantized with the same float64 operations, and the input halves are exact.
//!
//! Only plans whose evaluable widths all take the integer sum of squares ([`SumPath::Int`], the
//! default K32A12C27 at every Qwen width) or none ([`SumPath::Off`], uncapped rings) are split: the
//! reference's float64 sum is sequential and cannot be split exactly. Other plans run on one thread.
//!
//! Constant time: the partition, the loops and the pool tasks depend on public shapes only; the
//! selected width, its scale and `qmax` are mask selects; the selected width's `q` is recomputed in
//! the pack pass with the same float64 operations as each candidate (one quantization per element,
//! whatever the width). Every scratch and partial holding a secret-derived value is wiped.

use super::*;
use crate::prepare::pool;

/// Elements per column block of the fixed partition (public).
pub const SPLIT_BLOCK: usize = 2048;

/// Raw output pointer shared by tasks that write pairwise disjoint ranges.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
// SAFETY: tasks write disjoint ranges; the submitter waits for every task before reading.
unsafe impl<T> Send for Shared<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// The pointer (a method, so closures capture the whole `Sync` wrapper, not the raw field).
    #[inline(always)]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

/// Whether [`prepare_split`] splits a call of `rows x n` with this plan over `cores` tasks.
pub fn split_applies(n: usize, prm: PrepareParams, cores: usize) -> bool {
    if cores < 2 || n < 2 * SPLIT_BLOCK || prm.validate().is_err() {
        return false;
    }
    matches!(plan_for(n, prm).sum, SumPath::Int | SumPath::Off) && pool::global().workers() > 0
}

/// Blocks `[a, b)` of task `i` of `tasks` over `nblk` blocks.
#[inline(always)]
fn task_blocks(nblk: usize, tasks: usize, i: usize) -> (usize, usize) {
    (nblk * i / tasks, nblk * (i + 1) / tasks)
}

/// [`super::prepare_into_field`] with each row split into column blocks over `cores` pool tasks (see the
/// module docs). Falls back to the one-thread kernel when the split does not apply. Same outputs and
/// errors; on error every output is zeroed.
#[allow(clippy::too_many_arguments)]
pub fn prepare_split(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    half: Option<InputHalf<'_>>,
    field: u64,
    out: PrepOut<'_>,
    cores: usize,
) -> Result<(), PrepareError> {
    if !split_applies(n, prm, cores) || half.is_some_and(|h| h.c > MAX_SPLIT_C) {
        return prepare_into_field(x, r, n, prm, half, field, out, 1);
    }
    // the same validation as the one-thread kernel (sizes, field, word types)
    prm.validate()?;
    let nb = (prm.k / 8) as usize;
    let rows = out.sx.len();
    let c = half.map_or(0, |h| h.c);
    if field != 0 {
        if field != crate::prime::P31 || prm.k > 32 || n > crate::prime::MAX_LEN {
            return Err(PrepareError::Invalid("the prime check needs field = 2^31 - 1 and K <= 32".into()));
        }
        if let Some(h) = half {
            if !matches!(h.uw, Src::U32(_)) {
                return Err(PrepareError::Invalid("the prime input half needs uw as uint32 words".into()));
            }
        }
    }
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
    let plan = plan_for(n, prm);
    let res = match prm.k {
        8 => split_rows::<u32, 1>(x, r, n, prm, &plan, half, field, out, cores),
        16 => split_rows::<u32, 2>(x, r, n, prm, &plan, half, field, out, cores),
        24 => split_rows::<u32, 3>(x, r, n, prm, &plan, half, field, out, cores),
        32 => split_rows::<u32, 4>(x, r, n, prm, &plan, half, field, out, cores),
        40 => split_rows::<u64, 5>(x, r, n, prm, &plan, half, field, out, cores),
        48 => split_rows::<u64, 6>(x, r, n, prm, &plan, half, field, out, cores),
        _ => split_rows::<u64, 7>(x, r, n, prm, &plan, half, field, out, cores),
    };
    match res {
        SplitRes::Done(r) => r,
        // unreachable (the window always holds the reference's width): the one-thread kernel's
        // full search, which rewrites every output
        SplitRes::Fallback(out) => prepare_into_field(x, r, n, prm, half, field, out, 1),
    }
}

enum SplitRes<'a> {
    Done(Result<(), PrepareError>),
    Fallback(PrepOut<'a>),
}

/// Run `f(task)` for `tasks` tasks on the pool (fail closed: a panicking task is an internal error).
fn run_tasks(tasks: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
    pool::global().run(tasks, f)
}

#[allow(clippy::too_many_arguments)]
fn split_rows<'a, W: Word, const NB: usize>(
    x: &[f32],
    r: &[i64],
    n: usize,
    prm: PrepareParams,
    plan: &WidthPlan,
    half: Option<InputHalf<'_>>,
    field: u64,
    out: PrepOut<'a>,
    cores: usize,
) -> SplitRes<'a> {
    let rows = out.sx.len();
    let nblk = n.div_ceil(SPLIT_BLOCK);
    let tasks = cores.min(nblk).max(1);
    let w = plan.w as usize;
    let c = half.map_or(0, |h| h.c);
    let blk = |b: usize| (b * SPLIT_BLOCK, ((b + 1) * SPLIT_BLOCK).min(n));

    // ---- phase A: max |x| bits and sum x^2 per (row, block) ----
    let mut amax_p = vec![0u32; rows * nblk];
    let mut ss_p = vec![0f64; rows * nblk];
    {
        let (ap, sp) = (Shared(amax_p.as_mut_ptr()), Shared(ss_p.as_mut_ptr()));
        let ok = run_tasks(tasks, &|i| {
            let (b0, b1) = task_blocks(nblk, tasks, i);
            for t in 0..rows {
                for b in b0..b1 {
                    let (j0, j1) = blk(b);
                    let xs = &x[t * n + j0..t * n + j1];
                    let (bits, ss) = phase_a(xs);
                    // SAFETY: (t, b) is written by this task only; the vectors outlive the run.
                    unsafe {
                        *ap.ptr().add(t * nblk + b) = bits;
                        *sp.ptr().add(t * nblk + b) = ss;
                    }
                }
            }
        });
        if !ok {
            return fail(out, &mut [&mut ss_p], &mut amax_p, "internal error in a prepare task");
        }
    }
    let mut amax = vec![0f64; rows];
    let mut start = vec![0u32; rows];
    let mut nonfinite = false;
    for t in 0..rows {
        let bits = amax_p[t * nblk..(t + 1) * nblk].iter().fold(0u32, |m, &b| m.max(b));
        nonfinite |= bits >= INF_BITS;
        let a = f64::from(f32::from_bits(bits.min(INF_BITS - 1)));
        amax[t] = a;
        let ss: f64 = ss_p[t * nblk..(t + 1) * nblk].iter().fold(0.0, |s, v| s + v);
        let tau = ss.sqrt() / if a > 0.0 { a } else { 1.0 };
        start[t] = plan.start(tau);
    }
    wipe(&mut ss_p);
    wipe(&mut amax_p);
    if nonfinite {
        let mut o = out;
        zero_prep(&mut o);
        wipe(&mut amax);
        return SplitRes::Done(Err(PrepareError::Overflow("non-finite activation values cannot be quantized".into())));
    }

    // ---- phase B: exact integer sum of q_b^2 per (row, width, block) ----
    let mut sq_p = vec![0i64; rows * w * nblk];
    if prm.cap != 0 {
        let qp = Shared(sq_p.as_mut_ptr());
        let ok = run_tasks(tasks, &|i| {
            let (b0, b1) = task_blocks(nblk, tasks, i);
            let mut q = [0i64; SPLIT_BLOCK];
            for t in 0..rows {
                for wi in 0..w {
                    let b = start[t] + wi as u32;
                    let qm = qmax_of(b);
                    let s = if amax[t] > 0.0 { amax[t] / qm } else { 1.0 };
                    for bk in b0..b1 {
                        let (j0, j1) = blk(bk);
                        let v = phase_b(&x[t * n + j0..t * n + j1], s, qm, &mut q[..j1 - j0]);
                        // SAFETY: written by this task only.
                        unsafe { *qp.ptr().add((t * w + wi) * nblk + bk) = v };
                    }
                }
            }
            wipe(&mut q);
        });
        if !ok {
            wipe(&mut amax);
            wipe(&mut sq_p);
            return fail(out, &mut [], &mut [], "internal error in a prepare task");
        }
    }
    // the reference's width test on the merged sums; mask selects (as prepare_row)
    let limit = (1u64 << prm.cap) as f64;
    let mut s_sel = vec![0u64; rows];
    let mut qm_sel = vec![0u64; rows];
    let mut overflow = false;
    for t in 0..rows {
        let (mut sel, mut qsel, mut any) = (0u64, 0u64, false);
        for wi in 0..w {
            let b = start[t] + wi as u32;
            let qm = qmax_of(b);
            let s = if amax[t] > 0.0 { amax[t] / qm } else { 1.0 };
            let ss = sq_p[(t * w + wi) * nblk..(t * w + wi + 1) * nblk].iter().fold(0i64, |a, v| a.wrapping_add(*v));
            let fit = prm.cap == 0 || (ss as f64).sqrt() * prm.wnorm * NORM_SLACK <= limit;
            let mk = 0u64.wrapping_sub(u64::from(fit));
            sel = (sel & !mk) | (s.to_bits() & mk);
            qsel = (qsel & !mk) | (qm.to_bits() & mk);
            any |= fit;
        }
        if !any {
            if start[t] <= 2 {
                overflow = true;
            } else {
                // unreachable: hand the call to the one-thread kernel's fallback search
                wipe(&mut sq_p);
                wipe(&mut amax);
                wipe(&mut s_sel);
                return SplitRes::Fallback(out);
            }
        }
        s_sel[t] = sel;
        qm_sel[t] = qsel;
    }
    wipe(&mut sq_p);
    wipe(&mut amax);
    if overflow {
        let mut o = out;
        zero_prep(&mut o);
        wipe(&mut s_sel);
        return SplitRes::Done(Err(PrepareError::Overflow(format!(
            "K{}A{}C{}: activations cannot be bounded by 2^{} for this linear",
            prm.k, prm.a, prm.cap, prm.cap
        ))));
    }

    // ---- phase C: quantize with the selected width, mask, pack, x_hat, input-half partials ----
    // ring: one word partial per (row, vector, block); prime: two lazy sums (reduced) per (row, vector, block)
    let per = if field != 0 { 2 } else { 1 };
    let mut half_p = vec![0u64; rows * c * nblk * per];
    let PrepOut { payload, sx, xhat, rhs } = out;
    {
        let pp = Shared(payload.as_mut_ptr());
        let xp = Shared(xhat.as_mut_ptr());
        let want_xhat = !xhat.is_empty();
        let hp = Shared(half_p.as_mut_ptr());
        let kmask = ((1u64 << prm.k) - 1) as i64;
        let ok = run_tasks(tasks, &|i| {
            let (b0, b1) = task_blocks(nblk, tasks, i);
            let mut sc = PackScratch::<W>::new();
            for t in 0..rows {
                let s = f64::from_bits(s_sel[t]);
                let qm = f64::from_bits(qm_sel[t]);
                for bk in b0..b1 {
                    let (j0, j1) = blk(bk);
                    let len = j1 - j0;
                    let base = t * n + j0;
                    // SAFETY: [base, base + len) of the payload / x_hat belongs to this task only.
                    let pl = unsafe { std::slice::from_raw_parts_mut(pp.ptr().add(base * NB), len * NB) };
                    let xh = if want_xhat {
                        Some(unsafe { std::slice::from_raw_parts_mut(xp.ptr().add(base), len) })
                    } else {
                        None
                    };
                    let mut parts = [0u64; 2 * MAX_SPLIT_C];
                    phase_c::<W, NB>(
                        &x[base..base + len],
                        &r[base..base + len],
                        s,
                        qm,
                        kmask,
                        prm.k,
                        pl,
                        xh,
                        half.map(|h| (h, j0, n)),
                        field,
                        &mut sc,
                        &mut parts[..c * per],
                    );
                    for (ci, v) in parts[..c * per].iter().enumerate() {
                        // SAFETY: (t, vector, block) slots are written by this task only.
                        unsafe { *hp.ptr().add(((t * c + ci / per) * nblk + bk) * per + ci % per) = *v };
                    }
                    wipe(&mut parts);
                }
            }
            sc.wipe();
        });
        if !ok {
            let mut o = PrepOut { payload, sx, xhat, rhs };
            zero_prep(&mut o);
            wipe(&mut half_p);
            wipe(&mut s_sel);
            return SplitRes::Done(Err(PrepareError::Invalid("internal error in a prepare task".into())));
        }
    }
    // merge the input halves in block order
    if c > 0 {
        let kmask = (1u64 << prm.k) - 1;
        for t in 0..rows {
            for i in 0..c {
                let o = (t * c + i) * nblk * per;
                rhs[t * c + i] = if field != 0 {
                    let (mut a, mut au) = (0u64, 0u64);
                    for bk in 0..nblk {
                        a += half_p[o + bk * 2]; // each < 2^31: nblk <= 2^19 terms fit
                        au += half_p[o + bk * 2 + 1];
                    }
                    crate::prime::finish_dot_pub(a, au)
                } else {
                    let s = half_p[o..o + nblk].iter().fold(W::default(), |s, v| s.wadd(W::trunc(*v)));
                    s.widen() & kmask
                };
            }
        }
    }
    wipe(&mut half_p);
    for t in 0..rows {
        sx[t] = f64::from_bits(s_sel[t]);
    }
    wipe(&mut s_sel);
    SplitRes::Done(Ok(()))
}

/// Largest `c` of a split call (more check vectors take the one-thread kernel).
pub const MAX_SPLIT_C: usize = 16;

/// Zero the outputs and the given partials; an internal-error result.
fn fail<'a>(out: PrepOut<'a>, f: &mut [&mut Vec<f64>], u: &mut [u32], msg: &str) -> SplitRes<'a> {
    let mut o = out;
    zero_prep(&mut o);
    for v in f.iter_mut() {
        wipe(&mut v[..]);
    }
    wipe(u);
    SplitRes::Done(Err(PrepareError::Invalid(msg.into())))
}

/// Per-task scratch of the pack pass (block-sized).
struct PackScratch<W: Word> {
    q: Vec<i64>,
    xw: Vec<W>,
    ub: Vec<W>,
    xq: Vec<i32>,
    uq: Vec<u32>,
}

impl<W: Word> PackScratch<W> {
    fn new() -> Self {
        PackScratch {
            q: vec![0; SPLIT_BLOCK],
            xw: vec![W::default(); SPLIT_BLOCK],
            ub: vec![W::default(); SPLIT_BLOCK],
            xq: vec![0; SPLIT_BLOCK],
            uq: vec![0; SPLIT_BLOCK],
        }
    }
    fn wipe(&mut self) {
        wipe(&mut self.q);
        wipe(&mut self.xw);
        wipe(&mut self.ub);
        wipe(&mut self.xq);
        wipe(&mut self.uq);
    }
}

// ---- the per-block kernels (AVX2 builds selected at run time; same IEEE / integer results) ----

#[inline(always)]
fn phase_a_body(xs: &[f32]) -> (u32, f64) {
    (row_amax_bits(xs), sumsq(xs))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn phase_a_avx2(xs: &[f32]) -> (u32, f64) {
    phase_a_body(xs)
}

fn phase_a(xs: &[f32]) -> (u32, f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected.
        return unsafe { phase_a_avx2(xs) };
    }
    phase_a_body(xs)
}

#[inline(always)]
fn phase_b_body(xs: &[f32], s: f64, qm: f64, q: &mut [i64]) -> i64 {
    quant_ss::<true, false>(xs, s, qm, q).0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn phase_b_avx2(xs: &[f32], s: f64, qm: f64, q: &mut [i64]) -> i64 {
    phase_b_body(xs, s, qm, q)
}

fn phase_b(xs: &[f32], s: f64, qm: f64, q: &mut [i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected.
        return unsafe { phase_b_avx2(xs, s, qm, q) };
    }
    phase_b_body(xs, s, qm, q)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn phase_c_body<W: Word, const NB: usize>(
    xs: &[f32],
    rs: &[i64],
    s: f64,
    qm: f64,
    kmask: i64,
    k: u32,
    payload: &mut [u8],
    xhat: Option<&mut [i64]>,
    half: Option<(InputHalf<'_>, usize, usize)>,
    field: u64,
    sc: &mut PackScratch<W>,
    parts: &mut [u64],
) {
    let len = xs.len();
    let q = &mut sc.q[..len];
    quant_exact(xs, s, qm, q);
    let xw = &mut sc.xw[..len];
    for j in 0..len {
        let v = (q[j].wrapping_add(rs[j]) & kmask) as u64;
        xw[j] = W::trunc(v);
        payload[j * NB..(j + 1) * NB].copy_from_slice(&v.to_le_bytes()[..NB]);
    }
    if let Some(xh) = xhat {
        for (o, v) in xh.iter_mut().zip(xw.iter()) {
            *o = v.widen() as i64;
        }
    }
    let Some((h, j0, n)) = half else { return };
    if field != 0 {
        let shift = 32 - k;
        let xq = &mut sc.xq[..len];
        for (o, (v, r)) in xq.iter_mut().zip(xw.iter().zip(rs)) {
            *o = (((v.widen() as u32).wrapping_sub(*r as u32)) << shift) as i32 >> shift;
        }
        if let Src::U32(uw) = h.uw {
            for i in 0..h.c {
                let (a, au) = crate::prime::dot_lazy(&uw[i * n + j0..i * n + j0 + len], xq);
                parts[2 * i] = crate::prime::reduce(a);
                parts[2 * i + 1] = crate::prime::reduce(au);
            }
        }
    } else {
        for i in 0..h.c {
            let d = match h.uw.words::<W>(i * n + j0, i * n + j0 + len) {
                Some(us) => dot(xw, us),
                None => {
                    let ub = &mut sc.ub[..len];
                    h.uw.fill::<W, NB>(i * n + j0, i * n + j0 + len, ub);
                    dot(xw, ub)
                }
            };
            parts[i] = d.widen();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn phase_c_avx2<W: Word, const NB: usize>(
    xs: &[f32],
    rs: &[i64],
    s: f64,
    qm: f64,
    kmask: i64,
    k: u32,
    payload: &mut [u8],
    xhat: Option<&mut [i64]>,
    half: Option<(InputHalf<'_>, usize, usize)>,
    field: u64,
    sc: &mut PackScratch<W>,
    parts: &mut [u64],
) {
    phase_c_body::<W, NB>(xs, rs, s, qm, kmask, k, payload, xhat, half, field, sc, parts)
}

#[allow(clippy::too_many_arguments)]
fn phase_c<W: Word, const NB: usize>(
    xs: &[f32],
    rs: &[i64],
    s: f64,
    qm: f64,
    kmask: i64,
    k: u32,
    payload: &mut [u8],
    xhat: Option<&mut [i64]>,
    half: Option<(InputHalf<'_>, usize, usize)>,
    field: u64,
    sc: &mut PackScratch<W>,
    parts: &mut [u64],
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected.
        return unsafe { phase_c_avx2::<W, NB>(xs, rs, s, qm, kmask, k, payload, xhat, half, field, sc, parts) };
    }
    phase_c_body::<W, NB>(xs, rs, s, qm, kmask, k, payload, xhat, half, field, sc, parts)
}

// ---- split r keystream ---------------------------------------------------------------------------

/// Elements of `r` per task of [`fill_ring_split`] (any partition gives the same bits: `r` is a function
/// of the stream position only).
const R_TASK_ELEMS: usize = 4096;

/// `fill_ring` over `cores` pool tasks (fixed element ranges). `out` must be zero (the keystream is
/// XORed in). Same bits as one call.
pub fn fill_ring_split(key: &[u8; 32], elem_offset: u64, out: &mut [i64], k: u32, cores: usize) -> Result<(), String> {
    let len = out.len();
    if cores < 2 || len < 2 * R_TASK_ELEMS || pool::global().workers() == 0 {
        return crate::store::fill_ring(key, elem_offset, out, k);
    }
    let nt = len.div_ceil(R_TASK_ELEMS);
    let tasks = cores.min(nt);
    let op = Shared(out.as_mut_ptr());
    let errs: Vec<Mutex<Option<String>>> = (0..tasks).map(|_| Mutex::new(None)).collect();
    let ok = pool::global().run(tasks, &|i| {
        let (a, b) = task_blocks(nt, tasks, i);
        let (e0, e1) = (a * R_TASK_ELEMS, (b * R_TASK_ELEMS).min(len));
        if e1 <= e0 {
            return;
        }
        // SAFETY: [e0, e1) belongs to this task only; `out` outlives the run.
        let part = unsafe { std::slice::from_raw_parts_mut(op.ptr().add(e0), e1 - e0) };
        if let Err(e) = crate::store::fill_ring(key, elem_offset + e0 as u64, part, k) {
            *errs[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
        }
    });
    let err = errs.into_iter().find_map(|e| e.into_inner().unwrap_or_else(|p| p.into_inner()));
    if !ok || err.is_some() {
        wipe(out);
        return Err(err.unwrap_or_else(|| "internal error in an r task".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    type Outs = (Result<(), PrepareError>, Vec<u8>, Vec<f64>, Vec<i64>, Vec<u64>);

    #[allow(clippy::too_many_arguments)]
    fn run(
        x: &[f32],
        r: &[i64],
        n: usize,
        prm: PrepareParams,
        uw: Option<&[u32]>,
        c: usize,
        field: u64,
        cores: usize,
    ) -> Outs {
        let rows = x.len() / n;
        let nb = (prm.k / 8) as usize;
        let (mut pl, mut sx, mut xh, mut rh) =
            (vec![7u8; rows * n * nb], vec![0.0; rows], vec![0i64; rows * n], vec![0u64; rows * c]);
        let half = uw.map(|u| InputHalf { uw: Src::U32(u), c });
        let out = PrepOut { payload: &mut pl, sx: &mut sx, xhat: &mut xh, rhs: &mut rh };
        let res = if cores == 0 {
            prepare_into_field(x, r, n, prm, half, field, out, 1)
        } else {
            prepare_split(x, r, n, prm, half, field, out, cores)
        };
        (res, pl, sx, xh, rh)
    }

    #[test]
    fn split_prepare_is_identical_for_any_core_count() {
        let mut s = 42u64;
        for (k, a, cap, n, rows, kind) in [
            (32u32, 12u32, 27u32, 5000usize, 1usize, 0u8),
            (32, 12, 27, 8192 + 7, 2, 1),
            (16, 8, 0, 4096, 2, 0),
            (24, 12, 20, 6000, 1, 2),
        ] {
            let wnorm = if cap == 0 { 1.0 } else { 0.3 * 127.0 * (n as f64).sqrt() };
            let prm = PrepareParams { k, a, cap, wnorm };
            let x: Vec<f32> = (0..rows * n)
                .map(|i| {
                    let v = ((rng(&mut s) >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32;
                    if kind == 1 && i % n == 17 {
                        300.0
                    } else if kind == 2 {
                        v * 1e-3
                    } else {
                        v * 4.0
                    }
                })
                .collect();
            let kmask = (1u64 << k) - 1;
            let r: Vec<i64> = (0..rows * n).map(|_| (rng(&mut s) & kmask) as i64).collect();
            let c = 3;
            let p31 = crate::prime::P31;
            let uw_ring: Vec<u32> = (0..c * n).map(|_| rng(&mut s) as u32).collect();
            let uw_p: Vec<u32> = (0..c * n).map(|_| (rng(&mut s) % p31) as u32).collect();
            for (uw, field) in [(None, 0u64), (Some(&uw_ring[..]), 0), (Some(&uw_p[..]), p31)] {
                let cc = if uw.is_some() { c } else { 0 };
                let want = run(&x, &r, n, prm, uw, cc, field, 0);
                assert!(want.0.is_ok());
                for cores in [2usize, 3, 4, 9] {
                    let got = run(&x, &r, n, prm, uw, cc, field, cores);
                    assert!(got.0.is_ok());
                    assert!(
                        got.1 == want.1 && got.2 == want.2 && got.3 == want.3 && got.4 == want.4,
                        "k={k} n={n} cores={cores}"
                    );
                }
            }
            // non-finite input: the same error, outputs zeroed
            let mut xb = x.clone();
            xb[n - 1] = f32::NAN;
            let got = run(&xb, &r, n, prm, None, 0, 0, 4);
            assert!(matches!(got.0, Err(PrepareError::Overflow(_))) && got.1.iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn split_prime_output_matches() {
        let mut s = 7u64;
        let (rows, m, c, k) = (2usize, 3 * 8192 + 100, 1usize, 32u32);
        let p31 = crate::prime::P31;
        let u: Vec<u32> = (0..c * m).map(|_| (rng(&mut s) % p31) as u32).collect();
        let wr: Vec<u32> = (0..rows * m).map(|_| rng(&mut s) as u32).collect();
        let yq: Vec<i32> = (0..rows * m).map(|_| (rng(&mut s) % 2001) as i32 - 1000).collect();
        let y: Vec<u8> = yq.iter().zip(&wr).flat_map(|(q, w)| (*q as u32).wrapping_add(*w).to_le_bytes()).collect();
        let mut rhs: Vec<u64> = (0..rows).map(|t| crate::prime::dot_mod_p(&u, &yq[t * m..(t + 1) * m])).collect();
        for bound in [1000u64, 999] {
            let want = crate::prime::output_half(&y, Src::U32(&wr), &u, &rhs, c, m, k, bound);
            for cores in [2usize, 3, 4] {
                assert_eq!(crate::prime::output_half_split(&y, Src::U32(&wr), &u, &rhs, c, m, k, bound, cores), want);
            }
        }
        let ok = crate::prime::output_half(&y, Src::U32(&wr), &u, &rhs, c, m, k, 1000);
        assert_eq!(ok.0, crate::finish::STATUS_OK);
        rhs[1] = (rhs[1] + 1) % p31;
        for cores in [0usize, 2, 4] {
            assert_eq!(
                crate::prime::output_half_split(&y, Src::U32(&wr), &u, &rhs, c, m, k, 1000, cores),
                (crate::finish::STATUS_FREIVALDS, 1)
            );
        }
    }

    #[test]
    fn split_r_matches_one_call() {
        let key = [9u8; 32];
        for (len, off) in [(10_000usize, 0u64), (3 * 4096 + 5, 12_345)] {
            let mut a = vec![0i64; len];
            let mut b = vec![0i64; len];
            crate::store::fill_ring(&key, off, &mut a, 32).unwrap();
            fill_ring_split(&key, off, &mut b, 32, 4).unwrap();
            assert_eq!(a, b);
        }
    }
}
