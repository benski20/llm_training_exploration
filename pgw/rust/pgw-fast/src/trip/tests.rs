//! Property tests: the trip kernels against the existing kernels (`quantize_mask_pack_into`, the
//! descending width search, and `finish_group_run`) on random and adversarial inputs, every ring.

use super::*;
use crate::finish::{finish_group_run, Finished, STATUS_RANGE};
use crate::prepare::quantize_mask_pack_into;

/// splitmix64
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const RINGS: [u32; 7] = [8, 16, 24, 32, 40, 48, 56];

/// Trial multiplier (`PGW_TRIP_STRESS=50 cargo test --release trip` for a long run).
fn stress() -> usize {
    std::env::var("PGW_TRIP_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// A row of one of several shapes of activation distribution (some built to hit exact ties).
fn gen_row(g: &mut Rng, n: usize, kind: u64) -> Vec<f32> {
    match kind {
        0 => (0..n).map(|_| ((g.unit() + g.unit() + g.unit() - 1.5) * 4.0) as f32).collect(),
        1 => {
            // heavy-tailed: small values and one or two spikes (low ||x|| / max|x|)
            let mut v: Vec<f32> = (0..n).map(|_| ((g.unit() - 0.5) * 0.01) as f32).collect();
            let i = g.below(n as u64) as usize;
            v[i] = (300.0 * (g.unit() + 0.5)) as f32 * if g.next() & 1 == 0 { 1.0 } else { -1.0 };
            v
        }
        2 => vec![0.0; n],
        3 => {
            // constant magnitude, random signs (maximal ||x|| / max|x|)
            let c = (g.unit() * 10.0 + 0.1) as f32;
            (0..n).map(|_| if g.next() & 1 == 0 { c } else { -c }).collect()
        }
        4 => {
            // half-integers on a grid: exact ties for the integer scale
            let amax = 2047.0f32;
            let mut v: Vec<f32> = (0..n).map(|_| (g.below(4095) as f32 - 2047.0) + 0.5).collect();
            v[0] = amax;
            v
        }
        5 => (0..n).map(|_| f32::from_bits((g.next() as u32) & 0x007f_ffff)).collect(), // subnormal
        6 => {
            let mut v: Vec<f32> = (0..n).map(|_| ((g.unit() - 0.5) * 1e6) as f32).collect();
            v[g.below(n as u64) as usize] = -0.0;
            v
        }
        _ => (0..n).map(|_| f32::from_bits((g.next() as u32) & 0x7f7f_ffff)).collect(), // any finite
    }
}

fn reference(x: &[f32], r: &[i64], n: usize, p: PrepareParams) -> Result<(Vec<u8>, Vec<u64>, Vec<i64>), PrepareError> {
    let rows = x.len() / n;
    let mut xh = vec![0i64; rows * n];
    let mut sx = vec![0.0f64; rows];
    let mut pl = vec![0u8; rows * n * (p.k as usize / 8)];
    quantize_mask_pack_into(x, r, n, p, &mut xh, &mut sx, &mut pl)?;
    Ok((pl, sx.iter().map(|v| v.to_bits()).collect(), xh))
}

/// `uw . x_hat mod 2^K` per row, the obvious way.
fn naive_rhs(xh: &[i64], uw: &[u64], c: usize, n: usize, k: u32) -> Vec<u64> {
    let mask = u64::mask(k);
    let rows = xh.len() / n;
    let mut out = vec![0u64; rows * c];
    for t in 0..rows {
        for i in 0..c {
            let mut s = 0u64;
            for j in 0..n {
                s = s.wrapping_add(uw[i * n + j].wrapping_mul(xh[t * n + j] as u64));
            }
            out[t * c + i] = s & mask;
        }
    }
    out
}

struct Got {
    payload: Vec<u8>,
    sx: Vec<u64>,
    xhat: Vec<i64>,
    rhs: Vec<u64>,
}

fn run_trip(
    x: &[f32],
    r: &[i64],
    n: usize,
    p: PrepareParams,
    uw: Option<(&[u64], usize)>,
    threads: usize,
) -> Result<Got, PrepareError> {
    let rows = x.len() / n;
    let mut payload = vec![0xA5u8; rows * n * (p.k as usize / 8)];
    let mut sx = vec![-1.0f64; rows];
    let mut xhat = vec![-1i64; rows * n];
    let c = uw.map_or(0, |u| u.1);
    let mut rhs = vec![7u64; rows * c];
    let half = uw.map(|(u, c)| InputHalf { uw: Src::U64(u), c });
    let out = PrepOut { payload: &mut payload, sx: &mut sx, xhat: &mut xhat, rhs: &mut rhs };
    match prepare_into(x, r, n, p, half, out, threads) {
        Ok(()) => Ok(Got { payload, sx: sx.iter().map(|v| v.to_bits()).collect(), xhat, rhs }),
        Err(e) => {
            assert!(payload.iter().all(|b| *b == 0), "payload not zeroed after an error");
            assert!(rhs.iter().all(|b| *b == 0), "rhs not zeroed after an error");
            assert!(sx.iter().all(|b| b.to_bits() == 0), "sx not zeroed after an error");
            assert!(xhat.iter().all(|b| *b == 0), "xhat not zeroed after an error");
            Err(e)
        }
    }
}

fn check_case(g: &mut Rng, x: &[f32], n: usize, p: PrepareParams, label: &str) {
    let rows = x.len() / n;
    let mask = u64::mask(p.k);
    let r: Vec<i64> = (0..rows * n).map(|_| (g.next() & mask) as i64).collect();
    let c = 1 + g.below(3) as usize;
    let uw: Vec<u64> = (0..c * n).map(|_| g.next() & mask).collect();
    let want = reference(x, &r, n, p);
    for threads in [1usize, 0] {
        let got = run_trip(x, &r, n, p, Some((&uw, c)), threads);
        match (&want, got) {
            (Ok((pl, sx, xh)), Ok(gt)) => {
                assert_eq!(&gt.sx, sx, "{label}: sx (width choice) differs");
                assert_eq!(&gt.xhat, xh, "{label}: x_hat differs");
                assert_eq!(&gt.payload, pl, "{label}: payload differs");
                assert_eq!(gt.rhs, naive_rhs(xh, &uw, c, n, p.k), "{label}: input half differs");
            }
            (Err(a), Err(b)) => assert_eq!(a, &b, "{label}: different errors"),
            (a, b) => panic!("{label}: reference {:?} vs trip {:?}", a.as_ref().err(), b.err()),
        }
    }
    // the window always contained the choice (the non-constant-time fallback never ran)
    assert_eq!(WINDOW_MISSES.load(std::sync::atomic::Ordering::Relaxed), 0, "{label}: window missed");
}

/// Bit-identical to `quantize_mask_pack_into` (hence to the descending search and to Python) on
/// every ring, for random parameters and row distributions.
#[test]
fn prepare_matches_existing_kernel_every_ring() {
    let mut g = Rng(0x5eed_0001);
    for &k in &RINGS {
        let amax_bits = (k - 1).min(48);
        for trial in 0..60 * stress() {
            let a = 2 + g.below(u64::from(amax_bits - 1)) as u32;
            let cap = match trial % 4 {
                0 => 0,
                1 => 1 + g.below(62) as u32,
                _ => (k.saturating_sub(3)).max(1),
            };
            let n = [1usize, 3, 17, 64, 300][trial % 5];
            let rows = 1 + g.below(4) as usize;
            let x: Vec<f32> = (0..rows)
                .flat_map(|_| {
                    let kd = g.below(8);
                    gen_row(&mut g, n, kd)
                })
                .collect();
            let wnorm = match g.below(5) {
                0 => 127.0 * (n as f64).sqrt() * g.unit(),
                1 => 1.0 + g.unit() * 50.0,
                2 => 10f64.powf(g.unit() * 12.0 - 2.0),
                3 => 1e30,
                _ => [0.0, -3.0, f64::NAN, f64::INFINITY][g.below(4) as usize],
            };
            let p = PrepareParams { k, a, cap, wnorm };
            check_case(&mut g, &x, n, p, &format!("K{k}A{a}C{cap} n={n} wnorm={wnorm} trial {trial}"));
        }
    }
}

/// `wnorm` placed exactly on (and one ulp around) the reference's decision boundary of some width:
/// the widths whose sums of squares straddle the threshold are inside the window.
#[test]
fn prepare_boundary_wnorms() {
    let mut g = Rng(0x5eed_0002);
    for &k in &RINGS {
        let a = (k - 1).min(48).min(20);
        for trial in 0..80 * stress() {
            let n = [5usize, 64, 777][trial % 3];
            let cap = [16u32, 20, 27, 40][trial % 4];
            let x = {
                let kd = g.below(8);
                gen_row(&mut g, n, kd)
            };
            let bits = row_amax_bits(&x);
            let amax = f64::from(f32::from_bits(bits));
            let b = 2 + g.below(u64::from(a - 1)) as u32;
            let qm = qmax_of(b);
            let s = if amax > 0.0 { amax / qm } else { 1.0 };
            let mut q = vec![0i64; n];
            quant_exact(&x, s, qm, &mut q);
            let ss = sum_squares(&q, qm);
            if ss == 0.0 {
                continue;
            }
            let limit = (1u64 << cap) as f64;
            let w0 = limit / (ss.sqrt() * NORM_SLACK);
            for wnorm in [w0, f64::from_bits(w0.to_bits() + 1), f64::from_bits(w0.to_bits() - 1), w0 * (1.0 + 1e-7)] {
                let p = PrepareParams { k, a, cap, wnorm };
                check_case(&mut g, &x, n, p, &format!("boundary K{k}A{a}C{cap} b={b} n={n}"));
            }
        }
    }
}

/// Error precedence across rows (non-finite outranks an earlier cap overflow) and blocks.
#[test]
fn prepare_errors_like_reference() {
    let mut g = Rng(3);
    let n = 4096;
    let p = PrepareParams { k: 32, a: 12, cap: 27, wnorm: 3000.0 };
    let mut x: Vec<f32> = (0..24).flat_map(|_| gen_row(&mut g, n, 0)).collect();
    check_case(&mut g, &x, n, p, "parallel ok");
    x[5 * n + 3] = 1e30; // row 5: too large a norm? no: amax rescales; make overflow via wnorm below
    x[17 * n + 9] = f32::NAN;
    check_case(&mut g, &x, n, p, "nan in row 17");
    x[17 * n + 9] = 0.0;
    let p2 = PrepareParams { wnorm: 1e30, ..p };
    check_case(&mut g, &x, n, p2, "overflow everywhere");
}

/// Prefill-sized calls on the pool: the same bytes as one thread.
#[test]
fn prepare_parallel_blocks() {
    let mut g = Rng(4);
    for &(rows, n) in &[(24usize, 4096usize), (40, 896), (3, 18944)] {
        let x: Vec<f32> = (0..rows)
            .flat_map(|_| {
                let kd = g.below(8);
                gen_row(&mut g, n, kd)
            })
            .collect();
        for k in [32u32, 48] {
            let p = PrepareParams { k, a: 12, cap: 27, wnorm: 40.0 * (n as f64).sqrt() };
            check_case(&mut g, &x, n, p, &format!("parallel rows={rows} n={n} K{k}"));
        }
    }
}

/// The window has the public size and always contains `[lo, hi]`; for the 7B linears it is 2.
#[test]
fn window_covers_lo_hi() {
    let mut g = Rng(5);
    for _ in 0..300 {
        let a = 2 + g.below(47) as u32;
        let cap = 1 + g.below(62) as u32;
        let n = 1 + g.below(50_000) as usize;
        let wnorm = 10f64.powf(g.unit() * 14.0 - 3.0);
        let plan = WidthPlan::new(n, PrepareParams { k: 56, a: a.min(48), cap, wnorm });
        assert!(plan.w >= 1 && (plan.w < a.max(2) || a == 2));
        for _ in 0..200 {
            let tau = if g.next() & 1 == 0 { g.unit() * (n as f64).sqrt() } else { 10f64.powf(g.unit() * 8.0 - 4.0) };
            let start = plan.start(tau);
            assert!(start >= 2 && start + plan.w - 1 <= plan.a, "window out of range");
            if plan.bounded {
                let (lo, hi) = plan.lo_hi(tau);
                assert!(start <= lo, "window starts above lo");
                assert!(hi < start + plan.w, "window ends below hi: lo={lo} hi={hi} start={start} w={}", plan.w);
            }
        }
    }
    for (n, wnorm) in [(3584usize, 1500.0f64), (3584, 4000.0), (18944, 5000.0), (18944, 15000.0), (896, 400.0)] {
        let plan = WidthPlan::new(n, PrepareParams { k: 32, a: 12, cap: 27, wnorm });
        assert_eq!(plan.window(), 2, "n={n} wnorm={wnorm}");
    }
    let uncapped = WidthPlan::new(100, PrepareParams { k: 40, a: 12, cap: 0, wnorm: 0.0 });
    assert_eq!((uncapped.window(), uncapped.start(3.0)), (1, 12));
    let degenerate = WidthPlan::new(100, PrepareParams { k: 32, a: 12, cap: 27, wnorm: f64::NAN });
    assert_eq!((degenerate.window(), degenerate.start(3.0)), (11, 2));
}

// ---- finish ------------------------------------------------------------------------------------

struct FinCase {
    k: u32,
    rows: usize,
    m: usize,
    n: usize,
    c: usize,
    y: Vec<u8>,
    wr: Vec<u64>,
    xh: Vec<i64>,
    payload: Vec<u8>,
    u: Vec<u64>,
    uw: Vec<u64>,
    rhs: Vec<u64>,
    sx: Vec<f64>,
    sc: Vec<f32>,
    bias: Vec<f32>,
}

/// A genuine call: float rows prepared by the trip kernel, `y_hat = W x_hat`, `wr = W r`.
fn fin_case(g: &mut Rng, k: u32, rows: usize, m: usize, n: usize, c: usize) -> FinCase {
    let mask = u64::mask(k);
    let w: Vec<i64> = (0..m * n).map(|_| (g.below(255) as i64) - 127).collect();
    let wnorm =
        (0..m).map(|j| (0..n).map(|i| (w[j * n + i] * w[j * n + i]) as f64).sum::<f64>().sqrt()).fold(0.0, f64::max);
    let a = 12.min(k - 1);
    let cap = if k >= 24 { (k - 3).min(27) } else { 0 };
    let p = PrepareParams { k, a, cap, wnorm };
    let x: Vec<f32> = (0..rows)
        .flat_map(|_| {
            let kd = g.below(5);
            gen_row(g, n, kd)
        })
        .collect();
    let r: Vec<i64> = (0..rows * n).map(|_| (g.next() & mask) as i64).collect();
    let u: Vec<u64> = (0..c * m).map(|_| g.next() & mask).collect();
    let uw: Vec<u64> = (0..c * n)
        .map(|idx| {
            let (i, col) = (idx / n, idx % n);
            (0..m).fold(0u64, |s, j| s.wrapping_add(u[i * m + j].wrapping_mul(w[j * n + col] as u64))) & mask
        })
        .collect();
    let mut payload = vec![0u8; rows * n * (k as usize / 8)];
    let mut sx = vec![0.0; rows];
    let mut xh = vec![0i64; rows * n];
    let mut rhs = vec![0u64; rows * c];
    prepare_into(
        &x,
        &r,
        n,
        p,
        Some(InputHalf { uw: Src::U64(&uw), c }),
        PrepOut { payload: &mut payload, sx: &mut sx, xhat: &mut xh, rhs: &mut rhs },
        1,
    )
    .unwrap();
    let mut y = Vec::with_capacity(rows * m * (k as usize / 8));
    let mut wr = Vec::with_capacity(rows * m);
    for t in 0..rows {
        for j in 0..m {
            let (mut a1, mut b1) = (0u64, 0u64);
            for i in 0..n {
                let wv = w[j * n + i] as u64;
                a1 = a1.wrapping_add(wv.wrapping_mul(xh[t * n + i] as u64));
                b1 = b1.wrapping_add(wv.wrapping_mul(r[t * n + i] as u64));
            }
            y.extend_from_slice(&(a1 & mask).to_le_bytes()[..k as usize / 8]);
            wr.push(b1 & mask);
        }
    }
    let sc = (0..m).map(|j| 1e-3 * (j as f32 + 1.0)).collect();
    let bias = (0..m).map(|j| j as f32 * 0.25 - 1.0).collect();
    FinCase { k, rows, m, n, c, y, wr, xh, payload, u, uw, rhs, sx, sc, bias }
}

fn old_finish(cs: &FinCase, y: &[u8], u: &[u64], bound: u64) -> Finished {
    // SAFETY: read-only (`wipe = false`).
    let wr = unsafe { WrBuf::new(cs.wr.as_ptr() as *mut u8, cs.wr.len(), true, false) };
    let g = Group {
        parts: vec![Part { y_hat: y, wr, x_hat: Some(Src::Bytes(&cs.payload)), sx: &cs.sx }],
        checks: Some(GroupChecks { u: Src::U64(u), uw: Src::U64(&cs.uw), c: cs.c, n: cs.n }),
        scale: &cs.sc,
        bias: Some(&cs.bias),
        k: cs.k,
        bound,
        m: cs.m,
    };
    finish_group_run(&g, 1)
}

fn new_finish(cs: &FinCase, y: &[u8], u: &[u64], bound: u64, mode: u8, threads: usize) -> (Vec<f32>, u8, usize) {
    let mut wr = cs.wr.clone();
    // SAFETY: `wr` is owned here and outlives the call; wiped by the kernel.
    let wrb = unsafe { WrBuf::new(wr.as_mut_ptr().cast(), wr.len(), true, true) };
    let check = match mode {
        0 => CheckMode::Full { u: Src::U64(u), uw: Src::U64(&cs.uw), x_hat: Src::Bytes(&cs.payload), c: cs.c, n: cs.n },
        1 => CheckMode::Split { u: Src::U64(u), rhs: &cs.rhs, c: cs.c },
        _ => CheckMode::Off,
    };
    let args = FinishArgs { y_hat: y, wr: wrb, sx: &cs.sx, scale: &cs.sc, bias: Some(&cs.bias), k: cs.k, bound, check };
    let mut out = vec![f32::NAN; cs.rows * cs.m];
    let (st, bad) = finish_into(&args, &mut out, threads);
    assert!(wr.iter().all(|v| *v == 0), "wr not wiped");
    (out, st, bad)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// finish_into (full, split and unchecked) == finish_group_run: outputs, statuses, bad rows; the
/// output half alone gives the same verdict.
#[test]
fn finish_matches_existing_kernel_every_ring() {
    let mut g = Rng(0x5eed_0006);
    for &k in &RINGS {
        for &(rows, m, n, c) in &[(1usize, 37usize, 29usize, 1usize), (3, 130, 70, 3), (2, 4500, 9, 2)] {
            let cs = fin_case(&mut g, k, rows, m, n, c);
            let bound = u64::MAX >> 1;
            let (want, st, _) = old_finish(&cs, &cs.y, &cs.u, bound);
            assert_eq!(st, STATUS_OK, "K{k}");
            let want = bits(&want.unwrap());
            for mode in 0..3u8 {
                for threads in [1usize, 0] {
                    let (out, st, bad) = new_finish(&cs, &cs.y, &cs.u, bound, mode, threads);
                    assert_eq!((st, bad), (STATUS_OK, 0), "K{k} mode {mode}");
                    assert_eq!(bits(&out), want, "K{k} mode {mode} rows={rows} m={m}");
                }
            }
            assert_eq!(freivalds_output(&cs.y, Src::U64(&cs.u), &cs.rhs, c, m, k), (STATUS_OK, 0));
            // tamper with the last row: the same verdict everywhere
            let mut ybad = cs.y.clone();
            let at = (rows * m - 1) * (k as usize / 8);
            ybad[at] ^= 1;
            let old = old_finish(&cs, &ybad, &cs.u, bound);
            assert_eq!((old.1, old.2), (STATUS_FREIVALDS, 1), "K{k}");
            for mode in 0..2u8 {
                let (out, st, bad) = new_finish(&cs, &ybad, &cs.u, bound, mode, 1);
                assert_eq!((st, bad), (old.1, old.2), "K{k} mode {mode}");
                assert!(out.iter().all(|v| *v == 0.0), "output not zeroed after a failed check");
            }
            assert_eq!(freivalds_output(&ybad, Src::U64(&cs.u), &cs.rhs, c, m, k), (STATUS_FREIVALDS, 1));
            // all-zero u
            let uz = vec![0u64; c * m];
            assert_eq!(old_finish(&cs, &cs.y, &uz, bound).1, STATUS_U_ZERO);
            for mode in 0..2u8 {
                assert_eq!(new_finish(&cs, &cs.y, &uz, bound, mode, 1).1, STATUS_U_ZERO);
            }
            assert_eq!(freivalds_output(&cs.y, Src::U64(&uz), &cs.rhs, c, m, k).0, STATUS_U_ZERO);
            // range check at the largest |yq|
            let (_, st0, _) = old_finish(&cs, &cs.y, &cs.u, 0);
            for mode in 0..3u8 {
                assert_eq!(new_finish(&cs, &cs.y, &cs.u, 0, mode, 1).1, st0, "K{k} mode {mode} range");
            }
            assert!(st0 == STATUS_RANGE || st0 == STATUS_OK);
            let _ = &cs.xh;
        }
    }
}

/// Prefill-sized finish on the pool == scoped threads == one thread.
#[test]
fn finish_parallel_blocks() {
    let mut g = Rng(7);
    let cs = fin_case(&mut g, 32, 24, 9000, 16, 2);
    let (want, st, _) = old_finish(&cs, &cs.y, &cs.u, u64::MAX >> 1);
    assert_eq!(st, STATUS_OK);
    let want = bits(&want.unwrap());
    for mode in 0..3u8 {
        for threads in [0usize, 1, 3] {
            let (out, st, _) = new_finish(&cs, &cs.y, &cs.u, u64::MAX >> 1, mode, threads);
            assert_eq!(st, STATUS_OK);
            assert_eq!(bits(&out), want, "mode {mode} threads {threads}");
        }
    }
}

/// Reusing the same output buffers for different calls gives the per-call results (no state leaks
/// from one trip into the next).
#[test]
fn buffers_reused_across_calls() {
    let mut g = Rng(8);
    let p = PrepareParams { k: 32, a: 12, cap: 27, wnorm: 2500.0 };
    let n = 3584;
    let mut payload = vec![0u8; n * 4];
    let mut sx = vec![0.0; 1];
    for i in 0..20 {
        let x = gen_row(&mut g, n, i % 8);
        let r: Vec<i64> = (0..n).map(|_| (g.next() & 0xffff_ffff) as i64).collect();
        prepare_into(
            &x,
            &r,
            n,
            p,
            None,
            PrepOut { payload: &mut payload, sx: &mut sx, xhat: &mut [], rhs: &mut [] },
            1,
        )
        .unwrap();
        let (pl, s, _) = reference(&x, &r, n, p).unwrap();
        assert_eq!(payload, pl);
        assert_eq!(sx[0].to_bits(), s[0]);
    }
}

/// The summation path is public: `int_sum_ok` is monotone in the width (so `b <= b_int` is the
/// reference's choice), the default ring is integer-only, and large-A capped rings run both.
#[test]
fn sum_path_is_public_and_matches_reference_condition() {
    let mut g = Rng(9);
    for _ in 0..400 {
        let a = 2 + g.below(47) as u32;
        let cap = 1 + g.below(62) as u32;
        let n = [1usize, 7, 300, 3584, 18944, 1 << 20, 1 << 30][g.below(7) as usize];
        let plan = WidthPlan::new(n, PrepareParams { k: 56, a, cap, wnorm: 1.0 + g.unit() * 1e6 });
        for b in 2..=a {
            assert_eq!(int_sum_ok(n, qmax_of(b)), b <= plan.b_int, "n={n} A{a} b={b}");
        }
        let want = if plan.b_int >= a {
            SumPath::Int
        } else if plan.b_int < 2 {
            SumPath::Float
        } else {
            SumPath::Both
        };
        assert_eq!(plan.sum_path(), want);
    }
    for n in [3584usize, 18944] {
        let p = PrepareParams { k: 32, a: 12, cap: 27, wnorm: 4000.0 };
        assert_eq!(WidthPlan::new(n, p).sum_path(), SumPath::Int);
    }
    let big = PrepareParams { k: 56, a: 48, cap: 53, wnorm: (1u64 << 25) as f64 };
    assert_eq!(WidthPlan::new(18944, big).sum_path(), SumPath::Both);
    assert_eq!(WidthPlan::new(18944, PrepareParams { cap: 0, ..big }).sum_path(), SumPath::Off);
}

/// The width of a row from its scale (`amax / Q_b`).
fn width_of(x: &[f32], sx: u64) -> u32 {
    let amax = f64::from(f32::from_bits(row_amax_bits(x)));
    let s = f64::from_bits(sx);
    (2..=48).find(|&b| amax / qmax_of(b) == s).expect("scale of no width")
}

/// Regression (review): on a capped ring with a large A the reference sums squares as integers for
/// small widths and as sequential floats for large ones. A Gaussian row (small width) and a spike
/// row (large width) must run the same summation work, and stay bit-identical to the reference.
#[test]
fn large_a_rows_with_different_widths_run_the_same_path() {
    let mut g = Rng(10);
    let n = 18944;
    for wexp in [18u32, 22, 25, 28, 31] {
        let p = PrepareParams { k: 56, a: 48, cap: 53, wnorm: (1u64 << wexp) as f64 };
        let gauss = gen_row(&mut g, n, 0);
        let spike = gen_row(&mut g, n, 1);
        let r: Vec<i64> = (0..n).map(|_| (g.next() & u64::mask(56)) as i64).collect();
        let mut logs = Vec::new();
        let mut widths = Vec::new();
        for x in [&gauss, &spike] {
            PATH_LOG.with(|l| l.borrow_mut().clear());
            let got = run_trip(x, &r, n, p, None, 1).expect("fits");
            logs.push(PATH_LOG.with(|l| std::mem::take(&mut *l.borrow_mut())));
            widths.push(width_of(x, got.sx[0]));
            let (pl, sx, xh) = reference(x, &r, n, p).expect("reference fits");
            assert_eq!((got.sx, got.xhat, got.payload), (sx, xh, pl), "2^{wexp}: differs from the reference");
        }
        let w = WidthPlan::new(n, p).window() as usize;
        assert_eq!(logs[0], vec![SumPath::Both; w], "2^{wexp}");
        assert_eq!(logs[0], logs[1], "2^{wexp}: rows ran different summation paths");
        if wexp == 25 {
            // the case from the review: widths on both sides of b_int (25 at n = 18944)
            let b_int = WidthPlan::new(n, p).b_int;
            assert!(widths[0] <= b_int && widths[1] > b_int, "widths {widths:?}, b_int {b_int}");
        }
    }
    // windows straddling b_int: wnorm placed so that the Gaussian row's choice is b_int or b_int + 1
    for _ in 0..6 * stress() {
        let x = gen_row(&mut g, n, 0);
        let b = 24 + g.below(3) as u32;
        let qm = qmax_of(b);
        let amax = f64::from(f32::from_bits(row_amax_bits(&x)));
        let mut q = vec![0i64; n];
        quant_exact(&x, amax / qm, qm, &mut q);
        let w0 = (1u64 << 53) as f64 / (sum_squares(&q, qm).sqrt() * NORM_SLACK);
        for wnorm in [w0, f64::from_bits(w0.to_bits() + 1), w0 * 0.75] {
            let p = PrepareParams { k: 56, a: 48, cap: 53, wnorm };
            check_case(&mut g, &x, n, p, &format!("straddle b={b} wnorm={wnorm}"));
        }
    }
}

/// Status 3 (all-zero `u`) zeroes a reused output buffer like every other failure.
#[test]
fn finish_u_zero_clears_reused_output() {
    let mut g = Rng(11);
    let cs = fin_case(&mut g, 32, 2, 300, 40, 2);
    let uz = vec![0u64; cs.c * cs.m];
    for mode in 0..2u8 {
        let mut wr = cs.wr.clone();
        // SAFETY: `wr` is owned here and outlives the call.
        let wrb = unsafe { WrBuf::new(wr.as_mut_ptr().cast(), wr.len(), true, true) };
        let check = match mode {
            0 => CheckMode::Full {
                u: Src::U64(&uz),
                uw: Src::U64(&cs.uw),
                x_hat: Src::Bytes(&cs.payload),
                c: cs.c,
                n: cs.n,
            },
            _ => CheckMode::Split { u: Src::U64(&uz), rhs: &cs.rhs, c: cs.c },
        };
        let args = FinishArgs {
            y_hat: &cs.y,
            wr: wrb,
            sx: &cs.sx,
            scale: &cs.sc,
            bias: Some(&cs.bias),
            k: cs.k,
            bound: u64::MAX >> 1,
            check,
        };
        let mut out = vec![1.5f32; cs.rows * cs.m]; // the previous trip's outputs
        assert_eq!(finish_into(&args, &mut out, 1).0, STATUS_U_ZERO);
        assert!(out.iter().all(|v| v.to_bits() == 0), "mode {mode}: stale outputs survived status 3");
        assert!(wr.iter().all(|v| *v == 0), "wr not wiped");
    }
}
