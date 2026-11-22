//! Row-deterministic attention over the KV cache (`float_ops = "native"`, opt-in; see [`crate::layer`]).
//!
//! Each output row depends only on that row's query and the cache prefix it sees: the same bits
//! whatever the chunk size, the thread count, the instruction set or the number of rows in the pass
//! (so a speculative verify row equals plain decoding by construction). Per query head `h` (KV head
//! `h / groups`) and row at position `p`:
//!
//! * `s_j = dot(q, k_j) * scale` for `j = 0..=p`: the dot in float32 (16 lane sums over `d mod 16`, then a
//!   fixed tree), widened to float64 and scaled;
//! * `m = max_j s_j`, `e_j = exp(s_j - m)` in float64 ([`crate::layer::exp_f64`]) rounded to float32,
//!   `den = sum_j e_j` in float64 (`j` order);
//! * `o_d = (sum_j e_j * v_jd) / den`: the sum in float32 (`j` order), the quotient in float64, rounded.
//!
//! [`float_rope_attend`] also applies RoPE (torch's `apply_rope` bits: `x * cos + cat(x2, x1) *
//! cat(-sin1, sin2)`, two float32 products and a sum) to q and k of a `qkv` result and writes k and v
//! into the cache at the pass's positions before attending: one call per layer between the `qkv` and
//! `o` linears.

use crate::layer::{exp_f64, isa_dispatch, verr};
use numpy::{PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(float_rope_attend, m)?)?;
    m.add_function(wrap_pyfunction!(float_attend, m)?)?;
    Ok(())
}

/// `dot(a, b)` in float32: 16 lane sums (`d mod 16`, in order), then a fixed pairwise tree.
#[inline(always)]
fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut l = [0f32; 16];
    let n = a.len().min(b.len());
    let n16 = n / 16 * 16;
    let (a16, b16) = (&a[..n16], &b[..n16]);
    for (ca, cb) in a16.chunks_exact(16).zip(b16.chunks_exact(16)) {
        for j in 0..16 {
            l[j] += ca[j] * cb[j];
        }
    }
    for (j, idx) in (n16..n).enumerate() {
        l[j] += a[idx] * b[idx];
    }
    let mut w = 16;
    while w > 1 {
        w /= 2;
        for j in 0..w {
            l[j] += l[j + w];
        }
    }
    l[0]
}

/// Shapes of one attention problem.
#[derive(Clone, Copy)]
pub struct Dims {
    pub nh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub scale: f64,
}

/// Attention of one query row `q (nh * hd)` over `len` positions of `k`, `v` (`(len, kvh, hd)`, row
/// stride `kvh * hd`) into `out (nh * hd)`.
///
/// Per query head: `s_j = (dot32(q, k_j) as f64) * scale`; `m = max_j s_j`; `e_j = exp(s_j - m)` in
/// float64, rounded to float32; `den = sum_j e_j` in float64 (`j` order); `acc_d = sum_j e_j * v_jd` in
/// float32 (`j` order); `o_d = (acc_d / den) as f32` in float64.
#[inline(always)]
pub fn attend_row(q: &[f32], k: &[f32], v: &[f32], len: usize, dm: Dims, out: &mut [f32]) {
    let Dims { nh, kvh, hd, scale } = dm;
    let groups = nh / kvh;
    let stride = kvh * hd;
    let mut s = vec![0f64; len];
    let mut e = vec![0f32; groups * len];
    let mut den = vec![0f64; groups];
    let mut acc = vec![0f32; groups * hd];
    for g in 0..kvh {
        for h in 0..groups {
            let qh = &q[(g * groups + h) * hd..(g * groups + h + 1) * hd];
            for j in 0..len {
                s[j] = dot32(qh, &k[j * stride + g * hd..j * stride + (g + 1) * hd]) as f64 * scale;
            }
            let mut m = f64::NEG_INFINITY;
            for &x in s.iter() {
                m = if x > m { x } else { m };
            }
            let eh = &mut e[h * len..(h + 1) * len];
            for j in 0..len {
                eh[j] = exp_f64(s[j] - m) as f32;
            }
            let mut d = 0f64;
            for &x in eh.iter() {
                d += x as f64;
            }
            den[h] = d;
        }
        acc.iter_mut().for_each(|a| *a = 0.0);
        for j in 0..len {
            let vj = &v[j * stride + g * hd..j * stride + (g + 1) * hd];
            for h in 0..groups {
                let ej = e[h * len + j];
                let a = &mut acc[h * hd..(h + 1) * hd];
                for (ad, &vd) in a.iter_mut().zip(vj) {
                    *ad += ej * vd;
                }
            }
        }
        for h in 0..groups {
            let o = &mut out[(g * groups + h) * hd..(g * groups + h + 1) * hd];
            for d in 0..hd {
                o[d] = (acc[h * hd + d] as f64 / den[h]) as f32;
            }
        }
    }
}

#[inline(always)]
fn attend_chunk_body(q: &[f32], k: &[f32], v: &[f32], t0: usize, start: usize, dm: Dims, o: &mut [f32]) {
    let w = dm.nh * dm.hd;
    for (i, orow) in o.chunks_mut(w).enumerate() {
        let t = t0 + i;
        attend_row(&q[t * w..(t + 1) * w], k, v, start + t + 1, dm, orow);
    }
}
isa_dispatch!(fn attend_chunk => attend_chunk_body(q: &[f32], k: &[f32], v: &[f32], t0: usize, start: usize,
                                                   dm: Dims, o: &mut [f32]));

/// Rows `t` of `q (rows, nh * hd)` at positions `start + t` attend to `k`, `v` positions `[0, start + t]`.
/// Rows are split over `threads` scoped threads (0 or 1: the calling thread); each row's bits are the
/// same either way.
#[allow(clippy::too_many_arguments)]
pub fn attend_rows(q: &[f32], k: &[f32], v: &[f32], rows: usize, start: usize, dm: Dims, out: &mut [f32],
                   threads: usize) {
    let w = dm.nh * dm.hd;
    let run = |t0: usize, o: &mut [f32]| {
        attend_chunk(q, k, v, t0, start, dm, o);
    };
    let threads = threads.max(1).min(rows.max(1));
    if threads <= 1 {
        run(0, out);
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|sc| {
        for (c, o) in out.chunks_mut(per * w).enumerate() {
            let run = &run;
            sc.spawn(move || run(c * per, o));
        }
    });
}

/// Threads for a pass: one for decode-sized work, more for long prompts (bits do not depend on it).
fn auto_threads(rows: usize, start: usize, threads: usize) -> usize {
    if threads > 0 {
        return threads;
    }
    let work = rows * (start + rows);
    if rows < 4 || work < 1 << 14 {
        1
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8)
    }
}

/// RoPE on one head vector into `out`: `x * cos + cat(x2, x1) * cat(-sin1, sin2)` (float32).
#[inline]
fn rope_head(x: &[f32], cos: &[f32], sin: &[f32], out: &mut [f32]) {
    let h = x.len() / 2;
    for d in 0..h {
        out[d] = x[d] * cos[d] + x[d + h] * -sin[d];
    }
    for d in h..2 * h {
        out[d] = x[d] * cos[d] + x[d - h] * sin[d];
    }
}

/// One layer's attention block of a pass: RoPE on q and k of `y (rows, (nh + 2 kvh) hd)` (columns
/// `[q | k | v]`), k and v written into `kc`, `vc` (`(layers, cap, kvh, hd)`) at `[layer, start + t]`,
/// then row-deterministic causal attention into `out (rows, nh * hd)`. `threads`: 0 = automatic.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (y, cos, sin, kc, vc, layer, start, nh, scale, out, threads = 0))]
fn float_rope_attend<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyArray2<f32>>,
    cos: &Bound<'py, PyArray2<f32>>,
    sin: &Bound<'py, PyArray2<f32>>,
    kc: &Bound<'py, PyArray4<f32>>,
    vc: &Bound<'py, PyArray4<f32>>,
    layer: usize,
    start: usize,
    nh: usize,
    scale: f64,
    out: &Bound<'py, PyArray2<f32>>,
    threads: usize,
) -> PyResult<()> {
    let ks = kc.shape().to_vec();
    let (layers, cap, kvh, hd) = (ks[0], ks[1], ks[2], ks[3]);
    let rows = y.shape()[0];
    let ycols = y.shape()[1];
    if vc.shape() != ks.as_slice() || layer >= layers || kvh == 0 || nh % kvh != 0 || hd % 2 != 0
        || ycols != (nh + 2 * kvh) * hd || cos.shape() != [rows, hd] || sin.shape() != [rows, hd]
        || out.shape() != [rows, nh * hd] || start + rows > cap
    {
        return Err(verr("float_rope_attend: shapes"));
    }
    for (a, n) in [(y.is_c_contiguous(), "y"), (cos.is_c_contiguous(), "cos"), (sin.is_c_contiguous(), "sin"),
                   (kc.is_c_contiguous(), "kc"), (vc.is_c_contiguous(), "vc"), (out.is_c_contiguous(), "out")] {
        if !a {
            return Err(verr(format!("float_rope_attend: {n} must be C-contiguous float32")));
        }
    }
    let yr = y.try_readonly().map_err(|e| verr(format!("y: {e}")))?;
    let cr = cos.try_readonly().map_err(|e| verr(format!("cos: {e}")))?;
    let sr = sin.try_readonly().map_err(|e| verr(format!("sin: {e}")))?;
    let mut kw = kc.try_readwrite().map_err(|e| verr(format!("kc: {e}")))?;
    let mut vw = vc.try_readwrite().map_err(|e| verr(format!("vc: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (ys, cs, ss) = (yr.as_slice()?, cr.as_slice()?, sr.as_slice()?);
    let (kall, vall, os) = (kw.as_slice_mut()?, vw.as_slice_mut()?, ow.as_slice_mut()?);
    let dm = Dims { nh, kvh, hd, scale };
    let threads = auto_threads(rows, start, threads);
    py.allow_threads(|| {
        let lsz = cap * kvh * hd;
        let kl = &mut kall[layer * lsz..(layer + 1) * lsz];
        let vl = &mut vall[layer * lsz..(layer + 1) * lsz];
        let stride = kvh * hd;
        let mut q = vec![0f32; rows * nh * hd];
        for t in 0..rows {
            let yt = &ys[t * ycols..(t + 1) * ycols];
            let (c, s) = (&cs[t * hd..(t + 1) * hd], &ss[t * hd..(t + 1) * hd]);
            for h in 0..nh {
                rope_head(&yt[h * hd..(h + 1) * hd], c, s, &mut q[(t * nh + h) * hd..(t * nh + h + 1) * hd]);
            }
            let p = start + t;
            for g in 0..kvh {
                let src = &yt[(nh + g) * hd..(nh + g + 1) * hd];
                rope_head(src, c, s, &mut kl[p * stride + g * hd..p * stride + (g + 1) * hd]);
                let vs = &yt[(nh + kvh + g) * hd..(nh + kvh + g + 1) * hd];
                vl[p * stride + g * hd..p * stride + (g + 1) * hd].copy_from_slice(vs);
            }
        }
        let len = start + rows;
        attend_rows(&q, &kl[..len * stride], &vl[..len * stride], rows, start, dm, os, threads);
    });
    Ok(())
}

/// Row-deterministic causal attention of `q (rows, nh, hd)` over `k`, `v` `(len, kvh, hd)`; row `t` sees
/// positions `[0, start + t]`; `out (rows, nh * hd)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (q, k, v, start, scale, out, threads = 0))]
fn float_attend<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyArray3<f32>>,
    k: &Bound<'py, PyArray3<f32>>,
    v: &Bound<'py, PyArray3<f32>>,
    start: usize,
    scale: f64,
    out: &Bound<'py, PyArray2<f32>>,
    threads: usize,
) -> PyResult<()> {
    let (rows, nh, hd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (len, kvh) = (k.shape()[0], k.shape()[1]);
    if v.shape() != k.shape() || k.shape()[2] != hd || kvh == 0 || nh % kvh != 0 || start + rows > len
        || out.shape() != [rows, nh * hd]
    {
        return Err(verr("float_attend: shapes"));
    }
    if !(q.is_c_contiguous() && k.is_c_contiguous() && v.is_c_contiguous() && out.is_c_contiguous()) {
        return Err(verr("float_attend: arrays must be C-contiguous float32"));
    }
    let qr = q.try_readonly().map_err(|e| verr(format!("q: {e}")))?;
    let kr = k.try_readonly().map_err(|e| verr(format!("k: {e}")))?;
    let vr = v.try_readonly().map_err(|e| verr(format!("v: {e}")))?;
    let mut ow = out.try_readwrite().map_err(|e| verr(format!("out: {e}")))?;
    let (qs, ks, vs) = (qr.as_slice()?, kr.as_slice()?, vr.as_slice()?);
    let os = ow.as_slice_mut()?;
    let dm = Dims { nh, kvh, hd, scale };
    let threads = auto_threads(rows, start, threads);
    py.allow_threads(|| attend_rows(qs, ks, vs, rows, start, dm, os, threads));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
            })
            .collect()
    }

    #[test]
    fn rows_threads_and_blocks_agree_and_match_naive() {
        let dm = Dims { nh: 14, kvh: 2, hd: 64, scale: 0.125 };
        let (len, w, st) = (37usize, 14 * 64, 2 * 64);
        let q = data(len * w, 1);
        let k = data(len * st, 2);
        let v = data(len * st, 3);
        // whole block from 0 on 1 and 3 threads
        let mut a = vec![0f32; len * w];
        let mut b = vec![0f32; len * w];
        attend_rows(&q, &k, &v, len, 0, dm, &mut a, 1);
        attend_rows(&q, &k, &v, len, 0, dm, &mut b, 3);
        assert_eq!(a, b);
        // each row alone (decode) equals its row in the block
        for t in 0..len {
            let mut o = vec![0f32; w];
            attend_rows(&q[t * w..(t + 1) * w], &k[..(t + 1) * st], &v[..(t + 1) * st], 1, t, dm, &mut o, 1);
            assert_eq!(&o[..], &a[t * w..(t + 1) * w], "row {t}");
        }
        // close to a naive float64 softmax
        for t in [0usize, 5, 36] {
            for h in 0..14 {
                let g = h / 7;
                let sc: Vec<f64> = (0..=t)
                    .map(|j| (0..64).map(|d| q[t * w + h * 64 + d] as f64 * k[j * st + g * 64 + d] as f64).sum::<f64>()
                        * 0.125)
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let den: f64 = e.iter().sum();
                for d in 0..64 {
                    let want: f64 = (0..=t).map(|j| e[j] * v[j * st + g * 64 + d] as f64).sum::<f64>() / den;
                    let got = a[t * w + h * 64 + d] as f64;
                    assert!((got - want).abs() <= 1e-5 * want.abs().max(1e-1), "t={t} h={h} d={d} {got} {want}");
                }
            }
        }
    }

    #[test]
    fn rope_matches_reference_form() {
        let x = data(64, 9);
        let c = data(64, 10);
        let s = data(64, 11);
        let mut o = vec![0f32; 64];
        rope_head(&x, &c, &s, &mut o);
        for d in 0..64 {
            let rot = if d < 32 { -x[d + 32] } else { x[d - 32] };
            assert_eq!(o[d], x[d] * c[d] + rot * s[d]);
        }
    }
}
