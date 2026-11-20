//! Lane-interleaved NTT kernels and the block driver.
//!
//! Rows of the big operand are processed in blocks of [`L`] `= 16`: coefficient `j` of polynomial
//! `l` of all 16 rows sits in one `[u32; 16]` ("lanes"), so every butterfly, multiply-accumulate and
//! CRT step is the same operation on 16 independent rows. The loops are written over lanes with
//! fixed trip counts and compiled three times (AVX-512, AVX2, baseline x86-64 / portable), picked at
//! run time ([`Isa`]); LLVM turns the lane loops into `vpmuludq`/`vpmulld`/`vpminud` vectors. Every
//! instruction set gives the same bits (all arithmetic is exact integer arithmetic).
//!
//! Per block of rows and per prime: residues of the big operand (one `redc` per word), `rank`
//! forward NTTs per row (skipped when the operand was transformed once with [`transform`]), then for
//! every secret: `rank` pointwise multiply-accumulates per coefficient, one inverse NTT, and CRT +
//! reduction mod `2^64` into the output row.

use crate::arith::{params, Prime, MAC_CHUNK, N, NP};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Rows per block (vector lanes).
pub const L: usize = 16;
/// Secrets whose accumulators are held in registers together.
pub const GB: usize = 4;
/// Coefficients per multiply-accumulate tile (`rank * JB` lanes stay in L1 across all secrets).
pub const JB: usize = 16;

/// 16 lanes (one AVX-512 register).
pub(crate) type Lanes = [u32; L];
/// Columns converted per tile in [`load_residues`] (keeps the destination lines in L1).
const TILE: usize = 64;

/// The three hot kernels, per instruction set: [`Portable`] (auto-vectorised lane loops, compiled
/// for baseline x86-64 and for AVX2) and `Avx512` (intrinsics, `crate::avx512`).
pub(crate) trait Kern {
    fn ntt_fwd(a: &mut [Lanes], pr: &Prime);
    fn ntt_inv(a: &mut [Lanes], pr: &Prime);
    #[allow(clippy::too_many_arguments)]
    fn mac(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, pr: &Prime, out: &mut [Lanes]);
    /// Residues of 16 rows into lanes, `[0, 2p)` (`tmp`: `NP * L * TILE` words of scratch).
    fn load_residues(big: &Big, row0: usize, dim: usize, cs: usize, ce: usize, out: &mut [Lanes], tmp: &mut [u32]);
}

pub(crate) struct Portable;

impl Kern for Portable {
    #[inline(always)]
    fn ntt_fwd(a: &mut [Lanes], pr: &Prime) {
        ntt_fwd(a, pr)
    }
    #[inline(always)]
    fn ntt_inv(a: &mut [Lanes], pr: &Prime) {
        ntt_inv(a, pr)
    }
    #[inline(always)]
    fn mac(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, pr: &Prime, out: &mut [Lanes]) {
        mac(a, sh, rank, gp, pr, out)
    }
    #[inline(always)]
    fn load_residues(big: &Big, row0: usize, dim: usize, cs: usize, ce: usize, out: &mut [Lanes], tmp: &mut [u32]) {
        load_residues(big, row0, dim, cs, ce, out, tmp)
    }
}

#[inline(always)]
pub(crate) fn redc(t: u64, p: u32, pn: u32) -> u32 {
    let m = (t as u32).wrapping_mul(pn);
    (t.wrapping_add((m as u64).wrapping_mul(p as u64)) >> 32) as u32
}

#[inline(always)]
pub(crate) fn mmul(a: u32, b: u32, p: u32, pn: u32) -> u32 {
    redc((a as u64).wrapping_mul(b as u64), p, pn)
}

/// `x mod m` for `x < 2m` (`min` with the wrapped difference).
#[inline(always)]
pub(crate) fn red(x: u32, m: u32) -> u32 {
    x.min(x.wrapping_sub(m))
}

/// Forward negacyclic NTT (Cooley-Tukey, bit-reversed output) of 16 polynomials at once.
/// Input in `[0, 4p)`; output in `[0, p)`.
#[inline(always)]
pub(crate) fn ntt_fwd(a: &mut [Lanes], pr: &Prime) {
    let (p, pn) = (pr.p, pr.pn);
    let p2 = 2 * p;
    // Harvey's lazy butterflies: values in [0, 4p) (4p < 2^31), one reduction of x per butterfly
    let mut k = 0usize;
    let mut len = N / 2;
    while len > 0 {
        let last = len == 1;
        let mut start = 0;
        while start < N {
            k += 1;
            let z = pr.zetas[k];
            for j in start..start + len {
                let mut x = a[j];
                let mut y = a[j + len];
                for i in 0..L {
                    let u = red(x[i], p2);
                    let t = mmul(y[i], z, p, pn);
                    let (s, d) = (u.wrapping_add(t), u.wrapping_add(p2).wrapping_sub(t));
                    if last {
                        x[i] = red(red(s, p2), p);
                        y[i] = red(red(d, p2), p);
                    } else {
                        x[i] = s;
                        y[i] = d;
                    }
                }
                a[j] = x;
                a[j + len] = y;
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Inverse negacyclic NTT (Gentleman-Sande) without the final `1/256` (folded into the secret).
/// Input and output in `[0, 2p)`.
#[inline(always)]
pub(crate) fn ntt_inv(a: &mut [Lanes], pr: &Prime) {
    let (p, pn) = (pr.p, pr.pn);
    let p2 = 2 * p;
    let mut k = N;
    let mut len = 1;
    while len < N {
        let mut start = 0;
        while start < N {
            k -= 1;
            let z = pr.izetas[k];
            for j in start..start + len {
                let mut x = a[j];
                let mut y = a[j + len];
                for i in 0..L {
                    let u = x[i];
                    let v = y[i];
                    x[i] = red(u.wrapping_add(v), p2);
                    y[i] = mmul(u.wrapping_add(p2).wrapping_sub(v), z, p, pn);
                }
                a[j] = x;
                a[j + len] = y;
            }
            start += 2 * len;
        }
        len <<= 1;
    }
}

/// The big operand of a product.
#[derive(Clone, Copy)]
pub enum Big<'a> {
    /// `(rows, dim)` 64-bit words.
    U64 { data: &'a [u64], rows: usize },
    /// `(rows, npl, dim)` byte planes (`byte ^ 0x80` as int8) of bytes `first .. first + npl`.
    Planes { data: &'a [i8], rows: usize, npl: usize, first: usize },
    /// Already transformed by [`transform`] (layout `[block][prime][l][j][lane]`).
    Pre { data: &'a [u32], rows: usize, bits: u32 },
}

impl Big<'_> {
    pub fn rows(&self) -> usize {
        match *self {
            Big::U64 { rows, .. } | Big::Planes { rows, .. } | Big::Pre { rows, .. } => rows,
        }
    }

    /// Bits of the largest value (for the exactness bound).
    pub fn bits(&self) -> u32 {
        match *self {
            Big::U64 { .. } => 64,
            Big::Pre { bits, .. } => bits,
            Big::Planes { npl, first, .. } => 8 * (npl + first) as u32,
        }
    }
}

/// Residues of rows `row0 .. row0 + L` (zero past `rows`) into `out` (`[prime][l][j][lane]`),
/// values in `[0, 2p)`; `tmp` holds `NP * L * TILE` words. Converts `TILE` columns of all 16 rows
/// at a time (vectorised along the columns), then transposes them into lanes.
#[inline(always)]
fn load_residues(big: &Big, row0: usize, dim: usize, cs: usize, ce: usize, out: &mut [Lanes], tmp: &mut [u32]) {
    let ps = &params().primes;
    let rows = big.rows();
    let nrows = rows.saturating_sub(row0).min(L);
    let mut acc = [0u64; TILE];
    for c0 in (cs..ce).step_by(TILE) {
        let w = TILE.min(ce - c0);
        for lane in 0..L {
            let r = row0 + lane;
            for (pi, pr) in ps.iter().enumerate() {
                let (p, pn) = (pr.p, pr.pn);
                let t = &mut tmp[(pi * L + lane) * TILE..(pi * L + lane) * TILE + TILE];
                if lane >= nrows {
                    t.fill(0);
                    continue;
                }
                match *big {
                    Big::U64 { data, .. } => {
                        let row = &data[r * dim + c0..r * dim + c0 + w];
                        let (cl, ch) = (pr.c_lo as u64, pr.c_hi as u64);
                        for c in 0..w {
                            let v = row[c];
                            // lo * (R mod p) + hi * (2^32 R mod p) < 2^33 p; redc < 3p; one min -> [0, 2p)
                            let x = (v & 0xFFFF_FFFF).wrapping_mul(cl).wrapping_add((v >> 32).wrapping_mul(ch));
                            t[c] = red(redc(x, p, pn), 2 * p);
                        }
                    }
                    Big::Planes { data, npl, first, .. } => {
                        acc[..w].fill(0);
                        for d in 0..npl {
                            let cb = pr.c_byte[first + d] as u64;
                            let pl = &data[(r * npl + d) * dim + c0..(r * npl + d) * dim + c0 + w];
                            for c in 0..w {
                                // sum of <= 8 terms < 2^8 p: below 2^11 p < R p, so redc < 2p
                                acc[c] = acc[c].wrapping_add((((pl[c] as u8) ^ 0x80) as u64).wrapping_mul(cb));
                            }
                        }
                        for c in 0..w {
                            t[c] = redc(acc[c], p, pn);
                        }
                    }
                    Big::Pre { .. } => unreachable!(),
                }
            }
        }
        for pi in 0..NP {
            let src = &tmp[pi * L * TILE..(pi + 1) * L * TILE];
            let dst = &mut out[pi * dim + c0..pi * dim + c0 + w];
            for (c, d) in dst.iter_mut().enumerate() {
                for (lane, x) in d.iter_mut().enumerate() {
                    *x = src[lane * TILE + c];
                }
            }
        }
    }
}

/// Residues + forward NTTs of one block (`out`: `NP * rank * N` lanes, values in `[0, p)`).
#[inline(always)]
fn transform_block<K: Kern>(big: &Big, row0: usize, rank: usize, out: &mut [Lanes], tmp: &mut [u32]) {
    let dim = rank * N;
    let ps = &params().primes;
    for l in 0..rank {
        // one polynomial at a time, so its residues are still in L1/L2 for the NTTs
        K::load_residues(big, row0, dim, l * N, (l + 1) * N, out, tmp);
        for (pi, pr) in ps.iter().enumerate() {
            let o = (pi * rank + l) * N;
            K::ntt_fwd(&mut out[o..o + N], pr);
        }
    }
}

/// Overwrites secret-derived memory with zeros. Volatile stores plus a compiler fence, so the
/// writes are not elided as dead stores before the memory is freed.
pub fn wipe<T: Copy + Default>(v: &mut [T]) {
    for x in v.iter_mut() {
        // SAFETY: `x` is a valid, aligned, exclusive reference.
        unsafe { std::ptr::write_volatile(x, T::default()) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// A buffer of secret material that is wiped (see [`wipe`]) when dropped.
pub struct Wiping<T: Copy + Default>(pub Vec<T>);

impl<T: Copy + Default> Wiping<T> {
    /// The contents, handing the wiping duty to the caller.
    pub fn into_inner(mut self) -> Vec<T> {
        std::mem::take(&mut self.0)
    }
}

impl<T: Copy + Default> Drop for Wiping<T> {
    fn drop(&mut self) {
        wipe(&mut self.0);
    }
}

impl<T: Copy + Default> std::ops::Deref for Wiping<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Copy + Default> std::ops::DerefMut for Wiping<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// Secrets in NTT form: `shat[prime][(j * rank + l) * gp + g]` (scaled by `R / 256`, `< p`).
/// The transformed secrets are wiped when dropped.
pub struct Secrets {
    pub g: usize,
    pub gp: usize,
    pub rank: usize,
    pub l1: u64,
    pub(crate) shat: Wiping<u32>,
}

impl Secrets {
    /// `s`: `(g, rank, 256)` signed coefficients.
    pub fn new(s: &[i8], g: usize, rank: usize) -> Secrets {
        assert_eq!(s.len(), g * rank * N);
        let gp = g.div_ceil(GB) * GB;
        let mut l1 = 0u64;
        for gi in 0..g {
            let n: u64 = s[gi * rank * N..(gi + 1) * rank * N].iter().map(|&v| (v as i64).unsigned_abs()).sum();
            l1 = l1.max(n);
        }
        let ps = &params().primes;
        let mut shat = Wiping(vec![0u32; NP * N * rank * gp]);
        let polys = g * rank;
        let mut buf = Wiping(vec![[0u32; L]; N]);
        for (pi, pr) in ps.iter().enumerate() {
            let (p, pn) = (pr.p, pr.pn);
            let sh = &mut shat[pi * N * rank * gp..(pi + 1) * N * rank * gp];
            for q0 in (0..polys).step_by(L) {
                for (j, v) in buf.iter_mut().enumerate() {
                    for (lane, x) in v.iter_mut().enumerate() {
                        let q = q0 + lane;
                        *x = if q < polys {
                            let c = s[q * N + j] as i64;
                            (if c < 0 { c + p as i64 } else { c }) as u32
                        } else {
                            0
                        };
                    }
                }
                ntt_fwd(&mut buf, pr);
                for (j, v) in buf.iter().enumerate() {
                    for (lane, &x) in v.iter().enumerate() {
                        let q = q0 + lane;
                        if q < polys {
                            let (gi, l) = (q / rank, q % rank);
                            sh[(j * rank + l) * gp + gi] = red(mmul(x, pr.sscale, p, pn), p);
                        }
                    }
                }
            }
        }
        drop(buf);
        Secrets { g, gp, rank, l1, shat }
    }
}

/// Pointwise multiply-accumulate of one prime for all `gp` secrets: `out[g][j] = sum_l a[l][j] *
/// s[g][l][j]` in `[0, 2p)`. Coefficients are taken [`JB`] at a time and every secret is applied
/// to them while they are in L1, so a block's transformed rows are read from memory once per call.
#[inline(always)]
pub(crate) fn mac(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, pr: &Prime, out: &mut [Lanes]) {
    for jb in (0..N).step_by(JB) {
        for g0 in (0..gp).step_by(GB) {
            mac_part(a, sh, rank, gp, g0, jb, pr, out);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn mac_part(a: &[Lanes], sh: &[u32], rank: usize, gp: usize, g0: usize, jb: usize, pr: &Prime, out: &mut [Lanes]) {
    let (p, pn) = (pr.p, pr.pn);
    let (p2, p4) = (2 * p, 4 * p);
    for j in jb..jb + JB {
        let mut part = [[0u32; L]; GB];
        let mut l0 = 0;
        while l0 < rank {
            let l1 = (l0 + MAC_CHUNK).min(rank);
            let mut acc = [[0u64; L]; GB];
            for l in l0..l1 {
                let av = &a[l * N + j];
                let sb = &sh[(j * rank + l) * gp + g0..(j * rank + l) * gp + g0 + GB];
                for g in 0..GB {
                    let s = sb[g] as u64;
                    for i in 0..L {
                        acc[g][i] = acc[g][i].wrapping_add((av[i] as u64).wrapping_mul(s));
                    }
                }
            }
            for g in 0..GB {
                for i in 0..L {
                    let u = red(red(redc(acc[g][i], p, pn), p4), p2);
                    part[g][i] = if l0 == 0 { u } else { red(part[g][i].wrapping_add(u), p2) };
                }
            }
            l0 = l1;
        }
        for g in 0..GB {
            out[(g0 + g) * N + j] = part[g];
        }
    }
}

#[derive(Clone, Copy)]
struct OutPtr(*mut u64);
#[derive(Clone, Copy)]
struct OutPtr32(*mut u32);
// SAFETY: workers write disjoint rows / blocks.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}
unsafe impl Send for OutPtr32 {}
unsafe impl Sync for OutPtr32 {}

/// CRT of three residue polynomials (each `[0, 2p_i)`) -> exact integer mod 2^64, written to rows
/// `row0 .. row0 + nrows` at columns `col0 .. col0 + 256` (row stride `stride`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn crt_write(r: &[Lanes], g: usize, gp: usize, out: OutPtr, row0: usize, nrows: usize, stride: usize, col0: usize) {
    let prm = params();
    let [q0, q1, q2] = &prm.primes;
    let c = &prm.crt;
    let (p0, p1, p2) = (q0.p, q1.p, q2.p);
    let half2 = (p2 - 1) / 2;
    let o0 = g * N;
    let o1 = (gp + g) * N;
    let o2 = (2 * gp + g) * N;
    // results of all 16 rows first, then each row's 256 words written contiguously (a lane-major
    // scatter would touch 16 output pages per coefficient)
    let mut buf = [[0u64; L]; N];
    for (j, x) in buf.iter_mut().enumerate() {
        let (v0, v1, v2) = (r[o0 + j], r[o1 + j], r[o2 + j]);
        for i in 0..L {
            let a0 = red(v0[i], p0);
            let a1 = red(v1[i], p1);
            let a2 = red(v2[i], p2);
            let t1 = red(mmul(a1.wrapping_add(p1).wrapping_sub(red(a0, p1)), c.inv01, p1, q1.pn), p1);
            let u = mmul(t1, c.p0_m2, p2, q2.pn);
            let v = red(red(red(a0, p2).wrapping_add(u), 2 * p2), p2);
            let t2 = red(mmul(a2.wrapping_add(p2).wrapping_sub(v), c.inv012, p2, q2.pn), p2);
            // centred last digit: t2 - p2 (as a wrapped 64-bit word) when t2 > (p2 - 1) / 2
            let t2s = if t2 > half2 { (t2 as u64).wrapping_sub(p2 as u64) } else { t2 as u64 };
            x[i] = (a0 as u64).wrapping_add((p0 as u64).wrapping_mul(t1 as u64)).wrapping_add(c.p01.wrapping_mul(t2s));
        }
    }
    for i in 0..nrows {
        // SAFETY: row row0 + i, columns col0 .. col0 + 256 are inside the output and owned by this block.
        let row = unsafe { std::slice::from_raw_parts_mut(out.0.add((row0 + i) * stride + col0), N) };
        for (o, x) in row.iter_mut().zip(buf.iter()) {
            *o = x[i];
        }
    }
}

/// Scratch buffers of one worker.
struct Scratch {
    ahat: Vec<Lanes>,
    tmp: Vec<u32>,
    res: Vec<Lanes>,
}

impl Scratch {
    fn new(rank: usize) -> Scratch {
        Scratch { ahat: vec![[0; L]; NP * rank * N], tmp: vec![0; NP * L * TILE], res: vec![] }
    }
}

/// One block of rows: products with every secret, written to `out`.
#[inline(always)]
fn product_block<K: Kern>(big: &Big, sec: &Secrets, blk: usize, sc: &mut Scratch, out: OutPtr) {
    let rank = sec.rank;
    let ps = &params().primes;
    let row0 = blk * L;
    let nrows = (big.rows() - row0).min(L);
    let per_block = NP * rank * N;
    let ahat: &[Lanes] = match *big {
        Big::Pre { data, .. } => {
            let s = &data[blk * per_block * L..(blk + 1) * per_block * L];
            // SAFETY: [u32; L] has the alignment of u32 and the data is a multiple of L words.
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Lanes, per_block) }
        }
        _ => {
            transform_block::<K>(big, row0, rank, &mut sc.ahat, &mut sc.tmp);
            &sc.ahat
        }
    };
    let stride = sec.g * N;
    let gp = sec.gp;
    let shat_len = N * rank * gp;
    if sc.res.len() < NP * gp * N {
        sc.res.resize(NP * gp * N, [0; L]);
    }
    for (pi, pr) in ps.iter().enumerate() {
        let a = &ahat[pi * rank * N..(pi + 1) * rank * N];
        let sh = &sec.shat[pi * shat_len..(pi + 1) * shat_len];
        let res = &mut sc.res[pi * gp * N..(pi + 1) * gp * N];
        K::mac(a, sh, rank, gp, pr, res);
        for g in 0..sec.g {
            K::ntt_inv(&mut res[g * N..(g + 1) * N], pr);
        }
    }
    for g in 0..sec.g {
        crt_write(&sc.res, g, gp, out, row0, nrows, stride, g * N);
    }
}

/// Instruction set (every choice gives the same bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Auto,
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    pub fn parse(name: &str) -> Option<Isa> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Isa::Auto,
            "scalar" | "baseline" => Isa::Scalar,
            "avx2" => Isa::Avx2,
            "avx512" | "avx512f" => Isa::Avx512,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Auto => "auto",
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }

    /// The request if this CPU has it, else the best available one below it.
    pub fn resolve(self) -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            let mut want = if self == Isa::Auto { Isa::Avx512 } else { self };
            if want == Isa::Avx512
                && !(std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl"))
            {
                want = Isa::Avx2;
            }
            if want == Isa::Avx2 && !std::is_x86_feature_detected!("avx2") {
                want = Isa::Scalar;
            }
            want
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Isa::Scalar
        }
    }
}

enum Job<'a> {
    Product { big: Big<'a>, sec: &'a Secrets, out: OutPtr },
    Transform { big: Big<'a>, rank: usize, out: OutPtr32 },
}

#[inline(always)]
fn do_block<K: Kern>(job: &Job, blk: usize, sc: &mut Scratch) {
    match job {
        Job::Product { big, sec, out } => product_block::<K>(big, sec, blk, sc, *out),
        Job::Transform { big, rank, out } => {
            transform_block::<K>(big, blk * L, *rank, &mut sc.ahat, &mut sc.tmp);
            let n = NP * rank * N * L;
            // SAFETY: block `blk` owns words blk*n .. (blk+1)*n of the destination.
            unsafe {
                std::ptr::copy_nonoverlapping(sc.ahat.as_ptr() as *const u32, out.0.add(blk * n), n);
            }
        }
    }
}

fn blocks_portable(job: &Job, next: &AtomicUsize, nblk: usize, sc: &mut Scratch) {
    loop {
        let b = next.fetch_add(1, Ordering::Relaxed);
        if b >= nblk {
            break;
        }
        do_block::<Portable>(job, b, sc);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blocks_avx2(job: &Job, next: &AtomicUsize, nblk: usize, sc: &mut Scratch) {
    loop {
        let b = next.fetch_add(1, Ordering::Relaxed);
        if b >= nblk {
            break;
        }
        do_block::<Portable>(job, b, sc);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx2")]
unsafe fn blocks_avx512(job: &Job, next: &AtomicUsize, nblk: usize, sc: &mut Scratch) {
    loop {
        let b = next.fetch_add(1, Ordering::Relaxed);
        if b >= nblk {
            break;
        }
        do_block::<crate::avx512::Avx512>(job, b, sc);
    }
}

fn run_blocks(job: &Job, nblk: usize, rank: usize, threads: usize, isa: Isa) {
    let isa = isa.resolve();
    let next = AtomicUsize::new(0);
    let t = (if threads == 0 { default_threads() } else { threads }).clamp(1, nblk.max(1));
    let work = |sc: &mut Scratch| {
        // SAFETY: `resolve` only returns instruction sets this CPU has.
        #[cfg(target_arch = "x86_64")]
        match isa {
            Isa::Avx512 => return unsafe { blocks_avx512(job, &next, nblk, sc) },
            Isa::Avx2 => return unsafe { blocks_avx2(job, &next, nblk, sc) },
            _ => {}
        }
        blocks_portable(job, &next, nblk, sc)
    };
    if t == 1 {
        work(&mut Scratch::new(rank));
    } else {
        std::thread::scope(|s| {
            for _ in 0..t {
                s.spawn(|| work(&mut Scratch::new(rank)));
            }
        });
    }
}

pub fn default_threads() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::thread::available_parallelism().map_or(1, |v| v.get()).min(64))
}

/// Errors of [`product`] / [`transform`].
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Shape(String),
    /// The secrets' L1 norm is too large for an exact product (the norm itself is not reported: for a
    /// ternary secret it is its Hamming weight).
    Bound {
        big_bits: u32,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Shape(s) => f.write_str(s),
            Error::Bound { big_bits } => write!(
                f,
                "exactness bound exceeded: a big operand < 2^{big_bits} times these secrets exceeds the CRT range"
            ),
        }
    }
}

/// `big[:rows] @ S mod 2^64` for the block-negacyclic matrices `S` of every secret: `(rows, g*256)`
/// row-major, columns `g*256 .. g*256 + 256` for secret `g`.
pub fn product(big: Big, dim: usize, sec: &Secrets, threads: usize, isa: Isa) -> Result<Vec<u64>, Error> {
    if dim != sec.rank * N {
        return Err(Error::Shape(format!("big operand has {dim} columns, secrets need {}", sec.rank * N)));
    }
    check_shape(&big, sec.rank)?;
    if !crate::arith::bound_ok(big.bits(), sec.l1) {
        return Err(Error::Bound { big_bits: big.bits() });
    }
    let rows = big.rows();
    let total = rows * sec.g * N;
    let mut out: Vec<u64> = Vec::with_capacity(total);
    if total == 0 {
        return Ok(out);
    }
    let job = Job::Product { big, sec, out: OutPtr(out.as_mut_ptr()) };
    run_blocks(&job, rows.div_ceil(L), sec.rank, threads, isa);
    // SAFETY: the blocks cover every row and every block writes all g*256 columns of its rows.
    unsafe { out.set_len(total) };
    Ok(out)
}

/// Residues + forward NTTs of every row (layout of [`Big::Pre`]); `rows` padded to a multiple of 16.
/// Shape checks shared by [`product`] and [`transform`].
pub fn check_shape(big: &Big, rank: usize) -> Result<(), Error> {
    let dim = rank * N;
    if rank == 0 {
        return Err(Error::Shape("rank must be positive".into()));
    }
    let ok = match *big {
        Big::U64 { data, rows } => data.len() == rows * dim,
        Big::Planes { data, rows, npl, first } => npl >= 1 && first + npl <= 8 && data.len() == rows * npl * dim,
        Big::Pre { data, rows, bits } => (1..=64).contains(&bits) && data.len() >= rows.div_ceil(L) * NP * rank * N * L,
    };
    if ok {
        Ok(())
    } else {
        Err(Error::Shape(format!("big operand does not match {} rows of rank {rank}", big.rows())))
    }
}

pub fn transform(big: Big, rank: usize, threads: usize, isa: Isa) -> Result<Vec<u32>, Error> {
    check_shape(&big, rank)?;
    if matches!(big, Big::Pre { .. }) {
        return Err(Error::Shape("operand is already transformed".into()));
    }
    let nblk = big.rows().div_ceil(L);
    let n = nblk * NP * rank * N * L;
    let mut out: Vec<u32> = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let job = Job::Transform { big, rank, out: OutPtr32(out.as_mut_ptr()) };
    run_blocks(&job, nblk, rank, threads, isa);
    // SAFETY: every block was written.
    unsafe { out.set_len(n) };
    Ok(out)
}

/// Recover the secrets of a block-negacyclic matrix `S` (`dim x cols` int8, row-major, built like
/// `pgw.hemask.lwe.secret_matrix` and concatenated along columns): `(cols/256, dim/256, 256)`, or
/// `None` when `S` does not have that structure (or holds -128, whose negation wraps in int8).
pub fn extract_secrets(smat: &[i8], dim: usize, cols: usize) -> Option<Vec<i8>> {
    if dim == 0 || cols == 0 || !dim.is_multiple_of(N) || !cols.is_multiple_of(N) || smat.len() != dim * cols {
        return None;
    }
    let (rank, g) = (dim / N, cols / N);
    // dropped (and wiped) on every early return; handed over unwiped only on success
    let mut s = Wiping(vec![0i8; g * rank * N]);
    for gi in 0..g {
        for l in 0..rank {
            let first = &smat[(l * N) * cols + gi * N..(l * N) * cols + gi * N + N];
            if first.contains(&i8::MIN) {
                return None;
            }
            s[(gi * rank + l) * N..(gi * rank + l + 1) * N].copy_from_slice(first);
            let sp = &s[(gi * rank + l) * N..(gi * rank + l + 1) * N];
            for t in 1..N {
                let row = &smat[(l * N + t) * cols + gi * N..(l * N + t) * cols + gi * N + N];
                // row t: m[t][j] = s[j - t] for j >= t, -s[j - t + 256] for j < t
                for j in 0..t {
                    if row[j] != -sp[j + N - t] {
                        return None;
                    }
                }
                if row[t..] != sp[..N - t] {
                    return None;
                }
            }
        }
    }
    Some(s.into_inner())
}

#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx2")]
pub unsafe fn profile_parts(reps: usize, intrinsics: bool) -> Vec<(&'static str, f64)> {
    if intrinsics {
        profile_parts_k::<crate::avx512::Avx512>(reps)
    } else {
        profile_parts_k::<Portable>(reps)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx2")]
unsafe fn profile_parts_k<K: Kern>(reps: usize) -> Vec<(&'static str, f64)> {
    use std::time::Instant;
    let rank = 16;
    let dim = rank * N;
    let a: Vec<u64> = (0..L * dim as u64 as usize).map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
    let big = Big::U64 { data: &a, rows: L };
    let mut sc = Scratch::new(rank);
    sc.res.resize(NP * GB * N, [0; L]);
    let pr = &params().primes[0];
    let mut out = vec![];
    let t = Instant::now();
    for _ in 0..reps {
        K::load_residues(&big, 0, dim, 0, dim, &mut sc.ahat, &mut sc.tmp);
    }
    out.push(("load_residues per row", t.elapsed().as_secs_f64() / (reps * L) as f64));
    let t = Instant::now();
    for _ in 0..reps {
        for l in 0..rank {
            K::ntt_fwd(&mut sc.ahat[l * N..(l + 1) * N], pr);
        }
    }
    out.push(("one fwd NTT per row", t.elapsed().as_secs_f64() / (reps * L * rank) as f64));
    let t = Instant::now();
    for _ in 0..reps {
        for l in 0..rank {
            K::ntt_inv(&mut sc.ahat[l * N..(l + 1) * N], pr);
        }
    }
    out.push(("one inv NTT per row", t.elapsed().as_secs_f64() / (reps * L * rank) as f64));
    let s: Vec<i8> = (0..GB * rank * N).map(|i| (i % 3) as i8 - 1).collect();
    let sec = Secrets::new(&s, GB, rank);
    let shat_len = N * rank * sec.gp;
    let t = Instant::now();
    for _ in 0..reps {
        K::mac(&sc.ahat[..rank * N], &sec.shat[..shat_len], rank, sec.gp, pr, &mut sc.res[..GB * N]);
    }
    out.push(("mac (one prime, GB secrets) per row", t.elapsed().as_secs_f64() / (reps * L) as f64));
    let mut o = vec![0u64; L * N];
    let t = Instant::now();
    for _ in 0..reps {
        crt_write(&sc.res, 0, GB, OutPtr(o.as_mut_ptr()), 0, L, N, 0);
    }
    out.push(("crt per row (one secret)", t.elapsed().as_secs_f64() / (reps * L) as f64));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::bound_ok;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Schoolbook `sum_l a_l * s_l` mod 2^64 (negacyclic).
    fn reference(a: &[u64], s: &[i8], rank: usize) -> Vec<u64> {
        let mut out = vec![0u64; N];
        for l in 0..rank {
            for t in 0..N {
                for u in 0..N {
                    let prod = a[l * N + t].wrapping_mul(s[l * N + u] as i64 as u64);
                    let k = t + u;
                    if k < N {
                        out[k] = out[k].wrapping_add(prod);
                    } else {
                        out[k - N] = out[k - N].wrapping_sub(prod);
                    }
                }
            }
        }
        out
    }

    fn check(rows: usize, rank: usize, g: usize, max: bool, isa: Isa) {
        let mut seed = 0x1234_5678_9abc_def1u64 ^ (rows * 31 + rank * 7 + g) as u64;
        let dim = rank * N;
        let a: Vec<u64> = (0..rows * dim).map(|_| if max { u64::MAX } else { rng(&mut seed) }).collect();
        let s: Vec<i8> = (0..g * rank * N).map(|_| if max { -1 } else { (rng(&mut seed) % 3) as i8 - 1 }).collect();
        let sec = Secrets::new(&s, g, rank);
        let got = product(Big::U64 { data: &a, rows }, dim, &sec, 2, isa).unwrap();
        let pre = transform(Big::U64 { data: &a, rows }, rank, 2, isa).unwrap();
        let got2 = product(Big::Pre { data: &pre, rows, bits: 64 }, dim, &sec, 3, isa).unwrap();
        assert_eq!(got, got2);
        for r in 0..rows {
            for gi in 0..g {
                let want = reference(&a[r * dim..(r + 1) * dim], &s[gi * rank * N..(gi + 1) * rank * N], rank);
                assert_eq!(&got[r * g * N + gi * N..r * g * N + gi * N + N], &want[..], "row {r} secret {gi}");
            }
        }
    }

    #[test]
    fn product_matches_schoolbook() {
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
            check(17, 2, 5, false, isa);
            check(3, 1, 1, true, isa);
        }
    }

    #[test]
    fn large_rank_uses_several_chunks() {
        check(2, MAC_CHUNK + 3, 1, false, Isa::Auto);
    }

    #[test]
    fn rank16_all_max_at_bound() {
        check(1, 16, 1, true, Isa::Auto);
    }

    /// The largest products the CRT admits, not just the protocol's (`2^76` at rank 16): rank 129 with
    /// every secret coefficient at `+-127` (L1 = 129 * 256 * 127, the largest rank whose 64-bit
    /// products fit) and `A = 2^64 - 1`, so coefficient 255 is `Z = +-(2^64 - 1) L1 = +-0.99998 HALF`.
    /// Several MAC chunks (`MAC_CHUNK = 48 < 129`), every instruction set; rank 130 is refused.
    #[test]
    fn near_half_products_are_exact() {
        let (rank, rows) = (129usize, 2usize);
        let dim = rank * N;
        let half = params().crt.half;
        let l1 = (rank * N * 127) as u128;
        let z = (u64::MAX as u128) * l1;
        assert!(z <= half && z * 100_000 >= half * 99_997, "Z = {z}, HALF = {half}");
        assert!(bound_ok(64, l1 as u64) && !bound_ok(64, (130 * N * 127) as u64));
        let mut seed = 0x5eed_0129u64;
        // row 0 all-max, row 1 random words
        let a: Vec<u64> = (0..rows * dim).map(|i| if i < dim { u64::MAX } else { rng(&mut seed) }).collect();
        for case in 0..3 {
            let s: Vec<i8> = (0..rank * N)
                .map(|_| match case {
                    0 => 127,
                    1 => -127,
                    _ => {
                        if rng(&mut seed) & 1 == 0 {
                            127
                        } else {
                            -127
                        }
                    }
                })
                .collect();
            let sec = Secrets::new(&s, 1, rank);
            assert_eq!(sec.l1 as u128, l1);
            let want: Vec<u64> = (0..rows).flat_map(|r| reference(&a[r * dim..(r + 1) * dim], &s, rank)).collect();
            if case < 2 {
                // Z_255 = a * sum(s) exactly; mod 2^64 that is -sum(s)
                let zs = (u64::MAX as i128) * s.iter().map(|&v| v as i128).sum::<i128>();
                assert_eq!(want[N - 1], zs as u64);
            }
            for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
                let got = product(Big::U64 { data: &a, rows }, dim, &sec, 2, isa).unwrap();
                assert_eq!(got, want, "case {case} isa {isa:?}");
            }
        }
        let s130 = vec![127i8; 130 * N];
        let sec = Secrets::new(&s130, 1, 130);
        let a130 = vec![u64::MAX; 130 * N];
        let r = product(Big::U64 { data: &a130, rows: 1 }, 130 * N, &sec, 1, Isa::Auto);
        assert_eq!(r, Err(Error::Bound { big_bits: 64 }));
        assert!(!r.unwrap_err().to_string().contains(&(130 * N * 127).to_string()));
    }

    #[test]
    fn bound_is_enforced() {
        let s = vec![3i8; 64 * 128 * N]; // L1 = 3 * 2^21 > 2^21.99 (the limit for a 64-bit big operand)
        let sec = Secrets::new(&s, 1, 64 * 128);
        let a = vec![0u64; 64 * 128 * N];
        let r = product(Big::U64 { data: &a, rows: 1 }, 64 * 128 * N, &sec, 1, Isa::Auto);
        assert!(matches!(r, Err(Error::Bound { .. })));
    }

    #[test]
    fn extract_roundtrip() {
        let mut seed = 99u64;
        let rank = 2;
        let g = 2;
        let s: Vec<i8> = (0..g * rank * N).map(|_| (rng(&mut seed) % 3) as i8 - 1).collect();
        let (dim, cols) = (rank * N, g * N);
        let mut m = vec![0i8; dim * cols];
        for gi in 0..g {
            for l in 0..rank {
                for t in 0..N {
                    for j in 0..N {
                        let v =
                            if j >= t { s[(gi * rank + l) * N + j - t] } else { -s[(gi * rank + l) * N + j + N - t] };
                        m[(l * N + t) * cols + gi * N + j] = v;
                    }
                }
            }
        }
        assert_eq!(extract_secrets(&m, dim, cols).unwrap(), s);
        m[(N + 5) * cols + 7] ^= 1;
        assert!(extract_secrets(&m, dim, cols).is_none());
    }
}
