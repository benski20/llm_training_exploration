//! Modular arithmetic of `pks-v3`: lazy Montgomery (`R = 2^32`) over primes `p < 2^30` (so `4p < 2^32`),
//! negacyclic NTTs of any power-of-two length (Cooley-Tukey forward to bit-reversed order,
//! Gentleman-Sande inverse without the `N^-1` factor), single-polynomial and `L`-lane interleaved
//! (`x[c * L + l]`: every stage is a contiguous 16-word loop, one AVX-512 vector of u32).
//!
//! Every function is branch-free in its data (constant time for secret inputs): reductions use
//! `min(v, v - p)` (`vpminud`/`cmov`), Montgomery products have no data-dependent control flow.
//! Every operation that may wrap says so (`wrapping_*`): the workspace builds with overflow checks.

/// Lanes of the interleaved kernels (16 x u32 = one AVX-512 register).
pub const L: usize = 16;
/// Largest column (points) of the column-wise NTT passes.
const MAXPTS: usize = 32;

use zeroize::Zeroize;

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    let mulmod = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let powm = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        r
    };
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powm(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// The `count` largest primes below `2^bits` that are `1 mod step` (same rule as `pks.ntt_primes`).
pub fn ntt_primes(bits: u32, count: usize, step: u64) -> Vec<u32> {
    let mut c = ((1u64 << bits) - 1) / step * step + 1;
    let mut out = Vec::new();
    while out.len() < count {
        if c < (1u64 << bits) && is_prime(c) {
            out.push(c as u32);
        }
        c -= step;
    }
    out
}

pub fn powmod(b: u64, mut e: u64, p: u64) -> u64 {
    let (mut r, mut b) = (1u64, b % p);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    r
}

fn primitive_root(p: u64) -> u64 {
    let mut fac = Vec::new();
    let (mut n, mut f) = (p - 1, 2u64);
    while f * f <= n {
        if n % f == 0 {
            fac.push(f);
            while n % f == 0 {
                n /= f;
            }
        }
        f += 1;
    }
    if n > 1 {
        fac.push(n);
    }
    (2..p).find(|&g| fac.iter().all(|&q| powmod(g, (p - 1) / q, p) != 1)).expect("generator")
}

#[inline(always)]
pub fn mont(a: u32, b: u32, p: u32, pn: u32) -> u32 {
    let t = (a as u64).wrapping_mul(b as u64);
    let m = (t as u32).wrapping_mul(pn);
    (t.wrapping_add((m as u64).wrapping_mul(p as u64)) >> 32) as u32
}

/// `[0, 2q) -> [0, q)` (branch-free).
#[inline(always)]
pub fn red(v: u32, q: u32) -> u32 {
    v.min(v.wrapping_sub(q))
}

/// One NTT prime with Montgomery constants and twiddle tables for length `n`.
#[derive(Clone, Debug)]
pub struct Prime {
    pub p: u32,
    pub pn: u32,
    pub n: usize,
    /// `psi^brv(k) R mod p`.
    pub(crate) zm: Vec<u32>,
    /// `psi^-brv(k) R mod p`.
    pub(crate) izm: Vec<u32>,
    /// `R^2 mod p`.
    pub r2: u32,
    /// `N^-1 R^2 mod p`: `mont(x, ninv_r2) = x N^-1 R` (hint form of a plain value).
    pub ninv_r2: u32,
    /// `N^-1 R^3 mod p`: `mont(x R^-1, ninv_r3) = x N^-1 R` (hint form of a value read with [`Prime::res56`]).
    pub ninv_r3: u32,
}

impl Prime {
    pub fn new(p: u32, n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2);
        assert!(p < (1 << 30) && (p as u64 - 1).is_multiple_of(2 * n as u64));
        let pl = p as u64;
        let lg = n.trailing_zeros();
        let psi = powmod(primitive_root(pl), (pl - 1) / (2 * n as u64), pl);
        assert_eq!(powmod(psi, n as u64, pl), pl - 1);
        let r = (1u64 << 32) % pl;
        let brv = |k: usize| (k as u32).reverse_bits() >> (32 - lg);
        let zm = (0..n).map(|k| (powmod(psi, brv(k) as u64, pl) * r % pl) as u32).collect();
        let ipsi = powmod(psi, pl - 2, pl);
        let izm = (0..n).map(|k| (powmod(ipsi, brv(k) as u64, pl) * r % pl) as u32).collect();
        let mut inv = 1u64;
        for _ in 0..32 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(pl.wrapping_mul(inv)));
        }
        let pn = (inv as u32).wrapping_neg();
        let r2 = (r * r % pl) as u32;
        let ninv = powmod(n as u64, pl - 2, pl);
        let ninv_r2 = (ninv * r % pl * r % pl) as u32;
        let ninv_r3 = (ninv_r2 as u64 * r % pl) as u32;
        Prime { p, pn, n, zm, izm, r2, ninv_r2, ninv_r3 }
    }

    #[inline(always)]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        mont(a, b, self.p, self.pn)
    }

    #[inline(always)]
    pub fn reduce(&self, v: u32) -> u32 {
        red(v, self.p)
    }

    /// `[0, 4p) -> [0, p)`.
    #[inline(always)]
    pub fn reduce4(&self, v: u32) -> u32 {
        red(red(v, 2 * self.p), self.p)
    }

    /// Plain residue -> Montgomery form `x R mod p`, in `[0, p)`.
    pub fn to_mont(&self, x: u32) -> u32 {
        self.reduce(self.mul(x, self.r2))
    }

    /// `x R^-1 mod p` in `[0, 2p)` for any `x < 2^56` (`x = hi 2^32 + lo`: `hi + lo R^-1`). No division.
    #[inline(always)]
    pub fn res56(&self, x: u64) -> u32 {
        let hi = (x >> 32) as u32; // < 2^24 < p
        let lo = mont(x as u32, 1, self.p, self.pn); // lo R^-1, in [0, p] (lo < 2^32, b = 1)
        red(hi.wrapping_add(lo), 2 * self.p) // < p + 2^24 + p... reduced into [0, 2p)
    }

    /// Residue of a signed small value `|x| < 2^29` (branch-free), in `[0, p)`.
    #[inline(always)]
    pub fn res_small(&self, x: i32) -> u32 {
        let m = (x >> 31) as u32; // all ones when negative
        (x as u32).wrapping_add(self.p & m)
    }

    /// Forward negacyclic NTT of one polynomial, `[0, 2p)` in and out.
    #[inline(always)]
    pub fn fwd(&self, a: &mut [u32]) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let (mut len, mut k) = (n / 2, 1usize);
        while len >= 1 {
            let nb = n / (2 * len);
            for b in 0..nb {
                let z = self.zm[k + b];
                let (lo, hi) = a[2 * b * len..2 * (b + 1) * len].split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let t = mont(*y, z, p, pn);
                    let u = *x;
                    *x = red(u.wrapping_add(t), p2);
                    *y = red(u.wrapping_add(p2).wrapping_sub(t), p2);
                }
            }
            k += nb;
            len /= 2;
        }
    }

    /// Inverse NTT of one polynomial without the `N^-1` factor, `[0, 2p)` in and out.
    #[inline(always)]
    pub fn inv(&self, a: &mut [u32]) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let mut len = 1usize;
        while len < n {
            let nb = n / (2 * len);
            for b in 0..nb {
                let z = self.izm[nb + b];
                let (lo, hi) = a[2 * b * len..2 * (b + 1) * len].split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = red(u.wrapping_add(w), p2);
                    *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                }
            }
            len *= 2;
        }
    }

    /// [`Prime::fwd`] on `L` interleaved polynomials (`n * L` words).
    #[inline(always)]
    pub fn fwd_l(&self, a: &mut [u32]) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let (mut len, mut k) = (n / 2, 1usize);
        while len >= 1 {
            let nb = n / (2 * len);
            for b in 0..nb {
                let z = self.zm[k + b];
                let base = 2 * b * len * L;
                let (lo, hi) = a[base..base + 2 * len * L].split_at_mut(len * L);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let t = mont(*y, z, p, pn);
                    let u = *x;
                    *x = red(u.wrapping_add(t), p2);
                    *y = red(u.wrapping_add(p2).wrapping_sub(t), p2);
                }
            }
            k += nb;
            len /= 2;
        }
    }

    /// `a = INTT(h * t)` on `L` interleaved polynomials without the `N^-1` factor: `h` interleaved
    /// (`n * L`), `t` one polynomial broadcast to every lane (`< p`). The pointwise product and the first
    /// stages run per 256-coefficient sub-block while it is in L1 (16 KiB), the remaining stages as whole
    /// passes. Same values as `mont` then [`Prime::inv_l`].
    #[inline(always)]
    pub fn inv_l_mul(&self, a: &mut [u32], h: &[u32], t: &[u32]) {
        const B: usize = 256;
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let sub = B.min(n);
        for sb in 0..n / sub {
            let base = sb * sub;
            let blk = &mut a[base * L..(base + sub) * L];
            let hs = &h[base * L..(base + sub) * L];
            for c in 0..sub {
                let tc = t[base + c];
                for l in 0..L {
                    blk[c * L + l] = mont(hs[c * L + l], tc, p, pn);
                }
            }
            let mut len = 1usize;
            while len < sub {
                let nb = n / (2 * len);
                let b0 = base / (2 * len);
                for bb in 0..sub / (2 * len) {
                    let z = self.izm[nb + b0 + bb];
                    let off = 2 * bb * len * L;
                    let (lo, hi) = blk[off..off + 2 * len * L].split_at_mut(len * L);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        let (u, w) = (*x, *y);
                        *x = red(u.wrapping_add(w), p2);
                        *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                    }
                }
                len *= 2;
            }
        }
        let mut len = sub;
        while len < n {
            let nb = n / (2 * len);
            for b in 0..nb {
                let z = self.izm[nb + b];
                let base = 2 * b * len * L;
                let (lo, hi) = a[base..base + 2 * len * L].split_at_mut(len * L);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = red(u.wrapping_add(w), p2);
                    *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                }
            }
            len *= 2;
        }
    }

    /// Inverse (Gentleman-Sande) stages `len = from, 2 from, .., n/2` on `L` interleaved polynomials,
    /// column by column: the `n / from` points `c0 + from k` of column `c0` are the only words these stages
    /// combine, so all `log2(n / from)` stages of one column run while its lines are in L1 (one pass over
    /// the array instead of one per stage). Same values as the corresponding stages of [`Prime::inv_l`].
    #[inline(always)]
    pub fn inv_l_cols(&self, a: &mut [u32], from: usize) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let pts = n / from;
        assert!(pts <= MAXPTS);
        // the column's lines are `from * 64` bytes apart (same L1 set): copy them to a contiguous buffer
        let mut buf = [[0u32; L]; MAXPTS];
        for c0 in 0..from {
            for k in 0..pts {
                buf[k].copy_from_slice(&a[(c0 + from * k) * L..(c0 + from * k + 1) * L]);
            }
            let mut lk = 1usize; // half-block length in points
            while lk < pts {
                let nb = pts / (2 * lk); // = n / (2 len)
                for bk in 0..nb {
                    let z = self.izm[nb + bk];
                    let (lo, hi) = buf[2 * bk * lk..2 * (bk + 1) * lk].split_at_mut(lk);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        for l in 0..L {
                            let (u, w) = (x[l], y[l]);
                            x[l] = red(u.wrapping_add(w), p2);
                            y[l] = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                        }
                    }
                }
                lk *= 2;
            }
            for k in 0..pts {
                a[(c0 + from * k) * L..(c0 + from * k + 1) * L].copy_from_slice(&buf[k]);
            }
        }
        buf.zeroize();
    }

    /// Forward (Cooley-Tukey) stages `len = n/2 .. to` column by column (see [`Prime::inv_l_cols`]), then
    /// the stages `len < to` per `to`-coefficient sub-block (in L1). Same values as [`Prime::fwd_l`].
    #[inline(always)]
    pub fn fwd_l_fast(&self, a: &mut [u32], to: usize) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let to = to.min(n / 2).max(n / MAXPTS);
        let pts = n / to;
        let mut buf = [[0u32; L]; MAXPTS];
        for c0 in 0..to {
            for k in 0..pts {
                buf[k].copy_from_slice(&a[(c0 + to * k) * L..(c0 + to * k + 1) * L]);
            }
            let mut lk = pts / 2;
            while lk >= 1 {
                let nb = pts / (2 * lk);
                for bk in 0..nb {
                    let z = self.zm[nb + bk];
                    let (lo, hi) = buf[2 * bk * lk..2 * (bk + 1) * lk].split_at_mut(lk);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        for l in 0..L {
                            let t = mont(y[l], z, p, pn);
                            let u = x[l];
                            x[l] = red(u.wrapping_add(t), p2);
                            y[l] = red(u.wrapping_add(p2).wrapping_sub(t), p2);
                        }
                    }
                }
                lk /= 2;
            }
            for k in 0..pts {
                a[(c0 + to * k) * L..(c0 + to * k + 1) * L].copy_from_slice(&buf[k]);
            }
        }
        buf.zeroize();
        // stages len = to/2 .. 1 inside each sub-block of `to` coefficients
        for sb in 0..pts {
            let base = sb * to;
            let blk = &mut a[base * L..(base + to) * L];
            let mut len = to / 2;
            while len >= 1 {
                let nb = n / (2 * len);
                let b0 = base / (2 * len);
                for bb in 0..to / (2 * len) {
                    let z = self.zm[nb + b0 + bb];
                    let off = 2 * bb * len * L;
                    let (lo, hi) = blk[off..off + 2 * len * L].split_at_mut(len * L);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        let t = mont(*y, z, p, pn);
                        let u = *x;
                        *x = red(u.wrapping_add(t), p2);
                        *y = red(u.wrapping_add(p2).wrapping_sub(t), p2);
                    }
                }
                len /= 2;
            }
        }
    }

    /// [`Prime::inv_l_mul`] with the stages `len >= 256` done column-wise ([`Prime::inv_l_cols`]): two
    /// passes over the array instead of `log2(n) - 7`. Same values.
    #[inline(always)]
    pub fn inv_l_mul_fast(&self, a: &mut [u32], h: &[u32], t: &[u32]) {
        const B: usize = 256;
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let sub = B.min(n);
        for sb in 0..n / sub {
            let base = sb * sub;
            let blk = &mut a[base * L..(base + sub) * L];
            let hs = &h[base * L..(base + sub) * L];
            for c in 0..sub {
                let tc = t[base + c];
                let o: &mut [u32; L] = (&mut blk[c * L..(c + 1) * L]).try_into().unwrap();
                let hv: &[u32; L] = (&hs[c * L..(c + 1) * L]).try_into().unwrap();
                for l in 0..L {
                    o[l] = mont(hv[l], tc, p, pn);
                }
            }
            let mut len = 1usize;
            while len < sub {
                let nb = n / (2 * len);
                let b0 = base / (2 * len);
                for bb in 0..sub / (2 * len) {
                    let z = self.izm[nb + b0 + bb];
                    let off = 2 * bb * len * L;
                    let (lo, hi) = blk[off..off + 2 * len * L].split_at_mut(len * L);
                    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                        let (u, w) = (*x, *y);
                        *x = red(u.wrapping_add(w), p2);
                        *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                    }
                }
                len *= 2;
            }
        }
        if sub < n {
            self.inv_l_cols(a, sub);
        }
    }

    /// [`Prime::inv`] on `L` interleaved polynomials (`n * L` words).
    #[inline(always)]
    pub fn inv_l(&self, a: &mut [u32]) {
        let (p, pn, p2, n) = (self.p, self.pn, 2 * self.p, self.n);
        let mut len = 1usize;
        while len < n {
            let nb = n / (2 * len);
            for b in 0..nb {
                let z = self.izm[nb + b];
                let base = 2 * b * len * L;
                let (lo, hi) = a[base..base + 2 * len * L].split_at_mut(len * L);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = red(u.wrapping_add(w), p2);
                    *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                }
            }
            len *= 2;
        }
    }
}

/// Instruction set of the kernels (all compiled in, picked at run time; identical results).
/// `Avx512Ifma` is `Avx512` plus the IFMA decryption (one composite NTT modulo `q1`, [`crate::ifma`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
    Avx512Ifma,
}

impl Isa {
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("popcnt")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("bmi2")
            {
                if std::arch::is_x86_feature_detected!("avx512ifma") {
                    return Isa::Avx512Ifma;
                }
                return Isa::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("popcnt")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("bmi2")
            {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }

    /// `auto` (detect, or `PGW_PKS_ISA`), `avx512ifma`, `avx512`, `avx2`, `scalar`; capped at what the CPU
    /// has (each level includes the ones below it).
    pub fn parse(name: &str) -> Option<Isa> {
        let best = Isa::detect();
        let want = match name {
            "auto" | "" => match std::env::var("PGW_PKS_ISA").unwrap_or_default().as_str() {
                "" | "auto" => best,
                other => return Isa::parse(other),
            },
            "avx512ifma" => Isa::Avx512Ifma,
            "avx512" => Isa::Avx512,
            "avx2" => Isa::Avx2,
            "scalar" => Isa::Scalar,
            _ => return None,
        };
        Some(want.min(best))
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Avx512Ifma => "avx512ifma",
        }
    }
}

/// Defines `$vis fn $name(isa: Isa, args..)` that runs `$inner(args..)` (an `#[inline(always)]` function)
/// compiled for AVX-512, AVX2 or the baseline: the same code with other target features (auto-vectorised
/// lane loops), so every ISA gives identical results by construction.
#[macro_export]
macro_rules! multiversion {
    ($vis:vis fn $name:ident ( $($a:ident : $t:ty),* $(,)? ) $(-> $r:ty)? => $inner:path) => {
        #[allow(clippy::too_many_arguments)]
        $vis fn $name(isa: $crate::arith::Isa, $($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx2,fma,bmi2,popcnt")]
                unsafe fn v512($($a: $t),*) $(-> $r)? { $inner($($a),*) }
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx2,fma,bmi2,popcnt")]
                unsafe fn v256($($a: $t),*) $(-> $r)? { $inner($($a),*) }
                match isa {
                    // SAFETY: `Isa::parse` caps the ISA at what this CPU supports.
                    $crate::arith::Isa::Avx512 | $crate::arith::Isa::Avx512Ifma => return unsafe { v512($($a),*) },
                    $crate::arith::Isa::Avx2 => return unsafe { v256($($a),*) },
                    $crate::arith::Isa::Scalar => {}
                }
            }
            let _ = isa;
            $inner($($a),*)
        }
    };
}

/// Bit unpacking of `pgw.hemask.wire.pack_bits` (fields of 64/32/16/8 bits, then a `bits mod 8` tail stream).
pub fn unpack_bits(data: &[u8], count: usize, bits: u32) -> Result<Vec<u64>, String> {
    if !(1..=64).contains(&bits) || !count.is_multiple_of(8) || data.len() != count * bits as usize / 8 {
        return Err(format!("packed data: {} bytes is not {count} values of {bits} bits", data.len()));
    }
    let mut out = vec![0u64; count];
    let (mut off, mut pos) = (0u32, 0usize);
    while bits - off >= 8 {
        let w = [64u32, 32, 16, 8].into_iter().find(|&c| c <= bits - off).unwrap();
        let wb = (w / 8) as usize;
        let src = &data[pos..pos + count * wb];
        for (i, o) in out.iter_mut().enumerate() {
            let mut v = 0u64;
            for b in 0..wb {
                v |= (src[i * wb + b] as u64) << (8 * b);
            }
            *o |= v << off;
        }
        pos += count * wb;
        off += w;
    }
    let t = bits - off;
    if t > 0 {
        let src = &data[pos..];
        let tb = t as usize;
        for g in 0..count / 8 {
            let mut word = 0u64;
            for b in 0..tb {
                word |= (src[g * tb + b] as u64) << (8 * b);
            }
            for k in 0..8 {
                out[8 * g + k] |= ((word >> (t as usize * k)) & ((1u64 << t) - 1)) << off;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_round_trip_and_lanes() {
        for &(bits, n) in &[(24u32, 8192usize), (30, 8192), (30, 4096)] {
            let pr = Prime::new(ntt_primes(bits, 1, 16384)[0], n);
            let a: Vec<u32> = (0..n as u64).map(|i| ((i * 2654435761) % pr.p as u64) as u32).collect();
            let mut b = a.clone();
            pr.fwd(&mut b);
            let mut lanes = vec![0u32; n * L];
            for c in 0..n {
                lanes[c * L + 3] = a[c];
            }
            pr.fwd_l(&mut lanes);
            for c in 0..n {
                assert_eq!(pr.reduce(lanes[c * L + 3]), pr.reduce(b[c]));
            }
            pr.inv(&mut b);
            let ninv = powmod(n as u64, pr.p as u64 - 2, pr.p as u64);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(*x as u64, pr.reduce(*y) as u64 * ninv % pr.p as u64);
            }
        }
    }

    #[test]
    fn fused_ntt_passes_equal_plain() {
        for &(bits, n) in &[(24u32, 8192usize), (30, 8192), (30, 4096)] {
            let pr = Prime::new(ntt_primes(bits, 1, 16384)[0], n);
            let mut s = 7u64;
            let mut rnd = |m: u32| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 20) % m as u64) as u32
            };
            let h: Vec<u32> = (0..n * L).map(|_| rnd(2 * pr.p)).collect();
            let t: Vec<u32> = (0..n).map(|_| rnd(pr.p)).collect();
            let (mut a, mut b) = (vec![0u32; n * L], vec![0u32; n * L]);
            pr.inv_l_mul(&mut a, &h, &t);
            pr.inv_l_mul_fast(&mut b, &h, &t);
            assert_eq!(a, b);
            for to in [1usize, 16, 256, 512] {
                let (mut x, mut y) = (h.clone(), h.clone());
                pr.fwd_l(&mut x);
                pr.fwd_l_fast(&mut y, to);
                assert_eq!(x, y);
            }
        }
    }

    #[test]
    fn res56_is_x_rinv() {
        let pr = Prime::new(ntt_primes(30, 1, 16384)[0], 8192);
        let rinv = powmod(1u64 << 32, pr.p as u64 - 2, pr.p as u64);
        for x in [0u64, 1, (1 << 56) - 1, 0x00AB_CDEF_0123_4567, (1 << 32) - 1, 1 << 32] {
            let got = pr.reduce(pr.res56(x)) as u64;
            assert_eq!(got, (x % pr.p as u64) * rinv % pr.p as u64);
        }
        for x in [-5i32, 0, 7, -21] {
            assert_eq!(pr.res_small(x) as i64, (x as i64).rem_euclid(pr.p as i64));
        }
    }

    #[test]
    fn unpack_matches_layout() {
        // 36 bits: field 32 + tail 4
        let vals: Vec<u64> = (0..16u64).map(|i| (i * 0x1_2345_6789) & ((1 << 36) - 1)).collect();
        let mut data = Vec::new();
        for v in &vals {
            data.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        for g in 0..2 {
            let mut w = 0u32;
            for k in 0..8 {
                w |= ((vals[8 * g + k] >> 32) as u32 & 15) << (4 * k);
            }
            data.extend_from_slice(&w.to_le_bytes());
        }
        assert_eq!(unpack_bits(&data, 16, 36).unwrap(), vals);
    }
}
