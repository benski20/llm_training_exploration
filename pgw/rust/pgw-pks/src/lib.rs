//! # pgw-pks (research prototype, scheme `pks-v3`; not wired into the factory)
//!
//! Customer-side kernels of the compact-hint encrypted mask scheme of
//! `pgw/research/plans/he-compact-hint-pks.md` (numpy reference: `pgw.hemask.pks`):
//!
//! * [`decrypt_block`]: one 16-row block of one 256-slot group of a packed key-switched reply. With
//!   the compact hint `c1` (NTT domain, two ~2^24 primes, `N^-1` and the Montgomery factor folded in)
//!   and the fresh reply secret `t'` (NTT domain), `v = c0 + c1 t' mod q1` costs one pointwise product
//!   and one length-4096 inverse NTT per prime; the mask is `round(2^32 v / q1) mod 2^32`.
//! * [`pad_block`]: the ring-LWE upload pad `a_C t mod 2^56` of one 16-feature block (three ~2^30
//!   primes, exact centred CRT: `|a t| < 2^56 * 4096 = 2^68`, range `> 2^89`).
//!
//! Arithmetic: lazy Montgomery with `R = 2^32`, all primes `< 2^30` so `4p < 2^32`; NTT values stay in
//! `[0, 2p)`. Every operation that may wrap says so (`wrapping_*`): the workspace builds with
//! overflow checks.

use std::sync::OnceLock;

pub const N: usize = 4096;
/// Output rows packed per reply ciphertext (`N / 256`).
pub const ROWS: usize = 16;
/// Slots per group (module degree seen by the provider).
pub const SLOTS: usize = 256;

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
    let powmod = |mut b: u64, mut e: u64| {
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
        let mut x = powmod(a, d);
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

/// The `count` largest primes below `2^bits` that are `1 mod 2N` (same rule as `pks.ntt_primes`).
pub fn ntt_primes(bits: u32, count: usize) -> Vec<u32> {
    let step = 2 * N as u64;
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

fn powmod(b: u64, mut e: u64, p: u64) -> u64 {
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

/// One prime with Montgomery constants and twiddle tables.
#[derive(Clone, Debug)]
pub struct Prime {
    pub p: u32,
    pn: u32,
    /// `psi^brv(k) R mod p`.
    zm: Vec<u32>,
    /// `psi^-brv(k) R mod p`.
    izm: Vec<u32>,
    /// `R^2 mod p`.
    r2: u32,
    /// `N^-1 R^2 mod p`: `mont_mul(x, ninv_r2) = x N^-1 R` (hint form).
    ninv_r2: u32,
}

#[inline(always)]
fn mont(a: u32, b: u32, p: u32, pn: u32) -> u32 {
    let t = (a as u64).wrapping_mul(b as u64);
    let m = (t as u32).wrapping_mul(pn);
    (t.wrapping_add((m as u64).wrapping_mul(p as u64)) >> 32) as u32
}

#[inline(always)]
fn red2(v: u32, p2: u32) -> u32 {
    v.min(v.wrapping_sub(p2))
}

impl Prime {
    pub fn new(p: u32) -> Self {
        assert!(p < (1 << 30) && (p as u64 - 1).is_multiple_of(2 * N as u64));
        let pl = p as u64;
        let psi = powmod(primitive_root(pl), (pl - 1) / (2 * N as u64), pl);
        assert_eq!(powmod(psi, N as u64, pl), pl - 1);
        let r = (1u64 << 32) % pl;
        let brv = |k: usize| (k as u32).reverse_bits() >> (32 - 12);
        let zm = (0..N).map(|k| (powmod(psi, brv(k) as u64, pl) * r % pl) as u32).collect();
        let ipsi = powmod(psi, pl - 2, pl);
        let izm = (0..N).map(|k| (powmod(ipsi, brv(k) as u64, pl) * r % pl) as u32).collect();
        let mut inv = 1u64;
        for _ in 0..32 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(pl.wrapping_mul(inv)));
        }
        let pn = (inv as u32).wrapping_neg();
        let r2 = (r * r % pl) as u32;
        let ninv_r2 = (powmod(N as u64, pl - 2, pl) * r % pl * r % pl) as u32;
        Prime { p, pn, zm, izm, r2, ninv_r2 }
    }

    #[inline(always)]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        mont(a, b, self.p, self.pn)
    }

    /// Plain residue -> hint form `x N^-1 R mod p` (in `[0, p)`).
    pub fn to_hint_form(&self, x: u32) -> u32 {
        self.reduce(self.mul(x, self.ninv_r2))
    }

    /// Plain residue -> Montgomery form `x R mod p`.
    pub fn to_mont(&self, x: u32) -> u32 {
        self.reduce(self.mul(x, self.r2))
    }

    #[inline(always)]
    pub fn reduce(&self, v: u32) -> u32 {
        v.min(v.wrapping_sub(self.p))
    }

    /// Forward negacyclic NTT, input in `[0, 2p)`, output in `[0, 2p)` (bit-reversed order).
    pub fn fwd(&self, a: &mut [u32]) {
        let (p, pn, p2) = (self.p, self.pn, 2 * self.p);
        let (mut len, mut k) = (N / 2, 1usize);
        while len >= 1 {
            let nb = N / (2 * len);
            for b in 0..nb {
                let z = self.zm[k + b];
                let (lo, hi) = a[2 * b * len..2 * (b + 1) * len].split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let t = mont(*y, z, p, pn);
                    let u = *x;
                    *x = red2(u.wrapping_add(t), p2);
                    *y = red2(u.wrapping_add(p2).wrapping_sub(t), p2);
                }
            }
            k += nb;
            len /= 2;
        }
    }

    /// Inverse NTT without the `N^-1` factor (fold it into an operand), `[0, 2p)` in and out.
    pub fn inv(&self, a: &mut [u32]) {
        let (p, pn, p2) = (self.p, self.pn, 2 * self.p);
        let mut len = 1usize;
        while len < N {
            let nb = N / (2 * len);
            for b in 0..nb {
                let z = self.izm[nb + b];
                let (lo, hi) = a[2 * b * len..2 * (b + 1) * len].split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = red2(u.wrapping_add(w), p2);
                    *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                }
            }
            len *= 2;
        }
    }
}

/// Constants of the reply modulus `q1 = p0 p1` (customer decryption).
pub struct ReplyCtx {
    pub pr: [Prime; 2],
    pub q1: u64,
    /// `p0^-1 mod p1` in Montgomery form.
    inv01_m: u32,
    /// `2^32 / q1` as f64.
    scale: f64,
}

/// Constants of the upload pad (three primes, exact CRT to `mod 2^56`).
pub struct PadCtx {
    pub pr: [Prime; 3],
    inv01_m: u32,
    inv012_m: u32,
}

pub fn reply_ctx() -> &'static ReplyCtx {
    static C: OnceLock<ReplyCtx> = OnceLock::new();
    C.get_or_init(|| {
        let ps = ntt_primes(24, 2);
        let pr = [Prime::new(ps[0]), Prime::new(ps[1])];
        let q1 = ps[0] as u64 * ps[1] as u64;
        let inv01 = powmod(ps[0] as u64, ps[1] as u64 - 2, ps[1] as u64) as u32;
        let inv01_m = pr[1].to_mont(inv01);
        ReplyCtx { pr, q1, inv01_m, scale: (1u64 << 32) as f64 / q1 as f64 }
    })
}

pub fn pad_ctx() -> &'static PadCtx {
    static C: OnceLock<PadCtx> = OnceLock::new();
    C.get_or_init(|| {
        let ps = ntt_primes(30, 3);
        let pr = [Prime::new(ps[0]), Prime::new(ps[1]), Prime::new(ps[2])];
        let (p0, p1, p2) = (ps[0] as u64, ps[1] as u64, ps[2] as u64);
        let inv01_m = pr[1].to_mont(powmod(p0, p1 - 2, p1) as u32);
        let inv012_m = pr[2].to_mont(powmod(p0 * p1 % p2, p2 - 2, p2) as u32);
        PadCtx { pr, inv01_m, inv012_m }
    })
}

/// Decrypt one 16-row block of one group: `hint[i]` the compact hint in hint form mod prime `i`
/// (`N` values `< p`), `t_hat[i]` the forward NTT of `t'` mod prime `i` (`< p`), `c0r` the provider's
/// reply rounded to `yb` bits. Writes `out[u * m + row0 + j] = mask of (slot u, row row0 + j)` for the
/// first `slots` slots. `scratch` holds `2 N` words; it is overwritten (wipe it after the last call).
#[allow(clippy::too_many_arguments)]
pub fn decrypt_block(
    ctx: &ReplyCtx,
    hint: [&[u32]; 2],
    t_hat: [&[u32]; 2],
    c0r: &[u64],
    yb: u32,
    slots: usize,
    m: usize,
    row0: usize,
    out: &mut [u32],
    scratch: &mut [u32],
) {
    let (s0, s1) = scratch.split_at_mut(N);
    for (i, buf) in [s0, s1].into_iter().enumerate() {
        let pr = &ctx.pr[i];
        for ((d, &h), &t) in buf.iter_mut().zip(hint[i]).zip(t_hat[i]) {
            *d = pr.mul(h, t);
        }
        pr.inv(buf);
    }
    let (p0, p1) = (ctx.pr[0].p, ctx.pr[1].p);
    let c0scale = (2.0f64).powi(32 - yb as i32);
    for u in 0..slots {
        let dst = &mut out[u * m + row0..u * m + row0 + ROWS];
        for (j, d) in dst.iter_mut().enumerate() {
            let c = ROWS * u + j;
            let r0 = ctx.pr[0].reduce(red2(scratch[c], 2 * p0));
            let r1 = ctx.pr[1].reduce(red2(scratch[N + c], 2 * p1));
            let r0m1 = if r0 >= p1 { r0 - p1 } else { r0 };
            let k = ctx.pr[1].reduce(ctx.pr[1].mul(r1.wrapping_add(p1).wrapping_sub(r0m1), ctx.inv01_m));
            let w = (r0 as u64).wrapping_add((p0 as u64).wrapping_mul(k as u64)); // c1 t' mod q1, in [0, q1)
            let x = c0r[c] as f64 * c0scale + w as f64 * ctx.scale;
            *d = (x.round() as u64 & 0xFFFF_FFFF) as u32;
        }
    }
}

/// Upload pad of one 16-feature block: `a_hat[i]` the public `a_C` in hint form (`N^-1 R` folded)
/// mod prime `i`, `t_hat[i]` the forward NTT of the upload secret. Writes `a_C t mod 2^56`.
pub fn pad_block(ctx: &PadCtx, a_hat: [&[u32]; 3], t_hat: [&[u32]; 3], out: &mut [u64], scratch: &mut [u32]) {
    for i in 0..3 {
        let pr = &ctx.pr[i];
        let buf = &mut scratch[i * N..(i + 1) * N];
        for ((d, &h), &t) in buf.iter_mut().zip(a_hat[i]).zip(t_hat[i]) {
            *d = pr.mul(h, t);
        }
        pr.inv(buf);
    }
    let (p0, p1, p2) = (ctx.pr[0].p, ctx.pr[1].p, ctx.pr[2].p);
    let p01 = p0 as u64 * p1 as u64;
    let e = (p01 as u128 * p2 as u128) as u64; // E mod 2^64 (wrapping use)
    for (c, o) in out.iter_mut().enumerate() {
        let r0 = ctx.pr[0].reduce(red2(scratch[c], 2 * p0));
        let r1 = ctx.pr[1].reduce(red2(scratch[N + c], 2 * p1));
        let r2 = ctx.pr[2].reduce(red2(scratch[2 * N + c], 2 * p2));
        let r0m1 = if r0 >= p1 { r0 - p1 } else { r0 };
        let k1 = ctx.pr[1].reduce(ctx.pr[1].mul(r1.wrapping_add(p1).wrapping_sub(r0m1), ctx.inv01_m));
        // x01 = r0 + p0 k1 (< p0 p1), reduced mod p2
        let x01 = (r0 as u64).wrapping_add((p0 as u64).wrapping_mul(k1 as u64));
        let x01m2 = (x01 % p2 as u64) as u32;
        let k2 = ctx.pr[2].reduce(ctx.pr[2].mul(r2.wrapping_add(p2).wrapping_sub(x01m2), ctx.inv012_m));
        let mut x = x01.wrapping_add(p01.wrapping_mul(k2 as u64));
        if k2 > p2 / 2 {
            x = x.wrapping_sub(e);
        }
        *o = x & ((1u64 << 56) - 1);
    }
}

/// Forward NTT of a ternary secret mod prime `pr` (output `< p`).
pub fn secret_hat(pr: &Prime, t: &[i8]) -> Vec<u32> {
    let mut v: Vec<u32> = t.iter().map(|&x| if x < 0 { pr.p - 1 } else { x as u32 }).collect();
    pr.fwd(&mut v);
    v.iter_mut().for_each(|x| *x = pr.reduce(red2(*x, 2 * pr.p)));
    v
}

/// Plain coefficients `< p` -> hint form in the NTT domain (forward NTT, then `N^-1 R`).
pub fn hint_form(pr: &Prime, coeffs: &[u32]) -> Vec<u32> {
    let mut v = coeffs.to_vec();
    pr.fwd(&mut v);
    v.iter_mut().for_each(|x| *x = pr.to_hint_form(pr.reduce(red2(*x, 2 * pr.p))));
    v
}

/// Lanes of the batched kernels: 16 independent polynomials interleaved coefficient-major
/// (`x[c * LANES + l]`), so every NTT stage is a contiguous 16-lane loop (one AVX-512 vector of u32).
pub const LANES: usize = 16;

impl Prime {
    /// [`Prime::inv`] on `LANES` interleaved polynomials (`N * LANES` words).
    pub fn inv16(&self, a: &mut [u32]) {
        let (p, pn, p2) = (self.p, self.pn, 2 * self.p);
        let mut len = 1usize;
        while len < N {
            let nb = N / (2 * len);
            for b in 0..nb {
                let z = self.izm[nb + b];
                let base = 2 * b * len * LANES;
                let (lo, hi) = a[base..base + 2 * len * LANES].split_at_mut(len * LANES);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = red2(u.wrapping_add(w), p2);
                    *y = mont(u.wrapping_add(p2).wrapping_sub(w), z, p, pn);
                }
            }
            len *= 2;
        }
    }
}

/// [`decrypt_block`] for `LANES` blocks of one group at once (blocks `blk0 .. blk0 + 16`): `hint16[i]`
/// is the hint of those blocks interleaved (`N * LANES`, hint form), `t_hat[i]` the group's secret,
/// `c0r16` the reply interleaved the same way. Writes `out[u * m + 16 * (blk0 + l) + j]`.
#[allow(clippy::too_many_arguments)]
pub fn decrypt_block16(
    ctx: &ReplyCtx,
    hint16: [&[u32]; 2],
    t_hat: [&[u32]; 2],
    c0r16: &[u64],
    yb: u32,
    slots: usize,
    m: usize,
    blk0: usize,
    out: &mut [u32],
    scratch: &mut [u32],
) {
    let (s0, s1) = scratch.split_at_mut(N * LANES);
    for (i, buf) in [s0, s1].into_iter().enumerate() {
        let pr = &ctx.pr[i];
        let (p, pn) = (pr.p, pr.pn);
        for c in 0..N {
            let t = t_hat[i][c];
            let src = &hint16[i][c * LANES..(c + 1) * LANES];
            let dst = &mut buf[c * LANES..(c + 1) * LANES];
            for (d, &h) in dst.iter_mut().zip(src) {
                *d = mont(h, t, p, pn);
            }
        }
        pr.inv16(buf);
    }
    let (p0, p1) = (ctx.pr[0].p, ctx.pr[1].p);
    let (pn1, inv) = (ctx.pr[1].pn, ctx.inv01_m);
    let c0scale = (2.0f64).powi(32 - yb as i32);
    let mut row = [0u32; LANES];
    for u in 0..slots {
        for j in 0..ROWS {
            let c = ROWS * u + j;
            let a0 = &scratch[c * LANES..(c + 1) * LANES];
            let a1 = &scratch[N * LANES + c * LANES..N * LANES + (c + 1) * LANES];
            let cr = &c0r16[c * LANES..(c + 1) * LANES];
            for l in 0..LANES {
                let r0 = red2(red2(a0[l], 2 * p0), p0);
                let r1 = red2(red2(a1[l], 2 * p1), p1);
                let r0m1 = red2(r0, p1);
                let k = red2(mont(r1.wrapping_add(p1).wrapping_sub(r0m1), inv, p1, pn1), p1);
                let w = (r0 as u64).wrapping_add((p0 as u64).wrapping_mul(k as u64));
                let x = cr[l] as f64 * c0scale + w as f64 * ctx.scale;
                row[l] = ((x + 0.5) as u64 & 0xFFFF_FFFF) as u32; // x >= 0: truncation = round half up
            }
            for (l, &v) in row.iter().enumerate() {
                out[u * m + ROWS * (blk0 + l) + j] = v;
            }
        }
    }
}

/// Interleave `LANES` polynomials (`polys[l]`, `N` each) coefficient-major.
pub fn interleave<T: Copy + Default>(polys: &[&[T]]) -> Vec<T> {
    let mut v = vec![T::default(); N * LANES];
    for (l, p) in polys.iter().enumerate() {
        for c in 0..N {
            v[c * LANES + l] = p[c];
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
    }

    fn naive_mod(a: &[u128], t: &[i8], modulus: u128) -> Vec<u128> {
        let mut acc = vec![0i128; N];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &tj) in t.iter().enumerate() {
                if tj == 0 {
                    continue;
                }
                let v = (ai % modulus) as i128 * tj as i128;
                if i + j < N {
                    acc[i + j] += v;
                } else {
                    acc[i + j - N] -= v;
                }
            }
        }
        acc.iter().map(|&v| v.rem_euclid(modulus as i128) as u128).collect()
    }

    fn ternary(g: &mut Lcg) -> Vec<i8> {
        (0..N).map(|_| (g.next() % 3) as i8 - 1).collect()
    }

    #[test]
    fn primes_match_python_prototype() {
        assert_eq!(ntt_primes(24, 2), vec![16760833, 16736257]);
        assert_eq!(ntt_primes(30, 2), vec![1073692673, 1073668097]);
    }

    #[test]
    fn ntt_round_trip() {
        let pr = Prime::new(ntt_primes(24, 1)[0]);
        let mut g = Lcg(1);
        let a: Vec<u32> = (0..N).map(|_| (g.next() % pr.p as u64) as u32).collect();
        let mut b = a.clone();
        pr.fwd(&mut b);
        pr.inv(&mut b);
        let ninv = powmod(N as u64, pr.p as u64 - 2, pr.p as u64);
        for (x, y) in a.iter().zip(&b) {
            let y = pr.reduce(red2(*y, 2 * pr.p)) as u64 * ninv % pr.p as u64;
            assert_eq!(*x as u64, y);
        }
    }

    #[test]
    fn decrypt_block_exact() {
        let ctx = reply_ctx();
        let q1 = ctx.q1 as u128;
        let yb = 36u32;
        let mut g = Lcg(7);
        let t = ternary(&mut g);
        let c1: Vec<u128> = (0..N).map(|_| g.next() as u128 % q1).collect();
        let msg: Vec<u64> = (0..N).map(|_| g.next() & 0xFFFF_FFFF).collect();
        let c1t = naive_mod(&c1, &t, q1);
        // c0 = round(q1 m / 2^32) + noise - c1 t', noise up to q1 / 2^36
        let c0r: Vec<u64> = (0..N)
            .map(|c| {
                let noise = (g.next() % (1 << 12)) as i128 - (1 << 11);
                let base = ((q1 * msg[c] as u128 + (1u128 << 31)) >> 32) as i128;
                let c0 = (base + noise - c1t[c] as i128).rem_euclid(q1 as i128) as u128;
                (((c0 << yb) + q1 / 2) / q1) as u64 & ((1 << yb) - 1)
            })
            .collect();
        let hint: Vec<Vec<u32>> = (0..2)
            .map(|i| {
                let pr = &ctx.pr[i];
                hint_form(pr, &c1.iter().map(|&x| (x % pr.p as u128) as u32).collect::<Vec<_>>())
            })
            .collect();
        let th: Vec<Vec<u32>> = (0..2).map(|i| secret_hat(&ctx.pr[i], &t)).collect();
        let m = 32usize;
        let mut out = vec![0u32; SLOTS * m];
        let mut scratch = vec![0u32; 2 * N];
        decrypt_block(ctx, [&hint[0], &hint[1]], [&th[0], &th[1]], &c0r, yb, SLOTS, m, 16, &mut out, &mut scratch);
        for u in 0..SLOTS {
            for j in 0..ROWS {
                assert_eq!(out[u * m + 16 + j] as u64, msg[ROWS * u + j], "slot {u} row {j}");
            }
        }
    }

    #[test]
    fn batched_decrypt_matches_single() {
        let ctx = reply_ctx();
        let mut g = Lcg(11);
        let t = ternary(&mut g);
        let th: Vec<Vec<u32>> = (0..2).map(|i| secret_hat(&ctx.pr[i], &t)).collect();
        let m = LANES * ROWS;
        let hints: Vec<Vec<Vec<u32>>> = (0..2)
            .map(|i| {
                let pr = &ctx.pr[i];
                (0..LANES)
                    .map(|_| hint_form(pr, &(0..N).map(|_| (g.next() % pr.p as u64) as u32).collect::<Vec<_>>()))
                    .collect()
            })
            .collect();
        let c0: Vec<Vec<u64>> = (0..LANES).map(|_| (0..N).map(|_| g.next() & ((1 << 36) - 1)).collect()).collect();
        let mut want = vec![0u32; SLOTS * m];
        let mut scratch = vec![0u32; 2 * N * LANES];
        for b in 0..LANES {
            decrypt_block(
                ctx,
                [&hints[0][b], &hints[1][b]],
                [&th[0], &th[1]],
                &c0[b],
                36,
                SLOTS,
                m,
                ROWS * b,
                &mut want,
                &mut scratch,
            );
        }
        let h16: Vec<Vec<u32>> =
            (0..2).map(|i| interleave(&hints[i].iter().map(|v| v.as_slice()).collect::<Vec<_>>())).collect();
        let c16 = interleave(&c0.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
        let mut got = vec![0u32; SLOTS * m];
        decrypt_block16(ctx, [&h16[0], &h16[1]], [&th[0], &th[1]], &c16, 36, SLOTS, m, 0, &mut got, &mut scratch);
        assert_eq!(got, want);
    }

    #[test]
    fn pad_block_exact() {
        let ctx = pad_ctx();
        let mut g = Lcg(9);
        let t = ternary(&mut g);
        let a: Vec<u128> = (0..N).map(|_| (g.next() as u128) & ((1 << 56) - 1)).collect();
        let want = naive_mod(&a, &t, 1u128 << 56);
        let ah: Vec<Vec<u32>> = (0..3)
            .map(|i| {
                let pr = &ctx.pr[i];
                hint_form(pr, &a.iter().map(|&x| (x % pr.p as u128) as u32).collect::<Vec<_>>())
            })
            .collect();
        let th: Vec<Vec<u32>> = (0..3).map(|i| secret_hat(&ctx.pr[i], &t)).collect();
        let mut out = vec![0u64; N];
        let mut scratch = vec![0u32; 3 * N];
        pad_block(ctx, [&ah[0], &ah[1], &ah[2]], [&th[0], &th[1], &th[2]], &mut out, &mut scratch);
        for c in 0..N {
            assert_eq!(out[c] as u128, want[c], "coeff {c}");
        }
    }
}
