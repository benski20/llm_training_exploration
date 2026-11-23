//! The `pks-v3` kernels (numpy reference: `pgw.hemask.pks`).
//!
//! Customer: [`encrypt_item`] (ring-LWE upload, today's wire layout), [`reply_key`] (key-switching key and
//! the NTT of the reply secret), [`decrypt_item`] (packed replies + compact hint -> `W8 r mod 2^32`).
//! Provider and hint build: [`keyswitch`] (modulus switch, packing, hybrid key switch, ModDown; reply
//! rounding or the hint's NTT form).
//!
//! Rings: upload `Z[Y]/(Y^4096+1)` mod `2^56` (pad primes `EP`, exact CRT), reply `Z[Z]/(Z^8192+1)`
//! mod `q1 = p0 p1` (`Q1`, two ~2^24 primes) with special modulus `P = p2 p3` (`PS`, two ~2^30 primes);
//! all reply primes are `1 mod 16384`. Secrets: ternary; noise: CBD(21); streams: ChaCha20 keyed by
//! 32-byte stream keys the caller derives (`SeedExpander.stream_key`), so every sample equals the
//! numpy reference's.

use crate::arith::{mont, ntt_primes, powmod, red, Isa, Prime, L};
use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use chacha20::ChaCha20;
use std::sync::OnceLock;
use zeroize::Zeroize;

/// Slots per group (module degree the provider sees).
pub const D: usize = 256;
/// Upload ring degree and module rank (`NU = RANK * D`).
pub const NU: usize = 4096;
pub const RANK: usize = 16;
/// Reply ring degree (production) and rows packed per reply ciphertext.
pub const NR: usize = 8192;
pub const PACK: usize = NR / D;
/// Mask ring and upload modulus.
pub const K: u32 = 32;
pub const QBITS: u32 = 56;
pub const ETA: u32 = 21;
const M56: u64 = (1u64 << 56) - 1;

pub struct Ctx {
    /// `Q1` then `PS` (degree `NR`).
    pub qp: [Prime; 4],
    /// Pad primes (degree `NU`).
    pub ep: [Prime; 3],
    pub q1: u64,
    pub pmod: u64,
    /// `p0^-1 mod p1`, Montgomery form.
    inv01_m: u32,
    /// pad CRT: `e0^-1 mod e1` (Montgomery), `(e0 e1)^-1 R^2 mod e2`.
    einv01_m: u32,
    einv012_r2: u32,
}

pub fn ctx() -> &'static Ctx {
    static C: OnceLock<Ctx> = OnceLock::new();
    C.get_or_init(|| {
        let q = ntt_primes(24, 2, 2 * NR as u64);
        let ps = ntt_primes(30, 2, 2 * NR as u64);
        let e = ntt_primes(30, 3, 2 * NU as u64);
        let qp = [Prime::new(q[0], NR), Prime::new(q[1], NR), Prime::new(ps[0], NR), Prime::new(ps[1], NR)];
        let ep = [Prime::new(e[0], NU), Prime::new(e[1], NU), Prime::new(e[2], NU)];
        let (p0, p1) = (q[0] as u64, q[1] as u64);
        let pmod = ps[0] as u64 * ps[1] as u64;
        let inv01_m = qp[1].to_mont(powmod(p0, p1 - 2, p1) as u32);
        let (e0, e1, e2) = (e[0] as u64, e[1] as u64, e[2] as u64);
        let einv01_m = ep[1].to_mont(powmod(e0, e1 - 2, e1) as u32);
        let r2e2 = ((1u128 << 64) % e2 as u128) as u64;
        let einv012_r2 = (powmod(e0 * e1 % e2, e2 - 2, e2) * r2e2 % e2) as u32;
        Ctx { q1: p0 * p1, pmod, inv01_m, einv01_m, einv012_r2, qp, ep }
    })
}

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

/// `Vec` wiped on drop (secrets and secret-dependent scratch).
pub struct Wiping<T: Zeroize>(pub Vec<T>);

impl<T: Zeroize> Drop for Wiping<T> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// Raw pointer for disjoint writes from scoped threads.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

fn keystream(key: &[u8; 32], offset: u64, out: &mut [u8]) {
    let mut c = ChaCha20::new(key.into(), &[0u8; 12].into());
    c.seek(offset);
    out.fill(0);
    c.apply_keystream(out);
}

// ---- sampling (same streams and values as pgw.hemask.lwe._ternary / _cbd) -------------------------

/// `n` uniform ternary values from the stream `key` (`_ternary`'s first stream: bytes < 252 kept, mod 3,
/// minus 1). Rejection reveals only how many bytes were skipped, never a kept value. `None` if the
/// first `2n + 64` bytes do not suffice (probability < 2^-1000; the caller falls back to the reference).
pub fn ternary(key: &[u8; 32], n: usize) -> Option<Wiping<i8>> {
    let mut raw = Wiping(vec![0u8; 2 * n + 64]);
    keystream(key, 0, &mut raw.0);
    let mut out = Wiping(vec![0i8; n]);
    let mut k = 0usize;
    for &b in raw.0.iter() {
        if k == n {
            break;
        }
        let keep = (b < 252) as usize;
        out.0[k.min(n - 1)] = if keep == 1 { (b % 3) as i8 - 1 } else { out.0[k.min(n - 1)] };
        k += keep;
    }
    if k < n {
        return None;
    }
    Some(out)
}

/// CBD(eta) values `count` starting at value index `start` of the stream `key` (8 bytes per value:
/// `popcount(w & mask) - popcount((w >> eta) & mask)`).
fn cbd_into(key: &[u8; 32], start: usize, out: &mut [i32]) {
    let mut raw = Wiping(vec![0u8; 8 * out.len()]);
    keystream(key, 8 * start as u64, &mut raw.0);
    let mask = (1u64 << ETA) - 1;
    for (o, ch) in out.iter_mut().zip(raw.0.chunks_exact(8)) {
        let w = u64::from_le_bytes(ch.try_into().unwrap());
        *o = (w & mask).count_ones() as i32 - ((w >> ETA) & mask).count_ones() as i32;
    }
}

// ---- customer: upload encryption ----------------------------------------------------------------

/// Forward NTT of a ternary polynomial mod `pr`, reduced to `[0, p)`.
fn ternary_hat(pr: &Prime, t: &[i8]) -> Vec<u32> {
    let mut v: Vec<u32> = t.iter().map(|&x| pr.res_small(x as i32)).collect();
    pr.fwd(&mut v);
    v.iter_mut().for_each(|x| *x = pr.reduce(*x));
    v
}

/// One encryption item: a linear with `n` inputs, `groups` 256-slot groups.
pub struct EncItem<'a> {
    /// Stream key of the linear's public ring elements `a_C` (`pgw/hemask/pks-v3/A`, linear).
    pub a_key: [u8; 32],
    pub n: usize,
    /// Masks `r`, `rows x n` (values mod 2^32 in the low bits), `rows <= 256 groups`.
    pub r: &'a [i64],
    pub rows: usize,
    /// Upload secrets, `groups x NU` ternary.
    pub t: &'a [i8],
    /// Noise stream keys per group (`pks-e`).
    pub e_keys: &'a [[u8; 32]],
}

#[inline(always)]
fn enc_chunk(cx: &Ctx, it: &EncItem<'_>, c0: usize, that: &[Vec<Vec<u32>>], b: Shared<u64>, a16: &mut [u32], s16: &mut [u32]) {
    let nblk = it.n / RANK;
    let groups = it.e_keys.len();
    let cols = groups * D;
    let lanes = (nblk - c0).min(L);
    // a_C of blocks c0 .. c0 + lanes, interleaved: residues (x R^-1), forward NTT, hint form
    let mut raw = vec![0u8; lanes * NU * 8];
    keystream(&it.a_key, (c0 * NU * 8) as u64, &mut raw);
    for (i, pr) in cx.ep.iter().enumerate() {
        let a = &mut a16[i * NU * L..(i + 1) * NU * L];
        for c in 0..NU {
            for l in 0..L {
                a[c * L + l] = if l < lanes {
                    let o = (l * NU + c) * 8;
                    pr.res56(u64::from_le_bytes(raw[o..o + 8].try_into().unwrap()) & M56)
                } else {
                    0
                };
            }
        }
        pr.fwd_l(a);
        for x in a.iter_mut() {
            *x = pr.reduce(pr.mul(*x, pr.ninv_r3));
        }
    }
    let (e0, e1, e2) = (cx.ep[0].p, cx.ep[1].p, cx.ep[2].p);
    let (pn1, pn2) = (cx.ep[1].pn, cx.ep[2].pn);
    let p01 = e0 as u64 * e1 as u64;
    let emod = (p01 as u128 * e2 as u128) as u64; // E mod 2^64
    let half2 = e2 / 2;
    let mut noise = Wiping(vec![0i32; L * RANK * D]);
    for g in 0..groups {
        for i in 0..3 {
            let pr = &cx.ep[i];
            let (p, pn) = (pr.p, pr.pn);
            let (a, s) = (&a16[i * NU * L..(i + 1) * NU * L], &mut s16[i * NU * L..(i + 1) * NU * L]);
            let th = &that[g][i];
            for c in 0..NU {
                let t = th[c];
                for l in 0..L {
                    s[c * L + l] = mont(a[c * L + l], t, p, pn);
                }
            }
            pr.inv_l(s);
        }
        // noise of the chunk's rows (row f = 16 (c0 + l) + k is stream values f*256 .. f*256 + 255)
        cbd_into(&it.e_keys[g], c0 * RANK * D, &mut noise.0[..lanes * RANK * D]);
        // u outer, then (k, lane): contiguous NTT-output reads, row-local writes and r reads
        for u in 0..D {
            let row = g * D + u;
            let rrow = if row < it.rows { Some(&it.r[row * it.n..(row + 1) * it.n]) } else { None };
            for k in 0..RANK {
                let cb = (RANK * u + k) * L;
                for l in 0..lanes {
                    let c = cb + l;
                    let r0 = red(red(s16[c], 2 * e0), e0);
                    let r1 = red(red(s16[NU * L + c], 2 * e1), e1);
                    let r2 = s16[2 * NU * L + c]; // [0, 2 e2)
                    let r0m1 = red(r0, e1);
                    let k1 = red(mont(r1.wrapping_add(e1).wrapping_sub(r0m1), cx.einv01_m, e1, pn1), e1);
                    let x01 = (r0 as u64).wrapping_add((e0 as u64).wrapping_mul(k1 as u64)); // < e0 e1
                    let y = cx.ep[2].res56(x01); // x01 R^-1 (x01 < 2^60: hi < 2^28 < e2), [0, 2 e2)
                    let r2r = mont(r2, 1, e2, pn2); // r2 R^-1, [0, e2]
                    let z = r2r.wrapping_add(2 * e2).wrapping_sub(y); // (r2 - x01) R^-1, < 4 e2
                    let k2 = red(mont(z, cx.einv012_r2, e2, pn2), e2);
                    let mut x = x01.wrapping_add(p01.wrapping_mul(k2 as u64));
                    let neg = ((half2 as i64 - k2 as i64) >> 63) as u64; // all ones when k2 > e2/2
                    x = x.wrapping_sub(emod & neg);
                    let f = (c0 + l) * RANK + k;
                    let msg = match rrow { Some(rr) => (rr[f] as u64) & 0xFFFF_FFFF, None => 0 };
                    let e = noise.0[(l * RANK + k) * D + u];
                    let v = x.wrapping_add(e as i64 as u64).wrapping_add(msg << (QBITS - K)) & M56;
                    // SAFETY: rows 16 c0 .. 16 (c0 + lanes) of b are written by this chunk only.
                    unsafe { *b.0.add(f * cols + g * D + u) = v };
                }
            }
        }
    }
}

#[inline(always)]
fn enc_worker(it: &EncItem<'_>, that: &[Vec<Vec<u32>>], bp: Shared<u64>, th: usize, threads: usize) {
    let cx = ctx();
    let chunks = (it.n / RANK).div_ceil(L);
    let mut a16 = Wiping(vec![0u32; 3 * NU * L]);
    let mut s16 = Wiping(vec![0u32; 3 * NU * L]);
    for ch in (th..chunks).step_by(threads) {
        enc_chunk(cx, it, ch * L, that, bp, &mut a16.0, &mut s16.0);
    }
}

crate::multiversion!(fn enc_worker_isa(it: &EncItem<'_>, that: &[Vec<Vec<u32>>], bp: Shared<u64>, th: usize,
    threads: usize) => enc_worker);

fn encrypt_inner(isa: Isa, it: &EncItem<'_>, b: &mut [u64], threads: usize) {
    let cx = ctx();
    let groups = it.e_keys.len();
    let mut that: Vec<Vec<Vec<u32>>> = (0..groups)
        .map(|g| cx.ep.iter().map(|pr| ternary_hat(pr, &it.t[g * NU..(g + 1) * NU])).collect())
        .collect();
    let chunks = (it.n / RANK).div_ceil(L);
    let bp = Shared(b.as_mut_ptr());
    let threads = threads.clamp(1, chunks.max(1));
    std::thread::scope(|s| {
        for th in 0..threads {
            let that = &that;
            s.spawn(move || enc_worker_isa(isa, it, that, bp, th, threads));
        }
    });
    for g in that.iter_mut() {
        for v in g.iter_mut() {
            v.zeroize();
        }
    }
}

/// Ring-LWE upload of one item: writes `b` (`n x 256 groups`, mod 2^56): `b[16 C + k, 256 g + u] =
/// (a_C t_g)[16 u + k] + e_g[16 C + k, u] + 2^24 r[256 g + u, 16 C + k]`.
pub fn encrypt_item(isa: Isa, it: &EncItem<'_>, b: &mut [u64], threads: usize) -> Result<(), Error> {
    let groups = it.e_keys.len();
    if it.n == 0 || it.n % RANK != 0 || groups == 0 || it.rows > groups * D || it.r.len() != it.rows * it.n
        || it.t.len() != groups * NU || b.len() != it.n * groups * D
    {
        return err("encrypt_item: inconsistent shapes");
    }
    encrypt_inner(isa, it, b, threads);
    Ok(())
}

// ---- customer: reply key -------------------------------------------------------------------------

/// The public key-switching `a`-parts `k_l` (NTT domain, `4 x RANK x NR`, `< p`) in Montgomery form.
pub fn k_mont(k_hat: &[u32]) -> Result<Vec<u32>, Error> {
    let cx = ctx();
    if k_hat.len() != 4 * RANK * NR {
        return err("k_hat must be 4 x 16 x 8192");
    }
    let mut out = vec![0u32; k_hat.len()];
    for i in 0..4 {
        let pr = &cx.qp[i];
        for (o, &v) in out[i * RANK * NR..(i + 1) * RANK * NR].iter_mut().zip(&k_hat[i * RANK * NR..(i + 1) * RANK * NR]) {
            if v >= pr.p {
                return err("k_hat value out of range");
            }
            *o = pr.to_mont(v);
        }
    }
    Ok(out)
}

#[inline(always)]
fn ksk_inner(km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32]) {
    let cx = ctx();
    let mut e = Wiping(vec![0i32; RANK * NR]);
    cbd_into(e_key, 0, &mut e.0);
    let mut m16 = Wiping(vec![0u32; NR * L]);
    for i in 0..4 {
        let pr = &cx.qp[i];
        let th = Wiping(ternary_hat(pr, tr));
        if i < 2 {
            trhat[i * NR..(i + 1) * NR].copy_from_slice(&th.0);
        }
        // m_l = e_l + P iota(t_l): iota(t_l)[32 u] = t[16 u + l]
        let pm = (cx.pmod % pr.p as u64) as u32;
        for c in 0..NR {
            for l in 0..L {
                let mut v = pr.res_small(e.0[l * NR + c]);
                if c % PACK == 0 {
                    let tv = pr.res_small(t[RANK * (c / PACK) + l] as i32); // [0, p)
                    v = pr.reduce(v.wrapping_add(pr.reduce(pr.mul(pr.mul(tv, pm), pr.r2))));
                }
                m16.0[c * L + l] = v;
            }
        }
        pr.fwd_l(&mut m16.0);
        let (p, pn) = (pr.p, pr.pn);
        for l in 0..RANK {
            let kk = &km[(i * RANK + l) * NR..(i * RANK + l + 1) * NR];
            let out = &mut beta[(i * RANK + l) * NR..(i * RANK + l + 1) * NR];
            for c in 0..NR {
                let kt = mont(kk[c], th.0[c], p, pn); // k t' (k in Montgomery form), [0, 2p)
                out[c] = red(red(kt.wrapping_add(m16.0[c * L + l]), 2 * p), p);
            }
        }
    }
}

crate::multiversion!(fn ksk_isa(km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32]) => ksk_inner);

/// Reply key of one group: `beta_l = k_l t' + e_l + P iota(t_l) mod q1 P` in the NTT domain
/// (`beta`: `4 x RANK x NR`, `< p`), and the NTT of `t'` mod the two `Q1` primes (`trhat`: `2 x NR`, `< p`).
pub fn reply_key(isa: Isa, km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32])
    -> Result<(), Error> {
    if km.len() != 4 * RANK * NR || t.len() != NU || tr.len() != NR || beta.len() != 4 * RANK * NR
        || trhat.len() != 2 * NR
    {
        return err("reply_key: inconsistent shapes");
    }
    ksk_isa(isa, km, t, tr, e_key, beta, trhat);
    Ok(())
}

// ---- customer: decryption ---------------------------------------------------------------------------

/// The reply values: unpacked words, or the `pack_bits` layout at 36 bits (a u32 field, then a nibble
/// stream) read in place.
#[derive(Clone, Copy)]
pub enum C0<'a> {
    Words(&'a [u64]),
    Packed36 { data: &'a [u8], count: usize },
}

/// Decryption of one item: `m` output rows (`nb = m / 32` blocks), `groups` groups.
pub struct DecItem<'a> {
    /// Compact hint in NTT hint form: `(nb, 2, NR, 3)` bytes (24-bit little-endian residues `< p`).
    pub hint: &'a [u8],
    pub m: usize,
    /// Reply `c0` rounded to `yb` bits, `(groups, nb, NR)`.
    pub c0r: C0<'a>,
    pub yb: u32,
    /// NTT of `t'_g` mod the two `Q1` primes, `(groups, 2, NR)`, `< p`.
    pub trhat: &'a [u32],
    /// Valid slots per group (`<= 256`).
    pub rows: &'a [usize],
}

#[inline(always)]
fn dec_chunk(cx: &Ctx, it: &DecItem<'_>, b0: usize, outs: &[Shared<u32>], h16: &mut [u32], s16: &mut [u32]) {
    let nb = it.m / PACK;
    let lanes = (nb - b0).min(L);
    for i in 0..2 {
        let h = &mut h16[i * NR * L..(i + 1) * NR * L];
        for l in 0..L {
            if l < lanes {
                let src = &it.hint[((b0 + l) * 2 + i) * NR * 3..((b0 + l) * 2 + i + 1) * NR * 3];
                for c in 0..NR {
                    h[c * L + l] = src[3 * c] as u32 | (src[3 * c + 1] as u32) << 8 | (src[3 * c + 2] as u32) << 16;
                }
            } else {
                for c in 0..NR {
                    h[c * L + l] = 0;
                }
            }
        }
    }
    let (p0, p1) = (cx.qp[0].p, cx.qp[1].p);
    let pn1 = cx.qp[1].pn;
    let c0scale = 2f64.powi(K as i32 - it.yb as i32);
    let wscale = (1u64 << K) as f64 / cx.q1 as f64;
    let mut tmp = [0u32; L * PACK];
    let mut wv = [0u64; L * PACK];
    for (g, &rows) in it.rows.iter().enumerate() {
        for i in 0..2 {
            let pr = &cx.qp[i];
            let th = &it.trhat[(g * 2 + i) * NR..(g * 2 + i + 1) * NR];
            pr.inv_l_mul(&mut s16[i * NR * L..(i + 1) * NR * L], &h16[i * NR * L..(i + 1) * NR * L], th);
        }
        let gbase = g * nb * NR;
        let out = outs[g];
        for u in 0..rows {
            // c1 t' mod q1 in (j, lane) order: contiguous, vectorised
            let a0 = &s16[PACK * u * L..PACK * (u + 1) * L];
            let a1 = &s16[NR * L + PACK * u * L..NR * L + PACK * (u + 1) * L];
            for ((w, &x0), &x1) in wv.iter_mut().zip(a0).zip(a1) {
                let r0 = red(red(x0, 2 * p0), p0);
                let r1 = red(red(x1, 2 * p1), p1);
                let r0m1 = red(r0, p1);
                let k = red(mont(r1.wrapping_add(p1).wrapping_sub(r0m1), cx.inv01_m, p1, pn1), p1);
                *w = (r0 as u64).wrapping_add((p0 as u64).wrapping_mul(k as u64));
            }
            for l in 0..lanes {
                let idx0 = gbase + (b0 + l) * NR + PACK * u;
                let dst = &mut tmp[l * PACK..(l + 1) * PACK];
                match it.c0r {
                    C0::Words(v) => {
                        for (j, d) in dst.iter_mut().enumerate() {
                            let x = v[idx0 + j] as f64 * c0scale + wv[j * L + l] as f64 * wscale;
                            *d = ((x + 0.5) as u64 & 0xFFFF_FFFF) as u32; // x >= 0: truncation rounds half up
                        }
                    }
                    C0::Packed36 { data, count } => {
                        let lo = &data[4 * idx0..4 * (idx0 + PACK)];
                        let hi = &data[4 * count + idx0 / 2..4 * count + (idx0 + PACK) / 2];
                        for (j, d) in dst.iter_mut().enumerate() {
                            let v = u32::from_le_bytes(lo[4 * j..4 * j + 4].try_into().unwrap()) as u64
                                | (((hi[j / 2] >> (4 * (j & 1))) & 15) as u64) << 32;
                            let x = v as f64 * c0scale + wv[j * L + l] as f64 * wscale;
                            *d = ((x + 0.5) as u64 & 0xFFFF_FFFF) as u32;
                        }
                    }
                }
            }
            // SAFETY: columns 32 b0 .. 32 (b0 + lanes) of every row are written by this chunk only.
            unsafe {
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), out.0.add(u * it.m + PACK * b0), lanes * PACK);
            }
        }
    }
    wv.zeroize();
    tmp.zeroize();
}

#[inline(always)]
fn dec_worker(it: &DecItem<'_>, ptrs: &[Shared<u32>], th: usize, threads: usize) {
    let cx = ctx();
    let chunks = (it.m / PACK).div_ceil(L);
    let mut h16 = vec![0u32; 2 * NR * L];
    let mut s16 = Wiping(vec![0u32; 2 * NR * L]);
    for ch in (th..chunks).step_by(threads) {
        dec_chunk(cx, it, ch * L, ptrs, &mut h16, &mut s16.0);
    }
}

crate::multiversion!(fn dec_worker_isa(it: &DecItem<'_>, ptrs: &[Shared<u32>], th: usize, threads: usize)
    => dec_worker);

fn decrypt_inner(isa: Isa, it: &DecItem<'_>, outs: &mut [&mut [u32]], threads: usize) {
    let chunks = (it.m / PACK).div_ceil(L);
    let ptrs: Vec<Shared<u32>> = outs.iter_mut().map(|o| Shared(o.as_mut_ptr())).collect();
    let threads = threads.clamp(1, chunks.max(1));
    std::thread::scope(|s| {
        for th in 0..threads {
            let ptrs = &ptrs;
            s.spawn(move || dec_worker_isa(isa, it, ptrs, th, threads));
        }
    });
}

/// `outs[g]` (`rows[g] x m`, row-major) `= round(2^32 (c0r_g 2^-yb + c1 t'_g / q1)) mod 2^32`.
pub fn decrypt_item(isa: Isa, it: &DecItem<'_>, outs: &mut [&mut [u32]], threads: usize) -> Result<(), Error> {
    let groups = it.rows.len();
    let nb = it.m / PACK;
    if it.m == 0 || it.m % PACK != 0 || it.hint.len() != nb * 2 * NR * 3
        || it.trhat.len() != groups * 2 * NR || outs.len() != groups || !(K..=QBITS).contains(&it.yb)
        || !match it.c0r {
            C0::Words(v) => v.len() == groups * nb * NR,
            C0::Packed36 { data, count } => it.yb == 36 && count == groups * nb * NR && data.len() == count * 36 / 8,
        }
    {
        return err("decrypt_item: inconsistent shapes");
    }
    for (o, &r) in outs.iter().zip(it.rows) {
        if r > D || o.len() != r * it.m {
            return err("decrypt_item: bad output size");
        }
    }
    decrypt_inner(isa, it, outs, threads);
    Ok(())
}

// ---- provider (and hint build): key switch -------------------------------------------------------

/// One key-switch call for one linear (`m` rows): `hs` is `round(H q1 / 2^56) mod q1` (`m x NU`, the
/// provider's precomputed switched hint), `ops` the per-group operands (`beta_g`, or the public `k` for the
/// hint build) in the NTT domain (`(groups, 4, RANK, NR)`, `< p`). With `y` (`m x 256 groups`, `W8 b mod
/// 2^56`) the output is the reply `c0` rounded to `yb` bits, `(groups, nb, NR)`; without it (hint build,
/// one group) the output is `c1` in NTT hint form, `(nb, 2, NR)` u32 residues.
pub struct KsItem<'a> {
    pub hs: &'a [u64],
    pub m: usize,
    pub ops: &'a [u32],
    pub groups: usize,
    pub y: Option<&'a [u64]>,
    pub yb: u32,
}

#[inline(always)]
fn ks_block(cx: &Ctx, it: &KsItem<'_>, b: usize, opt: &[Vec<u32>], out: Shared<u64>, a16: &mut [u32], acc: &mut [u32]) {
    let nb = it.m / PACK;
    let q1 = cx.q1;
    let half = q1.div_ceil(2);
    // A~_l[32 u + j] = hs[32 b + j, 256 l + u], centred lift, residues (x R^-1), forward NTT
    for i in 0..4 {
        let pr = &cx.qp[i];
        let q1r = pr.reduce(pr.res56(q1));
        let a = &mut a16[i * NR * L..(i + 1) * NR * L];
        for u in 0..D {
            for j in 0..PACK {
                let row = &it.hs[(PACK * b + j) * NU..(PACK * b + j + 1) * NU];
                let c = PACK * u + j;
                for l in 0..L {
                    let v = row[D * l + u];
                    let r = pr.reduce(pr.res56(v));
                    a[c * L + l] = if v >= half { red(r.wrapping_add(pr.p).wrapping_sub(q1r), pr.p) } else { r };
                }
            }
        }
        pr.fwd_l(a);
    }
    let (p0, p1) = (cx.qp[0].p as u64, cx.qp[1].p as u64);
    let (p2, p3) = (cx.qp[2].p as u64, cx.qp[3].p as u64);
    let pm = cx.pmod;
    let inv23 = powmod(p2, p3 - 2, p3);
    let pinv = [powmod(pm % p0, p0 - 2, p0), powmod(pm % p1, p1 - 2, p1)];
    let inv01 = powmod(p0, p1 - 2, p1);
    for g in 0..it.groups {
        for i in 0..4 {
            let pr = &cx.qp[i];
            let (p, pn) = (pr.p, pr.pn);
            let a = &a16[i * NR * L..(i + 1) * NR * L];
            let o = &opt[g * 4 + i];
            let s = &mut acc[i * NR..(i + 1) * NR];
            for c in 0..NR {
                let mut v = 0u32;
                for l in 0..L {
                    v = red(v.wrapping_add(mont(a[c * L + l], o[c * L + l], p, pn)), 2 * p);
                }
                s[c] = v;
            }
            pr.inv(s);
            for x in s.iter_mut() {
                *x = pr.reduce(*x);
            }
        }
        let mut res = [0u64; 4];
        for c in 0..NR {
            for i in 0..4 {
                res[i] = acc[i * NR + c] as u64;
            }
            if let Some(y) = it.y {
                // body = round(y q1 / 2^56) mod q1 at coefficient 32 u + j = row 32 b + j, slot u
                let (u, j) = (c / PACK, c % PACK);
                let yv = y[(PACK * b + j) * it.groups * D + g * D + u] & M56;
                let body = (((yv as u128 * q1 as u128) + (1u128 << 55)) >> 56) as u64 % q1;
                for i in 0..2 {
                    let pp = cx.qp[i].p as u64;
                    let pb = (body % pp) * (pm % pp) % pp;
                    res[i] = (pb + pp - res[i]) % pp;
                }
                res[2] = (p2 - res[2]) % p2;
                res[3] = (p3 - res[3]) % p3;
            }
            // ModDown: [C]_P centred, then (r_i - [C]_P) P^-1 mod p_i
            let k = ((res[3] + p3 - res[2] % p3) % p3) * inv23 % p3;
            let mut cp = (res[2] + p2 * k) as i64;
            if cp >= pm.div_ceil(2) as i64 {
                cp -= pm as i64;
            }
            let mut o = [0u64; 2];
            for i in 0..2 {
                let pp = cx.qp[i].p as u64;
                let cpm = cp.rem_euclid(pp as i64) as u64;
                o[i] = (res[i] + pp - cpm) % pp * pinv[i] % pp;
            }
            if it.y.is_some() {
                let kk = (o[1] + p1 - o[0] % p1) % p1 * inv01 % p1;
                let cv = o[0] + p0 * kk; // [0, q1)
                let num = ((cv as u128) << it.yb) + (q1 / 2) as u128;
                let mut est = (num as f64 / q1 as f64) as u64;
                while est as u128 * q1 as u128 > num {
                    est -= 1;
                }
                while (est as u128 + 1) * q1 as u128 <= num {
                    est += 1;
                }
                // SAFETY: block b of group g is written by this call only.
                unsafe { *out.0.add((g * nb + b) * NR + c) = est & ((1u64 << it.yb) - 1) };
            } else {
                for i in 0..2 {
                    acc[i * NR + c] = o[i] as u32; // c1 mod p_i (rows 0, 1 of acc are done with)
                }
            }
        }
        if it.y.is_none() {
            for i in 0..2 {
                let pr = &cx.qp[i];
                let s = &mut acc[i * NR..(i + 1) * NR];
                pr.fwd(s);
                for c in 0..NR {
                    let v = pr.reduce(pr.mul(s[c], pr.ninv_r2));
                    // SAFETY: as above
                    unsafe { *out.0.add((b * 2 + i) * NR + c) = v as u64 };
                }
            }
        }
    }
}

#[inline(always)]
fn ks_worker(it: &KsItem<'_>, opt: &[Vec<u32>], op: Shared<u64>, th: usize, threads: usize) {
    let cx = ctx();
    let mut a16 = vec![0u32; 4 * NR * L];
    let mut acc = vec![0u32; 4 * NR];
    for b in (th..it.m / PACK).step_by(threads) {
        ks_block(cx, it, b, opt, op, &mut a16, &mut acc);
    }
}

crate::multiversion!(fn ks_worker_isa(it: &KsItem<'_>, opt: &[Vec<u32>], op: Shared<u64>, th: usize, threads: usize)
    => ks_worker);

fn keyswitch_inner(isa: Isa, it: &KsItem<'_>, out: &mut [u64], threads: usize) {
    let cx = ctx();
    // operands interleaved per coefficient (lane = component l), scaled to op N^-1 R^2
    let opt: Vec<Vec<u32>> = (0..it.groups * 4)
        .map(|gi| {
            let (g, i) = (gi / 4, gi % 4);
            let pr = &cx.qp[i];
            let mut v = vec![0u32; NR * L];
            for l in 0..RANK {
                let src = &it.ops[((g * 4 + i) * RANK + l) * NR..((g * 4 + i) * RANK + l + 1) * NR];
                for c in 0..NR {
                    v[c * L + l] = pr.reduce(pr.mul(src[c], pr.ninv_r3));
                }
            }
            v
        })
        .collect();
    let nb = it.m / PACK;
    let op = Shared(out.as_mut_ptr());
    let threads = threads.clamp(1, nb.max(1));
    std::thread::scope(|s| {
        for th in 0..threads {
            let opt = &opt;
            s.spawn(move || ks_worker_isa(isa, it, opt, op, th, threads));
        }
    });
}

pub fn keyswitch(isa: Isa, it: &KsItem<'_>, out: &mut [u64], threads: usize) -> Result<(), Error> {
    let nb = it.m / PACK;
    if it.m == 0 || it.m % PACK != 0 || it.hs.len() != it.m * NU || it.groups == 0
        || it.ops.len() != it.groups * 4 * RANK * NR
    {
        return err("keyswitch: inconsistent shapes");
    }
    match it.y {
        Some(y) => {
            if y.len() != it.m * it.groups * D || out.len() != it.groups * nb * NR || !(K..=QBITS).contains(&it.yb) {
                return err("keyswitch: bad reply shapes");
            }
        }
        None => {
            if it.groups != 1 || out.len() != nb * 2 * NR {
                return err("keyswitch: the hint build takes one operand and writes (nb, 2, NR)");
            }
        }
    }
    let cx = ctx();
    for (gi, chunk) in it.ops.chunks_exact(RANK * NR).enumerate() {
        let p = cx.qp[gi % 4].p;
        if chunk.iter().any(|&v| v >= p) {
            return err("keyswitch: operand residue out of range");
        }
    }
    if it.hs.iter().any(|&v| v >= cx.q1) {
        return err("keyswitch: switched hint value out of range");
    }
    keyswitch_inner(isa, it, out, threads);
    Ok(())
}
