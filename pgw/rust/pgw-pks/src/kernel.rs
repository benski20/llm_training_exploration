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

/// ChaCha20 (zero nonce) keystream bytes `offset ..` of the stream `key`. Block-aligned ranges use the
/// 16-block AVX-512 or 8-block AVX2 copy (same bytes, tested against the `chacha20` crate).
fn keystream(fast: Vx, key: &[u8; 32], offset: u64, out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if offset % 64 == 0 && out.len() % 64 == 0 && offset / 64 + (out.len() / 64) as u64 <= 1 << 32 {
        // SAFETY: `fast` comes from `Vx::of` of an ISA capped at what the CPU supports (`Isa::parse`).
        match fast {
            Vx::Z => return unsafe { crate::simd::chacha20_blocks(key, (offset / 64) as u32, out) },
            Vx::Y => return unsafe { crate::simd2::chacha20_blocks(key, (offset / 64) as u32, out) },
            Vx::Port => {}
        }
    }
    let _ = fast;
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
    keystream(Vx::of(Isa::parse("auto").unwrap_or(Isa::Scalar)), key, 0, &mut raw.0);
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
fn cbd_into(fast: Vx, key: &[u8; 32], start: usize, out: &mut [i32]) {
    // 1 KiB of keystream (16 ChaCha blocks, 128 values) at a time: no allocation, stays in L1
    const CH: usize = 128;
    let mut raw = [0u8; 8 * CH];
    let mask = (1u64 << ETA) - 1;
    for (ci, oc) in out.chunks_mut(CH).enumerate() {
        let bytes = &mut raw[..8 * oc.len()];
        keystream(fast, key, 8 * (start + ci * CH) as u64, bytes);
        for (o, ch) in oc.iter_mut().zip(bytes.chunks_exact(8)) {
            let w = u64::from_le_bytes(ch.try_into().unwrap());
            *o = (w & mask).count_ones() as i32 - ((w >> ETA) & mask).count_ones() as i32;
        }
    }
    raw.zeroize();
    std::hint::black_box(&raw);
}

#[inline(always)]
fn cbd_kernel(fast: Vx, key: &[u8; 32], start: usize, out: &mut [i32]) {
    cbd_into(fast, key, start, out)
}

crate::multiversion!(fn cbd_isa(fast: Vx, key: &[u8; 32], start: usize, out: &mut [i32]) => cbd_kernel);

/// CBD(21) noise values `start .. start + out.len()` of the stream `key` (the upload / reply-key noise
/// sampler; benchmark and test hook). Same values for every ISA.
pub fn cbd(isa: Isa, key: &[u8; 32], start: usize, out: &mut [i32]) {
    cbd_isa(isa, Vx::of(isa), key, start, out)
}

/// ChaCha20 keystream bytes (test hook for the AVX-512 copy).
pub fn keystream_isa(isa: Isa, key: &[u8; 32], offset: u64, out: &mut [u8]) {
    keystream(Vx::of(isa), key, offset, out)
}

/// Which hand-written vector copy a kernel uses: `Z` (AVX-512), `Y` (AVX2) or the portable code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vx {
    Port,
    Y,
    Z,
}

impl Vx {
    /// The copy for an ISA that `Isa::parse` capped at what this CPU supports.
    pub fn of(isa: Isa) -> Vx {
        match isa {
            Isa::Avx512 if cfg!(target_arch = "x86_64") => Vx::Z,
            Isa::Avx2 if cfg!(target_arch = "x86_64") => Vx::Y,
            _ => Vx::Port,
        }
    }
}

/// `INTT(h * t)` without `N^-1` on `L` interleaved polynomials (hand-written AVX-512 / AVX2 copies, else
/// the portable code; identical values).
#[inline(always)]
fn inv_mul(fast: Vx, pr: &Prime, a: &mut [u32], h: &[u32], t: &[u32]) {
    // SAFETY: `fast` comes from `Vx::of` of an ISA that `Isa::parse` capped at what the CPU supports.
    #[cfg(target_arch = "x86_64")]
    match fast {
        Vx::Z => return unsafe { crate::vntt::inv_l_mul_512(pr, a, h, t) },
        Vx::Y => return unsafe { crate::vntt::inv_l_mul_256(pr, a, h, t) },
        Vx::Port => {}
    }
    let _ = fast;
    pr.inv_l_mul(a, h, t)
}

/// Forward NTT on `L` interleaved polynomials (vector copies as in `inv_mul`).
#[inline(always)]
fn fwd(fast: Vx, pr: &Prime, a: &mut [u32]) {
    // SAFETY: as in `inv_mul`.
    #[cfg(target_arch = "x86_64")]
    match fast {
        Vx::Z => return unsafe { crate::vntt::fwd_l_512(pr, a) },
        Vx::Y => return unsafe { crate::vntt::fwd_l_256(pr, a) },
        Vx::Port => {}
    }
    let _ = fast;
    pr.fwd_l_fast(a, 256)
}

/// Benchmark hook: one NTT kernel (`which % 4`: 0 portable `inv_l_mul`, 1 `inv_mul`, 2 portable `fwd_l`,
/// 3 `fwd`; `which / 4`: 0 portable, 1 AVX-512 copy, 2 AVX2 copy), `reps` times.
#[inline(always)]
fn ntt_kernel(pr: &Prime, which: u32, a: &mut [u32], h: &[u32], t: &[u32], reps: usize) {
    let fast = match which / 4 {
        1 if Isa::detect() == Isa::Avx512 => Vx::Z,
        2 if Isa::detect() != Isa::Scalar => Vx::Y,
        _ => Vx::Port,
    };
    for _ in 0..reps {
        match which % 4 {
            0 => pr.inv_l_mul(a, h, t),
            1 => inv_mul(fast, pr, a, h, t),
            2 => {
                a.copy_from_slice(h);
                pr.fwd_l(a)
            }
            _ => {
                a.copy_from_slice(h);
                fwd(fast, pr, a)
            }
        }
    }
}

crate::multiversion!(pub fn ntt_bench(pr: &Prime, which: u32, a: &mut [u32], h: &[u32], t: &[u32], reps: usize)
    => ntt_kernel);

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
#[allow(clippy::too_many_arguments)]
fn enc_chunk(fast: Vx, cx: &Ctx, it: &EncItem<'_>, c0: usize, that: &[Vec<Vec<u32>>], b: Shared<u64>, a16: &mut [u32],
             s16: &mut [u32], raw: &mut [u8], xb: &mut [u64]) {
    let nblk = it.n / RANK;
    let groups = it.e_keys.len();
    let cols = groups * D;
    let lanes = (nblk - c0).min(L);
    // a_C of blocks c0 .. c0 + lanes, interleaved: residues (x R^-1), forward NTT, hint form (public data)
    for l in 0..L {
        if l < lanes {
            keystream(fast, &it.a_key, ((c0 + l) * NU * 8) as u64, raw);
            for (c, w) in raw.chunks_exact(8).enumerate() {
                let v = u64::from_le_bytes(w.try_into().unwrap()) & M56;
                for (i, pr) in cx.ep.iter().enumerate() {
                    a16[i * NU * L + c * L + l] = pr.res56(v);
                }
            }
        } else {
            for i in 0..3 {
                for c in 0..NU {
                    a16[i * NU * L + c * L + l] = 0;
                }
            }
        }
    }
    for (i, pr) in cx.ep.iter().enumerate() {
        let a = &mut a16[i * NU * L..(i + 1) * NU * L];
        fwd(fast, pr, a);
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
            let (a, s) = (&a16[i * NU * L..(i + 1) * NU * L], &mut s16[i * NU * L..(i + 1) * NU * L]);
            inv_mul(fast, pr, s, a, &that[g][i]);
        }
        // noise of the chunk's rows (row f = 16 (c0 + l) + k is stream values f*256 .. f*256 + 255)
        cbd_into(fast, &it.e_keys[g], c0 * RANK * D, &mut noise.0[..lanes * RANK * D]);
        #[cfg(target_arch = "x86_64")]
        if fast == Vx::Z {
            let pr2 = &cx.ep[2];
            let kc = crate::simd::EncConsts {
                e: [e0, e1, e2],
                pn: [cx.ep[0].pn, pn1, pn2],
                inv01_m: cx.einv01_m,
                e0_m2: pr2.to_mont(e0 % e2),
                inv012_m: pr2.to_mont(powmod(p01 % e2 as u64, e2 as u64 - 2, e2 as u64) as u32),
                p01,
                emod,
            };
            // SAFETY: rows 16 c0 .. 16 (c0 + lanes) of b are written by this chunk only (columns g D ..
            // g D + 255); the mask gather reads only rows < it.rows of r; `fast` implies AVX-512.
            unsafe {
                crate::simd::enc_out(&kc, s16.as_ptr(), NU, lanes, c0, cols, it.n, it.r.as_ptr().wrapping_add(g * D * it.n),
                                     it.rows.saturating_sub(g * D).min(D), noise.0.as_ptr(), b.0.add(g * D));
            }
            continue;
        }
        // pass 1: the centred CRT of the pad for all 16 lanes (contiguous, vectorised over the lanes)
        for (c, xo) in xb.iter_mut().enumerate() {
            {
                {
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
                    *xo = x;
                }
            }
        }
        // pass 2: one b row segment (256 slots) per feature f = 16 (c0 + l) + k, written sequentially
        let valid = it.rows.saturating_sub(g * D).min(D);
        for l in 0..lanes {
            for k in 0..RANK {
                let f = (c0 + l) * RANK + k;
                let er = &noise.0[(l * RANK + k) * D..(l * RANK + k + 1) * D];
                let mut row = [0u64; D];
                for (u, o) in row.iter_mut().enumerate() {
                    let msg = if u < valid { (it.r[(g * D + u) * it.n + f] as u64) & 0xFFFF_FFFF } else { 0 };
                    *o = xb[(RANK * u + k) * L + l].wrapping_add(er[u] as i64 as u64).wrapping_add(msg << (QBITS - K)) & M56;
                }
                // SAFETY: rows 16 c0 .. 16 (c0 + lanes) of b are written by this chunk only.
                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), b.0.add(f * cols + g * D), D) };
            }
        }
    }
}

#[inline(always)]
fn enc_worker(fast: Vx, it: &EncItem<'_>, that: &[Vec<Vec<u32>>], bp: Shared<u64>, th: usize, threads: usize) {
    let cx = ctx();
    let chunks = (it.n / RANK).div_ceil(L);
    let mut a16 = Wiping(vec![0u32; 3 * NU * L]);
    let mut s16 = Wiping(vec![0u32; 3 * NU * L]);
    let mut raw = vec![0u8; NU * 8];
    let mut xb = Wiping(vec![0u64; if fast == Vx::Z { 0 } else { NU * L }]);
    for ch in (th..chunks).step_by(threads) {
        enc_chunk(fast, cx, it, ch * L, that, bp, &mut a16.0, &mut s16.0, &mut raw, &mut xb.0);
    }
}

crate::multiversion!(fn enc_worker_isa(fast: Vx, it: &EncItem<'_>, that: &[Vec<Vec<u32>>], bp: Shared<u64>, th: usize,
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
            s.spawn(move || enc_worker_isa(isa, Vx::of(isa), it, that, bp, th, threads));
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
fn ksk_inner(fast: Vx, km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32]) {
    let cx = ctx();
    let mut e = Wiping(vec![0i32; RANK * NR]);
    cbd_into(fast, e_key, 0, &mut e.0);
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
        fwd(fast, pr, &mut m16.0);
        let (p, pn) = (pr.p, pr.pn);
        #[cfg(target_arch = "x86_64")]
        if fast == Vx::Z {
            // SAFETY: km, beta hold 4 x RANK x NR words, th NR, m16 NR L; AVX-512 (`fast`).
            unsafe {
                crate::simd::ksk_out(p, pn, NR, km[i * RANK * NR..].as_ptr(), th.0.as_ptr(), m16.0.as_ptr(),
                                     beta[i * RANK * NR..].as_mut_ptr());
            }
            continue;
        }
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

crate::multiversion!(fn ksk_isa(fast: Vx, km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32]) => ksk_inner);

/// Reply key of one group: `beta_l = k_l t' + e_l + P iota(t_l) mod q1 P` in the NTT domain
/// (`beta`: `4 x RANK x NR`, `< p`), and the NTT of `t'` mod the two `Q1` primes (`trhat`: `2 x NR`, `< p`).
pub fn reply_key(isa: Isa, km: &[u32], t: &[i8], tr: &[i8], e_key: &[u8; 32], beta: &mut [u32], trhat: &mut [u32])
    -> Result<(), Error> {
    if km.len() != 4 * RANK * NR || t.len() != NU || tr.len() != NR || beta.len() != 4 * RANK * NR
        || trhat.len() != 2 * NR
    {
        return err("reply_key: inconsistent shapes");
    }
    ksk_isa(isa, Vx::of(isa), km, t, tr, e_key, beta, trhat);
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
fn dec_chunk(fast: Vx, cx: &Ctx, it: &DecItem<'_>, b0: usize, outs: &[Shared<u32>], h16: &mut [u32], s16: &mut [u32]) {
    let nb = it.m / PACK;
    let lanes = (nb - b0).min(L);
    for i in 0..2 {
        let h = &mut h16[i * NR * L..(i + 1) * NR * L];
        #[cfg(target_arch = "x86_64")]
        if fast == Vx::Z {
            let src: [*const u8; L] =
                std::array::from_fn(|l| it.hint[((b0 + l.min(lanes - 1)) * 2 + i) * NR * 3..].as_ptr());
            // SAFETY: each source has 3 NR bytes (masked 48-byte loads), h has NR L words; AVX-512 (`fast`).
            unsafe { crate::simd::load_hint(&src, lanes, NR, h.as_mut_ptr()) };
            continue;
        }
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
            inv_mul(fast, pr, &mut s16[i * NR * L..(i + 1) * NR * L], &h16[i * NR * L..(i + 1) * NR * L], th);
        }
        let gbase = g * nb * NR;
        let out = outs[g];
        #[cfg(target_arch = "x86_64")]
        if fast == Vx::Z {
            let kc = crate::simd::DecConsts { p0, p1, pn1, inv01_m: cx.inv01_m, c0scale, wscale };
            let src = match it.c0r {
                C0::Words(v) => crate::simd::C0p::Words(v.as_ptr()),
                C0::Packed36 { data, count } => {
                    crate::simd::C0p::Packed36 { lo: data.as_ptr(), hi: data[4 * count..].as_ptr() }
                }
            };
            for u in 0..rows {
                let idx0: [usize; L] = std::array::from_fn(|l| gbase + (b0 + l.min(lanes - 1)) * NR + PACK * u);
                // SAFETY: columns 32 b0 .. 32 (b0 + lanes) of every row are written by this chunk only; reads
                // stay inside c0r (idx0 + 32 <= count) and s16; `fast` implies AVX-512 (Isa::parse).
                let outp: [*mut u32; L] =
                    std::array::from_fn(|l| unsafe { out.0.add(u * it.m + PACK * (b0 + l.min(lanes - 1))) });
                unsafe {
                    crate::simd::dec_combine(&kc, s16.as_ptr().add(PACK * u * L), s16.as_ptr().add(NR * L + PACK * u * L),
                                             lanes, src, &idx0, &outp);
                }
            }
            continue;
        }
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
fn dec_worker(fast: Vx, it: &DecItem<'_>, ptrs: &[Shared<u32>], th: usize, threads: usize) {
    let cx = ctx();
    let chunks = (it.m / PACK).div_ceil(L);
    let mut h16 = vec![0u32; 2 * NR * L];
    let mut s16 = Wiping(vec![0u32; 2 * NR * L]);
    for ch in (th..chunks).step_by(threads) {
        dec_chunk(fast, cx, it, ch * L, ptrs, &mut h16, &mut s16.0);
    }
}

crate::multiversion!(fn dec_worker_isa(fast: Vx, it: &DecItem<'_>, ptrs: &[Shared<u32>], th: usize, threads: usize)
    => dec_worker);

fn decrypt_inner(isa: Isa, it: &DecItem<'_>, outs: &mut [&mut [u32]], threads: usize) {
    let chunks = (it.m / PACK).div_ceil(L);
    let ptrs: Vec<Shared<u32>> = outs.iter_mut().map(|o| Shared(o.as_mut_ptr())).collect();
    let threads = threads.clamp(1, chunks.max(1));
    std::thread::scope(|s| {
        for th in 0..threads {
            let ptrs = &ptrs;
            s.spawn(move || dec_worker_isa(isa, Vx::of(isa), it, ptrs, th, threads));
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

#[cfg(test)]
mod tests {
    //! Bit identity of every ISA copy and thread count (the AVX-512 kernels against the portable code,
    //! which the Python tests check against the numpy reference), and the noise sampler's distribution.
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn key(&mut self) -> [u8; 32] {
            std::array::from_fn(|_| self.next() as u8)
        }
    }

    fn isas() -> Vec<Isa> {
        let mut v = vec![Isa::Scalar];
        for name in ["avx2", "avx512"] {
            let i = Isa::parse(name).unwrap();
            if !v.contains(&i) {
                v.push(i);
            }
        }
        v
    }

    #[test]
    fn keystream_copies_equal_the_crate() {
        let mut g = Lcg(1);
        let key = g.key();
        for &(off, len) in &[(0u64, 64usize), (0, 1024), (64, 960), (64 * 1000, 64 * 37), (8 * 4096 * 5, 4096 * 8), (3, 77)] {
            let mut want = vec![0u8; len];
            keystream(Vx::Port, &key, off, &mut want);
            for isa in isas() {
                let mut got = vec![0u8; len];
                keystream_isa(isa, &key, off, &mut got);
                assert_eq!(got, want, "isa {isa:?} offset {off} len {len}");
            }
        }
    }

    #[test]
    fn cbd_identical_and_distributed_as_cbd21() {
        let mut g = Lcg(2);
        let key = g.key();
        let n = 1 << 20;
        let mut want = vec![0i32; n];
        cbd(Isa::Scalar, &key, 5, &mut want);
        for isa in isas() {
            let mut got = vec![0i32; n];
            cbd(isa, &key, 5, &mut got);
            assert_eq!(got, want);
        }
        // same values when drawn in pieces (stream offsets)
        let mut parts = vec![0i32; n];
        let (a, b) = parts.split_at_mut(12345);
        cbd(Isa::parse("auto").unwrap(), &key, 5, a);
        cbd(Isa::parse("auto").unwrap(), &key, 5 + 12345, b);
        assert_eq!(parts, want);
        // distribution: CBD(21) = Binomial(42, 1/2) - 21; chi-square over [-12, 12] (tails pooled), moments
        let mut hist = [0u64; 43];
        for &v in &want {
            assert!((-21..=21).contains(&v));
            hist[(v + 21) as usize] += 1;
        }
        let mut binom = [0f64; 43];
        let mut c = 1f64;
        for (k, b) in binom.iter_mut().enumerate() {
            *b = c / 2f64.powi(42);
            c = c * (42 - k) as f64 / (k + 1) as f64;
        }
        let (mut chi2, mut lo_o, mut lo_e) = (0f64, 0f64, 0f64);
        for k in 0..43 {
            let (o, e) = (hist[k] as f64, binom[k] * n as f64);
            if (9..=33).contains(&k) {
                chi2 += (o - e) * (o - e) / e;
            } else {
                lo_o += o;
                lo_e += e;
            }
        }
        chi2 += (lo_o - lo_e) * (lo_o - lo_e) / lo_e;
        // 25 cells + pooled tails = 25 degrees of freedom; P(chi2 > 70) < 1e-5
        assert!(chi2 < 70.0, "chi2 {chi2}");
        let mean = want.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = want.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02 && (var - 10.5).abs() < 0.1, "mean {mean} var {var}");
    }

    #[test]
    fn encrypt_identical_across_isas_and_threads() {
        let mut g = Lcg(3);
        for &(n, groups, rows) in &[(48usize, 2usize, 300usize), (272, 1, 256), (16 * 40, 2, 257)] {
            let r: Vec<i64> = (0..rows * n).map(|_| g.next() as i64 - (1 << 52)).collect();
            let t: Vec<i8> = (0..groups * NU).map(|_| (g.next() % 3) as i8 - 1).collect();
            let ek: Vec<[u8; 32]> = (0..groups).map(|_| g.key()).collect();
            let it = EncItem { a_key: g.key(), n, r: &r, rows, t: &t, e_keys: &ek };
            let mut want = vec![0u64; n * groups * D];
            encrypt_item(Isa::Scalar, &it, &mut want, 1).unwrap();
            for isa in isas() {
                for threads in [1usize, 2, 3] {
                    let mut got = vec![0u64; n * groups * D];
                    encrypt_item(isa, &it, &mut got, threads).unwrap();
                    assert_eq!(got, want, "{isa:?} threads {threads} n {n}");
                }
            }
        }
    }

    #[test]
    fn reply_key_identical_across_isas() {
        let mut g = Lcg(4);
        let cx = ctx();
        let k_hat: Vec<u32> = (0..4 * RANK * NR).map(|i| (g.next() % cx.qp[i / (RANK * NR)].p as u64) as u32).collect();
        let km = k_mont(&k_hat).unwrap();
        let t: Vec<i8> = (0..NU).map(|_| (g.next() % 3) as i8 - 1).collect();
        let tr: Vec<i8> = (0..NR).map(|_| (g.next() % 3) as i8 - 1).collect();
        let ek = g.key();
        let (mut wb, mut wt) = (vec![0u32; 4 * RANK * NR], vec![0u32; 2 * NR]);
        reply_key(Isa::Scalar, &km, &t, &tr, &ek, &mut wb, &mut wt).unwrap();
        for isa in isas() {
            let (mut b, mut th) = (vec![0u32; 4 * RANK * NR], vec![0u32; 2 * NR]);
            reply_key(isa, &km, &t, &tr, &ek, &mut b, &mut th).unwrap();
            assert!(b == wb && th == wt, "{isa:?}");
        }
    }

    #[test]
    fn decrypt_identical_across_isas_and_threads() {
        let mut g = Lcg(5);
        let cx = ctx();
        // nb = 20 blocks: one full 16-lane chunk and one with 4 lanes; partial groups
        for &(m, ref rows, yb) in &[(640usize, vec![256usize, 77], 36u32), (64, vec![3], 36), (96, vec![256, 256], 54)] {
            let nb = m / PACK;
            let groups = rows.len();
            let hint: Vec<u8> = (0..nb * 2 * NR)
                .flat_map(|i| {
                    let v = (g.next() % cx.qp[i / NR % 2].p as u64) as u32;
                    [v as u8, (v >> 8) as u8, (v >> 16) as u8]
                })
                .collect();
            let trhat: Vec<u32> = (0..groups * 2 * NR).map(|i| (g.next() % cx.qp[(i / NR) % 2].p as u64) as u32).collect();
            let words: Vec<u64> = (0..groups * nb * NR).map(|_| g.next() & ((1u64 << yb) - 1)).collect();
            let mut packed = Vec::new();
            if yb == 36 {
                for w in &words {
                    packed.extend_from_slice(&(*w as u32).to_le_bytes());
                }
                for ch in words.chunks(8) {
                    let mut x = 0u32;
                    for (k, w) in ch.iter().enumerate() {
                        x |= ((w >> 32) as u32 & 15) << (4 * k);
                    }
                    packed.extend_from_slice(&x.to_le_bytes());
                }
                assert_eq!(unpack_bits_ref(&packed, words.len()), words);
            }
            let total: usize = rows.iter().sum();
            let run = |isa: Isa, threads: usize, c0r: C0<'_>| -> Vec<u32> {
                let mut out = vec![0u32; total * m];
                let mut outs: Vec<&mut [u32]> = Vec::new();
                let mut rest = out.as_mut_slice();
                for &r in rows {
                    let (a, b) = std::mem::take(&mut rest).split_at_mut(r * m);
                    outs.push(a);
                    rest = b;
                }
                let it = DecItem { hint: &hint, m, c0r, yb, trhat: &trhat, rows };
                decrypt_item(isa, &it, &mut outs, threads).unwrap();
                out
            };
            let want = run(Isa::Scalar, 1, C0::Words(&words));
            for isa in isas() {
                for threads in [1usize, 2, 3] {
                    assert_eq!(run(isa, threads, C0::Words(&words)), want, "{isa:?} {threads} words m {m}");
                    if yb == 36 {
                        let c = C0::Packed36 { data: &packed, count: words.len() };
                        assert_eq!(run(isa, threads, c), want, "{isa:?} {threads} packed m {m}");
                    }
                }
            }
        }
    }

    fn unpack_bits_ref(data: &[u8], count: usize) -> Vec<u64> {
        crate::arith::unpack_bits(data, count, 36).unwrap()
    }
}
