//! Customer CPU of pks-v3 at real shapes (random data: the cost does not depend on the values).
//!
//! `pks-bench decrypt <m> <groups> <threads> [isa]`: reply decryption of one linear with `m` outputs;
//! `pks-bench encrypt <n> <groups> <threads> [isa]`: upload encryption (pad + CBD noise + message) of a
//! linear with `n` inputs; `pks-bench ksk <reps>`: one reply key (per linear and group);
//! `pks-bench keyswitch <m> <groups> <threads>`: the provider's CPU key switch; per-kernel micro timings:
//! `pks-bench ntt <reps> 1 1 [isa]` (16-lane NTTs), `pks-bench cbd <count> 1 1 [isa]` (noise sampler),
//! `pks-bench encout <reps>` (AVX-512 upload output on L2-resident data; timing only, dummy constants).
//! `isa`: auto, avx512ifma, avx512, avx2, scalar. Env: `PKS_REPS` (repetitions, best wall time reported),
//! `PKS_PACKED=1` (36-bit replies read in place), `PKS_ROWS` (slots per group). Prints JSON lines with the
//! best wall time and the process CPU averaged over the repetitions, in ns per output (input) and slot.

use pgw_pks::arith::Isa;
use pgw_pks::kernel::{self, DecItem, EncItem, KsItem, D, NR, NU, PACK, RANK};
use std::time::Instant;

fn cpu_s() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> (f64, f64) {
    // best wall time; process CPU time (10 ms ticks) averaged over all repetitions
    f();
    let mut best = f64::INFINITY;
    let c0 = cpu_s();
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    (best, (cpu_s() - c0) / reps as f64)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("decrypt");
    let size: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let groups: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let threads: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let isa = Isa::parse(a.get(5).map(String::as_str).unwrap_or("auto")).expect("isa");
    let reps: usize = std::env::var("PKS_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let cx = kernel::ctx();
    let mut g = Lcg(42);
    let key = |g: &mut Lcg| -> [u8; 32] { std::array::from_fn(|_| g.next() as u8) };
    match mode {
        "decrypt" => {
            let nb = size / PACK;
            let hint: Vec<u8> = (0..nb * 2 * NR * 3).map(|i| (g.next() as u8) & if i % 3 == 2 { 0x7f } else { 0xff }).collect();
            let trhat: Vec<u32> = (0..groups * 2 * NR).map(|i| (g.next() % cx.qp[(i / NR) % 2].p as u64) as u32).collect();
            let c0r: Vec<u64> = (0..groups * nb * NR).map(|_| g.next() & ((1 << 36) - 1)).collect();
            let rows = vec![std::env::var("PKS_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(D); groups];
            let r0 = rows[0];
            let mut out = vec![0u32; groups * r0 * size];
            let packed: Vec<u8> = (0..c0r.len() * 36 / 8).map(|_| g.next() as u8).collect();
            let c0 = if std::env::var("PKS_PACKED").is_ok() { kernel::C0::Packed36 { data: &packed, count: c0r.len() } } else { kernel::C0::Words(&c0r) };
            let it = DecItem { hint: &hint, m: size, c0r: c0, yb: 36, trhat: &trhat, rows: &rows };
            let (s, c) = time(reps, || {
                let mut rest = out.as_mut_slice();
                let mut outs: Vec<&mut [u32]> = Vec::new();
                for _ in 0..groups {
                    let (a, b) = std::mem::take(&mut rest).split_at_mut(r0 * size);
                    outs.push(a);
                    rest = b;
                }
                kernel::decrypt_item(isa, &it, &mut outs, threads).unwrap();
            });
            let per = (size * groups * D) as f64;
            println!(
                "{{\"mode\":\"decrypt\",\"isa\":\"{}\",\"m\":{size},\"groups\":{groups},\"threads\":{threads},\"ns_per_output_slot\":{:.3},\"cpu_ns_per_output_slot\":{:.3}}}",
                isa.name(), s * 1e9 / per, c * 1e9 / per
            );
        }
        "encrypt" => {
            let rows = groups * D;
            let r: Vec<i64> = (0..rows * size).map(|_| (g.next() & 0xFFFF_FFFF) as i64).collect();
            let t: Vec<i8> = (0..groups * NU).map(|_| (g.next() % 3) as i8 - 1).collect();
            let ek: Vec<[u8; 32]> = (0..groups).map(|_| key(&mut g)).collect();
            let it = EncItem { a_key: key(&mut g), n: size, r: &r, rows, t: &t, e_keys: &ek };
            let mut b = vec![0u64; size * groups * D];
            let (s, c) = time(reps, || kernel::encrypt_item(isa, &it, &mut b, threads).unwrap());
            let per = (size * groups * D) as f64;
            println!(
                "{{\"mode\":\"encrypt\",\"isa\":\"{}\",\"n\":{size},\"groups\":{groups},\"threads\":{threads},\"ns_per_input_slot\":{:.3},\"cpu_ns_per_input_slot\":{:.3}}}",
                isa.name(), s * 1e9 / per, c * 1e9 / per
            );
        }
        "ksk" => {
            let k_hat: Vec<u32> = (0..4 * RANK * NR).map(|i| (g.next() % cx.qp[i / (RANK * NR)].p as u64) as u32).collect();
            let km = kernel::k_mont(&k_hat).unwrap();
            let t: Vec<i8> = (0..NU).map(|_| (g.next() % 3) as i8 - 1).collect();
            let tr: Vec<i8> = (0..NR).map(|_| (g.next() % 3) as i8 - 1).collect();
            let ek = key(&mut g);
            let mut beta = vec![0u32; 4 * RANK * NR];
            let mut trhat = vec![0u32; 2 * NR];
            let (s, c) = time(size.clamp(1, 20), || kernel::reply_key(isa, &km, &t, &tr, &ek, &mut beta, &mut trhat).unwrap());
            println!("{{\"mode\":\"ksk\",\"isa\":\"{}\",\"ms\":{:.3},\"cpu_ms\":{:.3}}}", isa.name(), s * 1e3, c * 1e3);
        }
        "vecmat" => {
            // the factory's check-vector products x @ W8 (W8 `size x groups` int8): wrapping 32-bit and exact p31,
            // `PKS_C` vectors (default 2); prints ns per weight element and vector
            let (m, n) = (size, groups);
            let c: usize = std::env::var("PKS_C").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            let w: Vec<i8> = (0..m * n).map(|_| g.next() as i8).collect();
            let x: Vec<u32> = (0..c * m).map(|_| g.next() as u32).collect();
            let u: Vec<u64> = (0..c * m).map(|_| g.next() % pgw_pks::vecmat::P31).collect();
            let mut o32 = vec![0u32; c * n];
            let mut op = vec![0u64; c * n];
            let (s0, _) = time(reps, || pgw_pks::vecmat::vecmat_wrap32_isa(isa, &x, c, &w, m, n, &mut o32, threads).unwrap());
            let (s1, _) = time(reps, || pgw_pks::vecmat::vecmat_p31_isa(isa, &u, c, &w, m, n, &mut op, threads).unwrap());
            let per = (m * n * c) as f64;
            // a digest of the outputs: identical across ISAs
            let h = o32.iter().fold(0u64, |a, &v| a.wrapping_mul(31).wrapping_add(u64::from(v)))
                ^ op.iter().fold(0u64, |a, &v| a.wrapping_mul(37).wrapping_add(v));
            println!(
                "{{\"mode\":\"vecmat\",\"isa\":\"{}\",\"m\":{m},\"n\":{n},\"c\":{c},\"threads\":{threads},\"wrap32_ms\":{:.2},\"p31_ms\":{:.2},\"wrap32_ns_per_elem_vec\":{:.4},\"p31_ns_per_elem_vec\":{:.4},\"digest\":\"{h:016x}\"}}",
                isa.name(), s0 * 1e3, s1 * 1e3, s0 * 1e9 / per, s1 * 1e9 / per
            );
        }
        "checkrows" => {
            // the factory's every-row self-check: `size` rows, m = groups outputs, n = PKS_N inputs, PKS_C vectors
            let (rows, m) = (size, groups);
            let n: usize = std::env::var("PKS_N").ok().and_then(|v| v.parse().ok()).unwrap_or(896);
            let c: usize = std::env::var("PKS_C").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            let r: Vec<i64> = (0..rows * n).map(|_| g.next() as u32 as i64).collect();
            let wr: Vec<u32> = (0..rows * m).map(|_| g.next() as u32).collect();
            let cu: Vec<u32> = (0..m * c).map(|_| g.next() as u32).collect();
            let cw: Vec<u32> = (0..n * c).map(|_| g.next() as u32).collect();
            let mut bad = vec![0u8; rows];
            let (s, _) = time(reps, || pgw_pks::vecmat::check_rows32_isa(isa, &r, n, &wr, m, &cu, &cw, c, u32::MAX, &mut bad, threads).unwrap());
            let nbad: usize = bad.iter().map(|&b| usize::from(b)).sum();
            println!(
                "{{\"mode\":\"checkrows\",\"isa\":\"{}\",\"rows\":{rows},\"m\":{m},\"n\":{n},\"c\":{c},\"threads\":{threads},\"ms\":{:.2},\"ns_per_row_elem\":{:.4},\"bad\":{nbad}}}",
                isa.name(), s * 1e3, s * 1e9 / (rows * (m + n)) as f64
            );
        }
        "keyswitch" => {
            let hs: Vec<u64> = (0..size * NU).map(|_| g.next() % cx.q1).collect();
            let ops: Vec<u32> = (0..groups * 4 * RANK * NR).map(|i| (g.next() % cx.qp[(i / (RANK * NR)) % 4].p as u64) as u32).collect();
            let y: Vec<u64> = (0..size * groups * D).map(|_| g.next() & ((1 << 56) - 1)).collect();
            let mut out = vec![0u64; groups * (size / PACK) * NR];
            let it = KsItem { hs: &hs, m: size, ops: &ops, groups, y: Some(&y), yb: 36, raw: false };
            let (s, c) = time(1, || kernel::keyswitch(isa, &it, &mut out, threads).unwrap());
            let per = (size * groups * D) as f64;
            println!(
                "{{\"mode\":\"keyswitch\",\"isa\":\"{}\",\"m\":{size},\"groups\":{groups},\"threads\":{threads},\"ns_per_output_slot\":{:.3},\"cpu_ns_per_output_slot\":{:.3}}}",
                isa.name(), s * 1e9 / per, c * 1e9 / per
            );
        }
        "ntt" => {
            // per-kernel micro timing (ns per coefficient-lane): inverse NTT (decrypt), forward NTT
            let pr = &cx.qp[0];
            let n = NR;
            let h: Vec<u32> = (0..n * 16).map(|_| (g.next() % pr.p as u64) as u32).collect();
            let t: Vec<u32> = (0..n).map(|_| (g.next() % pr.p as u64) as u32).collect();
            let mut a = vec![0u32; n * 16];
            let per = (n * 16) as f64;
            let reps = size.clamp(1, 1000);
            let (s0, _) = time(5, || kernel::ntt_bench(isa, pr, 0, &mut a, &h, &t, reps));
            let (s1, _) = time(5, || kernel::ntt_bench(isa, pr, match isa { Isa::Avx512 | Isa::Avx512Ifma => 5, Isa::Avx2 => 9, _ => 1 }, &mut a, &h, &t, reps));
                        let (s2, _) = time(5, || kernel::ntt_bench(isa, pr, 2, &mut a, &h, &t, reps));
            let (s3, _) = time(5, || kernel::ntt_bench(isa, pr, match isa { Isa::Avx512 | Isa::Avx512Ifma => 7, Isa::Avx2 => 11, _ => 3 }, &mut a, &h, &t, reps));
            let f = |s: f64| s * 1e9 / per / reps as f64;
            println!("{{\"mode\":\"ntt\",\"inv_l_mul\":{:.3},\"inv_l_mul_fast\":{:.3},\"fwd_l\":{:.3},\"fwd_l_fast\":{:.3}}}", f(s0), f(s1), f(s2), f(s3));
        }
        #[cfg(target_arch = "x86_64")]
        "encout" => {
            // the upload CRT/output kernel alone on L2-resident data (one 16-block chunk, one group)
            let e = [cx.ep[0].p, cx.ep[1].p, cx.ep[2].p];
            let s16: Vec<u32> = (0..3 * NU * 16).map(|i| (g.next() % (2 * e[i / (NU * 16)] as u64)) as u32).collect();
            let noise: Vec<i32> = (0..16 * RANK * D).map(|_| (g.next() % 43) as i32 - 21).collect();
            let n = 256;
            let r: Vec<i64> = (0..D * n).map(|_| g.next() as i64).collect();
            let mut b = vec![0u64; n * D];
            let kc = pgw_pks::simd::EncConsts { e, pn: [cx.ep[0].pn, cx.ep[1].pn, cx.ep[2].pn], inv01_m: 1, e0_m2: 1,
                inv012_m: 1, p01: e[0] as u64 * e[1] as u64, emod: 12345 };
            let reps = size.max(1);
            let (s, _) = time(5, || for _ in 0..reps {
                unsafe { pgw_pks::simd::enc_out(&kc, s16.as_ptr(), NU, 16, 0, D, n, r.as_ptr(), D, noise.as_ptr(), b.as_mut_ptr()) }
            });
            println!("{{\"mode\":\"encout\",\"ns_per_value\":{:.3}}}", s * 1e9 / (reps * NU * 16) as f64);
        }
        "cbd" => {
            let mut out = vec![0i32; size];
            let k = key(&mut g);
            let (s, _) = time(5, || kernel::cbd(isa, &k, 0, &mut out));
            println!("{{\"mode\":\"cbd\",\"isa\":\"{}\",\"ns_per_sample\":{:.3}}}", isa.name(), s * 1e9 / size as f64);
        }
        _ => eprintln!("modes: decrypt | encrypt | ksk | keyswitch | ntt | cbd"),
    }
}
