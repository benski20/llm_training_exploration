//! Customer CPU of pks-v3 at real shapes (random data: the cost does not depend on the values).
//!
//! `pks-bench decrypt <m> <groups> <threads> [isa]`: reply decryption of one linear with `m` outputs;
//! `pks-bench encrypt <n> <groups> <threads> [isa]`: upload encryption (pad + CBD noise + message) of a
//! linear with `n` inputs; `pks-bench ksk <reps>`: one reply key (per linear and group);
//! `pks-bench keyswitch <m> <groups> <threads>`: the provider's CPU key switch. Prints JSON lines with
//! wall and process-CPU ns per output (input) and slot.

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
    let (mut best, mut best_cpu) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..reps {
        let (t0, c0) = (Instant::now(), cpu_s());
        f();
        best = best.min(t0.elapsed().as_secs_f64());
        best_cpu = best_cpu.min(cpu_s() - c0);
    }
    (best, best_cpu)
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
        "keyswitch" => {
            let hs: Vec<u64> = (0..size * NU).map(|_| g.next() % cx.q1).collect();
            let ops: Vec<u32> = (0..groups * 4 * RANK * NR).map(|i| (g.next() % cx.qp[(i / (RANK * NR)) % 4].p as u64) as u32).collect();
            let y: Vec<u64> = (0..size * groups * D).map(|_| g.next() & ((1 << 56) - 1)).collect();
            let mut out = vec![0u64; groups * (size / PACK) * NR];
            let it = KsItem { hs: &hs, m: size, ops: &ops, groups, y: Some(&y), yb: 36 };
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
            let (s1, _) = time(5, || kernel::ntt_bench(isa, pr, if isa == Isa::Avx512 { 5 } else { 1 }, &mut a, &h, &t, reps));
                        let (s2, _) = time(5, || kernel::ntt_bench(isa, pr, 2, &mut a, &h, &t, reps));
            let (s3, _) = time(5, || kernel::ntt_bench(isa, pr, if isa == Isa::Avx512 { 7 } else { 3 }, &mut a, &h, &t, reps));
            let f = |s: f64| s * 1e9 / per / reps as f64;
            println!("{{\"mode\":\"ntt\",\"inv_l_mul\":{:.3},\"inv_l_mul_fast\":{:.3},\"fwd_l\":{:.3},\"fwd_l_fast\":{:.3}}}", f(s0), f(s1), f(s2), f(s3));
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
