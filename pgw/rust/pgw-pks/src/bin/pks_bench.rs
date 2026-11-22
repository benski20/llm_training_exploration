//! Customer CPU of pks-v3 at real shapes: `pks-bench decrypt <m> <groups> <threads>` times the reply
//! decryption of one linear with `m` output rows for `groups` 256-slot groups (random hint, secret and
//! reply: the cost does not depend on the values); `pks-bench pad <n> <groups> <threads>` times the
//! ring-LWE upload pad of a linear with `n` inputs. Prints ns per output (input) and slot.

use pgw_pks::{
    decrypt_block, decrypt_block16, hint_form, pad_block, pad_ctx, reply_ctx, secret_hat, LANES, N, ROWS, SLOTS,
};
use std::time::Instant;

/// Process CPU seconds (all threads; utime + stime from /proc/self/stat, 100 Hz ticks). On a shared,
/// oversubscribed host this is the meaningful cost (wall time includes waiting for a core).
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("decrypt");
    let rows: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let groups: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let threads: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let reps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    let blocks = rows / ROWS;
    let mut g = Lcg(42);
    let tern: Vec<i8> = (0..N).map(|_| (g.next() % 3) as i8 - 1).collect();
    match mode {
        "decrypt" => {
            let ctx = reply_ctx();
            let hint: Vec<Vec<Vec<u32>>> = (0..2)
                .map(|i| {
                    let pr = &ctx.pr[i];
                    (0..blocks)
                        .map(|_| hint_form(pr, &(0..N).map(|_| (g.next() % pr.p as u64) as u32).collect::<Vec<_>>()))
                        .collect()
                })
                .collect();
            let th: Vec<Vec<u32>> = (0..2).map(|i| secret_hat(&ctx.pr[i], &tern)).collect();
            let replies: Vec<Vec<u64>> =
                (0..groups).map(|_| (0..blocks * N).map(|_| g.next() & ((1 << 36) - 1)).collect()).collect();
            let mut best = f64::INFINITY;
            let mut best_cpu = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let c0 = cpu_s();
                std::thread::scope(|s| {
                    for th_i in 0..threads {
                        let (hint, th, replies) = (&hint, &th, &replies);
                        s.spawn(move || {
                            let mut out = vec![0u32; SLOTS * ROWS];
                            let mut scratch = vec![0u32; 2 * N];
                            for b in (th_i..blocks).step_by(threads) {
                                for rep in replies.iter() {
                                    decrypt_block(
                                        ctx,
                                        [&hint[0][b], &hint[1][b]],
                                        [&th[0], &th[1]],
                                        &rep[b * N..(b + 1) * N],
                                        36,
                                        SLOTS,
                                        ROWS,
                                        0,
                                        &mut out,
                                        &mut scratch,
                                    );
                                }
                            }
                            std::hint::black_box(&out);
                        });
                    }
                });
                best = best.min(t0.elapsed().as_secs_f64());
                best_cpu = best_cpu.min(cpu_s() - c0);
            }
            let per = best * 1e9 / (rows * groups * SLOTS) as f64;
            println!(
                "{{\"mode\":\"decrypt\",\"m\":{rows},\"groups\":{groups},\"threads\":{threads},\"s\":{best:.4},\"ns_per_output_slot\":{per:.3},\"cpu_ns_per_output_slot\":{:.3}}}",
                best_cpu * 1e9 / (rows * groups * SLOTS) as f64
            );
        }
        "decrypt16" => {
            let ctx = reply_ctx();
            let nb16 = blocks / LANES;
            let hint: Vec<Vec<Vec<u32>>> = (0..2)
                .map(|i| {
                    let pr = &ctx.pr[i];
                    (0..nb16).map(|_| (0..N * LANES).map(|_| (g.next() % pr.p as u64) as u32).collect()).collect()
                })
                .collect();
            let th: Vec<Vec<u32>> = (0..2).map(|i| secret_hat(&ctx.pr[i], &tern)).collect();
            let replies: Vec<Vec<u64>> =
                (0..groups).map(|_| (0..nb16 * N * LANES).map(|_| g.next() & ((1 << 36) - 1)).collect()).collect();
            let m = LANES * ROWS;
            let mut best = f64::INFINITY;
            let mut best_cpu = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let c0 = cpu_s();
                std::thread::scope(|s| {
                    for th_i in 0..threads {
                        let (hint, th, replies) = (&hint, &th, &replies);
                        s.spawn(move || {
                            let mut out = vec![0u32; SLOTS * m];
                            let mut scratch = vec![0u32; 2 * N * LANES];
                            for b in (th_i..nb16).step_by(threads) {
                                for rep in replies.iter() {
                                    decrypt_block16(
                                        ctx,
                                        [&hint[0][b], &hint[1][b]],
                                        [&th[0], &th[1]],
                                        &rep[b * N * LANES..(b + 1) * N * LANES],
                                        36,
                                        SLOTS,
                                        m,
                                        0,
                                        &mut out,
                                        &mut scratch,
                                    );
                                }
                            }
                            std::hint::black_box(&out);
                        });
                    }
                });
                best = best.min(t0.elapsed().as_secs_f64());
                best_cpu = best_cpu.min(cpu_s() - c0);
            }
            let rows = nb16 * LANES * ROWS;
            let per = best * 1e9 / (rows * groups * SLOTS) as f64;
            println!(
                "{{\"mode\":\"decrypt16\",\"m\":{rows},\"groups\":{groups},\"threads\":{threads},\"s\":{best:.4},\"ns_per_output_slot\":{per:.3},\"cpu_ns_per_output_slot\":{:.3}}}",
                best_cpu * 1e9 / (rows * groups * SLOTS) as f64
            );
        }
        "pad" => {
            let ctx = pad_ctx();
            let ah: Vec<Vec<Vec<u32>>> = (0..3)
                .map(|i| {
                    let pr = &ctx.pr[i];
                    (0..blocks)
                        .map(|_| hint_form(pr, &(0..N).map(|_| (g.next() % pr.p as u64) as u32).collect::<Vec<_>>()))
                        .collect()
                })
                .collect();
            let th: Vec<Vec<u32>> = (0..3).map(|i| secret_hat(&ctx.pr[i], &tern)).collect();
            let mut best = f64::INFINITY;
            let mut best_cpu = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let c0 = cpu_s();
                std::thread::scope(|s| {
                    for th_i in 0..threads {
                        let (ah, th) = (&ah, &th);
                        s.spawn(move || {
                            let mut out = vec![0u64; N];
                            let mut scratch = vec![0u32; 3 * N];
                            for b in (th_i..blocks).step_by(threads) {
                                for _ in 0..groups {
                                    pad_block(
                                        ctx,
                                        [&ah[0][b], &ah[1][b], &ah[2][b]],
                                        [&th[0], &th[1], &th[2]],
                                        &mut out,
                                        &mut scratch,
                                    );
                                }
                            }
                            std::hint::black_box(&out);
                        });
                    }
                });
                best = best.min(t0.elapsed().as_secs_f64());
                best_cpu = best_cpu.min(cpu_s() - c0);
            }
            let per = best * 1e9 / (rows * groups * SLOTS) as f64;
            println!(
                "{{\"mode\":\"pad\",\"n\":{rows},\"groups\":{groups},\"threads\":{threads},\"s\":{best:.4},\"ns_per_input_slot\":{per:.3},\"cpu_ns_per_input_slot\":{:.3}}}",
                best_cpu * 1e9 / (rows * groups * SLOTS) as f64
            );
        }
        _ => eprintln!("usage: pks-bench decrypt|pad <rows> <groups> <threads> [reps]"),
    }
}
