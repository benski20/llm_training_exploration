//! Microbenchmark: `cargo run --release -p pgw-ntt --no-default-features --example ntt_bench -- [rows] [threads] [isa]`.
//!
//! Prints ns per row for: transform (residues + forward NTTs), product from a transformed operand
//! (G = 1 and 4 secrets), and product straight from 6 byte planes (decrypt path, hint not transformed).

use pgw_ntt::arith::N;
use pgw_ntt::kernel::{product, transform, Big, Isa, Secrets};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("parts") {
        parts();
        return;
    }
    let rows: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let isa = Isa::parse(args.get(3).map_or("auto", |s| s.as_str())).unwrap();
    let rank = 16;
    let dim = rank * N;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let a: Vec<u64> = (0..rows * dim).map(|_| next()).collect();
    let planes: Vec<i8> = (0..rows * 6 * dim).map(|_| next() as i8).collect();
    let mk = |g: usize, next: &mut dyn FnMut() -> u64| {
        let s: Vec<i8> = (0..g * dim).map(|_| (next() % 3) as i8 - 1).collect();
        Secrets::new(&s, g, rank)
    };
    let s1 = mk(1, &mut next);
    let s4 = mk(4, &mut next);
    let s16 = mk(16, &mut next);
    println!("isa {} threads {threads} rows {rows} rank {rank}", isa.resolve().name());
    let t = Instant::now();
    let pre = transform(Big::U64 { data: &a, rows }, rank, threads, isa).unwrap();
    let el = t.elapsed().as_secs_f64();
    println!("transform u64      : {:8.1} us/row", el * 1e6 / rows as f64);
    for (name, s) in [("G=1", &s1), ("G=4", &s4), ("G=16", &s16)] {
        let t = Instant::now();
        let out = product(Big::Pre { data: &pre, rows, bits: 64 }, dim, s, threads, isa).unwrap();
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        println!("product pre {name:5}  : {:8.2} us/row/secret", el * 1e6 / (rows * s.g) as f64);
    }
    for (name, s) in [("G=1", &s1), ("G=4", &s4), ("G=16", &s16)] {
        let t = Instant::now();
        let out = product(Big::Planes { data: &planes, rows, npl: 6, first: 1 }, dim, s, threads, isa).unwrap();
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        println!("product planes {name:5}: {:8.2} us/row/secret", el * 1e6 / (rows * s.g) as f64);
    }
}

#[allow(dead_code)]
fn parts() {
    #[cfg(target_arch = "x86_64")]
    for (name, s) in
        unsafe { pgw_ntt::kernel::profile_parts(200, std::env::args().nth(2).as_deref() != Some("portable")) }
    {
        println!("{name:40}: {:8.3} us", s * 1e6);
    }
}
