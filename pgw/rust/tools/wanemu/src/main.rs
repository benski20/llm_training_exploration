//! Packet-level WAN emulator between two TUN devices, for containers without `tc netem`.
//!
//! `pgw-wanemu <tun-a> <tun-b>` creates two TUN devices (IFF_TUN | IFF_NO_PI) and forwards every IP packet read
//! from one to the other. Each direction is a link with its own parameters:
//!
//! * `rate_mbit` (0 = unlimited): a bottleneck that serialises packets at that rate (drop-tail when the backlog
//!   exceeds `queue_kb`), like a consumer access link;
//! * `delay_ms`: one-way propagation delay added after serialisation;
//! * `jitter_ms`: a normal deviate (sd) added to the delay, clamped at 0; packets never reorder (FIFO, as netem
//!   with a rate);
//! * `loss` (probability per packet) and `burst` (mean loss-burst length in packets, Gilbert-Elliott; 1 = Bernoulli).
//!
//! Direction `up` is a -> b (put the client behind `a`), `down` is b -> a. The kernels on both sides run real
//! TCP (slow start, delayed ACKs, loss recovery, cwnd validation) end to end: only packets are delayed, so the
//! transport behaves as over a real path, unlike a TCP-terminating proxy.
//!
//! Control on stdin, one command per line (answers on stdout, one line each):
//!
//! ```text
//! set <up|down|both> delay_ms=10 jitter_ms=0.5 loss=0.001 burst=1 rate_mbit=100 queue_kb=256
//! stats            -> {"up": {...}, "down": {...}}   (packets, bytes, drops, max backlog)
//! reset            -> zero the counters
//! quit
//! ```
//!
//! Timing: each direction has a reader thread (timestamps on arrival) and a writer thread that sleeps until
//! 150 us before a packet's due time and spins the rest (timer slack set to 1 ns), so a packet leaves within a
//! few microseconds of its due time on an idle core. A bench tool, never shipped; Linux only, needs
//! CAP_NET_ADMIN for the TUN devices.

use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
struct Params {
    delay: f64, // seconds
    jitter: f64,
    loss: f64,
    burst: f64,
    rate_bps: f64, // bits per second, 0 = unlimited
    queue_bytes: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { delay: 0.0, jitter: 0.0, loss: 0.0, burst: 1.0, rate_bps: 0.0, queue_bytes: 4.0 * 1024.0 * 1024.0 }
    }
}

#[derive(Default)]
struct Stats {
    packets: AtomicU64,
    bytes: AtomicU64,
    lost: AtomicU64,
    queue_drops: AtomicU64,
    max_backlog_us: AtomicU64,
    late_us_max: AtomicU64,
    late_us_sum: AtomicU64,
    sent: AtomicU64,
}

struct Link {
    params: Mutex<Params>,
    q: Mutex<VecDeque<(Instant, Vec<u8>)>>,
    cv: Condvar,
    stats: Stats,
}

fn tun_open(name: &str) -> i32 {
    let path = b"/dev/net/tun\0";
    let fd = unsafe { libc::open(path.as_ptr() as *const libc::c_char, libc::O_RDWR) };
    if fd < 0 {
        panic!("open /dev/net/tun: {}", std::io::Error::last_os_error());
    }
    let mut ifr = [0u8; 40];
    let nb = name.as_bytes();
    assert!(nb.len() < 16, "interface name too long");
    ifr[..nb.len()].copy_from_slice(nb);
    let flags: u16 = (libc::IFF_TUN | libc::IFF_NO_PI) as u16;
    ifr[16..18].copy_from_slice(&flags.to_ne_bytes());
    const TUNSETIFF: libc::c_ulong = 0x400454ca;
    let r = unsafe { libc::ioctl(fd, TUNSETIFF as _, ifr.as_mut_ptr()) };
    if r < 0 {
        panic!("TUNSETIFF {name}: {}", std::io::Error::last_os_error());
    }
    fd
}

/// xorshift64*: per-thread, seeded; bench randomness only.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next().max(1e-12);
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn reader(fd_in: i32, link: Arc<Link>, seed: u64) {
    let mut buf = vec![0u8; 65536];
    let mut rng = Rng(seed | 1);
    let mut link_free = Instant::now();
    let mut last_due = Instant::now();
    let mut bad = false; // Gilbert-Elliott state
    loop {
        let n = unsafe { libc::read(fd_in, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            eprintln!("read: {e}");
            return;
        }
        let n = n as usize;
        let now = Instant::now();
        let p = *link.params.lock().unwrap();
        link.stats.packets.fetch_add(1, Ordering::Relaxed);
        link.stats.bytes.fetch_add(n as u64, Ordering::Relaxed);
        // loss
        if p.loss > 0.0 {
            let lost = if p.burst <= 1.0 {
                rng.next() < p.loss
            } else {
                // enter bad with probability q, leave with 1/burst; stationary loss = q / (q + 1/burst)
                let r = 1.0 / p.burst;
                let q = p.loss * r / (1.0 - p.loss).max(1e-9);
                bad = if bad { rng.next() >= r } else { rng.next() < q };
                bad
            };
            if lost {
                link.stats.lost.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        }
        // bottleneck serialisation
        let mut depart = now;
        if p.rate_bps > 0.0 {
            let start = if link_free > now { link_free } else { now };
            let backlog = start.duration_since(now).as_secs_f64();
            if backlog * p.rate_bps / 8.0 > p.queue_bytes {
                link.stats.queue_drops.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            link.stats.max_backlog_us.fetch_max((backlog * 1e6) as u64, Ordering::Relaxed);
            link_free = start + Duration::from_secs_f64(n as f64 * 8.0 / p.rate_bps);
            depart = link_free;
        }
        let mut d = p.delay;
        if p.jitter > 0.0 {
            d = (d + p.jitter * rng.normal()).max(0.0);
        }
        let mut due = depart + Duration::from_secs_f64(d);
        if due < last_due {
            due = last_due; // FIFO: never reorder
        }
        last_due = due;
        let mut q = link.q.lock().unwrap();
        q.push_back((due, buf[..n].to_vec()));
        drop(q);
        link.cv.notify_one();
    }
}

fn writer(fd_out: i32, link: Arc<Link>) {
    loop {
        let (due, pkt) = {
            let mut q = link.q.lock().unwrap();
            loop {
                match q.front() {
                    None => q = link.cv.wait(q).unwrap(),
                    Some((due, _)) => {
                        let now = Instant::now();
                        if *due > now + Duration::from_micros(200) {
                            let wait = *due - now - Duration::from_micros(150);
                            q = link.cv.wait_timeout(q, wait).unwrap().0;
                        } else {
                            break;
                        }
                    }
                }
            }
            q.pop_front().unwrap()
        };
        while Instant::now() < due {
            std::hint::spin_loop();
        }
        let late = Instant::now().saturating_duration_since(due).as_micros() as u64;
        link.stats.late_us_max.fetch_max(late, Ordering::Relaxed);
        link.stats.late_us_sum.fetch_add(late, Ordering::Relaxed);
        link.stats.sent.fetch_add(1, Ordering::Relaxed);
        let r = unsafe { libc::write(fd_out, pkt.as_ptr() as *const libc::c_void, pkt.len()) };
        if r < 0 {
            // the device may be down for a moment (netns move); drop like a link would
            continue;
        }
    }
}

fn apply(p: &mut Params, kv: &str) -> Result<(), String> {
    let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad item {kv}"))?;
    let x: f64 = v.parse().map_err(|_| format!("bad value {kv}"))?;
    match k {
        "delay_ms" => p.delay = x / 1e3,
        "jitter_ms" => p.jitter = x / 1e3,
        "loss" => p.loss = x,
        "burst" => p.burst = x.max(1.0),
        "rate_mbit" => p.rate_bps = x * 1e6,
        "queue_kb" => p.queue_bytes = x * 1024.0,
        _ => return Err(format!("unknown key {k}")),
    }
    Ok(())
}

fn stats_json(l: &Link) -> String {
    let s = &l.stats;
    let sent = s.sent.load(Ordering::Relaxed).max(1);
    format!(
        "{{\"packets\":{},\"bytes\":{},\"lost\":{},\"queue_drops\":{},\"max_backlog_us\":{},\"late_us_max\":{},\"late_us_mean\":{:.2}}}",
        s.packets.load(Ordering::Relaxed),
        s.bytes.load(Ordering::Relaxed),
        s.lost.load(Ordering::Relaxed),
        s.queue_drops.load(Ordering::Relaxed),
        s.max_backlog_us.load(Ordering::Relaxed),
        s.late_us_max.load(Ordering::Relaxed),
        s.late_us_sum.load(Ordering::Relaxed) as f64 / sent as f64
    )
}

fn reset(l: &Link) {
    let s = &l.stats;
    for a in [&s.packets, &s.bytes, &s.lost, &s.queue_drops, &s.max_backlog_us, &s.late_us_max, &s.late_us_sum, &s.sent]
    {
        a.store(0, Ordering::Relaxed);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: pgw-wanemu <tun-a> <tun-b>   (commands on stdin; see the source)");
        std::process::exit(2);
    }
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 1 as libc::c_ulong, 0, 0, 0);
    }
    let fa = tun_open(&args[1]);
    let fb = tun_open(&args[2]);
    let mk = || {
        Arc::new(Link {
            params: Mutex::new(Params::default()),
            q: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            stats: Stats::default(),
        })
    };
    let up = mk(); // a -> b
    let down = mk(); // b -> a
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    {
        let (u, d) = (up.clone(), down.clone());
        thread::spawn(move || reader(fa, u, seed));
        thread::spawn(move || reader(fb, d, seed.rotate_left(17)));
    }
    {
        let (u, d) = (up.clone(), down.clone());
        thread::spawn(move || writer(fb, u));
        thread::spawn(move || writer(fa, d));
    }
    let stdout = std::io::stdout();
    println!("ready");
    stdout.lock().flush().ok();
    for line in std::io::stdin().lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let mut it = line.split_whitespace();
        let reply = match it.next() {
            Some("set") => {
                let which = it.next().unwrap_or("");
                let links: Vec<&Arc<Link>> = match which {
                    "up" => vec![&up],
                    "down" => vec![&down],
                    "both" => vec![&up, &down],
                    _ => vec![],
                };
                if links.is_empty() {
                    "error: set <up|down|both> k=v ...".to_string()
                } else {
                    let items: Vec<&str> = it.collect();
                    let mut err = None;
                    for l in links {
                        let mut p = l.params.lock().unwrap();
                        for kv in &items {
                            if let Err(e) = apply(&mut p, kv) {
                                err = Some(e);
                            }
                        }
                    }
                    err.map(|e| format!("error: {e}")).unwrap_or_else(|| "ok".into())
                }
            }
            Some("stats") => format!("{{\"up\":{},\"down\":{}}}", stats_json(&up), stats_json(&down)),
            Some("reset") => {
                reset(&up);
                reset(&down);
                "ok".into()
            }
            Some("quit") => break,
            _ => "error: unknown command".into(),
        };
        println!("{reply}");
        stdout.lock().flush().ok();
    }
}
