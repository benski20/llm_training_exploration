//! # pgw-lane
//!
//! Service side of the PGW trip lane (chat-speed plan WP-A1): a native front end next to the
//! WebSocket stream channel that serves decode-sized masked products (1..=8 rows per call) with no
//! Python on the hot path when CUDA graphs are available.
//!
//! * [`proto`]: the fixed binary header (public fields only) and the status / refusal codes.
//! * [`server`]: the listener (Unix domain socket with an SO_PEERCRED uid check, or TCP + TLS), single-use tickets issued
//!   by the Python service (`POST /pgw/v1/lane/open`, same authentication as compute), shape, size
//!   and deadline enforcement before a body is read, revocation, draining.
//! * [`admission`]: draining flag, global and per-tenant in-flight caps and the per-tenant row
//!   bucket, **shared** with the Python stream channel while a lane runs (one budget for both).
//! * [`stats`]: counters aggregated here and drained into the service's Prometheus collectors.
//! * [`exec`]: model/linear/ring tables, the executor trait, the executor chain and the device
//!   lock shared with the Python path (the one-call-per-device invariant).
//! * [`cuda`]: replay of torch-captured CUDA graphs through a `dlopen`ed `cuGraphLaunch`, hybrid
//!   spin-then-block wait. Registered only after a startup equality self-check on the device.
//!
//! The provider only ever sees one-time-padded `x_hat` and the resulting `y_hat`, exactly as on
//! the WebSocket path: the lane changes framing and latency, never content.
//! * [`tls`]: the TCP transport, TLS 1.3 only (rustls/ring): the service's certificate, client
//!   certificates verified against its client CA, tickets optionally bound to the client certificate
//!   that requested them. Same frames as the Unix domain socket.
//!
//! Python bindings live in `pgw-fast` (`lane_service.rs`).

pub mod admission;
pub mod cuda;
pub mod exec;
pub mod proto;
pub mod server;
pub mod stats;
mod sync;
pub mod tls;

/// Version of the lane wire format this crate implements.
pub const LANE_PROTOCOL_VERSION: u16 = proto::VERSION;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{Admission, Limits, Refusal};
    use crate::exec::{Call, Done, Executor, LinearEntry, ModelEntry, RingEntry};
    use crate::proto::{Kind, ReplyHeader, RequestHeader, Status, REPLY_HEADER_LEN, TICKET_LEN};
    use crate::server::{LaneServer, ServerConfig};
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// y[j] = sum_i x[i] * (j + 1) mod 2^32 over u32 LE elements (a stand-in "linear").
    struct Fake {
        calls: AtomicUsize,
        sleep: Duration,
    }

    impl Executor for Fake {
        fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(self.sleep);
            if std::time::Instant::now() > call.deadline {
                return Err(Refusal::new(Status::DeadlineExceeded));
            }
            let m = call.out_len / 4 / call.rows as usize;
            let n = call.body.len() / 4 / call.rows as usize;
            for r in 0..call.rows as usize {
                let x = &call.body[r * n * 4..(r + 1) * n * 4];
                let s = x.chunks(4).fold(0u32, |a, c| a.wrapping_add(u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                for j in 0..m {
                    out.extend_from_slice(&s.wrapping_mul(j as u32 + 1).to_le_bytes());
                }
            }
            Ok(Done { queue_us: 0, compute_us: 1, backend: "fake" })
        }
    }

    fn model() -> ModelEntry {
        ModelEntry {
            linears: vec![LinearEntry { n: 4, m: 3, kind: "qkv".into() }, LinearEntry { n: 2, m: 5, kind: "o".into() }],
            rings: vec![RingEntry { name: "K32A12C27".into(), bytes_per_element: 4 }],
        }
    }

    fn start(limits: Limits, max_rows: u16, sleep: Duration) -> (LaneServer, Arc<Fake>, tempdir::Dir) {
        let dir = tempdir::Dir::new();
        let fake = Arc::new(Fake { calls: AtomicUsize::new(0), sleep });
        let cfg = ServerConfig {
            path: dir.0.join("lane.sock"),
            max_rows,
            deadline_cap_ms: 1000,
            poll: Duration::from_millis(20),
            ..ServerConfig::default()
        };
        let s = LaneServer::start(cfg, Arc::new(Admission::new(limits)), Some(fake.clone())).unwrap();
        s.set_model(0, model());
        (s, fake, dir)
    }

    mod tempdir {
        pub struct Dir(pub std::path::PathBuf);
        impl Dir {
            pub fn new() -> Dir {
                static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let p = std::env::temp_dir().join(format!(
                    "pgw-lane-test-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                ));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    fn reply(c: &mut UnixStream) -> (ReplyHeader, Vec<u8>) {
        let mut hb = [0u8; REPLY_HEADER_LEN];
        c.read_exact(&mut hb).unwrap();
        let h = ReplyHeader::parse(&hb).unwrap();
        let mut body = vec![0u8; h.body_len as usize];
        c.read_exact(&mut body).unwrap();
        (h, body)
    }

    fn open(s: &LaneServer, tenant: &str) -> (UnixStream, ReplyHeader) {
        let ticket = [tenant.len() as u8 + 1; TICKET_LEN];
        s.add_ticket(ticket, tenant, 0, Duration::from_secs(5));
        let mut c = UnixStream::connect(s.path()).unwrap();
        let h = RequestHeader {
            kind: Kind::Open,
            request_id: [0; 16],
            linear: 0,
            rows: 0,
            ring: 0,
            deadline_ms: 0,
            body_len: TICKET_LEN as u32,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&ticket).unwrap();
        let (r, _) = reply(&mut c);
        (c, r)
    }

    fn call(c: &mut UnixStream, linear: u32, rows: u16, n: usize, id: u8) -> (ReplyHeader, Vec<u8>) {
        let body: Vec<u8> = (0..rows as usize * n).flat_map(|i| (i as u32 * 7 + id as u32).to_le_bytes()).collect();
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [id; 16],
            linear,
            rows,
            ring: 0,
            deadline_ms: 0,
            body_len: body.len() as u32,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&body).unwrap();
        reply(c)
    }

    #[test]
    fn open_call_and_reuse() {
        let (s, fake, _d) = start(Limits::default(), 8, Duration::ZERO);
        let (mut c, r) = open(&s, "t1");
        assert_eq!(r.status, Status::Ok);
        for id in 1..5u8 {
            let (h, body) = call(&mut c, 0, 2, 4, id);
            assert_eq!((h.status, h.request_id, h.body_len), (Status::Ok, [id; 16], 2 * 3 * 4));
            assert_eq!(body.len(), 24);
        }
        let (h, body) = call(&mut c, 1, 1, 2, 9);
        assert_eq!((h.status, body.len()), (Status::Ok, 20));
        assert_eq!(fake.calls.load(Ordering::SeqCst), 5);
        let st = s.shared().stats.drain();
        assert_eq!(st.requests[&("t1".to_owned(), "ok")].count, 5);
        assert_eq!(st.rows[&("t1".to_owned(), "qkv".to_owned())], 8);
        assert_eq!(st.connections, (1, 0));
        assert_eq!(s.shared().admission.inflight(), 0);
    }

    #[test]
    fn ticket_is_single_use_and_checked() {
        let (s, _f, _d) = start(Limits::default(), 8, Duration::ZERO);
        let ticket = [42u8; TICKET_LEN];
        s.add_ticket(ticket, "t", 0, Duration::from_secs(5));
        let open_with = |t: [u8; TICKET_LEN]| {
            let mut c = UnixStream::connect(s.path()).unwrap();
            let h = RequestHeader {
                kind: Kind::Open,
                request_id: [1; 16],
                linear: 0,
                rows: 0,
                ring: 0,
                deadline_ms: 0,
                body_len: TICKET_LEN as u32,
            };
            c.write_all(&h.encode()).unwrap();
            c.write_all(&t).unwrap();
            reply(&mut c).0.status
        };
        assert_eq!(open_with(ticket), Status::Ok);
        assert_eq!(open_with(ticket), Status::Unauthenticated); // used
        assert_eq!(open_with([43u8; TICKET_LEN]), Status::Unauthenticated); // unknown
        s.add_ticket([44u8; TICKET_LEN], "t", 0, Duration::ZERO);
        assert_eq!(open_with([44u8; TICKET_LEN]), Status::Unauthenticated); // expired
        s.add_ticket([45u8; TICKET_LEN], "t", 7, Duration::from_secs(5));
        assert_eq!(open_with([45u8; TICKET_LEN]), Status::Unauthenticated); // unknown model
                                                                            // a call before open
        let mut c = UnixStream::connect(s.path()).unwrap();
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [3; 16],
            linear: 0,
            rows: 1,
            ring: 0,
            deadline_ms: 0,
            body_len: 16,
        };
        c.write_all(&h.encode()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::NotOpen);
    }

    #[test]
    fn header_and_length_enforcement() {
        let (s, fake, _d) = start(Limits::default(), 2, Duration::ZERO);
        // rows above the lane's max but within the protocol max: fallback, connection kept
        let (mut c, _) = open(&s, "t");
        assert_eq!(call(&mut c, 0, 3, 4, 1).0.status, Status::Fallback);
        assert_eq!(call(&mut c, 0, 1, 4, 2).0.status, Status::Ok);
        // wrong body length: fatal
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [5; 16],
            linear: 0,
            rows: 1,
            ring: 0,
            deadline_ms: 0,
            body_len: 15,
        };
        c.write_all(&h.encode()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::BadLength);
        let mut rest = Vec::new();
        assert_eq!(c.read_to_end(&mut rest).unwrap(), 0); // closed
                                                          // unknown linear / ring: fatal
        let (mut c, _) = open(&s, "u");
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [6; 16],
            linear: 2,
            rows: 1,
            ring: 0,
            deadline_ms: 0,
            body_len: 16,
        };
        c.write_all(&h.encode()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::UnknownLinear);
        let (mut c, _) = open(&s, "v");
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [6; 16],
            linear: 0,
            rows: 1,
            ring: 1,
            deadline_ms: 0,
            body_len: 16,
        };
        c.write_all(&h.encode()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::UnknownLinear);
        // rows above the protocol max: fatal
        let (mut c, _) = open(&s, "w");
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [6; 16],
            linear: 0,
            rows: 9,
            ring: 0,
            deadline_ms: 0,
            body_len: 9 * 16,
        };
        c.write_all(&h.encode()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::BadHeader);
        // garbage header: fatal
        let (mut c, _) = open(&s, "x");
        c.write_all(&[0xAB; 48]).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::BadHeader);
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn quota_drain_deadline_and_revocation() {
        let lim = Limits { tenant_rows_per_s: Some(1.0), tenant_burst_rows: Some(3.0), ..Limits::default() };
        let (s, _f, _d) = start(lim, 8, Duration::ZERO);
        let (mut c, _) = open(&s, "t");
        assert_eq!(call(&mut c, 0, 2, 4, 1).0.status, Status::Ok);
        let (h, _) = call(&mut c, 0, 2, 4, 2);
        assert_eq!(h.status, Status::RateLimited);
        assert!(h.retry_after_ms >= 1);
        // the Python path draws on the same bucket
        assert_eq!(s.shared().admission.admit("t", 1, false), Ok(()));
        assert!(s.shared().admission.admit("t", 1, false).is_err());
        // draining: unavailable, connection kept
        let (mut c2, _) = open(&s, "u");
        s.shared().admission.set_draining(true);
        let (h, _) = call(&mut c2, 0, 1, 4, 3);
        assert_eq!((h.status, h.retry_after_ms), (Status::Unavailable, 1000));
        s.shared().admission.set_draining(false);
        assert_eq!(call(&mut c2, 0, 1, 4, 4).0.status, Status::Ok);
        // revocation closes the tenant's connections at the next call
        s.revoke_tenant("u");
        assert_eq!(call(&mut c2, 0, 1, 4, 5).0.status, Status::Unauthenticated);
        drop(s);
        // deadline: an executor slower than the call's deadline
        let (s, _f, _d2) = start(Limits::default(), 8, Duration::from_millis(30));
        let (mut c, _) = open(&s, "t");
        let body = [0u8; 16];
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [8; 16],
            linear: 0,
            rows: 1,
            ring: 0,
            deadline_ms: 5,
            body_len: 16,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&body).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::DeadlineExceeded);
        assert_eq!(call(&mut c, 0, 1, 4, 9).0.status, Status::Ok); // default deadline = cap (1 s)
    }

    #[test]
    fn no_executor_means_fallback_and_stop_removes_socket() {
        let dir = tempdir::Dir::new();
        let cfg =
            ServerConfig { path: dir.0.join("l.sock"), poll: Duration::from_millis(20), ..ServerConfig::default() };
        let s = LaneServer::start(cfg, Arc::new(Admission::new(Limits::default())), None).unwrap();
        s.set_model(0, model());
        let (mut c, _) = open(&s, "t");
        assert_eq!(call(&mut c, 0, 1, 4, 1).0.status, Status::Fallback);
        // a second server on a live socket is refused
        let cfg2 = ServerConfig { path: dir.0.join("l.sock"), ..ServerConfig::default() };
        assert!(LaneServer::start(cfg2, Arc::new(Admission::new(Limits::default())), None).is_err());
        drop(c);
        let p = s.path().to_path_buf();
        s.stop(Duration::from_secs(2));
        assert!(!p.exists());
        // a stale socket file is replaced
        let l = std::os::unix::net::UnixListener::bind(&p).unwrap();
        drop(l);
        let cfg3 = ServerConfig { path: p.clone(), ..ServerConfig::default() };
        let s3 = LaneServer::start(cfg3, Arc::new(Admission::new(Limits::default())), None).unwrap();
        s3.stop(Duration::from_secs(1));
    }

    #[test]
    fn peer_uid_is_ours() {
        let (a, _b) = UnixStream::pair().unwrap();
        // SAFETY: geteuid has no preconditions.
        assert_eq!(server::peer_uid(&a).unwrap(), unsafe { libc::geteuid() });
    }

    #[test]
    fn foreign_peer_uid_is_refused() {
        let dir = tempdir::Dir::new();
        // SAFETY: geteuid has no preconditions.
        let other = unsafe { libc::geteuid() }.wrapping_add(4242);
        let cfg = ServerConfig {
            path: dir.0.join("p.sock"),
            peer_uids: vec![other],
            poll: Duration::from_millis(20),
            ..ServerConfig::default()
        };
        let s = LaneServer::start(cfg, Arc::new(Admission::new(Limits::default())), None).unwrap();
        let mut c = UnixStream::connect(s.path()).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::PeerRefused);
    }

    fn start_with(cfg: impl FnOnce(&mut ServerConfig), ex: Arc<dyn Executor>) -> (LaneServer, tempdir::Dir) {
        let dir = tempdir::Dir::new();
        let mut c = ServerConfig {
            path: dir.0.join("lane.sock"),
            deadline_cap_ms: 1000,
            poll: Duration::from_millis(20),
            ..ServerConfig::default()
        };
        cfg(&mut c);
        let s = LaneServer::start(c, Arc::new(Admission::new(Limits::default())), Some(ex)).unwrap();
        s.set_model(0, model());
        (s, dir)
    }

    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(5) {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        cond()
    }

    #[test]
    fn connection_cap_refusal_is_fatal_busy() {
        let fake = Arc::new(Fake { calls: AtomicUsize::new(0), sleep: Duration::ZERO });
        let (s, _d) = start_with(|c| c.max_connections = 1, fake);
        let (mut c1, r) = open(&s, "t");
        assert_eq!(r.status, Status::Ok);
        let mut c2 = UnixStream::connect(s.path()).unwrap();
        let (h, _) = reply(&mut c2);
        assert_eq!((h.status, h.retry_after_ms), (Status::Busy, 1000));
        assert!(h.status.is_fatal() && h.status.is_fallback());
        let mut rest = Vec::new();
        assert_eq!(c2.read_to_end(&mut rest).unwrap(), 0); // closed, as the fatal class says
        assert_eq!(call(&mut c1, 0, 1, 4, 1).0.status, Status::Ok); // the held one is unaffected
        drop(c1);
        assert!(wait_for(|| s.connections() == 0));
        assert_eq!(open(&s, "u").1.status, Status::Ok);
    }

    #[test]
    fn open_frame_has_one_budget() {
        let fake = Arc::new(Fake { calls: AtomicUsize::new(0), sleep: Duration::ZERO });
        // stall timeout far above the open budget: the ticket read must still end at open_timeout
        let (s, _d) = start_with(
            |c| {
                c.open_timeout = Duration::from_millis(150);
                c.stall_timeout = Duration::from_secs(30);
            },
            fake,
        );
        let mut c = UnixStream::connect(s.path()).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let t0 = std::time::Instant::now();
        let h = RequestHeader {
            kind: Kind::Open,
            request_id: [0; 16],
            linear: 0,
            rows: 0,
            ring: 0,
            deadline_ms: 0,
            body_len: TICKET_LEN as u32,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&[1u8; 5]).unwrap(); // part of the ticket, then nothing
        let mut rest = Vec::new();
        assert_eq!(c.read_to_end(&mut rest).unwrap(), 0);
        let took = t0.elapsed();
        assert!(took < Duration::from_secs(2), "pre-auth hold {took:?}");
        assert!(wait_for(|| s.connections() == 0));
    }

    #[test]
    fn read_spin_serves_the_same_replies() {
        let fake = Arc::new(Fake { calls: AtomicUsize::new(0), sleep: Duration::ZERO });
        let (s, _d) = start_with(|c| c.read_spin = Duration::from_millis(5), fake.clone());
        let (mut c, r) = open(&s, "t1");
        assert_eq!(r.status, Status::Ok);
        for (id, gap_ms) in [(1u8, 0u64), (2, 2), (3, 12), (4, 0)] {
            std::thread::sleep(Duration::from_millis(gap_ms)); // inside and beyond the spin window
            let (h, body) = call(&mut c, 0, 2, 4, id);
            assert_eq!((h.status, h.request_id, body.len()), (Status::Ok, [id; 16], 24));
        }
        assert_eq!(fake.calls.load(Ordering::SeqCst), 4);
        let t0 = std::time::Instant::now();
        s.stop(Duration::from_secs(2)); // a spinning connection still sees the shutdown
        assert!(t0.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn tickets_are_matched_among_many() {
        let fake = Arc::new(Fake { calls: AtomicUsize::new(0), sleep: Duration::ZERO });
        let (s, _d) = start_with(|_| {}, fake);
        for b in 100..110u8 {
            let mut t = [b; TICKET_LEN];
            t[TICKET_LEN - 1] = 7; // all share a long prefix pattern with their neighbours
            s.add_ticket(t, &format!("t{b}"), 0, Duration::from_secs(5));
        }
        assert_eq!(s.pending_tickets(), 10);
        let (_c, r) = open(&s, "abc"); // ticket [4; 32]
        assert_eq!(r.status, Status::Ok);
        assert_eq!(s.pending_tickets(), 10);
        let mut t = [105u8; TICKET_LEN];
        t[TICKET_LEN - 1] = 7;
        let mut c = UnixStream::connect(s.path()).unwrap();
        let h = RequestHeader {
            kind: Kind::Open,
            request_id: [2; 16],
            linear: 0,
            rows: 0,
            ring: 0,
            deadline_ms: 0,
            body_len: TICKET_LEN as u32,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&t).unwrap();
        assert_eq!(reply(&mut c).0.status, Status::Ok);
        assert_eq!(s.pending_tickets(), 9);
        s.revoke_tenant("t101");
        assert_eq!(s.pending_tickets(), 8);
    }

    /// An executor that panics (a stand-in for any bug on a connection thread).
    struct Panics;

    impl Executor for Panics {
        fn execute(&self, _call: &Call<'_>, _out: &mut Vec<u8>) -> Result<Done, Refusal> {
            panic!("executor bug (expected in this test)");
        }
    }

    #[test]
    fn panic_on_a_connection_leaks_no_counts() {
        let (s, _d) = start_with(|_| {}, Arc::new(Panics));
        let (mut c, r) = open(&s, "t");
        assert_eq!(r.status, Status::Ok);
        let body = [0u8; 16];
        let h = RequestHeader {
            kind: Kind::Call,
            request_id: [1; 16],
            linear: 0,
            rows: 1,
            ring: 0,
            deadline_ms: 0,
            body_len: 16,
        };
        c.write_all(&h.encode()).unwrap();
        c.write_all(&body).unwrap();
        let mut rest = Vec::new();
        assert_eq!(c.read_to_end(&mut rest).unwrap(), 0); // the connection ended, no reply
        assert!(wait_for(|| s.connections() == 0));
        assert_eq!(s.shared().admission.inflight(), 0);
        assert_eq!(s.shared().admission.tenant_inflight("t"), 0);
        // the lane keeps serving
        s.set_executor(Some(Arc::new(Fake { calls: AtomicUsize::new(0), sleep: Duration::ZERO })));
        let (mut c, r) = open(&s, "t");
        assert_eq!(r.status, Status::Ok);
        assert_eq!(call(&mut c, 0, 1, 4, 2).0.status, Status::Ok);
        assert_eq!(s.shared().admission.inflight(), 0);
    }
}
