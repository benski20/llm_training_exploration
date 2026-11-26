//! Trip lane client (chat-speed plan WP-A3): `TripChannel`, one masked round trip per call.
//!
//! The lane carries exactly the bytes of the WebSocket stream channel's one-item compute request
//! (`x_hat` up, `y_hat` down) behind a fixed 36-byte binary header instead of msgpack; the provider
//! learns nothing new. The normative format is the docstring of `pgw.wire.lane` (golden vectors in
//! `pgw/tests/wire/lane_golden.bin`, checked here and in Python):
//!
//! | header | bytes | fields (little-endian) |
//! |---|---|---|
//! | HELLO | 40 | `"PGWH"`, version u16, kind u16 = 1, ticket \[32\] |
//! | REQUEST | 36 | `"PGWT"`, version u16, kind u8 = 2, ring u8, linear u16, rows u16, deadline_ms u32, body_len u32, request_id \[16\] |
//! | REPLY | 36 | `"PGWR"`, version u16, kind u8 (3 hello ack, 4 trip reply), status u8, linear u16, rows u16, compute_us u32, body_len u32, request_id \[16\] |
//!
//! `TripChannel` (Python class `pgw_fast.TripChannel`): a Unix-domain-socket connection, or TCP that is
//! always TLS (rustls, TLS 1.3, server verified against a caller-given CA bundle, ALPN `pgw-lane/1`).
//! `trip` releases the GIL, writes header and body with one `writev` (UDS), waits for the reply with a
//! hybrid wait (non-blocking polls for `spin_us`, then a blocking read bounded by the trip timeout) and
//! reads `y_hat` straight into the caller's buffer. Any I/O or protocol error closes the channel; the
//! caller (`pgw.wire.lane.LaneComputeClient`) then resends the byte-identical request on the regular path.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zeroize::Zeroize;

/// Lane wire version implemented here.
pub const LANE_VERSION: u16 = 1;
pub const HELLO_MAGIC: [u8; 4] = *b"PGWH";
pub const REQUEST_MAGIC: [u8; 4] = *b"PGWT";
pub const REPLY_MAGIC: [u8; 4] = *b"PGWR";
pub const KIND_HELLO: u16 = 1;
pub const KIND_TRIP: u8 = 2;
pub const KIND_HELLO_ACK: u8 = 3;
pub const KIND_TRIP_REPLY: u8 = 4;
pub const HELLO_LEN: usize = 40;
pub const REQUEST_LEN: usize = 36;
pub const REPLY_LEN: usize = 36;
pub const TICKET_LEN: usize = 32;
pub const ALPN: &[u8] = b"pgw-lane/1";

/// Reply statuses (`pgw.wire.lane.Status`).
/// Cache warm-up while a trip waits on the provider (opt-in, `PGW_LANE_WAIT_PREFETCH=1`; timing only).
///
/// A native driver registers, before a lane call, the memory its next steps read right after the reply
/// (this trip's `W r` rows and check vectors, the next trip's input-half vector); the lane call issues
/// software prefetches for those lines **after** the request has left and before it waits for the reply,
/// so the gateway's per-trip work after the reply runs from cache instead of DRAM. Prefetch instructions
/// never fault and never change memory: no value, check or byte on the wire depends on them.
pub mod wait_prefetch {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    /// At most this many bytes are prefetched per lane call (bounded work while the reply may arrive).
    const MAX_BYTES: usize = 4 << 20;

    thread_local! {
        static REGIONS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
    }

    /// `PGW_LANE_WAIT_PREFETCH` (read once; default off).
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            matches!(std::env::var("PGW_LANE_WAIT_PREFETCH").ok().as_deref().map(str::trim),
                     Some("1" | "true" | "yes" | "on"))
        })
    }

    /// Register `len` bytes at `ptr` for the calling thread's next lane call (no-op when off).
    pub fn add(ptr: *const u8, len: usize) {
        if len == 0 || ptr.is_null() || !enabled() {
            return;
        }
        REGIONS.with(|r| r.borrow_mut().push((ptr as usize, len)));
    }

    /// Register a slice (any element type).
    pub fn add_slice<T>(s: &[T]) {
        add(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s));
    }

    /// Forget the registered regions without prefetching them.
    pub fn clear() {
        REGIONS.with(|r| r.borrow_mut().clear());
    }

    /// Prefetch the registered regions (at most `MAX_BYTES`) and forget them.
    pub fn run() {
        REGIONS.with(|r| {
            let mut v = r.borrow_mut();
            let mut budget = MAX_BYTES;
            for &(a, len) in v.iter() {
                let len = len.min(budget);
                budget -= len;
                let mut off = 0;
                while off < len {
                    prefetch((a + off) as *const u8);
                    off += 64;
                }
                if budget == 0 {
                    break;
                }
            }
            v.clear();
        });
    }

    #[inline(always)]
    fn prefetch(p: *const u8) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a prefetch hint never faults and never reads or writes memory architecturally.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.cast::<i8>());
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = p;
    }
}

pub mod status {
    pub const OK: u8 = 0;
    pub const BAD_HEADER: u8 = 1;
    pub const BAD_TICKET: u8 = 2;
    pub const UNKNOWN_LINEAR: u8 = 3;
    pub const BAD_RING: u8 = 4;
    pub const TOO_MANY_ROWS: u8 = 5;
    pub const BAD_LENGTH: u8 = 6;
    pub const DEADLINE: u8 = 7;
    pub const BUSY: u8 = 8;
    pub const DRAINING: u8 = 9;
    pub const INTERNAL: u8 = 10;
    /// After these the server closes the connection.
    pub fn closing(s: u8) -> bool {
        matches!(s, BAD_HEADER | BAD_TICKET | DRAINING | INTERNAL)
    }
}

// ---------------------------------------------------------------------------------------------
// header codec (pure; shared with the golden-vector tests)
// ---------------------------------------------------------------------------------------------

/// A trip request header: the complete list of what the provider sees besides `x_hat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestHeader {
    pub ring: u8,
    pub linear: u16,
    pub rows: u16,
    pub deadline_ms: u32,
    pub body_len: u32,
    pub request_id: [u8; 16],
}

impl RequestHeader {
    pub fn encode(&self) -> [u8; REQUEST_LEN] {
        let mut b = [0u8; REQUEST_LEN];
        b[0..4].copy_from_slice(&REQUEST_MAGIC);
        b[4..6].copy_from_slice(&LANE_VERSION.to_le_bytes());
        b[6] = KIND_TRIP;
        b[7] = self.ring;
        b[8..10].copy_from_slice(&self.linear.to_le_bytes());
        b[10..12].copy_from_slice(&self.rows.to_le_bytes());
        b[12..16].copy_from_slice(&self.deadline_ms.to_le_bytes());
        b[16..20].copy_from_slice(&self.body_len.to_le_bytes());
        b[20..36].copy_from_slice(&self.request_id);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, String> {
        if b.len() != REQUEST_LEN {
            return Err("request header has the wrong size".into());
        }
        if b[0..4] != REQUEST_MAGIC || u16le(b, 4) != LANE_VERSION || b[6] != KIND_TRIP {
            return Err("not a lane v1 trip request".into());
        }
        let h = RequestHeader {
            ring: b[7],
            linear: u16le(b, 8),
            rows: u16le(b, 10),
            deadline_ms: u32le(b, 12),
            body_len: u32le(b, 16),
            request_id: b[20..36].try_into().unwrap(),
        };
        if h.rows == 0 {
            return Err("rows must be >= 1".into());
        }
        Ok(h)
    }
}

/// A reply header (trip reply or hello ack).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyHeader {
    pub kind: u8,
    pub status: u8,
    pub linear: u16,
    pub rows: u16,
    pub compute_us: u32,
    pub body_len: u32,
    pub request_id: [u8; 16],
}

impl ReplyHeader {
    pub fn encode(&self) -> [u8; REPLY_LEN] {
        let mut b = [0u8; REPLY_LEN];
        b[0..4].copy_from_slice(&REPLY_MAGIC);
        b[4..6].copy_from_slice(&LANE_VERSION.to_le_bytes());
        b[6] = self.kind;
        b[7] = self.status;
        b[8..10].copy_from_slice(&self.linear.to_le_bytes());
        b[10..12].copy_from_slice(&self.rows.to_le_bytes());
        b[12..16].copy_from_slice(&self.compute_us.to_le_bytes());
        b[16..20].copy_from_slice(&self.body_len.to_le_bytes());
        b[20..36].copy_from_slice(&self.request_id);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, String> {
        if b.len() != REPLY_LEN {
            return Err("reply header has the wrong size".into());
        }
        if b[0..4] != REPLY_MAGIC || u16le(b, 4) != LANE_VERSION || !(b[6] == KIND_HELLO_ACK || b[6] == KIND_TRIP_REPLY)
        {
            return Err("not a lane v1 reply".into());
        }
        let h = ReplyHeader {
            kind: b[6],
            status: b[7],
            linear: u16le(b, 8),
            rows: u16le(b, 10),
            compute_us: u32le(b, 12),
            body_len: u32le(b, 16),
            request_id: b[20..36].try_into().unwrap(),
        };
        if h.kind == KIND_HELLO_ACK
            && (h.linear != 0 || h.compute_us != 0 || h.body_len != 0 || h.request_id != [0; 16])
        {
            return Err("hello ack with non-zero fields".into());
        }
        if h.status != status::OK && h.body_len != 0 {
            return Err("refusal with a body".into());
        }
        Ok(h)
    }
}

pub fn encode_hello(ticket: &[u8; TICKET_LEN]) -> [u8; HELLO_LEN] {
    let mut b = [0u8; HELLO_LEN];
    b[0..4].copy_from_slice(&HELLO_MAGIC);
    b[4..6].copy_from_slice(&LANE_VERSION.to_le_bytes());
    b[6..8].copy_from_slice(&KIND_HELLO.to_le_bytes());
    b[8..40].copy_from_slice(ticket);
    b
}

pub fn decode_hello(b: &[u8]) -> Result<[u8; TICKET_LEN], String> {
    if b.len() != HELLO_LEN || b[0..4] != HELLO_MAGIC || u16le(b, 4) != LANE_VERSION || u16le(b, 6) != KIND_HELLO {
        return Err("not a lane v1 hello".into());
    }
    Ok(b[8..40].try_into().unwrap())
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

#[cfg(unix)]
pub use chan::{map_service_status, Channel, Framing, LaneError, TripChannel};

/// The connection and the Python class (Unix only: the UDS transport; the TCP+TLS transport is shared).
#[cfg(unix)]
mod chan {
    use super::*;
    use std::io::{self, IoSlice, Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};
    use std::os::unix::net::UnixStream;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use pgw_lane::proto::{
        Kind as SvcKind, ReplyHeader as SvcReply, RequestHeader as SvcRequest, REPLY_HEADER_LEN as SVC_REPLY_LEN,
    };
    use pgw_lane::shm::{self, Region};
    use pyo3::exceptions::{PyConnectionRefusedError, PyOSError};
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
    use std::os::fd::AsRawFd;
    use std::sync::atomic::Ordering;

    // ---------------------------------------------------------------------------------------------
    // connection
    // ---------------------------------------------------------------------------------------------

    enum Conn {
        Uds(UnixStream),
        Tls(Box<TlsConn>),
    }

    /// Bytes of one batched `recv` of TLS records (a reply of the widest 72B linear is ~240 KB).
    const TLS_RAW_BYTES: usize = 256 << 10;

    /// The TLS client end with batched reads: one `recv` takes everything the socket holds (up to
    /// [`TLS_RAW_BYTES`]) and the records are handed to rustls from memory, instead of rustls' own
    /// reads of at most 4 KiB each (a 150-240 KB reply took 37-60 `recv` calls). The bytes rustls sees,
    /// their order and every check are unchanged; only ciphertext sits in `raw`.
    struct TlsConn {
        conn: rustls::ClientConnection,
        sock: TcpStream,
        raw: Vec<u8>,
        /// `raw[lo..hi]`: received, not yet given to rustls.
        lo: usize,
        hi: usize,
    }

    impl TlsConn {
        fn new(s: rustls::StreamOwned<rustls::ClientConnection, TcpStream>) -> Self {
            let (conn, sock) = s.into_parts();
            TlsConn { conn, sock, raw: vec![0u8; TLS_RAW_BYTES], lo: 0, hi: 0 }
        }

        /// Plaintext into `buf` (at least one byte, or `Ok(0)` at a clean close). `dontwait`: one
        /// non-blocking `recv` (`MSG_DONTWAIT`; the socket stays in blocking mode), `WouldBlock` if the
        /// socket has nothing; else a blocking `recv` bounded by the socket's read timeout.
        fn read_plain(&mut self, buf: &mut [u8], dontwait: bool) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            // as rustls' own stream: anything rustls has to send (e.g. a key update answer) goes first
            while self.conn.wants_write() {
                match self.conn.write_tls(&mut self.sock) {
                    Ok(_) => {}
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            loop {
                match self.conn.reader().read(buf) {
                    Ok(n) => return Ok(n),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e),
                }
                if self.lo < self.hi {
                    let mut rd = &self.raw[self.lo..self.hi];
                    let n = self.conn.read_tls(&mut rd)?;
                    self.lo += n;
                    self.conn.process_new_packets().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    continue;
                }
                self.lo = 0;
                self.hi = 0;
                let flags = if dontwait { libc::MSG_DONTWAIT } else { 0 };
                // SAFETY: valid socket, writable buffer of `raw.len()` bytes.
                let n = unsafe {
                    libc::recv(self.sock.as_raw_fd(), self.raw.as_mut_ptr().cast(), self.raw.len(), flags)
                };
                if n < 0 {
                    return Err(io::Error::last_os_error());
                }
                if n == 0 {
                    // end of stream: rustls decides (Ok(0) after close_notify, else UnexpectedEof)
                    self.conn.read_tls(&mut &[][..])?;
                    self.conn.process_new_packets().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    return match self.conn.reader().read(buf) {
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            Err(io::Error::from(io::ErrorKind::UnexpectedEof))
                        }
                        r => r,
                    };
                }
                self.hi = n as usize;
            }
        }
    }

    impl Conn {
        fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
            match self {
                Conn::Uds(s) => s.set_nonblocking(nb),
                Conn::Tls(s) => s.sock.set_nonblocking(nb),
            }
        }

        fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> {
            match self {
                Conn::Uds(s) => s.set_read_timeout(d),
                Conn::Tls(s) => s.sock.set_read_timeout(d),
            }
        }

        fn set_write_timeout(&self, d: Option<Duration>) -> io::Result<()> {
            match self {
                Conn::Uds(s) => s.set_write_timeout(d),
                Conn::Tls(s) => s.sock.set_write_timeout(d),
            }
        }

        /// Header and body: one `writev` on a socket; on TLS both are sealed into records straight from the
        /// caller's buffers (no plaintext copy) and the records go out with vectored socket writes.
        fn send(&mut self, hdr: &[u8], body: &[u8], _scratch: &mut Vec<u8>) -> io::Result<()> {
            match self {
                Conn::Uds(s) => write_all_vectored(s, hdr, body),
                Conn::Tls(s) => {
                    let mut bufs = [IoSlice::new(hdr), IoSlice::new(body)];
                    let mut slices: &mut [IoSlice<'_>] = &mut bufs;
                    while !slices.is_empty() && slices.iter().any(|b| !b.is_empty()) {
                        let n = s.conn.writer().write_vectored(slices)?;
                        if n == 0 {
                            return Err(io::Error::new(io::ErrorKind::WriteZero, "TLS writer took nothing"));
                        }
                        IoSlice::advance_slices(&mut slices, n);
                    }
                    while s.conn.wants_write() {
                        match s.conn.write_tls(&mut s.sock) {
                            Ok(_) => {}
                            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                            Err(e) => return Err(e),
                        }
                    }
                    Ok(())
                }
            }
        }
    }

    impl Read for Conn {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match self {
                Conn::Uds(s) => s.read(buf),
                Conn::Tls(t) => t.read_plain(buf, false),
            }
        }
    }

    /// One non-blocking `recv` on a Unix stream socket (`MSG_DONTWAIT`): the socket itself stays in
    /// blocking mode, so a spin-then-block wait costs no `fcntl` calls.
    fn recv_dontwait(s: &UnixStream, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: valid socket, writable buffer of `buf.len()` bytes.
        let n = unsafe { libc::recv(s.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len(), libc::MSG_DONTWAIT) };
        if n < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(n as usize)
        }
    }

    /// Learned reply time of one (linear, rows) key, for the adaptive spin.
    #[derive(Clone, Copy)]
    struct Est {
        linear: u32,
        rows: u16,
        ns: u64,
    }

    /// Counters of a channel (`TripChannel.stats()`); nothing about payloads.
    #[derive(Clone, Copy, Default)]
    pub struct ChanStats {
        /// OK trips.
        pub trips: u64,
        /// Replies that arrived while spinning (no blocking wait).
        pub spin_hits: u64,
        /// Trips that fell back to a blocking wait (socket read or shm park).
        pub blocked: u64,
        /// Sum of request-sent to reply-header times, nanoseconds.
        pub wait_ns: u64,
        /// Sum of the whole trip (send to body read), nanoseconds.
        pub trip_ns: u64,
        /// Sum of the service's reported device-wait times (`queue_us`, service frames).
        pub queue_us: u64,
        /// Sum of the service's reported compute times (`compute_us`).
        pub compute_us: u64,
        /// Sum of the spin budgets used, nanoseconds.
        pub spin_ns: u64,
    }

    /// Client end of a shm segment ([`pgw_lane::shm`]).
    struct ShmClient {
        reg: Region,
        seq: u64,
    }

    fn write_all_vectored<W: Write>(w: &mut W, a: &[u8], b: &[u8]) -> io::Result<()> {
        let mut bufs = [IoSlice::new(a), IoSlice::new(b)];
        let mut slices: &mut [IoSlice<'_>] = &mut bufs;
        while !slices.is_empty() {
            match w.write_vectored(slices) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "lane socket closed while sending")),
                Ok(n) => IoSlice::advance_slices(&mut slices, n),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Errors of a trip (the channel is closed after any of them).
    #[derive(Debug)]
    pub enum LaneError {
        Io(io::Error),
        Protocol(String),
        Refused(u8),
        Closed,
    }

    impl From<io::Error> for LaneError {
        fn from(e: io::Error) -> Self {
            LaneError::Io(e)
        }
    }

    fn to_py(e: LaneError) -> PyErr {
        match e {
            LaneError::Io(e) => PyErr::from(e),
            LaneError::Protocol(m) => PyValueError::new_err(format!("lane protocol error: {m}")),
            LaneError::Refused(s) => PyConnectionRefusedError::new_err(format!("lane hello refused (status {s})")),
            LaneError::Closed => PyOSError::new_err("lane channel is closed"),
        }
    }

    /// Frame format of a connection: this module's (WP-A3 `PGWH`/`PGWT`) or the service lane's
    /// (`rust/pgw-lane`, WP-A1: `PGWL` 48-byte requests, 40-byte `PGWR` replies). Both carry the same public
    /// fields and the same packed `x_hat` / `y_hat` bytes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Framing {
        Lane,
        Service,
    }

    /// A service-lane (A1) status as this client's [`status`] (the mapping of
    /// `pgw.wire.service_lane.map_service_status`): per-call refusals are a non-closing `BUSY` (that call
    /// goes to the stream channel), every status after which the service closes the connection is a
    /// closing status here.
    pub fn map_service_status(code: u16) -> u8 {
        match code {
            0 => status::OK,
            1 | 2 | 3 | 5 => status::BUSY,
            4 => status::DEADLINE,
            16 | 21 => status::BAD_HEADER,
            19 | 20 => status::BAD_TICKET,
            23 => status::DRAINING,
            _ => status::INTERNAL,
        }
    }

    /// The client end of one lane connection (no Python types; `TripChannel` wraps it).
    pub struct Channel {
        conn: Option<Conn>,
        framing: Framing,
        nonblocking: bool,
        read_to: Option<Duration>,
        write_to: Option<Duration>,
        spin: Duration,
        /// Adaptive spin (opt-in, `set_spin_auto`): the cap; zero = the fixed `spin`.
        spin_auto: Duration,
        est: Vec<Est>,
        stats: ChanStats,
        shm: Option<ShmClient>,
        max_rows: u16,
        scratch: Vec<u8>,
    }

    impl Channel {
        fn new(conn: Conn, spin: Duration) -> Self {
            Channel {
                conn: Some(conn),
                framing: Framing::Lane,
                nonblocking: false,
                read_to: None,
                write_to: None,
                spin,
                spin_auto: Duration::ZERO,
                est: Vec::new(),
                stats: ChanStats::default(),
                shm: None,
                max_rows: 0,
                scratch: Vec::new(),
            }
        }

        /// Adaptive reply spin (opt-in): per (linear, rows) the channel learns the reply time (moving
        /// average of request-sent to reply-header) and busy-polls for 1.25x that + 20 us, at least the
        /// fixed spin and at most `cap` (zero turns it off). Waiting only: the bytes and their order are
        /// unchanged.
        pub fn set_spin_auto(&mut self, cap: Duration) {
            self.spin_auto = cap.min(Duration::from_millis(100));
        }

        /// Counters so far.
        pub fn stats(&self) -> ChanStats {
            self.stats
        }

        /// `"uds"`, `"shm"` or `"tls"` (`"closed"` once closed).
        pub fn transport(&self) -> &'static str {
            match (&self.conn, &self.shm) {
                (None, _) => "closed",
                (Some(Conn::Uds(_)), Some(_)) => "shm",
                (Some(Conn::Uds(_)), None) => "uds",
                (Some(Conn::Tls(_)), _) => "tls",
            }
        }

        fn spin_for(&self, linear: u32, rows: u16) -> Duration {
            if self.spin_auto.is_zero() {
                return self.spin;
            }
            match self.est.iter().find(|e| e.linear == linear && e.rows == rows) {
                None => self.spin_auto.max(self.spin),
                // A reply expected after the cap (a slow provider: a CPU GEMM, a far link) is not spun
                // for: a spin to the cap would almost never hit and only takes a core from the rest of
                // the process (measured: 400 us of CPU per trip, no hit, against a CPU provider). The
                // estimate keeps learning from blocked waits, so the spin returns when replies speed up.
                Some(e) if e.ns > self.spin_auto.as_nanos() as u64 => Duration::ZERO,
                Some(e) => {
                    let want = Duration::from_nanos(e.ns + e.ns / 4 + 20_000);
                    want.clamp(self.spin.min(self.spin_auto), self.spin_auto.max(self.spin))
                }
            }
        }

        fn learn(&mut self, linear: u32, rows: u16, waited: Duration) {
            if self.spin_auto.is_zero() {
                return;
            }
            // a wait far beyond the cap (a stall, a parked peer) teaches no more than the cap
            let ns = (waited.as_nanos() as u64).min(2 * self.spin_auto.as_nanos() as u64);
            match self.est.iter_mut().find(|e| e.linear == linear && e.rows == rows) {
                Some(e) => e.ns = e.ns - e.ns / 4 + ns / 4,
                None => {
                    if self.est.len() < 4096 {
                        self.est.push(Est { linear, rows, ns });
                    }
                }
            }
        }

        fn account(&mut self, spin: Duration, waited: Duration, total: Duration, blocked: bool, q: u32, c: u32) {
            let st = &mut self.stats;
            st.trips += 1;
            if blocked {
                st.blocked += 1;
            } else {
                st.spin_hits += 1;
            }
            st.wait_ns += waited.as_nanos() as u64;
            st.trip_ns += total.as_nanos() as u64;
            st.spin_ns += spin.as_nanos() as u64;
            st.queue_us += u64::from(q);
            st.compute_us += u64::from(c);
        }

        /// Move this service-lane UDS connection to a shared-memory segment (opt-in; [`pgw_lane::shm`]).
        /// `Ok(false)`: the service does not offer it, the connection stays on the socket. Errors close
        /// the channel.
        pub fn attach_shm(&mut self, timeout: Duration) -> Result<bool, LaneError> {
            if self.shm.is_some() {
                return Ok(true);
            }
            if self.framing != Framing::Service || !matches!(self.conn, Some(Conn::Uds(_))) {
                return Ok(false);
            }
            let res = (|| {
                let hdr = SvcRequest {
                    kind: SvcKind::Shm,
                    request_id: [0; 16],
                    linear: 0,
                    rows: 0,
                    ring: 0,
                    deadline_ms: 0,
                    body_len: 0,
                }
                .encode();
                self.set_nb(false)?;
                self.timeouts(Some(timeout), Some(timeout))?;
                let Some(Conn::Uds(sock)) = self.conn.as_mut() else { return Err(LaneError::Closed) };
                write_all_vectored(sock, &hdr, &[])?;
                let mut rb = [0u8; SVC_REPLY_LEN];
                let fd = shm::recv_with_fd(sock, &mut rb)?;
                let r = SvcReply::parse(&rb).ok_or_else(|| LaneError::Protocol("not a lane shm reply".into()))?;
                if r.body_len != 0 {
                    return Err(LaneError::Protocol("lane shm reply with a body".into()));
                }
                match (r.status.code(), fd) {
                    (0, Some(fd)) => {
                        let reg = Region::attach(&fd).map_err(LaneError::Io)?;
                        drop(fd); // the mapping stays
                        let seq = reg.req_seq().load(Ordering::SeqCst);
                        self.shm = Some(ShmClient { reg, seq });
                        Ok(true)
                    }
                    (0, None) => Err(LaneError::Protocol("lane shm reply without a segment".into())),
                    (c, _) if c == pgw_lane::proto::Status::Fallback.code() => Ok(false),
                    (c, _) => Err(LaneError::Refused(map_service_status(c))),
                }
            })();
            if res.is_err() {
                self.close();
            }
            res
        }

        /// Open over a connected Unix domain socket and run the hello.
        pub fn over_uds(
            sock: UnixStream,
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
        ) -> Result<Self, LaneError> {
            let mut ch = Channel::new(Conn::Uds(sock), spin);
            ch.hello(ticket, timeout)?;
            Ok(ch)
        }

        pub fn connect_uds(
            path: &str,
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
        ) -> Result<Self, LaneError> {
            let sock = UnixStream::connect(path)?;
            Self::over_uds(sock, ticket, timeout, spin)
        }

        /// The service lane (A1 frames) over a connected Unix domain socket; `max_rows` from the ticket.
        pub fn over_uds_service(
            sock: UnixStream,
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
            max_rows: u16,
        ) -> Result<Self, LaneError> {
            let mut ch = Channel::new(Conn::Uds(sock), spin);
            ch.framing = Framing::Service;
            ch.max_rows = max_rows;
            ch.hello(ticket, timeout)?;
            Ok(ch)
        }

        /// TCP + TLS 1.3 (rustls/ring), the server verified against `ca_pem` for `server_name`.
        pub fn connect_tls(
            host: &str,
            port: u16,
            server_name: &str,
            ca_pem: &[u8],
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
        ) -> Result<Self, LaneError> {
            Self::connect_tls_with(host, port, server_name, ca_pem, None, ticket, timeout, spin, Framing::Lane, 0)
        }

        /// The service lane (A1 frames) over TCP + TLS 1.3, optionally presenting a client certificate
        /// (`client` = PEM chain and PEM private key); `max_rows` from the ticket.
        #[allow(clippy::too_many_arguments)]
        pub fn connect_tls_service(
            host: &str,
            port: u16,
            server_name: &str,
            ca_pem: &[u8],
            client: Option<(&[u8], &[u8])>,
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
            max_rows: u16,
        ) -> Result<Self, LaneError> {
            Self::connect_tls_with(
                host,
                port,
                server_name,
                ca_pem,
                client,
                ticket,
                timeout,
                spin,
                Framing::Service,
                max_rows,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn connect_tls_with(
            host: &str,
            port: u16,
            server_name: &str,
            ca_pem: &[u8],
            client: Option<(&[u8], &[u8])>,
            ticket: &[u8; TICKET_LEN],
            timeout: Duration,
            spin: Duration,
            framing: Framing,
            max_rows: u16,
        ) -> Result<Self, LaneError> {
            let mut roots = rustls::RootCertStore::empty();
            for cert in CertificateDer::pem_slice_iter(ca_pem) {
                let cert = cert.map_err(|e| LaneError::Protocol(format!("CA bundle: {e}")))?;
                roots.add(cert).map_err(|e| LaneError::Protocol(format!("CA bundle: {e}")))?;
            }
            if roots.is_empty() {
                return Err(LaneError::Protocol("CA bundle holds no certificate".into()));
            }
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let builder = rustls::ClientConfig::builder_with_provider(provider)
                .with_protocol_versions(&[&rustls::version::TLS13])
                .map_err(|e| LaneError::Protocol(format!("TLS config: {e}")))?
                .with_root_certificates(roots);
            let mut cfg = match client {
                None => builder.with_no_client_auth(),
                Some((cert_pem, key_pem)) => {
                    let mut chain = Vec::new();
                    for c in CertificateDer::pem_slice_iter(cert_pem) {
                        chain.push(c.map_err(|e| LaneError::Protocol(format!("client certificate: {e}")))?);
                    }
                    if chain.is_empty() {
                        return Err(LaneError::Protocol("client certificate file holds no certificate".into()));
                    }
                    let key = PrivateKeyDer::from_pem_slice(key_pem)
                        .map_err(|e| LaneError::Protocol(format!("client key: {e}")))?;
                    builder
                        .with_client_auth_cert(chain, key)
                        .map_err(|e| LaneError::Protocol(format!("client certificate/key: {e}")))?
                }
            };
            cfg.alpn_protocols = vec![ALPN.to_vec()];
            let name = ServerName::try_from(server_name.to_string())
                .map_err(|e| LaneError::Protocol(format!("server name: {e}")))?;
            let mut tls = rustls::ClientConnection::new(Arc::new(cfg), name)
                .map_err(|e| LaneError::Protocol(format!("TLS: {e}")))?;
            tls.set_buffer_limit(None); // a whole request is sealed at once and flushed in `send`
            let mut last = io::Error::new(io::ErrorKind::NotFound, "host resolves to no address");
            let mut tcp = None;
            for addr in (host, port).to_socket_addrs()? {
                match TcpStream::connect_timeout(&addr, timeout) {
                    Ok(s) => {
                        tcp = Some(s);
                        break;
                    }
                    Err(e) => last = e,
                }
            }
            let tcp = tcp.ok_or(last)?;
            tcp.set_nodelay(true)?;
            tcp.set_read_timeout(Some(timeout))?;
            tcp.set_write_timeout(Some(timeout))?;
            let mut stream = rustls::StreamOwned::new(tls, tcp);
            while stream.conn.is_handshaking() {
                stream.conn.complete_io(&mut stream.sock)?;
            }
            match stream.conn.alpn_protocol() {
                None => {}
                Some(p) if p == ALPN => {}
                Some(_) => return Err(LaneError::Protocol("the server selected another ALPN protocol".into())),
            }
            let mut ch = Channel::new(Conn::Tls(Box::new(TlsConn::new(stream))), spin);
            ch.framing = framing;
            ch.max_rows = max_rows;
            ch.read_to = Some(timeout);
            ch.write_to = Some(timeout);
            ch.hello(ticket, timeout)?;
            Ok(ch)
        }

        fn hello(&mut self, ticket: &[u8; TICKET_LEN], timeout: Duration) -> Result<(), LaneError> {
            if self.framing == Framing::Service {
                return self.hello_service(ticket, timeout);
            }
            let mut hello = encode_hello(ticket);
            let deadline = Instant::now() + timeout;
            let res = (|| {
                self.timeouts(Some(timeout), Some(timeout))?;
                let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
                conn.send(&hello, &[], &mut self.scratch)?;
                let mut hdr = [0u8; REPLY_LEN];
                self.read_hybrid(&mut hdr, None, deadline)?;
                let h = ReplyHeader::decode(&hdr).map_err(LaneError::Protocol)?;
                if h.kind != KIND_HELLO_ACK {
                    return Err(LaneError::Protocol("expected a hello ack".into()));
                }
                if h.status != status::OK {
                    return Err(LaneError::Refused(h.status));
                }
                if h.rows == 0 {
                    return Err(LaneError::Protocol("hello ack serves no rows".into()));
                }
                self.max_rows = h.rows;
                Ok(())
            })();
            hello.zeroize();
            self.scratch.zeroize();
            if res.is_err() {
                self.close();
            }
            res
        }

        /// The service lane's `open` frame (48-byte header, kind 1, body = the ticket) and its 40-byte answer.
        fn hello_service(&mut self, ticket: &[u8; TICKET_LEN], timeout: Duration) -> Result<(), LaneError> {
            let hdr = SvcRequest {
                kind: SvcKind::Open,
                request_id: [0; 16],
                linear: 0,
                rows: 0,
                ring: 0,
                deadline_ms: 0,
                body_len: TICKET_LEN as u32,
            }
            .encode();
            let deadline = Instant::now() + timeout;
            let res = (|| {
                if self.max_rows == 0 {
                    return Err(LaneError::Protocol("the ticket serves no rows".into()));
                }
                self.timeouts(Some(timeout), Some(timeout))?;
                let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
                conn.send(&hdr, &ticket[..], &mut self.scratch)?;
                let mut rb = [0u8; SVC_REPLY_LEN];
                let _ = self.read_hybrid(&mut rb, None, deadline)?;
                let r = SvcReply::parse(&rb).ok_or_else(|| LaneError::Protocol("not a lane open reply".into()))?;
                if r.body_len != 0 {
                    return Err(LaneError::Protocol("lane open reply with a body".into()));
                }
                if r.status.code() != 0 {
                    return Err(LaneError::Refused(map_service_status(r.status.code())));
                }
                Ok(())
            })();
            if res.is_err() {
                self.close();
            }
            res
        }

        fn timeouts(&mut self, read: Option<Duration>, write: Option<Duration>) -> io::Result<()> {
            let conn = self.conn.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "closed"))?;
            if self.read_to != read {
                conn.set_read_timeout(read)?;
                self.read_to = read;
            }
            if self.write_to != write {
                conn.set_write_timeout(write)?;
                self.write_to = write;
            }
            Ok(())
        }

        fn set_nb(&mut self, nb: bool) -> io::Result<()> {
            if self.nonblocking != nb {
                self.conn
                    .as_ref()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected, "closed"))?
                    .set_nonblocking(nb)?;
                self.nonblocking = nb;
            }
            Ok(())
        }

        /// Fill `buf`: non-blocking polls until `spin_until` (if any), then blocking reads; never past
        /// `deadline` by more than one read timeout.
        fn read_hybrid(
            &mut self,
            buf: &mut [u8],
            spin_until: Option<Instant>,
            deadline: Instant,
        ) -> Result<bool, LaneError> {
            let mut pos = 0;
            let mut spin = spin_until;
            let mut blocked = false;
            while pos < buf.len() {
                if let Some(until) = spin {
                    // non-blocking polls without switching the socket's mode (no fcntl per trip)
                    self.set_nb(false)?;
                    let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
                    let got = match conn {
                        Conn::Uds(s) => recv_dontwait(s, &mut buf[pos..]),
                        Conn::Tls(t) => t.read_plain(&mut buf[pos..], true),
                    };
                    match got {
                        Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into()),
                        Ok(n) => pos += n,
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            if Instant::now() >= until {
                                spin = None;
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(e) => return Err(e.into()),
                    }
                } else {
                    blocked = true;
                    self.set_nb(false)?;
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(io::Error::new(io::ErrorKind::TimedOut, "lane trip timed out").into());
                    }
                    // Re-arm the timeout only when it would otherwise overshoot the deadline by > 10%.
                    let want = match self.read_to {
                        Some(t) if t <= left && t * 10 >= left * 9 => t,
                        _ => left,
                    };
                    let wto = self.write_to;
                    self.timeouts(Some(want), wto)?;
                    let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
                    match conn.read(&mut buf[pos..]) {
                        Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into()),
                        Ok(n) => pos += n,
                        Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {
                            if Instant::now() >= deadline {
                                return Err(io::Error::new(io::ErrorKind::TimedOut, "lane trip timed out").into());
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            Ok(blocked)
        }

        pub fn closed(&self) -> bool {
            self.conn.is_none()
        }

        pub fn max_rows(&self) -> u16 {
            self.max_rows
        }

        pub fn close(&mut self) {
            self.shm = None;
            if let Some(c) = self.conn.take() {
                match c {
                    Conn::Uds(s) => {
                        let _ = s.shutdown(std::net::Shutdown::Both);
                    }
                    Conn::Tls(mut s) => {
                        s.conn.send_close_notify();
                        let _ = s.conn.complete_io(&mut s.sock);
                        let _ = s.sock.shutdown(std::net::Shutdown::Both);
                    }
                }
            }
        }

        /// One trip. Returns `(status, compute_us)`; on OK, `out` holds the reply body (its length must be
        /// exactly `out.len()`). Any error closes the channel.
        pub fn trip(
            &mut self,
            h: &RequestHeader,
            body: &[u8],
            out: &mut [u8],
            timeout: Duration,
        ) -> Result<(u8, u32), LaneError> {
            let res = self.trip_inner(h, body, out, timeout);
            match &res {
                Ok((s, _)) if status::closing(*s) => self.close(),
                Err(_) => self.close(),
                _ => {}
            }
            res
        }

        fn trip_inner(
            &mut self,
            h: &RequestHeader,
            body: &[u8],
            out: &mut [u8],
            timeout: Duration,
        ) -> Result<(u8, u32), LaneError> {
            if self.conn.is_none() {
                return Err(LaneError::Closed);
            }
            if h.body_len as usize != body.len() || h.rows == 0 {
                return Err(LaneError::Protocol("header does not describe the body".into()));
            }
            if self.framing == Framing::Service {
                return self.trip_service(h, body, out, timeout);
            }
            let start = Instant::now();
            let deadline = start + timeout;
            let hdr = h.encode();
            self.set_nb(false)?;
            let rto = self.read_to;
            let wto = Some(match self.write_to {
                Some(t) if t <= timeout && t * 10 >= timeout * 9 => t,
                _ => timeout.max(Duration::from_millis(1)),
            });
            self.timeouts(rto, wto)?;
            let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
            conn.send(&hdr, body, &mut self.scratch)?;
            super::wait_prefetch::run();
            let sent = Instant::now();
            let spin = self.spin_for(u32::from(h.linear), h.rows);
            let spin_until = if spin.is_zero() { None } else { Some(sent + spin) };
            let mut rh = [0u8; REPLY_LEN];
            let blocked = self.read_hybrid(&mut rh, spin_until, deadline)?;
            let waited = sent.elapsed();
            let r = ReplyHeader::decode(&rh).map_err(LaneError::Protocol)?;
            if r.kind != KIND_TRIP_REPLY || r.request_id != h.request_id || r.linear != h.linear || r.rows != h.rows {
                return Err(LaneError::Protocol("reply does not answer the request".into()));
            }
            if r.status != status::OK {
                return Ok((r.status, r.compute_us));
            }
            if r.body_len as usize != out.len() {
                return Err(LaneError::Protocol("reply body has the wrong length".into()));
            }
            let spin_left = self.body_spin(spin_until);
            self.read_hybrid(out, spin_left, deadline)?;
            self.set_nb(false)?;
            self.learn(u32::from(h.linear), h.rows, waited);
            self.account(spin, waited, start.elapsed(), blocked, 0, r.compute_us);
            Ok((status::OK, r.compute_us))
        }

        /// Spin deadline for a reply body whose header has arrived (the body follows in the same write):
        /// what is left of the header's spin, and with the adaptive spin at least 50 us more.
        fn body_spin(&self, spin_until: Option<Instant>) -> Option<Instant> {
            let now = Instant::now();
            let left = spin_until.filter(|t| now < *t);
            if self.spin_auto.is_zero() {
                return left;
            }
            let more = now + Duration::from_micros(50);
            Some(left.map_or(more, |t| t.max(more)))
        }

        /// One service-lane (A1) call: the same public fields (request id, linear, rows, ring, deadline,
        /// body length) in the 48-byte header, the same body; the status is mapped to this client's codes.
        fn trip_service(
            &mut self,
            h: &RequestHeader,
            body: &[u8],
            out: &mut [u8],
            timeout: Duration,
        ) -> Result<(u8, u32), LaneError> {
            let start = Instant::now();
            let deadline = start + timeout;
            let hdr = SvcRequest {
                kind: SvcKind::Call,
                request_id: h.request_id,
                linear: u32::from(h.linear),
                rows: h.rows,
                ring: u16::from(h.ring),
                deadline_ms: h.deadline_ms,
                body_len: h.body_len,
            }
            .encode();
            if self.shm.is_some() {
                return self.trip_shm(h, &hdr, body, out, start, deadline);
            }
            self.set_nb(false)?;
            let rto = self.read_to;
            let wto = Some(match self.write_to {
                Some(t) if t <= timeout && t * 10 >= timeout * 9 => t,
                _ => timeout.max(Duration::from_millis(1)),
            });
            self.timeouts(rto, wto)?;
            let conn = self.conn.as_mut().ok_or(LaneError::Closed)?;
            conn.send(&hdr, body, &mut self.scratch)?;
            super::wait_prefetch::run();
            let sent = Instant::now();
            let spin = self.spin_for(u32::from(h.linear), h.rows);
            let spin_until = if spin.is_zero() { None } else { Some(sent + spin) };
            let mut rb = [0u8; SVC_REPLY_LEN];
            let blocked = self.read_hybrid(&mut rb, spin_until, deadline)?;
            let waited = sent.elapsed();
            let r = SvcReply::parse(&rb).ok_or_else(|| LaneError::Protocol("not a lane reply".into()))?;
            if r.request_id != h.request_id {
                return Err(LaneError::Protocol("reply does not answer the request".into()));
            }
            if r.status.code() != 0 {
                if r.body_len != 0 {
                    return Err(LaneError::Protocol("refusal with a body".into()));
                }
                return Ok((map_service_status(r.status.code()), r.compute_us));
            }
            if r.body_len as usize != out.len() {
                return Err(LaneError::Protocol("reply body has the wrong length".into()));
            }
            let spin_left = self.body_spin(spin_until);
            self.read_hybrid(out, spin_left, deadline)?;
            self.set_nb(false)?;
            self.learn(u32::from(h.linear), h.rows, waited);
            self.account(spin, waited, start.elapsed(), blocked, r.queue_us, r.compute_us);
            Ok((status::OK, r.compute_us))
        }

        /// One service-lane call through the shm segment: the same header and body bytes as on the
        /// socket, written into the segment; the reply header and body read back from it.
        fn trip_shm(
            &mut self,
            h: &RequestHeader,
            hdr: &[u8],
            body: &[u8],
            out: &mut [u8],
            start: Instant,
            deadline: Instant,
        ) -> Result<(u8, u32), LaneError> {
            let spin = self.spin_for(u32::from(h.linear), h.rows);
            let (Some(Conn::Uds(sock)), Some(sc)) = (self.conn.as_ref(), self.shm.as_mut()) else {
                return Err(LaneError::Closed);
            };
            let reg = &sc.reg;
            if body.len() > reg.req_cap() || out.len() > reg.rep_cap() {
                return Err(LaneError::Protocol("call does not fit the shm segment".into()));
            }
            reg.write(reg.req_body(), body);
            reg.write(shm::OFF_REQ_HDR, hdr);
            sc.seq = sc.seq.wrapping_add(1);
            let seq = sc.seq;
            reg.req_seq().store(seq, Ordering::SeqCst);
            if reg.srv_parked().load(Ordering::SeqCst) != 0 {
                shm::ring(sock)?;
            }
            super::wait_prefetch::run();
            let sent = Instant::now();
            let ready = || reg.rep_seq().load(Ordering::Acquire) == seq;
            let mut blocked = false;
            let spin_until = sent + spin;
            let mut i: u32 = 0;
            while !ready() {
                i = i.wrapping_add(1);
                if i % 32 == 0 && Instant::now() >= spin_until {
                    // park: flag, re-check, block on the doorbell (bounded by the deadline)
                    blocked = true;
                    loop {
                        if shm::park_or_ready(reg.cli_parked(), &ready) {
                            break;
                        }
                        let left = deadline.saturating_duration_since(Instant::now());
                        if left.is_zero() {
                            reg.cli_parked().store(0, Ordering::SeqCst);
                            return Err(io::Error::new(io::ErrorKind::TimedOut, "lane trip timed out").into());
                        }
                        sock.set_read_timeout(Some(left))?;
                        let w = shm::wait_bell(sock);
                        reg.cli_parked().store(0, Ordering::SeqCst);
                        match w? {
                            shm::Wake::Closed => return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into()),
                            shm::Wake::Rang | shm::Wake::Timeout => {}
                        }
                        if ready() {
                            break;
                        }
                    }
                    break;
                }
                std::hint::spin_loop();
            }
            let waited = sent.elapsed();
            let mut rb = [0u8; SVC_REPLY_LEN];
            reg.read(shm::OFF_REP_HDR, &mut rb);
            let r = SvcReply::parse(&rb).ok_or_else(|| LaneError::Protocol("not a lane reply".into()))?;
            if r.request_id != h.request_id {
                return Err(LaneError::Protocol("reply does not answer the request".into()));
            }
            if r.status.code() != 0 {
                if r.body_len != 0 {
                    return Err(LaneError::Protocol("refusal with a body".into()));
                }
                return Ok((map_service_status(r.status.code()), r.compute_us));
            }
            if r.body_len as usize != out.len() {
                return Err(LaneError::Protocol("reply body has the wrong length".into()));
            }
            reg.read(reg.rep_body(), out);
            // the socket's read timeout was changed while parked: let `timeouts` re-arm it
            if blocked {
                self.read_to = None;
            }
            self.learn(u32::from(h.linear), h.rows, waited);
            self.account(spin, waited, start.elapsed(), blocked, r.queue_us, r.compute_us);
            Ok((status::OK, r.compute_us))
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Python
    // ---------------------------------------------------------------------------------------------

    fn secs(timeout_s: f64) -> PyResult<Duration> {
        if !(timeout_s.is_finite() && timeout_s > 0.0 && timeout_s < 1e7) {
            return Err(PyValueError::new_err("timeout_s must be a positive number of seconds"));
        }
        Ok(Duration::from_secs_f64(timeout_s))
    }

    /// One lane connection (see the module docs). Thread-safe: one trip at a time (a lock).
    #[pyclass(module = "pgw_fast", name = "TripChannel", frozen)]
    pub struct TripChannel {
        inner: Mutex<Channel>,
        max_rows: u16,
    }

    impl TripChannel {
        fn wrap(ch: Channel) -> Self {
            let max_rows = ch.max_rows();
            TripChannel { inner: Mutex::new(ch), max_rows }
        }

        /// One trip for crate callers (the native per-trip driver, `tripnative.rs`): [`Channel::trip`]
        /// under the channel's lock. Call without the GIL. Any error closes the channel.
        pub(crate) fn trip_raw(
            &self,
            h: &RequestHeader,
            body: &[u8],
            out: &mut [u8],
            timeout: Duration,
        ) -> Result<(u8, u32), LaneError> {
            self.inner.lock().unwrap_or_else(|p| p.into_inner()).trip(h, body, out, timeout)
        }

        /// Largest rows per trip (crate callers; the hello ack's).
        pub(crate) fn max_rows_raw(&self) -> u16 {
            self.max_rows
        }
    }

    #[pymethods]
    impl TripChannel {
        /// Connect to the lane's Unix domain socket at `path` and present `ticket` (32 bytes).
        #[staticmethod]
        #[pyo3(signature = (path, ticket, timeout_s, spin_us = 50))]
        fn connect_uds(
            py: Python<'_>,
            path: String,
            ticket: &Bound<'_, PyAny>,
            timeout_s: f64,
            spin_us: u64,
        ) -> PyResult<Self> {
            let t = ticket_of(ticket)?;
            let to = secs(timeout_s)?;
            let spin = Duration::from_micros(spin_us.min(1_000_000));
            let ch = py.allow_threads(|| {
                let sock = connect_uds_timeout(&path, to)?;
                Channel::over_uds(sock, &t, to, spin)
            });
            Ok(Self::wrap(ch.map_err(to_py)?))
        }

        /// Connect to `host:port` with TLS 1.3, verifying the server as `server_name` against `ca_pem`
        /// (PEM bundle), and present `ticket`. There is no plaintext TCP variant.
        #[staticmethod]
        #[pyo3(signature = (host, port, server_name, ca_pem, ticket, timeout_s, spin_us = 50))]
        #[allow(clippy::too_many_arguments)]
        fn connect_tls(
            py: Python<'_>,
            host: String,
            port: u16,
            server_name: String,
            ca_pem: Vec<u8>,
            ticket: &Bound<'_, PyAny>,
            timeout_s: f64,
            spin_us: u64,
        ) -> PyResult<Self> {
            let t = ticket_of(ticket)?;
            let to = secs(timeout_s)?;
            let spin = Duration::from_micros(spin_us.min(1_000_000));
            let ch = py.allow_threads(|| Channel::connect_tls(&host, port, &server_name, &ca_pem, &t, to, spin));
            Ok(Self::wrap(ch.map_err(to_py)?))
        }

        /// Connect to the **service** lane's Unix domain socket (`rust/pgw-lane` frames, `pgw.service.lane`)
        /// and present `ticket`; `max_rows` is the ticket's.
        #[staticmethod]
        #[pyo3(signature = (path, ticket, timeout_s, max_rows, spin_us = 50))]
        fn connect_service_uds(
            py: Python<'_>,
            path: String,
            ticket: &Bound<'_, PyAny>,
            timeout_s: f64,
            max_rows: u16,
            spin_us: u64,
        ) -> PyResult<Self> {
            let t = ticket_of(ticket)?;
            let to = secs(timeout_s)?;
            let spin = Duration::from_micros(spin_us.min(1_000_000));
            let ch = py.allow_threads(|| {
                let sock = connect_uds_timeout(&path, to)?;
                Channel::over_uds_service(sock, &t, to, spin, max_rows)
            });
            Ok(Self::wrap(ch.map_err(to_py)?))
        }

        /// Connect to the **service** lane on `host:port` with TLS 1.3 (server verified as `server_name`
        /// against `ca_pem`; with `client_cert_pem` / `client_key_pem` it presents that client certificate)
        /// and present `ticket`; `max_rows` is the ticket's. There is no plaintext TCP variant.
        #[staticmethod]
        #[pyo3(signature = (host, port, server_name, ca_pem, ticket, timeout_s, max_rows, spin_us = 50,
                            client_cert_pem = None, client_key_pem = None))]
        #[allow(clippy::too_many_arguments)]
        fn connect_service_tls(
            py: Python<'_>,
            host: String,
            port: u16,
            server_name: String,
            ca_pem: Vec<u8>,
            ticket: &Bound<'_, PyAny>,
            timeout_s: f64,
            max_rows: u16,
            spin_us: u64,
            client_cert_pem: Option<Vec<u8>>,
            client_key_pem: Option<Vec<u8>>,
        ) -> PyResult<Self> {
            let t = ticket_of(ticket)?;
            let to = secs(timeout_s)?;
            let spin = Duration::from_micros(spin_us.min(1_000_000));
            let key = client_key_pem.map(zeroize::Zeroizing::new);
            let client = match (&client_cert_pem, &key) {
                (Some(c), Some(k)) => Some((c.as_slice(), k.as_slice())),
                (None, None) => None,
                _ => return Err(PyValueError::new_err("client_cert_pem and client_key_pem go together")),
            };
            let ch = py.allow_threads(|| {
                Channel::connect_tls_service(&host, port, &server_name, &ca_pem, client, &t, to, spin, max_rows)
            });
            Ok(Self::wrap(ch.map_err(to_py)?))
        }

        /// One trip: send the header and `body` (`x_hat`), read the reply into `out` (writable, exactly the
        /// reply's size). Returns `(status, compute_us)`. GIL released throughout. Raises `OSError` /
        /// `ValueError` (the channel is then closed).
        #[pyo3(signature = (ring, linear, rows, request_id, deadline_ms, body, out, timeout_s))]
        #[allow(clippy::too_many_arguments)]
        fn trip(
            &self,
            py: Python<'_>,
            ring: u8,
            linear: u16,
            rows: u16,
            request_id: &[u8],
            deadline_ms: u32,
            body: &Bound<'_, PyAny>,
            out: &Bound<'_, PyAny>,
            timeout_s: f64,
        ) -> PyResult<(u8, u32)> {
            let rid = rid_of(request_id)?;
            let to = secs(timeout_s)?;
            let bb = PyBuffer::<u8>::get(body).map_err(|_| PyTypeError::new_err("body must be a bytes-like object"))?;
            let ob = PyBuffer::<u8>::get(out).map_err(|_| PyTypeError::new_err("out must be a writable buffer"))?;
            if !bb.is_c_contiguous() || !ob.is_c_contiguous() {
                return Err(PyValueError::new_err("body and out must be C-contiguous"));
            }
            if ob.readonly() {
                return Err(PyValueError::new_err("out must be writable"));
            }
            let blen = bb.len_bytes();
            let olen = ob.len_bytes();
            if rows == 0 {
                return Err(PyValueError::new_err("rows must be >= 1"));
            }
            let body_len = u32::try_from(blen).map_err(|_| PyValueError::new_err("body too large"))?;
            // `src` and `dst` below must not alias (a `&[u8]` and a `&mut [u8]` over shared bytes is UB).
            let (bp, op) = (bb.buf_ptr() as usize, ob.buf_ptr() as usize);
            if blen > 0 && olen > 0 && bp < op.saturating_add(olen) && op < bp.saturating_add(blen) {
                return Err(PyValueError::new_err("body and out must not overlap"));
            }
            // SAFETY: both buffers are C-contiguous exports held (`bb`, `ob`) until this function returns, so
            // the memory stays valid and the exporter cannot resize it (bytearray / numpy refuse while
            // exported); `out` is writable. Python code may still touch `out` concurrently from another
            // thread, which is the caller's contract (it owns the buffer for the duration of the trip).
            let src: &[u8] = unsafe { std::slice::from_raw_parts(bb.buf_ptr().cast::<u8>(), blen) };
            let dst: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ob.buf_ptr().cast::<u8>(), olen) };
            let h = RequestHeader { ring, linear, rows, deadline_ms, body_len, request_id: rid };
            let res = py.allow_threads(|| {
                let mut ch = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                ch.trip(&h, src, dst, to)
            });
            drop(bb);
            drop(ob);
            res.map_err(to_py)
        }

        /// Close the connection (idempotent).
        fn close(&self, py: Python<'_>) {
            py.allow_threads(|| self.inner.lock().unwrap_or_else(|p| p.into_inner()).close());
        }

        #[getter]
        fn closed(&self) -> bool {
            self.inner.try_lock().map(|c| c.closed()).unwrap_or(false)
        }

        /// Largest rows per trip this connection serves (from the hello ack).
        #[getter]
        fn max_rows(&self) -> u16 {
            self.max_rows
        }

        /// Adaptive reply spin (opt-in): learn each (linear, rows) reply time and busy-poll about that
        /// long, at most `cap_us` (0 = back to the fixed `spin_us`). Waiting only; the bytes are unchanged.
        fn set_spin_auto(&self, py: Python<'_>, cap_us: u64) {
            py.allow_threads(|| {
                self.inner.lock().unwrap_or_else(|p| p.into_inner()).set_spin_auto(Duration::from_micros(cap_us))
            });
        }

        /// Move a service-lane UDS connection to a shared-memory segment the service passes over this
        /// socket (opt-in; same header and body bytes). False: the service does not offer it (the
        /// connection stays on the socket). Raises (and closes the channel) on a protocol or I/O error.
        #[pyo3(signature = (timeout_s = 5.0))]
        fn attach_shm(&self, py: Python<'_>, timeout_s: f64) -> PyResult<bool> {
            let to = secs(timeout_s)?;
            py.allow_threads(|| self.inner.lock().unwrap_or_else(|p| p.into_inner()).attach_shm(to)).map_err(to_py)
        }

        /// `"uds"`, `"shm"`, `"tls"` or `"closed"`.
        #[getter]
        fn transport(&self, py: Python<'_>) -> &'static str {
            py.allow_threads(|| self.inner.lock().unwrap_or_else(|p| p.into_inner()).transport())
        }

        /// Counters of OK trips: `trips`, `spin_hits`, `blocked`, `wait_us` (request sent to reply header),
        /// `trip_us` (whole trip), `queue_us` / `compute_us` (as the service reported them), `spin_us`
        /// (budgets): sums in microseconds.
        fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
            let st = py.allow_threads(|| self.inner.lock().unwrap_or_else(|p| p.into_inner()).stats());
            let d = pyo3::types::PyDict::new(py);
            d.set_item("trips", st.trips)?;
            d.set_item("spin_hits", st.spin_hits)?;
            d.set_item("blocked", st.blocked)?;
            d.set_item("wait_us", st.wait_ns as f64 / 1e3)?;
            d.set_item("trip_us", st.trip_ns as f64 / 1e3)?;
            d.set_item("queue_us", st.queue_us)?;
            d.set_item("compute_us", st.compute_us)?;
            d.set_item("spin_us", st.spin_ns as f64 / 1e3)?;
            Ok(d)
        }
    }

    fn connect_uds_timeout(path: &str, timeout: Duration) -> io::Result<UnixStream> {
        // `UnixStream::connect` blocks at most until the listener's backlog accepts; a local socket
        // answers at once or fails, so the timeout only bounds the hello that follows.
        let s = UnixStream::connect(path)?;
        s.set_read_timeout(Some(timeout))?;
        s.set_write_timeout(Some(timeout))?;
        Ok(s)
    }

    #[cfg(test)]
    mod spin_tests {
        use super::*;

        #[test]
        fn adaptive_spin_skips_replies_beyond_the_cap() {
            let (a, _b) = UnixStream::pair().unwrap();
            let mut ch = Channel::new(Conn::Uds(a), Duration::from_micros(50));
            ch.set_spin_auto(Duration::from_micros(400));
            // unknown key: the cap
            assert_eq!(ch.spin_for(1, 1), Duration::from_micros(400));
            // a fast reply: 1.25 x + 20 us, at least the fixed spin
            ch.learn(1, 1, Duration::from_micros(100));
            assert_eq!(ch.spin_for(1, 1), Duration::from_micros(145));
            ch.est[0].ns = 10_000;
            assert_eq!(ch.spin_for(1, 1), Duration::from_micros(50));
            // expected just inside the cap: spun up to the cap, as before
            ch.est[0].ns = 380_000;
            assert_eq!(ch.spin_for(1, 1), Duration::from_micros(400));
            // expected after the cap: no spin (blocking wait at once)
            ch.est[0].ns = 401_000;
            assert_eq!(ch.spin_for(1, 1), Duration::ZERO);
            // blocked waits keep teaching the estimate: faster replies bring the spin back
            for _ in 0..16 {
                ch.learn(1, 1, Duration::from_micros(150));
            }
            assert!(ch.spin_for(1, 1) > Duration::ZERO);
            // other keys are independent; the fixed spin (no auto) is unchanged
            assert_eq!(ch.spin_for(2, 1), Duration::from_micros(400));
            ch.set_spin_auto(Duration::ZERO);
            assert_eq!(ch.spin_for(1, 1), Duration::from_micros(50));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Python helpers and header functions
// ---------------------------------------------------------------------------------------------

fn ticket_of(obj: &Bound<'_, PyAny>) -> PyResult<zeroize::Zeroizing<[u8; TICKET_LEN]>> {
    let pb = PyBuffer::<u8>::get(obj).map_err(|_| PyTypeError::new_err("ticket must be bytes-like"))?;
    let v = zeroize::Zeroizing::new(pb.to_vec(obj.py())?);
    if v.len() != TICKET_LEN {
        return Err(PyValueError::new_err(format!("ticket must be {TICKET_LEN} bytes")));
    }
    let mut t = zeroize::Zeroizing::new([0u8; TICKET_LEN]);
    t.copy_from_slice(&v);
    Ok(t)
}

fn rid_of(b: &[u8]) -> PyResult<[u8; 16]> {
    b.try_into().map_err(|_| PyValueError::new_err("request_id must be 16 bytes"))
}

/// `pgw.wire.lane.pack_request` (golden-vector tests).
#[pyfunction]
fn lane_pack_request<'py>(
    py: Python<'py>,
    ring: u8,
    linear: u16,
    rows: u16,
    request_id: &[u8],
    deadline_ms: u32,
    body_len: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    if rows == 0 {
        return Err(PyValueError::new_err("rows must be >= 1"));
    }
    let h = RequestHeader { ring, linear, rows, deadline_ms, body_len, request_id: rid_of(request_id)? };
    Ok(PyBytes::new(py, &h.encode()))
}

/// `pgw.wire.lane.unpack_request` as a tuple in `REQUEST_FIELDS` order without magic/version/kind:
/// `(ring, linear, rows, deadline_ms, body_len, request_id)`.
#[pyfunction]
fn lane_unpack_request<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(u8, u16, u16, u32, u32, Bound<'py, PyBytes>)> {
    let h = RequestHeader::decode(data).map_err(PyValueError::new_err)?;
    Ok((h.ring, h.linear, h.rows, h.deadline_ms, h.body_len, PyBytes::new(py, &h.request_id)))
}

/// `pgw.wire.lane.pack_reply`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn lane_pack_reply<'py>(
    py: Python<'py>,
    kind: u8,
    status: u8,
    linear: u16,
    rows: u16,
    compute_us: u32,
    body_len: u32,
    request_id: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    if kind != KIND_HELLO_ACK && kind != KIND_TRIP_REPLY {
        return Err(PyValueError::new_err("reply kind must be HELLO_ACK or TRIP_REPLY"));
    }
    let h = ReplyHeader { kind, status, linear, rows, compute_us, body_len, request_id: rid_of(request_id)? };
    Ok(PyBytes::new(py, &h.encode()))
}

/// `pgw.wire.lane.unpack_reply` as `(kind, status, linear, rows, compute_us, body_len, request_id)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn lane_unpack_reply<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(u8, u8, u16, u16, u32, u32, Bound<'py, PyBytes>)> {
    let h = ReplyHeader::decode(data).map_err(PyValueError::new_err)?;
    Ok((h.kind, h.status, h.linear, h.rows, h.compute_us, h.body_len, PyBytes::new(py, &h.request_id)))
}

/// `pgw.wire.lane.pack_hello`.
#[pyfunction]
fn lane_pack_hello<'py>(py: Python<'py>, ticket: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let t = ticket_of(ticket)?;
    let mut b = encode_hello(&t);
    let out = PyBytes::new(py, &b);
    b.zeroize();
    Ok(out)
}

/// The client status for a service-lane (A1) status code (`pgw.wire.service_lane.map_service_status`).
#[cfg(unix)]
#[pyfunction]
fn lane_map_service_status(code: u16) -> u8 {
    chan::map_service_status(code)
}

/// Register this module's functions in `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(unix)]
    m.add_class::<TripChannel>()?;
    #[cfg(unix)]
    m.add_function(wrap_pyfunction!(lane_map_service_status, m)?)?;
    m.add_function(wrap_pyfunction!(lane_pack_request, m)?)?;
    m.add_function(wrap_pyfunction!(lane_unpack_request, m)?)?;
    m.add_function(wrap_pyfunction!(lane_pack_reply, m)?)?;
    m.add_function(wrap_pyfunction!(lane_unpack_reply, m)?)?;
    m.add_function(wrap_pyfunction!(lane_pack_hello, m)?)?;
    m.add("LANE_VERSION", LANE_VERSION)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::{self, Read, Write};
    use std::os::unix::net::UnixStream;
    use std::time::{Duration, Instant};

    const GOLDEN: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/wire/lane_golden.bin"));

    fn rid(seed: u8) -> [u8; 16] {
        let mut r = [0u8; 16];
        for (i, b) in r.iter_mut().enumerate() {
            *b = seed.wrapping_mul(31).wrapping_add(i as u8 * 7);
        }
        r
    }

    /// The golden file: `"PGWLGOLD" | version u16 | count u16 | count x (tag u8 | len u8 | bytes)`;
    /// tag 1/2/3 = valid hello/request/reply, tag | 0x80 = invalid (must be rejected).
    fn golden_records() -> Vec<(u8, Vec<u8>)> {
        assert_eq!(&GOLDEN[0..8], b"PGWLGOLD");
        assert_eq!(u16le(GOLDEN, 8), LANE_VERSION);
        let n = u16le(GOLDEN, 10) as usize;
        let mut at = 12;
        let mut out = Vec::new();
        for _ in 0..n {
            let tag = GOLDEN[at];
            let len = GOLDEN[at + 1] as usize;
            out.push((tag, GOLDEN[at + 2..at + 2 + len].to_vec()));
            at += 2 + len;
        }
        assert_eq!(at, GOLDEN.len());
        out
    }

    /// The valid records, field by field (the same values as `tests/wire/test_lane_protocol.py`).
    fn expected() -> (Vec<[u8; TICKET_LEN]>, Vec<RequestHeader>, Vec<ReplyHeader>) {
        let mut ticket = [0u8; TICKET_LEN];
        for (i, b) in ticket.iter_mut().enumerate() {
            *b = i as u8;
        }
        let hellos = vec![ticket, [0xA5; TICKET_LEN]];
        let reqs = vec![
            RequestHeader { ring: 0, linear: 0, rows: 1, deadline_ms: 0, body_len: 14336, request_id: rid(1) },
            RequestHeader { ring: 1, linear: 111, rows: 8, deadline_ms: 30000, body_len: 114688, request_id: rid(2) },
            RequestHeader {
                ring: 255,
                linear: 65535,
                rows: 65535,
                deadline_ms: u32::MAX,
                body_len: u32::MAX,
                request_id: [0xFF; 16],
            },
        ];
        let reps = vec![
            ReplyHeader {
                kind: KIND_HELLO_ACK,
                status: 0,
                linear: 0,
                rows: 8,
                compute_us: 0,
                body_len: 0,
                request_id: [0; 16],
            },
            ReplyHeader {
                kind: KIND_HELLO_ACK,
                status: status::BAD_TICKET,
                linear: 0,
                rows: 0,
                compute_us: 0,
                body_len: 0,
                request_id: [0; 16],
            },
            ReplyHeader {
                kind: KIND_TRIP_REPLY,
                status: 0,
                linear: 0,
                rows: 1,
                compute_us: 287,
                body_len: 18432,
                request_id: rid(1),
            },
            ReplyHeader {
                kind: KIND_TRIP_REPLY,
                status: status::BUSY,
                linear: 111,
                rows: 8,
                compute_us: 0,
                body_len: 0,
                request_id: rid(2),
            },
        ];
        (hellos, reqs, reps)
    }

    #[test]
    fn golden_vectors() {
        let (hellos, reqs, reps) = expected();
        let (mut hi, mut qi, mut ri, mut bad) = (0, 0, 0, 0);
        for (tag, b) in golden_records() {
            match tag {
                1 => {
                    assert_eq!(decode_hello(&b).unwrap(), hellos[hi]);
                    assert_eq!(encode_hello(&hellos[hi]).as_slice(), b.as_slice());
                    hi += 1;
                }
                2 => {
                    assert_eq!(RequestHeader::decode(&b).unwrap(), reqs[qi]);
                    assert_eq!(reqs[qi].encode().as_slice(), b.as_slice());
                    qi += 1;
                }
                3 => {
                    assert_eq!(ReplyHeader::decode(&b).unwrap(), reps[ri]);
                    assert_eq!(reps[ri].encode().as_slice(), b.as_slice());
                    ri += 1;
                }
                0x81 => {
                    assert!(decode_hello(&b).is_err());
                    bad += 1;
                }
                0x82 => {
                    assert!(RequestHeader::decode(&b).is_err());
                    bad += 1;
                }
                0x83 => {
                    assert!(ReplyHeader::decode(&b).is_err());
                    bad += 1;
                }
                t => panic!("unknown golden tag {t}"),
            }
        }
        assert_eq!((hi, qi, ri), (hellos.len(), reqs.len(), reps.len()));
        assert!(bad >= 8);
    }

    /// A minimal lane server on one end of a socket pair: hello ack, then answers each trip with
    /// `y = x` repeated `factor` times, or with `refuse` statuses first.
    fn serve(mut s: UnixStream, factor: usize, mut refuse: Vec<u8>) {
        let mut hello = [0u8; HELLO_LEN];
        s.read_exact(&mut hello).unwrap();
        let t = decode_hello(&hello).unwrap();
        let st = if t == [7u8; 32] { status::OK } else { status::BAD_TICKET };
        let ack = ReplyHeader {
            kind: KIND_HELLO_ACK,
            status: st,
            linear: 0,
            rows: 8,
            compute_us: 0,
            body_len: 0,
            request_id: [0; 16],
        };
        s.write_all(&ack.encode()).unwrap();
        if st != status::OK {
            return;
        }
        loop {
            let mut hdr = [0u8; REQUEST_LEN];
            if s.read_exact(&mut hdr).is_err() {
                return;
            }
            let h = RequestHeader::decode(&hdr).unwrap();
            let mut body = vec![0u8; h.body_len as usize];
            s.read_exact(&mut body).unwrap();
            if !refuse.is_empty() {
                let st = refuse.remove(0);
                let r = ReplyHeader {
                    kind: KIND_TRIP_REPLY,
                    status: st,
                    linear: h.linear,
                    rows: h.rows,
                    compute_us: 0,
                    body_len: 0,
                    request_id: h.request_id,
                };
                s.write_all(&r.encode()).unwrap();
                continue;
            }
            let y: Vec<u8> = body.iter().cycle().take(body.len() * factor).copied().collect();
            let r = ReplyHeader {
                kind: KIND_TRIP_REPLY,
                status: 0,
                linear: h.linear,
                rows: h.rows,
                compute_us: 5,
                body_len: y.len() as u32,
                request_id: h.request_id,
            };
            // the client may hang up after the header (wrong-size test): stop then
            if s.write_all(&r.encode()).and_then(|_| s.write_all(&y)).is_err() {
                return;
            }
        }
    }

    #[test]
    fn trips_over_a_socket_pair() {
        for spin_us in [0u64, 200] {
            let (a, b) = UnixStream::pair().unwrap();
            let srv = std::thread::spawn(move || serve(b, 2, vec![status::BUSY]));
            let mut ch =
                Channel::over_uds(a, &[7u8; 32], Duration::from_secs(5), Duration::from_micros(spin_us)).unwrap();
            assert_eq!(ch.max_rows(), 8);
            let body: Vec<u8> = (0..4096u32).map(|i| (i * 13) as u8).collect();
            let mut out = vec![0u8; body.len() * 2];
            let h = RequestHeader {
                ring: 0,
                linear: 3,
                rows: 2,
                deadline_ms: 0,
                body_len: body.len() as u32,
                request_id: rid(9),
            };
            // a refusal keeps the connection and leaves `out` untouched
            assert_eq!(ch.trip(&h, &body, &mut out, Duration::from_secs(5)).unwrap().0, status::BUSY);
            assert!(out.iter().all(|&v| v == 0));
            assert!(!ch.closed());
            for i in 0..20u8 {
                let h = RequestHeader { request_id: rid(i), ..h };
                let (st, us) = ch.trip(&h, &body, &mut out, Duration::from_secs(5)).unwrap();
                assert_eq!((st, us), (0, 5));
                assert_eq!(&out[..body.len()], body.as_slice());
                assert_eq!(&out[body.len()..], body.as_slice());
            }
            // wrong reply size: protocol error, channel closed
            let mut small = vec![0u8; 10];
            assert!(matches!(ch.trip(&h, &body, &mut small, Duration::from_secs(5)), Err(LaneError::Protocol(_))));
            assert!(ch.closed());
            srv.join().unwrap();
        }
    }

    /// The service-lane (A1) framing: open frame, an OK call, a non-closing refusal, then a reply for
    /// another request id (protocol error: the channel closes).
    #[test]
    fn service_frames_over_a_socket_pair() {
        use pgw_lane::proto::{
            Kind as SK, ReplyHeader as SR, RequestHeader as SQ, Status as SS, REQUEST_HEADER_LEN as SQ_LEN,
        };
        let (a, mut b) = UnixStream::pair().unwrap();
        let srv = std::thread::spawn(move || {
            let mut hb = [0u8; SQ_LEN];
            b.read_exact(&mut hb).unwrap();
            let h = SQ::parse(&hb).unwrap();
            assert_eq!((h.kind, h.rows, h.body_len), (SK::Open, 0, 32));
            let mut tk = [0u8; 32];
            b.read_exact(&mut tk).unwrap();
            assert_eq!(tk, [9u8; 32]);
            b.write_all(&SR::refusal(SS::Ok, [0; 16], 0).encode()).unwrap();
            let mut body = [0u8; 8];
            b.read_exact(&mut hb).unwrap();
            let h = SQ::parse(&hb).unwrap();
            assert_eq!((h.kind, h.linear, h.rows, h.ring, h.deadline_ms, h.body_len), (SK::Call, 3, 2, 1, 250, 8));
            b.read_exact(&mut body).unwrap();
            assert_eq!(body, [5u8; 8]);
            let r = SR {
                status: SS::Ok,
                request_id: h.request_id,
                queue_us: 0,
                compute_us: 7,
                retry_after_ms: 0,
                body_len: 4,
            };
            b.write_all(&r.encode()).unwrap();
            b.write_all(&[1, 2, 3, 4]).unwrap();
            b.read_exact(&mut hb).unwrap();
            let h = SQ::parse(&hb).unwrap();
            b.read_exact(&mut body).unwrap();
            b.write_all(&SR::refusal(SS::Fallback, h.request_id, 0).encode()).unwrap();
            b.read_exact(&mut hb).unwrap();
            b.read_exact(&mut body).unwrap();
            b.write_all(&SR::refusal(SS::Fallback, [0xEE; 16], 0).encode()).unwrap();
        });
        let mut ch =
            Channel::over_uds_service(a, &[9u8; 32], Duration::from_secs(5), Duration::from_micros(20), 4).unwrap();
        assert_eq!(ch.max_rows(), 4);
        let h = RequestHeader { ring: 1, linear: 3, rows: 2, deadline_ms: 250, body_len: 8, request_id: rid(1) };
        let mut out = [0u8; 4];
        assert_eq!(ch.trip(&h, &[5u8; 8], &mut out, Duration::from_secs(5)).unwrap(), (status::OK, 7));
        assert_eq!(out, [1, 2, 3, 4]);
        let h2 = RequestHeader { request_id: rid(2), ..h };
        assert_eq!(ch.trip(&h2, &[5u8; 8], &mut out, Duration::from_secs(5)).unwrap().0, status::BUSY);
        assert!(!ch.closed());
        let h3 = RequestHeader { request_id: rid(3), ..h };
        assert!(matches!(ch.trip(&h3, &[5u8; 8], &mut out, Duration::from_secs(5)), Err(LaneError::Protocol(_))));
        assert!(ch.closed());
        srv.join().unwrap();
        for (code, want) in [(0u16, status::OK), (4, status::DEADLINE), (5, status::BUSY), (19, status::BAD_TICKET)] {
            assert_eq!(map_service_status(code), want);
        }
        for code in [16u16, 17, 18, 19, 20, 21, 22, 23, 99] {
            assert!(status::closing(map_service_status(code)), "{code}");
        }
    }

    #[test]
    fn bad_ticket_is_refused() {
        let (a, b) = UnixStream::pair().unwrap();
        let srv = std::thread::spawn(move || serve(b, 1, vec![]));
        let r = Channel::over_uds(a, &[1u8; 32], Duration::from_secs(5), Duration::ZERO);
        assert!(matches!(r, Err(LaneError::Refused(status::BAD_TICKET))));
        srv.join().unwrap();
    }

    #[test]
    fn timeout_closes_the_channel() {
        let (a, b) = UnixStream::pair().unwrap();
        let srv = std::thread::spawn(move || {
            let mut b = b;
            let mut hello = [0u8; HELLO_LEN];
            b.read_exact(&mut hello).unwrap();
            let ack = ReplyHeader {
                kind: KIND_HELLO_ACK,
                status: 0,
                linear: 0,
                rows: 1,
                compute_us: 0,
                body_len: 0,
                request_id: [0; 16],
            };
            b.write_all(&ack.encode()).unwrap();
            let mut sink = Vec::new();
            let _ = b.read_to_end(&mut sink); // never answers
        });
        let mut ch = Channel::over_uds(a, &[7u8; 32], Duration::from_secs(5), Duration::from_micros(100)).unwrap();
        let body = [1u8; 64];
        let mut out = [0u8; 64];
        let h = RequestHeader { ring: 0, linear: 0, rows: 1, deadline_ms: 0, body_len: 64, request_id: rid(3) };
        let t0 = Instant::now();
        let r = ch.trip(&h, &body, &mut out, Duration::from_millis(80));
        assert!(matches!(r, Err(LaneError::Io(ref e)) if e.kind() == io::ErrorKind::TimedOut), "{r:?}");
        assert!(t0.elapsed() < Duration::from_secs(2));
        assert!(ch.closed());
        srv.join().unwrap();
    }

    #[test]
    fn header_rejects() {
        let h = RequestHeader { ring: 0, linear: 0, rows: 1, deadline_ms: 0, body_len: 4, request_id: [1; 16] };
        let mut b = h.encode();
        b[6] = KIND_HELLO_ACK;
        assert!(RequestHeader::decode(&b).is_err());
        let mut b = h.encode();
        b[10] = 0;
        assert!(RequestHeader::decode(&b).is_err());
        let r = ReplyHeader {
            kind: KIND_TRIP_REPLY,
            status: status::BUSY,
            linear: 0,
            rows: 1,
            compute_us: 0,
            body_len: 4,
            request_id: [1; 16],
        };
        assert!(ReplyHeader::decode(&r.encode()).is_err());
    }
}
