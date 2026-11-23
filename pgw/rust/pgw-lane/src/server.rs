//! The lane listener: a Unix domain socket, or TCP with TLS 1.3 ([`crate::tls`]); one thread per
//! connection (a connection carries one call at a time; a gateway opens one per worker), blocking
//! reads with periodic shutdown checks.
//!
//! Per connection: SO_PEERCRED uid check (UDS) or a completed TLS 1.3 handshake (TCP; client
//! certificate verified when presented), then an `open` frame with a single-use ticket (bound to a
//! tenant and a model, and on TCP optionally to the client certificate that asked for it) whose
//! header and ticket must both arrive within `open_timeout` of the accept (one budget for the
//! handshake and the whole frame), then calls. A connection beyond `max_connections` gets the fatal
//! [`Status::Busy`] (UDS; a TCP connection is closed before the handshake) and is closed. Per call:
//! header parse, table and length checks **before** the body is read, the tenant's revocation
//! generation, shared admission, the executor (with the call's deadline), the reply. Refusals below
//! [`crate::proto::FATAL_MIN`] keep the connection; fatal ones close it after the reply. Both
//! transports carry the same frames.

use std::collections::HashMap;
use std::io::{ErrorKind, IoSlice, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::admission::{Admission, Refusal};
use crate::exec::{Call, Done, Executor, ModelEntry, ReplySink};
use crate::proto::{Kind, ReplyHeader, RequestHeader, Status, MAX_ROWS, REQUEST_HEADER_LEN, TICKET_LEN};
use crate::shm::{self, Region};
use crate::stats::Stats;
use crate::sync::{MutexExt, RwLockExt};
use crate::tls::TlsIo;

/// TCP + TLS listener settings (instead of the Unix domain socket).
#[derive(Clone, Debug)]
pub struct TlsListen {
    /// Address to bind (port 0 = any free port; [`LaneServer::local_addr`] tells which).
    pub addr: SocketAddr,
    /// TLS 1.3 server configuration ([`crate::tls::server_config`]).
    pub config: Arc<rustls::ServerConfig>,
}

/// Listener configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Socket path (created; a stale socket file is replaced, a live one is an error). Unused with `tls`.
    pub path: PathBuf,
    /// Listen on TCP with TLS instead of the Unix domain socket.
    pub tls: Option<TlsListen>,
    /// Allowed peer uids (empty = this process's effective uid only).
    pub peer_uids: Vec<u32>,
    /// Largest rows served (1..=8); calls with more rows (up to 8) get `Fallback`.
    pub max_rows: u16,
    /// Cap and default for a call's deadline, milliseconds.
    pub deadline_cap_ms: u32,
    /// Most concurrent connections.
    pub max_connections: usize,
    /// Time to send the whole `open` frame (header and ticket) after the accept.
    pub open_timeout: Duration,
    /// Longest pause inside a frame (header or body) before the connection is dropped.
    pub stall_timeout: Duration,
    /// Poll interval of blocking reads and `accept` (shutdown latency).
    pub poll: Duration,
    /// Before blocking for the next request of a connection, poll the socket without blocking for
    /// this long (zero = block at once). A decode client sends its next trip a few milliseconds after
    /// the reply; a blocking read then pays a scheduler wake-up (slow under gVisor). Costs one core
    /// per busy connection while it spins; changes nothing on the wire.
    pub read_spin: Duration,
    /// Opt-in: OK replies of a Unix-domain-socket connection are written straight from the executor's
    /// output buffer (CUDA graphs: the pinned `h_out`) with a non-blocking `writev` while the device
    /// lock is held, instead of first being copied into a reply buffer; whatever the socket does not take
    /// at once is copied and sent after the lock is released (a slow peer cannot hold the device). Same
    /// bytes.
    pub zero_copy: bool,
    /// Opt-in: answer a connection's `shm` frame by moving its calls to a sealed shared-memory segment
    /// ([`crate::shm`]); without it the frame gets `Fallback` and the connection stays on the socket.
    pub shm: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            path: PathBuf::new(),
            tls: None,
            peer_uids: Vec::new(),
            max_rows: MAX_ROWS,
            deadline_cap_ms: 120_000,
            max_connections: 256,
            open_timeout: Duration::from_secs(10),
            stall_timeout: Duration::from_secs(30),
            poll: Duration::from_millis(100),
            read_spin: Duration::ZERO,
            zero_copy: false,
            shm: false,
        }
    }
}

struct Ticket {
    key: [u8; TICKET_LEN],
    tenant: String,
    model: u32,
    expires: Instant,
    /// SHA-256 of the client certificate the connection must present (TLS only).
    cert: Option<[u8; 32]>,
}

/// One accepted connection.
enum Io {
    Uds(UnixStream),
    Tls(Box<TlsIo>),
}

impl Io {
    fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()> {
        match self {
            Io::Uds(s) => s.set_read_timeout(d),
            Io::Tls(s) => s.set_read_timeout(d),
        }
    }

    /// Header and body in one `writev` (UDS) or one batch of TLS records.
    fn send(&mut self, hdr: &[u8], body: &[u8]) -> std::io::Result<()> {
        match self {
            Io::Uds(s) => write_all_vectored(s, hdr, body),
            Io::Tls(s) => s.send(hdr, body),
        }
    }

    fn close(&mut self) {
        match self {
            Io::Uds(s) => {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
            Io::Tls(s) => s.close(),
        }
    }
}

impl Io {
    /// Spin until request bytes are readable, `spin` passes or shutdown (then the regular blocking read
    /// takes over). UDS: non-blocking peeks of the socket. TLS: non-blocking reads into the TLS
    /// session's own buffer (never a peek: rustls may hold part of a record already).
    fn spin_readable(&mut self, spin: Duration, sh: &Shared) {
        let t0 = Instant::now();
        match self {
            Io::Uds(s) => {
                let fd = s.as_raw_fd();
                let mut b = [0u8; 1];
                let mut i: u32 = 0;
                loop {
                    // SAFETY: a valid fd owned by `s`, a 1-byte local buffer; MSG_PEEK leaves the data queued.
                    let n = unsafe {
                        libc::recv(fd, b.as_mut_ptr() as *mut libc::c_void, 1, libc::MSG_PEEK | libc::MSG_DONTWAIT)
                    };
                    if n >= 0 {
                        return; // data (or EOF): the regular read handles it
                    }
                    let e = std::io::Error::last_os_error().kind();
                    if !matches!(e, ErrorKind::WouldBlock | ErrorKind::Interrupted) {
                        return;
                    }
                    i = i.wrapping_add(1);
                    if i % 32 == 0 && (t0.elapsed() >= spin || sh.shutdown.load(Ordering::Relaxed)) {
                        return;
                    }
                    std::hint::spin_loop();
                }
            }
            Io::Tls(s) => s.spin_for_data(spin, || sh.shutdown.load(Ordering::Relaxed)),
        }
    }
}

impl Read for Io {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Io::Uds(s) => s.read(buf),
            Io::Tls(s) => s.read(buf),
        }
    }
}

/// State shared by the listener, the connections and the owner.
pub struct Shared {
    /// Shared admission (also used by the Python path).
    pub admission: Arc<Admission>,
    /// Counters drained by Python.
    pub stats: Arc<Stats>,
    executor: RwLock<Option<Arc<dyn Executor>>>,
    models: RwLock<HashMap<u32, Arc<ModelEntry>>>,
    tickets: Mutex<Vec<Ticket>>,
    generations: Mutex<HashMap<String, u64>>,
    shutdown: AtomicBool,
    connections: AtomicUsize,
}

impl Shared {
    fn generation(&self, tenant: &str) -> u64 {
        self.generations.plock().get(tenant).copied().unwrap_or(0)
    }
}

/// A running lane listener.
pub struct LaneServer {
    shared: Arc<Shared>,
    cfg: ServerConfig,
    local: Option<SocketAddr>,
    listener: Mutex<Option<JoinHandle<()>>>,
    conns: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() }
}

/// The connecting process's uid (SO_PEERCRED).
pub fn peer_uid(s: &UnixStream) -> std::io::Result<u32> {
    let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: valid fd, out-pointer to a properly sized ucred.
    let rc = unsafe {
        libc::getsockopt(
            s.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut len,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(cred.uid)
}

fn bind(path: &Path) -> std::io::Result<UnixListener> {
    if let Ok(md) = std::fs::symlink_metadata(path) {
        if !md.file_type().is_socket() {
            return Err(std::io::Error::new(
                ErrorKind::AlreadyExists,
                format!("{} exists and is not a socket", path.display()),
            ));
        }
        if UnixStream::connect(path).is_ok() {
            return Err(std::io::Error::new(
                ErrorKind::AddrInUse,
                format!("{} is served by another process", path.display()),
            ));
        }
        std::fs::remove_file(path)?;
    }
    let l = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(l)
}

enum ReadErr {
    Closed,
    Stalled,
    Shutdown,
    Io,
}

/// Fill `buf`. While nothing of the frame has arrived and `idle` is set, waits indefinitely
/// (checking shutdown every poll); once a byte arrived, the frame must complete within `stall`.
fn read_full(s: &mut Io, buf: &mut [u8], sh: &Shared, idle: bool, stall: Duration) -> Result<(), ReadErr> {
    read_until(s, buf, sh, idle, stall, None)
}

/// [`read_full`] that also fails once `by` passes, whatever has arrived (the pre-auth budget).
fn read_until(
    s: &mut Io,
    buf: &mut [u8],
    sh: &Shared,
    idle: bool,
    stall: Duration,
    by: Option<Instant>,
) -> Result<(), ReadErr> {
    let mut got = 0;
    let mut since = Instant::now();
    while got < buf.len() {
        if by.is_some_and(|b| Instant::now() >= b) {
            return Err(ReadErr::Stalled);
        }
        match s.read(&mut buf[got..]) {
            Ok(0) => return Err(ReadErr::Closed),
            Ok(n) => {
                if got == 0 {
                    since = Instant::now();
                }
                got += n;
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => {
                if sh.shutdown.load(Ordering::SeqCst) {
                    return Err(ReadErr::Shutdown);
                }
                if (got > 0 || !idle) && since.elapsed() >= stall {
                    return Err(ReadErr::Stalled);
                }
            }
            Err(_) => return Err(ReadErr::Io),
        }
    }
    Ok(())
}

fn write_reply(s: &mut Io, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
    s.send(&h.encode(), body)
}

fn write_all_vectored(s: &mut UnixStream, hb: &[u8], body: &[u8]) -> std::io::Result<()> {
    let mut bufs = [IoSlice::new(hb), IoSlice::new(body)];
    let mut slices: &mut [IoSlice<'_>] = &mut bufs;
    while !slices.is_empty() && slices.iter().any(|b| !b.is_empty()) {
        match s.write_vectored(slices) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut slices, n),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn refuse(s: &mut Io, status: Status, id: [u8; 16], retry_after_ms: u32) -> std::io::Result<()> {
    write_reply(s, &ReplyHeader::refusal(status, id, retry_after_ms), &[])
}

/// Constant-time equality of two tickets (no early exit on the first differing byte).
fn ticket_eq(a: &[u8; TICKET_LEN], b: &[u8; TICKET_LEN]) -> bool {
    let mut d = 0u8;
    for i in 0..TICKET_LEN {
        d |= a[i] ^ b[i];
    }
    // keep the accumulated difference opaque so the loop is not turned into an early-exit compare
    std::hint::black_box(d) == 0
}

/// Take the live ticket equal to `key` out of `tickets`. Every stored ticket is compared in
/// constant time (the scan never stops early), so timing reveals at most whether one matched.
fn take_ticket(tickets: &mut Vec<Ticket>, key: &[u8; TICKET_LEN], now: Instant) -> Option<Ticket> {
    tickets.retain(|t| t.expires > now);
    let mut found = None;
    for (i, t) in tickets.iter().enumerate() {
        if ticket_eq(&t.key, key) {
            found = Some(i);
        }
    }
    found.map(|i| tickets.swap_remove(i))
}

/// Holds one admitted call in flight; exits the shared admission on drop (also on unwind).
struct Admitted<'a> {
    admission: &'a Admission,
    tenant: &'a str,
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        self.admission.exit(self.tenant);
    }
}

/// Counts one open connection; released on drop (also on unwind).
struct ConnSlot(Arc<Shared>);

impl Drop for ConnSlot {
    fn drop(&mut self) {
        self.0.connections.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Conn {
    tenant: String,
    model: Arc<ModelEntry>,
    model_id: u32,
    generation: u64,
}

/// What the transport established about the peer before the `open` frame.
enum Peer {
    /// Unix domain socket: SO_PEERCRED checked against the allowed uids.
    Uds,
    /// TLS: the SHA-256 of the verified client certificate, if one was presented.
    Tls(Option<[u8; 32]>),
}

fn open(s: &mut Io, sh: &Shared, cfg: &ServerConfig, peer: Peer, by: Instant) -> Option<Conn> {
    if let (Peer::Uds, Io::Uds(u)) = (&peer, &*s) {
        let allowed = match peer_uid(u) {
            Ok(uid) => {
                if cfg.peer_uids.is_empty() {
                    uid == effective_uid()
                } else {
                    cfg.peer_uids.contains(&uid)
                }
            }
            Err(_) => false,
        };
        if !allowed {
            let _ = refuse(s, Status::PeerRefused, [0; 16], 0);
            return None;
        }
    }
    // one budget for the whole open frame (and, on TLS, the handshake), counted from the accept: a
    // slow peer cannot hold a connection slot unauthenticated for longer than `open_timeout`
    let by = Some(by);
    let mut hb = [0u8; REQUEST_HEADER_LEN];
    if read_until(s, &mut hb, sh, false, cfg.open_timeout, by).is_err() {
        return None;
    }
    let h = match RequestHeader::parse(&hb) {
        Ok(h) => h,
        Err((st, id)) => {
            let _ = refuse(s, st, id, 0);
            return None;
        }
    };
    if h.kind != Kind::Open {
        let _ = refuse(s, Status::NotOpen, h.request_id, 0);
        return None;
    }
    let mut ticket = [0u8; TICKET_LEN];
    if read_until(s, &mut ticket, sh, false, cfg.open_timeout, by).is_err() {
        return None;
    }
    let entry = take_ticket(&mut sh.tickets.plock(), &ticket, Instant::now());
    ticket.fill(0);
    let entry = match entry {
        Some(t) => t,
        None => {
            let _ = refuse(s, Status::Unauthenticated, h.request_id, 0);
            return None;
        }
    };
    // A ticket issued to a gateway that presented a client certificate on `lane/open` is good only on
    // a TLS connection presenting that same certificate (the ticket is spent either way).
    if let Some(want) = entry.cert {
        let ok = match peer {
            Peer::Tls(Some(got)) => ticket_eq(&want, &got),
            _ => false,
        };
        if !ok {
            let _ = refuse(s, Status::Unauthenticated, h.request_id, 0);
            return None;
        }
    }
    let model = match sh.models.pread().get(&entry.model) {
        Some(m) => m.clone(),
        None => {
            let _ = refuse(s, Status::Unauthenticated, h.request_id, 0);
            return None;
        }
    };
    let generation = sh.generation(&entry.tenant);
    if refuse(s, Status::Ok, h.request_id, 0).is_err() {
        return None;
    }
    Some(Conn { tenant: entry.tenant, model, model_id: entry.model, generation })
}

enum Listener {
    Uds(UnixListener),
    Tcp(TcpListener),
}

impl Listener {
    /// The next connection, in blocking mode (the listener itself is non-blocking).
    fn accept(&self) -> std::io::Result<Accepted> {
        match self {
            Listener::Uds(l) => {
                let (s, _) = l.accept()?;
                s.set_nonblocking(false)?;
                Ok(Accepted::Uds(s))
            }
            Listener::Tcp(l) => {
                let (s, _) = l.accept()?;
                s.set_nonblocking(false)?;
                Ok(Accepted::Tcp(s))
            }
        }
    }
}

/// A freshly accepted connection, before its transport set-up.
enum Accepted {
    Uds(UnixStream),
    Tcp(TcpStream),
}

fn serve(acc: Accepted, sh: Arc<Shared>, cfg: ServerConfig) {
    let by = Instant::now() + cfg.open_timeout;
    let (mut s, peer) = match acc {
        Accepted::Uds(u) => (Io::Uds(u), Peer::Uds),
        Accepted::Tcp(t) => {
            let Some(tl) = cfg.tls.as_ref() else { return };
            let stop = || sh.shutdown.load(Ordering::SeqCst);
            match TlsIo::accept(t, tl.config.clone(), cfg.poll, cfg.open_timeout, cfg.stall_timeout, stop) {
                Ok(io) => {
                    let d = io.peer_cert_digest();
                    (Io::Tls(Box::new(io)), Peer::Tls(d))
                }
                Err(_) => {
                    sh.stats.connection(false);
                    return;
                }
            }
        }
    };
    let _ = s.set_read_timeout(Some(cfg.poll));
    let conn = open(&mut s, &sh, &cfg, peer, by);
    sh.stats.connection(conn.is_some());
    let Some(conn) = conn else {
        s.close();
        return;
    };
    serve_calls(&mut s, &sh, &cfg, conn);
    s.close();
}

/// Where a connection's call bodies come from and its replies go: the socket, or (opt-in) a
/// shared-memory segment ([`crate::shm`]). Same headers and bodies either way.
trait Frames {
    /// Read the call's body (`buf.len()` bytes, already checked against the tables).
    fn body(&mut self, buf: &mut [u8], sh: &Shared, cfg: &ServerConfig) -> bool;
    /// Write a reply (blocking).
    fn reply(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()>;
    /// True if [`Frames::reply_direct`] may be used (zero-copy replies from the executor's buffer).
    fn direct(&self) -> bool;
    /// Reply from inside the executor (the device lock may be held): never waits for the peer; what
    /// cannot go out at once is kept for [`Frames::finish`].
    fn reply_direct(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()>;
    /// Complete a [`Frames::reply_direct`] (after the executor returned).
    fn finish(&mut self) -> std::io::Result<()>;
}

/// The socket transport (`zero_copy`: replies are written straight from the executor's buffer with a
/// non-blocking `writev`; a remainder the socket does not take at once is copied and sent after).
struct SockFrames<'a> {
    io: &'a mut Io,
    zero_copy: bool,
    pending: Vec<u8>,
}

impl Frames for SockFrames<'_> {
    fn body(&mut self, buf: &mut [u8], sh: &Shared, cfg: &ServerConfig) -> bool {
        read_full(self.io, buf, sh, false, cfg.stall_timeout).is_ok()
    }

    fn reply(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
        write_reply(self.io, h, body)
    }

    fn direct(&self) -> bool {
        self.zero_copy && matches!(self.io, Io::Uds(_))
    }

    fn reply_direct(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
        let hb = h.encode();
        self.pending.clear();
        let Io::Uds(s) = &mut *self.io else {
            self.pending.extend_from_slice(&hb);
            self.pending.extend_from_slice(body);
            return Ok(());
        };
        let total = hb.len() + body.len();
        let mut sent = 0usize;
        while sent < total {
            let (a, b): (&[u8], &[u8]) =
                if sent < hb.len() { (&hb[sent..], body) } else { (&body[sent - hb.len()..], &[]) };
            let iov = [
                libc::iovec { iov_base: a.as_ptr() as *mut libc::c_void, iov_len: a.len() },
                libc::iovec { iov_base: b.as_ptr() as *mut libc::c_void, iov_len: b.len() },
            ];
            // SAFETY: zeroed msghdr pointing at two iovecs over live slices.
            let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
            msg.msg_iov = iov.as_ptr() as *mut libc::iovec;
            msg.msg_iovlen = if b.is_empty() { 1 } else { 2 };
            // SAFETY: valid socket and msghdr; never blocks (MSG_DONTWAIT).
            let n = unsafe { libc::sendmsg(s.as_raw_fd(), &msg, libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL) };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                match e.kind() {
                    ErrorKind::Interrupted => continue,
                    ErrorKind::WouldBlock => break,
                    _ => return Err(e),
                }
            }
            sent += n as usize;
        }
        if sent < hb.len() {
            self.pending.extend_from_slice(&hb[sent..]);
            self.pending.extend_from_slice(body);
        } else if sent < total {
            self.pending.extend_from_slice(&body[sent - hb.len()..]);
        }
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let r = self.io.send(&self.pending, &[]);
        self.pending.clear();
        r
    }
}

/// The shared-memory transport of one connection (after a `shm` frame).
struct ShmFrames<'a> {
    sock: &'a UnixStream,
    reg: Region,
    /// Sequence number of the request being served (echoed in `rep_seq`).
    seq: u64,
}

impl ShmFrames<'_> {
    fn publish(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
        if body.len() > self.reg.rep_cap() {
            return Err(std::io::Error::new(ErrorKind::InvalidData, "reply exceeds the shm segment"));
        }
        self.reg.write(self.reg.rep_body(), body);
        self.reg.write(shm::OFF_REP_HDR, &h.encode());
        self.reg.rep_seq().store(self.seq, Ordering::SeqCst);
        if self.reg.cli_parked().load(Ordering::SeqCst) != 0 {
            shm::ring(self.sock)?;
        }
        Ok(())
    }
}

impl Frames for ShmFrames<'_> {
    fn body(&mut self, buf: &mut [u8], _sh: &Shared, _cfg: &ServerConfig) -> bool {
        if buf.len() > self.reg.req_cap() {
            return false;
        }
        self.reg.read(self.reg.req_body(), buf);
        true
    }

    fn reply(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
        self.publish(h, body)
    }

    fn direct(&self) -> bool {
        true
    }

    fn reply_direct(&mut self, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
        self.publish(h, body)
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Per-connection scratch buffers.
#[derive(Default)]
struct Scratch {
    body: Vec<u8>,
    out: Vec<u8>,
}

/// The zero-copy reply sink handed to the executor: counts the call and replies from its buffer.
struct Direct<'f, 's, F: Frames> {
    f: &'f mut F,
    sh: &'s Shared,
    tenant: &'s str,
    kind: &'s str,
    id: [u8; 16],
    rows: u64,
    want_in: u64,
    out_len: usize,
    received: Instant,
    err: Option<std::io::Error>,
}

impl<F: Frames> ReplySink for Direct<'_, '_, F> {
    fn deliver(&mut self, d: Done, body: &[u8]) {
        if body.len() != self.out_len {
            // cannot happen with a well-formed executor; refuse rather than send a wrong length
            self.sh.stats.request(self.tenant, Status::Internal.name(), self.received.elapsed().as_secs_f64());
            if let Err(e) = self.f.reply_direct(&ReplyHeader::refusal(Status::Internal, self.id, 0), &[]) {
                self.err = Some(e);
            }
            return;
        }
        let rh = ReplyHeader {
            status: Status::Ok,
            request_id: self.id,
            queue_us: d.queue_us,
            compute_us: d.compute_us,
            retry_after_ms: 0,
            body_len: self.out_len as u32,
        };
        // counted before the reply is written: a client that saw the reply sees the counters
        let st = &self.sh.stats;
        st.computed(
            self.tenant,
            self.kind,
            d.backend,
            self.rows,
            self.want_in,
            self.out_len as u64,
            d.compute_us as f64 * 1e-6,
        );
        st.request(self.tenant, Status::Ok.name(), self.received.elapsed().as_secs_f64());
        if let Err(e) = self.f.reply_direct(&rh, body) {
            self.err = Some(e);
        }
    }
}

/// Validate and serve one parsed request header (`h`, read at `received`). False: close the connection.
fn handle_call<F: Frames>(
    f: &mut F,
    sh: &Arc<Shared>,
    cfg: &ServerConfig,
    conn: &Conn,
    h: RequestHeader,
    received: Instant,
    sc: &mut Scratch,
) -> bool {
    let id = h.request_id;
    let fatal = |f: &mut F, st: Status| {
        sh.stats.request(&conn.tenant, st.name(), received.elapsed().as_secs_f64());
        let _ = f.reply(&ReplyHeader::refusal(st, id, 0), &[]);
    };
    if h.kind != Kind::Call {
        fatal(f, Status::NotOpen);
        return false;
    }
    if sh.generation(&conn.tenant) != conn.generation {
        fatal(f, Status::Unauthenticated);
        return false;
    }
    let (Some(lin), Some(ring)) = (conn.model.linears.get(h.linear as usize), conn.model.rings.get(h.ring as usize))
    else {
        fatal(f, Status::UnknownLinear);
        return false;
    };
    if h.rows > MAX_ROWS {
        fatal(f, Status::BadHeader);
        return false;
    }
    let bpe = ring.bytes_per_element as u64;
    let want_in = h.rows as u64 * lin.n as u64 * bpe;
    if h.body_len as u64 != want_in {
        fatal(f, Status::BadLength);
        return false;
    }
    sc.body.resize(want_in as usize, 0);
    if !f.body(&mut sc.body, sh, cfg) {
        return false;
    }
    let out_len = (h.rows as u64 * lin.m as u64 * bpe) as usize;
    let cap = cfg.deadline_cap_ms.max(1);
    let ms = if h.deadline_ms == 0 { cap } else { h.deadline_ms.min(cap) };
    let deadline = received + Duration::from_millis(ms as u64);
    let mut direct_err: Option<std::io::Error> = None;
    let reply = if h.rows > cfg.max_rows {
        Err(Refusal::new(Status::Fallback))
    } else {
        match sh.admission.admit(&conn.tenant, h.rows as u64, true) {
            Err(r) => Err(r),
            Ok(()) => {
                let admitted = Admitted { admission: &sh.admission, tenant: &conn.tenant };
                let call = Call {
                    tenant: &conn.tenant,
                    model: conn.model_id,
                    linear: h.linear,
                    ring: h.ring,
                    rows: h.rows,
                    body: &sc.body,
                    out_len,
                    received,
                    deadline,
                };
                sc.out.clear();
                let ex = sh.executor.pread().clone();
                let r = match ex {
                    Some(ex) if f.direct() => {
                        let mut sink = Direct {
                            f: &mut *f,
                            sh,
                            tenant: &conn.tenant,
                            kind: &lin.kind,
                            id,
                            rows: h.rows as u64,
                            want_in,
                            out_len,
                            received,
                            err: None,
                        };
                        let r = ex.execute_to(&call, &mut sc.out, &mut sink);
                        direct_err = sink.err.take();
                        r
                    }
                    Some(ex) => ex.execute(&call, &mut sc.out).map(|d| (d, false)),
                    None => Err(Refusal::new(Status::Fallback)),
                };
                drop(admitted);
                match r {
                    Ok((_, true)) => {
                        // replied from the executor's buffer; send what the socket did not take at once
                        return direct_err.is_none() && f.finish().is_ok();
                    }
                    Ok((d, false)) if sc.out.len() == out_len => Ok(d),
                    Ok(_) => Err(Refusal::new(Status::Internal)),
                    Err(e) => Err(e),
                }
            }
        }
    };
    match reply {
        Ok(d) => {
            let rh = ReplyHeader {
                status: Status::Ok,
                request_id: id,
                queue_us: d.queue_us,
                compute_us: d.compute_us,
                retry_after_ms: 0,
                body_len: out_len as u32,
            };
            // counted before the reply is written: a client that saw the reply sees the counters
            sh.stats.computed(
                &conn.tenant,
                &lin.kind,
                d.backend,
                h.rows as u64,
                want_in,
                out_len as u64,
                d.compute_us as f64 * 1e-6,
            );
            sh.stats.request(&conn.tenant, Status::Ok.name(), received.elapsed().as_secs_f64());
            f.reply(&rh, &sc.out).is_ok()
        }
        Err(r) => {
            sh.stats.request(&conn.tenant, r.status.name(), received.elapsed().as_secs_f64());
            f.reply(&ReplyHeader::refusal(r.status, id, r.retry_after_ms), &[]).is_ok() && !r.status.is_fatal()
        }
    }
}

/// Body capacities of a connection's shm segment: the largest request and reply of its model's tables
/// at [`MAX_ROWS`] rows.
fn shm_caps(model: &ModelEntry) -> Option<(usize, usize)> {
    let bpe = model.rings.iter().map(|r| r.bytes_per_element as u64).max()?;
    let n = model.linears.iter().map(|l| l.n as u64).max()?;
    let m = model.linears.iter().map(|l| l.m as u64).max()?;
    let req = MAX_ROWS as u64 * n * bpe;
    let rep = MAX_ROWS as u64 * m * bpe;
    (req > 0 && rep > 0 && req <= shm::MAX_CAP as u64 && rep <= shm::MAX_CAP as u64)
        .then_some((req as usize, rep as usize))
}

fn serve_calls(s: &mut Io, sh: &Arc<Shared>, cfg: &ServerConfig, conn: Conn) {
    let mut sc = Scratch::default();
    let mut hb = [0u8; REQUEST_HEADER_LEN];
    loop {
        if !cfg.read_spin.is_zero() {
            s.spin_readable(cfg.read_spin, sh);
        }
        if read_full(s, &mut hb, sh, true, cfg.stall_timeout).is_err() {
            return;
        }
        let received = Instant::now();
        let h = match RequestHeader::parse(&hb) {
            Ok(h) => h,
            Err((st, id)) => {
                sh.stats.request(&conn.tenant, st.name(), 0.0);
                let _ = refuse(s, st, id, 0);
                return;
            }
        };
        if h.kind == Kind::Shm {
            let attached = match (&*s, cfg.shm, shm_caps(&conn.model)) {
                (Io::Uds(u), true, Some((req, rep))) => match Region::create(req, rep) {
                    Ok((fd, reg)) => {
                        let ok = ReplyHeader::refusal(Status::Ok, h.request_id, 0).encode();
                        if shm::send_with_fd(u, &ok, fd.as_raw_fd()).is_err() {
                            return;
                        }
                        drop(fd); // the peer holds its own descriptor; our mapping stays
                        Some(reg)
                    }
                    Err(_) => None,
                },
                _ => None,
            };
            match attached {
                Some(reg) => {
                    if let Io::Uds(u) = &*s {
                        serve_shm(u, reg, sh, cfg, &conn, &mut sc);
                    }
                    return;
                }
                None => {
                    // not offered here (or not possible): the connection stays on the socket
                    if refuse(s, Status::Fallback, h.request_id, 0).is_err() {
                        return;
                    }
                    continue;
                }
            }
        }
        let mut f = SockFrames { io: &mut *s, zero_copy: cfg.zero_copy, pending: Vec::new() };
        if !handle_call(&mut f, sh, cfg, &conn, h, received, &mut sc) {
            return;
        }
    }
}

/// Calls of a connection moved to a shm segment: spin on the request sequence for the read spin
/// (500 us when that is 0), then park on the socket until the doorbell, a close or shutdown.
fn serve_shm(sock: &UnixStream, reg: Region, sh: &Arc<Shared>, cfg: &ServerConfig, conn: &Conn, sc: &mut Scratch) {
    let spin = if cfg.read_spin.is_zero() { Duration::from_micros(500) } else { cfg.read_spin };
    let mut f = ShmFrames { sock, reg, seq: 0 };
    // the segment is fresh (all zero): the first request is seq 1, possibly stored before we get here
    let mut last = 0u64;
    loop {
        let t0 = Instant::now();
        let mut i: u32 = 0;
        let cur = loop {
            let c = f.reg.req_seq().load(Ordering::Acquire);
            if c != last {
                break Some(c);
            }
            i = i.wrapping_add(1);
            if i % 64 == 0 && (t0.elapsed() >= spin || sh.shutdown.load(Ordering::Relaxed)) {
                break None;
            }
            std::hint::spin_loop();
        };
        let Some(cur) = cur else {
            if sh.shutdown.load(Ordering::SeqCst) {
                return;
            }
            if shm::park_or_ready(f.reg.srv_parked(), || f.reg.req_seq().load(Ordering::SeqCst) != last) {
                continue;
            }
            let w = shm::wait_bell(sock);
            f.reg.srv_parked().store(0, Ordering::SeqCst);
            match w {
                Ok(shm::Wake::Rang) | Ok(shm::Wake::Timeout) => continue,
                Ok(shm::Wake::Closed) | Err(_) => return,
            }
        };
        last = cur;
        f.seq = cur;
        let mut hb = [0u8; REQUEST_HEADER_LEN];
        f.reg.read(shm::OFF_REQ_HDR, &mut hb); // one private copy, validated below
        let received = Instant::now();
        let h = match RequestHeader::parse(&hb) {
            Ok(h) => h,
            Err((st, id)) => {
                sh.stats.request(&conn.tenant, st.name(), 0.0);
                let _ = f.reply(&ReplyHeader::refusal(st, id, 0), &[]);
                return;
            }
        };
        if !handle_call(&mut f, sh, cfg, conn, h, received, sc) {
            return;
        }
    }
}

impl LaneServer {
    /// Bind the socket and start accepting.
    pub fn start(
        cfg: ServerConfig,
        admission: Arc<Admission>,
        executor: Option<Arc<dyn Executor>>,
    ) -> std::io::Result<LaneServer> {
        if cfg.max_rows == 0 || cfg.max_rows > MAX_ROWS {
            return Err(std::io::Error::new(ErrorKind::InvalidInput, "max_rows must be in 1..=8"));
        }
        let listener = match &cfg.tls {
            None => {
                let l = bind(&cfg.path)?;
                l.set_nonblocking(true)?;
                Listener::Uds(l)
            }
            Some(t) => {
                let l = TcpListener::bind(t.addr)?;
                l.set_nonblocking(true)?;
                Listener::Tcp(l)
            }
        };
        let local = match &listener {
            Listener::Tcp(l) => Some(l.local_addr()?),
            Listener::Uds(_) => None,
        };
        let shared = Arc::new(Shared {
            admission,
            stats: Arc::new(Stats::default()),
            executor: RwLock::new(executor),
            models: RwLock::new(HashMap::new()),
            tickets: Mutex::new(Vec::new()),
            generations: Mutex::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
            connections: AtomicUsize::new(0),
        });
        let conns: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));
        let (sh, c2, cfg2) = (shared.clone(), conns.clone(), cfg.clone());
        let handle = std::thread::Builder::new().name("pgw-lane-accept".into()).spawn(move || {
            while !sh.shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok(acc) => {
                        if sh.connections.load(Ordering::SeqCst) >= cfg2.max_connections {
                            match acc {
                                Accepted::Uds(s) => {
                                    let mut io = Io::Uds(s);
                                    let _ = refuse(&mut io, Status::Busy, [0; 16], 1000);
                                }
                                // no TLS session yet to refuse in: close before the handshake
                                Accepted::Tcp(t) => {
                                    let _ = t.shutdown(std::net::Shutdown::Both);
                                }
                            }
                            continue;
                        }
                        sh.connections.fetch_add(1, Ordering::SeqCst);
                        let slot = ConnSlot(sh.clone());
                        let (sh3, cfg3) = (sh.clone(), cfg2.clone());
                        // a failed spawn drops the closure, and with it the slot
                        let h = std::thread::Builder::new().name("pgw-lane-conn".into()).spawn(move || {
                            let _slot = slot;
                            // a panic ends this connection only: the guards release its admission
                            // and connection counts during unwinding, locks recover from poisoning
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(acc, sh3, cfg3)));
                        });
                        if let Ok(h) = h {
                            let mut v = c2.plock();
                            v.retain(|j| !j.is_finished());
                            v.push(h);
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                        std::thread::sleep(cfg2.poll.min(Duration::from_millis(20)));
                    }
                    Err(_) => std::thread::sleep(cfg2.poll),
                }
            }
        })?;
        Ok(LaneServer { shared, cfg, local, listener: Mutex::new(Some(handle)), conns })
    }

    /// Shared state (admission, stats).
    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// Socket path.
    pub fn path(&self) -> &Path {
        &self.cfg.path
    }

    /// Configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.cfg
    }

    /// Replace the executor (calls in flight finish on the old one).
    pub fn set_executor(&self, ex: Option<Arc<dyn Executor>>) {
        *self.shared.executor.pwrite() = ex;
    }

    /// Publish (or replace) a model under slot `id`.
    pub fn set_model(&self, id: u32, model: ModelEntry) {
        self.shared.models.pwrite().insert(id, Arc::new(model));
    }

    /// Remove a model: tickets for it stop working; open connections keep their table until they
    /// close (the Python side unloads only after draining).
    pub fn remove_model(&self, id: u32) {
        self.shared.models.pwrite().remove(&id);
    }

    /// Accept `ticket` (single use) for `tenant` and model `model` until `ttl` from now.
    pub fn add_ticket(&self, ticket: [u8; TICKET_LEN], tenant: &str, model: u32, ttl: Duration) {
        self.add_bound_ticket(ticket, tenant, model, ttl, None);
    }

    /// [`Self::add_ticket`], and with `cert` (SHA-256 of a DER client certificate, [`crate::tls::cert_digest`])
    /// the ticket opens only a TLS connection that presents that certificate.
    pub fn add_bound_ticket(
        &self,
        ticket: [u8; TICKET_LEN],
        tenant: &str,
        model: u32,
        ttl: Duration,
        cert: Option<[u8; 32]>,
    ) {
        let now = Instant::now();
        let mut t = self.shared.tickets.plock();
        t.retain(|v| v.expires > now);
        t.push(Ticket { key: ticket, tenant: tenant.to_owned(), model, expires: now + ttl, cert });
    }

    /// The bound TCP address (TLS listener), `None` for the Unix domain socket.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local
    }

    /// Unused tickets.
    pub fn pending_tickets(&self) -> usize {
        let now = Instant::now();
        self.shared.tickets.plock().iter().filter(|v| v.expires > now).count()
    }

    /// Close every connection of `tenant` at its next call and drop its unused tickets.
    pub fn revoke_tenant(&self, tenant: &str) {
        *self.shared.generations.plock().entry(tenant.to_owned()).or_default() += 1;
        self.shared.tickets.plock().retain(|v| v.tenant != tenant);
    }

    /// Open connections.
    pub fn connections(&self) -> usize {
        self.shared.connections.load(Ordering::SeqCst)
    }

    /// Stop accepting, close idle connections (within one poll interval; a call in flight finishes
    /// first) and remove the socket file. Waits up to `timeout` for connection threads.
    pub fn stop(&self, timeout: Duration) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.listener.plock().take() {
            let _ = h.join();
        }
        if self.cfg.tls.is_none() {
            let _ = std::fs::remove_file(&self.cfg.path);
        }
        let t0 = Instant::now();
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.conns.plock());
        for h in handles {
            while !h.is_finished() && t0.elapsed() < timeout {
                std::thread::sleep(Duration::from_millis(5));
            }
            if h.is_finished() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for LaneServer {
    fn drop(&mut self) {
        if !self.shared.shutdown.load(Ordering::SeqCst) {
            self.stop(Duration::from_secs(1));
        }
    }
}
