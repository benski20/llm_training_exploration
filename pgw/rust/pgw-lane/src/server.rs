//! The lane listener: a Unix domain socket, one thread per connection (a connection carries one
//! call at a time; a gateway opens one per worker), blocking reads with periodic shutdown checks.
//!
//! Per connection: SO_PEERCRED uid check, then an `open` frame with a single-use ticket (bound to a
//! tenant and a model) within `open_timeout`, then calls. Per call: header parse, table and length
//! checks **before** the body is read, the tenant's revocation generation, shared admission, the
//! executor (with the call's deadline), the reply. Refusals below [`crate::proto::FATAL_MIN`] keep the
//! connection; fatal ones close it after the reply.

use std::collections::HashMap;
use std::io::{ErrorKind, IoSlice, Read, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::admission::{Admission, Refusal};
use crate::exec::{Call, Executor, ModelEntry};
use crate::proto::{Kind, ReplyHeader, RequestHeader, Status, MAX_ROWS, REQUEST_HEADER_LEN, TICKET_LEN};
use crate::stats::Stats;

/// Listener configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Socket path (created; a stale socket file is replaced, a live one is an error).
    pub path: PathBuf,
    /// Allowed peer uids (empty = this process's effective uid only).
    pub peer_uids: Vec<u32>,
    /// Largest rows served (1..=8); calls with more rows (up to 8) get `Fallback`.
    pub max_rows: u16,
    /// Cap and default for a call's deadline, milliseconds.
    pub deadline_cap_ms: u32,
    /// Most concurrent connections.
    pub max_connections: usize,
    /// Time to present the ticket after connecting.
    pub open_timeout: Duration,
    /// Longest pause inside a frame (header or body) before the connection is dropped.
    pub stall_timeout: Duration,
    /// Poll interval of blocking reads and `accept` (shutdown latency).
    pub poll: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            path: PathBuf::new(),
            peer_uids: Vec::new(),
            max_rows: MAX_ROWS,
            deadline_cap_ms: 120_000,
            max_connections: 256,
            open_timeout: Duration::from_secs(10),
            stall_timeout: Duration::from_secs(30),
            poll: Duration::from_millis(100),
        }
    }
}

struct Ticket {
    tenant: String,
    model: u32,
    expires: Instant,
}

/// State shared by the listener, the connections and the owner.
pub struct Shared {
    /// Shared admission (also used by the Python path).
    pub admission: Arc<Admission>,
    /// Counters drained by Python.
    pub stats: Arc<Stats>,
    executor: RwLock<Option<Arc<dyn Executor>>>,
    models: RwLock<HashMap<u32, Arc<ModelEntry>>>,
    tickets: Mutex<HashMap<[u8; TICKET_LEN], Ticket>>,
    generations: Mutex<HashMap<String, u64>>,
    shutdown: AtomicBool,
    connections: AtomicUsize,
}

impl Shared {
    fn generation(&self, tenant: &str) -> u64 {
        self.generations.lock().unwrap().get(tenant).copied().unwrap_or(0)
    }
}

/// A running lane listener.
pub struct LaneServer {
    shared: Arc<Shared>,
    cfg: ServerConfig,
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
fn read_full(s: &mut UnixStream, buf: &mut [u8], sh: &Shared, idle: bool, stall: Duration) -> Result<(), ReadErr> {
    let mut got = 0;
    let mut since = Instant::now();
    while got < buf.len() {
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

fn write_reply(s: &mut UnixStream, h: &ReplyHeader, body: &[u8]) -> std::io::Result<()> {
    let hb = h.encode();
    let mut bufs = [IoSlice::new(&hb), IoSlice::new(body)];
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

fn refuse(s: &mut UnixStream, status: Status, id: [u8; 16], retry_after_ms: u32) -> std::io::Result<()> {
    write_reply(s, &ReplyHeader::refusal(status, id, retry_after_ms), &[])
}

struct Conn {
    tenant: String,
    model: Arc<ModelEntry>,
    model_id: u32,
    generation: u64,
}

fn open(s: &mut UnixStream, sh: &Shared, cfg: &ServerConfig) -> Option<Conn> {
    let uid = match peer_uid(s) {
        Ok(u) => u,
        Err(_) => {
            let _ = refuse(s, Status::PeerRefused, [0; 16], 0);
            return None;
        }
    };
    let allowed = if cfg.peer_uids.is_empty() { uid == effective_uid() } else { cfg.peer_uids.contains(&uid) };
    if !allowed {
        let _ = refuse(s, Status::PeerRefused, [0; 16], 0);
        return None;
    }
    let mut hb = [0u8; REQUEST_HEADER_LEN];
    if read_full(s, &mut hb, sh, false, cfg.open_timeout).is_err() {
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
    if read_full(s, &mut ticket, sh, false, cfg.stall_timeout).is_err() {
        return None;
    }
    let entry = sh.tickets.lock().unwrap().remove(&ticket);
    ticket.fill(0);
    let entry = match entry {
        Some(t) if t.expires > Instant::now() => t,
        _ => {
            let _ = refuse(s, Status::Unauthenticated, h.request_id, 0);
            return None;
        }
    };
    let model = match sh.models.read().unwrap().get(&entry.model) {
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

fn serve(mut s: UnixStream, sh: Arc<Shared>, cfg: ServerConfig) {
    let _ = s.set_read_timeout(Some(cfg.poll));
    let conn = open(&mut s, &sh, &cfg);
    sh.stats.connection(conn.is_some());
    let Some(conn) = conn else { return };
    let mut body: Vec<u8> = Vec::new();
    let mut out: Vec<u8> = Vec::new();
    let mut hb = [0u8; REQUEST_HEADER_LEN];
    loop {
        if read_full(&mut s, &mut hb, &sh, true, cfg.stall_timeout).is_err() {
            return;
        }
        let received = Instant::now();
        let h = match RequestHeader::parse(&hb) {
            Ok(h) => h,
            Err((st, id)) => {
                sh.stats.request(&conn.tenant, st.name(), 0.0);
                let _ = refuse(&mut s, st, id, 0);
                return;
            }
        };
        let id = h.request_id;
        let fatal = |s: &mut UnixStream, st: Status| {
            sh.stats.request(&conn.tenant, st.name(), received.elapsed().as_secs_f64());
            let _ = refuse(s, st, id, 0);
        };
        if h.kind != Kind::Call {
            fatal(&mut s, Status::NotOpen);
            return;
        }
        if sh.generation(&conn.tenant) != conn.generation {
            fatal(&mut s, Status::Unauthenticated);
            return;
        }
        let (Some(lin), Some(ring)) =
            (conn.model.linears.get(h.linear as usize), conn.model.rings.get(h.ring as usize))
        else {
            fatal(&mut s, Status::UnknownLinear);
            return;
        };
        if h.rows > MAX_ROWS {
            fatal(&mut s, Status::BadHeader);
            return;
        }
        let bpe = ring.bytes_per_element as u64;
        let want_in = h.rows as u64 * lin.n as u64 * bpe;
        if h.body_len as u64 != want_in {
            fatal(&mut s, Status::BadLength);
            return;
        }
        body.resize(want_in as usize, 0);
        if read_full(&mut s, &mut body, &sh, false, cfg.stall_timeout).is_err() {
            return;
        }
        let out_len = (h.rows as u64 * lin.m as u64 * bpe) as usize;
        let cap = cfg.deadline_cap_ms.max(1);
        let ms = if h.deadline_ms == 0 { cap } else { h.deadline_ms.min(cap) };
        let deadline = received + Duration::from_millis(ms as u64);
        let reply = if h.rows > cfg.max_rows {
            Err(Refusal::new(Status::Fallback))
        } else {
            match sh.admission.admit(&conn.tenant, h.rows as u64, true) {
                Err(r) => Err(r),
                Ok(()) => {
                    let call = Call {
                        tenant: &conn.tenant,
                        model: conn.model_id,
                        linear: h.linear,
                        ring: h.ring,
                        rows: h.rows,
                        body: &body,
                        out_len,
                        received,
                        deadline,
                    };
                    out.clear();
                    let ex = sh.executor.read().unwrap().clone();
                    let r = match ex {
                        Some(ex) => ex.execute(&call, &mut out),
                        None => Err(Refusal::new(Status::Fallback)),
                    };
                    sh.admission.exit(&conn.tenant);
                    match r {
                        Ok(d) if out.len() == out_len => Ok(d),
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
                if write_reply(&mut s, &rh, &out).is_err() {
                    return;
                }
            }
            Err(r) => {
                sh.stats.request(&conn.tenant, r.status.name(), received.elapsed().as_secs_f64());
                if refuse(&mut s, r.status, id, r.retry_after_ms).is_err() || r.status.is_fatal() {
                    return;
                }
            }
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
        let listener = bind(&cfg.path)?;
        listener.set_nonblocking(true)?;
        let shared = Arc::new(Shared {
            admission,
            stats: Arc::new(Stats::default()),
            executor: RwLock::new(executor),
            models: RwLock::new(HashMap::new()),
            tickets: Mutex::new(HashMap::new()),
            generations: Mutex::new(HashMap::new()),
            shutdown: AtomicBool::new(false),
            connections: AtomicUsize::new(0),
        });
        let conns: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));
        let (sh, c2, cfg2) = (shared.clone(), conns.clone(), cfg.clone());
        let handle = std::thread::Builder::new().name("pgw-lane-accept".into()).spawn(move || {
            while !sh.shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        let _ = s.set_nonblocking(false);
                        if sh.connections.load(Ordering::SeqCst) >= cfg2.max_connections {
                            let _ = refuse(&mut s, Status::Unavailable, [0; 16], 1000);
                            continue;
                        }
                        sh.connections.fetch_add(1, Ordering::SeqCst);
                        let (sh3, cfg3) = (sh.clone(), cfg2.clone());
                        let h = std::thread::Builder::new().name("pgw-lane-conn".into()).spawn(move || {
                            serve(s, sh3.clone(), cfg3);
                            sh3.connections.fetch_sub(1, Ordering::SeqCst);
                        });
                        match h {
                            Ok(h) => {
                                let mut v = c2.lock().unwrap();
                                v.retain(|j| !j.is_finished());
                                v.push(h);
                            }
                            Err(_) => {
                                sh.connections.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                        std::thread::sleep(cfg2.poll.min(Duration::from_millis(20)));
                    }
                    Err(_) => std::thread::sleep(cfg2.poll),
                }
            }
        })?;
        Ok(LaneServer { shared, cfg, listener: Mutex::new(Some(handle)), conns })
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
        *self.shared.executor.write().unwrap() = ex;
    }

    /// Publish (or replace) a model under slot `id`.
    pub fn set_model(&self, id: u32, model: ModelEntry) {
        self.shared.models.write().unwrap().insert(id, Arc::new(model));
    }

    /// Remove a model: tickets for it stop working; open connections keep their table until they
    /// close (the Python side unloads only after draining).
    pub fn remove_model(&self, id: u32) {
        self.shared.models.write().unwrap().remove(&id);
    }

    /// Accept `ticket` (single use) for `tenant` and model `model` until `ttl` from now.
    pub fn add_ticket(&self, ticket: [u8; TICKET_LEN], tenant: &str, model: u32, ttl: Duration) {
        let now = Instant::now();
        let mut t = self.shared.tickets.lock().unwrap();
        t.retain(|_, v| v.expires > now);
        t.insert(ticket, Ticket { tenant: tenant.to_owned(), model, expires: now + ttl });
    }

    /// Unused tickets.
    pub fn pending_tickets(&self) -> usize {
        let now = Instant::now();
        self.shared.tickets.lock().unwrap().values().filter(|v| v.expires > now).count()
    }

    /// Close every connection of `tenant` at its next call and drop its unused tickets.
    pub fn revoke_tenant(&self, tenant: &str) {
        *self.shared.generations.lock().unwrap().entry(tenant.to_owned()).or_default() += 1;
        self.shared.tickets.lock().unwrap().retain(|_, v| v.tenant != tenant);
    }

    /// Open connections.
    pub fn connections(&self) -> usize {
        self.shared.connections.load(Ordering::SeqCst)
    }

    /// Stop accepting, close idle connections (within one poll interval; a call in flight finishes
    /// first) and remove the socket file. Waits up to `timeout` for connection threads.
    pub fn stop(&self, timeout: Duration) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.listener.lock().unwrap().take() {
            let _ = h.join();
        }
        let _ = std::fs::remove_file(&self.cfg.path);
        let t0 = Instant::now();
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.conns.lock().unwrap());
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
