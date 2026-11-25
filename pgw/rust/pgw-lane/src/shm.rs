//! Opt-in shared-memory transport of the Unix-domain-socket lane (co-located gateway and service,
//! different users on one host).
//!
//! After a successful `open` on a UDS connection the client may send a [`crate::proto::Kind::Shm`]
//! frame. A service with `ServerConfig::shm` creates an anonymous, sealed memory file
//! (`memfd_create`, sealed against shrinking and growing, so neither side can make the other's
//! mapping fault), maps it, and passes the descriptor back over the **already authenticated**
//! connection with `SCM_RIGHTS` (the OK reply). The segment never has a path: only the two processes
//! of this connection hold it, so its access control is exactly the socket's (SO_PEERCRED uid check
//! and the single-use ticket); a process of either uid could equally read the other's memory with
//! ptrace. From then on every call of the connection goes through the segment, with the same
//! 48-byte request header, the same 40-byte reply header and the same body bytes as on the socket;
//! the socket stays open as the doorbell (one byte when the other side is parked) and for close
//! detection. What the provider sees is unchanged: the same `x_hat` bytes, one call at a time.
//!
//! Layout (offsets in bytes, all integers little-endian / native atomics):
//!
//! | offset | what |
//! |---|---|
//! | 0 | magic `b"PGWSHM1\0"`, then `req_cap` u64, `rep_cap` u64 (written by the service before it sends the fd) |
//! | 64 | `req_seq` (u64, client stores: a request is ready) |
//! | 128 | `rep_seq` (u64, service stores the request's seq: its reply is ready) |
//! | 192 | `srv_parked` (u32, service blocks on the socket: ring the doorbell) |
//! | 256 | `cli_parked` (u32, client blocks on the socket) |
//! | 320 | request header (48 B) |
//! | 384 | reply header (40 B) |
//! | 4096 | request body (`req_cap`) |
//! | 4096 + round_up(req_cap, 4096) | reply body (`rep_cap`) |
//!
//! Waiting is a spin on the sequence word, then a park: set the own `*_parked` flag (SeqCst), re-check
//! the sequence, block in `recv` on the socket. The other side stores the sequence (SeqCst) and then
//! reads the flag; if set it writes one doorbell byte. The SeqCst order guarantees that one of the
//! two sees the other's store, so no wake-up is lost; a stale doorbell only causes a spurious wake.
//!
//! The service treats the segment as untrusted input: it copies the header once and validates the
//! copy exactly as a socket frame, then copies the (length-checked) body once; a client that
//! scribbles over the segment concurrently can only change the bytes of its own call, as it could on
//! the socket.

#[cfg_attr(not(target_os = "linux"), allow(unused_imports))]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Segment magic.
pub const MAGIC: [u8; 8] = *b"PGWSHM1\0";
const OFF_REQ_SEQ: usize = 64;
const OFF_REP_SEQ: usize = 128;
const OFF_SRV_PARKED: usize = 192;
const OFF_CLI_PARKED: usize = 256;
/// Request header area.
pub const OFF_REQ_HDR: usize = 320;
/// Reply header area.
pub const OFF_REP_HDR: usize = 384;
const OFF_REQ_BODY: usize = 4096;
const PAGE: usize = 4096;
/// Largest body area accepted on either side (64 MiB).
pub const MAX_CAP: usize = 64 << 20;

fn round_up(v: usize) -> usize {
    v.div_ceil(PAGE) * PAGE
}

/// Total segment size for the two body capacities.
pub fn segment_len(req_cap: usize, rep_cap: usize) -> usize {
    OFF_REQ_BODY + round_up(req_cap) + round_up(rep_cap)
}

/// A mapped segment (either side). Unmapped on drop.
pub struct Region {
    base: *mut u8,
    len: usize,
    req_cap: usize,
    rep_cap: usize,
}

// SAFETY: the mapping is plain shared memory; every access goes through atomics or raw copies, and
// a `Region` is used by one thread at a time (its connection's).
unsafe impl Send for Region {}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: `base..base+len` is the mapping created in `map`.
        unsafe { libc::munmap(self.base.cast(), self.len) };
    }
}

fn os_err(what: &str) -> std::io::Error {
    let e = std::io::Error::last_os_error();
    std::io::Error::new(e.kind(), format!("{what}: {e}"))
}

fn map(fd: RawFd, len: usize) -> std::io::Result<*mut u8> {
    // SAFETY: a fresh shared mapping of a memory file of at least `len` bytes (checked by callers).
    let p =
        unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
    if p == libc::MAP_FAILED {
        return Err(os_err("mmap"));
    }
    Ok(p.cast())
}

impl Region {
    /// Service side: a new sealed memory file sized for the two capacities, mapped, header written.
    #[cfg(target_os = "linux")]
    pub fn create(req_cap: usize, rep_cap: usize) -> std::io::Result<(OwnedFd, Region)> {
        if req_cap == 0 || rep_cap == 0 || req_cap > MAX_CAP || rep_cap > MAX_CAP {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "shm capacity out of range"));
        }
        let len = segment_len(req_cap, rep_cap);
        // SAFETY: NUL-terminated name; flags as documented.
        let fd = unsafe { libc::memfd_create(c"pgw-lane".as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(os_err("memfd_create"));
        }
        // SAFETY: a descriptor we own from here on.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: valid fd.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(os_err("ftruncate"));
        }
        let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
        // SAFETY: valid fd, F_ADD_SEALS with an int argument.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
            return Err(os_err("F_ADD_SEALS"));
        }
        let base = map(fd.as_raw_fd(), len)?;
        let r = Region { base, len, req_cap, rep_cap };
        // SAFETY: the first 24 bytes of our fresh mapping (zero-filled by the kernel).
        unsafe {
            std::ptr::copy_nonoverlapping(MAGIC.as_ptr(), base, 8);
            std::ptr::copy_nonoverlapping((req_cap as u64).to_le_bytes().as_ptr(), base.add(8), 8);
            std::ptr::copy_nonoverlapping((rep_cap as u64).to_le_bytes().as_ptr(), base.add(16), 8);
        }
        Ok((fd, r))
    }

    /// Client side: map a segment received from the service after checking its seals and size.
    #[cfg(target_os = "linux")]
    pub fn attach(fd: &OwnedFd) -> std::io::Result<Region> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_owned());
        // SAFETY: valid fd.
        let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 {
            return Err(os_err("F_GET_SEALS"));
        }
        if seals & (libc::F_SEAL_SHRINK | libc::F_SEAL_GROW) != (libc::F_SEAL_SHRINK | libc::F_SEAL_GROW) {
            return Err(bad("shm segment is not sealed against resizing"));
        }
        // SAFETY: zeroed stat, valid fd.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
            return Err(os_err("fstat"));
        }
        let size = st.st_size as usize;
        if size < OFF_REQ_BODY + 2 * PAGE || size > segment_len(MAX_CAP, MAX_CAP) {
            return Err(bad("shm segment size out of range"));
        }
        let base = map(fd.as_raw_fd(), size)?;
        let mut r = Region { base, len: size, req_cap: 0, rep_cap: 0 };
        let mut h = [0u8; 24];
        // SAFETY: the mapping has at least 24 bytes.
        unsafe { std::ptr::copy_nonoverlapping(base, h.as_mut_ptr(), 24) };
        let req = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
        let rep = u64::from_le_bytes(h[16..24].try_into().unwrap()) as usize;
        if h[..8] != MAGIC || req == 0 || rep == 0 || req > MAX_CAP || rep > MAX_CAP || segment_len(req, rep) != size {
            return Err(bad("not a lane shm segment"));
        }
        r.req_cap = req;
        r.rep_cap = rep;
        Ok(r)
    }

    /// Non-Linux build: sealed memory files are Linux-only, so the shared-memory lane is unavailable.
    #[cfg(not(target_os = "linux"))]
    pub fn create(_req_cap: usize, _rep_cap: usize) -> std::io::Result<(OwnedFd, Region)> {
        Err(unsupported())
    }

    /// Non-Linux build: see [`Region::create`].
    #[cfg(not(target_os = "linux"))]
    pub fn attach(_fd: &OwnedFd) -> std::io::Result<Region> {
        Err(unsupported())
    }

    /// Request body capacity.
    pub fn req_cap(&self) -> usize {
        self.req_cap
    }

    /// Reply body capacity.
    pub fn rep_cap(&self) -> usize {
        self.rep_cap
    }

    fn u64_at(&self, off: usize) -> &AtomicU64 {
        // SAFETY: 8-aligned offset inside the page-aligned mapping; lives as long as `self`.
        unsafe { &*(self.base.add(off) as *const AtomicU64) }
    }

    fn u32_at(&self, off: usize) -> &AtomicU32 {
        // SAFETY: as above.
        unsafe { &*(self.base.add(off) as *const AtomicU32) }
    }

    /// Request sequence word.
    pub fn req_seq(&self) -> &AtomicU64 {
        self.u64_at(OFF_REQ_SEQ)
    }

    /// Reply sequence word.
    pub fn rep_seq(&self) -> &AtomicU64 {
        self.u64_at(OFF_REP_SEQ)
    }

    /// Service parked flag.
    pub fn srv_parked(&self) -> &AtomicU32 {
        self.u32_at(OFF_SRV_PARKED)
    }

    /// Client parked flag.
    pub fn cli_parked(&self) -> &AtomicU32 {
        self.u32_at(OFF_CLI_PARKED)
    }

    /// Copy `src` into the segment at `off` (bounds checked).
    pub fn write(&self, off: usize, src: &[u8]) {
        assert!(off.checked_add(src.len()).is_some_and(|e| e <= self.len));
        // SAFETY: in bounds; the other process may access the same bytes concurrently, which can only
        // change the data of the call in flight (never memory safety of this process).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.base.add(off), src.len()) };
    }

    /// Copy segment bytes at `off` into `dst` (bounds checked).
    pub fn read(&self, off: usize, dst: &mut [u8]) {
        assert!(off.checked_add(dst.len()).is_some_and(|e| e <= self.len));
        // SAFETY: as in `write`.
        unsafe { std::ptr::copy_nonoverlapping(self.base.add(off), dst.as_mut_ptr(), dst.len()) };
    }

    /// Offset of the request body.
    pub fn req_body(&self) -> usize {
        OFF_REQ_BODY
    }

    /// Offset of the reply body.
    pub fn rep_body(&self) -> usize {
        OFF_REQ_BODY + round_up(self.req_cap)
    }
}

/// Send `data` with `fd` attached (`SCM_RIGHTS`) on a connected Unix stream socket.
pub fn send_with_fd(sock: &UnixStream, data: &[u8], fd: RawFd) -> std::io::Result<()> {
    let mut iov = libc::iovec { iov_base: data.as_ptr() as *mut libc::c_void, iov_len: data.len() };
    // SAFETY: CMSG_SPACE of one int.
    let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    let mut cbuf = vec![0u8; space];
    // SAFETY: zeroed msghdr filled with pointers to locals that outlive the call.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cbuf.as_mut_ptr().cast();
    msg.msg_controllen = space as _;
    // SAFETY: the control buffer holds one cmsg of one fd.
    unsafe {
        let c = libc::CMSG_FIRSTHDR(&msg);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
        std::ptr::copy_nonoverlapping((&fd as *const RawFd).cast::<u8>(), libc::CMSG_DATA(c), 4);
    }
    let mut sent = 0usize;
    // the first sendmsg carries the descriptor; any rest of `data` follows as plain writes
    loop {
        // SAFETY: valid socket and msghdr.
        let n = unsafe { libc::sendmsg(sock.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        sent += n as usize;
        break;
    }
    if sent < data.len() {
        use std::io::Write;
        (&*sock).write_all(&data[sent..])?;
    }
    Ok(())
}

/// Receive exactly `buf.len()` bytes; the first `recvmsg` may carry one descriptor (`SCM_RIGHTS`),
/// which is returned. Any further descriptors are closed. Blocks (honours the socket's timeout).
#[cfg(target_os = "linux")]
pub fn recv_with_fd(sock: &UnixStream, buf: &mut [u8]) -> std::io::Result<Option<OwnedFd>> {
    // SAFETY: CMSG_SPACE of 4 ints (room to see and close extras).
    let space = unsafe { libc::CMSG_SPACE((4 * std::mem::size_of::<RawFd>()) as u32) } as usize;
    let mut cbuf = vec![0u8; space];
    let mut got = 0usize;
    let mut out: Option<OwnedFd> = None;
    while got < buf.len() {
        let mut iov = libc::iovec { iov_base: buf[got..].as_mut_ptr().cast(), iov_len: buf.len() - got };
        // SAFETY: as in `send_with_fd`.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cbuf.as_mut_ptr().cast();
        msg.msg_controllen = space as _;
        // SAFETY: valid socket and msghdr; MSG_CMSG_CLOEXEC marks received fds close-on-exec.
        let n = unsafe { libc::recvmsg(sock.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        // SAFETY: walking the control messages the kernel wrote into `cbuf`.
        unsafe {
            let mut c = libc::CMSG_FIRSTHDR(&msg);
            while !c.is_null() {
                if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                    let data = libc::CMSG_DATA(c);
                    let nfd = ((*c).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / std::mem::size_of::<RawFd>();
                    for i in 0..nfd {
                        let mut fd: RawFd = -1;
                        std::ptr::copy_nonoverlapping(data.add(i * 4), (&mut fd as *mut RawFd).cast::<u8>(), 4);
                        let owned = OwnedFd::from_raw_fd(fd);
                        if out.is_none() {
                            out = Some(owned);
                        } // else dropped (closed)
                    }
                }
                c = libc::CMSG_NXTHDR(&msg, c);
            }
        }
        got += n as usize;
    }
    Ok(out)
}

/// Non-Linux build: descriptor passing is only used by the (Linux-only) shared-memory lane.
#[cfg(not(target_os = "linux"))]
pub fn recv_with_fd(_sock: &UnixStream, _buf: &mut [u8]) -> std::io::Result<Option<OwnedFd>> {
    Err(unsupported())
}

#[cfg(not(target_os = "linux"))]
fn unsupported() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, "the shared-memory lane needs Linux (memfd seals)")
}

/// Ring the doorbell: one byte, never blocking (a full socket buffer already holds a wake-up).
pub fn ring(sock: &UnixStream) -> std::io::Result<()> {
    let b = [1u8];
    loop {
        // SAFETY: valid socket, 1-byte local buffer.
        let n = unsafe { libc::send(sock.as_raw_fd(), b.as_ptr().cast(), 1, libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL) };
        if n >= 0 {
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        match e.kind() {
            std::io::ErrorKind::Interrupted => continue,
            std::io::ErrorKind::WouldBlock => return Ok(()),
            _ => return Err(e),
        }
    }
}

/// Outcome of a blocking doorbell wait.
pub enum Wake {
    /// One or more doorbell bytes (or a spurious wake): re-check the sequence.
    Rang,
    /// The socket's read timeout passed.
    Timeout,
    /// The peer closed the connection.
    Closed,
}

/// Block on the socket for doorbell bytes (drains what is there), up to the socket's read timeout.
pub fn wait_bell(sock: &UnixStream) -> std::io::Result<Wake> {
    let mut b = [0u8; 64];
    loop {
        // SAFETY: valid socket, local buffer.
        let n = unsafe { libc::recv(sock.as_raw_fd(), b.as_mut_ptr().cast(), b.len(), 0) };
        if n > 0 {
            return Ok(Wake::Rang);
        }
        if n == 0 {
            return Ok(Wake::Closed);
        }
        let e = std::io::Error::last_os_error();
        match e.kind() {
            std::io::ErrorKind::Interrupted => continue,
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => return Ok(Wake::Timeout),
            _ => return Err(e),
        }
    }
}

/// Park protocol helper: set `flag`, and return true if `ready()` already holds (then the flag is
/// cleared again and the caller must not block).
pub fn park_or_ready(flag: &AtomicU32, ready: impl Fn() -> bool) -> bool {
    flag.store(1, Ordering::SeqCst);
    if ready() {
        flag.store(0, Ordering::SeqCst);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_attach_and_fd_passing() {
        let (fd, srv) = Region::create(10_000, 3).unwrap();
        let (a, b) = UnixStream::pair().unwrap();
        send_with_fd(&a, b"hello", fd.as_raw_fd()).unwrap();
        let mut buf = [0u8; 5];
        let got = recv_with_fd(&b, &mut buf).unwrap().expect("a descriptor");
        assert_eq!(&buf, b"hello");
        let cli = Region::attach(&got).unwrap();
        assert_eq!((cli.req_cap(), cli.rep_cap()), (10_000, 3));
        assert_eq!(cli.rep_body(), 4096 + 12288);
        cli.write(cli.req_body(), &[9, 8, 7]);
        let mut r = [0u8; 3];
        srv.read(srv.req_body(), &mut r);
        assert_eq!(r, [9, 8, 7]);
        cli.req_seq().store(5, Ordering::SeqCst);
        assert_eq!(srv.req_seq().load(Ordering::SeqCst), 5);
        // sealed: the client cannot shrink (or grow) the segment under the service's mapping
        // SAFETY: valid fd.
        assert_ne!(unsafe { libc::ftruncate(got.as_raw_fd(), 4096) }, 0);
        assert_ne!(unsafe { libc::ftruncate(got.as_raw_fd(), 1 << 30) }, 0);
    }

    #[test]
    fn unsealed_or_foreign_segments_are_refused() {
        // SAFETY: plain memfd without seals.
        let fd = unsafe { libc::memfd_create(c"x".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        unsafe { libc::ftruncate(fd.as_raw_fd(), segment_len(4096, 4096) as libc::off_t) };
        assert!(Region::attach(&fd).is_err());
    }

    #[test]
    fn doorbell() {
        let (a, b) = UnixStream::pair().unwrap();
        b.set_read_timeout(Some(std::time::Duration::from_millis(5))).unwrap();
        assert!(matches!(wait_bell(&b).unwrap(), Wake::Timeout));
        ring(&a).unwrap();
        ring(&a).unwrap();
        assert!(matches!(wait_bell(&b).unwrap(), Wake::Rang));
        drop(a);
        assert!(matches!(wait_bell(&b).unwrap(), Wake::Closed));
    }
}
