//! TLS 1.3 for the TCP lane (rustls with the `ring` provider).
//!
//! The TCP listener never speaks plaintext: every accepted connection completes a TLS 1.3 handshake
//! (the service's own certificate and key, the same files as its HTTPS listener) before the lane's
//! `open` frame is read. ALPN `pgw-lane/1` is selected when the client offers it (a client offering
//! only other protocols is refused by rustls). Client certificates are verified against the
//! service's client CA when one is configured (optional, or required with `require_client_cert`),
//! and a ticket issued to a gateway that presented a certificate on `lane/open` is **bound** to the
//! SHA-256 of that certificate: the lane connection must present the same certificate
//! ([`cert_digest`], checked in the server's `open`). Session tickets are off (one connection per
//! worker, opened once), so no resumption state exists.

use std::io::{self, ErrorKind, IoSlice, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// ALPN protocol of the lane (= `pgw.wire.lane.LANE_ALPN`, pgw-fast `lane::ALPN`).
pub const ALPN: &[u8] = b"pgw-lane/1";

/// SHA-256 of a DER certificate (the ticket binding).
pub fn cert_digest(der: &[u8]) -> [u8; 32] {
    let d = ring::digest::digest(&ring::digest::SHA256, der);
    let mut out = [0u8; 32];
    out.copy_from_slice(d.as_ref());
    out
}

fn certs(pem: &[u8], what: &str) -> Result<Vec<CertificateDer<'static>>, String> {
    let mut v = Vec::new();
    for c in CertificateDer::pem_slice_iter(pem) {
        v.push(c.map_err(|e| format!("{what}: {e}"))?);
    }
    if v.is_empty() {
        return Err(format!("{what} holds no certificate"));
    }
    Ok(v)
}

/// Server configuration: TLS 1.3 only, `cert_pem` (chain, leaf first) and `key_pem`, ALPN
/// `pgw-lane/1`; with `client_ca_pem`, client certificates are verified against it (and required when
/// `require_client_cert`).
pub fn server_config(
    cert_pem: &[u8],
    key_pem: &[u8],
    client_ca_pem: Option<&[u8]>,
    require_client_cert: bool,
) -> Result<Arc<rustls::ServerConfig>, String> {
    let chain = certs(cert_pem, "server certificate")?;
    let key = PrivateKeyDer::from_pem_slice(key_pem).map_err(|e| format!("server key: {e}"))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| format!("TLS config: {e}"))?;
    let builder = match client_ca_pem {
        Some(ca) => {
            let mut roots = rustls::RootCertStore::empty();
            for c in certs(ca, "client CA")? {
                roots.add(c).map_err(|e| format!("client CA: {e}"))?;
            }
            let b = rustls::server::WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider);
            let b = if require_client_cert { b } else { b.allow_unauthenticated() };
            builder.with_client_cert_verifier(b.build().map_err(|e| format!("client verifier: {e}"))?)
        }
        None if require_client_cert => return Err("require_client_cert needs a client CA".into()),
        None => builder.with_no_client_auth(),
    };
    let mut cfg = builder.with_single_cert(chain, key).map_err(|e| format!("server certificate/key: {e}"))?;
    cfg.alpn_protocols = vec![ALPN.to_vec()];
    cfg.send_tls13_tickets = 0;
    Ok(Arc::new(cfg))
}

/// One accepted TLS connection.
pub struct TlsIo {
    pub(crate) s: rustls::StreamOwned<rustls::ServerConnection, TcpStream>,
}

impl TlsIo {
    /// Complete the handshake on `tcp` within `budget` (polling `stop` every read timeout).
    /// `stall` bounds every socket write (a peer that stops reading cannot pin the connection thread).
    pub fn accept(
        tcp: TcpStream,
        cfg: Arc<rustls::ServerConfig>,
        poll: Duration,
        budget: Duration,
        stall: Duration,
        stop: impl Fn() -> bool,
    ) -> io::Result<TlsIo> {
        tcp.set_nodelay(true)?;
        tcp.set_read_timeout(Some(poll))?;
        tcp.set_write_timeout(Some(stall))?;
        let mut conn = rustls::ServerConnection::new(cfg).map_err(|e| io::Error::other(e.to_string()))?;
        // the whole reply of a call is queued at once and flushed below (no 64 KiB cap)
        conn.set_buffer_limit(None);
        let mut s = rustls::StreamOwned::new(conn, tcp);
        let by = Instant::now() + budget;
        while s.conn.is_handshaking() {
            match s.conn.complete_io(&mut s.sock) {
                Ok(_) => {}
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => {}
                Err(e) => return Err(e),
            }
            if s.conn.is_handshaking() && (Instant::now() >= by || stop()) {
                return Err(ErrorKind::TimedOut.into());
            }
        }
        while s.conn.wants_write() {
            s.conn.write_tls(&mut s.sock)?;
        }
        Ok(TlsIo { s })
    }

    /// SHA-256 of the verified client certificate (leaf), if the client presented one.
    pub fn peer_cert_digest(&self) -> Option<[u8; 32]> {
        self.s.conn.peer_certificates().and_then(|c| c.first()).map(|d| cert_digest(d.as_ref()))
    }

    /// Header and body as one batch of TLS records, written with vectored socket writes.
    pub fn send(&mut self, hdr: &[u8], body: &[u8]) -> io::Result<()> {
        let mut bufs = [IoSlice::new(hdr), IoSlice::new(body)];
        let mut slices: &mut [IoSlice<'_>] = &mut bufs;
        while !slices.is_empty() && slices.iter().any(|b| !b.is_empty()) {
            let n = self.s.conn.writer().write_vectored(slices)?;
            if n == 0 {
                return Err(ErrorKind::WriteZero.into());
            }
            IoSlice::advance_slices(&mut slices, n);
        }
        while self.s.conn.wants_write() {
            match self.s.conn.write_tls(&mut self.s.sock) {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    pub fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> {
        self.s.sock.set_read_timeout(d)
    }

    pub fn close(&mut self) {
        self.s.conn.send_close_notify();
        while self.s.conn.wants_write() {
            if self.s.conn.write_tls(&mut self.s.sock).is_err() {
                break;
            }
        }
        let _ = self.s.sock.shutdown(std::net::Shutdown::Both);
    }
}

impl Read for TlsIo {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.s.read(buf)
    }
}
