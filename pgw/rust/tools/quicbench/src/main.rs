//! QUIC (quinn) against TCP for the lane's traffic pattern: strictly sequential request/reply trips of given sizes
//! on one long-lived connection, over the emulated WAN of `benchmarks/wan_bench.py quic`.
//!
//! ```text
//! pgw-quicbench serve  --transport quic|tcp --addr IP:PORT [--cert PEM --key PEM] [--cc cubic|newreno|bbr]
//!                      [--init-window BYTES]
//! pgw-quicbench client --transport quic|tcp --addr IP:PORT [--ca PEM --name HOST] [--cc ..] [--init-window B]
//!                      --sizes UP:DOWN,... --iters N [--pass UP:DOWN,... --passes N]
//! ```
//!
//! Framing on the one bidirectional stream (QUIC) or the socket (TCP, `TCP_NODELAY`): an 8-byte header (request
//! bytes, reply bytes, little endian), the request; the server answers with that many reply bytes. QUIC runs TLS
//! 1.3 (rustls/ring, the server certificate verified against `--ca`), TCP runs bare (TLS does not change loss
//! behaviour; the real lane's TLS cost is measured by `wan_bench.py lane`). The congestion controller of QUIC is the
//! application's own on both ends (`--cc`, `--init-window`): the point of the comparison, since a TCP client's
//! controller is its OS's. Prints one JSON object: per size p50 / p90 / p99 ms, per pass p50 ms.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

const ALPN: &[u8] = b"pgw-bench";

struct Args {
    cmd: String,
    transport: String,
    addr: SocketAddr,
    cert: Option<String>,
    key: Option<String>,
    ca: Option<String>,
    name: String,
    cc: String,
    init_window: Option<u64>,
    sizes: Vec<(usize, usize)>,
    pass: Vec<(usize, usize)>,
    iters: usize,
    passes: usize,
}

fn parse_sizes(s: &str) -> Vec<(usize, usize)> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| {
            let (a, b) = x.split_once(':').expect("UP:DOWN");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect()
}

fn args() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let mut a = Args {
        cmd: v.get(1).cloned().unwrap_or_default(),
        transport: "quic".into(),
        addr: "127.0.0.1:9600".parse().unwrap(),
        cert: None,
        key: None,
        ca: None,
        name: "localhost".into(),
        cc: "cubic".into(),
        init_window: None,
        sizes: vec![],
        pass: vec![],
        iters: 50,
        passes: 3,
    };
    let mut i = 2;
    while i + 1 < v.len() {
        let val = v[i + 1].clone();
        match v[i].as_str() {
            "--transport" => a.transport = val,
            "--addr" => a.addr = val.parse().expect("addr"),
            "--cert" => a.cert = Some(val),
            "--key" => a.key = Some(val),
            "--ca" => a.ca = Some(val),
            "--name" => a.name = val,
            "--cc" => a.cc = val,
            "--init-window" => a.init_window = Some(val.parse().unwrap()),
            "--sizes" => a.sizes = parse_sizes(&val),
            "--pass" => a.pass = parse_sizes(&val),
            "--iters" => a.iters = val.parse().unwrap(),
            "--passes" => a.passes = val.parse().unwrap(),
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    a
}

fn transport_config(a: &Args) -> quinn::TransportConfig {
    use quinn::congestion::{BbrConfig, CubicConfig, NewRenoConfig};
    let mut t = quinn::TransportConfig::default();
    t.stream_receive_window((64u32 << 20).into());
    t.receive_window((128u32 << 20).into());
    t.send_window(128 << 20);
    t.max_idle_timeout(Some(std::time::Duration::from_secs(120).try_into().unwrap()));
    t.keep_alive_interval(Some(std::time::Duration::from_secs(10)));
    match a.cc.as_str() {
        "bbr" => {
            let mut c = BbrConfig::default();
            if let Some(w) = a.init_window {
                c.initial_window(w);
            }
            t.congestion_controller_factory(Arc::new(c));
        }
        "newreno" => {
            let mut c = NewRenoConfig::default();
            if let Some(w) = a.init_window {
                c.initial_window(w);
            }
            t.congestion_controller_factory(Arc::new(c));
        }
        _ => {
            let mut c = CubicConfig::default();
            if let Some(w) = a.init_window {
                c.initial_window(w);
            }
            t.congestion_controller_factory(Arc::new(c));
        }
    }
    t
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn pem_certs(path: &str) -> Vec<rustls::pki_types::CertificateDer<'static>> {
    use rustls::pki_types::pem::PemObject;
    rustls::pki_types::CertificateDer::pem_file_iter(path).unwrap().map(|c| c.unwrap()).collect()
}

async fn serve(a: Args) {
    let zero = Arc::new(vec![0u8; 64 << 20]);
    if a.transport == "tcp" {
        let l = tokio::net::TcpListener::bind(a.addr).await.unwrap();
        println!("ready");
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            s.set_nodelay(true).unwrap();
            let zero = zero.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 64 << 20];
                loop {
                    let mut h = [0u8; 8];
                    if s.read_exact(&mut h).await.is_err() {
                        return;
                    }
                    let n = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
                    let m = u32::from_le_bytes(h[4..].try_into().unwrap()) as usize;
                    if s.read_exact(&mut buf[..n]).await.is_err() || s.write_all(&zero[..m]).await.is_err() {
                        return;
                    }
                }
            });
        }
    }
    use rustls::pki_types::pem::PemObject;
    let key = rustls::pki_types::PrivateKeyDer::from_pem_file(a.key.as_ref().unwrap()).unwrap();
    let mut tls = rustls::ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(pem_certs(a.cert.as_ref().unwrap()), key)
        .unwrap();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    let qc = quinn::crypto::rustls::QuicServerConfig::try_from(tls).unwrap();
    let mut sc = quinn::ServerConfig::with_crypto(Arc::new(qc));
    sc.transport_config(Arc::new(transport_config(&a)));
    let ep = quinn::Endpoint::server(sc, a.addr).unwrap();
    println!("ready");
    while let Some(inc) = ep.accept().await {
        let zero = zero.clone();
        tokio::spawn(async move {
            let Ok(conn) = inc.await else { return };
            let Ok((mut send, mut recv)) = conn.accept_bi().await else { return };
            let mut buf = vec![0u8; 64 << 20];
            loop {
                let mut h = [0u8; 8];
                if recv.read_exact(&mut h).await.is_err() {
                    return;
                }
                let n = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
                let m = u32::from_le_bytes(h[4..].try_into().unwrap()) as usize;
                if recv.read_exact(&mut buf[..n]).await.is_err() || send.write_all(&zero[..m]).await.is_err() {
                    return;
                }
            }
        });
    }
}

enum Io {
    Tcp(tokio::net::TcpStream),
    Quic(quinn::SendStream, quinn::RecvStream, quinn::Connection),
}

impl Io {
    async fn trip(&mut self, up: usize, down: usize, body: &[u8], buf: &mut [u8]) {
        let mut h = [0u8; 8];
        h[..4].copy_from_slice(&(up as u32).to_le_bytes());
        h[4..].copy_from_slice(&(down as u32).to_le_bytes());
        match self {
            Io::Tcp(s) => {
                let mut msg = Vec::with_capacity(8 + up);
                msg.extend_from_slice(&h);
                msg.extend_from_slice(&body[..up]);
                s.write_all(&msg).await.unwrap();
                s.read_exact(&mut buf[..down]).await.unwrap();
            }
            Io::Quic(send, recv, _) => {
                let mut msg = Vec::with_capacity(8 + up);
                msg.extend_from_slice(&h);
                msg.extend_from_slice(&body[..up]);
                send.write_all(&msg).await.unwrap();
                recv.read_exact(&mut buf[..down]).await.unwrap();
            }
        }
    }
}

fn pct(xs: &mut [f64], q: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if xs.is_empty() {
        return 0.0;
    }
    xs[((q * xs.len() as f64) as usize).min(xs.len() - 1)]
}

async fn client(a: Args) {
    let t0 = Instant::now();
    let mut io = if a.transport == "tcp" {
        let s = tokio::net::TcpStream::connect(a.addr).await.unwrap();
        s.set_nodelay(true).unwrap();
        Io::Tcp(s)
    } else {
        let mut roots = rustls::RootCertStore::empty();
        for c in pem_certs(a.ca.as_ref().unwrap()) {
            roots.add(c).unwrap();
        }
        let mut tls = rustls::ClientConfig::builder_with_provider(provider())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
        tls.alpn_protocols = vec![ALPN.to_vec()];
        let qc = quinn::crypto::rustls::QuicClientConfig::try_from(tls).unwrap();
        let mut cc = quinn::ClientConfig::new(Arc::new(qc));
        cc.transport_config(Arc::new(transport_config(&a)));
        let mut ep = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap()).unwrap();
        ep.set_default_client_config(cc);
        let conn = ep.connect(a.addr, &a.name).unwrap().await.unwrap();
        let (send, recv) = conn.open_bi().await.unwrap();
        Io::Quic(send, recv, conn)
    };
    let connect_ms = t0.elapsed().as_secs_f64() * 1e3;
    let maxup = a.sizes.iter().chain(a.pass.iter()).map(|x| x.0).max().unwrap_or(8);
    let maxdown = a.sizes.iter().chain(a.pass.iter()).map(|x| x.1).max().unwrap_or(8);
    let body = vec![7u8; maxup];
    let mut buf = vec![0u8; maxdown];
    // warm the connection (window growth) like a running decode
    for _ in 0..20 {
        for &(u, d) in &a.sizes {
            io.trip(u, d, &body, &mut buf).await;
        }
    }
    let mut out = String::from("{");
    out +=
        &format!("\"transport\":\"{}\",\"cc\":\"{}\",\"connect_ms\":{:.3},\"sizes\":[", a.transport, a.cc, connect_ms);
    let mut first = true;
    let mut per: Vec<Vec<f64>> = vec![vec![]; a.sizes.len()];
    for _ in 0..a.iters {
        for (k, &(u, d)) in a.sizes.iter().enumerate() {
            let t = Instant::now();
            io.trip(u, d, &body, &mut buf).await;
            per[k].push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    for (k, &(u, d)) in a.sizes.iter().enumerate() {
        let xs = &mut per[k];
        if !first {
            out += ",";
        }
        first = false;
        out += &format!(
            "{{\"up\":{u},\"down\":{d},\"p50_ms\":{:.3},\"p90_ms\":{:.3},\"p99_ms\":{:.3}}}",
            pct(xs, 0.5),
            pct(xs, 0.9),
            pct(xs, 0.99)
        );
    }
    out += "],\"pass_ms\":[";
    let mut ps = vec![];
    for i in 0..(if a.pass.is_empty() { 0 } else { a.passes }) {
        let t = Instant::now();
        for &(u, d) in &a.pass {
            io.trip(u, d, &body, &mut buf).await;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        ps.push(ms);
        if i > 0 {
            out += ",";
        }
        out += &format!("{ms:.3}");
    }
    out += &format!("],\"pass_p50_ms\":{:.3},\"pass_trips\":{}", pct(&mut ps, 0.5), a.pass.len());
    if let Io::Quic(_, _, c) = &io {
        let st = c.stats();
        out += &format!(
            ",\"quic\":{{\"lost_packets\":{},\"sent_packets\":{},\"cwnd\":{},\"rtt_ms\":{:.3}}}",
            st.path.lost_packets,
            st.path.sent_packets,
            st.path.cwnd,
            st.path.rtt.as_secs_f64() * 1e3
        );
    }
    out += "}";
    println!("{out}");
}

fn main() {
    let a = args();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    match a.cmd.as_str() {
        "serve" => rt.block_on(serve(a)),
        "client" => rt.block_on(client(a)),
        _ => {
            eprintln!("usage: pgw-quicbench serve|client ... (see the source)");
            std::process::exit(2);
        }
    }
}
