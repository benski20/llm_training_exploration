#!/usr/bin/env python3
"""Round-trip time from here to the PGW provider (run it on the client side, e.g. the client test pod).

    python rtt_probe.py https://203.0.113.7:41234 --ca-sha256 <64 hex> [-n 200]
    PGW_API_KEY=pgw_test_... python rtt_probe.py            # URL and CA pin taken from the SDK key

Measures TCP connect, the TLS handshake and ``GET /healthz`` round trips on one kept-alive TLS connection
(verified against the pinned provider CA; nothing secret is sent). Every token of a masked chat makes tens of
sequential round trips, so the p50 here, times the trips per token, is the floor of the per-token latency:
1 ms of RTT at 26 trips per token costs 26 ms per token. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import ssl
import statistics
import sys
import time
from urllib.parse import urlsplit


def _pinned_context(host: str, port: int, ca_sha256: str | None) -> tuple[ssl.SSLContext, str | None]:
    """Fetch the CA from ``/pgw/v1/enroll/ca`` unverified, check its SHA-256 against the pin, trust only it."""
    probe = ssl.create_default_context()
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=10) as s, probe.wrap_socket(s, server_hostname=host) as t:
        t.sendall(f"GET /pgw/v1/enroll/ca HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while chunk := t.recv(65536):
            data += chunk
    pem = data.split(b"\r\n\r\n", 1)[-1].decode("ascii", "replace")
    der = ssl.PEM_cert_to_DER_cert(pem[pem.index("-----BEGIN"):pem.index("-----END CERTIFICATE-----") + 25])
    got = hashlib.sha256(der).hexdigest()
    if ca_sha256 and not hmac.compare_digest(got, ca_sha256.lower()):
        raise SystemExit(f"CA pin mismatch: expected {ca_sha256[:16]}..., got {got[:16]}... (interception?)")
    ctx = ssl.create_default_context(cadata=ssl.DER_cert_to_PEM_cert(der))
    return ctx, got if not ca_sha256 else None


def probe(url: str, ca_sha256: str | None, n: int) -> dict[str, object]:
    u = urlsplit(url)
    host, port = u.hostname or "", u.port or 443
    ctx, unpinned = _pinned_context(host, port, ca_sha256)
    t0 = time.perf_counter()
    sock = socket.create_connection((host, port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t1 = time.perf_counter()
    tls = ctx.wrap_socket(sock, server_hostname=host)
    t2 = time.perf_counter()
    req = f"GET /healthz HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()
    rtts: list[float] = []
    try:
        for _ in range(n):
            s = time.perf_counter()
            tls.sendall(req)
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += tls.recv(4096)
            head, _, body = buf.partition(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            while len(body) < length:
                body += tls.recv(4096)
            rtts.append((time.perf_counter() - s) * 1000)
    finally:
        tls.close()
    rtts.sort()
    q = statistics.quantiles(rtts, n=100) if len(rtts) >= 2 else rtts * 99
    return {"url": url, "n": n, "tcp_connect_ms": round((t1 - t0) * 1000, 3),
            "tls_handshake_ms": round((t2 - t1) * 1000, 3), "rtt_p50_ms": round(statistics.median(rtts), 3),
            "rtt_p90_ms": round(q[89], 3), "rtt_p99_ms": round(q[98], 3), "rtt_min_ms": round(rtts[0], 3),
            "per_token_floor_ms_at_26_trips": round(26 * statistics.median(rtts), 1),
            **({"warning": f"no CA pin given; trusted on first use: {unpinned}"} if unpinned else {})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url", nargs="?", default=os.environ.get("PGW_PROVIDER_URL"))
    ap.add_argument("--ca-sha256", default=os.environ.get("PGW_PROVIDER_CA_SHA256"))
    ap.add_argument("-n", type=int, default=200)
    a = ap.parse_args(argv)
    url, pin = a.url, a.ca_sha256
    key = os.environ.get("PGW_API_KEY")
    if (not url or not pin) and key:
        try:
            from pgw.enroll.sdk_key import SDKKey

            k = SDKKey.parse(key)
            url, pin = url or k.service_url, pin or k.ca_fingerprint_hex
        except Exception as exc:
            raise SystemExit(f"PGW_API_KEY is not usable: {type(exc).__name__}") from None
    if not url:
        raise SystemExit("give the provider URL (or set PGW_API_KEY / PGW_PROVIDER_URL)")
    print(json.dumps(probe(url, pin, a.n), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
