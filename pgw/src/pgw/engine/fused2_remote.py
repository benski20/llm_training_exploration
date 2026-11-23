"""fused2 over a socket: the provider's pair op as a separate process (opt-in e2e harness, default off).

This is the minimal remote path for one exact end-to-end fused2 run (2 provider round trips per layer):

* :class:`Fused2Service` is the provider's op on **masked ring elements only**: ``single(k1, x_hat) =
  W1 x_hat`` and ``pair(k1, k2, r_hat, x_hat) = (H_hat, Z_hat)`` with ``H_hat = r_hat + hmul1 * (W1
  x_hat)``, ``Z_hat = W2' H_hat`` (mod ``2^K``), the ring products on the provider's device (CUDA via
  :class:`pgw.core.ringmm.PreparedWeight`, any ``K <= 56``). It holds public weights only (``w8``, ``hmul``
  from :meth:`pgw.engine.fused2.Fused2Weights.save` with ``provider_only=True``).
* :func:`serve` runs it on a Unix socket or loopback TCP; :class:`RemoteFused2Provider` is the gateway's
  client (the same ``single`` / ``pair`` interface as the in-process :class:`pgw.engine.fused2.Fused2Provider`),
  so :class:`pgw.engine.fused2.Fused2MaskedBackend` does the masking, unmasking, range checks and both F_p
  Freivalds checks unchanged: an exact, private and integrity-checked trip, the same bits as the plain
  reference.

Frame format (little endian). Request: ``b"F2Q1"``, ``u8 op`` (1 single, 2 pair), ``u8 K``, ``u16 len(k1)``,
``u16 len(k2)``, ``u32 rows``, ``k1``, ``k2`` (utf-8), then ``x_hat`` (``rows * n1`` ring elements of ``K/8``
bytes) and for a pair ``r_hat`` (``rows * m1``). Reply: ``b"F2R1"``, ``u8 status`` (0 ok), ``u32 len``,
``f32 compute_ms``, payload (``y_hat``, or ``H_hat`` then ``Z_hat``; on error utf-8 text).

Not the production wire: no TLS, no authentication, no batching across sessions, one connection at a
time; meant for the co-located measurement (both processes in one container, loopback or a 0600 Unix
socket). What the production path still lacks is listed in ``research/plans/fused2-two-trips-per-layer.md``.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import threading
import time
from typing import Any

import torch

from pgw.core.errors import ConfigError, ProtocolError
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.ringmm import PreparedWeight

__all__ = ["Fused2Service", "RemoteFused2Provider", "serve"]

_REQ = struct.Struct("<4sBBHHI")
_REP = struct.Struct("<4sBIf")
_MAGIC_Q, _MAGIC_R = b"F2Q1", b"F2R1"


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("fused2 peer closed the connection")
        got += k
    return buf


class Fused2Service:
    """The provider's fused2 op (public weights only; inputs are uniform ring elements)."""

    def __init__(self, weights: Any, k: int, device: str = "cpu"):
        self.k = int(k)
        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                           ("cpu" if device == "auto" else device))
        self.device = dev
        self._w: dict[str, PreparedWeight] = {}
        self._hmul: dict[str, torch.Tensor] = {}
        for lin in weights.linears():
            self._w[lin.key] = PreparedWeight(lin.w8.to(dev), device=dev, cache_float=False)
            if lin.hmul is not None:
                self._hmul[lin.key] = lin.hmul.to(dev)
        self.calls = 0

    def dims(self, key: str) -> tuple[int, int]:
        pw = self._w.get(key)
        if pw is None:
            raise ProtocolError(f"fused2: unknown linear {key!r}")
        return pw.out_features, pw.in_features

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self._w[k1].matmul(x_hat.to(self.device), self.k).cpu()

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        mask = (1 << self.k) - 1
        y = self._w[k1].matmul(x_hat.to(self.device), self.k)
        hm = self._hmul.get(k1)
        if hm is not None:
            y = (y * hm) & mask
        h_hat = (r_hat.to(self.device) + y) & mask
        z_hat = self._w[k2].matmul(h_hat, self.k)
        return h_hat.cpu(), z_hat.cpu()


def _handle(svc: Fused2Service, conn: socket.socket) -> None:
    if conn.family != socket.AF_UNIX:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        try:
            hdr = _recv_exact(conn, _REQ.size)
        except ConnectionError:
            return
        magic, op, k, l1, l2, rows = _REQ.unpack(hdr)
        try:
            if magic != _MAGIC_Q or k != svc.k or op not in (1, 2):
                raise ProtocolError("fused2: bad frame")
            names = bytes(_recv_exact(conn, l1 + l2)).decode()
            k1, k2 = names[:l1], names[l1:]
            m1, n1 = svc.dims(k1)
            nb = k // 8
            x_hat = unpack_ring(_recv_exact(conn, rows * n1 * nb), (rows, n1), k)
            t0 = time.perf_counter()
            if op == 1:
                payload = pack_ring(svc.single(k1, x_hat), k)
            else:
                svc.dims(k2)
                r_hat = unpack_ring(_recv_exact(conn, rows * m1 * nb), (rows, m1), k)
                h_hat, z_hat = svc.pair(k1, k2, r_hat, x_hat)
                payload = pack_ring(h_hat, k) + pack_ring(z_hat, k)
            ms = (time.perf_counter() - t0) * 1e3
            conn.sendall(_REP.pack(_MAGIC_R, 0, len(payload), ms) + payload)
        except (ProtocolError, KeyError, ValueError, RuntimeError) as e:
            msg = str(e).encode()[:4096]
            conn.sendall(_REP.pack(_MAGIC_R, 1, len(msg), 0.0) + msg)
            return


def serve(svc: Fused2Service, address: str, *, ready: threading.Event | None = None) -> None:
    """Serve ``svc`` on ``unix:/path`` (mode 0600) or ``tcp:127.0.0.1:port`` (loopback only), one connection
    at a time, until the process ends."""
    if address.startswith("unix:"):
        path = address[5:]
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            srv.bind(path)
        finally:
            os.umask(old)
    elif address.startswith("tcp:"):
        host, port = address[4:].rsplit(":", 1)
        if host not in ("127.0.0.1", "localhost"):
            raise ConfigError("fused2 harness: loopback only (no TLS on this socket)")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, int(port)))
    else:
        raise ConfigError(f"fused2 address must be unix:/path or tcp:127.0.0.1:port, got {address!r}")
    srv.listen(4)
    if ready is not None:
        ready.set()
    while True:
        conn, _ = srv.accept()
        with conn:
            _handle(svc, conn)


class RemoteFused2Provider:
    """Gateway-side client of :func:`serve` (the :class:`pgw.engine.fused2.Fused2Provider` interface)."""

    def __init__(self, address: str, k: int, dims: dict[str, tuple[int, int]], *, connect_timeout_s: float = 600.0):
        self.k = int(k)
        self._dims = dict(dims)
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                if address.startswith("unix:"):
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(address[5:])
                else:
                    host, port = address[4:].rsplit(":", 1)
                    s = socket.create_connection((host, int(port)))
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        self._s = s
        self._lock = threading.Lock()
        #: Provider-reported compute time (ms) and round trips (wall, ms) of every call.
        self.compute_ms: list[float] = []
        self.trip_ms: list[float] = []
        self.bytes_up = 0
        self.bytes_down = 0
        self._rows = 0

    def _call(self, op: int, k1: str, k2: str, blobs: list[bytes]) -> bytes:
        a, b = k1.encode(), k2.encode()
        rows = int(self._rows)
        hdr = _REQ.pack(_MAGIC_Q, op, self.k, len(a), len(b), rows) + a + b
        with self._lock:
            t0 = time.perf_counter()
            self._s.sendall(b"".join([hdr, *blobs]))
            magic, status, n, ms = _REP.unpack(_recv_exact(self._s, _REP.size))
            payload = bytes(_recv_exact(self._s, n))
            self.trip_ms.append((time.perf_counter() - t0) * 1e3)
        if magic != _MAGIC_R:
            raise ProtocolError("fused2: bad reply")
        if status:
            raise ProtocolError("fused2 provider error: " + payload.decode(errors="replace"))
        self.compute_ms.append(ms)
        self.bytes_up += len(hdr) + sum(len(x) for x in blobs)
        self.bytes_down += _REP.size + n
        return payload

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        m1, _ = self._dims[k1]
        self._rows = x_hat.shape[0]
        out = self._call(1, k1, "", [pack_ring(x_hat, self.k)])
        return unpack_ring(out, (self._rows, m1), self.k)

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m1, _ = self._dims[k1]
        m2, _ = self._dims[k2]
        rows = self._rows = x_hat.shape[0]
        out = self._call(2, k1, k2, [pack_ring(x_hat, self.k), pack_ring(r_hat, self.k)])
        nb = self.k // 8
        cut = rows * m1 * nb
        return unpack_ring(out[:cut], (rows, m1), self.k), unpack_ring(out[cut:], (rows, m2), self.k)

    def close(self) -> None:
        self._s.close()


def main(argv: list[str] | None = None) -> None:
    from pgw.engine.fused2 import Fused2Weights

    ap = argparse.ArgumentParser(description="fused2 provider pair op (opt-in e2e harness)")
    ap.add_argument("serve", choices=["serve"])
    ap.add_argument("--weights", required=True, help="Fused2Weights.save(provider_only=True) file")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--address", required=True, help="unix:/path or tcp:127.0.0.1:port")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    w = Fused2Weights.load(args.weights)
    svc = Fused2Service(w, args.k, args.device)
    # warm the device kernels once per linear (first-call setup off the measured path)
    for lin in w.linears():
        svc.single(lin.key, torch.zeros((1, lin.n), dtype=torch.int64))
    print(f"fused2 provider: {len(w.linears())} linears on {svc.device}, K={args.k}, {args.address}", flush=True)
    serve(svc, args.address)


if __name__ == "__main__":
    main()
