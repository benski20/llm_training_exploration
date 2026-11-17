"""Recording (and optionally tampering) reverse proxy between the gateway and the compute service.

It sits on the network path exactly where an observer of the service would: everything the
service receives passes through it. Tests use it to

* capture every ``POST /pgw/v1/compute`` body (decoded to ring tensors) and assert that it is
  uniform ring noise, never the plaintext quantized activations;
* simulate a malicious or faulty service by corrupting one element of the next compute response.

Plain HTTP/1.1 on loopback (the gateway allows plain http only to loopback hosts).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import torch

from pgw.core import RingParams, unpack_ring
from pgw.wire.codec import decode_message, encode_message
from pgw.wire.messages import COMPUTE_PATH, ComputeRequest, ComputeResponse, ComputeResult

_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "upgrade",
        "proxy-connection", "te", "trailer"}


@dataclass(frozen=True)
class CapturedItem:
    """One compute item as the service received it."""

    request_id: str
    linear: str
    rows: int
    x_hat: torch.Tensor  # int64 (rows, in_features), values in [0, 2^K)


@dataclass(frozen=True)
class CapturedRequest:
    """Raw view of one HTTP request that reached the service."""

    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class RecordingProxy:
    """Threaded reverse proxy ``127.0.0.1:<port>`` -> ``upstream``.

    Args:
        upstream: Base URL of the real compute service.
        in_features: ``{linear id: in_features}`` used to decode ``x_hat`` (from the pack).
        ring: Ring of the traffic.
    """

    def __init__(self, upstream: str, in_features: dict[str, int], ring: RingParams):
        self.upstream = upstream.rstrip("/")
        self.in_features = in_features
        self.ring = ring
        self.items: list[CapturedItem] = []
        self.raw: list[CapturedRequest] = []
        self.tamper_next = 0  # number of upcoming compute responses to corrupt
        self.tampered = 0
        self._lock = threading.Lock()
        self._client = httpx.Client(base_url=self.upstream, timeout=300.0, trust_env=False)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

            def _forward(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                proxy._record(self.command, self.path, headers, body)
                up = proxy._client.request(self.command, self.path, content=body, headers=headers)
                out = up.content
                if self.path == COMPUTE_PATH and up.status_code == 200:
                    out = proxy._maybe_tamper(out)
                self.send_response(up.status_code)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP and k.lower() != "content-encoding":
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = _forward  # noqa: N815 - stdlib naming
            do_POST = _forward  # noqa: N815

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="e2e-proxy", daemon=True)
        self._thread.start()

    # ---- recording --------------------------------------------------------------------------------
    def _record(self, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        with self._lock:
            self.raw.append(CapturedRequest(method, path, headers, body))
        if method != "POST" or path != COMPUTE_PATH:
            return
        req = decode_message(body, ComputeRequest)
        assert req.ring == self.ring.name
        decoded = [
            CapturedItem(req.request_id, it.linear, it.rows,
                         unpack_ring(it.x_hat, (it.rows, self.in_features[it.linear]), self.ring))
            for it in req.items
        ]
        with self._lock:
            self.items.extend(decoded)

    def _maybe_tamper(self, body: bytes) -> bytes:
        with self._lock:
            if self.tamper_next <= 0:
                return body
            self.tamper_next -= 1
            self.tampered += 1
        resp = decode_message(body, ComputeResponse)
        first = resp.items[0]
        y = bytearray(first.y_hat)
        y[0] ^= 0x01  # one wrong ring element (lowest bit): a minimal, plausible-looking error
        items = (ComputeResult(first.linear, first.rows, bytes(y)), *resp.items[1:])
        return encode_message(ComputeResponse(resp.request_id, items, resp.timing))

    # ---- helpers ----------------------------------------------------------------------------------
    def take(self) -> list[CapturedItem]:
        """Return and clear the captured compute items."""
        with self._lock:
            out, self.items = self.items, []
            return out

    def take_raw(self) -> list[CapturedRequest]:
        """Return and clear the raw captured requests."""
        with self._lock:
            out, self.raw = self.raw, []
            return out

    def close(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()
        self._client.close()
