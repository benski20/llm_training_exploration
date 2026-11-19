"""Recording (and optionally tampering) reverse proxy between the gateway and the compute service.

It sits on the network path exactly where an observer of the service would: everything the
service receives passes through it. Tests use it to

* capture every ``POST /pgw/v1/compute`` body (decoded to ring tensors) and assert that it is
  uniform ring noise, never the plaintext quantized activations;
* simulate a malicious or faulty service by corrupting one element of the next compute response.

Plain HTTP/1.1 and the WebSocket compute stream on loopback (the gateway allows plain http only to
loopback hosts). Stream frames are recorded and can be tampered with like HTTP bodies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
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
    """Reverse proxy ``127.0.0.1:<port>`` -> ``upstream`` for HTTP and the WebSocket compute stream.

    Args:
        upstream: Base URL of the real compute service.
        in_features: ``{linear id: in_features}`` used to decode ``x_hat`` (from the pack).
        ring: Ring of the traffic.
    """

    def __init__(self, upstream: str, in_features: dict[str, int], ring: RingParams):
        import asyncio
        import socket

        import uvicorn

        self.upstream = upstream.rstrip("/")
        self.in_features = in_features
        self.ring = ring
        self.items: list[CapturedItem] = []
        self.raw: list[CapturedRequest] = []
        self.tamper_next = 0  # number of upcoming compute responses to corrupt
        self.tampered = 0
        self.stream_frames = 0  # compute requests that arrived over the WebSocket stream
        self._lock = threading.Lock()
        proxy = self
        client = httpx.AsyncClient(base_url=self.upstream, timeout=300.0, trust_env=False)

        async def http(scope: Any, receive: Any, send: Any) -> None:
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body"):
                    break
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
                       if k.decode("latin-1").lower() not in _HOP}
            path = scope["path"] + (("?" + scope["query_string"].decode()) if scope.get("query_string") else "")
            proxy._record(scope["method"], scope["path"], headers, body)
            up = await client.request(scope["method"], path, content=body, headers=headers)
            out = up.content
            if scope["path"] == COMPUTE_PATH and up.status_code == 200:
                out = proxy._maybe_tamper(out)
            hdrs = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in up.headers.items()
                    if k.lower() not in _HOP and k.lower() != "content-encoding"]
            hdrs.append((b"content-length", str(len(out)).encode()))
            await send({"type": "http.response.start", "status": up.status_code, "headers": hdrs})
            await send({"type": "http.response.body", "body": out})

        async def ws(scope: Any, receive: Any, send: Any) -> None:
            from websockets.asyncio.client import connect

            fwd = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
                   if k.decode("latin-1").lower() in ("authorization", "pgw-protocol")}
            await receive()  # websocket.connect
            proxy._record("WS", scope["path"], fwd, b"")
            try:
                upstream_ws = await connect("ws" + self.upstream[4:] + scope["path"], additional_headers=fwd,
                                            max_size=None, compression=None)
            except Exception:
                await send({"type": "websocket.close", "code": 1011})
                return
            await send({"type": "websocket.accept"})

            async def down() -> None:
                async for frame in upstream_ws:
                    data = frame if isinstance(frame, bytes) else frame.encode()
                    await send({"type": "websocket.send", "bytes": proxy._maybe_tamper_frame(data)})

            task = asyncio.ensure_future(down())
            try:
                while True:
                    msg = await receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    data = msg.get("bytes") or b""
                    proxy._record("POST", COMPUTE_PATH, fwd, data, stream=True)
                    await upstream_ws.send(data)
            finally:
                task.cancel()
                await upstream_ws.close()

        async def app(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                await http(scope, receive, send)
            elif scope["type"] == "websocket":
                await ws(scope, receive, send)
            elif scope["type"] == "lifespan":
                while True:
                    m = await receive()
                    if m["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    else:
                        await send({"type": "lifespan.shutdown.complete"})
                        return

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = int(sock.getsockname()[1])
        self.url = f"http://127.0.0.1:{self.port}"
        cfg = uvicorn.Config(app, log_level="warning", access_log=False, ws="websockets", lifespan="on",
                             ws_max_size=1 << 30)
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=lambda: self._server.run(sockets=[sock]), name="e2e-proxy",
                                        daemon=True)
        self._thread.start()
        import time

        for _ in range(200):
            if self._server.started:
                break
            time.sleep(0.02)

    # ---- recording --------------------------------------------------------------------------------
    def _record(self, method: str, path: str, headers: dict[str, str], body: bytes, *, stream: bool = False) -> None:
        with self._lock:
            self.raw.append(CapturedRequest(method, path, headers, body))
            if stream:
                self.stream_frames += 1
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

    def _maybe_tamper_frame(self, frame: bytes) -> bytes:
        from pgw.wire.codec import decode_any

        try:
            msg = decode_any(frame)
        except Exception:
            return frame
        return self._maybe_tamper(frame) if isinstance(msg, ComputeResponse) else frame

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
        self._server.should_exit = True
        self._thread.join(10)
