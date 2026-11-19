"""Client demo server: the wire tap (gateway -> service) and the web UI backend, in one process.

Started by ``run.py``; configuration comes from DEMO_* environment variables.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import msgpack
import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from pgw.core import RingParams
from pgw.core.packing import unpack_ring_u64

HERE = Path(__file__).resolve().parent
SERVICE = os.environ["DEMO_SERVICE_URL"]
GATEWAY = os.environ["DEMO_GATEWAY_URL"]
API_KEY = os.environ["DEMO_API_KEY"]
ADMIN = os.environ["DEMO_ADMIN"]
MODEL = os.environ["DEMO_MODEL"]
RING = RingParams.from_name(os.environ.get("DEMO_RING", "K32A12C27"))
NB = RING.k // 8
HALF = 1 << (RING.k - 1)
SYSTEM = os.environ.get("DEMO_SYSTEM", "You are a helpful assistant. Answer in at most three short sentences.")


class Tap:
    def __init__(self) -> None:
        self.calls = 0
        self.bytes_up = 0
        self.bytes_down = 0
        self.tampered = 0
        self.tamper_armed = False
        self.recent: collections.deque[dict[str, Any]] = collections.deque(maxlen=14)
        self.hist = np.zeros(32, dtype=np.int64)  # histogram of everything sent, over the full ring
        self.sample: list[str] = []
        self.sample_linear = ""


TAP = Tap()
_client = httpx.AsyncClient(timeout=120.0, trust_env=False)


def _observe(body: bytes) -> None:
    try:
        msg = msgpack.unpackb(body, raw=False)
        items = msg.get("items") or []
    except Exception:
        return
    for it in items[:1]:
        xh = it.get("x_hat") or b""
        n = min(len(xh) // NB, 8192)
        if n == 0:
            continue
        v = unpack_ring_u64(xh[: n * NB], (n,), RING)
        TAP.hist += np.bincount((v >> np.uint64(RING.k - 5)).astype(np.int64), minlength=32)[:32]
        signed = [int(x) - (1 << RING.k) if int(x) >= HALF else int(x) for x in v[:6]]
        TAP.sample = [f"{s:+,d}" for s in signed]
        TAP.sample_linear = str(it.get("linear", ""))
    TAP.recent.appendleft({
        "t": time.time(),
        "linear": ",".join(str(i.get("linear", "?")) for i in items[:2]) + ("..." if len(items) > 2 else ""),
        "rows": sum(int(i.get("rows", 0)) for i in items),
        "bytes": len(body),
    })


def _tamper(body: bytes) -> bytes:
    """Flip one bit inside the first ``y_hat`` blob, in place (the message length does not change)."""
    msg = msgpack.unpackb(body, raw=False)
    y = msg["items"][0]["y_hat"]
    off = body.find(y)
    if off < 0 or not y:
        raise ValueError("y_hat not found")
    out = bytearray(body)
    out[off + len(y) // 2] ^= 0x01
    return bytes(out)


async def _read_message(r: asyncio.StreamReader) -> tuple[bytes, bytes] | None:
    """One HTTP/1.1 message: (head, body). Supports Content-Length and chunked bodies."""
    try:
        head = await r.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    lower = head.lower()
    if b"transfer-encoding: chunked" in lower:
        body = b""
        raw = b""
        while True:
            line = await r.readuntil(b"\r\n")
            raw += line
            size = int(line.split(b";")[0].strip(), 16)
            chunk = await r.readexactly(size + 2)
            raw += chunk
            body += chunk[:-2]
            if size == 0:
                break
        return head, raw  # forwarded as-is
    n = 0
    for line in lower.split(b"\r\n"):
        if line.startswith(b"content-length:"):
            n = int(line.split(b":", 1)[1])
    return head, (await r.readexactly(n) if n else b"")


async def _relay(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
    """Byte-level relay gateway -> service. Requests and replies pass through unchanged (except the
    one deliberately corrupted reply when the UI arms the tamper switch)."""
    host, port = SERVICE.split("://", 1)[1].rsplit(":", 1)
    sr, sw = await asyncio.open_connection(host, int(port))
    try:
        while True:
            req = await _read_message(cr)
            if req is None:
                break
            head, body = req
            is_compute = head.startswith(b"POST ") and b"/compute " in head.split(b"\r\n", 1)[0]
            if is_compute:
                TAP.calls += 1
                TAP.bytes_up += len(body)
                if TAP.calls % 5 == 1:  # sample; 5 is coprime to the 4 linears per layer
                    _observe(body)
            sw.write(head + body)
            await sw.drain()
            resp = await _read_message(sr)
            if resp is None:
                break
            rhead, rbody = resp
            if is_compute and rhead.startswith(b"HTTP/1.1 200"):
                if TAP.tamper_armed and b"chunked" not in rhead.lower():
                    try:
                        rbody = _tamper(rbody)
                        TAP.tamper_armed = False
                        TAP.tampered += 1
                    except Exception as exc:
                        print(f"tamper failed: {exc}", flush=True)
                TAP.bytes_down += len(rbody)
            cw.write(rhead + rbody)
            await cw.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        sw.close()
        cw.close()


# ---- UI -------------------------------------------------------------------------------------------

async def index(_: Request) -> Response:
    return FileResponse(HERE / "index.html", headers={"cache-control": "no-store"})


async def chat(request: Request) -> Response:
    body = await request.json()
    payload = {"model": MODEL, "stream": True, "temperature": float(body.get("temperature", 0.3)),
               "max_tokens": max(1, min(int(body.get("max_tokens", 90)), 400)),
               "messages": [{"role": "system", "content": SYSTEM}, *body.get("messages", [])[-8:]]}

    async def upstream(q: asyncio.Queue[bytes | str | None]) -> None:
        try:
            async with _client.stream("POST", GATEWAY + "/v1/chat/completions", json=payload,
                                      headers={"Authorization": f"Bearer {API_KEY}"}) as r:
                if r.status_code != 200:
                    err = (await r.aread()).decode(errors="replace")
                    await q.put(f"data: {json.dumps({'demo_error': err[:600], 'status': r.status_code})}\n\n")
                    return
                async for chunk in r.aiter_raw():
                    await q.put(chunk)
        except httpx.HTTPError as exc:
            await q.put(f"data: {json.dumps({'demo_error': str(exc)[:300], 'status': 0})}\n\n")
        finally:
            await q.put(None)

    async def gen() -> Any:
        # Masked prefill can take a while before the first token. Send an SSE comment every few
        # seconds so proxies and port forwarders (GitHub Codespaces) keep the connection open.
        q: asyncio.Queue[bytes | str | None] = asyncio.Queue()
        task = asyncio.create_task(upstream(q))
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=5.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                yield item
        finally:
            task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"cache-control": "no-store"})


async def state(_: Request) -> Response:
    gw: dict[str, Any] = {}
    try:
        r = await _client.get(GATEWAY + "/admin/status", headers={"Authorization": f"Bearer {ADMIN}"})
        m = r.json()["models"][0]
        b = m.get("backend", {})
        gw = {"round_trips": b.get("round_trips"), "freivalds_checked": b.get("freivalds_checked"),
              "integrity_failures": b.get("integrity_failures"), "masks_available": m.get("masks_available"),
              "breaker": m.get("breaker") or m.get("integrity_breaker"), "ready": m.get("ready"),
              "noise": (m.get("noise_monitor") or {}).get("state"), "raw_keys": sorted(m)}
    except Exception as exc:
        gw = {"error": str(exc)[:200]}
    tot = int(TAP.hist.sum())
    return JSONResponse({
        "model": MODEL, "ring": RING.name if hasattr(RING, "name") else os.environ.get("DEMO_RING"),
        "fingerprint": os.environ.get("DEMO_FP", "")[:16],
        "tap": {"calls": TAP.calls, "bytes_up": TAP.bytes_up, "bytes_down": TAP.bytes_down,
                "tampered": TAP.tampered, "tamper_armed": TAP.tamper_armed, "recent": list(TAP.recent),
                "hist": (TAP.hist / tot).tolist() if tot else [], "sample": TAP.sample,
                "sample_linear": TAP.sample_linear, "values_seen": tot},
        "gateway": gw,
    }, headers={"cache-control": "no-store"})


async def tamper(_: Request) -> Response:
    TAP.tamper_armed = True
    return JSONResponse({"armed": True})


async def reset(_: Request) -> Response:
    r = await _client.post(GATEWAY + f"/admin/models/{MODEL}/reset", json={"breaker": True, "noise": True},
                           headers={"Authorization": f"Bearer {ADMIN}"})
    TAP.tamper_armed = False
    return JSONResponse({"status": r.status_code, "body": r.json() if r.content else None})


async def ping(_: Request) -> Response:
    return Response("ok")


ui_app = Starlette(routes=[
    Route("/", index), Route("/api/ping", ping), Route("/api/chat", chat, methods=["POST"]), Route("/api/state", state),
    Route("/api/tamper", tamper, methods=["POST"]), Route("/api/reset", reset, methods=["POST"]),
])


async def main() -> None:
    tap = await asyncio.start_server(_relay, "127.0.0.1", int(os.environ["DEMO_TAP_PORT"]))
    ui = uvicorn.Server(uvicorn.Config(ui_app, host=os.environ.get("DEMO_HOST", "127.0.0.1"),
                                       port=int(os.environ["DEMO_PORT"]), log_level="warning"))
    async with tap:
        await asyncio.gather(tap.serve_forever(), ui.serve())


if __name__ == "__main__":
    asyncio.run(main())
