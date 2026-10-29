"""Serve a vt HTTP server's handler as an ASGI app, in-process: no socket between the platform's proxy and
the handler. On Modal, the proxy in front of a web_server reads each reply back over a local TCP socket;
live, it sometimes failed to read a reply the server had fully sent ("Response payload is not
completed"), and the client then waited about 270 s for nothing. Here the handler (unchanged: batching,
streaming, heartbeats, limits) runs in a worker thread with the request in memory, and what it writes is
parsed and handed to ASGI as it comes.

    srv = make_server(bot, port=0, ...)        # vt.verichat_server: built, never served
    app = asgi_from(srv)                       # an ASGI app (uvicorn, Modal's asgi_app)
"""
import asyncio
import io
import traceback
from concurrent.futures import ThreadPoolExecutor

DROP = {b"transfer-encoding", b"content-length", b"expect", b"connection"}


class _Writer:
    """The handler's wfile: every write goes to the event loop's queue, in order."""

    def __init__(self, loop, queue):
        self.loop, self.queue = loop, queue

    def write(self, b):
        if b:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, bytes(b))
        return len(b)

    def flush(self):
        pass


def _request_bytes(scope, body):
    path = scope.get("raw_path") or scope["path"].encode()
    if scope.get("query_string") and b"?" not in path:
        path += b"?" + scope["query_string"]
    head = [scope["method"].encode() + b" " + path + b" HTTP/1.1"]
    head += [k + b": " + v for k, v in scope.get("headers", []) if k.lower() not in DROP]
    head += [b"Content-Length: " + str(len(body)).encode(), b"Connection: close"]
    return b"\r\n".join(head) + b"\r\n\r\n" + body


def _parse_head(raw):
    """b"HTTP/1.0 200 OK\\r\\nName: value..." -> (status, [(name, value)]) with ASGI's lowercase names."""
    lines = raw.split(b"\r\n")
    status = int(lines[0].split(b" ")[1])
    headers = []
    for line in lines[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            if k.strip().lower() not in (b"connection", b"transfer-encoding"):
                headers.append((k.strip().lower(), v.strip()))
    return status, headers


def asgi_from(server, workers=64):
    handler_cls = server.RequestHandlerClass
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vt-asgi")

    def run(raw, client, writer):
        h = handler_cls.__new__(handler_cls)              # the handler, without a socket
        h.request, h.server, h.client_address = None, server, client
        h.rfile, h.wfile = io.BytesIO(raw), writer
        h.close_connection = True
        try:
            h.handle_one_request()
        except Exception:                                 # the handler's own 500s cover model failures
            traceback.print_exc()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        body = b""
        while True:
            m = await receive()
            if m["type"] == "http.disconnect":
                return
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        loop, queue = asyncio.get_running_loop(), asyncio.Queue()
        client = tuple(scope.get("client") or ("127.0.0.1", 0))
        done = loop.run_in_executor(pool, run, _request_bytes(scope, body), client, _Writer(loop, queue))
        done.add_done_callback(lambda _: queue.put_nowait(None))
        buf, started = b"", False
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if started:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
                continue
            buf += chunk
            if b"\r\n\r\n" in buf:                        # the status line and headers are complete
                head, rest = buf.split(b"\r\n\r\n", 1)
                status, headers = _parse_head(head)
                await send({"type": "http.response.start", "status": status, "headers": headers})
                started = True
                if rest:
                    await send({"type": "http.response.body", "body": rest, "more_body": True})
        if not started:                                   # the handler wrote nothing usable
            await send({"type": "http.response.start", "status": 500,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"error": "internal error"}'})
            return
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app
