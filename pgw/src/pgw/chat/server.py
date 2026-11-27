"""The localhost server of ``pgw chat``: the built web app plus a small JSON / SSE API over the client library.

Security model (the user's own machine; the browser on it is the only intended client):

* **Loopback only.** The socket is bound to ``127.0.0.1``; :class:`ChatServer` refuses any other address.
* **Per-launch token.** 256 random bits drawn at start. Every ``/api/`` call must carry ``Authorization: Bearer
  <token>`` (constant-time compare), so other local users and other web pages cannot call the API. The browser gets
  the token in the URL *fragment* (``/#token=...``), which browsers never send to a server, keeps it in
  ``sessionStorage`` and removes it from the address bar.
* **DNS rebinding.** Every request (static files included) must name this server in ``Host`` (``127.0.0.1:<port>``
  or ``localhost:<port>``). ``/api/`` requests that carry ``Origin`` must come from this origin; state-changing ones
  (``POST``, ``DELETE``) must carry it; ``Sec-Fetch-Site``, when sent, must be ``same-origin`` or ``none``; bodies
  must be ``application/json``. There is no CORS: preflights are refused.
* **Strict CSP** on every response (no inline script or style, nothing from another origin), framing denied,
  ``no-referrer``, ``nosniff``; API replies ``no-store``.
* **No content in logs.** The request log of :mod:`http.server` is off; errors are logged by type only.
* Nothing is sent anywhere by this module: prompts go to the client library only, which masks them before they
  leave the process.

API (JSON; ``/api/chat`` answers with Server-Sent Events, ``data:`` lines of ``{"type": ...}``):
``GET /api/info``, ``GET /api/status?model=``, ``POST /api/warmup {model}``, ``GET /api/conversations``,
``GET|DELETE /api/conversations/<id>``, ``DELETE /api/conversations``,
``POST /api/chat {conversation_id?, model, message | regenerate: true, temperature?, max_tokens?, seed?}``
(events ``start``, ``delta {text}``, ``done {message}``, ``error {error, message}``), ``POST /api/chat/<gen>/stop``.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import queue
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from pgw.chat.backend import Backend, device_profile, recommended_model
from pgw.chat.status import error_view
from pgw.chat.store import ConversationStore, valid_id

__all__ = ["CSP", "DEFAULT_STATIC_DIR", "SECURITY_HEADERS", "ChatServer"]

log = logging.getLogger("pgw.chat")

DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
       "connect-src 'self'; manifest-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CSP,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}

_MEDIA = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
          ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8", ".json": "application/json",
          ".ico": "image/x-icon"}

MAX_BODY = 1 << 20
MAX_MESSAGE_CHARS = 32_000
MAX_TOKENS_LIMIT = 8192
_GEN_RE = re.compile(r"g_[0-9a-f]{16}")
_NOT_BUILT = (b"<!doctype html><meta charset=utf-8><title>PGW Chat</title><p>PGW Chat is not built in this "
              b"installation (pgw/chat/static is missing). Build it with scripts/build-chat.sh.")


class _Generation:
    def __init__(self, gid: str, conversation_id: str):
        self.id = gid
        self.conversation_id = conversation_id
        self.stop = threading.Event()


class ChatServer:
    """The ``pgw chat`` HTTP server.

    Args:
        backend: The SDK (:class:`pgw.chat.backend.SDKBackend`) or a test double.
        store: Where conversations live (:class:`DiskStore` or :class:`MemoryStore`).
        port: TCP port on 127.0.0.1 (0 = a free one).
        host: Must be ``127.0.0.1`` (any other value raises ``ValueError``).
        token: The API token (default: 256 fresh random bits).
        static_dir: The built web app.
        default_model: The model the UI selects first (default: the one recommended for this machine).
        heartbeat_s: Interval of SSE keep-alive comments (they also detect a closed browser tab).
        history_dir: Shown on the privacy screen (where conversations are kept).
        demo: The backend gives canned answers (the page shows a banner).
    """

    def __init__(self, backend: Backend, store: ConversationStore, *, port: int = 0, host: str = "127.0.0.1",
                 token: str | None = None, static_dir: Path | None = None, default_model: str | None = None,
                 heartbeat_s: float = 1.0, max_tokens: int = 1024, history_dir: Path | None = None,
                 demo: bool = False):
        if host != "127.0.0.1":
            raise ValueError("pgw chat binds to 127.0.0.1 only")
        self.backend = backend
        self.store = store
        self.token = token or secrets.token_urlsafe(32)
        if len(self.token) < 32:
            raise ValueError("the token must have at least 32 characters")
        self.static_dir = Path(static_dir or DEFAULT_STATIC_DIR)
        self.device = device_profile()
        self.default_model = default_model or recommended_model(self.device)
        self.heartbeat_s = heartbeat_s
        self.max_tokens = max_tokens
        self.history_dir = history_dir
        self.demo = demo
        self._gens: dict[str, _Generation] = {}
        self._gmu = threading.Lock()
        handler = type("Handler", (_Handler,), {"app": self})
        self.httpd = _Server((host, port), handler)
        self.port = int(self.httpd.server_address[1])
        self.allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @property
    def open_url(self) -> str:
        """The URL that hands the browser its token (in the fragment, never sent to the server)."""
        return f"{self.url}#token={self.token}"

    def start(self) -> ChatServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="pgw-chat-http", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)

    def stop_all(self) -> None:
        with self._gmu:
            for g in self._gens.values():
                g.stop.set()

    def shutdown(self) -> None:
        self.stop_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---- generation registry ------------------------------------------------------------------------
    def _begin(self, cid: str) -> _Generation | None:
        with self._gmu:
            if any(g.conversation_id == cid for g in self._gens.values()):
                return None
            g = _Generation("g_" + secrets.token_hex(8), cid)
            self._gens[g.id] = g
            return g

    def _end(self, g: _Generation) -> None:
        with self._gmu:
            self._gens.pop(g.id, None)

    def _stop(self, gid: str) -> bool:
        with self._gmu:
            g = self._gens.get(gid)
        if g is None:
            return False
        g.stop.set()
        return True

    def _stop_conversation(self, cid: str) -> None:
        with self._gmu:
            for g in self._gens.values():
                if g.conversation_id == cid:
                    g.stop.set()

    def active(self) -> list[str]:
        with self._gmu:
            return [g.conversation_id for g in self._gens.values()]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request: Any, client_address: Any) -> None:  # no tracebacks (could quote anything)
        import sys

        exc = sys.exc_info()[1]
        log.warning("pgw chat: request failed (%s)", type(exc).__name__ if exc else "unknown")


class _ClientGoneError(Exception):
    pass


class _FailedError(Exception):
    def __init__(self, exc: BaseException):
        super().__init__(type(exc).__name__)
        self.exc = exc


class _Handler(BaseHTTPRequestHandler):
    app: ChatServer
    protocol_version = "HTTP/1.1"
    server_version = "pgw-chat"
    sys_version = ""
    timeout = 120

    def log_message(self, format: str, *args: Any) -> None:
        return

    # ---- plumbing ------------------------------------------------------------------------------------
    def _headers(self, status: int, ctype: str, length: int | None, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        extra = {"Cache-Control": "no-store"}
        if self.close_connection:
            extra["Connection"] = "close"
        self._headers(status, "application/json; charset=utf-8", len(body), extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self.close_connection = True  # an unread request body must not be parsed as the next request
        self._json(status, {"error": {"code": code, "message": message}})

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").strip().lower() in self.app.allowed_hosts

    def _api_guard(self, unsafe: bool) -> bool:
        """Origin, fetch-site and token checks of an ``/api/`` call; answers the request when one fails."""
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in self.app.allowed_origins:
            self._error(403, "forbidden_origin", "cross-origin requests are refused")
            return False
        if unsafe and origin is None:
            self._error(403, "forbidden_origin", "state-changing requests must carry Origin")
            return False
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site.strip().lower() not in ("same-origin", "none"):
            self._error(403, "forbidden_origin", "cross-site requests are refused")
            return False
        auth = self.headers.get("Authorization") or ""
        given = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        if not given or not hmac.compare_digest(given.encode("utf-8", "replace"), self.app.token.encode()):
            self._error(401, "unauthenticated", "missing or wrong token: open the link pgw chat printed")
            return False
        return True

    def _read_body(self) -> bool:
        """Read the whole request body now (an unread body would be parsed as the next request on this
        connection); refuses one above :data:`MAX_BODY`."""
        self._raw = b""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self._error(413, "payload_too_large", "request body too large")
            return False
        if self.headers.get("Transfer-Encoding"):
            self._error(411, "length_required", "send a Content-Length")
            return False
        self._raw = self.rfile.read(n) if n else b""
        return True

    def _body(self) -> dict[str, Any] | None:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(415, "unsupported_media_type", "send JSON (Content-Type: application/json)")
            return None
        try:
            data = json.loads(self._raw.decode("utf-8")) if self._raw else {}
        except (UnicodeDecodeError, ValueError):
            self._error(400, "bad_request", "malformed JSON")
            return None
        if not isinstance(data, dict):
            self._error(400, "bad_request", "expected a JSON object")
            return None
        return data

    # ---- dispatch ------------------------------------------------------------------------------------
    def _dispatch(self) -> None:
        if not self._host_ok():
            self.close_connection = True
            self._error(421 if self.headers.get("Host") else 400, "bad_host",
                        "this server answers only to 127.0.0.1 and localhost")
            return
        if not self._read_body():
            return
        path = urlsplit(self.path).path
        if path.startswith("/api/") or path == "/api":
            if not self._api_guard(self.command in ("POST", "DELETE", "PUT", "PATCH")):
                self.close_connection = True
                return
            try:
                self._api(path)
            except (_ClientGoneError, BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            return
        if self.command not in ("GET", "HEAD"):
            self._error(405, "method_not_allowed", "method not allowed")
            return
        self._static(path)

    def do_GET(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_PATCH(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:
        self.close_connection = True
        if not self._host_ok():
            self._error(421, "bad_host", "this server answers only to 127.0.0.1 and localhost")
            return
        self._error(403, "forbidden_origin", "cross-origin requests are refused")

    # ---- static files --------------------------------------------------------------------------------
    def _static(self, path: str) -> None:
        root = self.app.static_dir
        rel = path.lstrip("/")
        if rel == "" or rel.startswith("c/"):
            rel = "index.html"  # the single-page app's routes
        parts = Path(rel).parts
        target = (root / rel).resolve()
        try:
            inside = target.is_relative_to(root.resolve()) and not any(p.startswith(".") for p in parts)
        except OSError:
            inside = False
        if not inside or not target.is_file():
            if rel == "index.html":
                self._headers(503, "text/html; charset=utf-8", len(_NOT_BUILT), {"Cache-Control": "no-store"})
                if self.command != "HEAD":
                    self.wfile.write(_NOT_BUILT)
                return
            self._error(404, "not_found", "not found")
            return
        data = target.read_bytes()
        cache = "public, max-age=31536000, immutable" if rel.startswith("assets/") else "no-cache"
        self._headers(200, _MEDIA.get(target.suffix, "application/octet-stream"), len(data), {"Cache-Control": cache})
        if self.command != "HEAD":
            self.wfile.write(data)

    # ---- API -----------------------------------------------------------------------------------------
    def _api(self, path: str) -> None:
        app, m = self.app, self.command
        parts = [p for p in path.split("/")[2:] if p]
        if parts == ["info"] and m == "GET":
            self._json(200, self._info())
        elif parts == ["status"] and m == "GET":
            q = parse_qs(urlsplit(self.path).query)
            model = (q.get("model") or [app.default_model])[0]
            if self._model_ok(model):
                self._json(200, app.backend.status(model))
        elif parts == ["warmup"] and m == "POST":
            body = self._body()
            if body is not None and self._model_ok(body.get("model")):
                app.backend.warmup(str(body["model"]))
                self._json(202, {"ok": True})
        elif parts == ["conversations"] and m == "GET":
            self._json(200, {"conversations": app.store.summaries(), "active": app.active()})
        elif parts == ["conversations"] and m == "DELETE":
            app.stop_all()
            self._json(200, {"deleted": app.store.delete_all()})
        elif len(parts) == 2 and parts[0] == "conversations" and m in ("GET", "DELETE"):
            cid = parts[1]
            conv = app.store.get(cid) if valid_id(cid) else None
            if conv is None:
                self._error(404, "not_found", "no such conversation")
            elif m == "GET":
                self._json(200, conv)
            else:
                app._stop_conversation(cid)
                self._json(200, {"deleted": app.store.delete(cid)})
        elif parts == ["chat"] and m == "POST":
            self._chat()
        elif len(parts) == 3 and parts[0] == "chat" and parts[2] == "stop" and m == "POST":
            ok = bool(_GEN_RE.fullmatch(parts[1])) and app._stop(parts[1])
            self._json(200 if ok else 404, {"stopped": ok})
        else:
            self._error(404, "not_found", "no such endpoint")

    def _model_ok(self, model: Any) -> bool:
        from pgw.sdk.options import SUPPORTED_MODELS

        if not isinstance(model, str) or model not in SUPPORTED_MODELS:
            self._error(400, "unsupported_model", "pick one of the supported models")
            return False
        return True

    def _info(self) -> dict[str, Any]:
        import shutil

        from pgw import __version__

        app = self.app
        free = None
        with contextlib.suppress(OSError):
            free = round(shutil.disk_usage(app.history_dir or Path.home()).free / 2**30, 1)
        return {"app": "PGW Chat", "version": __version__, "deployment": "local", "demo": app.demo,
                "default_model": app.default_model, "models": app.backend.models(),
                "device": {**app.device, "free_disk_gb": free},
                "recommended_model": recommended_model(app.device, free),
                "settings": app.backend.settings(),
                "history": {"persistent": app.store.persistent,
                            "location": str(app.history_dir) if app.history_dir and app.store.persistent else None},
                "limits": {"max_message_chars": MAX_MESSAGE_CHARS, "max_tokens": app.max_tokens}}

    def _params(self, body: dict[str, Any]) -> dict[str, Any] | None:
        p: dict[str, Any] = {"max_tokens": self.app.max_tokens}
        t = body.get("temperature")
        if t is not None:
            if not isinstance(t, int | float) or isinstance(t, bool) or not 0 <= t <= 2:
                self._error(400, "bad_request", "temperature must be between 0 and 2")
                return None
            p["temperature"] = float(t)
        mt = body.get("max_tokens")
        if mt is not None:
            if not isinstance(mt, int) or isinstance(mt, bool) or not 1 <= mt <= MAX_TOKENS_LIMIT:
                self._error(400, "bad_request", f"max_tokens must be between 1 and {MAX_TOKENS_LIMIT}")
                return None
            p["max_tokens"] = mt
        seed = body.get("seed")
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                self._error(400, "bad_request", "seed must be an integer")
                return None
            p["seed"] = seed
        return p

    def _chat(self) -> None:
        app = self.app
        body = self._body()
        if body is None:
            return
        model = body.get("model")
        if not self._model_ok(model):
            return
        regenerate = body.get("regenerate") is True
        text = body.get("message")
        if not regenerate and (not isinstance(text, str) or not text.strip()):
            self._error(400, "bad_request", "message is required")
            return
        if isinstance(text, str) and len(text) > MAX_MESSAGE_CHARS:
            self._error(413, "payload_too_large", f"a message may have at most {MAX_MESSAGE_CHARS} characters")
            return
        params = self._params(body)
        if params is None:
            return
        cid = body.get("conversation_id")
        if cid is None:
            if regenerate:
                self._error(400, "bad_request", "regenerate needs a conversation")
                return
            cid = app.store.create(str(model))["id"]
        elif not valid_id(cid) or app.store.get(cid) is None:
            self._error(404, "not_found", "no such conversation")
            return
        gen = app._begin(cid)
        if gen is None:
            self._error(409, "busy", "this conversation is already answering")
            return
        try:
            self._run_chat(gen, cid, str(model), None if regenerate else text, params)
        finally:
            app._end(gen)

    def _history(self, cid: str) -> list[dict[str, str]]:
        conv = self.app.store.get(cid) or {"messages": []}
        out = []
        for msg in conv["messages"]:
            role, content = msg.get("role"), msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                out.append({"role": role, "content": content})
        return out

    def _run_chat(self, gen: _Generation, cid: str, model: str, text: str | None, params: dict[str, Any]) -> None:
        app = self.app
        if text is None:
            app.store.truncate_to_last_user(cid)
            user_msg = None
        else:
            user_msg = app.store.append(cid, {"role": "user", "content": text}, model=model)
        messages = self._history(cid)
        if not messages or messages[-1]["role"] != "user":
            self._error(400, "bad_request", "nothing to answer")
            return
        self.close_connection = True
        self._headers(200, "text/event-stream; charset=utf-8", None,
                      {"Cache-Control": "no-store", "Connection": "close", "X-Accel-Buffering": "no"})

        def send(obj: dict[str, Any]) -> None:
            try:
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError as exc:
                raise _ClientGoneError from exc

        conv = app.store.get(cid) or {}
        t0 = time.monotonic()
        parts: list[str] = []
        status, err, ttft, finish, usage = "done", None, None, None, None
        stream: Any = None
        q: queue.Queue[tuple[str, Any]] = queue.Queue()
        pump: threading.Thread | None = None
        gone = False
        try:
            send({"type": "start", "conversation_id": cid, "generation_id": gen.id, "model": model,
                  "title": conv.get("title") or "", "user_message": user_msg})
            try:
                stream = app.backend.stream(model, messages, params, cid)
            except Exception as exc:
                raise _FailedError(exc) from exc

            def run(s: Any = stream) -> None:
                try:
                    for d in s:
                        q.put(("delta", d))
                    q.put(("end", None))
                except BaseException as exc:  # handed to the request thread
                    q.put(("error", exc))

            pump = threading.Thread(target=run, name="pgw-chat-stream", daemon=True)
            pump.start()
            last_beat = time.monotonic()
            while True:
                if gen.stop.is_set():
                    status = "stopped"
                    break
                try:
                    kind, val = q.get(timeout=min(0.25, app.heartbeat_s))
                except queue.Empty:
                    if time.monotonic() - last_beat >= app.heartbeat_s:
                        try:  # a keep-alive comment; fails once the tab was closed or the fetch aborted
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except OSError as exc:
                            raise _ClientGoneError from exc
                        last_beat = time.monotonic()
                    continue
                if kind == "delta":
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    parts.append(val)
                    send({"type": "delta", "text": val})
                    last_beat = time.monotonic()
                elif kind == "end":
                    finish, usage = stream.finish_reason, stream.usage
                    break
                else:
                    raise _FailedError(val)
        except _ClientGoneError:
            gone, status = True, "stopped"
        except _FailedError as f:
            status, err = "error", error_view(f.exc)
        finally:
            if stream is not None and status != "done":
                stream.close()  # stop generating; the request's unused masks are burned, never reused
            content = "".join(parts)
            elapsed = time.monotonic() - t0
            n_out = (usage or {}).get("completion_tokens") or len(parts)
            decode_s = elapsed - (ttft or 0.0)
            record: dict[str, Any] = {
                "role": "assistant", "content": content, "model": model, "status": status,
                # every token shown passed its integrity checks before the library released it
                "verified": err is None and bool(content),
                "finish_reason": finish, "usage": usage, "error": err,
                "elapsed_s": round(elapsed, 3), "ttft_s": None if ttft is None else round(ttft, 3),
                "tokens_per_s": round((n_out - 1) / decode_s, 2) if n_out > 1 and decode_s > 0 else None}
            saved = None
            if content or err is not None:
                with contextlib.suppress(KeyError):
                    saved = app.store.append(cid, record, model=model)
            if not gone:
                with contextlib.suppress(_ClientGoneError):
                    send({"type": "error" if err else "done", "message": saved or record, "error": err})
            if pump is not None:
                pump.join(timeout=0.1)
