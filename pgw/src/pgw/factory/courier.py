"""Segment courier: deliver partner-export segments from a mask-service VM to a gateway VM in the same VPC.

The payload is the existing partner export (SPEC 7.7): each ``*.pgwm`` in the factory's outbox has its DEK
HPKE-sealed to the gateway store's recipient key, an Ed25519 ``.sig`` sidecar from the factory's signing key, and
the destination store id in its signed header. The courier adds only transport; it never sees a key:

* **Outbox server** (mask-service VM, ``pgw factory outbox-serve``): ``GET /pgw/v1/outbox`` lists complete
  segments (with their signature), ``GET /pgw/v1/outbox/<name>`` streams a file, ``DELETE /pgw/v1/outbox/<name>``
  retires a segment and its signature after the gateway imported it. Bearer token required (constant-time
  compare); optional TLS. Names are confined to the outbox.
* **Puller** (gateway VM, ``pgw maskstore pull``): downloads segment + signature into a private inbox, imports
  them with :func:`pgw.maskstore.admin.import_files` (verifies the signature against
  ``PGW_MASKSTORE_TRUSTED_PARTNER_KEYS``, the destination, every chunk's AEAD; re-wraps the DEK under the local
  KEK; deletes the local copy), and only then deletes the remote copy. A file the store already holds (mask
  reuse refusal) is deleted remotely as well; other failures leave it for the operator. Pulling stops while the
  store holds ``max_available_slots`` or more for the segment's model and ring (back-pressure).

Confidentiality and integrity do not depend on the channel: a network observer or a forged server gets sealed
files the gateway refuses unless they carry a trusted signature and its own store id. The token only stops other
VPC hosts from listing or deleting outbox files (denial of service).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError

__all__ = ["OutboxServer", "pull_once"]

log = logging.getLogger(__name__)

_PREFIX = "/pgw/v1/outbox"
_SUFFIX = ".pgwm"
_SIG = ".sig"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.pgwm(\.sig)?$")
_COPY_BYTES = 1 << 20


def _read_token(token_file: str | Path) -> bytes:
    tok = Path(token_file).read_text().strip().encode()
    if len(tok) < 16:
        raise ConfigError("courier token must be at least 16 characters")
    return tok


class OutboxServer:
    """HTTP(S) server over an :class:`~pgw.factory.sinks.ExportSink` outbox (see module docstring).

    Args:
        outbox: The export sink's ``<export_dir>/outbox``.
        token_file: File with the shared bearer token (also given to the gateway's puller).
        host, port: Bind address (the VPC-internal address of the mask-service VM; port 0 picks one).
        tls_cert_file, tls_key_file: Serve HTTPS (recommended; the gateway pins the CA with ``--ca-file``).
    """

    def __init__(self, outbox: str | Path, token_file: str | Path, host: str = "127.0.0.1", port: int = 8470, *,
                 tls_cert_file: str | Path | None = None, tls_key_file: str | Path | None = None):
        self.outbox = Path(outbox)
        self.outbox.mkdir(parents=True, exist_ok=True)
        token = _read_token(token_file)
        root = self.outbox.resolve()
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            server_version = "pgw-outbox/1"
            sys_version = ""

            def log_message(self, fmt: str, *args: Any) -> None:  # route to logging, no secrets in paths
                log.debug("outbox %s - " + fmt, self.client_address[0], *args)

            def _auth(self) -> bool:
                got = self.headers.get("Authorization", "")
                ok = got.startswith("Bearer ") and hmac.compare_digest(got[7:].strip().encode(), token)
                if not ok:
                    self._send(401, {"error": "unauthorized"})
                return ok

            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _path(self) -> Path | None:
                name = urllib.parse.unquote(self.path[len(_PREFIX) + 1:]) if self.path.startswith(_PREFIX + "/") \
                    else ""
                if not _NAME.match(name):
                    self._send(404, {"error": "not found"})
                    return None
                p = (root / name).resolve()
                if p.parent != root:
                    self._send(404, {"error": "not found"})
                    return None
                return p

            def do_GET(self) -> None:
                if not self._auth():
                    return
                if self.path.rstrip("/") == _PREFIX:
                    self._send(200, {"segments": list_outbox(root)})
                    return
                p = self._path()
                if p is None:
                    return
                try:
                    f = p.open("rb")
                except OSError:
                    self._send(404, {"error": "not found"})
                    return
                with f:
                    size = os.fstat(f.fileno()).st_size
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    shutil.copyfileobj(f, self.wfile, _COPY_BYTES)

            def do_DELETE(self) -> None:
                if not self._auth():
                    return
                p = self._path()
                if p is None:
                    return
                if not p.name.endswith(_SUFFIX):
                    self._send(400, {"error": "delete the segment, not its signature"})
                    return
                with lock:
                    existed = p.exists()
                    from pgw.maskstore.admin import _retire_source

                    _retire_source(p)  # segment (renamed first) and its .sig, then fsync of the directory
                log.info("outbox: segment %s delivered and removed", p.name)
                self._send(200 if existed else 404, {"deleted": p.name, "existed": existed})

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.tls = tls_cert_file is not None
        if tls_cert_file is not None:
            if tls_key_file is None:
                raise ConfigError("outbox server: tls_cert_file needs tls_key_file")
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(str(tls_cert_file), str(tls_key_file))
            self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = self.httpd.server_address[0]
        return f"{'https' if self.tls else 'http'}://{host}:{self.port}"

    def start(self) -> OutboxServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="pgw-outbox", daemon=True)
        self._thread.start()
        log.info("outbox server on %s for %s", self.url, self.outbox)
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def list_outbox(outbox: Path) -> list[dict[str, Any]]:
    """Complete, signed segments in ``outbox`` (public header fields only)."""
    from pgw.factory.sinks import _segment_info

    out: list[dict[str, Any]] = []
    for p in sorted(outbox.glob(f"*{_SUFFIX}")):
        sig = p.with_name(p.name + _SIG)
        if not sig.exists():
            continue  # the factory places the signature first; unsigned files are never offered
        try:
            info = _segment_info(p)
        except Exception as exc:  # unreadable header: skip
            log.warning("outbox: skipping %s: %r", p.name, exc)
            continue
        out.append({"name": p.name, "sig": sig.name, "segment_id": info.segment_id, "fingerprint": info.fingerprint,
                    "ring": info.ring, "slots": info.slots, "size": info.size})
    return out


# ---- gateway side --------------------------------------------------------------------------------------------
def _opener(ca_file: str | Path | None) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [urllib.request.ProxyHandler({})]  # VPC-internal: never via an egress proxy
    if ca_file is not None:
        handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=str(ca_file))))
    return urllib.request.build_opener(*handlers)


def _req(opener: Any, method: str, url: str, token: bytes, timeout: float) -> Any:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ConfigError(f"mask service outbox URL must be http(s), got {url!r}")
    rq = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {token.decode()}"})  # noqa: S310
    return opener.open(rq, timeout=timeout)


def _download(opener: Any, url: str, token: bytes, dest: Path, timeout: float) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with _req(opener, "GET", url, token, timeout) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, _COPY_BYTES)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)


def pull_once(base_url: str, token_file: str | Path, inbox: str | Path, *, ca_file: str | Path | None = None,
              max_available_slots: int | None = None, timeout_s: float = 300.0,
              store_overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fetch, import and retire every offered segment once; one result entry per segment.

    ``max_available_slots``: skip segments of a ``(fingerprint, ring)`` of which the store already holds at
    least this many available slots. ``store_overrides`` go to :class:`pgw.maskstore.MaskStoreSettings`.
    """
    from pgw.maskstore.admin import import_files, open_store

    token = _read_token(token_file)
    inbox_p = Path(inbox)
    inbox_p.mkdir(parents=True, exist_ok=True, mode=0o700)
    opener = _opener(ca_file)
    base = base_url.rstrip("/") + _PREFIX
    with _req(opener, "GET", base, token, timeout_s) as resp:
        offered = json.loads(resp.read().decode())["segments"]
    overrides = dict(store_overrides or {})
    results: list[dict[str, Any]] = []
    for seg in offered:
        name, sig = str(seg["name"]), str(seg["sig"])
        if not (_NAME.match(name) and _NAME.match(sig) and sig == name + _SIG and name.endswith(_SUFFIX)):
            results.append({"file": name, "ok": False, "error": "bad name offered"})
            continue
        if max_available_slots is not None:
            with open_store(**overrides) as store:
                avail = sum(i.slots_available for i in store.inventory(seg["fingerprint"], seg["ring"]))
            if avail >= max_available_slots:
                results.append({"file": name, "ok": True, "skipped": "store full", "available": avail})
                continue
        local = inbox_p / name
        try:
            _download(opener, f"{base}/{urllib.parse.quote(sig)}", token, inbox_p / sig, timeout_s)
            _download(opener, f"{base}/{urllib.parse.quote(name)}", token, local, timeout_s)
        except (OSError, urllib.error.URLError) as exc:
            for p in (local, inbox_p / sig, local.with_name(local.name + ".part")):
                p.unlink(missing_ok=True)
            results.append({"file": name, "ok": False, "error": f"download failed: {exc!r}"})
            continue
        res = import_files([local], **overrides)[0]  # deletes the local copy (and .sig) on success
        retire = bool(res.get("ok")) or res.get("code") == "mask_reuse"
        if not res.get("ok"):
            for p in (local, inbox_p / sig):
                p.unlink(missing_ok=True)
        if retire:
            try:
                with _req(opener, "DELETE", f"{base}/{urllib.parse.quote(name)}", token, timeout_s):
                    pass
            except (OSError, urllib.error.URLError) as exc:  # imported; a later pull retries the delete
                res["warning"] = f"remote delete failed: {exc!r}"
        res["file"] = name
        results.append(res)
    return results
