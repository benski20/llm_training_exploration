"""Expose the verified mTLS client certificate to the ASGI app when serving natively with hypercorn.

Hypercorn (0.18) negotiates TLS and verifies client certificates, but it does not publish the peer
certificate in the ASGI scope (there is no ``scope["extensions"]["tls"]``). Without it the wire
router never sees a client identity: ``mtls_tenants_file``-only deployments reject everything and
the token-vs-certificate tenant cross-check in
:meth:`pgw.service.auth.TenantAuthenticator.authenticate` silently never runs.

This module closes that gap with two small pieces:

* :func:`install_hypercorn_peer_cert_shim` wraps ``hypercorn.asyncio.tcp_server.TCPServer.run``
  so that, for every TLS connection, the RFC 4514 subject of the *verified* peer certificate is
  stored in a :class:`contextvars.ContextVar`. Hypercorn runs each connection in its own asyncio
  task and spawns the per-request app tasks from it, so the value is visible to exactly the
  requests of that connection (HTTP/1.1 keep-alive and every HTTP/2 stream) and nothing else.
* :class:`PeerCertMiddleware` copies that value into
  ``scope["extensions"]["tls"]["client_cert_name"]`` (overwriting whatever was there).

If hypercorn changes its internals the shim refuses to install (:func:`peer_cert_supported`
returns False) and :func:`pgw.service.server.serve_async` refuses to start a configuration that
needs certificate identities, instead of failing open.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "PeerCertMiddleware",
    "install_hypercorn_peer_cert_shim",
    "peer_cert_subject",
    "peer_cert_supported",
    "subject_from_der",
]

log = logging.getLogger(__name__)

_PEER_SUBJECT: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgw_tls_peer_subject", default=None)
_INSTALL_LOCK = threading.Lock()
_installed: bool | None = None  # None = not attempted yet
_MARKER = "__pgw_peer_cert_shim__"


def subject_from_der(der: bytes) -> str | None:
    """RFC 4514 subject string (``CN=gw1,O=Acme``) of a DER certificate, or ``None`` if unparsable."""
    try:
        from cryptography import x509

        return x509.load_der_x509_certificate(der).subject.rfc4514_string() or None
    except Exception:  # an unparsable cert simply yields no identity
        return None


def peer_cert_subject() -> str | None:
    """Subject of the verified client certificate of the current connection (``None`` if none)."""
    return _PEER_SUBJECT.get()


def _subject_of_writer(writer: Any) -> str | None:
    ssl_object = writer.get_extra_info("ssl_object")
    if ssl_object is None:
        return None
    der = ssl_object.getpeercert(binary_form=True)
    if not der:
        return None
    return subject_from_der(der)


def install_hypercorn_peer_cert_shim() -> bool:
    """Install the per-connection peer-certificate capture into hypercorn (idempotent).

    Returns:
        True when installed (or already installed), False when hypercorn's internals do not look
        as expected (then no certificate identity is available to the app).
    """
    global _installed
    with _INSTALL_LOCK:
        if _installed is not None:
            return _installed
        try:
            from hypercorn.asyncio import tcp_server
        except ImportError:  # pragma: no cover - hypercorn is a service dependency
            _installed = False
            return False
        cls = getattr(tcp_server, "TCPServer", None)
        orig = getattr(cls, "run", None)
        if cls is None or orig is None or not callable(orig):  # pragma: no cover - hypercorn API change
            log.error("hypercorn.asyncio.tcp_server.TCPServer.run not found; mTLS identities unavailable")
            _installed = False
            return False
        if getattr(orig, _MARKER, False):
            _installed = True
            return True

        @functools.wraps(orig)
        async def run(self: Any) -> None:
            writer = getattr(self, "writer", None)
            subject = None
            if writer is not None:
                try:
                    subject = _subject_of_writer(writer)
                except Exception:  # never break the connection over identity capture
                    log.exception("could not read the TLS peer certificate")
            _PEER_SUBJECT.set(subject)
            await orig(self)

        setattr(run, _MARKER, True)
        cls.run = run
        _installed = True
        log.debug("installed hypercorn TLS peer-certificate shim")
        return True


def peer_cert_supported() -> bool:
    """Whether verified client certificates reach the app when serving with hypercorn."""
    return install_hypercorn_peer_cert_shim()


Scope = dict[str, Any]
Receive = Callable[[], Awaitable[Any]]
Send = Callable[[Any], Awaitable[None]]


class PeerCertMiddleware:
    """ASGI middleware publishing the connection's verified client certificate subject.

    Writes ``scope["extensions"]["tls"]["client_cert_name"]`` from :func:`peer_cert_subject`, and
    removes any value already present when the connection has no verified certificate (so nothing
    upstream can inject one).
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]):
        self.app = app

    def __getattr__(self, name: str) -> Any:
        # keep ``app.state`` and friends reachable through the wrapper
        return getattr(self.app, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") in ("http", "websocket"):
            subject = _PEER_SUBJECT.get()
            scope = dict(scope)
            ext = dict(scope.get("extensions") or {})
            tls = dict(ext.get("tls") or {})
            tls.pop("client_cert_name", None)
            tls.pop("client_cert_subject", None)
            if subject:
                tls["client_cert_name"] = subject
            ext["tls"] = tls
            scope["extensions"] = ext
        await self.app(scope, receive, send)
