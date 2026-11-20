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
  ``scope["extensions"]["tls"]["client_cert_name"]`` and the certificate itself (PEM) into
  ``scope["extensions"]["tls"]["client_cert_chain"]`` (overwriting whatever was there). The
  certificate is what certificate renewal compares the submitted certificate against
  (:func:`peer_cert_der`).

If hypercorn changes its internals the shim refuses to install (:func:`peer_cert_supported`
returns False) and :func:`pgw.service.server.serve_async` refuses to start a configuration that
needs certificate identities, instead of failing open.

Under uvicorn (``server="uvicorn"``, which does not publish the peer certificate either),
:func:`uvicorn_peer_cert_protocol` subclasses uvicorn's HTTP and WebSocket protocol classes: when
a connection is made, the verified subject is read from the connection's own TLS transport and the
protocol's ASGI app is replaced, for that connection only, by a wrapper that publishes it (again
overwriting anything already in the scope). No context variables or global state are involved, and
:func:`uvicorn_peer_cert_supported` checks the hook points before the server starts.
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
    "peer_cert_der",
    "peer_cert_subject",
    "peer_cert_supported",
    "subject_from_der",
    "uvicorn_peer_cert_protocol",
    "uvicorn_peer_cert_supported",
]

log = logging.getLogger(__name__)

_PEER_SUBJECT: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgw_tls_peer_subject", default=None)
_PEER_DER: contextvars.ContextVar[bytes | None] = contextvars.ContextVar("pgw_tls_peer_der", default=None)
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


def _peer_of_writer(writer: Any) -> tuple[str | None, bytes | None]:
    """``(subject, DER)`` of the verified peer certificate of a TLS transport/stream writer."""
    ssl_object = writer.get_extra_info("ssl_object")
    if ssl_object is None:
        return None, None
    der = ssl_object.getpeercert(binary_form=True)
    if not der:
        return None, None
    subject = subject_from_der(der)
    return subject, (bytes(der) if subject else None)


def _der_to_pem(der: bytes) -> str:
    import ssl

    return ssl.DER_cert_to_PEM_cert(der)


def peer_cert_der(scope: Any) -> bytes | None:
    """DER of the client certificate published in ``scope["extensions"]["tls"]["client_cert_chain"]``
    (the first entry, PEM) by this module or the trusted-proxy middleware; ``None`` if absent or
    unparsable."""
    tls = (scope.get("extensions") or {}).get("tls") or {}
    chain = tls.get("client_cert_chain")
    if not isinstance(chain, list | tuple) or not chain or not isinstance(chain[0], str):
        return None
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        cert = x509.load_pem_x509_certificate(chain[0].encode("ascii"))
        return cert.public_bytes(serialization.Encoding.DER)
    except Exception:  # an unparsable certificate is no certificate
        return None


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
            subject: str | None = None
            der: bytes | None = None
            if writer is not None:
                try:
                    subject, der = _peer_of_writer(writer)
                except Exception:  # never break the connection over identity capture
                    log.exception("could not read the TLS peer certificate")
            _PEER_SUBJECT.set(subject)
            _PEER_DER.set(der)
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
            scope = _publish(scope, _PEER_SUBJECT.get(), _PEER_DER.get())
        await self.app(scope, receive, send)


def _publish(scope: Scope, subject: str | None, der: bytes | None = None) -> Scope:
    """Copy of ``scope`` whose TLS extension carries exactly ``subject`` and the certificate ``der``
    (as ``client_cert_chain = [PEM]``); nothing when ``None``."""
    scope = dict(scope)
    ext = dict(scope.get("extensions") or {})
    tls = dict(ext.get("tls") or {})
    tls.pop("client_cert_name", None)
    tls.pop("client_cert_subject", None)
    tls.pop("client_cert_chain", None)
    if subject:
        tls["client_cert_name"] = subject
        if der:
            tls["client_cert_chain"] = [_der_to_pem(der)]
    ext["tls"] = tls
    scope["extensions"] = ext
    return scope


# ---------------------------------------------------------------------------------------------
# uvicorn
# ---------------------------------------------------------------------------------------------
class _ConnectionPeerApp:
    """The ASGI app of one uvicorn connection: publishes that connection's verified subject."""

    __slots__ = ("app", "der", "subject")

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], subject: str | None,
                 der: bytes | None = None):
        self.app = app
        self.subject = subject
        self.der = der

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") in ("http", "websocket"):
            scope = _publish(scope, self.subject, self.der)
        await self.app(scope, receive, send)


async def _refuse(scope: Scope, receive: Receive, send: Send) -> None:  # pragma: no cover - API drift guard
    """Fail closed when the per-connection identity could not be attached."""
    if scope.get("type") == "websocket":
        await send({"type": "websocket.close", "code": 1011, "reason": "identity unavailable"})
    elif scope.get("type") == "http":
        await send({"type": "http.response.start", "status": 503, "headers": [(b"content-length", b"0")]})
        await send({"type": "http.response.body", "body": b""})


def uvicorn_peer_cert_protocol(base: type[Any]) -> type[Any]:
    """Subclass of a uvicorn protocol class (HTTP or WebSocket) that publishes the connection's
    verified client-certificate subject in every scope it creates.

    uvicorn hands the upgraded WebSocket protocol the same TLS transport, so the HTTP and the
    WebSocket protocol of a connection both read the certificate the handshake verified.
    """

    class PeerCertProtocol(base):  # type: ignore[misc]
        def connection_made(self, transport: Any) -> None:
            app = getattr(self, "app", None)
            subject: str | None = None
            der: bytes | None = None
            try:
                subject, der = _peer_of_writer(transport)
            except Exception:  # never break the connection over identity capture
                log.exception("could not read the TLS peer certificate")
            # the protocol instance serves exactly this connection: wrap its app for it alone
            self.app = _ConnectionPeerApp(app, subject, der) if callable(app) else _refuse
            super().connection_made(transport)

    PeerCertProtocol.__name__ = PeerCertProtocol.__qualname__ = f"PeerCert{base.__name__}"
    return PeerCertProtocol


def uvicorn_peer_cert_supported(*protocol_classes: type[Any]) -> bool:
    """Whether :func:`uvicorn_peer_cert_protocol` can hook these uvicorn protocol classes: each must
    keep its ASGI app in the instance attribute ``app`` (set in ``__init__``) and receive its
    transport through ``connection_made``."""
    import inspect

    for cls in protocol_classes:
        try:
            init_src = inspect.getsource(cls.__init__)
        except (OSError, TypeError):  # pragma: no cover - no source available: cannot verify
            return False
        if "self.app = " not in init_src or not callable(getattr(cls, "connection_made", None)):
            log.error("uvicorn protocol %s does not look as expected; mTLS identities unavailable", cls.__name__)
            return False
    return True
