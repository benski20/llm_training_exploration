"""The placement controller's HTTP API for pods (TLS with the controller's own certificate; pods pin its CA hash).

    POST /placement/v1/register   {token, pod_id, pod_name, dc, model, fingerprints, selftests, csr_pem,
                                   boot_timings} -> {chain_pem, root_pem, client_ca_pem, leaf_sha256}
                                   (pgw.placement.registry.Registry.register; the pod's public address comes from
                                   RunPod's API, ``lookup``, never from the request)
                                   (+ ``pod_secret``: a per-pod secret, stored hashed, for the two routes below)
    POST /placement/v1/ready      {pod_id, pod_secret} -> 204 once the controller's own probe passed and the pod
                                   is in the next signed directory (Registry.activate)
    GET  /placement/v1/tenants    (headers PGW-Pod-Id, PGW-Pod-Secret) the tenant credential *digests* a serving pod
                                   needs: the tokens file (SHA-256 of service tokens) and the client-certificate
                                   subject map; never a token, never a private key

Every route answers 403 to anything it cannot tie to a registration it issued.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from pgw.placement.registry import RegistrationError, Registry

__all__ = ["create_controller_app"]

log = logging.getLogger(__name__)
MAX_BODY = 64 * 1024


def create_controller_app(registry: Registry, *, lookup: Callable[[str], tuple[str, int, int | None]],
                          probe: Callable[[str, Any], Mapping[str, Any]], root_pem: bytes, client_ca_pem: bytes,
                          tenants: Callable[[], Mapping[str, Any]] | None = None) -> Any:
    """``lookup(pod_id) -> (public host, service port, lane port)`` from RunPod's API; ``probe(url, cert)`` the
    client-like check of :meth:`Registry.activate`; ``tenants()`` the digests a pod serves with."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    secrets_of: dict[str, bytes] = {}  # pod id -> SHA-256 of a per-pod secret returned at registration

    async def _json(request: Request) -> Any:
        body = await request.body()
        if len(body) > MAX_BODY:
            raise RegistrationError("body too large")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise RegistrationError("not JSON") from exc

    def _authorised(pod_id: str, secret: str) -> bool:
        d = secrets_of.get(pod_id)
        return d is not None and hmac.compare_digest(d, hashlib.sha256(secret.encode()).digest())

    async def register(request: Request) -> Response:
        try:
            req = await _json(request)
            pod_id = str(req.get("pod_id", ""))
            host, port, lane = lookup(pod_id)  # RunPod's view of the pod, not the pod's claim
            rec = registry.register(req, public_host=host, public_port=port, public_lane_port=lane)
        except (RegistrationError, KeyError, ValueError) as exc:
            log.warning("pod registration refused: %s", exc)
            return JSONResponse({"error": "refused"}, status_code=403)
        import secrets

        sec = secrets.token_urlsafe(32)
        secrets_of[rec.pod_id] = hashlib.sha256(sec.encode()).digest()
        return JSONResponse({"chain_pem": rec.cert.chain_pem.decode(), "root_pem": root_pem.decode(),
                             "client_ca_pem": client_ca_pem.decode(), "leaf_sha256": rec.cert.sha256,
                             "pod_secret": sec, "url": rec.url})

    async def ready(request: Request) -> Response:
        try:
            req = await _json(request)
            pod_id = str(req.get("pod_id", ""))
            if not _authorised(pod_id, str(req.get("pod_secret", ""))):
                return Response(status_code=403)
            registry.activate(pod_id, probe)
        except RegistrationError as exc:
            log.warning("pod activation refused: %s", exc)
            return JSONResponse({"error": "refused"}, status_code=403)
        return Response(status_code=204)

    async def tenant_digests(request: Request) -> Response:
        pod_id = request.headers.get("pgw-pod-id", "")
        sec = request.headers.get("pgw-pod-secret", "")
        rec = registry.pods.get(pod_id)
        if rec is None or not _authorised(pod_id, sec) or rec.state not in ("certified", "serving", "draining"):
            return Response(status_code=403)
        return JSONResponse(dict(tenants()) if tenants is not None else {"tokens": [], "subjects": {}},
                            headers={"Cache-Control": "no-store"})

    return Starlette(routes=[Route("/placement/v1/register", register, methods=["POST"]),
                             Route("/placement/v1/ready", ready, methods=["POST"]),
                             Route("/placement/v1/tenants", tenant_digests, methods=["GET"])])


def tls_probe(url: str, cert: Any, *, root_pem: bytes, token: str | None = None,
              timeout_s: float = 10.0) -> dict[str, Any]:
    """Connect to a pod like a client would (TLS verified against ``root_pem`` only, the pod's address as the
    name): the served leaf's SHA-256, ``/readyz`` and the fingerprints of ``/pgw/v1/models`` (with the controller's
    probe tenant token, when given)."""
    import socket
    import ssl
    from urllib.parse import urlsplit

    import httpx

    u = urlsplit(url)
    ctx = ssl.create_default_context(cadata=root_pem.decode())
    with socket.create_connection((u.hostname, u.port or 443), timeout=timeout_s) as raw, \
            ctx.wrap_socket(raw, server_hostname=u.hostname) as s:
        der = s.getpeercert(binary_form=True) or b""
    out: dict[str, Any] = {"leaf_sha256": hashlib.sha256(der).hexdigest()}
    with httpx.Client(base_url=url, verify=ctx, timeout=timeout_s, trust_env=False) as c:
        out["ready"] = c.get("/readyz").status_code == 200
        if token:
            r = c.get("/pgw/v1/models", headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 200:
                out["fingerprints"] = [str(m.get("fingerprint")) for m in r.json().get("models", [])]
    return out
