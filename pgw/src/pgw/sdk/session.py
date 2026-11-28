"""Session bootstrap: enroll this installation with the provider once per SDK key, then reuse the files.

Reuses the gateway's enrollment machinery (:mod:`pgw.enroll.client`): the provider CA is fetched over an
unverified connection and accepted only if its SHA-256 equals the pin in the SDK key (nothing secret is sent
before that); the client key is generated locally and never leaves this host; the enrollment request goes over
TLS verified against the pinned CA only (no system trust store).

``POST /pgw/v1/sdk/enroll`` (provider endpoint), ``Authorization: Bearer <SDK key>``, JSON body::

    {"client_name": "<host-derived name>", "csr_pem": "<PEM CSR>", "sdk_version": "<pgw version>"}

Response (200), the same fields as ``POST /pgw/v1/enroll`` plus optional extensions::

    {"tenant": str, "service_token": str, "client_cert_pem": str | null,
     "models": [{"name", "fingerprint", "ring", "rings", "masked_head"?: bool, ...}],
     "lane": {"transport": "tcp", "port": int} | null}       # optional: the provider's TCP+TLS trip lane

Files (``<cache>/sessions/<key_id>/``): ``secrets/service_token`` and ``secrets/client.key`` (0600),
``client.crt``, ``service-ca.pem``, ``enrollment.json`` (public metadata). They are written under an exclusive
lock, so concurrent first uses in several processes enroll once.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from pgw.core.errors import ConfigError
from pgw.core.fileio import atomic_write_text, write_private
from pgw.sdk.cache import CacheLayout, ensure_private_dir, file_lock
from pgw.sdk.errors import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    IntegrityError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from pgw.sdk.keys import SdkKey

__all__ = ["SDK_ENROLL_PATH", "Session", "ensure_session", "load_session"]

log = logging.getLogger(__name__)

SDK_ENROLL_PATH = "/pgw/v1/sdk/enroll"
_FILES = {"token": "secrets/service_token", "client_key": "secrets/client.key", "client_cert": "client.crt",
          "ca": "service-ca.pem", "meta": "enrollment.json"}


@dataclass
class Session:
    """An enrolled installation (no secret values in ``repr``).

    This is the ``session`` object handed to the provisioning function
    (:func:`pgw.provision.client.provision`): everything needed to open an authenticated, CA-pinned
    connection to the provider.

    Attributes:
        key_id: Public id of the SDK key.
        service_url: The provider's base URL.
        tenant: Tenant id assigned by the provider.
        dir: The session directory.
        token_file: Service bearer token (0600).
        ca_file: The pinned provider CA (PEM), or ``None`` for plain http to loopback.
        client_cert_file / client_key_file: The mTLS client certificate and its key (0600), if issued.
        models: The provider's model entries (``name``, ``fingerprint``, ``ring``, ...).
        lane: ``{"transport": "tcp", "port": int}`` when the provider offers its TCP+TLS trip lane.
        enrolled_at: Unix time of the enrollment.
        directory_key: The signed endpoint directory's public key (64 hex) the provider sent at enrollment over
            the CA-pinned channel, or ``None`` (pgw.sdk.directory; a version-2 SDK key's own pin takes precedence).
    """

    key_id: str
    service_url: str
    tenant: str
    dir: Path
    token_file: Path
    ca_file: Path | None
    client_cert_file: Path | None
    client_key_file: Path | None
    models: list[dict[str, Any]] = field(default_factory=list)
    lane: dict[str, Any] | None = None
    enrolled_at: float = 0.0
    ca_fingerprint: str | None = None
    directory_key: str | None = None

    def token(self) -> SecretStr:
        """The service bearer token (read from its 0600 file)."""
        return SecretStr(self.token_file.read_text(encoding="utf-8").strip())

    def model_entry(self, name: str) -> dict[str, Any] | None:
        """The provider's entry for model ``name`` (exact name), or ``None``."""
        for m in self.models:
            if str(m.get("name", "")).lower() == name.lower():
                return m
        return None

    def http_client(self, *, timeout_s: float = 60.0, transport: Any = None) -> Any:
        """A blocking ``httpx.Client`` to the provider: CA pinned, client certificate presented, bearer token set."""
        import ssl

        import httpx

        kw: dict[str, Any] = {"timeout": timeout_s, "follow_redirects": False, "trust_env": True,
                              "headers": {"Authorization": "Bearer " + self.token().get_secret_value()}}
        if transport is not None:
            kw["transport"] = transport
        elif self.service_url.startswith("https://"):
            ctx = ssl.create_default_context(cafile=str(self.ca_file)) if self.ca_file else ssl.create_default_context()
            if self.client_cert_file is not None and self.client_key_file is not None:
                ctx.load_cert_chain(str(self.client_cert_file), str(self.client_key_file))
            kw["verify"] = ctx
        return httpx.Client(base_url=self.service_url, **kw)

    def public(self) -> dict[str, Any]:
        """JSON view without secrets (``pgw sdk status``)."""
        return {"key_id": self.key_id, "service_url": self.service_url, "tenant": self.tenant,
                "enrolled_at": self.enrolled_at, "ca_fingerprint": self.ca_fingerprint,
                "client_cert": self.client_cert_file is not None, "lane": self.lane,
                "models": [{"name": m.get("name"), "fingerprint": m.get("fingerprint"), "ring": m.get("ring"),
                            "masked_head": m.get("masked_head")} for m in self.models]}


def _paths(d: Path) -> dict[str, Path]:
    return {k: d / v for k, v in _FILES.items()}


def load_session(key: SdkKey, layout: CacheLayout) -> Session | None:
    """The cached session of ``key`` if it is complete and was made for this key, provider and CA pin."""
    d = layout.session_dir(key.key_id_hex)
    p = _paths(d)
    try:
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (meta.get("key_id") != key.key_id_hex or meta.get("service_url") != key.service_url
            or meta.get("ca_fingerprint") != (key.ca_fingerprint_hex if key.pinned else None)
            or not p["token"].is_file()):
        return None
    if key.pinned and not _ca_file_matches(p["ca"], key.ca_sha256):
        # a missing or replaced CA file must not fall back to the system trust store (or trust another CA):
        # enroll again, which fetches the CA and checks it against the key's pin
        log.warning("the cached provider CA of SDK key %s is missing or does not match the key's pin; enrolling "
                    "again", key.key_id_hex)
        return None
    has_cert = bool(meta.get("client_cert")) and p["client_cert"].is_file() and p["client_key"].is_file()
    if not _tls_material_ok(p["ca"] if p["ca"].is_file() else None, key,
                            (p["client_cert"], p["client_key"]) if has_cert else None):
        log.warning("the cached session files of SDK key %s are damaged; enrolling again", key.key_id_hex)
        return None
    return Session(
        key_id=key.key_id_hex, service_url=key.service_url, tenant=str(meta.get("tenant", "")), dir=d,
        token_file=p["token"], ca_file=p["ca"] if p["ca"].is_file() else None,
        client_cert_file=p["client_cert"] if has_cert else None, client_key_file=p["client_key"] if has_cert else None,
        models=list(meta.get("models") or []), lane=meta.get("lane"), enrolled_at=float(meta.get("enrolled_at", 0)),
        ca_fingerprint=meta.get("ca_fingerprint"), directory_key=_directory_key(meta.get("directory_key")))


def _directory_key(v: Any) -> str | None:
    """A 64-hex Ed25519 public key, or ``None`` (anything else is ignored: routing then stays on the enrolled URL)."""
    if not isinstance(v, str):
        return None
    try:
        from pgw.sdk.directory import load_public_key

        return load_public_key(v).hex()
    except ValueError:
        return None


def _ca_file_matches(path: Path, pin: bytes) -> bool:
    """Whether ``path`` holds exactly one certificate, the one ``pin`` names."""
    import hmac

    from pgw.enroll.service import pinned_ca

    try:
        fp, _ = pinned_ca(path.read_text(encoding="ascii"))
    except (OSError, ValueError, ConfigError):
        return False
    return hmac.compare_digest(fp, pin)


def _tls_material_ok(ca: Path | None, key: SdkKey, pair: tuple[Path, Path] | None) -> bool:
    """The cached CA still matches the key's pin and the client certificate loads with its key (a damaged or
    half-written file, e.g. after a full disk or a crash, makes the session unusable: it is enrolled again)."""
    import ssl

    try:
        if key.service_url.startswith("https://"):
            if ca is None:
                return False
            if key.pinned and not _ca_file_matches(ca, bytes.fromhex(key.ca_fingerprint_hex)):
                return False
        if pair is not None:
            ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_cert_chain(str(pair[0]), str(pair[1]))
    except (OSError, ValueError, ssl.SSLError, ConfigError):
        return False
    return True


def ensure_session(key: SdkKey, layout: CacheLayout, *, force: bool = False, transport: Any = None,
                   timeout_s: float = 30.0, client_name: str | None = None) -> Session:
    """The enrolled session of ``key``: from the cache, or enrolled now (once per host, under a lock).

    Args:
        force: Enroll again even if a cached session exists (e.g. after the provider revoked its token).
        transport: httpx transport override (tests).

    Raises:
        AuthenticationError: the provider does not accept the key (unknown, revoked).
        IntegrityError: the provider's CA does not match the key's pin (possible interception).
        APIConnectionError / RateLimitError / PermissionDeniedError / InternalServerError.
    """
    d = layout.session_dir(key.key_id_hex)
    ensure_private_dir(layout.root)
    ensure_private_dir(d)
    with file_lock(d / ".enroll.lock"):
        if not force:
            s = load_session(key, layout)
            if s is not None and not _cert_expired(s):
                return s
            if s is not None:  # an idle cache: the provider no longer accepts the certificate; enroll again
                log.warning("the client certificate of SDK key %s has expired; enrolling again", key.key_id_hex)
        s = _enroll(key, d, transport=transport, timeout_s=timeout_s, client_name=client_name)
        if _cert_expired(s, margin_s=0.0) or _cert_not_yet_valid(s):
            log.error("the client certificate the provider just issued is not valid by this machine's clock: the "
                      "clock is wrong (check it, e.g. 'date -u', and enable NTP time sync)")
        return s


def _cert_window(s: Session) -> tuple[float, float] | None:
    if s.client_cert_file is None:
        return None
    from pgw.wire.client import client_cert_validity

    return client_cert_validity(s.client_cert_file)


def _cert_expired(s: Session, margin_s: float = 60.0) -> bool:
    """The session's client certificate has expired (or expires within ``margin_s``) by this machine's clock."""
    w = _cert_window(s)
    return w is not None and time.time() >= w[1] - margin_s


def _cert_not_yet_valid(s: Session) -> bool:
    w = _cert_window(s)
    return w is not None and time.time() < w[0]


def _enroll(key: SdkKey, d: Path, *, transport: Any, timeout_s: float, client_name: str | None) -> Session:
    import ssl

    import httpx

    import pgw
    from pgw.enroll.client import EnrollmentError, _client, _default_name, _fetch_pinned_ca, _key_and_csr

    try:  # CA pin check before anything secret is sent (SdkKey has the fields EnrollmentCode has)
        ca_pem = _fetch_pinned_ca(key, transport, timeout_s)  # type: ignore[arg-type]
    except EnrollmentError as exc:
        if str(exc.code) == "integrity_check_failed":
            raise IntegrityError(exc.message, code=str(exc.code)) from exc
        raise APIConnectionError(exc.message, code=str(exc.code)) from exc
    verify: Any = False
    if key.service_url.startswith("https://"):
        if ca_pem is None:  # pragma: no cover - an https key always pins (SdkKey refuses otherwise)
            raise IntegrityError("the provider sent no CA certificate for a pinned https key")
        verify = ssl.create_default_context(cadata=ca_pem)
    name = client_name or _default_name()
    key_pem, csr_pem = _key_and_csr(name)
    body = {"client_name": name, "gateway_name": name, "csr_pem": csr_pem, "sdk_version": pgw.__version__}
    try:
        with _client(key, verify, transport, timeout_s) as h:  # type: ignore[arg-type]
            r = h.post(SDK_ENROLL_PATH, json=body, headers={"Authorization": key.bearer()})
    except httpx.HTTPError as exc:
        raise APIConnectionError(f"cannot reach the provider at {key.service_url} ({type(exc).__name__})") from exc
    if r.status_code != 200:
        try:
            msg = str(r.json()["error"]["message"])
        except (ValueError, KeyError, TypeError):
            msg = f"HTTP {r.status_code}"
        if r.status_code == 401:
            raise AuthenticationError(f"the provider did not accept this API key (unknown, revoked or expired): {msg}")
        if r.status_code == 403:
            raise PermissionDeniedError(f"the provider refused this API key: {msg}")
        if r.status_code == 404:
            raise APIConnectionError("the provider does not offer SDK enrollment (POST /pgw/v1/sdk/enroll); "
                                     "check the API key's provider URL", code="backend_unavailable")
        if r.status_code == 429:
            raise RateLimitError(f"too many enrollment attempts: {msg}")
        if r.status_code == 400:
            raise BadRequestError(f"enrollment refused: {msg}")
        raise InternalServerError(f"enrollment failed: {msg}")
    try:
        data = r.json()
        tenant, token, models = str(data["tenant"]), str(data["service_token"]), list(data["models"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InternalServerError("the provider sent an invalid enrollment response") from exc
    if not token or any(ch.isspace() for ch in token):
        raise InternalServerError("the provider sent an invalid service token")
    lane = data.get("lane") if isinstance(data.get("lane"), dict) else None
    if lane is not None:
        try:
            lane = {"transport": str(lane.get("transport", "tcp")), "port": int(lane["port"])}
            if not 0 < lane["port"] < 65536:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            lane = None
    p = _paths(d)
    ensure_private_dir(d / "secrets")
    write_private(p["token"], token + "\n", append=False)
    if ca_pem is not None:
        atomic_write_text(p["ca"], ca_pem, mode=0o644)
    else:
        p["ca"].unlink(missing_ok=True)
    cert_pem = data.get("client_cert_pem")
    has_cert = isinstance(cert_pem, str) and bool(cert_pem.strip())
    if has_cert:
        write_private(p["client_key"], key_pem.decode("ascii"), append=False)
        atomic_write_text(p["client_cert"], str(cert_pem), mode=0o644)
    else:
        for k in ("client_key", "client_cert"):
            p[k].unlink(missing_ok=True)
    clean_models = [{k: v for k, v in m.items() if k != "pack_path"} for m in models
                    if isinstance(m, dict) and m.get("name") and m.get("fingerprint")]
    meta = {"key_id": key.key_id_hex, "env": key.env, "service_url": key.service_url, "tenant": tenant,
            "client_name": name, "enrolled_at": time.time(),
            "ca_fingerprint": key.ca_fingerprint_hex if key.pinned else None, "client_cert": has_cert,
            "models": clean_models, "lane": lane}
    dkey = _directory_key(data.get("endpoint_directory_key"))
    if dkey is not None:
        meta["directory_key"] = dkey
    atomic_write_text(p["meta"], json.dumps(meta, indent=2) + "\n", mode=0o600)
    log.info("enrolled SDK key %s with %s as tenant %s (%d model(s))", key.key_id_hex, key.service_url, tenant,
             len(clean_models))
    s = load_session(key, CacheLayout(d.parent.parent))
    if s is None:  # pragma: no cover - just written
        raise ConfigError("enrollment files could not be read back")
    return s
