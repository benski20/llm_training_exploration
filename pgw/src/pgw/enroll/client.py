"""Gateway side of enrollment (``pgw gateway enroll CODE --config-dir DIR`` and the setup wizard).

Flow:

1. Parse the code locally (checksum: typos are caught before any network call).
2. ``GET /pgw/v1/enroll/ca`` and compare the SHA-256 of the returned CA certificate with the
   fingerprint in the code. On a mismatch, stop: someone may be intercepting the connection. Nothing
   secret has been sent at that point.
3. Generate the gateway's client key locally (it never leaves the gateway) and a CSR.
4. ``POST /pgw/v1/enroll`` over TLS verified against the pinned CA *only* (no system trust store).
5. Write ``secrets/service_token`` (0600), ``service-ca.pem``, ``secrets/client.key`` (0600) and
   ``client.crt`` (when the service signed one), ``models.json`` and ``enrollment.json`` (metadata).
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, ErrorCode, PGWError
from pgw.core.fileio import atomic_write_text
from pgw.enroll.code import EnrollmentCode

__all__ = ["ENROLLMENT_FILES", "EnrollResult", "EnrollmentError", "enroll_gateway", "enrollment_paths"]

log = logging.getLogger(__name__)

#: Files written under the config directory (relative paths).
ENROLLMENT_FILES = {
    "token": "secrets/service_token",
    "ca": "service-ca.pem",
    "client_key": "secrets/client.key",
    "client_cert": "client.crt",
    "models": "models.json",
    "meta": "enrollment.json",
}


class EnrollmentError(PGWError):
    """Enrollment failed; the message is written for the gateway's administrator."""


@dataclass
class EnrollResult:
    """Outcome of a successful enrollment (no secrets)."""

    tenant: str
    service_url: str
    models: list[dict[str, Any]]
    files: dict[str, Path] = field(default_factory=dict)
    client_cert: bool = False
    ca_fingerprint: str | None = None

    def public(self) -> dict[str, Any]:
        """JSON view for the console."""
        return {"tenant": self.tenant, "service_url": self.service_url, "models": self.models,
                "client_cert": self.client_cert, "ca_fingerprint": self.ca_fingerprint,
                "files": {k: str(v) for k, v in self.files.items()}}


def enrollment_paths(config_dir: Path) -> dict[str, Path]:
    """Absolute paths of the files enrollment writes."""
    return {k: Path(config_dir) / v for k, v in ENROLLMENT_FILES.items()}


def _default_name() -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname() or "gateway").strip("-.") or "gateway"
    return name[:64]


def _client(code: EnrollmentCode, verify: Any, transport: Any, timeout: float) -> Any:
    import httpx

    # trust_env: honour HTTPS_PROXY (CONNECT keeps TLS end to end, so the pin still holds)
    kw: dict[str, Any] = {"timeout": timeout, "follow_redirects": False, "trust_env": True}
    if transport is not None:
        kw["transport"] = transport
    else:
        kw["verify"] = verify
    return httpx.Client(base_url=code.service_url, **kw)


def _fetch_pinned_ca(code: EnrollmentCode, transport: Any, timeout: float) -> str | None:
    """The service CA (PEM) whose SHA-256 equals the code's pin; ``None`` for an unpinned loopback code."""
    import httpx

    from pgw.enroll.service import ca_fingerprint

    # Unverified on purpose: the certificate is checked against the pin below before any secret is sent.
    try:
        with _client(code, False, transport, timeout) as c:
            r = c.get("/pgw/v1/enroll/ca")
    except httpx.HTTPError as exc:
        raise EnrollmentError(f"cannot reach the compute service at {code.service_url} "
                              f"({type(exc).__name__}). Check the network and firewall (outbound HTTPS).",
                              code=ErrorCode.BACKEND_UNAVAILABLE) from exc
    if r.status_code == 404 and not code.pinned:
        return None
    if r.status_code != 200:
        raise EnrollmentError(f"the compute service does not offer enrollment (HTTP {r.status_code}); ask your "
                              "provider to enable it", code=ErrorCode.BACKEND_UNAVAILABLE)
    pem = r.text
    try:
        fp = ca_fingerprint(pem)
    except ConfigError as exc:
        raise EnrollmentError("the compute service sent an invalid CA certificate", code=ErrorCode.INTEGRITY) from exc
    if code.pinned and not hmac.compare_digest(fp, code.ca_sha256):
        raise EnrollmentError(
            "the compute service's certificate authority does not match the enrollment code. The connection may "
            "be intercepted (for example by a TLS-inspecting proxy). Enrollment stopped; nothing was sent. "
            f"Expected CA fingerprint {code.ca_fingerprint_hex[:16]}..., got {fp.hex()[:16]}...",
            code=ErrorCode.INTEGRITY)
    return str(pem)


def _key_and_csr(gateway_name: str) -> tuple[bytes, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, gateway_name)]))
           .sign(key, hashes.SHA256()))
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    return key_pem, csr.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _check_writable(config_dir: Path) -> None:
    try:
        (config_dir / "secrets").mkdir(parents=True, exist_ok=True, mode=0o700)
        probe = config_dir / f".write-test-{time.time_ns()}"
        probe.write_text("")
        probe.unlink()
    except OSError as exc:
        raise ConfigError(f"cannot write to the configuration directory {config_dir}: {exc.strerror}") from exc


def enroll_gateway(
    code: str | EnrollmentCode,
    config_dir: Path,
    *,
    gateway_name: str | None = None,
    force: bool = False,
    packs_dir: Path | None = None,
    request_client_cert: bool = True,
    transport: Any = None,
    timeout_s: float = 30.0,
    progress: Callable[[str], None] | None = None,
) -> EnrollResult:
    """Enroll this gateway with the compute service named in ``code``.

    Args:
        code: The enrollment code (string as pasted, or parsed).
        config_dir: Where to write the token, CA, client key/cert, ``models.json`` and ``enrollment.json``.
        gateway_name: Name recorded by the service (default: host name).
        force: Overwrite files of an earlier enrollment.
        packs_dir: Directory for model packs referenced from ``models.json`` (default ``<config_dir>/packs``).
        request_client_cert: Send a CSR so the service can issue an mTLS client certificate.
        transport: httpx transport override (tests).
        timeout_s: Per-request timeout.
        progress: Optional callback for plain-language progress lines.

    Raises:
        EnrollmentCodeError: the code is malformed or mistyped (before any network call).
        EnrollmentError: service unreachable, CA pin mismatch, code refused.
        ConfigError: files exist (without ``force``) or the directory is not writable.
    """
    import httpx

    say = progress or (lambda msg: None)
    c = code if isinstance(code, EnrollmentCode) else EnrollmentCode.parse(code)
    config_dir = Path(config_dir)
    paths = enrollment_paths(config_dir)
    existing = [p for k, p in paths.items() if p.exists()]
    if existing and not force:
        raise ConfigError(f"this gateway is already enrolled ({existing[0]} exists); pass --force to replace "
                          "the earlier enrollment")
    _check_writable(config_dir)
    name = gateway_name or _default_name()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ConfigError("gateway name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")

    say(f"Contacting {c.service_url} and checking its certificate authority")
    ca_pem = _fetch_pinned_ca(c, transport, timeout_s)
    verify: Any = False
    if c.service_url.startswith("https://"):
        assert ca_pem is not None
        verify = ssl.create_default_context(cadata=ca_pem)  # the pinned CA only, no system trust store
    key_pem: bytes | None = None
    body: dict[str, Any] = {"secret": c.secret_hex, "gateway_name": name}
    if request_client_cert:
        key_pem, body["csr_pem"] = _key_and_csr(name)
    say("Presenting the enrollment code")
    try:
        with _client(c, verify, transport, timeout_s) as h:
            r = h.post("/pgw/v1/enroll", json=body)
    except httpx.HTTPError as exc:
        raise EnrollmentError(f"enrollment request failed ({type(exc).__name__}); nothing was changed",
                              code=ErrorCode.BACKEND_UNAVAILABLE) from exc
    if r.status_code == 401:
        raise EnrollmentError("the compute service did not accept this enrollment code: it is unknown, already "
                              "used, revoked or expired. Ask your provider for a new code.",
                              code=ErrorCode.UNAUTHENTICATED)
    if r.status_code == 429:
        raise EnrollmentError("too many enrollment attempts; wait a minute and try again",
                              code=ErrorCode.RATE_LIMITED)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = f"HTTP {r.status_code}"
        raise EnrollmentError(f"enrollment failed: {msg}", code=ErrorCode.BACKEND_UNAVAILABLE)
    try:
        data = r.json()
        tenant, token, models = str(data["tenant"]), str(data["service_token"]), list(data["models"])
    except (ValueError, KeyError, TypeError) as exc:
        raise EnrollmentError("the compute service sent an invalid enrollment response",
                              code=ErrorCode.PROTOCOL) from exc
    if not token or any(ch.isspace() for ch in token):
        raise EnrollmentError("the compute service sent an invalid token", code=ErrorCode.PROTOCOL)

    say("Saving the credentials")
    from pgw.core.fileio import write_private

    write_private(paths["token"], token + "\n", append=False)
    files = {"token": paths["token"]}
    if ca_pem is not None:
        atomic_write_text(paths["ca"], ca_pem, mode=0o644)
        files["ca"] = paths["ca"]
    cert_pem = data.get("client_cert_pem")
    if key_pem is not None and isinstance(cert_pem, str) and cert_pem.strip():
        write_private(paths["client_key"], key_pem.decode("ascii"), append=False)
        atomic_write_text(paths["client_cert"], cert_pem, mode=0o644)
        files["client_key"], files["client_cert"] = paths["client_key"], paths["client_cert"]
    else:
        for k in ("client_key", "client_cert"):  # never leave a stale pair from an earlier enrollment
            if paths[k].exists():
                paths[k].unlink()

    packs = Path(packs_dir) if packs_dir is not None else config_dir / "packs"
    entries = [model_entry(m, packs / _safe_dirname(str(m["name"])), c.service_url, files) for m in models
               if isinstance(m, dict) and m.get("name") and m.get("fingerprint")]
    atomic_write_text(paths["models"], json.dumps(entries, indent=2) + "\n", mode=0o600)
    files["models"] = paths["models"]
    meta: dict[str, Any] = {"tenant": tenant, "service_url": c.service_url, "gateway_name": name,
                            "enrolled_at": time.time(),
            "ca_fingerprint": c.ca_fingerprint_hex if c.pinned else None, "models": models}
    atomic_write_text(paths["meta"], json.dumps(meta, indent=2) + "\n", mode=0o644)
    files["meta"] = paths["meta"]
    log.info("enrolled as tenant %s with %s (%d model(s))", tenant, c.service_url, len(entries))
    return EnrollResult(tenant=tenant, service_url=c.service_url, models=models, files=files,
                        client_cert="client_cert" in files, ca_fingerprint=meta["ca_fingerprint"])


def _safe_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".") or "model"


def model_entry(m: dict[str, Any], pack_path: Path, service_url: str, files: dict[str, Path]) -> dict[str, Any]:
    """A ``models.json`` entry (masked mode, pinned fingerprint, enrolled credentials), validated."""
    from pgw.gateway.config import GatewayModelConfig

    e: dict[str, Any] = {"name": m["name"], "pack_path": str(pack_path), "fingerprint": m["fingerprint"],
                         "mode": "masked", "service_url": service_url,
                         "service_token_file": str(files["token"])}
    if m.get("ring"):
        e["ring"] = m["ring"]
    if "ca" in files:
        e["service_ca_file"] = str(files["ca"])
    if "client_cert" in files:
        e["client_cert_file"], e["client_key_file"] = str(files["client_cert"]), str(files["client_key"])
    GatewayModelConfig(**e)  # raises on anything invalid
    return e
