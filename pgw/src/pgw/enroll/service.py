"""Compute-service side of enrollment: ``GET /pgw/v1/enroll/ca`` and ``POST /pgw/v1/enroll``.

``GET /pgw/v1/enroll/ca``
    The service CA certificate (PEM, public). A new gateway fetches it once, checks its SHA-256
    against the fingerprint in the enrollment code and from then on verifies the service's TLS
    certificate against that CA only. Nothing secret is sent before that check.

``POST /pgw/v1/enroll``  (JSON ``{"secret": <hex>, "gateway_name": str, "csr_pem"?: str}``)
    Consumes a single-use code (constant-time match, atomic, not expired, not revoked), issues a
    fresh tenant bearer token (only its hash is stored, as with ``pgw service token create``) and,
    when an issuing CA is configured and a CSR is sent, signs a short-lived client certificate for
    the gateway's locally generated key. Rate-limited per client IP. Every attempt is audited
    (JSON lines) without secrets, tokens or certificates.

``POST /pgw/v1/enroll/renew``  (JSON ``{"csr_pem": str, "cert_pem": str}``)
    Renews an enrolled gateway's client certificate before it expires. Needs the tenant bearer token,
    the current client certificate presented in the TLS handshake (proof that the caller holds its key)
    and the same certificate, byte for byte (DER), in ``cert_pem``: issued by this service's client CA,
    not expired, its CN the TLS-verified one and belonging to the token's tenant. The new certificate keeps
    the CN. Audited as ``enroll.renewed`` with the old and new serials. An expired certificate is never accepted (the
    gateway then has to enroll again).

Failures of the code itself (unknown, used, revoked, expired) all return the same 401.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pgw.core.errors import ConfigError, ErrorCode, PGWError, RateLimitedError
from pgw.enroll.store import EnrollmentRecord, EnrollmentRejectedError, EnrollmentStore

__all__ = [
    "ENROLL_CA_PATH", "ENROLL_PATH", "ENROLL_RENEW_PATH", "EnrollmentService", "add_enroll_routes", "ca_fingerprint",
    "load_ca_pem", "sign_client_csr",
]

log = logging.getLogger(__name__)
T = TypeVar("T")

ENROLL_PATH = "/pgw/v1/enroll"
ENROLL_CA_PATH = "/pgw/v1/enroll/ca"
ENROLL_RENEW_PATH = "/pgw/v1/enroll/renew"
_MAX_BODY = 64 << 10
_GATEWAY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ---- certificates ------------------------------------------------------------------------------------
def load_ca_pem(path: Path) -> str:
    """PEM text of the first certificate in ``path`` (validated).

    Raises:
        ConfigError: unreadable file or no certificate in it.
    """
    from cryptography import x509

    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read CA certificate {path}: {exc.strerror}") from exc
    try:
        certs = x509.load_pem_x509_certificates(data)
    except ValueError as exc:
        raise ConfigError(f"{Path(path).name} does not contain a PEM certificate") from exc
    from cryptography.hazmat.primitives import serialization

    return certs[0].public_bytes(serialization.Encoding.PEM).decode("ascii")


def ca_fingerprint(pem: str | bytes) -> bytes:
    """SHA-256 of the DER encoding of the first certificate in ``pem`` (what enrollment codes pin)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    data = pem.encode("ascii") if isinstance(pem, str) else pem
    try:
        cert = x509.load_pem_x509_certificates(data)[0]
    except (ValueError, IndexError) as exc:
        raise ConfigError("not a PEM certificate") from exc
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).digest()


def sign_client_csr(csr_pem: str, ca_cert_file: Path, ca_key_file: Path, common_name: str,
                    validity_days: float) -> tuple[str, int]:
    """Sign a gateway client certificate for the CSR's public key.

    The CSR's subject and extensions are ignored: the certificate gets ``CN=common_name``, clientAuth
    only, no CA rights and ``validity_days`` of lifetime. The CSR signature must verify (proof that
    the caller holds the private key).

    Returns:
        ``(certificate PEM, serial number)``.

    Raises:
        PGWError(protocol): malformed CSR, bad signature or a weak key.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    pub = check_csr(csr_pem).public_key()
    ca_cert = x509.load_pem_x509_certificate(Path(ca_cert_file).read_bytes())
    ca_key = serialization.load_pem_private_key(Path(ca_key_file).read_bytes(), password=None)
    now = dt.datetime.now(dt.UTC)
    serial = x509.random_serial_number()
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(pub)
        .serial_number(serial)
        # backdated for clock skew: 5 minutes, or a tenth of a short lifetime (so 2/3-lifetime renewal still works)
        .not_valid_before(now - min(dt.timedelta(minutes=5), dt.timedelta(days=validity_days) / 10))
        .not_valid_after(now + dt.timedelta(days=validity_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=isinstance(pub, rsa.RSAPublicKey),
                                     content_commitment=False, data_encipherment=False, key_agreement=False,
                                     key_cert_sign=False, crl_sign=False, encipher_only=False,
                                     decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(pub), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),  # type: ignore[arg-type]
                       critical=False)
        .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode("ascii"), serial


def check_csr(csr_pem: str) -> Any:
    """Parse a CSR and check its signature and key strength; returns the CSR.

    Raises:
        PGWError(protocol): malformed CSR, bad signature or a weak/unsupported key.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    try:
        csr = x509.load_pem_x509_csr(csr_pem.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise PGWError("csr_pem is not a PEM certificate signing request", code=ErrorCode.PROTOCOL) from exc
    if not csr.is_signature_valid:
        raise PGWError("the CSR signature does not verify", code=ErrorCode.PROTOCOL)
    pub = csr.public_key()
    if isinstance(pub, rsa.RSAPublicKey):
        if pub.key_size < 2048:
            raise PGWError("RSA keys must have at least 2048 bits", code=ErrorCode.PROTOCOL)
    elif isinstance(pub, ec.EllipticCurvePublicKey):
        if pub.curve.name not in ("secp256r1", "secp384r1"):
            raise PGWError("EC keys must use P-256 or P-384", code=ErrorCode.PROTOCOL)
    else:
        raise PGWError("unsupported CSR key type (use EC P-256/P-384 or RSA >= 2048)", code=ErrorCode.PROTOCOL)
    return csr


# ---- rate limiting and audit -------------------------------------------------------------------------
class _IPRateLimiter:
    """Failed attempts per client address per minute, plus a global ceiling on failures from all addresses
    (so rotating addresses does not buy unlimited code guesses). Successful calls are not counted.

    Admission is atomic: :meth:`acquire` counts the attempt as in flight (against both limits) until
    :meth:`release` settles it, as a failure or not. So N concurrent attempts cannot all pass the check
    before the first of their failures is recorded.
    """

    def __init__(self, per_minute: int, *, global_per_minute: int = 300, max_clients: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.global_per_minute = global_per_minute
        self.max_clients = max_clients
        self.clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._global: deque[float] = deque()
        self._inflight: dict[str, int] = {}
        self._global_inflight = 0
        self._lock = threading.Lock()

    def _queue(self, client: str, now: float) -> deque[float]:
        q = self._hits.pop(client, None) or deque()
        while q and q[0] <= now - 60.0:
            q.popleft()
        self._hits[client] = q
        while len(self._hits) > self.max_clients:
            self._hits.popitem(last=False)
        return q

    def acquire(self, client: str) -> None:
        """Reserve one attempt by ``client``; raise :class:`RateLimitedError` if its recent failures plus its
        attempts in flight (or everyone's together) reach the limit. Pair with :meth:`release`."""
        now = self.clock()
        with self._lock:
            g = self._global
            while g and g[0] <= now - 60.0:
                g.popleft()
            q = self._queue(client, now)
            mine = self._inflight.get(client, 0)
            for queue, busy, limit in ((g, self._global_inflight, self.global_per_minute),
                                       (q, mine, self.per_minute)):
                if len(queue) + busy >= limit:
                    retry = queue[0] + 60.0 - now if queue else 1.0
                    raise RateLimitedError("too many failed enrollment attempts; try again later",
                                           details={"retry_after_s": max(1.0, retry), "limit": "enroll"})
            self._inflight[client] = mine + 1
            self._global_inflight += 1

    def release(self, client: str, *, failed: bool) -> None:
        """Settle an attempt reserved with :meth:`acquire`; ``failed`` counts it as a failure."""
        now = self.clock()
        with self._lock:
            n = self._inflight.get(client, 0) - 1
            if n > 0:
                self._inflight[client] = n
            else:
                self._inflight.pop(client, None)
            self._global_inflight = max(0, self._global_inflight - 1)
            if failed:
                self._queue(client, now).append(now)
                self._global.append(now)
                while len(self._global) > self.global_per_minute:
                    self._global.popleft()

    def guarded(self, client: str, fn: Callable[[], T]) -> T:
        """Run ``fn`` as one reserved attempt; a :class:`PGWError` with a failure code counts as a failure."""
        self.acquire(client)
        failed = False
        try:
            return fn()
        except PGWError as exc:
            failed = exc.code in _FAILURE_CODES
            raise
        finally:
            self.release(client, failed=failed)


#: Errors that count as a failed attempt for the limiter (a refused code or credential, a malformed body);
#: not "service busy" or internal errors, which are not the caller's doing.
_FAILURE_CODES = frozenset({ErrorCode.UNAUTHENTICATED, ErrorCode.PERMISSION_DENIED, ErrorCode.PROTOCOL})


class _Audit:
    """JSON-lines audit of enrollment attempts. Fields are fixed; no secret ever reaches them."""

    FIELDS = ("ts", "event", "enrollment_id", "tenant", "gateway_name", "client", "reason", "cert_serial",
              "prev_cert_serial", "token_id", "models")

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()

    def event(self, event: str, **kw: Any) -> None:
        rec = {"ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"), "event": event}
        rec.update({k: v for k, v in kw.items() if k in self.FIELDS and v is not None})
        log.info("enrollment %s %s", event, json.dumps({k: v for k, v in rec.items() if k not in ("ts", "event")},
                                                       sort_keys=True))
        if self.path is None:
            return
        from pgw.core.fileio import write_private

        with self._lock:
            try:
                write_private(self.path, json.dumps(rec, sort_keys=True) + "\n", append=True)
            except PGWError as exc:  # never fail an enrollment because the audit disk is full: log loudly
                log.error("cannot write enrollment audit record: %s", exc.message)


# ---- the service -------------------------------------------------------------------------------------
class EnrollmentService:
    """Enrollment logic, independent of the web framework (tested directly and via the routes).

    Args:
        settings: :class:`pgw.service.config.ServiceSettings` with ``enrollment_file`` set.
        authenticator: The running service's :class:`pgw.service.auth.TenantAuthenticator` (new tokens
            and certificate subjects are added to it, so they work without a restart).
        models: ``() -> [{"name", "fingerprint", "rings", "pack_path"}]`` of the served models.
    """

    def __init__(self, settings: Any, authenticator: Any, models: Callable[[], list[dict[str, Any]]]):
        assert settings.tokens_file is not None
        self.settings = settings
        self.authenticator = authenticator
        self.models = models
        # Without enrollment_file only the credential issuing is used (SDK keys, pgw.service.accounts).
        self.store = EnrollmentStore(settings.enrollment_file) if settings.enrollment_file is not None else None
        self.limiter = _IPRateLimiter(settings.enrollment_rate_limit_per_minute)
        audit_path = settings.enrollment_audit_file or (
            Path(settings.enrollment_file).with_suffix(".audit.jsonl") if settings.enrollment_file is not None
            else None)
        self.audit = _Audit(audit_path)
        self.ca_pem: str | None = load_ca_pem(settings.enrollment_ca_file) if settings.enrollment_ca_file else None
        self._lock = threading.Lock()
        self._manifest_cache: dict[str, dict[str, Any]] = {}

    def _model_entries(self, allowed: list[str]) -> list[dict[str, Any]]:
        out = []
        for m in self.models():
            if allowed and m["name"] not in allowed:
                continue
            fp = m["fingerprint"]
            meta = self._manifest_cache.get(fp)
            if meta is None:
                meta = _manifest_meta(m.get("pack_path"))
                self._manifest_cache[fp] = meta
            rings = list(m.get("rings") or [])
            out.append({"name": m["name"], "fingerprint": fp, "ring": rings[0] if rings else None, "rings": rings,
                        **meta})
        return out

    def enroll(self, body: Any, client: str) -> dict[str, Any]:
        """Handle one enrollment request (blocking: run it in a worker thread).

        Raises:
            RateLimitedError, PGWError(protocol), EnrollmentRejectedError, PGWError(unavailable).
        """
        return self.limiter.guarded(client, lambda: self._enroll(body, client))

    def _enroll(self, body: Any, client: str) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise PGWError("body must be a JSON object", code=ErrorCode.PROTOCOL)
        secret_hex, name, csr = body.get("secret"), body.get("gateway_name"), body.get("csr_pem")
        if not isinstance(secret_hex, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", secret_hex):
            self.audit.event("enroll.failed", client=client, reason="malformed_secret")
            raise EnrollmentRejectedError("malformed_secret")
        if not isinstance(name, str) or not _GATEWAY_NAME_RE.match(name):
            raise PGWError("gateway_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}", code=ErrorCode.PROTOCOL)
        if csr is not None and (not isinstance(csr, str) or len(csr) > 16 << 10):
            raise PGWError("csr_pem must be a PEM string", code=ErrorCode.PROTOCOL)
        s = self.settings
        sign = isinstance(csr, str) and bool(csr) and s.client_issuing_ca_cert_file is not None
        if sign:
            check_csr(str(csr))  # before the code is consumed: a bad CSR must not burn it

        def check(rec: EnrollmentRecord) -> list[dict[str, Any]]:
            models = self._model_entries(rec.models)
            if not models:
                raise PGWError("none of the models of this enrollment code is served right now; try again later "
                               "or ask your provider", code=ErrorCode.UNAVAILABLE)
            return models

        if self.store is None:
            raise PGWError("this service does not accept enrollment codes", code=ErrorCode.UNAVAILABLE)
        with self._lock:
            try:
                rec, models = self.store.consume(bytes.fromhex(secret_hex), name, check)
            except EnrollmentRejectedError as exc:
                r = exc.record
                self.audit.event("enroll.failed", client=client, gateway_name=name, reason=exc.reason,
                                 enrollment_id=r.id if r else None, tenant=r.tenant if r else None)
                raise
            except PGWError as exc:
                self.audit.event("enroll.failed", client=client, gateway_name=name, reason=str(exc.code))
                raise
            out, trec, serial, _cn = self.issue_credentials(rec.tenant, name, str(csr) if sign else None)
            out["models"] = models
        self.audit.event("enroll.ok", client=client, gateway_name=name, enrollment_id=rec.id, tenant=rec.tenant,
                         token_id=trec.token_id,
                         cert_serial=f"{serial:x}" if serial is not None else None,
                         models=",".join(m["name"] for m in models))
        return out


    def issue_credentials(self, tenant: str, name: str, csr: str | None,
                          ) -> tuple[dict[str, Any], Any, int | None, str | None]:
        """Issue a tenant service token and, for a (checked) ``csr`` with an issuing CA configured, a client
        certificate ``CN=pgw-gw.<tenant>.<name>``; register both with the running authenticator (and the token
        hash in ``tokens_file``, the CN in ``mtls_tenants_file``). Shared by code enrollment and SDK keys.

        Returns:
            ``(reply without "models", token record, certificate serial or None, CN or None)``.
        """
        s = self.settings
        sign = csr is not None and s.client_issuing_ca_cert_file is not None
        # Without a client certificate the token is marked token-only: it stays usable when the
        # service requires mTLS for compute (require_client_cert_for_compute).
        token, trec = _issue_token(tenant, Path(s.tokens_file), token_only=not sign)
        if self.authenticator is not None:
            self.authenticator.add_token_hash(trec.digest_hex, tenant, token_id=trec.token_id,
                                              token_only=trec.token_only)
        out: dict[str, Any] = {
            "tenant": tenant, "service_token": token, "ca_bundle_pem": self.ca_pem, "service_url": s.public_url,
        }
        serial: int | None = None
        cn: str | None = None
        if sign:
            cn = f"pgw-gw.{tenant}.{name}"[:64]
            cert_pem, serial = sign_client_csr(str(csr), s.client_issuing_ca_cert_file,
                                               s.client_issuing_ca_key_file, cn, s.client_cert_validity_days)
            out["client_cert_pem"] = cert_pem
            out["client_cert_cn"] = cn
            if s.mtls_tenants_file is not None:
                from pgw.core.fileio import write_private

                write_private(Path(s.mtls_tenants_file), f"{tenant}:{cn}\n", append=True)
            if self.authenticator is not None:
                self.authenticator.add_subject(cn, tenant)
        return out, trec, serial, cn


    def renew(self, body: Any, client: str, authorization: str | None, cert_subject: str | None,
              cert_der: bytes | None = None) -> dict[str, Any]:
        """Renew a gateway client certificate (blocking: run it in a worker thread). See the module docstring.

        Args:
            cert_subject: Subject of the verified client certificate of the TLS handshake.
            cert_der: DER of that certificate (``cert_pem`` in the body must be byte-identical to it).

        Raises:
            RateLimitedError, AuthError (401), PGWError(permission_denied / protocol / unavailable).
        """
        return self.limiter.guarded(client, lambda: self._renew(body, client, authorization, cert_subject, cert_der))

    def _renew(self, body: Any, client: str, authorization: str | None, cert_subject: str | None,
               cert_der: bytes | None) -> dict[str, Any]:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.x509.oid import NameOID

        from pgw.core.errors import AuthError
        from pgw.service.auth import parse_subject_cn

        s = self.settings
        if s.client_issuing_ca_cert_file is None or s.client_issuing_ca_key_file is None:
            raise PGWError("this service does not issue client certificates", code=ErrorCode.UNAVAILABLE)

        def deny(reason: str, msg: str, tenant: str | None = None, code: ErrorCode = ErrorCode.PERMISSION_DENIED,
                 ) -> PGWError:
            self.audit.event("enroll.renew_failed", client=client, reason=reason, tenant=tenant)
            return PGWError(msg, code=code)

        if self.authenticator is None or not authorization:
            self.audit.event("enroll.renew_failed", client=client, reason="no_token")
            raise AuthError("certificate renewal needs the tenant bearer token")
        if not cert_subject:
            raise deny("no_client_cert", "certificate renewal needs the current client certificate in the TLS "
                       "handshake; an expired certificate cannot be renewed (enroll the gateway again)")
        try:
            ident = self.authenticator.authenticate(authorization, cert_subject)
        except PGWError as exc:
            self.audit.event("enroll.renew_failed", client=client, reason=str(exc.code))
            raise
        tenant = ident.tenant_id
        token_id = getattr(ident, "token_id", None)
        if not isinstance(body, dict):
            raise PGWError("body must be a JSON object", code=ErrorCode.PROTOCOL)
        csr, cert_pem = body.get("csr_pem"), body.get("cert_pem")
        for v, what in ((csr, "csr_pem"), (cert_pem, "cert_pem")):
            if not isinstance(v, str) or not v or len(v) > 16 << 10:
                raise PGWError(f"{what} must be a PEM string", code=ErrorCode.PROTOCOL)
        check_csr(str(csr))
        try:
            cur = x509.load_pem_x509_certificate(str(cert_pem).encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise PGWError("cert_pem is not a PEM certificate", code=ErrorCode.PROTOCOL) from exc
        # The certificate renewed must be exactly the one presented in the TLS handshake (whose key the
        # caller proved it holds), not merely one with the same subject.
        if not cert_der:
            raise deny("no_client_cert_der", "certificate renewal needs the client certificate itself from the TLS "
                       "handshake (native mTLS, or trusted_client_cert_pem_header behind a TLS-terminating proxy)",
                       tenant)
        if not hmac.compare_digest(cur.public_bytes(serialization.Encoding.DER), bytes(cert_der)):
            raise deny("cert_mismatch", "the certificate does not match the TLS client certificate or the tenant",
                       tenant)
        ca_cert = x509.load_pem_x509_certificate(Path(s.client_issuing_ca_cert_file).read_bytes())
        try:
            cur.verify_directly_issued_by(ca_cert)
        except Exception as exc:
            raise deny("foreign_cert", "the certificate was not issued by this service", tenant) from exc
        now = dt.datetime.now(dt.UTC)
        if not cur.not_valid_before_utc <= now < cur.not_valid_after_utc:
            raise deny("expired_cert", "the client certificate has expired (or is not yet valid); enroll the "
                       "gateway again", tenant)
        cns = cur.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        cn = str(cns[0].value) if cns else ""
        mapped = getattr(self.authenticator, "tenant_for_subject", lambda _s: None)(cert_subject)
        if (not cn or cn != parse_subject_cn(cert_subject) or not cn.startswith(f"pgw-gw.{tenant}.")
                or mapped != tenant):
            raise deny("cert_mismatch", "the certificate does not match the TLS client certificate or the tenant",
                       tenant)
        new_pem, serial = sign_client_csr(str(csr), s.client_issuing_ca_cert_file, s.client_issuing_ca_key_file,
                                          cn, s.client_cert_validity_days)
        self.audit.event("enroll.renewed", client=client, tenant=tenant, gateway_name=cn, token_id=token_id,
                         cert_serial=f"{serial:x}", prev_cert_serial=f"{cur.serial_number:x}")
        return {"client_cert_pem": new_pem, "client_cert_cn": cn}


def _issue_token(tenant: str, tokens_file: Path, *, token_only: bool) -> tuple[str, Any]:
    from pgw.service.auth import issue_token_record

    return issue_token_record(tenant, tokens_file, token_only=token_only)


def _manifest_meta(pack_path: Any) -> dict[str, Any]:
    """Pack-build parameters a gateway needs to rebuild the identical pack from the open weights."""
    meta: dict[str, Any] = {"hf_repo": None, "revision": None, "license": None, "plain_dtype": "float32"}
    if pack_path is None:
        return meta
    try:
        man = json.loads((Path(pack_path) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return meta
    src = man.get("source") or {}
    meta.update(hf_repo=src.get("repo"), revision=src.get("revision"), license=src.get("license"),
                plain_dtype=(man.get("quant") or {}).get("plain_dtype", "float32"))
    if man.get("name"):
        meta["pack_name"] = man["name"]
    return meta


def add_enroll_routes(app: Any, settings: Any, authenticator: Any, models: Callable[[], list[dict[str, Any]]]) -> Any:
    """Register the enrollment routes on the service's FastAPI app; returns the :class:`EnrollmentService`."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, Response

    from pgw.core.client_ip import client_ip

    svc = EnrollmentService(settings, authenticator, models)

    def _err(exc: PGWError) -> Response:
        headers = {"Cache-Control": "no-store"}
        retry = exc.details.get("retry_after_s")
        if isinstance(retry, int | float):
            headers["Retry-After"] = str(max(1, round(retry)))
        msg = "internal error" if exc.code == ErrorCode.INTERNAL else exc.message
        return JSONResponse({"error": {"code": str(exc.code), "message": msg}}, status_code=exc.http_status,
                            headers=headers)

    async def get_ca(request: Request) -> Response:
        if svc.ca_pem is None:
            return JSONResponse({"error": {"code": str(ErrorCode.CONFIG),
                                           "message": "this service has no enrollment CA configured"}},
                                status_code=404)
        return PlainTextResponse(svc.ca_pem, media_type="application/x-pem-file")

    async def post_enroll(request: Request) -> Response:
        client = client_ip(request)
        try:
            length = request.headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > _MAX_BODY):
                raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
            raw = await request.body()
            if len(raw) > _MAX_BODY:
                raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise PGWError("body must be JSON", code=ErrorCode.PROTOCOL) from exc
            out = await asyncio.to_thread(svc.enroll, body, client)
        except PGWError as exc:
            return _err(exc)
        except Exception as exc:  # never leak details
            log.error("enrollment failed: %s", type(exc).__name__)
            return _err(PGWError("internal error"))
        return JSONResponse(out, headers={"Cache-Control": "no-store"})

    async def post_renew(request: Request) -> Response:
        from pgw.service.tls_peer import peer_cert_der
        from pgw.wire.server import client_cert_subject

        client = client_ip(request)
        try:
            length = request.headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > _MAX_BODY):
                raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
            raw = await request.body()
            if len(raw) > _MAX_BODY:
                raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
            try:
                body = json.loads(raw)
            except ValueError as exc:
                raise PGWError("body must be JSON", code=ErrorCode.PROTOCOL) from exc
            out = await asyncio.to_thread(svc.renew, body, client, request.headers.get("authorization"),
                                          client_cert_subject(request.scope), peer_cert_der(request.scope))
        except PGWError as exc:
            return _err(exc)
        except Exception as exc:  # never leak details
            log.error("certificate renewal failed: %s", type(exc).__name__)
            return _err(PGWError("internal error"))
        return JSONResponse(out, headers={"Cache-Control": "no-store"})

    app.add_route(ENROLL_CA_PATH, get_ca, methods=["GET"], include_in_schema=False)
    app.add_route(ENROLL_RENEW_PATH, post_renew, methods=["POST"], include_in_schema=False)
    if svc.store is not None:  # codes need enrollment_file; SDK keys (pgw.service.accounts) only use the CA/renew
        app.add_route(ENROLL_PATH, post_enroll, methods=["POST"], include_in_schema=False)
    app.state.pgw_enroll = svc
    return svc
