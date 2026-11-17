"""Tenant authentication for the compute service (bearer tokens + optional mTLS identity).

Bearer tokens
    ``tokens_file`` holds one tenant per line: ``tenant_id:<sha256 hex of the token>``. Blank
    lines and ``#`` comments are ignored. Only hashes are stored server side; presented tokens are
    hashed and compared in constant time against every entry (no early exit, no timing oracle on
    which tenant matched).

mTLS identity
    When the TLS layer verified a client certificate, its subject (or common name) can be mapped
    to a tenant with ``mtls_tenants_file`` (lines ``tenant_id:<subject or CN>``). The subject
    reaches the app through the ASGI TLS extension or, behind a TLS-terminating proxy, through a
    configured trusted header (see :func:`pgw.service.app.create_app`). A request with a mapped
    certificate does not need a token; if both are present they must name the same tenant.

Neither tokens nor their hashes are ever logged.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError

__all__ = ["TenantAuthenticator", "TenantIdentity", "hash_token", "parse_subject_cn"]

log = logging.getLogger(__name__)

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ANON = "anonymous"


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a token (the form stored in ``tokens_file``)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_subject_cn(subject: str) -> str | None:
    """Common name from an RFC 4514 / OpenSSL-style subject (``CN=gw1,O=Acme`` or ``/O=Acme/CN=gw1``)."""
    for part in re.split(r"[,/]", subject):
        k, _, v = part.strip().partition("=")
        if k.strip().upper() == "CN" and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class TenantIdentity:
    """Authenticated caller (used for quotas, billing counters and metrics labels)."""

    tenant_id: str
    via: str  # "token" | "mtls" | "anonymous"


def _read_map(path: Path, what: str, value_re: re.Pattern[str] | None) -> list[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {what} file {path}: {exc.strerror}", details={"path": str(path)}) from exc
    out: list[tuple[str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tenant, sep, value = line.partition(":")
        tenant, value = tenant.strip(), value.strip()
        if not sep or not _TENANT_RE.match(tenant) or not value:
            raise ConfigError(f"{what} file {path}: line {lineno} must be 'tenant_id:value'",
                              details={"path": str(path), "line": lineno})
        if value_re is not None:
            value = value.lower()
            if not value_re.match(value):
                raise ConfigError(f"{what} file {path}: line {lineno}: value must be a SHA-256 hex digest",
                                  details={"path": str(path), "line": lineno})
        out.append((tenant, value))
    return out


class TenantAuthenticator:
    """Verifies ``Authorization: Bearer <token>`` against SHA-256 hashes (constant-time compare)
    and/or the mTLS client certificate subject.

    Args:
        tokens_file: Token hash file (see module docstring), or ``None``.
        allow_unauthenticated: Dev only: accept requests without credentials as tenant
            ``anonymous`` (credentials that *are* presented must still be valid).
        mtls_tenants_file: Certificate subject -> tenant map, or ``None``.
        token_hashes: Extra ``{sha256_hex: tenant}`` entries (tests, programmatic setup).
        mtls_subjects: Extra ``{subject_or_cn: tenant}`` entries.

    Raises:
        ConfigError: unreadable/malformed files, or no way to authenticate anyone while
            unauthenticated access is disabled.
    """

    def __init__(
        self,
        tokens_file: Path | None,
        *,
        allow_unauthenticated: bool = False,
        mtls_tenants_file: Path | None = None,
        token_hashes: dict[str, str] | None = None,
        mtls_subjects: dict[str, str] | None = None,
    ):
        self.allow_unauthenticated = allow_unauthenticated
        self._tokens: list[tuple[bytes, str]] = []
        self._subjects: dict[str, str] = {}
        if tokens_file is not None:
            for tenant, h in _read_map(Path(tokens_file), "tokens", _HEX64_RE):
                self._tokens.append((bytes.fromhex(h), tenant))
        for h, tenant in (token_hashes or {}).items():
            if not _HEX64_RE.match(h.lower()) or not _TENANT_RE.match(tenant):
                raise ConfigError("token_hashes entries must be {sha256_hex: tenant_id}")
            self._tokens.append((bytes.fromhex(h.lower()), tenant))
        if mtls_tenants_file is not None:
            for tenant, subj in _read_map(Path(mtls_tenants_file), "mTLS tenants", None):
                self._subjects[subj] = tenant
        for subj, tenant in (mtls_subjects or {}).items():
            if not _TENANT_RE.match(tenant):
                raise ConfigError("mtls_subjects values must be tenant ids")
            self._subjects[subj] = tenant
        if not self._tokens and not self._subjects and not allow_unauthenticated:
            raise ConfigError(
                "no tenant credentials configured: set tokens_file and/or mtls_tenants_file "
                "(or allow_unauthenticated for development)"
            )
        if allow_unauthenticated:
            log.warning("compute service accepts UNAUTHENTICATED requests (development mode)")
        log.info("tenant auth: %d token(s), %d certificate subject(s)", len(self._tokens), len(self._subjects))

    @property
    def tenants(self) -> set[str]:
        """All configured tenant ids."""
        return {t for _, t in self._tokens} | set(self._subjects.values())

    def _match_token(self, token: str) -> str | None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        found: str | None = None
        for h, tenant in self._tokens:  # scan all entries: constant time w.r.t. which one matches
            if hmac.compare_digest(h, digest):
                found = tenant
        return found

    def _match_subject(self, subject: str) -> str | None:
        t = self._subjects.get(subject)
        if t is None:
            cn = parse_subject_cn(subject)
            if cn is not None:
                t = self._subjects.get(cn)
        return t

    def authenticate(self, authorization: str | None, client_cert_subject: str | None = None) -> TenantIdentity:
        """Return the identity or raise :class:`pgw.core.AuthError`.

        Raises:
            AuthError: missing/invalid credentials (HTTP 401).
            PGWError(permission_denied): token and certificate name different tenants, or the
                certificate is not mapped to any tenant (HTTP 403).
        """
        cert_tenant: str | None = None
        if client_cert_subject:
            cert_tenant = self._match_subject(client_cert_subject)
            if cert_tenant is None and self._subjects:
                raise PGWError("client certificate is not authorized", code=ErrorCode.PERMISSION_DENIED)
        token_tenant: str | None = None
        if authorization:
            scheme, _, token = authorization.strip().partition(" ")
            token = token.strip()
            if scheme.lower() != "bearer" or not token:
                raise AuthError("malformed Authorization header; expected 'Bearer <token>'")
            token_tenant = self._match_token(token)
            if token_tenant is None:
                raise AuthError("invalid bearer token")
        if token_tenant is not None and cert_tenant is not None and token_tenant != cert_tenant:
            raise PGWError("bearer token and client certificate belong to different tenants",
                           code=ErrorCode.PERMISSION_DENIED)
        if token_tenant is not None:
            return TenantIdentity(token_tenant, "token")
        if cert_tenant is not None:
            return TenantIdentity(cert_tenant, "mtls")
        if self.allow_unauthenticated:
            return TenantIdentity(_ANON, "anonymous")
        raise AuthError("missing credentials: send 'Authorization: Bearer <token>' or a client certificate")
