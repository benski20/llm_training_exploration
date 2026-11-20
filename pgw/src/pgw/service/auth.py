"""Tenant authentication for the compute service (bearer tokens + optional mTLS identity).

Bearer tokens
    ``tokens_file`` holds one token per line: ``tenant_id:token_id:<sha256 hex of the token>`` with an
    optional ``:not_after`` (unix seconds, empty for none) and an optional ``:flags`` field
    (``token-only``: the token was issued without a client certificate, see below). The older form
    ``tenant_id:<sha256 hex>`` still loads (its token id is ``legacy-`` plus the first 8 hex digits of
    the hash). Blank lines and ``#`` comments are ignored. Only hashes are stored server side;
    presented tokens are hashed and compared in constant time against every entry (no early exit, no
    timing oracle on which tenant matched).

    The file is re-read when its modification time or size changes (checked at most every
    ``reload_interval_s``), rebuilding the whole set: a revoked token (``pgw service token revoke ID``)
    stops working within one interval, and the compute channels it opened are closed (see
    :meth:`TenantIdentity.still_valid`).

mTLS identity
    When the TLS layer verified a client certificate, its subject (or common name) can be mapped
    to a tenant with ``mtls_tenants_file`` (lines ``tenant_id:<subject or CN>``). The subject
    reaches the app through the ASGI TLS extension or, behind a TLS-terminating proxy, through a
    configured trusted header (see :func:`pgw.service.app.create_app`). A request with a mapped
    certificate does not need a token; if both are present they must name the same tenant.

    With ``require_client_cert`` (``ServiceSettings.require_client_cert_for_compute``) a token alone is
    refused (403) unless it is marked ``token-only``: compute needs a verified, mapped certificate.

Neither tokens nor their hashes are ever logged; audit records carry the token id only.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError

__all__ = [
    "TOKEN_ONLY_FLAG", "TenantAuthenticator", "TenantIdentity", "TokenRecord", "hash_token", "issue_tenant_token",
    "issue_token_record", "new_token", "parse_subject_cn", "read_token_records", "revoke_token", "valid_tenant_id",
]

log = logging.getLogger(__name__)

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ANON = "anonymous"
TOKEN_ONLY_FLAG = "token-only"  # noqa: S105 - a flag name, not a secret


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a token (the form stored in ``tokens_file``)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def valid_tenant_id(tenant: str) -> bool:
    """Whether ``tenant`` is a valid tenant id (``[A-Za-z0-9][A-Za-z0-9._-]{0,63}``)."""
    return bool(_TENANT_RE.fullmatch(tenant))


@dataclass(frozen=True)
class TokenRecord:
    """One line of the tokens file (no secret: the token's SHA-256 only)."""

    tenant: str
    token_id: str
    digest_hex: str
    not_after: float | None = None
    token_only: bool = False
    legacy: bool = False

    def line(self) -> str:
        """The tokens-file line (always the current format)."""
        out = f"{self.tenant}:{self.token_id}:{self.digest_hex}"
        if self.not_after is not None or self.token_only:
            out += f":{int(self.not_after) if self.not_after is not None else ''}"
        if self.token_only:
            out += f":{TOKEN_ONLY_FLAG}"
        return out

    def expired(self, now: float) -> bool:
        return self.not_after is not None and now >= self.not_after


def _parse_token_line(line: str, path: Path, lineno: int) -> TokenRecord:
    def bad(why: str) -> ConfigError:
        return ConfigError(f"tokens file {path}: line {lineno}: {why}", details={"path": str(path), "line": lineno})

    parts = [p.strip() for p in line.split(":")]
    if len(parts) == 2:  # legacy tenant:sha256
        tenant, digest = parts[0], parts[1].lower()
        if not _TENANT_RE.match(tenant) or not digest:
            raise bad("must be 'tenant_id:token_id:sha256[:not_after[:flags]]' (or the older 'tenant_id:sha256')")
        if not _HEX64_RE.match(digest):
            raise bad("value must be a SHA-256 hex digest")
        return TokenRecord(tenant, f"legacy-{digest[:8]}", digest, legacy=True)
    if not 3 <= len(parts) <= 5:
        raise bad("must be 'tenant_id:token_id:sha256[:not_after[:flags]]'")
    tenant, token_id, digest = parts[0], parts[1], parts[2].lower()
    if not _TENANT_RE.match(tenant):
        raise bad("invalid tenant id")
    if not _TOKEN_ID_RE.match(token_id):
        raise bad("invalid token id")
    if not _HEX64_RE.match(digest):
        raise bad("value must be a SHA-256 hex digest")
    not_after: float | None = None
    if len(parts) >= 4 and parts[3]:
        if not parts[3].isdigit():
            raise bad("not_after must be unix seconds")
        not_after = float(parts[3])
    flags = {f.strip() for f in parts[4].split(",") if f.strip()} if len(parts) == 5 else set()
    if flags - {TOKEN_ONLY_FLAG}:
        raise bad(f"unknown flag(s) {sorted(flags - {TOKEN_ONLY_FLAG})}")
    return TokenRecord(tenant, token_id, digest, not_after, TOKEN_ONLY_FLAG in flags)


def read_token_records(path: Path) -> list[TokenRecord]:
    """All records of a tokens file (both line formats).

    Raises:
        ConfigError: unreadable file or a malformed line (duplicate token ids included).
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read tokens file {path}: {exc.strerror}", details={"path": str(path)}) from exc
    out: list[TokenRecord] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rec = _parse_token_line(line, path, lineno)
        if rec.token_id in seen and not rec.legacy:
            raise ConfigError(f"tokens file {path}: line {lineno}: duplicate token id {rec.token_id}",
                              details={"path": str(path), "line": lineno})
        seen.add(rec.token_id)
        out.append(rec)
    return out


_TOKENS_FILE_LOCK = threading.Lock()


def _lock_path(tokens_file: Path) -> Path:
    return Path(tokens_file).with_name(f".{Path(tokens_file).name}.lock")


def new_token() -> str:
    """A fresh tenant bearer token (256 random bits)."""
    return "pgwt-" + secrets.token_urlsafe(32)


def issue_token_record(tenant: str, tokens_file: Path, *, token_only: bool = False,
                       ttl_s: float | None = None, token: str | None = None) -> tuple[str, TokenRecord]:
    """Create a tenant bearer token (or register ``token`` from :func:`new_token`) and append its record
    to ``tokens_file`` (mode 0600).

    Returns ``(token, record)``; only the hash is stored. Shared by ``pgw service token create`` and
    enrollment (``POST /pgw/v1/enroll``).

    Raises:
        ConfigError: invalid tenant id or unwritable tokens file.
    """
    from pgw.core.fileio import file_lock, write_private

    if not valid_tenant_id(tenant):
        raise ConfigError("tenant id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    if ttl_s is not None and ttl_s <= 0:
        raise ConfigError("token lifetime must be positive")
    token = token or new_token()
    rec = TokenRecord(tenant, "tk_" + secrets.token_hex(6), hash_token(token),
                      None if ttl_s is None else float(int(time.time() + ttl_s)), token_only)
    with _TOKENS_FILE_LOCK, file_lock(_lock_path(tokens_file)):
        write_private(Path(tokens_file), rec.line() + "\n", append=True)
    return token, rec


def issue_tenant_token(tenant: str, tokens_file: Path, *, token_only: bool = False) -> tuple[str, str]:
    """Like :func:`issue_token_record`; returns ``(token, sha256 hex)``."""
    token, rec = issue_token_record(tenant, tokens_file, token_only=token_only)
    return token, rec.digest_hex


def revoke_token(tokens_file: Path, token_id: str) -> TokenRecord:
    """Remove the token ``token_id`` from ``tokens_file`` (atomic rewrite; comments and other lines kept).

    Running services pick the change up within their reload interval.

    Raises:
        ConfigError: unknown token id, or the file cannot be read or written.
    """
    from pgw.core.fileio import atomic_write_text, file_lock

    path = Path(tokens_file)
    with _TOKENS_FILE_LOCK, file_lock(_lock_path(path)):
        found: TokenRecord | None = None
        kept: list[str] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigError(f"cannot read tokens file {path}: {exc.strerror}") from exc
        for lineno, raw in enumerate(lines, 1):
            line = raw.strip()
            if line and not line.startswith("#"):
                rec = _parse_token_line(line, path, lineno)
                if rec.token_id == token_id:
                    found = rec
                    continue
            kept.append(raw)
        if found is None:
            raise ConfigError(f"no token with id {token_id} in {path}")
        atomic_write_text(path, "".join(f"{x}\n" for x in kept), mode=0o600)
    return found


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
    token_id: str | None = field(default=None, compare=False)
    _valid: Callable[[], bool] | None = field(default=None, compare=False, repr=False)

    def still_valid(self) -> bool:
        """False once the credential behind this identity was revoked or expired (long-lived channels
        re-check this and close)."""
        return self._valid() if self._valid is not None else True


def _read_map(path: Path, what: str) -> list[tuple[str, str]]:
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
        out.append((tenant, value))
    return out


class TenantAuthenticator:
    """Verifies ``Authorization: Bearer <token>`` against SHA-256 hashes (constant-time compare)
    and/or the mTLS client certificate subject.

    Args:
        tokens_file: Token hash file (see module docstring), or ``None``. Reloaded when it changes.
        allow_unauthenticated: Dev only: accept requests without credentials as tenant
            ``anonymous`` (credentials that *are* presented must still be valid).
        mtls_tenants_file: Certificate subject -> tenant map, or ``None``.
        token_hashes: Extra ``{sha256_hex: tenant}`` entries (tests, programmatic setup).
        mtls_subjects: Extra ``{subject_or_cn: tenant}`` entries.
        allow_empty: Start with no credentials at all (enrollment adds tenants at run time).
        require_client_cert: Refuse (403) a token without a verified, mapped client certificate unless
            the token is marked token-only.
        reload_interval_s: How often (at most) ``tokens_file`` is checked for changes.
        clock: Monotonic clock (tests). Token expiry uses ``wall_clock``.

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
        allow_empty: bool = False,
        require_client_cert: bool = False,
        reload_interval_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.allow_unauthenticated = allow_unauthenticated
        self.require_client_cert = require_client_cert
        self.tokens_file = Path(tokens_file) if tokens_file is not None else None
        self.reload_interval_s = reload_interval_s
        self._clock = clock
        self._wall = wall_clock
        self._reload_lock = threading.Lock()
        self._static: list[tuple[bytes, TokenRecord]] = []
        self._added: list[tuple[bytes, TokenRecord]] = []
        self._file: list[tuple[bytes, TokenRecord]] = []
        self._tokens: list[tuple[bytes, TokenRecord]] = []
        self._ids: dict[str, TokenRecord] = {}
        self._file_sig: tuple[int, int] | None = None
        self._next_check = 0.0
        self._subjects: dict[str, str] = {}
        if self.tokens_file is not None:
            self._file_sig = self._signature()
            self._file = [(bytes.fromhex(r.digest_hex), r) for r in read_token_records(self.tokens_file)]
            self._next_check = clock() + reload_interval_s
        for h, tenant in (token_hashes or {}).items():
            if not _HEX64_RE.match(h.lower()) or not _TENANT_RE.match(tenant):
                raise ConfigError("token_hashes entries must be {sha256_hex: tenant_id}")
            h = h.lower()
            self._static.append((bytes.fromhex(h), TokenRecord(tenant, f"static-{h[:8]}", h)))
        self._rebuild()
        if mtls_tenants_file is not None:
            for tenant, subj in _read_map(Path(mtls_tenants_file), "mTLS tenants"):
                self._subjects[subj] = tenant
        for subj, tenant in (mtls_subjects or {}).items():
            if not _TENANT_RE.match(tenant):
                raise ConfigError("mtls_subjects values must be tenant ids")
            self._subjects[subj] = tenant
        if not self._tokens and not self._subjects and not allow_unauthenticated and not allow_empty:
            raise ConfigError(
                "no tenant credentials configured: set tokens_file and/or mtls_tenants_file "
                "(or allow_unauthenticated for development)"
            )
        if allow_unauthenticated:
            log.warning("compute service accepts UNAUTHENTICATED requests (development mode)")
        log.info("tenant auth: %d token(s), %d certificate subject(s)%s", len(self._tokens), len(self._subjects),
                 ", client certificate required for tokens" if require_client_cert else "")

    # ---- token set ---------------------------------------------------------------------------------
    def _signature(self) -> tuple[int, int] | None:
        assert self.tokens_file is not None
        try:
            st = os.stat(self.tokens_file)
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _rebuild(self) -> None:
        tokens = [*self._file, *self._added, *self._static]
        self._tokens = tokens  # swapped as a whole: readers never see a half-built list
        self._ids = {r.token_id: r for _, r in tokens}

    def maybe_reload(self, *, force: bool = False) -> bool:
        """Re-read ``tokens_file`` if it changed (at most every ``reload_interval_s``); True if reloaded.

        A file that cannot be parsed (e.g. mid-edit by hand) keeps the previous set and logs an error.
        """
        if self.tokens_file is None or (not force and self._clock() < self._next_check):
            return False
        with self._reload_lock:
            if not force and self._clock() < self._next_check:
                return False
            self._next_check = self._clock() + self.reload_interval_s
            sig = self._signature()
            if sig == self._file_sig and not force:
                return False
            try:
                recs = read_token_records(self.tokens_file)
            except ConfigError as exc:
                log.error("tokens file not reloaded (keeping the previous set): %s", exc.message)
                return False
            self._file_sig = sig
            self._file = [(bytes.fromhex(r.digest_hex), r) for r in recs]
            self._added = []  # the file is the truth: enrollment writes it before adding a token here
            self._rebuild()
            log.info("tokens file reloaded: %d token(s)", len(self._tokens))
            return True

    def add_token_hash(self, digest_hex: str, tenant: str, *, token_id: str | None = None,
                       token_only: bool = False, not_after: float | None = None) -> None:
        """Accept a new token (by SHA-256 hex) for ``tenant`` without waiting for a reload (enrollment)."""
        if not _HEX64_RE.match(digest_hex.lower()) or not _TENANT_RE.match(tenant):
            raise ConfigError("add_token_hash needs a sha256 hex digest and a tenant id")
        h = digest_hex.lower()
        rec = TokenRecord(tenant, token_id or f"added-{h[:8]}", h, not_after, token_only)
        with self._reload_lock:
            self._added = [*self._added, (bytes.fromhex(h), rec)]
            self._rebuild()

    def add_subject(self, subject: str, tenant: str) -> None:
        """Map a client-certificate subject or CN to ``tenant`` without a restart (enrollment)."""
        if not _TENANT_RE.match(tenant) or not subject:
            raise ConfigError("add_subject needs a subject and a tenant id")
        self._subjects = {**self._subjects, subject: tenant}

    @property
    def tenants(self) -> set[str]:
        """All configured tenant ids."""
        return {r.tenant for _, r in self._tokens} | set(self._subjects.values())

    def token_valid(self, token_id: str) -> bool:
        """Whether ``token_id`` is still configured and unexpired (reloads the file if due)."""
        self.maybe_reload()
        rec = self._ids.get(token_id)
        return rec is not None and not rec.expired(self._wall())

    def _match_token(self, token: str) -> TokenRecord | None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        found: TokenRecord | None = None
        for h, rec in self._tokens:  # scan all entries: constant time w.r.t. which one matches
            if hmac.compare_digest(h, digest):
                found = rec
        return found

    def _match_subject(self, subject: str) -> str | None:
        t = self._subjects.get(subject)
        if t is None:
            cn = parse_subject_cn(subject)
            if cn is not None:
                t = self._subjects.get(cn)
        return t

    def tenant_for_subject(self, subject: str) -> str | None:
        """Tenant a client-certificate subject (or its CN) is mapped to, if any."""
        return self._match_subject(subject)

    def authenticate(self, authorization: str | None, client_cert_subject: str | None = None) -> TenantIdentity:
        """Return the identity or raise :class:`pgw.core.AuthError`.

        Raises:
            AuthError: missing/invalid/expired credentials (HTTP 401).
            PGWError(permission_denied): token and certificate name different tenants, the
                certificate is not mapped to any tenant, or a client certificate is required and the
                token is not token-only (HTTP 403).
        """
        self.maybe_reload()
        cert_tenant: str | None = None
        if client_cert_subject:
            cert_tenant = self._match_subject(client_cert_subject)
            if cert_tenant is None and self._subjects:
                raise PGWError("client certificate is not authorized", code=ErrorCode.PERMISSION_DENIED)
        rec: TokenRecord | None = None
        if authorization:
            scheme, _, token = authorization.strip().partition(" ")
            token = token.strip()
            if scheme.lower() != "bearer" or not token:
                raise AuthError("malformed Authorization header; expected 'Bearer <token>'")
            rec = self._match_token(token)
            if rec is None:
                raise AuthError("invalid bearer token")
            if rec.expired(self._wall()):
                raise AuthError("bearer token expired")
        if rec is not None and cert_tenant is not None and rec.tenant != cert_tenant:
            raise PGWError("bearer token and client certificate belong to different tenants",
                           code=ErrorCode.PERMISSION_DENIED)
        if rec is not None:
            if self.require_client_cert and cert_tenant is None and not rec.token_only:
                raise PGWError("this service requires the gateway's client certificate (mTLS) for compute; the "
                               "certificate was not presented, has expired or is not registered",
                               code=ErrorCode.PERMISSION_DENIED)
            tid = rec.token_id
            return TenantIdentity(rec.tenant, "token", tid, lambda: self.token_valid(tid))
        if cert_tenant is not None:
            return TenantIdentity(cert_tenant, "mtls")
        if self.allow_unauthenticated:
            return TenantIdentity(_ANON, "anonymous")
        raise AuthError("missing credentials: send 'Authorization: Bearer <token>' or a client certificate")
