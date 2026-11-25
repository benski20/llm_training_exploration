"""Thin client between the developer platform and the provider's accounts API.

The platform never talks to anything else: organisations, SDK keys, usage, the request log and the
model catalogue all come through :class:`AccountsClient`. Two implementations ship here:

* :class:`HttpAccountsClient`: the provider's admin API (``/pgw/v1/admin/...``, appendix A4 of
  ``docs/sdk/CONTRACT.md``), for a platform process separate from the service.
* any in-process object with the same methods (``pgw.service.accounts`` on the provider service);
  :func:`pgw.platform.fixtures.FixtureAccounts` is the in-memory one used by tests and the demo.

Every value crossing this boundary is parsed into the dataclasses below by **allow-list**: a request
record has metadata only (time, key, model, status, verification, timing, token counts). Any other field
an implementation returns (a prompt, a completion, a message) is dropped here and can never reach a page.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pgw.core.errors import ErrorCode, PGWError

__all__ = [
    "AccountsClient", "AccountsError", "AuditEvent", "CreatedKey", "HttpAccountsClient", "ModelInfo", "Org",
    "Principal",
    "RequestPage", "RequestRecord", "SdkKey", "UsageRow",
]

Role = Literal["admin", "viewer"]
_ROLES = {"owner": "admin", "admin": "admin", "member": "viewer", "viewer": "viewer", "developer": "admin"}
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MODEL_STATUS = {"available", "degraded", "unavailable", "preview"}


class AccountsError(PGWError):
    """The accounts API refused or failed a call (message is safe to show).

    ``kind`` is one of ``not_found``, ``denied``, ``invalid``, ``unavailable``, ``unsupported`` (the
    accounts service does not offer this data, e.g. no request log).
    """

    def __init__(self, message: str, *, kind: str = "unavailable", code: ErrorCode | None = None):
        super().__init__(message, code=code or {"denied": ErrorCode.PERMISSION_DENIED,
                                                "invalid": ErrorCode.BAD_SHAPE}.get(kind, ErrorCode.UNAVAILABLE))
        self.kind = kind


def _s(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _opt_s(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _i(v: Any) -> int:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else 0


def _opt_f(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


@dataclass(frozen=True)
class Principal:
    """A signed-in platform user, always scoped to one organisation."""

    org_id: str
    role: Role
    name: str | None = None
    email: str | None = None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> Principal:
        role = _ROLES.get(_s(d.get("role")), "viewer")
        return cls(org_id=_s(d.get("org_id")), role=role,  # type: ignore[arg-type]
                   name=_opt_s(d.get("name")), email=_opt_s(d.get("email")))


@dataclass(frozen=True)
class Org:
    id: str
    name: str
    created_at: float | None = None
    plan: str | None = None
    region: str | None = None
    #: Days request metadata is kept (content is never stored at all).
    retention_days: int | None = None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> Org:
        org_id = _s(d.get("id")) or _s(d.get("org_id"))
        return cls(id=org_id, name=_s(d.get("name")) or org_id or "Organisation",
                   created_at=_opt_f(d.get("created_at")),
                   plan=_opt_s(d.get("plan")), region=_opt_s(d.get("region")),
                   retention_days=_i(d.get("retention_days")) or None)


@dataclass(frozen=True)
class SdkKey:
    """An SDK key's metadata. The secret itself exists only in :class:`CreatedKey`, once."""

    id: str
    name: str
    #: Non-secret display form, e.g. ``pgw_live_7Q2M…``.
    prefix: str
    env: str = "live"
    #: Models the key may use; ``None`` means every model of the organisation.
    models: tuple[str, ...] | None = None
    created_at: float | None = None
    last_used_at: float | None = None
    revoked_at: float | None = None
    created_by: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> SdkKey:
        models = d.get("models")
        scoped = tuple(m for m in models if isinstance(m, str)) if isinstance(models, list) else None
        key_id = _s(d.get("id")) or _s(d.get("key_id"))
        # the accounts API's public key id (``sk_<16 hex>``) doubles as the display form when no prefix is given
        prefix = _s(d.get("prefix")) or (f"{key_id[:7]}…{key_id[-4:]}" if len(key_id) > 12 else key_id)
        revoked = _opt_f(d.get("revoked_at"))
        if revoked is None and d.get("state") in ("revoked", "expired"):
            revoked = _opt_f(d.get("expires_at")) or 0.0
        return cls(id=key_id, name=_s(d.get("name"), "Untitled key"), prefix=prefix,
                   env=_s(d.get("env"), "live"), models=scoped or None, created_at=_opt_f(d.get("created_at")),
                   last_used_at=_opt_f(d.get("last_used_at")), revoked_at=revoked,
                   created_by=_opt_s(d.get("created_by")))


@dataclass(frozen=True)
class CreatedKey:
    key: SdkKey
    #: The full key, returned exactly once by the accounts API; the platform shows it once and keeps nothing.
    secret: str = field(repr=False)


@dataclass(frozen=True)
class UsageRow:
    """Usage of one key on one model on one UTC day.

    Fields the accounts service does not meter are ``None`` (the provider meters token positions and compute
    round trips; request counts, the input/output split and verification are reported by the client library,
    where results are verified). Pages show a dash for them.
    """

    day: str
    key_id: str
    model: str
    requests: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Requests whose every result passed verification.
    verified: int | None = None
    errors: int = 0
    #: Token positions processed (input and output); defaults to input + output.
    total_tokens: int | None = None
    #: Compute round trips (provider metering).
    trips: int | None = None

    @property
    def tokens(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> UsageRow | None:
        day = _s(d.get("day"))
        if not _DAY_RE.match(day):
            return None

        def opt(k: str) -> int | None:
            v = d.get(k)
            return _i(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

        return cls(day=day, key_id=_s(d.get("key_id")), model=_s(d.get("model")), requests=opt("requests"),
                   prompt_tokens=opt("prompt_tokens"), completion_tokens=opt("completion_tokens"),
                   verified=opt("verified"), errors=_i(d.get("errors")), total_tokens=opt("tokens"),
                   trips=opt("trips"))


@dataclass(frozen=True)
class RequestRecord:
    """One request in the log: metadata only, by construction (see the module docstring)."""

    id: str
    ts: float
    key_id: str
    model: str
    status: Literal["ok", "error"] = "ok"
    #: Every masked result of the request passed its integrity check.
    verified: bool = True
    #: Data left the customer's process masked (always true for SDK traffic; false only for dev modes).
    private: bool = True
    latency_ms: float | None = None
    ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stream: bool = False
    #: Error class on failure (an error code, never a message that could echo input).
    error_code: str | None = None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> RequestRecord:
        status: Literal["ok", "error"] = "error" if d.get("status") == "error" else "ok"
        code = _s(d.get("error_code"))
        return cls(id=_s(d.get("id")), ts=_opt_f(d.get("ts")) or 0.0, key_id=_s(d.get("key_id")),
                   model=_s(d.get("model")), status=status, verified=d.get("verified") is True,
                   private=d.get("private", True) is True, latency_ms=_opt_f(d.get("latency_ms")),
                   ttft_ms=_opt_f(d.get("ttft_ms")), prompt_tokens=_i(d.get("prompt_tokens")),
                   completion_tokens=_i(d.get("completion_tokens")), stream=d.get("stream") is True,
                   error_code=code if re.fullmatch(r"[a-z0-9_.-]{1,64}", code) else None)


@dataclass(frozen=True)
class RequestPage:
    records: list[RequestRecord]
    #: Cursor for the next (older) page, ``None`` at the end.
    next_before: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    status: str = "available"
    params: str | None = None
    context_length: int | None = None
    description: str | None = None
    #: p50 time to first token (ms) and decode throughput (tokens/s) the provider measures, if any.
    ttft_ms: float | None = None
    tokens_per_s: float | None = None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> ModelInfo:
        status = _s(d.get("status"), "available")
        return cls(id=_s(d.get("id")), name=_s(d.get("name")) or _s(d.get("id")),
                   status=status if status in _MODEL_STATUS else "unavailable", params=_opt_s(d.get("params")),
                   context_length=_i(d.get("context_length")) or None, description=_opt_s(d.get("description")),
                   ttft_ms=_opt_f(d.get("ttft_ms")), tokens_per_s=_opt_f(d.get("tokens_per_s")))


@dataclass(frozen=True)
class AuditEvent:
    """An organisation audit-log entry (key created or revoked, enrollment, quota...): never a secret."""

    ts: float
    event: str
    actor: str | None = None
    key_id: str | None = None
    reason: str | None = None

    @classmethod
    def parse(cls, d: Mapping[str, Any]) -> AuditEvent:
        raw = d.get("detail")
        detail: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
        reason = _s(detail.get("reason"))
        return cls(ts=_opt_f(d.get("ts")) or 0.0, event=_s(d.get("event"))[:64], actor=_opt_s(d.get("actor")),
                   key_id=_opt_s(d.get("key_id")),
                   reason=reason if re.fullmatch(r"[a-z0-9_.-]{1,64}", reason) else None)


@runtime_checkable
class AccountsClient(Protocol):
    """What the platform needs from the accounts service (all calls org-scoped by the caller).

    Methods are synchronous; the platform runs them on a worker thread.
    """

    def authenticate(self, token: str) -> Principal | None:
        """Principal for a console sign-in token (an organisation admin or viewer token), else ``None``."""

    def principal_for_sso(self, *, sub: str, email: str | None, role: str) -> Principal | None:
        """Principal for a single-sign-on identity (``role`` from the IdP claim mapping), else ``None``."""

    def org(self, org_id: str) -> Org: ...

    def list_keys(self, org_id: str) -> list[SdkKey]: ...

    def create_key(self, org_id: str, *, name: str, env: str, models: Sequence[str] | None,
                   actor: str) -> CreatedKey: ...

    def revoke_key(self, org_id: str, key_id: str, *, actor: str) -> SdkKey: ...

    def usage(self, org_id: str, *, since: str, until: str) -> list[UsageRow]:
        """Rows for UTC days ``since``..``until`` inclusive (``YYYY-MM-DD``)."""

    def requests(self, org_id: str, *, limit: int, before: str | None = None, key_id: str | None = None,
                 model: str | None = None, status: str | None = None) -> RequestPage: ...

    def models(self, org_id: str) -> list[ModelInfo]: ...

    # Optional: ``audit(org_id, *, limit) -> list[AuditEvent]`` (shown on Settings when present).


class HttpAccountsClient:
    """:class:`AccountsClient` over the provider's admin API (``/pgw/v1/admin``, CONTRACT appendix A4).

    Sign-in: a user signs in with an **organisation admin token** (``pgwa-...`` created with
    ``pgw service admin-token create --org ORG``); :meth:`authenticate` checks it by listing the organisations it
    may see, which must be exactly one. All later calls use the platform's own token (a global admin token,
    ``service_token``), always scoped to the session's organisation by the platform.

    Args:
        base_url: Provider service URL (``https://provider.example``).
        service_token: The platform's admin token (sent as a bearer token; never logged).
        sso_org: Organisation of every single-sign-on user (the IdP decides who may sign in).
        verify: TLS verification (``True``, a CA bundle path or an ``ssl.SSLContext``).
        transport: httpx transport (tests).
    """

    timeout_s = 10.0

    def __init__(self, base_url: str, service_token: str, *, sso_org: str | None = None, verify: Any = True,
                 transport: Any = None):
        import httpx

        self.sso_org = sso_org
        self._http = httpx.Client(base_url=base_url.rstrip("/") + "/pgw/v1/admin", timeout=self.timeout_s,
                                  verify=verify, transport=transport,
                                  headers={"Authorization": f"Bearer {service_token}"})

    def close(self) -> None:
        self._http.close()

    def _call(self, method: str, path: str, *, json: Any = None, params: Mapping[str, Any] | None = None,
              token: str | None = None) -> Any:
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token is not None else None
        try:
            r = self._http.request(method, path, json=json, headers=headers,
                                   params={k: v for k, v in (params or {}).items() if v is not None})
        except httpx.HTTPError as exc:
            raise AccountsError("the accounts service is unreachable", kind="unavailable") from exc
        if r.status_code == 404:
            raise AccountsError("not found", kind="not_found")
        if r.status_code in (401, 403):
            raise AccountsError("the accounts service refused the request", kind="denied")
        if r.status_code >= 400:
            msg = "the accounts service failed the request"
            try:
                err = r.json().get("error", {})
                if r.status_code < 500 and isinstance(err.get("message"), str):
                    msg = err["message"][:200]
            except (ValueError, AttributeError):
                pass
            raise AccountsError(msg, kind="invalid" if r.status_code < 500 else "unavailable")
        return r.json() if r.content else None

    def authenticate(self, token: str) -> Principal | None:
        if not token or any(c.isspace() for c in token):
            return None
        try:
            d = self._call("GET", "/orgs", token=token)
        except AccountsError as exc:
            if exc.kind in ("denied", "not_found"):
                return None
            raise
        orgs = [o for o in (d or {}).get("orgs", []) if isinstance(o, dict)]
        if len(orgs) != 1 or orgs[0].get("disabled_at"):
            return None  # a global token, or a disabled organisation: the platform is per organisation
        return Principal(org_id=_s(orgs[0].get("org_id")), role="admin")

    def principal_for_sso(self, *, sub: str, email: str | None, role: str) -> Principal | None:
        if not self.sso_org:
            return None
        return Principal(org_id=self.sso_org, role="admin" if role == "admin" else "viewer", email=email)

    def org(self, org_id: str) -> Org:
        return Org.parse(self._call("GET", f"/orgs/{org_id}"))

    def list_keys(self, org_id: str) -> list[SdkKey]:
        d = self._call("GET", f"/orgs/{org_id}/keys")
        return [SdkKey.parse(k) for k in d.get("keys", []) if isinstance(k, dict)]

    def create_key(self, org_id: str, *, name: str, env: str, models: Sequence[str] | None,
                   actor: str) -> CreatedKey:
        body: dict[str, Any] = {"name": name, "env": env}
        if models:
            body["models"] = list(models)
        d = self._call("POST", f"/orgs/{org_id}/keys", json=body)
        return CreatedKey(key=SdkKey.parse(d["record"]), secret=str(d["key"]))

    def revoke_key(self, org_id: str, key_id: str, *, actor: str) -> SdkKey:
        return SdkKey.parse(self._call("POST", f"/orgs/{org_id}/keys/{key_id}/revoke"))

    def usage(self, org_id: str, *, since: str, until: str) -> list[UsageRow]:
        d = self._call("GET", f"/orgs/{org_id}/usage", params={"since": since, "until": until})
        rows = (UsageRow.parse(r) for r in d.get("usage", []) if isinstance(r, dict))
        return [r for r in rows if r is not None]

    def requests(self, org_id: str, *, limit: int, before: str | None = None, key_id: str | None = None,
                 model: str | None = None, status: str | None = None) -> RequestPage:
        try:
            d = self._call("GET", f"/orgs/{org_id}/requests", params={
                "limit": limit, "before": before, "key_id": key_id, "model": model, "status": status})
        except AccountsError as exc:
            if exc.kind == "not_found":
                raise AccountsError("request logging is not enabled on this provider", kind="unsupported") from exc
            raise
        recs = [RequestRecord.parse(r) for r in d.get("requests", []) if isinstance(r, dict)]
        nxt = d.get("next_before")
        return RequestPage(recs, nxt if isinstance(nxt, str) and nxt else None)

    def models(self, org_id: str) -> list[ModelInfo]:
        try:
            d = self._call("GET", f"/orgs/{org_id}/models")
        except AccountsError as exc:
            if exc.kind == "not_found":
                return []
            raise
        return [ModelInfo.parse(m) for m in d.get("models", []) if isinstance(m, dict)]

    def audit(self, org_id: str, *, limit: int = 50) -> list[AuditEvent]:
        d = self._call("GET", f"/orgs/{org_id}/audit", params={"limit": limit})
        return [AuditEvent.parse(ev) for ev in d.get("events", []) if isinstance(ev, dict)]
