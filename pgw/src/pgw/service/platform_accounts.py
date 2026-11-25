"""The developer platform's view of the provider's accounts, in the service process.

:class:`ServiceAccountsClient` implements :class:`pgw.platform.client.AccountsClient` over the same store and runtime
that serve the admin API (``/pgw/v1/admin``, ``docs/sdk/CONTRACT.md`` appendix A4): the console mounted on the service
(``PGW_PLATFORM_ENABLED=1``) and the admin API see the same organisations, keys, usage, request log and audit log.
A console running as a separate process uses :class:`pgw.platform.client.HttpAccountsClient` against the admin API
instead (``PGW_PLATFORM_ACCOUNTS_URL``).

What it can show honestly:

* tokens, compute round trips: metered by the provider on the masked traffic;
* requests, the input/output token split, errors and the request log: reported by the client library after each
  request (counts and timing only; never any content) and recorded by the provider;
* "checked": the client reported that every integrity check of the request passed. The provider cannot check a
  result itself (it only ever sees masked data); the console labels it as checked client-side.

Every value goes through the platform's allow-list parsers, so nothing but metadata can reach a page.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from pgw.core.errors import PGWError
from pgw.platform.client import (
    AccountsError,
    AuditEvent,
    CreatedKey,
    ModelInfo,
    Org,
    Principal,
    RequestPage,
    RequestRecord,
    SdkKey,
    UsageRow,
)
from pgw.service.accounts import Accounts, served_models

__all__ = ["ServiceAccountsClient"]


def _err(exc: PGWError) -> AccountsError:
    code = str(exc.code)
    kind = ("not_found" if code in ("model_not_found", "not_found") or "no organisation" in exc.message
            or "no SDK key" in exc.message else "denied" if code in ("permission_denied", "unauthenticated")
            else "invalid" if code in ("config", "protocol", "bad_shape") else "unavailable")
    return AccountsError(exc.message[:200], kind=kind)


class ServiceAccountsClient:
    """:class:`pgw.platform.client.AccountsClient` over :class:`pgw.service.accounts.Accounts` (in process).

    Args:
        accounts: The service's accounts runtime (its store is the admin API's).
        sso_org: Organisation of every single-sign-on user (``PGW_PLATFORM_OIDC_ORG``).
    """

    def __init__(self, accounts: Accounts, *, sso_org: str | None = None):
        self.accounts = accounts
        self.store = accounts.store
        self.sso_org = sso_org

    def _call(self, fn: Any) -> Any:
        try:
            return fn()
        except AccountsError:
            raise
        except PGWError as exc:
            raise _err(exc) from None
        finally:
            with contextlib.suppress(Exception):
                self.store.close()  # platform calls run on short-lived worker threads

    # ---- sign-in ------------------------------------------------------------------------------------
    def authenticate(self, token: str) -> Principal | None:
        """An organisation admin token (``pgw service admin-token create --org ORG``) of an enabled organisation."""
        if not token or any(c.isspace() for c in token):
            return None

        def go() -> Principal | None:
            rec = self.store.admin_for_token(token)
            if rec is None or not rec.get("org_id"):
                return None  # unknown, revoked, or a global token: the console is per organisation
            org = self.store.get_org(str(rec["org_id"]))
            if org.get("disabled_at"):
                return None
            return Principal(org_id=str(org["org_id"]), role="admin", name=str(rec.get("name") or "") or None)

        return self._call(go)  # type: ignore[no-any-return]

    def principal_for_sso(self, *, sub: str, email: str | None, role: str) -> Principal | None:
        if not self.sso_org:
            return None
        return Principal(org_id=self.sso_org, role="admin" if role == "admin" else "viewer", email=email)

    # ---- data ---------------------------------------------------------------------------------------
    def org(self, org_id: str) -> Org:
        return Org.parse(self._call(lambda: self.store.get_org(org_id)))

    def list_keys(self, org_id: str) -> list[SdkKey]:
        return [SdkKey.parse(k.public()) for k in self._call(lambda: self.store.list_keys(org_id))]

    def create_key(self, org_id: str, *, name: str, env: str, models: Sequence[str] | None,
                   actor: str) -> CreatedKey:
        s = self.accounts.settings

        def go() -> CreatedKey:
            key, rec = self.store.create_key(org_id, name, str(s.public_url), self.accounts.ca_sha256,
                                             env=env or s.sdk_key_env, models=list(models or []),
                                             actor=f"console:{actor}")
            self.accounts.refresh()
            return CreatedKey(key=SdkKey.parse(rec.public()), secret=key.encode())

        return self._call(go)  # type: ignore[no-any-return]

    def revoke_key(self, org_id: str, key_id: str, *, actor: str) -> SdkKey:
        def go() -> SdkKey:
            rec = self.store.get_key(key_id)
            if rec.org_id != org_id:
                raise AccountsError("not found", kind="not_found")
            return SdkKey.parse(self.accounts.revoke_key(key_id, actor=f"console:{actor}"))

        return self._call(go)  # type: ignore[no-any-return]

    def usage(self, org_id: str, *, since: str, until: str) -> list[UsageRow]:
        with contextlib.suppress(Exception):
            self.accounts.flush()  # the last seconds of metering, so a page shows what just happened
        rows = self._call(lambda: self.store.usage(org_id=org_id, since=since, until=until))
        out = (UsageRow.parse(r) for r in rows)
        return [r for r in out if r is not None]

    def requests(self, org_id: str, *, limit: int, before: str | None = None, key_id: str | None = None,
                 model: str | None = None, status: str | None = None) -> RequestPage:
        if before is not None and not str(before).isdigit():
            before = None
        recs = self._call(lambda: self.store.requests(org_id, limit=limit, before=int(before) if before else None,
                                                      key_id=key_id or None, model=model or None,
                                                      status=status or None))
        nxt = recs[-1]["id"] if len(recs) >= max(1, limit) else None
        return RequestPage([RequestRecord.parse(r) for r in recs], nxt)

    def models(self, org_id: str) -> list[ModelInfo]:
        return [ModelInfo.parse(m) for m in served_models(self.accounts)]

    def audit(self, org_id: str, *, limit: int = 50) -> list[AuditEvent]:
        return [AuditEvent.parse(ev) for ev in self._call(lambda: self.store.audit_events(org_id, limit=limit))]
