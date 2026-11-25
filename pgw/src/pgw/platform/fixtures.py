"""In-memory :class:`~pgw.platform.client.AccountsClient` with deterministic sample data.

Used by the platform tests, the screenshot script and ``python -m pgw.platform --demo``. It is a
stand-in for ``pgw.service.accounts`` and is never selected implicitly by a production service.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import random
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Literal

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

__all__ = ["DEMO_ORG_ID", "FixtureAccounts"]

DEMO_ORG_ID = "org_northwind"
_DAY = 86_400.0

_MODELS = [
    ModelInfo("qwen2.5-7b-instruct", "Qwen2.5 7B Instruct", "available", "7.6B", 32_768,
              "General-purpose assistant; the strongest model of this release.", 310.0, 41.0),
    ModelInfo("qwen2.5-3b-instruct", "Qwen2.5 3B Instruct", "available", "3.1B", 32_768,
              "Balanced quality and latency for chat and extraction.", 190.0, 68.0),
    ModelInfo("qwen2.5-1.5b-instruct", "Qwen2.5 1.5B Instruct", "available", "1.5B", 32_768,
              "Fast classification, routing and short answers.", 120.0, 104.0),
    ModelInfo("qwen2.5-0.5b-instruct", "Qwen2.5 0.5B Instruct", "available", "0.5B", 32_768,
              "Smallest and fastest; tests and lightweight tasks.", 80.0, 162.0),
]


def _digest(v: str) -> bytes:
    return hashlib.sha256(v.encode()).digest()


class FixtureAccounts:
    """Sample organisation ``Northwind Labs`` with five keys, 90 days of usage and a request log.

    Args:
        admin_token, viewer_token: Sign-in tokens for the two roles (``None`` disables that role).
        clock: Wall clock; the sample data is laid out relative to it.
        seed: Seed of the sample data.
    """

    def __init__(self, admin_token: str | None = "demo", viewer_token: str | None = "demo-viewer", *,  # noqa: S107
                 clock: Callable[[], float] = time.time, seed: int = 7):
        self._tokens: list[tuple[bytes, Literal["admin", "viewer"]]] = []
        if admin_token:
            self._tokens.append((_digest(admin_token), "admin"))
        if viewer_token:
            self._tokens.append((_digest(viewer_token), "viewer"))
        self.clock = clock
        self._lock = threading.Lock()
        now = clock()
        self._org = Org(DEMO_ORG_ID, "Northwind Labs", created_at=now - 140 * _DAY, plan="Enterprise",
                        region="eu-west", retention_days=30)
        m7, m3, m15, m05 = (m.id for m in _MODELS)
        self._keys: dict[str, SdkKey] = {}
        for k in [
            SdkKey("key_prod01", "Production backend", "pgw_live_…7QXA", "live", None, now - 96 * _DAY,
                   now - 95, None, "maya@northwind.example"),
            SdkKey("key_supp02", "Support assistant", "pgw_live_…K2MD", "live", (m7,), now - 61 * _DAY,
                   now - 41 * 60, None, "maya@northwind.example"),
            SdkKey("key_stag03", "Staging", "pgw_test_…P9TE", "test", (m15, m05), now - 44 * _DAY,
                   now - 5 * 3600, None, "jonas@northwind.example"),
            SdkKey("key_ci0004", "CI smoke tests", "pgw_test_…2HWR", "test", (m05,), now - 30 * _DAY,
                   now - 3 * _DAY, None, "ci@northwind.example"),
            SdkKey("key_old005", "Legacy batch job", "pgw_live_…V6NB", "live", (m3,), now - 120 * _DAY,
                   now - 13 * _DAY, now - 12 * _DAY, "jonas@northwind.example"),
        ]:
            self._keys[k.id] = k
        self._usage = self._make_usage(now, seed)
        self._log = self._make_log(now, seed)
        self._audit: list[AuditEvent] = [
            AuditEvent(now - 41 * 60, "sdk.enroll.ok", None, "key_supp02"),
            AuditEvent(now - 5 * 3600, "sdk.enroll.failed", None, "key_old005", "revoked"),
            AuditEvent(now - 2 * _DAY, "quota.exceeded", None, "key_stag03", "daily_token_quota"),
            AuditEvent(now - 12 * _DAY, "key.revoked", "jonas@northwind.example", "key_old005"),
            AuditEvent(now - 30 * _DAY, "key.created", "ci@northwind.example", "key_ci0004"),
            AuditEvent(now - 44 * _DAY, "key.created", "jonas@northwind.example", "key_stag03"),
        ]

    # ---- sample data ---------------------------------------------------------------------------------
    def _make_usage(self, now: float, seed: int) -> list[UsageRow]:
        rng = random.Random(seed)  # noqa: S311 - sample data
        today = datetime.fromtimestamp(now, UTC).date()
        plan = {  # key -> (model, base requests/day, prompt tokens/request, completion tokens/request)
            "key_prod01": [("qwen2.5-7b-instruct", 5200, 820, 260), ("qwen2.5-3b-instruct", 2300, 540, 180),
                           ("qwen2.5-1.5b-instruct", 1500, 210, 40)],
            "key_supp02": [("qwen2.5-7b-instruct", 1900, 1350, 320)],
            "key_stag03": [("qwen2.5-1.5b-instruct", 260, 300, 90), ("qwen2.5-0.5b-instruct", 180, 120, 60)],
            "key_ci0004": [("qwen2.5-0.5b-instruct", 90, 60, 24)],
            "key_old005": [("qwen2.5-3b-instruct", 700, 900, 400)],
        }
        rows: list[UsageRow] = []
        for back in range(89, -1, -1):
            d = today - timedelta(days=back)
            growth = 0.55 + 0.45 * (89 - back) / 89
            weekday = 0.62 if d.weekday() >= 5 else 1.0
            for key_id, lines in plan.items():
                k = self._keys[key_id]
                day_start = datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()
                if k.created_at and day_start + _DAY < k.created_at:
                    continue
                if k.revoked_at and day_start > k.revoked_at:
                    continue
                if key_id == "key_ci0004" and back < 3:
                    continue
                for model, base, pt, ct in lines:
                    n = int(base * growth * weekday * rng.uniform(0.85, 1.15))
                    if back == 0:
                        n = int(n * (datetime.fromtimestamp(now, UTC).hour + 1) / 24)
                    errors = int(n * rng.uniform(0.0004, 0.0022))
                    rows.append(UsageRow(d.isoformat(), key_id, model, n, int(n * pt * rng.uniform(0.9, 1.1)),
                                         int(n * ct * rng.uniform(0.9, 1.1)), n - errors, errors))
        return rows

    def _make_log(self, now: float, seed: int) -> list[RequestRecord]:
        rng = random.Random(seed + 1)  # noqa: S311 - sample data
        weights = [("key_prod01", "qwen2.5-7b-instruct", 34), ("key_prod01", "qwen2.5-3b-instruct", 16),
                   ("key_prod01", "qwen2.5-1.5b-instruct", 11), ("key_supp02", "qwen2.5-7b-instruct", 14),
                   ("key_stag03", "qwen2.5-1.5b-instruct", 3), ("key_stag03", "qwen2.5-0.5b-instruct", 2)]
        speed = {m.id: (m.ttft_ms or 200.0, m.tokens_per_s or 50.0) for m in _MODELS}
        pop = [(k, m) for k, m, w in weights for _ in range(w)]
        out: list[RequestRecord] = []
        t = now - 95
        for i in range(600):
            key_id, model = rng.choice(pop)
            ttft0, tps = speed[model]
            pt = int(rng.lognormvariate(6.3, 0.7))
            ct = max(1, int(rng.lognormvariate(5.0, 0.8)))
            ttft = ttft0 * rng.uniform(0.8, 1.5) + pt * 0.02
            err = rng.random() < 0.012
            code = rng.choice(["rate_limited", "deadline_exceeded", "client_closed"]) if err else None
            latency = ttft + (0 if err else ct / tps * 1000 * rng.uniform(0.9, 1.2))
            out.append(RequestRecord(
                id=f"req_{hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:16]}", ts=t, key_id=key_id, model=model,
                status="error" if err else "ok", verified=not err, private=True, latency_ms=round(latency, 1),
                ttft_ms=round(ttft, 1), prompt_tokens=pt, completion_tokens=0 if err else ct,
                stream=rng.random() < 0.7, error_code=code))
            t -= rng.expovariate(1 / 38.0)
        return out

    # ---- AccountsClient ------------------------------------------------------------------------------
    def authenticate(self, token: str) -> Principal | None:
        if not token:
            return None
        d = _digest(token)
        role: Literal["admin", "viewer"] | None = None
        for ref, r in self._tokens:
            if hmac.compare_digest(d, ref):
                role = r
        if role is None:
            return None
        return Principal(DEMO_ORG_ID, role, name="Maya Lindqvist" if role == "admin" else "Read-only viewer",
                         email="maya@northwind.example" if role == "admin" else None)

    def principal_for_sso(self, *, sub: str, email: str | None, role: str) -> Principal | None:
        if not email or not email.endswith("@northwind.example"):
            return None
        return Principal(DEMO_ORG_ID, "admin" if role == "admin" else "viewer", name=None, email=email)

    def _check(self, org_id: str) -> None:
        if org_id != DEMO_ORG_ID:
            raise AccountsError("not found", kind="not_found")

    def org(self, org_id: str) -> Org:
        self._check(org_id)
        return self._org

    def list_keys(self, org_id: str) -> list[SdkKey]:
        self._check(org_id)
        with self._lock:
            return sorted(self._keys.values(), key=lambda k: (k.revoked_at is not None, -(k.created_at or 0)))

    def create_key(self, org_id: str, *, name: str, env: str, models: Sequence[str] | None,
                   actor: str) -> CreatedKey:
        self._check(org_id)
        known = {m.id for m in _MODELS}
        if models and not set(models) <= known:
            raise AccountsError("unknown model in the key's scope", kind="invalid")
        body = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
        secret = f"pgw_{env}_{body}"
        key = SdkKey(f"key_{secrets.token_hex(4)}", name, f"pgw_{env}_…{body[-4:]}", env,
                     tuple(models) if models else None, self.clock(), None, None, actor)
        with self._lock:
            self._keys[key.id] = key
            self._audit.insert(0, AuditEvent(self.clock(), "key.created", actor, key.id))
        return CreatedKey(key, secret)

    def revoke_key(self, org_id: str, key_id: str, *, actor: str) -> SdkKey:
        self._check(org_id)
        with self._lock:
            k = self._keys.get(key_id)
            if k is None:
                raise AccountsError("no such key", kind="not_found")
            if k.revoked_at is None:
                k = replace(k, revoked_at=self.clock())
                self._keys[key_id] = k
                self._audit.insert(0, AuditEvent(self.clock(), "key.revoked", actor, key_id))
            return k

    def usage(self, org_id: str, *, since: str, until: str) -> list[UsageRow]:
        self._check(org_id)
        return [r for r in self._usage if since <= r.day <= until]

    def requests(self, org_id: str, *, limit: int, before: str | None = None, key_id: str | None = None,
                 model: str | None = None, status: str | None = None) -> RequestPage:
        self._check(org_id)
        recs = self._log
        if before:
            idx = next((i for i, r in enumerate(recs) if r.id == before), None)
            recs = recs[idx + 1:] if idx is not None else []
        recs = [r for r in recs if (not key_id or r.key_id == key_id) and (not model or r.model == model)
                and (not status or r.status == status)]
        page = recs[:limit]
        return RequestPage(page, page[-1].id if len(recs) > limit else None)

    def models(self, org_id: str) -> list[ModelInfo]:
        self._check(org_id)
        return list(_MODELS)

    def audit(self, org_id: str, *, limit: int = 50) -> list[AuditEvent]:
        self._check(org_id)
        with self._lock:
            return list(self._audit[:limit])
