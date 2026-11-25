"""Provider accounts: organisations, SDK keys, installations, usage metering, quotas and the audit log.

Persistence is one SQLite database (WAL, ``PGW_SERVICE_ACCOUNTS_DB``) shared by the running service and the
admin CLI (``pgw service keys ...``, ``pgw service usage``). Nothing secret is stored: SDK keys and admin tokens
are kept as SHA-256 digests (:func:`pgw.enroll.sdk_key.secret_digest`), compared in constant time, and never
logged; audit records carry key ids only.

Runtime (:class:`Accounts`, in the service process)

* ``POST /pgw/v1/sdk/enroll``: authenticated by the SDK key; reuses the enrollment machinery
  (:class:`pgw.enroll.service.EnrollmentService`) to issue the per-installation service token and client
  certificate exactly as ``POST /pgw/v1/enroll`` does for a code. One tenant per key: ``sdk.<key hex>``.
* Gate (:attr:`pgw.service.auth.TenantAuthenticator.tenant_gate`): a revoked or expired key, or a disabled
  organisation, makes every credential of its tenant fail authentication, and long-lived channels (stream,
  lane) close at their next validity check.
* Policy (:attr:`pgw.service.handler.ServiceHandler.accounts`): per call, a model outside the key's scope is
  refused (403); the key's row bucket and the key's and organisation's daily token quotas are enforced (429).
  Usage (trips, rows, tokens, maskgen columns, bytes) is counted in memory and flushed to the database by a
  background thread; the hot path never touches SQLite.

Tenants that do not start with ``sdk.`` (gateways enrolled with codes, hand-made tokens) are not affected.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError, RateLimitedError
from pgw.enroll.sdk_key import SDKKey, SDKKeyError, format_key_id, parse_key_id, secret_digest

__all__ = [
    "SDK_ENROLL_PATH", "SDK_TENANT_PREFIX", "SDK_USAGE_PATH", "Accounts", "AccountsStore", "KeyRecord",
    "KeyRejectedError", "add_admin_routes", "add_sdk_routes", "attest_mac", "day_of", "revoke_key_everywhere",
    "tenant_for_key",
]

log = logging.getLogger(__name__)

SDK_ENROLL_PATH = "/pgw/v1/sdk/enroll"
SDK_USAGE_PATH = "/pgw/v1/sdk/usage"
ADMIN_PREFIX = "/pgw/v1/admin"
SDK_TENANT_PREFIX = "sdk."
ADMIN_TOKEN_PREFIX = "pgwa-"  # noqa: S105 - a prefix, not a secret
ATTEST_DOMAIN = b"pgw-admin-attest-v1"
_ORG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_BODY = 64 << 10
_REJECTED = "invalid, revoked or expired SDK key"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orgs (
    org_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL, disabled_at REAL,
    daily_token_quota INTEGER);
CREATE TABLE IF NOT EXISTS sdk_keys (
    key_id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES orgs(org_id), name TEXT NOT NULL, env TEXT NOT NULL,
    digest BLOB NOT NULL, models TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL, revoked_at REAL,
    last_used_at REAL, daily_token_quota INTEGER, rows_per_s REAL, burst_rows INTEGER, max_installations INTEGER);
CREATE INDEX IF NOT EXISTS sdk_keys_org ON sdk_keys(org_id);
CREATE TABLE IF NOT EXISTS installations (
    installation_id TEXT PRIMARY KEY, key_id TEXT NOT NULL REFERENCES sdk_keys(key_id), name TEXT NOT NULL,
    tenant TEXT NOT NULL, token_id TEXT NOT NULL, cert_serial TEXT, cert_cn TEXT, created_at REAL NOT NULL,
    revoked_at REAL, client TEXT);
CREATE INDEX IF NOT EXISTS installations_key ON installations(key_id, created_at);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL, key_id TEXT NOT NULL, model TEXT NOT NULL, trips INTEGER NOT NULL DEFAULT 0,
    rows INTEGER NOT NULL DEFAULT 0, tokens REAL NOT NULL DEFAULT 0, maskgen_cols INTEGER NOT NULL DEFAULT 0,
    maskgen_requests INTEGER NOT NULL DEFAULT 0, bytes_in INTEGER NOT NULL DEFAULT 0,
    bytes_out INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (day, key_id, model));
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, event TEXT NOT NULL, org_id TEXT, key_id TEXT,
    actor TEXT, client TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS audit_org ON audit(org_id, id);
CREATE TABLE IF NOT EXISTS admin_tokens (
    token_id TEXT PRIMARY KEY, org_id TEXT, name TEXT NOT NULL, digest BLOB NOT NULL UNIQUE, created_at REAL NOT NULL,
    revoked_at REAL);
"""
_SCHEMA_VERSION = "1"
_USAGE_FIELDS = ("trips", "rows", "tokens", "maskgen_cols", "maskgen_requests", "bytes_in", "bytes_out")
_LIMIT_FIELDS = ("daily_token_quota", "rows_per_s", "burst_rows", "max_installations")


def day_of(ts: float) -> str:
    """UTC day ``YYYY-MM-DD`` of a unix time (usage and quotas are per UTC day)."""
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _seconds_to_midnight(ts: float) -> float:
    d = dt.datetime.fromtimestamp(ts, dt.UTC)
    nxt = (d + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (nxt - d).total_seconds())


def tenant_for_key(key_hex: str) -> str:
    """Tenant id of an SDK key (``sdk.<16 hex>``)."""
    return SDK_TENANT_PREFIX + key_hex


def attest_mac(admin_token_digest: bytes, nonce: bytes, ca_sha256: bytes) -> bytes:
    """``HMAC-SHA256(SHA-256(admin token), "pgw-admin-attest-v1" | nonce | ca_sha256)`` (``/pgw/v1/admin/attest``)."""
    return hmac.new(admin_token_digest, ATTEST_DOMAIN + nonce + ca_sha256, hashlib.sha256).digest()


def _admin_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class KeyRejectedError(AuthError):
    """The SDK key is unknown, revoked, expired, of another env or its organisation is disabled.

    The reply is the same in every case (no oracle); :attr:`reason` goes to the audit log only.
    """

    def __init__(self, reason: str, key_hex: str | None = None, org_id: str | None = None):
        super().__init__(_REJECTED)
        self.reason = reason
        self.key_hex = key_hex
        self.org_id = org_id


@dataclass
class KeyRecord:
    """One SDK key as stored (the digest never leaves :class:`AccountsStore`)."""

    key_hex: str
    org_id: str
    name: str
    env: str
    models: list[str]
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None
    last_used_at: float | None = None
    daily_token_quota: int | None = None
    rows_per_s: float | None = None
    burst_rows: int | None = None
    max_installations: int | None = None
    installations: int = 0

    @property
    def key_id(self) -> str:
        return "sk_" + self.key_hex

    @property
    def tenant(self) -> str:
        return tenant_for_key(self.key_hex)

    def state(self, now: float | None = None) -> str:
        """``active`` | ``revoked`` | ``expired``."""
        now = time.time() if now is None else now
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at is not None and now >= self.expires_at:
            return "expired"
        return "active"

    def limits(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _LIMIT_FIELDS}

    def public(self) -> dict[str, Any]:
        """JSON view (contract A4 ``key``)."""
        return {"key_id": self.key_id, "org_id": self.org_id, "name": self.name, "env": self.env,
                "models": list(self.models), "created_at": self.created_at, "expires_at": self.expires_at,
                "revoked_at": self.revoked_at, "last_used_at": self.last_used_at, "state": self.state(),
                "limits": self.limits(), "installations": self.installations}


# ---------------------------------------------------------------------------------------------------- store
class AccountsStore:
    """SQLite store (WAL). Safe for several threads and processes (``busy_timeout``, short transactions).

    Args:
        path: The database file (created with mode 0600; its directory should be private).
        clock: Wall clock (tests).
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.clock = clock
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init_db()

    # ---- connection ---------------------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=True)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout = 10000")
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = c
        return c

    @contextlib.contextmanager
    def _tx(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        c = self._conn()
        c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def _init_db(self) -> None:
        import os

        with self._init_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            try:
                c = self._conn()
                mode = c.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).lower() != "wal":  # pragma: no cover - e.g. a network filesystem
                    log.warning("accounts database %s: journal_mode is %s, not WAL", self.path, mode)
                c.executescript(_SCHEMA)
                c.execute("INSERT OR IGNORE INTO meta (k, v) VALUES ('schema', ?)", (_SCHEMA_VERSION,))
            except sqlite3.Error as exc:
                raise ConfigError(f"cannot open the accounts database {self.path}: {exc}") from exc
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(self.path) + suffix)
                with contextlib.suppress(OSError):
                    if p.exists():
                        os.chmod(p, 0o600)

    def close(self) -> None:
        """Close this thread's connection."""
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ---- audit --------------------------------------------------------------------------------------
    def audit(self, event: str, *, org_id: str | None = None, key_hex: str | None = None, actor: str | None = None,
              client: str | None = None, detail: Mapping[str, Any] | None = None,
              conn: sqlite3.Connection | None = None) -> None:
        """Append an audit record (never pass secrets: only ids, names, reasons and counts)."""
        row = (self.clock(), event, org_id, ("sk_" + key_hex) if key_hex else None, actor, client,
               json.dumps(dict(detail), sort_keys=True, default=str) if detail else None)
        log.info("accounts %s org=%s key=%s actor=%s%s", event, org_id, row[3], actor,
                 f" {row[6]}" if row[6] else "")
        try:
            (conn or self._conn()).execute(
                "INSERT INTO audit (ts, event, org_id, key_id, actor, client, detail) VALUES (?,?,?,?,?,?,?)", row)
        except sqlite3.Error as exc:  # never fail the operation because the audit write failed: log loudly
            log.error("cannot write accounts audit record %s: %s", event, exc)

    def audit_events(self, org_id: str | None = None, *, limit: int = 100, before: int | None = None,
                     ) -> list[dict[str, Any]]:
        """Newest first."""
        q, args = "SELECT * FROM audit WHERE 1=1", []
        if org_id is not None:
            q += " AND org_id = ?"
            args.append(org_id)
        if before is not None:
            q += " AND id < ?"
            args.append(int(before))
        q += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        out = []
        for r in self._conn().execute(q, args):
            d = dict(r)
            d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
            out.append(d)
        return out

    # ---- organisations ------------------------------------------------------------------------------
    def create_org(self, org_id: str, name: str | None = None, *, daily_token_quota: int | None = None,
                   actor: str | None = None) -> dict[str, Any]:
        """Create an organisation.

        Raises:
            ConfigError: invalid id/quota, or the id exists.
        """
        if not isinstance(org_id, str) or not _ORG_RE.fullmatch(org_id):
            raise ConfigError("organisation id must match [a-z0-9][a-z0-9_-]{0,47}")
        _check_quota(daily_token_quota, "daily_token_quota")
        name = (name or org_id).strip()[:200] or org_id
        with self._tx() as c:
            if c.execute("SELECT 1 FROM orgs WHERE org_id = ?", (org_id,)).fetchone():
                raise ConfigError(f"organisation {org_id!r} exists")
            c.execute("INSERT INTO orgs (org_id, name, created_at, daily_token_quota) VALUES (?,?,?,?)",
                      (org_id, name, self.clock(), daily_token_quota))
            self.audit("org.created", org_id=org_id, actor=actor, conn=c,
                       detail={"daily_token_quota": daily_token_quota})
        return self.get_org(org_id)

    def get_org(self, org_id: str) -> dict[str, Any]:
        """Raises: PGWError(model_not_found-like 404 'config'): unknown organisation."""
        r = self._conn().execute(
            "SELECT o.*, (SELECT COUNT(*) FROM sdk_keys k WHERE k.org_id = o.org_id AND k.revoked_at IS NULL "
            "AND (k.expires_at IS NULL OR k.expires_at > ?)) AS keys_active FROM orgs o WHERE o.org_id = ?",
            (self.clock(), org_id)).fetchone()
        if r is None:
            raise _not_found(f"no organisation {org_id!r}")
        return dict(r)

    def list_orgs(self) -> list[dict[str, Any]]:
        now = self.clock()
        return [dict(r) for r in self._conn().execute(
            "SELECT o.*, (SELECT COUNT(*) FROM sdk_keys k WHERE k.org_id = o.org_id AND k.revoked_at IS NULL "
            "AND (k.expires_at IS NULL OR k.expires_at > ?)) AS keys_active FROM orgs o ORDER BY o.org_id", (now,))]

    def set_org_disabled(self, org_id: str, disabled: bool, *, actor: str | None = None) -> dict[str, Any]:
        """Disable (every key of the organisation stops working) or re-enable an organisation."""
        with self._tx() as c:
            if not c.execute("SELECT 1 FROM orgs WHERE org_id = ?", (org_id,)).fetchone():
                raise _not_found(f"no organisation {org_id!r}")
            c.execute("UPDATE orgs SET disabled_at = ? WHERE org_id = ?", (self.clock() if disabled else None, org_id))
            self.audit("org.disabled" if disabled else "org.enabled", org_id=org_id, actor=actor, conn=c)
        return self.get_org(org_id)

    # ---- keys ---------------------------------------------------------------------------------------
    def create_key(self, org_id: str, name: str, service_url: str, ca_sha256: bytes | None, *, env: str = "live",
                   models: list[str] | None = None, expires_in_s: float | None = None,
                   daily_token_quota: int | None = None, rows_per_s: float | None = None,
                   burst_rows: int | None = None, max_installations: int | None = None,
                   actor: str | None = None) -> tuple[SDKKey, KeyRecord]:
        """Create an SDK key; returns ``(key, record)``. Only the key's digest is stored.

        Raises:
            ConfigError: invalid arguments; PGWError(404): unknown organisation.
        """
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ConfigError("key name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        models = [m.strip() for m in (models or []) if m and m.strip()]
        for m in models:
            if not _MODEL_RE.fullmatch(m):
                raise ConfigError(f"invalid model name {m!r}")
        models = sorted(set(models))
        if expires_in_s is not None and not 0 < float(expires_in_s) <= 3 * 365 * 86400:
            raise ConfigError("expires_in_s must be positive and at most 3 years")
        _check_quota(daily_token_quota, "daily_token_quota")
        _check_quota(burst_rows, "burst_rows")
        _check_quota(max_installations, "max_installations")
        if rows_per_s is not None and not (isinstance(rows_per_s, int | float) and 0 < float(rows_per_s) < 1e9):
            raise ConfigError("rows_per_s must be positive")
        key = SDKKey.new(env, service_url, ca_sha256)
        now = self.clock()
        rec = KeyRecord(key_hex=key.key_id.hex(), org_id=org_id, name=name, env=env, models=models, created_at=now,
                        expires_at=None if expires_in_s is None else now + float(expires_in_s),
                        daily_token_quota=daily_token_quota,
                        rows_per_s=None if rows_per_s is None else float(rows_per_s), burst_rows=burst_rows,
                        max_installations=max_installations)
        with self._tx() as c:
            org = c.execute("SELECT disabled_at FROM orgs WHERE org_id = ?", (org_id,)).fetchone()
            if org is None:
                raise _not_found(f"no organisation {org_id!r}")
            c.execute("INSERT INTO sdk_keys (key_id, org_id, name, env, digest, models, created_at, expires_at, "
                      "daily_token_quota, rows_per_s, burst_rows, max_installations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (rec.key_hex, org_id, name, env, key.digest, json.dumps(models), now, rec.expires_at,
                       daily_token_quota, rec.rows_per_s, burst_rows, max_installations))
            self.audit("key.created", org_id=org_id, key_hex=rec.key_hex, actor=actor, conn=c,
                       detail={"name": name, "env": env, "models": models, "expires_at": rec.expires_at,
                               **{k: v for k, v in rec.limits().items() if v is not None}})
        return key, rec

    _KEY_SELECT = ("SELECT k.*, (SELECT COUNT(*) FROM installations i WHERE i.key_id = k.key_id AND i.revoked_at IS "
                   "NULL) AS n_inst FROM sdk_keys k")

    @staticmethod
    def _key_rec(r: sqlite3.Row) -> KeyRecord:
        return KeyRecord(key_hex=r["key_id"], org_id=r["org_id"], name=r["name"], env=r["env"],
                         models=json.loads(r["models"]), created_at=r["created_at"], expires_at=r["expires_at"],
                         revoked_at=r["revoked_at"], last_used_at=r["last_used_at"],
                         daily_token_quota=r["daily_token_quota"], rows_per_s=r["rows_per_s"],
                         burst_rows=r["burst_rows"], max_installations=r["max_installations"],
                         installations=r["n_inst"] if "n_inst" in r.keys() else 0)

    def get_key(self, key_id: str) -> KeyRecord:
        """By ``sk_<hex>`` or the bare hex. Raises PGWError(404) when unknown."""
        hx = _key_hex(key_id)
        r = self._conn().execute(self._KEY_SELECT + " WHERE k.key_id = ?", (hx,)).fetchone()
        if r is None:
            raise _not_found(f"no SDK key {key_id!r}")
        return self._key_rec(r)

    def list_keys(self, org_id: str | None = None) -> list[KeyRecord]:
        q, args = self._KEY_SELECT, []
        if org_id is not None:
            q += " WHERE k.org_id = ?"
            args.append(org_id)
        return [self._key_rec(r) for r in self._conn().execute(q + " ORDER BY k.created_at", args)]

    def active_key_states(self) -> list[tuple[KeyRecord, float | None, int | None]]:
        """Every key with its organisation's ``disabled_at`` and ``daily_token_quota`` (the runtime cache)."""
        rows = self._conn().execute(
            "SELECT k.*, o.disabled_at AS org_disabled_at, o.daily_token_quota AS org_quota FROM sdk_keys k "
            "JOIN orgs o ON o.org_id = k.org_id").fetchall()
        return [(self._key_rec(r), r["org_disabled_at"], r["org_quota"]) for r in rows]

    def verify_key(self, key: SDKKey) -> tuple[KeyRecord, dict[str, Any]]:
        """Check a presented key (constant-time digest compare). Returns ``(record, org)``.

        Raises:
            KeyRejectedError: unknown, wrong secret, revoked, expired, wrong env or organisation disabled.
        """
        hx = key.key_id.hex()
        r = self._conn().execute(
            "SELECT k.*, o.disabled_at AS org_disabled_at, o.daily_token_quota AS org_quota, o.name AS org_name "
            "FROM sdk_keys k JOIN orgs o ON o.org_id = k.org_id WHERE k.key_id = ?", (hx,)).fetchone()
        stored = bytes(r["digest"]) if r is not None else b"\x00" * 32
        ok = hmac.compare_digest(stored, secret_digest(key.key_id, key.secret))
        if r is None:
            raise KeyRejectedError("unknown_key", hx)
        rec = self._key_rec(r)
        if not ok:
            raise KeyRejectedError("bad_secret", hx, rec.org_id)
        st = rec.state(self.clock())
        if st != "active":
            raise KeyRejectedError(st, hx, rec.org_id)
        if key.env != rec.env:
            raise KeyRejectedError("wrong_env", hx, rec.org_id)
        if r["org_disabled_at"] is not None:
            raise KeyRejectedError("org_disabled", hx, rec.org_id)
        return rec, {"org_id": rec.org_id, "name": r["org_name"], "daily_token_quota": r["org_quota"]}

    def revoke_key(self, key_id: str, *, actor: str | None = None) -> tuple[KeyRecord, list[str]]:
        """Revoke a key and its installations; returns ``(record, token ids of its installations)``.

        Idempotent (a revoked key is returned unchanged with its token ids, so a retry can finish the cascade).
        """
        hx = _key_hex(key_id)
        now = self.clock()
        with self._tx() as c:
            r = c.execute("SELECT org_id, revoked_at FROM sdk_keys WHERE key_id = ?", (hx,)).fetchone()
            if r is None:
                raise _not_found(f"no SDK key {key_id!r}")
            tokens = [t[0] for t in c.execute("SELECT token_id FROM installations WHERE key_id = ?", (hx,))]
            if r["revoked_at"] is None:
                c.execute("UPDATE sdk_keys SET revoked_at = ? WHERE key_id = ?", (now, hx))
                c.execute("UPDATE installations SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL", (now, hx))
                self.audit("key.revoked", org_id=r["org_id"], key_hex=hx, actor=actor, conn=c,
                           detail={"installations": len(tokens)})
        return self.get_key(hx), tokens

    def touch_keys(self, key_hexes: list[str], ts: float) -> None:
        if not key_hexes:
            return
        with self._tx() as c:
            c.executemany("UPDATE sdk_keys SET last_used_at = MAX(COALESCE(last_used_at, 0), ?) WHERE key_id = ?",
                          [(ts, k) for k in key_hexes])

    # ---- installations ------------------------------------------------------------------------------
    def installation_counts(self, key_hex: str, since: float) -> tuple[int, int]:
        """``(active installations, installations created since ``since``)``."""
        c = self._conn()
        active = c.execute("SELECT COUNT(*) FROM installations WHERE key_id = ? AND revoked_at IS NULL",
                           (key_hex,)).fetchone()[0]
        recent = c.execute("SELECT COUNT(*) FROM installations WHERE key_id = ? AND created_at >= ?",
                           (key_hex, since)).fetchone()[0]
        return int(active), int(recent)

    def add_installation(self, key_hex: str, name: str, tenant: str, token_id: str, *, cert_serial: str | None,
                         cert_cn: str | None, client: str | None) -> str:
        inst = "in_" + secrets.token_hex(6)
        with self._tx() as c:
            c.execute("INSERT INTO installations (installation_id, key_id, name, tenant, token_id, cert_serial, "
                      "cert_cn, created_at, client) VALUES (?,?,?,?,?,?,?,?,?)",
                      (inst, key_hex, name, tenant, token_id, cert_serial, cert_cn, self.clock(), client))
            c.execute("UPDATE sdk_keys SET last_used_at = ? WHERE key_id = ?", (self.clock(), key_hex))
        return inst

    def list_installations(self, org_id: str | None = None, key_id: str | None = None) -> list[dict[str, Any]]:
        q = ("SELECT i.* FROM installations i JOIN sdk_keys k ON k.key_id = i.key_id WHERE 1=1")
        args: list[Any] = []
        if org_id is not None:
            q += " AND k.org_id = ?"
            args.append(org_id)
        if key_id is not None:
            q += " AND i.key_id = ?"
            args.append(_key_hex(key_id))
        out = []
        for r in self._conn().execute(q + " ORDER BY i.created_at", args):
            d = dict(r)
            d["key_id"] = "sk_" + d["key_id"]
            d.pop("cert_cn", None)
            out.append(d)
        return out

    # ---- usage --------------------------------------------------------------------------------------
    def add_usage(self, rows: Mapping[tuple[str, str, str], Mapping[str, float]]) -> None:
        """Add counters: ``{(day, key_hex, model): {field: value}}`` (one transaction)."""
        if not rows:
            return
        sets = ", ".join(f"{f} = {f} + excluded.{f}" for f in _USAGE_FIELDS)
        # column names come from the fixed _USAGE_FIELDS tuple; every value is a bound parameter
        sql = (f"INSERT INTO usage (day, key_id, model, {', '.join(_USAGE_FIELDS)}) "  # noqa: S608
               f"VALUES (?,?,?{',?' * len(_USAGE_FIELDS)}) ON CONFLICT(day, key_id, model) DO UPDATE SET {sets}")
        with self._tx() as c:
            c.executemany(sql, [(d, k, m, *(v.get(f, 0) for f in _USAGE_FIELDS)) for (d, k, m), v in rows.items()])

    def usage(self, *, org_id: str | None = None, key_id: str | None = None, since: str | None = None,
              until: str | None = None) -> list[dict[str, Any]]:
        """Usage rows (newest day first); days are ``YYYY-MM-DD`` (UTC), inclusive."""
        for d in (since, until):
            if d is not None and not _DAY_RE.fullmatch(d):
                raise ConfigError("days are YYYY-MM-DD")
        q = "SELECT u.* FROM usage u JOIN sdk_keys k ON k.key_id = u.key_id WHERE 1=1"
        args: list[Any] = []
        if org_id is not None:
            q += " AND k.org_id = ?"
            args.append(org_id)
        if key_id is not None:
            q += " AND u.key_id = ?"
            args.append(_key_hex(key_id))
        if since is not None:
            q += " AND u.day >= ?"
            args.append(since)
        if until is not None:
            q += " AND u.day <= ?"
            args.append(until)
        out = []
        for r in self._conn().execute(q + " ORDER BY u.day DESC, u.key_id, u.model", args):
            d = dict(r)
            d["key_id"] = "sk_" + d["key_id"]
            out.append(d)
        return out

    def day_totals(self, day: str) -> dict[str, tuple[float, int]]:
        """``{key_hex: (tokens, trips)}`` for one day."""
        return {r[0]: (float(r[1]), int(r[2])) for r in self._conn().execute(
            "SELECT key_id, SUM(tokens), SUM(trips) FROM usage WHERE day = ? GROUP BY key_id", (day,))}

    # ---- admin tokens -------------------------------------------------------------------------------
    def create_admin_token(self, name: str, org_id: str | None = None, *, actor: str | None = None,
                           ) -> tuple[str, dict[str, Any]]:
        """New admin token (``pgwa-...``), global or scoped to ``org_id``; returns ``(token, record)``."""
        if not _NAME_RE.fullmatch(name or ""):
            raise ConfigError("token name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        token = ADMIN_TOKEN_PREFIX + secrets.token_urlsafe(32)
        tid = "at_" + secrets.token_hex(6)
        with self._tx() as c:
            if org_id is not None and not c.execute("SELECT 1 FROM orgs WHERE org_id = ?", (org_id,)).fetchone():
                raise _not_found(f"no organisation {org_id!r}")
            c.execute("INSERT INTO admin_tokens (token_id, org_id, name, digest, created_at) VALUES (?,?,?,?,?)",
                      (tid, org_id, name, _admin_digest(token), self.clock()))
            self.audit("admin_token.created", org_id=org_id, actor=actor, conn=c, detail={"token_id": tid,
                                                                                          "name": name})
        return token, {"token_id": tid, "org_id": org_id, "name": name}

    def list_admin_tokens(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn().execute(
            "SELECT token_id, org_id, name, created_at, revoked_at FROM admin_tokens ORDER BY created_at")]

    def revoke_admin_token(self, token_id: str, *, actor: str | None = None) -> dict[str, Any]:
        with self._tx() as c:
            r = c.execute("SELECT org_id FROM admin_tokens WHERE token_id = ?", (token_id,)).fetchone()
            if r is None:
                raise _not_found(f"no admin token {token_id!r}")
            c.execute("UPDATE admin_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE token_id = ?",
                      (self.clock(), token_id))
            self.audit("admin_token.revoked", org_id=r["org_id"], actor=actor, conn=c, detail={"token_id": token_id})
        return {"token_id": token_id, "revoked": True}

    def admin_for_token(self, token: str) -> dict[str, Any] | None:
        """The active admin token record for ``token`` (lookup by its SHA-256), or ``None``."""
        r = self._conn().execute("SELECT token_id, org_id, name, digest FROM admin_tokens WHERE digest = ? AND "
                                 "revoked_at IS NULL", (_admin_digest(token),)).fetchone()
        if r is None or not hmac.compare_digest(bytes(r["digest"]), _admin_digest(token)):
            return None
        return {"token_id": r["token_id"], "org_id": r["org_id"], "name": r["name"]}


def _check_quota(v: Any, what: str) -> None:
    if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
        raise ConfigError(f"{what} must be a positive integer")


def _key_hex(key_id: str) -> str:
    if isinstance(key_id, str) and re.fullmatch(r"[0-9a-f]{16}", key_id):
        return key_id
    return parse_key_id(key_id).hex()


def _not_found(msg: str) -> PGWError:
    return PGWError(msg, code=ErrorCode.MODEL_NOT_FOUND)  # 404


def revoke_key_everywhere(store: AccountsStore, key_id: str, tokens_file: Path | None, *,
                          actor: str | None = None) -> dict[str, Any]:
    """Revoke an SDK key, its installations and their service tokens (removed from ``tokens_file``).

    Running services refuse the key's tenant within their refresh interval and drop the tokens at their next
    tokens-file reload (and at once in the process that calls this through :meth:`Accounts.revoke_key`).
    """
    from pgw.service.auth import revoke_token

    rec, token_ids = store.revoke_key(key_id, actor=actor)
    removed = 0
    if tokens_file is not None:
        for tid in token_ids:
            try:
                revoke_token(tokens_file, tid)
                removed += 1
            except ConfigError:  # already removed (a retried revocation) or never written
                pass
    return {**rec.public(), "tokens_removed": removed}


# -------------------------------------------------------------------------------------------------- runtime
@dataclass
class _ModelInfo:
    name: str
    ref_kind: str
    ref_count: int


@dataclass
class _KeyState:
    rec: KeyRecord
    org_disabled: bool
    org_quota: int | None
    allowed: frozenset[str] | None
    bucket: Any = None
    over_quota_logged: str = ""

    def active(self, now: float) -> bool:
        return self.rec.state(now) == "active" and not self.org_disabled


@dataclass
class _Pending:
    counters: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)

    def add(self, key: tuple[str, str, str], **kw: float) -> None:
        d = self.counters.get(key)
        if d is None:
            d = self.counters[key] = dict.fromkeys(_USAGE_FIELDS, 0)
        for k, v in kw.items():
            d[k] += v


class Accounts:
    """The accounts runtime of one service process (see the module docstring).

    Args:
        store: The database.
        settings: :class:`pgw.service.config.ServiceSettings` (``accounts_*``, ``sdk_*``, ``tokens_file``).
        registry: The model registry (fingerprint -> model name; token accounting).
        authenticator: The service's authenticator (revocations take effect there at once).
        clock: Wall clock (tests).
    """

    def __init__(self, store: AccountsStore, settings: Any, *, registry: Any = None, authenticator: Any = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.settings = settings
        self.registry = registry
        self.authenticator = authenticator
        self.clock = clock
        self.enroll_service: Any = None  # pgw.enroll.service.EnrollmentService, set by add_sdk_routes
        self.ca_sha256: bytes | None = None
        self._lock = threading.RLock()
        self._enroll_lock = threading.Lock()
        self._keys: dict[str, _KeyState] = {}  # tenant -> state
        self._models: dict[str, _ModelInfo | None] = {}  # fingerprint -> info
        self._pending = _Pending()
        self._flushing = _Pending()
        self._day = day_of(clock())
        self._db_today: dict[str, tuple[float, int]] = {}  # key_hex -> (tokens, trips) flushed today
        self._lane_models: dict[str, set[str]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.refresh_s = float(getattr(settings, "accounts_refresh_s", 2.0))
        self.flush_s = float(getattr(settings, "accounts_flush_s", 5.0))
        self.refresh()

    # ---- lifecycle ----------------------------------------------------------------------------------
    def start(self) -> None:
        """Start the refresh/flush thread."""
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="pgw-accounts", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the thread and flush pending usage."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.flush()

    def _loop(self) -> None:
        next_flush = time.monotonic() + self.flush_s
        while not self._stop.wait(self.refresh_s):
            try:
                if time.monotonic() >= next_flush:
                    self.flush()
                    next_flush = time.monotonic() + self.flush_s
                self.refresh()
            except Exception as exc:  # keep metering alive; the next round retries
                log.error("accounts refresh/flush failed: %s: %s", type(exc).__name__, exc)
        self.store.close()

    def refresh(self) -> None:
        """Reload keys, organisations and today's flushed totals from the database."""
        now = self.clock()
        states = self.store.active_key_states()
        today = day_of(now)
        totals = self.store.day_totals(today)
        with self._lock:
            new: dict[str, _KeyState] = {}
            for rec, org_disabled, org_quota in states:
                t = rec.tenant
                old = self._keys.get(t)
                st = _KeyState(rec, org_disabled is not None, org_quota,
                               frozenset(rec.models) if rec.models else None)
                if old is not None and (old.rec.rows_per_s, old.rec.burst_rows) == (rec.rows_per_s, rec.burst_rows):
                    st.bucket = old.bucket
                    st.over_quota_logged = old.over_quota_logged
                if st.bucket is None and rec.rows_per_s:
                    from pgw.service.handler import TokenBucket

                    st.bucket = TokenBucket(rec.rows_per_s, rec.burst_rows or max(1, math.ceil(2 * rec.rows_per_s)))
                if old is not None and not old.active(now):  # a revocation seen here stays (no resurrection)
                    st.rec.revoked_at = st.rec.revoked_at or old.rec.revoked_at
                new[t] = st
            self._keys = new
            if today != self._day:
                self._day = today
            self._db_today = totals

    def flush(self) -> None:
        """Write pending usage to the database (called by the thread every ``flush_s`` and at stop)."""
        with self._lock:
            pend, self._pending = self._pending, _Pending()
            self._flushing = pend
        if not pend.counters:
            return
        try:
            self.store.add_usage(pend.counters)
            self.store.touch_keys(sorted({k for _, k, _ in pend.counters}), self.clock())
        except Exception:
            with self._lock:  # keep the counts for the next flush
                for k, v in pend.counters.items():
                    self._pending.add(k, **v)
                self._flushing = _Pending()
            raise
        with self._lock:
            self._db_today = self.store.day_totals(self._day)
            self._flushing = _Pending()

    # ---- model info ---------------------------------------------------------------------------------
    def _model(self, fingerprint: str) -> _ModelInfo | None:
        info = self._models.get(fingerprint, False)
        if info is not False:
            return info  # type: ignore[return-value]
        try:
            m = self.registry.get(fingerprint) if self.registry is not None else None
        except PGWError:
            m = None
        if m is None:
            return None  # not cached: the pack may still be loading
        kinds = [s.id.key.split(".", 1)[1] for s in m.specs]
        ref = kinds[0] if kinds else ""
        out = _ModelInfo(str(m.name), ref, max(1, kinds.count(ref)))
        self._models[fingerprint] = out
        return out

    # ---- gate (authenticator) -----------------------------------------------------------------------
    def check(self, tenant: str) -> None:
        """Authentication gate: raise for an SDK tenant whose key is revoked/expired/unknown or org disabled."""
        if not tenant.startswith(SDK_TENANT_PREFIX):
            return
        st = self._keys.get(tenant)
        if st is None or not st.active(self.clock()):
            raise AuthError("credentials revoked or expired")

    def active(self, tenant: str) -> bool:
        """Validity re-check of long-lived channels (stream, lane)."""
        if not tenant.startswith(SDK_TENANT_PREFIX):
            return True
        st = self._keys.get(tenant)
        return st is not None and st.active(self.clock())

    # ---- policy (handler, lane) ---------------------------------------------------------------------
    def _today_tokens(self, key_hex: str) -> float:
        with self._lock:
            tok = self._db_today.get(key_hex, (0.0, 0))[0]
            for pend in (self._pending, self._flushing):
                for (d, k, _m), v in pend.counters.items():
                    if k == key_hex and d == self._day:
                        tok += v["tokens"]
        return tok

    def _org_tokens(self, org_id: str) -> float:
        return sum(self._today_tokens(st.rec.key_hex) for st in self._keys.values() if st.rec.org_id == org_id)

    def over_quota(self, tenant: str) -> str | None:
        """``"key"`` / ``"org"`` when today's token quota of the key / its organisation is used up, else ``None``."""
        st = self._keys.get(tenant)
        if st is None:
            return None
        q = st.rec.daily_token_quota
        if q is not None and self._today_tokens(st.rec.key_hex) >= q:
            return "key"
        if st.org_quota is not None and self._org_tokens(st.rec.org_id) >= st.org_quota:
            return "org"
        return None

    def admit(self, tenant: str, fingerprint: str, rows: int) -> None:
        """Per-call policy of an SDK tenant (others pass at once).

        Raises:
            AuthError: key revoked/expired (also checked at authentication; here for long-lived channels).
            PGWError(permission_denied): the model is outside the key's scope.
            RateLimitedError: the key's row rate, or the key's or organisation's daily token quota.
        """
        if not tenant.startswith(SDK_TENANT_PREFIX):
            return
        st = self._keys.get(tenant)
        now = self.clock()
        if st is None or not st.active(now):
            raise AuthError("credentials revoked or expired")
        if st.allowed is not None:
            info = self._model(fingerprint)
            if info is None or info.name not in st.allowed:
                raise PGWError("this SDK key is not allowed to use this model", code=ErrorCode.PERMISSION_DENIED)
        if st.rec.daily_token_quota is not None or st.org_quota is not None:
            which = self.over_quota(tenant)
            if which is not None:
                if st.over_quota_logged != self._day + which:
                    st.over_quota_logged = self._day + which
                    self.store_audit_async("quota.exceeded", st, {"quota": which, "day": self._day})
                raise RateLimitedError(f"daily token quota of this {'SDK key' if which == 'key' else 'organisation'} "
                                       "is used up", details={"retry_after_s": round(_seconds_to_midnight(now)),
                                                              "limit": f"{which}_daily_tokens"})
        if st.bucket is not None and rows > 0:
            wait = st.bucket.take(rows)
            if wait > 0:
                raise RateLimitedError("SDK key row rate limit exceeded", details={"retry_after_s": round(wait, 3)})

    def store_audit_async(self, event: str, st: _KeyState, detail: dict[str, Any]) -> None:
        threading.Thread(target=lambda: self._audit_safely(event, st, detail), daemon=True).start()

    def _audit_safely(self, event: str, st: _KeyState, detail: dict[str, Any]) -> None:
        try:
            self.store.audit(event, org_id=st.rec.org_id, key_hex=st.rec.key_hex, detail=detail)
        finally:
            self.store.close()

    def _add(self, tenant: str, fingerprint: str | None, **kw: float) -> None:
        st = self._keys.get(tenant)
        if st is None:
            return
        info = self._model(fingerprint) if fingerprint else None
        name = info.name if info is not None else "(unattributed)"
        day = day_of(self.clock())
        with self._lock:
            if day != self._day:  # new UTC day: yesterday's flushed totals no longer count against the quota
                self._day = day
                self._db_today = {}
            self._pending.add((day, st.rec.key_hex, name), **kw)

    def record_rows(self, tenant: str, fingerprint: str, kinds: Mapping[str, int], bytes_in: int = 0,
                    bytes_out: int = 0, trips: int = 1) -> None:
        """Count one compute call: ``kinds`` = rows per linear kind."""
        if not tenant.startswith(SDK_TENANT_PREFIX):
            return
        info = self._model(fingerprint)
        rows = sum(kinds.values())
        tokens = kinds.get(info.ref_kind, 0) / info.ref_count if info is not None else 0.0
        self._add(tenant, fingerprint, trips=trips, rows=rows, tokens=tokens, bytes_in=bytes_in, bytes_out=bytes_out)

    def record_maskgen(self, tenant: str, fingerprint: str, cols: int, bytes_in: int = 0, bytes_out: int = 0) -> None:
        """Count one encrypted maskgen request (``cols`` ciphertext columns over all items)."""
        if not tenant.startswith(SDK_TENANT_PREFIX):
            return
        self._add(tenant, fingerprint, maskgen_cols=cols, maskgen_requests=1, bytes_in=bytes_in, bytes_out=bytes_out)

    def lane_opened(self, tenant: str, fingerprint: str) -> None:
        """A lane ticket for ``fingerprint`` was issued to ``tenant`` (attribution of the lane's counters)."""
        if tenant.startswith(SDK_TENANT_PREFIX):
            with self._lock:
                self._lane_models.setdefault(tenant, set()).add(fingerprint)

    def record_lane(self, drain: Mapping[str, Any]) -> list[str]:
        """Fold a lane counter drain (``pgw_fast.LaneService.drain_stats``) into usage; returns the SDK tenants now
        over quota or revoked (their lane connections should be closed).

        The lane counts per tenant and linear kind, not per model: a tenant's lane usage is attributed to the
        model of its lane tickets when it opened tickets for one model, else to ``(unattributed)``.
        """
        trips: dict[str, int] = {}
        rows: dict[str, dict[str, int]] = {}
        io: dict[str, tuple[int, int]] = {}
        for tenant, status, _b, _t, count in drain.get("requests", ()):
            if tenant.startswith(SDK_TENANT_PREFIX) and status == "ok":
                trips[tenant] = trips.get(tenant, 0) + int(count)
        for tenant, kind, n in drain.get("rows", ()):
            if tenant.startswith(SDK_TENANT_PREFIX):
                rows.setdefault(tenant, {})[kind] = rows.get(tenant, {}).get(kind, 0) + int(n)
        for tenant, bin_, bout in drain.get("bytes", ()):
            if tenant.startswith(SDK_TENANT_PREFIX):
                a, b = io.get(tenant, (0, 0))
                io[tenant] = (a + int(bin_), b + int(bout))
        for tenant in set(trips) | set(rows) | set(io):
            fps = self._lane_models.get(tenant) or set()
            fp = next(iter(fps)) if len(fps) == 1 else None
            kinds = rows.get(tenant, {})
            info = self._model(fp) if fp else None
            tokens = kinds.get(info.ref_kind, 0) / info.ref_count if info is not None else 0.0
            bi, bo = io.get(tenant, (0, 0))
            self._add(tenant, fp, trips=trips.get(tenant, 0), rows=sum(kinds.values()), tokens=tokens, bytes_in=bi,
                      bytes_out=bo)
        with self._lock:
            dead = [t for t in self._lane_models if not self.active(t) or self.over_quota(t)]
            for t in dead:  # reported once; a new ticket (refused while the reason lasts) adds the tenant again
                self._lane_models.pop(t, None)
        return dead

    # ---- keys (runtime side) ------------------------------------------------------------------------
    def revoke_key(self, key_id: str, *, actor: str | None = None) -> dict[str, Any]:
        """Revoke a key everywhere and refuse its tenant in this process at once."""
        out = revoke_key_everywhere(self.store, key_id, getattr(self.settings, "tokens_file", None), actor=actor)
        t = tenant_for_key(_key_hex(key_id))
        with self._lock:
            st = self._keys.get(t)
            if st is not None:
                st.rec.revoked_at = st.rec.revoked_at or self.clock()
        if self.authenticator is not None:
            with contextlib.suppress(Exception):
                self.authenticator.maybe_reload(force=True)
        return out

    def limits_view(self, rec: KeyRecord, org: Mapping[str, Any]) -> dict[str, Any]:
        return {**rec.limits(), "org_daily_token_quota": org.get("daily_token_quota")}

    # ---- SDK enrollment -----------------------------------------------------------------------------
    def enroll(self, authorization: str | None, body: Any, client: str) -> dict[str, Any]:
        """``POST /pgw/v1/sdk/enroll`` (blocking: run in a worker thread). See contract appendix A2.

        Raises:
            RateLimitedError, KeyRejectedError (401), PGWError(permission_denied / protocol / unavailable).
        """
        svc = self.enroll_service
        assert svc is not None
        return svc.limiter.guarded(client, lambda: self._enroll(authorization, body, client))

    def _key_from_header(self, authorization: str | None, client: str) -> SDKKey:
        scheme, _, token = (authorization or "").strip().partition(" ")
        try:
            if scheme.lower() != "bearer" or not token.strip():
                raise SDKKeyError("missing SDK key")
            return SDKKey.parse(token.strip())
        except SDKKeyError as exc:
            self.store.audit("sdk.enroll.failed", client=client, detail={"reason": "malformed_key"})
            raise KeyRejectedError("malformed_key") from exc

    def authenticate_key(self, authorization: str | None, client: str, event: str) -> tuple[KeyRecord, dict[str, Any]]:
        """Verify the SDK key in ``authorization``; audits failures as ``event``.

        Raises:
            KeyRejectedError: (401) for every kind of bad key.
        """
        key = self._key_from_header(authorization, client)
        try:
            return self.store.verify_key(key)
        except KeyRejectedError as exc:
            self.store.audit(event, org_id=exc.org_id, key_hex=exc.key_hex, client=client,
                             detail={"reason": exc.reason})
            raise

    def _enroll(self, authorization: str | None, body: Any, client: str) -> dict[str, Any]:
        from pgw.enroll.service import check_csr

        rec, org = self.authenticate_key(authorization, client, "sdk.enroll.failed")

        def fail(reason: str, exc: PGWError, **detail: Any) -> PGWError:
            self.store.audit("sdk.enroll.failed", org_id=rec.org_id, key_hex=rec.key_hex, client=client,
                             detail={"reason": reason, **detail})
            return exc

        if not isinstance(body, dict):
            raise PGWError("body must be a JSON object", code=ErrorCode.PROTOCOL)
        name = body.get("installation_name", body.get("gateway_name"))
        csr, want = body.get("csr_pem"), body.get("models")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise PGWError("installation_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}", code=ErrorCode.PROTOCOL)
        if csr is not None and (not isinstance(csr, str) or len(csr) > 16 << 10):
            raise PGWError("csr_pem must be a PEM string", code=ErrorCode.PROTOCOL)
        if want is not None and (not isinstance(want, list) or len(want) > 64
                                 or not all(isinstance(m, str) and _MODEL_RE.fullmatch(m) for m in want)):
            raise PGWError("models must be a list of model names", code=ErrorCode.PROTOCOL)
        s = self.settings
        sign = isinstance(csr, str) and bool(csr) and s.client_issuing_ca_cert_file is not None
        if sign:
            check_csr(str(csr))
        scope = set(rec.models)
        wanted = sorted(set(want or []))
        if scope and wanted and not set(wanted) <= scope:
            raise fail("model_scope", PGWError("a requested model is outside this SDK key's scope",
                                               code=ErrorCode.PERMISSION_DENIED), models=wanted)
        allowed = wanted or sorted(scope)
        now = self.clock()
        tenant = rec.tenant
        quota_which = None
        if rec.daily_token_quota is not None and self._today_tokens(rec.key_hex) >= rec.daily_token_quota:
            quota_which = "key"
        elif org.get("daily_token_quota") is not None and self._org_tokens(rec.org_id) >= org["daily_token_quota"]:
            quota_which = "org"
        if quota_which:
            raise fail("quota", RateLimitedError("today's token quota is used up", details={
                "retry_after_s": round(_seconds_to_midnight(now)), "limit": f"{quota_which}_daily_tokens"}))
        per_hour = int(getattr(s, "sdk_enrolls_per_key_per_hour", 60))
        with self._enroll_lock:
            active, recent = self.store.installation_counts(rec.key_hex, now - 3600.0)
            if recent >= per_hour:
                raise fail("enroll_rate", RateLimitedError("too many enrollments with this SDK key; try again later",
                                                           details={"retry_after_s": 60.0, "limit": "sdk_enroll"}))
            if rec.max_installations is not None and active >= rec.max_installations:
                raise fail("install_limit", RateLimitedError(
                    "this SDK key has reached its installation limit; revoke it and create a new key, or ask for a "
                    "higher limit", details={"retry_after_s": 3600.0, "limit": "max_installations"}))
            models = self.enroll_service._model_entries(allowed)
            if not models:
                raise PGWError("none of this SDK key's models is served right now; try again later",
                               code=ErrorCode.UNAVAILABLE)
            # the key must be known to the gate before its first compute call
            with self._lock:
                if tenant not in self._keys:
                    self._keys[tenant] = _KeyState(rec, False, org.get("daily_token_quota"),
                                                   frozenset(rec.models) if rec.models else None)
            out, trec, serial, cn = self.enroll_service.issue_credentials(tenant, name, csr if sign else None)
            inst = self.store.add_installation(rec.key_hex, name, tenant, trec.token_id,
                                               cert_serial=f"{serial:x}" if serial is not None else None, cert_cn=cn,
                                               client=client)
        self.store.audit("sdk.enroll.ok", org_id=rec.org_id, key_hex=rec.key_hex, client=client,
                         detail={"installation_id": inst, "name": name, "token_id": trec.token_id,
                                 "cert_serial": f"{serial:x}" if serial is not None else None,
                                 "models": [m["name"] for m in models]})
        out["models"] = models
        out["sdk"] = {"key_id": rec.key_id, "org_id": rec.org_id, "installation_id": inst, "env": rec.env,
                      "models_scope": list(rec.models), "limits": self.limits_view(rec, org),
                      "expires_at": rec.expires_at}
        return out

    def usage_for_key(self, authorization: str | None, client: str, days: int) -> dict[str, Any]:
        """``GET /pgw/v1/sdk/usage`` (blocking)."""
        svc = self.enroll_service
        assert svc is not None
        return svc.limiter.guarded(client, lambda: self._usage_for_key(authorization, client, days))

    def _usage_for_key(self, authorization: str | None, client: str, days: int) -> dict[str, Any]:
        rec, org = self.authenticate_key(authorization, client, "sdk.usage.failed")
        now = self.clock()
        since = day_of(now - (days - 1) * 86400.0)
        rows = self.store.usage(key_id=rec.key_hex, since=since)
        for r in rows:
            r.pop("key_id", None)
        tok_trips = self._db_today.get(rec.key_hex, (0.0, 0))
        return {"key_id": rec.key_id, "org_id": rec.org_id, "state": rec.state(now), "models_scope": list(rec.models),
                "limits": self.limits_view(rec, org),
                "today": {"tokens": round(self._today_tokens(rec.key_hex), 3), "trips": tok_trips[1]},
                "usage": rows}


# -------------------------------------------------------------------------------------------------- routes
def _json_error(exc: PGWError) -> Any:
    from starlette.responses import JSONResponse

    headers = {"Cache-Control": "no-store"}
    retry = exc.details.get("retry_after_s")
    if isinstance(retry, int | float):
        headers["Retry-After"] = str(max(1, round(retry)))
    msg = "internal error" if exc.code == ErrorCode.INTERNAL else exc.message
    return JSONResponse({"error": {"code": str(exc.code), "message": msg}}, status_code=exc.http_status,
                        headers=headers)


async def _read_json(request: Any) -> Any:
    length = request.headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > _MAX_BODY):
        raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        raise PGWError("request body too large", code=ErrorCode.PAYLOAD_TOO_LARGE)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise PGWError("body must be JSON", code=ErrorCode.PROTOCOL) from exc


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def route(request: Any) -> Any:
        from starlette.responses import JSONResponse

        try:
            out = await fn(request)
        except PGWError as exc:
            return _json_error(exc)
        except Exception as exc:  # never leak details
            log.error("accounts route %s failed: %s", request.url.path, type(exc).__name__)
            return _json_error(PGWError("internal error"))
        return JSONResponse(out, headers={"Cache-Control": "no-store"})

    return route


def add_sdk_routes(app: Any, accounts: Accounts, enroll_service: Any) -> None:
    """``POST /pgw/v1/sdk/enroll`` and ``GET /pgw/v1/sdk/usage``."""
    import asyncio

    from pgw.core.client_ip import client_ip

    accounts.enroll_service = enroll_service

    async def post_enroll(request: Any) -> Any:
        body = await _read_json(request)
        return await asyncio.to_thread(accounts.enroll, request.headers.get("authorization"), body,
                                       client_ip(request))

    async def get_usage(request: Any) -> Any:
        raw = request.query_params.get("days", "31")
        if not raw.isdigit() or not 1 <= int(raw) <= 92:
            raise PGWError("days must be 1..92", code=ErrorCode.PROTOCOL)
        return await asyncio.to_thread(accounts.usage_for_key, request.headers.get("authorization"),
                                       client_ip(request), int(raw))

    app.add_route(SDK_ENROLL_PATH, _guard(post_enroll), methods=["POST"], include_in_schema=False)
    app.add_route(SDK_USAGE_PATH, _guard(get_usage), methods=["GET"], include_in_schema=False)


def add_admin_routes(app: Any, accounts: Accounts) -> None:
    """The admin API (contract appendix A4). Only registered with ``accounts_admin_api``."""
    import asyncio

    from pgw.core.client_ip import client_ip

    store = accounts.store
    s = accounts.settings
    boot = [bytes.fromhex(h) for h in (getattr(s, "accounts_admin_token_sha256", None) or [])]

    def admin(request: Any) -> dict[str, Any]:
        scheme, _, token = (request.headers.get("authorization") or "").strip().partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise AuthError("admin token required")
        digest = _admin_digest(token)
        hit = False
        for h in boot:  # constant time over the bootstrap hashes
            hit |= hmac.compare_digest(h, digest)
        if hit:
            return {"token_id": "bootstrap", "org_id": None}
        rec = store.admin_for_token(token)
        if rec is None:
            store.audit("admin.denied", client=client_ip(request), detail={"reason": "bad_token"})
            raise AuthError("invalid admin token")
        return rec

    def scoped(who: Mapping[str, Any], org_id: str, request: Any) -> None:
        if who.get("org_id") is not None and who["org_id"] != org_id:
            store.audit("admin.denied", org_id=org_id, actor=str(who.get("token_id")), client=client_ip(request),
                        detail={"reason": "org_scope"})
            raise PGWError("this admin token is limited to another organisation", code=ErrorCode.PERMISSION_DENIED)

    def run(fn: Callable[[], Any]) -> Any:
        return asyncio.to_thread(fn)  # worker threads keep their own SQLite connection

    def actor(who: Mapping[str, Any]) -> str:
        return f"admin:{who.get('token_id')}"

    async def orgs(request: Any) -> Any:
        who = await run(lambda: admin(request))
        if request.method == "POST":
            if who.get("org_id") is not None:
                raise PGWError("only a global admin token can create organisations", code=ErrorCode.PERMISSION_DENIED)
            b = await _read_json(request)
            if not isinstance(b, dict):
                raise PGWError("body must be a JSON object", code=ErrorCode.PROTOCOL)
            return await run(lambda: store.create_org(str(b.get("org_id", "")), b.get("name"),
                                                      daily_token_quota=b.get("daily_token_quota"), actor=actor(who)))
        all_orgs = await run(store.list_orgs)
        if who.get("org_id") is not None:
            all_orgs = [o for o in all_orgs if o["org_id"] == who["org_id"]]
        return {"orgs": all_orgs}

    async def org(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid = request.path_params["org"]
        scoped(who, oid, request)
        return await run(lambda: store.get_org(oid))

    async def keys(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid = request.path_params["org"]
        scoped(who, oid, request)
        if request.method == "POST":
            b = await _read_json(request)
            if not isinstance(b, dict):
                raise PGWError("body must be a JSON object", code=ErrorCode.PROTOCOL)
            models = b.get("models") or []
            if not isinstance(models, list):
                raise PGWError("models must be a list", code=ErrorCode.PROTOCOL)

            def create() -> dict[str, Any]:
                key, rec = store.create_key(
                    oid, str(b.get("name", "")), str(s.public_url), accounts.ca_sha256,
                    env=str(b.get("env") or s.sdk_key_env), models=[str(m) for m in models],
                    expires_in_s=b.get("expires_in_s"), daily_token_quota=b.get("daily_token_quota"),
                    rows_per_s=b.get("rows_per_s"), burst_rows=b.get("burst_rows"),
                    max_installations=b.get("max_installations"), actor=actor(who))
                accounts.refresh()
                return {"key": key.encode(), "record": rec.public()}

            return await run(create)
        return {"keys": [k.public() for k in await run(lambda: store.list_keys(oid))]}

    async def revoke(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid, kid = request.path_params["org"], request.path_params["key_id"]
        scoped(who, oid, request)

        def go() -> dict[str, Any]:
            rec = store.get_key(kid)
            if rec.org_id != oid:
                raise _not_found(f"no SDK key {kid!r} in {oid!r}")
            return accounts.revoke_key(kid, actor=actor(who))

        return await run(go)

    async def installations(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid = request.path_params["org"]
        scoped(who, oid, request)
        kid = request.query_params.get("key_id")
        return {"installations": await run(lambda: store.list_installations(oid, kid))}

    async def usage(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid = request.path_params["org"]
        scoped(who, oid, request)
        q = request.query_params
        return {"usage": await run(lambda: store.usage(org_id=oid, key_id=q.get("key_id"), since=q.get("since"),
                                                         until=q.get("until")))}

    async def audit(request: Any) -> Any:
        who = await run(lambda: admin(request))
        oid = request.path_params["org"]
        scoped(who, oid, request)
        q = request.query_params
        lim, before = q.get("limit", "100"), q.get("before")
        if not lim.isdigit() or (before is not None and not before.isdigit()):
            raise PGWError("limit and before must be integers", code=ErrorCode.PROTOCOL)
        return {"events": await run(lambda: store.audit_events(oid, limit=int(lim),
                                                               before=int(before) if before else None))}

    async def attest(request: Any) -> Any:
        b = await _read_json(request)
        nonce = b.get("nonce") if isinstance(b, dict) else None
        if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce):
            raise PGWError("nonce must be 64 lowercase hex digits", code=ErrorCode.PROTOCOL)
        ca = accounts.ca_sha256 or bytes(32)
        return {"ca_sha256": ca.hex(), "macs": [attest_mac(h, bytes.fromhex(nonce), ca).hex() for h in boot]}

    p = ADMIN_PREFIX
    app.add_route(f"{p}/attest", _guard(attest), methods=["POST"], include_in_schema=False)
    app.add_route(f"{p}/orgs", _guard(orgs), methods=["GET", "POST"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}", _guard(org), methods=["GET"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}/keys", _guard(keys), methods=["GET", "POST"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}/keys/{key_id}/revoke", _guard(revoke), methods=["POST"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}/installations", _guard(installations), methods=["GET"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}/usage", _guard(usage), methods=["GET"], include_in_schema=False)
    app.add_route(p + "/orgs/{org}/audit", _guard(audit), methods=["GET"], include_in_schema=False)


def format_key_id_hex(key_hex: str) -> str:
    """``sk_`` form of a stored key hex."""
    return format_key_id(bytes.fromhex(key_hex))
