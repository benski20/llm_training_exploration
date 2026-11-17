"""API keys, rate limiting and quotas for the gateway.

Keys look like ``pgw-<key_id>-<secret>``: ``key_id`` is 12 lowercase hex characters (public, used in
logs and the audit trail), ``secret`` is 43 url-safe base64 characters (256 bits). Only
``SHA-256("pgw-api-key-v1\\0" || key_id || "\\0" || secret)`` is stored; verification is a lookup
by ``key_id`` plus a constant-time hash comparison.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pgw.core.errors import AuthError, ConfigError, RateLimitedError

__all__ = [
    "SCOPES", "ApiKey", "ApiKeyStore", "KeyNotFoundError", "QuotaTracker", "RateLimiter", "hash_secret",
    "parse_api_key",
]

log = logging.getLogger(__name__)

#: Known scopes. ``inference`` = OpenAI API, ``admin`` = ``/admin/*`` (alternative to the admin token).
SCOPES: frozenset[str] = frozenset({"inference", "admin"})
_KEY_RE = re.compile(r"^pgw-([0-9a-f]{12})-([A-Za-z0-9_-]{43})$")
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_FILE_VERSION = 1


class KeyNotFoundError(ConfigError):
    """Unknown ``key_id`` in an admin operation (HTTP 404 at the admin API)."""


def hash_secret(key_id: str, secret: str) -> str:
    """Stored form of a key secret (hex SHA-256, domain-separated and bound to ``key_id``)."""
    h = hashlib.sha256()
    h.update(b"pgw-api-key-v1\0")
    h.update(key_id.encode())
    h.update(b"\0")
    h.update(secret.encode())
    return h.hexdigest()


def parse_api_key(presented: str) -> tuple[str, str]:
    """Split ``pgw-<key_id>-<secret>``; raises :class:`AuthError` on malformed keys."""
    m = _KEY_RE.match(presented.strip())
    if m is None:
        raise AuthError("malformed API key")
    return m.group(1), m.group(2)


@dataclass(frozen=True)
class ApiKey:
    """Stored API key record (only the SHA-256 hash of the secret is stored)."""

    key_id: str
    hashed: str
    name: str
    scopes: tuple[str, ...] = ("inference",)
    models: tuple[str, ...] = ()  # empty = all
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    created_at: str = ""
    disabled: bool = False
    meta: dict[str, str] = field(default_factory=dict)
    daily_token_quota: int | None = None
    max_concurrent_requests: int | None = None

    def allows_model(self, model: str) -> bool:
        """Whether this key may use ``model``."""
        return not self.models or model in self.models

    def has_scope(self, scope: str) -> bool:
        """Whether this key carries ``scope``."""
        return scope in self.scopes

    def public(self) -> dict[str, Any]:
        """Record without the hash (admin listing)."""
        d = asdict(self)
        d.pop("hashed", None)
        d["scopes"] = list(self.scopes)
        d["models"] = list(self.models)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ApiKey:
        """Inverse of :func:`dataclasses.asdict` (unknown keys ignored)."""
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        kw["scopes"] = tuple(kw.get("scopes", ("inference",)))
        kw["models"] = tuple(kw.get("models", ()))
        kw["meta"] = {str(k): str(v) for k, v in dict(kw.get("meta") or {}).items()}
        return cls(**kw)


#: Anonymous identity used only with ``allow_anonymous`` in development.
ANONYMOUS_KEY = ApiKey(key_id="anonymous000", hashed="", name="anonymous", scopes=("inference",))


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically with mode 0600 (temp file + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class ApiKeyStore:
    """File-backed key store. Keys look like ``pgw-<key_id>-<secret>``; verification is by
    ``key_id`` lookup + constant-time hash comparison.

    The file (JSON, mode 0600) is re-read when its modification time changes, so keys created or
    revoked with ``pgw gateway keys`` take effect in a running gateway. ``path=None`` keeps the keys
    in memory only (tests, ephemeral setups).
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._keys: dict[str, ApiKey] = {}
        self._mtime: float | None = None
        if self.path is not None:
            if self.path.exists():
                self._load()
            else:
                log.info("API key file %s does not exist yet; starting with no keys", self.path)

    # ---- persistence ------------------------------------------------------------------------------
    def _load(self) -> None:
        assert self.path is not None
        try:
            st = self.path.stat()
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._keys, self._mtime = {}, None
            return
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read API key file {self.path.name}: {type(exc).__name__}") from exc
        if st.st_mode & 0o077:
            log.warning("API key file %s is readable by group/others; use mode 0600", self.path)
        if not isinstance(raw, dict) or raw.get("version") != _FILE_VERSION or not isinstance(raw.get("keys"), list):
            raise ConfigError(f"API key file {self.path.name} has an unsupported format")
        keys: dict[str, ApiKey] = {}
        for entry in raw["keys"]:
            k = ApiKey.from_dict(entry)
            if not _ID_RE.match(k.key_id):
                raise ConfigError("API key file contains an invalid key_id")
            keys[k.key_id] = k
        self._keys = keys
        self._mtime = st.st_mtime_ns

    def _maybe_reload(self) -> None:
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            if self._mtime is not None:
                self._keys, self._mtime = {}, None
            return
        if mtime != self._mtime:
            self._load()

    def _save(self) -> None:
        if self.path is None:
            return
        data = {"version": _FILE_VERSION, "keys": [asdict(k) for k in self._keys.values()]}
        _atomic_write_json(self.path, data)
        self._mtime = self.path.stat().st_mtime_ns

    # ---- API ----------------------------------------------------------------------------------------
    def create(
        self,
        name: str,
        *,
        scopes: tuple[str, ...] = ("inference",),
        models: tuple[str, ...] = (),
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        daily_token_quota: int | None = None,
        max_concurrent_requests: int | None = None,
        meta: dict[str, str] | None = None,
    ) -> tuple[ApiKey, str]:
        """Create a key; returns the record and the plaintext key (shown once)."""
        name = name.strip()
        if not name or len(name) > 128:
            raise ConfigError("key name must be 1..128 characters")
        bad = set(scopes) - SCOPES
        if bad or not scopes:
            raise ConfigError(f"invalid scopes {sorted(bad)}; known: {sorted(SCOPES)}")
        for v, what in ((requests_per_minute, "requests_per_minute"), (tokens_per_minute, "tokens_per_minute"),
                        (daily_token_quota, "daily_token_quota"), (max_concurrent_requests, "max_concurrent_requests")):
            if v is not None and v < 1:
                raise ConfigError(f"{what} must be >= 1")
        with self._lock:
            self._maybe_reload()
            key_id = secrets.token_hex(6)
            while key_id in self._keys:  # pragma: no cover - 48-bit collision
                key_id = secrets.token_hex(6)
            secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
            rec = ApiKey(
                key_id=key_id,
                hashed=hash_secret(key_id, secret),
                name=name,
                scopes=tuple(dict.fromkeys(scopes)),
                models=tuple(dict.fromkeys(models)),
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
                created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                meta=dict(meta or {}),
                daily_token_quota=daily_token_quota,
                max_concurrent_requests=max_concurrent_requests,
            )
            self._keys[key_id] = rec
            self._save()
        log.info("API key created", extra={"key_id": key_id, "scopes": ",".join(rec.scopes)})
        return rec, f"pgw-{key_id}-{secret}"

    def verify(self, presented: str) -> ApiKey:
        """Return the key record or raise :class:`pgw.core.AuthError`."""
        key_id, secret = parse_api_key(presented)
        with self._lock:
            self._maybe_reload()
            rec = self._keys.get(key_id)
        expected = rec.hashed if rec is not None else hash_secret("000000000000", "x")
        ok = hmac.compare_digest(hash_secret(key_id, secret), expected)
        if rec is None or not ok:
            raise AuthError("invalid API key", details={"key_id": key_id})
        if rec.disabled:
            raise AuthError("API key revoked", details={"key_id": key_id})
        return rec

    def get(self, key_id: str) -> ApiKey | None:
        """Record by id (or ``None``)."""
        with self._lock:
            self._maybe_reload()
            return self._keys.get(key_id)

    def revoke(self, key_id: str) -> None:
        """Disable a key."""
        with self._lock:
            self._maybe_reload()
            rec = self._keys.get(key_id)
            if rec is None:
                raise KeyNotFoundError(f"unknown API key {key_id!r}", details={"key_id": key_id})
            self._keys[key_id] = replace(rec, disabled=True)
            self._save()
        log.info("API key revoked", extra={"key_id": key_id})

    def list(self) -> list[ApiKey]:
        """All key records (no secrets)."""
        with self._lock:
            self._maybe_reload()
            return sorted(self._keys.values(), key=lambda k: (k.created_at, k.key_id))

    def __len__(self) -> int:
        return len(self.list())


# =================================================================================================
# rate limiting
# =================================================================================================
@dataclass
class _Bucket:
    capacity: float
    rate: float  # per second
    level: float
    t: float

    def refill(self, now: float) -> None:
        if now > self.t:
            self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now

    def wait_for(self, amount: float) -> float:
        """Seconds until ``amount`` is available (0 if available now)."""
        short = amount - self.level
        return 0.0 if short <= 0 else short / self.rate


class RateLimiter:
    """Token-bucket limits per key: requests/minute and tokens/minute (prompt + completion).

    ``check_request`` takes one request token and *reserves* ``est_tokens`` from the token bucket
    (requests larger than the bucket are admitted when the bucket is full). ``record_tokens``
    settles the difference between actual usage and the reservation afterwards, so the bucket may go
    into debt and delays the key's next requests accordingly.

    Per-key overrides come from :attr:`ApiKey.requests_per_minute` / :attr:`ApiKey.tokens_per_minute`.

    Args:
        requests_per_minute: Default request rate (burst = one minute's worth).
        tokens_per_minute: Default token rate.
        clock: Injectable monotonic clock (tests).
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int, *, clock: Callable[[], float] | None = None):
        if requests_per_minute < 1 or tokens_per_minute < 1:
            raise ConfigError("rate limits must be >= 1")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._req: dict[str, _Bucket] = {}
        self._tok: dict[str, _Bucket] = {}

    def _buckets(self, key: ApiKey, now: float) -> tuple[_Bucket, _Bucket]:
        rpm = float(key.requests_per_minute or self.requests_per_minute)
        tpm = float(key.tokens_per_minute or self.tokens_per_minute)
        rb = self._req.get(key.key_id)
        if rb is None or rb.capacity != rpm:
            rb = self._req[key.key_id] = _Bucket(rpm, rpm / 60.0, rpm, now)
        tb = self._tok.get(key.key_id)
        if tb is None or tb.capacity != tpm:
            tb = self._tok[key.key_id] = _Bucket(tpm, tpm / 60.0, tpm, now)
        rb.refill(now)
        tb.refill(now)
        return rb, tb

    def check_request(self, key: ApiKey, est_tokens: int) -> None:
        """Admit or raise :class:`pgw.core.RateLimitedError` (with ``retry_after_s`` in details)."""
        est = max(0, int(est_tokens))
        with self._lock:
            now = self._clock()
            rb, tb = self._buckets(key, now)
            need_tok = min(float(est), tb.capacity)
            wait = max(rb.wait_for(1.0), tb.wait_for(need_tok))
            if wait > 0:
                which = "requests" if rb.wait_for(1.0) >= tb.wait_for(need_tok) else "tokens"
                raise RateLimitedError(
                    f"rate limit exceeded ({which} per minute)",
                    details={"retry_after_s": round(max(wait, 0.001), 3), "limit": which},
                )
            rb.level -= 1.0
            tb.level -= float(est)

    def record_tokens(self, key: ApiKey, tokens: int, *, reserved: int = 0) -> None:
        """Charge actual usage after completion (minus what ``check_request`` reserved)."""
        with self._lock:
            _, tb = self._buckets(key, self._clock())
            tb.level -= float(int(tokens) - int(reserved))

    def forget(self, key_id: str) -> None:
        """Drop the buckets of a key (after revocation)."""
        with self._lock:
            self._req.pop(key_id, None)
            self._tok.pop(key_id, None)


class QuotaTracker:
    """Per-key daily token quotas (UTC calendar day), optionally persisted to a JSON file.

    Args:
        default_daily_tokens: Quota for keys without :attr:`ApiKey.daily_token_quota` (``None`` = unlimited).
        state_file: Where usage is persisted (atomic rewrite on every update); ``None`` = memory only.
        today: Injectable ``() -> "YYYY-MM-DD"`` (tests).
    """

    def __init__(
        self,
        default_daily_tokens: int | None = None,
        state_file: Path | None = None,
        *,
        today: Callable[[], str] | None = None,
    ):
        self.default_daily_tokens = default_daily_tokens
        self.state_file = Path(state_file) if state_file is not None else None
        self._today = today or (lambda: datetime.now(UTC).strftime("%Y-%m-%d"))
        self._lock = threading.Lock()
        self._day = self._today()
        self._used: dict[str, int] = {}
        self._reserved: dict[str, int] = {}
        if self.state_file is not None and self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                if raw.get("day") == self._day:
                    self._used = {str(k): int(v) for k, v in dict(raw.get("used", {})).items()}
            except (OSError, ValueError, AttributeError) as exc:
                log.warning("ignoring unreadable quota state file: %s", type(exc).__name__)

    def _quota(self, key: ApiKey) -> int | None:
        return key.daily_token_quota if key.daily_token_quota is not None else self.default_daily_tokens

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            self._used.clear()

    def reserve(self, key: ApiKey, est_tokens: int) -> None:
        """Reserve ``est_tokens`` of today's quota or raise :class:`RateLimitedError` (quota exceeded)."""
        quota = self._quota(key)
        with self._lock:
            self._roll()
            if quota is not None:
                used = self._used.get(key.key_id, 0) + self._reserved.get(key.key_id, 0)
                if used + est_tokens > quota:
                    raise RateLimitedError(
                        "daily token quota exceeded",
                        details={"quota": quota, "used": used, "limit": "daily_tokens", "retry_after_s": 3600.0},
                    )
            self._reserved[key.key_id] = self._reserved.get(key.key_id, 0) + int(est_tokens)

    def settle(self, key: ApiKey, reserved: int, actual: int) -> None:
        """Release the reservation and record actual usage."""
        with self._lock:
            self._roll()
            left = self._reserved.get(key.key_id, 0) - int(reserved)
            if left > 0:
                self._reserved[key.key_id] = left
            else:
                self._reserved.pop(key.key_id, None)
            self._used[key.key_id] = self._used.get(key.key_id, 0) + int(actual)
            snapshot = {"day": self._day, "used": dict(self._used)}
        if self.state_file is not None:
            try:
                _atomic_write_json(self.state_file, snapshot)
            except OSError as exc:
                log.error("cannot persist quota state: %s", type(exc).__name__)

    def usage(self, key_ids: Iterable[str] | None = None) -> dict[str, int]:
        """Today's recorded usage per key."""
        with self._lock:
            self._roll()
            if key_ids is None:
                return dict(self._used)
            return {k: self._used.get(k, 0) for k in key_ids}
