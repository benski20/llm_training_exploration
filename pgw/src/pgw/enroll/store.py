"""Service-side enrollment records (``pgw service enroll create|list|revoke`` and ``POST /pgw/v1/enroll``).

The store is one JSON file (mode 0600) guarded by an ``flock`` on ``<file>.lock`` so the CLI and the
running service can both change it. A record holds only the SHA-256 of the code's secret. Consuming a
code (check + mark used) happens under the lock in one read-modify-write, so a code works at most once
even with concurrent requests or several service processes on the same file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pgw.core.errors import AuthError, ConfigError, ErrorCode, PGWError
from pgw.core.fileio import atomic_write_text, file_lock

__all__ = ["EnrollmentRecord", "EnrollmentRejectedError", "EnrollmentStore", "parse_ttl"]

T = TypeVar("T")

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
#: Used, revoked or expired records are dropped this long after they stopped being usable.
_RETENTION_S = 90 * 86400.0
_MAX_TTL_S = 30 * 86400.0


def parse_ttl(text: str) -> float:
    """``"24h"`` / ``"30m"`` / ``"7d"`` / ``"3600"`` (seconds) -> seconds (at most 30 days).

    Raises:
        ConfigError: unparsable, non-positive or longer than 30 days.
    """
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", text or "")
    if not m:
        raise ConfigError(f"invalid TTL {text!r}: use e.g. 30m, 24h or 7d")
    seconds = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    if seconds <= 0 or seconds > _MAX_TTL_S:
        raise ConfigError("TTL must be positive and at most 30 days")
    return seconds


class EnrollmentRejectedError(AuthError):
    """The presented code is unknown, used, revoked or expired.

    The HTTP reply is the same in every case (no oracle); :attr:`reason` goes to the audit log only.
    """

    def __init__(self, reason: str, record: EnrollmentRecord | None = None):
        super().__init__("invalid, used or expired enrollment code")
        self.reason = reason
        self.record = record


@dataclass
class EnrollmentRecord:
    """One enrollment code (never the secret itself)."""

    id: str
    tenant: str
    secret_sha256: str
    created_at: float
    expires_at: float
    models: list[str] = field(default_factory=list)
    note: str | None = None
    used_at: float | None = None
    used_by: str | None = None
    revoked_at: float | None = None

    def state(self, now: float | None = None) -> str:
        """``active`` | ``used`` | ``revoked`` | ``expired``."""
        now = time.time() if now is None else now
        if self.revoked_at is not None:
            return "revoked"
        if self.used_at is not None:
            return "used"
        if now >= self.expires_at:
            return "expired"
        return "active"

    def public(self) -> dict[str, Any]:
        """JSON view without the secret hash (``pgw service enroll list``)."""
        d = asdict(self)
        d.pop("secret_sha256")
        d["state"] = self.state()
        return d


class EnrollmentStore:
    """JSON-file store of enrollment records.

    Args:
        path: The records file (created 0600 on first write).
        clock: Time source (tests).
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.clock = clock
        self._thread_lock = threading.Lock()

    # ---- file I/O ----------------------------------------------------------------------------------
    def _load(self) -> list[EnrollmentRecord]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ConfigError(f"cannot read enrollment file {self.path}: {exc.strerror}") from exc
        try:
            data = json.loads(text) if text.strip() else {"records": []}
            return [EnrollmentRecord(**r) for r in data.get("records", [])]
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"enrollment file {self.path} is corrupt: {type(exc).__name__}") from exc

    def _save(self, records: list[EnrollmentRecord]) -> None:
        now = self.clock()
        keep = [r for r in records if r.state(now) == "active"
                or now - max(r.used_at or 0, r.revoked_at or 0, r.expires_at) < _RETENTION_S]
        atomic_write_text(self.path, json.dumps({"version": 1, "records": [asdict(r) for r in keep]}, indent=1) + "\n")

    def _locked(self, fn: Callable[[list[EnrollmentRecord]], tuple[T, bool]]) -> T:
        with self._thread_lock, file_lock(self.lock_path):
            records = self._load()
            out, changed = fn(records)
            if changed:
                self._save(records)
            return out

    # ---- operations --------------------------------------------------------------------------------
    def create(self, tenant: str, ttl_s: float, models: list[str] | None = None, *,
               note: str | None = None) -> tuple[EnrollmentRecord, bytes]:
        """New single-use record; returns ``(record, secret)``. The secret is not stored.

        Raises:
            ConfigError: invalid tenant id, model name or TTL.
        """
        from pgw.enroll.code import SECRET_BYTES

        if not _TENANT_RE.match(tenant):
            raise ConfigError("tenant id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        models = [m.strip() for m in (models or []) if m.strip()]
        for m in models:
            if not _MODEL_RE.match(m):
                raise ConfigError(f"invalid model name {m!r}")
        if not 0 < ttl_s <= _MAX_TTL_S:
            raise ConfigError("TTL must be positive and at most 30 days")
        secret = secrets.token_bytes(SECRET_BYTES)
        now = self.clock()
        rec = EnrollmentRecord(id=secrets.token_hex(6), tenant=tenant, secret_sha256=_hash(secret), created_at=now,
                               expires_at=now + ttl_s, models=models, note=note)

        def add(records: list[EnrollmentRecord]) -> tuple[None, bool]:
            records.append(rec)
            return None, True

        self._locked(add)
        return rec, secret

    def list(self) -> list[EnrollmentRecord]:
        """All retained records (oldest first)."""
        return self._locked(lambda records: (list(records), False))

    def revoke(self, record_id: str) -> EnrollmentRecord:
        """Revoke an unused code.

        Raises:
            PGWError(model_not_found-like 404): unknown id.
        """
        def rev(records: list[EnrollmentRecord]) -> tuple[EnrollmentRecord, bool]:
            for r in records:
                if r.id == record_id:
                    if r.revoked_at is None:
                        r.revoked_at = self.clock()
                        return r, True
                    return r, False
            raise PGWError(f"no enrollment code with id {record_id!r}", code=ErrorCode.CONFIG)

        return self._locked(rev)

    def consume(self, secret: bytes, used_by: str,
                check: Callable[[EnrollmentRecord], T]) -> tuple[EnrollmentRecord, T]:
        """Atomically find the active record for ``secret``, run ``check(record)`` and mark it used.

        ``check`` runs under the lock before the record is marked; if it raises, the code stays
        unused (e.g. none of its models is served right now).

        Raises:
            EnrollmentRejectedError: unknown, used, revoked or expired code (all look the same to callers).
        """
        digest = bytes.fromhex(_hash(secret))

        def use(records: list[EnrollmentRecord]) -> tuple[tuple[EnrollmentRecord, T], bool]:
            found: EnrollmentRecord | None = None
            for r in records:  # compare against every record: no timing hint about which matched
                if hmac.compare_digest(bytes.fromhex(r.secret_sha256), digest):
                    found = r
            if found is None:
                raise EnrollmentRejectedError("unknown_code")
            st = found.state(self.clock())
            if st != "active":
                raise EnrollmentRejectedError(f"code_{st}", found)
            result = check(found)
            found.used_at = self.clock()
            found.used_by = used_by
            return (found, result), True

        return self._locked(use)


def _hash(secret: bytes) -> str:
    return hashlib.sha256(b"pgw-enroll-secret-v1" + secret).hexdigest()
