"""The client library's persistent integrity halt: one record per (SDK key, model), surviving restarts.

An integrity failure of the provider (a failed Freivalds, p31 or range check) opens the in-process gateway's
circuit breaker. This module persists that state, so a restart of the application does not reset it (security
review 2026-11, S11; the factory's own halt is persisted the same way, F3):

* ``<state dir>/sessions/<key id>/integrity/<model>.json``: ``halted``, the failure count, timestamps and a short
  event log (no secrets, no content), written atomically with mode 0600;
* authenticated with HMAC-SHA256 under ``<state dir>/sessions/<key id>/integrity.key`` (32 random bytes, 0600)
  and its own domain label :data:`DOMAIN`, over the canonical JSON of the record, the key id and the model.

Reading fails closed: a record whose MAC is missing or wrong, that is not valid JSON, names another key or model,
or whose MAC key is gone, reads as *halted* (``tampered``). No record at all means no failure was ever recorded.

Only an explicit operator action clears it: ``pgw sdk clear-halt --model ...`` (:func:`clear_halt`) or
``OpenAI.reset_integrity_breaker(model)``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pgw.sdk.cache import ensure_private_dir, file_lock

__all__ = ["DOMAIN", "HaltRecord", "HaltStore", "clear_halt"]

#: HMAC domain label of the halt records (distinct from every other pgw MAC domain).
DOMAIN = b"pgw-sdk-integrity-halt-v1\x00"
_VERSION = 1
_KEY_FILE = "integrity.key"
_DIR = "integrity"
_MAX_EVENTS = 20
_MODEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


@dataclass
class HaltRecord:
    """The persisted state of one (key, model). ``tampered`` is set on read only (never written)."""

    model: str
    halted: bool = False
    failures: int = 0
    halted_at: float | None = None
    last_failure_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    tampered: str | None = None

    @property
    def reason(self) -> str | None:
        """Why the model is halted (``None`` when it is not)."""
        if self.tampered:
            return (f"the integrity halt record of {self.model!r} cannot be verified ({self.tampered}); "
                    "treated as halted (fail closed)")
        if self.halted:
            return (f"{self.failures} integrity failure(s) of the provider recorded for {self.model!r} "
                    "(persisted across restarts)")
        return None

    def public(self) -> dict[str, Any]:
        """Status view (no secrets)."""
        return {"halted": self.halted or bool(self.tampered), "failures": self.failures,
                "halted_at": self.halted_at, "last_failure_at": self.last_failure_at, "tampered": self.tampered,
                "reason": self.reason}


def _check_model(model: str) -> str:
    if not isinstance(model, str) or not _MODEL_RE.fullmatch(model):
        raise ValueError(f"invalid model name for an integrity halt record: {model!r}")
    return model


def _body(key_id: str, rec: HaltRecord) -> bytes:
    data = {"v": _VERSION, "key_id": key_id, "model": rec.model, "halted": bool(rec.halted),
            "failures": int(rec.failures), "halted_at": rec.halted_at, "last_failure_at": rec.last_failure_at,
            "events": rec.events[-_MAX_EVENTS:]}
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


class HaltStore:
    """The halt records of one SDK key (``state_dir`` is the key's ``session_state_dir``).

    Thread- and process-safe (``flock`` on ``integrity/.lock``). Reads are cached by the file's
    ``(mtime_ns, size, inode)``, so :meth:`load` is cheap enough to call on every request.
    """

    def __init__(self, state_dir: Path, key_id: str):
        self.state_dir = Path(state_dir)
        self.key_id = str(key_id)
        self._cache: dict[str, tuple[tuple[int, int, int], HaltRecord]] = {}

    @property
    def dir(self) -> Path:
        """``<state>/sessions/<key id>/integrity``."""
        return self.state_dir / _DIR

    def path(self, model: str) -> Path:
        """The record of ``model``."""
        return self.dir / f"{_check_model(model)}.json"

    @property
    def key_file(self) -> Path:
        """The MAC key (32 random bytes, 0600)."""
        return self.state_dir / _KEY_FILE

    # ---- MAC key -----------------------------------------------------------------------------------
    def _read_key(self) -> bytes | None:
        try:
            fd = os.open(self.key_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return None
            data = os.read(fd, 64)
        finally:
            os.close(fd)
        return data if len(data) == 32 else None

    def _key(self, *, create: bool) -> bytes | None:
        k = self._read_key()
        if k is not None or not create:
            return k
        ensure_private_dir(self.state_dir)
        k = secrets.token_bytes(32)
        tmp = self.key_file.with_name(f".{_KEY_FILE}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.write(fd, k)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp, self.key_file)  # never replaces a key another process just created
        except FileExistsError:
            pass
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
        return self._read_key()

    @staticmethod
    def _mac(key: bytes, body: bytes) -> str:
        return hmac.new(key, DOMAIN + body, hashlib.sha256).hexdigest()

    # ---- read ----------------------------------------------------------------------------------------
    def load(self, model: str) -> HaltRecord:
        """The record of ``model``: fresh when absent; *halted* (``tampered``) when it cannot be verified."""
        p = self.path(model)
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            self._cache.pop(model, None)
            return HaltRecord(model)
        except OSError as exc:
            return HaltRecord(model, halted=True, tampered=f"unreadable: {exc.strerror or type(exc).__name__}")
        sig = (st.st_mtime_ns, st.st_size, st.st_ino)
        hit = self._cache.get(model)
        if hit is not None and hit[0] == sig:
            return hit[1]
        rec = self._parse(model, p, st)
        self._cache[model] = (sig, rec)
        return rec

    def _parse(self, model: str, p: Path, st: os.stat_result) -> HaltRecord:
        def bad(why: str) -> HaltRecord:
            return HaltRecord(model, halted=True, tampered=why)

        if not stat.S_ISREG(st.st_mode):
            return bad("not a regular file")
        try:
            raw = p.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError):
            return bad("not valid JSON")
        if not isinstance(data, dict):
            return bad("not a JSON object")
        mac = data.pop("mac", None)
        if not isinstance(mac, str) or not mac:
            return bad("no MAC")
        key = self._read_key()
        if key is None:
            return bad(f"its MAC key {_KEY_FILE} is missing")
        try:
            rec = HaltRecord(model=str(data["model"]), halted=bool(data["halted"]), failures=int(data["failures"]),
                             halted_at=data.get("halted_at"), last_failure_at=data.get("last_failure_at"),
                             events=list(data.get("events") or []))
        except (KeyError, TypeError, ValueError):
            return bad("malformed fields")
        if data.get("v") != _VERSION or data.get("key_id") != self.key_id or rec.model != model:
            return bad("it names another key, model or version")
        if not hmac.compare_digest(self._mac(key, _body(self.key_id, rec)), mac):
            return bad("MAC mismatch")
        if data != json.loads(_body(self.key_id, rec)):  # a field outside the MAC'd body
            return bad("unexpected fields")
        return rec

    # ---- write ---------------------------------------------------------------------------------------
    def _write(self, rec: HaltRecord) -> None:
        from pgw.core.fileio import atomic_write_text

        key = self._key(create=True)
        assert key is not None
        body = _body(self.key_id, rec)
        data = json.loads(body)
        data["mac"] = self._mac(key, body)
        atomic_write_text(self.path(rec.model), json.dumps(data, indent=1, sort_keys=True) + "\n", mode=0o600)
        self._cache.pop(rec.model, None)

    def _locked(self) -> Any:
        ensure_private_dir(self.dir)
        return file_lock(self.dir / ".lock", timeout_s=30.0)

    def record_failure(self, model: str, reason: str = "integrity_check_failed") -> HaltRecord:
        """Count one integrity failure and halt the model (persisted before it returns)."""
        now = time.time()
        with self._locked():
            cur = self.load(model)
            failures = cur.failures + 1
            ev = {"at": now, "event": "failure", "reason": str(reason)[:80]}
            if cur.tampered:
                ev["note"] = f"previous record unverifiable: {cur.tampered}"[:120]
            rec = HaltRecord(model, halted=True, failures=failures,
                             halted_at=cur.halted_at if (cur.halted and not cur.tampered) else now,
                             last_failure_at=now, events=[*cur.events, ev][-_MAX_EVENTS:])
            self._write(rec)
        return self.load(model)

    def clear(self, model: str, note: str = "") -> HaltRecord:
        """Operator action: clear the halt and the failure count (keeps the event log). Returns the old record."""
        with self._locked():
            cur = self.load(model)
            ev = {"at": time.time(), "event": "cleared", "note": str(note)[:200],
                  "previous_failures": cur.failures}
            if cur.tampered:
                ev["previous_state"] = f"unverifiable: {cur.tampered}"[:120]
            self._write(HaltRecord(model, events=[*([] if cur.tampered else cur.events), ev][-_MAX_EVENTS:]))
        return cur

    def models(self) -> list[str]:
        """Models with a record."""
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if _MODEL_RE.fullmatch(p.stem))


def clear_halt(state_root: Path, model: str, *, key_id: str | None = None, note: str = "") -> list[dict[str, Any]]:
    """``pgw sdk clear-halt``: clear ``model``'s halt for one key (``key_id``) or for every key under
    ``<state_root>/sessions`` that has a record of it. Returns one entry per record that was halted or counted
    failures (``key_id``, ``failures``, ``tampered``)."""
    _check_model(model)
    root = Path(state_root) / "sessions"
    if key_id is not None:
        dirs = [root / key_id]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out: list[dict[str, Any]] = []
    for d in dirs:
        store = HaltStore(d, d.name)
        if not store.path(model).exists():
            continue
        cur = store.load(model)
        if not (cur.halted or cur.failures or cur.tampered):
            continue
        store.clear(model, note)
        out.append({"key_id": d.name, "failures": cur.failures, "tampered": cur.tampered,
                    "halted_at": cur.halted_at})
    return out
