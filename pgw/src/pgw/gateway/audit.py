"""Structured audit log of the gateway (who did what, when, with which result; never content).

One JSON object per line on the ``pgw.audit`` logger, optionally also appended to a dedicated file
(``PGW_GATEWAY_AUDIT_LOG_FILE``, mode 0600). Records contain ids, model names, token counts, mask
slot counts, durations and error codes. They never contain prompts, completions, activations,
masks, API keys or other secrets (SPEC 11.2); :meth:`AuditLog.event` enforces an allowlist of
field names so that content cannot be added by accident.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["AUDIT_FIELDS", "AuditLog"]

log = logging.getLogger("pgw.audit")

#: Field names an audit record may carry. Anything else is dropped (and counted).
AUDIT_FIELDS: frozenset[str] = frozenset({
    "request_id", "key_id", "key_name", "model", "endpoint", "status", "code", "stream",
    "prompt_tokens", "completion_tokens", "masks_reserved", "masks_consumed", "duration_ms", "ttft_ms",
    "finish_reason", "client", "fingerprint", "ring", "mode", "action", "target", "scopes", "reason",
    "queue_ms", "prompts", "retry_after_s", "masks_burned", "method", "subject", "email", "prefix_reused",
})


class AuditLog:
    """Audit trail writer.

    Args:
        path: Optional JSON-lines file (appended, created with mode 0600).
        enabled: Disable entirely (tests).
    """

    def __init__(self, path: Path | None = None, *, enabled: bool = True):
        self.path = Path(path) if path is not None else None
        self.enabled = enabled
        self.dropped_fields = 0
        self._lock = threading.Lock()
        self._fh: Any = None
        if self.path is not None and enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._fh = os.fdopen(fd, "a", encoding="utf-8", buffering=1)

    def event(self, event: str, **fields: Any) -> dict[str, Any]:
        """Record ``event`` with allowlisted ``fields``; returns the record written."""
        rec: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z",
            "event": event,
        }
        for k, v in fields.items():
            if k not in AUDIT_FIELDS:
                self.dropped_fields += 1
                continue
            if v is None:
                continue
            rec[k] = v if isinstance(v, str | int | float | bool) else str(v)
        if not self.enabled:
            return rec
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        log.info(line)
        if self._fh is not None:
            with self._lock:
                self._fh.write(line + "\n")
        return rec

    def close(self) -> None:
        """Flush and close the file."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
