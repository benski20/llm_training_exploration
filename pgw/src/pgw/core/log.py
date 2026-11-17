"""Logging setup shared by all PGW processes.

Rules (SPEC.md section 11): never log activations, masked payloads, masks, seeds, prompt or
completion text, API keys. Log ids, shapes, sizes, timings and error codes only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

__all__ = ["JsonFormatter", "configure_logging"]

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra ``logger.info(..., extra={...})`` fields are included."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v if isinstance(v, str | int | float | bool | type(None)) else repr(v)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(level: str | int = "INFO", json_lines: bool = False) -> None:
    """Configure the root logger once (idempotent: replaces handlers installed by a previous call)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_pgw", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler._pgw = True  # type: ignore[attr-defined]
    handler.setFormatter(
        JsonFormatter() if json_lines else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
