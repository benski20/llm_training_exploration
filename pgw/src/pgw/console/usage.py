"""Usage statistics and audit-event listing from the gateway's JSON-lines audit log.

The audit file is append-only and chronological, so both views read it from the end backwards in
blocks and stop as soon as they have what they need; ``before`` cursors are located by binary search
on the timestamps. Every read is bounded (``max_scan_bytes``) and usage results are cached briefly.
Only the current file is read (rotated files are not). Records are reduced to the audit allowlist
(:data:`pgw.gateway.audit.AUDIT_FIELDS`), which never contains prompts or completions.
"""

from __future__ import annotations

import calendar
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from pgw.gateway.audit import AUDIT_FIELDS

__all__ = ["WINDOWS", "AuditReader", "format_ts", "parse_ts_ms"]

#: Usage windows: name -> (length s, bucket s).
WINDOWS: dict[str, tuple[int, int]] = {"1h": (3600, 60), "24h": (86_400, 900), "7d": (604_800, 3600)}

_BLOCK = 64 * 1024
_TS_RE = re.compile(rb'"ts": ?"([0-9T:.Z-]+)"')
_OUTPUT_FIELDS = AUDIT_FIELDS | {"ts", "event"}
#: Out-of-order tolerance when stopping a backwards scan (records are written under a lock, but the
#: timestamp is taken just before it).
_SLACK_MS = 5_000


def parse_ts_ms(s: str | bytes) -> int | None:
    """Audit timestamp ``YYYY-MM-DDTHH:MM:SS.mmmZ`` -> Unix milliseconds (``None`` if malformed)."""
    if isinstance(s, bytes):
        s = s.decode("ascii", "replace")
    try:
        if len(s) < 19 or s[4] != "-" or s[10] != "T":
            return None
        sec = calendar.timegm((int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]),
                               int(s[17:19]), 0, 0, 0))
        ms = int(s[20:23]) if len(s) >= 23 and s[19] == "." else 0
    except ValueError:
        return None
    return sec * 1000 + ms


def format_ts(ms: int) -> str:
    """Inverse of :func:`parse_ts_ms`."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ms // 1000)) + f".{ms % 1000:03d}Z"


def _line_ts(line: bytes) -> int | None:
    m = _TS_RE.search(line)
    return parse_ts_ms(m.group(1)) if m else None


@dataclass
class _Scan:
    """Backwards line reader over ``[0, end)`` of a file, reading at most ``max_bytes``."""

    fh: IO[bytes]
    end: int
    max_bytes: int
    truncated: bool = False

    def lines(self) -> Iterator[bytes]:
        pos = self.end
        floor = max(0, self.end - self.max_bytes)
        buf = b""
        while pos > floor:
            n = min(_BLOCK, pos - floor)
            pos -= n
            self.fh.seek(pos)
            buf = self.fh.read(n) + buf
            parts = buf.split(b"\n")
            buf = parts[0]
            for ln in reversed(parts[1:]):
                if ln.strip():
                    yield ln
        if pos == 0:
            if buf.strip():
                yield buf
        else:
            self.truncated = True


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[max(0, math.ceil(q * len(v)) - 1)], 1)


@dataclass
class _Agg:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    masks_consumed: int = 0
    durations: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)

    def add(self, rec: dict[str, Any]) -> None:
        self.requests += 1
        self.prompt_tokens += _int(rec.get("prompt_tokens"))
        self.completion_tokens += _int(rec.get("completion_tokens"))
        self.masks_consumed += _int(rec.get("masks_consumed"))
        if _int(rec.get("status")) >= 400:
            self.errors += 1
        d = rec.get("duration_ms")
        if isinstance(d, int | float) and not isinstance(d, bool):
            self.durations.append(float(d))
        t = rec.get("ttft_ms")
        if isinstance(t, int | float) and not isinstance(t, bool):
            self.ttfts.append(float(t))

    def summary(self) -> dict[str, Any]:
        return {
            "requests": self.requests, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "errors": self.errors,
            "p50_ms": _percentile(self.durations, 0.5), "p95_ms": _percentile(self.durations, 0.95),
            "ttft_p50_ms": _percentile(self.ttfts, 0.5), "masks_consumed": self.masks_consumed,
        }


def _int(v: Any) -> int:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else 0


def _status_matches(status: Any, want: str) -> bool:
    if not isinstance(status, int) or isinstance(status, bool):
        return False
    w = want.lower()
    if w == "error":
        return status >= 400
    if w == "ok":
        return status < 400
    if len(w) == 3 and w.endswith("xx") and w[0].isdigit():
        return status // 100 == int(w[0])
    return w.isdigit() and status == int(w)


class AuditReader:
    """Read-side of the audit JSON-lines file.

    Args:
        path: The audit file (``None``: no audit file configured; every view is empty).
        max_scan_bytes: Most bytes one call reads.
        cache_s: How long a usage result is reused.
        clock: Wall clock in seconds (tests).
    """

    def __init__(self, path: Path | None, *, max_scan_bytes: int = 128 << 20, cache_s: float = 5.0,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path) if path is not None else None
        self.max_scan_bytes = max_scan_bytes
        self.cache_s = cache_s
        self.clock = clock
        self._cache: dict[str, tuple[float, int, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _open(self) -> tuple[IO[bytes], int] | None:
        if self.path is None:
            return None
        try:
            fh = open(self.path, "rb")
        except OSError:
            return None
        return fh, os.fstat(fh.fileno()).st_size

    # ---- usage ---------------------------------------------------------------------------------------
    def usage(self, window: str, key_names: Callable[[str], str | None] | None = None) -> dict[str, Any]:
        """Aggregated ``inference`` events of the last ``window`` (``1h``, ``24h`` or ``7d``)."""
        length_s, bucket_s = WINDOWS[window]
        now = self.clock()
        size = -1
        if self.path is not None:
            try:
                size = os.stat(self.path).st_size
            except OSError:
                size = -1
        with self._lock:
            hit = self._cache.get(window)
            if hit is not None and now - hit[0] < self.cache_s and hit[1] == size:
                return hit[2]
        out = self._compute_usage(window, length_s, bucket_s, now, key_names)
        with self._lock:
            self._cache[window] = (now, size, out)
        return out

    def _compute_usage(self, window: str, length_s: int, bucket_s: int, now: float,
                       key_names: Callable[[str], str | None] | None) -> dict[str, Any]:
        now_ms = int(now * 1000)
        start_ms = now_ms - length_s * 1000
        first_bucket = (start_ms // 1000) // bucket_s * bucket_s
        n_buckets = int((now_ms // 1000 - first_bucket) // bucket_s) + 1
        buckets = [_Agg() for _ in range(n_buckets)]
        by_model: dict[str, _Agg] = {}
        by_key: dict[str, _Agg] = {}
        total = _Agg()
        truncated = False
        opened = self._open()
        if opened is not None:
            fh, fsize = opened
            with fh:
                scan = _Scan(fh, fsize, self.max_scan_bytes)
                for line in scan.lines():
                    ts = _line_ts(line)
                    if ts is None:
                        continue
                    if ts < start_ms:
                        if ts < start_ms - _SLACK_MS:
                            break
                        continue
                    if ts > now_ms or (b'"event": "inference"' not in line and b'"event":"inference"' not in line):
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(rec, dict) or rec.get("event") != "inference":
                        continue
                    i = int((ts // 1000 - first_bucket) // bucket_s)
                    if 0 <= i < n_buckets:
                        buckets[i].add(rec)
                    total.add(rec)
                    by_model.setdefault(str(rec.get("model") or ""), _Agg()).add(rec)
                    by_key.setdefault(str(rec.get("key_id") or ""), _Agg()).add(rec)
                truncated = scan.truncated
        keys_out: dict[str, Any] = {}
        for kid, agg in by_key.items():
            keys_out[kid] = {"name": key_names(kid) if key_names else None, **agg.summary()}
        return {
            "window": window,
            "bucket_s": bucket_s,
            "series": [{"t": first_bucket + i * bucket_s, **b.summary()} for i, b in enumerate(buckets)],
            "by_model": {m: a.summary() for m, a in by_model.items()},
            "by_key": keys_out,
            "totals": total.summary(),
            "source": "audit_log" if self.path is not None else None,
            "truncated": truncated,
        }

    # ---- events --------------------------------------------------------------------------------------
    def _offset_at(self, fh: IO[bytes], size: int, target_ms: int) -> int:
        """Start offset of the first line with ``ts >= target_ms`` (or ``size``), by binary search."""
        lo, hi = 0, size  # lo is always a line start whose line is older than the target (or 0)
        while hi - lo > _BLOCK:
            mid = (lo + hi) // 2
            fh.seek(mid)
            fh.readline()
            pos = fh.tell()
            ts = None
            line_start = pos
            while pos < hi:
                line = fh.readline()
                if not line:
                    break
                ts = _line_ts(line)
                if ts is not None:
                    break
                pos = fh.tell()
                line_start = pos
            if ts is None or line_start >= hi:
                break
            if ts < target_ms:
                lo = line_start
            else:
                hi = line_start
        fh.seek(lo)
        pos = lo
        while pos < hi:
            line = fh.readline()
            if not line:
                break
            ts = _line_ts(line)
            if ts is not None and ts >= target_ms:
                return pos
            pos = fh.tell()
        return min(pos, size)

    def events(self, *, limit: int, before_ms: int | None = None, key_id: str | None = None,
               model: str | None = None, status: str | None = None, event: str | None = None) -> dict[str, Any]:
        """Newest-first events older than ``before_ms`` matching the filters.

        Events sharing the timestamp of the last one returned are all included (the page can then
        exceed ``limit``), so ``next_before`` never skips events.
        """
        out: list[dict[str, Any]] = []
        next_before: str | None = None
        truncated = False
        opened = self._open()
        if opened is None:
            return {"events": out, "next_before": None, "truncated": False}
        fh, size = opened
        with fh:
            end = size if before_ms is None else self._offset_at(fh, size, before_ms)
            scan = _Scan(fh, end, self.max_scan_bytes)
            last_ts: int | None = None
            more = False
            for line in scan.lines():
                ts = _line_ts(line)
                if ts is None or (before_ms is not None and ts >= before_ms):
                    continue
                if len(out) >= limit and ts != last_ts:
                    more = True
                    break
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if key_id is not None and rec.get("key_id") != key_id:
                    continue
                if model is not None and rec.get("model") != model:
                    continue
                if event is not None and rec.get("event") != event:
                    continue
                if status is not None and not _status_matches(rec.get("status"), status):
                    continue
                out.append({k: v for k, v in rec.items() if k in _OUTPUT_FIELDS and isinstance(k, str)})
                last_ts = ts
            truncated = scan.truncated
            if more and last_ts is not None:
                next_before = format_ts(last_ts)
            elif truncated:
                # the scan bound was hit: continue from the oldest line examined
                oldest = self._oldest_scanned(fh, end)
                next_before = format_ts(oldest) if oldest is not None else None
        return {"events": out, "next_before": next_before, "truncated": truncated}

    def _oldest_scanned(self, fh: IO[bytes], end: int) -> int | None:
        floor = max(0, end - self.max_scan_bytes)
        fh.seek(floor)
        fh.readline()
        while fh.tell() < end:
            line = fh.readline()
            if not line:
                return None
            ts = _line_ts(line)
            if ts is not None:
                return ts + 1  # "before" is exclusive: keep this line for the next page
        return None
