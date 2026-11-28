"""Content-free demand reports and their aggregation (the placement controller's only input about users).

What a client sends (``POST /pgw/v1/placement/report``, at most once per model and interval, authenticated like
every other call, on the endpoint it is using)::

    {"v": 1, "model": "qwen2.5-7b-instruct", "endpoint": "<region label of the endpoint in use>",
     "rtt_bucket": "<=5", "peak_streams": 3}

Nothing else is accepted (:func:`validate_report` refuses unknown fields, so nothing can be smuggled in a free-form
field): no prompt, no answer, no token ids, no per-request timing, no client identifier beyond the authenticated
tenant the endpoint already knows. The coarse region (a metro) is derived **server-side** from the connection's
source address, which the provider sees on every connection anyway; the address is used for that lookup and
dropped (:class:`Aggregator` never stores it).

Why this reveals nothing new:

* the provider already sees the client IP, the model, the endpoint, the timing and volume of every trip (and so
  the RTT, which is the gap between a reply and the next request) and how many streams run at once (rows per lane
  call); the report only puts those already-visible facts in a form the controller can sum;
* reports are aggregated **before storing**: per hour, per (metro, model, endpoint, RTT bucket) the number of
  distinct installations (counted with a per-hour random salt that is discarded when the hour closes, so
  installations cannot be linked across hours from the stored data) and the sum of peak streams;
* buckets with fewer than ``k_min`` installations are folded into ``(metro="other")`` at export (small-count
  suppression), and aggregates older than ``retention_days`` are deleted;
* off unless the operator enables it (``ServiceSettings.demand_reports``) and the client library sends it only
  with routing on (``routing="signed"``) and ``usage_reports`` on.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Aggregator", "RegionResolver", "ReportError", "validate_report"]

_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FIELDS = {"v", "model", "endpoint", "rtt_bucket", "peak_streams"}


class ReportError(ValueError):
    """A malformed demand report (refused whole)."""


def _buckets() -> frozenset[str]:
    from pgw.sdk.directory import RTT_BUCKETS

    return frozenset([f"<={b:g}" for b in RTT_BUCKETS] + [f">{RTT_BUCKETS[-1]:g}"])


def validate_report(r: Any, *, models: Iterable[str]) -> dict[str, Any]:
    """The report with exactly the allowed fields and values, or :class:`ReportError`."""
    if not isinstance(r, Mapping) or set(r) - _FIELDS or not {"v", "model", "endpoint", "rtt_bucket"} <= set(r):
        raise ReportError(f"a demand report has exactly the fields {sorted(_FIELDS)}")
    if r["v"] != 1:
        raise ReportError("report version must be 1")
    model, ep, b = r["model"], r["endpoint"], r["rtt_bucket"]
    if not isinstance(model, str) or model not in set(models):
        raise ReportError("unknown model")
    if not isinstance(ep, str) or not _LABEL.fullmatch(ep):
        raise ReportError("endpoint must be a region label")
    if b not in _buckets():
        raise ReportError("unknown rtt bucket")
    ps = r.get("peak_streams", 1)
    if not isinstance(ps, int) or isinstance(ps, bool) or not 0 <= ps <= 256:
        raise ReportError("peak_streams must be an integer in [0, 256]")
    return {"v": 1, "model": model, "endpoint": ep, "rtt_bucket": b, "peak_streams": ps}


class RegionResolver:
    """Source address -> coarse region (a metro label), from an operator table of CIDR prefixes (e.g. derived from a
    GeoIP database at /20 or coarser). Unknown addresses map to ``"unknown"``; the address itself is never kept."""

    def __init__(self, table: Mapping[str, str] | None = None):
        self._nets = sorted(((ipaddress.ip_network(k, strict=False), v) for k, v in (table or {}).items()),
                            key=lambda t: -t[0].prefixlen)
        for _, v in self._nets:
            if not _LABEL.fullmatch(v):
                raise ValueError(f"region label {v!r} must match {_LABEL.pattern}")

    def __call__(self, ip: str | None) -> str:
        try:
            a = ipaddress.ip_address(ip or "")
        except ValueError:
            return "unknown"
        for net, name in self._nets:
            if a.version == net.version and a in net:
                return name
        return "unknown"


@dataclass
class _Hour:
    start: int
    salt: bytes
    seen: dict[tuple[str, str, str, str], set[bytes]]
    streams: dict[tuple[str, str, str, str], int]


class Aggregator:
    """Hourly aggregation of demand reports (thread-safe). Only the closed hours' counts are ever stored or
    exported; the open hour holds salted, hour-scoped digests of the tenants (to count each installation once) and
    is discarded with its salt when it closes."""

    def __init__(self, *, models: Iterable[str], resolver: RegionResolver | None = None, k_min: int = 5,
                 retention_days: float = 30.0, clock: Any = time.time):
        self.models = frozenset(models)
        self.resolver = resolver or RegionResolver()
        self.k_min, self.retention_s, self.clock = int(k_min), retention_days * 86400.0, clock
        self._hour: _Hour | None = None
        #: (hour start, metro, model, endpoint, rtt bucket) -> (installations, sum of peak streams)
        self.closed: dict[tuple[int, str, str, str, str], tuple[int, int]] = {}
        self._lock = threading.Lock()

    def _roll(self, now: float) -> _Hour:
        h0 = int(now // 3600) * 3600
        if self._hour is None or self._hour.start != h0:
            if self._hour is not None:
                self._close(self._hour)
            self._hour = _Hour(h0, secrets.token_bytes(32), defaultdict(set), defaultdict(int))
        return self._hour

    def _close(self, h: _Hour) -> None:
        for k, s in h.seen.items():
            self.closed[(h.start, *k)] = (len(s), h.streams[k])
        h.seen.clear()
        h.streams.clear()
        h.salt = b""
        cutoff = self.clock() - self.retention_s
        for k in [k for k in self.closed if k[0] < cutoff]:
            del self.closed[k]

    def add(self, report: Any, *, tenant: str, source_ip: str | None) -> None:
        """Validate and count one report; ``source_ip`` is resolved to a metro and dropped here."""
        r = validate_report(report, models=self.models)
        metro = self.resolver(source_ip)
        del source_ip
        with self._lock:
            h = self._roll(self.clock())
            k = (metro, r["model"], r["endpoint"], r["rtt_bucket"])
            tag = hmac.new(h.salt, tenant.encode("utf-8"), hashlib.sha256).digest()[:16]
            first = tag not in h.seen[k]
            h.seen[k].add(tag)
            if first:  # one installation's peak counts once per hour and bucket
                h.streams[k] += r["peak_streams"]

    def flush(self) -> None:
        """Close the open hour now (tests, shutdown)."""
        with self._lock:
            if self._hour is not None:
                self._close(self._hour)
                self._hour = None

    def export(self) -> list[dict[str, Any]]:
        """The closed hours' aggregates, small counts folded into ``metro="other"`` (k-anonymity)."""
        with self._lock:
            self._roll(self.clock())
            out: dict[tuple[int, str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
            for (h, metro, model, ep, b), (n, s) in self.closed.items():
                key = (h, metro if n >= self.k_min else "other", model, ep, b)
                out[key][0] += n
                out[key][1] += s
            rows = [{"hour": h, "metro": m, "model": md, "endpoint": ep, "rtt_bucket": b, "installations": n,
                     "peak_streams": s} for (h, m, md, ep, b), (n, s) in sorted(out.items())]
            return [r for r in rows if r["metro"] != "other" or r["installations"] >= self.k_min]


def demand_by_metro(rows: Iterable[Mapping[str, Any]], hour: int | None = None) -> dict[tuple[str, str], float]:
    """(metro, model) -> peak concurrent streams in ``hour`` (default: the latest hour present)."""
    rows = list(rows)
    if not rows:
        return {}
    h = max(int(r["hour"]) for r in rows) if hour is None else hour
    out: dict[tuple[str, str], float] = defaultdict(float)
    for r in rows:
        if int(r["hour"]) == h and r["metro"] not in ("other", "unknown"):
            out[(str(r["metro"]), str(r["model"]))] += float(r["peak_streams"])
    return dict(out)
