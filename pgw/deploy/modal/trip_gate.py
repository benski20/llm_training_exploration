"""Client-reported trip latency of the Modal demo, and the ``--max-trip-ms`` gate of ``launch.py``.

No Modal imports: the customer container uses it during ``launch.py --e2e`` and the launcher uses it
to decide whether to relaunch the pair; the tests import it directly.

The measurement is the gateway's own view of each masked round trip (the compute client's wall time
from sending ``x_hat`` to having ``y_hat``):

* ``pgw_gateway_round_trip_seconds`` (Prometheus histogram from ``/metrics``, buckets 0.25-25 ms plus a
  tail; :data:`pgw.gateway.metrics.HOP_BUCKETS`). The delta between two scrapes gives the trip count and
  the median. Its bucket bounds include 1 ms exactly, so "p50 above 1 ms" is decided from the counts
  without interpolation.
* ``client_timing`` from ``/admin/status`` (cumulative sums per rows-per-request class): the mean
  round trip of single-row (decode) trips and its split into service queue + compute and the rest
  (network, framing, Python on both ends).

Nothing here reaches the provider: both sources are local to the customer's gateway.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "TRIP_METRIC",
    "gate",
    "hist_delta",
    "parse_histogram",
    "quantile",
    "quantile_bucket",
    "trip_latency",
]

#: The gateway's per-round-trip histogram (labels ``model``, ``linear_kind``; summed over both here).
TRIP_METRIC = "pgw_gateway_round_trip_seconds"

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)")
_LE = re.compile(r'(?:^|,)\s*le="(?P<le>[^"]*)"')


def _num(s: str) -> float:
    return math.inf if s in ("+Inf", "Inf", "inf") else float(s)


def parse_histogram(text: str, name: str = TRIP_METRIC) -> dict[str, Any]:
    """``{"buckets": {le_seconds: cumulative_count}, "count": n, "sum": seconds}`` of one histogram in a
    Prometheus text exposition, summed over all label sets. Empty buckets and zeros if it is absent."""
    buckets: dict[float, float] = {}
    count = total = 0.0
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if m is None:
            continue
        n, labels, value = m.group("name"), m.group("labels") or "", m.group("value")
        try:
            v = float(value)
        except ValueError:
            continue
        if n == name + "_bucket":
            le = _LE.search(labels)
            if le is not None:
                b = _num(le.group("le"))
                buckets[b] = buckets.get(b, 0.0) + v
        elif n == name + "_count":
            count += v
        elif n == name + "_sum":
            total += v
    return {"buckets": dict(sorted(buckets.items())), "count": count, "sum": total}


def hist_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """``after - before`` (observations made between two scrapes)."""
    b0 = before.get("buckets") or {}
    return {
        "buckets": {le: c - b0.get(le, 0.0) for le, c in sorted((after.get("buckets") or {}).items())},
        "count": float(after.get("count", 0.0)) - float(before.get("count", 0.0)),
        "sum": float(after.get("sum", 0.0)) - float(before.get("sum", 0.0)),
    }


def quantile_bucket(h: dict[str, Any], q: float) -> tuple[float, float] | None:
    """``(lo, hi]`` (seconds) of the bucket that holds the ``q`` quantile; ``None`` without observations.

    The quantile is the smallest observation with at least ``q * count`` observations at or below it,
    so it lies in the first bucket whose cumulative count reaches that rank."""
    buckets = h.get("buckets") or {}
    n = float(h.get("count") or (buckets.get(math.inf, 0.0) if buckets else 0.0))
    if n <= 0 or not buckets:
        return None
    rank = q * n
    lo = 0.0
    for le, cum in sorted(buckets.items()):
        if cum >= rank and cum > 0:
            return lo, le
        lo = le
    return lo, math.inf


def quantile(h: dict[str, Any], q: float) -> float | None:
    """Estimate of the ``q`` quantile in seconds, interpolated linearly inside its bucket (the same rule
    as Prometheus' ``histogram_quantile``); the last finite bound when it falls in the ``+Inf`` bucket."""
    buckets = sorted((h.get("buckets") or {}).items())
    n = float(h.get("count") or (buckets[-1][1] if buckets else 0.0))
    if n <= 0 or not buckets:
        return None
    rank = q * n
    lo, prev = 0.0, 0.0
    for le, cum in buckets:
        if cum >= rank and cum > 0:
            if math.isinf(le):
                return lo
            inside = cum - prev
            return lo + (le - lo) * ((rank - prev) / inside if inside > 0 else 1.0)
        lo, prev = le, cum
    return lo


def _ms(s: float | None) -> float | None:
    return None if s is None or math.isinf(s) else round(s * 1e3, 4)


def _class_delta(t0: dict[str, Any] | None, t1: dict[str, Any] | None, cls: str) -> dict[str, float]:
    a, b = (t0 or {}).get(cls) or {}, (t1 or {}).get(cls) or {}
    return {k: float(v) - float(a.get(k) or 0) for k, v in b.items() if isinstance(v, int | float)}


def trip_latency(metrics_before: str, metrics_after: str, timing_before: dict[str, Any] | None = None,
                 timing_after: dict[str, Any] | None = None) -> dict[str, Any]:
    """Round-trip latency between two snapshots of the gateway: the ``/metrics`` texts and (optional)
    one model's ``client_timing`` from ``/admin/status``.

    Returns ``trips``, ``mean_ms``, ``p50_ms`` / ``p90_ms`` (interpolated estimates), ``p50_bucket_ms``
    (the exact bucket that holds the median) and, from ``client_timing``, the single-row (decode) mean
    and its split ``rows_1_service_ms`` (queue + compute) / ``rows_1_wire_ms``. All trips count in the
    histogram (prefill trips too; they are a small minority in a chat of tens of tokens)."""
    if not metrics_before or not metrics_after:  # a failed scrape: a one-sided delta would be cumulative
        return {"trips": 0, "source": "missing /metrics snapshot"}
    h = hist_delta(parse_histogram(metrics_before), parse_histogram(metrics_after))
    n = round(h["count"])
    out: dict[str, Any] = {"trips": n, "source": f"{TRIP_METRIC} delta + client_timing"}
    if n > 0:
        out["mean_ms"] = round(h["sum"] / n * 1e3, 4)
        out["p50_ms"] = _ms(quantile(h, 0.5))
        out["p90_ms"] = _ms(quantile(h, 0.9))
        b = quantile_bucket(h, 0.5)
        if b is not None:
            out["p50_bucket_ms"] = [_ms(b[0]), _ms(b[1])]
        out["histogram"] = {("+Inf" if math.isinf(le) else f"{le * 1e3:g}ms"): round(c)
                            for le, c in h["buckets"].items()}
    d = _class_delta(timing_before, timing_after, "1")
    trips1 = int(d.get("trips", 0))
    if trips1 > 0:
        out["rows_1_trips"] = trips1
        out["rows_1_mean_ms"] = round(d.get("rtt_s", 0.0) / trips1 * 1e3, 4)
        out["rows_1_service_ms"] = round((d.get("queue_s", 0.0) + d.get("compute_s", 0.0)) / trips1 * 1e3, 4)
        out["rows_1_wire_ms"] = round(d.get("wire_s", 0.0) / trips1 * 1e3, 4)
    return out


def gate(latency: dict[str, Any] | None, max_trip_ms: float) -> tuple[bool | None, str]:
    """``(passed, reason)`` for "median client-reported trip ≤ ``max_trip_ms``".

    ``passed`` is ``None`` when there is no measurement (no trips). The decision uses the bucket that
    holds the median: entirely at or below the limit passes, entirely above fails; only when the limit
    falls strictly inside that bucket is the interpolated ``p50_ms`` compared (never for a limit on a
    bucket bound, e.g. 1 ms)."""
    if not latency or not latency.get("trips"):
        return None, "no round trips measured"
    lo, hi = (latency.get("p50_bucket_ms") or [None, None])
    p50 = latency.get("p50_ms")
    shown = f"p50 {p50} ms (bucket {lo}-{hi} ms) over {latency['trips']} trips"
    if hi is not None and hi <= max_trip_ms:
        return True, f"{shown} <= {max_trip_ms} ms"
    if lo is not None and lo >= max_trip_ms:
        return False, f"{shown} > {max_trip_ms} ms"
    if hi is None and lo is not None and lo < max_trip_ms:  # +Inf bucket starting below the limit: unbounded
        return False, f"{shown}: median in the open-ended bucket above {lo} ms"
    passed = p50 is not None and p50 <= max_trip_ms
    return passed, f"{shown} {'<=' if passed else '>'} {max_trip_ms} ms (interpolated inside the bucket)"
