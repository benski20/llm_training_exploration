"""Prometheus metric definitions (names are part of the contract; see SPEC.md section 11.1).

:data:`METRIC_NAMES` is the contract: components emit only the metrics listed there, and labels
never carry prompt data. :class:`PGWMetrics` creates collectors from that table, lazily and at most
once per registry, so the gateway, the service and the factory can share one registry in one
process (or each use a private one) without "Duplicated timeseries" errors.

Example::

    m = PGWMetrics()                       # process-wide shared PGW registry
    m.get("pgw_service_rows_total").labels(tenant="t", linear_kind="qkv").inc(32)
    body, content_type = m.render()        # Prometheus text exposition

``prometheus_client`` is optional at import time: if it is missing (or ``enabled=False``) every
collector is a no-op and :meth:`PGWMetrics.render` returns an empty page.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "LATENCY_BUCKETS",
    "METRIC_NAMES",
    "ROW_BUCKETS",
    "PGWMetrics",
    "default_registry",
    "reset_default_registry",
    "safe_label",
]

log = logging.getLogger(__name__)

#: name -> (type, labels, help). Components must only emit metrics listed here.
METRIC_NAMES: dict[str, tuple[str, tuple[str, ...], str]] = {
    # service
    "pgw_service_requests_total": ("counter", ("tenant", "code"), "Compute requests by result code"),
    "pgw_service_rows_total": ("counter", ("tenant", "linear_kind"), "Masked rows computed"),
    "pgw_service_request_seconds": ("histogram", ("tenant",), "End-to-end compute request latency"),
    "pgw_service_kernel_seconds": ("histogram", ("linear_kind", "backend"), "Kernel time per batch"),
    "pgw_service_batch_rows": ("histogram", ("linear_kind",), "Rows per coalesced batch"),
    "pgw_service_queue_depth": ("gauge", (), "Queued compute items"),
    "pgw_service_inflight": ("gauge", (), "In-flight requests"),
    "pgw_service_bytes_in_total": ("counter", ("tenant",), "Request payload bytes"),
    "pgw_service_bytes_out_total": ("counter", ("tenant",), "Response payload bytes"),
    # gateway
    "pgw_gateway_requests_total": ("counter", ("model", "endpoint", "status"), "API requests"),
    "pgw_gateway_tokens_total": ("counter", ("model", "kind"), "Prompt/completion tokens"),
    "pgw_gateway_ttft_seconds": ("histogram", ("model",), "Time to first token"),
    "pgw_gateway_tpot_seconds": ("histogram", ("model",), "Time per output token"),
    "pgw_gateway_round_trip_seconds": ("histogram", ("model", "linear_kind"), "Masked linear round trip"),
    "pgw_gateway_network_seconds": (
        "histogram", ("model", "linear_kind"),
        "Round trip minus the service's queue and compute time (network, framing, client and server Python)",
    ),
    "pgw_gateway_service_seconds": (
        "histogram", ("model", "linear_kind"), "Service-reported queue plus compute time per round trip",
    ),
    "pgw_gateway_integrity_failures_total": ("counter", ("model", "check"), "Freivalds/range check failures"),
    "pgw_gateway_active_sessions": ("gauge", ("model",), "Active generation sessions"),
    "pgw_gateway_noise_alarm": ("gauge", ("model",), "1 if the outbound-noise monitor alarms"),
    "pgw_gateway_model_ready": ("gauge", ("model",), "1 if the model can take requests now"),
    "pgw_gateway_service_attempts_total": (
        "counter", ("model", "code"), "Compute-service HTTP attempts by result code (ok, error code)"
    ),
    "pgw_gateway_service_retries_total": ("counter", ("model",), "Compute-service attempts that were retried"),
    "pgw_gateway_round_trip_failures_total": ("counter", ("model", "code"), "Failed masked round trips"),
    "pgw_gateway_admission_active": ("gauge", ("model",), "Admitted sessions"),
    "pgw_gateway_admission_waiting": ("gauge", ("model",), "Requests waiting in the admission queue"),
    "pgw_gateway_admission_rejected_total": ("counter", ("model", "reason"), "Requests refused by admission"),
    "pgw_gateway_queue_wait_seconds": ("histogram", ("model",), "Time spent in the admission queue"),
    "pgw_gateway_masks_burned_total": ("counter", ("model",), "Mask slots reserved by requests but never used"),
    "pgw_gateway_client_cert_expiry_seconds": (
        "gauge", ("model",), "Seconds until the configured compute-service client certificate expires",
    ),
    "pgw_masks_open_reservations": ("gauge", (), "Open mask reservations in the store"),
    "pgw_masks_unshredded_bytes": ("gauge", (), "Bytes of fully consumed segment files not yet shredded"),
    # mask store / factory
    "pgw_masks_available_slots": ("gauge", ("fingerprint", "ring"), "Unconsumed mask slots"),
    "pgw_masks_consumed_slots_total": ("counter", ("fingerprint", "ring"), "Consumed (incl. burned) slots"),
    "pgw_masks_burned_slots_total": ("counter", ("fingerprint", "ring"), "Reserved but unused slots"),
    "pgw_masks_low_water": ("gauge", ("fingerprint", "ring"), "1 if below low water"),
    "pgw_factory_slots_generated_total": ("counter", ("fingerprint", "ring", "device"), "Slots generated"),
    "pgw_factory_slots_per_second": ("gauge", ("fingerprint", "ring", "device"), "Generation throughput"),
    "pgw_factory_last_success_timestamp": ("gauge", (), "Unix time of the last successful scheduler plan/run"),
    "pgw_factory_errors_total": ("counter", (), "Failed factory runs"),
}

#: Buckets for ``*_seconds`` histograms (sub-millisecond kernels up to multi-minute requests).
LATENCY_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)
#: Buckets for row-count histograms (``*_rows``).
ROW_BUCKETS: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

_TEXT_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_LABEL_OK = re.compile(r"[^A-Za-z0-9_.:/@+-]")
_MAX_LABEL_LEN = 64


def safe_label(value: object, *, max_len: int = _MAX_LABEL_LEN) -> str:
    """Normalize a label value: restricted charset, bounded length.

    Labels may only carry ids (model names, tenants, linear kinds, codes, short fingerprints). This
    helper is a guard against accidentally putting free text into a label: anything outside
    ``[A-Za-z0-9_.:/@+-]`` becomes ``_`` and the value is truncated to ``max_len`` characters.
    """
    s = _LABEL_OK.sub("_", str(value))
    return s[:max_len]


class _Noop:
    """Collector stand-in used when metrics are disabled or unavailable."""

    def labels(self, *_a: Any, **_kw: Any) -> _Noop:
        return self

    def inc(self, *_a: Any, **_kw: Any) -> None:
        return None

    def dec(self, *_a: Any, **_kw: Any) -> None:
        return None

    def set(self, *_a: Any, **_kw: Any) -> None:
        return None

    def observe(self, *_a: Any, **_kw: Any) -> None:
        return None

    def time(self) -> Any:  # pragma: no cover - trivial
        import contextlib

        return contextlib.nullcontext()


_NOOP = _Noop()

# ---- shared registry and per-registry collector cache ----------------------------------------------
_lock = threading.RLock()
_default_registry: Any = None
# registry -> {metric name -> collector}; weak so private registries (tests) can be collected.
_collectors: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()


def _prometheus() -> Any | None:
    try:
        import prometheus_client
    except ImportError:  # pragma: no cover - dependency of every server extra
        return None
    return prometheus_client


def default_registry() -> Any:
    """The process-wide PGW ``CollectorRegistry`` (created on first use).

    It is separate from ``prometheus_client.REGISTRY`` so library defaults (process/GC collectors)
    do not leak into the contract, and so tests can :func:`reset_default_registry`.

    Raises:
        RuntimeError: ``prometheus_client`` is not installed.
    """
    global _default_registry
    pc = _prometheus()
    if pc is None:
        raise RuntimeError("prometheus_client is not installed")
    with _lock:
        if _default_registry is None:
            _default_registry = pc.CollectorRegistry(auto_describe=True)
        return _default_registry


def reset_default_registry() -> None:
    """Drop the shared registry (tests; a new one is created on next use)."""
    global _default_registry
    with _lock:
        if _default_registry is not None:
            _collectors.pop(_default_registry, None)
        _default_registry = None


def _buckets_for(name: str) -> tuple[float, ...]:
    return ROW_BUCKETS if name.endswith("_rows") else LATENCY_BUCKETS


def _find_registered(registry: Any, name: str) -> Any | None:
    """Collector already registered under ``name`` by someone else (private API, best effort)."""
    names = getattr(registry, "_names_to_collectors", None)
    if not isinstance(names, dict):
        return None
    return names.get(name) or names.get(name.removesuffix("_total"))


class PGWMetrics:
    """Lazily creates Prometheus collectors from :data:`METRIC_NAMES` on a registry.

    ``metrics.get("pgw_service_rows_total").labels(tenant="t", linear_kind="qkv").inc(32)``

    Several :class:`PGWMetrics` on the same registry share collectors (so the service, the
    gateway and an embedded factory can all emit into one registry).

    Args:
        registry: A ``prometheus_client.CollectorRegistry``; ``None`` uses :func:`default_registry`.
        enabled: ``False`` makes every collector a no-op.
        prefixes: Optional allowed name prefixes (e.g. ``("pgw_service_",)``); other contract names
            raise ``KeyError`` from :meth:`get`. Default: every contract metric.
    """

    def __init__(self, registry: Any | None = None, *, enabled: bool = True, prefixes: Iterable[str] | None = None):
        self._prefixes = tuple(prefixes) if prefixes is not None else None
        self._hooks: list[Callable[[], None]] = []
        self.registry: Any = None
        self.enabled = False
        if not enabled:
            return
        pc = _prometheus()
        if pc is None:
            log.warning("prometheus_client is not installed; PGW metrics disabled")
            return
        try:  # only the contract's series: no *_created samples (process-wide setting)
            pc.disable_created_metrics()
        except AttributeError:  # pragma: no cover - old prometheus_client
            pass
        self.registry = registry if registry is not None else default_registry()
        self.enabled = True

    # ---- collectors ----------------------------------------------------------------------------
    def _check(self, name: str) -> tuple[str, tuple[str, ...], str]:
        spec = METRIC_NAMES.get(name)
        if spec is None:
            raise KeyError(f"{name!r} is not a contract metric (pgw.observability.METRIC_NAMES)")
        if self._prefixes is not None and not name.startswith(self._prefixes):
            raise KeyError(f"{name!r} is outside this component's metric families {self._prefixes}")
        return spec

    def get(self, name: str) -> Any:
        """Collector for ``name`` (``KeyError`` if not in :data:`METRIC_NAMES`); no-op stub if disabled."""
        kind, labels, help_ = self._check(name)
        if not self.enabled:
            return _NOOP
        with _lock:
            cache = _collectors.setdefault(self.registry, {})
            col = cache.get(name)
            if col is None:
                col = self._create(name, kind, labels, help_)
                cache[name] = col
            return col

    def _create(self, name: str, kind: str, labels: tuple[str, ...], help_: str) -> Any:
        import prometheus_client as pc

        try:
            if kind == "counter":  # prometheus_client appends _total itself
                return pc.Counter(name.removesuffix("_total"), help_, labels, registry=self.registry)
            if kind == "gauge":
                return pc.Gauge(name, help_, labels, registry=self.registry)
            if kind == "histogram":
                return pc.Histogram(name, help_, labels, registry=self.registry, buckets=_buckets_for(name))
        except ValueError:
            # Registered on this registry outside PGWMetrics (e.g. a component's own fallback).
            existing = _find_registered(self.registry, name)
            if existing is not None:
                return existing
            raise
        raise ValueError(f"unknown metric type {kind!r} for {name}")  # pragma: no cover - contract

    def preregister(self, names: Iterable[str] | None = None) -> None:
        """Create collectors up front so they appear (with no samples) before first use."""
        for name in names if names is not None else METRIC_NAMES:
            if self._prefixes is None or name.startswith(self._prefixes):
                self.get(name)

    def value(self, name: str, **labels: str) -> float | None:
        """Current value of one series (counters: the ``_total`` sample; histograms: ``_count``).

        Returns ``None`` if the series does not exist (or metrics are disabled). For tests, the
        dashboard and ``/admin/status``; not a hot-path API.
        """
        kind, _, _ = self._check(name)
        if not self.enabled:
            return None
        sample = name if kind != "histogram" else name + "_count"
        if kind == "counter" and not sample.endswith("_total"):
            sample += "_total"
        v = self.registry.get_sample_value(sample, labels)
        return None if v is None else float(v)

    # ---- exposition ------------------------------------------------------------------------------
    def add_collect_hook(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` before every :meth:`render` (refresh gauges from live state)."""
        self._hooks.append(fn)

    def _run_hooks(self) -> None:
        for fn in self._hooks:
            try:
                fn()
            except Exception:
                log.exception("metrics collect hook failed")

    def render(self) -> tuple[bytes, str]:
        """Prometheus text exposition ``(body, content_type)``."""
        self._run_hooks()
        if not self.enabled:
            return b"", _TEXT_CONTENT_TYPE
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def asgi_app(self) -> Any:
        """ASGI app serving the exposition (mount it at ``/metrics``).

        Collect hooks run on every scrape. Works without FastAPI (plain ASGI).
        """
        metrics = self

        async def app(scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
            if scope["type"] == "lifespan":  # pragma: no cover - when mounted standalone
                while True:
                    msg = await receive()
                    if msg["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif msg["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            if scope["type"] != "http":  # pragma: no cover - websockets etc.
                return
            if scope.get("method", "GET") not in ("GET", "HEAD"):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 405,
                        "headers": [(b"allow", b"GET, HEAD"), (b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"method not allowed\n"})
                return
            body, ctype = metrics.render()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", ctype.encode()),
                        (b"content-length", str(len(body)).encode()),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"" if scope.get("method") == "HEAD" else body})

        return app

    def start_http_server(self, port: int, addr: str = "127.0.0.1") -> Any:
        """Serve ``/metrics`` on a background thread (factory / batch jobs without an HTTP app).

        Returns whatever ``prometheus_client.start_http_server`` returns (server, thread) on
        recent versions. Collect hooks are not run by this server; set gauges eagerly instead.

        Raises:
            RuntimeError: Metrics are disabled.
        """
        if not self.enabled:
            raise RuntimeError("metrics are disabled")
        from prometheus_client import start_http_server

        log.info("serving metrics", extra={"addr": addr, "port": port})
        return start_http_server(port, addr=addr, registry=self.registry)
