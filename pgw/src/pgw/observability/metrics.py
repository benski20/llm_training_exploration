"""Prometheus metric definitions (names are part of the contract; see SPEC.md section 11.1)."""

from __future__ import annotations

from typing import Any

__all__ = ["METRIC_NAMES", "PGWMetrics"]

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
    "pgw_gateway_integrity_failures_total": ("counter", ("model", "check"), "Freivalds/range check failures"),
    "pgw_gateway_active_sessions": ("gauge", ("model",), "Active generation sessions"),
    "pgw_gateway_noise_alarm": ("gauge", ("model",), "1 if the outbound-noise monitor alarms"),
    # mask store / factory
    "pgw_masks_available_slots": ("gauge", ("fingerprint", "ring"), "Unconsumed mask slots"),
    "pgw_masks_consumed_slots_total": ("counter", ("fingerprint", "ring"), "Consumed (incl. burned) slots"),
    "pgw_masks_burned_slots_total": ("counter", ("fingerprint", "ring"), "Reserved but unused slots"),
    "pgw_masks_low_water": ("gauge", ("fingerprint", "ring"), "1 if below low water"),
    "pgw_factory_slots_generated_total": ("counter", ("fingerprint", "ring", "device"), "Slots generated"),
    "pgw_factory_slots_per_second": ("gauge", ("fingerprint", "ring", "device"), "Generation throughput"),
}


class PGWMetrics:
    """Lazily creates Prometheus collectors from :data:`METRIC_NAMES` on a registry.

    ``metrics.get("pgw_service_rows_total").labels(tenant="t", linear_kind="qkv").inc(32)``
    """

    def __init__(self, registry: Any | None = None, *, enabled: bool = True):
        raise NotImplementedError("observability team: PGWMetrics")

    def get(self, name: str) -> Any:
        """Collector for ``name`` (KeyError if not in :data:`METRIC_NAMES`); no-op stub if disabled."""
        raise NotImplementedError

    def asgi_app(self) -> Any:
        """ASGI app serving ``/metrics`` for this registry."""
        raise NotImplementedError
