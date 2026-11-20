"""Compute service settings (env prefix ``PGW_SERVICE_``).

Examples::

    PGW_SERVICE_PACKS='["/srv/packs/qwen2.5-7b"]'
    PGW_SERVICE_DEVICES='["cuda:0","cuda:1"]'  PGW_SERVICE_PLACEMENT=by_layer
    PGW_SERVICE_TOKENS_FILE=/etc/pgw/tenants.txt
    PGW_SERVICE_TLS__CERT_FILE=/etc/pgw/tls/server.pem PGW_SERVICE_TLS__KEY_FILE=/etc/pgw/tls/server.key
    PGW_SERVICE_TLS__CA_FILE=/etc/pgw/tls/clients-ca.pem PGW_SERVICE_TLS__REQUIRE_CLIENT_CERT=true
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from pgw.core.errors import PGWError
from pgw.core.ring import RingParams
from pgw.core.settings import PGWSettings, TLSConfig

__all__ = ["ServiceSettings"]


class ServiceSettings(PGWSettings):
    """Compute service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_SERVICE_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    # ---- listener ------------------------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 - service is meant to listen on all interfaces
    port: int = Field(8443, ge=0, le=65535)
    metrics_port: int | None = Field(
        None, ge=0, le=65535, description="Serve /metrics on a separate plain-HTTP port (recommended)"
    )
    metrics_host: str = "127.0.0.1"
    metrics_on_main_port: bool = Field(
        False,
        description="Also expose /metrics on the main listener. Off by default: the metrics carry per-tenant "
        "series (request counts, rows, bytes, latency) that other tenants must not read",
    )
    workers_keep_alive_s: float = Field(75.0, gt=0, description="HTTP keep-alive timeout")
    server: Literal["hypercorn", "uvicorn"] = Field(
        "uvicorn",
        description="HTTP server of the main listener. 'uvicorn' (default): HTTP/1.1 (httptools) with the "
        "`websockets` framing for the /pgw/v1/stream channel, which ComputeClient uses by default (lowest per-call "
        "overhead). 'hypercorn': HTTP/2 + HTTP/1.1, for HTTP-only clients that want HTTP/2 multiplexing (its "
        "pure-Python WebSocket framing costs more per stream frame). Both publish verified mTLS identities",
    )
    tls: TLSConfig = TLSConfig()

    # ---- models and devices --------------------------------------------------------------
    packs: list[Path] = Field(default_factory=list, description="Model pack directories to serve")
    verify_packs: Literal["full", "fingerprint"] = Field(
        "full", description="Pack verification at startup ('full' hashes every linear shard)"
    )
    rings: list[str] = Field(default_factory=lambda: ["K32A12C27", "K40A12", "K48A20", "K32A8"])
    device: str = Field("auto", description="Single device: auto|cpu|cuda|cuda:i (used when devices is empty)")
    devices: list[str] = Field(
        default_factory=list, description="Devices for multi-GPU placement, e.g. ['cuda:0','cuda:1']"
    )
    placement: Literal["single", "by_layer"] = Field(
        "by_layer", description="How linears are spread over `devices` (contiguous layer ranges by bytes)"
    )
    backend: Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"] = "auto"
    cache_float_weights: bool = Field(True, description="CPU float backends keep float weight copies")
    warmup: bool = Field(True, description="Run one dummy product per linear at startup")
    cuda_graphs: bool = Field(
        False,
        description="CUDA devices: replay captured CUDA graphs for small calls (rows * K/8 <= the largest of "
        "`cuda_graph_buckets`), one graph per (linear, K, bucket), captured at warm-up or on first use and checked "
        "against the eager result. "
        "Off by default: capture is unreliable on some torch builds (2.11 + T4). "
        "Falls back to eager per shape if a capture fails. No effect on CPU",
    )
    cuda_graph_buckets: list[int] = Field(
        default_factory=lambda: [32, 64, 128],
        description="Padded GEMM row counts that get a CUDA graph (multiples of 32). With K=32 (4 byte planes) "
        "32/64/128 cover up to 8/16/32 rows",
    )

    # ---- batching / scheduling -----------------------------------------------------------
    batch_window_ms: float = Field(
        0.0, ge=0,
        description="Extra wait to coalesce rows for one linear. 0 (default) = no idle wait: a batch is sent to its "
        "device queue at once and keeps accepting rows until a worker picks it up, so batching still happens "
        "under load without adding latency to every round trip.",
    )
    max_batch_rows: int = Field(4096, gt=0)
    cpu_workers: int = Field(2, gt=0, description="Concurrent kernel executions on the CPU device")
    gpu_workers_per_device: int = Field(1, gt=0, description="Concurrent kernel executions per GPU")
    max_queued_rows: int = Field(1 << 20, gt=0, description="Above this many queued rows, reply 'unavailable'")
    batch_inline_max_rows: int = Field(
        8, ge=0,
        description="Run batches of at most this many rows (decode and speculative-verify calls) directly on the "
        "event loop when their device is completely idle: saves the kernel pool's two thread hand-offs (measured "
        "0.2-0.4 ms per call) at the cost of blocking the loop for that kernel's time. A busy device always queues "
        "and coalesces in the pool. 0 = always use the pool",
    )
    codec_offload_bytes: int = Field(
        1 << 20, ge=0, description="Payloads above this size are (un)packed in a worker thread"
    )

    # ---- limits / admission --------------------------------------------------------------
    max_request_bytes: int = Field(256 << 20, gt=4096)
    max_items: int = Field(256, gt=0)
    max_rows_per_item: int = Field(8192, gt=0)
    stream_channel: bool = Field(
        True, description="Serve the persistent WebSocket compute channel /pgw/v1/stream (lower per-call overhead "
        "than one HTTP request per call; same authentication, limits and checks)",
    )
    max_inflight_requests: int = Field(256, gt=0, description="Above this, reply 'unavailable'")
    max_inflight_bytes: int = Field(
        1 << 30, gt=0,
        description="Request-body bytes buffered at once across all connections; checked against Content-Length "
        "before a body is read (a refused request gets 'unavailable' + Retry-After, nothing is buffered)",
    )
    max_deadline_s: float | None = Field(120.0, gt=0, description="Server-side cap on compute time per request")
    tenant_max_inflight: int | None = Field(None, gt=0, description="Per-tenant concurrent request cap")
    tenant_rows_per_s: float | None = Field(None, gt=0, description="Per-tenant row rate (token bucket)")
    tenant_burst_rows: int | None = Field(None, gt=0, description="Token bucket size (default: 2 s of rate)")
    retry_after_s: float = Field(1.0, ge=0, description="Retry-After hint for 'unavailable' replies")

    # ---- auth ----------------------------------------------------------------------------
    tokens_file: Path | None = Field(None, description="Tenant tokens: lines 'tenant_id:sha256(token)'")
    mtls_tenants_file: Path | None = Field(
        None, description="Client certificate subjects: lines 'tenant_id:<subject or CN>'"
    )
    trusted_client_cert_header: str | None = Field(
        None,
        description="Header carrying the verified client-cert subject from a TLS-terminating proxy. "
        "Only set when every request passes through that proxy.",
    )
    allow_unauthenticated: bool = Field(False, description="Dev only")

    # ---- lifecycle / misc ----------------------------------------------------------------
    drain_timeout_s: float = Field(30.0, ge=0, description="Max wait for in-flight work on shutdown")
    metrics_enabled: bool = True
    server_name: str = "pgw-service"

    @field_validator("rings")
    @classmethod
    def _valid_rings(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for name in v:
            try:
                p = RingParams.from_name(name)
            except PGWError as exc:
                raise ValueError(exc.message) from exc
            if p.name not in out:
                out.append(p.name)
        if not out:
            raise ValueError("at least one ring must be served")
        return out

    @field_validator("cuda_graph_buckets")
    @classmethod
    def _valid_buckets(cls, v: list[int]) -> list[int]:
        from pgw.service.gpu_exec import validate_buckets

        try:
            return list(validate_buckets(v))
        except PGWError as exc:
            raise ValueError(exc.message) from exc

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.tls.require_client_cert and (not self.tls.enabled or self.tls.ca_file is None):
            raise ValueError("tls.require_client_cert needs tls.cert_file, tls.key_file and tls.ca_file")
        if (
            self.mtls_tenants_file is not None
            and not self.trusted_client_cert_header
            and (not self.tls.enabled or self.tls.ca_file is None)
        ):
            raise ValueError(
                "mtls_tenants_file needs verified client certificates: set tls.cert_file, tls.key_file and "
                "tls.ca_file (native mTLS) or trusted_client_cert_header (TLS-terminating proxy)"
            )
        if self.metrics_port is not None and self.metrics_port == self.port and self.port != 0:
            raise ValueError("metrics_port must differ from port")
        return self

    @property
    def ring_params(self) -> list[RingParams]:
        """Parsed served rings."""
        return [RingParams.from_name(r) for r in self.rings]

    @property
    def device_list(self) -> list[str]:
        """Devices used for placement (``devices`` or ``[device]``)."""
        return list(self.devices) if self.devices else [self.device]
