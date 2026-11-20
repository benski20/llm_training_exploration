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

__all__ = ["PENDING_SPEED_SETTINGS", "ServiceSettings"]

#: Chat-speed settings of :class:`ServiceSettings` whose implementation has not landed yet, with the
#: work package that builds it. A value other than the default (today's behaviour) is refused while the
#: field is listed; the implementing package removes its entry.
PENDING_SPEED_SETTINGS: dict[str, str] = {
    "lane": "WP-A1 (service trip lane, rust/pgw-lane)",
}


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
    triton_gemv: bool = Field(
        True,
        description="CUDA devices: serve calls with K <= 32 and at most 16 rows (decode, speculative verify) with the "
        "fused Triton ring GEMV (one kernel, wrapping 32-bit integer arithmetic, bit-identical results). Used only "
        "if triton imports and a startup self-check against the digit-GEMM path passes on the device; the first call "
        "per (linear, K, rows) is cross-checked too. false = always use the digit-GEMM path. No effect on CPU",
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
    require_client_cert_for_compute: bool | None = Field(
        None,
        description="Compute and model calls need a verified client certificate mapped to the token's tenant; a "
        "token alone gets 403 unless it was issued token-only (enrolled without a CSR, or 'pgw service token create "
        "--token-only'). The TLS layer keeps client certificates optional so enrollment, renewal and probes work "
        "without one. Off unless set to true (None = off; it never turns on implicitly). When on, needs verifiable "
        "client certificates (tls.ca_file or trusted_client_cert_header) and mtls_tenants_file (so the "
        "certificate-to-tenant mappings written at enrollment survive restarts)",
    )
    trusted_client_cert_pem_header: str | None = Field(
        None,
        description="Header carrying the verified client certificate itself (URL-encoded PEM, e.g. nginx "
        "$ssl_client_escaped_cert) from the TLS-terminating proxy that sets trusted_client_cert_header. Certificate "
        "renewal behind such a proxy needs it (the renewed certificate must be the one presented in the handshake). "
        "Only set when every request passes through that proxy.",
    )
    tokens_reload_interval_s: float = Field(
        2.0, gt=0, description="How often (at most) tokens_file is checked for changes (revocations take effect "
        "and channels of revoked tokens close within this interval)",
    )

    # ---- enrollment (pgw service enroll ..., POST /pgw/v1/enroll) --------------------------
    enrollment_file: Path | None = Field(
        None, description="Enrollment records (JSON, 0600; only hashes of the code secrets). Setting it enables "
        "POST /pgw/v1/enroll; needs tokens_file and public_url",
    )
    public_url: str | None = Field(
        None, description="Base URL gateways use to reach this service (https), written into enrollment codes",
    )
    enrollment_ca_file: Path | None = Field(
        None, description="PEM certificate of the CA that issued this service's TLS certificate. Its SHA-256 goes "
        "into enrollment codes (gateways pin it on first contact) and enrolled gateways receive it as service CA",
    )
    enrollment_audit_file: Path | None = Field(
        None, description="Append-only JSON-lines audit of enrollment attempts (never secrets or tokens; default: "
        "<enrollment_file>.audit.jsonl)",
    )
    enrollment_rate_limit_per_minute: int = Field(
        10, ge=1, description="Failed enrollment and certificate-renewal attempts per client IP per minute (behind a "
        "reverse proxy set PGW_TRUSTED_PROXY_CIDRS); at most 300 failures per minute from all addresses together",
    )
    client_issuing_ca_cert_file: Path | None = Field(
        None, description="Optional CA certificate (PEM) used to sign gateway client certificates at enrollment "
        "(when the gateway sends a CSR). Needs client_issuing_ca_key_file",
    )
    client_issuing_ca_key_file: Path | None = Field(
        None, description="Private key (PEM, 0600) of client_issuing_ca_cert_file",
    )
    client_cert_validity_days: float = Field(
        30.0, gt=0, le=397, description="Lifetime of client certificates signed at enrollment",
    )

    # ---- trip lane (chat-speed plan; WP-A0 scaffold) ----------------------------------------
    # Every default is today's behaviour; a value other than the default is refused while the field is
    # listed in PENDING_SPEED_SETTINGS (not built yet).
    lane: Literal["off", "uds", "tcp"] = Field(
        "off", description="Native trip lane next to the stream channel (binary header, same masked bytes, CUDA-graph "
        "replay for rows 1..lane_max_rows). 'off' (default) = no lane; 'uds' = a Unix domain socket at lane_socket "
        "(peer credentials checked); 'tcp' = lane_port, TLS required",
    )
    lane_socket: Path | None = Field(None, description="Unix domain socket path of the lane (lane = 'uds')")
    lane_port: int | None = Field(None, ge=0, le=65535, description="TCP port of the lane (lane = 'tcp')")
    lane_max_rows: int = Field(
        8, ge=1, le=8, description="Largest rows per item the lane serves; bigger calls use the stream channel",
    )

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
        if (
            self.require_client_cert_for_compute
            and not self.trusted_client_cert_header
            and (not self.tls.enabled or self.tls.ca_file is None)
        ):
            raise ValueError("require_client_cert_for_compute needs verified client certificates: set "
                             "tls.cert_file, tls.key_file and tls.ca_file, or trusted_client_cert_header")
        if self.require_client_cert_for_compute and self.mtls_tenants_file is None:
            raise ValueError("require_client_cert_for_compute needs mtls_tenants_file: without it the "
                             "certificate-to-tenant mappings made at enrollment are lost on restart and every "
                             "enrolled gateway's compute calls are refused")
        if self.trusted_client_cert_pem_header and not self.trusted_client_cert_header:
            raise ValueError("trusted_client_cert_pem_header needs trusted_client_cert_header (the same proxy)")
        if (self.client_issuing_ca_cert_file is None) != (self.client_issuing_ca_key_file is None):
            raise ValueError("client_issuing_ca_cert_file and client_issuing_ca_key_file must be set together")
        if self.public_url is not None:
            from pgw.enroll.code import validate_service_url

            try:
                validate_service_url(self.public_url)
            except PGWError as exc:
                raise ValueError(f"public_url: {exc.message}") from exc
        if self.enrollment_file is not None:
            if self.tokens_file is None:
                raise ValueError("enrollment_file needs tokens_file (enrolled gateways get a tenant token)")
            if self.public_url is None:
                raise ValueError("enrollment_file needs public_url (the URL written into enrollment codes)")
            if self.public_url.startswith("https://") and self.enrollment_ca_file is None:
                raise ValueError("enrollment_file with an https public_url needs enrollment_ca_file (the CA "
                                 "gateways pin on first contact)")
        if self.metrics_port is not None and self.metrics_port == self.port and self.port != 0:
            raise ValueError("metrics_port must differ from port")
        for name, package in PENDING_SPEED_SETTINGS.items():
            value = getattr(self, name)
            if value != type(self).model_fields[name].default:
                raise ValueError(f"{name}={value!r} is not available in this build (pending {package}); "
                                 "leave it at its default")
        if self.lane == "uds" and self.lane_socket is None:
            raise ValueError("lane = 'uds' needs lane_socket")
        if self.lane == "tcp":
            if self.lane_port is None:
                raise ValueError("lane = 'tcp' needs lane_port")
            if not self.tls.enabled:
                raise ValueError("lane = 'tcp' needs TLS (tls.cert_file and tls.key_file): the lane is never plain TCP")
            if self.lane_port == self.port and self.port != 0:
                raise ValueError("lane_port must differ from port")
        return self

    @property
    def compute_requires_client_cert(self) -> bool:
        """Effective ``require_client_cert_for_compute``: on only when explicitly set to true."""
        return bool(self.require_client_cert_for_compute)

    @property
    def ring_params(self) -> list[RingParams]:
        """Parsed served rings."""
        return [RingParams.from_name(r) for r in self.rings]

    @property
    def device_list(self) -> list[str]:
        """Devices used for placement (``devices`` or ``[device]``)."""
        return list(self.devices) if self.devices else [self.device]
