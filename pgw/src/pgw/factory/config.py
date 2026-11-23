"""Factory settings (env prefix ``PGW_FACTORY_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from pgw.core.errors import ConfigError
from pgw.core.ring import RingParams
from pgw.core.settings import PGWSettings
from pgw.maskstore.format import DEFAULT_CHUNK_BYTES, MAX_CHUNK_BYTES

__all__ = ["FactorySettings"]


class FactorySettings(PGWSettings):
    """Mask factory configuration.

    Example (env)::

        PGW_FACTORY_PACK_PATH=/packs/llama3-8b
        PGW_FACTORY_RINGS='["K32A12C27"]'
        PGW_FACTORY_LOW_WATER_SLOTS=50000
        PGW_FACTORY_TARGET_SLOTS=500000
        PGW_FACTORY_MAX_SLOTS_PER_HOUR=200000   # leave GPU headroom for other jobs
        PGW_FACTORY_NICE=10
    """

    model_config = SettingsConfigDict(
        env_prefix="PGW_FACTORY_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    pack_path: Path = Field(description="Model pack directory")
    extra_pack_paths: tuple[Path, ...] = Field((), description="Further packs kept topped up by the same factory")
    ring: str = "K32A12C27"
    rings: tuple[str, ...] = Field((), description="Rings to produce (default: just `ring`)")
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:1'")
    backend: Literal["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"] = "auto"
    segment_slots: int = Field(4096, gt=0, description="Token positions per segment")
    first_segment_slots: int | None = Field(
        None, gt=0,
        description="When a (model, ring) has no masks at all (a new store), produce one segment of this many slots "
        "first, then continue with segment_slots: a new gateway can serve after a small segment instead of waiting "
        "for a full one (minutes with encrypted generation over a slow link). Unset = segment_slots",
    )
    chunk_slots: int | None = Field(
        None, gt=0,
        description="Rows per GEMM batch (factory throughput and memory; not the stored chunk size). Unset = "
        "1024 on CUDA when the per-chunk working set of the widest linear fits half of the device memory left "
        "outside gpu_memory_fraction plus 1 GiB (halved down to 256 otherwise), 256 on the CPU and for "
        "encrypted generation",
    )
    segment_chunk_slots: int | None = Field(
        None, gt=0,
        description="Slots per encrypted W r chunk in the segment file. Unset = sized so that the widest "
        "linear's chunk is about segment_chunk_bytes. Every open gateway session caches one decrypted chunk "
        "per linear, so large chunks multiply gateway memory by max_concurrent_sessions",
    )
    segment_chunk_bytes: int = Field(
        DEFAULT_CHUNK_BYTES, gt=0, le=MAX_CHUNK_BYTES,
        description="Target plaintext bytes per encrypted chunk of the widest linear (auto chunk sizing)",
    )
    min_free_bytes: int = Field(
        2 << 30, ge=0,
        description="Stop producing when a further segment would leave less than this free on the target volume",
    )
    max_store_bytes: int = Field(
        0, ge=0, description="Cap on bytes of segment files held in the store / outbox (0 = only the free-space guard)"
    )
    checks: int | None = Field(
        None, ge=0, le=8,
        description="Freivalds vectors per linear per segment; unset = smallest number meeting 20 bits of "
        "soundness for the ring and model (gateways and stores refuse weaker segments by default)",
    )
    check_field: Literal["ring", "p31"] = Field(
        "ring",
        description="Freivalds check format of new segments (SPEC 5.1): 'ring' = vectors over Z_2^K (default); "
        "'p31' = one vector over F_p, p = 2^31 - 1, checked on the unmasked values (K32 rings; 31 bits per "
        "call with checks=1; gateways need prime_check=true to accept such segments)",
    )
    gpu_memory_fraction: float = Field(0.85, gt=0, le=1, description="Above this, stream weights from host RAM")
    factory_id: str = Field("local", description="Recorded in segment headers")
    output: Literal["store", "export"] = Field("store", description="Write into a local MaskStore or export files")
    export_dir: Path | None = None
    recipient_public_key_file: Path | None = Field(
        None, description="export: customer's raw 32-byte X25519 public key (or hex/base64 text) for HPKE wrapping"
    )
    destination_store_id: str | None = Field(
        None,
        description="export: store id of the customer mask store the segments are for ('pgw maskstore status'); "
        "recorded in the signed header so one file can never be imported into two stores",
    )
    partner_signing_key_file: Path | None = Field(
        None, description="export: partner Ed25519 signing key; each exported segment gets a .sig sidecar (SPEC 7.7)"
    )
    poll_interval_s: float = Field(30.0, gt=0, description="Scheduler inventory poll period")
    max_segments_per_run: int = Field(0, ge=0, description="0 = unlimited")
    low_water_slots: int | None = Field(None, ge=0, description="Start refilling below this (default: store setting)")
    target_slots: int | None = Field(None, ge=0, description="Refill up to this (default: store setting)")
    # ---- resource governance ----------------------------------------------------------------
    max_slots_per_hour: float = Field(0.0, ge=0, description="Rate limit on produced slots; 0 = unlimited")
    duty_cycle: float = Field(1.0, gt=0, le=1, description="Fraction of wall time spent computing (idle in between)")
    nice: int = Field(0, ge=0, le=19, description="Process niceness increment (low CPU priority)")
    cpu_threads: int = Field(0, ge=0, description="torch intra-op threads; 0 = torch default")
    pipeline: bool = Field(True, description="Overlap r generation, GEMMs and encryption in threads")
    self_check_rows: int = Field(
        8, ge=-1, description="Rows per chunk whose W8.r output is verified before writing (-1 = all, 0 = off)"
    )
    gpu_prng: Literal["auto"] | bool = Field(
        "auto",
        description="Expand mask rows r (ChaCha20 + digit split) on the factory GPU instead of the CPU: 'auto' = on "
        "for CUDA factories once the kernel's self-test against the CPU expander passed, false = always CPU, "
        "true = on (still self-tested; never used when the test fails). Needs self_check_rows != 0 (the output "
        "self-check re-derives sampled r rows on the CPU). Same derivation and output",
    )
    masked_head: bool = Field(
        False, description="Masks for the masked LM head (opt-in; pgw.engine.masked_head): every pack is wrapped in "
        "MaskedHeadPack, so segments are made for its derived fingerprint and carry the head linear "
        "'<num_layers>.lm_head' (its int8 weight derived here from the pack's hash-checked plain head; the pack "
        "needs its plain tensors). For gateways with head_mode = 'provider_int8'. The encrypted factory extends "
        "the base pack's hint (same public seed) instead of rebuilding it. Default off",
    )
    skip_linear_kinds: str = Field(
        "", description="Comma-separated linear kinds (qkv, o, gate_up, down) whose masks are NOT generated: "
        "opt-in, for gateways whose local_linears is fixed in configuration (e.g. 'attention' with local_max_rows "
        "unset: set 'qkv,o') and that therefore never send those linears to the service. Segments then list only "
        "the generated linears in their header. A gateway that needs a skipped linear never uses such a segment: "
        "it retires it (burned, never reused) and the factory must produce full segments again (unset this). "
        "Not supported with in-memory mask rings (their layout is fixed). Default: every linear",
    )
    schedule: Literal["standard", "fused2"] = Field(
        "standard", description="Masks for the gateway's linear schedule: 'standard' (default: one entry per pack "
        "linear) or 'fused2' (opt-in, for gateways with schedule = 'fused2'): fused2 pair-mask segments "
        "(pgw.factory.fused2_gen.Fused2MaskGenerator: one entry per fused2 linear, p61 check words, schedule "
        "'fused2' headers) under the fused2 lane fingerprint, for every pack and ring (rings must be capped, K a "
        "multiple of 8, e.g. K56A12C53). The public fused2 weights are derived from each pack like the gateway's "
        "and the service's (fused2_migrate_bits) or loaded from fused2_weights_path. Encrypted mode (he_service_url) "
        "computes each product as independent uniform digits on the existing HE schemes (mlwe v2 at K40 with "
        "14-bit digits, or pks-v3 at K32 with 8-bit digits) recombined exactly mod 2^K",
    )
    fused2_migrate_bits: int = Field(
        8, ge=0, le=8, description="schedule = 'fused2': residual-grid migration bits (must match the gateway's "
        "fused2_migrate_bits and the service's PGW_SERVICE_FUSED2_FROM_PACK; the lane fingerprint binds them)",
    )
    fused2_weights_path: Path | None = Field(
        None, description="schedule = 'fused2': a Fused2Weights.save() file (provider-only is enough) instead of "
        "deriving the weights from the pack (one pack only)",
    )
    # ---- GPU mask service (pgw.factory.mask_service; customer-cloud deployment, opt-in) ----------
    mask_service: bool = Field(
        False, description="Run as the customer-cloud GPU mask service (deploy/customer-cloud): the plain W8 r "
        "factory on the customer's own GPU from its own copy of the public pack. Before the first segment a "
        "throw-away segment is generated on the device and compared bit for bit with an independent CPU reference "
        "(refuses to start on any difference); device memory is scrubbed after every segment; encrypted "
        "generation, self_check_rows=0, a CPU device (unless mask_service_allow_cpu) and unsigned or unbound "
        "export are refused. Delivery: output='store' (gateway store on the same host) or output='export' (HPKE + "
        "signature, pulled by the gateway with 'pgw maskstore pull'). Off by default",
    )
    mask_service_allow_cpu: bool = Field(
        False, description="mask_service: allow running on a CPU device (tests, small models, measurement)")
    mask_service_check_rows: int = Field(
        4, ge=2, le=64, description="mask_service start-up check: sampled W r rows per checked linear")
    mask_service_check_linears: int = Field(
        8, ge=0, description="mask_service start-up check: linears compared with the CPU reference (first and last "
        "layer plus evenly spaced ones; 0 = every linear)")
    # ---- encrypted mask generation (pgw.hemask) -----------------------------------------------
    he_service_url: str | None = Field(
        None, description="Compute service URL: get W8.r from the provider on MLWE-encrypted masks instead of "
        "computing it locally (no local GEMMs; the provider learns nothing about the masks)",
    )
    he_token_file: Path | None = Field(None, description="File with the service bearer token (encrypted mode)")
    he_ca_file: Path | None = Field(None, description="CA bundle for the service (encrypted mode)")
    he_client_cert_file: Path | None = Field(
        None, description="mTLS client certificate (PEM) presented to the service (encrypted mode): the gateway's "
        "enrolled client.crt, needed when the service requires client certificates for compute. Renewals of the "
        "file are picked up by new connections",
    )
    he_client_key_file: Path | None = Field(None, description="Private key of he_client_cert_file (PEM, 0600)")
    he_hint_dir: Path | None = Field(None, description="Where the one-time hint W8.A is kept (default: <state>/hemask)")
    he_rank: int = Field(16, ge=8, le=32, description="MLWE module rank (LWE dimension = 256 x rank)")
    he_params_ack: str = Field(
        "", description="Opt in to an unreviewed HE parameter profile: ranks below 16 are refused unless this is the "
        "profile's acknowledgement string (rank 12: pgw.hemask.mlwe_ring.R12_ACK). See "
        "research/plans/he-structured-encryption.md",
    )
    he_scheme: Literal["mlwe", "pks-v3"] = Field(
        "mlwe", description="Encrypted mask scheme: 'mlwe' (default: SPEC 7.9, wire v1/v2, hint W8.A of 4096 "
        "coefficients per output row) or 'pks-v3' (compact hint of 256 coefficients per row, packed key-switched "
        "replies, wire v3: research/plans/he-compact-hint-pks.md; needs a K32 ring, a service with "
        "PGW_SERVICE_MASKGEN_V3=true and he_pks_ack until the external review signs off)",
    )
    he_pks_ack: str = Field(
        "", description="Opt in to the unreviewed pks-v3 scheme: must equal pgw.hemask.pks.PKS_V3_ACK "
        "('pks-v3-unreviewed-n8192') while research/reviews/pks-v3-review-packet.md is open",
    )
    he_encrypt: Literal["dense", "ntt_pre", "ntt_dev", "gpu_ntt"] = Field(
        "dense", description="Gateway encryption product: 'dense' (byte-plane GEMM on the int8 engine), 'ntt_pre' "
        "(public matrix kept in NTT form, pgw.hemask.mlwe_ring; same ciphertexts bit for bit, ~10x+ fewer CPU "
        "operations; needs pgw_ntt) or 'ntt_dev' (the same NTT product, and decryption's hint product, on the "
        "gateway GPU, pgw.hemask.ntt_dev; self-tested at start-up, falls back to the CPU) or 'gpu_ntt' (the gateway's "
        "own GPU, pgw.hemask.gpuntt: A in NTT form on the device, fused kernels on a low-priority stream; "
        "self-tested and cross-checked bit for bit against the CPU path at start-up, else 'ntt_pre' on the CPU)",
    )
    he_decrypt: Literal["auto", "dense", "ntt", "ntt_pre", "gpu_ntt"] = Field(
        "auto", description="Gateway decryption product on the CPU (pgw.hemask.backend.HEMaskBackend): 'dense' "
        "(byte-plane GEMM on the int8 engine plus the native combine), 'ntt' (fused native NTT decryption on the "
        "stored hint, no extra memory), 'ntt_pre' ('ntt' with the hints of the linears that fit he_hint_ntt_gb "
        "kept in NTT form in RAM), 'auto' (default: hints in NTT form within he_hint_ntt_gb, else per pass size "
        "the faster of 'dense' and 'ntt' as measured at start-up on public random data, 'ntt' only if both gave "
        "the same bits), 'gpu_ntt' (on the gateway's own GPU, pgw.hemask.gpuntt, hint planes cached on the device; "
        "self-tested and cross-checked bit for bit against the CPU path at start-up, else 'auto'). Same bits in "
        "every mode; a native NTT error falls back to 'dense', a GPU error to the CPU",
    )
    he_hint_ntt_gb: float = Field(
        0.0, ge=0, description="RAM (GB) for hints kept in NTT form ('ntt_pre' / 'auto'): 12 bytes per coefficient, "
        "twice the stored hint (Qwen2.5-0.5B: about 15 GB for all linears). 0 = none (default)",
    )
    he_service_urls: tuple[str, ...] = Field(
        (), description="Encrypted mode with several compute services (provider GPUs or hosts): segments are sharded "
        "over them, never their contents: every request of one segment goes to one service. Worker k of a fleet "
        "starts on url[k % len] and keeps it (a failed segment moves the next one to the next service); with fewer "
        "workers than services each worker rotates per segment. Overrides he_service_url",
    )
    he_groups_per_call: int = Field(4, ge=1, le=16, description="256-slot groups per provider request")
    he_max_inflight: int = Field(2, ge=1, le=4, description="Provider requests in flight at once (encrypted mode)")
    he_pipeline_depth: int = Field(
        0, ge=0, le=8, description="Encrypted mode: chunks (across linears) outstanding at once, so that encryption, "
        "the provider's work and decryption of consecutive chunks overlap (0 = stages in series per chunk). Same "
        "requests, ciphertexts and masks; the provider sees the same sizes and counts",
    )
    he_timeout_s: float = Field(600.0, gt=0, description="Budget per provider request")
    he_wire_v2: bool = Field(
        True, description="Offer maskgen wire v2 (bit-packed: q-3 bit uploads, K+3 bit replies; about 10% fewer "
        "bytes than v1) for every linear whose noise budget allows it (pgw.hemask.client). Every factory row is "
        "verified after decryption either way",
    )
    he_bulk_port: int | None = Field(
        None, ge=1, le=65535, description="Port of the service's bulk maskgen channel (service maskgen_bulk_port; "
        "same host and TLS settings as he_service_url). Several times the HTTP route's throughput; requests fall "
        "back to HTTP when it is unreachable. Off (None) by default",
    )
    he_bulk_split: int = Field(
        1, ge=1, le=8, description="With he_bulk_port: a request of several 256-slot groups is sent as up to this "
        "many concurrent requests of whole groups (parallel TLS and packing)",
    )
    # ---- horizontal scale-out (pgw.factory.fleet) ------------------------------------------------
    workers: int = Field(
        1, ge=1, le=64,
        description="Factory worker processes feeding the same mask store concurrently (pgw.factory.fleet). Workers "
        "coordinate through an in-flight ledger next to the store so that together they stop at target_slots; each "
        "has its own state directory and lock. 1 = the classic single-process factory",
    )
    devices: tuple[str, ...] = Field(
        (), description="Per-worker devices for local generation: worker k uses devices[k % len] (e.g. "
        "'[\"cuda:0\", \"cuda:1\"]'). Empty = `device` for every worker",
    )
    # ---- crash safety -----------------------------------------------------------------------
    state_dir: Path | None = Field(None, description="Factory state/lock directory (default: store root)")
    error_backoff_s: float = Field(10.0, gt=0, description="Initial back-off after a failed run (doubles to 600 s)")
    # ---- observability ----------------------------------------------------------------------
    metrics_port: int | None = Field(
        None, ge=1, le=65535, description="Serve /metrics and /healthz on this internal port (unset = off)"
    )
    metrics_host: str = Field("0.0.0.0", description="Bind address of the metrics/health listener")  # noqa: S104
    health_max_stale_s: float = Field(
        3600.0, gt=0,
        description="/healthz fails when no scheduler plan/segment completed for this long (hung GPU, lock wait)",
    )
    health_max_consecutive_errors: int = Field(
        5, ge=1, description="/healthz fails after this many consecutive failed runs"
    )

    @field_validator("ring")
    @classmethod
    def _valid_ring(cls, v: str) -> str:
        return RingParams.from_name(v).name

    @field_validator("rings")
    @classmethod
    def _valid_rings(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        out: list[str] = []
        for r in v:
            name = RingParams.from_name(r).name
            if name not in out:
                out.append(name)
        return tuple(out)

    @field_validator("skip_linear_kinds")
    @classmethod
    def _valid_skip(cls, v: str) -> str:
        kinds = [k.strip() for k in str(v or "").split(",") if k.strip()]
        allowed = ("qkv", "o", "gate_up", "down")
        bad = [k for k in kinds if k not in allowed]
        if bad:
            raise ValueError(f"skip_linear_kinds: unknown kinds {bad} (allowed: {', '.join(allowed)})")
        if set(kinds) >= set(allowed):
            raise ValueError("skip_linear_kinds cannot skip every linear kind")
        return ",".join(k for k in allowed if k in kinds)  # canonical order, de-duplicated

    @property
    def skip_kinds(self) -> frozenset[str]:
        """Linear kinds whose masks are not generated (``skip_linear_kinds``)."""
        return frozenset(k for k in self.skip_linear_kinds.split(",") if k)

    @model_validator(mode="after")
    def _check(self) -> FactorySettings:
        if self.output == "export" and self.export_dir is None:
            raise ValueError("output='export' requires export_dir")
        lw, tg = self.low_water_slots, self.target_slots
        if lw is not None and tg is not None and tg < lw:
            raise ValueError("target_slots must be >= low_water_slots")
        if self.workers > 1 and self.output != "store":
            raise ValueError("workers > 1 needs output='store' (export staging is per process)")
        if (self.he_client_cert_file is None) != (self.he_client_key_file is None):
            raise ValueError("he_client_cert_file and he_client_key_file must be set together")
        if self.mask_service:
            from pgw.factory.mask_service import check_mask_service_settings

            try:
                check_mask_service_settings(self)
            except ConfigError as exc:
                raise ValueError(str(exc)) from exc
        from pgw.hemask.mlwe_ring import check_rank

        try:
            check_rank(self.he_rank, self.he_params_ack)
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc
        if self.he_scheme == "pks-v3":
            from pgw.hemask.pks import PKS_V3_ACK

            if self.he_pks_ack.strip() != PKS_V3_ACK:
                raise ValueError("he_scheme='pks-v3' has not had its external cryptographic review; set "
                                 f"he_pks_ack={PKS_V3_ACK!r} to opt in (research/reviews/pks-v3-review-packet.md)")
            # fused2: the K56 masks are recombined from K32 products of 8-bit digits (the HE ring stays K32)
            bad = [r for r in self.ring_names if not r.upper().startswith("K32")] if self.schedule != "fused2" else []
            if bad:
                raise ValueError(f"he_scheme='pks-v3' covers K32 mask rings only, not {bad}")
        if self.schedule == "fused2":
            for r in self.ring_names:
                rp = RingParams.from_name(r)
                if rp.k % 8 or not rp.cap or rp.cap + 1 >= 61 or rp.k > 56:
                    raise ValueError(f"schedule='fused2' needs capped rings with K a multiple of 8 (<= 56) and "
                                     f"2^(C+1) < 2^61 - 1, e.g. K56A12C53; not {r}")
            if self.masked_head or self.skip_kinds:
                raise ValueError("schedule='fused2' makes every fused2 linear (no masked_head, no skip_linear_kinds)")
            if self.check_field != "ring":
                raise ValueError("schedule='fused2' segments carry p61 check words (leave check_field = 'ring')")
            if self.fused2_weights_path is not None and len(self.all_pack_paths) > 1:
                raise ValueError("fused2_weights_path serves one pack only")
        return self

    @property
    def he_urls(self) -> tuple[str, ...]:
        """Compute services for encrypted generation (``he_service_urls``, else ``he_service_url``; may be empty)."""
        if self.he_service_urls:
            return tuple(dict.fromkeys(self.he_service_urls))
        return (self.he_service_url,) if self.he_service_url else ()

    @property
    def ring_names(self) -> tuple[str, ...]:
        """Rings to produce, de-duplicated."""
        return self.rings or (self.ring,)

    @property
    def all_pack_paths(self) -> tuple[Path, ...]:
        """Primary plus extra packs."""
        return (self.pack_path, *[p for p in self.extra_pack_paths if p != self.pack_path])
