"""Gateway settings (env prefix ``PGW_GATEWAY_``; models usually from a JSON/YAML file).

Production defaults are fail-closed:

* every model runs in ``masked`` mode; the local ``plain``/``float`` modes (no compute service, used
  for development and tests) are refused unless ``allow_dev_modes`` is set;
* Freivalds checks run on every compute response (``freivalds_rate = 1.0``);
* the gateway only talks to the configured compute-service endpoints (egress allowlist);
* anonymous access is refused unless ``allow_dev_modes`` is set;
* masked models must pin their pack fingerprint, and the pack is fully hashed at start-up;
* an integrity failure keeps the circuit breaker open until an admin reset;
* the outbound-noise monitor is required for masked models;
* mask segments must give at least ``min_soundness_bits`` of Freivalds soundness.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from pgw.core.errors import ConfigError
from pgw.core.ring import RingParams
from pgw.core.settings import PGWSettings, TLSConfig

__all__ = [
    "PENDING_SPEED_SETTINGS", "SETUP_MODEL_OPTION_KEYS", "GatewayModelConfig", "GatewaySettings", "load_models_file",
    "origin_of",
]

log = logging.getLogger(__name__)

#: :attr:`GatewaySettings.setup_model_options` keys: speed settings that never change outputs or weaken a
#: check (exactness is kept bit for bit; privacy is unchanged), so a deployment may preset them for the wizard.
SETUP_MODEL_OPTION_KEYS: frozenset[str] = frozenset({
    "lane", "lane_uds_dir", "lane_pool_wait_s", "local_linears", "local_linears_device", "local_rtt_ms",
    "local_max_rows", "lm_head_device", "float_threads",
    "spec_rows", "spec_strict_passes", "spec_mask_rows", "spec_mask_credit", "spec_lookup", "spec_lookup_min_n",
    "spec_tokens", "inline_max_rows",
    "batch_lm_head", "engine_pingpong", "max_concurrent_sessions", "draft_pack_path", "draft_device",
})
#: Draft sources among the presets: a preset that enables one must also enable strict fixed-row speculation
#: (``spec_rows`` + ``spec_strict_passes``), so a deployment preset can never switch on the variable-row mode,
#: whose row and pass counts depend on how predictable the text is.
_PRESET_DRAFT_SOURCES = ("spec_lookup", "draft_pack_path")

#: Modes that run without the compute service (no privacy protocol involved). Development only.
DEV_MODES: frozenset[str] = frozenset({"plain", "float"})

#: Shortest accepted admin token (guessable tokens would expose /admin/*).
_MIN_ADMIN_TOKEN_LEN = 16

#: Chat-speed settings of :class:`GatewayModelConfig` whose implementation has not landed yet, with the
#: work package that builds it. A value other than the field's default (today's behaviour) is refused
#: while the field is listed, so a configuration can never ask for a mode (e.g. strict-pass
#: speculation) that silently does not happen. The implementing package removes its entry.
PENDING_SPEED_SETTINGS: dict[str, str] = {
    "head_mode": "WP-B2 (provider int8 LM head)",
    "schedule": "WP-B4 (fused2 schedule)",
    "prefix_cache": "WP-B3 (prefix cache)",
}


def _refuse_pending_speed_settings(cfg: BaseModel, pending: dict[str, str], where: str = "") -> None:
    """Raise ``ValueError`` if a field listed in ``pending`` is set to anything but its default."""
    for name, package in pending.items():
        value = getattr(cfg, name)
        if value != type(cfg).model_fields[name].default:
            raise ValueError(f"{where}{name}={value!r} is not available in this build (pending {package}); "
                             "leave it at its default")


def origin_of(url: str) -> tuple[str, str, int]:
    """``(scheme, host, port)`` of an ``http(s)`` URL (default ports filled in), lower-cased.

    Raises:
        ConfigError: not an absolute http(s) URL.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"not an absolute http(s) URL: {url!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"invalid port in URL {url!r}") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, parts.hostname.lower(), int(port)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def network_path(url: str | None) -> str:
    """Label of the network path to a compute service URL, shown next to round-trip figures (admin status,
    start-up log): ``loopback`` (service on this host), ``private <ip>`` (a private or unique-local
    address, e.g. a cloud's private network), ``public <host>`` (anything else) or ``unknown``."""
    try:
        _scheme, host, _port = origin_of(url or "")
    except ConfigError:
        return "unknown"
    host = host.strip("[]")
    if _is_loopback(host):
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return f"public {host}"
    return f"private {ip}" if ip.is_private else f"public {ip}"


class GatewayModelConfig(BaseModel):
    """One served model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name clients use in 'model'", min_length=1, max_length=128)
    pack_path: Path
    fingerprint: str | None = Field(None, description="Pin the expected pack fingerprint (64 hex)")
    mode: Literal["masked", "plain", "float"] = Field(
        "masked", description="'masked' = production; 'plain'/'float' = local dev/testing only (no service)"
    )
    ring: str = "K32A12C27"
    draft_pack_path: Path | None = Field(
        None, description="Speculative decoding: a small model pack with the same tokenizer (e.g. Qwen2.5-0.5B for "
        "Qwen2.5-7B), run locally in plaintext on the gateway to propose tokens the target checks in one masked pass",
    )
    spec_tokens: int = Field(
        2, ge=0, le=16, description="Draft tokens proposed per speculative pass (0 disables speculation). Speculation "
        "runs only with a draft model (draft_pack_path) or prompt lookup (spec_lookup), both off by default. A "
        "draft model on a CPU-only gateway does not pay off (the draft passes cost more than the round trips "
        "saved); use a gateway GPU (draft_device) or prompt lookup",
    )
    spec_min_tokens: int = Field(
        2, ge=0, le=16, description="Adaptive draft length (draft model only): draft tokens always proposed; the "
        "full spec_tokens only while the draft is confident (spec_confidence). Equal to spec_tokens (the default): "
        "one fixed size, and no per-token confidence check",
    )
    spec_confidence: float = Field(0.3, ge=0.0, lt=1.0, description="Draft top-probability needed to extend a draft")
    spec_row_budget: float = Field(
        2.0, ge=1.0, le=17.0, description="Mask rows reserved per requested token when speculating (speculation "
        "only ever uses rows the request already holds; unused rows are burned when the request ends). Rejected "
        "drafts use rows; when the spare rows run out the request finishes without speculation. 2.0 suits "
        "spec_tokens=2; raise it with longer drafts",
    )
    spec_max_batch: int = Field(
        1, ge=1, le=1024, description="Speculate only in engine steps with at most this many sequences. Verify rows "
        "of one sequence slow every linear of a shared step, so concurrent streams lose more than they gain",
    )
    spec_lookup: int = Field(
        0, ge=0, le=8, description="Prompt-lookup drafting: propose what followed the latest earlier match of the "
        "last N (down to spec_lookup_min_n) context tokens, before (or without) a draft model; free to compute, "
        "strong on RAG, code and summaries. Rejected proposals use mask rows. 0 disables; 3 is a good value",
    )
    spec_lookup_min_n: int = Field(
        3, ge=2, le=8, description="Shortest context match prompt lookup accepts (no 1-gram coincidences). A "
        "proposal is always spec_tokens long: a shorter match is completed by the draft model, or dropped",
    )
    draft_device: str | None = Field(None, description="Device of the draft model (default: the model's device)")
    service_url: str | None = None
    service_token: SecretStr | None = None
    service_token_file: Path | None = Field(
        None, description="File holding the service bearer token (read at start-up; exclusive with service_token)"
    )
    service_ca_file: Path | None = None
    client_cert_file: Path | None = None
    client_key_file: Path | None = None
    service_timeout_s: float = Field(60.0, gt=0, description="Budget per compute round trip incl. retries")
    service_max_retries: int = Field(3, ge=0)
    service_deadline_ms: int | None = Field(None, gt=0, description="Deadline sent with each compute request")
    freivalds_rate: float = Field(1.0, ge=0.0, le=1.0)
    max_context: int | None = Field(None, gt=1)
    max_concurrent_sessions: int = Field(32, ge=1, description="Sequences generating at once (engine batch)")
    max_queue: int = Field(256, ge=0, description="Requests waiting for a session slot before 503")
    queue_timeout_s: float = Field(30.0, gt=0, description="Max wait in the request queue")
    device: str = "cpu"
    inline_max_rows: int = Field(
        8, ge=0, description="Masked round trips of at most this many rows (decode) run their gateway-side work "
        "(verify and unmask the results, the plaintext step, mask the next calls) on the event loop when the model's "
        "worker thread is idle, saving two thread switches per round trip (about 0.3 ms each measured on a busy "
        "4-vCPU host). Steps with the LM head always use the worker thread. 0 = always use the worker thread",
    )
    batch_lm_head: bool | None = Field(
        None, description="One LM-head pass for all rows of a decode step (every concurrent sequence's last "
        "position and all rows of speculative verify passes): the float32 head weights are read once per step "
        "instead of once per row. Outputs are bit-identical either way (the LM head uses a fixed summation order, "
        "so a row's logits do not depend on the rows it shares a pass with); this only trades speed. None = "
        "automatic: on with the native kernel (pgw_fast), off with the slower torch fallback",
    )
    engine_pingpong: bool = Field(
        False, description="Two-cohort ping-pong: an engine step with at least engine_pingpong_min_sessions "
        "sequences runs as two concurrent half-steps, so the gateway masks and unmasks one half while the other "
        "half's round trip is with the service. Outputs and per-session mask use are identical; the service sees "
        "two round trips per linear (half the rows each) instead of one. Not used with a draft model",
    )
    engine_pingpong_min_sessions: int = Field(
        5, ge=2, le=1024, description="Smallest engine step (sequences) that ping-pong splits (default: more than 4)",
    )

    # ---- chat-speed plan (WP-A0 scaffold). Every default is today's behaviour; a value other than the
    # default is refused while its package is listed in PENDING_SPEED_SETTINGS (not built yet).
    lane: Literal["off", "auto", "required", "stream", "tls"] = Field(
        "off", description="Native trip lane to the service (a binary header over a Unix domain socket; the provider "
        "sees the same masked bytes as on the stream channel). 'off' (default) = the WebSocket stream channel only; "
        "'auto' = the lane when the service offers one (lane = 'uds' on a service on this host, or a socket inside "
        "lane_uds_dir), else the stream channel; any call the lane cannot take is resent byte-identically on the "
        "stream channel. 'required' = as 'auto', but the model does not start (stays 'connecting') until the "
        "service grants a lane ticket. 'auto' and 'required' take UDS tickets only. With the lane the "
        "gateway also uses the native trip engine (constant-time masking kernels, deferred Freivalds output half) "
        "and runs decode passes synchronously on its plaintext thread. 'stream' = the blocking stream lane for a "
        "remote service (any network, TLS as configured): decode trips go over the same WebSocket stream endpoint "
        "with the same msgpack bytes, but are written and read with blocking socket calls on the plaintext thread "
        "(no event-loop hand-off per trip); TripEngine and synchronous decode as with the UDS lane; any call it "
        "cannot complete is resent byte-identically on the stream channel. 'tls' = the native TCP+TLS trip lane of a "
        "service with lane = 'tcp' (needs an https service_url with service_ca_file; TLS 1.3 to the service_url "
        "host on the port the ticket names, the server verified against service_ca_file, the client certificate "
        "presented when configured and bound to the ticket by the service; the lane port must be on egress_allowlist "
        "as https://<service host>:<port>, else the lane is not used): binary header instead of msgpack, rustls "
        "instead of Python ssl, the GIL released for the whole trip; the same masked bytes, request ids and call "
        "schedule. A call the lane cannot take, and every lane failure, goes to the blocking stream lane ('stream') "
        "with the byte-identical request, and from there to the stream channel",
    )
    lane_uds_dir: Path | None = Field(
        None, description="Directory the service's lane socket must be in (for a service on this host that is not "
        "reached on a loopback URL). None = accept a lane socket only from a service on a loopback URL",
    )
    lane_pool_wait_s: float = Field(
        0.01, ge=0, le=1.0, description="When every lane connection is busy, wait this long for one before sending "
        "the call on the stream channel",
    )
    local_linears: Literal["off", "attention", "auto", "attention_down", "all", "auto_kinds"] = Field(
        "off", description="Exact local linears: 'off' (default) = every linear goes to the service; 'attention' = "
        "qkv and o are computed exactly on the gateway from the int8 pack (same integers, so the same bits) and only "
        "gate_up/down are sent (half the round trips per token; the gateway then holds the attention int8 weights "
        "in memory); 'auto' = 'attention' when one layer's local qkv + o costs less than two round trips, decided "
        "once at start-up (local_rtt_ms, else a start-up measurement of a few content-free round trips; public "
        "configuration, never per token). Pays off on links slower than about 2 ms. R1 (research/plans/"
        "fewer-round-trips.md): 'attention_down' = also down (one round trip per layer), 'all' = every linear "
        "(no round trip), 'auto_kinds' = each kind local iff its calibrated local time is below one round trip",
    )
    local_max_rows: int | None = Field(
        None, ge=1, description="Local linears apply only to engine steps of at most this many rows in total "
        "(decode); larger steps (prefill) send every linear, so a long prompt never runs the big linears on the "
        "gateway CPU. None = every step (today's behaviour). Depends only on row counts, already public",
    )
    local_linears_device: Literal["cpu", "cuda", "auto"] = Field(
        "cpu", description="Where the exact integer products of the local linears run: 'cpu' (default), 'cuda' (a "
        "gateway GPU) or 'auto' (the GPU when it is faster). The GPU is used only after a start-up check found its "
        "integer outputs identical to the CPU's; quantization, the range check and dequantization stay on the CPU, "
        "so the outputs are the same bits either way",
    )
    lm_head_device: Literal["cpu", "cuda"] = Field(
        "cpu", description="Where the float32 LM head runs: 'cpu' (default, the native kernel) or 'cuda' (a "
        "gateway GPU: the same fixed summation order as one GPU kernel, pgw.engine.lm_head_gpu). The GPU is used "
        "only after a start-up check found its logits identical to the CPU kernel's bit for bit; otherwise the "
        "CPU is kept. PGW_LM_HEAD_DEVICE overrides it",
    )
    local_rtt_ms: float | None = Field(
        None, gt=0, description="Round trip (ms) local_linears = 'auto' compares with. None = measured at start-up "
        "with a few mask-free dummy round trips (uniform random payloads; no mask is used)",
    )
    spec_rows: int | None = Field(
        None, ge=2, le=8, description="Strict speculation with a fixed row count: every decode pass sends exactly "
        "this many rows per sequence (the pending token, the drafts, then masked random filler rows), so the "
        "provider never sees how many drafts were proposed. Needs spec_strict_passes and a draft source "
        "(spec_lookup or draft_pack_path). Each pass then uses spec_rows mask rows per sequence, so a request "
        "reserves prompt + max_tokens + (max_tokens - 1) * (spec_rows - 1) mask slots. None (default) = today's "
        "variable-row speculation (1 + the drafts of that pass) when spec_lookup / draft_pack_path is set",
    )
    spec_mask_rows: float | None = Field(
        None, ge=1, le=8, description="Padded strict speculation with a public mask charge (L3; needs spec_rows and "
        "spec_strict_passes): filler rows and dummy passes are padding rows (ChaCha20 ring elements like a mask's r, "
        "no mask, Freivalds-checked like real rows; the gateway's work per call depends only on the row count) and "
        "every decode pass, dummy passes included, charges C = this many mask slots (1 <= C <= spec_rows, "
        "fractions allowed, e.g. 1.25); drafts spend the credit earlier passes left. Masks per generated token: C "
        "instead of spec_rows. The store is asked for slots on a public schedule only (prompt + floor(C x pass "
        "index)), so mask demand reveals nothing about the drafts; a request reserves prompt + floor(C x "
        "(max_tokens - 1)) slots. None (default) = masked filler rows (spec_rows masks per pass)",
    )
    spec_mask_credit: int | None = Field(
        None, ge=0, le=1 << 20, description="With spec_mask_rows: cap of the draft credit charged per request (B "
        "slots): the schedule becomes prompt + j + min(B, floor((C - 1) x j)), so a high C front-loads the draft "
        "budget into the answer's passes and a long answer costs about 1 + B / L masks per token. Public, like C. "
        "None (default) = no cap",
    )
    spec_strict_passes: bool = Field(
        False, description="Strict-pass speculation (needs spec_rows): after the answer, dummy passes on random "
        "data run the real code path until the provider has seen as many passes as tokens generated, so the pass "
        "count reveals nothing beyond the length L it already sees. The answer streams first; the next request "
        "of the same conversation (API key and OpenAI 'user') starts after those passes. False (default) = off",
    )
    float_threads: int | None = Field(
        1, ge=1, le=256, description="Torch threads for the gateway's plaintext float ops of decode passes "
        "(norms, RoPE, attention); enabled only when a start-up self-check shows the same bits as the default "
        "thread count, else the process thread count is kept. 1 (default) measured 2-5 ms per token faster on a "
        "4-vCPU gateway with one chat and 36% less float time per engine step with 32 chats (0.5B shapes). Only "
        "CPU engines. None = the process thread count, or PGW_FLOAT_THREADS",
    )
    float_ops: Literal["torch", "native"] = Field(
        "torch", description="Gateway plaintext float path (norms, RoPE, attention): 'torch' (default, today's "
        "reference); 'native' = the pinned native formulation (pgw_fast: fixed-order RMSNorm, SiLU and "
        "row-deterministic attention; a new reference: logits differ from 'torch' in the last bits, so the plaintext "
        "reference compared against must use it too, e.g. PGW_FLOAT_OPS=native). Unset: PGW_FLOAT_OPS or 'torch'",
    )
    head_mode: Literal["gateway", "provider_int8"] = Field(
        "gateway", description="LM head: 'gateway' (default) = the exact float32 head on the gateway CPU; "
        "'provider_int8' = an int8 head as one more masked linear on the service (a new reference)",
    )
    schedule: Literal["standard", "fused2"] = Field(
        "standard", description="Masked linear schedule per layer: 'standard' (default, 4 round trips: qkv, o, "
        "gate_up, down); 'fused2' = two composite round trips (a new reference, gated on a 7B quality check)",
    )
    prefix_cache: Literal["off", "conversation"] = Field(
        "off", description="Reuse the KV cache of earlier turns: 'off' (default); 'conversation' = within one "
        "conversation only (the provider then sees prefill rows = the uncached suffix length, a function of public "
        "token counts)",
    )

    @field_validator("ring")
    @classmethod
    def _ring(cls, v: str) -> str:
        try:
            return RingParams.from_name(v).name
        except ConfigError as exc:
            raise ValueError(exc.message) from exc

    @field_validator("fingerprint")
    @classmethod
    def _fp(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip().lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("fingerprint must be 64 lowercase hex characters")
        return v

    @field_validator("service_url")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip().rstrip("/")
        try:
            origin_of(v)
        except ConfigError as exc:
            raise ValueError(exc.message) from exc
        return v

    @model_validator(mode="after")
    def _masked_needs_service(self) -> GatewayModelConfig:
        if self.mode == "masked" and not self.service_url:
            raise ValueError(f"model {self.name!r}: masked mode needs service_url")
        if (self.client_cert_file is None) != (self.client_key_file is None):
            raise ValueError("client_cert_file and client_key_file must be set together")
        if self.service_token is not None and self.service_token_file is not None:
            raise ValueError("service_token and service_token_file are mutually exclusive")
        _refuse_pending_speed_settings(self, PENDING_SPEED_SETTINGS, f"model {self.name!r}: ")
        if self.spec_strict_passes and self.spec_rows is None:
            raise ValueError(f"model {self.name!r}: spec_strict_passes needs spec_rows (a fixed row count per pass)")
        if self.spec_rows is not None:
            if not self.spec_strict_passes:
                raise ValueError(f"model {self.name!r}: spec_rows needs spec_strict_passes (without it the pass count "
                                 "reveals how many drafts were accepted)")
            if self.spec_lookup <= 0 and self.draft_pack_path is None:
                raise ValueError(f"model {self.name!r}: spec_rows needs a draft source "
                                 "(spec_lookup or draft_pack_path)")
            if self.spec_tokens == 0:
                raise ValueError(f"model {self.name!r}: spec_rows needs spec_tokens > 0 (0 disables speculation)")
        if self.spec_mask_rows is not None:
            if self.spec_rows is None or not self.spec_strict_passes:
                raise ValueError(f"model {self.name!r}: spec_mask_rows needs spec_rows and spec_strict_passes")
            if self.spec_mask_rows > self.spec_rows:
                raise ValueError(f"model {self.name!r}: spec_mask_rows must be at most spec_rows")
            if self.freivalds_rate < 1.0:
                # padding rows are not range-checked (no W r): with sampled checks a provider that corrupts a row
                # would learn from the (missing) range-check abort whether that row was a real draft
                raise ValueError(f"model {self.name!r}: spec_mask_rows needs freivalds_rate = 1.0")
        if self.spec_mask_credit is not None and self.spec_mask_rows is None:
            raise ValueError(f"model {self.name!r}: spec_mask_credit needs spec_mask_rows")
        if self.speculates and self.device.split(":", 1)[0].strip().lower() != "cpu":
            # Verify passes are bit-identical to plain decoding only with the CPU float path: on a GPU the
            # float kernels' results depend on how many rows a pass carries (Kaggle T4, 2026-11-21: greedy
            # answers of a CUDA target diverged once drafts were accepted). The draft may use a GPU
            # (draft_device): it only proposes, the target decides every token.
            raise ValueError(f"model {self.name!r}: speculation (spec_lookup / draft_pack_path) needs device='cpu' "
                             "for the target (verify passes are not bit-identical on a GPU); put the draft on the "
                             "GPU with draft_device instead")
        if self.lane != "off" and self.mode != "masked":
            raise ValueError(f"model {self.name!r}: lane needs masked mode")
        return self

    @property
    def speculates(self) -> bool:
        """Speculative decoding is on (a draft source and spec_tokens > 0)."""
        return (self.spec_lookup > 0 or self.draft_pack_path is not None) and self.spec_tokens > 0

    def resolved_service_token(self) -> SecretStr | None:
        """The service bearer token: ``service_token`` or the stripped contents of ``service_token_file``.

        Raises:
            ConfigError: the token file cannot be read or is empty.
        """
        if self.service_token_file is None:
            return self.service_token
        try:
            text = Path(self.service_token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"model {self.name!r}: cannot read service_token_file ({exc.strerror or type(exc).__name__})"
            ) from exc
        if not text or any(c.isspace() for c in text):
            raise ConfigError(f"model {self.name!r}: service_token_file must contain exactly one token")
        return SecretStr(text)

    @property
    def ring_params(self) -> RingParams:
        """Parsed ring."""
        return RingParams.from_name(self.ring)


class MaskRingSettings(BaseModel):
    """In-memory mask ring (``PGW_GATEWAY_MASK_RING__*``; design ``research/plans/mask-ringbuffer-design.md``).

    Off by default. When enabled, masks from an embedded (in-process) factory are handed to the gateway
    through locked RAM instead of the disk store; the store stays as fallback/overflow tier unless
    ``ring_only``. Enabling it makes the gateway process non-dumpable and checks swap at start-up.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(False, description="Build per-model RAM rings and the tiered mask source")
    capacity_bytes: int | None = Field(
        None, gt=0, description="Locked RAM for all rings (split evenly across masked models unless per_model_bytes). "
        "Sizing: burst_s x slots/s x slot_bytes, e.g. 10 s x 100 slots/s x 5.56 MB = 5.6 GB for 7B at K32",
    )
    per_model_bytes: dict[str, int] = Field(default_factory=dict, description="{model name: bytes} override")
    segment_slots: int | None = Field(None, gt=0, description="Slots per RAM segment (default: ~8 slabs, 16..256)")
    prefer: Literal["ring", "store"] = Field("ring", description="Tier reserve() tries first")
    ring_only: bool = Field(False, description="No disk store at all (needs embed_factory)")
    ring_only_wait_s: float = Field(
        2.0, ge=0, description="ring_only: a first reservation waits this long for the factory (never extensions)"
    )
    max_session_fraction: float = Field(
        0.5, gt=0, le=1, description="ring_only: max_context + mask_window_tokens must fit in this fraction of the "
        "ring's capacity (else start-up fails)",
    )
    ring_low_fraction: float = Field(0.5, gt=0, le=1, description="Factory priority threshold")
    embed_factory: bool = Field(False, description="Run the mask factory in-process (v1)")
    factory_process: bool = Field(
        False, description="Run the mask factory as a supervised child filling shared-memory slabs (v2: one sealed "
        "memfd per slab over a private socketpair; separate process and CPU budget). Excludes embed_factory",
    )
    run_dir: Path | None = Field(None, description="Factory state/lock directory in ring_only mode "
                                 "(default <data_dir>/maskring)")
    swap_policy: Literal["refuse", "warn"] = Field(
        "refuse", description="Unencrypted disk swap at start-up: refuse to start, or only warn"
    )
    init_on_free_policy: Literal["refuse", "warn"] = Field(
        "warn", description="factory_process (v2) when the kernel does not zero freed pages (init_on_free=1): "
        "consumed sealed slabs are dropped unzeroed; refuse to start, or only warn"
    )
    allow_unlocked: bool = Field(False, description="DEV ONLY (needs allow_dev_modes): run when mlock/madvise fail")


class GatewaySettings(PGWSettings):
    """Gateway configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PGW_GATEWAY_", env_nested_delimiter="__", extra="ignore", frozen=True
    )

    host: str = "127.0.0.1"
    port: int = 8080
    models: list[GatewayModelConfig] = Field(default_factory=list)
    models_file: Path | None = Field(None, description="JSON/YAML file with a list of GatewayModelConfig")
    maskstore_root: Path = Path("/var/lib/pgw/masks")
    api_keys_file: Path | None = Field(None, description="Hashed API keys (managed with 'pgw gateway keys')")
    admin_token: SecretStr | None = None
    admin_token_file: Path | None = Field(
        None,
        description="Read admin_token from this file (a mounted Docker/Kubernetes secret; read at start-up). "
        "Mutually exclusive with admin_token",
    )
    console_viewer_tokens_file: Path | None = Field(
        None, description="Console read-only (viewer) tokens, one per line; admins sign in with admin_token",
    )
    allow_anonymous: bool = Field(False, description="Dev only (needs allow_dev_modes)")
    allow_dev_modes: bool = Field(
        False, description="Permit 'plain'/'float' models, sampled Freivalds and anonymous access. Never in production"
    )
    requests_per_minute: int = Field(600, ge=1, description="Default per-key request rate")
    tokens_per_minute: int = Field(200_000, ge=1, description="Default per-key token rate (prompt + completion)")
    daily_token_quota: int | None = Field(None, ge=1, description="Default per-key tokens per UTC day")
    quota_state_file: Path | None = Field(None, description="Persist per-key daily usage here (JSON)")
    max_prompt_tokens: int = Field(32_768, ge=1)
    default_max_tokens: int = Field(512, ge=1, description="max_tokens when the request gives none")
    max_tokens_limit: int = Field(8192, ge=1, description="Upper bound on a request's max_tokens")
    max_prompts_per_request: int = Field(16, ge=1, description="Prompts in one /v1/completions list")
    max_body_bytes: int = Field(4 << 20, ge=1024, description="Max API request body")
    tls: TLSConfig = TLSConfig()
    noise_monitor: bool = Field(True, description="Statistically check outbound payloads look uniform")
    noise_monitor_fail_closed: bool = Field(True, description="Block traffic if the monitor alarms")
    noise_monitor_required: bool | None = Field(
        None,
        description="Refuse to start masked models when the noise monitor cannot be constructed "
        "(unset: required unless allow_dev_modes)",
    )
    noise_sample_rate: float = Field(0.05, gt=0.0, le=1.0)
    egress_allowlist: list[str] = Field(
        default_factory=list,
        description="Extra allowed outbound origins (scheme://host:port). Service URLs are always allowed",
    )
    require_service_tls: bool = Field(
        True, description="Refuse plain-http service URLs except loopback (e.g. a local TLS sidecar)"
    )
    integrity_cooldown_s: float | None = Field(
        None,
        gt=0,
        description="Circuit-breaker cool-down after an integrity failure. Default None: only an admin reset "
        "re-opens traffic (SPEC 3.2: an adaptive attacker is cut off)",
    )
    integrity_max_auto_resets: int = Field(
        2, ge=0, description="With a cool-down: after this many automatic re-opens the breaker stays open"
    )
    min_soundness_bits: float = Field(
        20.0, ge=0, description="Refuse mask segments giving fewer bits of Freivalds soundness for any linear"
    )
    pack_verify: Literal["full", "fingerprint"] = Field(
        "full",
        description="Pack verification at start-up: 'full' hashes plain tensors, tokenizer and linears "
        "(scales/biases); 'fingerprint' (manifest only) needs allow_dev_modes",
    )
    audit_log_file: Path | None = Field(None, description="Append-only JSON-lines audit log (no content)")
    drain_timeout_s: float = Field(
        30.0, ge=0, description="On SIGTERM: max wait for open requests/streams before connections are closed"
    )
    ops_port: int | None = Field(
        None, ge=1, le=65535,
        description="Internal plain-HTTP listener serving only /healthz, /readyz and /metrics (for probes and "
        "scraping when the API port requires client certificates). Bind it to a pod-internal interface",
    )
    ops_host: str = Field("0.0.0.0", description="Bind address of the ops listener")  # noqa: S104
    mask_window_tokens: int = Field(
        64,
        ge=0,
        description="Reserve masks for the prompt plus this many generated tokens, then extend in windows of "
        "the same size while generating, so an early stop or disconnect burns at most one window instead of "
        "max_tokens slots. 0 = reserve prompt + max_tokens up front",
    )
    mask_burn_weight: float = Field(
        1.0,
        ge=0,
        description="Rate limits and quotas charge max(tokens, burn_weight x mask slots reserved) per request, "
        "so a key cannot drain the shared mask store with huge max_tokens and early stops",
    )
    mask_cache_budget_bytes: int = Field(
        4 << 30,
        ge=0,
        description="Start-up check: warn when max_concurrent_sessions x the decrypted mask chunks a session "
        "caches exceeds this (0 = no check)",
    )
    maskstore_gc_interval_s: float = Field(
        600.0, ge=0, description="Run mask store GC (dead leases, consumed segments, old audit rows); 0 = never"
    )
    quota_flush_interval_s: float = Field(
        5.0, gt=0, description="Persist quota usage at most this often (off the event loop) and on shutdown"
    )
    connect_retry_max_s: float = Field(
        60.0, gt=0, description="Max back-off between compute-service connection attempts of a model not yet ready"
    )
    metrics_require_admin: bool = Field(False, description="Protect /metrics with the admin token")
    cors_origins: list[str] = Field(default_factory=list)
    dashboard_enabled: bool = True
    console_enabled: bool = Field(
        True, description="Serve the operator console at /console/ (sign-in with admin_token or a viewer token)"
    )
    console_token_login_enabled: bool = Field(
        True, description="Allow console sign-in with the admin or a viewer token. Set false when SSO is mandatory "
        "(needs console OIDC; the admin token keeps working as a Bearer token for /admin/* scripts)",
    )
    # ---- first-run setup (pgw gateway enroll, console setup wizard) ---------------------------------------
    setup_mode: bool = Field(
        False, description="Start the first-run setup wizard (/console/) instead of serving models. Also used "
        "automatically when no model is configured or a wizard run under config_dir is unfinished",
    )
    config_dir: Path = Field(
        Path("/etc/pgw"), description="Configuration directory the enrollment and setup wizard write to "
        "(credentials, models.json, gateway.env, setup state)",
    )
    data_dir: Path = Field(
        Path("/var/lib/pgw"), description="Data directory the setup wizard uses for model packs, the mask "
        "store and gateway state",
    )
    setup_model_options: dict[str, Any] = Field(
        default_factory=dict, description="Speed settings the setup wizard writes into the model's models.json "
        "entry, e.g. {\"local_linears\": \"auto\", \"float_threads\": 1} (PGW_GATEWAY_SETUP_MODEL_OPTIONS as "
        "JSON). Only the speed settings in SETUP_MODEL_OPTION_KEYS are accepted (never a security setting such "
        "as freivalds_rate); the entry is validated like any models.json entry",
    )
    factory_env_file: Path | None = Field(
        None, description="Run the encrypted mask factory as a supervised child process of the gateway with the "
        "PGW_FACTORY_* / PGW_MASKSTORE_* settings in this file (written by the setup wizard)",
    )
    mask_ring: MaskRingSettings = Field(
        default_factory=MaskRingSettings, description="In-memory mask ring (off by default; PGW_GATEWAY_MASK_RING__*)",
    )
    # ---- console single sign-on (OpenID Connect, authorization code flow + PKCE; off unless issuer + client id) --
    console_oidc_issuer: str | None = Field(
        None, description="OIDC issuer URL (https), exactly as the provider's discovery document states it",
    )
    console_oidc_client_id: str | None = Field(None, description="Client id registered at the OIDC provider")
    console_oidc_client_secret_file: Path | None = Field(
        None, description="File holding the OIDC client secret (confidential client; never set the secret inline)",
    )
    console_oidc_redirect_url: str | None = Field(
        None, description="Redirect URI registered at the provider, e.g. "
        "https://gw.example.com/console/api/oidc/callback (path must end with /console/api/oidc/callback)",
    )
    console_oidc_scopes: str = Field("openid email profile", description="Space-separated scopes (must include openid)")
    console_oidc_role_claim: str = Field(
        "groups", description="ID-token claim holding the user's groups or roles (dotted path for nested claims, "
        "e.g. realm_access.roles)",
    )
    console_oidc_admin_values: list[str] = Field(
        default_factory=list, description="Values of the role claim that grant the admin role (JSON list)",
    )
    console_oidc_viewer_values: list[str] = Field(
        default_factory=list, description="Values of the role claim that grant the read-only viewer role (JSON "
        "list); users with neither an admin nor a viewer value are refused (403)",
    )
    console_oidc_allowed_domains: list[str] = Field(
        default_factory=list, description="Optional allowlist of email domains (JSON list); empty = any domain",
    )
    console_oidc_ca_file: Path | None = Field(
        None, description="CA bundle (PEM) to verify the OIDC provider's TLS certificate (default: system/certifi)",
    )
    console_oidc_label: str = Field(
        "SSO", min_length=1, max_length=40, description="Provider name on the sign-in button ('Sign in with <label>')",
    )

    @model_validator(mode="before")
    @classmethod
    def _admin_token_from_file(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("admin_token_file") in (None, ""):
            return data
        if data.get("admin_token") not in (None, ""):
            raise ValueError("admin_token and admin_token_file are mutually exclusive")
        try:
            text = Path(data["admin_token_file"]).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read admin_token_file ({exc.strerror or type(exc).__name__})") from exc
        if not text or any(c.isspace() for c in text):
            raise ValueError("admin_token_file must contain exactly one token")
        return {**data, "admin_token": text}

    @field_validator("setup_model_options")
    @classmethod
    def _setup_model_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        bad = sorted(set(v) - SETUP_MODEL_OPTION_KEYS)
        if bad:
            raise ValueError(f"setup_model_options: {', '.join(bad)} cannot be preset (allowed: "
                             f"{', '.join(sorted(SETUP_MODEL_OPTION_KEYS))})")
        drafts = [k for k in _PRESET_DRAFT_SOURCES if v.get(k) not in (None, 0, "", False)]
        if drafts and (v.get("spec_rows") is None or v.get("spec_strict_passes") is not True):
            raise ValueError(f"setup_model_options: {', '.join(drafts)} needs spec_rows and spec_strict_passes=true "
                             "(presets enable strict fixed-row speculation only)")
        return v

    @field_validator("admin_token")
    @classmethod
    def _admin_token(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and len(v.get_secret_value().strip()) < _MIN_ADMIN_TOKEN_LEN:
            raise ValueError(f"admin_token must have at least {_MIN_ADMIN_TOKEN_LEN} characters")
        return v

    @model_validator(mode="after")
    def _policy(self) -> GatewaySettings:
        for o in self.egress_allowlist:
            try:
                origin_of(o)
            except ConfigError as exc:
                raise ValueError(exc.message) from exc
        self._check_console_oidc()
        self._check_mask_ring()
        return self

    def _check_mask_ring(self) -> None:
        mr = self.mask_ring
        if not mr.enabled:
            return
        if mr.capacity_bytes is None and not mr.per_model_bytes:
            raise ValueError("mask_ring.enabled needs mask_ring.capacity_bytes (or per_model_bytes)")
        if mr.embed_factory and self.factory_env_file is not None:
            raise ValueError("mask_ring.embed_factory and factory_env_file are mutually exclusive "
                             "(one GPU must never run two factories)")
        if mr.embed_factory and mr.factory_process:
            raise ValueError("mask_ring.embed_factory and mask_ring.factory_process are mutually exclusive")
        if mr.factory_process and self.factory_env_file is not None:
            raise ValueError("mask_ring.factory_process and factory_env_file are mutually exclusive "
                             "(one GPU must never run two factories)")
        if mr.ring_only and not (mr.embed_factory or mr.factory_process):
            raise ValueError("mask_ring.ring_only needs embed_factory or factory_process (nothing else fills the ring)")
        if mr.allow_unlocked and not self.allow_dev_modes:
            raise ValueError("mask_ring.allow_unlocked is a dev-only setting (needs allow_dev_modes)")

    @property
    def console_oidc_enabled(self) -> bool:
        """Console single sign-on is configured (issuer and client id set)."""
        return bool(self.console_oidc_issuer) and bool(self.console_oidc_client_id)

    def _check_console_oidc(self) -> None:
        if bool(self.console_oidc_issuer) != bool(self.console_oidc_client_id):
            raise ValueError("console_oidc_issuer and console_oidc_client_id must be set together")
        if not self.console_oidc_enabled:
            if not self.console_token_login_enabled and self.console_enabled:
                raise ValueError("console_token_login_enabled=false needs console SSO (console_oidc_issuer and "
                                 "console_oidc_client_id); otherwise nobody can sign in to the console")
            return
        assert self.console_oidc_issuer is not None
        for name, url in (("console_oidc_issuer", self.console_oidc_issuer),
                          ("console_oidc_redirect_url", self.console_oidc_redirect_url)):
            if not url:
                raise ValueError(f"{name} is required for console SSO")
            try:
                scheme, host, _ = origin_of(url)
            except ConfigError as exc:
                raise ValueError(f"{name}: {exc.message}") from exc
            if scheme != "https" and not _is_loopback(host):
                raise ValueError(f"{name} must use https (plain http only to loopback, for tests)")
            if urlsplit(url).query or urlsplit(url).fragment:
                raise ValueError(f"{name} must not have a query or fragment")
        assert self.console_oidc_redirect_url is not None
        if not urlsplit(self.console_oidc_redirect_url).path.endswith("/console/api/oidc/callback"):
            raise ValueError("console_oidc_redirect_url must end with /console/api/oidc/callback")
        if self.console_oidc_client_secret_file is None:
            raise ValueError("console SSO needs console_oidc_client_secret_file (confidential client)")
        if "openid" not in self.console_oidc_scopes.split():
            raise ValueError("console_oidc_scopes must include 'openid'")
        if not self.console_oidc_role_claim.strip():
            raise ValueError("console_oidc_role_claim must not be empty")
        if not [v for v in (*self.console_oidc_admin_values, *self.console_oidc_viewer_values) if v.strip()]:
            raise ValueError("console SSO needs console_oidc_admin_values and/or console_oidc_viewer_values "
                             "(users without a mapped role are refused)")
        for d in self.console_oidc_allowed_domains:
            if not d.strip().lstrip("@") or any(c.isspace() or c in "/@" for c in d.strip().lstrip("@")):
                raise ValueError(f"console_oidc_allowed_domains: invalid domain {d!r}")

    def all_models(self) -> list[GatewayModelConfig]:
        """Models from ``models`` plus ``models_file`` (names must be unique)."""
        out = list(self.models)
        if self.models_file is not None:
            out += load_models_file(self.models_file)
        names = [m.name for m in out]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ConfigError(f"duplicate model names: {sorted(dup)}")
        return out

    def validate_policy(self, models: list[GatewayModelConfig] | None = None) -> None:
        """Enforce the production policy (fail closed). Raises :class:`ConfigError`."""
        models = self.all_models() if models is None else models
        if self.allow_anonymous and not self.allow_dev_modes:
            raise ConfigError("allow_anonymous requires allow_dev_modes (never in production)")
        for m in models:
            if m.mode in DEV_MODES and not self.allow_dev_modes:
                raise ConfigError(
                    f"model {m.name!r}: mode {m.mode!r} is for development only (set allow_dev_modes)",
                    details={"model": m.name, "mode": m.mode},
                )
            if m.mode == "masked":
                if m.fingerprint is None and not self.allow_dev_modes:
                    raise ConfigError(
                        f"model {m.name!r}: pin the pack fingerprint (fingerprint: <64 hex from 'pgw inspect'); "
                        "an unpinned pack could be swapped together with its manifest",
                        details={"model": m.name},
                    )
                if m.freivalds_rate < 1.0 and not self.allow_dev_modes:
                    raise ConfigError(
                        f"model {m.name!r}: freivalds_rate must be 1.0 in production (every response is checked)"
                    )
                assert m.service_url is not None
                scheme, host, _ = origin_of(m.service_url)
                if scheme == "http" and not _is_loopback(host):
                    if self.require_service_tls:
                        raise ConfigError(
                            f"model {m.name!r}: service_url must use https (plain http only to loopback)",
                            details={"model": m.name},
                        )
                    log.warning(
                        "model %s: plain-http compute service on a non-loopback host (require_service_tls=false); "
                        "only safe when the network layer (service mesh mTLS) encrypts and authenticates it",
                        m.name,
                    )
        if self.pack_verify != "full" and not self.allow_dev_modes:
            raise ConfigError("pack_verify must be 'full' in production (set allow_dev_modes for 'fingerprint')")
        if self.noise_monitor_required is False and not self.allow_dev_modes:
            raise ConfigError("noise_monitor_required=false needs allow_dev_modes")
        if self.allow_dev_modes:
            log.warning("gateway running with allow_dev_modes=true: NOT a production configuration")

    @property
    def noise_monitor_is_required(self) -> bool:
        """Effective ``noise_monitor_required`` (default: required outside dev mode)."""
        if self.noise_monitor_required is None:
            return not self.allow_dev_modes
        return self.noise_monitor_required

    def egress_origins(self, models: list[GatewayModelConfig] | None = None) -> frozenset[tuple[str, str, int]]:
        """Allowed outbound origins: configured service URLs plus ``egress_allowlist``."""
        models = self.all_models() if models is None else models
        out = {origin_of(m.service_url) for m in models if m.service_url}
        out |= {origin_of(o) for o in self.egress_allowlist}
        return frozenset(out)


def load_models_file(path: Path) -> list[GatewayModelConfig]:
    """Parse a JSON or YAML list of :class:`GatewayModelConfig` (or ``{"models": [...]}``).

    Relative ``pack_path`` values are resolved against the file's directory.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"models file {p} does not exist")
    text = p.read_text(encoding="utf-8")
    data: Any
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - dependency
                raise ConfigError("PyYAML is required for YAML model files") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except (ValueError, OSError) as exc:
        raise ConfigError(f"cannot parse models file {p.name}: {type(exc).__name__}") from exc
    if isinstance(data, dict):
        data = data.get("models")
    if not isinstance(data, list):
        raise ConfigError(f"models file {p.name} must contain a list of models")
    out: list[GatewayModelConfig] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ConfigError(f"models file entry {i} must be an object")
        e = dict(entry)
        pp = e.get("pack_path")
        if isinstance(pp, str) and not Path(pp).is_absolute():
            e["pack_path"] = str((p.parent / pp).resolve())
        try:
            out.append(GatewayModelConfig(**e))
        except ValueError as exc:
            raise ConfigError(f"invalid model entry {i} in {p.name}: {exc}") from exc
    return out
