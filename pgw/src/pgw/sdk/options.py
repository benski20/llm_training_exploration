"""Options of the client library and the models it supports.

Every option has a default that needs no configuration. The defaults for a remote GPU provider are the
measured, exact stack of the 7B 28 tok/s run (STATUS.md, deploy/modal/README.md): every linear on the provider
(``local_linears = "off"``), native float ops, the native per-trip and decode drivers, the masked LM head
(when the provider serves it), the prime-field (p31) check, incremental mask reservations, exact conversation
prefix reuse, and the pks-v3 encrypted mask factory in low-priority background processes. Speculation (strict,
exact and draft-independent on the wire, but it multiplies the mask demand) is opt-in (``speculation``); modes that
are not proven exact and safe stay off.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from pgw.sdk.errors import UnsupportedModelError

__all__ = ["PREVIEW_ENV", "SUPPORTED_MODELS", "SDKOptions", "canonical_model", "default_mask_workers",
           "preview_models"]

#: Models of this release (provider names). Anything else is refused with a clear message.
SUPPORTED_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
)

#: Measurement-only PREVIEW of models that this release pins (``pgw.sdk.provisioning.PINNED_MODELS``, with a baked
#: fingerprint) but does not support: a comma list of model names, off (empty) by default. It is not a support
#: statement (sizing, defaults and docs are untested for these models); it exists so that the pinned 14B / 32B / 72B
#: models can be measured end to end. It relaxes no check: the provider's pack is verified against the baked pin
#: exactly as for a supported model, ``model_fingerprints`` cannot replace a preview model's pin, and a listed name
#: without a baked pin is refused.
PREVIEW_ENV = "PGW_SDK_PREVIEW_MODELS"


def _model_key(name: str) -> str:
    return re.sub(r"^(qwen/)", "", name.strip().lower())


def preview_models(raw: str | None = None) -> tuple[str, ...]:
    """The preview models :data:`PREVIEW_ENV` (or ``raw``) enables: listed, pinned with a baked fingerprint and not
    already supported.

    Raises:
        UnsupportedModelError: a listed name has no baked pin.
    """
    from pgw.sdk.provisioning import PINNED_MODELS

    text = os.environ.get(PREVIEW_ENV, "") if raw is None else raw
    out: list[str] = []
    for item in text.split(","):
        n = _model_key(item)
        if not n or n in SUPPORTED_MODELS or n in out:
            continue
        pin = PINNED_MODELS.get(n)
        if pin is None or not pin.fingerprint:
            pinned = ", ".join(m for m, p in PINNED_MODELS.items() if p.fingerprint and m not in SUPPORTED_MODELS)
            raise UnsupportedModelError(
                f"{PREVIEW_ENV}: {item.strip()!r} has no baked pin in this release and cannot be previewed "
                f"(pinned preview candidates: {pinned or 'none'})", details={"model": item.strip()})
        out.append(n)
    return tuple(out)


def canonical_model(name: str) -> str:
    """``"Qwen/Qwen2.5-7B-Instruct"`` / ``"qwen2.5-7b-instruct"`` -> ``"qwen2.5-7b-instruct"``.

    A pinned model outside :data:`SUPPORTED_MODELS` is accepted only while :data:`PREVIEW_ENV` lists it
    (measurement-only preview, off by default: :func:`preview_models`).

    Raises:
        UnsupportedModelError: not one of :data:`SUPPORTED_MODELS` nor an enabled preview model.
    """
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedModelError("model is required (supported: " + ", ".join(SUPPORTED_MODELS) + ")")
    n = _model_key(name)
    if n in SUPPORTED_MODELS:
        return n
    if os.environ.get(PREVIEW_ENV, "").strip() and n in preview_models():
        return n
    raise UnsupportedModelError(
        f"model {name!r} is not supported by this release of pgw (supported: {', '.join(SUPPORTED_MODELS)})",
        details={"model": name})


def default_mask_workers(cpus: int | None = None) -> int:
    """Background mask workers: a quarter of the CPUs, at least 1, at most 8 (each at low priority)."""
    n = cpus if cpus is not None else (os.cpu_count() or 2)
    return max(1, min(8, n // 4))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class SDKOptions:
    """Client options (all optional). Environment variables in brackets.

    Attributes:
        cache_dir: Cache root [``PGW_CACHE_DIR``] (default ``~/.cache/pgw``).
        state_dir: State root for the mask store's KEK and rollback counter [``PGW_STATE_DIR``].
        timeout_s: Budget of one request, queueing and mask wait included.
        max_retries: Retries of a request after a transport failure, only before any token was returned.
        ready_timeout_s: How long a first request may wait for enrollment, provisioning and connection.
        mask_scheme: ``"pks-v3"`` (default; the compact encrypted scheme, pending external review, used with the
            library's explicit acknowledgement), ``"mlwe"`` (the reviewed encrypted scheme; large hint) or
            ``"plain"`` (masks computed locally from the int8 weights; no provider involvement).
        mask_workers: Factory worker processes per model [``PGW_MASK_WORKERS``] (default: CPUs / 4, 1..8).
        mask_threads: Torch threads per factory worker (default 1).
        mask_nice: Niceness increment of the factory processes (default 10: low priority).
        mask_store_gb: Disk budget of the mask stock per model; sets the factory target [``PGW_MASK_STORE_GB``].
        mask_target_slots / mask_low_water_slots: Explicit stock target / refill threshold (slots).
        mask_segment_slots: Slots per factory segment (default 1024, capped by the stock target; small: masks become
            usable sooner; pks-v3: rounded down to whole 256-slot groups, and four groups per segment cost about
            25 % less client CPU per mask slot than one).
        mask_wait_s: How long a request waits for masks when the stock is empty before failing (back-pressure).
            During a model's first fill on a host (the factory still building its one-time setup and making
            progress) a request without an explicit ``timeout`` waits up to ``ready_timeout_s`` instead. While the
            factory is visibly making the segment a request waits for and expects it within the request's
            ``timeout_s``, the request waits for it (never beyond ``timeout_s`` or its deadline).
        factory_stall_s: [``PGW_FACTORY_STALL_S``] How long the background mask factory may make no progress
            because the provider cannot be reached (connection refused or reset, TLS failure, 5xx, timeouts; the
            library also checks the provider's ``/readyz``) or because its runs keep failing, before requests
            waiting for masks fail at once with ``APIConnectionError`` / ``NotReadyError`` naming the reason
            (default 120 s). The factory keeps retrying with back-off and resumes by itself when the provider is
            back. A long first-time setup that makes progress is never cut short by this.
        maskgen_stall_s: [``PGW_SDK_MASKGEN_STALL_S``] The maskgen watchdog (default 30 s; 0 turns it off): while
            the factory makes a segment it reads this key's own maskgen progress from the provider
            (``GET /pgw/v1/maskgen/status``); when one of its items has been running there this long without
            finishing (at least 4x the slowest recent item, and the provider's warm-up bound, 300 s, for the first
            item of a shape), the provider's mask generation counts as stalled (a wedged GPU) and requests waiting
            for masks fail at once with ``ProviderStalledError`` (an ``APIConnectionError``, retryable). Providers
            without the endpoint: off. The client's own work (encryption, the hint build) never counts.
        mask_admission: [``PGW_MASK_ADMISSION``] ``"entitled"`` (default): a request starts only when the mask
            stock covers the whole remaining prompt + ``max_tokens`` of every running request (a running answer
            never waits for masks; N concurrent requests need a stock of about N x (prompt + max_tokens) slots).
            ``"overcommit"`` (opt-in): a request starts when the stock covers its prompt and one chunk for every
            running request; an answer that finds
            the stock short waits for the factory (up to 30 s, then fails closed), so a burst slows every answer to
            the factory's rate instead of queueing new requests. Single use of masks is the same in both.
        factory: Run the background mask factory (``False``: consume existing stock only).
        lane: ``"auto"`` (the provider's TCP+TLS trip lane when offered, else the blocking WebSocket stream lane),
            ``"stream"`` or ``"off"`` (the async stream channel only).
        native: The native trip / decode drivers (the measured stack; exact). ``False``: the Python trip path.
            Process-wide (environment switches read when a model starts): the first client's choice applies.
        masked_head: ``"auto"`` (the int8 LM head as a masked linear when the provider serves it), ``"on"``
            (required), ``"off"``.
        prefix_cache: Exact KV prefix reuse (``research/reviews/prefix-cache.md``): the plaintext KV cache of
            finished requests stays in this process's RAM, and a later request whose prompt starts with the same
            tokens (a conversation's earlier turns, a shared system prompt) runs only the rest of its prompt: no
            round-trip rows and no masks for the cached tokens; answers and logprobs bit-identical. ``"auto"``
            (default): on only when the masked LM head is not used (as before); ``True``: on, with or without the
            masked head (opt-in: the provider then sees a shorter prompt pass for a request that hits the cache,
            i.e. *that* it continued an earlier prompt); ``False``: off.
        prefix_cache_mb: Memory budget of the stored KV caches per model and process (MiB, default 1024; least
            recently used entries are wiped first). Live requests' caches are not counted.
        prefix_cache_ttl_s: A stored KV cache unused this long is wiped (default 600 s).
        prefix_cache_scope: Which requests may reuse each other's prefixes: ``"user"`` (default: the same OpenAI
            ``user``, so one application user's timing cannot reveal another's prompts) or ``"client"`` (every
            request of this client: a system prompt shared by all users is computed once).
        hint: How the pks-v3 hint is obtained on first use of a model [``PGW_HINT``]: ``"local"`` (default; the
            background factory builds it from the fingerprint-checked public weights: one-time, about 20-30 min of
            CPU at 7B on 4 cores, then cached) or ``"provider"`` (opt-in; the provider computes the expensive part
            and the client verifies it with a >= 40-bit check; streams about 44 GB at 7B).
        factory_weights: Where the background factory's check material comes from [``PGW_FACTORY_WEIGHTS``].
            ``"pack"`` (default): the int8 weights (the factory pack: 6.5 / 31 / 70 GB at 7B / 32B / 72B) stay in the
            cache. ``"stream"`` (opt-in; ``mask_scheme="pks-v3"``): no int8 weights at rest. Each refill pass streams
            them from ``factory_weights_source``, checks every tensor against the pinned fingerprint, pre-draws the
            check material of the next ``check_stock_segments`` segments (plus private self-check sets) into a
            KEK-encrypted single-use stock and drops them: about 1 % more network than pks-v3 itself at 64
            segments per pass, a stock of 0.6 / 1.9 / 3.1 GB, and the factory pauses (requests wait, then fail
            closed) when the stock is empty and no source is reachable (``docs/sdk/scaling.md``).
        factory_weights_source: ``"provider"`` (default: the provider's provisioning endpoint) or ``"hf"`` (the
            pinned public checkpoint, packed per pass: needs its size in temporary disk)
            [``PGW_FACTORY_WEIGHTS_SOURCE``].
            Both are untrusted and checked in full.
        check_stock_segments: Segments of check material a pass pre-draws with ``factory_weights="stream"``
            (default 64; the next pass starts at a quarter) [``PGW_CHECK_STOCK_SEGMENTS``].
        model_fingerprints: Extra or replacement pack fingerprint pins ``{model: fingerprint}``. The library pins
            every supported model (``pgw.sdk.provisioning.PINNED_MODELS``) and refuses a provider serving another
            pack; set this only for a custom pack you have checked yourself.
        usage_reports: Send the provider a content-free record of each finished request (model, token counts,
            latency, status, whether every integrity check passed) for the developer console's request log. The
            provider already sees the same counts and timing in the masked traffic; never any content.
        max_concurrent: Requests generating at once per model.
        speculation: Strict speculative decoding [``PGW_SPECULATION``] (opt-in; ``"off"`` by default). Every decode
            pass verifies a fixed ``spec_rows`` rows (the pending token, the drafts, masked filler rows), so one pass
            of ``4 x layers + 1`` round trips yields 1 to ``spec_rows`` tokens and the answer streams that many
            times faster on the round-trip-bound path. Bit-identical to plain decoding (the verifier keeps only
            tokens the target model itself samples). The provider's view stays independent of the drafts: fixed
            rows per pass, and after the answer the client runs dummy passes until the provider has seen exactly
            one pass per generated token (as without speculation); the next request of the same ``user`` waits
            for them. Costs: ``spec_rows`` mask slots per pass (the factory must make ``spec_rows`` times the
            masks) and ``spec_rows`` rows of provider work per trip. Drafts: ``"lookup"`` (prompt lookup: no
            weights, no model; strong when the answer copies the prompt or the conversation: code edits, RAG,
            summaries, rewrites; about 1.0-1.1 tokens per pass on open chat) or ``"draft_head"`` (a trained
            target-feature draft head: a few MB, one small CPU step per draft row, on this machine only; it
            drafts open chat too. The library ships a head per supported model, pinned by SHA-256
            (``pgw.sdk.draft_heads``); ``draft_head_path`` replaces it with your own file).
        spec_rows: Rows per verify pass with ``speculation`` (2-8; default 3: up to 2 drafts per pass).
        draft_head_path: Your own draft head file for ``speculation="draft_head"`` (default: the pinned head
            shipped for the model).
        spec_pass_gap_ms: With ``speculation``: the least time from the end of one pass to the start of the next
            (default 8 ms; 0 = off). The client work between passes (sampling, drafting, handing tokens to the
            application, and the application waking up when the answer is complete) differs a little between the
            answer's passes and the dummy passes after it; this floor hides it from the provider's view of pass
            timing (security review 2026-11, S20). It adds the floor minus the usual gap (a few ms) per pass.
        schedule: The masked linear schedule [``PGW_SCHEDULE``] (:mod:`pgw.sdk.schedule`): ``"standard"``
            (default: ``4 x layers + 1`` round trips per token, K32), ``"fused"`` (opt-in: ``2 x layers`` round
            trips per token, the LM head folded into the last trip, K56 values: 1.75x the bytes per element; for
            high-RTT links such as consumer devices on the internet) or ``"auto"`` (measure the link when the model
            starts and pick ``fused`` when it is served, the RTT is at least ``schedule_auto_rtt_ms`` and its
            estimated time per token is lower, bandwidth included). ``fused`` is a **different pinned reference**
            (its tokens and logprobs are not bit-identical to ``standard``), so the choice is fixed per model for
            the life of the client and never changes during an answer or a conversation; the decision and its
            measurements are in ``status()``. Pin ``"standard"`` or ``"fused"`` for one reference across restarts.
        schedule_auto_rtt_ms: ``auto``: the least measured RTT (ms) at which ``fused`` may be picked (default 3).
        schedule_auto_margin: ``auto``: ``fused`` must be estimated at least this fraction faster (default 0.1).
        routing: Region routing [``PGW_ROUTING``] (:mod:`pgw.sdk.directory`, docs/sdk/wan.md): ``"off"`` (default:
            the SDK key's provider URL only) or ``"signed"`` (opt-in: fetch the provider's signed endpoint directory,
            verify it with the pinned key, signature, expiry and version; probe the endpoints with payload-free TCP
            handshakes and use the nearest healthy one serving the model's fingerprint; move a model only between
            requests, never during an answer; fail over on a connection failure before the first token). Without a
            pinned directory key or a valid directory it uses the provider URL only. The mask stock is shared across
            endpoints of the same fingerprint (single use is the client's own ledger).
        routing_refresh_s: ``signed``: seconds between directory refreshes and probe rounds (default 300; a failure
            refreshes at once).
        log_level: Enable the library's structured (JSON) logs on stderr at this level [``PGW_SDK_LOG``]; ``None``
            = silent (records still reach handlers the application configures).
    """

    cache_dir: Path | None = None
    state_dir: Path | None = None
    timeout_s: float = 600.0
    max_retries: int = 2
    ready_timeout_s: float = 1800.0
    mask_scheme: Literal["pks-v3", "mlwe", "plain"] = "pks-v3"
    mask_workers: int | None = None
    mask_threads: int = 1
    mask_nice: int = 10
    mask_store_gb: float = 8.0
    mask_target_slots: int | None = None
    mask_low_water_slots: int | None = None
    mask_segment_slots: int = 1024
    mask_wait_s: float = 120.0
    factory_stall_s: float = 120.0
    maskgen_stall_s: float = 30.0
    mask_admission: Literal["entitled", "overcommit"] = "entitled"
    factory: bool = True
    lane: Literal["auto", "stream", "off"] = "auto"
    native: bool = True
    masked_head: Literal["auto", "on", "off"] = "auto"
    prefix_cache: bool | Literal["auto"] = "auto"
    prefix_cache_mb: float = 1024.0
    prefix_cache_ttl_s: float = 600.0
    prefix_cache_scope: Literal["user", "client"] = "user"
    hint: Literal["local", "provider"] = "local"
    factory_weights: Literal["pack", "stream"] = "pack"
    factory_weights_source: Literal["provider", "hf"] = "provider"
    check_stock_segments: int = 64
    model_fingerprints: dict[str, str] = field(default_factory=dict)
    usage_reports: bool = True
    max_concurrent: int = 8
    speculation: Literal["off", "lookup", "draft_head"] = "off"
    spec_rows: int = 3
    draft_head_path: Path | None = None
    spec_pass_gap_ms: float = 8.0
    log_level: str | None = None
    schedule: Literal["standard", "fused", "auto"] = "standard"
    schedule_auto_rtt_ms: float = 3.0
    schedule_auto_margin: float = 0.1
    routing: Literal["off", "signed"] = "off"
    routing_refresh_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> SDKOptions:
        """Defaults, then ``PGW_*`` environment variables, then ``overrides`` (unknown names raise TypeError)."""
        known = {f.name for f in fields(cls)}
        bad = sorted(set(overrides) - known)
        if bad:
            raise TypeError(f"unknown pgw client option(s): {', '.join(bad)}")
        base: dict[str, Any] = {}
        scheme = os.environ.get("PGW_MASK_SCHEME", "").strip()
        if scheme:
            base["mask_scheme"] = scheme
        w = _env_int("PGW_MASK_WORKERS", None)
        if w is not None:
            base["mask_workers"] = w
        base["mask_store_gb"] = _env_float("PGW_MASK_STORE_GB", cls.mask_store_gb)
        hint = os.environ.get("PGW_HINT", "").strip()
        if hint:
            base["hint"] = hint
        fw = os.environ.get("PGW_FACTORY_WEIGHTS", "").strip()
        if fw:
            base["factory_weights"] = fw
        fws = os.environ.get("PGW_FACTORY_WEIGHTS_SOURCE", "").strip()
        if fws:
            base["factory_weights_source"] = fws
        cs = _env_int("PGW_CHECK_STOCK_SEGMENTS", None)
        if cs is not None:
            base["check_stock_segments"] = cs
        stall = os.environ.get("PGW_FACTORY_STALL_S", "").strip()
        if stall:
            base["factory_stall_s"] = float(stall)
        mstall = os.environ.get("PGW_SDK_MASKGEN_STALL_S", "").strip()
        if mstall:
            base["maskgen_stall_s"] = float(mstall)
        adm = os.environ.get("PGW_MASK_ADMISSION", "").strip()
        if adm:
            base["mask_admission"] = adm
        spec = os.environ.get("PGW_SPECULATION", "").strip()
        if spec:
            base["speculation"] = spec
        sch = os.environ.get("PGW_SCHEDULE", "").strip()
        if sch:
            base["schedule"] = sch
        rt = os.environ.get("PGW_ROUTING", "").strip()
        if rt:
            base["routing"] = rt
        lvl = os.environ.get("PGW_SDK_LOG", "").strip()
        if lvl:
            base["log_level"] = lvl
        base.update({k: v for k, v in overrides.items() if v is not None})
        opts = cls(**base)
        opts.validate()
        return opts

    def validate(self) -> None:
        """Raise ``ValueError`` on an invalid combination."""
        if self.mask_admission not in ("entitled", "overcommit"):
            raise ValueError("mask_admission must be 'entitled' or 'overcommit'")
        if self.mask_scheme not in ("pks-v3", "mlwe", "plain"):
            raise ValueError("mask_scheme must be 'pks-v3', 'mlwe' or 'plain'")
        if self.lane not in ("auto", "stream", "off"):
            raise ValueError("lane must be 'auto', 'stream' or 'off'")
        if self.prefix_cache not in (True, False, "auto"):
            raise ValueError("prefix_cache must be True, False or 'auto'")
        if not 1.0 <= float(self.prefix_cache_mb) <= float(1 << 20):
            raise ValueError("prefix_cache_mb must be between 1 and 1048576")
        if not 1.0 <= float(self.prefix_cache_ttl_s) <= 7 * 86400.0:
            raise ValueError("prefix_cache_ttl_s must be between 1 and 604800 seconds")
        if self.prefix_cache_scope not in ("user", "client"):
            raise ValueError("prefix_cache_scope must be 'user' or 'client'")
        if self.hint not in ("local", "provider"):
            raise ValueError("hint must be 'local' or 'provider'")
        if self.factory_weights not in ("pack", "stream"):
            raise ValueError("factory_weights must be 'pack' or 'stream'")
        if self.factory_weights_source not in ("provider", "hf"):
            raise ValueError("factory_weights_source must be 'provider' or 'hf'")
        if self.factory_weights == "stream" and self.mask_scheme != "pks-v3":
            raise ValueError("factory_weights='stream' needs mask_scheme='pks-v3' (the masks themselves need the "
                             "int8 weights otherwise)")
        if not 2 <= int(self.check_stock_segments) <= 4096:
            raise ValueError("check_stock_segments must be between 2 and 4096")
        preview = preview_models() if os.environ.get(PREVIEW_ENV, "").strip() else ()
        for m, fp in (self.model_fingerprints or {}).items():
            if not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{64}", fp.lower()):
                raise ValueError(f"model_fingerprints[{m!r}] must be 64 hex digits")
            if _model_key(str(m)) in preview:
                raise ValueError(f"model_fingerprints[{m!r}]: a preview model ({PREVIEW_ENV}) is verified against "
                                 "its baked pin only")
        if self.masked_head not in ("auto", "on", "off"):
            raise ValueError("masked_head must be 'auto', 'on' or 'off'")
        if self.mask_workers is not None and not 1 <= self.mask_workers <= 64:
            raise ValueError("mask_workers must be between 1 and 64")
        if not 0 <= self.mask_nice <= 19:
            raise ValueError("mask_nice must be between 0 and 19")
        if self.speculation not in ("off", "lookup", "draft_head"):
            raise ValueError("speculation must be 'off', 'lookup' or 'draft_head'")
        if not 2 <= int(self.spec_rows) <= 8:
            raise ValueError("spec_rows must be between 2 and 8")
        if self.draft_head_path is not None and self.speculation != "draft_head":
            raise ValueError("draft_head_path is used only with speculation='draft_head'")
        if not 0.0 <= float(self.spec_pass_gap_ms) <= 1000.0:
            raise ValueError("spec_pass_gap_ms must be between 0 and 1000")
        if not 1.0 <= float(self.factory_stall_s) <= 86400.0:
            raise ValueError("factory_stall_s must be between 1 and 86400 seconds")
        if not (float(self.maskgen_stall_s) == 0.0 or 1.0 <= float(self.maskgen_stall_s) <= 86400.0):
            raise ValueError("maskgen_stall_s must be 0 (off) or between 1 and 86400 seconds")
        if self.schedule not in ("standard", "fused", "auto"):
            raise ValueError("schedule must be 'standard', 'fused' or 'auto'")
        if self.schedule != "standard":
            if self.prefix_cache is True:
                raise ValueError("prefix_cache=True is not combined with schedule='fused' / 'auto' yet")
            if self.factory_weights == "stream":
                raise ValueError("schedule='fused' / 'auto' needs factory_weights='pack' (the K56 masks need W8)")
            if self.lane != "auto":
                raise ValueError("schedule='fused' / 'auto' needs lane='auto' (its pair items ride the trip lane)")
            if self.masked_head == "off":
                raise ValueError("schedule='fused' / 'auto' folds the LM head into the last trip on the provider: "
                                 "masked_head='off' contradicts it")
            if not self.native:
                raise ValueError("schedule='fused' / 'auto' runs on the native float ops (native=True)")
        if not 0.0 <= float(self.schedule_auto_rtt_ms) <= 10000.0 or not 0.0 <= float(self.schedule_auto_margin) < 1:
            raise ValueError("schedule_auto_rtt_ms must be in [0, 10000] and schedule_auto_margin in [0, 1)")
        if self.routing not in ("off", "signed"):
            raise ValueError("routing must be 'off' or 'signed'")
        if not 10.0 <= float(self.routing_refresh_s) <= 86400.0:
            raise ValueError("routing_refresh_s must be between 10 and 86400 seconds")
        if self.timeout_s <= 0 or self.mask_wait_s < 0 or self.max_retries < 0 or self.max_concurrent < 1:
            raise ValueError("timeouts must be positive and max_concurrent >= 1")

    def with_(self, **kw: Any) -> SDKOptions:
        """A copy with some options replaced."""
        return replace(self, **kw)

    @property
    def workers(self) -> int:
        """Effective factory workers per model."""
        return self.mask_workers if self.mask_workers is not None else default_mask_workers()
