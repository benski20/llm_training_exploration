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

__all__ = ["SUPPORTED_MODELS", "SDKOptions", "canonical_model", "default_mask_workers"]

#: Models of this release (provider names). Anything else is refused with a clear message.
SUPPORTED_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
)


def canonical_model(name: str) -> str:
    """``"Qwen/Qwen2.5-7B-Instruct"`` / ``"qwen2.5-7b-instruct"`` -> ``"qwen2.5-7b-instruct"``.

    Raises:
        UnsupportedModelError: not one of :data:`SUPPORTED_MODELS`.
    """
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedModelError("model is required (supported: " + ", ".join(SUPPORTED_MODELS) + ")")
    n = name.strip().lower()
    n = re.sub(r"^(qwen/)", "", n)
    if n in SUPPORTED_MODELS:
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
            During a model's first fill on a host (the factory still building its one-time setup) a request without
            an explicit ``timeout`` waits up to ``ready_timeout_s`` instead.
        factory: Run the background mask factory (``False``: consume existing stock only).
        lane: ``"auto"`` (the provider's TCP+TLS trip lane when offered, else the blocking WebSocket stream lane),
            ``"stream"`` or ``"off"`` (the async stream channel only).
        native: The native trip / decode drivers (the measured stack; exact). ``False``: the Python trip path.
            Process-wide (environment switches read when a model starts): the first client's choice applies.
        masked_head: ``"auto"`` (the int8 LM head as a masked linear when the provider serves it), ``"on"``
            (required), ``"off"``.
        prefix_cache: Exact conversation prefix reuse (the KV cache of the previous turn, in RAM). ``"auto"``
            (default): on when the masked LM head is not used (the engine proves prefix reuse exact only with the
            gateway's own head); ``True``: on, and the masked head off; ``False``: off.
        hint: How the pks-v3 hint is obtained on first use of a model [``PGW_HINT``]: ``"local"`` (default; the
            background factory builds it from the fingerprint-checked public weights: one-time, about 20-30 min of
            CPU at 7B on 4 cores, then cached) or ``"provider"`` (opt-in; the provider computes the expensive part
            and the client verifies it with a >= 40-bit check; streams about 44 GB at 7B).
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
            target-feature draft head file, ``draft_head_path``: a few MB, one small CPU step per draft row).
        spec_rows: Rows per verify pass with ``speculation`` (2-8; default 3: up to 2 drafts per pass).
        draft_head_path: The draft head file for ``speculation="draft_head"`` (a public artefact for this model).
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
    factory: bool = True
    lane: Literal["auto", "stream", "off"] = "auto"
    native: bool = True
    masked_head: Literal["auto", "on", "off"] = "auto"
    prefix_cache: bool | Literal["auto"] = "auto"
    hint: Literal["local", "provider"] = "local"
    model_fingerprints: dict[str, str] = field(default_factory=dict)
    usage_reports: bool = True
    max_concurrent: int = 8
    speculation: Literal["off", "lookup", "draft_head"] = "off"
    spec_rows: int = 3
    draft_head_path: Path | None = None
    log_level: str | None = None
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
        spec = os.environ.get("PGW_SPECULATION", "").strip()
        if spec:
            base["speculation"] = spec
        lvl = os.environ.get("PGW_SDK_LOG", "").strip()
        if lvl:
            base["log_level"] = lvl
        base.update({k: v for k, v in overrides.items() if v is not None})
        opts = cls(**base)
        opts.validate()
        return opts

    def validate(self) -> None:
        """Raise ``ValueError`` on an invalid combination."""
        if self.mask_scheme not in ("pks-v3", "mlwe", "plain"):
            raise ValueError("mask_scheme must be 'pks-v3', 'mlwe' or 'plain'")
        if self.lane not in ("auto", "stream", "off"):
            raise ValueError("lane must be 'auto', 'stream' or 'off'")
        if self.prefix_cache not in (True, False, "auto"):
            raise ValueError("prefix_cache must be True, False or 'auto'")
        if self.prefix_cache is True and self.masked_head == "on":
            raise ValueError("prefix_cache=True and masked_head='on' do not combine (prefix reuse is proven exact "
                             "with the gateway's own LM head only)")
        if self.hint not in ("local", "provider"):
            raise ValueError("hint must be 'local' or 'provider'")
        for m, fp in (self.model_fingerprints or {}).items():
            if not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{64}", fp.lower()):
                raise ValueError(f"model_fingerprints[{m!r}] must be 64 hex digits")
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
        if self.speculation == "draft_head" and self.draft_head_path is None:
            raise ValueError("speculation='draft_head' needs draft_head_path (the trained head file)")
        if self.draft_head_path is not None and self.speculation != "draft_head":
            raise ValueError("draft_head_path is used only with speculation='draft_head'")
        if self.timeout_s <= 0 or self.mask_wait_s < 0 or self.max_retries < 0 or self.max_concurrent < 1:
            raise ValueError("timeouts must be positive and max_concurrent >= 1")

    def with_(self, **kw: Any) -> SDKOptions:
        """A copy with some options replaced."""
        return replace(self, **kw)

    @property
    def workers(self) -> int:
        """Effective factory workers per model."""
        return self.mask_workers if self.mask_workers is not None else default_mask_workers()
