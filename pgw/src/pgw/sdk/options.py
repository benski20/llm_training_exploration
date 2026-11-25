"""Options of the client library and the models it supports.

Every option has a default that needs no configuration. The defaults for a remote GPU provider are the
measured, exact stack of the 7B 28 tok/s run (STATUS.md, deploy/modal/README.md): every linear on the provider
(``local_linears = "off"``), native float ops, the native per-trip and decode drivers, the masked LM head
(when the provider serves it), the prime-field (p31) check, incremental mask reservations, exact conversation
prefix reuse, and the pks-v3 encrypted mask factory in low-priority background processes. Speculation and other
modes that are not proven exact and safe stay off.
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
        mask_segment_slots: Slots per factory segment (small: masks become usable sooner).
        mask_wait_s: How long a request waits for masks when the stock is empty before failing (back-pressure).
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
        max_concurrent: Requests generating at once per model.
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
    mask_segment_slots: int = 256
    mask_wait_s: float = 120.0
    factory: bool = True
    lane: Literal["auto", "stream", "off"] = "auto"
    native: bool = True
    masked_head: Literal["auto", "on", "off"] = "auto"
    prefix_cache: bool | Literal["auto"] = "auto"
    max_concurrent: int = 8
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
        if self.masked_head not in ("auto", "on", "off"):
            raise ValueError("masked_head must be 'auto', 'on' or 'off'")
        if self.mask_workers is not None and not 1 <= self.mask_workers <= 64:
            raise ValueError("mask_workers must be between 1 and 64")
        if not 0 <= self.mask_nice <= 19:
            raise ValueError("mask_nice must be between 0 and 19")
        if self.timeout_s <= 0 or self.mask_wait_s < 0 or self.max_retries < 0 or self.max_concurrent < 1:
            raise ValueError("timeouts must be positive and max_concurrent >= 1")

    def with_(self, **kw: Any) -> SDKOptions:
        """A copy with some options replaced."""
        return replace(self, **kw)

    @property
    def workers(self) -> int:
        """Effective factory workers per model."""
        return self.mask_workers if self.mask_workers is not None else default_mask_workers()
