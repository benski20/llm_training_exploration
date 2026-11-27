"""Shared value types: linear identifiers, specs and architecture names.

A model exposes ``4 * num_layers`` masked linear *groups*, identified by :class:`LinearId` whose
canonical string form is ``"<layer>.<kind>"`` (e.g. ``"3.qkv"``). The canonical execution order
inside a layer is ``qkv, o, gate_up, down``. These strings appear in the model pack manifest, on the
wire and in mask store segments, and must match byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from pgw.core.errors import ConfigError

__all__ = ["LINEAR_ORDER", "LinearId", "LinearKind", "LinearSpec", "ModelArch", "linear_ids"]


class LinearKind(StrEnum):
    """The four masked linear groups of a decoder layer."""

    QKV = "qkv"  #: q_proj, k_proj, v_proj concatenated along the output dim (in that order)
    O = "o"  #: o_proj  # noqa: E741
    GATE_UP = "gate_up"  #: gate_proj, up_proj concatenated along the output dim (in that order)
    DOWN = "down"  #: down_proj
    #: The LM head as a masked linear (opt-in, :mod:`pgw.engine.masked_head`): id ``"<num_layers>.lm_head"``,
    #: ``(vocab, hidden)``, after the last layer. Never part of :data:`LINEAR_ORDER` / :func:`linear_ids`.
    LM_HEAD = "lm_head"


LINEAR_ORDER: tuple[LinearKind, ...] = (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP, LinearKind.DOWN)

_ID_RE = re.compile(r"^(\d+)\.(qkv|o|gate_up|down|lm_head)$")


@dataclass(frozen=True, slots=True, order=True)
class LinearId:
    """Identifier of one masked linear group, e.g. ``LinearId(3, LinearKind.QKV)`` == ``"3.qkv"``."""

    layer: int
    kind: LinearKind
    #: Cached canonical string (hot path: dictionary keys of every masked call).
    _key: str = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ConfigError(f"layer index must be >= 0, got {self.layer}")
        object.__setattr__(self, "kind", LinearKind(self.kind))
        object.__setattr__(self, "_key", f"{self.layer}.{self.kind.value}")

    @property
    def key(self) -> str:
        """Canonical string form ``"<layer>.<kind>"``."""
        return self._key

    @property
    def ordinal(self) -> int:
        """Position in the model-wide execution order (``4 * layer + index(kind)``)."""
        if self.kind == LinearKind.LM_HEAD:  # masked LM head "<num_layers>.lm_head": after the last down
            return 4 * self.layer
        return 4 * self.layer + LINEAR_ORDER.index(self.kind)

    @classmethod
    def parse(cls, s: str) -> LinearId:
        """Parse ``"3.qkv"``."""
        m = _ID_RE.match(s)
        if not m:
            raise ConfigError(f"bad linear id {s!r}; expected '<layer>.<qkv|o|gate_up|down|lm_head>'")
        return cls(int(m.group(1)), LinearKind(m.group(2)))

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True, slots=True)
class LinearSpec:
    """Shape of one masked linear group: ``y (m) = W (m, n) x (n) [+ b]``."""

    id: LinearId
    out_features: int
    in_features: int
    has_bias: bool = False

    def __post_init__(self) -> None:
        if self.out_features <= 0 or self.in_features <= 0:
            raise ConfigError(f"{self.id}: dimensions must be positive")


class ModelArch(StrEnum):
    """Supported decoder architectures (HF ``model_type``)."""

    QWEN2 = "qwen2"
    LLAMA = "llama"
    #: Qwen3 dense: Llama-style linears (no biases) plus a per-head RMSNorm of q and k before RoPE (gateway side).
    QWEN3 = "qwen3"


def linear_ids(num_layers: int) -> list[LinearId]:
    """All linear ids of a model in canonical execution order."""
    return [LinearId(i, k) for i in range(num_layers) for k in LINEAR_ORDER]
