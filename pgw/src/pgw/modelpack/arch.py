"""Mapping from Hugging Face parameter names to PGW linear groups and plain tensors.

This table is part of the contract (SPEC.md section 8.2): the order of ``parts`` defines the row
order of concatenated groups (``qkv`` = q, k, v; ``gate_up`` = gate, up), which the engine relies on
when splitting outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pgw.core.errors import ConfigError
from pgw.core.types import LinearId, LinearKind, ModelArch

if TYPE_CHECKING:
    from pgw.modelpack.manifest import ArchConfig

__all__ = [
    "HF_LINEAR_PARTS",
    "HF_PLAIN_TENSORS",
    "KNOWN_ROPE_TYPES",
    "arch_config_from_hf",
    "expected_linear_shape",
    "hf_linear_parts",
    "hf_plain_tensor_names",
]

log = logging.getLogger(__name__)

#: RoPE scaling types the engine implements (others are recorded but logged as a warning).
KNOWN_ROPE_TYPES: frozenset[str] = frozenset({"linear", "llama3", "yarn"})

#: (part name, HF suffix under ``model.layers.{i}.``) per linear kind; identical for Qwen2 and Llama.
HF_LINEAR_PARTS: dict[LinearKind, tuple[tuple[str, str], ...]] = {
    LinearKind.QKV: (("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj")),
    LinearKind.O: (("o_proj", "self_attn.o_proj"),),
    LinearKind.GATE_UP: (("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj")),
    LinearKind.DOWN: (("down_proj", "mlp.down_proj"),),
}

#: PGW plain tensor name -> HF parameter name (``{i}`` = layer index). ``lm_head`` may be absent
#: when ``tie_word_embeddings`` is true; the loader then returns ``embed_tokens``.
HF_PLAIN_TENSORS: dict[str, str] = {
    "embed_tokens": "model.embed_tokens.weight",
    "final_norm": "model.norm.weight",
    "lm_head": "lm_head.weight",
    "layers.{i}.input_norm": "model.layers.{i}.input_layernorm.weight",
    "layers.{i}.post_attn_norm": "model.layers.{i}.post_attention_layernorm.weight",
}


def hf_linear_parts(arch: ModelArch | str, layer: int, kind: LinearKind) -> list[tuple[str, str, str | None]]:
    """``[(part_name, hf_weight_name, hf_bias_name_or_None), ...]`` for one linear group.

    Qwen2 has biases on q/k/v only; Llama has no linear biases.
    """
    a = ModelArch(arch)
    if a not in (ModelArch.QWEN2, ModelArch.LLAMA):  # pragma: no cover - future archs
        raise ConfigError(f"unsupported architecture {arch}")
    out: list[tuple[str, str, str | None]] = []
    for part, suffix in HF_LINEAR_PARTS[LinearKind(kind)]:
        base = f"model.layers.{layer}.{suffix}"
        has_bias = a == ModelArch.QWEN2 and kind == LinearKind.QKV
        out.append((part, base + ".weight", base + ".bias" if has_bias else None))
    return out


def hf_plain_tensor_names(num_layers: int, tie_word_embeddings: bool) -> dict[str, str]:
    """PGW plain tensor name -> HF name for a model with ``num_layers`` layers."""
    out: dict[str, str] = {}
    for k, v in HF_PLAIN_TENSORS.items():
        if k == "lm_head" and tie_word_embeddings:
            continue
        if "{i}" in k:
            for i in range(num_layers):
                out[k.format(i=i)] = v.format(i=i)
        else:
            out[k] = v
    return out


# ---- HF config -> ArchConfig --------------------------------------------------------------------
def _eos_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, list | tuple):
        return [int(x) for x in v]
    raise ConfigError(f"bad eos_token_id {v!r}")


def _rope(cfg: Mapping[str, Any], allow_unsupported: bool = False) -> tuple[float, dict[str, Any] | None]:
    """``(rope_theta, rope_scaling)`` from either the classic or the transformers-5 config layout.

    Raises:
        ConfigError: a RoPE scaling type the engine does not implement (unless ``allow_unsupported``).
    """
    params = dict(cfg.get("rope_parameters") or {})
    theta = cfg.get("rope_theta", params.pop("rope_theta", None))
    params.pop("rope_theta", None)
    scaling = cfg.get("rope_scaling") or (params or None)
    if scaling is not None:
        scaling = dict(scaling)
        scaling.pop("rope_theta", None)
        if "rope_type" not in scaling and "type" in scaling:
            scaling["rope_type"] = scaling.pop("type")
        rtype = scaling.get("rope_type", "default")
        if rtype == "default":
            scaling = None
        elif rtype not in KNOWN_ROPE_TYPES:
            if not allow_unsupported:
                raise ConfigError(
                    f"rope scaling type {rtype!r} is not supported by the PGW engine "
                    f"(supported: default, {', '.join(sorted(KNOWN_ROPE_TYPES))}); a pack built from it could "
                    "not be served. Use --allow-unsupported-rope only for packs served by another engine",
                    details={"rope_type": rtype},
                )
            log.warning("rope scaling type %r is not supported by the PGW engine; recorded anyway "
                        "(allow_unsupported_rope)", rtype)
    return float(theta if theta is not None else 10000.0), scaling


def arch_config_from_hf(
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None = None,
    *,
    allow_unsupported_rope: bool = False,
) -> ArchConfig:
    """Build an :class:`~pgw.modelpack.manifest.ArchConfig` from an HF ``config.json`` dict.

    Only dense decoder-only ``qwen2`` and ``llama`` models with SiLU MLPs are supported. EOS ids
    are the union (in order) of ``config.eos_token_id`` and ``generation_config.eos_token_id``.

    Raises:
        ConfigError: unsupported architecture or feature (sliding window, attention/MLP biases on
            Llama, non-SiLU activation, inconsistent head dimensions, a RoPE scaling type outside
            :data:`KNOWN_ROPE_TYPES` unless ``allow_unsupported_rope``).
    """
    from pgw.modelpack.manifest import ArchConfig

    model_type = config.get("model_type")
    try:
        arch = ModelArch(str(model_type))
    except ValueError as exc:
        raise ConfigError(
            f"unsupported model_type {model_type!r}; supported: {[a.value for a in ModelArch]}",
            details={"model_type": model_type},
        ) from exc
    act = config.get("hidden_act", "silu")
    if act != "silu":
        raise ConfigError(f"unsupported hidden_act {act!r} (only 'silu')")
    if arch == ModelArch.QWEN2 and config.get("use_sliding_window"):
        raise ConfigError("Qwen2 checkpoints with use_sliding_window=true are not supported")
    if arch == ModelArch.LLAMA and (config.get("attention_bias") or config.get("mlp_bias")):
        raise ConfigError("Llama checkpoints with attention_bias/mlp_bias are not supported")
    if int(config.get("pretraining_tp", 1) or 1) != 1:
        log.info("ignoring pretraining_tp=%s (weights are stored unsplit)", config.get("pretraining_tp"))
    try:
        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        kv = int(config.get("num_key_value_heads") or heads)
        layers = int(config["num_hidden_layers"])
        inter = int(config["intermediate_size"])
        vocab = int(config["vocab_size"])
    except KeyError as exc:
        raise ConfigError(f"config.json is missing required key {exc.args[0]!r}") from exc
    head_dim = int(config.get("head_dim") or hidden // heads)
    if heads % kv:
        raise ConfigError(f"num_attention_heads={heads} is not a multiple of num_key_value_heads={kv}")
    if config.get("head_dim") is None and hidden % heads:
        raise ConfigError(f"hidden_size={hidden} is not divisible by num_attention_heads={heads}")
    theta, scaling = _rope(config, allow_unsupported_rope)
    eos: list[int] = []
    for v in _eos_list(config.get("eos_token_id")) + _eos_list((generation_config or {}).get("eos_token_id")):
        if v not in eos:
            eos.append(v)
    bos = config.get("bos_token_id")
    if bos is None and generation_config:
        bos = generation_config.get("bos_token_id")
    return ArchConfig(
        arch=arch,
        hidden_size=hidden,
        intermediate_size=inter,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=kv,
        head_dim=head_dim,
        vocab_size=vocab,
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        rope_theta=theta,
        rope_scaling=scaling,
        max_position_embeddings=int(config.get("max_position_embeddings", 2048)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        bos_token_id=int(bos) if bos is not None else None,
        eos_token_ids=tuple(eos),
    )


def expected_linear_shape(arch: ArchConfig, lid: LinearId) -> tuple[int, int, list[tuple[str, int]]]:
    """``(out_features, in_features, [(part, rows), ...])`` implied by ``arch`` for one linear group."""
    q = arch.num_heads * arch.head_dim
    kv = arch.num_kv_heads * arch.head_dim
    h, i = arch.hidden_size, arch.intermediate_size
    parts: dict[LinearKind, tuple[int, list[tuple[str, int]]]] = {
        LinearKind.QKV: (h, [("q_proj", q), ("k_proj", kv), ("v_proj", kv)]),
        LinearKind.O: (q, [("o_proj", h)]),
        LinearKind.GATE_UP: (h, [("gate_proj", i), ("up_proj", i)]),
        LinearKind.DOWN: (i, [("down_proj", h)]),
    }
    n, p = parts[lid.kind]
    return sum(r for _, r in p), n, p
