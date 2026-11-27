"""What the engine needs from a model pack, plus an in-memory pack built from an HF checkpoint.

The engine never reads pack files itself. It talks to any object implementing :class:`EnginePack`,
which :class:`pgw.modelpack.ModelPack` satisfies. :class:`InMemoryPack` implements the same
protocol directly from a Hugging Face checkpoint (directory, state dict or ``PreTrainedModel``),
which is used for development, tests and the "self-factory" mode where no on-disk pack exists yet.

The int8 quantization is the normative one (:func:`pgw.core.quantize_weight_rows`), so an
``InMemoryPack`` and a real pack built from the same checkpoint produce identical engine outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from pgw.core.errors import ConfigError, ShapeError
from pgw.core.fingerprint import compute_fingerprint
from pgw.core.ring import quantize_weight_rows
from pgw.core.types import LinearId, LinearKind, LinearSpec, ModelArch, linear_ids
from pgw.modelpack.arch import hf_linear_parts, hf_plain_tensor_names
from pgw.modelpack.loader import LinearWeights
from pgw.modelpack.manifest import ArchConfig

__all__ = ["EnginePack", "InMemoryPack", "arch_config_from_hf", "linear_split_sizes"]

log = logging.getLogger(__name__)


@runtime_checkable
class EnginePack(Protocol):
    """Subset of :class:`pgw.modelpack.ModelPack` used by the engine."""

    @property
    def fingerprint(self) -> str:
        """Model fingerprint (64 hex chars)."""
        ...

    @property
    def name(self) -> str:
        """Served model name."""
        ...

    @property
    def arch(self) -> ArchConfig:
        """Architecture hyper-parameters."""
        ...

    def linear_specs(self) -> list[LinearSpec]:
        """All linear group specs in canonical order."""
        ...

    def load_linear(self, lid: LinearId | str, device: torch.device | str = "cpu") -> LinearWeights:
        """int8 weight, scale and bias of one linear group."""
        ...

    def load_plain(
        self, name: str, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Embeddings, norms and LM head."""
        ...


def linear_split_sizes(arch: ArchConfig, kind: LinearKind) -> tuple[int, ...]:
    """Row sizes of the parts of a concatenated linear group (SPEC 8.2 order).

    ``qkv`` -> ``(q, k, v)``, ``gate_up`` -> ``(gate, up)``, ``o``/``down`` -> single part.
    """
    hd = arch.head_dim
    if kind == LinearKind.QKV:
        return (arch.num_heads * hd, arch.num_kv_heads * hd, arch.num_kv_heads * hd)
    if kind == LinearKind.O:
        return (arch.hidden_size,)
    if kind == LinearKind.GATE_UP:
        return (arch.intermediate_size, arch.intermediate_size)
    return (arch.hidden_size,)


def _eos_list(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def arch_config_from_hf(
    config: Mapping[str, Any] | Any, generation_config: Mapping[str, Any] | None = None
) -> ArchConfig:
    """Build an :class:`ArchConfig` from an HF config (dict or ``PretrainedConfig``).

    Handles both the transformers 4.x (``rope_theta`` + ``rope_scaling``) and 5.x
    (``rope_parameters``) layouts. ``rope_scaling`` is ``None`` for plain RoPE.
    """
    cfg: dict[str, Any] = dict(config.to_dict() if hasattr(config, "to_dict") else config)
    arch_name = str(cfg.get("model_type"))
    try:
        arch = ModelArch(arch_name)
    except ValueError as exc:
        raise ConfigError(f"unsupported architecture {arch_name!r}; supported: qwen2, llama, qwen3") from exc
    act = cfg.get("hidden_act", "silu")
    if act != "silu":
        raise ConfigError(f"unsupported activation {act!r}; only silu is supported")
    if arch == ModelArch.QWEN3 and (cfg.get("use_sliding_window") or cfg.get("attention_bias")):
        raise ConfigError("Qwen3 checkpoints with use_sliding_window or attention_bias are not supported")
    heads = int(cfg["num_attention_heads"])
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg.get("head_dim") or hidden // heads)

    rope_params = cfg.get("rope_parameters")
    scaling = cfg.get("rope_scaling")
    theta = cfg.get("rope_theta")
    if isinstance(rope_params, Mapping):
        if "full_attention" in rope_params and isinstance(rope_params["full_attention"], Mapping):
            rope_params = rope_params["full_attention"]
        theta = rope_params.get("rope_theta", theta)
        rtype = rope_params.get("rope_type", rope_params.get("type", "default"))
        if rtype not in (None, "default"):
            scaling = {k: v for k, v in rope_params.items() if k != "rope_theta"}
    if isinstance(scaling, Mapping):
        rtype = scaling.get("rope_type", scaling.get("type", "default"))
        scaling = None if rtype in (None, "default") else dict(scaling)
    if theta is None:
        theta = 10000.0

    eos = _eos_list(cfg.get("eos_token_id"))
    if generation_config is not None:
        for e in _eos_list(generation_config.get("eos_token_id")):
            if e not in eos:
                eos = (*eos, e)
    return ArchConfig(
        arch=arch,
        hidden_size=hidden,
        intermediate_size=int(cfg["intermediate_size"]),
        num_layers=int(cfg["num_hidden_layers"]),
        num_heads=heads,
        num_kv_heads=int(cfg.get("num_key_value_heads") or heads),
        head_dim=head_dim,
        vocab_size=int(cfg["vocab_size"]),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_theta=float(theta),
        rope_scaling=scaling,
        max_position_embeddings=int(cfg.get("max_position_embeddings", 4096)),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        bos_token_id=cfg.get("bos_token_id") if isinstance(cfg.get("bos_token_id"), int) else None,
        eos_token_ids=eos,
    )


def _tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


class InMemoryPack:
    """An :class:`EnginePack` held in memory, built from an HF checkpoint.

    Use :meth:`from_hf` (directory with ``config.json`` + safetensors) or :meth:`from_state_dict`.
    The fingerprint is a SHA-256 over the architecture and every tensor's bytes (domain-separated
    from real pack fingerprints, so the two can never be confused).

    Args:
        name: Served model name.
        arch: Architecture config.
        linears: ``{"<layer>.<kind>": LinearWeights}`` for all ``4 * num_layers`` groups.
        plain: Plain tensors keyed by pack names (``embed_tokens``, ``final_norm``, ``lm_head``
            (optional when tied), ``layers.<i>.input_norm``, ``layers.<i>.post_attn_norm``).
        tokenizer_dir: Directory holding tokenizer files (optional).
    """

    def __init__(
        self,
        name: str,
        arch: ArchConfig,
        linears: Mapping[str, LinearWeights],
        plain: Mapping[str, torch.Tensor],
        tokenizer_dir: str | Path | None = None,
    ):
        self._name = name
        self._arch = arch
        self._linears = dict(linears)
        self._plain = {k: v.detach() for k, v in plain.items()}
        self._tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else None
        self.manifest = None
        self._validate()
        self._fingerprint = self._compute_fingerprint()

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        arch: ArchConfig,
        *,
        name: str = "in-memory",
        tokenizer_dir: str | Path | None = None,
    ) -> InMemoryPack:
        """Quantize an HF-named state dict (``model.layers.{i}.self_attn.q_proj.weight`` ...)."""
        linears: dict[str, LinearWeights] = {}
        for lid in linear_ids(arch.num_layers):
            parts = hf_linear_parts(arch.arch, lid.layer, lid.kind)
            try:
                w = torch.cat([state_dict[wn].detach().float() for _, wn, _ in parts], dim=0)
            except KeyError as exc:
                raise ConfigError(f"checkpoint is missing tensor {exc.args[0]!r}") from exc
            biases = [state_dict[bn].detach().float() for _, _, bn in parts if bn is not None and bn in state_dict]
            w8, sw = quantize_weight_rows(w)
            bias = torch.cat(biases).float() if biases else None
            if bias is not None and bias.numel() != w8.shape[0]:
                raise ConfigError(f"{lid}: bias size {bias.numel()} != rows {w8.shape[0]}")
            linears[lid.key] = LinearWeights(lid, w8, sw, bias)
        plain: dict[str, torch.Tensor] = {}
        tied = arch.tie_word_embeddings or "lm_head.weight" not in state_dict
        for pname, hname in hf_plain_tensor_names(arch.num_layers, tied, arch.arch).items():
            if hname not in state_dict:
                raise ConfigError(f"checkpoint is missing tensor {hname!r}")
            plain[pname] = state_dict[hname].detach().float()
        if tied and not arch.tie_word_embeddings:
            arch = arch.model_copy(update={"tie_word_embeddings": True})
        return cls(name, arch, linears, plain, tokenizer_dir)

    @classmethod
    def from_hf(cls, source: str | Path | Any, *, name: str | None = None) -> InMemoryPack:
        """Build from an HF checkpoint directory or a loaded ``PreTrainedModel``."""
        if hasattr(source, "state_dict") and hasattr(source, "config"):
            arch = arch_config_from_hf(source.config)
            return cls.from_state_dict(source.state_dict(), arch, name=name or str(source.config.model_type))
        path = Path(source)
        cfg_file = path / "config.json"
        if not cfg_file.is_file():
            raise ConfigError(f"{path} is not an HF checkpoint directory (no config.json)")
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        gen_file = path / "generation_config.json"
        gen = json.loads(gen_file.read_text(encoding="utf-8")) if gen_file.is_file() else None
        arch = arch_config_from_hf(cfg, gen)
        from safetensors.torch import load_file

        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise ConfigError(f"{path}: no *.safetensors files")
        sd: dict[str, torch.Tensor] = {}
        for f in files:
            sd.update(load_file(str(f)))
        tok = path if (path / "tokenizer_config.json").is_file() or (path / "tokenizer.json").is_file() else None
        log.info("built in-memory pack from %s (%d tensors)", path.name, len(sd))
        return cls.from_state_dict(sd, arch, name=name or path.name, tokenizer_dir=tok)

    # ---- protocol ------------------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Content fingerprint (64 hex chars)."""
        return self._fingerprint

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def arch(self) -> ArchConfig:
        """Architecture config."""
        return self._arch

    @property
    def tokenizer_dir(self) -> Path | None:
        """Tokenizer directory, if any."""
        return self._tokenizer_dir

    def linear_specs(self) -> list[LinearSpec]:
        """Specs in canonical order."""
        out = []
        for lid in linear_ids(self._arch.num_layers):
            lw = self._linears[lid.key]
            out.append(LinearSpec(lid, int(lw.w8.shape[0]), int(lw.w8.shape[1]), lw.bias is not None))
        return out

    def load_linear(self, lid: LinearId | str, device: torch.device | str = "cpu") -> LinearWeights:
        """Weights of one linear group (moved to ``device``)."""
        key = str(lid)
        if key not in self._linears:
            raise ConfigError(f"unknown linear {key!r}")
        lw = self._linears[key]
        if torch.device(device) == lw.w8.device:
            return lw
        return LinearWeights(
            lw.id, lw.w8.to(device), lw.scale.to(device), None if lw.bias is None else lw.bias.to(device)
        )

    def load_plain(
        self, name: str, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Plain tensor; ``lm_head`` falls back to ``embed_tokens`` when tied."""
        if name == "lm_head" and "lm_head" not in self._plain:
            name = "embed_tokens"
        if name not in self._plain:
            raise ConfigError(f"unknown plain tensor {name!r}")
        return self._plain[name].to(device=device, dtype=dtype)

    # ---- internals -----------------------------------------------------------------------------
    def _validate(self) -> None:
        a = self._arch
        for lid in linear_ids(a.num_layers):
            if lid.key not in self._linears:
                raise ConfigError(f"missing linear {lid}")
            lw = self._linears[lid.key]
            m = sum(linear_split_sizes(a, lid.kind))
            n = a.intermediate_size if lid.kind == LinearKind.DOWN else (
                a.num_heads * a.head_dim if lid.kind == LinearKind.O else a.hidden_size)
            if tuple(lw.w8.shape) != (m, n):
                raise ShapeError(f"{lid}: weight shape {tuple(lw.w8.shape)} != expected {(m, n)}")
        emb = self._plain.get("embed_tokens")
        if emb is None or tuple(emb.shape) != (a.vocab_size, a.hidden_size):
            raise ShapeError("embed_tokens missing or wrong shape")

    def _compute_fingerprint(self) -> str:
        doc: dict[str, Any] = {
            "format": "pgw-engine-inmemory",
            "arch": self._arch.model_dump(mode="json"),
            "linears": {
                k: [_tensor_digest(v.w8), _tensor_digest(v.scale), None if v.bias is None else _tensor_digest(v.bias)]
                for k, v in sorted(self._linears.items())
            },
            "plain": {k: _tensor_digest(v) for k, v in sorted(self._plain.items())},
        }
        return compute_fingerprint(doc)
