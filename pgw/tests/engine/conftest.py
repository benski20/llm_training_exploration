"""Engine test fixtures: in-memory packs built from the shared tiny HF checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RingParams
from pgw.core.types import LinearId
from pgw.engine import (
    ChatTokenizer,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    TransformerModel,
)
from pgw.modelpack.arch import hf_linear_parts


def dequantize_into_hf(model: Any, pack: Any) -> Any:
    """Overwrite an HF model's linear weights with the pack's dequantized int8 weights (in place)."""
    sd = model.state_dict()
    with torch.no_grad():
        for spec in pack.linear_specs():
            lw = pack.load_linear(spec.id)
            w = lw.w8.to(torch.float32) * lw.scale.to(torch.float32)[:, None]
            row = 0
            for _, wn, _ in hf_linear_parts(pack.arch.arch, spec.id.layer, spec.id.kind):
                n = sd[wn].shape[0]
                sd[wn].copy_(w[row : row + n].to(sd[wn].dtype))
                row += n
    return model


def save_tiny_variant(out: Path, arch: str, seed: int, **overrides: Any) -> Path:
    """Tiny random HF checkpoint with config overrides (e.g. rope scaling, untied head)."""
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    from tests.conftest import TINY_ARCH, _save_byte_tokenizer

    torch.manual_seed(seed)
    cfg_kw = {**TINY_ARCH, **overrides}
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**cfg_kw))
    else:
        model = LlamaForCausalLM(LlamaConfig(**cfg_kw))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
            elif name.endswith(".bias"):
                p.normal_(0, 0.02)
    model.eval().save_pretrained(out, safe_serialization=True)
    _save_byte_tokenizer(out)
    return out


@pytest.fixture(scope="session")
def qwen_pack(tiny_hf_dir: Path) -> InMemoryPack:
    """Tiny Qwen2 in-memory pack."""
    return InMemoryPack.from_hf(tiny_hf_dir, name="tiny-qwen2")


@pytest.fixture(scope="session")
def llama_pack(tiny_llama_hf_dir: Path) -> InMemoryPack:
    """Tiny Llama in-memory pack."""
    return InMemoryPack.from_hf(tiny_llama_hf_dir, name="tiny-llama")


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_hf_dir: Path) -> ChatTokenizer:
    """Byte-level tokenizer of the tiny checkpoints."""
    return ChatTokenizer(tiny_hf_dir)


class TwoExtentReservation:
    """Reservation spanning two segments (different seeds, hence different Freivalds vectors)."""

    def __init__(self, pack: Any, ring: RingParams, first: int, second: int):
        self.ring = ring
        self.parts = [
            LocalMaskReservation(pack, ring, first, seed=bytes([1]) * 32),
            LocalMaskReservation(pack, ring, second, seed=bytes([2]) * 32),
        ]
        self.boundaries_seen: list[int] = []

    def _cur(self, lid: LinearId) -> LocalMaskReservation:
        for p in self.parts:
            if p.remaining(lid) > 0:
                return p
        return self.parts[-1]

    def remaining(self, lid: LinearId) -> int:
        return sum(p.remaining(lid) for p in self.parts)

    def rows_until_boundary(self, lid: LinearId) -> int:
        return self._cur(lid).remaining(lid)

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.boundaries_seen.append(rows)
        return self._cur(lid).take(lid, rows)

    def checks(self, lid: LinearId) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cur(lid).checks(lid)

    def close(self) -> None:
        for p in self.parts:
            p.close()


def plain_and_masked(pack: Any, ring: RingParams, **masked_kw: Any) -> tuple[TransformerModel, TransformerModel, Any]:
    """Models over PlainBackend and MaskedBackend (in-process client) for the same pack/ring."""
    client = LocalComputeClient(pack, record=masked_kw.pop("record", False))
    plain = TransformerModel(pack, PlainBackend(pack, ring))
    masked = TransformerModel(pack, MaskedBackend(pack, None, client, ring, pack.fingerprint, **masked_kw))
    return plain, masked, client
