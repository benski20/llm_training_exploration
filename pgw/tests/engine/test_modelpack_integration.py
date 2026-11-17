"""Engine over a real on-disk model pack (skipped until pgw.modelpack is implemented)."""

from __future__ import annotations

from typing import Any

import torch

from pgw.core import RING_K40_A12
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LocalComputeClient,
    LocalMaskSource,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)

IDS = [256, 72, 105, 33, 10, 65, 66]


def test_engine_on_modelpack_matches_in_memory(tiny_pack: Any, qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    on_disk = TransformerModel(tiny_pack, PlainBackend(tiny_pack, ring))
    in_mem = TransformerModel(qwen_pack, PlainBackend(qwen_pack, ring))
    a = on_disk.forward(torch.tensor(IDS), on_disk.new_cache(16), all_logits=True)
    b = in_mem.forward(torch.tensor(IDS), in_mem.new_cache(16), all_logits=True)
    assert torch.equal(a, b)


def test_masked_generation_on_modelpack(tiny_pack: Any) -> None:
    ring = RING_K40_A12
    tok = ChatTokenizer.from_pack(tiny_pack)
    src = LocalMaskSource(tiny_pack)
    masked = TransformerModel(
        tiny_pack, MaskedBackend(tiny_pack, None, LocalComputeClient(tiny_pack), ring, tiny_pack.fingerprint)
    )
    plain = TransformerModel(tiny_pack, PlainBackend(tiny_pack, ring))
    p = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    rm = Generator(masked, tok, session_factory=lambda n: src.reserve(tiny_pack.fingerprint, ring, n)).generate(IDS, p)
    rp = Generator(plain, tok).generate(IDS, p)
    assert rm.token_ids == rp.token_ids and rm.text == rp.text
