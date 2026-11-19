"""Pipelined prefill: large masked round trips are split in two overlapping halves, results unchanged."""

from __future__ import annotations

import asyncio

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.engine import InMemoryPack, LocalMaskReservation
from pgw.engine.model import SequenceInput, TransformerModel
from tests.engine.conftest import plain_and_masked


@pytest.mark.parametrize("prompt_len", [3, 16, 37])
def test_pipelined_prefill_is_identical(qwen_pack: InMemoryPack, prompt_len: int) -> None:
    plain, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    ids = [int(t) for t in torch.randint(0, 250, (prompt_len,), generator=torch.Generator().manual_seed(prompt_len))]
    ref = plain.forward_batch([SequenceInput(ids, plain.new_cache(64), all_logits=True)])[0]
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, prompt_len + 4)
    be = masked.backend

    async def run() -> torch.Tensor:
        cache = masked.new_cache(64)
        out = (await masked.aforward_batch([SequenceInput(ids, cache, res, all_logits=True)]))[0]
        nxt = (await masked.aforward_batch([SequenceInput([7], cache, res)]))[0]  # decode after
        assert torch.equal(nxt, plain_next(ids))
        return out

    def plain_next(prefix: list[int]) -> torch.Tensor:
        c = plain.new_cache(64)
        plain.forward_batch([SequenceInput(prefix, c)])
        return plain.forward_batch([SequenceInput([7], c)])[0]

    trips0 = be.stats.round_trips
    got = asyncio.run(run())
    assert torch.equal(got, ref)
    layers = qwen_pack.arch.num_layers
    expected = 4 * layers * (2 if prompt_len >= TransformerModel.pipeline_min_rows else 1) + 4 * layers
    assert be.stats.round_trips - trips0 == expected  # pipelined halves only for the prefill
    assert be.stats.integrity_failures == 0
