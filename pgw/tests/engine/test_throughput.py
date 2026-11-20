"""Multi-session hot path: grouped finish across sessions, inline decode hops, opt-in batched LM head.

Every optimisation must keep outputs bit-identical to the plaintext reference run alone (the LM-head
batching is opt-in precisely because it does not)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, native
from pgw.engine import Generator, LocalComputeClient, MaskedBackend, PlainBackend, SamplingParams, TransformerModel
from pgw.engine.model import SequenceInput

RING = DEFAULT_RING
PROMPTS = [[3, 17, 42, 5], [9, 9, 1], [100, 7, 8, 64, 2], [11]]
PARAMS = SamplingParams(max_tokens=7, temperature=0.0, ignore_eos=True, logprobs=0)


@pytest.fixture
def store(tmp_path: Path, tiny_pack: Any) -> Any:
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore

    s = MaskStore.open(tmp_path / "store", FileKeyProvider(tmp_path / "kek", create=True), fsync=False)
    MaskGenerator(tiny_pack, RING).generate_into_store(s, 200, chunk_slots=16, factory_id="test")
    yield s
    s.close()


def _reference(pack: Any) -> list[Any]:
    async def run() -> list[Any]:
        gen = Generator(TransformerModel(pack, PlainBackend(pack, RING)), None)
        try:
            return [await gen.agenerate(p, PARAMS) for p in PROMPTS]  # one at a time
        finally:
            await gen.aclose()

    return asyncio.run(run())


async def _masked(pack: Any, store: Any, **model_attrs: Any) -> tuple[list[Any], MaskedBackend]:
    be = MaskedBackend(pack, None, LocalComputeClient(pack), RING, pack.fingerprint, min_soundness_bits=20)
    model = TransformerModel(pack, be)
    for k, v in model_attrs.items():
        setattr(model, k, v)
    gen = Generator(model, None, session_factory=lambda n: store.reserve(pack.fingerprint, RING, n))
    try:
        return list(await asyncio.gather(*(gen.agenerate(p, PARAMS) for p in PROMPTS))), be
    finally:
        await gen.aclose()
        model.close()


@pytest.mark.skipif(not native.native_available("finish_group"), reason="pgw_fast.finish_group not built")
async def test_sessions_share_check_pass_and_stay_exact(tiny_pack: Any, store: Any, monkeypatch: Any) -> None:
    ref = await asyncio.to_thread(_reference, tiny_pack)
    group_sizes: list[int] = []
    fast = native.fast
    real = fast.finish_group

    def spy_group(parts: Any, *a: Any, **k: Any) -> Any:
        group_sizes.append(len(parts))
        return real(parts, *a, **k)

    class Fast:
        def __getattr__(self, name: str) -> Any:
            return spy_group if name == "finish_group" else getattr(fast, name)

    monkeypatch.setattr(native, "fast", Fast())
    got, be = await _masked(tiny_pack, store)
    for a, b in zip(got, ref, strict=True):
        assert a.token_ids == b.token_ids and a.logprobs == b.logprobs  # bit-identical, batch or not
    # Concurrent sessions reserved from one segment: their rows were verified in shared passes.
    assert max(group_sizes) == len(PROMPTS)
    assert be.stats.integrity_failures == 0 and be.stats.freivalds_skipped == 0


async def test_inline_hops_change_nothing(tiny_pack: Any, store: Any) -> None:
    ref = await asyncio.to_thread(_reference, tiny_pack)
    for rows in (0, 8, 1 << 20):  # never inline / decode only / everything but the LM head
        got, _ = await _masked(tiny_pack, store, inline_max_rows=rows)
        for a, b in zip(got, ref, strict=True):
            assert a.token_ids == b.token_ids and a.logprobs == b.logprobs


def test_batched_lm_head_is_opt_in(tiny_pack: Any) -> None:
    model = TransformerModel(tiny_pack, PlainBackend(tiny_pack, RING))
    assert model.batch_lm_head is False
    prompts = PROMPTS[:3]

    def run(batched: bool) -> list[torch.Tensor]:
        model.batch_lm_head = batched
        caches = [model.new_cache(64) for _ in prompts]
        return model.forward_batch([SequenceInput(p, c) for p, c in zip(prompts, caches, strict=True)])

    alone = [model.forward(p, model.new_cache(64)) for p in prompts]
    default = run(False)
    batched = run(True)
    for a, d, b in zip(alone, default, batched, strict=True):
        assert torch.equal(a, d)  # default: identical to running alone
        assert b.shape == a.shape and torch.allclose(a, b, rtol=0, atol=1e-4)  # opt-in: last bits only
        assert int(a.argmax()) == int(b.argmax())
    # One sequence, or sequences that need all positions / row logits, take the unbatched path.
    model.batch_lm_head = True
    assert torch.equal(model.forward(prompts[0], model.new_cache(64)), alone[0])
    model.close()


def test_inline_only_when_masks_cannot_block(tiny_pack: Any, store: Any) -> None:
    from pgw.gateway.masks import WindowedReservation

    fp = tiny_pack.fingerprint
    be = MaskedBackend(tiny_pack, None, LocalComputeClient(tiny_pack), RING, fp)
    model = TransformerModel(tiny_pack, be)
    w = WindowedReservation(store.reserve(fp, RING, 3), lambda n: store.reserve(fp, RING, n), cap=20, window=2)
    cache = model.new_cache(32)

    def safe(ids: list[int]) -> bool:
        return model._inline_safe([SequenceInput(ids, cache, w)], be)

    assert not safe([1])  # first pass of a reservation: its keys and check vectors are not cached yet
    model.forward([1, 2], cache, session=w)
    assert safe([3])  # one row left in the current extent
    assert not safe([3, 4])  # would extend the window (a durable store transaction)
    model.forward([3], cache, session=w)
    assert not safe([4])
    w.close()
