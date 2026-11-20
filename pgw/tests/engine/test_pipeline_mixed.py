"""Mixed-batch pipelining: a 1-row decode next to prefills still splits the prefills in halves."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, MaskExhaustedError
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LocalMaskReservation
from pgw.engine.model import SequenceInput, TransformerModel
from tests.engine.conftest import plain_and_masked


def _ids(n: int, seed: int) -> list[int]:
    return [int(t) for t in torch.randint(0, 250, (n,), generator=torch.Generator().manual_seed(seed))]


def _plain_ref(plain: TransformerModel, steps: list[list[int]]) -> torch.Tensor:
    """Logits of the last step of one sequence fed step by step through the plain model."""
    cache = plain.new_cache(128)
    out = None
    for ids in steps:
        out = plain.forward_batch([SequenceInput(ids, cache, all_logits=True)])[0]
    assert out is not None
    return out


def _lids(pack: InMemoryPack) -> list[LinearId]:
    return [LinearId(i, k) for i in range(pack.arch.num_layers) for k in LinearKind]


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (1, 2, 0)])
@pytest.mark.parametrize("min_rows", [16, 0])
def test_mixed_batch_identical_and_masks_in_order(qwen_pack: InMemoryPack, order: tuple[int, ...],
                                                  min_rows: int) -> None:
    """[1-row decode, 40-row prompt, 40-row prompt] (any order): same logits split or not; each session
    consumes exactly its rows; the split doubles the round trips of that pass."""
    plain, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    masked.pipeline_min_rows = min_rows
    be = masked.backend
    head = _ids(5, 1)
    prompts = [[7], _ids(40, 2), _ids(40, 3)]  # session 0 decodes after its 5-token prompt
    refs = [_plain_ref(plain, [head, [7]]), _plain_ref(plain, [prompts[1]]), _plain_ref(plain, [prompts[2]])]
    slots = [5 + 1 + 3, 40 + 3, 40 + 3]
    res = [LocalMaskReservation(qwen_pack, DEFAULT_RING, n) for n in slots]
    caches = [masked.new_cache(128) for _ in range(3)]

    async def run() -> list[torch.Tensor]:
        await masked.aforward_batch([SequenceInput(head, caches[0], res[0])])
        seqs = [SequenceInput(prompts[i], caches[i], res[i], all_logits=True) for i in order]
        return await masked.aforward_batch(seqs)

    trips0 = be.stats.round_trips
    out = asyncio.run(run())
    layers = qwen_pack.arch.num_layers
    assert be.stats.round_trips - trips0 == 4 * layers + 4 * layers * (2 if min_rows else 1)
    for pos, i in enumerate(order):
        assert torch.equal(out[pos], refs[i]), i
    used = [5 + 1, 40, 40]
    for r, n, want in zip(res, slots, used, strict=True):
        for lid in _lids(qwen_pack):
            assert n - int(r.remaining(lid)) == want, lid
    assert be.stats.integrity_failures == 0


def test_mixed_split_takes_half_a_first(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per session, the first half's rows are prepared (masked) before the second half's; the 1-row
    call is only in the first half; halves are rejoined by call position."""
    _, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    be = masked.backend
    seen: list[list[tuple[int, int]]] = []
    real = be.prepare_calls

    def spy(calls: Any) -> Any:
        seen.append([(id(c.session), int(c.x.shape[0])) for c in calls])
        return real(calls)

    monkeypatch.setattr(be, "prepare_calls", spy)
    res = [LocalMaskReservation(qwen_pack, DEFAULT_RING, 50) for _ in range(3)]
    caches = [masked.new_cache(128) for _ in range(3)]

    async def run() -> None:
        await masked.aforward_batch([SequenceInput([1, 2], caches[0], res[0])])
        seen.clear()
        await masked.aforward_batch([SequenceInput(_ids(21, 4), caches[1], res[1]),
                                     SequenceInput([3], caches[0], res[0]),
                                     SequenceInput(_ids(30, 5), caches[2], res[2])])

    asyncio.run(run())
    a, b, c = (id(r) for r in res)
    assert seen[0] == [(b, 10), (a, 1), (c, 15)]
    assert seen[1] == [(b, 11), (c, 15)]
    assert len(seen) == 2 * 4 * qwen_pack.arch.num_layers
    assert seen[0::2] == [seen[0]] * (len(seen) // 2)
    assert seen[1::2] == [seen[1]] * (len(seen) // 2)


def test_only_decode_rows_do_not_split(qwen_pack: InMemoryPack) -> None:
    """Many 1-row calls (no call of 2+ rows) are one round trip per linear as before."""
    _, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    masked.pipeline_min_rows = 2
    be = masked.backend
    res = [LocalMaskReservation(qwen_pack, DEFAULT_RING, 4) for _ in range(3)]
    caches = [masked.new_cache(16) for _ in range(3)]

    async def run() -> None:
        await masked.aforward_batch([SequenceInput([i + 1], caches[i], res[i]) for i in range(3)])

    trips0 = be.stats.round_trips
    asyncio.run(run())
    assert be.stats.round_trips - trips0 == 4 * qwen_pack.arch.num_layers


def test_exhausted_second_half_wipes_and_raises(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session that runs out in the second half: MaskExhaustedError, every prepared request is
    finished or discarded (wiped), the other sessions' second-half rows are not taken."""
    _, masked, _ = plain_and_masked(qwen_pack, DEFAULT_RING)
    be = masked.backend
    prepared: list[Any] = []
    done: list[Any] = []
    real_prep, real_fin, real_disc = be.prepare_calls, be.finish_prepared, be.discard

    def prep(calls: Any) -> Any:
        p = real_prep(calls)
        prepared.append(p)
        return p

    def fin(p: Any, *a: Any) -> Any:
        done.append(p)
        return real_fin(p, *a)

    def disc(p: Any) -> None:
        done.append(p)
        real_disc(p)

    monkeypatch.setattr(be, "prepare_calls", prep)
    monkeypatch.setattr(be, "finish_prepared", fin)
    monkeypatch.setattr(be, "discard", disc)
    ok = LocalMaskReservation(qwen_pack, DEFAULT_RING, 50)
    dec = LocalMaskReservation(qwen_pack, DEFAULT_RING, 10)
    short = LocalMaskReservation(qwen_pack, DEFAULT_RING, 30)  # 20 rows in half A fit, 20 more do not
    caches = [masked.new_cache(128) for _ in range(3)]

    async def run() -> None:
        await masked.aforward_batch([SequenceInput([1, 2], caches[1], dec)])
        await masked.aforward_batch([SequenceInput(_ids(40, 6), caches[0], ok),
                                     SequenceInput([3], caches[1], dec),
                                     SequenceInput(_ids(40, 7), caches[2], short)])

    with pytest.raises(MaskExhaustedError):
        asyncio.run(run())
    assert prepared and all(any(p is d for d in done) for p in prepared)
    lid = LinearId(0, LinearKind.QKV)
    assert 50 - int(ok.remaining(lid)) == 20  # only its first half was taken
    assert 10 - int(dec.remaining(lid)) == 3
    assert 30 - int(short.remaining(lid)) == 20
