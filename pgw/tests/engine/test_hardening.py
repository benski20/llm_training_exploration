"""Hardening regressions: mask hygiene on failure paths, fingerprint binding, ownership, loop robustness."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K40_A12,
    ConfigError,
    FingerprintMismatchError,
    IntegrityError,
    LinearId,
    RingOverflowError,
    ServiceUnavailableError,
)
from pgw.engine import (
    ChatTokenizer,
    Generator,
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    SamplingParams,
)
from tests.engine.conftest import plain_and_masked

QKV0 = LinearId.parse("0.qkv")
PROMPT = [256, 72, 105, 33, 10]


def _x(rows: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, n, generator=g) * 3


class SpyReservation(LocalMaskReservation):
    """Records the ``W r`` tensors handed out so tests can check they were wiped."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.handed: list[torch.Tensor] = []

    def take(self, lid: LinearId, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, wr = super().take(lid, rows)
        self.handed.append(wr)
        return r, wr


class _Down(LocalComputeClient):
    def compute_sync(self, request: Any) -> Any:
        raise ConnectionError("network down")


def test_reservation_fingerprint_must_match(qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), RING_K40_A12, qwen_pack.fingerprint)
    other = LocalMaskReservation(qwen_pack, RING_K40_A12, 2)
    other.fingerprint = llama_pack.fingerprint
    with pytest.raises(FingerprintMismatchError):
        be.forward(LinearCall(QKV0, _x(1, qwen_pack.arch.hidden_size), session=other))
    assert other.remaining(QKV0) == 2


def test_bad_activation_in_later_call_consumes_no_masks(qwen_pack: InMemoryPack) -> None:
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint)
    n = qwen_pack.arch.hidden_size
    a = LocalMaskReservation(qwen_pack, RING_K40_A12, 4)
    b = LocalMaskReservation(qwen_pack, RING_K40_A12, 4)
    bad = _x(2, n)
    bad[1, 3] = float("nan")
    with pytest.raises(RingOverflowError):
        be.forward_many([LinearCall(QKV0, _x(2, n), session=a), LinearCall(QKV0, bad, session=b)])
    assert a.remaining(QKV0) == 4 and b.remaining(QKV0) == 4 and client.requests == 0


def test_wr_wiped_after_transport_failure_and_success(qwen_pack: InMemoryPack) -> None:
    n = qwen_pack.arch.hidden_size
    be = MaskedBackend(qwen_pack, None, _Down(qwen_pack), RING_K40_A12, qwen_pack.fingerprint)
    res = SpyReservation(qwen_pack, RING_K40_A12, 4)
    with pytest.raises(ServiceUnavailableError):
        be.forward(LinearCall(QKV0, _x(2, n), session=res))
    assert res.handed and all(int(w.abs().sum()) == 0 for w in res.handed)
    assert res.remaining(QKV0) == 2  # consumed rows are never handed out again

    ok = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), RING_K40_A12, qwen_pack.fingerprint)
    res2 = SpyReservation(qwen_pack, RING_K40_A12, 4)
    ok.forward(LinearCall(QKV0, _x(2, n), session=res2))
    assert all(int(w.abs().sum()) == 0 for w in res2.handed)


def test_wr_wiped_after_integrity_failure(qwen_pack: InMemoryPack) -> None:
    n = qwen_pack.arch.hidden_size
    tripped: list[BaseException] = []
    client = LocalComputeClient(qwen_pack, tamper=lambda _lid, y: y + 1)
    be = MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint, on_integrity_failure=tripped.append)
    res = SpyReservation(qwen_pack, RING_K40_A12, 4)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(2, n), session=res))
    assert len(tripped) == 1 and all(int(w.abs().sum()) == 0 for w in res.handed)


def test_noise_monitor_alarm_wipes_and_blocks_send(qwen_pack: InMemoryPack) -> None:
    class Alarm:
        def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None) -> None:
            raise IntegrityError("noise monitor alarm")

    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint, noise_monitor=Alarm())
    res = SpyReservation(qwen_pack, RING_K40_A12, 4)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(2, qwen_pack.arch.hidden_size), session=res))
    assert client.requests == 0 and all(int(w.abs().sum()) == 0 for w in res.handed)


def test_zero_row_call_has_output_shape_and_no_round_trip(qwen_pack: InMemoryPack) -> None:
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint)
    res = LocalMaskReservation(qwen_pack, RING_K40_A12, 2)
    n = qwen_pack.arch.hidden_size
    m = sum(qwen_pack.arch.num_heads * qwen_pack.arch.head_dim if i == 0 else
            qwen_pack.arch.num_kv_heads * qwen_pack.arch.head_dim for i in range(3))
    y = be.forward(LinearCall(QKV0, torch.zeros((0, n)), session=res))
    assert tuple(y.shape) == (0, m) and client.requests == 0 and res.remaining(QKV0) == 2
    # mixed with a non-empty call: one round trip, correct shapes
    ys = be.forward_many([LinearCall(QKV0, torch.zeros((0, n)), session=res), LinearCall(QKV0, _x(1, n), session=res)])
    assert tuple(ys[0].shape) == (0, m) and tuple(ys[1].shape) == (1, m) and client.requests == 1


async def test_async_cancellation_wipes(qwen_pack: InMemoryPack) -> None:
    gate = asyncio.Event()

    class Slow(LocalComputeClient):
        async def compute(self, request: Any) -> Any:
            gate.set()
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

    be = MaskedBackend(qwen_pack, None, Slow(qwen_pack), RING_K40_A12, qwen_pack.fingerprint)
    res = SpyReservation(qwen_pack, RING_K40_A12, 4)
    task = asyncio.create_task(be.aforward(LinearCall(QKV0, _x(2, qwen_pack.arch.hidden_size), session=res)))
    await gate.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert res.handed and all(int(w.abs().sum()) == 0 for w in res.handed)


def test_own_session_closed_on_finish_and_on_validation_error(
    qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer
) -> None:
    _, masked, _ = plain_and_masked(qwen_pack, RING_K40_A12)
    gen = Generator(masked, tiny_tokenizer)
    sp = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True)
    res = LocalMaskReservation(qwen_pack, RING_K40_A12, len(PROMPT) + 2)
    gen.generate(PROMPT, sp, session=res, own_session=True)
    assert res._closed
    borrowed = LocalMaskReservation(qwen_pack, RING_K40_A12, len(PROMPT) + 2)
    gen.generate(PROMPT, sp, session=borrowed)
    assert not borrowed._closed  # caller keeps ownership by default
    bad = LocalMaskReservation(qwen_pack, RING_K40_A12, 4)
    with pytest.raises(ConfigError):
        gen.generate([], sp, session=bad, own_session=True)
    assert bad._closed


def test_session_factory_session_closed_when_state_setup_fails(
    qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, masked, _ = plain_and_masked(qwen_pack, RING_K40_A12)
    made: list[LocalMaskReservation] = []

    def factory(slots: int) -> LocalMaskReservation:
        made.append(LocalMaskReservation(qwen_pack, RING_K40_A12, slots))
        return made[-1]

    gen = Generator(masked, tiny_tokenizer, session_factory=factory)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise MemoryError("no room for the cache")

    monkeypatch.setattr(masked, "new_cache", boom)
    with pytest.raises(MemoryError):
        gen.generate(PROMPT, SamplingParams(max_tokens=2))
    assert len(made) == 1 and made[0]._closed


async def test_batching_loop_survives_non_pgw_error_in_isolation(
    qwen_pack: InMemoryPack, tiny_tokenizer: ChatTokenizer
) -> None:
    """A session that hits a non-PGW error during exhaustion isolation fails alone; others finish."""
    from pgw.core import MaskExhaustedError

    _, masked, _ = plain_and_masked(qwen_pack, RING_K40_A12)
    gen = Generator(masked, tiny_tokenizer)
    sp = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True)

    class Broken(LocalMaskReservation):
        def remaining(self, lid: LinearId) -> int:
            raise RuntimeError("store backend crashed")

    good = LocalMaskReservation(qwen_pack, RING_K40_A12, len(PROMPT) + 3)
    short = LocalMaskReservation(qwen_pack, RING_K40_A12, 2)
    broken = Broken(qwen_pack, RING_K40_A12, len(PROMPT) + 3)

    async def run(session: Any) -> Any:
        try:
            return await asyncio.wait_for(gen.agenerate(PROMPT, sp, session=session), 30)
        except Exception as exc:
            return exc

    # The precheck sees ``short`` first, so the batch fails with exhaustion and is isolated per
    # session; ``broken`` then raises a plain RuntimeError, which must not kill the loop.
    results = await asyncio.gather(run(short), run(good), run(broken))
    assert isinstance(results[0], MaskExhaustedError)
    assert not isinstance(results[1], BaseException) and results[1].completion_tokens == 3
    assert isinstance(results[2], RuntimeError)
    await gen.aclose()
