"""Gateway CPU parallelism (plan P2-c): ``prepare_many`` and the single-use mask prefetch change no byte.

Differential tests against the existing per-item path with no prefetch: two mask stores hold
identical segments (same seeds), so the same reservations exist in both; every request payload the
"provider" receives and every output must be identical, with the same mask consumption.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K32_A12_C27, RingParams
from pgw.core import native as core_native
from pgw.engine import (
    ChatTokenizer,
    Generator,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SamplingParams,
    TransformerModel,
)
from pgw.observability.noise import OutboundNoiseMonitor

RING = RING_K32_A12_C27
native_many = pytest.mark.skipif(not core_native.native_available("quantize_mask_pack_many"),
                                 reason="pgw_fast.quantize_mask_pack_many not built")


class RecordingClient:
    """The in-process service, keeping every request item exactly as the provider receives it."""

    def __init__(self, pack: Any):
        self.inner = LocalComputeClient(pack)
        self.items: list[tuple[str, int, bytes]] = []
        self.ids: list[str] = []

    def compute_sync(self, request: Any) -> Any:
        self.ids.append(request.request_id)
        self.items.extend((it.linear, it.rows, bytes(it.x_hat)) for it in request.items)
        return self.inner.compute_sync(request)


def _store(tmp_path: Path, name: str, pack: Any, ring: RingParams) -> Any:
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore

    store = MaskStore.open(tmp_path / name, FileKeyProvider(tmp_path / f"{name}.kek", create=True), fsync=False,
                           low_water_slots=0, min_free_bytes=0)
    gen = MaskGenerator(pack, ring)
    for i, n in enumerate((13, 11, 17, 9, 20, 30)):  # odd sizes: calls and stages meet extent ends
        gen.generate_into_store(store, n, chunk_slots=8, factory_id="test", seed=bytes([i + 1]) * 32,
                                store_chunk_slots=4)
    return store


def _backend(pack: Any, client: Any, **kw: Any) -> MaskedBackend:
    return MaskedBackend(pack, None, client, RING, pack.fingerprint, **kw)


@pytest.fixture
def twin(tmp_path: Path, tiny_pack: Any) -> Any:
    a, b = _store(tmp_path, "a", tiny_pack, RING), _store(tmp_path, "b", tiny_pack, RING)
    yield a, b
    a.close()
    b.close()


def _waiting(backend: MaskedBackend) -> MaskedBackend:
    """Let every queued prefetch finish before the next prepare (so the staged rows are used)."""
    orig = backend._prepare

    def prep(calls: Any) -> Any:
        assert backend.wait_prefetch(10.0)
        return orig(calls)

    backend._prepare = prep  # type: ignore[method-assign]
    return backend


@native_many
def test_multi_session_generation_is_byte_identical(twin: Any, tiny_pack: Any) -> None:
    sa, sb = twin
    tok = ChatTokenizer.from_pack(tiny_pack)
    ca, cb = RecordingClient(tiny_pack), RecordingClient(tiny_pack)
    ba = _backend(tiny_pack, ca, prepare_many=False, mask_prefetch=False)
    bb = _waiting(_backend(tiny_pack, cb, prepare_many=True, mask_prefetch=True))
    prompts = [[256, 72, 105, 33, 10], [256, 65], [256, 66, 67, 68, 69, 70, 71, 72]]
    params = SamplingParams(max_tokens=7, temperature=0, ignore_eos=True)
    results = []
    reservations = []
    for store, backend in ((sa, ba), (sb, bb)):
        sessions = [store.reserve(tiny_pack.fingerprint, RING, 18, session_id=f"s{i}") for i in range(3)]
        reservations.append(sessions)
        gen = Generator(TransformerModel(tiny_pack, backend), tok)
        results.append(gen.generate_many(prompts, params, sessions=sessions))
    plain = Generator(TransformerModel(tiny_pack, PlainBackend(tiny_pack, RING)), tok)
    for ra, rb, p in zip(results[0], results[1], prompts, strict=True):
        assert ra.token_ids == rb.token_ids == plain.generate(p, params).token_ids
    assert ca.items == cb.items  # the provider sees exactly the same bytes
    assert bb.prefetched_rows > 0
    for s_a, s_b in zip(*reservations, strict=True):
        for key in s_a.linear_keys:
            assert s_a.taken(key) == s_b.taken(key)  # same consumption per session and linear
        s_a.close()
        s_b.close()
        assert s_b.closed and not s_b._stage  # staged leftovers zeroed and dropped
    bb.close()


@native_many
def test_speculative_rows_prefetch_single_session(twin: Any, tiny_pack: Any) -> None:
    """Calls of 1+k = 3 rows on one session (speculative verification) are prefetched for."""
    sa, sb = twin
    ca, cb = RecordingClient(tiny_pack), RecordingClient(tiny_pack)
    ba = _backend(tiny_pack, ca, prepare_many=False, mask_prefetch=False)
    bb = _waiting(_backend(tiny_pack, cb))
    ra = sa.reserve(tiny_pack.fingerprint, RING, 40)
    rb = sb.reserve(tiny_pack.fingerprint, RING, 40)
    specs = [s for s in tiny_pack.linear_specs()][:4]
    g = torch.Generator().manual_seed(3)
    for step in range(12):
        rows = 3 if step % 4 else 1
        for spec in specs:
            x = torch.randn(rows, spec.in_features, generator=g) * 2
            ya = ba.forward_many([LinearCall(spec.id, x, ra)])
            yb = bb.forward_many([LinearCall(spec.id, x, rb)])
            assert torch.equal(ya[0], yb[0])
    assert ca.items == cb.items
    assert bb.prefetched_rows > 0
    ra.close()
    rb.close()
    assert rb.staged_burned >= 0 and not rb._stage
    bb.close()


def test_single_stream_decode_does_not_prefetch(twin: Any, tiny_pack: Any) -> None:
    _, sb = twin
    bb = _backend(tiny_pack, RecordingClient(tiny_pack))
    res = sb.reserve(tiny_pack.fingerprint, RING, 8)
    spec = next(iter(tiny_pack.linear_specs()))
    for _ in range(3):
        for rows in (1, 2, 40 if res.remaining(spec.id) >= 40 else 1):
            if res.remaining(spec.id) < rows:
                break
            bb.forward_many([LinearCall(spec.id, torch.randn(rows, spec.in_features), res)])
    assert bb.wait_prefetch(5.0)
    assert bb.prefetched_rows == 0 and bb._prefetch_pool is None and not res._stage
    res.close()


@native_many
@pytest.mark.parametrize("wait", [True, False], ids=["staged", "racing"])
def test_single_session_decode_prefetch_is_byte_identical(twin: Any, tiny_pack: Any, wait: bool) -> None:
    """L2: with ``prefetch_decode`` (default with synchronous trips) one session's 1-row decode stages its next
    rows in the background: same payloads, outputs and consumption as no prefetch, across extent ends,
    whether the stage is ready (``staged``) or still running when the next take comes (``racing``)."""
    sa, sb = twin
    tok = ChatTokenizer.from_pack(tiny_pack)
    ca, cb = RecordingClient(tiny_pack), RecordingClient(tiny_pack)
    ba = _backend(tiny_pack, ca, mask_prefetch=False)
    bb = _backend(tiny_pack, cb)
    assert not bb.prefetch_decode  # off without a synchronous-trip client
    bb.prefetch_decode = True
    if wait:
        bb = _waiting(bb)
    params = SamplingParams(max_tokens=24, temperature=0, ignore_eos=True)
    prompt = [256, 72, 105, 33]
    outs, sessions = [], []
    for store, backend in ((sa, ba), (sb, bb)):
        res = store.reserve(tiny_pack.fingerprint, RING, 40)
        sessions.append(res)
        outs.append(Generator(TransformerModel(tiny_pack, backend), tok).generate(prompt, params, session=res))
    assert outs[0].token_ids == outs[1].token_ids
    assert outs[0].logprobs == outs[1].logprobs
    assert ca.items == cb.items  # the provider sees exactly the same bytes
    # request ids are uuid4 on every path (filler-only trips use uuid4 too): no per-path id format
    assert ca.ids and all(uuid.UUID(hex=i).version == 4 and uuid.UUID(hex=i).hex == i for i in ca.ids + cb.ids)
    assert bb.wait_prefetch(10.0) and bb.prefetched_rows > 0
    for key in sessions[0].linear_keys:
        assert sessions[0].taken(key) == sessions[1].taken(key)
    for s in sessions:
        s.close()
    assert not sessions[1]._stage
    bb.close()


@native_many
@pytest.mark.parametrize("noise", [False, True])
def test_prepare_many_equals_per_item(qwen_pack: Any, noise: bool) -> None:
    """Several sessions, a prefill block, segment boundaries and item splits: the same request
    bytes and outputs as one native call per item (with and without the noise monitor)."""
    pack = qwen_pack

    def make(many: bool) -> tuple[MaskedBackend, RecordingClient]:
        client = RecordingClient(pack)
        mon = OutboundNoiseMonitor(RING, seed=7, always_sample_first=2, sample_rate=0.3) if noise else None
        return _backend(pack, client, prepare_many=many, mask_prefetch=False, noise_monitor=mon,
                        max_rows_per_item=5), client

    (ba, ca), (bb, cb) = make(False), make(True)
    assert bb.prepare_many and not ba.prepare_many
    specs = list(pack.linear_specs())
    g = torch.Generator().manual_seed(1)

    def sessions() -> list[LocalMaskReservation]:
        return [LocalMaskReservation(pack, RING, 64, seed=bytes([i + 9]) * 32) for i in range(3)]

    sa, sb = sessions(), sessions()
    for step, rows_per_session in enumerate([(12, 1, 3), (1, 1, 1), (7, 0, 2), (1, 3, 1)]):
        spec = specs[step % len(specs)]
        xs = [torch.randn(r, spec.in_features, generator=g) * (1 + step) for r in rows_per_session]
        ya = ba.forward_many([LinearCall(spec.id, x, s) for x, s in zip(xs, sa, strict=True)])
        yb = bb.forward_many([LinearCall(spec.id, x, s) for x, s in zip(xs, sb, strict=True)])
        assert all(torch.equal(a, b) for a, b in zip(ya, yb, strict=True))
    assert ca.items == cb.items
    if noise:
        ra, rb = ba.noise_monitor.report(), bb.noise_monitor.report()  # type: ignore[union-attr]
        assert (ra.calls, ra.sampled_calls, ra.rows_checked) == (rb.calls, rb.sampled_calls, rb.rows_checked)


@native_many
def test_prepare_many_failure_wipes_and_raises_like_per_item(qwen_pack: Any) -> None:
    from pgw.core import RingOverflowError

    pack = qwen_pack
    spec = next(iter(pack.linear_specs()))
    for many in (False, True):
        b = _backend(pack, RecordingClient(pack), prepare_many=many, mask_prefetch=False)
        meta = b._meta.get(spec.id)
        # a weight-row norm no activation bit width fits under the cap: a ring overflow in the kernel
        b._meta._cache[spec.id.key] = dataclasses.replace(meta, wnorm=2.0**60)
        s = [LocalMaskReservation(pack, RING, 8, seed=bytes([i + 1]) * 32) for i in range(2)]
        wiped: list[torch.Tensor] = []
        orig_take = b._take

        def take(*a: Any, _orig: Any = orig_take, _w: list[torch.Tensor] = wiped) -> Any:
            out = _orig(*a)
            _w.extend([out[1], out[2]])
            return out

        b._take = take  # type: ignore[method-assign]
        with pytest.raises(RingOverflowError):
            b.forward_many([LinearCall(spec.id, torch.randn(1, spec.in_features), r) for r in s])
        assert wiped and all(not bool(t.any()) for t in wiped)  # every taken r and W r zeroed


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch, qwen_pack: Any) -> None:
    monkeypatch.setenv("PGW_PREPARE_MANY", "0")
    monkeypatch.setenv("PGW_MASK_PREFETCH", "off")
    b = _backend(qwen_pack, RecordingClient(qwen_pack))
    assert not b.prepare_many and not b.mask_prefetch
    monkeypatch.setenv("PGW_MASK_PREFETCH", "1")
    assert _backend(qwen_pack, RecordingClient(qwen_pack)).mask_prefetch
    assert not _backend(qwen_pack, RecordingClient(qwen_pack), mask_prefetch=False).mask_prefetch


@native_many
@pytest.mark.parametrize("wait", [True, False], ids=["staged", "racing"])
def test_prompt_prefetch_is_byte_identical(twin: Any, tiny_pack: Any, wait: bool) -> None:
    """``prompt_prefetch`` (PGW_PROMPT_PREFETCH): a prompt pass stages each next remote linear's rows ahead;
    same payloads, outputs, logprobs and consumption as no prefetch, across extent ends."""
    import asyncio

    sa, sb = twin
    tok = ChatTokenizer.from_pack(tiny_pack)
    ca, cb = RecordingClient(tiny_pack), RecordingClient(tiny_pack)
    ba = _backend(tiny_pack, ca, mask_prefetch=False)
    bb = _backend(tiny_pack, cb)
    assert not bb.prompt_prefetch  # off by default
    bb.prompt_prefetch = True
    if wait:
        bb = _waiting(bb)
    staged: list[int] = []
    orig = bb.prefetch_prompt

    def spy(items: Any) -> bool:
        staged.extend(int(r) for _, _, r in items)
        return orig(items)

    bb.prefetch_prompt = spy  # type: ignore[method-assign]
    params = SamplingParams(max_tokens=6, temperature=0, ignore_eos=True)
    prompt = [256, *range(65, 85)]
    outs, sessions = [], []
    for store, backend in ((sa, ba), (sb, bb)):
        res = store.reserve(tiny_pack.fingerprint, RING, 40)
        sessions.append(res)
        gen = Generator(TransformerModel(tiny_pack, backend), tok)
        outs.append(asyncio.run(gen.agenerate(prompt, params, session=res)))
    assert outs[0].token_ids == outs[1].token_ids
    assert outs[0].logprobs == outs[1].logprobs
    assert ca.items == cb.items  # the provider sees exactly the same bytes
    assert staged and set(staged) == {len(prompt)}  # prompt rows only (a public count), never decode rows
    assert bb.wait_prefetch(10.0) and bb.prefetched_rows > 0
    for key in sessions[0].linear_keys:
        assert sessions[0].taken(key) == sessions[1].taken(key)
    for s in sessions:
        s.close()
    assert not sessions[1]._stage
    bb.close()
