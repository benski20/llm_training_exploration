"""The masked remote LM head (``pgw.engine.masked_head``, opt-in): exact against its pinned plaintext
reference (the same wrapped pack on ``PlainBackend``), Freivalds-checked and range-checked like every
linear (a tampered head result is refused), domain-separated fingerprint, single-use masks."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.core.errors import IntegrityError, PGWError
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LocalComputeClient, LocalMaskReservation, MaskedBackend, PlainBackend
from pgw.engine.masked_head import MaskedHeadPack, enable_masked_head, head_linear_id, quantize_head
from pgw.engine.model import SequenceInput, TransformerModel


def _ids(n: int, seed: int) -> list[int]:
    return [int(t) for t in torch.randint(0, 250, (n,), generator=torch.Generator().manual_seed(seed))]


def _model(pack: Any, masked: bool, client: Any = None, **kw: Any) -> TransformerModel:
    if masked:
        be: Any = MaskedBackend(pack, None, client or LocalComputeClient(pack), DEFAULT_RING, pack.fingerprint, **kw)
        be.defer_min_out = 0
    else:
        be = PlainBackend(pack, DEFAULT_RING)
    model = TransformerModel(pack, be)
    enable_masked_head(model)
    return model


def _run(model: TransformerModel, pack: Any, masked: bool, mode: str = "sync") -> dict[str, torch.Tensor]:
    def drive(seqs: list[SequenceInput]) -> list[Any]:
        return model.forward_batch(seqs) if mode == "sync" else asyncio.run(model.aforward_batch(seqs))

    out: dict[str, torch.Tensor] = {}
    res = (lambda n, s: LocalMaskReservation(pack, DEFAULT_RING, n, seed=bytes([s]) * 32)) if masked else \
        (lambda n, s: None)
    ca, cb = model.new_cache(128), model.new_cache(128)
    ra, rb = res(48, 1), res(48, 2)
    out["a_prefill"] = drive([SequenceInput(_ids(20, 1), ca, ra, all_logits=True)])[0]
    for i in range(3):
        out[f"a_decode{i}"] = drive([SequenceInput([11 + i], ca, ra)])[0]
    drive([SequenceInput(_ids(9, 2), cb, rb)])
    for i in range(2):
        la, lb = drive([SequenceInput([20 + i], ca, ra), SequenceInput([30 + i], cb, rb)])
        out[f"ab{i}_a"], out[f"ab{i}_b"] = la, lb
    for r in (ra, rb):
        if r is not None:
            r.close()
    return out


@pytest.fixture(scope="module")
def head_pack(qwen_pack: InMemoryPack) -> MaskedHeadPack:
    return MaskedHeadPack(qwen_pack)


def test_pack_wrapper(qwen_pack: InMemoryPack, head_pack: MaskedHeadPack) -> None:
    lid = head_linear_id(qwen_pack.arch.num_layers)
    assert lid.key == f"{qwen_pack.arch.num_layers}.lm_head" and LinearId.parse(lid.key) == lid
    assert lid.ordinal == max(s.id.ordinal for s in qwen_pack.linear_specs()) + 1
    specs = head_pack.linear_specs()
    assert specs[:-1] == qwen_pack.linear_specs() and specs[-1].id == lid
    assert (specs[-1].out_features, specs[-1].in_features) == (qwen_pack.arch.vocab_size, qwen_pack.arch.hidden_size)
    assert head_pack.fingerprint != qwen_pack.fingerprint and len(head_pack.fingerprint) == 64
    assert MaskedHeadPack(qwen_pack).fingerprint == head_pack.fingerprint  # deterministic
    lw = head_pack.load_linear(lid)
    w = qwen_pack.load_plain("lm_head", dtype=torch.float32)
    from pgw.core.ring import quantize_weight_rows

    w8, sw = quantize_weight_rows(w)  # the blockwise quantizer is the reference one, bit for bit
    assert torch.equal(lw.w8, w8) and torch.equal(lw.scale, sw)
    b8, bs = quantize_head(w[:5])
    assert torch.equal(b8, w8[:5]) and torch.equal(bs, sw[:5])
    meta = head_pack.load_linear_meta(lid)
    assert meta.bias is None and meta.row_norm_sq_max is not None
    assert LinearKind.LM_HEAD.value == "lm_head" and LinearKind.LM_HEAD not in (
        k for _, k in [(0, x) for x in (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP, LinearKind.DOWN)])


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_masked_equals_plain(head_pack: MaskedHeadPack, mode: str) -> None:
    plain = _model(head_pack, False)
    ref = _run(plain, head_pack, False)
    plain.close()
    model = _model(head_pack, True)
    try:
        got = _run(model, head_pack, True, mode)
        assert model.backend.stats.integrity_failures == 0
        # one head call per sequence and pass: the head linear went over the (masked) wire
        assert model._trip_plan()[0] == 4 * head_pack.arch.num_layers + 1
    finally:
        model.close()
    assert ref.keys() == got.keys()
    for k in ref:
        assert ref[k].dtype == torch.float32
        assert torch.equal(ref[k], got[k]), k


def test_trip_engine_equals_plain(head_pack: MaskedHeadPack, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core import native

    if not native.native_available("trip_prepare_into"):
        pytest.skip("pgw_fast trip kernels not built")
    monkeypatch.setenv("PGW_TRIP_FAST", "1")
    plain = _model(head_pack, False)
    ref = _run(plain, head_pack, False)
    plain.close()
    model = _model(head_pack, True, trip_engine=True, sync_trips=True)
    try:
        model.sync_decode = True
        got = _run(model, head_pack, True)
    finally:
        model.close()
    for k in ref:
        assert torch.equal(ref[k], got[k]), k


def test_close_to_float_head(qwen_pack: InMemoryPack, head_pack: MaskedHeadPack) -> None:
    """The int8 head is a new pinned reference: close to the float head, same argmax here."""
    ref = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    got = _model(head_pack, False)
    try:
        ids = _ids(16, 5)
        a = ref.forward_batch([SequenceInput(ids, ref.new_cache(32), None, all_logits=True)])[0]
        b = got.forward_batch([SequenceInput(ids, got.new_cache(32), None, all_logits=True)])[0]
    finally:
        ref.close()
        got.close()
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert float((a - b).abs().max()) < 0.05 * float(a.abs().max())


class _Tamper(LocalComputeClient):
    """Adds 1 to the first output element of every head result (an active provider)."""

    def __init__(self, pack: Any, key: str):
        super().__init__(pack)
        self.key = key

    def _tamper(self, resp: Any) -> Any:
        from pgw.wire.messages import ComputeResponse, ComputeResult

        items = []
        for it in resp.items:
            y = bytearray(bytes(it.y_hat))
            if it.linear == self.key:
                y[0] = (y[0] + 1) % 256
            items.append(ComputeResult(it.linear, it.rows, bytes(y)))
        return ComputeResponse(resp.request_id, tuple(items))

    async def compute(self, req: Any) -> Any:
        return self._tamper(await super().compute(req))

    def compute_sync(self, req: Any) -> Any:
        return self._tamper(super().compute_sync(req))


def test_tampered_head_refused(head_pack: MaskedHeadPack) -> None:
    key = head_linear_id(head_pack.arch.num_layers).key
    model = _model(head_pack, True, client=_Tamper(head_pack, key))
    model.backend.freivalds_rate = 1.0
    res = LocalMaskReservation(head_pack, DEFAULT_RING, 16, seed=bytes([9]) * 32)
    try:
        with pytest.raises((IntegrityError, PGWError)):
            model.forward_batch([SequenceInput(_ids(4, 3), model.new_cache(16), res)])
            model.backend.join_checks()
        assert model.backend.stats.integrity_failures >= 1
    finally:
        res.close()
        model.close()


def test_compact_embedding_rows_exact(qwen_pack: InMemoryPack) -> None:
    """``PGW_EMBED_COMPACT``: a bfloat16-exact table kept in bfloat16 gives the same passes bit for bit."""
    from pgw.engine.lm_head import compact_head

    ref = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    got = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    try:
        exact = ref.embed_tokens.to(torch.bfloat16).to(torch.float32)  # a bfloat16 checkpoint's table
        ref.embed_tokens = exact
        got.embed_tokens = compact_head(exact.clone())
        assert got.embed_tokens.dtype == torch.bfloat16
        ids = _ids(12, 4)
        a = ref.forward_batch([SequenceInput(ids, ref.new_cache(16), None, all_logits=True)])[0]
        b = got.forward_batch([SequenceInput(ids, got.new_cache(16), None, all_logits=True)])[0]
        assert torch.equal(a, b)
    finally:
        ref.close()
        got.close()


def test_default_off(qwen_pack: InMemoryPack) -> None:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    try:
        assert model.head_mode is None
        assert all(s.id.kind != LinearKind.LM_HEAD for s in qwen_pack.linear_specs())
    finally:
        model.close()
