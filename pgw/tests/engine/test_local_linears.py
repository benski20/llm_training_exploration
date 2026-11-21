"""Chat-speed WP-A4b, E2: exact local linears in :class:`MaskedBackend`.

With ``local_linears="attention"`` the gateway computes qkv and o itself from the pack's int8 weights
with the masked path's quantization and an exact integer product: local == masked == plain, bit for
bit; the provider receives only gate_up and down items; the local linears take no mask rows. The
``auto`` policy is decided once from a start-up calibration (public), never per token.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, ConfigError, IntegrityError, RingParams
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SequenceInput,
    TransformerModel,
)
from pgw.engine import backends as B
from tests.engine.test_model_hooks import scenario_run


def _ref(pack: Any) -> dict[str, torch.Tensor]:
    model = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
    model.pipeline_min_rows = 16
    try:
        return scenario_run(model, pack, "sync", False)
    finally:
        model.close()


@pytest.fixture(scope="module")
def refs(qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> dict[str, dict[str, torch.Tensor]]:
    return {"qwen": _ref(qwen_pack), "llama": _ref(llama_pack)}


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("trip", [False, True], ids=["today", "trip"])
@pytest.mark.parametrize("which", ["qwen", "llama"])
def test_local_equals_masked_equals_plain(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                         refs: dict[str, dict[str, torch.Tensor]], which: str, trip: bool,
                                         mode: str) -> None:
    pack = qwen_pack if which == "qwen" else llama_pack
    layers = pack.arch.num_layers
    client = LocalComputeClient(pack, record=True)
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint, local_linears="attention",
                       trip_engine=trip)
    model = TransformerModel(pack, be)
    model.pipeline_min_rows = 16
    try:
        assert model._trip_plan() == (2 * layers, True)
        got = scenario_run(model, pack, mode, True)
        ref = refs[which]
        for k in ref:
            assert torch.equal(ref[k], got[k]), k
        # the provider received only gate_up / down items
        assert client.received
        assert {lid for lid, _ in client.received} <= {f"{i}.{k}" for i in range(layers) for k in ("gate_up", "down")}
        st = be.stats.snapshot()
        assert st["local_calls"] > 0 and st["round_trips"] == client.requests
    finally:
        model.close()


def test_local_linears_take_no_masks(qwen_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       local_linears="attention")
    model = TransformerModel(qwen_pack, be)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    model.forward_batch([SequenceInput([1, 2, 3], model.new_cache(8), res)])
    for i in range(qwen_pack.arch.num_layers):
        assert res.remaining(LinearId(i, LinearKind.QKV)) == 8 and res.remaining(LinearId(i, LinearKind.O)) == 8
        assert res.remaining(LinearId(i, LinearKind.GATE_UP)) == 5 and res.remaining(LinearId(i, LinearKind.DOWN)) == 5


@pytest.mark.parametrize("ring", [DEFAULT_RING, RING_K40_A12], ids=lambda r: r.name)
@pytest.mark.parametrize("lid", ["0.qkv", "1.o"])
def test_forward_local_equals_plain_backend(qwen_pack: InMemoryPack, ring: RingParams, lid: str) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), ring, qwen_pack.fingerprint,
                       local_linears="attention")
    plain = PlainBackend(qwen_pack, ring)
    lin = LinearId.parse(lid)
    n = qwen_pack.load_linear(lin).w8.shape[1]
    g = torch.Generator().manual_seed(3)
    calls = [LinearCall(lin, torch.randn(t, n, generator=g) * s) for t, s in ((1, 1.0), (5, 30.0), (3, 1e-3))]
    for c, y in zip(calls, be.forward_local(calls), strict=True):
        assert torch.equal(y, plain.forward(c))
    with pytest.raises(ConfigError):
        be.forward_local([LinearCall(LinearId(0, LinearKind.DOWN), torch.zeros(1, qwen_pack.arch.intermediate_size))])


def test_policy_modes(qwen_pack: InMemoryPack) -> None:
    def make(**kw: Any) -> MaskedBackend:
        return MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint, **kw)

    be = make()
    assert be.local_policy["mode"] == "off" and not be.local_linear(LinearId(0, LinearKind.QKV))
    with pytest.raises(ConfigError):
        make(local_linears="everything")
    be = make(local_linears="attention")
    assert be.local_linear(LinearId(0, LinearKind.QKV)) and be.local_linear(LinearId(1, LinearKind.O))
    assert not be.local_linear(LinearId(0, LinearKind.GATE_UP)) and not be.local_linear(LinearId(1, LinearKind.DOWN))
    assert be.local_policy["kinds"] == ("o", "qkv")
    # weights are loaded at configuration (start-up), never on a request
    assert be._local_w is not None and len(be._local_w._cache) == 2 * qwen_pack.arch.num_layers


def test_auto_calibration(qwen_pack: InMemoryPack) -> None:
    def make(**kw: Any) -> MaskedBackend:
        return MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint, **kw)

    slow_link = make(local_linears="auto", local_rtt_s=1.0)  # a 1 s round trip: local always wins
    assert slow_link.local_policy["kinds"] == ("o", "qkv") and slow_link.local_policy["local_s"] > 0
    fast_link = make(local_linears="auto", local_rtt_s=1e-9)  # a free round trip: never local
    assert fast_link.local_policy["kinds"] == () and fast_link._local_w is None
    assert not fast_link.local_linear(LinearId(0, LinearKind.QKV))
    unknown = make(local_linears="auto")  # no measurement yet: stays off
    assert unknown.local_policy["kinds"] == () and "no round-trip" in unknown.local_policy["reason"]
    # with measured decode trips (stats), auto uses their mean
    be = make()
    be.stats.add_many(16, 16, 16, 0, 0, 16, 0, 16 * 2.0, 0.0, decode=True)
    assert be.configure_local_linears("auto") == B.LOCAL_ATTENTION_KINDS
    assert be.local_policy["rtt_s"] == pytest.approx(2.0)
    assert be.configure_local_linears("off") == frozenset() and be._local_w is None


def test_local_linear_device_choice(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    """The device check compares integer outputs with the CPU's; without CUDA the CPU is kept."""
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert be.configure_local_linears("attention", device="auto") == B.LOCAL_ATTENTION_KINDS
    assert be.local_policy["device"] == "cpu" and be.local_policy["device_reason"] == "no CUDA device"
    with pytest.raises(ConfigError):
        be.configure_local_linears("attention", device="tpu")
    # the check itself (run on CPU tensors as the "device"): identical outputs pass, a wrong product fails
    policy: dict[str, Any] = {}
    assert be._local_device_check(torch.device("cpu"), 2, policy, need_faster=False) == torch.device("cpu")
    assert policy["device_check"]["identical"]
    real = be._local_int_product

    def wrong(w: Any, xq: torch.Tensor) -> torch.Tensor:
        y = real(w, xq)
        return y + 1 if w.w8t.data_ptr() not in cpu_ptrs else y

    cpu_ptrs = {be._local_w.get(s.id).w8t.data_ptr() for s in qwen_pack.linear_specs()  # type: ignore[union-attr]
                if s.id.kind.value in B.LOCAL_ATTENTION_KINDS}
    monkeypatch.setattr(be, "_local_int_product", wrong)
    policy = {}
    assert be._local_device_check(torch.device("cpu"), 2, policy, need_faster=False) is None
    assert "differ" in policy["device_reason"]


def test_local_linear_refuses_out_of_range_outputs_like_the_masked_path(qwen_pack: InMemoryPack,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       local_linears="attention")
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(2, qwen_pack.arch.hidden_size)
    be.forward_local([LinearCall(lid, x)])
    bound = DEFAULT_RING.output_bound(qwen_pack.arch.hidden_size)
    real = B.exact_int_matmul

    def too_big(xq: torch.Tensor, w8t: torch.Tensor, value_bits: int) -> torch.Tensor:
        y = real(xq, w8t, value_bits)
        y[1, 0] = bound + 1
        return y

    monkeypatch.setattr(B, "exact_int_matmul", too_big)
    with pytest.raises(IntegrityError) as ei:
        be.forward_local([LinearCall(lid, x)])
    assert ei.value.details["bound"] == bound
    be.close()
