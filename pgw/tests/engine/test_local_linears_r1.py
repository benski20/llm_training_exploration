"""R1 (research/plans/fewer-round-trips.md): more exact local linears, all default off.

``attention_down`` (qkv, o, down on the gateway: one round trip per layer), ``all`` (no round trip) and
``auto_kinds`` (per-kind start-up calibration) must give logits bit-identical to the plaintext
reference; the provider must receive only the remaining kinds; ``local_max_rows`` sends every linear
for large (prefill) steps and keeps decode steps local, still bit-identical.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from pgw.core import ConfigError
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import (
    InMemoryPack,
    LocalComputeClient,
    LocalMaskReservation,
    MaskedBackend,
    PlainBackend,
    SequenceInput,
    TransformerModel,
)
from pgw.engine import backends as B
from tests.engine.test_model_hooks import scenario_run


@pytest.fixture(scope="module")
def ref(qwen_pack: InMemoryPack) -> dict[str, torch.Tensor]:
    model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    model.pipeline_min_rows = 16
    try:
        return scenario_run(model, qwen_pack, "sync", False)
    finally:
        model.close()


REMOTE = {"attention_down": ("gate_up",), "all": ()}


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("policy", ["attention_down", "all"])
@pytest.mark.parametrize("max_rows", [None, 2], ids=["nocap", "cap2"])
def test_r1_modes_bit_identical(qwen_pack: InMemoryPack, ref: dict[str, torch.Tensor], policy: str,
                                mode: str, max_rows: int | None) -> None:
    layers = qwen_pack.arch.num_layers
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, local_linears=policy,
                       local_max_rows=max_rows)
    model = TransformerModel(qwen_pack, be)
    model.pipeline_min_rows = 16
    try:
        assert model._trip_plan() == (len(REMOTE[policy]) * layers, True)
        got = scenario_run(model, qwen_pack, mode, True)
        for k in ref:
            assert torch.equal(ref[k], got[k]), k
        kinds = {lid.split(".", 1)[1] for lid, _ in client.received}
        rows = [int(x.shape[0]) for _, x in client.received]
        if max_rows is None:
            assert kinds <= set(REMOTE[policy])
            if policy == "all":
                assert client.requests == 0 and not client.received
        else:
            # prefill steps (more rows than the cap) sent every linear; capped steps sent only the rest
            assert kinds == {"qkv", "o", "gate_up", "down"}
            for lid, x in client.received:
                if lid.split(".", 1)[1] not in REMOTE[policy]:
                    assert int(x.shape[0]) > 0
            assert max(rows) > max_rows
        assert be.stats.snapshot()["local_calls"] > 0
    finally:
        model.close()


def test_r1_cap_sends_prefill_and_keeps_decode_local(qwen_pack: InMemoryPack) -> None:
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint, local_linears="all",
                       local_max_rows=1)
    model = TransformerModel(qwen_pack, be)
    res = LocalMaskReservation(qwen_pack, DEFAULT_RING, 8)
    cache = model.new_cache(8)
    model.forward_batch([SequenceInput([1, 2, 3], cache, res)])  # 3 rows > cap: every linear remote
    n_prefill = client.requests
    assert n_prefill == 4 * qwen_pack.arch.num_layers
    model.forward_batch([SequenceInput([4], cache, res)])  # 1 row: every linear local, nothing sent
    assert client.requests == n_prefill
    for i in range(qwen_pack.arch.num_layers):
        for kind in (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP, LinearKind.DOWN):
            assert res.remaining(LinearId(i, kind)) == 5  # prefill took 3 rows, decode none
    res.close()
    model.close()


def test_r1_policy_and_auto_kinds(qwen_pack: InMemoryPack) -> None:
    def make(**kw: Any) -> MaskedBackend:
        return MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint, **kw)

    be = make(local_linears="attention_down")
    assert be.local_policy["kinds"] == ("down", "o", "qkv") and be.local_policy["max_rows"] is None
    assert not be.local_linear(LinearId(0, LinearKind.GATE_UP)) and be.local_linear(LinearId(0, LinearKind.DOWN))
    assert make(local_linears="all").local_policy["kinds"] == ("down", "gate_up", "o", "qkv")
    with pytest.raises(ConfigError):
        make(local_linears="all", local_max_rows=0)
    slow = make(local_linears="auto_kinds", local_rtt_s=1.0)  # a 1 s trip: every kind local
    assert slow.local_policy["kinds"] == tuple(sorted(B.LOCAL_ALL_KINDS))
    assert set(slow.local_policy["local_s_by_kind"]) == set(B.LOCAL_ALL_KINDS)
    fast = make(local_linears="auto_kinds", local_rtt_s=1e-9)  # free trips: nothing local
    assert fast.local_policy["kinds"] == () and fast._local_w is None
    assert make(local_linears="auto_kinds").local_policy["kinds"] == ()  # no measurement: off
    # the default is unchanged: off, no cap
    d = make()
    assert d.local_policy["kinds"] == () and d.local_max_rows is None and d.local_rows_ok(10**6)


@pytest.mark.parametrize("value_bits", [12, 20])
@pytest.mark.parametrize("rows", [1, 3, 17])
def test_exact_int_matmul_stacked_digits_matches_float64(value_bits: int, rows: int) -> None:
    """R1 critique: the CPU path stacks the activation digits as rows of one int8 GEMM (one pass over
    the weight). It must equal the float64 product exactly, extremes included."""
    from pgw.engine.backends import exact_int_matmul

    g = torch.Generator().manual_seed(rows * 100 + value_bits)
    n, m = 1000, 37
    hi = (1 << (value_bits - 1)) - 1
    xq = torch.randint(-hi, hi + 1, (rows, n), generator=g, dtype=torch.int64)
    xq[0, :5] = torch.tensor([hi, -hi, 0, hi, -hi])
    w8t = torch.randint(-128, 128, (n, m), generator=g, dtype=torch.int8)
    w8t[:5, 0] = torch.tensor([127, -128, 5, -128, 127], dtype=torch.int8)
    ref = (xq.to(torch.float64) @ w8t.to(torch.float64)).round().to(torch.int64)
    assert torch.equal(exact_int_matmul(xq, w8t, value_bits), ref)
