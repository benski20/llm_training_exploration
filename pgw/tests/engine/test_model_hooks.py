"""WP-A5 engine hooks: every default is bitwise-equal to the engine before the refactor.

Two references, both bit for bit:

* ``model_hooks_golden.json``: SHA-256 digests of the logits of :func:`scenarios`, captured with the
  engine **before** the WP-A5 change (commit 6810bf9). Float bits depend on the CPU kernels, torch and
  the tiny checkpoints' random init, so the digests are compared only where they were captured
  (same ``env_key``); elsewhere that test skips and the in-process checks below still run.
* In-process: with the hook objects installed explicitly (``TorchFloatOps``, ``StandardSchedule``,
  ``GatewayHead``), and with the new paths (thread scope, synchronous decode loop, causal SDPA,
  ``pipeline_blocks``) on and off, every scenario gives the same bits.

Recapture (only on purpose, with the pre-change engine checked out)::

    PYTHONPATH=src:. python -m tests.engine.test_model_hooks --capture
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LocalComputeClient, LocalMaskReservation, MaskedBackend, PlainBackend
from pgw.engine.backends import LinearCall
from pgw.engine.model import (
    GatewayHead,
    PassState,
    SequenceInput,
    StandardSchedule,
    TorchFloatOps,
    TransformerModel,
)

GOLDEN = Path(__file__).with_name("model_hooks_golden.json")


def env_key() -> str:
    """Where float bits are reproducible: torch, transformers, CPU ISA level and machine."""
    import transformers

    cap = torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends, "cpu") else "?"
    return f"torch={torch.__version__} transformers={transformers.__version__} cpu={cap} {platform.machine()}"


def _ids(n: int, seed: int) -> list[int]:
    return [int(t) for t in torch.randint(0, 250, (n,), generator=torch.Generator().manual_seed(seed))]


def _res(pack: Any, slots: int, seed: int) -> LocalMaskReservation:
    return LocalMaskReservation(pack, DEFAULT_RING, slots, seed=bytes([seed]) * 32)


def _rows(x: Any) -> torch.Tensor:
    """Tensor of a step output (RowLogits: all rows stacked)."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack([x[i] for i in range(len(x))])


def _drive(model: TransformerModel, mode: str, seqs: list[SequenceInput]) -> list[Any]:
    if mode == "sync":
        return model.forward_batch(seqs)
    return asyncio.run(model.aforward_batch(seqs))


def scenario_run(model: TransformerModel, pack: Any, mode: str, masked: bool) -> dict[str, torch.Tensor]:
    """One fixed sequence of passes; returns ``{name: logits}``.

    ``mode``: ``sync`` (forward_batch) or ``async`` (aforward_batch: the split-hop path with a
    masked backend). Covers prefill from an empty cache (all rows), chunked prefill (start > 0),
    1-row decode, two sequences in lockstep, a speculative verify pass with a serial tail
    (RowLogits) and a mixed batch (prefill next to a 1-row decode).
    """
    out: dict[str, torch.Tensor] = {}
    sess = (lambda n, s: _res(pack, n, s)) if masked else (lambda n, s: None)
    # sequence A: 40-row prefill, 3 decode steps, then a verify pass of 4 rows (serial tail 3)
    ca = model.new_cache(256)
    ra = sess(64, 1)
    out["a_prefill"] = _rows(_drive(model, mode, [SequenceInput(_ids(40, 1), ca, ra, all_logits=True)])[0])
    for i in range(3):
        out[f"a_decode{i}"] = _rows(_drive(model, mode, [SequenceInput([11 + i], ca, ra)])[0])
    rl = _drive(model, mode, [SequenceInput([5, 6, 7, 8], ca, ra, last_logits=4, serial_tail=3)])[0]
    out["a_verify"] = _rows(rl)
    # sequence B: chunked prefill (13 then 7 rows) and C: 1-row prompt; then B and C in lockstep
    cb, cc = model.new_cache(256), model.new_cache(256)
    rb, rc = sess(48, 2), sess(48, 3)
    _drive(model, mode, [SequenceInput(_ids(13, 2), cb, rb)])
    out["b_chunk2"] = _rows(_drive(model, mode, [SequenceInput(_ids(7, 3), cb, rb, all_logits=True)])[0])
    _drive(model, mode, [SequenceInput([42], cc, rc)])
    for i in range(2):
        lb, lc = _drive(model, mode, [SequenceInput([20 + i], cb, rb), SequenceInput([30 + i], cc, rc)])
        out[f"bc_decode{i}_b"], out[f"bc_decode{i}_c"] = _rows(lb), _rows(lc)
    # mixed batch: a 30-row prompt next to a decode row of B (pipelined halves when masked + async)
    cd = model.new_cache(256)
    rd = sess(40, 4)
    ld, lb = _drive(model, mode, [SequenceInput(_ids(30, 4), cd, rd, all_logits=True), SequenceInput([9], cb, rb)])
    out["d_prompt"], out["mixed_b"] = _rows(ld), _rows(lb)
    for r in (ra, rb, rc, rd):
        if r is not None:
            r.close()
    return out


def build(pack: Any, masked: bool, **attrs: Any) -> TransformerModel:
    if masked:
        backend: Any = MaskedBackend(pack, None, LocalComputeClient(pack), DEFAULT_RING, pack.fingerprint)
    else:
        backend = PlainBackend(pack, DEFAULT_RING)
    model = TransformerModel(pack, backend)
    model.pipeline_min_rows = 16
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


#: (name, masked, mode, model attributes)
CASES: tuple[tuple[str, bool, str, dict[str, Any]], ...] = (
    ("plain_sync", False, "sync", {}),
    ("plain_sync_lazyhead", False, "sync", {"batch_lm_head": False}),
    ("masked_async", True, "async", {}),
    ("masked_sync", True, "sync", {}),
)


def scenarios(packs: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """All cases for every pack; ``extra`` model attributes are applied on top of each case's."""
    res: dict[str, dict[str, torch.Tensor]] = {}
    for pname, pack in packs.items():
        for cname, masked, mode, attrs in CASES:
            model = build(pack, masked, **{**attrs, **(extra or {})})
            try:
                res[f"{pname}/{cname}"] = scenario_run(model, pack, mode, masked)
            finally:
                model.close()
    return res


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:32]


def digests(res: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, str]]:
    return {case: {k: digest(v) for k, v in outs.items()} for case, outs in res.items()}


def _packs(qwen: Any, llama: Any) -> dict[str, Any]:
    return {"qwen": qwen, "llama": llama}


# ---- tests --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference(qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> dict[str, dict[str, torch.Tensor]]:
    return scenarios(_packs(qwen_pack, llama_pack))


def _same(a: dict[str, dict[str, torch.Tensor]], b: dict[str, dict[str, torch.Tensor]]) -> None:
    assert a.keys() == b.keys()
    for case in a:
        assert a[case].keys() == b[case].keys()
        for k in a[case]:
            assert torch.equal(a[case][k], b[case][k]), f"{case}/{k}"


def test_golden_digests_from_before_the_change(reference: dict[str, dict[str, torch.Tensor]]) -> None:
    """Defaults reproduce the logits the pre-refactor engine produced (same machine class)."""
    gold = json.loads(GOLDEN.read_text())
    if gold["env_key"] != env_key():
        pytest.skip(f"golden digests were captured on {gold['env_key']!r}, here {env_key()!r}")
    assert digests(reference) == gold["digests"]


def test_masked_equals_plain(reference: dict[str, dict[str, torch.Tensor]]) -> None:
    for p in ("qwen", "llama"):
        for name in ("masked_async", "masked_sync", "plain_sync_lazyhead"):
            ref, got = reference[f"{p}/plain_sync"], reference[f"{p}/{name}"]
            for k in ref:
                assert torch.equal(ref[k], got[k]), f"{p}/{name}/{k}"


def test_explicit_default_hooks_are_bitwise_equal(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                                  reference: dict[str, dict[str, torch.Tensor]]) -> None:
    extra = {"float_ops": TorchFloatOps(), "schedule": StandardSchedule(), "head_mode": GatewayHead()}
    _same(reference, scenarios(_packs(qwen_pack, llama_pack), extra))


@pytest.mark.parametrize("extra", [{"causal_sdpa": False}, {"pipeline_blocks": 4}, {"pipeline_min_rows": 0},
                                   {"inline_max_rows": 0}], ids=lambda d: next(iter(d)))
def test_speed_paths_are_bitwise_equal(qwen_pack: InMemoryPack, llama_pack: InMemoryPack,
                                       reference: dict[str, dict[str, torch.Tensor]], extra: dict[str, Any]) -> None:
    _same(reference, scenarios(_packs(qwen_pack, llama_pack), extra))


def test_custom_hooks_are_called(qwen_pack: InMemoryPack, reference: dict[str, dict[str, torch.Tensor]]) -> None:
    """Subclassed hooks see every op / layer / head pass; a head mode may add a round trip (counted by
    the trip plan); delegating to the defaults keeps the bits."""
    counts: dict[str, int] = {}

    def tick(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    class Ops(TorchFloatOps):
        def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
            tick("rms_norm")
            return super().rms_norm(x, w, eps)

        def norm_rows(self, h: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
            tick("norm_rows")
            return super().norm_rows(h, w, eps)

        def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            tick("rope")
            return super().rope(x, cos, sin)

        def attention(self, model: TransformerModel, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      start: int) -> torch.Tensor:
            tick("attention")
            return super().attention(model, q, k, v, start)

        def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            tick("silu_mul")
            return super().silu_mul(gate, up)

    class Sched(StandardSchedule):
        def layer(self, model: TransformerModel, layer: int, st: PassState) -> Any:
            tick("layer")
            return (yield from super().layer(model, layer, st))

    class ExtraTripHead(GatewayHead):
        def linears(self, model: TransformerModel) -> Any:
            return (model._lids[-1][3],)

        def steps(self, model: TransformerModel, seqs: Any, hs: list[torch.Tensor]) -> Any:
            tick("head")
            inter = model.split_gate_up[0]
            ys = yield [LinearCall(model._lids[-1][3], torch.zeros(1, inter), s.session) for s in seqs]
            assert len(ys) == len(seqs)
            return model._logits(seqs, hs)

    model = build(qwen_pack, False, float_ops=Ops(), schedule=Sched(), head_mode=ExtraTripHead())
    assert model._trip_plan() == (4 * qwen_pack.arch.num_layers + 1, False)
    trips: list[int] = []
    real = model.backend.forward_many

    def counted(calls: Any) -> list[torch.Tensor]:
        trips.append(len(calls))
        return real(calls)

    model.backend.forward_many = counted  # type: ignore[method-assign]
    got = scenario_run(model, qwen_pack, "sync", False)
    ref = reference["qwen/plain_sync"]
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    passes = counts["head"]
    assert len(trips) == passes * (4 * qwen_pack.arch.num_layers + 1)
    assert counts["layer"] == passes * qwen_pack.arch.num_layers
    for name in ("rms_norm", "norm_rows", "rope", "attention", "silu_mul"):
        assert counts.get(name, 0) > 0, name


def test_capture_hidden(qwen_pack: InMemoryPack) -> None:
    """Hidden states of the chosen layers (-1: embeddings); the last layer's rows give the logits."""
    from pgw.engine.lm_head import lm_head_logits
    from pgw.engine.model import rms_norm

    model = build(qwen_pack, False)
    n = qwen_pack.arch.num_layers
    assert model.eagle_capture_layers() == (min(2, n - 1), n // 2, max(n - 3, 0))
    cache = model.new_cache(64)
    ids = _ids(9, 5)
    plain = SequenceInput(ids, model.new_cache(64), all_logits=True)
    want = (-1, 0, n - 1)
    seq = SequenceInput(ids, cache, all_logits=True, capture_hidden=want)
    out, ref = model.forward_batch([seq, plain])
    assert plain.hidden is None
    assert seq.hidden is not None and sorted(seq.hidden) == sorted(want)
    assert torch.equal(out, ref)
    assert torch.equal(seq.hidden[-1], model.embed_tokens[torch.tensor(ids)])
    assert all(t.shape == (9, qwen_pack.arch.hidden_size) for t in seq.hidden.values())
    hn = rms_norm(seq.hidden[n - 1], model.final_norm, model.eps).to(torch.float32)
    assert torch.equal(lm_head_logits(hn, model._head), out)
    # decode: one row each, the captured tensors are not changed by later passes
    keep = seq.hidden[0].clone()
    dec = SequenceInput([4], cache, capture_hidden=(0,))
    model.forward_batch([dec])
    assert dec.hidden is not None and dec.hidden[0].shape == (1, qwen_pack.arch.hidden_size)
    assert torch.equal(seq.hidden[0], keep)


def _local_attention(model: TransformerModel, pack: Any) -> list[str]:
    """Make the model's masked backend compute qkv and o locally (exact plain product)."""
    be = model.backend
    plain = PlainBackend(pack, DEFAULT_RING)
    local_calls: list[str] = []

    def forward_local(calls: list[LinearCall]) -> list[torch.Tensor]:
        local_calls.extend(str(c.id) for c in calls)
        return plain.forward_many(calls)

    be.local_linear = lambda lid: lid.kind in (LinearKind.QKV, LinearKind.O)  # type: ignore[attr-defined]
    be.forward_local = forward_local  # type: ignore[attr-defined]
    return local_calls


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_local_linears_step_past_without_a_hop(qwen_pack: InMemoryPack, mode: str,
                                               reference: dict[str, dict[str, torch.Tensor]]) -> None:
    """qkv/o computed on the gateway: same bits as all-remote and as plain; the provider receives only
    gate_up/down items (half the round trips) and the local linears take no masks."""
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, DEFAULT_RING, qwen_pack.fingerprint)
    model = TransformerModel(qwen_pack, be)
    local = _local_attention(model, qwen_pack)
    layers = qwen_pack.arch.num_layers
    assert model._trip_plan() == (2 * layers, True)
    got = scenario_run(model, qwen_pack, mode, True)
    ref = reference["qwen/plain_sync"]
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert local and {lid for lid, _ in client.received} <= {f"{i}.{k}" for i in range(layers)
                                                            for k in ("gate_up", "down")}
    assert be.stats.round_trips == client.requests
    assert client.requests % (2 * layers) == 0
    res = _res(qwen_pack, 8, 9)
    cache = model.new_cache(16)
    if mode == "sync":
        model.forward_batch([SequenceInput([1, 2], cache, res)])
    else:
        asyncio.run(model.aforward_batch([SequenceInput([1, 2], cache, res)]))
    assert res.remaining(LinearId(0, LinearKind.QKV)) == 8 and res.remaining(LinearId(0, LinearKind.O)) == 8
    assert res.remaining(LinearId(0, LinearKind.GATE_UP)) == 6


def _recorded_run(pack: Any, sync: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, torch.Tensor],
                                                                                   list[tuple[str, torch.Tensor]],
                                                                                   list[int]]:
    client = LocalComputeClient(pack, record=True)
    be = MaskedBackend(pack, None, client, DEFAULT_RING, pack.fingerprint)
    sync_rows: list[int] = []

    def ok(rows: int) -> bool:
        sync_rows.append(rows)
        return True

    if sync:
        be.sync_trip_ok = ok  # type: ignore[attr-defined]
    model = TransformerModel(pack, be)
    model.sync_decode = sync
    out = scenario_run(model, pack, "async", True)
    model.close()
    return out, client.received, sync_rows


def test_sync_decode_loop_same_bits_and_same_provider_bytes(qwen_pack: InMemoryPack,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A decode pass as one plaintext-thread call over synchronous trips: same logits, and the provider
    receives the same byte-identical x_hat items (fixed mask seeds) as with the split-hop path."""
    split_out, split_rx, _ = _recorded_run(qwen_pack, False, monkeypatch)
    sync_out, sync_rx, sync_rows = _recorded_run(qwen_pack, True, monkeypatch)
    assert split_out.keys() == sync_out.keys()
    for k in split_out:
        assert torch.equal(split_out[k], sync_out[k]), k
    # the in-process service records pipelined halves in completion order (they compute concurrently)
    def items(rx: list[tuple[str, torch.Tensor]]) -> list[tuple[str, tuple[int, ...], bytes]]:
        return sorted((lid, tuple(x.shape), x.numpy().tobytes()) for lid, x in rx)

    assert items(split_rx) == items(sync_rx)
    assert sync_rows and max(sync_rows) <= 2 * 8  # asked only for decode-sized passes


def test_sync_decode_only_for_decode_passes(qwen_pack: InMemoryPack) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    model = TransformerModel(qwen_pack, be)
    model.sync_decode = True
    seq1 = [SequenceInput([1], model.new_cache(8))]
    seq9 = [SequenceInput(list(range(9)), model.new_cache(16))]
    assert not model._sync_pass(seq1, be)  # backend without sync_trip_ok
    be.sync_trip_ok = lambda rows: rows <= 4  # type: ignore[attr-defined]
    assert model._sync_pass(seq1, be) and not model._sync_pass(seq9, be)
    assert not model._sync_pass([SequenceInput([1], model.new_cache(8)) for _ in range(5)], be)  # backend says no
    model.sync_decode = False
    assert not model._sync_pass(seq1, be)


def test_sync_decode_env(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    assert build(qwen_pack, False).sync_decode is False
    monkeypatch.setenv("PGW_SYNC_DECODE", "1")
    assert build(qwen_pack, False).sync_decode is True


def main(argv: list[str]) -> None:  # pragma: no cover - golden capture helper
    if "--capture" not in argv:
        raise SystemExit("usage: python -m tests.engine.test_model_hooks --capture")
    import tempfile

    from tests.conftest import _save_tiny

    with tempfile.TemporaryDirectory() as d:
        packs = {"qwen": InMemoryPack.from_hf(_save_tiny(Path(d) / "q", "qwen2", 0), name="tiny-qwen2"),
                 "llama": InMemoryPack.from_hf(_save_tiny(Path(d) / "l", "llama", 1), name="tiny-llama")}
        res = scenarios(packs)
    GOLDEN.write_text(json.dumps({"env_key": env_key(), "digests": digests(res)}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

