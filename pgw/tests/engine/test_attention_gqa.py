"""GQA attention: SDPA ``enable_gqa`` only where it is bit-identical to ``repeat_interleave``."""

from __future__ import annotations

import pytest
import torch

from pgw.core import DEFAULT_RING
from pgw.engine import InMemoryPack, PlainBackend
from pgw.engine import model as model_mod
from pgw.engine.model import GQA_CHECK_LENGTHS, SequenceInput, TransformerModel, _attend, gqa_self_check

needs_gqa = pytest.mark.skipif(not model_mod._sdpa_gqa_supported(), reason="torch < 2.5 has no enable_gqa")


@needs_gqa
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(("heads", "kv_heads", "head_dim"), [(4, 2, 16), (28, 4, 128), (14, 2, 64)])
def test_gqa_equality_grid(threads: int, heads: int, kv_heads: int, head_dim: int) -> None:
    """Both forms over T=1 / T>1 (causal), the plan's lengths, cache-like views and a serial tail."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        groups = heads // kv_heads
        scale = head_dim**-0.5
        gen = torch.Generator().manual_seed(heads * 1000 + threads)
        cap = max(GQA_CHECK_LENGTHS) + 8
        kbuf = torch.randn((2, cap, kv_heads, head_dim), generator=gen)
        vbuf = torch.randn((2, cap, kv_heads, head_dim), generator=gen)
        for length in GQA_CHECK_LENGTHS:
            k, v = kbuf[0, :length], vbuf[0, :length]  # a view of a bigger buffer, as KVCache.append returns
            for t in sorted({1, min(length, 7), length}):
                q = torch.randn((t, heads, head_dim), generator=gen) * 3
                a = _attend(q, k, v, length - t, scale, groups, False)
                b = _attend(q, k, v, length - t, scale, groups, True)
                assert torch.equal(a, b), (length, t)
            # serial tail: each row alone over its own prefix (speculative verify)
            q = torch.randn((min(length, 3), heads, head_dim), generator=gen)
            st0 = length - q.shape[0]
            for j in range(q.shape[0]):
                kj, vj = k[: st0 + j + 1], v[: st0 + j + 1]
                assert torch.equal(_attend(q[j : j + 1], kj, vj, st0 + j, scale, groups, False),
                                   _attend(q[j : j + 1], kj, vj, st0 + j, scale, groups, True))
        assert gqa_self_check(heads, kv_heads, head_dim, torch.float32, "cpu", scale)
    finally:
        torch.set_num_threads(old)


def test_gqa_form_matches_old_form_through_the_model(qwen_pack: InMemoryPack) -> None:
    """Prefill, decode and a serial-tail verify pass give the same logits with either form."""
    on = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    off = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    off.gqa_sdpa = False
    assert not off._use_gqa()
    if model_mod._sdpa_gqa_supported():
        assert on._use_gqa()  # the self-check passes for the tiny shapes on this CPU
    ids = [int(t) for t in torch.randint(0, 250, (37,), generator=torch.Generator().manual_seed(5))]
    outs = []
    for m in (on, off):
        cache = m.new_cache(64)
        prefill = m.forward_batch([SequenceInput(ids, cache, all_logits=True)])[0]
        dec = m.forward_batch([SequenceInput([9], cache)])[0]
        ver = m.forward_batch([SequenceInput([3, 4, 5], cache, last_logits=3, serial_tail=3)])[0]
        outs.append((prefill, dec, [ver[i] for i in range(3)]))
    (p1, d1, v1), (p2, d2, v2) = outs
    assert torch.equal(p1, p2)
    assert torch.equal(d1, d2)
    assert all(torch.equal(a, b) for a, b in zip(v1, v2, strict=True))


def test_failed_self_check_keeps_repeat_interleave(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(model_mod, "_gqa_verdicts", {})
    monkeypatch.setattr(model_mod, "gqa_self_check", lambda *a, **k: False)
    used: list[bool] = []
    real = model_mod._attend

    def spy(*args: object) -> torch.Tensor:
        used.append(bool(args[-1]))
        return real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(model_mod, "_attend", spy)
    with caplog.at_level("WARNING", logger="pgw.engine.model"):
        m = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    assert not m._use_gqa()
    assert any("not bit-identical" in r.getMessage() for r in caplog.records)
    m.forward_batch([SequenceInput([1, 2, 3], m.new_cache(16))])
    assert used and not any(used)


def test_self_check_is_cached_per_thread_count(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_mod, "_gqa_verdicts", {})
    calls: list[int] = []

    def fake(*args: object, **kw: object) -> bool:
        calls.append(torch.get_num_threads())
        return True

    monkeypatch.setattr(model_mod, "gqa_self_check", fake)
    m = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))  # same key: cached
    assert len(calls) == 1
    old = torch.get_num_threads()
    torch.set_num_threads(1 if old != 1 else 2)
    try:
        m._use_gqa()
        assert len(calls) == 2  # a new thread count is checked again
    finally:
        torch.set_num_threads(old)


def test_self_check_rejects_a_differing_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GQA form that differs in the last bit (or raises) is rejected."""
    real = torch.nn.functional.scaled_dot_product_attention

    def off_by_ulp(*args: object, **kw: object) -> torch.Tensor:
        out = real(*args, **kw)  # type: ignore[arg-type]
        return torch.nextafter(out, out + 1) if kw.get("enable_gqa") else out

    monkeypatch.setattr(model_mod.F, "scaled_dot_product_attention", off_by_ulp)
    if model_mod._sdpa_gqa_supported():
        assert not gqa_self_check(4, 2, 16, torch.float32, "cpu", 0.25)

    def raises(*args: object, **kw: object) -> torch.Tensor:
        if kw.get("enable_gqa"):
            raise RuntimeError("unsupported")
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(model_mod.F, "scaled_dot_product_attention", raises)
    assert not gqa_self_check(4, 2, 16, torch.float32, "cpu", 0.25)
    assert gqa_self_check(4, 4, 16, torch.float32, "cpu", 0.25)  # no groups: nothing to check


def test_lm_head_property_on_another_device(qwen_pack: InMemoryPack) -> None:
    """A head kept on the CPU for a non-CPU model reads as a float32 copy on the model's device (so
    callers building their own GPU product keep working); assigning replaces the head."""
    m = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    assert m.lm_head is m._head
    m.device = torch.device("meta")  # stands in for cuda: no data is touched
    view = m.lm_head
    assert view.device.type == "meta" and view.dtype == torch.float32 and m._head.device.type == "cpu"
    new = torch.zeros((3, 4), device="meta", dtype=torch.float16)
    m.lm_head = new
    assert m.lm_head is new and m._head is new


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_model_keeps_head_on_cpu_and_matches(qwen_pack: InMemoryPack) -> None:  # pragma: no cover - GPU
    from pgw.engine.lm_head import native_lm_head

    cpu = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    gpu = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING, device="cuda"), device="cuda")
    if native_lm_head():
        assert gpu._head.device.type == "cpu"
    ids = [1, 2, 3, 4]
    a = cpu.forward_batch([SequenceInput(ids, cpu.new_cache(16))])[0]
    b = gpu.forward_batch([SequenceInput(ids, gpu.new_cache(16))])[0]
    assert a.shape == b.shape and b.device.type == ("cpu" if native_lm_head() else "cuda")
