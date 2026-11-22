"""The fixed-order LM head as a Triton kernel (:mod:`pgw.engine.lm_head_gpu`), checked on the CPU.

Under ``TRITON_INTERPRET=1`` the kernel runs with numpy float32 operations, which are correctly rounded like
the PTX ``mul.rn``/``add.rn`` of the device build, so these tests check the summation order (chunks, tail,
tree) against the native CPU kernel bit for bit. On a GPU the gateway repeats the check at start-up
(:meth:`GpuLmHead.self_check`) before using it.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

pytest.importorskip("triton")


@pytest.fixture
def mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    import pgw.engine.lm_head_gpu as m

    m = importlib.reload(m)  # triton.jit picks the interpreter when it wraps the kernel
    yield m
    monkeypatch.delenv("TRITON_INTERPRET")
    importlib.reload(m)


@pytest.mark.parametrize(("v", "d"), [(200, 128), (130, 100), (70, 20)])
def test_interpreted_kernel_equals_the_cpu_kernel(mod: Any, v: int, d: int) -> None:
    from pgw.engine.lm_head import lm_head_logits

    g = torch.Generator().manual_seed(v * d)
    w = (torch.randn(v, d, generator=g) * 0.05).to(torch.bfloat16)
    head = mod.GpuLmHead(w, torch.device("cpu"), block_v=64, max_rows=3)
    assert head.self_check(w, rows=2) is None
    h = torch.randn(5, d, generator=g)  # more rows than max_rows: two launches
    assert torch.equal(head.logits(h).view(torch.int32), lm_head_logits(h, w).view(torch.int32))


def test_self_check_catches_a_different_order(mod: Any) -> None:
    w = (torch.randn(64, 48) * 0.1).to(torch.bfloat16)
    head = mod.GpuLmHead(w, torch.device("cpu"), block_v=32, max_rows=2)
    other = (w.to(torch.float32) * 1.05).to(torch.bfloat16)
    assert head.self_check(other) is not None


def test_attached_head_routes_lm_head_logits(mod: Any) -> None:
    from pgw.engine import lm_head as L

    w = (torch.randn(50, 32) * 0.1).to(torch.bfloat16)

    class Spy:
        calls = 0

        def logits(self, h: torch.Tensor) -> torch.Tensor:
            Spy.calls += 1
            return L.lm_head_reference(h, w)

    h = torch.randn(2, 32)
    L.attach_device_head(w, Spy())
    try:
        assert torch.equal(L.lm_head_logits(h, w), L.lm_head_reference(h, w))
        assert Spy.calls == 1
        assert L.device_head(w.clone()) is None
    finally:
        L.attach_device_head(w, None)
    L.lm_head_logits(h, w)
    assert Spy.calls == 1


def test_enable_without_cuda_keeps_the_cpu(mod: Any) -> None:
    class M:
        _head = torch.zeros(4, 16, dtype=torch.bfloat16)
        lm_head_fn = None

    pol = mod.enable_gpu_lm_head(M(), "cuda")
    assert pol["device"] == "cpu" and pol["reason"]
