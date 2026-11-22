"""Local linears on a gateway GPU (:mod:`pgw.engine.local_gpu`): the backend wiring, on the CPU.

The GPU path is the masked trip with a zero mask: the native prepare kernel (``r = 0``) quantizes into the
K = 32 payload, the device computes ``x_hat @ W8^T mod 2^32``, the native finish (``W r = 0``) range-checks
and rescales. ``_CpuProducts`` below is the same three steps with the product in numpy (wrapping int32, as
the fused ring GEMV), so these tests check on any host that this composition gives the CPU reference's
bits, that the backend's self-check catches a wrong product, and that refusals match. The CUDA graphs
themselves are checked on a GPU by ``benchmarks/local_gpu_bench.py`` and the start-up self-check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import IntegrityError
from pgw.core.native import native_available
from pgw.core.ring import DEFAULT_RING
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LinearCall, LocalComputeClient, MaskedBackend, PlainBackend, TransformerModel
from pgw.engine import backends as B
from tests.engine.test_model_hooks import scenario_run

pytestmark = pytest.mark.skipif(not native_available("trip_finish_into"), reason="needs the native trip kernels")


class _CpuProducts:
    """:class:`pgw.engine.local_gpu.LocalGpuProducts` with the product in numpy (no CUDA)."""

    def __init__(self, device: torch.device, k: int, a: int, cap: int, *, graph_rows: tuple[int, ...],
                 max_n: int, max_m: int, wrong: bool = False, block: int = 0):
        from pgw.core.native import fast

        assert k == 32
        self.fast, self.device, self.k, self.a, self.cap = fast, device, k, a, cap
        self.graph_rows, self.fused_rows = graph_rows, 1
        self.w: dict[str, np.ndarray] = {}
        self.wrong = wrong
        self.calls = 0
        self.prefill_block_rows = block
        self.max_rows_seen = 0

    def add(self, key: str, w8: torch.Tensor) -> None:
        self.w[key] = w8.contiguous().numpy().astype(np.int64)  # (m, n)

    def has(self, key: str) -> bool:
        return key in self.w

    def forward(self, key: str, xf: np.ndarray, wnorm: float, scale: np.ndarray, bias: np.ndarray | None,
                bound: int, out: np.ndarray, sx: np.ndarray) -> int:
        w = self.w[key]
        rows, n = xf.shape
        self.max_rows_seen = max(self.max_rows_seen, rows)
        payload = np.empty(rows * n * 4, dtype=np.uint8)
        self.fast.trip_prepare_into(xf, np.zeros((rows, n), dtype=np.int64), self.k, self.a, self.cap, wnorm,
                                    payload, sx)
        x = payload.view(np.int32).reshape(rows, n).astype(np.int64)
        y = (x @ w.T) & 0xFFFFFFFF  # mod 2^32, as the fused ring GEMV
        if self.wrong:
            y[0, 0] ^= 1
        yb = y.astype(np.uint32).view(np.uint8).reshape(-1)
        status, _ = self.fast.trip_finish_into(yb, np.zeros((rows, w.shape[0]), dtype=np.int64), sx, scale, bias,
                                               self.k, bound, out)
        self.calls += 1
        return int(status)


def _backend(pack: InMemoryPack, mode: str = "attention_down", **kw: Any) -> tuple[MaskedBackend, dict[str, Any]]:
    be = MaskedBackend(pack, None, LocalComputeClient(pack), DEFAULT_RING, pack.fingerprint, local_linears=mode)
    policy: dict[str, Any] = {}
    gp = be._local_gpu_build(torch.device("cpu"), be._local_kinds, 2, policy,
                             lambda *a, **k2: _CpuProducts(*a, **k2, **kw), lambda d, k: None, need_faster=False)
    be._local_gpu = gp
    return be, policy


def test_gpu_path_composition_is_bit_identical(qwen_pack: InMemoryPack) -> None:
    be, policy = _backend(qwen_pack)
    assert be._local_gpu is not None and policy["gpu_graphs_reason"] == "identical outputs"
    assert policy["gpu_graphs_check"]["linears"] == 3 * qwen_pack.arch.num_layers
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    g = torch.Generator().manual_seed(5)
    for lid in (LinearId(0, LinearKind.QKV), LinearId(1, LinearKind.O), LinearId(1, LinearKind.DOWN)):
        n = qwen_pack.load_linear(lid).w8.shape[1]
        calls = [LinearCall(lid, torch.randn(t, n, generator=g) * s) for t, s in ((1, 1.0), (5, 30.0), (3, 1e-3))]
        before = be._local_gpu.calls
        for c, y in zip(calls, be.forward_local(calls), strict=True):
            assert torch.equal(y, plain.forward(c))
        assert be._local_gpu.calls > before  # served by the GPU path (calls of one linear are grouped)
    be.close()


def test_prompt_blocks_are_bit_identical(qwen_pack: InMemoryPack) -> None:
    """``PGW_LOCAL_GPU_BLOCK_ROWS``: a prompt-sized call runs in row blocks (the last one short); same bits."""
    be, policy = _backend(qwen_pack, block=4)
    assert be._local_gpu is not None and policy["gpu_graphs_reason"] == "identical outputs"
    plain = PlainBackend(qwen_pack, DEFAULT_RING)
    g = torch.Generator().manual_seed(9)
    lid = LinearId(1, LinearKind.DOWN)
    n = qwen_pack.load_linear(lid).w8.shape[1]
    x = torch.randn(11, n, generator=g) * 5
    x[9, 3] = 400.0
    before = be._local_gpu.calls
    (y,) = be.forward_local([LinearCall(lid, x)])
    assert torch.equal(y, plain.forward(LinearCall(lid, x)))
    assert be._local_gpu.calls - before == 3 and be._local_gpu.max_rows_seen == 4
    be.close()


def test_gpu_path_in_a_model_run_equals_plain(qwen_pack: InMemoryPack) -> None:
    ref_model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    ref_model.pipeline_min_rows = 16
    try:
        ref = scenario_run(ref_model, qwen_pack, "sync", False)
    finally:
        ref_model.close()
    be, _ = _backend(qwen_pack)
    model = TransformerModel(qwen_pack, be)
    model.pipeline_min_rows = 16
    try:
        got = scenario_run(model, qwen_pack, "sync", True)
    finally:
        model.close()
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert be._local_gpu.calls > 0


def test_self_check_rejects_a_wrong_product(qwen_pack: InMemoryPack) -> None:
    be, policy = _backend(qwen_pack, wrong=True)
    assert be._local_gpu is None and "differ" in policy["gpu_graphs_reason"]
    be.close()


def test_unavailable_and_disabled_keep_the_cpu(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint,
                       local_linears="attention")
    policy: dict[str, Any] = {}
    assert be._local_gpu_build(torch.device("cpu"), be._local_kinds, 2, policy, _CpuProducts,
                               lambda d, k: "no reason", need_faster=False) is None
    assert policy["gpu_graphs_reason"] == "no reason"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert be.configure_local_linears("attention", device="cuda") == B.LOCAL_ATTENTION_KINDS
    assert be._local_gpu is None and be.local_policy["device"] == "cpu"
    assert be.local_policy["gpu_graphs_reason"] == "no CUDA device"
    monkeypatch.setenv("PGW_LOCAL_GPU_GRAPHS", "0")
    be.configure_local_linears("attention", device="cuda")
    assert be.local_policy["gpu_graphs_reason"] == "PGW_LOCAL_GPU_GRAPHS=0"
    be.close()


def test_gpu_path_refuses_out_of_range_like_the_masked_path(qwen_pack: InMemoryPack,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    be, _ = _backend(qwen_pack, mode="attention")
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(2, qwen_pack.arch.hidden_size)
    be.forward_local([LinearCall(lid, x)])
    real = be._local_gpu.forward

    def too_big(key: str, xf: np.ndarray, wnorm: float, scale: np.ndarray, bias: Any, bound: int, out: np.ndarray,
                sx: np.ndarray) -> int:
        real(key, xf, wnorm, scale, bias, bound, out, sx)
        return 2

    monkeypatch.setattr(be._local_gpu, "forward", too_big)
    with pytest.raises(IntegrityError) as ei:
        be.forward_local([LinearCall(lid, x)])
    assert ei.value.details["bound"] == DEFAULT_RING.output_bound(qwen_pack.arch.hidden_size)
    be.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_gpu_graphs_equal_plain(qwen_pack: InMemoryPack) -> None:  # pragma: no cover - GPU only
    ref_model = TransformerModel(qwen_pack, PlainBackend(qwen_pack, DEFAULT_RING))
    ref_model.pipeline_min_rows = 16
    try:
        ref = scenario_run(ref_model, qwen_pack, "sync", False)
    finally:
        ref_model.close()
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), DEFAULT_RING, qwen_pack.fingerprint)
    be.configure_local_linears("attention_down", device="cuda")
    assert be._local_gpu is not None, be.local_policy.get("gpu_graphs_reason")
    model = TransformerModel(qwen_pack, be)
    model.pipeline_min_rows = 16
    try:
        got = scenario_run(model, qwen_pack, "sync", True)
    finally:
        model.close()
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert be._local_gpu.stats["graph_calls"] > 0 and be._local_gpu.stats["eager_calls"] > 0
