"""GPU tests of the service trip lane's CUDA-graph replay (chat-speed WP-A1).

Marked ``gpu`` (skipped without CUDA). For Qwen2.5-7B and 0.5B linear shapes at rows 1..8 the lane's
natively launched graphs ``[H2D, memset, GEMV, D2H]`` must give the same ``y_hat`` bytes as the
digit-GEMM path and as the exact CPU product, through the real listener; an injected mismatch in the
startup self-check must disable graph replay (calls then take the kernel path, still exact).

    pytest -q tests/service/test_lane_graph.py
"""

from __future__ import annotations

import secrets
import shutil
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import LinearId, LinearKind, LinearSpec, RingParams, pack_ring, ringmm_reference, unpack_ring
from pgw.service import ModelRegistry, ServiceMetrics, ServiceSettings
from pgw.service.gpu_exec import device_executor
from pgw.service.kernels import ComputeKernel
from pgw.service.lane import (
    REPLY_HEADER,
    LaneManager,
    LaneStatus,
    encode_call,
    encode_open,
    lane_native_available,
    parse_reply,
)
from pgw.service.registry import RegisteredModel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not lane_native_available(), reason="no native lane")]

RING = RingParams.from_name("K32A12C27")
SHAPES = {
    "7b": {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)},
    "0.5b": {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)},
}


def _model(name: str, seed: int) -> tuple[RegisteredModel, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    specs, kernels, weights = [], {}, {}
    for kind, (m, n) in SHAPES[name].items():
        spec = LinearSpec(LinearId(0, LinearKind(kind)), m, n)
        w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
        w[0] = -128  # extreme rows
        w[1] = 127
        specs.append(spec)
        kernels[spec.id.key] = ComputeKernel(spec, w, device="cuda:0", cuda_graphs=False, triton_gemv=True)
        weights[spec.id.key] = w
    fp = (secrets.token_hex(32))
    return RegisteredModel(name, fp, specs, kernels, rings=(RING.name,)), weights


class _Handler:
    """Kernel-path executor for calls without a graph (the service's ``lane_compute``)."""

    def __init__(self, reg: ModelRegistry):
        self.reg = reg
        self.lane_admission: Any = None
        self.draining = False
        self.calls = 0

    async def lane_compute(self, fp: str, linear: str, ring: RingParams, rows: int, body: bytes, tenant: str) -> bytes:
        k = self.reg.get(fp).kernel(linear)
        self.calls += 1
        return pack_ring(k.run_compact(unpack_ring(body, (rows, k.in_features), ring), ring), ring)


@pytest.fixture(scope="module")
def models() -> dict[str, tuple[RegisteredModel, dict[str, torch.Tensor]]]:
    return {name: _model(name, i) for i, name in enumerate(SHAPES)}


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pgwl"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


async def _manager(models: dict[str, Any], sock_dir: Path, monkeypatch: pytest.MonkeyPatch,
                   inject: bool = False) -> tuple[LaneManager, _Handler]:
    import pgw.service.lane as lane_mod

    reg = ModelRegistry("cuda:0", rings=(RING.name,))
    for m, _w in models.values():
        reg.register(m)
    handler = _Handler(reg)
    state = SimpleNamespace(registry=reg, handler=handler, metrics=ServiceMetrics(enabled=False))
    s = ServiceSettings(device="cuda:0", allow_unauthenticated=True, rings=[RING.name], lane="uds",
                        lane_socket=sock_dir / "lane.sock")
    if inject:
        real = lane_mod.capture_lane_graphs
        monkeypatch.setattr(lane_mod, "capture_lane_graphs", lambda mgr: real(mgr, inject_mismatch=True))
    mgr = LaneManager(s, state)
    assert await mgr.start()
    return mgr, handler


def _call(path: str, ticket: bytes, calls: list[tuple[int, int, bytes]]) -> list[tuple[Any, bytes]]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(path)

    def recv(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            c = s.recv(n - len(buf))
            assert c, "lane closed"
            buf += c
        return bytes(buf)

    def reply() -> tuple[Any, bytes]:
        h = parse_reply(recv(REPLY_HEADER.size))
        return h, recv(h.body_len)

    s.sendall(encode_open(ticket))
    assert reply()[0].status == LaneStatus.OK
    out = []
    for linear, rows, body in calls:
        s.sendall(encode_call(secrets.token_bytes(16), linear, rows, 0, body))
        out.append(reply())
    s.close()
    return out


def _x(rows: int, n: int, seed: int) -> torch.Tensor:
    x = torch.from_numpy(np.random.default_rng(seed).integers(0, 1 << 32, size=(rows, n), dtype=np.int64))
    x[0, : n // 2] = (1 << 32) - 1
    return x


async def test_graph_replay_equals_digit_path_all_shapes(models: dict[str, Any], sock_dir: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    mgr, handler = await _manager(models, sock_dir, monkeypatch)
    try:
        rep = mgr.graph_report
        assert rep.disabled is None, rep
        assert rep.captured == rep.checked == 2 * 4 * 8 and mgr.svc.graph_count == 64
        # the Python path and the lane share one device lock
        assert type(device_executor("cuda:0").lock).__name__ == "LaneDeviceLock"
        ex = device_executor("cuda:0")
        for name, (model, weights) in models.items():
            info = mgr.open_ticket(None, model.fingerprint)
            calls, want = [], []
            for li, lin in enumerate(info["linears"]):
                k = model.kernel(lin)
                for rows in range(1, 9):
                    x = _x(rows, k.in_features, li * 10 + rows)
                    calls.append((li, rows, pack_ring(x, RING)))
                    ref = ex.run(x, 32, k._pw._w_dev, k._pw._rowsum(), k.out_features, graphs=None)
                    want.append(pack_ring(ref, RING))
                    if rows in (1, 8) and name == "0.5b":  # exact CPU product too
                        exact = torch.from_numpy(ringmm_reference(x, weights[lin], 32))
                        assert want[-1] == pack_ring(exact, RING)
            got = await asyncio.to_thread(_call, info["socket"], bytes.fromhex(info["ticket"]), calls)
            for (h, body), w in zip(got, want, strict=True):
                assert h.status == LaneStatus.OK
                assert body == w
        assert handler.calls == 0  # every call was a native graph replay
        d = mgr.drain_metrics()
        assert any(b == "lane_cuda_graph" for _k, b, *_ in d["kernel"])
    finally:
        await mgr.stop()


async def test_injected_mismatch_disables_graphs(models: dict[str, Any], sock_dir: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    mgr, handler = await _manager(models, sock_dir, monkeypatch, inject=True)
    try:
        assert mgr.graph_report.disabled and "differs" in mgr.graph_report.disabled
        assert mgr.svc.graphs_disabled_reason
        model, weights = models["0.5b"]
        info = mgr.open_ticket(None, model.fingerprint)
        lin = info["linears"][0]
        x = _x(2, model.kernel(lin).in_features, 3)
        (h, body), = await asyncio.to_thread(_call, info["socket"], bytes.fromhex(info["ticket"]),
                                             [(0, 2, pack_ring(x, RING))])
        assert h.status == LaneStatus.OK
        assert body == pack_ring(torch.from_numpy(ringmm_reference(x, weights[lin], 32)), RING)
        assert handler.calls == 1  # served by the kernel path, not a graph
    finally:
        await mgr.stop()
