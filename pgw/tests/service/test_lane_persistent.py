"""Persistent trip worker protocol, tensor-parallel gather and multi-tenant coalescing through the native lane, on CPU.

The resident CUDA kernel is replaced by :class:`pgw.service.persistent.EmulatedWorker` (same control block,
same buffers, same order of words); the host side is the lane's real native executor
(``rust/pgw-lane/src/persistent.rs``) and coalescer (``coalesce.rs``). Every reply is compared byte for byte
with the exact product and with the stream path.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import RingParams, pack_ring
from pgw.core.native import fast
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.service import ComputeKernel, ServiceSettings, ShardedKernel
from pgw.service.lane import LaneManager, LaneStatus, lane_native_available
from pgw.service.persistent import (
    CTRL_BYTES,
    KERNEL_SOURCE,
    MAX_ROWS,
    PersistentGroup,
    exec_mode,
    start_lane_persistent,
)

from .conftest import TOKEN_A, TOKEN_B, asgi_client, expected, make_req, rand_x, running_app
from .test_lane_server import Lane, _open

K32 = RingParams.from_name("K32A8")

needs_native = pytest.mark.skipif(
    not lane_native_available() or not hasattr(getattr(fast, "LaneService", None), "register_persistent_group"),
    reason="pgw_fast without the persistent lane executor (rebuild it)",
)


def _w(m: int, n: int, seed: int) -> torch.Tensor:
    return torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(seed))


def _kernels(tp: int) -> list[Any]:
    dims = [(48, 32), (37, 20), (16, 64)]
    out: list[Any] = []
    for i, (m, n) in enumerate(dims):
        spec = LinearSpec(LinearId(0, list(LinearKind)[i]), m, n)
        w = _w(m, n, i)
        out.append(ComputeKernel(spec, w, device="cpu") if tp == 0 else ShardedKernel(spec, w, ["cpu"] * tp))
    return out


def test_defaults_off_and_kernel_source_sane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_SERVICE_EXEC", raising=False)
    assert exec_mode() == "graph"
    monkeypatch.setenv("PGW_SERVICE_EXEC", "persistent")
    assert exec_mode() == "persistent"
    monkeypatch.setenv("PGW_SERVICE_EXEC", "bogus")
    with pytest.raises(Exception, match="PGW_SERVICE_EXEC"):
        exec_mode()
    # the CUDA source names the same control words as the host side (offsets in u32 words)
    assert "#define C_DONE 32" in KERNEL_SOURCE and "#define C_STATUS 33" in KERNEL_SOURCE
    assert "#define C_LINEAR 1" in KERNEL_SOURCE and "#define C_ROWS 2" in KERNEL_SOURCE
    assert CTRL_BYTES == 256 and MAX_ROWS == 4
    assert KERNEL_SOURCE.count("{") == KERNEL_SOURCE.count("}")


def test_coalesce_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service.lane import _coalesce_wanted

    monkeypatch.delenv("PGW_SERVICE_COALESCE", raising=False)
    assert _coalesce_wanted() is False  # default off
    monkeypatch.setenv("PGW_SERVICE_COALESCE", "1")
    assert _coalesce_wanted() is True
    monkeypatch.setenv("PGW_SERVICE_COALESCE", "maybe")
    with pytest.raises(Exception, match="PGW_SERVICE_COALESCE"):
        _coalesce_wanted()


@pytest.mark.parametrize("tp", [0, 1, 3])
def test_emulated_group_matches_kernels(tp: int) -> None:
    kerns = _kernels(tp)
    g = PersistentGroup(list(range(max(tp, 1))), emulate=True)
    idx = [g.add_kernel(k) for k in kerns]
    assert idx == [0, 1, 2] and g.add_kernel(kerns[1]) == 1
    g.start()
    try:
        for k, t in zip(kerns, idx, strict=True):
            for rows in range(1, MAX_ROWS + 1):
                x = rand_x(rows, k.in_features, K32, seed=rows * 7 + t)
                x[0] = K32.modulus - 1
                got = g.run(t, pack_ring(x, K32), rows)
                assert got == pack_ring(k.run_compact(x, K32), K32)
        with pytest.raises(ValueError):
            g.run(0, b"\0" * 8, 1)
        g.workers[-1].fail = True
        with pytest.raises(RuntimeError):
            g.run(0, pack_ring(rand_x(1, kerns[0].in_features, K32), K32), 1)
        g.workers[-1].fail = False
        g.workers[0].stall = True
        with pytest.raises(TimeoutError):
            g.run(0, pack_ring(rand_x(1, kerns[0].in_features, K32), K32), 1, timeout_s=0.05)
    finally:
        g.stop()


# ---- through the native lane -----------------------------------------------------------------------
@pytest.fixture
def sock_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pgwp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _settings(tiny_pack_dir: Path, sock_dir: Path, tokens: Path | None = None, **kw: Any) -> ServiceSettings:
    base: dict[str, Any] = {
        "packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "fingerprint", "warmup": False,
        "rings": ["K32A8", "K40A12"], "lane": "uds", "lane_socket": sock_dir / "lane.sock",
    }
    if tokens is None:
        base["allow_unauthenticated"] = True
    else:
        base["tokens_file"] = tokens
    base.update(kw)
    return ServiceSettings(**base)


@pytest.fixture
def persistent_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the lane with the persistent executor on CPU: pretend a GPU exists (graph capture then turns
    itself off: no CUDA), the workers are started by the test with ``emulate=True``."""
    monkeypatch.setenv("PGW_SERVICE_EXEC", "persistent")
    monkeypatch.setattr(LaneManager, "_any_cuda", lambda self: True)


async def _lane(app: Any, fp: str, token: str | None = None) -> tuple[Lane, dict[str, Any]]:
    status, info = await _open(app, fp, token)
    assert status == 200, info
    lane = await asyncio.to_thread(Lane, info["socket"], bytes.fromhex(info["ticket"]))
    assert lane.open_status == LaneStatus.OK
    return lane, info


def _trips(mgr: LaneManager) -> int:
    return sum(w.trips for g in mgr._persistent for w in g.workers)


@needs_native
@pytest.mark.parametrize("tp", [0, 3])
async def test_lane_persistent_is_bit_identical(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str,
                                                pack_weights: dict[str, torch.Tensor], persistent_lane: None,
                                                tp: int) -> None:
    kw: dict[str, Any] = {"tp_devices": ["cpu"] * tp} if tp else {}
    async with running_app(_settings(tiny_pack_dir, sock_dir, **kw)) as app:
        mgr: LaneManager = app.state.pgw.lane
        assert mgr.svc.persistent_count == 0  # nothing on CPU until the emulated workers start
        rep = await asyncio.to_thread(start_lane_persistent, mgr, emulate=True)
        assert rep.disabled is None, rep.disabled
        assert rep.groups == 1 and rep.linears == len(pack_weights) and rep.checked == rep.linears
        assert len(mgr._persistent[0].workers) == max(tp, 1)
        mgr.svc.set_persistent_timeout_us(60_000_000)  # emulated workers on a loaded CI box: no spurious disable
        lane, info = await _lane(app, pack_fp)
        client = asgi_client(app)
        try:
            dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
            for li, name in enumerate(info["linears"][:4]):
                _m, n = dims[name]
                for rows in (1, 2, 4):
                    x = rand_x(rows, n, K32, seed=li * 10 + rows)
                    before = _trips(mgr)
                    h, out = await asyncio.to_thread(lane.call, li, rows, 0, pack_ring(x, K32))
                    assert h.status == LaneStatus.OK
                    assert _trips(mgr) == before + max(tp, 1)  # served by the workers (one trip per shard)
                    assert out == pack_ring(expected(x, pack_weights[name], K32), K32)
                    resp = await client.compute(make_req(pack_fp, K32, [(name, x)]))
                    assert out == resp.items[0].y_hat
                # rows above the persistent cap and other rings take the kernel path, same bytes
                x = rand_x(6, n, K32, seed=li)
                before = _trips(mgr)
                h, out = await asyncio.to_thread(lane.call, li, 6, 0, pack_ring(x, K32))
                assert h.status == LaneStatus.OK and _trips(mgr) == before
                assert out == pack_ring(expected(x, pack_weights[name], K32), K32)
            d = mgr.drain_metrics()
            assert any(b == "lane_persistent" for _k, b, *_ in d["kernel"])
        finally:
            await client.aclose()
            lane.close()


@needs_native
async def test_lane_persistent_mismatch_and_stall_disable(tiny_pack_dir: Path, sock_dir: Path, pack_fp: str,
                                                          pack_weights: dict[str, torch.Tensor],
                                                          persistent_lane: None) -> None:
    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        mgr: LaneManager = app.state.pgw.lane
        rep = await asyncio.to_thread(start_lane_persistent, mgr, emulate=True, inject_mismatch=True)
        assert rep.disabled is not None and "differs" in rep.disabled
        assert mgr.svc.persistent_disabled_reason == rep.disabled
        lane, info = await _lane(app, pack_fp)
        name = info["linears"][0]
        _m, n = dict(app.state.pgw.registry.get(pack_fp).linear_dims)[name]
        x = rand_x(1, n, K32, seed=3)
        h, out = await asyncio.to_thread(lane.call, 0, 1, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.OK and out == pack_ring(expected(x, pack_weights[name], K32), K32)
        lane.close()

    async with running_app(_settings(tiny_pack_dir, sock_dir)) as app:
        mgr = app.state.pgw.lane
        rep = await asyncio.to_thread(start_lane_persistent, mgr, emulate=True, rows_list=[1])
        assert rep.disabled is None
        mgr.svc.set_persistent_timeout_us(30_000)
        for g in mgr._persistent:
            for w in g.workers:
                w.stall = True
        lane, info = await _lane(app, pack_fp)
        x = rand_x(1, n, K32, seed=4)
        h, _ = await asyncio.to_thread(lane.call, 0, 1, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.INTERNAL and h.status.fallback  # the client resends on the stream channel
        assert "did not answer" in mgr.svc.persistent_disabled_reason
        h, out = await asyncio.to_thread(lane.call, 0, 1, 0, pack_ring(x, K32))  # kernel path from now on
        assert h.status == LaneStatus.OK and out == pack_ring(expected(x, pack_weights[name], K32), K32)
        lane.close()


@needs_native
async def test_lane_coalesces_tenants_without_mixing(tiny_pack_dir: Path, sock_dir: Path, tokens_file: Path,
                                                     pack_fp: str, pack_weights: dict[str, torch.Tensor],
                                                     persistent_lane: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent trips of two tenants on one linear share device calls; each reply holds only its own rows,
    per-tenant accounting is per call, and a lone trip still runs alone."""
    monkeypatch.setenv("PGW_SERVICE_COALESCE", "1")
    async with running_app(_settings(tiny_pack_dir, sock_dir, tokens_file, tp_devices=["cpu", "cpu"])) as app:
        mgr: LaneManager = app.state.pgw.lane
        # the lane's own drain loop (every DRAIN_INTERVAL_S) also drains the counters: keep every drain, so the
        # accounting check below sees all rows whatever the timing (it failed when the box was busy)
        drained: list[dict[str, Any]] = []
        first_drain = mgr.drain_metrics

        def recording_drain() -> dict[str, Any]:
            d = first_drain()
            drained.append(d)
            return d

        monkeypatch.setattr(mgr, "drain_metrics", recording_drain)
        rep = await asyncio.to_thread(start_lane_persistent, mgr, emulate=True, rows_list=[1, 2])
        assert rep.disabled is None
        lanes = [(await _lane(app, pack_fp, tok))[0] for tok in (TOKEN_A, TOKEN_B, TOKEN_A, TOKEN_B)]
        name = (await _open(app, pack_fp, TOKEN_A))[1]["linears"][1]
        _m, n = dict(app.state.pgw.registry.get(pack_fp).linear_dims)[name]
        # lone trip: alone, never merged
        x = rand_x(1, n, K32, seed=99)
        h, out = await asyncio.to_thread(lanes[0].call, 1, 1, 0, pack_ring(x, K32))
        assert h.status == LaneStatus.OK and out == pack_ring(expected(x, pack_weights[name], K32), K32)
        st0 = mgr.svc.coalesce_stats()
        assert st0["alone"] == 1 and st0["merged_calls"] == 0
        for g in mgr._persistent:
            for w in g.workers:
                w.delay_s = 0.02  # trips overlap
        errors: list[str] = []
        calls_per_lane = 6

        def client(ix: int) -> None:
            rng = np.random.default_rng(ix)
            for i in range(calls_per_lane):
                x = torch.from_numpy(rng.integers(0, K32.modulus, size=(1, n), dtype=np.int64))
                h, out = lanes[ix].call(1, 1, 0, pack_ring(x, K32))
                if h.status != LaneStatus.OK or out != pack_ring(expected(x, pack_weights[name], K32), K32):
                    errors.append(f"lane {ix} call {i}: {h.status}")

        threads = [threading.Thread(target=client, args=(i,)) for i in range(len(lanes))]
        def run_all() -> None:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        await asyncio.to_thread(run_all)
        assert errors == []
        st = mgr.svc.coalesce_stats()
        assert st["merged_batches"] >= 1 and 2 <= st["max_rows"] <= 8, st
        assert st["merged_calls"] + st["alone"] - st0["alone"] + st["retried"] >= len(lanes) * calls_per_lane
        mgr.drain_metrics()
        rows: dict[tuple[str, str], int] = {}
        for d in drained:
            for t, k, r in d.get("rows", ()):
                rows[(t, k)] = rows.get((t, k), 0) + int(r)
        kind = name.split(".", 1)[1]
        assert rows[("tenant-a", kind)] == 1 + 2 * calls_per_lane  # per-tenant accounting per call, unchanged
        assert rows[("tenant-b", kind)] == 2 * calls_per_lane
        for ln in lanes:
            ln.close()
