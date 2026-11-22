"""Output-row tensor parallel kernels (:mod:`pgw.service.tensor_parallel`) on the fake multi-device CPU backend.

Every test compares the sharded reply byte for byte with the single-device kernel (and the reference).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K32_A12_C27,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    PGWError,
    RingParams,
    ShapeError,
    pack_ring,
    unpack_ring,
)
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.service import BatchScheduler, ComputeKernel, ModelRegistry, ServiceSettings, ShardedKernel, TPGroup
from pgw.service.tensor_parallel import shard_bounds, validate_tp_devices
from tests.service.conftest import asgi_client, expected, make_req, rand_x, running_app

RINGS = (RING_K32_A8, RING_K32_A12_C27, RING_K40_A12, RING_K48_A20)


def _w(m: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)


def _spec(m: int, n: int, kind: LinearKind = LinearKind.GATE_UP) -> LinearSpec:
    return LinearSpec(LinearId(0, kind), m, n)


@pytest.fixture(scope="module")
def group3() -> Any:
    g = TPGroup(["cpu", "cpu", "cpu"])
    yield g
    g.close()


def test_shard_bounds() -> None:
    assert shard_bounds(10, 1) == [(0, 10)]
    for m in (1, 3, 7, 16, 63, 64, 65, 100, 4096 + 8, 29568):
        for n in range(1, 6):
            if m < n:
                with pytest.raises(ConfigError):
                    shard_bounds(m, n)
                continue
            b = shard_bounds(m, n)
            assert b[0][0] == 0 and b[-1][1] == m and len(b) == n
            assert all(b[i][1] == b[i + 1][0] for i in range(n - 1))
            assert all(m1 > m0 for m0, m1 in b)
            if m >= 16 * n:
                assert all(m0 % 16 == 0 for m0, _ in b)
                sizes = [m1 - m0 for m0, m1 in b]
                assert max(sizes) - min(sizes) <= 16
    with pytest.raises(ConfigError):
        shard_bounds(8, 0)


def test_validate_tp_devices() -> None:
    assert [str(d) for d in validate_tp_devices(["cpu", "cpu"])] == ["cpu", "cpu"]
    with pytest.raises(ConfigError):
        validate_tp_devices([])
    with pytest.raises(PGWError):
        validate_tp_devices(["cpu", "bogus-device"])


@pytest.mark.parametrize("tp", [1, 2, 3, 4])
@pytest.mark.parametrize("backend", ["cpu_f64", "auto"])
def test_sharded_matches_single_device_bytes(tp: int, backend: str) -> None:
    m, n = 77, 40  # uneven split, m not a multiple of the alignment
    w = _w(m, n, seed=tp)
    single = ComputeKernel(_spec(m, n), w, device="cpu", backend=backend)
    sk = ShardedKernel(_spec(m, n), w, ["cpu"] * tp, backend=backend)
    assert sk.tp == tp and sk.out_features == m and sk.in_features == n
    assert sk.linear == single.linear and sk.kind == single.kind and sk.backend == single.backend
    assert sum(s.out_features for s in sk.shards) == m
    for ring in RINGS:
        for rows in (0, 1, 2, 5, 17):
            x = rand_x(rows, n, ring, seed=rows + 7 * tp)
            ys, yt = single.run(x, ring), sk.run(x, ring)
            assert yt.dtype == torch.int64 and tuple(yt.shape) == (rows, m)
            assert torch.equal(yt, ys) and torch.equal(yt, expected(x, w, ring))
            assert pack_ring(sk.run_compact(x, ring), ring) == pack_ring(single.run_compact(x, ring), ring)
        # extreme operands: all-ones x and +-128 weights
        x = torch.full((3, n), ring.modulus - 1, dtype=torch.int64)
        assert torch.equal(sk.run(x, ring), single.run(x, ring))
    sk.group.close()


def test_sharded_extreme_weights(group3: TPGroup) -> None:
    m, n = 48, 64
    w = torch.full((m, n), -128, dtype=torch.int8)
    w[::2] = 127
    sk = ShardedKernel(_spec(m, n), w, group3, backend="cpu_f64")
    for ring in RINGS:
        x = torch.full((2, n), ring.modulus - 1, dtype=torch.int64)
        assert torch.equal(sk.run(x, ring), expected(x, w, ring))


def test_shape_errors(group3: TPGroup) -> None:
    sk = ShardedKernel(_spec(32, 16), _w(32, 16), group3, backend="cpu_f64")
    with pytest.raises(ShapeError):
        sk.run(torch.zeros((1, 15), dtype=torch.int64), RING_K40_A12)
    with pytest.raises(ShapeError):
        sk.run(torch.zeros((1, 16), dtype=torch.int32), RING_K40_A12)
    with pytest.raises(ShapeError):
        ShardedKernel(_spec(32, 16), _w(31, 16), group3)
    with pytest.raises(ConfigError):
        ShardedKernel(_spec(2, 16), _w(2, 16), group3)  # fewer rows than shards


def test_maskgen_entry_points_identical(group3: TPGroup) -> None:
    m, n, cols = 50, 24, 5
    w = _w(m, n, seed=3)
    single = ComputeKernel(_spec(m, n), w, device="cpu")
    sk = ShardedKernel(_spec(m, n), w, group3)
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2**63, size=(n, cols), dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    for nbytes in (8, 7, 4):
        bb = b & np.uint64((1 << (8 * nbytes)) - 1) if nbytes < 8 else b
        assert np.array_equal(sk.mulmod64(bb, nbytes), single.mulmod64(bb, nbytes))
    raw = b.astype("<u8").tobytes()
    for q_bits, drop in ((64, 0), (64, 16), (56, 8), (32, 0)):
        data = raw if q_bits == 64 else np.ascontiguousarray(
            b.astype("<u8").view(np.uint8).reshape(n * cols, 8)[:, : q_bits // 8]).tobytes()
        assert sk.mulmod64_wire(data, cols, q_bits, drop) == single.mulmod64_wire(data, cols, q_bits, drop)


def test_failure_isolation_and_recovery(group3: TPGroup) -> None:
    m, n = 64, 32
    w = _w(m, n, seed=9)
    sk = ShardedKernel(_spec(m, n), w, group3, backend="cpu_f64")
    finished: list[int] = []

    def fault(j: int) -> None:
        if j == 1:
            raise RuntimeError("device lost")
        time.sleep(0.05)  # the other shards are still running when shard 1 fails
        finished.append(j)

    sk.fault = fault
    x = rand_x(2, n, RING_K40_A12, seed=1)
    with pytest.raises(PGWError):
        sk.run(x, RING_K40_A12)
    # the call returned only after every other shard finished: nothing runs on behalf of a failed call
    assert sorted(finished) == [0, 2]
    sk.fault = None
    assert torch.equal(sk.run(x, RING_K40_A12), expected(x, w, RING_K40_A12))

    def pgw_fault(j: int) -> None:
        if j == 2:
            raise ShapeError("shard said no")

    sk.fault = pgw_fault
    with pytest.raises(ShapeError):  # PGWErrors keep their code
        sk.run(x, RING_K40_A12)


def test_concurrent_calls_keep_order_and_ownership(group3: TPGroup) -> None:
    m, n = 96, 48
    w = _w(m, n, seed=11)
    sk = ShardedKernel(_spec(m, n), w, group3, backend="cpu_f64")
    errors: list[str] = []

    def worker(t: int) -> None:
        for i in range(15):
            ring = RINGS[(t + i) % len(RINGS)]
            x = rand_x(1 + (i % 3), n, ring, seed=1000 * t + i)
            if not torch.equal(sk.run(x, ring), expected(x, w, ring)):
                errors.append(f"thread {t} call {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []


def test_registry_tp_on_tiny_pack(tiny_pack_dir: Path, pack_weights: dict[str, torch.Tensor]) -> None:
    single = ModelRegistry("cpu", rings=("K40A12", "K32A12C27"), verify="fingerprint")
    tp = ModelRegistry("cpu", rings=("K40A12", "K32A12C27"), verify="fingerprint", tp_devices=["cpu"] * 3)
    ms, mt = single.load(tiny_pack_dir), tp.load(tiny_pack_dir)
    assert mt.fingerprint == ms.fingerprint  # sharding is internal: the pack's identity is unchanged
    assert all(isinstance(k, ShardedKernel) and k.tp == 3 for k in mt.kernels.values())
    assert mt.devices() == ["cpu"]
    assert mt.linear_dims == ms.linear_dims
    summ = tp.device_summary()
    assert summ[0]["linears"] == 3 * len(mt.kernels)
    for i, (lid, w) in enumerate(pack_weights.items()):
        for ring in (RING_K40_A12, RING_K32_A12_C27):
            x = rand_x(1 + i % 4, w.shape[1], ring, seed=i)
            yt = mt.kernel(lid).run_compact(x, ring)
            assert pack_ring(yt, ring) == pack_ring(ms.kernel(lid).run_compact(x, ring), ring)
            assert torch.equal(unpack_ring(pack_ring(yt, ring), (x.shape[0], w.shape[0]), ring), expected(x, w, ring))
    assert tp.check_devices() == {"cpu": True}


async def test_batcher_coalesces_tenants_on_sharded_kernels(tiny_pack_dir: Path,
                                                            pack_weights: dict[str, torch.Tensor]) -> None:
    """Rows of concurrent trips (different tenants) share one sharded GEMM; each gets exactly its own rows."""
    reg = ModelRegistry("cpu", rings=("K40A12",), verify="fingerprint", tp_devices=["cpu"] * 2)
    fp = reg.load(tiny_pack_dir).fingerprint
    sched = BatchScheduler(reg, batch_window_ms=5.0)
    await sched.start()
    try:
        lid, w = next(iter(pack_weights.items()))
        ring = RING_K40_A12
        xs = [rand_x(1 + t % 2, w.shape[1], ring, seed=50 + t) for t in range(6)]
        outs = await asyncio.gather(*(sched.submit_timed(fp, ring, lid, x) for x in xs))
        assert max(t.batch_rows for _, t in outs) > 2  # at least two trips were coalesced
        for x, (y, _t) in zip(xs, outs, strict=True):
            assert torch.equal(y, expected(x, w, ring))
    finally:
        await sched.stop()


async def test_app_end_to_end_with_tp(tiny_pack_dir: Path, pack_fp: str, pack_weights: dict[str, torch.Tensor]) -> None:
    assert ServiceSettings().tp_devices == []  # default off
    s = ServiceSettings(packs=[tiny_pack_dir], device="cpu", verify_packs="fingerprint", allow_unauthenticated=True,
                        warmup=False, tp_devices=["cpu", "cpu", "cpu", "cpu"])
    async with running_app(s) as app:
        c = asgi_client(app)
        await c.connect(pack_fp, "K40A12")
        for ring in (RING_K40_A12, RING_K32_A12_C27):
            items = [(lid, rand_x(1 + i % 3, w.shape[1], ring, seed=i)) for i, (lid, w) in enumerate(pack_weights.items())]
            resp = await c.compute(make_req(pack_fp, ring, items, rid=f"tp-{ring.name}"))
            for (lid, x), res in zip(items, resp.items, strict=True):
                y = unpack_ring(res.y_hat, (x.shape[0], pack_weights[lid].shape[0]), ring)
                assert torch.equal(y, expected(x, pack_weights[lid], ring))
        await c.aclose()


def test_ring_params_roundtrip_sanity() -> None:
    # K32 compact int32 parts from every shard concatenate to the same packed bytes as int64
    sk = ShardedKernel(_spec(40, 8), _w(40, 8), ["cpu", "cpu"])
    ring = RingParams.from_name("K32A12C27")
    x = rand_x(2, 8, ring, seed=4)
    assert pack_ring(sk.run_compact(x, ring), ring) == pack_ring(expected(x, _w(40, 8), ring), ring)
    sk.group.close()
