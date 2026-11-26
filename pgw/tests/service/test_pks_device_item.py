"""pks-v3 provider hint recomputed per item on the device (``pgw.service.pks_gpu.item``, ``PKSProvider`` policy).

The torch path runs on CPU here (the same code runs on CUDA). Every reply is compared bit for bit with the host path
(GEMM, :func:`pgw.hemask.pks.switch_hint`, the CPU key switch) and decrypted by the customer (``pks_native.Customer``)
to ``W8 r mod 2^32``; the device memory of one item is measured (``AllocPeak``) against its model.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from pgw.core.ringmm import cpu_int8_available
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks import NR, PACK, RANK, D, PKSParams
from pgw.hemask.wire import V3_Y_BITS, pack_bits
from pgw.service import pks_gpu
from pgw.service.pks_provider import PKSProvider, _device_operands, hint_mode, pack_ksk
from tests.core.alloc_tracker import AllocPeak

PAR = PKSParams()
M56 = np.uint64((1 << 56) - 1)
CPU = torch.device("cpu")
INT8 = pytest.mark.skipif(not cpu_int8_available(), reason="needs exact CPU int8 GEMMs (torch._int_mm)")


def _weight(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w[0] = -128  # extreme rows: the int8 GEMM's worst partial sums
    w[-1] = 127
    return w


def _const(w8: np.ndarray, rows: int | None = None) -> torch.Tensor:
    scale = (128 * sum(1 << (8 * d) for d in range(7))) % (1 << 64)
    cu = w8.astype(np.int64).sum(1).view(np.uint64) * np.uint64(scale)
    c = cu.view(np.int64).reshape(-1, 1)
    if rows is not None and rows > c.shape[0]:
        c = np.concatenate([c, np.zeros((rows - c.shape[0], 1), dtype=np.int64)])
    return torch.from_numpy(c.copy())


def _host(w8: np.ndarray, a_seed: bytes, linear: str, b: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """The provider's host path: hint GEMM, host switch, the item's GEMM, CPU key switch."""
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, w8.shape[1], linear))
    hs = pks.switch_hint(matmul_mod64(w8, a) & M56, PAR)
    return pn.keyswitch(hs, ops, matmul_mod64(w8, b) & M56)


def _ops(groups: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                     for _ in range(groups)]).astype(np.uint32)


def test_a_columns_equal_host_expansion() -> None:
    ring = pks.public_ring_rows(b"r" * 32, 48, "x.q")
    src = pks_gpu._a_source(torch.from_numpy(ring.view(np.int64)))
    got = torch.cat([pks_gpu.a_columns(src, l0, l0 + 2) for l0 in range(0, RANK, 2)], dim=1)
    assert np.array_equal(got.numpy().view(np.uint64), pks.a_mlwe_rows(ring))


def test_tns_equal_reference() -> None:
    rng = np.random.default_rng(3)
    for p in pks.QP:
        x = rng.integers(0, p, (3, NR), dtype=np.uint64)
        z, zi, ninv = pks_gpu._dev_tables(p, CPU)
        nt = pks.Ntt.of(p)
        xt = torch.from_numpy(x.view(np.int64))
        assert np.array_equal(pks_gpu._fwd(xt, p, z).numpy().view(np.uint64), nt.fwd(x))
        assert np.array_equal(pks_gpu._inv(xt, p, zi, ninv).numpy().view(np.uint64), nt.inv(x))
        assert np.array_equal(xt.numpy().view(np.uint64), x)  # inputs untouched


@pytest.mark.parametrize(("m", "n", "groups", "pad", "rows"), [
    (64, 48, 1, False, None),  # one row block, unpadded weight (the CPU kernel's)
    (96, 80, 2, True, 32),  # three row blocks of one reply ciphertext, zero-padded weight (the CUDA kernel's)
])
def test_item_equals_host_path(m: int, n: int, groups: int, pad: bool, rows: int | None) -> None:
    w8 = _weight(m, n, 7 + m)
    rng = np.random.default_rng(m)
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    b[0] = (1 << 56) - 1
    ops = _ops(groups, 9)
    a_seed, linear = bytes(range(32)), "3.gate_up"
    if pad:
        mp, np_ = m + 32, -(-n // 32) * 32 + 32
        w = torch.zeros((mp, np_), dtype=torch.int8)
        w[:m, :n] = torch.from_numpy(w8)
        const = _const(w8, mp)
    else:
        w, const = torch.from_numpy(w8), _const(w8)
    work = None if rows is None else rows * pks_gpu._row_bytes(w.shape[1], D * groups, groups)
    if rows is not None:
        assert pks_gpu.item_rows(m, w.shape[1], D * groups, groups, work) == rows
    got = pks_gpu.item(w, const, m, n, pks_gpu.ring_rows(a_seed, n, linear, CPU), b, ops, work_bytes=work)
    assert got.shape == (groups, m // PACK, NR) and got.dtype == np.uint64
    assert np.array_equal(got, _host(w8, a_seed, linear, b, ops))


@pytest.mark.parametrize("native", [True, False])
def test_host_switch_equals_reference(monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    from pgw.service.pks_provider import switch_hint_host

    if not native:
        monkeypatch.setenv("PGW_PKS_NATIVE", "0")
    elif not pn.available():
        pytest.skip("pgw_pks not built")
    rng = np.random.default_rng(2)
    h = rng.integers(0, 1 << 63, (96, 4096), dtype=np.uint64) * np.uint64(2) + np.uint64(1)  # bits above 2^56 too
    h[0, :4] = [0, (1 << 56) - 1, 1 << 55, (1 << 64) - 1]
    assert np.array_equal(switch_hint_host(h), pks.switch_hint(h & M56, PAR))


def test_item_self_test_passes_on_cpu() -> None:
    pks_gpu.item_self_test(CPU)


def test_item_memory_within_model() -> None:
    """Measured peak tensor bytes of one item (CPU allocations of the same torch code) stay within
    :func:`pks_gpu.item_bytes`, and the row blocks bound it: a smaller work budget lowers the peak."""
    m, n, groups = 256, 96, 2
    w8 = _weight(m, n, 1)
    w, const = torch.from_numpy(w8), _const(w8)
    rng = np.random.default_rng(1)
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    ops = _ops(groups, 1)
    ring = pks_gpu.ring_rows(b"m" * 32, n, "0.o", CPU)
    peaks = {}
    for blocks in (1, 4):
        rows = m // blocks
        work = rows * pks_gpu._row_bytes(n, D * groups, groups)
        assert pks_gpu.item_rows(m, n, D * groups, groups, work) == rows
        with AllocPeak(exclude=[w, const, ring]) as t:
            pks_gpu.item(w, const, m, n, ring, b, ops, work_bytes=work)
        model = pks_gpu.item_bytes(m, n, n, D * groups, groups, work)
        assert t.peak <= model, (blocks, t.peak, model)
        assert t.peak >= 0.4 * model, (blocks, t.peak, model)  # the model is not vacuous
        peaks[blocks] = t.peak
    assert peaks[4] < 0.5 * peaks[1]


# ---- through the provider: device path == cache path, and the customer decrypts it ------------------------------

def _kernel(w8: np.ndarray, key: str = "0.o") -> Any:
    from pgw.service.kernels import ComputeKernel

    layer, kind = key.split(".")
    spec = LinearSpec(LinearId(int(layer), LinearKind(kind)), *w8.shape)
    return ComputeKernel(spec, torch.from_numpy(w8), device="cpu", backend="cpu_int8")


def _request(w8: np.ndarray, linear: str, rows: int, seed: int) -> tuple[Any, ...]:
    """A customer upload of ``rows`` random mask rows (``pks_native.Customer``), its stored compact hint built the
    customer's way, and the wire item."""
    n = w8.shape[1]
    rng = np.random.default_rng(seed)
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    a_seed, k_hat = os.urandom(32), pks.ks_a(os.urandom(32))
    cust = pn.Customer(k_hat)
    b, beta, secrets = cust.encrypt(a_seed, linear, r)
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, linear))
    hint = pn.keyswitch(pks.switch_hint(matmul_mod64(w8, a) & M56, PAR), k_hat[None].astype(np.uint32))
    del a
    item = SimpleNamespace(linear=linear, cols=b.shape[1], b=pack_bits(b.reshape(-1), 56), b_bits=56,
                           ksk=pack_ksk(beta))
    return r, a_seed, cust, hint, secrets, item


def _provider_round_trip(w8: np.ndarray, linear: str, rows: int, seed: int) -> None:
    kern = _kernel(w8, linear)
    r, a_seed, cust, hint, secrets, item = _request(w8, linear, rows, seed)
    dev = PKSProvider(mode="device")
    dev._item_ok["cpu"] = True  # the device self-test has its own test (slow on a CPU)
    host = PKSProvider(mode="cache")
    assert dev.use_device("fp", kern, linear) and not host.use_device("fp", kern, linear)
    y_dev = bytes(dev.run_item("fp", kern, item, a_seed))
    y_host = bytes(host.run_item("fp", kern, item, a_seed))
    assert dev.device_items == 1 and dev.builds == 0 and host.builds == 1
    assert y_dev == y_host
    assert len(y_dev) == -(-(item.cols // D) * (w8.shape[0] // PACK) * NR * V3_Y_BITS // 8)
    got = cust.decrypt(hint, y_dev, secrets)
    assert np.array_equal(got.astype(np.int64), ((w8.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)


@INT8
def test_provider_device_equals_cache_tiny() -> None:
    _provider_round_trip(_weight(64, 48, 3), "0.o", 300, 3)  # 2 groups, the last one partial


@INT8
@pytest.mark.slow
@pytest.mark.parametrize(("key", "m", "n"), [("0.o", 896, 896), ("0.down", 896, 4864)])
def test_provider_device_equals_cache_qwen05b_shapes(key: str, m: int, n: int) -> None:
    """Qwen2.5-0.5B's o_proj and down_proj shapes (the device key switch runs at CPU torch speed here)."""
    _provider_round_trip(_weight(m, n, 5), key, 256, 5)


# ---- policy ---------------------------------------------------------------------------------------------------

def test_hint_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, want in (("", "auto"), ("auto", "auto"), ("1", "device"), ("on", "device"), ("0", "cache"),
                      ("cache", "cache")):
        monkeypatch.setenv("PGW_SERVICE_PKS_DEVICE_HINT", raw)
        assert hint_mode() == want
        assert PKSProvider(cache_bytes=0).mode == want
    monkeypatch.setenv("PGW_SERVICE_PKS_DEVICE_HINT", "maybe")
    with pytest.raises(ValueError, match="PKS_DEVICE_HINT"):
        hint_mode()


def test_auto_policy_by_model_hint_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """``auto``: the device path on CUDA kernels exactly when one seed's hints for the model exceed the cache."""
    kern = SimpleNamespace(device=torch.device("cuda", 0), out_features=1024, in_features=512)
    monkeypatch.setattr("pgw.service.pks_provider._device_operands",
                        lambda k: [(None, None, 0, k.out_features, k.device)])
    pv = PKSProvider(cache_bytes=3 * 1024 * 4096 * 8, mode="auto")
    pv._item_ok["cuda:0"] = True
    dims = {f"{i}.o": (1024, 512) for i in range(3)}
    assert not pv.use_device("fp", kern, "0.o", dims)  # 3 linears' hints fit exactly
    dims["3.o"] = (1024, 512)
    assert pv.use_device("fp2", kern, "0.o", dims)
    # without dims: the linears seen so far (grows to the model's size)
    assert not pv.use_device("fp3", kern, "0.o") and not pv.use_device("fp3", kern, "1.o")
    assert not pv.use_device("fp3", kern, "2.o") and pv.use_device("fp3", kern, "3.o")
    assert pv.model_hint_bytes("fp3", "3.o", 1024) == 4 * 1024 * 4096 * 8
    # a failed device self-test keeps the cache path
    pv._item_ok["cuda:0"] = False
    assert not pv.use_device("fp2", kern, "0.o", dims)
    # never on CPU kernels in auto; never in cache mode
    cpu_kern = SimpleNamespace(device=CPU, out_features=1024, in_features=512)
    assert not pv.use_device("fp2", cpu_kern, "0.o", dims)
    assert not PKSProvider(cache_bytes=0, mode="cache").use_device("fp2", kern, "0.o", dims)


@INT8
def test_device_operands_of_kernels() -> None:
    from pgw.service.tensor_parallel import ShardedKernel

    w8 = _weight(128, 48, 2)
    [(w, const, r0, r1, dev)] = _device_operands(_kernel(w8))
    assert torch.equal(w, torch.from_numpy(w8)) and torch.equal(const, _const(w8))
    assert (r0, r1, dev) == (0, 128, CPU)
    assert _device_operands(SimpleNamespace(device=CPU, out_features=64, mulmod64=None)) is None
    spec = LinearSpec(LinearId(0, LinearKind("o")), 128, 48)
    parts = _device_operands(ShardedKernel(spec, torch.from_numpy(w8), ["cpu"] * 2, backend="cpu_int8"))
    assert [(p[2], p[3]) for p in parts] == [(0, 64), (64, 128)]
    assert torch.equal(parts[1][0], torch.from_numpy(w8[64:]))
    # shard bounds inside a reply ciphertext (96 rows over 2 shards: 48): the cache path serves it
    spec96 = LinearSpec(LinearId(0, LinearKind("o")), 96, 48)
    assert _device_operands(ShardedKernel(spec96, torch.from_numpy(w8[:96]), ["cpu"] * 2, backend="cpu_int8")) is None


@INT8
def test_tensor_parallel_device_item_equals_cache() -> None:
    from pgw.service.tensor_parallel import ShardedKernel

    w8 = _weight(128, 48, 6)
    spec = LinearSpec(LinearId(0, LinearKind("o")), 128, 48)
    sk = ShardedKernel(spec, torch.from_numpy(w8), ["cpu"] * 2, backend="cpu_int8")
    r, a_seed, cust, hint, secrets, item = _request(w8, "0.o", 256, 6)
    pv = PKSProvider(mode="device")
    pv._item_ok["cpu"] = True
    y = bytes(pv.run_item("fp", sk, item, a_seed))
    assert pv.device_items == 1
    assert y == bytes(PKSProvider(mode="cache").run_item("fp", _kernel(w8), item, a_seed))
    got = cust.decrypt(hint, y, secrets)
    assert np.array_equal(got.astype(np.int64), ((w8.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)


@INT8
def test_device_failure_falls_back_to_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    w8 = _weight(64, 48, 4)
    kern = _kernel(w8)
    _r, a_seed, _c, _h, _s, item = _request(w8, "0.o", 100, 4)
    pv = PKSProvider(mode="device")
    pv._item_ok["cpu"] = True

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("device lost")

    monkeypatch.setattr(pks_gpu, "item", boom)
    y = bytes(pv.run_item("fp", kern, item, a_seed))
    assert pv._item_ok["cpu"] is False and pv.device_items == 0 and pv.builds == 1
    assert y == bytes(PKSProvider(mode="cache").run_item("fp", kern, item, a_seed))


# ---- memory plan and the (estimated) item time model ------------------------------------------------------------

class _Traffic(TorchDispatchMode):
    """Bytes read + written by every non-view torch operator (inputs and outputs, each tensor once per op)."""

    VIEWS = frozenset({"view", "_unsafe_view", "t", "permute", "slice", "select", "unsqueeze", "transpose", "expand",
                       "alias", "as_strided", "reshape", "squeeze", "detach", "lift_fresh", "_reshape_alias"})

    def __init__(self) -> None:
        super().__init__()
        self.bytes = 0
        self.ops = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in self.VIEWS or name == "_int_mm":
            return out
        self.ops += 1
        seen: set[int] = set()
        for t in tree_flatten((args, kwargs, out))[0]:
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                self.bytes += t.numel() * t.element_size()
        return out


def test_traffic_constants_are_the_counted_ones() -> None:
    """memplan's per-unit device traffic is what the torch code moves (counted per operator)."""
    from pgw.service import memplan as mp

    rng = np.random.default_rng(1)
    rows = 64
    for groups in (1, 2):
        hs = torch.from_numpy(rng.integers(0, PAR.q1, (rows, 4096), dtype=np.int64))
        y = torch.from_numpy(rng.integers(0, 1 << 56, (rows, D * groups), dtype=np.int64))
        ops = torch.from_numpy(rng.integers(0, 1 << 20, (groups, 4, RANK, NR), dtype=np.int64))
        with _Traffic() as t:
            pks_gpu._keyswitch_t(hs, ops, y)
        want = mp.KS_ROW_BYTES[0] + mp.KS_ROW_BYTES[1] * groups
        assert abs(t.bytes / rows - want) <= 0.01 * want, (groups, t.bytes / rows, want)
        assert t.ops == 871
    h = torch.from_numpy(rng.integers(0, 1 << 56, (rows, 512), dtype=np.int64))
    with _Traffic() as t:
        pks_gpu._round_mul_shift(h, PAR.q1, 56) % PAR.q1
    assert t.bytes / (rows * 512) == mp.SWITCH_VALUE_BYTES
    n = 96
    w = torch.from_numpy(_weight(rows, n, 1))
    pl = pks_gpu.planes_of(torch.from_numpy(rng.integers(0, 1 << 56, (n, 512), dtype=np.int64)), n)
    with _Traffic() as t:
        pks_gpu.gemm_mod56(w, pl, torch.zeros((rows, 1), dtype=torch.int64))
    assert abs(t.bytes / (rows * 512) - mp.COMBINE_VALUE_BYTES) < 1  # + the (rows, 1) constant


def test_memplan_hint_policy_and_device_item_memory() -> None:
    from pgw.service import memplan as mp

    lins7 = mp.MODELS["qwen2.5-7b"].linears(True)
    assert abs(mp.pks_model_hint_bytes(lins7) / 1e9 - 50.5) < 0.1
    assert abs(mp.pks_model_hint_bytes(mp.MODELS["qwen2.5-72b"].linears(True)) / 1e9 - 229.8) < 0.1
    cfg = mp.MemConfig()
    assert mp.pks_device_hint(cfg, lins7)  # 50.5 GB per seed > the default 8 GB cache
    assert not mp.pks_device_hint(mp.MemConfig(pks_cache_bytes=64_000_000_000), lins7)
    assert not mp.pks_device_hint(mp.MemConfig(pks_hint="cache"), lins7)
    assert not mp.pks_device_hint(mp.MemConfig(maskgen_wire="v2"), lins7)
    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        dev = mp.plan_model(model, "h100-80", cfg).devices[0].serving["maskgen_items"]
        old = mp.plan_model(model, "h100-80", mp.MemConfig(pks_hint="cache")).devices[0].serving["maskgen_items"]
        assert dev < old, model  # recomputing in row blocks needs less device memory than a hint build
    assert mp.plan_model("qwen2.5-72b", "h100-80", cfg).fits


def test_item_time_estimates_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from pgw.service import memplan as mp

    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        before = mp.pks_throughput(model, "miss")["slots_per_s"]
        after = mp.pks_throughput(model, "device")["slots_per_s"]
        assert after > 10 * before, (model, before, after)  # type: ignore[operator]
    t = mp.pks_item_seconds(59136, 8192, 1024, "device")
    assert "host_switch" not in t and "hint_h2d" not in t
    assert max(t, key=t.__getitem__) == "key_switch"  # the remaining bottleneck: the torch NTTs
    with pytest.raises(ValueError, match="unknown path"):
        mp.pks_item_seconds(64, 64, 256, "nope")
    assert mp.main(["--pks-throughput", "--model", "qwen2.5-0.5b"]) == 0
    out = capsys.readouterr().out
    assert "ESTIMATES" in out and "| qwen2.5-0.5b | 14.9 | device |" in out
    assert mp.main(["--pks-throughput", "--model", "qwen2.5-0.5b", "--json"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 4


def test_memplan_fused_ntt_key_switch(capsys: pytest.CaptureFixture[str]) -> None:
    """The fused NTTs (``--pks-ntt fused``, the default: what ``auto`` picks on CUDA) take the key switch off the
    top of the 72B device item; the torch NTTs keep the previous estimate."""
    from pgw.service import memplan as mp

    torch_t = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="torch")
    fused_t = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="fused")
    assert torch_t == mp.pks_item_seconds(59136, 8192, 1024, "device")
    assert fused_t["key_switch"] < torch_t["key_switch"] / 10
    assert max(fused_t, key=fused_t.__getitem__) == "hint_gemm_switch"
    assert {k: v for k, v in fused_t.items() if k != "key_switch"} == \
        {k: v for k, v in torch_t.items() if k != "key_switch"}
    with pytest.raises(ValueError, match="unknown ntt"):
        mp.pks_item_seconds(64, 64, 256, "device", ntt="cuda")
    for model in ("qwen2.5-7b", "qwen2.5-72b"):
        f = mp.pks_throughput(model, "device")
        t = mp.pks_throughput(model, "device", mp.MemConfig(pks_ntt="torch"))
        assert f["ntt"] == "fused" and t["ntt"] == "torch"
        assert f["slots_per_s"] > 2 * t["slots_per_s"]  # type: ignore[operator]
    assert mp.main(["--pks-throughput", "--model", "qwen2.5-72b", "--pks-ntt", "torch", "--json"]) == 0
    assert all('"ntt": "torch"' in line for line in capsys.readouterr().out.splitlines())
