"""The pks-v3 device item's reply keys unpacked and range-checked on the device (``PGW_SERVICE_PKS_GPU_KEYS``,
``pgw.service.pks_gpu.PackedKeys`` / ``_unpack_ksk_dev``) and its uploads staged in pieces by the host thread pool
(``pgw.service.pks_gpu.staged_copy``).

The device unpack is plain torch integer code: the CPU tests run the same operators the CUDA path runs and compare
them bit for bit with the host unpack (``pks_provider.unpack_ksk``), the host path of ``PKSProvider`` and the
customer's decryption. They also cover the selection (self-test, ``auto`` / ``torch`` / ``fused``, the permanent
fallback) and that a customer's out-of-range key is refused (``ProtocolError``) before any key switch, without
disabling the device path. The CUDA tests are skipped without a GPU.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest
import torch

from pgw.core.errors import ProtocolError
from pgw.core.ringmm import cpu_int8_available
from pgw.hemask import pks
from pgw.hemask.pks import NR, PACK, RANK, D
from pgw.hemask.wire import V3_KSK_BITS, V3_KSK_BYTES, V3_Y_BITS, pack_bits
from pgw.service import pks_gpu
from pgw.service.pks_provider import PKSProvider, pack_ksk, unpack_ksk
from tests.service.test_pks_device_item import _const, _host, _kernel, _ops, _request, _weight

CPU = torch.device("cpu")
CUDA = torch.cuda.is_available()
INT8 = pytest.mark.skipif(not cpu_int8_available(), reason="needs exact CPU int8 GEMMs (torch._int_mm)")
GPU = pytest.mark.skipif(not CUDA, reason="needs CUDA")
#: every device-item choice on its plain path, so these tests run without Triton
PLAIN = ("PGW_SERVICE_PKS_GPU_NTT", "PGW_SERVICE_PKS_GPU_FWD4", "PGW_SERVICE_PKS_GPU_SWITCH",
         "PGW_SERVICE_PKS_GPU_RING", "PGW_SERVICE_PKS_GPU_WIRE")


@pytest.fixture(autouse=True)
def _plain(monkeypatch: pytest.MonkeyPatch) -> None:
    for e in PLAIN:
        monkeypatch.setenv(e, "torch")
    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_KEYS", raising=False)
    monkeypatch.setattr(pks_gpu.KEYS, "choice", {})


def _packed(beta: np.ndarray) -> bytes:
    """The wire bytes of ``beta`` (``(G, 4, 16, 8192)``) without range checks (``pack_bits`` per prime)."""
    out = bytearray()
    for g in range(beta.shape[0]):
        for i, bits in enumerate(V3_KSK_BITS):
            out += pack_bits(np.ascontiguousarray(beta[g, i], dtype=np.uint64).reshape(-1), bits)
    return bytes(out)


def _edge_keys(groups: int, seed: int) -> np.ndarray:
    beta = _ops(groups, seed).astype(np.uint64)
    for i, p in enumerate(pks.QP):
        beta[0, i, 0, :6] = [0, 1, p - 2, p - 1, (1 << 16) - 1, 1 << 16]
        beta[-1, i, -1] = p - 1
    beta[0, :, 2] = 0
    return beta


# ---- the unpack itself ------------------------------------------------------------------------------------------


def test_layout_is_the_device_one() -> None:
    assert pks_gpu.keys_layout_ok()
    assert V3_KSK_BYTES == RANK * NR * sum(V3_KSK_BITS) // 8


@pytest.mark.parametrize("groups", [1, 3])
def test_device_unpack_equals_host_unpack(groups: int) -> None:
    beta = _edge_keys(groups, 30 + groups)
    data = _packed(beta)
    assert data == bytes(pack_ksk(beta.astype(np.uint32)))  # the customer's packer: the same bytes
    want = unpack_ksk(data, groups)
    assert np.array_equal(want, beta)
    u32, bad = pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(data, CPU), groups)
    assert not bool(bad) and u32.dtype == torch.uint32 and tuple(u32.shape) == (4, groups, RANK, NR)
    got = u32.view(torch.int32).numpy().view(np.uint32)
    assert np.array_equal(got, want.transpose(1, 0, 2, 3))
    keys = pks_gpu._Keys(u32=u32)  # the torch key switch's int64 form, made from it
    assert keys.shape == (groups, 4, RANK, NR) and keys.u32() is u32
    assert np.array_equal(keys.t.numpy(), want.astype(np.int64))
    host = pks_gpu._Keys(torch.from_numpy(want.astype(np.int64)))
    assert torch.equal(host.u32().view(torch.int32), u32.view(torch.int32))
    with pytest.raises(ValueError):
        pks_gpu._Keys()


@pytest.mark.parametrize("prime", range(4))
def test_device_unpack_flags_residues_out_of_range(prime: int) -> None:
    p, bits = pks.QP[prime], V3_KSK_BITS[prime]
    for g in (0, 1):
        for v in (p, p + 1, (1 << bits) - 1):
            beta = _ops(2, 7).astype(np.uint64)
            beta[g, prime, RANK - 1, NR - 1 - g] = v
            data = _packed(beta)
            with pytest.raises(ProtocolError, match="out of range"):
                unpack_ksk(data, 2)
            _u, bad = pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(data, CPU), 2)
            assert bool(bad), (prime, g, v)


def test_packed_keys_size_and_host_cache() -> None:
    beta = _ops(2, 3)
    data = pack_ksk(beta)
    pk = pks_gpu.PackedKeys(data, 2)
    assert pk.shape == (2, 4, RANK, NR) and pk.groups == 2
    assert pk.host() is pk.host() and np.array_equal(pk.host(), beta)
    for bad in (bytes(data)[:-1], bytes(data) + b"\0"):
        with pytest.raises(ProtocolError, match="size mismatch"):
            pks_gpu.PackedKeys(bad, 2)


# ---- self-test and selection ------------------------------------------------------------------------------------


def test_keys_self_test_passes_and_detects_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    pks_gpu._keys_self_test(CPU)
    real = pks_gpu._unpack_ksk_dev

    def one_bit(raw: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, bad = real(raw, groups)
        u = u.view(torch.int32).clone()
        u[1, 0, 3, 5] ^= 1 << 9
        return u.view(torch.uint32), bad

    monkeypatch.setattr(pks_gpu, "_unpack_ksk_dev", one_bit)
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._keys_self_test(CPU)
    monkeypatch.setattr(pks_gpu, "_unpack_ksk_dev", lambda raw, g: (real(raw, g)[0], torch.zeros((), dtype=torch.bool)))
    with pytest.raises(RuntimeError, match="out of range"):
        pks_gpu._keys_self_test(CPU)
    monkeypatch.setattr(pks_gpu, "_unpack_ksk_dev", lambda raw, g: (real(raw, g)[0], torch.ones((), dtype=torch.bool)))
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._keys_self_test(CPU)


def test_keys_choice_env_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = pks_gpu.KEYS
    assert ch.env == "PGW_SERVICE_PKS_GPU_KEYS" and ch.mode() == "auto"
    assert ch.impl(CPU) == "torch" and ch.choice == {"cpu": "torch"}  # auto: CUDA devices only, no self-test on CPU
    for raw, want in (("torch", "torch"), ("host", "torch"), ("fused", "fused"), ("device", "fused")):
        monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", raw)
        assert ch.mode() == want
    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_KEYS")
    calls: list[str] = []
    monkeypatch.setattr(ch, "test", lambda dev: calls.append(str(dev)))
    assert ch.impl("cuda:2") == "fused" and calls == ["cuda:2"]
    monkeypatch.setattr(ch, "test", lambda dev: (_ for _ in ()).throw(RuntimeError("mismatch")))
    assert ch.impl("cuda:3") == "torch"  # a failing self-test keeps the host unpack
    assert pks_gpu.fused_status(CPU)["keys"] == "torch"


# ---- whole items -------------------------------------------------------------------------------------------------


def _item_inputs(groups: int = 2) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, int, int, bytes, np.ndarray,
                                            np.ndarray]:
    m, n = 2 * PACK, 3 * RANK
    w8 = _weight(m, n, 41)
    wpad = torch.zeros((m, 64), dtype=torch.int8)
    wpad[:, :n] = torch.from_numpy(w8)
    rng = np.random.default_rng(42)
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    return w8, wpad, _const(w8), m, n, os.urandom(32), b, _edge_keys(groups, 43).astype(np.uint32)


@INT8
@pytest.mark.parametrize("keys", ["torch", "fused"])
def test_item_packed_keys_equal_host_path(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    """``pks_gpu.item`` with the keys' wire bytes (unpacked on the device or on the host), the upload's wire bytes
    and the reply's wire bytes out equals ``pack_bits`` of the host path's reply; two row blocks."""
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", keys)
    w8, wpad, const, m, n, seed, b, ops = _item_inputs()
    groups = ops.shape[0]
    want = _host(w8, seed, "k", b, ops)
    work = pks_gpu._row_bytes(64, D * groups, groups) * PACK
    got = pks_gpu.item(wpad, const, m, n, pks_gpu.ring_rows(seed, n, "k", CPU), b,
                       pks_gpu.PackedKeys(pack_ksk(ops), groups), work_bytes=work)
    assert np.array_equal(got, want)
    pr = pks_gpu.item(wpad, const, m, n, pks_gpu.ring_rows(seed, n, "k", CPU),
                      pks_gpu.PackedUpload(pack_bits(b.reshape(-1), 56), n, D * groups),
                      pks_gpu.PackedKeys(pack_ksk(ops), groups), out=pks_gpu.PackedReply(groups, m // PACK))
    assert bytes(pr.buf) == bytes(pack_bits(want.reshape(-1), V3_Y_BITS))


@INT8
@pytest.mark.parametrize("keys", ["torch", "fused", "auto"])
def test_item_refuses_out_of_range_keys_before_any_key_switch(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", keys)
    if keys == "auto":
        monkeypatch.setattr(pks_gpu.KEYS, "choice", {"cpu": "fused"})  # as after a passed self-test on a GPU
    _w8, wpad, const, m, n, seed, b, ops = _item_inputs()
    beta = ops.astype(np.uint64)
    beta[1, 3, 0, 0] = pks.QP[3]
    calls: list[int] = []
    real = pks_gpu._keyswitch_t
    monkeypatch.setattr(pks_gpu, "_keyswitch_t", lambda *a: calls.append(1) or real(*a))
    with pytest.raises(ProtocolError, match="out of range"):
        pks_gpu.item(wpad, const, m, n, pks_gpu.ring_rows(seed, n, "k", CPU), b,
                     pks_gpu.PackedKeys(_packed(beta), 2))
    assert calls == []
    assert pks_gpu.KEYS.choice.get("cpu", "torch") == ("fused" if keys == "auto" else "torch")  # not a device fault


@INT8
def test_item_device_unpack_error_falls_back_for_good(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under ``auto`` with the device unpack chosen, an error of it falls back to the host unpack for good: the same
    bits."""
    monkeypatch.setattr(pks_gpu.KEYS, "choice", {"cpu": "fused"})
    w8, wpad, const, m, n, seed, b, ops = _item_inputs(1)
    want = _host(w8, seed, "k", b, ops)

    def boom(raw: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(pks_gpu, "_unpack_ksk_dev", boom)
    got = pks_gpu.item(wpad, const, m, n, pks_gpu.ring_rows(seed, n, "k", CPU), b, pks_gpu.PackedKeys(pack_ksk(ops), 1))
    assert np.array_equal(got, want) and pks_gpu.KEYS.choice["cpu"] == "torch"


@INT8
@pytest.mark.parametrize("keys", ["torch", "fused"])
def test_provider_packed_keys_equal_cache_and_decrypt(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    from tests.service.test_pks_device_item import _provider_round_trip

    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", keys)
    _provider_round_trip(_weight(64, 48, 8), "0.o", 300, 8)


@INT8
@pytest.mark.parametrize("keys", ["torch", "fused"])
def test_provider_refuses_bad_keys_and_keeps_the_device_path(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    """A customer's reply key out of range is refused (``ProtocolError``) on the device path as on the cache path,
    and it does not switch the provider's device path off (a customer must not be able to do that)."""
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", keys)
    w8 = _weight(64, 48, 9)
    kern = _kernel(w8)
    _r, a_seed, _cust, _hint, _secrets, item = _request(w8, "0.o", 256, 9)
    good = item.ksk
    beta = unpack_ksk(good, 1).astype(np.uint64)
    beta[0, 2, 7, 11] = (1 << V3_KSK_BITS[2]) - 1
    item.ksk = _packed(beta)
    dev = PKSProvider(mode="device")
    dev._item_ok["cpu"] = True
    host = PKSProvider(mode="cache")
    for prov in (dev, host):
        with pytest.raises(ProtocolError, match="out of range"):
            prov.run_item("fp", kern, item, a_seed)
    assert dev._item_ok["cpu"] is True and dev.device_items == 0 and dev.items == 0 and host.builds == 0
    item.ksk = bytes(good)[:-3]
    for prov in (dev, host):
        with pytest.raises(ProtocolError, match="size mismatch"):
            prov.run_item("fp", kern, item, a_seed)
    item.ksk = good
    assert bytes(dev.run_item("fp", kern, item, a_seed)) == bytes(host.run_item("fp", kern, item, a_seed))
    assert dev.device_items == 1


def test_item_self_test_covers_packed_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``item_self_test`` runs the wire forms with the keys' wire bytes too (here with the device unpack forced)."""
    if not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", "fused")
    seen: list[int] = []
    real = pks_gpu._unpack_ksk_dev
    monkeypatch.setattr(pks_gpu, "_unpack_ksk_dev", lambda raw, g: seen.append(g) or real(raw, g))
    pks_gpu.item_self_test(CPU)
    assert seen == [1]


# ---- staged uploads ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("size", [0, 5, 1 << 20, 3 * (1 << 20) + 17])
def test_staged_copy_in_order_and_exact(monkeypatch: pytest.MonkeyPatch, threads: str, size: int) -> None:
    monkeypatch.setenv("PGW_SERVICE_PKS_HOST_THREADS", threads)
    src = np.frombuffer(os.urandom(size), dtype=np.uint8)
    dst = np.zeros_like(src)
    seen: list[tuple[int, int]] = []

    def piece(a: int, b: int) -> None:
        assert np.array_equal(dst[a:b], src[a:b])  # the piece is there before its transfer is queued
        seen.append((a, b))

    pks_gpu.staged_copy(src, dst, piece, chunk=1 << 18)
    assert np.array_equal(dst, src)
    want = [(a, min(size, a + (1 << 18))) for a in range(0, size, 1 << 18)]
    if threads == "1" or len(want) <= 1:
        want = [(0, size)] if size else []
    assert seen == want
    with pytest.raises(ValueError):
        pks_gpu.staged_copy(src, np.zeros(size + 1, dtype=np.uint8), piece)


def test_staged_copy_error_waits_for_the_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error while queuing a piece propagates, and no copy keeps writing into the buffer afterwards."""
    monkeypatch.setenv("PGW_SERVICE_PKS_HOST_THREADS", "4")
    src = np.frombuffer(os.urandom(4 << 20), dtype=np.uint8)
    dst = np.zeros_like(src)
    running = threading.Event()

    def piece(a: int, b: int) -> None:
        running.set()
        raise RuntimeError("queue failed")

    with pytest.raises(RuntimeError, match="queue failed"):
        pks_gpu.staged_copy(src, dst, piece, chunk=1 << 16)
    assert running.is_set()
    snap = dst.copy()
    time.sleep(0.05)
    assert np.array_equal(dst, snap)  # every copy had finished or was cancelled when the error came out


# ---- memplan -----------------------------------------------------------------------------------------------------


def test_memplan_keys_and_staged_uploads() -> None:
    from pgw.service import memplan as mp

    prev = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="fused", kernels="r2")
    new = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="fused", kernels="all")
    assert "keys_host" in prev and "keys_host" not in new and new["keys_device"] < prev["keys_host"] / 10
    assert new["upload_h2d"] < prev["upload_h2d"]
    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        a = mp.pks_throughput(model, "device")
        b = mp.pks_throughput(model, "device", mp.MemConfig(pks_kernels="r2"))
        assert a["slots_per_s"] > b["slots_per_s"]  # type: ignore[operator]


# ---- CUDA -------------------------------------------------------------------------------------------------------


@GPU
def test_cuda_staged_upload_and_keys() -> None:  # pragma: no cover - GPU only
    dev = torch.device("cuda", 0)
    data = os.urandom(3 * pks_gpu.UPLOAD_CHUNK + 1234)
    got = pks_gpu._upload_bytes(data, dev)
    torch.cuda.synchronize(dev)
    assert bytes(got.cpu().numpy()) == data
    arr = np.frombuffer(os.urandom(8 * 999), dtype=np.uint64).reshape(999, 8)
    assert np.array_equal(pks_gpu._upload(arr, dev).cpu().numpy().view(np.uint64), arr)
    pks_gpu._keys_self_test(dev)
    beta = _edge_keys(3, 5)
    u32, bad = pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(_packed(beta), dev), 3)
    assert not bool(bad)
    assert np.array_equal(u32.view(torch.int32).cpu().numpy().view(np.uint32), beta.transpose(1, 0, 2, 3))
    beta[2, 1, 4, 4] = pks.QP[1]
    _u, bad = pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(_packed(beta), dev), 3)
    assert bool(bad)


@GPU
@pytest.mark.parametrize("keys", ["fused", "auto"])
def test_cuda_item_packed_keys(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:  # pragma: no cover - GPU only
    """On CUDA: the item with the keys' and the upload's wire bytes in and the reply's out equals the host path, and an
    out-of-range key is refused before any key switch (the range check read back through pinned memory)."""
    dev = torch.device("cuda", 0)
    for e in PLAIN:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_KEYS", keys)
    w8, wpad, const, m, n, seed, b, ops = _item_inputs()
    groups = ops.shape[0]
    want = _host(w8, seed, "k", b, ops)
    pr = pks_gpu.item(wpad.to(dev), const.to(dev), m, n, pks_gpu.ring_rows(seed, n, "k", dev),
                      pks_gpu.PackedUpload(pack_bits(b.reshape(-1), 56), n, D * groups),
                      pks_gpu.PackedKeys(pack_ksk(ops), groups), out=pks_gpu.PackedReply(groups, m // PACK))
    assert bytes(pr.buf) == bytes(pack_bits(want.reshape(-1), V3_Y_BITS))
    assert pks_gpu.KEYS.impl(dev) == "fused"
    beta = ops.astype(np.uint64)
    beta[0, 0, 0, 0] = pks.QP[0]
    with pytest.raises(ProtocolError, match="out of range"):
        pks_gpu.item(wpad.to(dev), const.to(dev), m, n, pks_gpu.ring_rows(seed, n, "k", dev), b,
                     pks_gpu.PackedKeys(_packed(beta), groups))
    assert pks_gpu.KEYS.impl(dev) == "fused"
