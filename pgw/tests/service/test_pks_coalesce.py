"""Coalesced pks-v3 device items (one hint computation for the groups of several items) and the opt-in common
upload seed (``pgw.hemask.pks.crs_a_seed``). Research note: ``research/reviews/mask-cost-consumer.md``.

Every coalesced reply is compared bit for bit with the same item computed alone (device path and host path) and
decrypted by its own customer to ``W8 r mod 2^32``. The torch device path runs on CPU here (same code on CUDA).
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import ProtocolError
from pgw.core.ringmm import cpu_int8_available
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks import NR, PACK, RANK, D, PKSParams
from pgw.hemask.wire import V3_KSK_BYTES, V3_Y_BITS, pack_bits
from pgw.service import pks_gpu
from pgw.service.pks_provider import PKSProvider, _Coalescer, pack_ksk

PAR = PKSParams()
M56 = np.uint64((1 << 56) - 1)
CPU = torch.device("cpu")
INT8 = pytest.mark.skipif(not cpu_int8_available(), reason="needs exact CPU int8 GEMMs (torch._int_mm)")
NATIVE = pytest.mark.skipif(not pn.available(), reason="pgw_pks not built")


def _weight(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w[0], w[-1] = -128, 127
    return w


def _const(w8: np.ndarray) -> torch.Tensor:
    scale = (128 * sum(1 << (8 * d) for d in range(7))) % (1 << 64)
    cu = w8.astype(np.int64).sum(1).view(np.uint64) * np.uint64(scale)
    return torch.from_numpy(cu.view(np.int64).reshape(-1, 1).copy())


def _ops(groups: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                     for _ in range(groups)]).astype(np.uint32)


@pytest.mark.parametrize(("packed", "groups"), [(False, [1, 2, 1]), (True, [1, 2, 1]), (True, [1, 1, 1, 1, 1])])
def test_multi_upload_item_equals_items_alone(packed: bool, groups: list[int]) -> None:
    """Items side by side (one-group parts are merged into 512-column plane blocks): the reply of each is the reply
    of the item alone."""
    m, n, a_seed, linear = 64, 48, bytes(range(32)), "2.o"
    w8 = _weight(m, n, 11)
    w, const = torch.from_numpy(w8), _const(w8)
    rng = np.random.default_rng(4)
    bs = [rng.integers(0, 1 << 56, (n, D * g), dtype=np.uint64) for g in groups]
    bs[-1][0] = (1 << 56) - 1
    ops = [_ops(g, 20 + i) for i, g in enumerate(groups)]
    alone = [pks_gpu.item(w, const, m, n, pks_gpu.ring_rows(a_seed, n, linear, CPU), b, o)
             for b, o in zip(bs, ops, strict=True)]
    parts: list[Any] = [pks_gpu.PackedUpload(pack_bits(b.reshape(-1), 56), n, b.shape[1]) if packed else b for b in bs]
    multi = pks_gpu.MultiUpload(parts)
    assert multi.shape == (n, D * sum(groups))
    out = pks_gpu.PackedReply(sum(groups), m // PACK) if packed else None
    got = pks_gpu.item(w, const, m, n, pks_gpu.ring_rows(a_seed, n, linear, CPU), multi, np.concatenate(ops), out=out)
    split = pks_gpu.split_reply(got, groups)
    for i, (b, o) in enumerate(zip(bs, ops, strict=True)):
        assert split[i] == bytes(pack_bits(alone[i].reshape(-1), V3_Y_BITS))
        a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, linear))  # the host path of the provider
        hs = pks.switch_hint(matmul_mod64(w8, a) & M56, PAR)
        if pn.available():
            assert np.array_equal(alone[i], pn.keyswitch(hs, o, matmul_mod64(w8, b) & M56))
    with pytest.raises(ValueError):
        pks_gpu.split_reply(got, [1])
    with pytest.raises(ValueError):
        pks_gpu.MultiUpload([bs[0], bs[0][:, :100]])


# ---- through the provider, two customers on the common seed ------------------------------------------------

def _kernel(w8: np.ndarray, key: str) -> Any:
    from pgw.service.kernels import ComputeKernel

    layer, kind = key.split(".")
    spec = LinearSpec(LinearId(int(layer), LinearKind(kind)), *w8.shape)
    return ComputeKernel(spec, torch.from_numpy(w8), device="cpu", backend="cpu_int8")


def _customer(w8: np.ndarray, linear: str, a_seed: bytes, rows: int, seed: int) -> tuple[Any, ...]:
    """One customer: its own private ``ks_seed`` (so its own compact hint), its own mask rows and fresh secrets, the
    upload seed ``a_seed`` (the common one in these tests)."""
    n = w8.shape[1]
    r = np.random.default_rng(seed).integers(0, 1 << 32, (rows, n), dtype=np.int64)
    k_hat = pks.ks_a(os.urandom(32))
    cust = pn.Customer(k_hat)
    b, beta, secrets = cust.encrypt(a_seed, linear, r)
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, linear))
    hint = pn.keyswitch(pks.switch_hint(matmul_mod64(w8, a) & M56, PAR), k_hat[None].astype(np.uint32))
    item = SimpleNamespace(linear=linear, cols=b.shape[1], b=pack_bits(b.reshape(-1), 56), b_bits=56,
                           ksk=pack_ksk(beta))
    return r, cust, hint, secrets, item


def _want(w8: np.ndarray, r: np.ndarray) -> np.ndarray:
    return ((w8.astype(np.int64) @ r.T) & 0xFFFFFFFF).T


@INT8
@NATIVE
def test_run_items_two_customers_common_seed() -> None:
    w8, linear = _weight(64, 48, 3), "0.o"
    kern = _kernel(w8, linear)
    a_seed = pks.crs_a_seed("fp-test")
    custs = [_customer(w8, linear, a_seed, rows, 30 + i) for i, rows in enumerate((256, 300, 77))]
    pv = PKSProvider(mode="device")
    pv._item_ok["cpu"] = True
    ys = pv.run_items("fp", kern, [c[4] for c in custs], a_seed)
    assert pv.coalesced_runs == 1 and pv.coalesced_items == 3 and pv.device_items == 1
    alone = PKSProvider(mode="device")
    alone._item_ok["cpu"] = True
    host = PKSProvider(mode="cache")
    for (r, cust, hint, secrets, item), y in zip(custs, ys, strict=True):
        assert bytes(y) == bytes(alone.run_item("fp", kern, item, a_seed)) == bytes(host.run_item("fp", kern, item,
                                                                                                    a_seed))
        assert np.array_equal(cust.decrypt(hint, y, secrets).astype(np.int64), _want(w8, r))
    # another customer's hint does not decrypt this customer's reply (private ks_seed: the hint is per customer)
    r0, c0, _h0, s0, _ = custs[0]
    assert not np.array_equal(c0.decrypt(custs[1][2], ys[0], s0).astype(np.int64), _want(w8, r0))


@INT8
@NATIVE
def test_run_items_refused_item_is_isolated() -> None:
    """One item's reply keys out of range: only that item is refused; the others get their own replies."""
    w8, linear = _weight(64, 48, 5), "1.o"
    kern = _kernel(w8, linear)
    a_seed = pks.crs_a_seed("fp-test")
    custs = [_customer(w8, linear, a_seed, 256, 40 + i) for i in range(3)]
    bad = custs[1][4]
    raw = bytearray(bad.ksk)
    cnt = RANK * NR  # 24-bit fields: all 16-bit low parts, then all 8-bit high parts
    raw[0:2] = b"\xff\xff"
    raw[2 * cnt] = 0xFF  # the first residue of the first prime: 2^24 - 1 >= p
    bad.ksk = bytes(raw)
    assert len(bad.ksk) == V3_KSK_BYTES
    pv = PKSProvider(mode="device")
    pv._item_ok["cpu"] = True
    res = pv.run_items("fp", kern, [c[4] for c in custs], a_seed, return_exceptions=True)
    assert isinstance(res[1], ProtocolError)
    for i in (0, 2):
        r, cust, hint, secrets, _ = custs[i]
        assert np.array_equal(cust.decrypt(hint, res[i], secrets).astype(np.int64), _want(w8, r))
    with pytest.raises(ProtocolError):
        pv.run_items("fp", kern, [c[4] for c in custs], a_seed)


@INT8
@NATIVE
def test_coalescer_group_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent ``run_item`` calls of one linear and seed become one device item (``PGW_SERVICE_PKS_COALESCE_MS``);
    every caller gets its own reply, bit-identical to the item alone."""
    monkeypatch.setenv("PGW_SERVICE_PKS_COALESCE_MS", "400")
    monkeypatch.setenv("PGW_SERVICE_PKS_COALESCE_GROUPS", "8")
    w8, linear = _weight(64, 48, 7), "3.o"
    kern = _kernel(w8, linear)
    a_seed = pks.crs_a_seed("fp-test")
    custs = [_customer(w8, linear, a_seed, 256, 50 + i) for i in range(3)]
    pv = PKSProvider(mode="device")
    pv._item_ok["cpu"] = True
    assert pv._coalescer is not None and pv._coalescer.max_groups == 8
    out: dict[int, Any] = {}

    def call(i: int) -> None:
        out[i] = pv.run_item("fp", kern, custs[i][4], a_seed)

    ths = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert pv.coalesced_runs == 1 and pv.coalesced_items == 3
    for i, (r, cust, hint, secrets, _) in enumerate(custs):
        assert np.array_equal(cust.decrypt(hint, out[i], secrets).astype(np.int64), _want(w8, r))
    # a full batch closes at once; another seed never joins
    co = _Coalescer(10.0, 1)
    assert bytes(co.submit(pv, "fp", kern, custs[0][4], a_seed, None)) == bytes(out[0])


def test_coalescing_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_SERVICE_PKS_COALESCE_MS", raising=False)
    assert PKSProvider(mode="device")._coalescer is None
    monkeypatch.setenv("PGW_SERVICE_PKS_COALESCE_MS", "0")
    assert PKSProvider(mode="device")._coalescer is None


def test_common_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    s = pks.crs_a_seed("abc")
    assert len(s) == 32 and s == pks.crs_a_seed("abc") and s != pks.crs_a_seed("abd")
    assert s != pks.crs_a_seed("abc", "v2")
    monkeypatch.delenv("PGW_PKS_CRS_A", raising=False)
    assert pks.upload_seed("abc") != pks.upload_seed("abc")  # fresh by default
    monkeypatch.setenv("PGW_PKS_CRS_A", "1")
    assert pks.upload_seed("abc") == s
