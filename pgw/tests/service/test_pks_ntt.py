"""Fused Triton NTTs of the pks-v3 provider key switch (``pgw.service.pks_ntt``) and their selection
(``PGW_SERVICE_PKS_GPU_NTT``, ``pgw.service.pks_gpu.ntt_impl``).

Without CUDA the kernels run in Triton's interpreter (``TRITON_INTERPRET=1``, set by ``tests/conftest.py``) on CPU
tensors: the same kernel code, executed with numpy integer semantics (uint32 wrap-around, ``umulhi``). Every result is
compared bit for bit with :class:`pgw.hemask.pks.Ntt`, the torch key switch and the CPU key switch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams
from pgw.service import pks_gpu

triton = pytest.importorskip("triton")
from pgw.service import pks_ntt  # noqa: E402

PAR = PKSParams()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu" and os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)


def _centred(hs: np.ndarray) -> np.ndarray:
    """``A~`` of :func:`pks_gpu._keyswitch_torch` (``(nb, 16, 8192)`` int64, centred)."""
    nb = hs.shape[0] // PACK
    at = hs.astype(np.int64).reshape(nb, PACK, RANK, D).transpose(0, 2, 3, 1).reshape(nb, RANK, NR)
    return np.where(at >= (PAR.q1 + 1) // 2, at - PAR.q1, at)


def _edge_rows(seed: int, nb: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hs = pks_gpu._edge_hint_rows(rng, nb * PACK)
    if nb > 1:  # one whole reply ciphertext at the centring boundary and one at q1 - 1
        hs[PACK:2 * PACK] = (PAR.q1 + 1) // 2
        hs[-PACK:] = PAR.q1 - 1
    return hs


@pytest.mark.parametrize("p", pks.QP)
def test_forward_equals_reference(p: int) -> None:
    hs = _edge_rows(3, 3)
    got = pks_ntt.forward(torch.from_numpy(hs.view(np.int64)).to(DEV), p).cpu().numpy().astype(np.uint64)
    want = pks.Ntt.of(p).fwd(np.mod(_centred(hs), p).astype(np.uint64))
    assert got.dtype == np.uint64 and np.array_equal(got, want)
    assert got.max() < p


@pytest.mark.parametrize("p", pks.QP)
def test_mac_inverse_equals_reference(p: int) -> None:
    rng = np.random.default_rng(p % 1000)
    nb, groups = 2, 3
    ah = rng.integers(0, p, (nb, RANK, NR), dtype=np.uint64)
    ah[0, 0] = p - 1  # the largest Montgomery products
    ah[0, 1] = 0
    ah[1, :, :8] = [0, 1, p - 1, p - 2, 1, 0, p - 1, p - 1]
    ops = rng.integers(0, p, (groups, RANK, NR), dtype=np.uint64)
    ops[0] = p - 1  # every slot of every key polynomial at p - 1
    ops[1, :, ::2] = 0
    ops[2, :, :8] = [0, 1, p - 1, p - 2, p - 1, p - 1, 0, 1]
    got = pks_ntt.mac_inverse(torch.from_numpy(ah.astype(np.uint32)).to(DEV),
                              torch.from_numpy(ops.astype(np.uint32)).to(DEV), p).cpu().numpy().astype(np.uint64)
    acc = np.zeros((groups, nb, NR), dtype=np.uint64)
    for li in range(RANK):  # products below 2^60, reduced per term
        acc = (acc + ah[None, :, li] * ops[:, None, li] % np.uint64(p)) % np.uint64(p)
    want = pks.Ntt.of(p).inv(acc)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("p", pks.QP)
def test_round_trip_and_unit_vectors(p: int) -> None:
    """``inv(fwd(x) * 1)`` is ``x`` (the keys at the Montgomery-free unit ``1`` in slot ``l = 0``), and the transforms
    of the unit vectors ``X^k`` in every slot ``k`` of one polynomial equal the reference."""
    nt = pks.Ntt.of(p)
    one = np.zeros((1, RANK, NR), dtype=np.uint32)
    one[0, 0] = 1
    hs = _edge_rows(5, 1)
    ah = pks_ntt.forward(torch.from_numpy(hs.view(np.int64)).to(DEV), p)
    back = pks_ntt.mac_inverse(ah, torch.from_numpy(one).to(DEV), p).cpu().numpy().astype(np.uint64)
    assert np.array_equal(back[0, 0], np.mod(_centred(hs)[0, 0], p).astype(np.uint64))
    # unit vectors: A~[0, l, i] = 1 for slot i = 32 u + j <-> hs[j, 256 l + u]; 16 polynomials per launch, 512 slots
    # each -> all 8192 slots over 16 launches, every slot at value 1 (and at -1 = q1 - 1 in the next row block)
    for k0 in range(0, NR, RANK):
        hs = np.zeros((2 * PACK, NU), dtype=np.uint64)
        for li in range(RANK):
            i = k0 + li
            hs[i % PACK, D * li + i // PACK] = 1
            hs[PACK + i % PACK, D * li + i // PACK] = PAR.q1 - 1
        got = pks_ntt.forward(torch.from_numpy(hs.view(np.int64)).to(DEV), p).cpu().numpy().astype(np.uint64)
        want = nt.fwd(np.mod(_centred(hs), p).astype(np.uint64))
        assert np.array_equal(got, want), k0
        if k0 >= 3 * RANK:  # every slot of the transform is checked in the first launches; the rest are spot checks
            break


@pytest.mark.parametrize("groups", [1, 2])
def test_fused_keyswitch_equals_torch_and_cpu(groups: int) -> None:
    rng = np.random.default_rng(20 + groups)
    m = 3 * PACK
    hs = _edge_rows(7 + groups, 3)
    y = rng.integers(0, 1 << 56, (m, D * groups), dtype=np.uint64)
    y[0, :4] = [0, 1, (1 << 56) - 1, 1 << 55]
    y[PACK] = (1 << 56) - 1
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP])
                    for _ in range(groups)])
    for k, p in enumerate(pks.QP):
        ops[0, k, 0] = p - 1
    keys = pks_gpu._Keys(torch.from_numpy(ops).to(DEV))
    hs_t = torch.from_numpy(hs.view(np.int64)).to(DEV)
    y_t = torch.from_numpy(y.view(np.int64)).to(DEV)
    got = pks_ntt.keyswitch(hs_t, keys.u32(), y_t).cpu()
    assert torch.equal(got, pks_gpu._keyswitch_torch(hs_t, keys.t, y_t).cpu())
    want = pn.keyswitch(hs, ops.astype(np.uint32), y)
    assert np.array_equal(got.numpy().astype(np.uint64), want)


@pytest.mark.parametrize("arch", [75, 90])
def test_kernels_lower_for_cuda(arch: int) -> None:
    """Every kernel compiles to a cubin for a T4 (sm_75) and an H100 (sm_90) with the launch signature, without a GPU
    (Triton's bundled ptxas). The interpreter does not run Triton's front end: a T4 run found that an integer
    argument equal to 1 (one group) became a compile-time constant there, so the kernels take no such argument and
    :class:`pks_ntt.Tables` refuses constants equal to 1 (``pks_ntt.compile_for``, in a process without the
    interpreter)."""
    import json
    import subprocess
    import sys

    code = f"import json; from pgw.service import pks_ntt; print(json.dumps(pks_ntt.compile_for({arch})))"
    env = dict(os.environ, TRITON_INTERPRET="0")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)  # noqa: S603
    if r.returncode and ("ptxas" in r.stderr.lower() or "No module named 'triton.backends" in r.stderr):
        pytest.skip(f"no CUDA compiler toolchain in this Triton: {r.stderr[-300:]}")
    assert r.returncode == 0, r.stderr[-3000:]
    sizes = json.loads(r.stdout.strip().splitlines()[-1])
    assert set(sizes) == {"fwd", "fwd4", "mac_inv", "tail"} and all(v > 0 for v in sizes.values())


def test_fused_self_test_passes() -> None:
    pks_gpu.fused_self_test(DEV)


def test_fused_self_test_detects_a_wrong_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    real = pks_ntt.keyswitch

    def off_by_one(*a: object) -> torch.Tensor:
        out = real(*a)  # type: ignore[arg-type]
        out[0, 0, 5] ^= 1
        return out

    monkeypatch.setattr(pks_ntt, "keyswitch", off_by_one)
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu.fused_self_test(DEV)


def test_ntt_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, want in (("", "auto"), ("auto", "auto"), ("torch", "torch"), ("0", "torch"), ("fused", "fused"),
                      ("Triton", "fused")):
        monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", raw)
        assert pks_gpu.ntt_mode() == want
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", "fast")
    with pytest.raises(ValueError):
        pks_gpu.ntt_mode()


def test_auto_selection_and_permanent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pks_gpu, "_ntt_choice", {})
    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_NTT", raising=False)
    assert pks_gpu.ntt_impl("cpu") == "torch"  # auto: CUDA devices only
    if not torch.cuda.is_available():
        # the self-test cannot place tensors on a missing CUDA device: an error keeps torch for good
        assert pks_gpu.ntt_impl("cuda:0") == "torch"
        assert pks_gpu._ntt_choice["cuda:0"] == "torch"
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", "fused")
    assert pks_gpu.ntt_impl("cpu") == "fused"
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", "torch")
    assert pks_gpu.ntt_impl("cuda:0") == "torch"
    # auto with the fused kernels chosen: an error falls back to torch (same bits) and stays there
    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_NTT")
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_FWD4", "torch")  # one keyswitch call per item: the per-prime forward
    rng = np.random.default_rng(4)
    hs = torch.from_numpy(rng.integers(0, PAR.q1, (PACK, NU), dtype=np.int64)).to(DEV)
    key = str(hs.device)  # "cuda:0", not "cuda"
    pks_gpu._ntt_choice[key] = "fused"
    y = torch.from_numpy(rng.integers(0, 1 << 56, (PACK, D), dtype=np.int64)).to(DEV)
    ops = torch.from_numpy(np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP])[None]).to(DEV)
    want = pks_gpu._keyswitch_torch(hs, ops, y)
    calls = []

    def boom(*a: object) -> torch.Tensor:
        calls.append(1)
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(pks_ntt, "keyswitch", boom)
    assert torch.equal(pks_gpu._keyswitch_t(hs, ops, y), want)
    assert pks_gpu._ntt_choice[key] == "torch" and calls == [1]
    assert torch.equal(pks_gpu._keyswitch_t(hs, ops, y), want)
    assert calls == [1]  # not retried
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", "fused")  # forced: errors propagate
    with pytest.raises(RuntimeError, match="launch failed"):
        pks_gpu._keyswitch_t(hs, ops, y)


def test_device_item_with_fused_ntt_equals_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pks_gpu.item`` (hint GEMM, switch, ``y`` GEMM, key switch) with the fused NTTs equals the host path."""
    from pgw.core.ringmm import cpu_int8_available
    from pgw.hemask.lwe import matmul_mod64

    if DEV.type == "cpu" and not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_NTT", "fused")
    rng = np.random.default_rng(9)
    m, n, groups = 2 * PACK, 3 * RANK, 2
    w8 = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w8[0], w8[-1] = -128, 127
    wpad = torch.zeros((m, 64), dtype=torch.int8)
    wpad[:, :n] = torch.from_numpy(w8)
    scale = (128 * sum(1 << (8 * d) for d in range(7))) % (1 << 64)
    const = torch.from_numpy((w8.astype(np.int64).sum(1).view(np.uint64) * np.uint64(scale)).view(np.int64)
                             .reshape(m, 1).copy())
    a_seed = bytes(range(32, 64))
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                    for _ in range(groups)]).astype(np.uint32)
    got = pks_gpu.item(wpad.to(DEV), const.to(DEV), m, n, pks_gpu.ring_rows(a_seed, n, "t", DEV), b, ops)
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, "t"))
    m56 = np.uint64((1 << 56) - 1)
    want = pn.keyswitch(pks.switch_hint(matmul_mod64(w8, a) & m56, PAR), ops, matmul_mod64(w8, b) & m56)
    assert np.array_equal(got, want)
