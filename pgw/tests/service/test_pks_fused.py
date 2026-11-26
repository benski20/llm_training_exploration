"""Fused kernels of the pks-v3 device item besides the NTTs (``pgw.service.pks_fused``), the 4-prime forward NTT
(``pgw.service.pks_ntt.forward4``), their selection (``pgw.service.pks_gpu.FusedChoice``: ``PGW_SERVICE_PKS_GPU_SWITCH``,
``..._RING``, ``..._FWD4``), the multi-threaded host ring-row stream and the pinned / overlapped copies.

Without CUDA the kernels run in Triton's interpreter (``TRITON_INTERPRET=1``, set by ``tests/conftest.py``) on CPU
tensors. Every result is compared bit for bit with the torch path, :func:`pgw.hemask.pks.switch_hint`,
:func:`pgw.hemask.pks.public_ring_rows`, :class:`pgw.hemask.pks.Ntt` and, end to end, the host path of
``PKSProvider`` and the customer's decryption.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pgw.hemask import pks
from pgw.hemask.lwe import matmul_mod64
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams
from pgw.service import pks_gpu

triton = pytest.importorskip("triton")
from pgw.service import pks_fused, pks_ntt  # noqa: E402
from pgw.service.kernels import _combine_planes, _combine_planes_np  # noqa: E402

PAR = PKSParams()
M56 = (1 << 56) - 1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERP = DEV.type == "cpu"
if INTERP and os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)
CHOICES = ("PGW_SERVICE_PKS_GPU_NTT", "PGW_SERVICE_PKS_GPU_FWD4", "PGW_SERVICE_PKS_GPU_SWITCH",
           "PGW_SERVICE_PKS_GPU_RING", "PGW_SERVICE_PKS_GPU_WIRE")


def _partials(rng: np.random.Generator, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    part = rng.integers(-(1 << 31), 1 << 31, (r, 7 * c), dtype=np.int64)
    part[0] = (1 << 31) - 1
    part[1] = -(1 << 31)
    part[2, ::2] = 0
    part[3, :c] = -(1 << 31)  # plane 0 at its minimum, the rest random
    part[4, 6 * c:] = (1 << 31) - 1  # the top plane at its maximum
    const = rng.integers(-(1 << 63), (1 << 63) - 1, (r, 1), dtype=np.int64)
    const[:5, 0] = [-1, 0, -(1 << 63), (1 << 63) - 1, 1 << 56]
    return part.astype(np.int32), const


@pytest.mark.parametrize(("r", "c"), [(40, 300), (32, 512), (1, 256), (17, 1)])
def test_combine_equals_torch_and_numpy(r: int, c: int) -> None:
    rng = np.random.default_rng(r * 1000 + c)
    part, const = _partials(rng, max(r, 5), c)
    part, const = part[:r], const[:r]
    p_t, c_t = torch.from_numpy(part).to(DEV), torch.from_numpy(const).to(DEV)
    want = _combine_planes([p_t[:, d * c:(d + 1) * c] for d in range(7)], c_t).bitwise_and_(M56)
    host = _combine_planes_np([part[:, d * c:(d + 1) * c] for d in range(7)], const.view(np.uint64)) & np.uint64(M56)
    assert np.array_equal(want.cpu().numpy().view(np.uint64), host)
    buf = torch.full((r, c + 6), -3, dtype=torch.int64, device=DEV)
    got = pks_fused.combine(p_t, c_t, buf[:, 2:2 + c], False)
    assert torch.equal(got, want)
    assert bool((buf[:, :2] == -3).all()) and bool((buf[:, 2 + c:] == -3).all())  # nothing outside the view
    sw = pks_fused.combine(p_t, c_t, torch.empty((r, c), dtype=torch.int64, device=DEV), True)
    assert torch.equal(sw, pks_gpu._round_mul_shift(want, PAR.q1, 56) % PAR.q1)
    assert np.array_equal(sw.cpu().numpy().view(np.uint64), pks.switch_hint(host, PAR))  # the host switch


def test_switch_edges_equal_host_switch() -> None:
    """Hint values at the rounding boundaries (``h q1 / 2^56`` at ``k + 1/2``), ``0`` and ``2^56 - 1`` (which rounds to
    ``q1`` and wraps to ``0``), through the fused switch alone: partials = the value's bytes, constant 0."""
    q1 = PAR.q1
    ks = np.array([0, 1, 2, q1 // 2, q1 - 2, q1 - 1], dtype=object)
    vals = [int(v) for k in ks for v in (((2 * k + 1) << 55) // q1, ((2 * k + 1) << 55) // q1 + 1) if v < 1 << 56]
    vals += [0, 1, M56, M56 - 1, 1 << 55, (1 << 55) - 1]
    h = np.array(vals, dtype=np.uint64)
    c = h.size
    part = np.stack([(h >> np.uint64(8 * d)) & np.uint64(255) for d in range(7)]).reshape(1, 7 * c).astype(np.int32)
    got = pks_fused.combine(torch.from_numpy(part).to(DEV), torch.zeros((1, 1), dtype=torch.int64, device=DEV),
                            torch.empty((1, c), dtype=torch.int64, device=DEV), True)
    assert np.array_equal(got.cpu().numpy().view(np.uint64)[0], pks.switch_hint(h, PAR))


def test_combine_rejects_bad_operands() -> None:
    p = torch.zeros((4, 7 * 8), dtype=torch.int32, device=DEV)
    c = torch.zeros((4, 1), dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError):
        pks_fused.combine(p.to(torch.int64), c, torch.empty((4, 8), dtype=torch.int64, device=DEV), True)
    with pytest.raises(ValueError):
        pks_fused.combine(p, c, torch.empty((4, 9), dtype=torch.int64, device=DEV), True)
    with pytest.raises(ValueError):
        pks_fused.combine(p, c, torch.empty((8, 4), dtype=torch.int64, device=DEV).t(), True)


@pytest.mark.parametrize(("seed", "n", "linear"), [(bytes(range(32)), RANK, "0.o"), (b"\x00" * 32, 3 * RANK, "27.down"),
                                                   (b"\xa5" * 32, 5 * RANK, "lm_head")])
def test_device_ring_rows_equal_host_stream(seed: bytes, n: int, linear: str) -> None:
    got = pks_fused.ring_rows(seed, n, linear, DEV)
    assert got.shape == (n // RANK, NU) and got.dtype == torch.int64
    assert np.array_equal(got.cpu().numpy().view(np.uint64), pks.public_ring_rows(seed, n, linear))


def test_device_ring_rows_reject_bad_sizes() -> None:
    with pytest.raises(ValueError):
        pks_fused.ring_rows(bytes(32), 24, "x", DEV)


@pytest.mark.parametrize("threads", ["1", "3"])
def test_host_ring_rows_threaded_equal_reference(monkeypatch: pytest.MonkeyPatch, threads: str) -> None:
    """The multi-threaded host stream (pieces of 64 ring rows at block-counter offsets) is the one-shot stream."""
    monkeypatch.setenv("PGW_SERVICE_PKS_HOST_THREADS", threads)
    seed = os.urandom(32)
    for n in (RANK, 64 * RANK, 150 * RANK):  # one piece, exactly one, three (the last partial)
        got = pks_gpu.host_ring_rows(seed, n, "3.gate_up")
        assert np.array_equal(got.view(np.uint64), pks.public_ring_rows(seed, n, "3.gate_up"))


def test_ring_rows_async_and_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = os.urandom(32)
    want = pks.public_ring_rows(seed, 2 * RANK, "1.o")
    for mode in ("torch", "fused"):
        monkeypatch.setenv("PGW_SERVICE_PKS_GPU_RING", mode)
        r = pks_gpu.ring_rows_async(seed, 2 * RANK, "1.o", DEV)
        t = r if isinstance(r, torch.Tensor) else r.result()
        assert np.array_equal(t.cpu().numpy().view(np.uint64), want), mode
        assert np.array_equal(pks_gpu.ring_rows(seed, 2 * RANK, "1.o", DEV).cpu().numpy().view(np.uint64), want)
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_RING", "torch")
    assert not isinstance(pks_gpu.ring_rows_async(seed, RANK, "1.o", DEV), torch.Tensor)


def _edge_rows(nb: int, seed: int) -> np.ndarray:
    hs = pks_gpu._edge_hint_rows(np.random.default_rng(seed), nb * PACK)
    if nb > 1:
        hs[PACK:2 * PACK] = (PAR.q1 + 1) // 2
        hs[-PACK:] = PAR.q1 - 1
    return hs


def test_forward4_equals_forward_and_reference() -> None:
    hs = _edge_rows(3, 2)
    got = pks_ntt.forward4(torch.from_numpy(hs.view(np.int64)).to(DEV)).cpu().numpy().astype(np.uint64)
    nb = hs.shape[0] // PACK
    at = hs.astype(np.int64).reshape(nb, PACK, RANK, D).transpose(0, 2, 3, 1).reshape(nb, RANK, NR)
    at = np.where(at >= (PAR.q1 + 1) // 2, at - PAR.q1, at)
    for k, p in enumerate(pks.QP):
        assert np.array_equal(got[k], pks.Ntt.of(p).fwd(np.mod(at, p).astype(np.uint64))), p


@pytest.mark.parametrize("groups", [1, 2])
def test_keyswitch_fwd4_equals_torch(groups: int) -> None:
    rng = np.random.default_rng(30 + groups)
    hs = _edge_rows(2, 30 + groups)
    y = rng.integers(0, 1 << 56, (hs.shape[0], D * groups), dtype=np.uint64)
    y[0, :3] = [0, (1 << 56) - 1, 1 << 55]
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP])
                    for _ in range(groups)])
    keys = pks_gpu._Keys(torch.from_numpy(ops).to(DEV))
    hs_t, y_t = torch.from_numpy(hs.view(np.int64)).to(DEV), torch.from_numpy(y.view(np.int64)).to(DEV)
    got = pks_ntt.keyswitch(hs_t, keys.u32(), y_t, fwd4=True)
    assert torch.equal(got.cpu(), pks_gpu._keyswitch_torch(hs_t, keys.t, y_t).cpu())


@pytest.mark.parametrize("arch", [75, 90])
def test_new_kernels_lower_for_cuda(arch: int) -> None:
    """The combine, ChaCha20 and 4-prime forward kernels compile to cubins for a T4 and an H100 with their launch
    signatures (no GPU, Triton's bundled ptxas), within the T4's 64 KiB of shared memory per block."""
    import json
    import subprocess
    import sys

    code = ("import json; from pgw.service import pks_fused, pks_ntt; s = {}; "
            f"a = pks_fused.compile_for({arch}, shared=s); b = pks_ntt.compile_for({arch}, shared=s); "
            "print(json.dumps([a, b, s]))")
    env = dict(os.environ, TRITON_INTERPRET="0")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900)  # noqa: S603
    if r.returncode and ("ptxas" in r.stderr.lower() or "No module named 'triton.backends" in r.stderr):
        pytest.skip(f"no CUDA compiler toolchain in this Triton: {r.stderr[-300:]}")
    assert r.returncode == 0, r.stderr[-3000:]
    a, b, shared = json.loads(r.stdout.strip().splitlines()[-1])
    assert set(a) == {"combine_switch", "combine", "chacha_ring"} and all(v > 0 for v in a.values())
    assert set(b) == {"fwd", "fwd4", "mac_inv", "tail"} and all(v > 0 for v in b.values())
    assert max(shared.values()) <= 64 * 1024, shared


# ---- self-tests and selection --------------------------------------------------------------------------------


def test_self_tests_pass() -> None:
    pks_gpu._switch_self_test(DEV)
    pks_gpu._ring_self_test(DEV)


@pytest.mark.slow
def test_fwd4_self_test_passes() -> None:
    pks_gpu._fwd4_self_test(DEV)


def test_self_tests_detect_one_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    real_c, real_r = pks_fused.combine, pks_fused.ring_rows

    def bad_combine(part: torch.Tensor, const: torch.Tensor, out: torch.Tensor, switch: bool) -> torch.Tensor:
        real_c(part, const, out, switch)
        out[0, 0] ^= 1
        return out

    def bad_ring(*a: object) -> torch.Tensor:
        t = real_r(*a)  # type: ignore[arg-type]
        t[-1, -1] ^= 1 << 40
        return t

    monkeypatch.setattr(pks_fused, "combine", bad_combine)
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._switch_self_test(DEV)
    monkeypatch.setattr(pks_fused, "ring_rows", bad_ring)
    with pytest.raises(RuntimeError, match="do not match"):
        pks_gpu._ring_self_test(DEV)
    real_f = pks_ntt.forward4
    memo: dict[tuple[int, int], torch.Tensor] = {}

    def good(hs: torch.Tensor) -> torch.Tensor:  # one interpreted launch per operand (the per-prime kernel's values)
        key = (hs.data_ptr(), int(hs.shape[0]))
        if key not in memo:
            memo[key] = real_f(hs)
        return memo[key]

    def bad_fwd4(hs: torch.Tensor) -> torch.Tensor:
        t = good(hs).clone()
        t.view(torch.int32)[2, 0, 0, 0] ^= 1  # (no uint32 xor on CUDA)
        return t

    monkeypatch.setattr(pks_ntt, "forward4", bad_fwd4)
    monkeypatch.setattr(pks_ntt, "forward", lambda hs, p: good(hs)[pks.QP.index(p)])
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._fwd4_self_test(DEV)


def test_choice_env_auto_and_permanent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def test(dev: object) -> None:
        calls.append(str(dev))

    ch = pks_gpu.FusedChoice("thing", "PGW_TEST_FUSED_THING", test)
    for raw, want in (("", "auto"), ("auto", "auto"), ("torch", "torch"), ("off", "torch"), ("host", "torch"),
                      ("fused", "fused"), ("on", "fused"), ("device", "fused")):
        monkeypatch.setenv("PGW_TEST_FUSED_THING", raw)
        assert ch.mode() == want
    monkeypatch.setenv("PGW_TEST_FUSED_THING", "fast")
    with pytest.raises(ValueError):
        ch.mode()
    monkeypatch.delenv("PGW_TEST_FUSED_THING")
    assert ch.impl("cpu") == "torch" and calls == []  # auto: CUDA devices only, no self-test
    assert ch.impl("cuda:3") == "fused" and calls == ["cuda:3"]  # self-test passed (the stub)
    assert ch.impl("cuda:3") == "fused" and calls == ["cuda:3"]  # once per device
    n = []

    def boom() -> int:
        n.append(1)
        raise RuntimeError("launch failed")

    assert ch.run("cuda:3", boom, lambda: 7) == 7  # an error falls back (same bits) ...
    assert ch.choice["cuda:3"] == "torch" and ch.run("cuda:3", boom, lambda: 8) == 8 and n == [1]  # ... for good
    bad = pks_gpu.FusedChoice("bad", "PGW_TEST_FUSED_THING", lambda dev: (_ for _ in ()).throw(RuntimeError("x")))
    assert bad.impl("cuda:0") == "torch"  # a failing self-test keeps the plain path
    monkeypatch.setenv("PGW_TEST_FUSED_THING", "fused")
    with pytest.raises(RuntimeError, match="launch failed"):
        bad.run("cuda:0", boom, lambda: 9)  # forced: errors propagate
    monkeypatch.setenv("PGW_TEST_FUSED_THING", "torch")
    assert bad.run("cuda:0", boom, lambda: 10) == 10


def test_fused_status_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    for e in CHOICES:
        monkeypatch.delenv(e, raising=False)
    assert pks_gpu.fused_status("cpu") == {"ntt": "torch", "fwd4": "torch", "switch": "torch", "ring": "torch",
                                           "wire": "torch", "keys": "torch"}
    # the 4-prime forward NTT is off unless set (slower on the T4); auto or fused enable it
    assert pks_gpu.FWD4.mode() == "torch"
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_FWD4", "auto")
    assert pks_gpu.FWD4.mode() == "auto"


# ---- the wire forms on the device ------------------------------------------------------------------------------


def test_reply_pack_device_and_host_equal_pack_bits() -> None:
    from pgw.hemask.wire import pack_bits

    rng = np.random.default_rng(40)
    groups, nb = 3, 4
    vals = rng.integers(0, 1 << 36, (groups, nb, NR), dtype=np.uint64)
    vals[0, 0, :6] = [0, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, (1 << 36) - 1]
    vals[2, 3] = (1 << 36) - 1
    want = bytes(pack_bits(vals.reshape(-1), 36))
    host = pks_gpu.PackedReply(groups, nb)
    for b0 in range(nb):  # row block by row block, as the item
        host.put_host(vals[:, b0:b0 + 1], b0, b0 + 1)
    assert bytes(host.buf) == want
    dev = pks_gpu.PackedReply(groups, nb)
    low, tail = pks_gpu._pack36_dev(torch.from_numpy(vals.view(np.int64)).to(DEV))
    dev.low[:] = low.cpu().numpy().view(np.uint32)
    dev.tail[:] = tail.cpu().numpy()
    assert bytes(dev.buf) == want


def test_upload_unpack_device_equals_unpack_bits() -> None:
    from pgw.hemask.wire import pack_bits

    rng = np.random.default_rng(41)
    n, cols = 32, 768
    b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
    b[0, :6] = [0, (1 << 56) - 1, (1 << 32) - 1, 1 << 32, (1 << 48) - 1, 1 << 48]
    up = pks_gpu.PackedUpload(pack_bits(b.reshape(-1), 56), n, cols)
    assert np.array_equal(up.host(), b)
    raw = pks_gpu._upload_bytes(up.data, DEV)
    for c0, c1 in ((0, 512), (512, 768), (100, 101)):
        got = pks_gpu._unpack56_cols(raw, n, cols, c0, c1)
        assert np.array_equal(got.cpu().numpy().view(np.uint64), b[:, c0:c1])
    with pytest.raises(ValueError):
        pks_gpu.PackedUpload(b"\0" * 10, n, cols)


def test_wire_self_test_passes_and_detects_one_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    pks_gpu._wire_self_test(DEV)
    real = pks_gpu._pack36_dev

    def bad(res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low, tail = real(res)
        tail[0, 0, 0] ^= 1
        return low, tail

    monkeypatch.setattr(pks_gpu, "_pack36_dev", bad)
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._wire_self_test(DEV)
    monkeypatch.setattr(pks_gpu, "_pack36_dev", real)
    real_u = pks_gpu._unpack56_cols
    monkeypatch.setattr(pks_gpu, "_unpack56_cols", lambda *a: real_u(*a) ^ (1 << 50))
    with pytest.raises(RuntimeError, match="does not match"):
        pks_gpu._wire_self_test(DEV)


@pytest.mark.parametrize("wire", ["torch", "fused"])
def test_item_wire_forms_equal_host_pack(monkeypatch: pytest.MonkeyPatch, wire: str) -> None:
    """``pks_gpu.item`` with the upload's wire bytes in and the reply's wire bytes out (packed by the device or on
    the host, row block by row block, also at a shard offset) equals ``pack_bits`` of the host path's reply."""
    from pgw.core.ringmm import cpu_int8_available
    from pgw.hemask import pks_native as pn
    from pgw.hemask.wire import pack_bits

    if INTERP and not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    for e in CHOICES:
        monkeypatch.setenv(e, "torch")
    monkeypatch.setenv("PGW_SERVICE_PKS_GPU_WIRE", wire)
    m, n, groups = 2 * PACK, 2 * RANK, 2
    w8, wpad, const = _w_const(m, n, n, 21)
    rng = np.random.default_rng(22)
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                    for _ in range(groups)]).astype(np.uint32)
    seed = os.urandom(32)
    up = pks_gpu.PackedUpload(pack_bits(b.reshape(-1), 56), n, D * groups)
    pr = pks_gpu.PackedReply(groups, 2 * m // PACK)  # two shards' worth: this item fills the second half
    work = pks_gpu._row_bytes(n, D * groups, groups) * PACK  # one reply ciphertext per row block
    got = pks_gpu.item(wpad.to(DEV), const.to(DEV), m, n, pks_gpu.ring_rows(seed, n, "w", DEV), up, ops,
                       work_bytes=work, out=pr, out_b0=m // PACK)
    assert got is pr
    a = pks.a_mlwe_rows(pks.public_ring_rows(seed, n, "w"))
    m56 = np.uint64(M56)
    want = pn.keyswitch(pks.switch_hint(matmul_mod64(w8, a) & m56, PAR), ops, matmul_mod64(w8, b) & m56)
    full = np.zeros((groups, 2 * m // PACK, NR), dtype=np.uint64)
    full[:, m // PACK:] = want
    assert bytes(pr.buf) == bytes(pack_bits(full.reshape(-1), 36))  # the first shard's rows untouched (zero)


def test_switch_fallback_keeps_the_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gemm_into`` under ``auto`` with the fused combine chosen: an error falls back to torch for good."""
    from pgw.service.pks_gpu import SWITCH

    monkeypatch.delenv("PGW_SERVICE_PKS_GPU_SWITCH", raising=False)
    rng = np.random.default_rng(5)
    n, rows = 64, 32
    w = torch.from_numpy(rng.integers(-128, 128, (rows, n), dtype=np.int64).astype(np.int8)).to(DEV)
    key = str(w.device)  # "cuda:0", not "cuda"
    monkeypatch.setattr(SWITCH, "choice", {key: "fused"})
    pl = pks_gpu.planes_of(torch.from_numpy(rng.integers(0, 1 << 56, (n, 256), dtype=np.int64)).to(DEV), n)
    const = torch.from_numpy(rng.integers(-(1 << 62), 1 << 62, (rows, 1), dtype=np.int64)).to(DEV)
    want = pks_gpu._round_mul_shift(pks_gpu.gemm_mod56(w, pl, const), PAR.q1, 56) % PAR.q1
    out = torch.empty((rows, 256), dtype=torch.int64, device=DEV)
    pks_gpu.gemm_into(w, pl, const, out, True)
    assert torch.equal(out, want)

    def boom(*a: object) -> None:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(pks_fused, "combine", boom)
    out.zero_()
    pks_gpu.gemm_into(w, pl, const, out, True)
    assert torch.equal(out, want) and SWITCH.choice[key] == "torch"


# ---- whole items -----------------------------------------------------------------------------------------------


def _w_const(m: int, n: int, np_: int, seed: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    w8 = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w8[0], w8[-1] = -128, 127
    wpad = torch.zeros((m, np_), dtype=torch.int8)
    wpad[:, :n] = torch.from_numpy(w8)
    scale = (128 * sum(1 << (8 * d) for d in range(7))) % (1 << 64)
    const = torch.from_numpy((w8.astype(np.int64).sum(1).view(np.uint64) * np.uint64(scale)).view(np.int64)
                             .reshape(m, 1).copy())
    return w8, wpad, const


def test_item_all_fused_equals_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pks_gpu.item`` with every fused kernel forced (device ring rows, fused switch and recombination, 4-prime
    forward NTT, fused NTTs) equals the host path (numpy GEMM, ``pks.switch_hint``, CPU key switch); 2 groups, 2 row
    blocks, a zero-padded weight."""
    from pgw.core.ringmm import cpu_int8_available
    from pgw.hemask import pks_native as pn

    if INTERP and not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    for e in CHOICES:
        monkeypatch.setenv(e, "fused")
    m, n, groups = 2 * PACK, 3 * RANK, 2
    w8, wpad, const = _w_const(m, n, 64, 11)
    rng = np.random.default_rng(12)
    a_seed = os.urandom(32)
    b = rng.integers(0, 1 << 56, (n, D * groups), dtype=np.uint64)
    b[0] = (1 << 56) - 1
    ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                    for _ in range(groups)]).astype(np.uint32)
    ring = pks_gpu.ring_rows_async(a_seed, n, "5.up", DEV)
    assert isinstance(ring, torch.Tensor)  # generated on the device
    work = pks_gpu._row_bytes(64, D * groups, groups) * PACK  # one reply ciphertext per row block
    got = pks_gpu.item(wpad.to(DEV), const.to(DEV), m, n, ring, b, ops, work_bytes=work)
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, n, "5.up"))
    m56 = np.uint64(M56)
    want = pn.keyswitch(pks.switch_hint(matmul_mod64(w8, a) & m56, PAR), ops, matmul_mod64(w8, b) & m56)
    assert np.array_equal(got, want)


def test_provider_all_fused_equals_cache_and_decrypts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through ``PKSProvider``: the device path with every fused kernel forced returns the cache path's bytes, and the
    customer decrypts them to ``W8 r mod 2^32``."""
    from pgw.core.ringmm import cpu_int8_available

    if INTERP and not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    if DEV.type == "cuda":  # the helper's kernels live on the CPU; on CUDA this is the bench's job (``items``)
        pytest.skip("CPU kernels only")
    from tests.service.test_pks_device_item import _provider_round_trip, _weight

    for e in CHOICES:
        monkeypatch.setenv(e, "fused")
    _provider_round_trip(_weight(64, 48, 8), "0.o", 300, 8)


def test_item_cpu_ring_future_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the host stream (``RING=torch``) the item takes a future and resolves it after queuing the upload."""
    from pgw.core.ringmm import cpu_int8_available

    if INTERP and not cpu_int8_available():
        pytest.skip("needs exact CPU int8 GEMMs (torch._int_mm)")
    for e in CHOICES:
        monkeypatch.setenv(e, "torch")
    m, n = PACK, 2 * RANK
    _, wpad, const = _w_const(m, n, n, 3)
    rng = np.random.default_rng(4)
    b = rng.integers(0, 1 << 56, (n, D), dtype=np.uint64)
    ops = np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])[None].astype(np.uint32)
    seed = os.urandom(32)
    fut = pks_gpu.ring_rows_async(seed, n, "x", DEV)
    assert not isinstance(fut, torch.Tensor)
    got = pks_gpu.item(wpad.to(DEV), const.to(DEV), m, n, fut, b, ops)
    assert np.array_equal(got, pks_gpu.item(wpad.to(DEV), const.to(DEV), m, n,
                                            pks_gpu.ring_rows(seed, n, "x", DEV), b, ops))


# ---- memplan: the throughput model of the device item, calibrated on the Kaggle T4 ---------------------------

#: whole items measured on the Kaggle T4 (``results/kaggle/pks-ntt-t4-r2``, ``items``): (m, n, before s, after s),
#: before = the fused NTTs only, after = every kernel of this round but the device wire forms (packed on the host)
T4_ITEMS = {"7b gate_up": (37888, 3584, 1.698, 1.199), "7b down": (3584, 18944, 0.581, 0.578),
            "7b head": (152064, 3584, 6.616, 4.689), "32b gate_up": (55296, 5120, 2.572, 1.908),
            "32b down": (5120, 27648, 0.975, 0.877), "72b gate_up": (59136, 8192, 3.122, 2.541),
            "72b down": (8192, 29568, 1.402, 1.273)}


def test_memplan_reproduces_the_t4_items() -> None:
    from pgw.service import memplan as mp

    for name, (m, n, before, after) in T4_ITEMS.items():
        prev = mp.pks_item_seconds(m, n, 1024, "device", mp.T4_RATES, ntt="fused", kernels="ntt")
        assert abs(sum(prev.values()) / before - 1) < 0.25, (name, sum(prev.values()), before)
        new = mp.pks_item_seconds(m, n, 1024, "device", mp.T4_RATES, ntt="fused", kernels="r2")
        new.pop("reply_pack_device")  # the run packed on the host
        new["wire_host"], new["reply_d2h"] = prev["wire_host"], prev["reply_d2h"]
        assert abs(sum(new.values()) / after - 1) < 0.3, (name, sum(new.values()), after)


def test_memplan_all_kernels(capsys: pytest.CaptureFixture[str]) -> None:
    from pgw.service import memplan as mp

    prev = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="fused", kernels="ntt")
    new = mp.pks_item_seconds(59136, 8192, 1024, "device", ntt="fused", kernels="all")
    assert "wire_host" in prev and "wire_host" not in new and "ring_rows_host" not in new
    assert new["hint_gemm_switch"] < prev["hint_gemm_switch"] / 3
    assert new["key_switch"] == prev["key_switch"]  # the per-prime forward NTT stays the default
    f4 = mp.pks_item_seconds(59136, 8192, 1024, "device", mp.T4_RATES, ntt="fused", kernels="all", fwd4=True)
    t4 = mp.pks_item_seconds(59136, 8192, 1024, "device", mp.T4_RATES, ntt="fused", kernels="all")
    assert f4["key_switch"] > t4["key_switch"]  # the 4-prime forward is slower on the T4
    with pytest.raises(ValueError, match="unknown kernels"):
        mp.pks_item_seconds(64, 64, 256, "device", kernels="most")
    for model in ("qwen2.5-7b", "qwen2.5-72b"):
        a = mp.pks_throughput(model, "device")
        b = mp.pks_throughput(model, "device", mp.MemConfig(pks_kernels="ntt"))
        assert a["kernels"] == "all" and b["kernels"] == "ntt"
        assert a["slots_per_s"] > 5 * b["slots_per_s"]  # type: ignore[operator]
    assert mp.main(["--pks-throughput", "--model", "qwen2.5-72b", "--pks-gpu", "t4", "--pks-kernels", "ntt"]) == 0
    out = capsys.readouterr().out
    assert "Kaggle T4" in out and "device-item kernels: ntt" in out
