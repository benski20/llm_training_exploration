#!/usr/bin/env python3
"""Fused Triton kernels of the pks-v3 provider device item (``pgw.service.pks_ntt``, ``pgw.service.pks_fused``)
against the torch / host paths, on one GPU.

Sections (one JSON object, ``--json``):

* ``exact``: the fused transforms equal :class:`pgw.hemask.pks.Ntt` on every prime (random and edge-case
  polynomials), the fused key switch equals the torch key switch and the CPU key switch
  (:func:`pgw.hemask.pks_native.keyswitch`) for 1 and 2 groups (the one-group path fixed after the first T4 run),
  and a larger random key switch equals torch on the device;
* ``exact_new``: the fused recombination + hint switch equals the torch path and the host ``pks.switch_hint`` (edge
  partials, rounding boundaries), the device ChaCha20 ring rows equal ``pks.public_ring_rows`` (up to 72B's
  29568 inputs), the threaded host stream equals it too, the 4-prime forward NTT equals the per-prime one and
  ``pks.Ntt``, the key switch with it equals torch, and every self-test passes (``fused_status`` under ``auto``);
* ``ntt_us``: microseconds per 8192-point transform (fused forward incl. the gather, 4-prime forward, fused MAC +
  inverse; torch forward / inverse) at several ``num_warps``;
* ``keyswitch``: microseconds per output row of the whole key switch (torch, fused, fused with the 4-prime forward)
  for 1 and 4 groups;
* ``stages``: per-stage seconds of the device item at 72B's gate_up shape (8192 inputs, 1024 columns) on a block
  of output rows, before (torch / host / pageable) and after (fused / device / pinned): ring rows, the upload and
  its wire unpack, the hint operand's planes, the GEMMs, the recombination + switch, the key switch, the reply copy
  and its wire pack; scaled to a 1024-slot cycle of every linear of 7B / 32B / 72B;
* ``round3``: the reply keys unpacked and range-checked on the device against the host unpack (exactness, residues
  out of range flagged, seconds per 4-group item) and the staged uploads (one thread against the thread pool's
  overlapped pieces, GB/s at 72B's upload sizes, next to the raw pinned transfer and the host copy rates);
* ``items``: one pks-v3 item (1024 columns) of the largest linears of 7B / 32B / 72B through ``ComputeKernel`` +
  ``PKSProvider`` (device hint path) in four configurations (``--configs``): ``torch`` (every kernel plain),
  ``prev`` (the fused NTTs only), ``r2`` (the second round: host reply keys, one-thread staging) and ``fused``
  (every device step under ``auto``); seconds and byte-identical replies;
* ``cycle``: key-switch seconds of one 1024-slot decode cycle per model (every linear), from the per-row times.

Runs on CPU too (``--cpu``, Triton interpreter, tiny shapes): a smoke test of the script only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

if "--cpu" in sys.argv:  # Triton reads it at import
    os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import torch

from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.pks import NR, NU, PACK, RANK, D, PKSParams
from pgw.service import memplan, pks_fused, pks_gpu, pks_ntt

PAR = PKSParams()
M56 = (1 << 56) - 1
LINEARS = {
    "qwen2.5-7b": [("0.gate_up", 37888, 3584), ("0.down", 3584, 18944), ("28.head", 152064, 3584)],
    "qwen2.5-32b": [("0.gate_up", 55296, 5120), ("0.down", 5120, 27648)],
    "qwen2.5-72b": [("0.gate_up", 59136, 8192), ("0.down", 8192, 29568)],
}
CHOICES = ("PGW_SERVICE_PKS_GPU_NTT", "PGW_SERVICE_PKS_GPU_FWD4", "PGW_SERVICE_PKS_GPU_SWITCH",
           "PGW_SERVICE_PKS_GPU_RING", "PGW_SERVICE_PKS_PINNED", "PGW_SERVICE_PKS_HOST_THREADS",
           "PGW_SERVICE_PKS_GPU_WIRE", "PGW_SERVICE_PKS_GPU_KEYS")
#: item configurations: every kernel plain; the fused NTTs only (before the second round); every kernel of the second
#: round under auto but the reply keys unpacked on the host and the uploads staged by one thread (``r2``: before the
#: third round); every device step under auto
_PKS_PLAIN = {"PGW_SERVICE_PKS_GPU_FWD4": "torch", "PGW_SERVICE_PKS_GPU_SWITCH": "torch",
              "PGW_SERVICE_PKS_GPU_RING": "torch", "PGW_SERVICE_PKS_GPU_WIRE": "torch",
              "PGW_SERVICE_PKS_GPU_KEYS": "torch", "PGW_SERVICE_PKS_PINNED": "0", "PGW_SERVICE_PKS_HOST_THREADS": "1"}
CONFIGS = {
    "torch": {**_PKS_PLAIN, "PGW_SERVICE_PKS_GPU_NTT": "torch"},
    "prev": _PKS_PLAIN,
    "r2": {"PGW_SERVICE_PKS_GPU_KEYS": "torch", "PGW_SERVICE_PKS_HOST_THREADS": "1"},
    "fused": {},
}


def _set_env(cfg: dict[str, str]) -> None:
    for k in CHOICES:
        os.environ.pop(k, None)
    os.environ.update(cfg)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn: Any, dev: torch.device, reps: int) -> float:
    """Median seconds of ``fn()`` over ``reps`` runs after one warm-up."""
    fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _keys(rng: np.random.Generator, groups: int) -> np.ndarray:
    return np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.int64) for p in pks.QP])
                     for _ in range(groups)])


def _centred(hs: np.ndarray) -> np.ndarray:
    nb = hs.shape[0] // PACK
    at = hs.astype(np.int64).reshape(nb, PACK, RANK, D).transpose(0, 2, 3, 1).reshape(nb, RANK, NR)
    return np.where(at >= (PAR.q1 + 1) // 2, at - PAR.q1, at)


def _all_true(d: dict[str, Any]) -> bool:
    return all(v is True or (isinstance(v, list) and all(x is True for x in v)) for k, v in d.items()
               if k not in ("all", "fused_status", "seconds"))


def exact(dev: torch.device, big_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(1)
    hs = pks_gpu._edge_hint_rows(rng, 3 * PACK)
    hs[PACK:2 * PACK] = (PAR.q1 + 1) // 2
    hs[-PACK:] = PAR.q1 - 1
    hs_t = torch.from_numpy(hs.view(np.int64)).to(dev)
    nb = hs.shape[0] // PACK
    at = _centred(hs)
    ok_f, ok_i = [], []
    for p in pks.QP:
        nt = pks.Ntt.of(p)
        ah = pks_ntt.forward(hs_t, p)
        ref = nt.fwd(np.mod(at, p).astype(np.uint64))
        ok_f.append(bool(np.array_equal(ah.cpu().numpy().astype(np.uint64), ref)))
        ops = rng.integers(0, p, (2, RANK, NR), dtype=np.uint64)
        ops[0] = p - 1
        ops[1, :, :4] = [0, 1, p - 1, p - 2]
        got = pks_ntt.mac_inverse(ah, torch.from_numpy(ops.astype(np.uint32)).to(dev), p).cpu().numpy()
        acc = np.zeros((2, nb, NR), dtype=np.uint64)
        for li in range(RANK):
            acc = (acc + ref[None, :, li] * ops[:, None, li] % np.uint64(p)) % np.uint64(p)
        ok_i.append(bool(np.array_equal(got.astype(np.uint64), nt.inv(acc))))
    out["fwd_equals_pks_ntt"] = ok_f
    out["mac_inv_equals_pks_ntt"] = ok_i
    for groups in (1, 2):  # one group: the path the first T4 run could not compile
        y = rng.integers(0, 1 << 56, (hs.shape[0], D * groups), dtype=np.uint64)
        y[0, :3] = [0, (1 << 56) - 1, 1 << 55]
        ops = _keys(rng, groups)
        keys = pks_gpu._Keys(torch.from_numpy(ops).to(dev))
        y_t = torch.from_numpy(y.view(np.int64)).to(dev)
        tq = pks_gpu._keyswitch_torch(hs_t, keys.t, y_t).cpu()
        cpu = pn.keyswitch(hs, ops.astype(np.uint32), y)
        for f4 in (False, True):
            fused = pks_ntt.keyswitch(hs_t, keys.u32(), y_t, fwd4=f4).cpu()
            tag = f"keyswitch_g{groups}{'_fwd4' if f4 else ''}"
            out[f"{tag}_fused_equals_torch"] = bool(torch.equal(fused, tq))
            out[f"{tag}_fused_equals_cpu"] = bool(np.array_equal(fused.numpy().astype(np.uint64), cpu))
        k1 = pks_gpu._Keys(keys.t[:1])  # one group, one reply ciphertext: grid dimensions of 1
        out[f"keyswitch_g{groups}_1ct_g1"] = bool(torch.equal(
            pks_ntt.keyswitch(hs_t[:PACK], k1.u32(), y_t[:PACK, :D]).cpu(),
            pks_gpu._keyswitch_torch(hs_t[:PACK], k1.t, y_t[:PACK, :D]).cpu()))
    try:
        pks_gpu.fused_self_test(dev)
        out["fused_self_test"] = True
    except Exception as e:
        out["fused_self_test"] = repr(e)
    hs_b = torch.from_numpy(rng.integers(0, PAR.q1, (big_rows, NU), dtype=np.int64)).to(dev)
    y_b = torch.from_numpy(rng.integers(0, 1 << 56, (big_rows, 4 * D), dtype=np.int64)).to(dev)
    kb = pks_gpu._Keys(torch.from_numpy(_keys(rng, 4)).to(dev))
    ref_b = pks_gpu._keyswitch_torch(hs_b, kb.t, y_b)
    out[f"keyswitch_{big_rows}rows_g4_fused_equals_torch"] = bool(torch.equal(pks_ntt.keyswitch(hs_b, kb.u32(), y_b),
                                                                             ref_b))
    out[f"keyswitch_{big_rows}rows_g4_fwd4_equals_torch"] = bool(torch.equal(
        pks_ntt.keyswitch(hs_b, kb.u32(), y_b, fwd4=True), ref_b))
    out["all"] = _all_true(out)
    return out


def _partials(rng: np.random.Generator, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    part = rng.integers(-(1 << 31), 1 << 31, (r, 7 * c), dtype=np.int64)
    part[0] = (1 << 31) - 1
    part[1] = -(1 << 31)
    part[2, ::2] = 0
    const = rng.integers(-(1 << 63), (1 << 63) - 1, (r, 1), dtype=np.int64)
    const[:4, 0] = [-1, 0, -(1 << 63), (1 << 63) - 1]
    return part.astype(np.int32), const


def exact_new(dev: torch.device, small: bool) -> dict[str, Any]:
    from pgw.service.kernels import _combine_planes, _combine_planes_np

    out: dict[str, Any] = {}
    rng = np.random.default_rng(21)
    for r, c in ((64, 300), (2048 if not small else 32, 512), (1, 256)):
        part, const = _partials(rng, max(r, 4), c)
        part, const = part[:r], const[:r]
        p_t, c_t = torch.from_numpy(part).to(dev), torch.from_numpy(const).to(dev)
        want = _combine_planes([p_t[:, d * c:(d + 1) * c] for d in range(7)], c_t).bitwise_and_(M56)
        host = _combine_planes_np([part[:, d * c:(d + 1) * c] for d in range(7)], const.view(np.uint64)) \
            & np.uint64(M56)
        got = pks_fused.combine(p_t, c_t, torch.empty((r, c), dtype=torch.int64, device=dev), False)
        sw = pks_fused.combine(p_t, c_t, torch.empty((r, c), dtype=torch.int64, device=dev), True)
        out[f"combine_{r}x{c}_equals_torch"] = bool(torch.equal(got, want))
        out[f"combine_{r}x{c}_equals_numpy"] = bool(np.array_equal(got.cpu().numpy().view(np.uint64), host))
        out[f"switch_{r}x{c}_equals_torch"] = bool(torch.equal(sw, pks_gpu._round_mul_shift(want, PAR.q1, 56)
                                                               % PAR.q1))
        out[f"switch_{r}x{c}_equals_host_switch_hint"] = bool(np.array_equal(sw.cpu().numpy().view(np.uint64),
                                                                             pks.switch_hint(host, PAR)))
    q1 = PAR.q1
    vals = [v for k in (0, 1, 2, q1 // 2, q1 - 2, q1 - 1) for v in (((2 * k + 1) << 55) // q1,
                                                                    ((2 * k + 1) << 55) // q1 + 1) if v < 1 << 56]
    vals += [0, 1, M56, M56 - 1, 1 << 55, (1 << 55) - 1]
    h = np.array(vals, dtype=np.uint64)
    part = np.stack([(h >> np.uint64(8 * d)) & np.uint64(255) for d in range(7)]).reshape(1, -1).astype(np.int32)
    sw = pks_fused.combine(torch.from_numpy(part).to(dev), torch.zeros((1, 1), dtype=torch.int64, device=dev),
                           torch.empty((1, h.size), dtype=torch.int64, device=dev), True)
    out["switch_rounding_boundaries_equal_host"] = bool(np.array_equal(sw.cpu().numpy().view(np.uint64)[0],
                                                                       pks.switch_hint(h, PAR)))
    ring_ok = []
    for n in ((16, 48) if small else (16, 3584, 8192, 29568)):
        seed = os.urandom(32)
        ref = pks.public_ring_rows(seed, n, f"{n}.down")
        ring_ok.append(bool(np.array_equal(pks_fused.ring_rows(seed, n, f"{n}.down", dev).cpu().numpy()
                                           .view(np.uint64), ref)))
        ring_ok.append(bool(np.array_equal(pks_gpu.host_ring_rows(seed, n, f"{n}.down").view(np.uint64), ref)))
    out["ring_rows_device_and_threaded_host_equal_public_ring_rows"] = ring_ok
    hs = pks_gpu._edge_hint_rows(rng, (3 if small else 64) * PACK)
    hs[PACK:2 * PACK] = (PAR.q1 + 1) // 2
    hs[-PACK:] = PAR.q1 - 1
    hs_t = torch.from_numpy(hs.view(np.int64)).to(dev)
    f4 = pks_ntt.forward4(hs_t)
    at = _centred(hs)
    out["fwd4_equals_forward"] = [bool(torch.equal(f4[k], pks_ntt.forward(hs_t, p))) for k, p in enumerate(pks.QP)]
    out["fwd4_equals_pks_ntt"] = [bool(np.array_equal(f4[k].cpu().numpy().astype(np.uint64),
                                                      pks.Ntt.of(p).fwd(np.mod(at, p).astype(np.uint64))))
                                  for k, p in enumerate(pks.QP)]
    for name, fn in (("fwd4_self_test", pks_gpu._fwd4_self_test), ("switch_self_test", pks_gpu._switch_self_test),
                     ("ring_self_test", pks_gpu._ring_self_test), ("item_self_test", pks_gpu.item_self_test)):
        try:
            fn(dev)
            out[name] = True
        except Exception as e:
            out[name] = repr(e)
    _set_env({})
    out["fused_status"] = pks_gpu.fused_status(dev)
    # every kernel ``auto`` chose on CUDA (the 4-prime forward NTT is off unless set: slower on the T4)
    chosen = {k: v for k, v in out["fused_status"].items() if k != "fwd4" or pks_gpu.FWD4.mode() != "torch"}
    out["all"] = _all_true(out) and (dev.type != "cuda" or set(chosen.values()) == {"fused"})
    return out


def ntt_us(dev: torch.device, polys: int, reps: int, warps: list[int]) -> dict[str, Any]:
    """Microseconds per transform."""
    rng = np.random.default_rng(2)
    nb = max(1, polys // RANK)
    hs = torch.from_numpy(rng.integers(0, PAR.q1, (nb * PACK, NU), dtype=np.int64)).to(dev)
    out: dict[str, Any] = {"polys": nb * RANK}
    p = pks.QP[2]
    groups = 4
    ops_u = pks_gpu._Keys(torch.from_numpy(_keys(rng, groups)).to(dev)).u32()
    y = torch.from_numpy(rng.integers(0, 1 << 56, (nb * PACK, D * groups), dtype=np.int64)).to(dev)
    for w in warps:
        os.environ["PGW_SERVICE_PKS_NTT_WARPS"] = str(w)
        t0 = time.perf_counter()
        ah = pks_ntt.forward(hs, p)
        pks_ntt.forward4(hs)
        pks_ntt.mac_inverse(ah, ops_u[2], p)
        _sync(dev)
        comp = time.perf_counter() - t0
        tf = _time(lambda: pks_ntt.forward(hs, p), dev, reps)
        tf4 = _time(lambda: pks_ntt.forward4(hs), dev, reps)
        ti = _time(lambda a=ah: pks_ntt.mac_inverse(a, ops_u[2], p), dev, reps)
        tk = _time(lambda: pks_ntt.keyswitch(hs, ops_u, y), dev, reps)
        tk4 = _time(lambda: pks_ntt.keyswitch(hs, ops_u, y, fwd4=True), dev, reps)
        out[f"warps_{w}"] = {"first_call_s": round(comp, 2), "fwd_us": round(tf / (nb * RANK) * 1e6, 3),
                             "fwd4_us": round(tf4 / (4 * nb * RANK) * 1e6, 3),
                             "mac_inv_us": round(ti / (nb * groups) * 1e6, 3),
                             "keyswitch_g4_us_per_row": round(tk / (nb * PACK) * 1e6, 3),
                             "keyswitch_g4_fwd4_us_per_row": round(tk4 / (nb * PACK) * 1e6, 3)}
    os.environ.pop("PGW_SERVICE_PKS_NTT_WARPS", None)
    z, zi, ninv = pks_gpu._dev_tables(p, dev)
    x = (hs.view(nb, PACK, RANK, D).permute(0, 2, 3, 1).reshape(nb, RANK, NR) % p)
    tf = _time(lambda: pks_gpu._fwd(x, p, z), dev, reps)
    acc = pks_gpu._fwd(x, p, z)[:, :groups].contiguous()
    ti = _time(lambda: pks_gpu._inv(acc, p, zi, ninv), dev, reps)
    out["torch"] = {"fwd_us": round(tf / (nb * RANK) * 1e6, 3), "inv_us": round(ti / (nb * groups) * 1e6, 3)}
    return out


def keyswitch_rows(dev: torch.device, rows: int, reps: int) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    hs = torch.from_numpy(rng.integers(0, PAR.q1, (rows, NU), dtype=np.int64)).to(dev)
    out: dict[str, Any] = {"rows": rows}
    for groups in (1, 4):
        y = torch.from_numpy(rng.integers(0, 1 << 56, (rows, D * groups), dtype=np.int64)).to(dev)
        keys = pks_gpu._Keys(torch.from_numpy(_keys(rng, groups)).to(dev))
        ops_u = keys.u32()
        tt = _time(lambda k=keys, y=y: pks_gpu._keyswitch_torch(hs, k.t, y), dev, reps)
        tf = _time(lambda o=ops_u, y=y: pks_ntt.keyswitch(hs, o, y), dev, reps)
        tf4 = _time(lambda o=ops_u, y=y: pks_ntt.keyswitch(hs, o, y, fwd4=True), dev, reps)
        ref = pks_gpu._keyswitch_torch(hs, keys.t, y)
        same = bool(torch.equal(ref, pks_ntt.keyswitch(hs, ops_u, y)))
        same4 = bool(torch.equal(ref, pks_ntt.keyswitch(hs, ops_u, y, fwd4=True)))
        out[f"g{groups}"] = {"torch_us_per_row": round(tt / rows * 1e6, 3),
                             "fused_us_per_row": round(tf / rows * 1e6, 3),
                             "fused_fwd4_us_per_row": round(tf4 / rows * 1e6, 3), "speedup": round(tt / tf, 2),
                             "identical": same and same4}
    return out


def stages(dev: torch.device, rows: int, reps: int, small: bool) -> dict[str, Any]:
    """Per-stage seconds of the device item at 72B's gate_up shape on ``rows`` output rows, before and after; unit
    costs (per output row, per input feature, per upload value) and their sums over a 1024-slot cycle per model."""
    from pgw.service.kernels import _combine_planes

    n, cols = (64, 256) if small else (8192, 1024)
    groups = cols // D
    rng = np.random.default_rng(5)
    seed = os.urandom(32)
    w = torch.randint(-128, 128, (rows, n), dtype=torch.int8, generator=torch.Generator().manual_seed(1)).to(dev)
    const = torch.from_numpy(rng.integers(-(1 << 62), 1 << 62, (rows, 1), dtype=np.int64)).to(dev)
    b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
    out: dict[str, Any] = {"rows": rows, "n": n, "cols": cols}
    t: dict[str, dict[str, float]] = {}
    # per input feature: the ring rows (n/16 x 4096 values)
    t["ring_rows"] = {"before_host_1thread": _time(lambda: torch.from_numpy(pks.public_ring_rows(seed, n, "x")
                                                                            .view(np.int64)).to(dev), dev, reps),
                      "host_threaded": _time(lambda: pks_gpu._upload(pks_gpu.host_ring_rows(seed, n, "x"), dev),
                                             dev, reps),
                      "after_device": _time(lambda: pks_fused.ring_rows(seed, n, "x", dev), dev, reps)}
    # per upload value: b, column block by column block (as pks_gpu.item)
    def up(pinned: bool) -> None:
        os.environ["PGW_SERVICE_PKS_PINNED"] = "1" if pinned else "0"
        for c0 in range(0, cols, pks_gpu.COL_BLOCK):
            pks_gpu._upload(b[:, c0:c0 + pks_gpu.COL_BLOCK], dev)

    from pgw.hemask.wire import pack_bits, unpack_bits

    packed_b = pack_bits(b.reshape(-1), 56)

    def up_raw() -> torch.Tensor:  # the upload's wire bytes (7 per value), pinned
        os.environ["PGW_SERVICE_PKS_PINNED"] = "1"
        return pks_gpu._upload_bytes(packed_b, dev)

    t["upload_b"] = {"before_pageable": _time(lambda: up(False), dev, reps),
                     "pinned_int64": _time(lambda: up(True), dev, reps),
                     "after_pinned_wire_bytes": _time(up_raw, dev, reps)}
    raw = up_raw()
    t["wire_unpack_b"] = {"before_host": _time(lambda: unpack_bits(packed_b, n * cols, 56), dev, reps),
                          "after_device": _time(lambda: [pks_gpu._unpack56_cols(raw, n, cols, c0, c0 + 512)
                                                         for c0 in range(0, cols, 512)], dev, reps)}
    os.environ.pop("PGW_SERVICE_PKS_PINNED", None)
    ring_t = pks_fused.ring_rows(seed, n, "x", dev) if dev.type == "cuda" else pks_gpu.ring_rows(seed, n, "x", dev)
    step = pks_gpu.COL_BLOCK // D

    def planes() -> list[torch.Tensor]:
        src = pks_gpu._a_source(ring_t)
        return [pks_gpu.planes_of(pks_gpu.a_columns(src, l0, l0 + step), n) for l0 in range(0, RANK, step)]

    t["a_planes"] = {"same": _time(planes, dev, reps)}
    apl = planes()
    bpl = [pks_gpu.planes_of(torch.from_numpy(b[:, c0:c0 + 512].copy().view(np.int64)).to(dev), n)
           for c0 in range(0, cols, 512)]

    def gemms(pls: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch._int_mm(w, p.view(torch.int8).view(-1, n).t()) for p in pls]

    t["hint_gemm"] = {"same": _time(lambda: gemms(apl), dev, reps)}
    t["y_gemm"] = {"same": _time(lambda: gemms(bpl), dev, reps)}
    parts_a, parts_b = gemms(apl), gemms(bpl)
    hs = torch.empty((rows, NU), dtype=torch.int64, device=dev)
    y = torch.empty((rows, cols), dtype=torch.int64, device=dev)

    def comb(parts: list[torch.Tensor], dst: torch.Tensor, switch: bool, fused: bool) -> None:
        for j, part in enumerate(parts):
            c = part.shape[1] // 7
            o = dst[:, j * c:(j + 1) * c]
            if fused:
                pks_fused.combine(part, const, o, switch)
            else:
                h = _combine_planes([part[:, d * c:(d + 1) * c] for d in range(7)], const).bitwise_and_(M56)
                o.copy_(pks_gpu._round_mul_shift(h, PAR.q1, 56) % PAR.q1 if switch else h)

    t["hint_switch_recombination"] = {"before_torch": _time(lambda: comb(parts_a, hs, True, False), dev, reps),
                                      "after_fused": _time(lambda: comb(parts_a, hs, True, True), dev, reps)}
    t["y_recombination"] = {"before_torch": _time(lambda: comb(parts_b, y, False, False), dev, reps),
                            "after_fused": _time(lambda: comb(parts_b, y, False, True), dev, reps)}
    ref_h = hs.clone()
    comb(parts_a, hs, True, False)
    same = bool(torch.equal(ref_h, hs))
    keys = pks_gpu._Keys(torch.from_numpy(_keys(rng, groups)).to(dev))
    u32 = keys.u32()
    if small:
        t["key_switch"] = {"before_fused": 0.0, "fwd4_fused": 0.0}
    else:
        t["key_switch"] = {"before_fused": _time(lambda: pks_ntt.keyswitch(hs, u32, y), dev, reps),
                           "fwd4_fused": _time(lambda: pks_ntt.keyswitch(hs, u32, y, fwd4=True), dev, reps)}
    res = torch.randint(0, 1 << 36, (groups, rows // PACK, NR), dtype=torch.int64, device=dev)
    dst = np.empty((groups, rows // PACK, NR), dtype=np.uint64)

    def back(pinned: bool) -> None:
        os.environ["PGW_SERVICE_PKS_PINNED"] = "1" if pinned else "0"
        cp = pks_gpu._ReplyCopies(dst, dev)
        cp.put(res, 0, rows // PACK)
        cp.finish()

    def back_packed() -> None:  # packed by the device into the reply's wire buffer (4.5 bytes per value back)
        os.environ["PGW_SERVICE_PKS_PINNED"] = "1"
        os.environ["PGW_SERVICE_PKS_GPU_WIRE"] = "fused"
        cp = pks_gpu._ReplyCopies(pks_gpu.PackedReply(groups, rows // PACK), dev)
        cp.put(res, 0, rows // PACK)
        cp.finish()

    t["reply_d2h"] = {"before_pageable": _time(lambda: back(False), dev, reps),
                      "pinned_int64": _time(lambda: back(True), dev, reps),
                      "after_device_packed": _time(back_packed, dev, reps)}
    t["wire_pack_reply"] = {"before_host": _time(lambda: pack_bits(dst.reshape(-1), 36), dev, reps),
                            "after_device": 0.0}  # inside reply_d2h's after_device_packed
    pr = pks_gpu.PackedReply(groups, rows // PACK)
    cp = pks_gpu._ReplyCopies(pr, dev)
    cp.put(res, 0, rows // PACK)
    cp.finish()
    out["reply_pack_identical"] = bytes(pr.buf) == bytes(pack_bits(res.cpu().numpy().view(np.uint64).reshape(-1), 36))
    os.environ.pop("PGW_SERVICE_PKS_PINNED", None)
    os.environ.pop("PGW_SERVICE_PKS_GPU_WIRE", None)
    out["seconds"] = {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in t.items()}
    out["recombination_identical"] = same
    # unit costs: per input feature (ring rows, a planes), per input x upload column (upload b), per output row
    per_feature = {"ring_rows", "a_planes"}
    per_upload = {"upload_b", "wire_unpack_b"}
    unit: dict[str, dict[str, float]] = {}
    for k, v in t.items():
        div = n if k in per_feature else n * cols if k in per_upload else rows
        unit[k] = {kk: vv / div for kk, vv in v.items()}
    out["unit_s"] = {k: {kk: float(f"{vv:.4g}") for kk, vv in v.items()} for k, v in unit.items()}

    def pick(v: dict[str, float], when: str) -> float:
        for key in v:
            if key.startswith(when) or key == "same":
                return v[key]
        return next(iter(v.values()))

    cyc: dict[str, Any] = {}
    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        lins = memplan.MODELS[model].linears(True)
        row = {}
        for when in ("before", "after"):
            st: dict[str, float] = {}
            for _, m, nn in lins:
                for k, v in unit.items():
                    mult = nn if k in per_feature else nn * cols if k in per_upload else m
                    # the GEMMs scale with the inputs too: unit cost per row at n inputs -> nn / n
                    scale = nn / n if k in ("hint_gemm", "y_gemm") else 1.0
                    st[k] = st.get(k, 0.0) + pick(v, when) * mult * scale
            tot = sum(st.values())
            row[when] = {"cycle_s": round(tot, 1), "slots_per_s": round(cols / tot, 2),
                         "stages_s": {k: round(v, 2) for k, v in sorted(st.items(), key=lambda kv: -kv[1])}}
        cyc[model] = row
    out["t4_cycle"] = cyc
    return out


def round3(dev: torch.device, small: bool, reps: int) -> dict[str, Any]:
    """The third round's device steps: the reply keys unpacked and range-checked on the device against the host unpack
    (exactness, the range check, seconds per 4-group item), and the staged uploads (one host thread, then one transfer,
    against the thread pool's pieces with their transfers queued as they arrive; GB/s at 72B's down and gate_up upload
    sizes, next to the raw pinned transfer)."""
    from pgw.core.errors import ProtocolError
    from pgw.hemask.wire import V3_KSK_BITS, pack_bits
    from pgw.service.pks_provider import pack_ksk, unpack_ksk

    rng = np.random.default_rng(31)
    out: dict[str, Any] = {"host_threads": pks_gpu._host_threads(), "cpus": os.cpu_count()}
    groups = 4
    beta = _keys(rng, groups).astype(np.uint64)
    for i, p in enumerate(pks.QP):
        beta[0, i, 0, :6] = [0, 1, p - 2, p - 1, 65535, 65536]
        beta[3, i, 15] = p - 1
    data = bytes(pack_ksk(beta.astype(np.uint32)))
    want = unpack_ksk(data, groups)
    u32, bad = pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(data, dev), groups)
    out["keys_device_equal_host"] = bool(not bool(bad) and np.array_equal(
        u32.view(torch.int32).cpu().numpy().view(np.uint32), want.transpose(1, 0, 2, 3)))
    flagged = []
    for i, (bits, p) in enumerate(zip(V3_KSK_BITS, pks.QP, strict=True)):
        for v in (p, (1 << bits) - 1):
            b1 = beta.copy()
            b1[2, i, 9, 99] = v
            raw = bytearray()
            for g in range(groups):
                for k, bk in enumerate(V3_KSK_BITS):
                    raw += pack_bits(np.ascontiguousarray(b1[g, k]).reshape(-1), bk)
            try:
                unpack_ksk(bytes(raw), groups)
                host_refused = False
            except ProtocolError:
                host_refused = True
            flagged.append(host_refused and bool(pks_gpu._unpack_ksk_dev(pks_gpu._upload_bytes(bytes(raw), dev),
                                                                       groups)[1]))
    out["keys_out_of_range_flagged"] = all(flagged)
    raw_t = pks_gpu._upload_bytes(data, dev)
    os.environ["PGW_SERVICE_PKS_PINNED"] = "1"
    out["keys_s"] = {
        "host_unpack": _time(lambda: unpack_ksk(data, groups), dev, reps),
        "host_unpack_and_upload_u32": _time(lambda: pks_gpu._host_keys(unpack_ksk(data, groups), dev), dev, reps),
        "device_unpack": _time(lambda: pks_gpu._unpack_ksk_dev(raw_t, groups), dev, reps),
        "device_upload_and_unpack": _time(lambda: pks_gpu._device_keys(pks_gpu.PackedKeys(data, groups), dev), dev,
                                          reps)}
    ups: dict[str, Any] = {}
    sizes = {"64KiB": 1 << 16} if small else {"72b_gate_up_b": 8192 * 1024 * 7, "72b_down_b": 29568 * 1024 * 7}
    ok = True
    for name, size in sizes.items():
        src = np.frombuffer(os.urandom(size), dtype=np.uint8)
        row: dict[str, float] = {}
        for label, threads in (("one_thread", "1"), ("pool", "")):
            if threads:
                os.environ["PGW_SERVICE_PKS_HOST_THREADS"] = threads
            else:
                os.environ.pop("PGW_SERVICE_PKS_HOST_THREADS", None)
            sec = _time(lambda: pks_gpu._upload_bytes(src, dev), dev, reps)  # noqa: B023 (called here)
            row[f"{label}_gbps"] = round(size / sec / 1e9, 2)
            got = pks_gpu._upload_bytes(src, dev)
            _sync(dev)
            ok = ok and bool(np.array_equal(got.cpu().numpy(), src))
        if dev.type == "cuda":
            pin = torch.from_numpy(src.copy()).pin_memory()
            row["raw_pinned_h2d_gbps"] = round(size / _time(lambda: pin.to(dev, non_blocking=True), dev, reps) / 1e9,  # noqa: B023
                                               2)
            dst = np.empty_like(src)
            row["host_copy_one_thread_gbps"] = round(size / _time(lambda: np.copyto(dst, src), dev, reps) / 1e9, 2)  # noqa: B023
            os.environ.pop("PGW_SERVICE_PKS_HOST_THREADS", None)
            row["host_copy_pool_gbps"] = round(size / _time(lambda: pks_gpu.staged_copy(src, dst, lambda a, b: None),  # noqa: B023
                                                            dev, reps) / 1e9, 2)
        ups[name] = row
    os.environ.pop("PGW_SERVICE_PKS_HOST_THREADS", None)
    os.environ.pop("PGW_SERVICE_PKS_PINNED", None)
    out["staged_upload"] = ups
    out["staged_upload_identical"] = ok
    out["keys_choice"] = pks_gpu.KEYS.impl(dev)
    out["all"] = bool(out["keys_device_equal_host"] and out["keys_out_of_range_flagged"] and ok and
                      (dev.type != "cuda" or out["keys_choice"] == "fused"))
    return out


def items(dev: torch.device, models: list[str], cols: int, cpu: bool, deadline: float,
          progress: Any = None, configs: tuple[str, ...] = tuple(CONFIGS), reps: int = 1) -> dict[str, Any]:
    from types import SimpleNamespace

    from pgw.core.types import LinearId, LinearKind, LinearSpec
    from pgw.hemask.wire import pack_bits
    from pgw.service.kernels import ComputeKernel
    from pgw.service.pks_provider import PKSProvider, pack_ksk

    rng = np.random.default_rng(7)
    groups = cols // D
    out: dict[str, Any] = {"cols": cols}
    warm: set[str] = set()
    for model in models:
        for key, m, n in LINEARS[model]:
            if time.time() > deadline:
                out[f"{model}:{key}"] = "skipped (time budget)"
                continue
            if cpu:
                m, n = min(m, 64), min(n, 64)
            layer, kind = key.split(".")
            lid = LinearId(int(layer), LinearKind.LM_HEAD if kind == "head" else LinearKind(kind))
            w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=torch.Generator().manual_seed(3))
            kern = ComputeKernel(LinearSpec(lid, m, n), w, device=dev)
            del w
            b = rng.integers(0, 1 << 56, (n, cols), dtype=np.uint64)
            ops = np.stack([np.stack([rng.integers(0, p, (RANK, NR), dtype=np.uint64) for p in pks.QP])
                            for _ in range(groups)]).astype(np.uint32)
            it = SimpleNamespace(linear=key, cols=cols, b=pack_bits(b.reshape(-1), 56), b_bits=56, ksk=pack_ksk(ops))
            seed = os.urandom(32)
            res: dict[str, Any] = {"m": m, "n": n}
            replies = {}
            for cfg in configs:
                env = CONFIGS[cfg]
                if time.time() > deadline:
                    break
                _set_env(env)
                pv = PKSProvider(mode="device")
                pv.use_device("fp", kern, key)  # self-tests outside the timing
                if cfg not in warm:  # warm-up once per configuration: kernel compilation, allocator
                    bytes(pv.run_item("fp", kern, it, seed))
                    warm.add(cfg)
                ts = []
                for _ in range(1 if cfg in ("torch", "prev") else reps):
                    _sync(dev)
                    t0 = time.perf_counter()
                    replies[cfg] = bytes(pv.run_item("fp", kern, it, seed))
                    ts.append(time.perf_counter() - t0)
                res[f"{cfg}_s"] = round(float(np.median(ts)), 4)
                res[f"{cfg}_device_items"] = pv.device_items
                if cfg == "fused":
                    res["fused_status"] = pks_gpu.fused_status(dev)
            _set_env({})
            res["identical"] = len(replies) == len(configs) and len(set(replies.values())) == 1
            if "fused_s" in res:
                for base in ("prev", "torch", "r2"):
                    if f"{base}_s" in res:
                        res[f"speedup_vs_{base}"] = round(res[f"{base}_s"] / max(res["fused_s"], 1e-9), 3)
            for kk in ("r2", "all"):
                res[f"model_t4_{kk}_s"] = round(sum(memplan.pks_item_seconds(
                    m, n, cols, "device", memplan.T4_RATES, ntt="fused", kernels=kk).values()), 3)
            out[f"{model}:{key}"] = res
            print(model, key, res, flush=True)
            if progress is not None:  # saved per item: a step timeout keeps what finished
                progress(out)
            del kern
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


def cycle(ks: dict[str, Any]) -> dict[str, Any]:
    """Key-switch seconds per 1024-slot cycle (every linear, 4 groups) from the per-row times."""
    g4 = ks["g4"]
    out = {}
    for model in ("qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"):
        rows = sum(m for _, m, _ in memplan.MODELS[model].linears(True))
        out[model] = {"rows": rows, "torch_s": round(rows * g4["torch_us_per_row"] / 1e6, 1),
                      "fused_s": round(rows * g4["fused_us_per_row"] / 1e6, 1),
                      "fused_fwd4_s": round(rows * g4["fused_fwd4_us_per_row"] / 1e6, 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU (Triton interpreter, tiny shapes)")
    ap.add_argument("--models", default="qwen2.5-7b,qwen2.5-32b,qwen2.5-72b")
    ap.add_argument("--warps", default="4,8,16")
    ap.add_argument("--budget-s", type=float, default=2400)
    ap.add_argument("--sections", default="exact,exact_new,ntt_us,keyswitch,stages,round3,items",
                    help="comma list of the sections to run")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="item configurations (comma list of CONFIGS)")
    ap.add_argument("--item-reps", type=int, default=1, help="timed runs per item and configuration (median)")
    a = ap.parse_args()
    want = set(a.sections.split(","))
    t_start = time.time()
    if a.cpu:
        os.environ.setdefault("TRITON_INTERPRET", "1")
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")
    import triton

    res: dict[str, Any] = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                           "torch": torch.__version__, "triton": triton.__version__}
    path = a.json

    def save() -> None:
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=1)

    def section(name: str, fn: Any) -> None:
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:
            import traceback

            res[name] = {"error": f"{type(e).__name__}: {str(e)[:800]}", "trace": traceback.format_exc()[-3000:]}
        print(name, round(time.time() - t0, 1), "s:", json.dumps(res[name])[:3000], flush=True)
        save()

    small = a.cpu
    if "exact" in want:
        section("exact", lambda: exact(dev, PACK * (2 if small else 64)))
    if "exact_new" in want:
        section("exact_new", lambda: exact_new(dev, small))
    if "ntt_us" in want:
        section("ntt_us", lambda: ntt_us(dev, 16 if small else 2048, 1 if small else 5,
                                         [8] if small else [int(w) for w in a.warps.split(",")]))
    best = None
    if isinstance(res.get("ntt_us"), dict):
        cand = {k: v["keyswitch_g4_us_per_row"] for k, v in res["ntt_us"].items() if k.startswith("warps_")}
        if cand:
            best = min(cand, key=cand.get).split("_")[1]
            os.environ["PGW_SERVICE_PKS_NTT_WARPS"] = best
    res["warps_used"] = best
    if "keyswitch" in want:
        section("keyswitch", lambda: keyswitch_rows(dev, PACK if small else 2048, 1 if small else 5))
        if isinstance(res.get("keyswitch"), dict) and "g4" in res["keyswitch"]:
            section("cycle", lambda: cycle(res["keyswitch"]))
    if "stages" in want:
        section("stages", lambda: stages(dev, PACK if small else 2048, 1 if small else 5, small))
    if "round3" in want:
        section("round3", lambda: round3(dev, small, 1 if small else 5))

    def partial(o: dict[str, Any]) -> None:
        res["items"] = o
        save()

    configs = tuple(a.configs.split(","))
    if "items" in want:
        section("items", lambda: items(dev, a.models.split(","), 1024 if not small else 256, small,
                                       t_start + a.budget_s, partial, configs, a.item_reps))
    res["total_s"] = round(time.time() - t_start, 1)
    save()
    ok = all(isinstance(res.get(s), dict) and res[s].get("all") is True
             for s in ("exact", "exact_new", "round3") if s in want)
    ok = ok and all(v.get("identical", False) for k, v in res.get("items", {}).items() if isinstance(v, dict))
    if "stages" in want:
        ok = ok and bool(res.get("stages", {}).get("recombination_identical"))
        ok = ok and bool(res.get("stages", {}).get("reply_pack_identical"))
    print("OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
