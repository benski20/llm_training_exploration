"""Gateway HE on a GPU: the device NTT (``pgw.hemask.ntt_dev``) vs the CPU NTT and the dense device path.

Times, per row of the public matrix / hint and per 256-slot group, at rank 16 / q = 2^56 (K32):

* ``enc_pad_*``: the encryption pad ``A[:n] @ S`` with ``A`` in NTT form, ``--groups`` groups per pass:
  ``cpu_ntt_pre`` (native ``pgw_ntt``), ``dev_torch`` (torch ops), ``dev_triton`` (fused Triton kernel);
* ``enc_full_*``: complete encryptions (secret and noise sampling on the CPU, pad, combine) through
  ``PretransformedEncryptor`` (CPU NTT vs device NTT), and the dense device path
  ``devmm.DeviceHE.encrypt`` (int8 GEMM, the ~55-65 slots/s baseline);
* ``dec_hint_*``: decryption's ``H @ [S_1 | ... | S_G]`` from the 6 hint byte planes: CPU
  ``ntt.big_planes_mm_small`` vs ``DeviceNTT.hint_mm`` (hint rows uploaded and transformed per pass).

Every device result is compared with the CPU result on the timed shapes (bit-identical or the script
fails). Per-row costs are extrapolated to the model like ``hemask_ntt_bench.py``: ms/slot = cost per
row and group x (sum of n, resp. m, over all linears) / 256.

    python benchmarks/hemask_ntt_dev_bench.py --preset qwen2.5-0.5b --groups 16 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from pgw.hemask import DEGREE, HEParams, ntt, ntt_dev, public_matrix, secret_matrix
from pgw.hemask.mlwe_ring import PretransformedEncryptor

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "llama-70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}),
}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def best(fn: Callable[[], object], reps: int) -> float:
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--groups", type=int, default=16, help="256-slot groups per pass")
    ap.add_argument("--rows", type=int, default=2048, help="rows timed on the device")
    ap.add_argument("--cpu-rows", type=int, default=512, help="rows timed on the CPU paths")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    layers, kinds = PRESETS[args.preset]
    sum_n = layers * sum(n for _m, n in kinds.values())
    sum_m = layers * sum(m for m, _n in kinds.values())
    params = HEParams(k=32, rank=args.rank)
    g, rows, crow = args.groups, args.rows, min(args.cpu_rows, args.rows)
    rng = np.random.default_rng(0)
    a = public_matrix(bytes(32), rows, params)
    hint = rng.integers(-128, 128, size=(rows, params.q // 8 - 1, params.dim), dtype=np.int8)
    secs = rng.integers(-1, 2, size=(g, args.rank, DEGREE)).astype(np.int8)
    smat = np.concatenate([secret_matrix(s) for s in secs], axis=1)
    res: dict[str, object] = {"preset": args.preset, "rank": args.rank, "groups": g, "rows": rows,
                              "sum_n": sum_n, "sum_m": sum_m, "device": args.device,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              "cpu_threads": torch.get_num_threads(), "pgw_ntt": ntt.available()}
    t: dict[str, float] = {}  # seconds per row and group
    ref_pad = ref_hint = None
    if ntt.available():
        from pgw.hemask.fastmm import to_planes

        cpu_a = ntt.TransformedOperand.from_planes(to_planes(a[:crow])[:, :7], first_byte=0)
        t["enc_pad_cpu_ntt_pre"] = best(lambda: cpu_a.mm_secrets(secs, rows=crow), args.reps) / (crow * g)
        ref_pad = cpu_a.mm_secrets(secs, rows=crow)
        t["dec_hint_cpu_ntt"] = best(lambda: ntt.big_planes_mm_small(hint[:crow], smat, first_byte=1),
                                     args.reps) / (crow * g)
        ref_hint = ntt.big_planes_mm_small(hint[:crow], smat, first_byte=1)
        enc_cpu = PretransformedEncryptor(a, params, groups_per_pass=min(g, 16))
        r = rng.integers(0, 1 << 32, size=(g * DEGREE, crow), dtype=np.int64)
        t["enc_full_cpu_ntt_pre"] = best(lambda: enc_cpu.encrypt(r), args.reps) / (crow * g)
    variants = [("dev_torch", False)] + ([] if args.no_triton else [("dev_triton", True)])
    status: dict[str, object] = {}
    for name, use_tri in variants:
        try:
            dn = ntt_dev.DeviceNTT(args.device, use_triton=use_tri)
            if use_tri and dn.triton is None:
                status[name] = "triton unavailable"
                continue
            t0 = time.perf_counter()
            ok = ntt_dev.self_test(dn, params)
            status[name] = {"self_test": ok, "self_test_s": round(time.perf_counter() - t0, 2)}
            if not ok:
                continue
            t0 = time.perf_counter()
            dev_a = ntt_dev.DeviceTransformedA(a, params.q, dn)
            _sync()
            res[f"{name}_transform_a_s_per_row"] = (time.perf_counter() - t0) / rows
            t[f"enc_pad_{name}"] = best(lambda d=dev_a: d.mm_secrets(secs, rows=rows), args.reps) / (rows * g)
            t[f"dec_hint_{name}"] = best(lambda d=dn: d.hint_mm(hint, smat, first_byte=1), args.reps) / (rows * g)
            if ref_pad is not None:
                assert np.array_equal(dev_a.mm_secrets(secs, rows=crow), ref_pad), name
                assert np.array_equal(dn.hint_mm(hint[:crow], smat, first_byte=1), ref_hint), name
            enc = PretransformedEncryptor(a, params, groups_per_pass=min(g, 16), device_ntt=dn)
            r = rng.integers(0, 1 << 32, size=(g * DEGREE, rows), dtype=np.int64)
            t[f"enc_full_{name}"] = best(lambda e=enc, x=r: e.encrypt(x), args.reps) / (rows * g)
        except Exception as e:
            status[name] = repr(e)
    if args.device.startswith("cuda"):
        try:
            from pgw.hemask.devmm import DeviceHE
            from pgw.hemask.fastmm import to_planes

            dh = DeviceHE(to_planes(a)[:, :7], params, args.device)
            r = rng.integers(0, 1 << 32, size=(g * DEGREE, rows), dtype=np.int64)
            t["enc_full_devmm_dense"] = best(lambda: dh.encrypt(r, rows), args.reps) / (rows * g)
        except Exception as e:
            status["devmm"] = repr(e)
    res["status"] = status
    per_slot = {k: 1e3 * v * (sum_n if k.startswith("enc") else sum_m) / DEGREE for k, v in t.items()}
    res["us_per_row_group"] = {k: round(v * 1e6, 4) for k, v in t.items()}
    res["ms_per_slot"] = {k: round(v, 3) for k, v in per_slot.items()}
    res["slots_per_s"] = {k: round(1e3 / v, 1) for k, v in per_slot.items()}
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    os.environ.setdefault("PGW_NTT_DEV_TRITON", "1")
    main()
