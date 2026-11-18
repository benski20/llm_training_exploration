"""Encrypted mask generation cost at real layer shapes (one layer of each kind, extrapolated per model).

Times, per 256-slot group:
- gateway encrypt (CPU),
- provider ``W8 b mod 2^64`` (ComputeKernel.mulmod64 on --device, cuda or cpu),
- gateway decrypt (CPU, including ``hint @ S``),
and the one-time hint build, then extrapolates to seconds per token for the whole model.

    python benchmarks/hemask_bench.py --preset qwen2.5-7b --device cuda --groups 4
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE, HEParams, encrypt_masks, matmul_mod64, public_matrix
from pgw.hemask.fastmm import big_planes_mm_small, int8_gemm_available, to_planes
from pgw.service.kernels import ComputeKernel

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "llama3-8b": (32, {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-7b", choices=sorted(PRESETS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--groups", type=int, default=4, help="256-slot groups per provider call")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    layers, shapes = PRESETS[args.preset]
    p = HEParams(k=40, rank=args.rank)
    rng = np.random.default_rng(0)
    cols = DEGREE * args.groups
    res: dict[str, object] = {"preset": args.preset, "device": args.device, "rank": args.rank, "groups": args.groups,
                              "torch": torch.__version__, "cpu_int8_fast": int8_gemm_available(),
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    per_kind = {}
    tot = {"encrypt": 0.0, "provider": 0.0, "decrypt": 0.0, "hint_build": 0.0, "hint_bytes": 0, "up": 0, "down": 0}
    for kind, (m, n) in shapes.items():
        w = rng.integers(-127, 128, size=(m, n)).astype(np.int8)
        a = public_matrix(bytes(32), n, p)
        t = time.perf_counter()
        fast = int8_gemm_available()
        hint = np.empty((m, 7, p.dim), dtype=np.int8)  # the backend's format: byte planes 1..7
        for c0 in range(0, p.dim, 512):
            hint[:, :, c0:c0 + 512] = to_planes(matmul_mod64(w, a[:, c0:c0 + 512]))[:, 1:]
        t_hint = time.perf_counter() - t
        a_planes = to_planes(a) if fast else None
        kern = ComputeKernel(LinearSpec(LinearId(0, kind), m, n), torch.from_numpy(w), device=args.device)
        r = rng.integers(0, 1 << 40, size=(cols, n), dtype=np.int64)
        t = time.perf_counter()
        cts = [encrypt_masks(r[g * DEGREE:(g + 1) * DEGREE], None if fast else a, p, a_planes=a_planes)
               for g in range(args.groups)]
        b = np.concatenate([c.b for c in cts], axis=1)
        t_enc = time.perf_counter() - t
        kern.mulmod64(b[:, :DEGREE].copy())  # warm-up
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        y = kern.mulmod64(b)
        t_prov = time.perf_counter() - t
        t = time.perf_counter()
        smat = np.concatenate([c.secret for c in cts], axis=1)  # one GEMM for all groups, as the backend
        v = y - big_planes_mm_small(hint, smat)
        wr = ((v + np.uint64(1 << (p.delta_bits - 1))) >> np.uint64(p.delta_bits)) & np.uint64((1 << 40) - 1)
        outs = [wr[:, g * DEGREE:(g + 1) * DEGREE].T.astype(np.int64) for g in range(args.groups)]
        t_dec = time.perf_counter() - t
        # exactness on a sample of rows
        ref = np.array((r[:8].astype(object) @ w.T.astype(object)) % (1 << 40), dtype=np.int64)
        exact = bool(np.array_equal(outs[0][:8], ref))
        per_kind[kind] = {"m": m, "n": n, "exact": exact, "hint_build_s": round(t_hint, 2),
                          "encrypt_ms_per_slot": 1e3 * t_enc / cols, "provider_ms_per_slot": 1e3 * t_prov / cols,
                          "decrypt_ms_per_slot": 1e3 * t_dec / cols}
        tot["encrypt"] += layers * t_enc / cols
        tot["provider"] += layers * t_prov / cols
        tot["decrypt"] += layers * t_dec / cols
        tot["hint_build"] += layers * t_hint
        tot["hint_bytes"] += layers * hint.nbytes
        tot["up"] += layers * n * 8
        tot["down"] += layers * m * (64 - HEParams.DROP_BITS) // 8
        print(kind, json.dumps(per_kind[kind]), flush=True)
        del hint, kern
    res["per_kind"] = per_kind
    res["per_token_full_model"] = {
        "gateway_encrypt_s": tot["encrypt"], "provider_s": tot["provider"], "gateway_decrypt_s": tot["decrypt"],
        "upload_MB": tot["up"] / 1e6, "download_MB": tot["down"] / 1e6,
    }
    res["one_time"] = {"hint_GB": tot["hint_bytes"] / 1e9, "hint_build_s_cpu": tot["hint_build"]}
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
