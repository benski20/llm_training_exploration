"""Client cost of a provider-built, client-verified pks-v3 hint at real shapes (pgw.provision), vs a local build.

Per linear shape (Qwen2.5-7B by default: qkv, o, gate_up, down, the masked head) it times, on ``--rows`` rows of
random int8 weights and the honest ``H`` for a fresh seed:

* ``check_init``  challenges and ``A X`` (once per linear, from the ring rows),
* ``weights``     ``U^T W8`` (per row of W8),
* ``h_check``     decoding the 7-byte wire form and ``U^T (H X)`` (per row of H),
* ``finish``      ``switch_hint`` + key switch (per row; identical in the local build),
* ``local_gemm``  ``H = W8 A`` on this CPU (per row; what the provider saves the client),

and extrapolates to the whole model (``--layers`` x the four linears + head). Bytes: ``H`` at 7 bytes per entry, the
int8 weights once, the stored hint. Run: ``python benchmarks/provision_hint_bench.py [--model 7b] [--rows 1024]``.
"""

# ruff: noqa: B023  (the closures are called immediately, inside the loop iteration)
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

SHAPES = {
    "7b": {"layers": 28, "linears": {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584),
                                     "down": (3584, 18944)}, "head": (152064, 3584)},
    "0.5b": {"layers": 24, "linears": {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896),
                                       "down": (896, 4864)}, "head": (151936, 896)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=sorted(SHAPES))
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--no-head", action="store_true")
    args = ap.parse_args()

    from pgw.core.prng import new_seed
    from pgw.hemask import pks
    from pgw.hemask.lwe import matmul_mod64
    from pgw.hemask.pks_backend import switch_hint_fast
    from pgw.hemask.pks_native import keyswitch
    from pgw.provision.hintcheck import LinearHCheck, decode_h56, encode_h56

    cfg = SHAPES[args.model]
    kinds = dict(cfg["linears"])
    if not args.no_head:
        kinds["head"] = cfg["head"]
    rng = np.random.default_rng(0)
    m56 = np.uint64((1 << 56) - 1)
    k_hat = pks.ks_a(new_seed())[None].astype(np.uint32)
    per: dict[str, dict[str, float]] = {}
    # warm-up (thread pools, BLAS, native kernels)
    ww = rng.integers(-127, 128, (64, 64), dtype=np.int8)
    s0 = new_seed()
    hh = matmul_mod64(ww, pks.a_mlwe_rows(pks.public_ring_rows(s0, 64, "w"))) & m56
    LinearHCheck(s0, "w", 64, 64).add_h(0, hh)
    keyswitch(switch_hint_fast(hh), k_hat)
    def timed(fn: object, *a: object) -> tuple[float, float, object]:
        w0, c0 = time.perf_counter(), time.process_time()
        r = fn(*a)  # type: ignore[operator]
        return time.perf_counter() - w0, time.process_time() - c0, r

    cpu: dict[str, dict[str, float]] = {}
    for kind, (m, n) in kinds.items():
        rows = min(args.rows, m)
        seed = new_seed()
        w = rng.integers(-127, 128, (rows, n), dtype=np.int8)
        t_a, _, a = timed(lambda: pks.a_mlwe_rows(pks.public_ring_rows(seed, n, "0.x")))
        t_gemm, c_gemm, h = timed(lambda: matmul_mod64(w, a) & m56)
        data = encode_h56(h)
        t_init, c_init, ck = timed(lambda: LinearHCheck(seed, "0.x", m, n))
        t_w, c_w, _ = timed(ck.add_weights, 0, w)

        def hcheck() -> object:
            hd = decode_h56(data, rows)
            ck.add_h(0, hd)
            return hd

        t_h, c_h, hd = timed(hcheck)
        t_fin, c_fin, _ = timed(lambda: keyswitch(switch_hint_fast(hd), k_hat))
        cpu[kind] = {"check_init": c_init, "weights": c_w / rows, "h_check": c_h / rows, "finish": c_fin / rows,
                     "local_gemm": c_gemm / rows}
        per[kind] = {"m": m, "n": n, "rows": rows, "check_init_s": t_init, "weights_s_per_row": t_w / rows,
                     "h_check_s_per_row": t_h / rows, "finish_s_per_row": t_fin / rows,
                     "local_gemm_s_per_row": t_gemm / rows, "a_expand_s": t_a}
        print(kind, json.dumps({k: (round(v * 1e6, 1) if "per_row" in k else round(v, 3)) for k, v in per[kind].items()
                                if k not in ("m", "n", "rows")}), "(per_row in us)", flush=True)
    total = {"check_init_s": 0.0, "weights_s": 0.0, "h_check_s": 0.0, "finish_s": 0.0, "local_gemm_s": 0.0}
    nbytes = {"h_bytes": 0, "weight_bytes": 0, "hint_bytes": 0}
    for kind, p in per.items():
        cnt = 1 if kind == "head" else cfg["layers"]
        m, n = int(p["m"]), int(p["n"])
        total["check_init_s"] += cnt * p["check_init_s"]
        total["weights_s"] += cnt * m * p["weights_s_per_row"]
        total["h_check_s"] += cnt * m * p["h_check_s_per_row"]
        total["finish_s"] += cnt * m * p["finish_s_per_row"]
        total["local_gemm_s"] += cnt * m * p["local_gemm_s_per_row"]
        nbytes["h_bytes"] += cnt * m * 4096 * 7
        nbytes["weight_bytes"] += cnt * m * n if kind != "head" else 0  # the head's W8 is derived locally
        nbytes["hint_bytes"] += cnt * (m // 32) * 2 * 8192 * 3
    ctot = {"check_init": 0.0, "weights": 0.0, "h_check": 0.0, "finish": 0.0, "local_gemm": 0.0}
    for kind, c in cpu.items():
        cnt = 1 if kind == "head" else cfg["layers"]
        m = int(per[kind]["m"])
        for k in ctot:
            ctot[k] += cnt * (c[k] if k == "check_init" else m * c[k])
    verify = total["check_init_s"] + total["weights_s"] + total["h_check_s"]
    out = {"model": args.model, "cpus": os.cpu_count(), "totals_s": {k: round(v, 1) for k, v in total.items()},
           "client_provider_path_s": round(verify + total["finish_s"], 1), "verify_s": round(verify, 1),
           "client_local_build_s": round(total["local_gemm_s"] + total["finish_s"], 1),
           "bytes_GB": {k: round(v / 1e9, 2) for k, v in nbytes.items()},
           "cpu_seconds": {k: round(v, 1) for k, v in ctot.items()},
           "cpu_seconds_provider_path": round(sum(v for k, v in ctot.items() if k != "local_gemm"), 1),
           "cpu_seconds_local_build": round(ctot["local_gemm"] + ctot["finish"], 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
