"""Client cost of the pks-v3 hint at real shapes (pgw.provision): local build and provider-built, client-verified.

Per linear shape (Qwen2.5-0.5B / 7B / 32B / 72B: qkv, o, gate_up, down, the masked head) it times, on ``--rows`` rows of
random int8 weights and the honest ``H`` for a fresh seed:

* ``check_init``  challenges and ``A X`` (once per linear, from the ring rows),
* ``weights``     ``U^T W8`` (per row of W8),
* ``h_check``     decoding the 7-byte wire form and ``U^T (H X)`` (per row of H),
* ``gemm_init``   ``A``'s byte planes (once per linear; ``--path new`` only),
* ``finish``      ``switch_hint`` + key switch (per row; identical in the local build),
* ``local_gemm``  ``H = W8 A`` on this CPU (per row; what the provider saves the client),

and extrapolates to the whole model (``--layers`` x the four linears + head). ``--path old``: the pre-2026-11-26
kernels (``matmul_mod64`` per row block, numpy switch, native key switch); ``--path new`` (default):
:class:`pgw.hemask.hint_gemm.HintGemm` (planes once, 7 planes, native recombination) and the fused native
``hint_keyswitch``. Bytes: ``H`` at 7 bytes per entry, the int8 weights once, the stored hint. CPU-seconds (all
threads) are the robust figure on a shared box. Run:
``PYTHONPATH=src python benchmarks/provision_hint_bench.py [--model 7b] [--rows 1024] [--path new]``.
"""

# ruff: noqa: B023  (the closures are called immediately, inside the loop iteration)
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

#: (layers, {kind: (out_features, in_features)}, head (vocab, hidden)) of Qwen2.5 (fused qkv and gate_up).
SHAPES = {
    "0.5b": {"layers": 24, "linears": {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896),
                                       "down": (896, 4864)}, "head": (151936, 896)},
    "7b": {"layers": 28, "linears": {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584),
                                     "down": (3584, 18944)}, "head": (152064, 3584)},
    "32b": {"layers": 64, "linears": {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120),
                                      "down": (5120, 27648)}, "head": (152064, 5120)},
    "72b": {"layers": 80, "linears": {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192),
                                      "down": (8192, 29568)}, "head": (152064, 8192)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=sorted(SHAPES))
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--path", default="new", choices=["new", "old"])
    ap.add_argument("--skip-check", action="store_true", help="local build only (no provider-path check timings)")
    ap.add_argument("--threads", type=int, default=0, help="torch and native threads (0: every CPU)")
    ap.add_argument("--reps", type=int, default=1, help="repeat the per-shape timings, keep the fastest")
    args = ap.parse_args()
    if args.threads:
        os.environ["PGW_PKS_THREADS"] = str(args.threads)
        import torch

        torch.set_num_threads(args.threads)

    from pgw.core.prng import new_seed
    from pgw.hemask import pks
    from pgw.hemask.hint_gemm import HintGemm, hint_rows
    from pgw.hemask.lwe import matmul_mod64
    from pgw.hemask.pks_native import keyswitch
    from pgw.provision.hintcheck import LinearHCheck, decode_h56, encode_h56

    def old_switch(h: np.ndarray) -> np.ndarray:  # the numpy switch of the old path (threads, as before)
        from concurrent.futures import ThreadPoolExecutor

        par = pks.PKSParams()
        nt = os.cpu_count() or 1
        out = np.empty(h.shape, dtype=np.uint64)
        step = -(-h.shape[0] // nt)

        def run(r0: int) -> None:
            out[r0:r0 + step] = pks.switch_hint(h[r0:r0 + step], par)

        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(run, range(0, h.shape[0], step)))
        return out

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
    keyswitch(old_switch(hh), k_hat)
    hint_rows(HintGemm(s0, 64, "w")(ww), k_hat)

    def timed(fn: object, *a: object) -> tuple[float, float, object]:
        best_w = best_c = float("inf")
        r: object = None
        for _ in range(max(1, args.reps)):
            w0, c0 = time.perf_counter(), time.process_time()
            r = fn(*a)  # type: ignore[operator]
            best_w, best_c = min(best_w, time.perf_counter() - w0), min(best_c, time.process_time() - c0)
        return best_w, best_c, r

    cpu: dict[str, dict[str, float]] = {}
    for kind, (m, n) in kinds.items():
        rows = min(args.rows, m)
        seed = new_seed()
        w = rng.integers(-127, 128, (rows, n), dtype=np.int8)
        if args.path == "new":
            t_gi, c_gi, g = timed(lambda: HintGemm(seed, n, "0.x"))
            t_gemm, c_gemm, h = timed(g, w)  # type: ignore[arg-type]
            g.close()  # type: ignore[attr-defined]
            t_fin, c_fin, _ = timed(lambda: hint_rows(h, k_hat))
        else:
            t_gi = c_gi = 0.0
            a = pks.a_mlwe_rows(pks.public_ring_rows(seed, n, "0.x"))
            t_gemm, c_gemm, h = timed(lambda: matmul_mod64(w, a) & m56)
            del a
            t_fin, c_fin, _ = timed(lambda: keyswitch(old_switch(h), k_hat))
        if args.skip_check:
            t_init = c_init = t_w = c_w = t_h = c_h = 0.0
        else:
            data = encode_h56(h)  # type: ignore[arg-type]
            t_init, c_init, _ = timed(lambda: LinearHCheck(seed, "0.x", m, n))
            t_w = c_w = t_h = c_h = float("inf")
            for _ in range(max(1, args.reps)):  # a check object takes each row once: a fresh one per repetition
                ck = LinearHCheck(seed, "0.x", m, n)
                w0, c0 = time.perf_counter(), time.process_time()
                ck.add_weights(0, w)
                t_w, c_w = min(t_w, time.perf_counter() - w0), min(c_w, time.process_time() - c0)
                w0, c0 = time.perf_counter(), time.process_time()
                ck.add_h(0, decode_h56(data, rows))
                t_h, c_h = min(t_h, time.perf_counter() - w0), min(c_h, time.process_time() - c0)
        cpu[kind] = {"check_init": c_init, "weights": c_w / rows, "h_check": c_h / rows, "finish": c_fin / rows,
                     "local_gemm": c_gemm / rows, "gemm_init": c_gi}
        per[kind] = {"m": m, "n": n, "rows": rows, "check_init_s": t_init, "weights_s_per_row": t_w / rows,
                     "h_check_s_per_row": t_h / rows, "finish_s_per_row": t_fin / rows,
                     "local_gemm_s_per_row": t_gemm / rows, "gemm_init_s": t_gi}
        print(kind, json.dumps({k: (round(v * 1e6, 1) if "per_row" in k else round(v, 3)) for k, v in per[kind].items()
                                if k not in ("m", "n", "rows")}), "(per_row in us)",
              json.dumps({k: round(v * (1 if k.endswith("init") else 1e6), 2) for k, v in cpu[kind].items()}),
              "(cpu; per row in us)", flush=True)
    total = {"check_init_s": 0.0, "weights_s": 0.0, "h_check_s": 0.0, "finish_s": 0.0, "local_gemm_s": 0.0,
             "gemm_init_s": 0.0}
    nbytes = {"h_bytes": 0, "weight_bytes": 0, "hint_bytes": 0}
    hint_rows_total = 0
    for kind, p in per.items():
        cnt = 1 if kind == "head" else cfg["layers"]
        m, n = int(p["m"]), int(p["n"])
        hint_rows_total += cnt * m
        total["check_init_s"] += cnt * p["check_init_s"]
        total["gemm_init_s"] += cnt * p["gemm_init_s"]
        total["weights_s"] += cnt * m * p["weights_s_per_row"]
        total["h_check_s"] += cnt * m * p["h_check_s_per_row"]
        total["finish_s"] += cnt * m * p["finish_s_per_row"]
        total["local_gemm_s"] += cnt * m * p["local_gemm_s_per_row"]
        nbytes["h_bytes"] += cnt * m * 4096 * 7
        nbytes["weight_bytes"] += cnt * m * n if kind != "head" else 0  # the head's W8 is derived locally
        nbytes["hint_bytes"] += cnt * (m // 32) * 2 * 8192 * 3
    ctot = {"check_init": 0.0, "weights": 0.0, "h_check": 0.0, "finish": 0.0, "local_gemm": 0.0, "gemm_init": 0.0}
    for kind, c in cpu.items():
        cnt = 1 if kind == "head" else cfg["layers"]
        m = int(per[kind]["m"])
        for k in ctot:
            ctot[k] += cnt * (c[k] if k.endswith("init") else m * c[k])
    verify = total["check_init_s"] + total["weights_s"] + total["h_check_s"]
    out = {"model": args.model, "path": args.path, "cpus": os.cpu_count(), "hint_rows": hint_rows_total,
           "totals_s": {k: round(v, 1) for k, v in total.items()},
           "client_provider_path_s": round(verify + total["finish_s"], 1), "verify_s": round(verify, 1),
           "client_local_build_s": round(total["gemm_init_s"] + total["local_gemm_s"] + total["finish_s"], 1),
           "bytes_GB": {k: round(v / 1e9, 2) for k, v in nbytes.items()},
           "cpu_seconds": {k: round(v, 1) for k, v in ctot.items()},
           "cpu_seconds_provider_path": round(sum(v for k, v in ctot.items()
                                                  if k not in ("local_gemm", "gemm_init")), 1),
           "cpu_seconds_local_build": round(ctot["gemm_init"] + ctot["local_gemm"] + ctot["finish"], 1),
           "cpu_us_per_hint_row_local": round(1e6 * (ctot["gemm_init"] + ctot["local_gemm"] + ctot["finish"])
                                              / hint_rows_total, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
