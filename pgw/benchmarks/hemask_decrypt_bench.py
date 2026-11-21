"""Gateway decryption of the encrypted mask factory, per path (``HEMaskBackend(decrypt=...)``), per slot.

Times one decryption pass (``G`` 256-slot groups) of each linear kind of one layer at real shapes, on
random hint planes and replies (cost does not depend on the values), for the paths

* ``dense``: today's byte-plane GEMM (``torch._int_mm`` on an int8 engine + the native combine; the
  non-fused NTT product on hosts without a fast int8 engine);
* ``ntt``: the fused native NTT decryption on the stored hint planes (no extra memory);
* ``ntt_pre``: the same with the hint held in NTT form (12 bytes per coefficient in RAM);

checks that all paths give the same bits, and extrapolates to milliseconds per slot for the whole model
(``layers x sum over kinds``). ``--hint-file`` puts the hint planes in a file and memory-maps them
(as the factory does), with ``--drop-cache`` evicting them from the page cache before every pass
(Linux, needs write access to /proc/sys/vm/drop_caches; otherwise it is reported as not done).

    python benchmarks/hemask_decrypt_bench.py --preset qwen2.5-0.5b --groups 2 16
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.core.ring import RingParams
from pgw.hemask import DEGREE, HEParams, ntt
from pgw.hemask.backend import HEMaskBackend
from pgw.hemask.fastmm import engine
from pgw.hemask.lwe import MaskCiphertext, secret_matrix

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
}


def _drop_cache() -> bool:
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")
        return True
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--groups", type=int, nargs="+", default=[2, 16], help="256-slot groups per pass")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--k", type=int, default=32, help="ring bits (32: q = 2^56)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=4096, help="hint rows timed per kind (scaled to m)")
    ap.add_argument("--kinds", nargs="*", default=None, help="subset of linear kinds")
    ap.add_argument("--paths", nargs="*", default=["dense", "ntt", "ntt_pre"])
    ap.add_argument("--hint-file", action="store_true", help="memory-map the hint planes from a file")
    ap.add_argument("--drop-cache", action="store_true", help="evict the hint file before every pass")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    layers, kinds = PRESETS[args.preset]
    if args.kinds:
        kinds = {k: v for k, v in kinds.items() if k in args.kinds}
    params = HEParams(k=args.k, rank=args.rank)
    ring = RingParams.from_name("K32A12C27" if args.k == 32 else "K40A12")
    npl = params.q // 8 - 1
    be: Any = object.__new__(HEMaskBackend)  # only the attributes the decrypt helpers read
    be.params, be.ring = params, ring
    rng = np.random.default_rng(0)
    res: dict[str, Any] = {"preset": args.preset, "rank": args.rank, "k": args.k, "dense_engine": engine(),
                           "ntt": ntt.info(), "torch_threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                           "hint_file": args.hint_file, "drop_cache": args.drop_cache, "per_kind": {}}
    tmp = Path(tempfile.mkdtemp(prefix="pgw-dec-bench-"))
    dropped = None
    per_slot: dict[str, dict[str, float]] = {}
    for kind, (m, _n) in kinds.items():
        rows = min(m, args.max_rows)
        hint = rng.integers(-128, 128, size=(rows, npl, params.dim), dtype=np.int8)
        if args.hint_file:
            path = tmp / f"{kind}.npy"
            np.save(path, hint)
            del hint
            hint = np.load(path, mmap_mode="r")
        pre = ntt.TransformedOperand.from_planes(np.asarray(hint), first_byte=1) if "ntt_pre" in args.paths else None
        kr: dict[str, Any] = {"m": m, "rows_timed": rows}
        for g in args.groups:
            cts = [MaskCiphertext(b=np.empty((0, DEGREE), np.uint64),
                                  secret=secret_matrix(rng.integers(-1, 2, size=(args.rank, DEGREE))), rows=DEGREE)
                   for _ in range(g)]
            y = rng.integers(0, 1 << 62, size=(rows, DEGREE * g), dtype=np.uint64)
            items = [(y, cts, 0)]
            outs: dict[str, np.ndarray] = {}
            for path in args.paths:
                out = np.zeros((DEGREE * g, rows), dtype=np.int64)
                best = float("inf")
                for _ in range(args.reps + 1):
                    if args.drop_cache:
                        dropped = _drop_cache()
                    t0 = time.perf_counter()
                    if path == "dense":
                        be._decrypt_dense(hint, items, out)
                    else:
                        be._decrypt_ntt(pre if path == "ntt_pre" else hint, items, out)
                    best = min(best, time.perf_counter() - t0)
                outs[path] = out
                us = best / (rows * g) * 1e6  # per hint row and group
                kr[f"g{g}_{path}_us_per_row_group"] = round(us, 3)
                per_slot.setdefault(f"g{g}_{path}", {})[kind] = us * m * layers / DEGREE / 1e3  # ms per slot
            ref = outs[args.paths[0]]
            kr[f"g{g}_identical"] = all(np.array_equal(ref, o) for o in outs.values())
        res["per_kind"][kind] = kr
        print(kind, json.dumps(kr), flush=True)
    res["page_cache_dropped"] = dropped
    res["ms_per_slot"] = {k: round(sum(v.values()), 3) for k, v in per_slot.items()}
    res["slots_per_s_decrypt_only"] = {k: round(1e3 / v, 1) for k, v in res["ms_per_slot"].items() if v > 0}
    print(json.dumps(res, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
