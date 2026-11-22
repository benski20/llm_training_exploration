"""Draft-model cost on a CPU-only gateway: ms per draft token of a Qwen2.5-0.5B draft.

Backends compared (``--backend``):

* ``torch-int8``: today's ``Int8DraftBackend`` (``torch._weight_int8pack_mm``, weight-only int8) with the
  ``int8_lm_head`` (torch threads = ``--threads``);
* ``native-int8``: ``NativeInt8DraftBackend`` (``pgw_fast.local_linear_into``: the target ring's quantizer,
  exact int16 x int8 products, columns over the native pool) with ``native_int8_lm_head``. The native
  pool size is fixed per process (``PGW_POOL_WORKERS`` = cores - 1), so run one process per core count.
* ``float``: ``FloatBackend`` float32 (reference cost).

Each run: a prompt prefill, then ``--tokens`` one-row draft steps (argmax fed back, as drafting does),
and ``--tokens`` four-row steps (a catch-up step's row count) for the multi-row cost. Prints JSON.

Usage (one process per core count)::

    PGW_POOL_WORKERS=0 python benchmarks/draft_cpu_bench.py --backend native-int8 --threads 1
    PGW_POOL_WORKERS=3 python benchmarks/draft_cpu_bench.py --backend native-int8 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

DEFAULT_PACK = Path.home() / ".pgw-client-demo/packs/qwen2.5-0.5b-instruct"


def build(pack_path: Path, backend: str, float_ops: str) -> object:
    from pgw.core import DEFAULT_RING
    from pgw.engine import FloatBackend, TransformerModel
    from pgw.engine.backends import Int8DraftBackend, NativeInt8DraftBackend, int8_lm_head, native_int8_lm_head
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(pack_path, verify="none")
    if backend == "torch-int8":
        model = TransformerModel(pack, Int8DraftBackend(pack, DEFAULT_RING))
        model.lm_head_fn = int8_lm_head(model.lm_head)
    elif backend == "native-int8":
        be = NativeInt8DraftBackend(pack, DEFAULT_RING)
        be.preload()
        model = TransformerModel(pack, be)
        model.lm_head_fn = native_int8_lm_head(model.lm_head, DEFAULT_RING)
    else:
        model = TransformerModel(pack, FloatBackend(pack, DEFAULT_RING, batch_rows=True))
    if float_ops == "native":
        model.set_float_ops("native")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--backend", choices=["torch-int8", "native-int8", "float"], default="native-int8")
    ap.add_argument("--threads", type=int, default=1, help="torch threads (and the label for the native pool)")
    ap.add_argument("--float-ops", choices=["torch", "native"], default="native")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", type=int, default=96)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pgw.engine import SequenceInput

    t0 = time.perf_counter()
    model = build(args.pack, args.backend, args.float_ops)
    load_s = time.perf_counter() - t0
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 150000, (args.prompt,), generator=g).tolist()
    cache = model.new_cache(args.prompt + 4 * args.tokens + 64)
    model.forward_batch([SequenceInput(prompt, cache)])
    tok = prompt[-1]
    one: list[float] = []
    toks: list[int] = []
    for i in range(args.tokens + 3):
        t = time.perf_counter()
        lg = model.forward_batch([SequenceInput([tok], cache)])[0]
        tok = int(torch.argmax(lg))
        dt = time.perf_counter() - t
        if i >= 3:
            one.append(dt * 1e3)
        toks.append(tok)
    four: list[float] = []
    base = cache.length
    for i in range(args.tokens // 2 + 2):
        cache.truncate(base)
        t = time.perf_counter()
        model.forward_batch([SequenceInput(toks[:4], cache)])
        dt = time.perf_counter() - t
        if i >= 2:
            four.append(dt * 1e3)
    print(json.dumps({
        "backend": args.backend, "threads": args.threads, "pool_workers": os.environ.get("PGW_POOL_WORKERS"),
        "float_ops": args.float_ops, "load_s": round(load_s, 1),
        "ms_per_token_1row_median": round(statistics.median(one), 2),
        "ms_per_token_1row_min": round(min(one), 2),
        "ms_4row_step_median": round(statistics.median(four), 2),
        "tokens": toks[:8], "cpus": os.cpu_count(), "load": os.getloadavg()[0],
    }), flush=True)


if __name__ == "__main__":
    main()
