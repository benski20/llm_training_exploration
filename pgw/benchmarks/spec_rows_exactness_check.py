"""Real-shape check: verify rows of an R-row pass vs single-row decoding, bit for bit (Qwen2.5-0.5B pack).

The exact integer reference backend (``PlainBackend``: the same integers as the masked path with every
linear remote) with the float path ``torch`` or ``native``. For each R a prompt is prefilled, then R tokens
are fed as one verify pass (serial tail, ``last_logits=R``) and, on a second cache, one at a time; the
script reports how many of the R logit rows differ and the largest absolute difference.

Usage: PYTHONPATH=src python benchmarks/spec_rows_exactness_check.py [--pack DIR] [--rows 2 4 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_PACK = Path.home() / ".pgw-client-demo/packs/qwen2.5-0.5b-instruct"


def main() -> None:
    from pgw.core import DEFAULT_RING
    from pgw.engine import PlainBackend, SequenceInput, TransformerModel
    from pgw.modelpack import ModelPack

    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--prompt", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    pack = ModelPack.open(args.pack, verify="none")
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 150000, (args.prompt,), generator=g).tolist()
    out = []
    for ops in ("torch", "native"):
        model = TransformerModel(pack, PlainBackend(pack, DEFAULT_RING))
        if ops == "native":
            model.set_float_ops("native")
        for r in args.rows:
            ids = torch.randint(0, 150000, (r,), generator=g).tolist()
            c1 = model.new_cache(args.prompt + r + 4)
            model.forward_batch([SequenceInput(prompt, c1)])
            lg = model.forward_batch([SequenceInput(ids, c1, last_logits=r, serial_tail=r - 1)])[0]
            c2 = model.new_cache(args.prompt + r + 4)
            model.forward_batch([SequenceInput(prompt, c2)])
            diff_rows, maxdiff = 0, 0.0
            for j in range(r):
                one = model.forward_batch([SequenceInput([ids[j]], c2)])[0].reshape(-1)
                row = lg[j]
                if not torch.equal(row, one):
                    diff_rows += 1
                    maxdiff = max(maxdiff, float((row - one).abs().max()))
            rec = {"float_ops": ops, "rows": r, "rows_differing": diff_rows, "max_abs_diff": maxdiff,
                   "torch_threads": args.threads}
            print(json.dumps(rec), flush=True)
            out.append(rec)


if __name__ == "__main__":
    main()
