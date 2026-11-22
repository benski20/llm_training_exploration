#!/usr/bin/env python3
"""Strict speculation before/after L3 on the CPU: masks per token, passes, user-visible tokens/s (masked path).

Runs the real :class:`pgw.engine.Generator` over :class:`MaskedBackend` with an in-process provider
(``LocalComputeClient``: the provider's ring GEMMs on this CPU) on a real pack, ``local_linears=attention``
(the Modal demo's placement: the provider sees gate_up and down). Configurations:

* ``plain``: no speculation (the reference; every other run must give the same tokens);
* ``filler R``: today's strict speculation (``spec_rows=R``, masked filler rows: R masks per pass);
* ``pad R/C``: L3 (``spec_mask_rows=C``): padding rows, C masks charged per pass.

Per configuration: generated tokens, provider passes (= L in strict mode), real passes, store masks per
token (what the reservation hands out: public), used masks per token (real masked rows), answer tokens/s
(first to last streamed token, what the user sees), and the dummy drain after the answer (the delay of a
follow-up turn of the same conversation).

Masks come from a bench-only reservation that cycles a few precomputed rows per linear (mask REUSE, never
do this outside a benchmark; outputs stay exact because each row's ``W r`` matches its ``r``), so the
numbers measure the gateway and provider work, not mask generation. CPU numbers: the provider's GEMMs run
on the same 4 shared vCPUs, so absolute tokens/s are far below a GPU provider's; the ratios between
configurations and the counts are what carry over.

Usage::

    PYTHONPATH=src python benchmarks/spec_no_compromise_bench.py --pack P [--prompts 2] [--max-tokens 48]
        [--configs plain,filler2,pad2/2,filler4,pad4/2,filler8,pad8/2,pad8/4] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


class _CycleReservation:
    """Bench only: ``rows`` precomputed mask rows per linear, handed out cyclically (mask reuse)."""

    def __init__(self, pack: Any, ring: Any, lids: list[Any], rows: int = 16):
        import torch

        from pgw.core.prng import SeedExpander

        self.ring = ring
        self.rows = rows
        self.r: dict[str, Any] = {}
        self.wr: dict[str, Any] = {}
        self.pos: dict[str, int] = {}
        self.uw: dict[str, Any] = {}
        exp = SeedExpander(os.urandom(32))
        for lid in lids:
            w = pack.load_linear(lid).w8.to(torch.int64)
            m, n = int(w.shape[0]), int(w.shape[1])
            r = exp.ring_uniform_rows(n, ring.k, 0, rows, "r", lid.key)
            self.r[lid.key] = r
            self.wr[lid.key] = (r @ w.t()) & ring.mask
            u = exp.ring_uniform((2, m), ring.k, "u", lid.key)
            self.uw[lid.key] = (u, (u @ w) & ring.mask)
            self.pos[lid.key] = 0
        self.taken = 0

    def remaining(self, lid: Any) -> int:
        return 1 << 30

    def rows_until_boundary(self, lid: Any) -> int:
        k = lid if isinstance(lid, str) else lid.key
        return self.rows - self.pos[k] % self.rows

    def take(self, lid: Any, rows: int) -> Any:
        k = lid if isinstance(lid, str) else lid.key
        p = self.pos[k] % self.rows
        self.pos[k] += rows
        return self.r[k][p:p + rows].clone(), self.wr[k][p:p + rows].clone()

    def checks(self, lid: Any) -> Any:
        k = lid if isinstance(lid, str) else lid.key
        u, uw = self.uw[k]
        return u.clone(), uw.clone()

    def close(self) -> None:
        pass


async def _run(gen: Any, ids: list[int], params: Any, session: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    stamps: list[float] = []
    toks: list[int] = []
    async for ev in gen.astream(ids, params, session=session, conversation="bench"):
        stamps.append(time.perf_counter())
        toks.append(ev.token_id)
    t_answer = time.perf_counter()
    await gen.wait_drained()
    t_drained = time.perf_counter()
    return {"tokens": toks, "t0": t0, "stamps": stamps, "answer_s": t_answer - t0, "drain_s": t_drained - t_answer}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--prompts", default="4,5", help="indices into spec_accept_bench.PROMPTS")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--configs", default="plain,filler2,pad2/2,filler4,pad4/2,filler8,pad8/2,pad8/4")
    ap.add_argument("--lookup", type=int, default=4)
    ap.add_argument("--local-linears", default="attention", help="'off' needs less memory (no local weights)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ref-file", default=None, help="plain answers (written by 'plain', read by the others)")
    args = ap.parse_args(argv)

    import torch
    from spec_accept_bench import PROMPTS

    from pgw.core.ring import RingParams
    from pgw.engine import ChatTokenizer, Generator, LocalComputeClient, MaskedBackend, SamplingParams, TransformerModel
    from pgw.modelpack import ModelPack

    ring = RingParams.from_name(args.ring)
    pack = ModelPack.open(args.pack, verify="fingerprint")
    tok = ChatTokenizer.from_pack(pack)
    client = LocalComputeClient(pack)
    backend = MaskedBackend(pack, None, client, ring, pack.fingerprint, local_linears=args.local_linears)
    model = TransformerModel(pack, backend)
    lids = [lid for layer in model._lids for lid in layer if not backend.local_linear(lid)]
    t0 = time.perf_counter()
    res = _CycleReservation(pack, ring, lids)
    print(f"[bench] precomputed masks in {time.perf_counter() - t0:.1f} s; torch threads {torch.get_num_threads()}",
          flush=True)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=0.8 if args.temperature else 1.0, seed=1234, ignore_eos=True)
    report: dict[str, Any] = {"pack": str(args.pack), "max_tokens": args.max_tokens, "prompts": args.prompts,
                              "temperature": args.temperature, "lookup": args.lookup, "rows": []}
    ref: dict[int, list[int]] = {}
    if args.ref_file and Path(args.ref_file).exists():  # plain answers from an earlier process (one config each)
        ref = {int(k): v for k, v in json.loads(Path(args.ref_file).read_text()).items()}
    for cfg in args.configs.split(","):
        kw: dict[str, Any] = {}
        if cfg.startswith("filler"):
            kw = {"spec_rows": int(cfg[6:]), "spec_strict": True, "lookup": args.lookup}
        elif cfg.startswith("pad"):
            r_s, c_s = cfg[3:].split("/")
            kw = {"spec_rows": int(r_s), "spec_strict": True, "lookup": args.lookup, "spec_mask_rows": float(c_s)}
        agg = {"tokens": 0, "passes": 0.0, "dummy": 0.0, "accepted": 0.0, "decode_s": 0.0, "answer_s": 0.0,
               "drain_s": 0.0, "used": 0, "store": 0, "identical": True}
        for pi in (int(x) for x in args.prompts.split(",")):
            ids = list(tok.apply_chat_template(PROMPTS[pi], add_generation_prompt=True))
            gen = Generator(model, tok, max_batch=1, **kw)
            before = dict(gen.spec_stats)
            out = asyncio.run(_run(gen, ids, params, res))
            asyncio.run(gen.aclose())
            if cfg == "plain":
                ref[pi] = out["tokens"]
                if args.ref_file:
                    Path(args.ref_file).write_text(json.dumps(ref))
            agg["identical"] &= out["tokens"] == ref.get(pi, out["tokens"])
            n = len(out["tokens"])
            agg["tokens"] += n
            st = gen.spec_stats
            agg["dummy"] += st.get("dummy_passes", 0) - before.get("dummy_passes", 0)
            agg["accepted"] += st.get("spec_accepted", 0)
            agg["decode_s"] += out["stamps"][-1] - out["stamps"][0]
            agg["answer_s"] += out["answer_s"]
            agg["drain_s"] += out["drain_s"]
            if "spec_mask_rows" in kw:  # real masked rows: pending tokens and the drafts sent
                used = int(st.get("spec_passes", 0) + st.get("spec_rows", 0))
            elif kw:  # masked filler rows: every row of every decode pass
                used = kw["spec_rows"] * (n - 1)
            else:
                used = n - 1
            agg["used"] += used
            if not kw:
                agg["store"] += n
            elif "spec_mask_rows" in kw:
                agg["store"] += gen.charge_target(0, n - 1)
            else:
                agg["store"] += kw["spec_rows"] * (n - 1)
            agg["passes"] += (st.get("spec_passes", 0) + 1) if kw else n
        toks = max(1, agg["tokens"])
        decode_toks = toks - len(args.prompts.split(","))
        row = {"config": cfg, "identical_to_plain": agg["identical"], "tokens": agg["tokens"],
               "real_passes": agg["passes"], "dummy_passes": agg["dummy"],
               "tokens_per_real_pass": round(toks / max(1, agg["passes"]), 3),
               "store_masks_per_token": round(agg["store"] / toks, 3),
               "used_masks_per_token": round(agg["used"] / toks, 3),
               "answer_tok_s": round(decode_toks / max(1e-9, agg["decode_s"]), 3),
               "answer_s": round(agg["answer_s"], 2), "drain_s": round(agg["drain_s"], 2)}
        import resource

        row["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
