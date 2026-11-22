#!/usr/bin/env python3
"""Masks, passes and tokens per pass of strict speculation under a public mask charge (L3).

Replays exact greedy target answers (``spec_accept_bench.py --dump-tokens``) through the generator's
draft budget rules, so the numbers are what ``Generator(spec_rows=R, spec_strict=True, spec_mask_rows=C)``
would do on those texts (greedy acceptance: a pass at position ``i`` accepts the longest correct prefix of
the drafts it sends). Per (R, C) it reports:

* ``tokens_per_pass``: generated tokens per real decode pass (the user-visible speedup factor tau);
* ``store_masks_per_token``: mask slots the store hands out per generated token (public: prompt + C per
  decode pass in strict mode, real and dummy passes alike; R with today's masked filler rows, C = None);
* ``used_masks_per_token``: real masked rows per token (pending token plus drafts sent; private);
* ``dummy_share``: dummy passes / all passes (the follow-up delay is that many passes).

Drafters: prompt lookup over the conversation (the gateway's ``_lookup_drafts_fixed``, n = ``--max-n`` down to
``--min-n``), optionally ``--tree 2``: up to two lookup candidates (the latest two distinct continuations) in a
token tree of R - 1 draft rows (an upper bound for tree verification, not implemented in the engine). A draft
model's proposals can be supplied as a ``model_m`` list per prompt in the dump (correct-prefix lengths); then the
drafter always proposes R - 1 tokens.

Usage::

    PYTHONPATH=src python benchmarks/spec_charge_sim.py --tokens tok.json [--rs 2,4,8] [--cs 1,2,3,4] [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _latest_matches(seq: list[int], k: int, max_n: int, min_n: int, want: int) -> list[list[int]]:
    """Up to ``want`` distinct continuations of the latest earlier matches of the longest suffix n-gram."""
    out: list[list[int]] = []
    L = len(seq)
    for n in range(min(max_n, L - 1), min_n - 1, -1):
        tail = seq[L - n:]
        for s in range(L - n - 1, -1, -1):
            if seq[s:s + n] == tail:
                cont = seq[s + n:s + n + k]
                if cont and cont not in out:
                    out.append(cont)
                    if len(out) >= want:
                        return out
        if out:
            return out
    return out


def lookup_tables(full: list[int], p: int, gen: list[int], k: int, max_n: int, min_n: int,
                  tree: int) -> tuple[list[int], list[int]]:
    """Per answer position ``i``: (proposal length sent by a chain drafter, correct prefix length).

    With ``tree`` = 2 the correct length is the best of two candidates (an upper bound of a 2-branch tree
    whose rows are split between the branches is computed in :func:`simulate`)."""
    from pgw.engine.generate import _lookup_drafts_fixed

    plen = [0] * len(gen)
    corr = [0] * len(gen)
    for i in range(1, len(gen)):
        ctx = full[: p + i]
        cands = [_lookup_drafts_fixed(ctx, k, max_n, min_n)] if tree <= 1 else _latest_matches(ctx, k, max_n, min_n,
                                                                                                 tree)
        best = 0
        for c in cands:
            m = 0
            while m < len(c) and i + m < len(gen) and c[m] == gen[i + m]:
                m += 1
            best = max(best, m)
        plen[i] = max((len(c) for c in cands), default=0)
        corr[i] = best
    return plen, corr


def charge(c: float, j: int, credit_cap: int | None) -> int:
    """Public charge of decode pass ``j`` beyond the prompt: ``j + min(cap, floor((C - 1) * j))``."""
    extra = int((c - 1) * j)
    return j + (extra if credit_cap is None else min(credit_cap, extra))


def simulate(plen: list[int], corr: list[int], n_gen: int, prompt: int, r: int, c: float | None,
             tree: int = 1, credit_cap: int | None = None) -> dict[str, float]:
    """One answer through the strict generator's rules. ``c`` None: masked filler rows (R masks per pass)."""
    i, passes, used = 1, 0, prompt
    while i < n_gen:
        passes += 1  # decode pass index j = passes
        k = r - 1
        left = n_gen - i  # tokens still to emit after this pass's first
        if c is None:
            d = min(plen[i], k)
        else:
            credit = prompt + charge(c, passes, credit_cap) - used - 1
            d = max(0, min(plen[i], k, credit, left - 1))
        if tree > 1 and d >= 2:  # two branches share the d rows: the longer branch gets ceil(d/2)
            a = min(corr[i], (d + 1) // 2)
        else:
            a = min(corr[i], d)
        used += 1 + d
        i += a + 1
    total_passes = max(1, n_gen)  # strict: prefill + decode + dummy passes = L
    store = prompt + (r * (n_gen - 1) if c is None else charge(c, n_gen - 1, credit_cap))
    if c is None:
        used = prompt + r * passes  # masked fillers: every row of a real pass is masked
    return {"real_decode_passes": passes, "dummy": total_passes - 1 - passes, "store": store, "used": used}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", required=True, help="spec_accept_bench --dump-tokens output")
    ap.add_argument("--rs", default="2,4,8")
    ap.add_argument("--cs", default="1,2,3,4,8")
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--tree", type=int, default=1)
    ap.add_argument("--credit-cap", type=int, default=None, help="cap of the extra (draft) slots charged")
    ap.add_argument("--draft-pack", default=None, help="draft model: its teacher-forced matches over the answers")
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-tokens", default=None, help="write the dump plus the draft's matches here (reuse)")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.tokens).read_text())
    prompts, gens = data["prompts"], data["answers"]
    if args.draft_pack and "model_m" not in data:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from spec_accept_bench import _run_lengths, build_draft, model_match

        draft, _ = build_draft(args.draft_pack, "K32A12C27", "cpu")
        dv = int(draft.arch.vocab_size)
        data["model_m"] = [_run_lengths(model_match(draft, p + g, len(p), g, dv)) for p, g in zip(prompts, gens,
                                                                                                    strict=True)]
        if args.save_tokens:
            Path(args.save_tokens).write_text(json.dumps(data))
    rs = [int(x) for x in args.rs.split(",")]
    cs: list[float | None] = [None] + [float(x) for x in args.cs.split(",")]
    kmax = max(rs) - 1
    tables = [lookup_tables(p + g, len(p), g, kmax, args.max_n, args.min_n, args.tree)
              for p, g in zip(prompts, gens, strict=True)]
    model_m = data.get("model_m")
    rows: list[dict[str, Any]] = []
    for r in rs:
        for c in cs:
            if c is not None and c > r:
                continue
            tot = {"tokens": 0, "real": 0, "dummy": 0, "store": 0, "used": 0, "prompt": 0}
            for j, ((plen, corr), p, g) in enumerate(zip(tables, prompts, gens, strict=True)):
                if model_m is not None:  # a draft model always proposes R - 1 tokens
                    plen = [r - 1] * len(g)
                    corr = [max(a, b) for a, b in zip(corr, model_m[j], strict=True)]
                s = simulate(plen, corr, len(g), len(p), r, c, args.tree, args.credit_cap)
                tot["tokens"] += len(g)
                tot["real"] += s["real_decode_passes"] + 1  # + the prefill pass
                tot["dummy"] += s["dummy"]
                tot["store"] += s["store"] - len(p)
                tot["used"] += s["used"] - len(p)
                tot["prompt"] += len(p)
            n = max(1, tot["tokens"])
            rows.append({"R": r, "C": c, "tokens_per_pass": round(n / tot["real"], 3),
                         "store_masks_per_token": round(tot["store"] / n, 3),
                         "used_masks_per_token": round(tot["used"] / n, 3),
                         "dummy_share": round(tot["dummy"] / n, 3)})
    base = sum(len(g) for g in gens)
    out = {"prompts": len(gens), "answer_tokens": base, "max_n": args.max_n, "min_n": args.min_n,
           "tree": args.tree, "credit_cap": args.credit_cap, "model_drafts": model_m is not None, "rows": rows}
    for row in rows:
        print(json.dumps(row))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
