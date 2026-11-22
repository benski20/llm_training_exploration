#!/usr/bin/env python3
"""Tokens per pass (tau) and mask slots per token of strict tree speculation, from exact target answers.

Replays exact greedy target answers (``spec_accept_bench.py --dump-tokens`` dumps, e.g. the Kaggle run 28
``tokens_q15.json`` / ``tokens_q3.json``) through the generator's tree acceptance rule: a pass emits the
target's token at the root, then follows the child whose token equals the target's next token while there is
one. On the target's own answer path a draft node's token is the draft's ``rank``-th choice given the true
prefix, so the draft only has to be run teacher-forced over each answer once: per answer position the rank
of the target's token in the draft's distribution (``ranks`` files, made by ``--draft`` or by
``draft_head_train.py``). This is exact for greedy decoding with a draft that sees the true prefix (a draft
model; a target-feature head fed the true tokens), up to float differences between this teacher-forced
draft and the engine's draft backend.

Mask slots: strict mode with the public charge ``prompt + j + min(B, floor((C - 1) j))`` before decode pass j
(``Generator.charge_target``): real rows of a pass are the pending token plus the longest node-order prefix of
the tree that the credit allows (the rest are padding rows); the store hands out ``C`` (capped) slots per
pass, L passes per answer of L tokens.

Usage::

    PYTHONPATH=src python benchmarks/tree_spec_sim.py --ranks ranks.json [--rs 2,3,4,6,8] [--json out.json]
    PYTHONPATH=src python benchmarks/tree_spec_sim.py --dump tokens_q15.json --draft Qwen/Qwen2.5-0.5B-Instruct \\
        --save-ranks ranks.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MAXRANK = 8  # ranks >= this count as "not in the draft's top choices"


# ---- topologies ------------------------------------------------------------------------------------------
def chain(r: int) -> list[tuple[int, int]]:
    return [(i, 0) for i in range(r - 1)]


def best_tree(r: int, depth_rank_p: list[list[float]]) -> list[tuple[int, int]]:
    """Static tree of ``r`` nodes (root included) maximising the expected accepted tokens when the target's
    token at depth d has rank k with probability ``p[d][k]`` independently (best-first growth: the value of a
    node is the product of its path's probabilities; children are added in rank order). Node order is the
    order of growth, so every prefix is a tree and the most valuable nodes come first."""
    nodes: list[tuple[int, int]] = []
    val = [1.0]
    depth = [0]
    nxt_rank = {0: 0}
    while len(val) < r:
        best, bv = None, -1.0
        for n in range(len(val)):
            k = nxt_rank.get(n, 0)
            d = depth[n]
            if d >= len(depth_rank_p) or k >= MAXRANK:
                continue
            v = val[n] * depth_rank_p[d][k]
            if v > bv:
                best, bv = n, v
        if best is None:
            break
        k = nxt_rank.get(best, 0)
        nodes.append((best, k))
        nxt_rank[best] = k + 1
        val.append(bv)
        depth.append(depth[best] + 1)
        nxt_rank[len(val) - 1] = 0
    return nodes


def topo_arrays(nodes: list[tuple[int, int]]) -> tuple[list[int], dict[int, dict[int, int]]]:
    parents = [-1] + [p for p, _ in nodes]
    kids: dict[int, dict[int, int]] = {}
    for i, (p, k) in enumerate(nodes, start=1):
        kids.setdefault(p, {})[k] = i
    return parents, kids


# ---- replay ----------------------------------------------------------------------------------------------
def replay(ranks: list[list[int]], nodes: list[tuple[int, int]], c: Fraction | None, b: int | None,
           prompt_lens: list[int], lookup_tok: list[list[list[int]]] | None = None,
           lookup_rows: int = 0) -> dict[str, float]:
    """Strict tree speculation over every answer: tau, slots per token, real rows per token.

    ``ranks[a][i]``: rank of answer token i under the draft at its true prefix (>= MAXRANK: absent).
    ``lookup_tok[a][i]``: optional prompt-lookup chain proposal at answer position i, compared with the answer
    (an extra root branch of ``lookup_rows`` rows; the pass keeps the better of the two branches)."""
    parents, kids = topo_arrays(nodes)
    r = len(parents) + lookup_rows
    toks = passes = real_rows = slots = 0
    for a, rk in enumerate(ranks):
        two_d = bool(rk) and isinstance(rk[0], list)
        L = len(rk)
        i, j, used = 0, 0, 0  # answer position, decode pass index (after the first token), masks used
        # the prefill pass emits token 0
        i = 1
        while i < L:
            j += 1  # decode pass j (1-based) charges up to target(j-1) in the generator: pass index before
            if c is None:
                credit = r - 1
            else:
                extra = math.floor((c - 1) * j)
                if b is not None:
                    extra = min(extra, b)
                target = j + extra  # slots held before decode pass j (prompt excluded): charge_target
                credit = max(0, min(r - 1, target - used - 1))
            n_tree = min(len(nodes), credit)
            n_look = min(lookup_rows, credit - n_tree) if lookup_rows else 0
            # walk the tree on the answer path using only nodes < n_tree + 1
            node, acc = 0, 0
            while i + acc < L - 0:
                if i + acc >= L - 1:
                    break
                x = rk[i + acc] if not two_d else (rk[i][acc] if acc < len(rk[i]) else MAXRANK)
                ch = kids.get(node, {}).get(x if x < MAXRANK else -1)
                if ch is None or ch > n_tree:
                    break
                node = ch
                acc += 1
            if n_look and lookup_tok is not None:
                prop = lookup_tok[a][i][:n_look]
                la = 0
                ans = ranks_answer[a]
                while la < len(prop) and i + la < L - 1 and prop[la] == ans[i + la]:
                    la += 1
                acc = max(acc, la)
            emitted = min(1 + acc, L - i)
            toks += emitted
            passes += 1
            used += 1 + n_tree + n_look
            real_rows += 1 + n_tree + n_look
            i += emitted
        # strict: L passes in all (L - 1 decode passes incl. dummies); slots are the public charge
        dec = L - 1
        if c is None:
            slots += dec * r
        else:
            extra = math.floor((c - 1) * dec)
            if b is not None:
                extra = min(extra, b)
            slots += dec + extra
        toks += 1  # token 0 from the prefill pass
        passes += 1
        real_rows += 1
    n_tok = sum(len(x) for x in ranks)
    return {"tau": toks / passes, "slots_per_token": slots / n_tok, "real_rows_per_token": real_rows / n_tok,
            "tokens": n_tok}


ranks_answer: list[list[int]] = []


def draft_ranks(dump: dict[str, Any], draft: str) -> list[list[int]]:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    m = AutoModelForCausalLM.from_pretrained(draft, dtype=torch.float32).eval()
    out = []
    for p, a in zip(dump["prompts"], dump["answers"], strict=True):
        ids = torch.tensor([p + a])
        with torch.no_grad():
            lg = m(ids).logits[0, len(p) - 1 : len(p) + len(a) - 1].float()
        tgt = torch.tensor(a)
        better = (lg > lg.gather(1, tgt[:, None])).sum(1)
        out.append([int(x) for x in better])
    return out


def depth_rank_probs(ranks: list[Any], depths: int = 8) -> list[list[float]]:
    """P(target token has draft rank k) per depth (1-D ranks: the same for every depth; 2-D ranks of a
    feature head: per depth, conditioned on the path above being on the answer)."""
    if ranks and ranks[0] and isinstance(ranks[0][0], list):
        out = []
        for d in range(depths):
            cnt, n = [0] * MAXRANK, 0
            for rk in ranks:
                for row in rk[1:]:
                    if d < len(row) and all(x == 0 for x in row[:d]):
                        n += 1
                        if row[d] < MAXRANK:
                            cnt[row[d]] += 1
            out.append([x / max(n, 1) for x in cnt])
        return out
    cnt = [0] * MAXRANK
    n = 0
    for rk in ranks:
        for x in rk[1:]:
            n += 1
            if x < MAXRANK:
                cnt[x] += 1
    p = [x / n for x in cnt]
    return [p for _ in range(depths)]


def expected_tau(nodes: list[tuple[int, int]], probs: list[float]) -> float:
    """Tokens per pass under the independence model: 1 + sum over nodes of the product of their path's rank
    probabilities (``probs[k]``: the target's token has draft rank k)."""
    val = [1.0]
    for par, k in nodes:
        val.append(val[par] * (probs[k] if k < len(probs) else 0.0))
    return sum(val)


def alpha_grid(shape: list[float], alphas: list[float], rs: list[int]) -> None:
    """tau of the chain and of the best static tree when rank 0 has probability alpha and the misses keep the
    measured rank shape (``shape``: P(rank k | not rank 0))."""
    print("alpha | " + " | ".join(f"R={r} chain / tree" for r in rs))
    for a in alphas:
        probs = [a] + [(1 - a) * x for x in shape]
        cells = []
        for r in rs:
            t = best_tree(r, [probs] * 32)
            cells.append(f"{expected_tau(chain(r), probs):.2f} / {expected_tau(t, probs):.2f}")
        print(f"{a:.2f} | " + " | ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", nargs="+", help="ranks json files: {name: [[rank per answer token]]}")
    ap.add_argument("--dump", help="token dump (prompts, answers) for --draft / --lookup")
    ap.add_argument("--draft", help="HF draft model: compute and save ranks")
    ap.add_argument("--save-ranks")
    ap.add_argument("--rs", default="2,3,4,5,6,8")
    ap.add_argument("--fit", help="ranks file to fit tree shapes on (default: the evaluated one)")
    ap.add_argument("--lookup-rows", default="", help="extra root branch from prompt lookup, e.g. 0,2,3")
    ap.add_argument("--alpha-grid", action="store_true", help="independence-model tau vs alpha (rank shape of --ranks)")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.alpha_grid:
        rk = json.load(open(args.ranks[0]))["ranks"]
        p0 = depth_rank_probs(rk)[0]
        shape = [x / (1 - p0[0]) for x in p0[1:]]
        print("measured rank probs", [round(x, 3) for x in p0[:5]])
        alpha_grid(shape, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9], [int(x) for x in args.rs.split(",")])
        return
    if args.draft:
        dump = json.load(open(args.dump))
        rk = draft_ranks(dump, args.draft)
        json.dump({"ranks": rk, "answers": dump["answers"], "prompt_lens": [len(p) for p in dump["prompts"]]},
                  open(args.save_ranks, "w"))
        print("saved", args.save_ranks)
        return
    results: dict[str, Any] = {}
    for f in args.ranks:
        d = json.load(open(f))
        rk = d["ranks"] if "ranks" in d else d["ranks2d"]
        global ranks_answer
        ranks_answer = d["answers"]
        fit = (lambda x: x.get("ranks", x.get("ranks2d")))(json.load(open(args.fit))) if args.fit else rk
        p = depth_rank_probs(fit)
        res: dict[str, Any] = {"rank_p": [x[:4] for x in p[:4]]}
        look = None
        lrows = [int(x) for x in args.lookup_rows.split(",") if x]
        if lrows:
            from pgw.engine.generate import _lookup_drafts_fixed

            kmax = max(lrows)
            look = [[_lookup_drafts_fixed(pr + an[:i], kmax, 4, 2) for i in range(len(an))]
                    for pr, an in zip(json.load(open(args.dump))["prompts"], d["answers"], strict=True)]
        for lr in lrows:
            for r in [int(x) for x in args.rs.split(",")]:
                if r - lr < 1:
                    continue
                for shape, nodes in (("lookup only", []), ("chain", chain(r - lr)), ("tree", best_tree(r - lr, p))):
                    if shape == "lookup only" and r - lr != 1:
                        continue
                    c, b = Fraction(r), 160
                    key = f"R={r} {shape}+lookup{lr} C={c} B={b}"
                    out = replay(rk, nodes, c, b, d["prompt_lens"], look, lr)
                    res[key] = out
                    print(f"{Path(f).stem:24s} {key:32s} tau {out['tau']:.2f} slots/tok "
                          f"{out['slots_per_token']:.2f} real rows/tok {out['real_rows_per_token']:.2f}")
        for r in [int(x) for x in args.rs.split(",")]:
            for shape, nodes in (("chain", chain(r)), ("tree", best_tree(r, p))):
                for c, b in ((None, None), (Fraction(r), 160), (Fraction(r), 320), (Fraction(2), None)):
                    if c is not None and c > r:
                        continue
                    key = f"R={r} {shape} C={'filler' if c is None else c}{'' if b is None else f' B={b}'}"
                    out = replay(rk, nodes, c, b, d["prompt_lens"])
                    out["nodes"] = nodes
                    res[key] = out
                    print(f"{Path(f).stem:24s} {key:32s} tau {out['tau']:.2f} slots/tok "
                          f"{out['slots_per_token']:.2f} real rows/tok {out['real_rows_per_token']:.2f}"
                          + (f"  {nodes}" if shape == "tree" and c is None else ""))
        results[Path(f).stem] = res
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
