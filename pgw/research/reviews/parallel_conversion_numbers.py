#!/usr/bin/env python3
"""Tables of ``research/reviews/parallel-conversion.md`` from the Kaggle results (stdlib only).

    python research/reviews/parallel_conversion_numbers.py [results dir]

Reads ``benchmarks/results/parallel_conversion/kaggle_t4_r2/<model>.json`` (``benchmarks/parallel_conversion.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "benchmarks/results/parallel_conversion/kaggle_t4_r2"
MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen3-1.7b", "qwen2.5-3b", "qwen3-4b"]
THRESH = (0.02, 0.05, 0.1, 0.2)


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    m = (n - 1) / 2
    num = sum((x - m) * (y - m) for x, y in zip(ra, rb, strict=True))
    return num / sum((x - m) ** 2 for x in ra)


def load(d: Path) -> dict[str, dict]:
    return {m: json.loads((d / f"{m}.json").read_text()) for m in MODELS if (d / f"{m}.json").is_file()}


def max_k(curve: list[dict], kl: float) -> dict:
    """The largest k of a (prefix) curve whose selection KL stays within ``kl`` (every smaller k too)."""
    best = curve[0]
    for p in sorted(curve, key=lambda p: p["k"]):
        if p["kl"] > kl:
            break
        best = p
    return best


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    res = load(d)
    print("## Sensitivity vs measured single-layer KL (Spearman over layers)\n")
    print("| model | L | rel_out | rel_mlp | rel_in | 1 - cos_in | most tolerant layers (single-layer KL) | "
          "least tolerant |")
    print("|---|---|---|---|---|---|---|---|")
    for m, r in res.items():
        L = r["layers"]
        single = [s["kl"] for s in r["single"]]
        sens = r["sensitivity"]
        c = {k: spearman(single, [s[k] for s in sens]) for k in ("rel_out", "rel_mlp", "rel_in")}
        c["cos"] = spearman(single, [1 - s["cos_in"] for s in sens])
        best = sorted(range(L), key=lambda i: single[i])
        print(f"| {m} | {L} | {c['rel_out']:.2f} | {c['rel_mlp']:.2f} | {c['rel_in']:.2f} | {c['cos']:.2f} | "
              + ", ".join(f"{i} ({single[i]:.3f})" for i in best[:6]) + " | "
              + ", ".join(f"{i} ({single[i]:.2f})" for i in sorted(range(L), key=lambda i: -single[i])[:3]) + " |")
    print("\n## Orderings: mean selection KL over k = 0, 2, 4, ..., L (lower is better)\n")
    names = ["sensitivity", "single_kl", "greedy", "late_first", "early_first"]
    print("| model | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m, r in res.items():
        print(f"| {m} | " + " | ".join(f"{r['order_area'][n]:.3f}" for n in names) + " |")
    print("\n## Training-free budget: most layers converted (greedy order) within a selection-KL budget\n")
    print("| model | L | trips 2L+1 | " + " | ".join(f"KL <= {t}: k / trips / chat top-1 / ppl x" for t in THRESH)
          + " |")
    print("|---|---|---|" + "---|" * len(THRESH))
    for m, r in res.items():
        L = r["layers"]
        cur = r["curves"]["greedy"]
        p0 = cur[0]["web_ppl"]
        cells = []
        for t in THRESH:
            p = max_k(cur, t)
            cells.append(f"{p['k']} / {p['trips']} / {p['chat_top1']:.3f} / {p['web_ppl'] / p0:.3f}")
        print(f"| {m} | {L} | {2 * L + 1} | " + " | ".join(cells) + " |")
    print("\n## Quality vs trips (greedy order, selection set): k / trips / KL / chat top-1 / web ppl\n")
    for m, r in res.items():
        L = r["layers"]
        step = max(1, L // 12)
        pts = [p for p in r["curves"]["greedy"] if p["k"] % step == 0 or p["k"] == L]
        print(f"* **{m}** (L = {L}): " + "; ".join(f"{p['k']} / {p['trips']} / {p['kl']:.3f} / "
                                                     f"{p['chat_top1']:.2f} / {p['web_ppl']:.1f}" for p in pts))
    print("\n## Report points (held-out harness sets, full KL; best ordering)\n")
    print("| model | k (share) | trips | FineWeb ppl | chat top-1 | chat KL | MMLU | ARC-E | ARC-C | HellaSwag |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in res.items():
        L = r["layers"]
        b = r["base_report"]
        rows = [(0, 2 * L + 1, b)] + [(p["k"], p["trips"], p) for p in r["report"]]
        for k, trips, p in rows:
            print(f"| {m} | {k} ({100 * k / L:.0f} %) | {trips} | {p['ppl_fineweb']:.2f} | "
                  f"{p['chat']['top1_agree']:.3f} | {p['chat']['kl']:.3f} | {p.get('mmlu', 0):.3f} | "
                  f"{p.get('arc_easy', 0):.3f} | {p.get('arc_challenge', 0):.3f} | {p.get('hellaswag', 0):.3f} |")
    print("\n## Best training-free map per model (greedy order, selection KL <= 0.1)\n")
    for m, r in res.items():
        L = r["layers"]
        p = max_k(r["curves"]["greedy"], 0.1)
        par = set(r["orders"]["greedy"][: p["k"]])
        lm = "".join("p" if i in par else "s" for i in range(L))
        print(f"* {m}: k = {p['k']}, {p['trips']} trips (from {2 * L + 1}), parallel layers "
              f"{sorted(par)}, map `{lm}`")


if __name__ == "__main__":
    main()
