"""Choose VeriChat's serving doubt score from per-token traces of the 32B (kaggle/chateval --trace).

For each candidate reply score: factual AUROC (wrong vs right TriviaQA replies) and, at the threshold
that flags at most 10% of everyday replies (poems, advice, code, small talk), the share of wrong and
right factual replies flagged.

    python experiments/serving_score.py results/kaggle/chateval-32t/chateval/traces.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from vt.correctness import auroc, auroc_ci  # noqa: E402


def content(tr):
    return [k for k, p in enumerate(tr["pieces"]) if any(ch.isalnum() for ch in p) and tr["risk"][k] is not None]


def claim_tokens(tr):
    """The tokens of each claim word: from its first token to the next word start."""
    ks, pieces = set(), tr["pieces"]
    for j, _, _ in tr["claims"]:
        k = j
        while k < len(pieces) and (k == j or not pieces[k][:1].isspace()) and pieces[k].strip(" .,;:") != "":
            ks.add(k)
            k += 1
    return sorted(k for k in ks if k < len(pieces) and tr["risk"][k] is not None)


CANDIDATES = {
    "claim P at decision (1 - min)": lambda tr, x: 1 - (x["reply_confidence"] if x["reply_confidence"] is not None else 1),
    "claim P (combined with token p)": lambda tr, x: 1 - min([(p + tp) / 2 for _, p, tp in tr["claims"]], default=1),
    "max risk, claim tokens": lambda tr, x: max([tr["risk"][k] for k in claim_tokens(tr)], default=0),
    "mean risk, claim tokens": lambda tr, x: float(np.mean([tr["risk"][k] for k in claim_tokens(tr)])) if claim_tokens(tr) else 0,
    "max risk, all content": lambda tr, x: max([tr["risk"][k] for k in content(tr)], default=0),
    "mean risk, all content": lambda tr, x: float(np.mean([tr["risk"][k] for k in content(tr)])) if content(tr) else 0,
    "p90 risk, all content": lambda tr, x: float(np.percentile([tr["risk"][k] for k in content(tr)], 90)) if content(tr) else 0,
    "min log-prob, all": lambda tr, x: -min([v for v in tr["logp"] if v is not None], default=0),
    "min log-prob, claim tokens": lambda tr, x: -min([tr["logp"][k] for k in claim_tokens(tr) if tr["logp"][k] is not None],
                                                     default=0),
}


def main(path, budget=0.10):
    T = json.load(open(path))
    ev = [x for x in T if x["kind"] == "everyday"]
    fa = [x for x in T if x["kind"] == "factual"]
    y = np.array([x["wrong"] for x in fa])
    out = {"everyday": len(ev), "factual": len(fa), "wrong": int(y.sum()),
           "everyday_with_claims": round(float(np.mean([bool(x["trace"]["claims"]) for x in ev])), 3), "scores": {}}
    for name, f in CANDIDATES.items():
        e = np.array([f(x["trace"], x) for x in ev])
        s = np.array([f(x["trace"], x) for x in fa])
        t = np.quantile(e, 1 - budget) + 1e-9               # at most `budget` of everyday replies above it
        out["scores"][name] = {"auroc": round(auroc(s, y), 3), "ci": [round(v, 3) for v in auroc_ci(s, y)],
                               "threshold": round(float(t), 4), "everyday_flagged": round(float((e > t).mean()), 3),
                               "wrong_flagged": round(float((s[y == 1] > t).mean()), 3),
                               "right_flagged": round(float((s[y == 0] > t).mean()), 3)}
    return out


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "scores"}))
    for k, v in sorted(r["scores"].items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{k:34s} AUROC {v['auroc']:.3f} {v['ci']}  at t={v['threshold']:.3f}: everyday {v['everyday_flagged']:.2f}"
              f"  wrong {v['wrong_flagged']:.2f}  right {v['right_flagged']:.2f}")
