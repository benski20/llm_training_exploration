"""Does acting on internal doubt mid-generation cut hallucinations? (vt/guided.py)

One greedy generation per question in flag mode records each claim's P(correct) from the probe and
the claim token's probability. Greedy text is the same in every mode up to the stop point, so this
gives the outcome of stop mode at any threshold: a reply is withheld if any claim is doubted.

At matched abstention rates it compares two triggers: the probe on the hidden state (ours, no extra
cost), and the claim token's probability (what active-retrieval methods such as FLARE use). Reports
the hallucination rate (wrong answers / all questions) and accuracy of what is still answered.

    python experiments/doubt_aware_eval.py --n 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.correctness import SYSTEM, CorrectnessHead, is_right  # noqa: E402
from vt.guided import DoubtAwareGenerator  # noqa: E402


def outcome(rows, score, keep_frac):
    """Withhold the (1 - keep_frac) replies with the lowest min-claim score; replies without claims are kept."""
    s = np.array([score(r) for r in rows])
    order = np.argsort(s, kind="mergesort")
    withheld = set(order[: int(round(len(rows) * (1 - keep_frac)))].tolist())
    answered = [r for i, r in enumerate(rows) if i not in withheld]
    wrong = sum(1 - r["right"] for r in answered)
    return {"abstain": round(1 - len(answered) / len(rows), 3), "hallucination": round(wrong / len(rows), 3),
            "accuracy_answered": round(float(np.mean([r["right"] for r in answered])), 3) if answered else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--heads", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-0.5B-Instruct_heads.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--offset", type=int, default=1000, help="TriviaQA validation index to start at (after the test set)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--out", default="results/doubt_aware_eval.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    head = CorrectnessHead.from_state(torch.load(a.heads, weights_only=True)["probe+output"])
    gen = DoubtAwareGenerator(model, tok, head.probe, head.layers[0], threshold=0.0)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(a.offset, a.offset + a.n))
    rows = json.load(open(a.partial)) if os.path.exists(a.partial) else []
    t0 = time.time()
    for i, x in enumerate(ds):
        if i < len(rows):
            continue
        r = gen.generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": x["question"]}], 40, "flag")
        rows.append({"q": x["question"], "reply": r["text"], "claims": r["claims"],
                     "right": is_right(r["text"], x["answer"]["normalized_aliases"] + [x["answer"]["value"]])})
        if (i + 1) % 10 == 0:
            json.dump(rows, open(a.partial, "w"))
            print(json.dumps({"done": i + 1, "acc": float(np.mean([r_["right"] for r_ in rows])),
                              "s": round(time.time() - t0)}), flush=True)
    json.dump(rows, open(a.partial, "w"))
    def probe(r): return min((c["p_correct"] for c in r["claims"]), default=1.0)
    def tokp(r): return min((c["token_p"] for c in r["claims"]), default=1.0)
    res = {"model": a.model, "n": len(rows), "accuracy": float(np.mean([r["right"] for r in rows])),
           "with_claim": float(np.mean([bool(r["claims"]) for r in rows])), "curves": {}}
    for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        res["curves"][f"answer_{int(keep * 100)}pct"] = {"probe": outcome(rows, probe, keep),
                                                         "token_probability": outcome(rows, tokp, keep)}
    res["examples"] = [{k: r[k] for k in ("q", "reply", "right")} | {"min_p": probe(r)} for r in rows[:10]]
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
