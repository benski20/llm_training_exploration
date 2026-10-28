"""What VeriChat costs on top of plain generation, on the same model and prompt.

  plain        model.generate, greedy, KV cache (without and with the document in the prompt)
  scored       VeriChat without documents: the probe reads the hidden states of every step
  provenance   VeriChat with documents: plus one forward pass and the exact decomposition
               (compared with plain generation on the same document prompt)
  judge        the general detector scoring an existing reply (one forward pass, no generation)

    python experiments/latency.py --model Qwen/Qwen2.5-1.5B-Instruct --head <correctness head> --detector <head>
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.verichat import VeriChat  # noqa: E402

DOC = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It was designed and built by "
       "the company of Gustave Eiffel and completed in 1889 for the World's Fair. It is 330 metres tall. "
       "It was the tallest man-made structure in the world until 1930.")
QS = ["Who designed the Eiffel Tower?", "When was the Eiffel Tower completed?", "How tall is the Eiffel Tower?",
      "What was the tallest structure before 1930?"]


def timed(f, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        out.append(time.perf_counter() - t)
    return sorted(out)[len(out) // 2], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--head", default="results/kaggle/qwen-correctness/correctness/Qwen2.5-1.5B-Instruct_heads.pt")
    ap.add_argument("--detector", help="a detector head for the same model (optional)")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/latency.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bot = VeriChat.load(a.model, a.head)
    tok, model = bot.tok, bot.model
    dev = model.get_input_embeddings().weight.device
    res = {"model": a.model, "device": str(dev), "threads": a.threads, "max_new_tokens": a.tokens, "per_question": []}
    for q in QS:
        msgs = [{"role": "user", "content": q}]
        n = {}

        def plain(docs, key):
            ids = tok.apply_chat_template(bot._messages(msgs, docs), add_generation_prompt=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
            n[key] = o.shape[1] - ids.shape[1]
            return o
        t_plain, _ = timed(lambda: plain([], "plain"), 3)
        t_plain_doc, _ = timed(lambda: plain([DOC], "plain_doc"), 3)
        t_scored, r1 = timed(lambda: bot.reply(msgs, documents=None, mode="flag", max_new_tokens=a.tokens), 3)
        t_prov, r2 = timed(lambda: bot.reply(msgs, documents=[DOC], mode="flag", max_new_tokens=a.tokens), 3)
        row = {"q": q, "plain_s": round(t_plain, 3), "plain_tokens": n["plain"], "plain_doc_s": round(t_plain_doc, 3),
               "scored_s": round(t_scored, 3), "scored_tokens": len(tok(r1["text"], add_special_tokens=False).input_ids),
               "provenance_s": round(t_prov, 3), "reply": r2["text"],
               "claims": [(c["text"], round(c["score"], 2), c.get("source")) for c in r2["claims"]]}
        if a.detector:
            from vt.detector import Detector
            det = Detector(model, tok, torch.load(a.detector, map_location="cpu", weights_only=True))
            row["judge_s"] = round(timed(lambda: det.judge(msgs, r2["text"], documents=[DOC]), 3)[0], 3)
        res["per_question"].append(row)
        print(json.dumps(row), flush=True)
    pq = res["per_question"]
    for k in ("plain_s", "plain_doc_s", "scored_s", "provenance_s", "judge_s"):
        if all(k in r for r in pq):
            res[k.replace("_s", "_mean_s")] = round(sum(r[k] for r in pq) / len(pq), 3)
    res["scored_overhead"] = round(res["scored_mean_s"] / res["plain_mean_s"] - 1, 3)
    res["provenance_overhead"] = round(res["provenance_mean_s"] / res["plain_doc_mean_s"] - 1, 3)
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
