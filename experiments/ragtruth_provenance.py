"""Exact per-token provenance as a RAG faithfulness signal, on RAGTruth (human-labelled hallucinated spans
in RAG responses from GPT-4, GPT-3.5, Llama-2 and Mistral).

Each response is read by a local model in one teacher-forced pass (`vt.qwen_glassbox.explain_all`):
the logit of every response token splits exactly into what the document contributed, what the
instruction, the chat template and the response so far contributed, and the model's own biases.
Tokens inside labelled hallucinations should draw less on the document. Reports token-level AUROC
of each share and response-level AUROC of the worst token.

    python experiments/ragtruth_provenance.py --n 40
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.correctness import auroc  # noqa: E402
from vt.qwen_glassbox import explain_all  # noqa: E402


def groups_for(tok, prompt, doc, reply_start_char):
    enc = tok(prompt, return_offsets_mapping=True)
    a = prompt.find(doc[:200]) if doc else -1
    b = a + len(doc) if a >= 0 else -1
    g = []
    for s, e in enc["offset_mapping"]:
        if s >= reply_start_char:
            g.append("answer")
        elif a >= 0 and s < b and e > a:
            g.append("document")
        else:
            g.append("instruction")
    return enc["input_ids"], enc["offset_mapping"], g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/ragtruth_provenance.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    ds = load_dataset("wandb/RAGTruth-processed", split="test").shuffle(seed=0)
    tok_rows, resp_rows, t0 = [], [], time.time()
    for x in ds:
        if len(resp_rows) >= a.n:
            break
        head = tok.apply_chat_template([{"role": "user", "content": x["input_str"]}], tokenize=False, add_generation_prompt=True)
        prompt = head + x["output"]
        ids, offs, groups = groups_for(tok, prompt, x["context"], len(head))
        if len(ids) > a.max_tokens or "document" not in groups:
            continue
        spans = [(l["start"] + len(head), l["end"] + len(head)) for l in json.loads(x["hallucination_labels"])]
        start = groups.index("answer") - 1                       # position predicting the first response token
        names, contrib, logits = explain_all(model, torch.tensor([ids]), groups, start=start)
        c = contrib.numpy()
        tot = np.abs(c).sum(0) + 1e-6
        k = {n: i for i, n in enumerate(names)}
        risks = []
        for j in range(c.shape[1]):
            s, e = offs[start + 1 + j]
            lab = int(any(s < hi and e > lo for lo, hi in spans))
            doc = c[k["document"], j]
            r = {"label": lab, "doc_share": float(doc / tot[j]), "doc": float(doc),
                 "bias_share": float(c[k["bias"], j] / tot[j]), "answer_share": float(c[k["answer"], j] / tot[j]),
                 "logit": float(logits[j]), "task": x["task_type"]}
            tok_rows.append(r)
            risks.append(-r["doc_share"])
        resp_rows.append({"hallucinated": int(bool(spans)), "worst": float(np.percentile(risks, 95)),
                          "mean_doc_share": float(-np.mean(risks)), "task": x["task_type"], "src_model": x["model"]})
        if len(resp_rows) % 10 == 0:
            print(json.dumps({"done": len(resp_rows), "tokens": len(tok_rows), "s": round(time.time() - t0)}), flush=True)
    y = [r["label"] for r in tok_rows]
    res = {"model": a.model, "responses": len(resp_rows), "tokens": len(tok_rows),
           "hallucinated_tokens": int(sum(y)), "hallucinated_responses": int(sum(r["hallucinated"] for r in resp_rows)),
           "token_auroc": {f: auroc([-r[f] if f == "doc_share" else r[f] for r in tok_rows], y)
                           for f in ("doc_share", "bias_share", "answer_share", "logit")},
           "response_auroc": {"worst_token_low_doc_share": auroc([r["worst"] for r in resp_rows], [r["hallucinated"] for r in resp_rows]),
                              "low_mean_doc_share": auroc([-r["mean_doc_share"] for r in resp_rows], [r["hallucinated"] for r in resp_rows])},
           "mean_doc_share": {"hallucinated": float(np.mean([r["doc_share"] for r in tok_rows if r["label"]] or [0])),
                              "supported": float(np.mean([r["doc_share"] for r in tok_rows if not r["label"]] or [0]))}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
