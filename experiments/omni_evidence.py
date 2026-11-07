"""Evidence shift for the corpus's document-grounded replies: each reply read with its documents and without them,
per token  shift = log p(token | documents, question, reply so far) - log p(token | question, reply so far).
Supported words rise, contradicted ones fall, unsupported ones barely move (experiments/evidence_shift.py). Stored per
reply and per unit, as one more label-free expert for the omni verifier.

    python experiments/omni_evidence.py --corpus corpus.jsonl --reader Qwen/Qwen3-1.7B --out /data/omni/q17_ev.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vt.omni import encode, token_units  # noqa: E402

EV = ("shift_mean", "shift_p10", "shift_min", "frac_down", "frac_up", "disagree_max")


def ev_feats(shift, dis):
    if len(shift) == 0:
        return np.zeros(len(EV), np.float32)
    return np.array([shift.mean(), np.percentile(shift, 10), shift.min(), (shift < -1).mean(), (shift > 1).mean(),
                     dis.max()], np.float32)


def strip_doc(r):
    """the conversation without its documents (None if they cannot be located in it)."""
    if not r["doc"]:
        return None
    msgs = [dict(m) for m in r["messages"]]
    u = msgs[-1]["content"]
    if r["doc"] in u:
        msgs[-1]["content"] = u.replace(r["doc"], "[documents withheld]")
    else:                                   # the document is formatted into the prompt: keep the instruction only
        msgs[-1]["content"] = u.strip().split("\n")[0][:300] + "\n[documents withheld]"
    msgs[0]["content"] = "You are a helpful assistant."
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16).eval()
    rows = [json.loads(l) for l in open(a.corpus)]
    ids, R, U_row, U_idx, U = [], [], [], [], []
    t0 = time.time()

    def logps(item):
        x, r0, roff = encode(tok, item, a.max_tokens)
        with torch.no_grad():
            lo = model(input_ids=torch.tensor([x])).logits[0, r0 - 1:-1].float().log_softmax(-1)
        return lo, x[r0:], roff

    for r in rows:
        nod = strip_doc(r)
        if nod is None:
            continue
        lw, rid, roff = logps(r)
        lo, rid2, _ = logps({"messages": nod, "response": r["response"]})
        n = min(len(rid), len(rid2))
        lw, lo = lw[:n], lo[:n]
        tgt = torch.tensor(rid[:n])
        a_w = lw.gather(-1, tgt[:, None])[:, 0]
        shift = (a_w - lo.gather(-1, tgt[:, None])[:, 0]).numpy()
        top = lw.argmax(-1)
        boost = lw.gather(-1, top[:, None])[:, 0] - lo.gather(-1, top[:, None])[:, 0]
        gap = lw.gather(-1, top[:, None])[:, 0] - a_w
        dis = torch.where((top != tgt) & (boost >= 0.5), gap, torch.zeros_like(gap)).numpy()
        ids.append(r["id"])
        R.append(ev_feats(shift, dis))
        tu = token_units(roff[:n], r["units"])
        for k in range(len(r["units"])):
            m = tu == k
            if m.any():
                U_row.append(r["id"])
                U_idx.append(k)
                U.append(ev_feats(shift[m], dis[m]))
        if len(ids) % 100 == 0:
            print(f"{len(ids)} replies, {(time.time() - t0) / len(ids):.2f} s each", flush=True)
            np.savez(a.out, ids=np.array(ids), ev=np.stack(R), u_row=np.array(U_row), u_idx=np.array(U_idx),
                     u_ev=np.stack(U))
    np.savez(a.out, ids=np.array(ids), ev=np.stack(R), u_row=np.array(U_row), u_idx=np.array(U_idx), u_ev=np.stack(U))
    print("done", len(ids), flush=True)


if __name__ == "__main__":
    main()
