"""NLI grounding experts for the corpus's document-grounded replies: MiniCheck (RoBERTa-Large, Apache-2.0) and
Vectara's HHEM-2.1-open, each scoring P(unsupported) for the whole reply and for each of its sentences (worst
sentence kept). Small cross-encoders (0.1-0.4B), cheap next to the reader, and strong exactly where the reader is
weakest (RAG). Stored per reply id for experiments/omni_final.py (--nli).

    python experiments/omni_nli.py --corpus corpus.jsonl --out /data/omni/nli.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NLI = ("minicheck_reply", "minicheck_worst", "hhem_reply", "hhem_worst")


def question_of(r):
    u = r["messages"][-1]["content"]
    if r["doc"] and r["doc"] in u:
        u = u.replace(r["doc"], " ")
    return " ".join(u.split())[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sentences", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N document-grounded replies")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from halubench_compare import MiniCheck
    from transformers import AutoModelForSequenceClassification
    mc = MiniCheck()
    hh = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model",
                                                            trust_remote_code=True).eval()   # its own model code
    rows = [json.loads(l) for l in open(a.corpus)]
    rows = [r for r in rows if r["doc"]]
    if a.limit:
        rows = rows[:a.limit]
    done = {}
    if os.path.exists(a.out):
        z = np.load(a.out)
        done = {int(i): v for i, v in zip(z["ids"], z["nli"])}
    t0, n0 = time.time(), len(done)
    for r in rows:
        if r["id"] in done:
            continue
        q, doc, resp = question_of(r), r["doc"][:12000], r["response"]
        sents = [resp[s:e].strip() for s, e in r["units"]][:a.max_sentences] or [resp]
        claim = f"{q} {resp}" if len(resp.split()) < 8 else resp            # a fragment answer reads with its question
        with torch.no_grad():
            m_reply = mc.p_unsupported(doc, claim)
            m_worst = max(mc.p_unsupported(doc, s) for s in sents) if len(sents) > 1 else m_reply
            pairs = [(doc, claim)] + ([(doc, s) for s in sents] if len(sents) > 1 else [])
            p = hh.predict(pairs).float().numpy()                            # P(consistent)
        h_reply, h_worst = 1 - p[0], float((1 - p[1:]).max()) if len(p) > 1 else 1 - p[0]
        done[r["id"]] = np.array([m_reply, m_worst, h_reply, h_worst], np.float32)
        k = len(done) - n0
        if k % 50 == 0:
            print(f"{len(done)}/{len(rows)} ({(time.time() - t0) / k:.2f} s each)", flush=True)
            np.savez(a.out, ids=np.array(list(done)), nli=np.stack(list(done.values())))
    np.savez(a.out, ids=np.array(list(done)), nli=np.stack(list(done.values())))
    print("done", len(done), flush=True)


if __name__ == "__main__":
    main()
