"""A multi-domain, multi-format verification corpus, and one-pass reader features for it.

The verifier has to judge any reply: a RAG answer, a closed-book chat answer, a long-form text, a
step-by-step solution. This builds one corpus that spans those formats, in ~20 *domains* (the unit of
transfer: a new customer is a new domain), and reads every reply once with a frozen reader, keeping

    per reply   mean residual state at a few layers over the reply's tokens, the state at its last token,
                and scalar output signals (token log-prob, entropy, rank-1 margin: mean / min / quantiles)
    per unit    (sentence, labelled segment, reasoning step) the same, pooled over the unit's tokens

so that every head, transfer and adaptation experiment afterwards runs on stored features in seconds.

Domains and labels (1 = hallucinated / wrong):
    ragtruth_{qa,summary,data2text}   RAG answers of 6 LLMs, human span labels -> sentence + reply labels
    halueval_{qa,dialogue,summary}    knowledge-grounded; right vs hallucinated reply (reply label)
    halueval_general                  ChatGPT replies to real user queries, closed book (reply label)
    truthfulqa                        closed-book misconceptions; correct vs incorrect answer
    halubench_{drop,pubmed,finance,covid}  passage QA (reply label)
    felm_{math,reasoning,wk,science,writing}  ChatGPT long-form replies, closed book, segment labels
    pb_{gsm8k,math,olympiad,omnimath}  ProcessBench: step-by-step solutions of 12 models, first wrong step

    python experiments/omni_corpus.py build --out /data/omni/corpus.jsonl
    python experiments/omni_corpus.py extract --corpus /data/omni/corpus.jsonl --reader Qwen/Qwen3-1.7B --out /data/omni/q17
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RAG_SYSTEM = "You are a helpful assistant. Answer using the provided information."
PLAIN_SYSTEM = "You are a helpful assistant."


def sentence_spans(text):
    """[(start, end)] of the sentences / lines of a text (no empty ones)."""
    out, start = [], 0
    for m in re.finditer(r"(?<!\d)[.!?](?=\s)|\n+", text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


def row(domain, fmt, user, response, doc="", units=None, unit_labels=None, label=0, system=None):
    """units: [(start, end)] char spans in the response (sentences if None); unit_labels: 0/1/-1 (unknown)."""
    if units is None:
        units = sentence_spans(response)
        unit_labels = [-1] * len(units)
    return {"domain": domain, "format": fmt,
            "messages": [{"role": "system", "content": system or (RAG_SYSTEM if doc else PLAIN_SYSTEM)},
                         {"role": "user", "content": user}],
            "response": response, "doc": doc, "units": units, "unit_labels": unit_labels, "label": int(label)}


def balanced(rows, n, seed=0):
    """Up to n rows, half of each label where possible."""
    rng = random.Random(seed)  # nosec B311: data sampling
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(len(pos), n // 2)
    out = pos[:k] + neg[:n - k]
    rng.shuffle(out)
    return out


def ragtruth(n_per_task):
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        for x in load_dataset("wandb/RAGTruth-processed", split=split):  # nosec B615: benchmark data
            resp = x["output"]
            hs = [(int(h["start"]), int(h["end"])) for h in json.loads(x["hallucination_labels"] or "[]")]
            units = sentence_spans(resp)
            ul = [int(any(s < b and e > a for s, e in hs)) for a, b in units]
            task = {"QA": "qa", "Summary": "summary", "Data2txt": "data2text"}[x["task_type"]]
            out.append(row(f"ragtruth_{task}", "rag", x["input_str"], resp, x["context"], units, ul, int(bool(hs)),
                           RAG_SYSTEM))
    res = []
    for t in ("qa", "summary", "data2text"):
        res += balanced([r for r in out if r["domain"] == f"ragtruth_{t}"], n_per_task, seed=1)
    return res


def halueval(n_pairs):
    from datasets import load_dataset
    out = []
    for sub in ("qa", "dialogue", "summarization"):
        rows = list(load_dataset("pminervini/HaluEval", sub)["data"])  # nosec B615: benchmark data
        random.Random(2).shuffle(rows)  # nosec B311
        for x in rows[:n_pairs]:
            if sub == "qa":
                u = f"Knowledge: {x['knowledge']}\n\nQuestion: {x['question']}"
                pair = (x["right_answer"], x["hallucinated_answer"])
                d, dom, fmt = x["knowledge"], "halueval_qa", "rag"
            elif sub == "dialogue":
                u = f"Knowledge: {x['knowledge']}\n\nDialogue so far:\n{x['dialogue_history']}\n\nReply as the assistant."
                pair = (x["right_response"], x["hallucinated_response"])
                d, dom, fmt = x["knowledge"], "halueval_dialogue", "rag"
            else:
                u = f"Summarize the following document.\n\n{x['document']}"
                pair = (x["right_summary"], x["hallucinated_summary"])
                d, dom, fmt = x["document"], "halueval_summary", "rag"
            out.append(row(dom, fmt, u, pair[0], d, label=0))
            out.append(row(dom, fmt, u, pair[1], d, label=1))
    rows = list(load_dataset("pminervini/HaluEval", "general")["data"])  # nosec B615
    gen = [row("halueval_general", "chat", x["user_query"], x["chatgpt_response"] or "", label=int(x["hallucination"] == "yes"))
           for x in rows if (x["chatgpt_response"] or "").strip()]
    return out + balanced(gen, 2 * n_pairs, seed=3)


def truthfulqa(n_q):
    from datasets import load_dataset
    rows = list(load_dataset("truthfulqa/truthful_qa", "generation")["validation"])  # nosec B615
    random.Random(4).shuffle(rows)  # nosec B311
    out = []
    for i, x in enumerate(rows[:n_q]):
        rng = random.Random(i)  # nosec B311
        out.append(row("truthfulqa", "chat", x["question"], x["best_answer"], label=0))
        if x["incorrect_answers"]:
            out.append(row("truthfulqa", "chat", x["question"], rng.choice(x["incorrect_answers"]), label=1))
    return out


def halubench(n):
    from datasets import load_dataset
    d = load_dataset("PatronusAI/HaluBench", split="test")  # nosec B615
    names = {"DROP": "drop", "pubmedQA": "pubmed", "FinanceBench": "finance", "covidQA": "covid"}
    out = []
    for raw, short in names.items():
        rows = [row(f"halubench_{short}", "rag", f"Context: {x['passage']}\n\nQuestion: {x['question']}",
                    str(x["answer"]), x["passage"], label=int(x["label"] == "FAIL"))
                for x in d if x["source_ds"] == raw]
        out += balanced(rows, n, seed=5)
    return out


def felm():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("hkust-nlp/felm", "all.jsonl", repo_type="dataset")  # nosec B615
    out = []
    for line in open(p):
        x = json.loads(line)
        try:
            segs, labs = x["segmented_response"], x["labels"]
            segs = ast.literal_eval(segs) if isinstance(segs, str) else segs
            labs = ast.literal_eval(labs) if isinstance(labs, str) else labs
        except (ValueError, SyntaxError):
            continue
        if not segs or len(segs) != len(labs):
            continue
        resp, units = "", []
        for s in segs:
            s = s.strip()
            if resp:
                resp += "\n" if "\n" in s[:1] or s[:1].isdigit() or s[:1] in "-*" else " "
            units.append((len(resp), len(resp) + len(s)))
            resp += s
        ul = [0 if l else 1 for l in labs]                           # FELM: True = factually correct
        dom = {"writing_rec": "writing", "wk": "wk"}.get(x["domain"], x["domain"])
        out.append(row(f"felm_{dom}", "longform", x["prompt"], resp, units=units, unit_labels=ul,
                       label=int(any(ul))))
    return out


def processbench(n):
    from datasets import load_dataset
    out = []
    for cfg in ("gsm8k", "math", "olympiadbench", "omnimath"):
        rows = []
        for x in load_dataset("Qwen/ProcessBench", split=cfg):  # nosec B615
            resp, units = "", []
            for s in x["steps"]:
                if resp:
                    resp += "\n\n"
                units.append((len(resp), len(resp) + len(s)))
                resp += s
            first = int(x["label"])
            ul = [(-1 if first < 0 else 0) if first < 0 or i < first else (1 if i == first else -1)
                  for i in range(len(units))]
            if first < 0:
                ul = [0] * len(units)
            rows.append(row(f"pb_{cfg.replace('olympiadbench', 'olympiad')}", "reasoning",
                            f"Solve the following problem step by step.\n\n{x['problem']}", resp, units=units,
                            unit_labels=ul, label=int(first >= 0)))
        out += balanced(rows, n, seed=6)
    return out


def cmd_build(a):
    rows = ragtruth(a.ragtruth) + halueval(a.halueval_pairs) + truthfulqa(a.truthfulqa) + halubench(a.halubench) + \
        felm() + processbench(a.processbench)
    for i, r in enumerate(rows):
        r["id"] = i
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter((r["domain"], r["label"]) for r in rows)
    for d in sorted({r["domain"] for r in rows}):
        print(f"{d:22s} n={c[(d, 0)] + c[(d, 1)]:4d} wrong={c[(d, 1)]}")
    print("total", len(rows))


# ---------------------------------------------------------------- extraction (vt.omni.Reader: the same code serves)

def cmd_extract(a):
    from vt.omni import SCALARS, Reader
    rows = [json.loads(l) for l in open(a.corpus)]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    rd = Reader(a.reader, layer_fracs=a.layer_fracs, max_tokens=a.max_tokens, threads=a.threads)
    meta = {"reader": a.reader, "layers": rd.layers, "n_layers": rd.model.config.num_hidden_layers,
            "scalars": SCALARS, "max_tokens": a.max_tokens}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    t0, done = time.time(), 0
    f16 = lambda x: np.asarray(x, np.float16)
    for s0 in range(0, len(rows), a.shard):
        path = os.path.join(a.out, f"shard_{s0 // a.shard:04d}.npz")
        if os.path.exists(path):
            continue
        acc = {k: [] for k in ("ids", "r_mean", "r_last", "r_sc", "j_state", "j_logit", "u_mean", "u_sc", "u_row",
                               "u_idx", "u_lab")}
        for r in rows[s0:s0 + a.shard]:
            f = rd.features(r)
            if f is None:
                continue
            acc["ids"].append(r["id"])
            for k in ("r_mean", "r_last", "j_state"):
                acc[k].append(f16(f[k]))
            acc["r_sc"].append(f["r_sc"])
            acc["j_logit"].append(f["j_logit"])
            for u in f["units"]:
                acc["u_mean"].append(f16(u["mean"]))
                acc["u_sc"].append(u["sc"])
                acc["u_row"].append(r["id"])
                acc["u_idx"].append(u["index"])
                acc["u_lab"].append(r["unit_labels"][u["index"]])
            done += 1
        np.savez(path, **{k: np.array(v) for k, v in acc.items()})
        el = time.time() - t0
        print(f"shard {s0 // a.shard} done; {done} replies in {el / 60:.1f} min ({el / max(done, 1):.2f} s/reply)",
              flush=True)


def load_features(out):
    """Concatenate the shards: dict of arrays, reply-level and unit-level."""
    import glob
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(out, "shard_*.npz")))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    cat["meta"] = json.load(open(os.path.join(out, "meta.json")))
    return cat


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--ragtruth", type=int, default=300, help="replies per RAGTruth task")
    b.add_argument("--halueval-pairs", type=int, default=150)
    b.add_argument("--truthfulqa", type=int, default=200, help="questions (2 replies each)")
    b.add_argument("--halubench", type=int, default=200)
    b.add_argument("--processbench", type=int, default=240)
    e = sub.add_parser("extract")
    e.add_argument("--corpus", required=True)
    e.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    e.add_argument("--out", required=True)
    e.add_argument("--max-tokens", type=int, default=2048)
    e.add_argument("--layer-fracs", type=float, nargs="+", default=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    e.add_argument("--shard", type=int, default=100)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"build": cmd_build, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    main()
