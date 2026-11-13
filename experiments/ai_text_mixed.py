"""Whole-document reading: does the detector catch an AI-written body behind a human-written opening?

The shipped detector (vt/aitext.py) read only a text's first 256 tokens, so an essay whose first paragraphs are
the student's own and whose rest is AI passes. HAP-E (browndw/human-ai-parallel-corpus, MIT) builds exactly that
document: a human text's first ~500 words (chunk 1) followed by a model's continuation. Its human counterpart is
chunk 1 followed by the real continuation (chunk 2). Both start with the same kind of human opening; only what
follows differs.

Per genre (academic, news, fiction): 200 human documents and 120 mixed ones (20 per generator: GPT-4o,
GPT-4o-mini, Llama-3-70B/8B base and instruct), from distinct source texts. Each document is read in up to four
consecutive 256-token windows (about 1,000 tokens: the human opening is about 650), every window's features kept,
so reading 1, 2, 3 or 4 windows is compared on the same pass. A document's score is its most AI-like window.
The promise (5% false alarms, 95% confidence, union of probe and Binoculars) is calibrated per genre on 150 human
documents and tested on the other 50 and the mixed ones, 50 times.

    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_mixed.py sample --hape <dir of the parquet files>
    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_mixed.py read
    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_mixed.py evaluate
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SCRATCH = os.environ.get("AI_TEXT_SCRATCH", "results/ai_text")
GENERATORS = ("gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "llama-3-70B-Instruct", "llama-3-8B-Instruct",
              "llama-3-70B", "llama-3-8B")
GENRES = ("acad", "news", "fic")
WINDOWS = 4


def sample(a):
    import pandas as pd

    def load(name):
        d = pd.read_parquet(os.path.join(a.hape, f"hape-text_{name}.parquet"))
        return d.assign(doc=d.doc_id.str.split("@").str[0]).set_index("doc").text

    rng = np.random.default_rng(0)
    c1, c2 = load("human-chunk-1"), load("human-chunk-2")
    gens = {g: load(g) for g in GENERATORS}
    rows = []
    for genre in GENRES:
        docs = rng.permutation([d for d in c2.index if d.startswith(genre + "_") and d in c1.index])
        hd, md = docs[:a.human], docs[a.human:a.human + a.per_generator * len(GENERATORS)]
        rows += [{"text": c1[d] + "\n\n" + c2[d], "genre": genre, "model": "human", "doc": d, "y": 0} for d in hd]
        for k, g in enumerate(GENERATORS):
            for d in md[k * a.per_generator:(k + 1) * a.per_generator]:
                rows.append({"text": c1[d] + "\n\n" + gens[g][d], "genre": genre, "model": g, "doc": d, "y": 1,
                             "human_chars": len(c1[d])})
    with open(f"{SCRATCH}/mixed_sample.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "documents")


def read(a):
    from vt.aitext import Reader
    with open(f"{SCRATCH}/mixed_sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    r = Reader(threads=a.threads)
    path = f"{SCRATCH}/mixed_features.npz"
    have = dict(np.load(path)) if os.path.exists(path) else None
    S, H, O, W = ([have[k]] for k in ("scalars", "hidden", "owner", "window")) if have else ([], [], [], [])
    start = int(have["owner"].max()) + 1 if have else 0
    t0 = time.time()
    for b in range(start, len(rows), a.docs_per_save):
        ws, own, idx = [], [], []
        for i in range(b, min(b + a.docs_per_save, len(rows))):
            w = r.windows(rows[i]["text"], most=WINDOWS)
            ws += w
            own += [i] * len(w)
            idx += list(range(len(w)))
        s, h = r.features_ids(ws, a.batch)
        S.append(s)
        H.append(h)
        O.append(np.array(own))
        W.append(np.array(idx))
        np.savez(path, scalars=np.concatenate(S), hidden=np.concatenate(H), owner=np.concatenate(O),
                 window=np.concatenate(W))
        done = min(b + a.docs_per_save, len(rows))
        print(f"{done}/{len(rows)} documents {(time.time() - t0) / (done - start):.2f}s a document", flush=True)


def evaluate(a):
    from vt.aitext import Detector, calibration_from_scores
    from vt.correctness import auroc
    with open(f"{SCRATCH}/mixed_sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    z = np.load(f"{SCRATCH}/mixed_features.npz")

    class NoReader:
        names = ()
    det = Detector(reader=NoReader())
    s = det.score_features(z["scalars"], z["hidden"])
    owner, win = z["owner"], z["window"]
    n = int(owner.max()) + 1
    rows = rows[:n]
    genre = np.array([r["genre"] for r in rows])
    model = np.array([r["model"] for r in rows])
    y = np.array([r["y"] for r in rows])
    res = {"documents": n, "windows_read": {}}
    for k in range(1, WINDOWS + 1):
        doc = {}
        for name in ("probe", "binoculars"):
            v = np.full(n, -np.inf)
            np.maximum.at(v, owner[win < k], s[name][win < k])
            doc[name] = v
        rng = np.random.default_rng(0)
        per = {}
        for g in GENRES:
            m = genre == g
            hum = np.where(m & (y == 0))[0]
            fa, ca, gen = [], [], {x: [] for x in GENERATORS}
            for _ in range(a.splits):
                cal = rng.choice(hum, size=a.cal, replace=False)
                test_h = np.setdiff1d(hum, cal)
                c = calibration_from_scores({x: doc[x][cal] for x in ("probe", "binoculars")}, a.fpr, 0.95)
                flag = np.zeros(n, bool)
                for x, t in c["thresholds"].items():
                    flag |= doc[x] > t
                fa.append(float(flag[test_h].mean()))
                ca.append(float(flag[m & (y == 1)].mean()))
                for x in GENERATORS:
                    gen[x].append(float(flag[m & (model == x)].mean()))
            per[g] = {"auroc_probe": round(float(auroc(doc["probe"][m], y[m].astype(float))), 3),
                      "false_alarms_mean": round(float(np.mean(fa)), 3),
                      "splits_over_5pct": round(float(np.mean(np.array(fa) > a.fpr)), 3),
                      "caught": round(float(np.mean(ca)), 3),
                      "caught_by_generator": {x: round(float(np.mean(v)), 3) for x, v in gen.items()}}
        per["mean"] = {key: round(float(np.mean([per[g][key] for g in GENRES])), 3)
                       for key in ("auroc_probe", "false_alarms_mean", "splits_over_5pct", "caught")}
        res["windows_read"][k] = per
        print(k, "windows", json.dumps(per["mean"]), flush=True)
    res["mean_windows_per_document"] = round(float(np.bincount(owner).mean()), 2)
    with open(a.report, "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "read", "evaluate"])
    ap.add_argument("--hape", default="")
    ap.add_argument("--human", type=int, default=200)
    ap.add_argument("--per-generator", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--docs-per-save", type=int, default=20)
    ap.add_argument("--cal", type=int, default=150)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--report", default="results/ai_text/hape_mixed.json")
    a = ap.parse_args()
    {"sample": sample, "read": read, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
