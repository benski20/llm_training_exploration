"""The shipped AI-text detector (vt/aitext.py) on generators and genres it never saw: HAP-E (Reinhart et al., 2024).

The probe head was trained on RAID only (2023-era generators: GPT-2/3/4, ChatGPT, Llama-2-chat, Mistral, MPT,
Cohere). HAP-E (browndw/human-ai-parallel-corpus, MIT) gives a human text's first 500 words to newer models and
keeps their continuation: GPT-4o (2024-08-06), GPT-4o-mini, Llama-3-70B and -8B, base and instruct. The human
reference is the text's real continuation. Six genres: academic, news, fiction, spoken, blogs, TV/movie scripts;
fiction, spoken and TV/movie have no RAID counterpart.

Per genre: 250 human continuations and 300 AI ones (50 per generator), from distinct documents, so no human test
text shares a prompt with an AI one. The product's promise is tested as a customer would use it: 150 of the
genre's human texts calibrate (5% false alarms, 95% confidence), the other 100 and the AI texts test, 50 times.

    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_modern.py sample --hape <dir of the parquet files>
    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_modern.py read
    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_modern.py evaluate
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
GENRES = ("acad", "blog", "fic", "news", "spok", "tvm")


def sample(a):
    import pandas as pd
    rng = np.random.default_rng(0)
    load = lambda f: pd.read_parquet(os.path.join(a.hape, f"hape-text_{f}.parquet"))       # noqa: E731
    human = load("human-chunk-2")
    gens = {g: load(g) for g in GENERATORS}
    human["doc"] = human.doc_id.str.split("@").str[0]
    rows = []
    for genre in GENRES:
        docs = human.doc[human.doc.str.startswith(genre + "_")].to_numpy()
        docs = rng.permutation(docs)
        need = a.human + a.per_generator * len(GENERATORS)
        assert len(docs) >= need, (genre, len(docs))
        hd, ad = docs[:a.human], docs[a.human:need]
        h = human.set_index("doc").loc[hd]
        rows += [{"text": t, "genre": genre, "model": "human", "doc": d, "y": 0} for d, t in zip(hd, h.text)]
        for k, g in enumerate(GENERATORS):
            gd = gens[g].assign(doc=gens[g].doc_id.str.split("@").str[0]).set_index("doc")
            for d in ad[k * a.per_generator:(k + 1) * a.per_generator]:
                rows.append({"text": gd.loc[d, "text"], "genre": genre, "model": g, "doc": d, "y": 1})
    os.makedirs(SCRATCH, exist_ok=True)
    with open(f"{SCRATCH}/hape_sample.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "texts")


def read(a):
    from vt.aitext import Reader
    with open(f"{SCRATCH}/hape_sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    path = f"{SCRATCH}/hape_features.npz"
    have = dict(np.load(path)) if os.path.exists(path) else None
    S, H = ([have["scalars"]], [have["hidden"]]) if have else ([], [])
    start = len(have["scalars"]) if have else 0
    r = Reader(threads=a.threads)
    t0, step = time.time(), a.batch * 25
    for b in range(start, len(rows), step):
        s, h = r.features([x["text"] for x in rows[b:b + step]], a.batch)
        S.append(s)
        H.append(h)
        np.savez(path, scalars=np.concatenate(S), hidden=np.concatenate(H))
        done = min(b + step, len(rows))
        print(f"{done}/{len(rows)} {(time.time() - t0) / (done - start):.2f}s a text", flush=True)


def evaluate(a):
    from vt.aitext import Detector, calibration_from_scores
    from vt.correctness import auroc
    with open(f"{SCRATCH}/hape_sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    z = np.load(f"{SCRATCH}/hape_features.npz")
    n = len(z["scalars"])
    rows = rows[:n]

    class NoReader:
        names = ()
    det = Detector(reader=NoReader())
    s = det.score_features(z["scalars"], z["hidden"])
    genre = np.array([r["genre"] for r in rows])
    model = np.array([r["model"] for r in rows])
    y = np.array([r["y"] for r in rows])
    rng = np.random.default_rng(0)
    res = {"texts": n, "head_sha256": det.head_sha, "by_genre": {}, "by_generator": {}}
    caught_by_gen = {g: [] for g in GENERATORS}
    for g in GENRES:
        m = genre == g
        hum = np.where(m & (y == 0))[0]
        row = {f"auroc_{k}": round(float(auroc(s[k][m], y[m].astype(float))), 3) for k in ("probe", "binoculars")}
        fa, ca, over = [], [], []
        per_gen = {x: [] for x in GENERATORS}
        for _ in range(a.splits):
            cal = rng.choice(hum, size=a.cal, replace=False)
            test_h = np.setdiff1d(hum, cal)
            c = calibration_from_scores({k: s[k][cal] for k in ("probe", "binoculars")}, a.fpr, 0.95)
            flag = np.zeros(n, bool)
            for k, t in c["thresholds"].items():
                flag |= s[k] > t
            fa.append(float(flag[test_h].mean()))
            over.append(fa[-1] > a.fpr)
            ca.append(float(flag[m & (y == 1)].mean()))
            for x in GENERATORS:
                per_gen[x].append(float(flag[m & (model == x)].mean()))
        row.update(false_alarms_mean=round(float(np.mean(fa)), 3), splits_over_5pct=round(float(np.mean(over)), 3),
                   caught=round(float(np.mean(ca)), 3),
                   caught_by_generator={x: round(float(np.mean(v)), 3) for x, v in per_gen.items()})
        for x in GENERATORS:
            caught_by_gen[x].append(np.mean(per_gen[x]))
        res["by_genre"][g] = row
        print(g, json.dumps(row), flush=True)
    for x in GENERATORS:
        mm = (model == x) | (y == 0)
        res["by_generator"][x] = {"auroc_probe": round(float(auroc(s["probe"][mm], y[mm].astype(float))), 3),
                                  "auroc_binoculars": round(float(auroc(s["binoculars"][mm], y[mm].astype(float))), 3),
                                  "caught_mean_over_genres": round(float(np.mean(caught_by_gen[x])), 3)}
    res["mean"] = {k: round(float(np.mean([res["by_genre"][g][k] for g in GENRES])), 3)
                   for k in ("auroc_probe", "auroc_binoculars", "false_alarms_mean", "splits_over_5pct", "caught")}
    with open(a.report, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["mean"]), json.dumps(res["by_generator"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "read", "evaluate"])
    ap.add_argument("--hape", default="")
    ap.add_argument("--human", type=int, default=250)
    ap.add_argument("--per-generator", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--cal", type=int, default=150)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--report", default="results/ai_text/hape_modern.json")
    a = ap.parse_args()
    {"sample": sample, "read": read, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
