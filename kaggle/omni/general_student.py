"""General-student data, one Kaggle shard: replies from many models on diverse prompts, labelled by a one-pass judge
(teacher) and read by the student's reader, in parallel on two T4s. No human labels are used for training.

    data     UltraFeedback (openbmb/UltraFeedback): instruction + 4 completions from different models, many sources
             (ShareGPT, Evol-Instruct, UltraChat, FLAN, FalseQA, ...). Excluded so the 21-domain test corpus stays
             unseen: the truthful_qa source. Sampled stratified by source, seeded, sharded.
    GPU 0    judge: Qwen/Qwen3-14B 4-bit (vt.omni.JudgeScorer, one forward pass, log-odds of "No") -> teacher score
    GPU 1    student reader: Qwen/Qwen3-4B (vt.omni.Reader): reply-mean and judge-state at three middle layers,
             output scalars, own-judge log-odds (float16). Shard 0 also reads the 5,107-reply test corpus
             (experiments/omni_corpus.py build) with the same reader, for evaluation.
    output   /kaggle/working/gs_judge_<shard>.npz, gs_feats_<shard>.npz, gs_corpus_feats.npz (shard 0)

Then, locally: experiments/general_student_fit.py trains the student on the judge scores only and tests it on all
21 corpus domains (never seen in training).

    python kaggle/kgrun.py --slug omni-gs-0 --script kaggle/omni/general_student.py \
        --bundle vt/omni.py vt/mathcheck.py vt/checkers/__init__.py vt/checkers/business_record.py \
                 experiments/omni_corpus.py kaggle/omni/general_student.py \
        --args "--shard 0 --n-shards 6 --per-shard 6000" --out results/kaggle/omni-gs-0 --wait
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SELF = os.path.join("/kaggle/working/src", "kaggle", "omni", "general_student.py")
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def sample_rows(shard, n_shards, per_shard, seed=0):
    """this shard's replies: [{"id", "source", "model", "question", "response"}]."""
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback", split="train")  # nosec B615: public benchmark data
    by_src = {}
    for i, x in enumerate(d):
        if x["source"] == "truthful_qa":
            continue                                        # the test corpus has a TruthfulQA domain
        for k, c in enumerate(x["completions"]):
            r = (c.get("response") or "").strip()
            if 20 <= len(r) <= 6000 and x["instruction"].strip():
                by_src.setdefault(x["source"], []).append(
                    {"id": f"uf{i}_{k}", "source": x["source"], "model": c.get("model", ""),
                     "question": x["instruction"], "response": r})
    rng = random.Random(seed)
    total = per_shard * n_shards
    srcs = sorted(by_src)
    quota = {s: min(len(by_src[s]), total // len(srcs)) for s in srcs}
    spare = total - sum(quota.values())
    for s in sorted(srcs, key=lambda s: -len(by_src[s])):  # give leftover room to the big sources
        extra = min(spare, len(by_src[s]) - quota[s])
        quota[s] += extra
        spare -= extra
    pool = []
    for s in srcs:
        rng.shuffle(by_src[s])
        pool += by_src[s][:quota[s]]
    rng.shuffle(pool)
    log("sources", {s: quota[s] for s in srcs}, "pool", len(pool))
    return pool[shard * per_shard:(shard + 1) * per_shard]


def judge_worker(a):
    import numpy as np

    from vt.omni import JudgeScorer
    rows = json.load(open(os.path.join(WORK, f"gs_rows_{a.shard}.json")))
    out = os.path.join(WORK, f"gs_judge_{a.shard}.npz")
    j = JudgeScorer(a.judge, load_in_4bit=True, max_tokens=3072)
    ids, sc = [], []
    for k, r in enumerate(rows):
        try:
            s = j({"response": r["response"], "doc": "", "question": r["question"]})
        except Exception as e:                              # one bad reply must not kill the shard
            log("judge error", r["id"], repr(e)[:120])
            s = float("nan")
        ids.append(r["id"])
        sc.append(s)
        if (k + 1) % 250 == 0:
            log(f"judge {k + 1}/{len(rows)}")
            np.savez(out, ids=np.array(ids), logodds_wrong=np.array(sc, np.float32), model=a.judge)
    np.savez(out, ids=np.array(ids), logodds_wrong=np.array(sc, np.float32), model=a.judge)
    log("judge done")


def reader_worker(a):
    import numpy as np

    from vt.omni import Reader, make_item

    rd = Reader(a.reader, max_tokens=a.max_tokens)
    nl = len(rd.layers)
    mid = [nl // 2 - 1, nl // 2, nl // 2 + 1]

    def read(items, ids, out):
        acc = {"ids": [], "r_mean": [], "j_state": [], "r_sc": [], "j_logit": []}
        for k, (i, it) in enumerate(zip(ids, items)):
            try:
                f = rd.features(it)
            except Exception as e:
                log("reader error", i, repr(e)[:120])
                f = None
            if f is None:
                continue
            acc["ids"].append(i)
            acc["r_mean"].append(np.asarray(f["r_mean"])[mid].astype(np.float16))
            acc["j_state"].append(np.asarray(f["j_state"])[mid].astype(np.float16))
            acc["r_sc"].append(f["r_sc"])
            acc["j_logit"].append(f["j_logit"])
            if (k + 1) % 500 == 0:
                log(f"reader {out} {k + 1}/{len(items)}")
        np.savez(out, **{k_: np.array(v) for k_, v in acc.items()}, layers=np.array(rd.layers)[mid],
                 reader=a.reader)

    rows = json.load(open(os.path.join(WORK, f"gs_rows_{a.shard}.json")))
    read([make_item(r["response"], r["question"]) for r in rows], [r["id"] for r in rows],
         os.path.join(WORK, f"gs_feats_{a.shard}.npz"))
    if a.shard == 0 and not a.no_corpus:                    # the test corpus, read by the same student reader
        sys.path.insert(0, "/kaggle/working/src/experiments")
        cp = os.path.join(WORK, "corpus.jsonl")
        subprocess.run([sys.executable, "/kaggle/working/src/experiments/omni_corpus.py", "build", "--out", cp],
                       check=True)
        corpus = [json.loads(l) for l in open(cp)]
        items = [{"messages": r["messages"], "response": r["response"], "units": r["units"], "doc": r["doc"],
                  "question": r["messages"][-1]["content"]} for r in corpus]
        read(items, [r["id"] for r in corpus], os.path.join(WORK, "gs_corpus_feats.npz"))
    log("reader done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=6)
    ap.add_argument("--per-shard", type=int, default=6000)
    ap.add_argument("--judge", default="Qwen/Qwen3-14B")
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--worker", choices=["judge", "reader"])
    ap.add_argument("--no-corpus", action="store_true", help="smoke tests: skip reading the test corpus")
    a = ap.parse_args()
    if a.worker == "judge":
        return judge_worker(a)
    if a.worker == "reader":
        return reader_worker(a)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43"], check=False)
    rows = sample_rows(a.shard, a.n_shards, a.per_shard)
    json.dump(rows, open(os.path.join(WORK, f"gs_rows_{a.shard}.json"), "w"))
    log("shard", a.shard, "rows", len(rows))
    base = [sys.executable, SELF if os.path.exists(SELF) else __file__, "--shard", str(a.shard), "--n-shards",
            str(a.n_shards), "--per-shard", str(a.per_shard), "--judge", a.judge, "--reader", a.reader,
            "--max-tokens", str(a.max_tokens)] + (["--no-corpus"] if a.no_corpus else [])
    env = dict(os.environ, PYTHONPATH="/kaggle/working/src:" + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen(base + ["--worker", "judge"], env=dict(env, CUDA_VISIBLE_DEVICES="0"),
                              stdout=open(os.path.join(WORK, "judge.log"), "w"), stderr=subprocess.STDOUT),
             subprocess.Popen(base + ["--worker", "reader"], env=dict(env, CUDA_VISIBLE_DEVICES="1"),
                              stdout=open(os.path.join(WORK, "reader.log"), "w"), stderr=subprocess.STDOUT)]
    for p in procs:
        p.wait()
    for name in ("judge.log", "reader.log"):
        print(f"--- {name} (tail)")
        print("".join(open(os.path.join(WORK, name)).readlines()[-15:]))
    log("all done", [p.returncode for p in procs])


if __name__ == "__main__":
    main()
