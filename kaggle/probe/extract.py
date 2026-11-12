"""Feature extraction for the large-scale grounding probe: one forward pass of a frozen reader per answer, sharded,
resumable, on a Kaggle CPU session (small readers) or Kaggle T4 GPUs (8B / 14B).

Each answer is laid out exactly as the server lays out a /judge request with documents (vt.omni.make_item: system
turn, documents then question, the answer, then the judge question), so a probe trained here can be served as is.
Per answer we keep, in float16:
    a_mean[L, P]   mean residual over the answer tokens at each residual layer L (projected to P dims)
    a_last[L, P]   the answer's last token
    a_judge[L, P]  the state after the judge question (the position that would say Yes / No)
    a_neu[K, P]    mean MLP-neuron activation (down_proj input) over the answer at K middle layers
    a_sc[10]       token-level output signals (vt.omni.scalar_feats); j_logit: log P(Yes) - log P(No)
Per sentence ("unit", vt.omni.sentence_spans) of the answer:
    u_res[L2, P]   mean residual at the middle residual layers; u_neu[K, P] mean MLP neurons; u_sc[10]
    u_lab          1 if the sentence overlaps a labelled hallucinated span, 0 if the answer has spans and it doesn't,
                   -1 when the source has no span labels
Projections are fixed Gaussian matrices from a seed (stored in meta), so serving reproduces them exactly.

    python kaggle/probe/extract.py --data data/probe_train/ragtruth.jsonl --reader Qwen/Qwen3-1.7B \
        --shard 0 --n-shards 8 --out /kaggle/working/feats
    (on Kaggle: --build clones the repository at --branch and runs experiments/probe_data.py to make the data)
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def load_records(paths, shard, n_shards, cap=0, caps=None):
    """this shard's records; with cap (or caps[file stem]), at most that many per file (a fixed pseudo-random subset)."""
    h = lambda r: int(hashlib.sha1(r["id"].encode()).hexdigest(), 16)
    recs = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if caps is not None and stem not in caps:
            continue
        c = (caps or {}).get(stem, cap)
        rs = [json.loads(line) for line in open(p)]
        if c and len(rs) > c:
            rs = sorted(rs, key=lambda r: hashlib.sha1(("cap" + r["id"]).encode()).hexdigest())[:c]
        recs += [r for r in rs if h(r) % n_shards == shard]      # deterministic sharding, independent of file order
    recs.sort(key=h)               # hash order mixes the sources, so a time cutoff keeps a balanced subset
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--build", action="store_true", help="clone the repo and build the data with probe_data.py")
    ap.add_argument("--branch", default="claude/keen-euler-0kccyk")
    ap.add_argument("--sources", default="", help="for --build: comma-separated sources (default all)")
    ap.add_argument("--eval-sets", default="", help="for --build: build these evaluation sets (probe_evalsets.py) "
                    "instead of the training data")
    ap.add_argument("--reader", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--res-fracs", default="0.3,0.4,0.5,0.6,0.7,0.85")
    ap.add_argument("--neu-fracs", default="0.45,0.55,0.65")
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=11.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0, help="at most this many answers per source file")
    ap.add_argument("--caps", default="", help="per source file: 'ragtruth=15000,fava=8000' (others skipped)")
    ap.add_argument("--gpu-workers", type=int, default=0, help="one model per GPU: split this shard over N "
                    "subprocesses (e.g. 2 on a Kaggle 2x T4 session with a 4-bit 14B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if a.build:
        repo = "/kaggle/working/repo" if os.path.isdir("/kaggle/working") else os.path.abspath("repo_clone")
        if not os.path.isdir(repo):
            subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", a.branch,
                            "https://github.com/benski20/llm_training_exploration", repo], check=True)
        if a.eval_sets:
            dd = os.path.join(repo, "data", "probe_eval_built")
            cmd = [sys.executable, os.path.join(repo, "experiments", "probe_evalsets.py"), "--out", dd,
                   "--sets", a.eval_sets]
        else:
            dd = os.path.join(repo, "data", "probe_train_built")
            cmd = [sys.executable, os.path.join(repo, "experiments", "probe_data.py"), "--out", dd]
            if a.sources:
                cmd += ["--sources", a.sources]
        log("building data:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo)
        a.data = sorted(glob.glob(os.path.join(dd, "*.jsonl")))
        root = repo
    if a.load_4bit:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46.1"], check=True)
    if a.gpu_workers > 1:                                    # one model per GPU, each a sub-shard of this shard
        from huggingface_hub import snapshot_download
        snapshot_download(a.reader, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        log("weights cached")
        hours = a.hours - (time.time() - T0) / 3600              # the workers get what is left of the budget
        script = os.path.join(root, "kaggle", "probe", "extract.py")    # the repo copy (no kernel prelude)
        os.makedirs(a.out, exist_ok=True)
        procs = []
        for g in range(a.gpu_workers):
            cmd = [sys.executable, script, "--data", *a.data, "--reader", a.reader, "--shard",
                   str(a.shard * a.gpu_workers + g), "--n-shards", str(a.n_shards * a.gpu_workers),
                   "--max-tokens", str(a.max_tokens), "--res-fracs", a.res_fracs, "--neu-fracs", a.neu_fracs,
                   "--proj", str(a.proj), "--seed", str(a.seed), "--hours", f"{hours:.3f}", "--cap", str(a.cap),
                   "--caps", a.caps,
                   "--out", a.out] + (["--load-4bit"] if a.load_4bit else []) + \
                  (["--limit", str(a.limit)] if a.limit else [])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(os.path.join(a.out, f"worker{g}.log"), "w"),
                                          stderr=subprocess.STDOUT))
            log("worker", g, "started")
        rc = [p.wait() for p in procs]
        log("workers done", rc)
        for g in range(a.gpu_workers):
            print(f"--- worker{g}.log (tail)\n" + "".join(open(os.path.join(a.out, f"worker{g}.log")).readlines()[-6:]))
        if any(rc):
            raise SystemExit(f"worker failure {rc}")
        return
    sys.path.insert(0, root)
    from vt.probe_features import ProbeReader
    caps = {k: int(v) for k, v in (x.split("=") for x in a.caps.split(",") if x)} if a.caps else None
    recs = load_records(a.data, a.shard, a.n_shards, a.cap, caps)
    if a.limit:
        recs = recs[:a.limit]
    log("shard", a.shard, "of", a.n_shards, ":", len(recs), "answers from", len(a.data), "files")
    os.makedirs(a.out, exist_ok=True)
    pr = ProbeReader(a.reader, a.load_4bit, tuple(map(float, a.res_fracs.split(","))),
                     tuple(map(float, a.neu_fracs.split(","))), a.proj, a.seed, a.max_tokens, a.threads)
    json.dump(pr.meta, open(os.path.join(a.out, "meta.json"), "w"))
    done = set()
    for p in glob.glob(os.path.join(a.out, f"shard{a.shard:03d}_*.npz")):      # resume
        done.update(np.load(p, allow_pickle=True)["ids"].tolist())
    todo = [r for r in recs if r["id"] not in done]
    log("already done", len(done), "to do", len(todo), "res layers", pr.res_layers, "neuron layers", pr.neu_layers)
    keys = ("ids", "label", "source", "task", "a_mean", "a_last", "a_judge", "a_neu", "a_sc", "j_logit", "n_tok",
            "truncated", "u_row", "u_idx", "u_lab", "u_res", "u_neu", "u_sc", "u_start", "u_end")

    def flush(buf, part):
        if buf["ids"]:
            np.savez(os.path.join(a.out, f"shard{a.shard:03d}_{part:04d}.npz"), **{k: np.array(v) for k, v in buf.items()})

    part = len(glob.glob(os.path.join(a.out, f"shard{a.shard:03d}_*.npz")))
    buf = {k: [] for k in keys}
    t_start = time.time()
    for k, r in enumerate(todo):
        if time.time() - T0 > a.hours * 3600:
            log("time budget reached")
            break
        f = pr.features(r.get("question") or "", r.get("documents") or [], r["response"], r.get("spans") or [],
                        bool(r.get("has_spans")), int(r["label"]))
        if f is None:
            continue
        row = len(buf["ids"])
        buf["ids"].append(r["id"])
        buf["label"].append(int(r["label"]))
        buf["source"].append(r.get("source", ""))
        buf["task"].append(r.get("task", ""))
        for key in ("a_mean", "a_last", "a_judge", "a_neu", "a_sc", "j_logit", "n_tok", "truncated"):
            buf[key].append(f[key])
        for u in f["units"]:
            buf["u_row"].append(row)
            buf["u_idx"].append(u["idx"])
            buf["u_lab"].append(u["lab"])
            buf["u_start"].append(u["start"])
            buf["u_end"].append(u["end"])
            buf["u_res"].append(u["res"])
            buf["u_neu"].append(u["neu"])
            buf["u_sc"].append(u["sc"])
        if len(buf["ids"]) >= 500:
            flush(buf, part)
            part += 1
            buf = {k_: [] for k_ in keys}
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t_start)
            log(f"{k + 1}/{len(todo)}  {rate * 3600:.0f} answers/h  ETA {(len(todo) - k - 1) / rate / 3600:.1f} h")
    flush(buf, part)
    log("done")


if __name__ == "__main__":
    main()
