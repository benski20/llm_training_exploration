"""Kaggle kernel: throughput of the one-pass judges (the "teacher as the product" cost), on Kaggle's T4s.

Each judge reads ~800-token verification prompts built from the omni corpus exactly as kaggle/omni/judge_calibration.py
does (same prompt, same Yes/No read-out), at batch sizes 1, 8 and 32 (padded, left-aligned; a batch that runs out of
memory is split in halves, which is what a server would do, and the split count is recorded).

    phase 1  Qwen/Qwen3-14B and microsoft/phi-4, 4-bit nf4, one per T4, in parallel
    phase 2  the same in fp16 spread over both T4s (device_map auto), where it fits (14B fp16 is ~28 GB: tight)

Output: /kaggle/working/throughput.json: per (judge, precision, batch): checks/s, p50/p95 latency of a check (its
batch's wall time), mean prompt tokens, OOM splits, and $/1,000 checks at an assumed T4 price.

    python kaggle/kgrun.py --slug omni-judge-tput --script kaggle/omni/judge_throughput.py \
        --bundle vt/omni.py experiments/omni_corpus.py kaggle/omni/judge_calibration.py kaggle/omni/judge_throughput.py \
        --out results/kaggle/omni-judge-tput --wait
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

SRC = "/kaggle/working/src"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SELF = os.path.join(SRC, "kaggle", "omni", "judge_throughput.py")
sys.path.insert(0, os.path.join(SRC, "kaggle", "omni"))
T4_USD_PER_HOUR = 0.35          # on-demand T4 (cloud list prices are ~$0.35/h per GPU); an assumption, stated in output
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def load(tag, precision):
    import torch
    import transformers
    from judge_calibration import JUDGES
    name, preq = JUDGES[tag]
    tok = transformers.AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if precision == "nf4":
        for src, quant in ((name, True), (preq, False)):
            try:
                kw = dict(dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
                if quant:
                    kw["quantization_config"] = transformers.BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
                return tok, transformers.AutoModelForCausalLM.from_pretrained(src, **kw).eval()
            except Exception as e:                                   # noqa: BLE001
                log(tag, "load failed", src, repr(e)[:200])
        raise RuntimeError("no 4-bit load")
    n = torch.cuda.device_count()
    return tok, transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
        max_memory={i: "14GiB" for i in range(n)}).eval()


def bench(tag, precision, a):
    import numpy as np
    import torch
    from judge_calibration import answer_ids, prompt
    rows = [json.loads(l) for l in open(a.corpus)]
    tok, model = load(tag, precision)
    on_cpu = [k for k, v in (getattr(model, "hf_device_map", None) or {}).items() if v in ("cpu", "disk")]
    if on_cpu:
        raise RuntimeError(f"does not fit on the GPUs ({len(on_cpu)} modules offloaded)")
    log(tag, precision, "loaded; GPU memory GB", [round(torch.cuda.memory_allocated(i) / 1e9, 1)
                                                  for i in range(torch.cuda.device_count())])
    yes, no = answer_ids(tok, ["Yes", "yes", " Yes", " yes"]), answer_ids(tok, ["No", "no", " No", " no"])
    pool = []
    for r in random.Random(0).sample(rows, len(rows)):  # nosec B311
        text, _ = prompt(tok, r, a.max_tokens)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if 600 <= len(ids) <= 1000:                                  # ~800-token checks
            pool.append(ids)
        if len(pool) >= a.n:
            break
    log(tag, precision, "checks", len(pool), "mean tokens", float(np.mean([len(x) for x in pool])))
    splits = [0]

    def run(batch):
        L = max(len(x) for x in batch)
        inp = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for k, x in enumerate(batch):
            inp[k, L - len(x):] = torch.tensor(x)
            att[k, L - len(x):] = 1
        dev = model.get_input_embeddings().weight.device

        def go(i, m):
            try:
                with torch.no_grad():
                    h = model.model(input_ids=i.to(dev), attention_mask=None if bool(m.all()) else m.to(dev)
                                    ).last_hidden_state[:, -1]
                    return torch.log_softmax(model.lm_head(h).float(), -1).cpu()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(i) == 1:
                    raise
                splits[0] += 1
                k = len(i) // 2
                return torch.cat([go(i[:k], m[:k]), go(i[k:], m[k:])])
        lp = go(inp, att)
        return (torch.logsumexp(lp[:, no], 1) - torch.logsumexp(lp[:, yes], 1)).tolist()

    run(pool[:2])                                                    # warm-up
    out = {}
    for bs in (1, 8, 32):
        splits[0] = 0
        lat, t = [], time.perf_counter()
        try:
            for b0 in range(0, len(pool), bs):
                batch = pool[b0:b0 + bs]
                torch.cuda.synchronize()
                s = time.perf_counter()
                run(batch)
                torch.cuda.synchronize()
                lat += [time.perf_counter() - s] * len(batch)
        except torch.OutOfMemoryError:
            out[str(bs)] = {"error": "out of memory at batch 1"}
            continue
        wall = time.perf_counter() - t
        cps = len(pool) / wall
        gpus = torch.cuda.device_count() if precision == "fp16" else 1
        out[str(bs)] = {"checks_per_s": round(cps, 3), "p50_s": round(float(np.percentile(lat, 50)), 3),
                        "p95_s": round(float(np.percentile(lat, 95)), 3), "oom_splits": splits[0], "gpus": gpus,
                        "usd_per_1000_checks": round(1000 / cps / 3600 * T4_USD_PER_HOUR * gpus, 4)}
        log(tag, precision, "batch", bs, out[str(bs)])
    return {"judge": tag, "precision": precision, "n_checks": len(pool),
            "mean_tokens": float(np.mean([len(x) for x in pool])), "by_batch": out}


def worker(a):
    try:
        r = bench(a.worker, a.precision, a)
    except Exception as e:                                           # noqa: BLE001
        log(a.worker, a.precision, "FAILED", repr(e)[:300])
        r = {"judge": a.worker, "precision": a.precision, "error": repr(e)[:300]}
    json.dump(r, open(os.path.join(WORK, f"tput_{a.worker}_{a.precision}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--precision", default="nf4")
    ap.add_argument("--corpus", default="/tmp/omni/corpus.jsonl")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-fp16", action="store_true")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43"], check=False)
    sys.path.insert(0, SRC)
    from judge_calibration import build_corpus
    a.corpus, _ = build_corpus()
    me = SELF if os.path.exists(SELF) else __file__
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    base = [sys.executable, me, "--corpus", a.corpus, "--n", str(a.n), "--max-tokens", str(a.max_tokens)]
    procs = [subprocess.Popen(base + ["--worker", t, "--precision", "nf4"], env=dict(env, CUDA_VISIBLE_DEVICES=str(g)),
                              stdout=open(os.path.join(WORK, f"{t}_nf4.log"), "w"), stderr=subprocess.STDOUT)
             for g, t in enumerate(("qwen14", "phi4"))]
    for p in procs:
        p.wait()
    if not a.no_fp16:
        for t in ("qwen14", "phi4"):
            subprocess.run(base + ["--worker", t, "--precision", "fp16"], env=env,
                           stdout=open(os.path.join(WORK, f"{t}_fp16.log"), "w"), stderr=subprocess.STDOUT)
    res = {"gpu": "Kaggle T4 (16 GB); Kaggle offers only T4/P100", "t4_usd_per_hour_assumed": T4_USD_PER_HOUR,
           "runs": [json.load(open(os.path.join(WORK, f))) for f in sorted(os.listdir(WORK)) if f.startswith("tput_")]}
    json.dump(res, open(os.path.join(WORK, "throughput.json"), "w"), indent=1)
    for f in sorted(os.listdir(WORK)):
        if f.endswith(".log"):
            print(f"--- {f} (tail)\n" + "".join(open(os.path.join(WORK, f)).readlines()[-12:]))
    log("done")


if __name__ == "__main__":
    main()
