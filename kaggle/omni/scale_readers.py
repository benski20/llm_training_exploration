"""Kaggle kernel: how omni-verifier accuracy scales with the frozen reader (Qwen3 1.7B / 4B / 8B / 14B-4bit).

For each reader, on the full corpus (experiments/omni_corpus.py build, 5,107 replies):
    1. features exactly as the product reads them: omni_corpus.py extract -> vt.omni.Reader.features, unchanged.
       Only the model load is adjusted here (monkeypatch of AutoModelForCausalLM.from_pretrained, no repo edits):
       fp16 on T4 (no bf16), sdpa attention, optional nf4 4-bit; a forward hook counts non-finite layer outputs
       (fp16 overflow) and the largest |activation| per layer.
       Readers that fit one T4 run two workers (one per GPU, alternate shards); larger ones span both GPUs.
    2. experiments/omni_final.py (LODO: every expert, adaptation, guarantee) on those features -> final_<tag>.json
    3. vt-omni train --parts mean judge -> heads/omni_qwen3_<size>.npz
    4. T4 seconds per reply (wall time of the extraction, per GPU held) -> timing_<tag>.json

Launched with kaggle/kgrun.py (bundled repo files under /kaggle/working/src):

    python kaggle/kgrun.py --slug omni-scale-smoke --script kaggle/omni/scale_readers.py --bundle <files> \
        --args "--smoke" --out results/kaggle/omni-scale-smoke --wait
    python kaggle/kgrun.py --slug omni-scale --script kaggle/omni/scale_readers.py --bundle <files> \
        --out results/kaggle/omni-scale --wait

Features stay in /tmp (not downloaded); outputs (reports, heads, timings, logs) go to /kaggle/working.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "/kaggle/working/src"
WORK = "/kaggle/working"
TMP = "/tmp/omni"
SELF = os.path.join(SRC, "kaggle", "omni", "scale_readers.py")     # this file, bundled (worker mode re-runs it)
CORPUS_MD5 = "b598848a44e70308b64ae5f0072b6eb0"                     # the repo's corpus.jsonl (5,107 replies)
READERS = {  # tag: (model, quant, gpus per worker)
    "q17": ("Qwen/Qwen3-1.7B", None, 1),
    "q4b": ("Qwen/Qwen3-4B", None, 1),
    "q8b": ("Qwen/Qwen3-8B", None, 2),
    "q14b4": ("Qwen/Qwen3-14B", "nf4", 1),
}
HEAD_NAME = {"q17": "omni_qwen3_1p7b_gpu", "q4b": "omni_qwen3_4b", "q8b": "omni_qwen3_8b", "q14b4": "omni_qwen3_14b_nf4"}
# CPU (bf16) features of the repo's runs, for a few corpus ids: [judge log-odds, mean token log-prob, |r_mean| layer 4]
CPU_REF = {"q17": {"4169": [15.875, -1.079, 397.8], "54": [-4.0, -1.574, 442.1], "1204": [17.375, -3.5, 422.2],
                   "3386": [15.5, -1.235, 402.0], "4070": [4.25, -1.342, 463.6], "1705": [13.0, -3.255, 405.3],
                   "1473": [9.75, -7.796, 403.7], "2429": [-1.5, -4.734, 452.0], "3701": [15.0, -2.026, 369.9],
                   "2809": [9.5, -4.343, 437.3], "4923": [14.5, -1.497, 395.3], "4993": [13.375, -0.83, 387.0]},
           "q4b": {"4169": [9.75, -1.22, 99.2], "54": [-0.5, -3.591, 85.4], "1204": [-5.0, -2.078, 94.5],
                   "3386": [13.75, -1.257, 91.7], "4070": [-15.0, -1.131, 99.9], "1705": [6.0, -2.954, 92.6],
                   "1473": [-28.625, -7.768, 82.0], "2429": [-18.5, -4.345, 99.4], "3701": [11.75, -1.6, 81.8],
                   "2809": [-2.5, -4.303, 97.7], "4923": [16.0, -1.229, 95.2], "4993": [14.625, -0.704, 87.8]}}
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def sh(cmd, logfile=None, env=None, cwd=None):
    log("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    out = open(logfile, "a") if logfile else None
    r = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT if out else None, env=env, cwd=cwd)
    if out:
        out.close()
    return r.returncode


# ---------------------------------------------------------------- worker: one extraction process (env OMNI_WORKER)

def patch_loading(dtype_name, quant, stats):
    """the product's Reader, loaded for a T4: dtype, sdpa, optional 4-bit; plus overflow accounting hooks."""
    import torch
    import transformers
    orig = transformers.AutoModelForCausalLM.from_pretrained.__func__

    def from_pretrained(cls, name, *args, **kw):
        kw.pop("torch_dtype", None)
        kw["dtype"] = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
        kw["attn_implementation"] = "sdpa"
        kw["device_map"] = "auto"
        if quant == "nf4":
            kw["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False)
        m = orig(cls, name, *args, **kw)
        layers = m.model.layers
        stats["_maxabs"] = [None] * len(layers)          # device tensors, no sync per forward

        def hook(i):
            def f(mod, inp, out):
                h = (out[0] if isinstance(out, tuple) else out).detach()
                v = h.abs().amax().float()                  # inf / nan if the layer overflowed
                stats["_maxabs"][i] = v if stats["_maxabs"][i] is None else torch.maximum(stats["_maxabs"][i], v)
            return f

        for i, L in enumerate(layers):
            L.register_forward_hook(hook(i))
        log("loaded", name, {k: str(v) for k, v in kw.items()}, "devices:", sorted({str(p.device) for p in m.parameters()}))
        return m

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)


def worker(cfg):
    import argparse as ap_
    import numpy as np
    sys.path[:0] = [SRC, os.path.join(SRC, "experiments")]
    stats = {"nonfinite_items": 0}
    patch_loading(cfg["dtype"], cfg["quant"], stats)
    import omni_corpus
    from vt import omni
    feat = omni.Reader.features
    stats["feat_seconds"] = 0.0

    def features(self, item):
        t1 = time.time()
        f = feat(self, item)                                # ends in .cpu(): synchronised
        stats["feat_seconds"] += time.time() - t1
        if f is not None:
            bad = not all(np.isfinite(np.asarray(f[k])).all() for k in ("r_mean", "r_last", "j_state", "r_sc", "j_logit"))
            bad = bad or any(not np.isfinite(u["mean"]).all() for u in f["units"])
            stats["nonfinite_items"] += int(bad)
        return f

    omni.Reader.features = features
    t = time.time()
    omni_corpus.cmd_extract(ap_.Namespace(corpus=cfg["corpus"], reader=cfg["reader"], out=cfg["out"], max_tokens=2048,
                                          layer_fracs=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0], shard=100, threads=None,
                                          limit=0))
    n = sum(len(np.load(p)["ids"]) for p in glob.glob(os.path.join(cfg["out"], "shard_*.npz")))
    import torch
    mx = [float(v) if v is not None else None for v in stats.pop("_maxabs")]
    stats.update(maxabs=mx, nonfinite_layers=[int(not np.isfinite(v)) if v is not None else 0 for v in mx])
    stats.update(n=n, seconds=time.time() - t, s_per_reply=stats["feat_seconds"] / max(n, 1),
                 gpu=torch.cuda.get_device_name(0), gpus=torch.cuda.device_count(),
                 peak_mem_gb=[torch.cuda.max_memory_allocated(d) / 1e9 for d in range(torch.cuda.device_count())])
    json.dump(stats, open(os.path.join(cfg["out"], "worker_stats.json"), "w"))
    log("worker done", json.dumps({k: v for k, v in stats.items() if k != "maxabs"}))


# ---------------------------------------------------------------- driver

def setup(need_bnb):
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    log("torch", __import__("torch").__version__, "transformers", transformers.__version__)
    if v < (4, 56):
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"])
    if need_bnb:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            sh([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"])


def build_corpus():
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, "corpus.jsonl")
    if not os.path.exists(path):
        rc = sh([sys.executable, os.path.join(SRC, "experiments", "omni_corpus.py"), "build", "--out", path],
                os.path.join(WORK, "build.log"))
        assert rc == 0, "corpus build failed (see build.log)"
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    n = sum(1 for _ in open(path))
    log("corpus", path, n, "replies, md5", md5, "(same as repo)" if md5 == CORPUS_MD5 else "(DIFFERS from repo)")
    return path, {"md5": md5, "n": n, "same_as_repo": md5 == CORPUS_MD5}


def extract(tag, corpus, out_dir, dtype="fp16", rows_filter=None):
    """run worker processes over the corpus; returns merged stats. Shards of 100 rows alternate between workers."""
    name, quant, gpw = READERS[tag]
    if dtype == "fp32" and tag != "q17":
        gpw = 2                                             # fp32 4B does not fit one T4
    import torch
    ngpu = torch.cuda.device_count()
    nw = max(1, ngpu // gpw)
    rows = [l for l in open(corpus)]
    if rows_filter:
        rows = [rows[i] for i in rows_filter]
    procs, outs = [], []
    t = time.time()
    for w in range(nw):
        part = [r for k, r in enumerate(rows) if (k // 100) % nw == w]
        if not part:
            continue
        cp = os.path.join(TMP, f"corpus_{tag}_{dtype}_w{w}.jsonl")
        open(cp, "w").writelines(part)
        od = os.path.join(TMP, f"x_{tag}_{dtype}_w{w}")
        outs.append(od)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in range(w * gpw, (w + 1) * gpw)),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false",
                   OMNI_WORKER=json.dumps({"corpus": cp, "reader": name, "out": od, "dtype": dtype, "quant": quant}))
        lf = open(os.path.join(WORK, f"extract_{tag}_{dtype}_w{w}.log"), "a")
        procs.append((subprocess.Popen([sys.executable, SELF], env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
    rcs = [p.wait() for p, _ in procs]
    for _, lf in procs:
        lf.close()
    wall = time.time() - t
    os.makedirs(out_dir, exist_ok=True)
    ws = []
    for w, od in enumerate(outs):
        for p in sorted(glob.glob(os.path.join(od, "shard_*.npz"))):
            shutil.move(p, os.path.join(out_dir, f"shard_w{w}_{os.path.basename(p)[6:]}"))
        if os.path.exists(os.path.join(od, "meta.json")):
            shutil.copy(os.path.join(od, "meta.json"), os.path.join(out_dir, "meta.json"))
        if os.path.exists(os.path.join(od, "worker_stats.json")):
            ws.append(json.load(open(os.path.join(od, "worker_stats.json"))))
    n = sum(s["n"] for s in ws)
    st = {"reader": name, "quant": quant, "dtype": dtype, "return_codes": rcs, "workers": len(outs),
          "gpus_per_worker": gpw, "n": n, "wall_s": wall, "worker": ws,
          # forward pass + feature pooling per reply, one worker; x gpus held = T4-seconds billed per reply
          "s_per_reply": (sum(s["feat_seconds"] for s in ws) / max(n, 1)) if ws else None,
          "t4_s_per_reply": (sum(s["feat_seconds"] * gpw for s in ws) / max(n, 1)) if ws else None,
          "incl_load_s_per_reply": (sum(s["seconds"] for s in ws) / max(n, 1)) if ws else None,
          "nonfinite_items": sum(s["nonfinite_items"] for s in ws),
          "nonfinite_layers": [max(x) for x in zip(*[s["nonfinite_layers"] for s in ws])] if ws else None,
          "maxabs_per_layer": [max(x) for x in zip(*[s["maxabs"] for s in ws])] if ws else None}
    log(tag, dtype, "extracted", n, "replies;", f"{st['t4_s_per_reply']} T4-s/reply;", "rc", rcs,
        "nonfinite items", st["nonfinite_items"])
    return st


def analyse(tag, corpus, feats, bg):
    """omni_final (LODO) + vt-omni train, in the background (CPU) while the GPUs read the next reader; one analysis
    at a time (memory)."""
    env = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="")
    os.makedirs(os.path.join(WORK, "heads"), exist_ok=True)
    cmd = (f"{sys.executable} {SRC}/experiments/omni_final.py --feats {feats} --corpus {corpus} "
           f"--report {WORK}/final_{tag}.json > {WORK}/final_{tag}.log 2>&1; "
           f"{sys.executable} -m vt.omni_cli train --feats {feats} --corpus {corpus} --parts mean judge "
           f"--out {WORK}/heads/{HEAD_NAME[tag]}.npz > {WORK}/train_{tag}.log 2>&1; touch {TMP}/done_{tag}")
    if bg:
        cmd = f"while [ ! -e {TMP}/done_{bg[-1][0]} ]; do sleep 20; done; " + cmd
    log("analyse", tag, "(background)")
    bg.append((tag, subprocess.Popen(["bash", "-c", cmd], env=env, cwd=SRC)))


def compare(tag, feats_a, feats_b=None):
    """features vs the repo's CPU (bf16) run for the reference ids, and fp16 vs fp32 when both were read."""
    import numpy as np
    sys.path[:0] = [SRC, os.path.join(SRC, "experiments")]
    from omni_corpus import load_features
    A = load_features(feats_a)
    pa = {int(i): k for k, i in enumerate(A["ids"])}
    out = {"vs_cpu_bf16": {}}
    for i, (jl, lp, nr) in CPU_REF.get(tag, {}).items():
        k = pa.get(int(i))
        if k is not None:
            out["vs_cpu_bf16"][i] = {"j_logit": [jl, round(float(A["j_logit"][k]), 3)],
                                     "lp_mean": [lp, round(float(A["r_sc"][k][0]), 3)],
                                     "norm_rmean_l4": [nr, round(float(np.linalg.norm(A["r_mean"][k][3].astype(np.float32))), 1)]}
    if feats_b:
        B = load_features(feats_b)
        pb = {int(i): k for k, i in enumerate(B["ids"])}
        cos, dj, dl = [], [], []
        for i, k in pa.items():
            if i in pb:
                a, b = A["r_mean"][k].astype(np.float32), B["r_mean"][pb[i]].astype(np.float32)
                cos.append(float(np.min([(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9) for x, y in zip(a, b)])))
                dj.append(abs(float(A["j_logit"][k] - B["j_logit"][pb[i]])))
                dl.append(abs(float(A["r_sc"][k][0] - B["r_sc"][pb[i]][0])))
        out["fp16_vs_fp32"] = {"n": len(cos), "min_layer_cosine_rmean": min(cos) if cos else None,
                               "median_layer_cosine_rmean": float(np.median(cos)) if cos else None,
                               "max_abs_dj_logit": max(dj) if dj else None, "mean_abs_dj_logit": float(np.mean(dj)) if dj else None,
                               "max_abs_dlp_mean": max(dl) if dl else None}
    log("compare", tag, json.dumps(out))
    return out


def smoke(a, corpus):
    """reference ids + the longest replies, fp16 (and fp32 where it fits) for each reader; a tiny end-to-end run."""
    rows = [json.loads(l) for l in open(corpus)]
    ref = [k for k, r in enumerate(rows) if str(r["id"]) in CPU_REF["q17"]]
    longest = sorted(range(len(rows)), key=lambda k: -(len(rows[k]["doc"]) + len(rows[k]["response"])
                                                        + sum(len(m["content"]) for m in rows[k]["messages"])))[:6]
    pick = sorted(set(ref + longest))
    rep = {}
    for tag in a.readers:
        r = {"fp16": extract(tag, corpus, os.path.join(TMP, f"smoke_{tag}_fp16"), "fp16", pick)}
        both = None
        if tag in ("q17", "q4b"):
            r["fp32"] = extract(tag, corpus, os.path.join(TMP, f"smoke_{tag}_fp32"), "fp32", pick)
            both = os.path.join(TMP, f"smoke_{tag}_fp32")
        try:
            r["compare"] = compare(tag, os.path.join(TMP, f"smoke_{tag}_fp16"), both)
        except Exception as e:  # noqa: BLE001
            r["compare"] = {"error": repr(e)}
        rep[tag] = r
        json.dump(rep, open(os.path.join(WORK, "smoke.json"), "w"), indent=1)
    # end to end on 10 replies per domain with the smallest reader: omni_final + train must run
    by_dom = {}
    for k, r in enumerate(rows):
        by_dom.setdefault(r["domain"], []).append(k)
    mini = sorted(k for v in by_dom.values() for k in v[:12])
    st = extract("q17", corpus, os.path.join(TMP, "mini_q17"), "fp16", mini)
    rep["mini_q17"] = st
    env = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="")
    rc1 = sh([sys.executable, f"{SRC}/experiments/omni_final.py", "--feats", os.path.join(TMP, "mini_q17"), "--corpus",
              corpus, "--report", f"{WORK}/final_mini.json", "--reps", "2"], f"{WORK}/final_mini.log", env)
    rc2 = sh([sys.executable, "-m", "vt.omni_cli", "train", "--feats", os.path.join(TMP, "mini_q17"), "--corpus", corpus,
              "--parts", "mean", "judge", "--out", f"{WORK}/heads/mini.npz"], f"{WORK}/train_mini.log", env, SRC)
    rep["mini_rc"] = {"omni_final": rc1, "train": rc2}
    json.dump(rep, open(os.path.join(WORK, "smoke.json"), "w"), indent=1)
    log("smoke done", json.dumps(rep["mini_rc"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="+", default=["q17", "q4b", "q8b", "q14b4"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--optional-after-h", type=float, default=7.0,
                    help="skip the 4-bit 14B reader if this many hours have passed before it starts")
    a = ap.parse_args()
    setup(any(READERS[t][1] for t in a.readers))
    os.makedirs(os.path.join(WORK, "heads"), exist_ok=True)
    corpus, cinfo = build_corpus()
    json.dump(cinfo, open(os.path.join(WORK, "corpus_info.json"), "w"))
    if a.smoke:
        return smoke(a, corpus)
    bg, timing = [], {}
    for tag in a.readers:
        if READERS[tag][1] and (time.time() - T0) / 3600 > a.optional_after_h:
            log("skip", tag, "(time)")
            continue
        feats = os.path.join(TMP, f"feats_{tag}")
        try:
            timing[tag] = extract(tag, corpus, feats)
            json.dump(timing, open(os.path.join(WORK, "timing.json"), "w"), indent=1)
            if timing[tag]["n"] > 0:
                analyse(tag, corpus, feats, bg)
        except Exception as e:  # noqa: BLE001
            log("FAILED", tag, repr(e))
            timing[tag] = {"error": repr(e)}
            json.dump(timing, open(os.path.join(WORK, "timing.json"), "w"), indent=1)
    for tag, p in bg:
        log("waiting for analysis", tag, "rc", p.wait())
    for tag in timing:
        f = os.path.join(WORK, f"final_{tag}.json")
        if os.path.exists(f):
            s = json.load(open(f))["summary"]
            log(tag, json.dumps({g: {k: round(v, 3) for k, v in x.items() if k in ("zero_label", "judge", "adapt_50",
                                                                                 "adapt_100")} for g, x in s.items()}))
    log("all done")


if __name__ == "__main__":
    if os.environ.get("OMNI_WORKER"):
        worker(json.loads(os.environ["OMNI_WORKER"]))
    else:
        main()
