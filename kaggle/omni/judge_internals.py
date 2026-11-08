"""Kaggle kernel: a white-box judge. Does a linear probe on an external judge's INTERNALS at the verdict position beat
the judge's own output P(Yes) / P(No) on held-out domains?

Same judges, prompts and loading as kaggle/omni/judge_calibration.py (one forward pass per reply, no generation):
    qwen14   Qwen/Qwen3-14B, 4-bit nf4 (quantized on load, fp16 compute), thinking disabled
    phi4     microsoft/phi-4 (14.7B), 4-bit nf4 (same)
For each of the 5,107 corpus replies we keep, besides the output log-odds log P(Yes) - log P(No) at the answer
position, the residual stream at the LAST prompt token (the state that produces the Yes / No) after decoder layers at
fractions 0.3, 0.4, 0.5, 0.6, 0.7, 0.85 and 1.0 of depth (pre final norm). The states (~730 MB per judge) stay on the
kernel's /tmp; the probing runs inside the kernel, after extraction, exactly as experiments/omni_eval.py does it:
global z-score -> PCA 256 (label-free, all replies) -> per-domain standardisation from each domain's unlabelled
replies -> leave-one-domain-out L2 logistic probe (l2 0.05, balanced weights). Reported per held-out domain and per
format, against the output logit, and the zero-label combination z(probe) + z(output) per domain.

Outputs (/kaggle/working): judge_internals.json (all AUROCs), judge_internals_<tag>.npz (ids, domain, label,
judge_output_logit, probe_best, probe_mid3, probe_last, probe_plus_logit: per-reply LODO scores; higher = more likely
wrong, except judge_output_logit = log P(Yes) - log P(No)), judge_<tag>.npz / .log (extraction checkpoints).

    python kaggle/kgrun.py --slug omni-judge-internals --script kaggle/omni/judge_internals.py \
        --bundle vt/omni.py experiments/omni_corpus.py experiments/omni_eval.py vt/correctness.py \
                 kaggle/omni/judge_internals.py --out results/kaggle/omni-judge-internals --wait
(the script bundles itself: the per-GPU workers re-run it as a subprocess.) Smoke: --args "--per-domain 6".
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback

SRC = "/kaggle/working/src"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TMP = "/tmp/omni"
SELF = os.path.join(SRC, "kaggle", "omni", "judge_internals.py")
CORPUS_MD5 = "b598848a44e70308b64ae5f0072b6eb0"
JUDGES = {"qwen14": ("Qwen/Qwen3-14B", "unsloth/Qwen3-14B-bnb-4bit"),
          "phi4": ("microsoft/phi-4", "unsloth/phi-4-bnb-4bit")}
FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
MID3 = (0.5, 0.6, 0.7)
SYSTEM = "You are a meticulous, impartial fact-checker. You verify answers written by an AI assistant."
Q_GROUNDED = ("Is the reply fully correct and faithful: is every statement in it supported by the documents / "
              "context in the task, with nothing contradicted, fabricated or unsupported?")
Q_REASON = ("Check the solution step by step. Is every step correct (no calculation, logic or reading error), "
            "so that the solution is fully correct?")
Q_CLOSED = "Is the reply fully correct: is every factual statement in it accurate, with nothing false or misleading?"
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


# ----------------------------------------------------------------- unchanged from judge_calibration.py

def build_corpus():
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, "corpus.jsonl")
    if not os.path.exists(path):
        rc = subprocess.run([sys.executable, os.path.join(SRC, "experiments", "omni_corpus.py"), "build", "--out",
                             path], stdout=open(os.path.join(WORK, "build.log"), "w"), stderr=subprocess.STDOUT).returncode
        assert rc == 0, "corpus build failed (see build.log)"
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    log("corpus md5", md5, "(same as repo)" if md5 == CORPUS_MD5 else "(DIFFERS from repo)")
    return path, md5


def cut(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= n:
        return text, False
    h = int(n * 0.7)
    return tok.decode(ids[:h]) + "\n[... truncated ...]\n" + tok.decode(ids[-(n - h):]), True


def prompt(tok, r, max_tokens):
    user = r["messages"][-1]["content"]
    doc = r.get("doc") or ""
    if "hallucinated span" in user and r["response"][:60] in user:
        user = user[:user.index(r["response"][:60])].rstrip()
        user = user[:user.rfind("\n")] if "\n" in user else user
        doc = ""
    if doc and " ".join(doc.split())[:100] in " ".join(user.split()):
        doc = ""
    q = Q_REASON if r["format"] == "reasoning" else (Q_GROUNDED if (r.get("doc") or "") else Q_CLOSED)
    reply, t1 = cut(tok, r["response"], 1800)
    budget = max_tokens - len(tok(reply, add_special_tokens=False)["input_ids"]) - 300
    if doc:
        user, t2 = cut(tok, user, budget // 2)
        doc, t3 = cut(tok, doc, budget // 2)
    else:
        user, t2 = cut(tok, user, budget)
        t3 = False
    body = "The task given to the assistant:\n<task>\n" + user + "\n</task>\n\n"
    if doc:
        body += "Source documents for the task:\n<documents>\n" + doc + "\n</documents>\n\n"
    body += "The assistant's reply:\n<reply>\n" + reply + "\n</reply>\n\n" + q + " Answer with only Yes or No."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text, bool(t1 or t2 or t3)


def answer_ids(tok, words):
    out = set()
    for w in words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            out.add(ids[0])
    return sorted(out)


# ----------------------------------------------------------------- extraction (+ last-token states via hooks)

CAP = {}


def add_hooks(model, layer_ids):
    layers = model.model.layers

    def mk(j):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            CAP[j] = o[:, -1].detach()          # last position (left padding: the last prompt token of every row)
        return hook
    for j, l in enumerate(layer_ids):
        layers[l - 1].register_forward_hook(mk(j))


def chunked_last_hidden(model, ids, chunk=512):
    from transformers import DynamicCache
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    for s in range(0, ids.shape[1], chunk):
        out = model.model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True)
    return out.last_hidden_state[:, -1]           # hooks: the last chunk's call wins = the last token


def last_logprobs(model, inp, att, nl):
    """(log-softmax at the last position, [B, nl, D] float32 states); halves the batch on CUDA OOM."""
    import torch
    try:
        CAP.clear()
        with torch.no_grad():
            if len(inp) == 1 and inp.shape[1] > 768:
                h = chunked_last_hidden(model, inp.cuda())
            else:
                mask = None if bool(att.all()) else att.cuda()
                h = model.model(input_ids=inp.cuda(), attention_mask=mask).last_hidden_state[:, -1]
            H = torch.stack([CAP[j] for j in range(nl)], 1).float().cpu()
            return torch.log_softmax(model.lm_head(h).float(), -1).cpu(), H
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(inp) == 1:
            raise
        k = len(inp) // 2
        a1, h1 = last_logprobs(model, inp[:k], att[:k], nl)
        a2, h2 = last_logprobs(model, inp[k:], att[k:], nl)
        return torch.cat([a1, a2]), torch.cat([h1, h2])


def worker(a):
    import numpy as np
    import torch
    import transformers
    tag = a.worker
    name, preq = JUDGES[tag]
    rows = [json.loads(l) for l in open(a.corpus)]
    if a.per_domain:
        by = {}
        for r in rows:
            by.setdefault(r["domain"], []).append(r)
        rows = [r for d in sorted(by) for r in random.Random(0).sample(by[d], min(a.per_domain, len(by[d])))]  # nosec B311
    order = list(range(len(rows)))
    random.Random(1).shuffle(order)  # nosec B311
    tok = transformers.AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = None
    for src in ((name, "nf4"), (preq, None)):
        try:
            kw = dict(dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
            if src[1]:
                kw["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            model = transformers.AutoModelForCausalLM.from_pretrained(src[0], **kw).eval()
            log(tag, "loaded", src[0], "mem GB", torch.cuda.memory_allocated() / 1e9)
            break
        except Exception as e:                                       # noqa: BLE001
            log(tag, "load failed", src[0], repr(e)[:300])
    assert model is not None
    nlay = len(model.model.layers)
    layer_ids = sorted({max(1, min(nlay, round(f * nlay))) for f in FRACS})
    dim = model.config.hidden_size
    add_hooks(model, layer_ids)
    log(tag, "layers", nlay, "hooked (1-based outputs)", layer_ids, "hidden", dim)
    yes = answer_ids(tok, ["Yes", "yes", " Yes", " yes", "YES"])
    no = answer_ids(tok, ["No", "no", " No", " no", "NO"])
    log(tag, "yes ids", yes, tok.convert_ids_to_tokens(yes), "no ids", no, tok.convert_ids_to_tokens(no))
    os.makedirs(TMP, exist_ok=True)
    mm = np.lib.format.open_memmap(os.path.join(TMP, f"states_{tag}.npy"), mode="w+", dtype=np.float32,
                                   shape=(len(rows), len(layer_ids), dim))
    res = {k: [] for k in ("ids", "logit", "p_correct", "yesno_mass", "n_tokens", "truncated")}
    outp = os.path.join(WORK, f"judge_{tag}.npz")
    t_start = time.time()
    first = True
    for b0 in range(0, len(order), 256):
        if time.time() - T0 > a.hours * 3600:
            log(tag, "time budget reached; stopping with", len(res["ids"]), "replies")
            break
        block = []
        for i in order[b0:b0 + 256]:
            text, tr = prompt(tok, rows[i], a.max_tokens)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            block.append((len(ids), i, ids, tr))
        if first:
            log(tag, "example prompt:\n" + tok.decode(block[0][2]))
        block.sort()
        j = 0
        while j < len(block):
            L = block[j][0]
            bs = 1 if L > 768 else max(1, min(a.max_batch, a.batch_tokens // max(L, 1)))
            chunk = block[j:j + bs]
            j += bs
            Lm = max(c[0] for c in chunk)
            inp = torch.full((len(chunk), Lm), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), Lm), dtype=torch.long)
            for k, c in enumerate(chunk):
                inp[k, Lm - c[0]:] = torch.tensor(c[2])
                att[k, Lm - c[0]:] = 1
            lp, H = last_logprobs(model, inp, att, len(layer_ids))
            for k, c in enumerate(chunk):
                ly = torch.logsumexp(lp[k, yes], 0).item()
                ln = torch.logsumexp(lp[k, no], 0).item()
                mm[len(res["ids"])] = H[k].numpy()                  # row = processing order, aligned with ids
                res["ids"].append(rows[c[1]]["id"])
                res["logit"].append(ly - ln)
                res["p_correct"].append(1 / (1 + np.exp(-(ly - ln))))
                res["yesno_mass"].append(float(np.exp(ly) + np.exp(ln)))
                res["n_tokens"].append(c[0])
                res["truncated"].append(c[3])
        mm.flush()
        n = len(res["ids"])
        if first:
            Hn = np.asarray(mm[:n])
            log(tag, "state norms per layer (median)", np.round(np.median(np.linalg.norm(Hn, axis=2), 0), 1).tolist(),
                "finite", bool(np.isfinite(Hn).all()))
            first = False
        np.savez(outp, model=name, layers=np.array(layer_ids), n_layers=nlay, fracs=np.array(FRACS),
                 **{k: np.array(v) for k, v in res.items()})
        log(tag, f"{n}/{len(order)}", f"{(time.time() - t_start) / n:.2f} s/reply",
            "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "mean yes/no mass", round(float(np.mean(res["yesno_mass"])), 3))
    log(tag, "done", len(res["ids"]))


# ----------------------------------------------------------------- probing (CPU numpy + GPU PCA if available)

def pca_scores(X, k, gpu=True):
    """global z-score, then the top-k principal component scores (Gram trick on GPU if available)."""
    import numpy as np
    X = (X - X.mean(0)) / (X.std(0) + 1e-4)
    k = min(k, X.shape[0] - 1)
    try:
        import torch
        if not (gpu and torch.cuda.is_available()):
            raise ImportError
        T = torch.from_numpy(X.astype(np.float32)).cuda()
        T -= T.mean(0)
        G = (T @ T.T).double()
        w, U = torch.linalg.eigh(G)
        w, U = w.flip(0)[:k].clamp_min(0), U.flip(1)[:, :k]
        out = (U * w.sqrt()).float().cpu().numpy()
        del T, G, U
        torch.cuda.empty_cache()
        return out
    except ImportError:
        from omni_eval import pca_fit
        mu, V = pca_fit(X, k)
        return (X - mu) @ V.T


def zdom(v, dom):
    import numpy as np
    out = np.zeros(len(v))
    for d in np.unique(dom):
        m = dom == d
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-9)
    return out


def analyse(tag, feats, logit, y, dom, fmt, layer_names, mid_idx, l2=0.05, pca=256, quick=False):
    """feats(j) -> [n, D] float32 states of layer j. Returns (report, per-reply scores)."""
    import numpy as np
    from omni_eval import auroc, in_domain, lodo

    def per(s):
        return {d: auroc(s[dom == d], y[dom == d]) for d in np.unique(dom)}

    rep, scores, X_of = {}, {}, {}
    out = -logit
    rep["output"] = per(out)
    scores["output"] = out
    sets = [(nm, [j]) for j, nm in enumerate(layer_names)]
    if quick:
        sets = sets[-2:]
    else:
        sets += [("mid3", list(mid_idx)), ("all", list(range(len(layer_names))))]
    for nm, js in sets:
        t = time.time()
        H = np.concatenate([feats(j) for j in js], 1)
        X = pca_scores(H, pca, gpu=not quick)            # the gate runs beside the workers: CPU only
        del H
        X_of[nm] = X
        r, s = lodo(X, y, dom, l2, True)
        rep[nm], scores[nm] = r, s
        log(tag, f"probe {nm:8s} LODO+dstd mean AUROC {np.nanmean(list(r.values())):.3f}  "
                 f"(output {np.nanmean(list(rep['output'].values())):.3f})  {time.time() - t:.0f}s")
    probes = [nm for nm, _ in sets]
    best = max(probes, key=lambda k: np.nanmean(list(rep[k].values())))
    rep["_best"] = best
    # combinations, zero labels: per-domain z-scores (unlabelled statistics)
    for nm in sorted({best, "mid3", probes[-1]} & set(probes)):
        rep[f"z({nm})+z(output)"] = per(zdom(scores[nm], dom) + zdom(out, dom))
    # the output log-odds as one more probe feature
    Xb = np.concatenate([X_of[best], out[:, None]], 1)
    rep["probe+logit_feature"], scores["probe+logit"] = lodo(Xb, y, dom, l2, True)
    if not quick:
        for l2_ in (0.005, 0.5):
            rep[f"{best}@l2={l2_}"], _ = lodo(X_of[best], y, dom, l2_, True)
        rep[f"{best}@in_domain_5fold"] = in_domain(X_of[best], y, dom, l2)
        # a probe with no per-domain standardisation (shows what the cheap covariate correction is worth)
        rep[f"{best}@no_dstd"], _ = lodo(X_of[best], y, dom, l2, False)
    summary = {}
    fmts = sorted(set(fmt.tolist()))
    dfmt = {d: fmt[dom == d][0] for d in np.unique(dom)}
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        row = {f: float(np.nanmean([v[d] for d in v if dfmt[d] == f])) for f in fmts}
        row["all"] = float(np.nanmean(list(v.values())))
        row["wins_vs_output"] = int(sum(v[d] > rep["output"][d] for d in v))
        summary[k] = row
    rep["_summary"] = summary
    log(tag, "summary (" + " / ".join(fmts) + " / all):")
    for k, row in summary.items():
        log(f"   {k:28s} " + "  ".join(f"{row[f]:.3f}" for f in fmts + ["all"]) + f"   wins {row['wins_vs_output']}/21")
    scores["best"] = scores[best]
    return rep, scores


def run_analysis(tags, corpus, quick=False):
    import numpy as np
    rows = {}
    for l in open(corpus):
        r = json.loads(l)
        rows[str(r["id"])] = r
    full, per_reply = {}, {}
    for tag in tags:
        p = os.path.join(WORK, f"judge_{tag}.npz")
        if not os.path.exists(p):
            log(tag, "no extraction output; skipped")
            continue
        z = np.load(p, allow_pickle=True)
        ids = [str(i) for i in z["ids"]]
        n = len(ids)
        mm = np.load(os.path.join(TMP, f"states_{tag}.npy"), mmap_mode="r")
        dom = np.array([rows[i]["domain"] for i in ids])
        fmt = np.array([rows[i]["format"] for i in ids])
        y = np.array([rows[i]["label"] for i in ids], float)
        L = [int(x) for x in z["layers"]]
        nl = int(z["n_layers"])
        names = [f"L{l}" for l in L]
        mid = [j for j, l in enumerate(L) if any(abs(l - round(f * nl)) == 0 for f in MID3)]
        log(tag, f"analysing {n} replies, {len(np.unique(dom))} domains, layers {L} of {nl}, mid3 {mid}")
        rep, sc = analyse(tag, lambda j: np.asarray(mm[:n, j], np.float32), z["logit"].astype(float), y, dom, fmt,
                          names, mid, quick=quick)
        rep["_meta"] = {"model": str(z["model"]), "n": n, "layers": L, "n_layers": nl}
        full[tag] = rep
        per_reply[tag] = dict(ids=np.array(ids), domain=dom, format=fmt, label=y.astype(int),
                              judge_output_logit=z["logit"], probe_best=sc["best"], probe_best_name=rep["_best"],
                              probe_mid3=sc.get("mid3", sc["best"]), probe_last=sc[names[-1]],
                              probe_plus_logit=sc["probe+logit"])
        if not quick:
            np.savez(os.path.join(WORK, f"judge_internals_{tag}.npz"), **per_reply[tag])
    if len(per_reply) == 2 and not quick:                      # both judges: zero-label sums of per-domain z-scores
        a_, b_ = (per_reply[t] for t in tags)
        pos = {i: k for k, i in enumerate(b_["ids"])}
        k_ = np.array([pos.get(i, -1) for i in a_["ids"]])
        ok = k_ >= 0
        dom, y, fmt = a_["domain"][ok], a_["label"][ok], a_["format"][ok]
        oa, ob = -a_["judge_output_logit"][ok], -b_["judge_output_logit"][k_[ok]]
        pa, pb = a_["probe_best"][ok], b_["probe_best"][k_[ok]]
        from omni_eval import auroc
        combos = {"outputs": zdom(oa, dom) + zdom(ob, dom),
                  "probes": zdom(pa, dom) + zdom(pb, dom),
                  "outputs+probes": zdom(oa, dom) + zdom(ob, dom) + zdom(pa, dom) + zdom(pb, dom)}
        both = {}
        for nm, s in combos.items():
            per = {d: auroc(s[dom == d], y[dom == d]) for d in np.unique(dom)}
            dfmt = {d: fmt[dom == d][0] for d in np.unique(dom)}
            row = {f: float(np.nanmean([per[d] for d in per if dfmt[d] == f])) for f in sorted(set(fmt.tolist()))}
            row["all"] = float(np.nanmean(list(per.values())))
            both[nm] = {"per_domain": per, "summary": row}
            log("both judges", nm, {k: round(v, 3) for k, v in row.items()})
        full["both_judges"] = both
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["qwen14", "phi4"])
    ap.add_argument("--per-domain", type=int, default=0, help="smoke test: this many replies per domain")
    ap.add_argument("--max-tokens", type=int, default=3584)
    ap.add_argument("--batch-tokens", type=int, default=6144)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--worker")
    ap.add_argument("--corpus")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    import transformers
    log("transformers", transformers.__version__)
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"])
    subprocess.run("nvidia-smi; df -h /tmp /root /kaggle/working; free -g", shell=True)
    corpus, md5 = build_corpus()
    procs = []
    for g, tag in enumerate(a.judges):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % 2), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, SELF, "--worker", tag, "--corpus", corpus, "--per-domain", str(a.per_domain),
               "--max-tokens", str(a.max_tokens), "--batch-tokens", str(a.batch_tokens), "--max-batch",
               str(a.max_batch), "--hours", str(a.hours)]
        procs.append((tag, subprocess.Popen(cmd, env=env, stdout=open(os.path.join(WORK, f"judge_{tag}.log"), "w"),
                                            stderr=subprocess.STDOUT)))
        log("started", tag, "on GPU", g % 2)
    gated, tries = False, 0
    while any(p.poll() is None for _, p in procs):
        time.sleep(60)
        for tag, _ in procs:
            lines = open(os.path.join(WORK, f"judge_{tag}.log")).read().strip().splitlines()
            log(tag, lines[-1][:200] if lines else "")
        # gate: once the first block is on disk, run the probing pipeline on it (CPU only) to catch bugs early
        if not gated and tries < 3 and os.path.exists(os.path.join(WORK, f"judge_{a.judges[0]}.npz")):
            tries += 1
            try:
                run_analysis([a.judges[0]], corpus, quick=True)
                gated = True
                log("GATE OK: probing pipeline ran on the first block")
            except Exception:                                    # noqa: BLE001  (a half-written checkpoint: retry)
                log(f"GATE attempt {tries} failed:\n" + traceback.format_exc())
    rc = {t: p.returncode for t, p in procs}
    log("extraction done", rc)
    info = {"corpus_md5": md5, "return_codes": rc, "minutes_extraction": (time.time() - T0) / 60}
    try:
        info["report"] = run_analysis(a.judges, corpus)
        json.dump(info, open(os.path.join(WORK, "judge_internals.json"), "w"), indent=1, default=float)
    except Exception:                                            # noqa: BLE001
        log("ANALYSIS FAILED:\n" + traceback.format_exc())
        import numpy as np
        for tag in a.judges:
            p = os.path.join(TMP, f"states_{tag}.npy")
            if os.path.exists(p):
                np.save(os.path.join(WORK, f"states_{tag}_f16.npy"), np.load(p, mmap_mode="r").astype(np.float16))
        json.dump(info, open(os.path.join(WORK, "judge_internals.json"), "w"), indent=1, default=float)
    log("all done", (time.time() - T0) / 60, "min")


if __name__ == "__main__":
    sys.exit(main())
