"""Kaggle kernel: an offline LLM judge's P(correct) for every reply of the omni corpus, for judge-assisted calibration
of the false-alarm guarantee (experiments/judge_calibration_sim.py runs the calibration simulation locally).

Judges (each distinct from the reader Qwen3-8B, both larger; one of another family), one per T4, in parallel:
    qwen14   Qwen/Qwen3-14B, 4-bit nf4 (quantized on load, fp16 compute), thinking disabled
    phi4     microsoft/phi-4 (14.7B), 4-bit nf4 (same)
Each judge sees the task (with its documents for grounded replies) and the reply, is asked a format-specific
verification question, and we read P(Yes) / (P(Yes) + P(No)) at the first answer token (one forward pass, no
generation: a budgeted thinking pass over 5,107 replies does not fit a Kaggle session on T4s).

Replies are processed in a seeded random order (blocks of 256, length-sorted inside a block for batching), so a run
stopped by --hours leaves a uniform random subset. Output: /kaggle/working/judge_<tag>.npz (ids, logit = log P(Yes) -
log P(No), p_correct, yesno_mass, n_tokens, truncated) + judge_<tag>.log, checkpointed every block.

    python kaggle/kgrun.py --slug omni-judge-smoke --script kaggle/omni/judge_calibration.py \
        --bundle vt/omni.py experiments/omni_corpus.py --args "--per-domain 4" --out results/kaggle/omni-judge-smoke --wait
    python kaggle/kgrun.py --slug omni-judge --script kaggle/omni/judge_calibration.py \
        --bundle vt/omni.py experiments/omni_corpus.py --out results/kaggle/omni-judge --wait
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

SRC = "/kaggle/working/src"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TMP = "/tmp/omni"
SELF = os.path.join(SRC, "kaggle", "omni", "judge_calibration.py")
CORPUS_MD5 = "b598848a44e70308b64ae5f0072b6eb0"
JUDGES = {"qwen14": ("Qwen/Qwen3-14B", "unsloth/Qwen3-14B-bnb-4bit"),
          "phi4": ("microsoft/phi-4", "unsloth/phi-4-bnb-4bit")}
SYSTEM = "You are a meticulous, impartial fact-checker. You verify answers written by an AI assistant."
Q_GROUNDED = ("Is the reply fully correct and faithful: is every statement in it supported by the documents / "
              "context in the task, with nothing contradicted, fabricated or unsupported?")
Q_REASON = ("Check the solution step by step. Is every step correct (no calculation, logic or reading error), "
            "so that the solution is fully correct?")
Q_CLOSED = "Is the reply fully correct: is every factual statement in it accurate, with nothing false or misleading?"
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


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
        # RAGTruth input_str is the annotation prompt (task + context, the reply, an instruction to label spans):
        # keep only the task and its context; the context is already inside it
        user = user[:user.index(r["response"][:60])].rstrip()
        user = user[:user.rfind("\n")] if "\n" in user else user        # drop the "Below is the answer:" header
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


def chunked_last_hidden(model, ids, chunk=512):
    """one long prompt, prefilled in chunks with a KV cache: attention memory O(chunk x L), not O(L^2)."""
    from transformers import DynamicCache
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    for s in range(0, ids.shape[1], chunk):
        out = model.model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True)
    return out.last_hidden_state[:, -1]


def last_logprobs(model, inp, att):
    """log-softmax of the next-token logits at the last position; halves the batch on CUDA OOM."""
    import torch
    try:
        with torch.no_grad():
            if len(inp) == 1 and inp.shape[1] > 768:
                h = chunked_last_hidden(model, inp.cuda())
            else:
                mask = None if bool(att.all()) else att.cuda()
                h = model.model(input_ids=inp.cuda(), attention_mask=mask).last_hidden_state[:, -1]
            return torch.log_softmax(model.lm_head(h).float(), -1).cpu()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(inp) == 1:
            raise
        k = len(inp) // 2
        return torch.cat([last_logprobs(model, inp[:k], att[:k]), last_logprobs(model, inp[k:], att[k:])])


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
    yes = answer_ids(tok, ["Yes", "yes", " Yes", " yes", "YES"])
    no = answer_ids(tok, ["No", "no", " No", " no", "NO"])
    log(tag, "yes ids", yes, tok.convert_ids_to_tokens(yes), "no ids", no, tok.convert_ids_to_tokens(no))
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
            first = False
        block.sort()
        j = 0
        while j < len(block):
            L = block[j][0]
            # long prompts one at a time with no padding mask (sdpa then uses its memory-efficient causal kernel;
            # a padded batch needs an explicit mask and the O(L^2) path, which runs out of memory on a T4)
            bs = 1 if L > 768 else max(1, min(a.max_batch, a.batch_tokens // max(L, 1)))
            chunk = block[j:j + bs]
            j += bs
            Lm = max(c[0] for c in chunk)
            inp = torch.full((len(chunk), Lm), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), Lm), dtype=torch.long)
            for k, c in enumerate(chunk):
                inp[k, Lm - c[0]:] = torch.tensor(c[2])
                att[k, Lm - c[0]:] = 1
            lp = last_logprobs(model, inp, att)
            for k, c in enumerate(chunk):
                ly = torch.logsumexp(lp[k, yes], 0).item()
                ln = torch.logsumexp(lp[k, no], 0).item()
                res["ids"].append(rows[c[1]]["id"])
                res["logit"].append(ly - ln)
                res["p_correct"].append(1 / (1 + np.exp(-(ly - ln))))
                res["yesno_mass"].append(float(np.exp(ly) + np.exp(ln)))
                res["n_tokens"].append(c[0])
                res["truncated"].append(c[3])
        np.savez(outp, model=name, **{k: np.array(v) for k, v in res.items()})
        n = len(res["ids"])
        log(tag, f"{n}/{len(order)}", f"{(time.time() - t_start) / n:.2f} s/reply",
            "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "mean yes/no mass", round(float(np.mean(res["yesno_mass"])), 3))
    log(tag, "done", len(res["ids"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["qwen14", "phi4"])
    ap.add_argument("--per-domain", type=int, default=0, help="smoke test: this many replies per domain")
    ap.add_argument("--max-tokens", type=int, default=3584)
    ap.add_argument("--batch-tokens", type=int, default=6144)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--hours", type=float, default=10.5)
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
    subprocess.run("nvidia-smi; df -h /tmp /root /kaggle/working", shell=True)
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
    while any(p.poll() is None for _, p in procs):
        time.sleep(120)
        for tag, _ in procs:
            lines = open(os.path.join(WORK, f"judge_{tag}.log")).read().strip().splitlines()
            log(tag, lines[-1][:200] if lines else "")
    json.dump({"corpus_md5": md5, "return_codes": {t: p.returncode for t, p in procs},
               "minutes": (time.time() - T0) / 60}, open(os.path.join(WORK, "judge_info.json"), "w"))
    log("all done", {t: p.returncode for t, p in procs})


if __name__ == "__main__":
    sys.exit(main())
