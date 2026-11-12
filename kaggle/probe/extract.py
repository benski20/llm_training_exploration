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


def load_records(paths, shard, n_shards):
    recs = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            # deterministic sharding by id hash, independent of file order
            if int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % n_shards == shard:
                recs.append(r)
    recs.sort(key=lambda r: r["id"])
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--build", action="store_true", help="clone the repo and build the data with probe_data.py")
    ap.add_argument("--branch", default="claude/keen-euler-0kccyk")
    ap.add_argument("--sources", default="", help="for --build: comma-separated sources (default all)")
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if a.build:
        repo = "/kaggle/working/repo" if os.path.isdir("/kaggle/working") else os.path.abspath("repo_clone")
        if not os.path.isdir(repo):
            subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", a.branch,
                            "https://github.com/benski20/llm_training_exploration", repo], check=True)
        dd = os.path.join(repo, "data", "probe_train_built")
        cmd = [sys.executable, os.path.join(repo, "experiments", "probe_data.py"), "--out", dd]
        if a.sources:
            cmd += ["--sources", a.sources]
        log("building data:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=repo)
        a.data = sorted(glob.glob(os.path.join(dd, "*.jsonl")))
        root = repo
    sys.path.insert(0, root)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.omni import encode, judge_suffix, make_item, scalar_feats, sentence_spans, token_units
    if a.threads:
        torch.set_num_threads(a.threads)
    recs = load_records(a.data, a.shard, a.n_shards)
    if a.limit:
        recs = recs[:a.limit]
    log("shard", a.shard, "of", a.n_shards, ":", len(recs), "answers from", len(a.data), "files")
    os.makedirs(a.out, exist_ok=True)

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.reader)
    kw = {"device_map": "auto"} if cuda else {}
    if cuda:
        kw["torch_dtype"] = torch.float16
        if a.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.reader, **kw).eval()
    cfg = model.config
    nL, d, inter = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
    res_layers = sorted({max(1, min(nL, int(round(f * nL)))) for f in map(float, a.res_fracs.split(","))})
    neu_layers = sorted({max(0, min(nL - 1, int(round(f * nL)))) for f in map(float, a.neu_fracs.split(","))})
    mid = [res_layers[len(res_layers) // 2 - 1], res_layers[len(res_layers) // 2],
           res_layers[min(len(res_layers) - 1, len(res_layers) // 2 + 1)]]
    g = torch.Generator().manual_seed(a.seed)
    Pr = (torch.randn(d, a.proj, generator=g) / a.proj ** 0.5)
    Pn = (torch.randn(inter, a.proj, generator=g) / a.proj ** 0.5)
    dev = model.get_input_embeddings().weight.device
    suffix = tok(judge_suffix(tok), add_special_tokens=False)["input_ids"]
    first = lambda ws: [tok(w, add_special_tokens=False)["input_ids"][0] for w in ws]
    yes, no = first(["Yes", "yes", " Yes"]), first(["No", "no", " No"])
    meta = {"reader": a.reader, "load_4bit": a.load_4bit, "n_layers": nL, "hidden": d, "intermediate": inter,
            "res_layers": res_layers, "unit_res_layers": mid, "neu_layers": neu_layers, "proj": a.proj,
            "seed": a.seed, "max_tokens": a.max_tokens, "layout": "vt.omni.make_item(response, question, documents)"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))

    captured = {}

    def hook(L):
        def f(mod, inp):
            captured[L] = inp[0]
        return f
    hooks = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(hook(L)) for L in neu_layers]

    done = set()
    for p in glob.glob(os.path.join(a.out, f"shard{a.shard:03d}_*.npz")):      # resume
        done.update(np.load(p, allow_pickle=True)["ids"].tolist())
    todo = [r for r in recs if r["id"] not in done]
    log("already done", len(done), "to do", len(todo), "res layers", res_layers, "neuron layers", neu_layers)

    keys_a = ("ids", "label", "source", "task", "a_mean", "a_last", "a_judge", "a_neu", "a_sc", "j_logit", "n_tok",
              "truncated")
    keys_u = ("u_row", "u_idx", "u_lab", "u_res", "u_neu", "u_sc", "u_start", "u_end")

    def flush(buf, part):
        if not buf["ids"]:
            return
        out = {k: np.array(v) for k, v in buf.items()}
        np.savez(os.path.join(a.out, f"shard{a.shard:03d}_{part:04d}.npz"), **out)

    part = len(glob.glob(os.path.join(a.out, f"shard{a.shard:03d}_*.npz")))
    buf = {k: [] for k in keys_a + keys_u}
    t_start = time.time()
    for k, r in enumerate(todo):
        if time.time() - T0 > a.hours * 3600:
            log("time budget reached")
            break
        it = make_item(r["response"], r.get("question") or "", r.get("documents") or [])
        ids, r0, roff = encode(tok, it, a.max_tokens, suffix)
        ns = len(suffix)
        n = len(ids) - r0 - ns
        if n <= 0:
            continue
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True,
                      logits_to_keep=n + ns + 1)
        lg = o.logits[0]
        last = lg[-1].float().log_softmax(-1)
        j_logit = float(torch.logsumexp(last[yes], 0) - torch.logsumexp(last[no], 0))
        logp = lg[:n].float().log_softmax(-1)
        tgt = torch.tensor(ids[r0:r0 + n], device=logp.device)
        lp = logp.gather(-1, tgt[:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
        top2 = logp.topk(2, -1).values
        margin = torch.where(lp >= top2[:, 0] - 1e-6, top2[:, 0] - top2[:, 1], lp - top2[:, 0])
        lp, ent, margin = lp.cpu().numpy(), ent.cpu().numpy(), margin.cpu().numpy()

        def proj(x, P):
            return (x.float().cpu() @ P)
        R = {L: proj(o.hidden_states[L][0, r0:r0 + n], Pr) for L in res_layers}          # n x P each
        J = np.stack([proj(o.hidden_states[L][0, -1:], Pr)[0].numpy() for L in res_layers])
        Nn = {L: proj(captured[L][0, r0:r0 + n], Pn) for L in neu_layers}
        units = sentence_spans(r["response"])
        tu = token_units(roff[:n], units)
        spans = r.get("spans") or []
        row = len(buf["ids"])
        buf["ids"].append(r["id"])
        buf["label"].append(int(r["label"]))
        buf["source"].append(r.get("source", ""))
        buf["task"].append(r.get("task", ""))
        buf["a_mean"].append(np.stack([R[L].mean(0).numpy() for L in res_layers]).astype(np.float16))
        buf["a_last"].append(np.stack([R[L][-1].numpy() for L in res_layers]).astype(np.float16))
        buf["a_judge"].append(J.astype(np.float16))
        buf["a_neu"].append(np.stack([Nn[L].mean(0).numpy() for L in neu_layers]).astype(np.float16))
        buf["a_sc"].append(scalar_feats(lp, ent, margin))
        buf["j_logit"].append(j_logit)
        buf["n_tok"].append(len(ids))
        buf["truncated"].append(int(n < len(tok(r["response"], add_special_tokens=False)["input_ids"])))
        for ui, (s, e) in enumerate(units):
            m = torch.from_numpy(tu == ui)
            if not bool(m.any()):
                continue
            lab = (1 if any(s < b and a_ < e for a_, b in spans) else 0) if r.get("has_spans") else -1
            if not r.get("has_spans") and int(r["label"]) == 0:
                lab = 0                                              # a faithful answer: every sentence is fine
            buf["u_row"].append(row)
            buf["u_idx"].append(ui)
            buf["u_lab"].append(lab)
            buf["u_start"].append(s)
            buf["u_end"].append(e)
            buf["u_res"].append(np.stack([R[L][m].mean(0).numpy() for L in mid]).astype(np.float16))
            buf["u_neu"].append(np.stack([Nn[L][m].mean(0).numpy() for L in neu_layers]).astype(np.float16))
            mm = m.numpy()
            buf["u_sc"].append(scalar_feats(lp[mm], ent[mm], margin[mm]))
        captured.clear()
        if len(buf["ids"]) >= 500:
            flush(buf, part)
            part += 1
            buf = {k_: [] for k_ in keys_a + keys_u}
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t_start)
            log(f"{k + 1}/{len(todo)}  {rate * 3600:.0f} answers/h  ETA {(len(todo) - k - 1) / rate / 3600:.1f} h")
    flush(buf, part)
    for h in hooks:
        h.remove()
    log("done")


if __name__ == "__main__":
    main()
