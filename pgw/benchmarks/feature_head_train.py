#!/usr/bin/env python3
"""Train a target-feature draft head (``pgw.engine.generate.FeatureHeadDrafter``) on a real target's features.

One GPU job (``deploy/modal/head_train.py`` runs it on Modal; ``--device cpu`` with a small target is the local
smoke test). Steps:

1. **Data.** First-turn user prompts of UltraChat 200k (``HuggingFaceH4/ultrachat_200k``, MIT licence;
   ``train_sft`` shard 0 for training, ``test_sft`` for the held-out evaluation, so prompts never overlap).
   (a) The target answers ``--n-gen`` training prompts greedily (bf16, batched HF ``generate``): the target's
   own distribution, what the head sees at inference. (b) ``--n-tf`` other training conversations are taken
   with the dataset's own first reply (prefill only, cheap). (c) ``--n-eval`` held-out prompts are answered
   greedily. One teacher-forced pass over every ``prompt + answer`` records, per position ``t``, the final
   hidden state ``f_t`` (after the final norm: what the gateway holds in plaintext), the next token
   ``x_{t+1}`` and the target's greedy prediction at ``t + 1`` (its next-next token, the head's label).
2. **Training.** The head ``z = W_in [f ; E[x]] + b_in; z = z + W_2 silu(W_1 z + b_1) + b_2;
   f' = W_out z; logits_V = (H_V W_out) z`` (``E`` the target's input embedding, ``H_V`` the target's LM-head
   rows of the reduced vocabulary ``V``: the most frequent label tokens) is unrolled ``--unroll`` steps on
   its own predicted features as at inference; loss: cross-entropy to the target's greedy token plus a
   feature regression; AdamW with weight decay and feature noise; 3% of the training sequences held out to
   pick the epoch. Several ``--configs`` (``d:V``) are trained on the same data.
3. **Export** (``FeatureHeadDrafter.from_file``): int8 per-row weights + scales, float biases, the vocabulary
   and ``U = H_V W_out`` (int8 per row): the gateway never needs the target's LM head.
4. **Evaluation** of the exported (int8-dequantized) head on the held-out answers: per pass start, the
   head unrolled from the true feature with the true tokens gives the rank of the target's token at each
   depth (``ranks2d``, the ``tree_spec_sim.py --ranks`` format: exact replay for greedy decoding);
   alpha = P(rank 0 at depth 1).

Outputs in ``--out``: ``head_d{d}_v{V}.pt`` (loadable head), ``ranks_d{d}_v{V}.json``, ``summary.json``,
``eval_features.pt`` (held-out features, for local re-evaluation), optionally ``train_features.pt``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---- data ------------------------------------------------------------------------------------------------
def load_conversations(split_file: str, n: int, seed: int, skip: int = 0) -> list[list[dict]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceH4/ultrachat_200k", split_file, repo_type="dataset")
    tab = pq.read_table(path, columns=["messages"])
    msgs = tab.column("messages").to_pylist()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(msgs), generator=g).tolist()
    out = []
    for i in order[skip:]:
        m = msgs[i]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            out.append(m)
        if len(out) >= n:
            break
    return out


def prompt_ids(tok, text: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


KV_TOKENS = 64000  # generation batch x (prompt + max_new): ~3.6 GB of 7B KV cache in bf16
PREFILL_TOKENS = 16384  # generation batch x prompt


@torch.no_grad()
def generate(model, tok, prompts: list[list[int]], max_new: int, batch: int, dev: str, deadline: float,
             label: str) -> list[tuple[list[int], list[int]]]:
    """Greedy answers (until an end token or ``max_new``), batched with left padding, longest-first."""
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: dict[int, list[int]] = {}
    ntok = 0
    b0 = 0
    shrink = 1
    while b0 < len(order):
        if time.time() > deadline:
            log(f"{label}: deadline reached after {len(out)} answers")
            break
        L = len(prompts[order[b0]])
        # memory: KV ~ batch x (L + max_new) tokens, prefill activations ~ batch x L tokens
        nb = max(1, min(batch, KV_TOKENS // (L + max_new), PREFILL_TOKENS // L) // shrink)
        idx = order[b0 : b0 + nb]
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i])
            am[r, L - len(prompts[i]):] = 1
        try:
            g = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new,
                               do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)
        except torch.OutOfMemoryError:
            shrink *= 2
            log(f"{label}: out of memory at batch {nb}; retrying with 1/{shrink}")
            torch.cuda.empty_cache()
            if nb == 1:
                raise
            continue
        b0 += len(idx)
        g = g[:, L:].tolist()
        for r, i in enumerate(idx):
            a = []
            for t in g[r]:
                a.append(int(t))
                if t in eos:
                    break
            out[i] = a
            ntok += len(a)
        log(f"{label}: {len(out)}/{len(prompts)} answers, {ntok} tokens (batch {len(idx)})")
    return [(prompts[i], out[i]) for i in sorted(out)]


@torch.no_grad()
def features(model, seqs: list[tuple[list[int], list[int]]], start_back: int, dev: str, max_tokens: int,
             deadline: float, label: str) -> list[dict]:
    """Teacher-forced pass: per sequence ``h`` (final-norm hidden state, bf16, positions from
    ``len(prompt) - start_back``), ``ids`` (all tokens), ``am`` (target argmax at those positions), ``s0``."""
    body = model.model
    head_w = model.get_output_embeddings().weight
    out = []
    order = sorted(range(len(seqs)), key=lambda i: -(len(seqs[i][0]) + len(seqs[i][1])))
    b0 = 0
    while b0 < len(order):
        if time.time() > deadline:
            log(f"{label}: deadline reached after {len(out)} sequences")
            break
        L = len(seqs[order[b0]][0]) + len(seqs[order[b0]][1])
        nb = max(1, max_tokens // L)
        idx = order[b0 : b0 + nb]
        b0 += nb
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        msk = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            s = seqs[i][0] + seqs[i][1]
            ids[r, : len(s)] = torch.tensor(s)
            msk[r, : len(s)] = 1
        h = body(input_ids=ids.to(dev), attention_mask=msk.to(dev)).last_hidden_state
        for r, i in enumerate(idx):
            p, a = seqs[i]
            s0 = max(0, len(p) - start_back)
            hr = h[r, s0 : len(p) + len(a)]
            am = torch.cat([(hr[c : c + 1024] @ head_w.T).argmax(-1) for c in range(0, len(hr), 1024)])
            out.append({"h": hr.to(torch.bfloat16).cpu(), "ids": p + a, "am": am.cpu(), "s0": s0, "plen": len(p)})
    log(f"{label}: {len(out)} sequences, {sum(len(x['h']) for x in out)} feature rows")
    return out


# ---- head ------------------------------------------------------------------------------------------------
class Head(torch.nn.Module):
    """The engine's head with fixed input/output scales (``fs`` per feature dimension, ``es`` for the token
    embedding) that :func:`export` folds into ``W_in`` and ``W_out``."""

    def __init__(self, hidden: int, d: int, fs: torch.Tensor, es: float):
        super().__init__()
        self.w_in = torch.nn.Linear(2 * hidden, d)
        self.w1 = torch.nn.Linear(d, 2 * d)
        self.w2 = torch.nn.Linear(2 * d, d)
        self.w_out = torch.nn.Linear(d, hidden, bias=False)
        self.register_buffer("in_scale", torch.cat([1.0 / fs, torch.full_like(fs, 1.0 / es)]))
        self.register_buffer("fs", fs.clone())

    def w_in_eff(self) -> torch.Tensor:
        return self.w_in.weight * self.in_scale[None, :]

    def w_out_eff(self) -> torch.Tensor:
        return self.w_out.weight * self.fs[:, None]

    def step(self, f, e, hv):
        z = self.w_in(torch.cat([f, e], dim=-1) * self.in_scale)
        z = z + self.w2(F.silu(self.w1(z)))
        return self.w_out(z) * self.fs, (z @ (hv @ self.w_out_eff()).T)


def q8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.detach().float().cpu()
    sc = w.abs().amax(dim=1) / 127.0
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    return torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8), sc


def export(head: Head, hv: torch.Tensor, vocab: list[int], target: str, extra: dict,
           keep_float: bool = True) -> dict:
    """The FeatureHeadDrafter file (format v2): int8 per-row weights + float32 scales, float32 biases, the
    per-input-column scales ``in_scale`` and ``U = H_V W_out``.

    ``W_in`` is stored in the trained (normalized) input space, as its feature half and token half, each
    quantized per row, with the input scales kept apart (``in_scale.f`` = 1 / feature std per dimension,
    ``in_scale.e`` = 1 / embedding rms): the loader folds them for float products and scales the input for
    the native int8 kernel. Folding them into ``W_in`` before quantizing (format v1) is the 7B export bug:
    the target's final-norm features have outlier dimensions (std up to ~100x the median), so the folded
    columns span 100-1000x within a row and one per-row int8 scale rounds the outlier columns' weights to
    0 or +-1 (the dimensions that carry most of the feature); with one scale over both halves (the first
    export) the whole feature half rounded to 0. The float weights are kept too (``float``; not read by
    the gateway) unless ``keep_float`` is false."""
    hidden = int(hv.shape[1])
    w_in = head.w_in.weight  # (d, 2H) in the normalized input space: x_norm = [f / fs ; e / es]
    ws = {"w_in_f": w_in[:, :hidden], "w_in_e": w_in[:, hidden:], "w1": head.w1.weight, "w2": head.w2.weight,
          "w_out": head.w_out_eff()}
    q = {k: q8(v) for k, v in ws.items()}
    w_out_dq = q["w_out"][0].float() * q["w_out"][1][:, None]
    u = hv.detach().float().cpu() @ w_out_dq  # (V, d)
    q["u"] = q8(u)
    ins = head.in_scale.detach().float().cpu()
    blob = {"format": "pgw-feature-head-v2", "target": target, "hidden": hidden,
            "d": int(head.w_in.weight.shape[0]), "vocab": [int(t) for t in vocab],
            "int8": {k: {"w8": v[0], "scale": v[1]} for k, v in q.items()},
            "in_scale": {"f": ins[:hidden].contiguous(), "e": ins[hidden:].contiguous()},
            "bias": {"b_in": head.w_in.bias.detach().float().cpu(), "b1": head.w1.bias.detach().float().cpu(),
                     "b2": head.w2.bias.detach().float().cpu()},
            **extra}
    if keep_float:
        w_eff = head.w_in_eff()
        fl = {"w_in_f": w_eff[:, :hidden], "w_in_e": w_eff[:, hidden:], "w1": head.w1.weight,
              "w2": head.w2.weight, "w_out": head.w_out_eff(), "b_in": head.w_in.bias}
        blob["float"] = {k: v.detach().float().cpu() for k, v in fl.items()}
    return blob


def dequant_head(blob: dict) -> dict[str, torch.Tensor]:
    """Float weights of a head file as the gateway's float kernel uses them (v1 and v2 layouts)."""
    w = {k: v["w8"].float() * v["scale"][:, None] for k, v in blob["int8"].items()}
    if "w_in_f" in w:
        wf, we = w.pop("w_in_f"), w.pop("w_in_e")
        if "in_scale" in blob:
            wf = wf * blob["in_scale"]["f"].float()[None, :]
            we = we * blob["in_scale"]["e"].float()[None, :]
        w["w_in"] = torch.cat([wf, we], dim=1)
    w.update(blob["bias"])
    return w


def train_head(feat: torch.Tensor, ids: torch.Tensor, am: torch.Tensor, win_tr: torch.Tensor,
               win_va: torch.Tensor, emb: torch.Tensor, hv: torch.Tensor, vid: torch.Tensor, d: int, args,
               deadline: float, label: str) -> tuple[Head, dict]:
    dev = feat.device
    hidden = feat.shape[1]
    torch.manual_seed(0)
    U = args.unroll
    sample = feat[win_tr[: 50000].to(dev)].float()
    fs = sample.std(0).clamp(min=float(sample.std(0).median()) * 0.1)
    es = float(emb[ids[win_tr[: 50000].to(dev)]].float().pow(2).mean().sqrt())
    head = Head(hidden, d, fs, es).to(dev)
    bs = args.batch
    steps_per_ep = (len(win_tr) + bs - 1) // bs
    total = max(40, args.epochs * steps_per_ep)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    labv = vid[am]  # reduced-vocab labels (-1: outside V)
    best, best_state, hist = -1.0, None, []
    use_amp = dev.type == "cuda"

    def run(idx, train: bool):
        f = feat[idx].float()
        if train and args.noise > 0:
            f = f + args.noise * head.fs * torch.randn_like(f)
        loss = torch.zeros((), device=dev)
        hits = []
        for k in range(U):
            j = idx + 1 + k
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                fp, lg = head.step(f, emb[ids[j]].float(), hv)
            fp, lg = fp.float(), lg.float()
            lab = labv[j]
            ok = lab >= 0
            loss = loss + F.cross_entropy(lg[ok], lab[ok]) * (0.8 ** k)
            loss = loss + args.reg * F.smooth_l1_loss(fp / head.fs, feat[j].float() / head.fs)
            hits.append(((lg.argmax(-1) == lab) & ok).float().sum() / max(1, len(idx)))
            f = fp
        return loss, hits

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        head.train()
        perm = win_tr[torch.randperm(len(win_tr))].to(dev)
        tl, nst = 0.0, 0
        for b in range(steps_per_ep):
            idx = perm[b * bs : (b + 1) * bs]
            loss, _ = run(idx, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step < total - 1:
                sched.step()
            step += 1
            tl += float(loss.detach())
            nst += 1
            if time.time() > deadline:
                break
        head.eval()
        with torch.no_grad():
            acc = torch.zeros(U)
            n = 0
            for v0 in range(0, len(win_va), 4096):
                idx = win_va[v0 : v0 + 4096].to(dev)
                _, hits = run(idx, False)
                acc += torch.tensor([float(h) * len(idx) for h in hits])
                n += len(idx)
            acc = (acc / max(n, 1)).tolist()
            tr_idx = win_tr[: len(win_va)].to(dev)
            _, th = run(tr_idx, False)
        rec = {"epoch": ep, "loss": round(tl / max(1, nst), 4), "val_top1_by_depth": [round(x, 4) for x in acc],
               "train_top1_d1": round(float(th[0]), 4), "s": round(time.time() - t0, 1)}
        hist.append(rec)
        log(f"{label} {rec}")
        if acc[0] > best:
            best = acc[0]
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if time.time() > deadline:
            log(f"{label}: training deadline reached")
            break
    head.load_state_dict(best_state)
    return head, {"history": hist, "best_val_top1_d1": best}


@torch.no_grad()
def eval_ranks(w: dict[str, torch.Tensor], emb: torch.Tensor, u: torch.Tensor, vid: torch.Tensor,
               ev: list[dict], depth: int) -> tuple[list, dict]:
    """ranks2d per held-out answer (as draft_head_train.evaluate, with the exported weights)."""
    dev = emb.device
    w = {k: v.to(dev) for k, v in w.items()}
    u = u.to(dev)
    ranks2d = []
    n1 = hit1 = n1am = hit1am = 0
    for e in ev:
        s, plen, s0 = e["ids"], e["plen"], e["s0"]
        hf = e["h"].to(dev).float()
        am = e["am"]
        n = len(s)
        # all pass starts at once: batch over i, unroll over depth
        qs = list(range(plen - 1, n - 1))  # q = plen + i - 1
        cur = hf[torch.tensor([q - 1 - s0 for q in qs], device=dev)]
        alive = list(range(len(qs)))
        rr = [[] for _ in qs]
        for k in range(depth):
            alive = [a for a in alive if qs[a] + k + 1 < n]
            if not alive:
                break
            ai = torch.tensor(alive, device=dev)
            toks = torch.tensor([s[qs[a] + k] for a in alive], device=dev)
            x = torch.cat([cur[ai], emb[toks].float()], dim=1)
            z = x @ w["w_in"].T + w["b_in"]
            z = z + F.silu(z @ w["w1"].T + w["b1"]) @ w["w2"].T + w["b2"]
            fp = z @ w["w_out"].T
            lg = z @ u.T
            tgt = torch.tensor([s[qs[a] + k + 1] for a in alive], device=dev)
            v = vid[tgt]
            lv = lg.gather(1, v.clamp(min=0)[:, None])
            rk = (lg > lv).sum(1)
            rk = torch.where(v >= 0, rk, torch.full_like(rk, 99)).tolist()
            for a, r in zip(alive, rk, strict=True):
                rr[a].append(int(r))
            if k == 0:
                n1 += len(alive)
                hit1 += sum(1 for r in rk if r == 0)
                # also against the teacher-forced argmax (bf16 batch effects aside, the same token)
                tam = torch.tensor([int(am[qs[a] - s0]) for a in alive], device=dev)
                top = lg.argmax(-1)
                vam = vid[tam]
                hit1am += int(((top == vam.clamp(min=0)) & (vam >= 0)).sum())
                n1am += len(alive)
            nc = cur.clone()
            nc[ai] = fp
            cur = nc
        ranks2d.append(rr)
    return ranks2d, {"alpha_top1_d1": hit1 / max(n1, 1), "alpha_top1_d1_vs_tf_argmax": hit1am / max(n1am, 1),
                     "positions": n1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-gen", type=int, default=1600)
    ap.add_argument("--n-tf", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-prompt", type=int, default=400)
    ap.add_argument("--tf-max-len", type=int, default=1024)
    ap.add_argument("--gen-batch", type=int, default=96)
    ap.add_argument("--tf-tokens", type=int, default=12288, help="tokens per teacher-forced forward")
    ap.add_argument("--configs", default="384:8192,512:8192,512:16384")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.05, help="feature noise (x feature rms) at the root")
    ap.add_argument("--reg", type=float, default=0.5, help="feature regression weight")
    ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=40.0, help="wall budget; stages stop at their share")
    ap.add_argument("--save-train-features", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-features", help="directory of an earlier run (train_features.pt, eval_features.pt): "
                    "train and evaluate only, without the target model")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    budget = args.budget_min * 60
    commit = COMMIT_HOOK[0]
    if args.from_features:
        src = Path(args.from_features)
        log(f"loading features from {src}")
        ev = torch.load(src / "eval_features.pt", weights_only=True)
        d0 = torch.load(src / "train_features.pt", weights_only=True, mmap=True)
        feat = d0["feat"].to(dev)
        ids, am = d0["ids"].to(dev), d0["am"].to(dev)
        win_tr, win_va = d0["win_tr"], d0["win_va"]
        del d0
        emb, lmh = load_embeddings(args.target, dev)
        cnt = Counter(dict(enumerate(torch.bincount(am.cpu(), minlength=emb.shape[0]).tolist())))
        log(f"training rows {int(feat.shape[0])}, windows train {len(win_tr)} val {len(win_va)}")
        summary: dict = {"target": args.target, "args": vars(args), "train_rows": int(feat.shape[0]),
                         "windows_train": len(win_tr), "windows_val": len(win_va), "eval_answers": len(ev),
                         "eval_tokens": sum(len(x["ids"]) - x["plen"] for x in ev), "configs": {}}
        train_configs(args, out, feat, ids, am, win_tr, win_va, emb, lmh, cnt, ev, summary, budget, commit)
        log("done")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(args.target)
    log("loading data")
    train_conv = load_conversations("data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
                                    args.n_gen + args.n_tf, seed=1)
    eval_conv = load_conversations("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", args.n_eval, seed=2)
    log(f"loading {args.target}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16, device_map=dev)
    except Exception as exc:
        log(f"device_map load failed ({exc!r}); loading on CPU then moving")
        model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(dev)
    model.eval()
    log("model loaded")
    gen_p = [prompt_ids(tok, c[0]["content"]) for c in train_conv[: args.n_gen]]
    gen_p = [p for p in gen_p if len(p) <= args.max_prompt]
    ev_p = [prompt_ids(tok, c[0]["content"]) for c in eval_conv]
    ev_p = [p for p in ev_p if len(p) <= args.max_prompt]
    tf_seqs = []
    for c in train_conv[args.n_gen :]:
        p = prompt_ids(tok, c[0]["content"])
        a = tok(c[1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) <= args.max_prompt:
            tf_seqs.append((p, a[: max(8, args.tf_max_len - len(p))]))
    # held-out answers first (they must exist), then training data
    ev_seqs = generate(model, tok, ev_p, args.max_new, args.gen_batch, dev, T0 + 0.15 * budget, "eval gen")
    ev = features(model, ev_seqs, 2, dev, args.tf_tokens, T0 + 0.9 * budget, "eval feat")
    torch.save(ev, out / "eval_features.pt")
    commit()
    gen_seqs = generate(model, tok, gen_p, args.max_new, args.gen_batch, dev, T0 + 0.42 * budget, "train gen")
    tr = features(model, gen_seqs, 1, dev, args.tf_tokens, T0 + 0.47 * budget, "train feat (target answers)")
    tr += features(model, tf_seqs, 1, dev, args.tf_tokens, T0 + 0.53 * budget, "train feat (dataset replies)")
    emb = model.get_input_embeddings().weight.detach().clone()  # (vocab, H) bf16
    lmh = model.get_output_embeddings().weight.detach().clone()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    # flat arrays; window starts t with t + unroll inside the same sequence
    cnt: Counter[int] = Counter()
    for x in tr:
        cnt.update(x["am"].tolist())
    feat = torch.cat([x["h"] for x in tr])
    for x in tr:
        x["h"] = feat[:0]  # free the per-sequence copies (lengths are kept below)
    feat = feat.to(dev)
    ids_l, am_l = [], []
    off = 0
    g = torch.Generator().manual_seed(3)
    is_val = (torch.rand(len(tr), generator=g) < 0.03).tolist()
    win_tr, win_va = [], []
    for si, x in enumerate(tr):
        T = len(x["am"])
        ids_l.append(torch.tensor(x["ids"][x["s0"] : x["s0"] + T]))
        am_l.append(x["am"])
        w = torch.arange(off, off + T - args.unroll - 1)
        (win_va if is_val[si] else win_tr).append(w)
        off += T
    ids = torch.cat(ids_l).to(dev)
    am = torch.cat(am_l).to(dev)
    win_tr = torch.cat(win_tr)
    win_va = torch.cat(win_va)
    n_tok = int(feat.shape[0])
    log(f"training rows {n_tok}, windows train {len(win_tr)} val {len(win_va)}; "
        f"gen answers {len(gen_seqs)} ({sum(len(a) for _, a in gen_seqs)} tok), dataset replies {len(tf_seqs)}")
    if args.save_train_features:
        def _save():
            torch.save({"feat": feat.cpu(), "ids": ids.cpu(), "am": am.cpu(), "win_tr": win_tr, "win_va": win_va},
                       out / "train_features.pt")
            commit()
        saver = threading.Thread(target=_save)
        saver.start()
    else:
        saver = None
    summary: dict = {"target": args.target, "args": vars(args), "train_rows": n_tok,
                     "windows_train": len(win_tr), "windows_val": len(win_va),
                     "gen_answers": len(gen_seqs), "gen_tokens": sum(len(a) for _, a in gen_seqs),
                     "dataset_replies": len(tf_seqs), "dataset_reply_tokens": sum(len(a) for _, a in tf_seqs),
                     "eval_answers": len(ev), "eval_tokens": sum(len(x["ids"]) - x["plen"] for x in ev),
                     "configs": {}}
    train_configs(args, out, feat, ids, am, win_tr, win_va, emb, lmh, cnt, ev, summary, budget, commit)
    if saver is not None:
        saver.join()
    log("done")


def load_embeddings(target: str, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's input embedding and LM head (bf16) from its safetensors shards, without the model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    try:
        idx = json.load(open(hf_hub_download(target, "model.safetensors.index.json")))["weight_map"]
    except Exception:  # one-file checkpoint
        idx = {"model.embed_tokens.weight": "model.safetensors"}
    out = []
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        name = name if name in idx else "model.embed_tokens.weight"  # tied
        with safe_open(hf_hub_download(target, idx[name]), "pt") as fh:
            out.append(fh.get_tensor(name).to(torch.bfloat16).to(dev))
    return out[0], out[1]


def train_configs(args, out: Path, feat, ids, am, win_tr, win_va, emb, lmh, cnt, ev, summary, budget, commit) -> None:
    dev = feat.device
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    for ci, (d, V) in enumerate(configs):
        label = f"d{d}_v{V}"
        try:
            vocab = [t for t, _ in cnt.most_common(V)]
            have = set(vocab)
            vocab += [t for t in range(emb.shape[0]) if t not in have][: V - len(vocab)]
            cover = sum(cnt[t] for t in vocab[:V]) / max(1, sum(cnt.values()))
            vid = torch.full((emb.shape[0],), -1, dtype=torch.long, device=dev)
            vid[torch.tensor(vocab, device=dev)] = torch.arange(len(vocab), device=dev)
            hv = lmh[torch.tensor(vocab, device=dev)].float()
            share = (T0 + 0.93 * budget - time.time()) / (len(configs) - ci) * 0.85
            head, info = train_head(feat, ids, am, win_tr, win_va, emb, hv, vid, d, args, time.time() + share, label)
            blob = export(head, hv, vocab, args.target, {"training": info})
            torch.save(blob, out / f"head_{label}.pt")
            w = dequant_head(blob)
            u = blob["int8"]["u"]["w8"].float() * blob["int8"]["u"]["scale"][:, None]
            ranks2d, al = eval_ranks(w, emb, u, vid, ev, args.depth)
            fw = dict(blob["float"])
            fw["w_in"] = torch.cat([fw.pop("w_in_f"), fw.pop("w_in_e")], dim=1)
            fw.update({k: v for k, v in blob["bias"].items() if k != "b_in"})
            _, al_f = eval_ranks(fw, emb, hv.float().cpu() @ fw["w_out"], vid, ev, 1)
            al["alpha_top1_d1_float_weights"] = al_f["alpha_top1_d1"]
            nw = (sum(int(v["w8"].numel()) for v in blob["int8"].values())
                  + sum(int(v.numel()) for v in blob["bias"].values()))
            rec = {"vocab_coverage_train_labels": cover, **al, "weights_per_step": nw, "training": info}
            summary["configs"][label] = rec
            json.dump({"ranks2d": ranks2d, "answers": [x["ids"][x["plen"]:] for x in ev],
                       "prompt_lens": [x["plen"] for x in ev], "weights_per_step": nw, "vocab": V, "d": d},
                      open(out / f"ranks_{label}.json", "w"))
            log(f"{label}: alpha {al['alpha_top1_d1']:.3f} (vs tf argmax {al['alpha_top1_d1_vs_tf_argmax']:.3f}), "
                f"coverage {cover:.3f}, {nw / 1e6:.2f} M weights")
        except Exception as exc:
            import traceback

            traceback.print_exc()
            summary["configs"][label] = {"error": repr(exc)}
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        commit()


COMMIT_HOOK = [lambda: None]

if __name__ == "__main__":
    main()
