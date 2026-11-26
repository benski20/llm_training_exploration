#!/usr/bin/env python3
"""Distil a target-feature draft head from the target's own answers to real chat prompts (one GPU job).

Runs on a Kaggle T4 / T4 x2 kernel (``benchmarks/kaggle/build.py`` job ``pgw-head-*``; ``--device cpu`` with the
0.5B target and tiny sizes is the local smoke test). Compared with ``feature_head_train.py`` (the first Modal run):
no teacher-forced pass (the features and the target's greedy labels are captured while the target generates, by
forward hooks on the final norm and the LM head), several times more data, fp16 on the T4, features kept in host
RAM as int8 (per-dimension scale, then per-row scale) and the v2 export (``feature_head_train.export``).

1. **Prompts** (licences at the pinned revisions in ``DATA``; all allow commercial use, none share-alike):
   first-turn user prompts of UltraChat 200k (``HuggingFaceH4/ultrachat_200k``, MIT), the root prompts of
   OpenAssistant OASST2 (``OpenAssistant/oasst2``, Apache-2.0; human-written, all languages) and the instructions
   of StarCoder2 Self-OSS-Instruct (``bigcode/self-oss-instruct-sc2-exec-filter-50k``, ODC-By 1.0; code). Held
   out before anything else: ``--n-eval-chat`` UltraChat ``test_sft`` prompts (open chat) and ``--n-eval-code``
   MBPP test-split problems (``google-research-datasets/mbpp``, CC BY 4.0; evaluation only, never trained on).
   Until 2026-11-26 the heads were trained on Stanford Alpaca (CC BY-NC 4.0, non-commercial) and Dolly 15k
   (CC BY-SA 3.0) as well; both are gone (``pgw/sdk/draft_heads/README.md``).
2. **Answers** by the target itself (distillation): batched HF ``generate``, greedy for most training prompts and
   sampled (``--sample-frac``, T = 0.7, top-p 0.9) for the rest, no repetition penalty. Per generated position:
   the final-norm hidden state (what the gateway holds), the token and the target's greedy (raw arg-max) token.
3. **Training** (``feature_head_train.Head``): unrolled ``--unroll`` steps on its own predicted features, CE to
   the target's greedy token + feature regression, feature noise, AdamW + one-cycle; 3% of the answers for
   validation (top-1 by depth), best epoch kept. Configs ``d:V`` are trained one per GPU in parallel.
4. **Export** v2 (int8, ``in_scale``) and **evaluation** of the exported head as the gateway loads it
   (``FeatureHeadDrafter`` float math): per held-out answer and pass start the rank of the target's token at
   depth 1..7 with the true tokens (``ranks2d``: exact replay for greedy decoding); tau for a chain of R - 1
   drafts at R = 2/3/4 via ``tree_spec_sim.replay``.

Outputs in ``--out``: ``head_<label>.pt`` (shipping file, no float copy), ``ranks_<label>_{chat,code}.json``,
``eval_<set>.json`` (held-out prompts and answers), ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import feature_head_train as fht  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


# ---- prompts ---------------------------------------------------------------------------------------------
#: Dataset repositories at fixed revisions (Hugging Face Hub): (repo, revision, licence).
DATA = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "MIT"),
    "oasst2": ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75", "Apache-2.0"),
    "selfoss": ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "356bb069eee815daa6e23e9a282eeefe1490ad44",
                "ODC-By-1.0"),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "CC-BY-4.0"),
}
#: What each dataset is used for (recorded in the head file and summary.json).
DATA_USE = {"ultrachat": "train (train_sft) + held-out chat eval (test_sft)", "oasst2": "train",
            "selfoss": "train (code)", "mbpp": "held-out code eval only (test split)"}
ULTRACHAT_FILES = {"test_sft": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                   "train_sft": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"}


def data_provenance() -> list[dict[str, str]]:
    return [{"repo": r, "revision": v, "license": lic, "use": DATA_USE[k]} for k, (r, v, lic) in DATA.items()]


def _file(name: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, rev, _ = DATA[name]
    return hf_hub_download(repo, path, repo_type="dataset", revision=rev)


def _ultrachat(split: str, n: int, seed: int) -> list[str]:
    import pyarrow.parquet as pq

    msgs = pq.read_table(_file("ultrachat", ULTRACHAT_FILES[split]), columns=["messages"]).column("messages")
    idx = list(range(len(msgs)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        m = msgs[i].as_py()
        if m and m[0]["role"] == "user" and m[0]["content"].strip():
            out.append(m[0]["content"])
        if len(out) >= n:
            break
    return out


def _oasst2() -> list[str]:
    """Root (first user turn) prompts of the OASST2 train split, all languages, not deleted, review passed."""
    import pyarrow.parquet as pq

    t = pq.read_table(_file("oasst2", "data/train-00000-of-00001-88ba0162028a73fc.parquet"),
                      columns=["parent_id", "role", "text", "deleted", "review_result"]).to_pylist()
    return [r["text"].strip() for r in t if r["parent_id"] is None and r["role"] == "prompter"
            and not r["deleted"] and r["review_result"] and r["text"].strip()]


def _selfoss() -> list[str]:
    import pyarrow.parquet as pq

    t = pq.read_table(_file("selfoss", "data/train-00000-of-00001.parquet"), columns=["instruction"]).to_pylist()
    return [r["instruction"].strip() for r in t if r["instruction"] and r["instruction"].strip()]


def _mbpp_code_eval(n_code: int) -> list[str]:
    """Held-out code prompts: a fixed sample of MBPP test-split problems, with the first test as the example."""
    import pyarrow.parquet as pq

    t = pq.read_table(_file("mbpp", "full/test-00000-of-00001.parquet"), columns=["text", "test_list"]).to_pylist()
    rows = [f"{r['text'].strip()} Your code should pass this test:\n\n{r['test_list'][0]}" for r in t]
    random.Random(5).shuffle(rows)
    return rows[:n_code]


def eval_prompts(n_chat: int, n_code: int) -> tuple[list[str], list[str]]:
    """The held-out evaluation prompts (open chat, code), as the distillation draws them."""
    return _ultrachat("test_sft", n_chat, seed=2), _mbpp_code_eval(n_code)


def load_prompts(args) -> tuple[list[str], list[str], list[str]]:
    """(train, eval chat, eval code). The eval sets come from splits / datasets never trained on."""
    rnd = random.Random(5)
    chat_eval, code_eval = eval_prompts(args.n_eval_chat, args.n_eval_code)
    n = args.n_train
    n_u, n_o = int(n * args.mix[0]), int(n * args.mix[1])
    n_c = n - n_u - n_o
    uc = _ultrachat("train_sft", n_u, seed=1)
    oa = _oasst2()
    rnd.shuffle(oa)
    co = _selfoss()
    rnd.shuffle(co)
    if len(oa) < n_o or len(co) < n_c:
        log(f"note: {len(oa)} OASST2 prompts for {n_o} asked, {len(co)} code prompts for {n_c} asked")
    train = uc + oa[:n_o] + co[:n_c]
    rnd.shuffle(train)
    return train, chat_eval, code_eval


def prompt_ids(tok, text: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


# ---- generation with captured features -------------------------------------------------------------------
class Capture:
    """Forward hooks: per forward call of ``generate`` the final-norm hidden state of the last position and the
    raw arg-max of the LM head (the target's greedy token)."""

    def __init__(self, model):
        self.feats: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.on = False
        body = model.model if hasattr(model, "model") else model.base_model
        self.h1 = body.norm.register_forward_hook(self._norm)
        self.h2 = model.get_output_embeddings().register_forward_hook(self._head)

    def _norm(self, mod, inp, out):
        if self.on:
            self.feats.append(out[:, -1].detach())

    def _head(self, mod, inp, out):
        if self.on:
            self.labels.append(out[:, -1].argmax(-1).detach())

    def start(self) -> None:
        self.feats, self.labels, self.on = [], [], True

    def stop(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.on = False
        f = torch.stack(self.feats, 0)  # (steps, B, H)
        g = torch.stack([x.to(f.device) for x in self.labels], 0)  # (steps, B)
        self.feats, self.labels = [], []
        return f, g


@torch.no_grad()
def generate(model, tok, cap: Capture, prompts: list[list[int]], *, max_new: int, batch: int, kv_tokens: int,
             prefill_tokens: int, sample: list[bool], deadline: float, label: str, sink,
             full=lambda: False) -> int:
    """Answers batched with left padding (longest prompts first); ``sink(prompt, answer, feats, labels)`` per
    answer (feats: (n, H) fp16 on the GPU, rows 0..n-1 = after the last prompt token, then after answer tokens
    0..n-2; labels: the target's greedy token at those rows). Returns the answers made."""
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    dev = model.get_input_embeddings().weight.device
    # random chunks (an early stop keeps a random sample), each sorted by length (little padding)
    perm = list(range(len(prompts)))
    random.Random(len(prompts)).shuffle(perm)
    ck = max(1, batch * 16)
    order = [i for c in range(0, len(perm), ck)
             for i in sorted(perm[c : c + ck], key=lambda i: (sample[i], -len(prompts[i])))]
    done = ntok = 0
    b0, shrink = 0, 1
    t_start = time.time()
    while b0 < len(order):
        if time.time() > deadline or full():
            log(f"{label}: {'deadline' if not full() else 'store full'} after {done} answers")
            break
        smp = sample[order[b0]]
        nb = max(1, min(batch, kv_tokens // (len(prompts[order[b0]]) + max_new),
                        prefill_tokens // len(prompts[order[b0]])) // shrink)
        idx = []
        for i in order[b0 : b0 + nb]:  # same decoding mode; a chunk boundary may raise the length: stop there
            if sample[i] != smp or (idx and len(prompts[i]) > len(prompts[idx[0]])):
                break
            idx.append(i)
        L = len(prompts[idx[0]])
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i])
            am[r, L - len(prompts[i]):] = 1
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=0) if smp else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        cap.start()
        try:
            g = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new,
                               repetition_penalty=1.0, pad_token_id=pad, **kw)
        except torch.OutOfMemoryError:
            cap.feats, cap.labels, cap.on = [], [], False
            shrink *= 2
            log(f"{label}: out of memory at batch {nb}; retrying with 1/{shrink}")
            torch.cuda.empty_cache()
            if nb == 1:
                raise
            continue
        f, lab = cap.stop()
        b0 += len(idx)
        g = g[:, L:].tolist()
        for r, i in enumerate(idx):
            a = []
            for t in g[r]:
                a.append(int(t))
                if t in eos:
                    break
            n = min(len(a), int(f.shape[0]))
            sink(prompts[i], a[:n], f[:n, r], lab[:n, r])
            done += 1
            ntok += n
        if done % (len(idx) * 5) < len(idx) or b0 >= len(order):
            el = time.time() - t_start
            log(f"{label}: {done}/{len(prompts)} answers, {ntok} tokens, {ntok / max(el, 1e-9):.0f} tok/s "
                f"(batch {len(idx)}, prompt {L}{', sampled' if smp else ''})")
    return done


class FeatureStore:
    """Training rows in host RAM: int8 features (per-dimension scale ``fs``, then a per-row scale), the token
    at the row and the target's greedy token at the row."""

    def __init__(self, cap_rows: int, hidden: int, fs: torch.Tensor):
        self.q = torch.empty((cap_rows, hidden), dtype=torch.int8)
        self.rs = torch.empty(cap_rows, dtype=torch.float32)
        self.ids = torch.empty(cap_rows, dtype=torch.long)
        self.am = torch.empty(cap_rows, dtype=torch.long)
        self.fs = fs.float().cpu()
        self.n = 0
        self.seqs: list[tuple[int, int]] = []  # (start row, rows)
        self.full = False
        self._fs_dev: dict[str, torch.Tensor] = {}

    def add(self, prompt: list[int], answer: list[int], f: torch.Tensor, lab: torch.Tensor) -> None:
        n = len(answer)
        if n < 4 or self.full:
            return
        if self.n + n > self.q.shape[0]:
            self.full = True
            return
        dev = f.device
        fs = self._fs_dev.setdefault(str(dev), self.fs.to(dev))
        x = f.float() / fs
        sc = (x.abs().amax(1) / 127.0).clamp(min=1e-8)
        q = torch.clamp(torch.round(x / sc[:, None]), -127, 127).to(torch.int8)
        s = slice(self.n, self.n + n)
        self.q[s] = q.cpu()
        self.rs[s] = sc.cpu()
        self.ids[s] = torch.tensor([prompt[-1]] + answer[: n - 1])
        self.am[s] = lab.cpu()
        self.seqs.append((self.n, n))
        self.n += n

    def gather(self, rows: torch.Tensor, dev: str) -> torch.Tensor:
        q = self.q.index_select(0, rows)
        rs = self.rs.index_select(0, rows)
        return q.to(dev, non_blocking=True).float() * rs.to(dev)[:, None] * self.fs.to(dev)[None, :]


# ---- training --------------------------------------------------------------------------------------------
def train_head(store: FeatureStore, win_tr: torch.Tensor, win_va: torch.Tensor, emb: torch.Tensor, hv: torch.Tensor,
               vid: torch.Tensor, d: int, args, deadline: float, label: str, dev: str):
    hidden = emb.shape[1]
    torch.manual_seed(0)
    U = args.unroll
    ids = store.ids
    am_all = store.am
    sample = store.gather(win_tr[:50000], dev)
    fs = sample.std(0).clamp(min=float(sample.std(0).median()) * 0.1)
    es = float(emb[ids[win_tr[:50000]]].float().pow(2).mean().sqrt())
    head = fht.Head(hidden, d, fs, es).to(dev)
    bs = args.batch
    steps_per_ep = (len(win_tr) + bs - 1) // bs
    total = max(40, args.epochs * steps_per_ep)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))
    use_amp = dev.startswith("cuda")
    emb_d = emb.to(dev)
    hv_d = hv.to(dev)
    vid_d = vid.to(dev)

    def hstep(f, e, uu):
        z = head.w_in(torch.cat([f, e], dim=-1) * head.in_scale)
        z = z + head.w2(F.silu(head.w1(z)))
        return head.w_out(z) * head.fs, z @ uu.T

    def batch_of(idx: torch.Tensor):
        rows = torch.cat([idx + k for k in range(U + 1)])  # (U + 1) * B rows: f_t, then the targets
        f = store.gather(rows, dev).view(U + 1, len(idx), hidden)
        toks = torch.stack([ids.index_select(0, idx + 1 + k) for k in range(U)]).to(dev)
        lab = vid_d[torch.stack([am_all.index_select(0, idx + 1 + k) for k in range(U)]).to(dev)]
        return f, toks, lab

    def run(f_all, toks, lab, train: bool):
        f = f_all[0]
        if train and args.noise > 0:
            f = f + args.noise * head.fs * torch.randn_like(f)
        loss = torch.zeros((), device=dev)
        hits = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            uu = hv_d @ head.w_out_eff()  # (V, d): once per batch, shared by the unrolled steps
        for k in range(U):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                fp, lg = hstep(f, emb_d[toks[k]].float(), uu)
            fp, lg = fp.float(), lg.float()
            ok = lab[k] >= 0
            loss = loss + F.cross_entropy(lg[ok], lab[k][ok]) * (0.8 ** k)
            loss = loss + args.reg * F.smooth_l1_loss(fp / head.fs, f_all[k + 1] / head.fs)
            hits.append(((lg.argmax(-1) == lab[k]) & ok).float().sum() / max(1, f.shape[0]))
            f = fp
        return loss, hits

    # prefetch batches on a helper thread (host gather + copy) while the GPU trains
    import queue

    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        head.train()
        perm = win_tr[torch.randperm(len(win_tr))]
        qq: queue.Queue = queue.Queue(maxsize=4)

        def producer(perm=perm, qq=qq):
            for b in range(steps_per_ep):
                qq.put(batch_of(perm[b * bs : (b + 1) * bs]))
            qq.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        tl, nst = 0.0, 0
        stop = False
        while True:
            item = qq.get()
            if item is None:
                break
            if stop:
                continue
            loss, _ = run(*item, True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step < total - 1:
                sched.step()
            step += 1
            tl += float(loss.detach())
            nst += 1
            if time.time() > deadline:
                stop = True
        th.join()
        head.eval()
        with torch.no_grad():
            acc = torch.zeros(U)
            n = 0
            for v0 in range(0, len(win_va), 4096):
                idx = win_va[v0 : v0 + 4096]
                _, hits = run(*batch_of(idx), False)
                acc += torch.tensor([float(h) * len(idx) for h in hits])
                n += len(idx)
            acc = (acc / max(n, 1)).tolist()
        rec = {"epoch": ep, "loss": round(tl / max(1, nst), 4), "val_top1_by_depth": [round(x, 4) for x in acc],
               "steps": nst, "s": round(time.time() - t0, 1)}
        hist.append(rec)
        log(f"{label} {rec}")
        if acc[0] > best:
            best = acc[0]
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if stop:
            log(f"{label}: training deadline reached")
            break
    head.load_state_dict(best_state)
    return head.cpu(), {"history": hist, "best_val_top1_d1": best}


# ---- evaluation ------------------------------------------------------------------------------------------
@torch.no_grad()
def eval_ranks(w: dict[str, torch.Tensor], emb: torch.Tensor, u: torch.Tensor, vid: torch.Tensor,
               ev: list[dict], depth: int, dev: str) -> tuple[list, dict]:
    """``ranks2d[a][i][k]``: rank of answer token ``i + k`` under the head's depth-``k`` draft in the pass whose
    pending token is answer token ``i - 1`` (true tokens, the head's own features after the root): the
    ``tree_spec_sim.replay`` format. Row 0 (pending = last prompt token) is not a decode pass: left empty."""
    w = {k: v.to(dev).float() for k, v in w.items()}
    u = u.to(dev).float()
    emb = emb.to(dev)
    vid = vid.to(dev)
    out = []
    n1 = hit1 = 0
    for e in ev:
        a = e["answer"]
        feats = e["feats"].to(dev).float()  # (n, H): row j = after (last prompt token, a[0], ...)[j]
        n = len(a)
        rr: list[list[int]] = [[] for _ in range(n)]
        starts = list(range(1, n))  # pending a[i-1], root feature row i-1 (it produced a[i-1])
        if starts:
            cur = feats[torch.tensor([i - 1 for i in starts], device=dev)]
            alive = list(range(len(starts)))
            for k in range(depth):
                alive = [x for x in alive if starts[x] + k < n]
                if not alive:
                    break
                ai = torch.tensor(alive, device=dev)
                toks = torch.tensor([a[starts[x] - 1 + k] for x in alive], device=dev)
                xin = torch.cat([cur[ai], emb[toks].float()], dim=1)
                z = xin @ w["w_in"].T + w["b_in"]
                z = z + F.silu(z @ w["w1"].T + w["b1"]) @ w["w2"].T + w["b2"]
                fp = z @ w["w_out"].T
                lg = z @ u.T
                tgt = torch.tensor([a[starts[x] + k] for x in alive], device=dev)
                v = vid[tgt]
                lv = lg.gather(1, v.clamp(min=0)[:, None])
                rk = torch.where(v >= 0, (lg > lv).sum(1), torch.full_like(v, 99)).tolist()
                for x, r in zip(alive, rk, strict=True):
                    rr[starts[x]].append(int(r))
                if k == 0:
                    n1 += len(alive)
                    hit1 += sum(1 for r in rk if r == 0)
                nc = cur.clone()
                nc[ai] = fp
                cur = nc
        out.append(rr)
    return out, {"alpha_d1": hit1 / max(n1, 1), "positions": n1}


def taus(ranks2d: list, prompt_lens: list[int]) -> dict:
    import tree_spec_sim as tss

    p = tss.depth_rank_probs(ranks2d)
    res = {"alpha_by_depth": [round(x[0], 4) for x in p[:6]]}
    for r in (2, 3, 4):
        res[f"tau_R{r}"] = round(tss.replay(ranks2d, tss.chain(r), None, None, prompt_lens)["tau"], 4)
    return res


# ---- main ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--mix", type=lambda s: [float(x) for x in s.split(",")], default=[0.75, 0.15, 0.10],
                    help="UltraChat, OASST2, Self-OSS-Instruct (code) shares of the training prompts")
    ap.add_argument("--n-eval-chat", type=int, default=200)
    ap.add_argument("--n-eval-code", type=int, default=120)
    ap.add_argument("--sample-frac", type=float, default=0.25)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--eval-max-new", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--kv-tokens", type=int, default=120000)
    ap.add_argument("--prefill-tokens", type=int, default=32768)
    ap.add_argument("--max-rows", type=int, default=8_000_000)
    ap.add_argument("--configs", default="256:16384")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=0.5)
    ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=160.0, help="wall budget of the whole job")
    ap.add_argument("--gen-share", type=float, default=0.6, help="share of the budget for training answers")
    ap.add_argument("--max-memory", default=None, help="per-GPU memory cap for the target, e.g. 12GiB")
    ap.add_argument("--compare-heads", nargs="*", default=[],
                    help="existing head files to evaluate on the same held-out answers (ranks_cmp<i>_<set>.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = args.budget_min * 60
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.revision)
    log("loading prompts")
    train_txt, chat_txt, code_txt = load_prompts(args)
    log(f"prompts: train {len(train_txt)}, eval chat {len(chat_txt)}, eval code {len(code_txt)}")
    log(f"loading {args.target}")
    ngpu = torch.cuda.device_count() if args.device == "cuda" else 0
    kw: dict = {"torch_dtype": dtype, "revision": args.revision}  # torch_dtype: older transformers too
    if ngpu > 1 and args.max_memory:
        kw.update(device_map="auto", max_memory={i: args.max_memory for i in range(ngpu)})
    else:
        kw.update(device_map=args.device if args.device != "cpu" else None)
    model = AutoModelForCausalLM.from_pretrained(args.target, **kw)
    model.eval()
    model.generation_config.repetition_penalty = 1.0
    log(f"model loaded ({getattr(model, 'hf_device_map', args.device)})")
    cap = Capture(model)
    hidden = int(model.config.hidden_size)

    def ids_of(txts: list[str]) -> list[list[int]]:
        return [p for p in (prompt_ids(tok, t) for t in txts) if len(p) <= args.max_prompt]

    gen_kw = dict(batch=args.gen_batch, kv_tokens=args.kv_tokens, prefill_tokens=args.prefill_tokens)
    evals: dict[str, list[dict]] = {}
    for name, txts in (("chat", chat_txt), ("code", code_txt)):
        ev: list[dict] = []
        pids = ids_of(txts)

        def sink(p, a, f, lab, ev=ev):
            ev.append({"prompt": p, "answer": a, "feats": f.to(torch.float16).cpu()})

        generate(model, tok, cap, pids, max_new=args.eval_max_new, sample=[False] * len(pids),
                 deadline=T0 + 0.12 * budget, label=f"eval {name}", sink=sink, **gen_kw)
        evals[name] = ev
        bad = sum(1 for e in ev if not torch.isfinite(e["feats"].float()).all())
        log(f"eval {name}: {len(ev)} answers, {bad} with non-finite features; sample: "
            f"{tok.decode(ev[0]['answer'][:60])!r}" if ev else f"eval {name}: no answers")
        if bad:
            raise RuntimeError(f"non-finite target features in {bad} {name} answers ({args.dtype} overflow?)")
        json.dump({"prompts": [e["prompt"] for e in ev], "answers": [e["answer"] for e in ev]},
                  open(out / f"eval_{name}.json", "w"))
    # per-dimension feature scale for the int8 store, from the held-out answers' features
    allf = torch.cat([e["feats"].float() for e in evals["chat"] + evals["code"]])
    fs = allf.std(0).clamp(min=float(allf.std(0).median()) * 0.05)
    store = FeatureStore(args.max_rows, hidden, fs)
    pids = ids_of(train_txt)
    rnd = random.Random(9)
    smp = [rnd.random() < args.sample_frac for _ in pids]
    generate(model, tok, cap, pids, max_new=args.max_new, sample=smp, deadline=T0 + args.gen_share * budget,
             label="train", sink=store.add, full=lambda: store.full, **gen_kw)
    log(f"training rows {store.n} from {len(store.seqs)} answers (store full: {store.full})")
    emb = model.get_input_embeddings().weight.detach().to("cpu", torch.float16)
    lmh = model.get_output_embeddings().weight.detach().to("cpu", torch.float16)
    cap.h1.remove()
    cap.h2.remove()
    del model
    torch.cuda.empty_cache() if ngpu else None
    # windows: start t with t + unroll + 1 inside the same answer
    g = torch.Generator().manual_seed(3)
    is_val = (torch.rand(len(store.seqs), generator=g) < 0.03).tolist()
    wtr, wva = [], []
    for si, (s0, n) in enumerate(store.seqs):
        w = torch.arange(s0, s0 + n - args.unroll - 1)
        (wva if is_val[si] else wtr).append(w)
    win_tr, win_va = torch.cat(wtr), torch.cat(wva)
    cnt = torch.bincount(store.am[: store.n], minlength=emb.shape[0])
    summary: dict = {"target": args.target, "revision": args.revision, "args": vars(args), "data": data_provenance(),
                     "train_rows": store.n,
                     "train_answers": len(store.seqs), "windows_train": len(win_tr), "windows_val": len(win_va),
                     "eval": {k: {"answers": len(v), "tokens": sum(len(e["answer"]) for e in v)}
                              for k, v in evals.items()}, "configs": {}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    devs = [f"cuda:{i}" for i in range(max(1, ngpu))] if ngpu else ["cpu"]
    deadline = T0 + 0.93 * budget - 6 * 60  # leave time for export and evaluation

    def one(ci: int, d: int, V: int) -> None:
        label = f"d{d}_v{V}"
        dev = devs[ci % len(devs)]
        try:
            vocab = cnt.argsort(descending=True)[:V].tolist()
            cover = float(cnt[torch.tensor(vocab)].sum() / max(1, int(cnt.sum())))
            vid = torch.full((emb.shape[0],), -1, dtype=torch.long)
            vid[torch.tensor(vocab)] = torch.arange(V)
            hv = lmh[torch.tensor(vocab)].float()
            # configs sharing a GPU run one after the other within the budget
            same = [c for j, c in enumerate(configs) if j % len(devs) == ci % len(devs)]
            pos = [j for j in range(len(configs)) if j % len(devs) == ci % len(devs)].index(ci)
            share = (deadline - time.time()) / (len(same) - pos)
            head, info = train_head(store, win_tr, win_va, emb, hv, vid, d, args, time.time() + share, label, dev)
            blob = fht.export(head, hv, vocab, args.target,
                              {"revision": args.revision, "training": info, "vocab_coverage": cover,
                               "training_data": data_provenance()},
                              keep_float=False)
            torch.save(blob, out / f"head_{label}.pt")
            w = fht.dequant_head(blob)
            u = blob["int8"]["u"]["w8"].float() * blob["int8"]["u"]["scale"][:, None]
            rec = {"vocab_coverage_train_labels": cover, "training": info,
                   "file_bytes": os.path.getsize(out / f"head_{label}.pt"),
                   "weights_per_step": sum(int(v["w8"].numel()) for v in blob["int8"].values())}
            for name, ev in evals.items():
                rk, al = eval_ranks(w, emb, u, vid, ev, args.depth, dev)
                plens = [len(e["prompt"]) for e in ev]
                rec[name] = {**al, **taus(rk, plens)}
                json.dump({"ranks2d": rk, "answers": [e["answer"] for e in ev], "prompt_lens": plens,
                           "weights_per_step": rec["weights_per_step"], "vocab": V, "d": d},
                          open(out / f"ranks_{label}_{name}.json", "w"))
            summary["configs"][label] = rec
            log(f"{label}: {json.dumps({k: rec[k] for k in ('chat', 'code')})}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            summary["configs"][label] = {"error": repr(exc)}

    threads = []
    for gi in range(len(devs)):
        mine = [(ci, d, V) for ci, (d, V) in enumerate(configs) if ci % len(devs) == gi]

        def lane(mine=mine):
            for c in mine:
                one(*c)

        threads.append(threading.Thread(target=lane))
        threads[-1].start()
    for t in threads:
        t.join()
    # earlier heads on the same held-out answers (the acceptance comparison)
    for ci, path in enumerate(args.compare_heads):
        label = f"cmp{ci}"
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            vocab = blob["vocab"]
            vid = torch.full((emb.shape[0],), -1, dtype=torch.long)
            vid[torch.tensor(vocab)] = torch.arange(len(vocab))
            w = fht.dequant_head(blob)
            u = blob["int8"]["u"]["w8"].float() * blob["int8"]["u"]["scale"][:, None]
            rec = {"file": os.path.basename(path), "d": int(blob["d"]), "vocab": len(vocab)}
            for name, ev in evals.items():
                rk, al = eval_ranks(w, emb, u, vid, ev, args.depth, devs[0])
                plens = [len(e["prompt"]) for e in ev]
                rec[name] = {**al, **taus(rk, plens)}
                json.dump({"ranks2d": rk, "answers": [e["answer"] for e in ev], "prompt_lens": plens,
                           "vocab": len(vocab), "d": int(blob["d"]), "file": rec["file"]},
                          open(out / f"ranks_{label}_{name}.json", "w"))
            summary.setdefault("compare", {})[label] = rec
            log(f"{label} ({rec['file']}): {json.dumps({k: rec[k] for k in ('chat', 'code')})}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            summary.setdefault("compare", {})[label] = {"file": path, "error": repr(exc)}
    summary["minutes"] = round((time.time() - T0) / 60, 1)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
