#!/usr/bin/env python3
"""Prototype target-feature draft head (EAGLE-style, a few MB per step) for strict tree speculation.

The gateway computes the target's residual stream in plaintext between round trips (only the linears are
remote), so after a verify pass it holds the final hidden state ``f`` (after the final norm) of the node that
produced the pending token ``x``. The head drafts from ``(f, x)``, autoregressively over its own predicted
features (``pgw.engine.generate.FeatureHeadDrafter`` runs the saved weights in the engine)::

    z = W_in [f ; E[x]] + b_in;  z = z + W_2 silu(W_1 z + b_1) + b_2;  f' = W_out z;  logits_V = (E_V W_out) z

``E`` is the target's tied embedding / LM head (public, on the gateway for the embedding lookup), ``V`` the
most frequent tokens. A step reads W_in, W_1, W_2, W_out and ``U = E_V W_out``: H = 1536, d = 384,
V = 8192 is ~5.5 M weights (5.5 MB int8), versus ~490 MB per step for a 0.5B draft model.

Data (``--data``, written by a teacher-forced pass of the target: ``eval_XXXX.npz`` for the exact answers of
``--answers`` and ``train_XXXX.npz`` for other text; each with ``h`` (T, H) fp16 final-norm features, ``ids``,
``am`` the target's argmax): self-distillation on the target's own predictions (features are the target's,
labels its argmax; on the exact answers the next answer token), unrolled ``--unroll`` steps on the head's own
predicted features as at inference. ``--folds 2``: the 24 chat answers are split in two halves by prompt; each
half is evaluated with a head trained on the other half plus the ``train_*`` text (prompts never overlap).

Evaluation: for every pass start on an exact answer the head is unrolled from the true feature with the true
tokens (the answer path), giving the rank of the target's token at each depth (``ranks2d``), which
``tree_spec_sim.py --ranks`` turns into tokens per pass for chains and trees exactly (greedy).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


class Head(torch.nn.Module):
    def __init__(self, hidden: int, d: int, emb_v: torch.Tensor):
        super().__init__()
        self.w_in = torch.nn.Linear(2 * hidden, d)
        self.w1 = torch.nn.Linear(d, 2 * d)
        self.w2 = torch.nn.Linear(2 * d, d)
        self.w_out = torch.nn.Linear(d, hidden, bias=False)
        self.register_buffer("emb_v", emb_v)  # (V, H): reduced LM head rows (frozen, public)

    def step(self, f: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.w_in(torch.cat([f, e], dim=-1))
        z = z + self.w2(F.silu(self.w1(z)))
        fp = self.w_out(z)
        return fp, fp @ self.emb_v.T


def weights_per_step(hidden: int, d: int, v: int) -> int:
    return 2 * hidden * d + d + 2 * d * d * 2 + 3 * d + d * hidden + v * d


def windows(items: list[tuple[np.ndarray, list[int], list[int], int]], unroll: int):
    """(f_t, tokens x_{t+1..t+U}, labels y_{t+1..t+U}, features f_{t+1..t+U}) for t >= start."""
    xs_f, xs_tok, xs_lab, xs_ft = [], [], [], []
    for h, s, lab, start in items:
        f = torch.from_numpy(h.astype(np.float32))
        for t in range(max(start, 0), len(s) - unroll - 1):
            xs_f.append(f[t])
            xs_tok.append(torch.tensor(s[t + 1 : t + 1 + unroll]))
            xs_lab.append(torch.tensor(lab[t + 1 : t + 1 + unroll]))
            xs_ft.append(f[t + 1 : t + 1 + unroll])
    return torch.stack(xs_f), torch.stack(xs_tok), torch.stack(xs_lab), torch.stack(xs_ft)


def train(head: Head, emb: torch.Tensor, data, vid: torch.Tensor, epochs: int, unroll: int, wd: float = 0.0) -> None:
    F0, TOK, LAB, FT = data
    LAB = vid[LAB]
    fscale = float(F0.pow(2).mean())
    nb = (len(F0) + 255) // 256
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * nb, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(F0))
        tot, acc, cnt = 0.0, 0.0, 0
        for i in range(0, len(F0), 256):
            idx = perm[i : i + 256]
            f = F0[idx]
            loss = torch.zeros(())
            for k in range(unroll):
                fp, lg = head.step(f, emb[TOK[idx, k]])
                lab = LAB[idx, k]
                ok = lab >= 0
                if ok.any():
                    loss = loss + F.cross_entropy(lg[ok], lab[ok])
                    if k == 0:
                        acc += float((lg[ok].argmax(-1) == lab[ok]).float().sum())
                        cnt += int(ok.sum())
                loss = loss + 0.2 * F.mse_loss(fp, FT[idx, k]) / fscale
                f = fp
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        print(f"  epoch {ep} loss {tot / nb:.3f} train top1 (step 1) {acc / max(cnt, 1):.3f} "
              f"{time.time() - t0:.0f}s", flush=True)


def evaluate(head: Head, emb: torch.Tensor, vid: torch.Tensor, h: np.ndarray, s: list[int], plen: int,
             depth: int) -> list[list[int]]:
    """ranks[i][k]: rank of answer token i + k under the head unrolled k + 1 steps from the pass at i."""
    f = torch.from_numpy(h.astype(np.float32))
    n = len(s)
    rows = []
    with torch.no_grad():
        for i in range(n - plen):
            q = plen + i - 1  # the pending token s_q of the pass whose root emits answer token i
            cur = f[q - 1]
            rr = []
            for k in range(depth):  # step k feeds s_{q+k} and predicts s_{q+k+1} = answer token i + k
                pos = q + k
                if pos + 1 >= n:
                    break
                fp, lg = head.step(cur[None], emb[s[pos]][None])
                v = int(vid[s[pos + 1]])
                rr.append(int((lg[0] > lg[0, v]).sum()) if v >= 0 else 99)
                cur = fp[0]
            rows.append(rr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--answers", required=True, help="exact target answers (tokens_q15.json)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    with safe_open(hf_hub_download(args.target, "model.safetensors"), "pt") as fh:
        emb = fh.get_tensor("model.embed_tokens.weight").to(torch.float32)
    ans = json.load(open(args.answers))
    plens = [len(p) for p in ans["prompts"]]
    ev = [np.load(f) for f in sorted(glob.glob(f"{args.data}/eval_*.npz"))]
    tr = [np.load(f) for f in sorted(glob.glob(f"{args.data}/train_*.npz"))]
    # other models' answers to the same prompts (x05_i, x3_i: prompt i), training only in the fold that does
    # not evaluate prompt i; labels are the target's argmax on that text
    others = {}
    for f in sorted(glob.glob(f"{args.data}/x*_*.npz")):
        others.setdefault(int(f.rsplit("_", 1)[1].split(".")[0]), []).append(np.load(f))
    n_ev = len(ev)
    print(f"{n_ev} answers with features, {len(tr)} extra text chunks "
          f"({sum(len(z['ids']) for z in tr)} tokens)", flush=True)
    text_items = [(z["h"], z["ids"].tolist(), z["am"].tolist(), 0) for z in tr]
    folds = [list(range(k, n_ev, args.folds)) for k in range(args.folds)]
    ranks2d: list[list[list[int]] | None] = [None] * n_ev
    for k, test in enumerate(folds):
        train_idx = [i for i in range(n_ev) if i not in test]
        chat = [(ev[i]["h"], ev[i]["ids"].tolist(), [*ev[i]["ids"].tolist()[1:], 0], plens[i] - 1)
                for i in train_idx]  # labels: the exact answer's next token (the target's greedy choice)
        chat += [(z["h"], z["ids"].tolist(), z["am"].tolist(), plens[i] - 1)
                 for i in train_idx for z in others.get(i, [])]
        cnt: Counter[int] = Counter()
        for _, s, lab, _ in text_items + chat:
            cnt.update(s)
            cnt.update(lab)
        vocab = [t for t, _ in cnt.most_common(args.vocab)]
        if len(vocab) < args.vocab:
            have = set(vocab)
            vocab += [t for t in range(emb.shape[0]) if t not in have][: args.vocab - len(vocab)]
        vid = torch.full((emb.shape[0],), -1, dtype=torch.long)
        vid[torch.tensor(vocab)] = torch.arange(len(vocab))
        head = Head(emb.shape[1], args.d, emb[torch.tensor(vocab)].clone())
        data = windows(text_items + chat, args.unroll)
        print(f"fold {k}: {len(data[0])} training windows, held-out answers {test}", flush=True)
        train(head, emb, data, vid, args.epochs, args.unroll, args.wd)
        head.eval()
        for i in test:
            ranks2d[i] = evaluate(head, emb, vid, ev[i]["h"], ev[i]["ids"].tolist(), plens[i], args.depth)
        sd = {"w_in": head.w_in.weight, "b_in": head.w_in.bias, "w1": head.w1.weight, "b1": head.w1.bias,
              "w2": head.w2.weight, "b2": head.w2.bias, "w_out": head.w_out.weight}
        torch.save({"weights": {n: v.detach().clone() for n, v in sd.items()}, "vocab": vocab},
                   f"{args.out}.fold{k}.pt")  # loads into pgw.engine.generate.FeatureHeadDrafter
    nw = weights_per_step(emb.shape[1], args.d, args.vocab)
    json.dump({"ranks2d": ranks2d, "answers": ans["answers"][:n_ev], "prompt_lens": plens[:n_ev],
               "weights_per_step": nw, "vocab": args.vocab, "d": args.d}, open(args.out, "w"))
    print(f"weights read per draft step: {nw / 1e6:.2f} M ({nw / 1e6:.1f} MB int8)")


if __name__ == "__main__":
    main()
