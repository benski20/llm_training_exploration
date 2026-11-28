#!/usr/bin/env python3
"""Training-free partial conversion to parallel blocks: per-layer sensitivity and quality-vs-trips curves.

Research script for ``research/reviews/parallel-conversion.md`` (angle 1: convert only the layers that tolerate it,
no retraining). One Kaggle T4 x 2 kernel (``benchmarks/kaggle/build_parallel_conversion.py``); ``--smoke`` runs a
tiny random model on the CPU.

A sequential pre-norm block is ``h1 = h + attn(n1 h); y = h1 + mlp(n2 h1)``: two sequential non-linear stops, two
fused round trips. Its parallel conversion reads the MLP input from the layer input, ``y = h + attn(n1 h) +
mlp(n2 h)``: one stop, one trip, the same weights. With a per-layer map a pass costs ``2 S + P (+ 1)`` trips
(``S`` sequential layers, ``P`` parallel ones; the ``+ 1`` is layer 0's input trip, the head folded into the last).

Per model, all in float16 on one GPU, the original model = every switch off:

1. **sensitivity** (no evaluation): on calibration text and chat, per layer, the change of the MLP input
   ``n2(h)`` vs ``n2(h + attn)`` (cosine, relative norm) and of the MLP output, relative to the MLP output and to
   the layer's output residual (``rel_out``: the one that predicts damage);
2. **single-layer conversion**: only layer ``i`` parallel; quality on a selection set (held-out FineWeb-Edu text:
   perplexity, KL to the original, top-1 agreement; UltraChat test prompts with the original's own greedy answers:
   KL, top-1 agreement);
3. **orderings**: convert ``k = 0..L`` layers in the order of (a) sensitivity ``rel_out``, (b) the measured
   single-layer KL, (c) depth position (last layers first / first layers first), (d) greedy marginal KL (at each
   step, among the ``--greedy-cands`` remaining layers with the lowest single-layer KL, the one that adds the least
   KL to the current set; measured on half the selection set);
4. **report points**: the best ordering (smallest mean selection KL over k) at a few ``k``, on the harness's
   held-out evaluation (``benchmarks/depth_prune_heal.py``: perplexity on 1024-token FineWeb-Edu documents, chat
   top-1 agreement / KL on 150 UltraChat ``test_sft`` prompts answered by the original, MMLU / ARC / HellaSwag
   subsets). The selection and report sets are disjoint.

KL on the selection set is the KL of the distributions coarsened to the original's top-64 tokens plus one "other"
bucket (a lower bound of the full KL by the data-processing inequality; the report points use the full KL).
Data: FineWeb-Edu (ODC-By 1.0), UltraChat 200k prompts (MIT; answers by the model itself), MMLU (MIT), ARC
(CC BY-SA 4.0), HellaSwag (MIT); eval only, nothing is trained.

Outputs (``--out``): ``<model>.json`` per model, ``log.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_prune_heal as H

T0 = time.time()
TOPK = 64

#: The models of the study: (repo, revision or None for the hub's current main, resolved and recorded).
MODELS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", None),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", None),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", None),
    "qwen3-4b": ("Qwen/Qwen3-4B", None),
}


def log(msg: str) -> None:
    H.log(msg)


# ---- switchable parallel conversion ------------------------------------------------------------------------
class SwitchNorm(nn.Module):
    """post_attention_layernorm replacement: ``active`` -> normalize the layer's *input* (stashed by a pre-hook),
    i.e. the parallel block; inactive -> the original norm of ``h + attn``. ``measure`` records the sensitivity."""

    def __init__(self, norm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.norm = norm
        object.__setattr__(self, "_mlp", mlp)  # not a submodule (it is the layer's own)
        self.active = False
        self.measure = False
        self.stash: torch.Tensor | None = None
        self.stats: dict[str, float] = {}
        self.count = 0

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        assert self.stash is not None
        if self.measure:
            self._measure(h1)
        out = self.norm(self.stash) if self.active else self.norm(h1)
        self.stash = None
        return out

    @torch.no_grad()
    def _measure(self, h1: torch.Tensor) -> None:
        h = self.stash
        ns, np_ = self.norm(h1).float(), self.norm(h).float()
        ms, mp = self._mlp(self.norm(h1)).float(), self._mlp(self.norm(h)).float()
        y = h1.float() + ms  # the layer's output residual
        dn, dm = (np_ - ns).norm(dim=-1), (mp - ms).norm(dim=-1)
        rec = {
            "cos_in": F.cosine_similarity(ns, np_, dim=-1).mean(),
            "rel_in": (dn / ns.norm(dim=-1).clamp_min(1e-6)).mean(),
            "rel_mlp": (dm / ms.norm(dim=-1).clamp_min(1e-6)).mean(),
            "rel_out": (dm / y.norm(dim=-1).clamp_min(1e-6)).mean(),
            "attn_rel": ((h1.float() - h.float()).norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-6)).mean(),
        }
        for k, v in rec.items():
            self.stats[k] = self.stats.get(k, 0.0) + float(v)
        self.count += 1


def install_switches(model) -> list[SwitchNorm]:
    sw = []
    for layer in H._layers(model):
        s = SwitchNorm(layer.post_attention_layernorm, layer.mlp)
        layer.post_attention_layernorm = s

        def pre(mod, args, kwargs, s=s):
            s.stash = args[0] if args else kwargs["hidden_states"]

        layer.register_forward_pre_hook(pre, with_kwargs=True)
        sw.append(s)
    return sw


def set_parallel(sw: list[SwitchNorm], layers) -> None:
    on = set(int(i) for i in layers)
    for i, s in enumerate(sw):
        s.active = i in on


def trips_per_pass(L: int, P: int) -> int:
    """Fused trips per pass with the head folded into the last trip: layer 0's input trip + 2 per sequential layer
    + 1 per parallel layer (``2L + 1`` all sequential; ``L + 1`` all parallel)."""
    return 1 + 2 * (L - P) + P


# ---- data ----------------------------------------------------------------------------------------------------
def chat_ids(tok, prompt: str, answer: list[int] | None = None, qwen3: bool = False) -> tuple[list[int], int]:
    kw = {"enable_thinking": False} if qwen3 else {}
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                **kw)
    p = tok(s, add_special_tokens=False)["input_ids"]
    return p + (answer or []), len(p)


@torch.no_grad()
def answers(model, tok, prompts: list[str], max_new: int, batch: int, qwen3: bool) -> list[list[int]]:
    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    enc = [chat_ids(tok, p, qwen3=qwen3)[0][-768:] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
    out: list[list[int]] = [[] for _ in prompts]
    for s in range(0, len(order), batch):
        ids = [enc[i] for i in order[s:s + batch]]
        m = max(len(x) for x in ids)
        x = torch.tensor([[pad] * (m - len(r)) + r for r in ids], device=dev)
        att = torch.tensor([[0] * (m - len(r)) + [1] * len(r) for r in ids], device=dev)
        g = model.generate(input_ids=x, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=pad, top_p=None, top_k=None, temperature=None)
        for j, i in enumerate(order[s:s + batch]):
            a = g[j, m:].tolist()
            cut = next((k + 1 for k, t in enumerate(a) if t in eos), len(a))
            out[i] = a[:cut]
    return out


# ---- selection-set evaluation (cached reference: top-64 of the original) ----------------------------------
class SelSet:
    """Sequences with target positions; the original's top-k log-probs cached on the CPU."""

    def __init__(self, seqs: list[tuple[torch.Tensor, int]], kind: str):
        self.seqs = seqs  # (ids 1-D, first scored position p: logits[p-1:-1] predict ids[p:])
        self.kind = kind
        self.ref: list[tuple[torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def logits(self, model, i: int) -> torch.Tensor:
        ids, p = self.seqs[i]
        dev = next(model.parameters()).device
        lg = model(input_ids=ids[None].to(dev)).logits[0, p - 1:-1].float()
        return lg

    @torch.no_grad()
    def cache(self, model) -> None:
        self.ref = []
        for i in range(len(self.seqs)):
            lp = F.log_softmax(self.logits(model, i), -1)
            v, ix = lp.topk(TOPK, dim=-1)
            self.ref.append((v.cpu(), ix.cpu()))

    @torch.no_grad()
    def score(self, model, idx: list[int] | None = None) -> dict[str, float]:
        kl = agree = nll = 0.0
        n = 0
        for i in idx if idx is not None else range(len(self.seqs)):
            ids, p = self.seqs[i]
            lp = F.log_softmax(self.logits(model, i), -1)
            rv, rix = (t.to(lp.device) for t in self.ref[i])
            pt = rv.exp()
            ls = lp.gather(1, rix)
            tail_t = (1 - pt.sum(-1)).clamp_min(1e-12)
            tail_s = (1 - ls.exp().sum(-1)).clamp_min(1e-12)
            kl += float((pt * (rv - ls)).sum() + (tail_t * (tail_t.log() - tail_s.log())).sum())
            agree += float((lp.argmax(-1) == rix[:, 0]).sum())
            y = ids[p:].to(lp.device)
            nll += float(F.nll_loss(lp, y, reduction="sum"))
            n += int(y.numel())
        return {"kl": kl / n, "top1": agree / n, "nll": nll / n, "tokens": n}


def sel_eval(model, sets: dict[str, SelSet], half: bool = False) -> dict[str, float]:
    r: dict[str, float] = {}
    for name, s in sets.items():
        idx = list(range(0, len(s.seqs), 2)) if half else None
        m = s.score(model, idx)
        r[f"{name}_kl"] = m["kl"]
        r[f"{name}_top1"] = m["top1"]
        if name == "web":
            r["web_ppl"] = math.exp(m["nll"])
    r["kl"] = sum(r[f"{n}_kl"] for n in sets) / len(sets)
    return r


# ---- report-point evaluation (the harness's metrics, full KL) ------------------------------------------------
@torch.no_grad()
def full_chat(model, items, ref: list[torch.Tensor] | None) -> tuple[dict[str, float], list[torch.Tensor]]:
    """Teacher-forced on the original's answers: top-1 agreement, full KL(orig || conv), NLL. ``ref`` = the
    original's log-probs per item (fp16 on the CPU), computed when None."""
    dev = next(model.parameters()).device
    agree = kl = nll = 0.0
    n = 0
    out = []
    for j, (ids, plen) in enumerate(items):
        x = torch.tensor([ids], device=dev)
        lp = F.log_softmax(model(input_ids=x).logits[0, plen - 1:-1].float(), -1)
        y = x[0, plen:]
        if ref is None:
            out.append(lp.half().cpu())
            pt = lp
        else:
            pt = ref[j].to(dev).float()
        if len(y) == 0:
            continue
        agree += float((lp.argmax(-1) == pt.argmax(-1)).sum())
        kl += float(F.kl_div(lp, pt, log_target=True, reduction="sum"))
        nll += float(F.cross_entropy(lp, y, reduction="sum"))
        n += len(y)
    return {"top1_agree": agree / n, "kl": kl / n, "answer_nll": nll / n, "tokens": n}, out


@torch.no_grad()
def ppl(model, data: torch.Tensor, bs: int) -> float:
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for s in range(0, len(data), bs):
        x = data[s:s + bs].to(dev)
        lg = model(input_ids=x).logits
        tot += float(F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                     reduction="sum"))
        n += x[:, 1:].numel()
    return math.exp(tot / n)


def report(model, tok, rp: dict, chat_ref) -> dict:
    t = time.time()
    r = {"ppl_fineweb": ppl(model, rp["ppl"], 2)}
    r["chat"], _ = full_chat(model, rp["chat"], chat_ref)
    for k, items in rp["mc"].items():
        r[k] = H.mc_eval(model, tok, items)
    r["seconds"] = time.time() - t
    return r


# ---- one model ------------------------------------------------------------------------------------------------
def run_model(key: str, args, dev: str, docs: list[str], prompts: dict[str, list[str]], mc: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_model = time.time()
    if args.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        repo, rev = "smoke", "smoke"
        cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                          num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                          max_position_embeddings=2048)
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg).to(dev)
        tok = AutoTokenizer.from_pretrained(args.smoke_tokenizer)
    else:
        repo, rev = MODELS[key]
        if rev is None:
            from huggingface_hub import HfApi

            rev = HfApi().model_info(repo).sha
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32,
                                                     attn_implementation="sdpa").to(dev)
    model.eval()
    qwen3 = "qwen3" in str(getattr(model.config, "model_type", ""))
    L = int(model.config.num_hidden_layers)
    log(f"[{key}] {repo}@{rev} on {dev}: L={L}, hidden {model.config.hidden_size}, {model.config.model_type}")
    rec: dict = {"model": key, "repo": repo, "revision": rev, "layers": L, "model_type": model.config.model_type,
                 "trips_all_sequential": trips_per_pass(L, 0), "trips_all_parallel": trips_per_pass(L, L)}

    def tok_docs(ds: list[str], n_tok: int) -> torch.Tensor:
        rows = [t[:n_tok] for t in (tok(d, add_special_tokens=False)["input_ids"] for d in ds) if len(t) >= n_tok]
        return torch.tensor(rows, dtype=torch.long)

    # report data: the harness's held-out sets (ppl docs[:n_ppl_docs] at 1024 tokens; 150 test_sft prompts)
    rp_ppl = tok_docs(docs[: args.n_ppl_docs], args.seq_len)
    ans_rep = answers(model, tok, prompts["report"], args.answer_tokens, args.gen_batch, qwen3)
    rp_chat = [chat_ids(tok, p, a, qwen3) for p, a in zip(prompts["report"], ans_rep)]
    # selection data (disjoint): later FineWeb-Edu docs, other test_sft prompts
    sel_web = tok_docs(docs[args.n_ppl_docs: args.n_ppl_docs + args.n_sel_docs * 3], args.sel_len)[: args.n_sel_docs]
    ans_sel = answers(model, tok, prompts["select"], args.sel_answer_tokens, args.gen_batch, qwen3)
    sel_chat = [chat_ids(tok, p, a, qwen3) for p, a in zip(prompts["select"], ans_sel)]
    sets = {"web": SelSet([(r, 1) for r in sel_web], "web"),
            "chat": SelSet([(torch.tensor(ids), pl) for ids, pl in sel_chat if len(ids) > pl], "chat")}
    # calibration (sensitivity only): other docs, train_sft prompts with the model's answers
    calib_web = tok_docs(docs[args.n_ppl_docs + args.n_sel_docs * 3:][: args.n_calib * 3], args.sel_len)[: args.n_calib]
    ans_cal = answers(model, tok, prompts["calib"], args.sel_answer_tokens, args.gen_batch, qwen3)
    calib = [r for r in calib_web] + [torch.tensor(chat_ids(tok, p, a, qwen3)[0]) for p, a in
                                      zip(prompts["calib"], ans_cal)]
    log(f"[{key}] data: report ppl {tuple(rp_ppl.shape)}, report chat {len(rp_chat)}, selection web "
        f"{tuple(sel_web.shape)} chat {len(sets['chat'].seqs)} ({sum(len(i) - p for i, p in sets['chat'].seqs)} "
        f"answer tokens), calibration {len(calib)} seqs; {time.time() - t_model:.0f}s")

    sw = install_switches(model)
    # 1. sensitivity
    for s in sw:
        s.measure = True
    with torch.no_grad():
        for x in calib:
            model(input_ids=x[None].to(dev))
    for s in sw:
        s.measure = False
    sens = [{k: v / max(s.count, 1) for k, v in s.stats.items()} for s in sw]
    rec["sensitivity"] = sens
    log(f"[{key}] sensitivity rel_out: " + " ".join(f"{i}:{d['rel_out']:.3f}" for i, d in enumerate(sens)))

    # reference (all off) + report baseline
    set_parallel(sw, [])
    for s in sets.values():
        s.cache(model)
    base_sel = sel_eval(model, sets)
    rec["base_selection"] = base_sel
    _, chat_ref = full_chat(model, rp_chat, None)
    rep_data = {"ppl": rp_ppl, "chat": rp_chat, "mc": mc}
    base_rep = report(model, tok, rep_data, chat_ref)
    rec["base_report"] = base_rep
    log(f"[{key}] base: sel {json.dumps({k: round(v, 4) for k, v in base_sel.items()})}; report ppl "
        f"{base_rep['ppl_fineweb']:.3f} chat top1 {base_rep['chat']['top1_agree']:.4f} "
        + " ".join(f"{k} {base_rep[k]:.3f}" for k in mc) + f" ({base_rep['seconds']:.0f}s)")
    budget_end = time.time() + args.model_min * 60

    # 2. single-layer conversion
    t = time.time()
    single = []
    for i in range(L):
        set_parallel(sw, [i])
        single.append(sel_eval(model, sets))
    rec["single"] = single
    log(f"[{key}] single-layer KL ({time.time() - t:.0f}s): " + " ".join(f"{i}:{d['kl']:.4f}"
                                                                       for i, d in enumerate(single)))
    by_single = sorted(range(L), key=lambda i: single[i]["kl"])
    by_sens = sorted(range(L), key=lambda i: sens[i]["rel_out"])

    # 3. orderings
    def curve(order: list[int], ks: list[int], label: str) -> list[dict]:
        pts = []
        for k in ks:
            set_parallel(sw, order[:k])
            r = sel_eval(model, sets)
            r.update(k=k, trips=trips_per_pass(L, k))
            pts.append(r)
        log(f"[{key}] curve {label}: " + " ".join(f"k{p['k']}:{p['kl']:.3f}/{p['web_ppl']:.2f}" for p in pts))
        return pts

    all_k = list(range(0, L + 1))
    half_k = sorted(set(list(range(0, L + 1, 2)) + [L]))
    orders = {"sensitivity": by_sens, "single_kl": by_single,
              "late_first": list(range(L - 1, -1, -1)), "early_first": list(range(L))}
    curves = {"sensitivity": curve(by_sens, all_k, "sensitivity"),
              "single_kl": curve(by_single, all_k, "single_kl")}
    curves["late_first"] = curve(orders["late_first"], half_k, "late_first")
    curves["early_first"] = curve(orders["early_first"], half_k, "early_first")
    # greedy marginal KL (lazy: a shortlist by single-layer KL), on half the selection set
    t = time.time()
    chosen: list[int] = []
    greedy_trace = []
    while len(chosen) < L and time.time() < budget_end:
        rest = [i for i in by_single if i not in chosen][: args.greedy_cands]
        best = None
        for c in rest:
            set_parallel(sw, chosen + [c])
            kl = sel_eval(model, sets, half=True)["kl"]
            if best is None or kl < best[1]:
                best = (c, kl)
        assert best is not None
        chosen.append(best[0])
        greedy_trace.append({"k": len(chosen), "layer": best[0], "kl_half": best[1]})
    if len(chosen) < L:
        log(f"[{key}] greedy stopped at k={len(chosen)} (time budget); the rest by single-layer KL")
        chosen += [i for i in by_single if i not in chosen]
    orders["greedy"] = chosen
    rec["greedy_trace"] = greedy_trace
    log(f"[{key}] greedy order ({time.time() - t:.0f}s): {chosen}")
    curves["greedy"] = curve(chosen, all_k, "greedy")
    rec["orders"] = orders
    rec["curves"] = curves

    # 4. report points on the best ordering (smallest mean selection KL over the common k)
    def area(name: str) -> float:
        pts = [p for p in curves[name] if p["k"] in half_k]
        return sum(p["kl"] for p in pts) / len(pts)

    rec["order_area"] = {n: area(n) for n in curves}
    best_name = min(curves, key=area)
    rec["best_order"] = best_name
    ks = sorted(set(int(x) for x in args.report_k.split(",")))
    pts_k = sorted(set(min(L, max(0, round(L * f / 100))) for f in ks) - {0})
    rep = []
    for k in pts_k:
        if time.time() > budget_end + args.report_extra_min * 60:
            log(f"[{key}] report points stopped at k={k} (time budget)")
            break
        set_parallel(sw, orders[best_name][:k])
        r = report(model, tok, rep_data, chat_ref)
        r.update(k=k, trips=trips_per_pass(L, k), layers=sorted(orders[best_name][:k]))
        rep.append(r)
        log(f"[{key}] report {best_name} k={k} ({r['trips']} trips): ppl {r['ppl_fineweb']:.3f} chat top1 "
            f"{r['chat']['top1_agree']:.4f} kl {r['chat']['kl']:.4f} " + " ".join(f"{m} {r[m]:.3f}" for m in mc))
    rec["report"] = rep
    rec["minutes"] = (time.time() - t_model) / 60
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return rec


def worker(args, keys: list[str], dev: str, shared: dict) -> None:
    for key in keys:
        try:
            rec = run_model(key, args, dev, shared["docs"], shared["prompts"], shared["mc"])
            rec["args"] = vars(args)
            with open(os.path.join(args.out, f"{key}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            log(f"[{key}] done in {rec['minutes']:.1f} min")
        except Exception as exc:  # one model failing must not cost the others
            import traceback

            log(f"[{key}] FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()


def _proc(args, keys: list[str], shared_path: str) -> None:
    H.LOG = open(os.path.join(args.out, "log.txt"), "a")
    worker(args, keys, "cuda:0", torch.load(shared_path, weights_only=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-4b,qwen2.5-0.5b,qwen2.5-1.5b;qwen2.5-3b,qwen3-1.7b",
                    help="';' separates the GPUs' queues, ',' the models of one queue")
    ap.add_argument("--out", default="out_parconv")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-ppl-docs", type=int, default=200)
    ap.add_argument("--n-chat-eval", type=int, default=150)
    ap.add_argument("--answer-tokens", type=int, default=256)
    ap.add_argument("--n-sel-docs", type=int, default=16)
    ap.add_argument("--sel-len", type=int, default=512)
    ap.add_argument("--n-sel-chat", type=int, default=40)
    ap.add_argument("--sel-answer-tokens", type=int, default=160)
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--greedy-cands", type=int, default=5)
    ap.add_argument("--report-k", default="25,40,50,60,75,100")
    ap.add_argument("--model-min", type=float, default=50, help="per-model budget before the report points")
    ap.add_argument("--report-extra-min", type=float, default=15)
    ap.add_argument("--text", default="fineweb", choices=["fineweb", "wikitext"],
                    help="held-out text: FineWeb-Edu (the harness's) or cached wikitext-103 (offline, no MC sets)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    H.LOG = open(os.path.join(args.out, "log.txt"), "a")
    queues = [[k for k in q.split(",") if k] for q in args.models.split(";")]
    ngpu = torch.cuda.device_count()
    if args.smoke:
        rnd = random.Random(0)
        words = "the a model layer token round trip client provider mask attention gate norm".split()
        docs = [" ".join(rnd.choice(words) for _ in range(1500)) for _ in range(args.n_ppl_docs + 200)]
        prompts = {"report": [f"Explain topic {i} briefly." for i in range(args.n_chat_eval)],
                   "select": [f"Tell me about item {i}." for i in range(args.n_sel_chat)],
                   "calib": [f"Describe thing {i}." for i in range(args.n_calib)]}
        mc: dict = {}
    elif args.text == "wikitext":  # offline CPU cross-check: wikitext-103 (CC BY-SA 4.0) from the local cache
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                               repo_type="dataset", local_files_only=True)
        text = "".join(pq.read_table(path).column("text").to_pylist()[:400000])
        n = args.n_ppl_docs + (args.n_sel_docs + args.n_calib) * 3 + 20
        docs = [text[i * 8000:(i + 1) * 8000] for i in range(n)]
        tp = H.ultrachat_prompts("test_sft", args.n_chat_eval + args.n_sel_chat, seed=2)
        prompts = {"report": tp[: args.n_chat_eval], "select": tp[args.n_chat_eval:],
                   "calib": H.ultrachat_prompts("train_sft", args.n_calib, seed=5)}
        mc = {}
    else:
        docs = H.fineweb_docs(args.n_ppl_docs + (args.n_sel_docs + args.n_calib) * 3 + 20)
        tp = H.ultrachat_prompts("test_sft", args.n_chat_eval + args.n_sel_chat, seed=2)
        prompts = {"report": tp[: args.n_chat_eval], "select": tp[args.n_chat_eval:],
                   "calib": H.ultrachat_prompts("train_sft", args.n_calib, seed=5)}
        try:
            mc = H.mc_sets(args.n_mc)
        except Exception as exc:
            log(f"multiple-choice sets failed to load: {type(exc).__name__}: {exc}")
            mc = {}
    log(f"devices {ngpu}; queues {queues}; mc " + ", ".join(f"{k} {len(v)}" for k, v in mc.items()))
    shared = {"docs": docs, "prompts": prompts, "mc": mc}
    if ngpu > 1 and len(queues) > 1:
        import multiprocessing as mp

        shared_path = os.path.join(args.out, "shared.pt")
        torch.save(shared, shared_path)
        ctx = mp.get_context("spawn")
        procs = []
        for i, q in enumerate(queues[:ngpu]):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            pr = ctx.Process(target=_proc, args=(args, q, shared_path))
            pr.start()
            procs.append(pr)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        for pr in procs:
            pr.join()
            log(f"worker exit code {pr.exitcode}")
        os.remove(shared_path)
    else:
        worker(args, [k for q in queues for k in q], "cuda:0" if ngpu else "cpu", shared)
    log("done")


if __name__ == "__main__":
    main()
