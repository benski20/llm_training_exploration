#!/usr/bin/env python3
"""Shallower students of a chat model: depth pruning and parallel blocks, healed by self-distillation.

Research script for ``research/reviews/consumer-model-depth.md`` (section 3). One Kaggle T4 x 2 kernel
(``benchmarks/kaggle/build_depth.py``); ``--smoke`` runs a tiny random model on the CPU.

Why: with the client evaluating every non-linear op, a token costs one round trip per *sequential non-linear
stop* (attention, SiLU gate) plus one: ``2L + 1`` with fused2 for a standard block. Two ways to cut the stops:

* **drop** ``n`` layers (Gromov et al. 2024: the contiguous block whose input and output hidden states have the
  smallest angular distance; the last layer is never dropped): ``2 (L - n) + 1`` trips.
* **parallel** blocks (GPT-J / PaLM form): ``y = x + attn(ln1 x) + mlp(ln2 x)`` instead of
  ``h = x + attn(ln1 x); y = h + mlp(ln2 h)``. Attention and the MLP read the same input, so ``qkv`` and
  ``gate_up`` share one trip and ``o`` + ``down`` (+ the next layer's input linears, fused2-style) another:
  ``L + 1`` trips with every weight kept.

Each student is healed by distillation from the unmodified model (KL(teacher || student) over the full vocabulary,
every position) on a token mix of FineWeb-Edu (ODC-By 1.0) and chat turns: UltraChat 200k *prompts* (MIT) with
the *teacher's own* greedy answers (no third-party model outputs are trained on). Evaluation (never trained on):
perplexity on held-out FineWeb-Edu documents; teacher-forced agreement on held-out UltraChat ``test_sft`` prompts
answered by the teacher (top-1 agreement with the teacher, KL, NLL); zero-shot multiple choice: MMLU (MIT),
ARC-Easy / ARC-Challenge (CC BY-SA 4.0), HellaSwag (MIT).

Outputs (``--out``): ``results.json`` (every variant before and after healing), ``log.txt``, sample answers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

#: Dataset repositories at fixed revisions: (repo, revision, licence, use).
DATA = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "MIT",
                  "train: prompts only (answers by the teacher); eval: test_sft prompts"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "main", "ODC-By-1.0", "train + held-out perplexity"),
    "mmlu": ("cais/mmlu", "main", "MIT", "eval only"),
    "arc": ("allenai/ai2_arc", "main", "CC-BY-SA-4.0", "eval only"),
    "hellaswag": ("Rowan/hellaswag", "main", "MIT", "eval only"),
}
ULTRACHAT_FILES = {"test_sft": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                   "train_sft": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"}
FINEWEB_FILE = "sample/10BT/000_00000.parquet"


def log(msg: str) -> None:
    line = f"[{(time.time() - T0) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    if LOG is not None:
        LOG.write(line + "\n")
        LOG.flush()


LOG = None


# ---- data --------------------------------------------------------------------------------------------------
def _hf_file(key: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, rev, _, _ = DATA[key]
    return hf_hub_download(repo, path, repo_type="dataset", revision=rev)


def ultrachat_prompts(split: str, n: int, seed: int) -> list[str]:
    import pyarrow.parquet as pq

    msgs = pq.read_table(_hf_file("ultrachat", ULTRACHAT_FILES[split]), columns=["messages"]).column("messages")
    idx = list(range(len(msgs)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        m = msgs[i].as_py()
        if m and m[0]["role"] == "user" and 20 < len(m[0]["content"]) < 2000:
            out.append(m[0]["content"])
        if len(out) >= n:
            break
    return out


def fineweb_docs(n: int) -> list[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(_hf_file("fineweb_edu", FINEWEB_FILE))
    out: list[str] = []
    for g in range(pf.num_row_groups):
        for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
            out.append(t)
        if len(out) >= n:
            break
    return out[:n]


def mc_sets(n_each: int) -> dict[str, list[dict]]:
    """Zero-shot multiple choice items: {"ctx": str, "choices": [str], "label": int, "kind": "norm"|"letter"}."""
    from datasets import load_dataset

    rnd = random.Random(7)
    sets: dict[str, list[dict]] = {}
    mm = load_dataset("cais/mmlu", "all", split="test")
    idx = rnd.sample(range(len(mm)), min(n_each, len(mm)))
    sets["mmlu"] = [{"ctx": (mm[i]["question"].strip() + "\n" + "\n".join(f"{c}. {t}" for c, t in
                                                                        zip("ABCD", mm[i]["choices"])) +
                             "\nAnswer:"), "choices": [" A", " B", " C", " D"], "label": int(mm[i]["answer"]),
                     "kind": "letter"} for i in idx]
    for cfg, name in (("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        items = []
        for r in ds:
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            items.append({"ctx": f"Question: {r['question']}\nAnswer:",
                          "choices": [" " + t for t in r["choices"]["text"]],
                          "label": labels.index(r["answerKey"]), "kind": "norm"})
        rnd.shuffle(items)
        sets[name] = items[:n_each]
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
    except Exception as exc:
        log(f"hellaswag not loaded: {type(exc).__name__}: {exc}")
        return sets
    idx = rnd.sample(range(len(hs)), min(n_each, len(hs)))

    def _pre(t: str) -> str:
        return t.replace(" [title]", ". ").replace("[header]", "").replace("  ", " ").strip()

    sets["hellaswag"] = [{"ctx": _pre(hs[i]["activity_label"] + ": " + hs[i]["ctx_a"] + " " +
                                      hs[i]["ctx_b"].capitalize()),
                          "choices": [" " + _pre(e) for e in hs[i]["endings"]], "label": int(hs[i]["label"]),
                          "kind": "norm"} for i in idx]
    return sets


def chat_ids(tok, prompt: str, answer: list[int] | None = None) -> tuple[list[int], int]:
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    p = tok(s, add_special_tokens=False)["input_ids"]
    return p + (answer or []), len(p)


@torch.no_grad()
def teacher_answers(model, tok, prompts: list[str], max_new: int, batch: int) -> list[list[int]]:
    """Greedy answers of the teacher (left-padded batches, longest first); EOS kept at the end."""
    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    enc = [chat_ids(tok, p)[0][-768:] for p in prompts]
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
        if (s // batch) % 10 == 0:
            log(f"  teacher answers {min(s + batch, len(order))}/{len(order)}")
    return out


def pack(seqs: list[list[int]], seq_len: int, sep: int) -> torch.Tensor:
    flat: list[int] = []
    for s in seqs:
        flat.extend(s)
        flat.append(sep)
    n = len(flat) // seq_len
    return torch.tensor(flat[: n * seq_len], dtype=torch.long).view(n, seq_len)


# ---- students ------------------------------------------------------------------------------------------------
def _layers(model) -> nn.ModuleList:
    return model.model.layers


def _set_depth(model, keep: list[int]) -> None:
    old = _layers(model)
    model.model.layers = nn.ModuleList([old[i] for i in keep])
    for j, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = j
    cfg = model.config
    lt = getattr(cfg, "layer_types", None)
    if isinstance(lt, list) and len(lt) == len(old):
        cfg.layer_types = [lt[i] for i in keep]
    cfg.num_hidden_layers = len(keep)


@torch.no_grad()
def angular_distances(model, batches: list[torch.Tensor]) -> list[list[float]]:
    """d[n][l] = mean over tokens of arccos(cos(h_l, h_{l+n})) / pi (h_l = input of layer l; Gromov et al.)."""
    L = len(_layers(model))
    acc = [[0.0] * (L + 1) for _ in range(L + 1)]
    cnt = 0
    for x in batches:
        hs = model(input_ids=x.to(next(model.parameters()).device), output_hidden_states=True).hidden_states
        # hidden_states[L] is after the final norm in HF; use the pre-norm residual of the last layer instead
        hs = [h.float() for h in hs[:L]]
        hs.append(None)
        for n in range(1, L):
            for l in range(0, L - n):  # l + n <= L - 1: the last layer is kept
                c = F.cosine_similarity(hs[l], hs[l + n], dim=-1).clamp(-1, 1)
                acc[n][l] += float(torch.arccos(c).mean() / math.pi)
        cnt += 1
    return [[v / cnt for v in row] for row in acc]


def drop_plan(dist: list[list[float]], L: int, n: int) -> list[int]:
    best = min(range(0, L - n), key=lambda l: dist[n][l])
    return [i for i in range(L) if not (best <= i < best + n)]


class _ParallelNorm(nn.Module):
    """post_attention_layernorm replacement: normalizes the layer's *input* (stashed by a pre-hook), so the
    standard decoder-layer forward computes ``x + attn(ln1 x) + mlp(ln2 x)``."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm
        self.stash: torch.Tensor | None = None

    def forward(self, _h: torch.Tensor) -> torch.Tensor:
        assert self.stash is not None
        return self.norm(self.stash)


def make_parallel(model, which: list[int] | None = None) -> None:
    layers = _layers(model)
    for i in which if which is not None else range(len(layers)):
        layer = layers[i]
        pn = _ParallelNorm(layer.post_attention_layernorm)
        layer.post_attention_layernorm = pn

        def pre(mod, args, kwargs, pn=pn):
            pn.stash = args[0] if args else kwargs["hidden_states"]

        layer.register_forward_pre_hook(pre, with_kwargs=True)


def trips(L_seq_stops: int) -> int:
    return L_seq_stops + 1


# ---- evaluation ----------------------------------------------------------------------------------------------
@torch.no_grad()
def ppl(model, data: torch.Tensor, bs: int) -> float:
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for s in range(0, len(data), bs):
        x = data[s:s + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(input_ids=x).logits
        lp = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        tot += float(lp)
        n += x[:, 1:].numel()
    return math.exp(tot / n)


@torch.no_grad()
def chat_agreement(student, teacher, items: list[tuple[list[int], int]]) -> dict[str, float]:
    """Teacher-forced on the teacher's greedy answers: top-1 agreement, mean KL(t||s), student NLL (answer tokens)."""
    dev_s = next(student.parameters()).device
    dev_t = next(teacher.parameters()).device
    agree = kl = nll = 0.0
    n = 0
    for ids, plen in items:
        x = torch.tensor([ids])
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev_s.type == "cuda"):
            ls = student(input_ids=x.to(dev_s)).logits[0, plen - 1:-1].float()
            lt = teacher(input_ids=x.to(dev_t)).logits[0, plen - 1:-1].float().to(dev_s)
        y = x[0, plen:].to(dev_s)
        if len(y) == 0:
            continue
        agree += float((ls.argmax(-1) == lt.argmax(-1)).sum())
        pt = F.log_softmax(lt, -1)
        kl += float(F.kl_div(F.log_softmax(ls, -1), pt, log_target=True, reduction="sum"))
        nll += float(F.cross_entropy(ls, y, reduction="sum"))
        n += len(y)
    return {"top1_agree": agree / n, "kl": kl / n, "answer_nll": nll / n, "tokens": n}


@torch.no_grad()
def mc_eval(model, tok, items: list[dict]) -> float:
    dev = next(model.parameters()).device
    correct = 0
    for it in items:
        ctx = tok(it["ctx"], add_special_tokens=False)["input_ids"][-900:]
        scores = []
        if it["kind"] == "letter":
            x = torch.tensor([ctx], device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(input_ids=x).logits[0, -1].float()
            for c in it["choices"]:
                scores.append(float(lg[tok(c, add_special_tokens=False)["input_ids"][0]]))
        else:
            conts = [tok(c, add_special_tokens=False)["input_ids"][:120] for c in it["choices"]]
            m = max(len(c) for c in conts)
            x = torch.full((len(conts), len(ctx) + m), tok.pad_token_id or 0, dtype=torch.long)
            for j, c in enumerate(conts):
                x[j, : len(ctx) + len(c)] = torch.tensor(ctx + c)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                lg = model(input_ids=x.to(dev)).logits.float()
            lp = F.log_softmax(lg, -1)
            for j, c in enumerate(conts):
                pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(c), device=dev)
                s = lp[j, pos, torch.tensor(c, device=dev)].sum()
                chars = len(it["choices"][j])
                scores.append(float(s) / max(chars, 1))  # acc_norm: per character
        correct += int(max(range(len(scores)), key=lambda k: scores[k]) == it["label"])
    return correct / max(len(items), 1)


@torch.no_grad()
def samples(model, tok, prompts: list[str], max_new: int) -> list[str]:
    dev = next(model.parameters()).device
    out = []
    for p in prompts:
        ids = torch.tensor([chat_ids(tok, p)[0]], device=dev)
        g = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False, top_p=None, top_k=None,
                           temperature=None, pad_token_id=tok.pad_token_id)
        out.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return out


def evaluate(model, teacher, tok, ev: dict, label: str) -> dict:
    model.eval()
    r = {"ppl_fineweb": ppl(model, ev["ppl"], ev["bs"]), "chat": chat_agreement(model, teacher, ev["chat"])}
    for k, items in ev["mc"].items():
        r[k] = mc_eval(model, tok, items)
    r["samples"] = samples(model, tok, ev["sample_prompts"], ev["sample_tokens"])
    log(f"  eval {label}: ppl {r['ppl_fineweb']:.3f} chat top1 {r['chat']['top1_agree']:.4f} kl "
        f"{r['chat']['kl']:.4f} " + " ".join(f"{k} {r[k]:.4f}" for k in ev["mc"]))
    return r


# ---- healing ---------------------------------------------------------------------------------------------
def heal(student, teacher, train: torch.Tensor, args, label: str, lr: float | None = None) -> dict:
    dev = next(student.parameters()).device
    student.train()
    if args.freeze_embed:
        student.get_input_embeddings().weight.requires_grad_(False)
        if student.get_output_embeddings() is not None:
            student.get_output_embeddings().weight.requires_grad_(False)
    if dev.type == "cuda":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr or args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = args.steps
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    mb, acc = args.micro_batch, args.accum
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(1))
    deadline = time.time() + args.heal_min * 60
    pos, done, hist = 0, 0, []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(acc):
            if pos + mb > len(perm):
                pos = 0
            x = train[perm[pos:pos + mb]].to(dev)
            pos += mb
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                lt = teacher(input_ids=x.to(next(teacher.parameters()).device)).logits.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                ls = student(input_ids=x).logits
            loss = 0.0
            V = ls.shape[-1]
            ls2, lt2 = ls.reshape(-1, V), lt.reshape(-1, V)
            ch = 1024
            n = ls2.shape[0]
            for c in range(0, n, ch):  # chunked fp32 KL keeps the vocab-wide softmax small
                s_ = F.log_softmax(ls2[c:c + ch].float(), -1)
                t_ = F.log_softmax(lt2[c:c + ch].float(), -1)
                loss = loss + F.kl_div(s_, t_, log_target=True, reduction="sum") / n
            scaler.scale(loss / acc).backward()
            tot += float(loss) / acc
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        done = step + 1
        hist.append(tot)
        if step % 50 == 0 or step == steps - 1:
            tps = done * acc * mb * train.shape[1] / (time.time() - t0)
            log(f"  heal {label} step {step}/{steps} kl {tot:.4f} lr {sched.get_last_lr()[0]:.2e} {tps:.0f} tok/s")
        if time.time() > deadline:
            log(f"  heal {label}: time budget reached at step {done}")
            break
    if dev.type == "cuda":
        student.gradient_checkpointing_disable()
    student.eval()
    return {"steps": done, "tokens": done * acc * mb * train.shape[1], "kl_first": hist[0],
            "kl_last50": sum(hist[-50:]) / len(hist[-50:]), "minutes": (time.time() - t0) / 60}


# ---- driver ----------------------------------------------------------------------------------------------
def load_model(args, dev: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    if args.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        cfg = Qwen2Config(vocab_size=args.smoke_vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                          max_position_embeddings=2048)
        torch.manual_seed(0)
        return Qwen2ForCausalLM(cfg).to(dev)
    AutoConfig.from_pretrained(args.model, revision=args.revision)
    return AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32,
                                                attn_implementation="sdpa").to(dev)


def variant_layers(v: str, L: int, dist) -> tuple[list[int], bool, int]:
    """(kept layers, parallel?, sequential non-linear stops) for a variant name."""
    if v == "base":
        return list(range(L)), False, 2 * L
    if v.startswith("drop"):
        n = round(L * int(v[4:]) / 100)
        keep = drop_plan(dist, L, n)
        return keep, False, 2 * len(keep)
    if v == "parallel":
        return list(range(L)), True, L
    if v.startswith("parallel_drop"):
        n = round(L * int(v[len("parallel_drop"):]) / 100)
        keep = drop_plan(dist, L, n)
        return keep, True, len(keep)
    raise SystemExit(f"unknown variant {v}")


def worker(args, variants: list[str], dev: str, shared: dict, out: list) -> None:
    import copy

    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, revision=None if args.smoke else args.revision)
    teacher = load_model(args, dev).half() if dev.startswith("cuda") else load_model(args, dev)
    teacher.eval()
    for v in variants:
        L = teacher.config.num_hidden_layers
        name, _, lr_s = v.partition("@")  # "parallel@2e-4": the variant healed at another peak learning rate
        lr = float(lr_s) if lr_s else None
        keep, par, stops = variant_layers(name, L, shared["dist"])
        log(f"variant {v} on {dev} (gpu {os.environ.get('CUDA_VISIBLE_DEVICES', '-')}): keep {len(keep)}/{L} layers {keep if len(keep) < L else 'all'} parallel {par} "
            f"-> {stops} sequential stops, {trips(stops)} trips per token (fused, client non-linearities)")
        student = load_model(args, dev)
        _set_depth(student, keep)
        if par:
            make_parallel(student)
        rec = {"variant": v, "lr": lr or args.lr, "keep": keep, "parallel": par, "layers": len(keep), "stops": stops,
               "trips_fused": trips(stops), "params_M": sum(p.numel() for p in student.parameters()) / 1e6}
        if name == "base":
            rec["before"] = evaluate(student, teacher, tok, shared["ev"], v)
        else:
            rec["before"] = evaluate(student, teacher, tok, shared["ev"], v + " (no heal)")
            rec["heal"] = heal(student, teacher, shared["train"], args, v, lr)
            rec["after"] = evaluate(student, teacher, tok, shared["ev"], v + " (healed)")
        out.append(rec)
        with open(os.path.join(args.out, f"result_{v}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        del student
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    del copy


def _proc_worker(args, variants: list[str], shared_path: str) -> None:
    global LOG
    LOG = open(os.path.join(args.out, "log.txt"), "a")
    shared = torch.load(shared_path, weights_only=False)
    worker(args, variants, "cuda:0", shared, [])


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--variants", default="base,drop25,drop50,parallel,parallel_drop25")
    ap.add_argument("--out", default="out_depth")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-fineweb", type=int, default=12000)
    ap.add_argument("--n-chat-train", type=int, default=3000)
    ap.add_argument("--n-chat-eval", type=int, default=150)
    ap.add_argument("--n-ppl-docs", type=int, default=200)
    ap.add_argument("--n-mc", type=int, default=600)
    ap.add_argument("--answer-tokens", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--heal-min", type=float, default=55)
    ap.add_argument("--freeze-embed", action="store_true", default=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-vocab", type=int, default=151936)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    LOG = open(os.path.join(args.out, "log.txt"), "a")
    from transformers import AutoTokenizer

    ngpu = torch.cuda.device_count()
    devs = [f"cuda:{i}" for i in range(ngpu)] or ["cpu"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, revision=None if args.smoke else args.revision)
    sep = tok.eos_token_id
    log(f"devices {devs}; data {json.dumps({k: v[:3] for k, v in DATA.items()})}")
    # data
    if args.smoke:  # synthetic text: no downloads
        rnd = random.Random(0)
        words = "the a model layer token round trip client provider mask attention gate norm".split()
        docs = [" ".join(rnd.choice(words) for _ in range(1500)) for _ in range(args.n_fineweb + args.n_ppl_docs)]
    else:
        docs = fineweb_docs(args.n_fineweb + args.n_ppl_docs)
    ppl_docs, train_docs = docs[: args.n_ppl_docs], docs[args.n_ppl_docs:]
    ppl_data = torch.stack([torch.tensor(t[: args.seq_len]) for t in
                            (tok(d, add_special_tokens=False)["input_ids"] for d in ppl_docs) if len(t) >= args.seq_len])
    log(f"ppl eval: {len(ppl_data)} FineWeb-Edu docs x {args.seq_len} tokens")
    if args.smoke:
        chat_train_p = [f"Tell me about item {i}." for i in range(args.n_chat_train)]
        chat_eval_p = [f"Explain topic {i} briefly." for i in range(args.n_chat_eval)]
    else:
        chat_train_p = ultrachat_prompts("train_sft", args.n_chat_train, seed=1)
        chat_eval_p = ultrachat_prompts("test_sft", args.n_chat_eval, seed=2)
    teacher = load_model(args, devs[0]).half() if ngpu else load_model(args, devs[0])
    teacher.eval()
    log(f"teacher answers for {len(chat_train_p)} train + {len(chat_eval_p)} eval prompts")
    ans_eval = teacher_answers(teacher, tok, chat_eval_p, args.answer_tokens, args.gen_batch)
    ans_train = teacher_answers(teacher, tok, chat_train_p, args.answer_tokens, args.gen_batch)
    chat_eval = [chat_ids(tok, p, a) for p, a in zip(chat_eval_p, ans_eval)]
    chat_train = [chat_ids(tok, p, a)[0] for p, a in zip(chat_train_p, ans_train)]
    web = [tok(d, add_special_tokens=False)["input_ids"] for d in train_docs]
    n_chat_tok = sum(len(c) for c in chat_train)
    random.Random(3).shuffle(web)
    web_keep, acc_ = [], 0
    for w in web:  # FineWeb-Edu tokens up to 2x the chat tokens (a ~1:2 chat:web mix)
        web_keep.append(w)
        acc_ += len(w)
        if acc_ > 2 * n_chat_tok:
            break
    train = torch.cat([pack(chat_train, args.seq_len, sep), pack(web_keep, args.seq_len, sep)])
    log(f"train: {len(train)} x {args.seq_len} tokens ({n_chat_tok / 1e6:.2f}M chat, {acc_ / 1e6:.2f}M FineWeb-Edu)")
    mc = {}
    if not args.smoke:
        try:
            mc = mc_sets(args.n_mc)
        except Exception as exc:  # a benchmark that fails to load must not cost the run
            log(f"multiple-choice sets failed to load: {type(exc).__name__}: {exc}")
    log("multiple choice: " + ", ".join(f"{k} {len(v)}" for k, v in mc.items()))
    calib = [ppl_data[i:i + 4] for i in range(0, min(32, len(ppl_data)), 4)]
    dist = angular_distances(teacher, calib)
    L = teacher.config.num_hidden_layers
    log("angular distance, best block per n: " + ", ".join(
        f"n={n}: l={min(range(L - n), key=lambda l: dist[n][l])} d={min(dist[n][:L - n]):.3f}" for n in range(1, L)))
    del teacher
    if ngpu:
        torch.cuda.empty_cache()
    ev = {"ppl": ppl_data, "bs": 4, "chat": chat_eval, "mc": mc,
          "sample_prompts": chat_eval_p[:3], "sample_tokens": 96}
    shared = {"dist": dist, "ev": ev, "train": train}
    variants = [v for v in args.variants.split(",") if v]
    results: list = []
    if len(devs) > 1:
        # one process per GPU (CUDA_VISIBLE_DEVICES), variants round-robin: two CUDA devices driven from threads of
        # one process hit illegal-address errors under gradient checkpointing's RNG forks (Kaggle run 1)
        import multiprocessing as mp

        shared_path = os.path.join(args.out, "shared.pt")
        torch.save(shared, shared_path)
        plan: dict[int, list[str]] = {i: [] for i in range(len(devs))}
        for i, v in enumerate(variants):
            plan[i % len(devs)].append(v)
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(len(devs)):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            pr = ctx.Process(target=_proc_worker, args=(args, plan[i], shared_path))
            pr.start()
            procs.append(pr)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        for pr in procs:
            pr.join()
            log(f"worker exit code {pr.exitcode}")
        for v in variants:
            f = os.path.join(args.out, f"result_{v}.json")
            if os.path.exists(f):
                results.append(json.load(open(f)))
        os.remove(shared_path)
    else:
        worker(args, variants, devs[0], shared, results)
    summary = {"model": args.model, "revision": args.revision, "data": DATA, "args": vars(args),
               "angular_distance": dist, "results": sorted(results, key=lambda r: variants.index(r["variant"]))}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
