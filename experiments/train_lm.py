"""Train a byte-level GPT with optional ternary weights and neuron-margin loss.

    # CPU toy run
    python experiments/train_lm.py --data input.txt --out runs/lm_base
    python experiments/train_lm.py --data input.txt --out runs/lm_tern_margin --ternary --margin-weight 0.1
    # GPT-2-small scale on pre-tokenised data (nanoGPT .bin format, GPT-2 BPE ids as uint16):
    torchrun --nproc_per_node 8 experiments/train_lm.py --data fineweb_train.bin --dtype uint16 \
        --preset gpt2-124m --block-size 1024 --batch-size 16 --grad-accum 4 --steps 20000 \
        --lr 6e-4 --ternary --margin-weight 0.5 --compile

Data is any text/byte file, read via np.memmap (no size limit). Training
resumes from <out>/ckpt.pt. Final metrics go to <out>/metrics.json.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.lm import GPT, GPTConfig  # noqa: E402


# (n_layer, n_head, n_embd); gpt2-* match the GPT-2 family sizes
PRESETS = {"tiny": (4, 4, 128), "small": (6, 6, 384), "gpt2-124m": (12, 12, 768),
           "gpt2-350m": (24, 16, 1024), "gpt2-774m": (36, 20, 1280)}


def get_batch(data, block, batch, device, gen):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix.tolist()])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, args, device, iters):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses, stable = [], []
    for _ in range(iters):
        x, y = get_batch(data, args.block_size, args.batch_size, device, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
        stable.append(model.stable_fraction())
    model.train()
    loss = float(np.mean(losses))
    return {"loss": loss, "bpb": loss / math.log(2), "stable_frac": float(np.mean(stable))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--margin-weight", type=float, default=0.0)
    ap.add_argument("--margin-eps", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=PRESETS, help="model size preset (overrides --n-*)")
    ap.add_argument("--dtype", default="uint8", choices=["uint8", "uint16"],
                    help="uint8: raw bytes (vocab 256); uint16: pre-tokenised ids, e.g. GPT-2 BPE")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="defaults to 256 for uint8, 50304 for uint16")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    if a.preset:
        a.n_layer, a.n_head, a.n_embd = PRESETS[a.preset]
    if a.vocab_size is None:
        a.vocab_size = 256 if a.dtype == "uint8" else 50304

    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}" if torch.cuda.is_available() else "cpu"
    else:
        rank, device = 0, "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    os.makedirs(a.out, exist_ok=True)

    raw = np.memmap(a.data, dtype=np.dtype(a.dtype), mode="r")
    split = int(0.9 * len(raw))
    train_data, val_data = raw[:split], raw[split:]

    cfg = GPTConfig(vocab_size=a.vocab_size, block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head,
                    n_embd=a.n_embd, ternary=a.ternary, margin_eps=a.margin_eps,
                    margin_weight=a.margin_weight)
    torch.manual_seed(a.seed)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    step, log = 0, []
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
        if master:
            print(f"resumed at step {step}")
    raw_model = model
    if a.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    gen = torch.Generator().manual_seed(a.seed * 1000 + rank + step)
    warmup = max(1, a.steps // 20)

    def lr_at(s):
        if s < warmup:
            return a.lr * (s + 1) / warmup
        return a.lr * 0.1 + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, a.steps - warmup)))

    t0 = time.time()
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for micro in range(a.grad_accum):
            x, y = get_batch(train_data, a.block_size, a.batch_size, device, gen)
            if ddp:
                model.require_backward_grad_sync = micro == a.grad_accum - 1
            with amp:
                _, loss = model(x, y)
                total = loss
                if a.margin_weight:
                    # ramp the regulariser in so early learning is unconstrained
                    total = loss + a.margin_weight * min(1.0, step / (0.3 * a.steps)) * raw_model.margin_loss()
            (total / a.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if master and (step % a.eval_every == 0 or step == a.steps):
            ev = evaluate(raw_model, val_data, a, device, a.eval_iters)
            ev.update(step=step, train_loss=loss.item(), elapsed=time.time() - t0)
            log.append(ev)
            print(f"step {step:5d}  train {loss.item():.3f}  val {ev['loss']:.3f} "
                  f"({ev['bpb']:.3f} bpb)  stable {ev['stable_frac']:.3f}  {ev['elapsed']:.0f}s", flush=True)
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "log": log, "config": asdict(cfg)}, ckpt_path)
    if master:
        final = log[-1]
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump({"args": vars(a), "config": asdict(cfg), "final": final, "log": log,
                       "params": sum(p.numel() for p in raw_model.parameters())}, f, indent=1)
        raw_model.eval()
    if master and a.dtype == "uint8":
        prompt = torch.tensor([list(b"ROMEO:\n")], device=device)
        sample = raw_model.generate(prompt, 200, temperature=0.8, top_k=20)[0].tolist()
        with open(os.path.join(a.out, "sample.txt"), "wb") as f:
            f.write(bytes(sample))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
