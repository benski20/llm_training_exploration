"""GPT + proved guard: guarantee a property of generated text.

1. Train a small byte-level GPT on balanced-bracket strings of length L.
2. Sample N strings and measure how many violate the spec.
3. Filter them with (a) a guard proved equal to the spec on all 2^L inputs and
   (b) an unverified guard with >99% held-out test accuracy.
4. Report leaks (invalid strings that pass) and false rejects, plus each
   guard's worst case over all 2^L inputs (by enumeration).

    python experiments/guarded_generation.py --out results/guarded_generation.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vt.artifact import save_artifact  # noqa: E402
from vt.global_verify import DyckSpec  # noqa: E402
from vt.guard import Guard  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402
from vt.model import MixerNet  # noqa: E402


def corpus(n, L, rng):
    seqs = DyckSpec()._balanced(n, L, rng)
    return b"".join(("".join("()"[t] for t in s) + "\n").encode() for s in seqs)


def train_gpt(data, steps, seed):
    torch.manual_seed(seed)
    m = GPT(GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=64))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    arr = np.frombuffer(data, dtype=np.uint8)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(arr) - 65, (32,), generator=g).tolist()
        x = torch.stack([torch.from_numpy(arr[i:i + 64].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(arr[i + 1:i + 65].astype(np.int64)) for i in ix])
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m, loss.item()


@torch.no_grad()
def sample(m, n, L, temperature, batch=2000):
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        idx = torch.full((b, 1), ord("\n"))
        for _ in range(L):
            logits, _ = m(idx[:, -64:])
            p = F.softmax(logits[:, -1] / temperature, -1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], 1)
        out += [bytes(r[1:].tolist()).decode("latin-1") for r in idx]
    return out


def worst_case(guard, L):
    spec = DyckSpec()
    seqs = list(itertools.product("()", repeat=L))
    strings = ["".join(s) for s in seqs]
    acc = guard.accepts(strings)
    lab = spec.label_batch(np.array([[c == ")" for c in s] for s in strings], dtype=np.int64))
    return {"invalid_accepted": int((acc & (lab == 0)).sum()), "valid_rejected": int((~acc & (lab == 1)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--proved", default="results/cegis/exact_dyck_L2-64/model_L16.npz")
    ap.add_argument("--proved-cert", default="results/cegis/exact_dyck_L2-64/certificate_L16.json")
    ap.add_argument("--unverified", default="results/unverified_guard_dyck16.npz",
                    help="unverified guard artifact (.npz), or a results/dyck16 .pt checkpoint")
    ap.add_argument("--out", default="results/guarded_generation.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    L, spec, rng = a.length, DyckSpec(), np.random.default_rng(0)

    t0 = time.time()
    gpt, loss = train_gpt(corpus(20000, L, rng), a.steps, 0)
    samples = sample(gpt, a.samples, L, a.temperature)
    well_formed = [s for s in samples if len(s) == L and set(s) <= set("()")]
    labels = {s: spec.label([c == ")" for c in s]) for s in set(well_formed)}

    guards = {"proved": Guard(a.proved, a.proved_cert)}
    # an accurate but unverified guard: the dense baseline from results/dyck16 (99.6% test acc)
    if a.unverified.endswith(".pt"):   # export the checkpoint once, as a committed artifact
        cfg = json.load(open(a.unverified.replace(".pt", ".json")))["config"]
        base = MixerNet(L, 2, cfg["d_model"], cfg["d_hidden"], cfg["n_blocks"], 2, ternary=cfg["ternary"])
        base.load_state_dict(torch.load(a.unverified))
        a.unverified = "results/unverified_guard_dyck16.npz"
        save_artifact(base.export(), a.unverified)
    guards["unverified (99.6% test acc)"] = Guard(a.unverified, None, alphabet="()", allow_unverified=True)

    raw_invalid = sum(1 for s in samples if not (s in labels and labels[s] == 1))
    report = {"gpt_final_loss": loss, "samples": len(samples), "raw_invalid": raw_invalid,
              "raw_invalid_frac": raw_invalid / len(samples), "guards": {}}
    for name, g in guards.items():
        acc = g.accepts(samples)
        leaks = sum(1 for s, k in zip(samples, acc) if k and labels.get(s, 0) != 1)
        false_rej = sum(1 for s, k in zip(samples, acc) if not k and labels.get(s, 0) == 1)
        report["guards"][name] = {"certified": g.proved, "accepted": int(acc.sum()), "leaks": leaks,
                                  "false_rejects": false_rej, "worst_case_all_inputs": worst_case(g, L)}
    report["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
