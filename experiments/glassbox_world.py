"""Can a model designed for interpretability be read more accurately than a standard
one probed after training? Hallucination detection on a synthetic knowledge world.

World: 1000 entities x 8 relations, each fact with one of 50 values. Facts are
  known      stated in the training corpus ("F e r v ;"), so storable in memory;
             split into known_qa (also asked in training QA) and known_heldout
             (never asked in training: does knowledge transfer to QA?)
  absent     never stated; unk_train facts are asked in training (answer
             "unknown"), unk_test facts are never seen at all.
Questions "C e r v ; ... Q e r ?" may carry context facts (which override
memory, including conflicting ones). Relations r6, r7 are stated but never
asked in training (held-out relations); test contexts can be longer than any
seen in training.

Models (similar parameter counts):
  glassbox : vt.glassbox.GlassBoxLM. Evidence stream, starved core, product-key
             memory, self-knowledge bottleneck trained on its own correctness,
             counterfactual memory-patch training.
  baseline : vt.lm.GPT (ALiBi, wide MLPs), "unknown" an ordinary token.

Readouts for "will this answer be right?" (AUROC, label = the model's own
forced answer is correct):
  built-in k (glassbox); label-free evidence readouts (glassbox); post-hoc
  linear probes on every layer of either model (best layer chosen on
  in-distribution validation data, as an interpretability researcher would).
Causal checks (glassbox): patching another fact's memory read; corrupting the
context token the evidence stream points to vs. a random one.

    python experiments/glassbox_world.py --out results/glassbox/run.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from interp_by_design import auroc, fit_probe  # noqa: E402
from vt.glassbox import GBConfig, GlassBoxLM  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402

N_ENT, N_REL, N_VAL = 1000, 8, 50
QA_RELS, HELD_RELS = list(range(6)), [6, 7]
PAD, FT, CT, QT, QM, SEMI, UNKNOWN = range(7)
BASE = 7
VOCAB = BASE + N_ENT + N_REL + N_REL * N_VAL
BLOCK = 64


def ent(i): return BASE + i
def rel(r): return BASE + N_ENT + r
def val(r, v): return BASE + N_ENT + N_REL + r * N_VAL + v


VALUES = torch.arange(val(0, 0), VOCAB)                  # all value tokens (contiguous)


def vidx(tok): return tok - val(0, 0)
ANS = torch.cat([VALUES, torch.tensor([UNKNOWN])])       # baseline answer space


class World:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.value = [[rng.randrange(N_VAL) for _ in range(N_REL)] for _ in range(N_ENT)]
        self.by = {s: [] for s in ("known_qa", "known_heldout", "unk_train", "unk_test")}
        for e in range(N_ENT):
            for r in range(N_REL):
                u = rng.random()
                s = "known_qa" if u < 0.42 else "known_heldout" if u < 0.6 else "unk_train" if u < 0.8 else "unk_test"
                self.by[s].append((e, r))
        self.known = self.by["known_qa"] + self.by["known_heldout"]
        self.qa_by_rel = {r: [f for f in self.by["known_qa"] if f[1] == r] for r in range(N_REL)}

    def statements(self, rng, n=6):
        seq, pos = [], []
        for e, r in rng.sample(self.known, n):
            seq += [FT, ent(e), rel(r)]
            pos.append((len(seq) - 1, val(r, self.value[e][r])))
            seq += [val(r, self.value[e][r]), SEMI]
        return seq, pos

    def question(self, rng, kind, rels, kmin, kmax, test):
        """kind: ctx | mem | absent. Returns (tokens, '?' index, gold token, kind)."""
        pool = {"mem": self.by["known_heldout" if test else "known_qa"],
                "absent": self.by["unk_test" if test else "unk_train"]}
        if kind == "ctx":
            e, r = rng.randrange(N_ENT), rng.choice(rels)
        else:
            while True:
                e, r = rng.choice(pool[kind])
                if r in rels:
                    break
        facts = []
        for _ in range(rng.randint(kmin, kmax)):
            e2 = e if rng.random() < 0.3 else rng.randrange(N_ENT)
            r2 = r if rng.random() < 0.3 else rng.choice(rels)
            if (e2, r2) != (e, r):
                v2 = self.value[e2][r2] if rng.random() < 0.7 else rng.randrange(N_VAL)
                facts.append((e2, r2, v2))
        if kind == "ctx":
            v = self.value[e][r] if rng.random() < 0.7 else rng.randrange(N_VAL)
            facts.insert(rng.randint(0, len(facts)), (e, r, v))
            gold = val(r, v)
        else:
            gold = val(r, self.value[e][r]) if kind == "mem" else UNKNOWN
        seq = []
        for e2, r2, v2 in facts:
            seq += [CT, ent(e2), rel(r2), val(r2, v2), SEMI]
        seq += [QT, ent(e), rel(r), QM]
        return seq, len(seq) - 1, gold, kind


def pad(seqs):
    T = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (T - len(s)) for s in seqs])


def qa_set(world, rng, n, rels, kmin, kmax, test):
    kinds = ["ctx", "mem", "absent"]
    return [world.question(rng, kinds[i % 3], rels, kmin, kmax, test) for i in range(n)]


# ---------------------------------------------------------------- model wrappers

MEM_EVIDENCE, IIT_POS = "pooled", "answer"        # v1 settings; v2: "per_position", "all"
MEM2 = -1                                          # v3: late memory read after block 3, match supervised
MEM_INIT = 1.0                                     # v4: 0.0, absence as the default state of memory
MEM_LAYER = 2                                      # v5: 3, memory after the last attention block
NULL_SLOT = False                                  # v6: a "nothing stored" option in every read


class Glass:
    def __init__(self):
        self.m = GlassBoxLM(GBConfig(vocab_size=VOCAB, block_size=BLOCK, mem_evidence=MEM_EVIDENCE,
                                     mem2_layer=MEM2, mem_value_init=MEM_INIT, mem_layer=MEM_LAYER,
                                     null_slot=NULL_SLOT))

    def parameters(self):
        return [p for p in self.m.parameters() if p.requires_grad]

    def qa_heads(self, x, qpos):
        logits, k, e, hs = self.m(x)
        b = torch.arange(x.shape[0])
        lv = F.log_softmax(logits[b, qpos][:, VALUES], -1)
        return lv, k[b, qpos], e[b, qpos], [h[b, qpos] for h in hs], logits

    def answer(self, x, qpos):
        lv, k, e, hs, _ = self.qa_heads(x, qpos)
        forced = VALUES[lv.argmax(-1)]
        abstain = torch.sigmoid(self.m.abstain_logit(k)) > 0.5
        return torch.where(abstain, torch.full_like(forced, UNKNOWN), forced), forced, k, e, hs


class Base:
    def __init__(self):
        self.m = GPT(GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=4, n_head=4, n_embd=128,
                               mlp_ratio=16, pos_encoding="alibi"))
        self._hs = []
        for blk in self.m.blocks:
            blk.register_forward_hook(lambda mod, i, o: self._hs.append(o))

    def parameters(self):
        return list(self.m.parameters())

    def run(self, x):
        self._hs = []
        logits, _ = self.m(x)
        return logits, self._hs

    def answer(self, x, qpos):
        logits, hs = self.run(x)
        b = torch.arange(x.shape[0])
        lq = logits[b, qpos]
        out = ANS[lq[:, ANS].argmax(-1)]
        forced = VALUES[lq[:, VALUES].argmax(-1)]
        return out, forced, None, None, [h[b, qpos] for h in hs]


def stmt_loss(model, world, rng, n):
    seqs, tgt = [], []
    for _ in range(n):
        s, p = world.statements(rng)
        seqs.append(s); tgt.append(p)
    x = pad(seqs)
    logits = model.m(x)[0] if isinstance(model, Glass) else model.run(x)[0]
    bi, ti, yi = zip(*[(b, t, vidx(y)) for b, ps in enumerate(tgt) for t, y in ps])
    if isinstance(model, Glass):                       # slots read while storing a stated fact
        mask = torch.zeros(x.shape, dtype=torch.bool)
        mask[list(bi), list(ti)] = True
        model.m.mem.accumulate(mask)
    lv = F.log_softmax(logits[list(bi), list(ti)][:, VALUES], -1)
    return F.nll_loss(lv, torch.tensor(yi))


def train(kind, world, steps, seed, log):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Glass() if kind == "glassbox" else Base()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for step in range(steps):
        lr = 2e-3 * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        qs = qa_set(world, rng, 64, QA_RELS, 0, 4, test=False)
        x, qpos = pad([q[0] for q in qs]), torch.tensor([q[1] for q in qs])
        gold = torch.tensor([q[2] for q in qs])
        is_unk = gold == UNKNOWN
        gidx = torch.where(is_unk, 0, gold - val(0, 0))
        loss = stmt_loss(model, world, rng, 32)
        logs = {}
        if kind == "glassbox":
            lv, k, _, _, _ = model.qa_heads(x, qpos)
            d = model.m.abstain_logit(k)
            nll = torch.where(is_unk, -F.logsigmoid(d), -(F.logsigmoid(-d) + lv.gather(1, gidx[:, None]).squeeze(1)))
            correct = ((lv.argmax(-1) == gidx) & ~is_unk).float()           # the model's own correctness
            aux = F.binary_cross_entropy_with_logits(k, correct)
            if MEM2 >= 0:                   # a late lookup that misses should mean "I will be wrong"
                m2 = model.m.mem2.trace["match"][torch.arange(64), qpos]
                aux = aux + F.binary_cross_entropy_with_logits(model.m.mem2.lscale * m2 + model.m.mem2.lbias,
                                                               correct)
            # interchange intervention: B's memory read at A's '?' must make A answer B's value
            pa = []
            while len(pa) < 16:
                fa, fb = rng.sample(world.qa_by_rel[rng.choice(QA_RELS)], 2)
                if world.value[fa[0]][fa[1]] != world.value[fb[0]][fb[1]]:
                    pa.append((fa, fb))
            iit = torch.tensor(0.0)
            if pa:
                xa = pad([[QT, ent(e), rel(r), QM] for (e, r), _ in pa])
                xb = pad([[QT, ent(e), rel(r), QM] for _, (e, r) in pa])
                model.m(xb)
                rb = model.m.mem.trace["read"].detach()
                mask = torch.zeros(xa.shape, dtype=torch.bool)
                mask[:, 3 if IIT_POS == "answer" else slice(None)] = True
                model.m.mem.patch = (mask, rb)
                la = F.log_softmax(model.m(xa)[0][:, 3][:, VALUES], -1)
                model.m.mem.patch = None
                yb = torch.tensor([vidx(val(r, world.value[e][r])) for _, (e, r) in pa])
                iit = F.nll_loss(la, yb)
            loss = loss + nll.mean() + 0.5 * aux + 0.5 * iit
            logs = {"qa": round(nll.mean().item(), 4), "aux": round(aux.item(), 4), "iit": round(iit.item(), 4)}
        else:
            logits, _ = model.run(x)
            lq = F.log_softmax(logits[torch.arange(64), qpos][:, ANS], -1)
            aidx = torch.where(is_unk, torch.full_like(gidx, len(VALUES)), gidx)
            nll = F.nll_loss(lq, aidx)
            loss = loss + nll
            logs = {"qa": round(nll.item(), 4)}
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            log({"model": kind, "step": step, "loss": round(loss.item(), 4), **logs})
    return model


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def collect(model, qs, bs=100):
    rows = []
    for i in range(0, len(qs), bs):
        chunk = qs[i:i + bs]
        x, qpos = pad([q[0] for q in chunk]), torch.tensor([q[1] for q in chunk])
        out, forced, k, e, hs = model.answer(x, qpos)
        for j, q in enumerate(chunk):
            r = {"kind": q[3], "gold": q[2], "out": int(out[j]), "forced_ok": int(forced[j]) == q[2],
                 "h": [h[j].numpy() for h in hs]}
            if k is not None:
                ctx_end = len(q[0]) - 4                                     # context = tokens before 'Q'
                r["k"] = float(k[j])
                T = x.shape[1]
                match = model.m.mem.trace["match"][j]
                if e.shape[-1] == 2 * T:                                   # evidence per memory read
                    w = e[j, T:]
                    r["mem_mass"] = float(w.sum())
                    r["match"] = float((w * match).sum() / w.sum().clamp_min(1e-9))
                else:
                    r["mem_mass"] = float(e[j, -1])
                    r["match"] = float(match[q[1]])
                r["ctx_mass"] = float(e[j, :ctx_end].sum())
                if MEM2 >= 0:
                    r["match2"] = float(model.m.mem2.trace["match"][j, q[1]])
                r["familiarity"] = float(model.m.mem.familiarity()[j, q[1]])
                if NULL_SLOT:
                    r["not_null"] = 1 - float(model.m.mem.trace["null"][j, q[1]])
            rows.append(r)
    return rows


def behaviour(rows):
    d = {}
    for kind in ("ctx", "mem", "absent"):
        rs = [r for r in rows if r["kind"] == kind]
        d[f"{kind}_acc"] = float(np.mean([r["out"] == r["gold"] for r in rs]))
    rs = [r for r in rows if r["kind"] == "absent"]
    d["hallucination_rate"] = float(np.mean([r["out"] != UNKNOWN for r in rs]))
    d["accuracy"] = float(np.mean([r["out"] == r["gold"] for r in rows]))
    return d


def readout_aurocs(rows, probe, layer):
    y = np.array([r["forced_ok"] for r in rows], dtype=int)
    if y.min() == y.max():
        return {}
    d = {"probe_best_layer": auroc(probe(np.stack([r["h"][layer] for r in rows])), y)}
    if "k" in rows[0]:
        d["k_builtin"] = auroc(np.array([r["k"] for r in rows]), y)
        d["evidence_support"] = auroc(np.array([r["mem_mass"] + r["ctx_mass"] for r in rows]), y)
        d["memory_match"] = auroc(np.array([r["match"] for r in rows]), y)
        d["memory_familiarity"] = auroc(np.array([r["familiarity"] for r in rows]), y)
        if "not_null" in rows[0]:
            d["memory_not_null"] = auroc(np.array([r["not_null"] for r in rows]), y)
        if "match2" in rows[0]:
            d["late_memory_match"] = auroc(np.array([r["match2"] for r in rows]), y)
    return d


def fit_best_probe(rows_fit, rows_val):
    y_fit = np.array([r["forced_ok"] for r in rows_fit], dtype=int)
    y_val = np.array([r["forced_ok"] for r in rows_val], dtype=int)
    best = None
    for L in range(len(rows_fit[0]["h"])):
        probe, _ = fit_probe(np.stack([r["h"][L] for r in rows_fit]), y_fit)
        a = auroc(probe(np.stack([r["h"][L] for r in rows_val])), y_val)
        if best is None or a > best[0]:
            best = (a, L, probe)
    return best


@torch.no_grad()
def causal_checks(model, world, rng, n=200):
    m = model.m
    # 1. memory patch: B's read at A's '?' -> A answers B's value
    follow = tot = 0
    for _ in range(n):
        while True:
            (ea, ra), (eb, rb) = rng.choice(world.by["known_heldout"]), rng.choice(world.by["known_heldout"])
            if ra == rb and world.value[ea][ra] != world.value[eb][rb]:
                break
        xa, xb = torch.tensor([[QT, ent(ea), rel(ra), QM]]), torch.tensor([[QT, ent(eb), rel(rb), QM]])
        ob, *_ = model.answer(xb, torch.tensor([3]))
        if int(ob[0]) != val(rb, world.value[eb][rb]):
            continue
        rbv = m.mem.trace["read"]
        m.mem.patch = (torch.tensor([[IIT_POS != "answer"] * 3 + [True]]), rbv)
        oa, *_ = model.answer(xa, torch.tensor([3]))
        m.mem.patch = None
        tot += 1
        follow += int(oa[0]) == int(ob[0])
    # 2. evidence-guided corruption on context questions answered correctly
    hit = ctrl = cnt = 0
    for q in qa_set(world, rng, 3 * n, QA_RELS, 2, 4, test=True):
        if q[3] != "ctx":
            continue
        x, qp = torch.tensor([q[0]]), torch.tensor([q[1]])
        o, _, _, e, _ = model.answer(x, qp)
        if int(o[0]) != q[2]:
            continue
        vpos = [i for i in range(len(q[0]) - 4) if q[0][i] >= val(0, 0)]
        if len(vpos) < 2:
            continue
        top = max(vpos, key=lambda i: float(e[0, i]))
        other = rng.choice([i for i in vpos if i != top])
        changed = []
        for p in (top, other):
            x2 = x.clone()
            r_ = (q[0][p] - val(0, 0)) // N_VAL
            x2[0, p] = val(r_, (q[0][p] - val(r_, 0) + 1 + rng.randrange(N_VAL - 1)) % N_VAL)
            o2, *_ = model.answer(x2, qp)
            changed.append(int(o2[0]) != int(o[0]))
        hit += changed[0]; ctrl += changed[1]; cnt += 1
    return {"memory_patch_follows": follow / max(1, tot), "memory_patch_n": tot,
            "corrupt_evidence_top_changes_answer": hit / max(1, cnt),
            "corrupt_random_context_changes_answer": ctrl / max(1, cnt), "corrupt_n": cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glassbox,baseline")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--mem-evidence", default="pooled", choices=["pooled", "per_position"])
    ap.add_argument("--iit-positions", default="answer", choices=["answer", "all"])
    ap.add_argument("--mem2-layer", type=int, default=-1)
    ap.add_argument("--mem-init", type=float, default=1.0)
    ap.add_argument("--mem-layer", type=int, default=2)
    ap.add_argument("--null-slot", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/glassbox/run.json")
    a = ap.parse_args()
    global MEM_EVIDENCE, IIT_POS, MEM2, MEM_INIT, MEM_LAYER, NULL_SLOT
    MEM_EVIDENCE, IIT_POS, MEM2, MEM_INIT, MEM_LAYER, NULL_SLOT = (
        a.mem_evidence, a.iit_positions, a.mem2_layer, a.mem_init, a.mem_layer, a.null_slot)
    torch.set_num_threads(a.threads)
    world = World(1234)
    erng = random.Random(99)
    probe_fit = qa_set(world, erng, 1200, QA_RELS, 0, 4, test=True)
    probe_val = qa_set(world, erng, 600, QA_RELS, 0, 4, test=True)
    tests = {"in_dist": qa_set(world, erng, a.n_test, QA_RELS, 0, 4, test=True),
             "held_relations": qa_set(world, erng, a.n_test, HELD_RELS, 0, 4, test=True),
             "long_context": qa_set(world, erng, a.n_test, QA_RELS, 6, 10, test=True)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = {}
    for kind in a.models.split(","):
        t0 = time.time()
        ckpt = a.out.replace(".json", f".{kind}.pt")
        model = Glass() if kind == "glassbox" else Base()
        if os.path.exists(ckpt):
            missing, unexpected = model.m.load_state_dict(torch.load(ckpt), strict=False)
            assert not unexpected and all(k.endswith(("lscale", "lbias", "usage")) for k in missing), (missing, unexpected)
        else:
            model = train(kind, world, a.steps, a.seed, lambda e: print(json.dumps(e), flush=True))
            torch.save(model.m.state_dict(), ckpt)
        model.m.eval()
        r = {"train_s": time.time() - t0, "params": sum(p.numel() for p in model.parameters())}
        _, L, probe = fit_best_probe(collect(model, probe_fit), collect(model, probe_val))
        r["probe_layer"] = L
        for name, qs in tests.items():
            rows = collect(model, qs)
            r[name] = {**behaviour(rows), "auroc": readout_aurocs(rows, probe, L)}
        if kind == "glassbox":
            r["causal"] = causal_checks(model, world, random.Random(7))
        print(json.dumps({kind: r}), flush=True)
        res[kind] = r
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
