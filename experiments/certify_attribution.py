"""Certified attribution on trained glass-box models.

For each test question, classify the model's decision:
  certified memory-free  proved: no memory content can change it (vt.certify)
  memory-dependent       witnessed: patching other memory reads changes it
  undetermined           neither
Memory-free certificates on questions whose fact is in the context mean
"provably answered from the conversation, not from stored knowledge".

    python experiments/certify_attribution.py --out results/certify_attribution.json
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from vt.certify import certified_argmax, memory_free_bounds  # noqa: E402
from vt.glassbox import GBConfig, GlassBoxLM  # noqa: E402


def witness(model, x, pos, decide, base, rng, tries=8):
    """A memory patch at every position that changes the decision, if one is found quickly."""
    V = model.mem.values.weight
    used = (model.mem.usage > 0).nonzero().squeeze(-1)
    pool = used if len(used) else torch.arange(V.shape[0])
    for _ in range(tries):
        slots = pool[torch.randint(len(pool), (x.shape[1],), generator=rng)]
        model.mem.patch = (torch.ones_like(x, dtype=torch.bool), V[slots][None])
        d = decide(x, pos)
        model.mem.patch = None
        if d != base:
            return True
    return False


def synthetic(path, mem_layer, n, seed):
    import glassbox_world as W
    m = GlassBoxLM(GBConfig(vocab_size=W.VOCAB, block_size=W.BLOCK, mem_layer=mem_layer))
    m.load_state_dict(torch.load(path), strict=False)
    m.eval()
    world = W.World(1234)
    qs = W.qa_set(world, random.Random(seed), n, W.QA_RELS, 0, 4, test=True)
    sp = torch.nn.functional.softplus(m.alpha)

    def decide(x, pos):
        logits, k, _, _ = m(x)
        if (m.beta - sp * k[0, pos]) > 0:
            return W.UNKNOWN
        return int(W.VALUES[logits[0, pos, W.VALUES].argmax()])

    g = torch.Generator().manual_seed(seed)
    rows = []
    for toks, pos, gold, kind in qs:
        x = torch.tensor([toks])
        with torch.no_grad():
            base = decide(x, pos)
            ll, lh, kl, kh, _, _ = memory_free_bounds(m, x)
            d_lo, d_hi = m.beta - sp * kh[0, pos], m.beta - sp * kl[0, pos]
            if d_lo > 0:
                cert = W.UNKNOWN
            elif d_hi < 0:
                c = certified_argmax(ll[0, pos], lh[0, pos], W.VALUES)
                cert = int(c) if int(c) >= 0 else None
            else:
                cert = None
            dep = cert is None and witness(m, x, pos, decide, base, g)
        assert cert is None or cert == base, "a certificate must agree with the actual decision"
        rows.append({"kind": kind, "right": base == gold,
                     "status": "certified_memory_free" if cert is not None else
                     ("memory_dependent" if dep else "undetermined")})
    return summarize(rows)


def synthetic_hard(path, n, seed):
    """Hard-gated memory: exact certificates from closed gates, checked against memory patches."""
    import glassbox_world as W
    from vt.glassbox import memory_free_positions
    m = GlassBoxLM(GBConfig(vocab_size=W.VOCAB, block_size=W.BLOCK, mem_gate="hard"))
    m.load_state_dict(torch.load(path), strict=False)
    m.eval()
    world = W.World(1234)
    qs = W.qa_set(world, random.Random(seed), n, W.QA_RELS, 0, 4, test=True)
    sp = torch.nn.functional.softplus(m.alpha)

    def decide(x, pos):
        logits, k, _, _ = m(x)
        if (m.beta - sp * k[0, pos]) > 0:
            return W.UNKNOWN
        return int(W.VALUES[logits[0, pos, W.VALUES].argmax()])

    g = torch.Generator().manual_seed(seed)
    rows, open_frac = [], []
    for toks, pos, gold, kind in qs:
        x = torch.tensor([toks])
        with torch.no_grad():
            base = decide(x, pos)
            free = bool(memory_free_positions(m, x)[0, pos])
            open_frac.append(float((m.mem.trace["gate"][0, :pos + 1] > 0).float().mean()))
            changed = witness(m, x, pos, decide, base, g)
        if free and changed:
            raise AssertionError("a certified memory-free decision changed under a memory patch")
        rows.append({"kind": kind, "right": base == gold,
                     "status": "certified_memory_free" if free else ("memory_dependent" if changed else "undetermined")})
    s = summarize(rows)
    s["gate_open_fraction"] = sum(open_frac) / len(open_frac)
    return s


def chat(path, tok_path, n, seed):
    from tokenizers import Tokenizer
    import glassbox_chat as C
    tok = Tokenizer.from_file(tok_path)
    m = GlassBoxLM(GBConfig(vocab_size=tok.get_vocab_size(), block_size=256, n_layer=6, n_head=4, n_embd=192,
                            mem_layer=3, mem_keys=128, mem_topk=8, fixed_embed=False, mlp_ratio=1,
                            track_evidence=False, sharpness=True, mem_value_init=0.0))
    st = torch.load(path)
    m.load_state_dict(st.get("model", st), strict=False)
    m.eval()
    world = C.ChatWorld(n_ent=300)
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        prompt, _, gold, kind = world.question(rng, world.KINDS[i % len(world.KINDS)], C.QA_RELS, test=True)
        ids = tok.encode(prompt).ids
        P = len(ids)
        out = []
        with torch.no_grad():
            for _ in range(16):                                        # greedy reply
                logits, _, _, _ = m(torch.tensor([ids + out]), evidence=False)
                nxt = int(logits[0, -1].argmax())
                if nxt == 0 or "\n" in tok.decode(out + [nxt]):
                    break
                out.append(nxt)
            if not out:
                continue
            x = torch.tensor([ids + out])
            ll, lh, _, _, _, _ = memory_free_bounds(m, x)
            certs = certified_argmax(ll[0, P - 1:P - 1 + len(out)], lh[0, P - 1:P - 1 + len(out)])
        n_cert = int((certs == torch.tensor(out)).sum())
        text = tok.decode(out)
        abst = any(mk in text.lower() for mk in C.ABSTAIN_MARKERS)
        right = abst if gold is None else (not abst and gold in text)
        rows.append({"kind": kind, "right": right, "certified_tokens": n_cert, "tokens": len(out),
                     "status": "certified_memory_free" if n_cert == len(out) else "undetermined"})
    s = summarize(rows)
    s["certified_token_fraction"] = sum(r["certified_tokens"] for r in rows) / sum(r["tokens"] for r in rows)
    return s


def summarize(rows):
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        d = {"n": len(rs), "right": sum(r["right"] for r in rs) / len(rs)}
        for st in ("certified_memory_free", "memory_dependent", "undetermined"):
            d[st] = sum(r["status"] == st for r in rs) / len(rs)
        cert = [r for r in rs if r["status"] == "certified_memory_free"]
        d["right_when_certified"] = sum(r["right"] for r in cert) / len(cert) if cert else None
        out[kind] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-chat", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard", nargs="*", default=[], help="hard-gated checkpoints to certify exactly")
    ap.add_argument("--out", default="results/certify_attribution.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if a.hard:
        for path in a.hard:
            name = os.path.basename(path).split(".glassbox")[0]
            res[name] = synthetic_hard(path, a.n, a.seed)
            print(json.dumps({name: res[name]}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        return
    for name, path, layer in (("v4_zero_memory", "results/glassbox/v4_zero_memory.glassbox.pt", 2),
                              ("v5_memory_after_last_block", "results/glassbox/v5_late_only_familiarity.glassbox.pt", 3)):
        res[name] = synthetic(path, layer, a.n, a.seed)
        print(json.dumps({name: res[name]}), flush=True)
    res["chat_v2"] = chat("results/glassbox_chat_v2/glassbox.glassbox.pt", "results/glassbox_chat_v2/tokenizer.json",
                          a.n_chat, a.seed)
    print(json.dumps({"chat_v2": res["chat_v2"]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
