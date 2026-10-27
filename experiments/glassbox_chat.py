"""Glass-box LM as a general chat model: are its built-in readouts better hallucination
and error detectors than post-hoc interpretability on a standard GPT?

Training data (one token stream, BPE vocabulary trained here):
  * Alpaca instruction/response pairs (CC BY-NC 4.0, research use), in chat form:
    "User: <instruction>\\n<input>\\nAssistant: <output>"
  * WikiText-2 paragraphs (general text and knowledge)
  * a fact world in natural language: invented people, 8 relations, invented values.
    Known facts are stated in the corpus ("Zorvik's employer is Talmora."); chat
    questions about them are answered, questions about facts never stated are
    answered "I don't know ...", and a fact given in the user's message overrides
    memory. Some known facts are never asked in training, and two relations are
    never asked at all, so test questions are new; absent test facts are never seen.

Models (similar parameter counts): glassbox (vt.glassbox: evidence stream, small MLP
core, product-key memory, per-token self-knowledge k that sets the sharpness of
the next-token distribution, trained on its own token correctness) vs baseline
(vt.lm.GPT, ALiBi).

Evaluation:
  token level, held-out Alpaca responses and WikiText: AUROC for "the model's top
    prediction is the next token", from k, max softmax probability (MSP), label-free
    evidence/memory readouts, and post-hoc linear probes (best layer) on each model;
  chat level, fact questions: greedy responses scored right / abstained /
    hallucinated, and AUROC of response-level readouts for "this response is right".

    python experiments/glassbox_chat.py --out results/glassbox_chat/run.json
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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from interp_by_design import auroc, fit_probe  # noqa: E402
from typed_wikitext import articles  # noqa: E402
from vt.glassbox import GBConfig, GlassBoxLM, memory_free_positions  # noqa: E402
from vt.glassbox2 import G2Config, Glassbox2  # noqa: E402
from vt.glassbox3 import G3Config, Glassbox3  # noqa: E402
from vt.lm import GPT, GPTConfig  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RELS = ["birthplace", "employer", "favorite food", "pet", "native language", "home planet",
        "favorite color", "instrument"]
QA_RELS, HELD_RELS = list(range(6)), [6, 7]
N_ENT, N_VAL = 800, 40
IDK = "I don't know"


# ---------------------------------------------------------------- fact world

def pseudo(rng, n_syl):
    c, v = "bdfgklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) + (rng.choice("nrsl") if rng.random() < 0.3 else "")
                   for _ in range(n_syl)).capitalize()


class World:
    KINDS = ["ctx", "mem", "absent"]

    def __init__(self, seed=1234, n_ent=N_ENT):
        rng = random.Random(seed)
        names = set()
        while len(names) < n_ent + N_VAL * len(RELS):
            names.add(pseudo(rng, rng.choice([2, 3])))
        names = sorted(names)
        rng.shuffle(names)
        self.n_ent = n_ent
        self.names = set(names)
        self.ents = names[:n_ent]
        self.vals = [names[n_ent + r * N_VAL:n_ent + (r + 1) * N_VAL] for r in range(len(RELS))]
        self.value = [[rng.randrange(N_VAL) for _ in RELS] for _ in range(n_ent)]
        self.by = {s: [] for s in ("known_qa", "known_heldout", "unk_train", "unk_test")}
        for e in range(n_ent):
            for r in range(len(RELS)):
                u = rng.random()
                s = "known_qa" if u < 0.42 else "known_heldout" if u < 0.6 else "unk_train" if u < 0.8 else "unk_test"
                self.by[s].append((e, r))
        self.known = self.by["known_qa"] + self.by["known_heldout"]

    def v(self, e, r, k=None):
        return self.vals[r][self.value[e][r] if k is None else k]

    def statement(self, rng, e, r):
        E, R, V = self.ents[e], RELS[r], self.v(e, r)
        return rng.choice([f"{E}'s {R} is {V}.", f"The {R} of {E} is {V}.", f"{E} has {V} as {R}."])

    def question(self, rng, kind, rels, test):
        pool = {"mem": self.by["known_heldout" if test else "known_qa"],
                "absent": self.by["unk_test" if test else "unk_train"]}
        if kind == "ctx":
            e, r = rng.randrange(self.n_ent), rng.choice(rels)
        else:
            while True:
                e, r = rng.choice(pool[kind])
                if r in rels:
                    break
        E, R = self.ents[e], RELS[r]
        notes = []
        for _ in range(rng.randint(0, 2)):
            e2, r2 = rng.randrange(self.n_ent), rng.choice(rels)
            if (e2, r2) != (e, r):
                notes.append(f"{self.ents[e2]}'s {RELS[r2]} is {self.v(e2, r2, rng.randrange(N_VAL))}.")
        if kind == "ctx":
            gold = self.v(e, r, self.value[e][r] if rng.random() < 0.7 else rng.randrange(N_VAL))
            notes.insert(rng.randint(0, len(notes)), f"{E}'s {R} is {gold}.")
        else:
            gold = self.v(e, r) if kind == "mem" else None
        note = ("Note: " + " ".join(notes) + " ") if notes else ""
        prompt = f"User: {note}What is {E}'s {R}?\nAssistant:"
        answer = f" {E}'s {R} is {gold}." if gold else f" {IDK} {E}'s {R}."
        return prompt, answer, gold, kind


# Chat v2: natural phrasings per relation, facts told in the user's message, multi-turn
# dialogues, invented names that are new in every sample, varied abstentions.
ASK = {"birthplace": ["Where was {E} born?", "What is {E}'s birthplace?", "Do you know where {E} was born?"],
       "employer": ["Who does {E} work for?", "Where does {E} work?", "What company employs {E}?"],
       "favorite food": ["What is {E}'s favorite food?", "What does {E} like to eat most?"],
       "pet": ["What pet does {E} have?", "What is the name of {E}'s pet?"],
       "native language": ["What language does {E} speak natively?", "What is {E}'s native language?"],
       "home planet": ["Which planet is {E} from?", "What is {E}'s home planet?"],
       "favorite color": ["What is {E}'s favorite color?", "Which color does {E} like best?"],
       "instrument": ["What instrument does {E} play?", "Which instrument is {E} known for playing?"]}
SAY = {"birthplace": ["{E} was born in {V}.", "{E}'s birthplace is {V}."],
       "employer": ["{E} works for {V}.", "{E}'s employer is {V}."],
       "favorite food": ["{E}'s favorite food is {V}.", "{E} loves eating {V}."],
       "pet": ["{E} has a pet called {V}.", "{E}'s pet is {V}."],
       "native language": ["{E} speaks {V} natively.", "{E}'s native language is {V}."],
       "home planet": ["{E} comes from the planet {V}.", "{E}'s home planet is {V}."],
       "favorite color": ["{E}'s favorite color is {V}.", "{E} likes {V} best."],
       "instrument": ["{E} plays the {V}.", "{E}'s instrument is the {V}."]}
TELL = ["I read that {S}", "Fact: {S}", "My friend told me that {S}", "{S}"]
DONT = ["I don't know {E}'s {R}.", "I'm not sure. I have no information about {E}'s {R}.",
        "Sorry, I don't know that about {E}."]
ABSTAIN_MARKERS = ("don't know", "not sure", "no information")


class ChatWorld(World):
    KINDS = ["ctx", "mem", "absent", "absent_fresh", "ctx_fresh"]

    def statement(self, rng, e, r):
        return rng.choice(SAY[RELS[r]]).format(E=self.ents[e], V=self.v(e, r))

    def fresh_name(self, rng):
        while True:
            n = pseudo(rng, rng.choice([2, 3]))
            if n not in self.names:
                return n

    def question(self, rng, kind, rels, test):
        pool = {"mem": self.by["known_heldout" if test else "known_qa"],
                "absent": self.by["unk_test" if test else "unk_train"]}
        r = rng.choice(rels)
        if kind in ("ctx", "ctx_fresh", "absent_fresh"):
            E = self.ents[rng.randrange(self.n_ent)] if kind == "ctx" else self.fresh_name(rng)
            e = self.ents.index(E) if kind == "ctx" else None
        else:
            while True:
                e, r = rng.choice(pool[kind])
                if r in rels:
                    break
            E = self.ents[e]
        R = RELS[r]
        notes = []
        for _ in range(rng.randint(0, 2)):
            e2, r2 = rng.randrange(self.n_ent), rng.choice(rels)
            if self.ents[e2] != E:
                notes.append(rng.choice(SAY[RELS[r2]]).format(E=self.ents[e2], V=self.v(e2, r2, rng.randrange(N_VAL))))
        gold = None
        if kind in ("ctx", "ctx_fresh"):
            k = self.value[e][r] if kind == "ctx" and rng.random() < 0.7 else rng.randrange(N_VAL)
            gold = self.vals[r][k]
            notes.insert(rng.randint(0, len(notes)), rng.choice(SAY[R]).format(E=E, V=gold))
        elif kind == "mem":
            gold = self.v(e, r)
        told = " ".join(rng.choice(TELL).format(S=n) for n in notes)
        turns = ""
        if rng.random() < 0.3:                                  # an earlier exchange in the same chat
            while True:                                         # only relations this split may ask about
                e0, r0 = rng.choice(self.by["known_qa"])
                if r0 in rels:
                    break
            turns = (f"User: {rng.choice(ASK[RELS[r0]]).format(E=self.ents[e0])}\nAssistant: "
                     f"{rng.choice(SAY[RELS[r0]]).format(E=self.ents[e0], V=self.v(e0, r0))}\n")
        q = rng.choice(ASK[R]).format(E=E)
        prompt = f"{turns}User: {(told + ' ') if told else ''}{q}\nAssistant:"
        answer = " " + (rng.choice(SAY[R]).format(E=E, V=gold) if gold else rng.choice(DONT).format(E=E, R=R))
        return prompt, answer, gold, kind


# ---------------------------------------------------------------- corpus and tokenizer

def alpaca():
    d = json.load(open(os.path.join(ROOT, "data", "alpaca", "alpaca_data.json")))
    out = []
    for x in d:
        q = x["instruction"] + ("\n" + x["input"] if x["input"] else "")
        out.append((f"User: {q}\nAssistant:", " " + x["output"]))
    return out[:-1000], out[-1000:]


def corpus(world, rng, al_train, n_qa=40000, stmt_reps=8):
    docs = [p + a for p, a in al_train]
    wiki = [p for _, ps in articles(os.path.join(ROOT, "data", "wikitext-2", "train.txt")) for p in ps]
    docs += wiki
    for _ in range(stmt_reps):
        facts = world.known[:]
        rng.shuffle(facts)
        for i in range(0, len(facts), 6):
            docs.append(" ".join(world.statement(rng, e, r) for e, r in facts[i:i + 6]))
    kinds = world.KINDS
    for i in range(n_qa):
        p, a, _, _ = world.question(rng, kinds[i % len(kinds)], QA_RELS, test=False)
        docs.append(p + a)
    rng.shuffle(docs)
    return docs


def unsure_stream(stream, tok):
    """The training token stream with <unsure> (id V) inserted where an abstention answer begins,
    so Glassbox-3 trains on exactly the tokens the other models see, plus the marker."""
    V, cue = tok.get_vocab_size(), tok.encode("Assistant:").ids
    starts = [tok.encode(" " + t.split("{")[0].rstrip()).ids for t in DONT]
    s = stream.tolist()
    out, last, n = [], 0, 0
    for i in range(len(s) - 16):
        if s[i:i + len(cue)] == cue and any(s[i + len(cue):i + len(cue) + len(d)] == d for d in starts):
            j = i + len(cue)
            out += s[last:j] + [V]
            last, n = j, n + 1
    out += s[last:]
    return torch.tensor(out, dtype=torch.long), n


def train_tokenizer(docs, path, vocab=4096):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<eos>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(path)
    return tok


# ---------------------------------------------------------------- models

class Glass:
    def __init__(self, V, block, mem_init=1.0, hard_gate=0.0):
        self.hard_gate = hard_gate
        self.m = GlassBoxLM(GBConfig(vocab_size=V, block_size=block, n_layer=6, n_head=4, n_embd=192,
                                     mem_layer=3, mem_keys=128, mem_topk=8, fixed_embed=False,
                                     mlp_ratio=1, track_evidence=False, sharpness=True,
                                     mem_value_init=mem_init, mem_gate="hard" if hard_gate > 0 else "sigmoid"))

    def params(self):
        return [p for p in self.m.parameters() if p.requires_grad]

    def loss(self, x, y):
        logits, k, _, _ = self.m(x)
        lm = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        right = (logits.argmax(-1) == y).float()
        aux = F.binary_cross_entropy_with_logits(k, right)
        info = {"lm": lm.item(), "aux": aux.item()}
        loss = lm + 0.2 * aux
        if self.hard_gate > 0:                       # read memory only where it is needed
            gate = self.m.mem.trace["gate"]
            loss = loss + self.hard_gate * gate.mean()
            info["gate_open"] = (gate > 0).float().mean().item()
        return loss, info

    @torch.no_grad()
    def readouts(self, x, evidence=True):
        logits, k, e, hs = self.m(x, evidence=evidence)
        lp = F.log_softmax(logits, -1)
        ro = {"k": k, "msp": lp.max(-1).values.exp(), "match": self.m.mem.trace["match"]}
        if e is not None:
            ro.update(mem_mass=e[..., -1], e=e)
        if self.hard_gate > 0:
            ro["free"] = memory_free_positions(self.m, x)
            self.m(x, evidence=False)                # restore this pass's trace
        return lp, ro, hs


class Glass2:
    """Glassbox-2: exact sparse features, exact attribution, logged hard-gated memory."""

    def __init__(self, V, block, mem_l1=0.002, mem_layer=3, k_active=16, attn_topk=0, mem_gate="hard_st"):
        self.mem_l1 = mem_l1
        self.m = Glassbox2(G2Config(vocab_size=V, block_size=block, n_layer=6, n_head=4, n_embd=192,
                                    n_features=1024, k_active=k_active, mem_layer=mem_layer, mem_keys=128,
                                    mem_topk=8, attn_topk=attn_topk, mem_gate=mem_gate))

    def params(self):
        return list(self.m.parameters())

    def loss(self, x, y):
        logits, k = self.m(x)
        lm = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        right = (logits.argmax(-1) == y).float()
        aux = F.binary_cross_entropy_with_logits(k, right)
        gate = self.m.trace["mem"]["gate"]
        return lm + 0.2 * aux + self.mem_l1 * gate.mean(), {
            "lm": lm.item(), "aux": aux.item(), "gate_open": (gate > 0).float().mean().item()}

    @torch.no_grad()
    def readouts(self, x, evidence=True):
        logits, k = self.m(x)
        lp = F.log_softmax(logits, -1)
        tr = self.m.trace
        return lp, {"k": k, "msp": lp.max(-1).values.exp(), "match": tr["mem"]["match"],
                    "free": self.m.memory_free()}, tr["hs"]


class Glass3(Glass2):
    """Glassbox-3: Glassbox-2 plus an epinet whose epistemic uncertainty u raises an explicit
    <unsure> token (id V, one past the tokenizer) and lowers the commitment k; trained with
    self-consistency replay: prompts on which the model's own sampled answers disagree are
    trained to <unsure> + a hedge. No label says which prompts those are."""

    HEDGE = " I'm not sure."

    def __init__(self, V, block, mem_l1, mem_layer, k_active, attn_topk, sc=None):
        self.mem_l1, self.V = mem_l1, V
        self.m = Glassbox3(G3Config(vocab_size=V + 1, block_size=block, n_layer=6, n_head=4, n_embd=192,
                                    n_features=1024, k_active=k_active, mem_layer=mem_layer, mem_keys=128,
                                    mem_topk=8, attn_topk=attn_topk, unsure_id=V))
        self.sc = sc                     # self-consistency replay: dict(tok, world, start, every, prompts, k)
        self.sc_rng = random.Random(77)
        self.sc_last = 0.0

    def loss(self, x, y):
        total, info = super().loss(x, y)
        self.m.record_features()
        epi = self.m.epi_loss(y)
        info.update(epi=epi.item(), u=self.m.trace["u"].mean().item(), sc_disagree=self.sc_last)
        return total + epi, info

    @torch.no_grad()
    def _sample(self, ids, k, max_new=16):
        """k sampled replies (token lists) to one prompt."""
        tok, V = self.sc["tok"], self.V
        x = torch.tensor([ids] * k)
        out = [[] for _ in range(k)]
        done = [False] * k
        for _ in range(max_new):
            logits, _ = self.m(x)
            nxt = torch.multinomial(F.softmax(logits[:, -1].float(), -1), 1)
            for j in range(k):
                t = int(nxt[j])
                if done[j]:
                    continue
                if t == 0 or t == V or "\n" in tok.decode(out[j] + [t]):
                    done[j] = True
                    if t == V:
                        out[j] = [V]
                    continue
                out[j].append(t)
            if all(done):
                break
            x = torch.cat([x, nxt], 1)
        return out

    def replay(self, step):
        sc = self.sc
        if not sc or step < sc["start"] or step % sc["every"]:
            return None
        tok, rng = sc["tok"], self.sc_rng
        self.m.eval()
        seqs, P = [], []
        n_dis = 0
        for _ in range(sc["prompts"]):
            prompt = sc["world"].question(rng, rng.choice(sc["world"].KINDS), QA_RELS, test=False)[0]
            ids = tok.encode(prompt).ids
            outs = [tok.decode(o).strip().lower() if self.V not in o else "<unsure>" for o in self._sample(ids, sc["k"])]
            top = max(outs.count(o) for o in outs)
            if top / len(outs) < sc["agree"]:                    # its own answers disagree: teach <unsure>
                n_dis += 1
                seqs.append(ids + [self.V] + tok.encode(self.HEDGE).ids + [0])
                P.append(len(ids))
        self.m.train()
        self.sc_last = n_dis / sc["prompts"]
        info = {"sc_disagree": self.sc_last}
        if not seqs:
            return None, info
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L - 1, dtype=torch.long)
        y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
        for i, (sq, p) in enumerate(zip(seqs, P)):
            x[i, :len(sq) - 1] = torch.tensor(sq[:-1])
            y[i, p - 1:len(sq) - 1] = torch.tensor(sq[p:])       # loss on the reply only
        logits, _ = self.m(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        return loss, info

    @torch.no_grad()
    def readouts(self, x, evidence=True):
        lp, ro, hs = super().readouts(x, evidence)
        tr = self.m.trace
        ro.update(u=tr["u"], novelty=self.m.novelty(), p_unsure=lp[..., self.V].exp())
        return lp, ro, hs


class Base:
    def __init__(self, V, block):
        self.m = GPT(GPTConfig(vocab_size=V, block_size=block, n_layer=6, n_head=4, n_embd=192,
                               mlp_ratio=8, pos_encoding="alibi"))
        self._hs = []
        for b in self.m.blocks:
            b.register_forward_hook(lambda mod, i, o: self._hs.append(o))

    def params(self):
        return list(self.m.parameters())

    def loss(self, x, y):
        self._hs = []
        _, lm = self.m(x, y)
        return lm, {"lm": lm.item()}

    @torch.no_grad()
    def readouts(self, x, evidence=True):
        self._hs = []
        logits, _ = self.m(x)
        lp = F.log_softmax(logits, -1)
        return lp, {"msp": lp.max(-1).values.exp()}, list(self._hs)


# ---------------------------------------------------------------- training

def train(model, stream, steps, block, bs, log, ckpt):
    opt = torch.optim.AdamW(model.params(), lr=1e-3, weight_decay=0.01)
    step = 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt)
        missing, unexpected = model.m.load_state_dict(st["model"], strict=False)
        assert not unexpected and all(k.endswith(("lscale", "lbias", "usage")) for k in missing), (missing, unexpected)
        if not missing:
            opt.load_state_dict(st["opt"])
        step = st["step"]
    g = torch.Generator().manual_seed(step)
    t0 = time.time()
    while step < steps:
        model.m.train()
        for gr in opt.param_groups:
            gr["lr"] = 1e-3 * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        ix = torch.randint(len(stream) - block - 1, (bs,), generator=g)
        x = torch.stack([stream[i:i + block] for i in ix.tolist()])
        y = torch.stack([stream[i + 1:i + block + 1] for i in ix.tolist()])
        loss, info = model.loss(x, y)
        extra = model.replay(step) if hasattr(model, "replay") else None
        if extra is not None:
            if extra[0] is not None:
                loss = loss + extra[0]
            info.update(extra[1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.params(), 1.0)
        opt.step(); step += 1
        if step % 250 == 0 or step == steps:
            log({"step": step, **{k: round(v, 4) for k, v in info.items()},
                 "s_per_step": round((time.time() - t0) / 250, 3)})
            t0 = time.time()
            torch.save({"model": model.m.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt)


# ---------------------------------------------------------------- evaluation

def encode_pair(tok, prompt, answer, block):
    p, a = tok.encode(prompt).ids, tok.encode(answer).ids
    ids = (p + a + [0])[:block + 1]
    return ids, len(p)


@torch.no_grad()
def token_rows(model, tok, pairs, block):
    """Per response token: label = top prediction correct, readouts, hidden states."""
    rows = []
    for prompt, answer in pairs:
        ids, P = encode_pair(tok, prompt, answer, block)
        x, y = torch.tensor([ids[:-1]]), torch.tensor(ids[1:])
        lp, ro, hs = model.readouts(x)
        for t in range(P - 1, len(ids) - 1):
            r = {"ok": int(lp[0, t].argmax() == y[t]), "nll": -float(lp[0, t, y[t]]),
                 "msp": float(ro["msp"][0, t]), "h": [h[0, t].numpy() for h in hs]}
            if "k" in ro:
                r.update(k=float(ro["k"][0, t]), match=float(ro["match"][0, t]))
                if "e" in ro:
                    r.update(mem_mass=float(ro["mem_mass"][0, t]), prompt_mass=float(ro["e"][0, t, :P].sum()))
            if "free" in ro:
                r["memory_free"] = bool(ro["free"][0, t])
            if "u" in ro:
                r.update(u=float(ro["u"][0, t]), novelty=float(ro["novelty"][0, t]))
            rows.append(r)
    return rows


def best_probe(fit, val):
    yf, yv = np.array([r["ok"] for r in fit]), np.array([r["ok"] for r in val])
    best = None
    for L in range(len(fit[0]["h"])):
        pr, _ = fit_probe(np.stack([r["h"][L] for r in fit]), yf)
        a = auroc(pr(np.stack([r["h"][L] for r in val])), yv)
        if best is None or a > best[0]:
            best = (a, L, pr)
    return best


def token_metrics(rows, probe, L):
    y = np.array([r["ok"] for r in rows])
    d = {"tokens": len(rows), "top1_acc": float(y.mean()),
         "bits_per_token": float(np.mean([r["nll"] for r in rows]) / math.log(2)),
         "auroc_msp": auroc(np.array([r["msp"] for r in rows]), y),
         "auroc_probe": auroc(probe(np.stack([r["h"][L] for r in rows])), y)}
    if "k" in rows[0]:
        d["auroc_k"] = auroc(np.array([r["k"] for r in rows]), y)
        d["auroc_memory_match"] = auroc(np.array([r["match"] for r in rows]), y)
        if "prompt_mass" in rows[0]:
            d["auroc_prompt_mass"] = auroc(np.array([r["prompt_mass"] for r in rows]), y)
    if "u" in rows[0]:
        d["auroc_neg_u"] = auroc(-np.array([r["u"] for r in rows]), y)
        d["auroc_neg_novelty"] = auroc(-np.array([r["novelty"] for r in rows]), y)
    if "memory_free" in rows[0]:
        free = np.array([r["memory_free"] for r in rows])
        d["certified_memory_free_tokens"] = float(free.mean())
        d["top1_acc_certified"] = float(y[free].mean()) if free.any() else None
        d["top1_acc_uncertified"] = float(y[~free].mean()) if (~free).any() else None
    return d


@torch.no_grad()
def respond(model, tok, prompt, block, max_new=24):
    ids = tok.encode(prompt).ids
    P = len(ids)
    V = tok.get_vocab_size()                    # ids >= V are special: Glassbox-3's <unsure>
    dec = lambda t: tok.decode([i for i in t if i < V])  # noqa: E731
    for i in range(max_new):
        lp, _, _ = model.readouts(torch.tensor([ids[-block:]]), evidence=False)
        if i == 0:
            lp[0, -1, 0] = -float("inf")                         # respond with at least one token
        nxt = int(lp[0, -1].argmax())
        if nxt == 0:
            break
        ids.append(nxt)
        if "\n" in dec(ids[P:]):
            break
    x = torch.tensor([ids[-block:]])
    lp, ro, hs = model.readouts(x)
    off = len(ids) - x.shape[1]
    span = range(max(0, P - 1 - off), x.shape[1] - 1)
    tgt = x[0, 1:]
    feats = {"msp": float(np.mean([float(ro["msp"][0, t]) for t in span])),
             "nll": float(np.mean([-float(lp[0, t, tgt[t]]) for t in span])),
             "h": [h[0, P - 1 - off].numpy() for h in hs]}
    if "k" in ro:
        feats.update(k=float(np.mean([float(ro["k"][0, t]) for t in span])),
                     k_min=float(min(float(ro["k"][0, t]) for t in span)),
                     match=float(np.mean([float(ro["match"][0, t]) for t in span])))
        if "e" in ro:
            feats["prompt_mass"] = float(np.mean([float(ro["e"][0, t, :P - off].sum()) for t in span]))
    if "u" in ro:                                   # epistemic uncertainty where the reply starts
        feats.update(u_first=float(ro["u"][0, P - 1 - off]),
                     u_mean=float(np.mean([float(ro["u"][0, t]) for t in span])),
                     novelty_first=float(ro["novelty"][0, P - 1 - off]),
                     p_unsure_first=float(ro["p_unsure"][0, P - 1 - off]),
                     unsure_emitted=any(i >= V for i in ids[P:]))
    if "free" in ro:                                # every reply token provably made without memory
        feats["reply_memory_free"] = bool(all(bool(ro["free"][0, t]) for t in span))
        feats["ids"] = x
    text = dec(ids[P:]).split("\n")[0]
    return ("<unsure>" + text if any(i >= V for i in ids[P:]) else text), feats


def chat_eval(model, tok, world, rng, n, rels, block, probe=None):
    rows = []
    kinds = world.KINDS
    for i in range(n):
        prompt, _, gold, kind = world.question(rng, kinds[i % len(kinds)], rels, test=True)
        out, f = respond(model, tok, prompt, block)
        abst = any(m in out.lower() for m in ABSTAIN_MARKERS + ("<unsure>",))
        right = abst if gold is None else (not abst and gold in out)
        rows.append({"kind": kind, "right": right, "abstained": abst, "out": out, **f})
    d = {f"{k}_right": float(np.mean([r["right"] for r in rows if r["kind"] == k])) for k in kinds}
    if "reply_memory_free" in rows[0]:
        d["certified_memory_free_replies"] = {k: float(np.mean([r["reply_memory_free"] for r in rows if r["kind"] == k]))
                                              for k in kinds}
        cert = [r for r in rows if r["reply_memory_free"]]
        d["right_when_certified"] = float(np.mean([r["right"] for r in cert])) if cert else None
        d["certificate_soundness_violations"] = soundness_check(model, [r["ids"] for r in cert[:20]])
    d["hallucination_rate"] = float(np.mean([not r["abstained"] for r in rows if r["kind"] == "absent"]))
    if "absent_fresh" in kinds:
        d["hallucination_rate_fresh_names"] = float(np.mean([not r["abstained"] for r in rows
                                                             if r["kind"] == "absent_fresh"]))
    answerable = [r for r in rows if r["kind"] in ("ctx", "mem", "ctx_fresh")]
    d["over_refusal_rate"] = float(np.mean([r["abstained"] for r in answerable]))
    if "unsure_emitted" in rows[0]:
        d["unsure_emitted"] = {k: float(np.mean([r["unsure_emitted"] for r in rows if r["kind"] == k])) for k in kinds}
    y = np.array([r["right"] for r in rows], dtype=int)
    if 0 < y.sum() < len(y):
        d["auroc_msp"] = auroc(np.array([r["msp"] for r in rows]), y)
        d["auroc_neg_nll"] = auroc(-np.array([r["nll"] for r in rows]), y)
        if "k" in rows[0]:
            d["auroc_k_mean"] = auroc(np.array([r["k"] for r in rows]), y)
            d["auroc_k_min"] = auroc(np.array([r["k_min"] for r in rows]), y)
            d["auroc_memory_match"] = auroc(np.array([r["match"] for r in rows]), y)
            if "prompt_mass" in rows[0]:
                d["auroc_prompt_mass"] = auroc(np.array([r["prompt_mass"] for r in rows]), y)
        if "u_first" in rows[0]:
            for key in ("u_first", "u_mean", "novelty_first", "p_unsure_first"):
                d[f"auroc_neg_{key}"] = auroc(-np.array([r[key] for r in rows]), y)
        if probe is not None:
            d["auroc_probe"] = auroc(probe[1](np.stack([r["h"][probe[0]] for r in rows])), y)
    d["examples"] = [{k: r[k] for k in ("kind", "right", "out")} for r in rows[:6]]
    return d, rows


@torch.no_grad()
def soundness_check(model, seqs):
    """Certified positions must be bit-identical when the memory content is replaced: a
    check of the certificate's logic on the trained model. Returns the number of violations."""
    m = model.m
    bad = 0
    saved = m.mem.values.weight.detach().clone()
    g2 = isinstance(m, Glassbox2)
    run = (lambda x: m(x)[0]) if g2 else (lambda x: m(x, evidence=False)[0])
    for x in seqs:
        if g2:
            m(x)
            free = m.memory_free()
        else:
            free = memory_free_positions(m, x)
        base = run(x)
        m.mem.values.weight.copy_(torch.randn_like(saved) * 10)
        other = run(x)
        m.mem.values.weight.copy_(saved)
        bad += int(not torch.equal(base[free], other[free]))
    return bad


def chat_probe(model, tok, world, block, n=300):
    """Post-hoc probe for 'this response will be right', on the hidden state at the
    last prompt token; fitted on one set of questions, layer chosen on another."""
    _, fit = chat_eval(model, tok, world, random.Random(21), n, QA_RELS, block)
    _, sel = chat_eval(model, tok, world, random.Random(22), n, QA_RELS, block)
    best = None
    for L in range(len(fit[0]["h"])):
        pr, _ = fit_probe(np.stack([r["h"][L] for r in fit]), np.array([r["right"] for r in fit], dtype=int))
        a = auroc(pr(np.stack([r["h"][L] for r in sel])), np.array([r["right"] for r in sel], dtype=int))
        if best is None or a > best[0]:
            best = (a, L, pr)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glassbox,baseline")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-chat", type=int, default=300)
    ap.add_argument("--n-tok-examples", type=int, default=300)
    ap.add_argument("--out", default="results/glassbox_chat/run.json")
    ap.add_argument("--world", default="v1", choices=["v1", "v2"], help="v2: chatbot-style phrasings and dialogues")
    ap.add_argument("--n-ent", type=int, default=N_ENT)
    ap.add_argument("--n-qa", type=int, default=40000)
    ap.add_argument("--mem-init", type=float, default=1.0)
    ap.add_argument("--hard-gate", type=float, default=0.0, help="L1 weight of a hard memory gate (chat v3)")
    ap.add_argument("--g2-mem-layer", type=int, default=3, help="Glassbox-2: block after which memory is read")
    ap.add_argument("--g2-k-active", type=int, default=16)
    ap.add_argument("--g2-attn-topk", type=int, default=0)
    ap.add_argument("--g2-mem-l1", type=float, default=0.002)
    ap.add_argument("--g2-mem-gate", default="hard_st", choices=["hard", "hard_st"],
                    help="hard: plain clamp (gates freeze when saturated); hard_st: straight-through")
    ap.add_argument("--sc-every", type=int, default=25, help="Glassbox-3: self-consistency replay period")
    ap.add_argument("--sc-start", type=int, default=1000)
    ap.add_argument("--sc-prompts", type=int, default=8)
    ap.add_argument("--sc-k", type=int, default=4)
    ap.add_argument("--sc-agree", type=float, default=0.75)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--data-dir", default=None, help="tokenizer and token stream (default: next to --out)")
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(0)
    world = (ChatWorld if a.world == "v2" else World)(n_ent=a.n_ent)
    al_train, al_val = alpaca()
    data_dir = a.data_dir or os.path.dirname(a.out)
    os.makedirs(data_dir, exist_ok=True)
    tok_path = os.path.join(data_dir, "tokenizer.json")
    stream_path = os.path.join(data_dir, "stream.pt")
    if os.path.exists(stream_path):
        tok, stream = Tokenizer.from_file(tok_path), torch.load(stream_path)
    else:
        docs = corpus(world, rng, al_train, n_qa=a.n_qa)
        tok = train_tokenizer(docs, tok_path)
        ids = []
        for enc in tok.encode_batch(docs):
            ids += enc.ids + [0]
        stream = torch.tensor(ids, dtype=torch.long)
        torch.save(stream, stream_path)
    V = tok.get_vocab_size()
    print(json.dumps({"vocab": V, "train_tokens": len(stream)}), flush=True)
    if a.prepare_only:
        return
    wiki_val = [p for _, ps in articles(os.path.join(ROOT, "data", "wikitext-2", "valid.txt")) for p in ps]
    erng = random.Random(5)
    n = a.n_tok_examples
    al_fit, al_sel, al_test = al_val[:n], al_val[n:2 * n], al_val[2 * n:3 * n]
    wiki_test = [("", " " + p) for p in erng.sample(wiki_val, n // 2)]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for kind in a.models.split(","):
        if kind in res:
            continue
        train_stream = stream
        torch.manual_seed(0)                       # the same initialization on every run
        if kind == "glassbox3":
            up = os.path.join(data_dir, "stream_unsure.pt")
            if not os.path.exists(up):
                train_stream, n_uns = unsure_stream(stream, tok)
                torch.save(train_stream, up)
                print(json.dumps({"unsure_marked_answers": n_uns, "train_tokens": len(train_stream)}), flush=True)
            train_stream = torch.load(up)
            model = Glass3(V, a.block, a.g2_mem_l1, a.g2_mem_layer, a.g2_k_active, a.g2_attn_topk,
                           sc=dict(tok=tok, world=world, start=a.sc_start, every=a.sc_every, prompts=a.sc_prompts,
                                   k=a.sc_k, agree=a.sc_agree) if a.sc_every else None)
        else:
            model = (Glass(V, a.block, a.mem_init, a.hard_gate) if kind == "glassbox" else
                     Glass2(V, a.block, a.g2_mem_l1, a.g2_mem_layer, a.g2_k_active, a.g2_attn_topk, a.g2_mem_gate)
                     if kind == "glassbox2" else Base(V, a.block))
        ckpt = a.out.replace(".json", f".{kind}.pt")
        t0 = time.time()
        train(model, train_stream, a.steps, a.block, a.batch, lambda e: print(json.dumps({kind: e}), flush=True), ckpt)
        model.m.eval()
        r = {"train_s": time.time() - t0, "params": sum(p.numel() for p in model.params())}
        _, L, probe = best_probe(token_rows(model, tok, al_fit, a.block), token_rows(model, tok, al_sel, a.block))
        r["probe_layer"] = L
        r["tokens_alpaca"] = token_metrics(token_rows(model, tok, al_test, a.block), probe, L)
        r["tokens_wikitext"] = token_metrics(token_rows(model, tok, [(p or " ", s) for p, s in wiki_test],
                                                        a.block), probe, L)
        print(json.dumps({kind: {k: v for k, v in r.items()}}), flush=True)
        cprobe = chat_probe(model, tok, world, a.block, a.n_chat)
        r["chat_probe_layer"] = cprobe[0]
        for name, rels in (("chat_facts", QA_RELS), ("chat_facts_held_relations", HELD_RELS)):
            d, rows = chat_eval(model, tok, world, random.Random(11), a.n_chat, rels, a.block, cprobe)
            r[name] = d
            print(json.dumps({kind: {name: {k: v for k, v in d.items() if k != "examples"}}}), flush=True)
            for row in rows:
                row.pop("ids", None)
        r["samples"] = []
        for p, _ in al_test[:5]:
            out, f = respond(model, tok, p, a.block, max_new=48)
            r["samples"].append({"prompt": p, "response": out,
                                 **{k: v for k, v in f.items() if k not in ("h", "ids")}})
        res[kind] = r
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
