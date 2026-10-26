"""Typed generation: a GPT-style text generator whose facts are checkable by construction.

Every output word has one of two types, fixed by the architecture:

  GEN   lowercase words and punctuation, generated freely from the byte vocabulary;
  COPY  a *fact word* (any word containing an uppercase letter, a digit or a
        non-ASCII character: names, numbers, dates, titles), emitted only as a
        pointer to a whole word of the source documents.

The vocabulary path assigns zero probability to uppercase letters, digits,
non-ASCII bytes and control bytes, so a fact word can enter the output only through a COPY
action. A fact the sources don't support is written as an explicit hole
(HOLE, rendered "[?]") instead of being invented: training data replaces
unsupported fact words by HOLE, so the model learns to mark them.

Generation returns a transcript (output offset, source offset, length) for
every copied word, and `check` verifies output against sources exactly,
without the model: every fact word in the output must be a whole-word copy
of the source at the stated offset.

Text is normalised so case carries meaning: a sentence-initial capital on a
common word ("The", "It") is lowercased, and `render` restores it.
"""
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lm import GPT, GPTConfig

HOLE, UNK, SEP, NL, SP = 1, 2, 3, 10, 32
FACT_BYTES = torch.tensor([b for b in range(256) if 65 <= b <= 90 or 48 <= b <= 57 or b >= 128])
# bytes the vocabulary path may never emit: fact bytes, and control bytes other than HOLE, UNK, newline
NO_GEN = torch.tensor([b for b in range(256) if b in FACT_BYTES.tolist() or
                       (b < 32 and b not in (HOLE, UNK, NL)) or b == 127])
_END = (".", "?", "!")


def is_fact(word: bytes) -> bool:
    return any(65 <= b <= 90 or 48 <= b <= 57 or b >= 128 for b in word)


# ---------------------------------------------------------------- text normalisation

def common_lowercase(paragraphs, min_count=3):
    """Words whose lowercase form is at least as frequent as their capitalised
    form in mid-sentence position: their sentence-initial capital carries no
    information and is removed by `normalize`."""
    low, cap = Counter(), Counter()
    for par in paragraphs:
        start = True
        for w in par.split(" "):
            if not start and w[:1].isupper():
                cap[w.lower()] += 1
            elif w.islower():
                low[w] += 1
            start = w in _END
    return {w for w, c in low.items() if c >= min_count and c >= cap[w]}


def normalize(par: str, common) -> bytes:
    out, start = [], True
    for w in par.split(" "):
        if w == "<unk>":
            out.append(chr(UNK))
        else:
            if start and w.isascii() and w.isalpha() and w[0].isupper() and \
                    (len(w) == 1 or w[1:].islower()) and w.lower() in common:
                w = w.lower()
            out.append(w)
        start = w in _END
    return " ".join(out).encode("utf-8")


def render(b: bytes) -> str:
    words, start, out = b.decode("utf-8", "replace").split(" "), True, []
    for w in words:
        out.append(w[:1].upper() + w[1:] if start and w[:1].islower() else w)
        start = w in _END
    return " ".join(out).replace(chr(HOLE), "[?]").replace(chr(UNK), "<unk>")


def fact_words(source: bytes):
    return {w for w in source.split() if is_fact(w)}


def holeify(target: bytes, source: bytes) -> bytes:
    """Replace every fact word of `target` that is not a word of `source` by HOLE."""
    have = fact_words(source)
    return b" ".join(w if not is_fact(w) or w in have else bytes([HOLE]) for w in target.split(b" "))


# ---------------------------------------------------------------- exact checker

def source_sentences(source: bytes):
    """Sentence index of every byte of `source`. Sentence 0 is the title (before
    " : "); later sentences end at the words ".", "?", "!"."""
    sid, k, i = [], 0, 0
    words = source.split(b" ")
    title_end = words.index(b":") if b":" in words else -1
    for n, w in enumerate(words):
        sid.extend([k] * (len(w) + 1))
        if n == title_end or w.decode("utf-8", "replace") in _END:
            k += 1
    return sid[:len(source)]


def check(source: bytes, output: bytes, transcript):
    """Exact, model-free verification. transcript: list of (out_off, src_off, length).

    ok:         every fact word of `output` is covered by a transcript entry that
                copies a whole word of `source` byte for byte.
    citations:  per output sentence, the source sentences its facts were copied
                from (the title, sentence 0, is shared context and not counted).
    cosourced:  every output sentence cites at most one source sentence."""
    entries = {o: (s, n) for o, s, n in transcript}
    sid = source_sentences(source)
    bad, n_fact, i, cites, cur = [], 0, 0, [], set()
    for w in output.replace(b"\n", b" ").split(b" "):
        if is_fact(w):
            n_fact += 1
            s, n = entries.get(i, (None, None))
            whole = s is not None and (s == 0 or source[s - 1:s] in (b" ", b"\n")) and \
                (s + n == len(source) or source[s + n:s + n + 1] in (b" ", b"\n"))
            if not (whole and n == len(w) and source[s:s + n] == w):
                bad.append(w.decode("utf-8", "replace"))
            elif sid[s] > 0:
                cur.add(sid[s])
        if w.decode("utf-8", "replace") in _END:
            cites.append(sorted(cur)); cur = set()
        i += len(w) + 1
    if cur:
        cites.append(sorted(cur))
    return {"ok": not bad, "fact_words": n_fact, "unsupported": bad,
            "holes": output.count(bytes([HOLE])), "citations": cites,
            "cosourced": all(len(c) <= 1 for c in cites)}


# ---------------------------------------------------------------- batching

def encode(examples, block):
    """examples: list of (source bytes, holeified target bytes). Returns tensors for
    teacher-forced training; position i predicts y[i] = x[i + 1]."""
    B, T = len(examples), block
    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.zeros(B, T, dtype=torch.long)
    region = torch.zeros(B, T, dtype=torch.bool)      # y[i] is a target byte
    boundary = torch.zeros(B, T, dtype=torch.bool)    # x[i] is a word boundary: COPY allowed
    inner = torch.zeros(B, T, dtype=torch.bool)       # y[i] is a non-initial byte of a fact word
    cand_id = torch.full((B, T), -1, dtype=torch.long)  # source fact word ending at j
    tgt_id = torch.full((B, T), -1, dtype=torch.long)   # fact word starting at y[i]
    for b, (src, tgt) in enumerate(examples):
        seq = src + bytes([SEP]) + tgt + bytes([NL])
        if len(seq) > T + 1:
            raise ValueError("example longer than block")
        ids = {}
        j = 0
        for w in src.split(b" "):
            if is_fact(w):
                cand_id[b, j + len(w) - 1] = ids.setdefault(w, len(ids))
            j += len(w) + 1
        L = len(src) + 1
        x[b, :len(seq) - 1] = torch.tensor(list(seq[:-1]))
        y[b, :len(seq) - 1] = torch.tensor(list(seq[1:]))
        region[b, L - 1:len(seq) - 1] = True
        k = L
        for w in tgt.split(b" "):
            if is_fact(w):
                if w not in ids:
                    raise ValueError("target fact word not in source: holeify first")
                tgt_id[b, k - 1] = ids[w]
                inner[b, k:k + len(w) - 1] = True
            k += len(w) + 1
        boundary[b] = (x[b] == SP) | (x[b] == SEP)
    return x, y, dict(region=region, boundary=boundary, inner=inner, cand_id=cand_id, tgt_id=tgt_id)


# ---------------------------------------------------------------- model

class TypedLM(nn.Module):
    """GPT body + masked vocabulary path + whole-word pointer over source fact words."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gpt = GPT(cfg)
        d = cfg.n_embd
        self.q, self.k = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self._h = None
        self.gpt.ln_f.register_forward_hook(lambda m, i, o: setattr(self, "_h", o))

    def heads(self, x, cand):
        logits, _ = self.gpt(x)
        h = self._h
        logits = logits.masked_fill(torch.isin(torch.arange(256), NO_GEN), -1e9)
        logp_gen = F.log_softmax(logits, -1)
        att = (self.q(h) @ self.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        logp_ptr = F.log_softmax(att.masked_fill(~cand[:, None, :], -1e4), -1)
        g = self.gate(h).squeeze(-1)
        return logp_gen, logp_ptr, F.logsigmoid(g), F.logsigmoid(-g)

    def token_logp(self, x, y, m):
        """log p of each target byte (0 for non-initial fact-word bytes, which a COPY determines)."""
        cand = m["cand_id"] >= 0
        logp_gen, logp_ptr, lg, l1g = self.heads(x, cand)
        allow = m["boundary"] & cand.any(-1, keepdim=True)
        gen = logp_gen.gather(-1, y[..., None]).squeeze(-1) + torch.where(allow, l1g, 0.0)
        match = (m["cand_id"][:, None, :] == m["tgt_id"][:, :, None]) & (m["tgt_id"][:, :, None] >= 0)
        copy = lg + torch.logsumexp(torch.where(match, logp_ptr, -1e4), -1)
        tok = torch.where(m["tgt_id"] >= 0, copy, gen)
        return torch.where(m["inner"], 0.0, tok)

    def loss(self, x, y, m):
        tok = self.token_logp(x, y, m)
        w = m["region"] & ~m["inner"]
        return -(tok * w).sum() / w.sum()

    @torch.no_grad()
    def generate(self, source: bytes, max_bytes=200, temperature=0.8, generator=None, cosource=False):
        """Sample a continuation. Returns (output bytes, transcript).
        cosource: once an output sentence has copied a fact from a source sentence,
        its later copies are restricted to that sentence (and the title)."""
        self.eval()
        sid = source_sentences(source)
        starts, j = {}, 0
        for w in source.split(b" "):
            if is_fact(w):
                starts[j + len(w) - 1] = (j, len(w))
            j += len(w) + 1
        prefix = list(source) + [SEP]
        T = self.gpt.config.block_size
        cand = torch.zeros(1, T, dtype=torch.bool)
        for e in starts:
            cand[0, e] = True
        out, transcript, cur = [], [], None
        while len(out) < max_bytes and len(prefix) + len(out) < T:
            x = torch.tensor([prefix + out])
            lgen, lptr, lg, l1g = self.heads(x, cand[:, :x.shape[1]])
            t = x.shape[1] - 1
            p_gen = (lgen[0, t] / temperature).softmax(-1)
            # a copy starts a word: only at the start of the output or after a space
            at_boundary = bool(starts) and (not out or out[-1] == SP)
            if at_boundary:
                g = lg[0, t].exp()
                ok = cand[0, :x.shape[1]].clone()
                if cosource and cur is not None:
                    for e in starts:
                        ok[e] = ok[e] and sid[e] in (0, cur)
                p_ptr = (lptr[0, t] / temperature).softmax(-1) * ok
                p = torch.cat([(1 - g) * p_gen, g * p_ptr / p_ptr.sum()])
            else:
                p = p_gen
            a = int(torch.multinomial(p, 1, generator=generator))
            if a < 256:
                out.append(a)
                if a == NL:
                    break
                if a == SP and len(out) >= 2 and out[-2] in b".?!" and (len(out) == 2 or out[-3] == SP):
                    cur = None                           # a new output sentence
            else:
                s, n = starts[a - 256]
                transcript.append((len(out), s, n))
                if sid[s] > 0:
                    cur = sid[s]
                out.extend(source[s:s + n])
                out.append(SP)                           # a copied word is always a whole word
        return bytes(out).rstrip(b"\n"), transcript
