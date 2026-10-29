"""Claim verification against evidence: extract a reply's check-worthy claims, look each up, and let a
chat model (the one already serving the chat) judge whether the evidence supports it.

    v = ClaimVerifier(model, tok, Wikipedia(full_text=True, pages=5, rerank=...))
    v.verify("Who designed the Eiffel Tower?", reply_text)
    # -> [{"claim", "verdict": "supported" | "unsupported" | "no evidence", "p_supported", "sources"}]

The internal probe says how sure the model was while it wrote; this says whether a source backs the
claim. They fail differently: the probe misses confident errors, evidence misses what a lookup cannot
find (experiments/evidence_verify.py compares them on the 32B's labeled replies).
"""
import re

from vt.claims import prompt_of

SYSTEM = "You check claims against evidence. Answer only Yes or No."


BOLD = "\x01"                                                   # marks a sentence the model put in bold


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        bold = bool(re.search(r"\*\*[^*]+\*\*", p))              # the model bolds its answer
        p = re.sub(r"[*_#`>]+", "", p).strip(" -•\t")             # markdown
        if len(p) > 3:
            out.append(p + BOLD if bold else p)
    return out


def check_worthy(s):
    """Names something, gives a number, or holds what the model put in bold (its answer: "**fresco**")."""
    w = s.rstrip(BOLD).split()
    return s.endswith(BOLD) or any(ch.isdigit() for ch in s) or any(t[:1].isupper() for t in w[1:])


def source(passage):
    """A short name for a passage: its Wikipedia title, a web page's title and site, else its opening words."""
    m = re.match(r"(.+?) \(([^()]+)\): ", passage)
    if m:
        return m.group(1) if m.group(2) == "Wikipedia" else f"{m.group(1)[:60]} ({m.group(2)})"
    return passage[:60].rsplit(" ", 1)[0] + "…" if len(passage) > 60 else passage


def gather(pairs, retriever, max_claims=3):
    """The lookup half, no model needed: [(question, reply text)] -> [{"question", "claims": [{"claim",
    "passages"}]}], every claim looked up in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    claims = [(k, q, c) for k, (q, text) in enumerate(pairs)
              for c in [s.rstrip(BOLD) for s in sentences(text) if check_worthy(s)][:max_claims]]
    found = []
    if claims:
        with ThreadPoolExecutor(max_workers=min(8, len(claims))) as pool:
            found = list(pool.map(lambda kqc: retriever(f"{kqc[1]} {kqc[2]}"), claims))
    out = [{"question": q, "claims": []} for q, _ in pairs]
    for (k, _, c), ps in zip(claims, found):
        out[k]["claims"].append({"claim": c, "passages": ps})
    return out


THREE_WAY = ("What does the evidence say about the claim?\nA) it supports the claim\nB) it contradicts the claim\n"
             "C) it does not say whether the claim is true\nAnswer with one letter.")


class ClaimVerifier:
    def __init__(self, model, tok, retriever, max_claims=3, supported=0.6, unsupported=0.2, max_tokens=1500,
                 batch=4, three_way=False):
        """supported / unsupported: P(Yes) at or above which a claim is supported, below which it is not
        (in between: the evidence is unclear, reported as "unclear"). three_way: ask whether the evidence
        supports, contradicts or does not say: evidence that is merely off the point then reads "not in the
        sources" instead of "unsupported", and the verdict leans on the model's own doubt there."""
        self.model, self.tok, self.retriever = model, tok, retriever
        self.max_claims, self.supported, self.unsupported = max_claims, supported, unsupported
        self.max_tokens, self.batch, self.three_way = max_tokens, batch, three_way

        def ids(*words):
            return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes, self.no = ids("Yes", " Yes"), ids("No", " No")
        self.letters = [ids(c, " " + c) for c in "ABC"]

    def claims(self, text):
        return [s.rstrip(BOLD) for s in sentences(text) if check_worthy(s)][:self.max_claims]

    def _prompt(self, question, claim, passages):
        ev = "\n\n".join(p[:900] for p in passages)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {question}\nClaim: {claim}\n\n"
                                            "Does the evidence support the claim? Answer Yes or No."}]
        return prompt_of(self.tok, msgs, thinking=False)

    def _prompt3(self, question, claim, passages):
        ev = "\n\n".join(p[:900] for p in passages)
        msgs = [{"role": "system", "content": "You check claims against evidence. Answer only A, B or C."},
                {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {question}\nClaim: {claim}\n\n{THREE_WAY}"}]
        return prompt_of(self.tok, msgs, thinking=False)

    def _choices(self, prompts, options):
        """The model's probabilities of each option (lists of token ids) as the next token of each prompt;
        batched, left-padded, one forward pass per batch."""
        import torch
        dev = next(self.model.parameters()).device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        out = []
        for i in range(0, len(prompts), self.batch):
            ids = [self.tok(p, add_special_tokens=False).input_ids[-self.max_tokens:] for p in prompts[i:i + self.batch]]
            n = max(len(x) for x in ids)
            x = torch.full((len(ids), n), pad, dtype=torch.long)
            m = torch.zeros((len(ids), n), dtype=torch.long)
            for b, t in enumerate(ids):
                x[b, n - len(t):], m[b, n - len(t):] = torch.tensor(t), 1
            with torch.no_grad():
                z = self.model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits[:, -1].float()
            lg = torch.stack([torch.logsumexp(z[:, o], -1) for o in options], -1)
            out += [[float(v) for v in row] for row in torch.softmax(lg, -1).cpu()]
        return out

    def judge(self, items):
        """items: [(question, claim, passages)] -> P(Yes) each."""
        return [p[0] for p in self._choices([self._prompt(*it) for it in items], [self.yes, self.no])]

    def judge3(self, items):
        """items: [(question, claim, passages)] -> (P(supports), P(contradicts), P(does not say)) each."""
        return [tuple(p) for p in self._choices([self._prompt3(*it) for it in items], self.letters)]

    def ask_yes_no(self, system, questions):
        """P(Yes) for plain yes/no questions (e.g. grading an answer against reference answers)."""
        return [p[0] for p in self._choices([prompt_of(self.tok, [{"role": "system", "content": system},
                                                                  {"role": "user", "content": q}], thinking=False)
                                             for q in questions], [self.yes, self.no])]

    def verify(self, question, text):
        return self.verify_batch([(question, text)])[0]

    def verify_batch(self, pairs):
        """[(question, reply text)] -> one verification list each: every claim of every reply looked up in
        parallel (the lookups wait on the network), then all of them judged in batched forward passes."""
        return self.judge_gathered(gather(pairs, self.retriever, self.max_claims))

    def judge_gathered(self, gathered):
        """gather()'s output -> one verification list per reply (the judging half: needs the model only)."""
        out = [[] for _ in gathered]
        claims = [(k, g["question"], c["claim"]) for k, g in enumerate(gathered) for c in g["claims"]]
        found = [c["passages"] for g in gathered for c in g["claims"]]
        if not claims:
            return out
        ask = [(q, c, ps) for (_, q, c), ps in zip(claims, found) if ps]
        probs = iter((self.judge3(ask) if self.three_way else self.judge(ask)) if ask else ())
        for (k, _, c), ps in zip(claims, found):
            if not ps:
                out[k].append({"claim": c, "verdict": "no evidence", "p_supported": None, "sources": []})
                continue
            p = next(probs)
            row = {"claim": c, "sources": sorted({source(s) for s in ps})}
            if self.three_way:
                a, b, n = p
                verdict = "supported" if a >= self.supported else "unsupported" if b >= self.supported else \
                    "not in the sources" if n >= 0.5 else "unclear"
                row.update(verdict=verdict, p_supported=round(a, 4), p_contradicted=round(b, 4),
                           p_not_stated=round(n, 4))
            else:
                verdict = "supported" if p >= self.supported else "unsupported" if p < self.unsupported else "unclear"
                row.update(verdict=verdict, p_supported=round(p, 4))
            out[k].append(row)
        return out
