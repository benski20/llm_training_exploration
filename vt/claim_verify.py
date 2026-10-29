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


def sentences(text):
    text = re.sub(r"[*_#`>]+", "", text)                           # markdown
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 3]


def check_worthy(s):
    """Names something or gives a number: a word capitalized mid-sentence, or a digit."""
    w = s.split()
    return any(ch.isdigit() for ch in s) or any(t[:1].isupper() for t in w[1:])


class ClaimVerifier:
    def __init__(self, model, tok, retriever, max_claims=3, supported=0.6, unsupported=0.2, max_tokens=1500,
                 batch=4):
        """supported / unsupported: P(Yes) at or above which a claim is supported, below which it is not
        (in between: the evidence is unclear, reported as "unclear")."""
        self.model, self.tok, self.retriever = model, tok, retriever
        self.max_claims, self.supported, self.unsupported = max_claims, supported, unsupported
        self.max_tokens, self.batch = max_tokens, batch
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")})

    def claims(self, text):
        return [s for s in sentences(text) if check_worthy(s)][:self.max_claims]

    def _prompt(self, question, claim, passages):
        ev = "\n\n".join(p[:900] for p in passages)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {question}\nClaim: {claim}\n\n"
                                            "Does the evidence support the claim? Answer Yes or No."}]
        return prompt_of(self.tok, msgs, thinking=False)

    def judge(self, items):
        """items: [(question, claim, passages)] -> P(Yes) each; batched, left-padded, one forward pass."""
        import torch
        dev = next(self.model.parameters()).device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        out = []
        for i in range(0, len(items), self.batch):
            ids = [self.tok(self._prompt(*it), add_special_tokens=False).input_ids[-self.max_tokens:]
                   for it in items[i:i + self.batch]]
            n = max(len(x) for x in ids)
            x = torch.full((len(ids), n), pad, dtype=torch.long)
            m = torch.zeros((len(ids), n), dtype=torch.long)
            for b, t in enumerate(ids):
                x[b, n - len(t):], m[b, n - len(t):] = torch.tensor(t), 1
            with torch.no_grad():
                z = self.model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits[:, -1].float()
            y, no = torch.logsumexp(z[:, self.yes], -1), torch.logsumexp(z[:, self.no], -1)
            out += [float(p) for p in torch.sigmoid(y - no).cpu()]
        return out

    def verify(self, question, text):
        from concurrent.futures import ThreadPoolExecutor
        cs = self.claims(text)
        if not cs:
            return []
        with ThreadPoolExecutor(max_workers=len(cs)) as pool:   # lookups wait on the network: in parallel
            found = list(zip(cs, pool.map(lambda c: self.retriever(f"{question} {c}"), cs)))
        ask = [(question, c, ps) for c, ps in found if ps]
        probs = iter(self.judge(ask)) if ask else iter(())
        out = []
        for c, ps in found:
            if not ps:
                out.append({"claim": c, "verdict": "no evidence", "p_supported": None, "sources": []})
                continue
            p = next(probs)
            verdict = "supported" if p >= self.supported else "unsupported" if p < self.unsupported else "unclear"
            out.append({"claim": c, "verdict": verdict, "p_supported": round(p, 4),
                        "sources": sorted({s.split(" (Wikipedia)")[0] for s in ps})})
        return out
