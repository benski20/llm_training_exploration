"""A verdict on every claim of every reply, whatever the message was: the reply is cut into units (code
blocks, then sentences), each unit is sorted by what kind of claim it is, and each is checked the way that
fits it:

    code          parsed, never run (vt.codecheck)                     parses (not run) | does not parse
    arithmetic    re-computed exactly (vt.mathcheck)                   correct | wrong
    fact          judged against evidence by the chat model (three-way: supports / contradicts / does not say):
                  the documents the reply was written from (the user's text, sources found first), else a
                  lookup of its own (at most max_lookups a reply)     supported | contradicted | not in the
                                                                       sources | no evidence | not checked
    not a claim   an opinion, a suggestion, a question, fiction, a greeting: nothing to be true or false
                                                                       not a factual claim

Which sentences are facts is decided by a small classifier (ClaimTyper: a sentence embedding and a layer
trained on ClaimBuster's human labels and on factual vs creative answers, experiments/train_claim_type.py),
so this works on any reply: a wrong date inside a poem is still a fact, "I hope this helps" never is. Only
evidence that contradicts a claim counts against it: a line of fiction the typer takes for a fact reads
"not in the sources", not "wrong".

    Ledger(ClaimTyper(), verifier, retriever).build_batch([{"text": reply, "question": q, "documents": docs}])
    # -> [[{"claim", "type", "verdict", "basis"?, "sources"?, "p_fact"?}, ...]]
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

from vt.claim_verify import BOLD, sentences, source
from vt.codecheck import FENCE, check_code
from vt.mathcheck import check_arithmetic

NOT_A_CLAIM = "not a factual claim"


class ClaimTyper:
    """P(a sentence is a factual claim), from a sentence embedding (CPU) and a trained two-way layer."""

    def __init__(self, path=None, embedder=None):
        import numpy as np
        z = np.load(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "claim_type_head.npz"))
        self.W, self.b = z["W"], z["b"]
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(str(z["embed"]), device="cpu")
        self.embedder = embedder

    def p_fact(self, texts):
        import numpy as np
        if not texts:
            return np.zeros(0)
        X = self.embedder.encode([t[:1000] for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
        Z = X @ self.W + self.b
        P = np.exp(Z - Z.max(1, keepdims=True))
        return (P / P.sum(1, keepdims=True))[:, 1]


def units(text, explicit=True):
    """The reply cut into checkable units: [{"claim", "type": "code" | "arithmetic" | None, ...}] in order;
    type None: a sentence still to be sorted (fact or not). explicit=False also reads "17 times 23 is 391"
    as arithmetic (in an answer to a calculation; elsewhere "1914-1918 is 4 years" is prose)."""
    out = []
    for m in FENCE.finditer(text):
        block = m.group(0)
        c = check_code(block)
        u = {"claim": m.group(2).strip()[:300], "type": "code"}
        if not c:                                             # a language it does not parse
            u.update(verdict="not checked", basis=f"{m.group(1) or 'unlabeled'} code is not parsed")
        elif c[0]["ok"]:
            u["verdict"] = "parses (not run)"
        else:
            u.update(verdict="does not parse", basis=c[0]["error"])
        out.append(u)
    prose = FENCE.sub(" ", text.replace("[unsure:]", ""))
    prose = re.sub(r"(?<=\d)\s*\*\s*(?=[\d(])", " × ", prose)      # 12 * 4: a product, not markdown emphasis
    for s in sentences(prose):
        s = s.rstrip(BOLD)
        sums = check_arithmetic(s, explicit=explicit)
        if not sums and len(re.findall(r"[A-Za-z]", s)) < 3:  # bullets, numbers alone, stray symbols
            continue
        if sums:
            bad = [x for x in sums if not x["ok"]]
            out.append({"claim": s, "type": "arithmetic", "verdict": "wrong" if bad else "correct",
                        "basis": (f"{bad[0]['expression']} is {bad[0]['computed']:g}, not {bad[0]['stated']:g}" if bad
                                  else f"{len(sums)} step(s) re-computed")})
        else:
            out.append({"claim": s, "type": None})
    return out


class Ledger:
    def __init__(self, typer, verifier=None, retriever=None, fact_threshold=0.5, max_lookups=4, max_facts=12):
        """verifier: a vt.claim_verify.ClaimVerifier (its judge3 does the judging; None: facts are typed but
        not judged). retriever: looks up facts of replies written without documents."""
        self.typer, self.verifier, self.retriever = typer, verifier, retriever
        self.fact_threshold, self.max_lookups, self.max_facts = fact_threshold, max_lookups, max_facts

    def build_batch(self, items):
        """items: [{"text", "question", "documents"?, "source_label"?, "fact_threshold"?, "calculation"?}] -> one
        ledger each (calculation: the message asked for one)."""
        ledgers = [units(it["text"], explicit=not it.get("calculation")) for it in items]
        todo = [(k, u) for k, led in enumerate(ledgers) for u in led if u["type"] is None]
        probs = self.typer.p_fact([u["claim"] for _, u in todo])
        facts = []
        for (k, u), p in zip(todo, probs):
            u["p_fact"] = round(float(p), 3)
            if p >= items[k].get("fact_threshold", self.fact_threshold):
                u["type"] = "fact"
                facts.append((k, u))
            else:
                u.update(type="not a claim", verdict=NOT_A_CLAIM)
        per = {}
        for k, u in facts:                                     # the most factual first when there are many
            per.setdefault(k, []).append(u)
        checked = []
        for k, fs in per.items():
            fs.sort(key=lambda u: -u["p_fact"])
            for u in fs[self.max_facts:]:
                u.update(verdict="not checked", basis="too many claims in one reply")
            checked += [(k, u) for u in fs[:self.max_facts]]
        self._evidence(items, checked)
        self._judge(items, checked)
        return ledgers

    def _evidence(self, items, checked):
        """Each fact's passages: its reply's documents, else a lookup of its own (in parallel, a few a reply)."""
        lookups, used = [], {}
        for k, u in checked:
            docs = items[k].get("documents")
            if docs:
                u["_passages"], u["_label"] = docs, items[k].get("source_label") or "the sources"
            elif self.retriever is not None and used.get(k, 0) < self.max_lookups:
                used[k] = used.get(k, 0) + 1
                lookups.append(u)
                u["_label"] = "the sources found"
            else:
                u["_passages"] = []
                u["_label"] = None if self.retriever is None else "limit"

        def find(u):
            k = next(k for k, v in checked if v is u)
            try:
                return self.retriever(f"{items[k].get('question') or ''} {u['claim']}".strip())
            except Exception as e:                             # a failed lookup: no evidence, not a failed reply
                print(f"ledger lookup failed: {e!r}", flush=True)
                return []
        if lookups:
            with ThreadPoolExecutor(max_workers=min(8, len(lookups))) as pool:
                for u, ps in zip(lookups, pool.map(find, lookups)):
                    u["_passages"] = ps

    def _judge(self, items, checked):
        ask = [(k, u) for k, u in checked if u.get("_passages") and self.verifier is not None]
        probs = self.verifier.judge3([(items[k].get("question") or "", u["claim"], u["_passages"]) for k, u in ask]) \
            if ask else []
        for (k, u), (a, b, n) in zip(ask, probs):
            where = u["_label"]
            u.update(p_supported=round(a, 4), p_contradicted=round(b, 4), p_not_stated=round(n, 4),
                     sources=sorted({source(p) for p in u["_passages"]}) if where == "the sources found" else [],
                     verdict="supported" if a >= self.verifier.supported else
                     "contradicted" if b >= self.verifier.supported else
                     f"not in {where}" if n >= 0.5 else "unclear")
        for _, u in checked:
            if "verdict" not in u:
                why = u.get("_label")
                u["verdict"] = ("not checked" if why == "limit" or self.verifier is None or why is None and
                                self.retriever is None else "no evidence")
                if why == "limit":
                    u["basis"] = "lookup limit for one reply"
            u.pop("_passages", None)
            u.pop("_label", None)


def overall(ledger):
    """One verdict for a reply from its ledger: anything shown wrong makes it likely wrong; facts all
    supported (and nothing wrong) likely right; facts unconfirmed uncertain or unverified; no factual claims
    at all (a poem, advice, code that parses) says so."""
    def of(t):
        return [u for u in ledger if u["type"] == t]
    wrong = [u for u in ledger if u["verdict"] in ("wrong", "does not parse", "contradicted")]
    if wrong:
        u = wrong[0]
        what = {"wrong": "arithmetic", "does not parse": "code", "contradicted": "the sources contradict"}[u["verdict"]]
        return {"verdict": "likely wrong", "doubt": 1.0,
                "basis": f"{what}: {u['claim'][:120]}" + (f" ({u['basis']})" if u.get("basis") else "")}
    facts = of("fact")
    supported = [u for u in facts if u["verdict"] == "supported"]
    if facts and len(supported) == len(facts):
        return {"verdict": "likely right", "doubt": 0.1, "basis": f"all {len(facts)} factual claim(s) supported"}
    if facts:
        return {"verdict": "uncertain" if supported else "unverified", "doubt": 0.5 if supported else None,
                "basis": f"{len(supported)} of {len(facts)} factual claim(s) supported"}
    if of("arithmetic"):
        return {"verdict": "likely right", "doubt": 0.0, "basis": f"arithmetic re-computed ({len(of('arithmetic'))})"}
    return {"verdict": "no factual claims", "doubt": None,
            "basis": "code parses (not run)" if of("code") else "opinion, advice or creative writing"}
