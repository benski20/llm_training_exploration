"""A verdict on every claim of every reply, one way for every query: each sentence (and code block) of the
reply is put to the same judge, the chat model already loaded, with the same question and all the evidence
there is:

    the conversation       what the user said and gave (a pasted article, documents): checked against first
    exact checks           any arithmetic the sentence states, re-computed (vt.mathcheck); any code, parsed
                           (vt.codecheck). Tools only add evidence; the judge decides what it means.
    a search               only for what the first pass leaves unsettled, then judged once more

    Does the evidence  A) support it   B) contradict it   C) not settle it   or is it  D) not a statement of fact
                       (an opinion, a suggestion, a question, a greeting, fiction)?

One forward pass per sentence (no generation), batched; lookups only for unsettled facts. A poem's lines come
back D and are never looked up, a wrong date inside one comes back C, is looked up, and is contradicted,
"17 x 23 = 401" is contradicted by its own re-computation, and a summary is checked against the text it
summarizes, without anything deciding beforehand what kind of message or sentence it was.

    Ledger(model, tok, retriever).build_batch([{"text": reply, "messages": conversation}])
    # -> [[{"claim", "verdict", "p": {...}, "evidence"?, "sources"?}, ...]]
"""
import re
from concurrent.futures import ThreadPoolExecutor

from vt.claim_verify import BOLD, choice_probs, option_ids, sentences, source
from vt.claims import prompt_of
from vt.codecheck import FENCE, check_code
from vt.mathcheck import check_arithmetic

VERDICTS = ("supported", "contradicted", "unverified", "not a factual claim")
QUESTION = ("Does the evidence support the statement?\n"
            "A) yes, the evidence supports it\n"
            "B) no, the evidence contradicts it (or shows it is wrong)\n"
            "C) the evidence does not settle it\n"
            "D) it is not a statement of fact: an opinion, a suggestion, an instruction, a question, a greeting, "
            "or fiction\n"
            "Answer with one letter.")
SYSTEM = "You check each statement of a chatbot's reply against evidence. Answer only A, B, C or D."


def units(text, max_units=24):
    """The reply cut into what can be judged: code blocks whole, then its sentences."""
    out = [{"claim": m.group(2).strip()[:1500], "code": m.group(0)} for m in FENCE.finditer(text)]
    prose = FENCE.sub(" ", text.replace("[unsure:]", ""))
    prose = re.sub(r"(?<=\d)\s*\*\s*(?=[\d(])", " × ", prose)        # 12 * 4: a product, not markdown emphasis
    for s in sentences(prose):
        s = s.rstrip(BOLD)
        if len(re.findall(r"[A-Za-z0-9]", s)) >= 3:
            out.append({"claim": s})
    return out[:max_units]


def exact_checks(unit):
    """Evidence no model can doubt: the arithmetic the statement states, re-computed; its code, parsed."""
    lines = []
    if "code" in unit:
        for c in check_code(unit["code"]):
            lines.append(f"This {c['language']} parses." if c["ok"] else
                         f"This {c['language']} does not parse: {c['error']}.")
    else:
        for m in check_arithmetic(unit["claim"]):
            lines.append(f"{m['expression']} = {m['computed']:g} exactly (the statement says {m['stated']:g}).")
    return lines


def conversation(messages, max_chars=6000):
    """The user's side of the conversation, newest last, as evidence."""
    said = [m["content"] for m in messages if m.get("role") == "user"]
    text = "\n\n".join(said)
    return text[-max_chars:]


class Ledger:
    def __init__(self, model, tok, retriever=None, batch=4, max_tokens=2000, max_lookups=3, max_units=24):
        self.model, self.tok, self.retriever = model, tok, retriever
        self.batch, self.max_tokens, self.max_lookups, self.max_units = batch, max_tokens, max_lookups, max_units
        self.letters = [option_ids(tok, c) for c in "ABCD"]

    def _prompt(self, said, documents, unit, found=()):
        parts = []
        if said:
            parts.append(f"The conversation (the user's messages):\n{said}")
        if documents:
            parts.append("Documents the user gave:\n" + "\n\n".join(d[:1500] for d in documents))
        checks = exact_checks(unit)
        if checks:
            parts.append("Exact checks:\n" + "\n".join(checks))
        if found:
            parts.append("Found by search:\n" + "\n\n".join(p[:900] for p in found))
        what = f"Code from the reply:\n{unit['code']}" if "code" in unit else f"Statement from the reply: {unit['claim']}"
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Evidence:\n\n" + "\n\n".join(parts or ["(none)"]) +
                 f"\n\n{what}\n\n{QUESTION}"}]
        return prompt_of(self.tok, msgs, thinking=False)

    def _judge(self, prompts):
        return choice_probs(self.model, self.tok, prompts, self.letters, self.batch, self.max_tokens)

    def build_batch(self, items):
        """items: [{"text": reply, "messages": conversation, "documents"?: the user's, "found"?: passages already
        looked up, "question"?}] -> a ledger each."""
        ledgers, todo = [], []
        for k, it in enumerate(items):
            said = conversation(it.get("messages") or ([{"role": "user", "content": it["question"]}]
                                                        if it.get("question") else []))
            it["_said"] = said
            us = units(it["text"], self.max_units)
            ledgers.append(us)
            todo += [(k, u) for u in us]
        if not todo:
            return ledgers
        first = self._judge([self._prompt(items[k]["_said"], items[k].get("documents"), u, items[k].get("found") or ())
                             for k, u in todo])
        for (_, u), p in zip(todo, first):
            self._set(u, p)
        self._search(items, [(k, u) for k, u in todo if u["verdict"] == "unverified"])
        for it in items:
            it.pop("_said", None)
        return ledgers

    def _set(self, u, p, sources=None):
        j = max(range(4), key=lambda i: p[i])
        u.update(verdict=VERDICTS[j], p={v: round(float(x), 4) for v, x in zip(VERDICTS, p)})
        checks = exact_checks(u)
        if checks:
            u["evidence"] = checks
        if sources:
            u["sources"] = sources

    def _search(self, items, unsettled):
        """Look up what the conversation and exact checks left unsettled (a few a reply), and judge it again."""
        if self.retriever is None or not unsettled:
            return
        per, picked = {}, []
        for k, u in unsettled:
            if per.get(k, 0) < self.max_lookups:
                per[k] = per.get(k, 0) + 1
                picked.append((k, u))

        def find(ku):
            k, u = ku
            q = items[k].get("question") or (items[k]["_said"].rsplit("\n\n", 1)[-1][:300])
            try:
                return self.retriever(f"{q} {u['claim'][:300]}".strip())
            except Exception as e:                                   # no lookup: the claim stays unverified
                print(f"ledger lookup failed: {e!r}", flush=True)
                return []
        with ThreadPoolExecutor(max_workers=min(8, len(picked))) as pool:
            found = list(pool.map(find, picked))
        again = [(k, u, ps) for (k, u), ps in zip(picked, found) if ps]
        if not again:
            return
        probs = self._judge([self._prompt(items[k]["_said"], items[k].get("documents"), u, ps) for k, u, ps in again])
        for (_, u, ps), p in zip(again, probs):
            self._set(u, p, sorted({source(x) for x in ps}))


def overall(ledger):
    """One verdict for a reply from its ledger: anything contradicted makes it likely wrong; every factual
    statement supported likely right; some unsettled uncertain (or unverified when none is supported); no
    factual statement at all says so."""
    if not ledger:
        return None
    wrong = [u for u in ledger if u["verdict"] == "contradicted"]
    if wrong:
        u = wrong[0]
        why = f" ({u['evidence'][0]})" if u.get("evidence") else ""
        return {"verdict": "likely wrong", "doubt": round(u["p"]["contradicted"], 4),
                "basis": f"contradicted: {u['claim'][:120]}{why}"}
    facts = [u for u in ledger if u["verdict"] != "not a factual claim"]
    supported = [u for u in facts if u["verdict"] == "supported"]
    if not facts:
        return {"verdict": "no factual claims", "doubt": None, "basis": "opinion, advice, wording or fiction"}
    if len(supported) == len(facts):
        return {"verdict": "likely right", "doubt": 0.1, "basis": f"all {len(facts)} factual statement(s) supported"}
    return {"verdict": "uncertain" if supported else "unverified", "doubt": 0.5 if supported else None,
            "basis": f"{len(supported)} of {len(facts)} factual statement(s) supported"}
