"""Claims in any reply, and conversation formats whose claims can be labelled automatically.

The correctness probe (vt/correctness.py) reads the hidden state at a claim's first token: the
first token of a name, number or date the reply asserts. This module finds those tokens in any
reply (short answer, paragraph, numbered list, reasoning chain, the last turn of a conversation)
and builds training conversations in several formats where each claim's truth is known:

  short       one question, a short answer                         (TriviaQA / NQ aliases)
  multi       k questions answered on numbered lines in one reply  (each line labelled)
  multiturn   the question asked after 1-3 earlier question/answer turns
  reasoning   think step by step, then a line "Answer: ..."        (TriviaQA aliases)
  math        GSM8K word problem, step by step, "Answer: <number>" (numeric match)

Every format reduces to: messages, and for each claim a reply-text span to search plus its gold
answers. `label_claims` turns a generated reply into (token index, label) pairs.
"""
import random
import re
import string

from vt.correctness import SKIP, norm

SYSTEMS = {
    "short": "You are a helpful assistant. Answer the question with a short answer.",
    "multi": "You are a helpful assistant. Answer each question on its own numbered line, briefly.",
    "multiturn": "You are a helpful assistant. Answer the question with a short answer.",
    "reasoning": "You are a helpful assistant. Think step by step, then give the final answer on a last line "
                 "that starts with 'Answer:'.",
    "math": "You are a helpful assistant. Solve the problem step by step, then give the final number on a last "
            "line that starts with 'Answer:'.",
}
FORMATS = tuple(SYSTEMS)


def word_starts(tok, ids):
    """Token indices that begin a word (a leading space or newline, or the first token)."""
    out = []
    for j in range(len(ids)):
        s = tok.decode(ids[j:j + 1])
        if j == 0 or s[:1].isspace() or (j > 0 and tok.decode(ids[j - 1:j]).endswith("\n")):
            out.append(j)
    return out


def claim_tokens(tok, ids, context="", lo=0, hi=None):
    """Indices (into ids[lo:hi]) of the first token of every capitalised or numeric word that does not
    already occur in `context` (the conversation so far): the reply's own assertions."""
    hi = len(ids) if hi is None else hi
    seen = set(norm(context).split())
    starts = [j for j in word_starts(tok, ids) if lo <= j < hi]
    out = []
    for a, b in zip(starts, starts[1:] + [hi]):
        w = tok.decode(ids[a:b]).strip().strip(string.punctuation + "\"'“”*#")
        if not w or w.lower() in SKIP or norm(w) in seen:
            continue
        if w[0].isupper() or w[0].isdigit():
            while a + 1 < b and not tok.decode(ids[a:a + 1]).strip():   # Qwen splits " 1958" into " ", "1", ...
                a += 1
            out.append(a)
    return out


def _num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return float(m[-1].replace(",", "").rstrip(".")) if m else None


def line_spans(tok, ids):
    """(start, end) token ranges of the reply's lines."""
    spans, start = [], 0
    for j in range(len(ids)):
        if "\n" in tok.decode(ids[j:j + 1]):
            if j > start:
                spans.append((start, j))
            start = j + 1
    if start < len(ids):
        spans.append((start, len(ids)))
    return spans


def answer_line(tok, ids):
    """Token range of the last line that starts with 'Answer:' (after any markdown), or None."""
    for a, b in reversed(line_spans(tok, ids)):
        if tok.decode(ids[a:b]).strip().lstrip("*# ").lower().startswith("answer"):
            return a, b
    return None


def label_claims(tok, item, ids):
    """[(token index, label 0/1)] for a generated reply `ids` to a built item. One labelled claim per
    question: the first claim token of the span that answers it. Lines/answers that are missing give
    no labels (they are not claims)."""
    fmt, out = item["format"], []
    if fmt in ("short", "multiturn"):
        c = claim_tokens(tok, ids, item["context"])
        text = tok.decode(ids)
        if c:
            out.append((c[0], int(any(norm(a) and norm(a) in norm(text) for a in item["golds"][0]))))
    elif fmt == "multi":
        lines = line_spans(tok, ids)
        numbered = {}
        for a, b in lines:
            m = re.match(r"\s*\**(\d+)[.)]", tok.decode(ids[a:b]))
            if m:
                numbered.setdefault(int(m.group(1)), (a, b))
        for k, golds in enumerate(item["golds"], 1):
            if k not in numbered:
                continue
            a, b = numbered[k]
            c = claim_tokens(tok, ids, item["context"], a, b)
            text = tok.decode(ids[a:b])
            if c:
                out.append((c[0], int(any(norm(g) and norm(g) in norm(text) for g in golds))))
    elif fmt in ("reasoning", "math"):
        span = answer_line(tok, ids)
        if span is None:
            return out
        a, b = span
        text = tok.decode(ids[a:b])
        c = claim_tokens(tok, ids, item["context"] + " answer", a, b)
        if fmt == "math":
            v, g = _num(text), _num(item["golds"][0][0])
            nums = [j for j in range(a, b) if any(ch.isdigit() for ch in tok.decode(ids[j:j + 1]))]
            if v is not None and nums:
                out.append((nums[0], int(abs(v - g) < 1e-6)))
        elif c:
            out.append((c[0], int(any(norm(g) and norm(g) in norm(text) for g in item["golds"][0]))))
    return out


def build(fmt, qa, rng, k=3):
    """One conversation item of format `fmt` from a pool of {"q", "aliases"} (or GSM8K
    {"q", "answer"}) items. Returns {"format", "messages", "golds", "context"}."""
    sys_ = {"role": "system", "content": SYSTEMS[fmt]}
    if fmt == "short":
        x = rng.choice(qa)
        msgs, golds = [sys_, {"role": "user", "content": x["q"]}], [x["aliases"]]
    elif fmt == "multi":
        xs = rng.sample(qa, k)
        body = "\n".join(f"{i}. {x['q']}" for i, x in enumerate(xs, 1))
        msgs, golds = [sys_, {"role": "user", "content": body}], [x["aliases"] for x in xs]
    elif fmt == "multiturn":
        xs = rng.sample(qa, rng.randint(2, 4))
        msgs = [sys_]
        for x in xs[:-1]:                               # earlier turns carry gold answers
            msgs += [{"role": "user", "content": x["q"]}, {"role": "assistant", "content": x["aliases"][-1]}]
        msgs.append({"role": "user", "content": xs[-1]["q"]})
        golds = [xs[-1]["aliases"]]
    elif fmt == "reasoning":
        x = rng.choice(qa)
        msgs, golds = [sys_, {"role": "user", "content": x["q"]}], [x["aliases"]]
    elif fmt == "math":
        x = rng.choice(qa)
        msgs, golds = [sys_, {"role": "user", "content": x["q"]}], [[x["answer"]]]
    else:
        raise ValueError(fmt)
    context = " ".join(m["content"] for m in msgs if m["role"] != "system")
    return {"format": fmt, "messages": msgs, "golds": golds, "context": context}


def make_items(fmt, pool, n, seed):
    rng = random.Random(seed)
    return [build(fmt, pool, rng) for _ in range(n)]
