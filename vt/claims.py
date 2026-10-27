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


def prompt_of(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def claim_features(model, tok, prompt_ids, reply_ids, positions, layers, batch=8):
    """One forward pass per conversation (batched, left-padded). For each requested claim position j
    (an index into the reply) returns the hidden states (len(layers), d) at the position that predicts
    it, and output signals: the claim token's probability, the least likely token in its word, and the
    reply's mean token log-probability. positions: list (per item) of lists of reply indices."""
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for i in range(0, len(prompt_ids), batch):
        P_, R_, J_ = prompt_ids[i:i + batch], reply_ids[i:i + batch], positions[i:i + batch]
        seqs = [p + (r or [int(tok.eos_token_id)]) for p, r in zip(P_, R_)]
        M = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), M), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), M), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, M - len(s):] = torch.tensor(s)
            mask[b, M - len(s):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                      output_hidden_states=True)
        hs = torch.stack([o.hidden_states[L + 1] for L in layers], 1)              # (B, nL, M, d)
        for b, (p, r, js) in enumerate(zip(P_, R_, J_)):
            r = r or [int(tok.eos_token_id)]
            off = M - len(seqs[b]) + len(p) - 1                                    # predicts r[0]
            with torch.no_grad():
                lp = F.log_softmax(o.logits[b, off:off + len(r)].float(), -1)
                tl = lp[torch.arange(len(r), device=dev), torch.tensor(r, device=dev)].cpu()
            starts = word_starts(tok, r) + [len(r)]
            rows = []
            for j in js:
                end = next((s for s in starts if s > j), len(r))
                rows.append({"h": hs[b, :, off + j].half().cpu(), "claim_p": float(tl[j].exp()),
                             "word_min_lp": float(tl[j:end].min()), "reply_mean_lp": float(tl.mean()),
                             "position": j / max(1, len(r))})
            out.append(rows)
    return out


OUTPUT_KEYS = ["claim_p", "word_min_lp", "reply_mean_lp"]
