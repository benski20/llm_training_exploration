"""Minimal pairs: does the omni stack catch a SINGLE subtle error in an otherwise correct reply?

Benchmarks average over natural errors, many of them gross. Here every test item is a pair: a reply the corpus
labels CORRECT (experiments/omni_corpus.py build, md5 checked) and a twin that differs in exactly one claim. The task
(question, documents) is identical within a pair. A scorer is right on a pair when it scores the twin higher.

Originals (label 0, no labelled error), long ones preferred (sampling weighted by length):
    longform   FELM math / reasoning / science / wk / writing (every segment labelled correct; closed book)
    rag        RAGTruth QA + summary (no hallucinated span), HaluEval dialogue right responses (with knowledge)
    reasoning  ProcessBench GSM8K / MATH / OlympiadBench / Omni-MATH solutions with label -1 (all steps correct)

Perturbations (one per twin; the edit site is drawn from an early / middle / late third of the reply, uniformly):
    number      rule: one figure / year changed by a small plausible amount (year +-1..3, small integer +-1,
                larger x0.75..1.5 at the same precision, decimal last digit); with documents, the new figure is
                absent from them
    qualifier   rule: one swap from a fixed list (always->usually, never->rarely, all->some, some of->all of,
                most->few, many->few, increase<->decrease, before<->after, more than<->less than, higher<->lower,
                larger<->smaller, largest<->smallest, maximum<->minimum, above<->below, earlier<->later, ...)
    negation    rule: add "not" after a copula / modal (is -> is not, can -> cannot) or remove one (is not -> is,
                doesn't -> does, cannot -> can); "not only / not just" skipped
    entity      LLM (Qwen3-8B, greedy, no thinking): rewrite one marked sentence replacing exactly one name / place /
                organisation / work with a plausible one of the same type; kept only if the word diff is one
                contiguous span of <= 4 words with a capitalised new word absent from the task, documents and reply
    addition    LLM (same): one new sentence with a specific plausible detail not in the source (closed book:
                plausible but false / unverifiable), inserted as its own sentence after a drawn sentence; kept only
                if it is one sentence of 6-45 words that is not in the source
    reasoning   rule, ProcessBench only: "intermediate" = the last "= N" result of one non-final step changed (later
                steps untouched: an arithmetic slip that does not propagate); "final" = the boxed answer changed
Sentences containing "language model" (refusal boilerplate) are never edited. Every pair records the edited unit
and the edit position as a fraction of the reply (character offset of the edit / reply length).

Scoring (Kaggle 2x T4), every reply of every pair plus a per-domain profile sample:
    reader     vt.omni.Reader("Qwen/Qwen3-8B") features, shipped head heads/omni_qwen3_8b.npz (OmniVerifier):
               reply head, worst unit, the reader's own judge, exact arithmetic flags; per-unit scores
    judges     vt.omni.JudgeScorer prompt, one pass each: Qwen3-14B and phi-4 (4-bit nf4, loaded as in
               kaggle/omni/judge_calibration.py: fp16 compute, one GPU each, fallback to unsloth pre-quantised)
    NLI        vt.omni.NLIScorer (HHEM-2.1-open + MiniCheck-RoBERTa-L; MiniCheck on the first 4 sentences, as shipped),
               on the GPU, replies with documents; also MiniCheck over all sentences as a diagnostic
    profile    zero labels: per source domain, a seeded sample of the corpus domain's replies (labels unused), read
               with the same stack, sets every standardisation (OmniVerifier.profile), as the headline stack does
    z-sum      OmniVerifier._combine with that profile: r + u + 2 j + 2 flags + NLI + 3 judge_qwen14 + 3 judge_phi4

    python kaggle/omni/nuance_pairs.py --build-local --corpus /home/user/omni/corpus.jsonl   # rule pairs, no models
    python kaggle/kgrun.py --slug omni-nuance --script kaggle/omni/nuance_pairs.py --bundle vt/omni.py \
        vt/mathcheck.py vt/correctness.py vt/checkers/__init__.py vt/checkers/business_record.py \
        experiments/omni_corpus.py heads/omni_qwen3_8b.npz --out results/kaggle/omni-nuance --wait [--args "--smoke 3"]
    python kaggle/omni/nuance_pairs.py --analyse results/kaggle/omni-nuance --report results/omni/nuance_pairs.json
"""
import argparse
import difflib
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np

SRC = "/kaggle/working/src"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
TMP = "/tmp/nuance"
SELF = os.path.join(SRC, "kaggle", "omni", "nuance_pairs.py")
CORPUS_MD5 = "b598848a44e70308b64ae5f0072b6eb0"
# the shipped head (17.9 MB) is too large for a Kaggle script bundle: fetched from the public repo at the commit that
# shipped it, md5-checked against heads/omni_qwen3_8b.npz
HEAD_URL = "https://raw.githubusercontent.com/benski20/llm_training_exploration/b284b37/heads/omni_qwen3_8b.npz"
HEAD_MD5 = "2a654983e708ff963cb52f9d7565cac2"
JUDGES = {"qwen14": ("Qwen/Qwen3-14B", "unsloth/Qwen3-14B-bnb-4bit"),
          "phi4": ("microsoft/phi-4", "unsloth/phi-4-bnb-4bit")}
SOURCES = {"longform": ["felm_math", "felm_reasoning", "felm_science", "felm_wk", "felm_writing"],
           "rag": ["ragtruth_qa", "ragtruth_summary", "halueval_dialogue"],
           "reasoning": ["pb_gsm8k", "pb_math", "pb_olympiad", "pb_omnimath"]}
FMT = {d: f for f, ds in SOURCES.items() for d in ds}
# pairs per perturbation type and format
TARGET = {"number": {"longform": 45, "rag": 45}, "qualifier": {"longform": 45, "rag": 45},
          "negation": {"longform": 45, "rag": 45}, "entity": {"longform": 45, "rag": 45},
          "addition": {"longform": 45, "rag": 45}, "reasoning_intermediate": {"reasoning": 80},
          "reasoning_final": {"reasoning": 60}}
LLM_TYPES = ("entity", "addition")
MIN_LEN = {"longform": 150, "rag": 40, "reasoning": 0}
PROFILE_N = 100
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def sentence_spans(text):
    out, start = [], 0
    for m in re.finditer(r"(?<!\d)[.!?](?=\s)|\n+", text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


# ---------------------------------------------------------------- edit sites (rule-based)

NUM = re.compile(r"(?<![\w.,$\\{^_/])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w]|\.\d|,\d{3})")
QUAL = [("always", "usually"), ("usually", "always"), ("never", "rarely"), ("rarely", "never"),
        ("all the", "some of the"), ("all of", "some of"), ("all these", "some of these"),
        ("all those", "some of those"), ("all", "some"), ("some of", "all of"), ("most of", "few of"), ("many", "few"), ("few", "many"),
        ("increase", "decrease"), ("decrease", "increase"), ("increases", "decreases"), ("decreases", "increases"),
        ("increased", "decreased"), ("decreased", "increased"), ("increasing", "decreasing"),
        ("decreasing", "increasing"), ("before", "after"), ("after", "before"), ("more than", "less than"),
        ("less than", "more than"), ("fewer than", "more than"), ("higher", "lower"), ("lower", "higher"),
        ("larger", "smaller"), ("smaller", "larger"), ("largest", "smallest"), ("smallest", "largest"),
        ("maximum", "minimum"), ("minimum", "maximum"), ("above", "below"), ("below", "above"),
        ("earlier", "later"), ("later", "earlier"), ("greater than", "less than"), ("positive", "negative"),
        ("negative", "positive"), ("older", "younger"), ("younger", "older"), ("longer", "shorter"),
        ("shorter", "longer"), ("rise", "fall"), ("rises", "falls"), ("often", "rarely"),
        ("frequently", "rarely"), ("commonly", "rarely")]
QUAL_SKIP = re.compile(r"after all|at all|all of a sudden|all in all|not only|if only|only if|above all|"
                       r"later on|more or less|first and foremost|all right", re.I)
NEG_ADD = re.compile(r"\b(is|are|was|were|can|could|will|would|should|must|might|has been|have been|had been)"
                     r" (?!not\b|n't|also\b|only\b|just\b|to\b|no\b|nothing\b|none\b|never\b)(?=[a-z])")
NEG_DEL = re.compile(r"\b(is|are|was|were|could|will|would|should|must|might|may|has|have|had|does|do|did) not "
                     r"(?!only\b|just\b|necessarily\b)")
NEG_CONTR = {"isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were", "doesn't": "does", "don't": "do",
             "didn't": "did", "can't": "can", "won't": "will", "shouldn't": "should", "couldn't": "could",
             "cannot": "can", "wouldn't": "would", "hasn't": "has", "haven't": "have"}
ENTITY_WORD = re.compile(r"(?<=[a-z,;:] )[A-Z][a-z]+(?: [A-Z][a-z]+)*")


def new_number(s, rng):
    """a small plausible change of the figure s (same format)."""
    comma = "," in s
    v = s.replace(",", "")
    if "." in v:
        dec = len(v.split(".")[1])
        x = float(v)
        step = 10 ** -dec
        d = max(step, round(abs(x) * rng.choice([0.1, 0.15, 0.2, 0.3]) / step) * step)
        y = x + d if rng.random() < 0.5 or x - d <= 0 else x - d
        out = f"{y:.{dec}f}"
    else:
        x = int(v)
        if 1000 <= x <= 2100 and len(v) == 4 and not comma:               # a year
            y = x + rng.choice([-3, -2, -1, 1, 2, 3])
        elif x < 2:
            return None
        elif x < 10:
            y = x + rng.choice([-1, 1])
        elif x < 100:
            y = x + rng.choice([-1, 1]) * max(1, round(x * rng.choice([0.1, 0.2, 0.25])))
        else:
            f = rng.choice([0.75, 0.8, 0.9, 1.1, 1.2, 1.25, 1.5])
            mag = 10 ** max(0, len(str(x)) - 2)
            y = int(round(x * f / mag) * mag)
            if y == x:
                y = x + mag
        out = f"{y:,}" if comma else str(y)
    return None if out == s or out.startswith("-") else out


def unit_sentences(text, a, b):
    """sentence spans inside a unit [a, b) (absolute offsets)."""
    return [(a + s, a + e) for s, e in sentence_spans(text[a:b])]


def is_list_marker(text, i):
    ls = text.rfind("\n", 0, i) + 1
    return bool(re.fullmatch(r"\s*(?:\d+[.)]|step|\(?[a-z]\))\s*", text[ls:i], re.I)) and \
        bool(re.match(r"\d+[.)]", text[i:i + 4]))


def sites(kind, r, rng):
    """[(unit_idx, a, b, replacement)] candidate single-claim edits of reply r (absolute character spans)."""
    t, doc = r["response"], (r.get("doc") or "").lower()
    question = judge_view(r)[0] if r.get("doc") else r["messages"][-1]["content"]
    out = []
    for k, (ua, ub) in enumerate(r["units"]):
        u = t[ua:ub]
        if "language model" in u.lower() or not u.strip():
            continue
        if kind == "number":
            for m in NUM.finditer(u):
                a = ua + m.start(1)
                if is_list_marker(t, a) or re.match(r"\s*words\b", u[m.end(1):]):
                    continue
                new = new_number(m.group(1), rng)
                if new is None or (doc and new.lower() in doc) or re.search(r"(?<![\d.,])" + re.escape(m.group(1)) +
                                                                            r"(?![\d])", question):
                    continue                                                 # a figure restated from the task: skip
                out.append((k, a, ua + m.end(1), new))
        elif kind == "qualifier":
            if QUAL_SKIP.search(u):
                continue
            for w, rep in QUAL:
                for m in re.finditer(r"\b" + re.escape(w) + r"\b", u, re.I):
                    if m.start() == 0 or u[:m.start()].rstrip().endswith((".", "\n")):
                        continue                                             # sentence-initial: often discourse use
                    if re.search(r"(?:\ba|\bhow|\bas|\bthe|\bso|\btoo)\s+$", u[:m.start()], re.I) and \
                            w in ("few", "many", "all"):
                        continue                                             # "a few", "how many", "the all-..."
                    if w == "all" and not re.match(r"\s+[a-z]+s\b", u[m.end():]):
                        continue                                             # "all" + plural noun only
                    if m.group(0)[0].isupper():
                        continue                                             # mid-sentence capital: a title
                    if any(x[1] <= ua + m.start() < x[2] or ua + m.start() <= x[1] < ua + m.end() for x in out):
                        continue                                             # overlaps a longer phrase already taken
                    out.append((k, ua + m.start(), ua + m.end(), rep))
        elif kind == "negation":
            for m in NEG_ADD.finditer(u):
                v = m.group(1)
                if re.search(r"\b(?:here|what|that|which|who)\s+$", u[:m.start()], re.I):
                    continue                                         # "here is the summary", relative clauses
                rep = "cannot " if v == "can" else (v.replace(" ", " not ", 1) + " " if " " in v else v + " not ")
                out.append((k, ua + m.start(), ua + m.end(), rep))
            for m in NEG_DEL.finditer(u):
                out.append((k, ua + m.start(), ua + m.end(), m.group(1) + " "))
            for w, rep in NEG_CONTR.items():
                for m in re.finditer(r"\b" + re.escape(w) + r"\b", u):
                    if not re.match(r"\s+(only|just|necessarily)\b", u[m.end():]):
                        out.append((k, ua + m.start(), ua + m.end(), rep))
        elif kind == "reasoning_intermediate":
            if k == len(r["units"]) - 1 or "boxed" in u:
                continue
            ms = [m for m in re.finditer(r"=\s*\$?\s*(-?\d+(?:\.\d+)?)(?![\d.,A-Za-z(^\\]|\s*(?:[+\-*/^×÷(]|\\times|\\cdot|\\div|\\frac))", u)]
            if ms:
                m = ms[-1]
                new = new_number(m.group(1).lstrip("-"), rng)
                if new is not None:
                    new = ("-" if m.group(1).startswith("-") else "") + new
                    out.append((k, ua + m.start(1), ua + m.end(1), new))
        elif kind == "reasoning_final":
            m = re.search(r"\\boxed\{", u)
            if m:
                n = NUM.search(u, m.end())
                if n and n.start() < (u.find("}", m.end()) if "}" in u[m.end():] else len(u)):
                    new = new_number(n.group(1), rng)
                    if new is not None:
                        out.append((k, ua + n.start(1), ua + n.end(1), new))
        elif kind in ("entity", "addition"):
            for a, b in unit_sentences(t, ua, ub):
                s = t[a:b]
                nw = len(s.split())
                if "language model" in s.lower() or not 6 <= nw <= 70:
                    continue
                if kind == "entity" and not ENTITY_WORD.search(s):
                    continue
                out.append((k, a, b, None))
    return out


def third(pos):
    return "early" if pos < 1 / 3 else ("middle" if pos < 2 / 3 else "late")


def pick_site(cands, t, rng):
    """an early / middle / late third drawn uniformly among those with candidates, then a site in it."""
    by = {}
    for c in cands:
        by.setdefault(third(c[1] / max(len(t), 1)), []).append(c)
    z = rng.choice(sorted(by))
    return rng.choice(by[z])


def apply_edit(r, k, a, b, new, insert=False):
    """the twin: response, units and messages (a copy of the reply embedded in the task is replaced too)."""
    t = r["response"]
    units = [tuple(u) for u in r["units"]]
    if insert:                                     # a new sentence after position b, as its own unit
        ins = " " + new
        L = len(ins)
        t2 = t[:b] + ins + t[b:]
        out_units = []
        for s, e in units:
            if e <= b:
                out_units.append((s, e))
            elif s >= b:
                out_units.append((s + L, e + L))
            else:                                  # b inside a unit: split it around the new sentence
                out_units += [(s, b), (b + L, e + L)]
        out_units.append((b + 1, b + L))
        out_units = sorted((s, e) for s, e in out_units if t2[s:e].strip())
        edit_k = out_units.index((b + 1, b + L))
        span = (b + 1, b + L)
    else:
        t2 = t[:a] + new + t[b:]
        d = len(new) - (b - a)
        out_units = [(s, e) if e <= a else ((s + d, e + d) if s >= b else (s, e + d)) for s, e in units]
        edit_k = [i for i, (s, e) in enumerate(out_units) if s <= a < e or (s <= a <= e and i == k)][0]
        span = (a, a + len(new))
    msgs = [dict(m) for m in r["messages"]]
    user = msgs[-1]["content"]
    if t in user:
        assert user.count(t) == 1
        msgs[-1]["content"] = user.replace(t, t2)
    return t2, [list(u) for u in out_units], msgs, edit_k, span


def weighted_order(rows, rng):
    """long replies first in expectation: Efraimidis-Spirakis keys u^(1/len)."""
    return sorted(rows, key=lambda r: -rng.random() ** (1.0 / max(len(r["response"]), 1)))


def choose_originals(rows, kind, fmt, n, rng, used):
    """up to n originals of a format with candidate sites, spread round-robin over the format's domains, preferring
    replies not yet used by another perturbation."""
    pools = {}
    for d in SOURCES[fmt]:
        pool = [r for r in rows if r["domain"] == d and r["label"] == 0 and len(r["response"]) >= MIN_LEN[fmt]]
        pool = weighted_order(pool, rng)
        pools[d] = [r for r in pool if r["id"] not in used] + [r for r in pool if r["id"] in used]
    out, ptr = [], {d: 0 for d in pools}
    while len(out) < n and any(ptr[d] < len(pools[d]) for d in pools):
        for d in SOURCES[fmt]:
            while ptr[d] < len(pools[d]):
                r = pools[d][ptr[d]]
                ptr[d] += 1
                srng = random.Random(f"{kind}-{r['id']}")  # nosec B311: seeded data generation
                c = sites(kind, r, srng)
                if c:
                    out.append((r, c, srng))
                    break
            if len(out) >= n:
                break
    return out


def build_pairs(rows, smoke=0, llm_factor=1.6):
    """rule-based twins, plus candidate (original, sentence) slots for the LLM edits."""
    rng = random.Random(20261108)  # nosec B311
    used, pairs, llm_jobs = set(), [], []
    for kind, per in TARGET.items():
        for fmt, n in per.items():
            n = min(n, smoke) if smoke else n
            want = int(np.ceil(n * llm_factor)) if kind in LLM_TYPES else n
            for r, cands, srng in choose_originals(rows, kind, fmt, want, rng, used):
                k, a, b, new = pick_site(cands, r["response"], srng)
                job = {"type": kind, "format": fmt, "domain": r["domain"], "orig_id": r["id"], "unit": k,
                       "a": a, "b": b, "target_n": n}
                if kind in LLM_TYPES:
                    llm_jobs.append(job)
                    continue
                p = finish_pair(r, job, new)
                if p is not None:
                    used.add(r["id"])
                    pairs.append(p)
    return pairs, llm_jobs


def finish_pair(r, job, new, insert=False):
    t2, units2, msgs2, ek, span = apply_edit(r, job["unit"], job["a"], job["b"], new, insert)
    before = "" if insert else r["response"][job["a"]:job["b"]]
    if not one_sentence_diff(r["response"], t2):
        return None
    p = dict(job, before=before, after=new, response=t2, units=units2, messages=msgs2, edit_unit=ek,
             edit_span=list(span), pos=round(span[0] / max(len(t2), 1), 4), n_units=len(units2),
             orig_len=len(r["response"]))
    p["third"] = third(p["pos"])
    return p


def one_sentence_diff(t1, t2):
    s1 = [t1[a:b].strip() for a, b in sentence_spans(t1)]
    s2 = [t2[a:b].strip() for a, b in sentence_spans(t2)]
    ops = [o for o in difflib.SequenceMatcher(a=s1, b=s2, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1:
        return False
    tag, i1, i2, j1, j2 = ops[0]
    return (i2 - i1) <= 1 and (j2 - j1) <= 1 or (tag == "replace" and i2 - i1 <= 2 and j2 - j1 <= 2)


# ---------------------------------------------------------------- LLM edits (Qwen3-8B, the reader's own weights)

ENTITY_PROMPT = """Below is a text and one sentence from it. Rewrite that sentence, replacing exactly ONE name of a \
person, place, organisation, country, date-free event or work (book, film, product) with a different, plausible \
name of the same type that does NOT appear anywhere in the text{src}. Change nothing else: same wording, same \
punctuation. Reply with the rewritten sentence only.

Text:
{text}

Sentence to rewrite:
{sent}"""
ADD_PROMPT = """Below is a text and one sentence from it. Write ONE new sentence that will be inserted right after \
that sentence. It must read naturally in the same style and state one specific, plausible-sounding detail (a figure, \
a date, a name or a cause) that is {what}. Do not repeat anything already said. Reply with the new sentence only.

Text:
{text}

Insert after this sentence:
{sent}"""


def norm_words(s):
    return [w.strip(".,;:!?\"'()[]").lower() for w in s.split()]


def check_entity(old, new, context):
    a, b = old.split(), new.split()
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1 or ops[0][0] != "replace":
        return "diff is not one replaced span"
    _, i1, i2, j1, j2 = ops[0]
    if i2 - i1 > 4 or j2 - j1 > 4:
        return "replaced span too long"
    newtoks = [w for w in b[j1:j2] if w[:1].isupper()]
    if not newtoks:
        return "no capitalised new word"
    ctx = context.lower()
    if any(w.strip(".,;:!?\"'()[]").lower() in ctx for w in newtoks if len(w) > 2):
        return "new name already in the text / source"
    return None


def check_addition(new, context):
    if len(sentence_spans(new)) != 1:
        return "not one sentence"
    if not 6 <= len(new.split()) <= 45:
        return "length"
    if new.strip().lower()[:60] in context.lower():
        return "already in the text"
    return None


def llm_edits(model, tok, rows_by_id, jobs, smoke):
    import torch
    pairs, rejected, done = [], [], {}
    dev = model.get_input_embeddings().weight.device
    for job in jobs:
        key = (job["type"], job["format"])
        if done.get(key, 0) >= job["target_n"]:
            continue
        r = rows_by_id[job["orig_id"]]
        t = r["response"]
        sent = t[job["a"]:job["b"]].strip()
        has_doc = bool(r.get("doc"))
        text = t if len(t) < 6000 else t[:6000]
        if job["type"] == "entity":
            prompt = ENTITY_PROMPT.format(src=" or in its source documents" if has_doc else "", text=text, sent=sent)
        else:
            what = ("NOT stated in or supported by the source documents the text was written from" if has_doc else
                    "plausible but false or unverifiable")
            prompt = ADD_PROMPT.format(what=what, text=text, sent=sent)
        msgs = [{"role": "user", "content": prompt}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
        out = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().split("\n")[0].strip().strip('"').strip()
        ctx = t + " " + r["messages"][-1]["content"] + " " + (r.get("doc") or "")
        if job["type"] == "entity":
            why = check_entity(sent, out, ctx)
            if why is None:
                lead = t[job["a"]:job["b"]]
                off = lead.index(sent)
                p = finish_pair(r, dict(job, a=job["a"] + off, b=job["a"] + off + len(sent)), out)
                if p is None:
                    why = "diff is not one sentence"
                else:                               # the edit span: the changed words only
                    a0 = os.path.commonprefix([sent, out])
                    p["before"], p["after"] = sent, out
                    p["edit_span"] = [p["edit_span"][0] + len(a0), p["edit_span"][0] + len(a0) + 1]
                    p["pos"] = round(p["edit_span"][0] / len(p["response"]), 4)
                    p["third"] = third(p["pos"])
        else:
            why = check_addition(out, ctx)
            if why is None:
                b = job["b"]
                while b > job["a"] and t[b - 1].isspace():
                    b -= 1
                p = finish_pair(r, dict(job, b=b), out, insert=True)
                if p is None:
                    why = "diff is not one sentence"
        if why is None:
            p["llm_output"] = out
            pairs.append(p)
            done[key] = done.get(key, 0) + 1
        else:
            rejected.append({"type": job["type"], "orig_id": job["orig_id"], "why": why, "out": out[:300]})
        if (len(pairs) + len(rejected)) % 25 == 0:
            log("llm edits", len(pairs), "kept", len(rejected), "rejected", dict(done))
    return pairs, rejected


# ---------------------------------------------------------------- kernel stages

def setup():
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    log("transformers", transformers.__version__)
    if v < (4, 56):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"])
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"])


def build_corpus():
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, "corpus.jsonl")
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.join(SRC, "experiments", "omni_corpus.py"), "build", "--out", path],
                       stdout=open(os.path.join(WORK, "build.log"), "w"), stderr=subprocess.STDOUT, check=True)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    log("corpus md5", md5, "same as repo" if md5 == CORPUS_MD5 else "DIFFERS")
    return path, md5


def profile_rows(rows, smoke):
    out = []
    for d in FMT:
        dr = [r for r in rows if r["domain"] == d]
        out += random.Random(f"profile-{d}").sample(dr, min(smoke or PROFILE_N, len(dr)))  # nosec B311
    return out


def judge_view(r):
    """the task + documents as the external judges read them (kaggle/omni/judge_calibration.py prompt())."""
    user = r["messages"][-1]["content"]
    doc = r.get("doc") or ""
    if "hallucinated span" in user and r["response"][:60] in user:
        user = user[:user.index(r["response"][:60])].rstrip()
        user = user[:user.rfind("\n")] if "\n" in user else user
        doc = ""
    if doc and " ".join(doc.split())[:100] in " ".join(user.split()):
        doc = ""
    return user, doc


def items(rows, pairs):
    """every reply to score: (key, reader row, judge item, format, has documents)."""
    by = {r["id"]: r for r in rows}
    out = []
    for r in rows:
        if r.get("_profile"):
            q, d = judge_view(r)
            out.append({"key": f"p{r['id']}", "row": r, "jq": q, "jd": d})
    seen = set()
    for p in pairs:
        r = by[p["orig_id"]]
        q, d = judge_view(r)
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append({"key": f"o{r['id']}", "row": r, "jq": q, "jd": d})
        tw = dict(r, response=p["response"], units=p["units"], messages=p["messages"])
        out.append({"key": f"t{p['pid']}", "row": tw, "jq": q, "jd": d})
    return out


class GPUNLI:
    """vt.omni.NLIScorer on a GPU: the shipped score ((HHEM + MiniCheck worst of the first 4 sentences) / 2) and,
    as a diagnostic, MiniCheck over all sentences (up to 40)."""

    def __init__(self, device):
        from vt.omni import NLIScorer
        self.s = NLIScorer()
        self.dev = device
        self.s.mc.to(device)
        self.hhem_dev = "cpu"
        try:
            self.s.hhem.to(device)
            self.s.hhem.predict([("The sky is blue.", "The sky is blue.")])
            self.hhem_dev = device
        except Exception as e:  # noqa: BLE001
            log("HHEM on GPU failed, CPU:", repr(e)[:200])
            self.s.hhem.to("cpu")

    def mc(self, doc, claim):
        import torch
        s = self.s
        w = doc.split()
        ch = [" ".join(w[i:i + s.chunk_words]) for i in range(0, max(1, len(w)), s.chunk_words)] or [""]
        x = s.mtok(ch, [claim] * len(ch), return_tensors="pt", padding=True, truncation="only_first", max_length=512)
        x = {k: v.to(self.dev) for k, v in x.items()}
        with torch.no_grad():
            p = torch.softmax(s.mc(**x).logits.float(), -1)[:, 1]
        return float(1 - p.max())

    def __call__(self, doc, question, reply):
        import torch
        doc = doc[:12000]
        claim = f"{question[-300:]} {reply}" if len(reply.split()) < 8 else reply
        sents = [reply[a:b].strip() for a, b in sentence_spans(reply)] or [reply]
        per = [self.mc(doc, x) for x in sents[:40]] if len(sents) > 1 else [self.mc(doc, claim)]
        with torch.no_grad():
            h = 1 - float(self.s.hhem.predict([(doc, claim)])[0])
        return {"nli": (h + max(per[:4])) / 2, "hhem": h, "mc4": max(per[:4]), "mc_all": max(per)}


def stage_reader(a):
    """GPUs 0+1: Qwen3-8B writes the LLM edits, then reads every reply; NLI on the replies with documents."""
    import torch
    from vt.omni import OmniVerifier, Reader
    rows = [json.loads(l) for l in open(a.corpus)]
    by = {r["id"]: r for r in rows}
    pairs, jobs = build_pairs(rows, a.smoke)
    log(len(pairs), "rule-based pairs;", len(jobs), "LLM edit slots")
    rd = Reader("Qwen/Qwen3-8B")
    assert rd.layers == [9, 14, 20, 25, 31, 36], rd.layers
    log("reader on", sorted({str(p.device) for p in rd.model.parameters()}))
    t = time.time()
    lp, rej = llm_edits(rd.model, rd.tok, by, jobs, a.smoke)
    log(f"LLM edits: {len(lp)} kept, {len(rej)} rejected, {(time.time() - t) / max(len(lp) + len(rej), 1):.1f} s each")
    pairs += lp
    for i, p in enumerate(pairs):
        p["pid"] = i
    json.dump(pairs, open(os.path.join(WORK, "pairs.json"), "w"))
    json.dump(rej, open(os.path.join(WORK, "llm_rejected.json"), "w"))
    for r in profile_rows(rows, a.smoke):
        r["_profile"] = True
    todo = items(rows, pairs)
    json.dump([{"key": x["key"], "jq": x["jq"], "jd": x["jd"], "response": x["row"]["response"],
                "format": FMT.get(x["row"]["domain"], "other"), "domain": x["row"]["domain"]} for x in todo],
              open(os.path.join(TMP, "judge_items.json"), "w"))
    json.dump({x["key"]: x["row"]["domain"] for x in todo}, open(os.path.join(WORK, "key_domains.json"), "w"))
    log(len(todo), "replies to read")
    v = OmniVerifier.load(a.head)
    nli = GPUNLI("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    log("NLI ready; HHEM on", nli.hhem_dev)
    from vt.mathcheck import check_arithmetic
    out = {"key": [], "X": [], "j_logit": [], "flags": [], "nli": [], "hhem": [], "mc4": [], "mc_all": [],
           "t_reader": [], "t_nli": [], "n_tokens_trunc": []}
    U, Urow, Uidx, viol = [], [], [], {}
    t_all = time.time()
    for k, x in enumerate(todo):
        r = x["row"]
        t = time.time()
        f = rd.features(r)
        torch.cuda.synchronize()
        t_r = time.time() - t
        if f is None:
            continue
        fl = [e["expression"] for e in check_arithmetic(r["response"], explicit=True) if not e["ok"]]
        t = time.time()
        n = nli(r["doc"], x["jq"], r["response"]) if r.get("doc") \
            else {"nli": np.nan, "hhem": np.nan, "mc4": np.nan, "mc_all": np.nan}
        t_n = time.time() - t
        out["key"].append(x["key"])
        out["X"].append(v.reply_vector(f).astype(np.float32))
        out["j_logit"].append(f["j_logit"])
        out["flags"].append(min(len(fl), 1))
        for kk in ("nli", "hhem", "mc4", "mc_all"):
            out[kk].append(n[kk])
        out["t_reader"].append(t_r)
        out["t_nli"].append(t_n)
        out["n_tokens_trunc"].append(len(f["units"]) < len(r["units"]))
        um = v.unit_matrix(f)
        U.append(um.astype(np.float32))
        Urow += [x["key"]] * len(um)
        Uidx += [u["index"] for u in f["units"]]
        if fl:
            viol[x["key"]] = fl
        if k % 50 == 0 or k == len(todo) - 1:
            np.savez(os.path.join(WORK, "reader_scores.npz"), **{kk: np.array(vv) for kk, vv in out.items()},
                     U=np.concatenate(U) if U else np.zeros((0, 1)), U_key=np.array(Urow), U_idx=np.array(Uidx))
            json.dump(viol, open(os.path.join(WORK, "arith_flags.json"), "w"))
            log(f"read {k + 1}/{len(todo)}", f"{(time.time() - t_all) / (k + 1):.2f} s/reply", x["key"],
                f"reader {t_r:.2f}s nli {t_n:.2f}s")
    log("reader stage done")


def stage_judge(a):
    """one GPU: one external judge, JudgeScorer's prompt and one-pass read, over every reply."""
    import torch
    import transformers
    from vt.omni import JudgeScorer
    name, preq = JUDGES[a.judge]
    js = JudgeScorer.__new__(JudgeScorer)                   # loaded as in judge_calibration.py (fp16, one GPU)
    js.torch, js.max_tokens = torch, 4096
    js.tok = transformers.AutoTokenizer.from_pretrained(name)
    js.model = None
    for src, q in ((name, True), (preq, False)):
        try:
            kw = dict(dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
            if q:
                kw["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            js.model = transformers.AutoModelForCausalLM.from_pretrained(src, **kw).eval()
            log(a.judge, "loaded", src, "GB", round(torch.cuda.memory_allocated() / 1e9, 1))
            break
        except Exception as e:  # noqa: BLE001
            log(a.judge, "load failed", src, repr(e)[:300])
    assert js.model is not None
    first = lambda ws: sorted({js.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws})
    js.yes, js.no = first(["Yes", " Yes", "yes"]), first(["No", " No", "no"])
    todo = json.load(open(os.path.join(TMP, "judge_items.json")))
    res = {"key": [], "logodds_wrong": [], "t": []}
    t_all = time.time()
    for k, x in enumerate(todo):
        it = {"question": x["jq"], "doc": x["jd"], "response": x["response"]}
        t = time.time()
        try:
            s = js(it, reasoning=x["format"] == "reasoning")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            js.max_tokens = 3072
            s = js(it, reasoning=x["format"] == "reasoning")
            js.max_tokens = 4096
        res["key"].append(x["key"])
        res["logodds_wrong"].append(s)
        res["t"].append(time.time() - t)
        if k % 100 == 0 or k == len(todo) - 1:
            np.savez(os.path.join(WORK, f"judge_{a.judge}.npz"), **{kk: np.array(vv) for kk, vv in res.items()})
            log(a.judge, f"{k + 1}/{len(todo)}", f"{(time.time() - t_all) / (k + 1):.2f} s/reply")
    log(a.judge, "done")


def run_worker(args, gpus, logname):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH=SRC)
    return subprocess.Popen([sys.executable, SELF] + args, env=env, stdout=open(os.path.join(WORK, logname), "w"),
                            stderr=subprocess.STDOUT)


def tail(name):
    try:
        lines = open(os.path.join(WORK, name)).read().strip().splitlines()
        return lines[-1][:240] if lines else ""
    except OSError:
        return ""


def kernel(a):
    global SELF
    if not os.path.exists(SELF):                           # not bundled: write this script (without the prelude)
        body = open(os.path.abspath(__file__)).read()
        body = body[body.index("# --- end bundle ---") + 21:] if "# --- end bundle ---" in body else body
        os.makedirs(TMP, exist_ok=True)
        SELF = os.path.join(TMP, "nuance_pairs.py")
        open(SELF, "w").write(body)
    setup()
    subprocess.run("nvidia-smi; df -h /tmp /kaggle/working", shell=True)
    corpus, md5 = build_corpus()
    head = os.path.join(SRC, "heads", "omni_qwen3_8b.npz")
    if not os.path.exists(head):
        import urllib.request
        os.makedirs(os.path.dirname(head), exist_ok=True)
        urllib.request.urlretrieve(HEAD_URL, head)  # nosec B310: fixed https URL, md5-checked
    hm = hashlib.md5(open(head, "rb").read()).hexdigest()
    log("head md5", hm)
    assert hm == HEAD_MD5, "head differs from heads/omni_qwen3_8b.npz"
    common = ["--corpus", corpus, "--head", head, "--smoke", str(a.smoke)]
    p = run_worker(["--stage", "reader"] + common, "0,1", "reader.log")
    while p.poll() is None:
        time.sleep(90)
        log("reader:", tail("reader.log"))
    log("reader rc", p.returncode)
    assert p.returncode == 0, open(os.path.join(WORK, "reader.log")).read()[-3000:]
    subprocess.run("rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B", shell=True)
    procs = [(j, run_worker(["--stage", "judge", "--judge", j] + common, str(g), f"judge_{j}.log"))
             for g, j in enumerate(JUDGES)]
    while any(q.poll() is None for _, q in procs):
        time.sleep(120)
        for j, _ in procs:
            log(j, tail(f"judge_{j}.log"))
    rc = {j: q.returncode for j, q in procs}
    log("judges rc", rc)
    json.dump({"corpus_md5": md5, "return_codes": rc, "minutes": (time.time() - T0) / 60, "smoke": a.smoke},
              open(os.path.join(WORK, "run_info.json"), "w"))
    rep = analyse(WORK, head)
    json.dump(rep, open(os.path.join(WORK, "nuance_pairs.json"), "w"), indent=1)
    log("done")


# ---------------------------------------------------------------- analysis (CPU, seconds)

def auroc(s, y):
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    r = rankdata(s)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_acc(so, st):
    d = np.asarray(st, float) - np.asarray(so, float)
    return np.where(d > 1e-9, 1.0, np.where(d < -1e-9, 0.0, 0.5))


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyse(d, head):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from vt.omni import OmniVerifier

    class Stored(OmniVerifier):
        def reply_vector(self, f):
            return f["X"]

        def unit_matrix(self, f):
            return f["U"]

    v = Stored.load(head)
    pairs = json.load(open(os.path.join(d, "pairs.json")))
    R = np.load(os.path.join(d, "reader_scores.npz"))
    J = {j: np.load(os.path.join(d, f"judge_{j}.npz")) for j in JUDGES if os.path.exists(os.path.join(d, f"judge_{j}.npz"))}
    jmap = {j: dict(zip(z["key"].tolist(), z["logodds_wrong"].tolist())) for j, z in J.items()}
    ukeys, uidx, Umat = R["U_key"], R["U_idx"], R["U"]
    feats = {}
    order = np.argsort(ukeys, kind="stable")
    starts = {}
    for i in order:
        starts.setdefault(str(ukeys[i]), []).append(i)
    for i, key in enumerate(R["key"].tolist()):
        ui = starts.get(key, [])
        feats[key] = {"X": R["X"][i], "U": Umat[ui] if ui else np.zeros((0, Umat.shape[1]), np.float32),
                      "uidx": uidx[ui] if ui else np.zeros(0, int), "j_logit": float(R["j_logit"][i]),
                      "nli": float(R["nli"][i]), "flags": float(R["flags"][i]), "mc_all": float(R["mc_all"][i]),
                      "judges": [jmap[j].get(key, np.nan) for j in JUDGES if j in jmap]}
    # profile per domain: the profile sample's replies (keys p<id>)
    dom_of = json.load(open(os.path.join(d, "key_domains.json")))
    profs = {}
    for dom in sorted({p["domain"] for p in pairs}):
        fs = [feats[k] for k, dd in dom_of.items() if dd == dom and k.startswith("p") and k in feats]
        if len(fs) >= 3:
            profs[dom] = v.profile(fs, name=dom)
    rows = []
    for p in pairs:
        ko, kt = f"o{p['orig_id']}", f"t{p['pid']}"
        if ko not in feats or kt not in feats or p["domain"] not in profs:
            continue
        prof = profs[p["domain"]]
        fo, ft = feats[ko], feats[kt]
        r, u = v._raw(prof, [fo, ft])
        j, n, fl, e = v._extras([fo, ft])
        z = prof["z"]
        zz = lambda x, k: np.where(np.isnan(x), 0.0, (np.nan_to_num(x) - z[k][0]) / z[k][1]) if k in z else np.zeros(2)
        ex = {"reader_reply": zz(r, "r"), "reader_worst_unit": zz(u, "u"), "reader_judge": zz(j, "j"),
              "arith_flag": fl, "nli": zz(n, "n") if "n" in z else n * np.nan,
              "nli_mc_all_sentences": np.array([fo["mc_all"], ft["mc_all"]])}
        for k, jn in enumerate([j_ for j_ in JUDGES if j_ in jmap]):
            ex[f"judge_{jn}"] = zz(e[:, k], f"e{k}")
        has_n = "n" in z and not np.isnan(n).any()
        ex["reader_stack (r+u+2j+2flag)"] = ex["reader_reply"] + ex["reader_worst_unit"] + 2 * ex["reader_judge"] + 2 * fl
        if has_n:
            ex["reader_stack + nli"] = ex["reader_stack (r+u+2j+2flag)"] + ex["nli"]
        ex["judges_only (qwen14+phi4)"] = sum(ex[f"judge_{jn}"] for jn in jmap)
        full = v._combine(prof, r, u, (j, n, fl, e))
        ex["FULL z-sum (stack + judges)"] = full
        # localisation: the reader's per-unit scores on the twin
        U = ft["U"]
        loc = {}
        if len(U):
            us = ((U - np.array(prof["umu"])) / np.array(prof["usd"])) @ v.art["uw"] + float(v.art["ub"])
            idx = list(ft["uidx"])
            if p["edit_unit"] in idx:
                rank = int((us > us[idx.index(p["edit_unit"])]).sum()) + 1
                loc = {"rank": rank, "n": len(us), "top1": rank == 1, "top3": rank <= 3}
                Uo = fo["U"]
                if p["type"] != "addition" and len(Uo) and p["edit_unit"] in list(fo["uidx"]):
                    uso = ((Uo - np.array(prof["umu"])) / np.array(prof["usd"])) @ v.art["uw"] + float(v.art["ub"])
                    loc["unit_score_up"] = bool(us[idx.index(p["edit_unit"])] > uso[list(fo["uidx"]).index(p["edit_unit"])])
        rows.append({"pid": p["pid"], "type": p["type"], "format": p["format"], "domain": p["domain"],
                     "third": p["third"], "pos": p["pos"], "orig_len": p["orig_len"],
                     "s": {k: [float(x[0]), float(x[1])] for k, x in ex.items()}, "loc": loc})
    # length terciles over the pairs
    L = np.array([x["orig_len"] for x in rows])
    q1, q2 = np.percentile(L, [100 / 3, 200 / 3]) if len(L) else (0, 0)
    for x in rows:
        x["len_tercile"] = "short" if x["orig_len"] <= q1 else ("medium" if x["orig_len"] <= q2 else "long")
    scorers = list(dict.fromkeys(k for x in rows for k in x["s"]))

    def table(sel):
        out = {"n_pairs": len(sel)}
        for sc in scorers:
            ss = [x for x in sel if sc in x["s"] and not np.isnan(x["s"][sc]).any()]
            if not ss:
                continue
            acc = pair_acc([x["s"][sc][0] for x in ss], [x["s"][sc][1] for x in ss])
            so = [x["s"][sc][0] for x in ss]
            st = [x["s"][sc][1] for x in ss]
            out[sc] = {"pair_acc": round(float(acc.mean()), 3), "ci95": boot(acc), "ties": round(float((acc == 0.5).mean()), 3),
                       "auroc": round(auroc(so + st, [0] * len(ss) + [1] * len(ss)), 3),
                       "mean_delta": round(float(np.mean(np.array(st) - np.array(so))), 3), "n": len(ss)}
        lo = [x["loc"] for x in sel if x["loc"]]
        if lo:
            out["localisation"] = {"n": len(lo), "top1": round(float(np.mean([l["top1"] for l in lo])), 3),
                                   "top3": round(float(np.mean([l["top3"] for l in lo])), 3),
                                   "chance_top1": round(float(np.mean([1 / l["n"] for l in lo])), 3),
                                   "chance_top3": round(float(np.mean([min(1, 3 / l["n"]) for l in lo])), 3),
                                   "mean_units": round(float(np.mean([l["n"] for l in lo])), 1)}
            up = [l["unit_score_up"] for l in lo if "unit_score_up" in l]
            if up:
                out["localisation"]["edited_unit_score_rose"] = round(float(np.mean(up)), 3)
        return out

    rep = {"n_pairs": len(rows), "length_tercile_bounds_chars": [float(q1), float(q2)],
           "profile": "per source domain: %d seeded corpus replies of the domain, labels unused (OmniVerifier.profile)"
                      % PROFILE_N, "overall": table(rows)}
    for key in ("type", "format", "domain", "third", "len_tercile"):
        rep[f"by_{key}"] = {g: table([x for x in rows if x[key] == g]) for g in sorted({x[key] for x in rows})}
    rep["by_type_format"] = {f"{t}|{f}": table([x for x in rows if x["type"] == t and x["format"] == f])
                             for t, f in sorted({(x["type"], x["format"]) for x in rows})}
    rep["by_type_third"] = {f"{t}|{h}": table([x for x in rows if x["type"] == t and x["third"] == h])
                            for t, h in sorted({(x["type"], x["third"]) for x in rows})}
    rep["per_pair"] = rows
    info = os.path.join(d, "run_info.json")
    if os.path.exists(info):
        rep["run_info"] = json.load(open(info))
    if os.path.exists(os.path.join(d, "llm_rejected.json")):
        rej = json.load(open(os.path.join(d, "llm_rejected.json")))
        from collections import Counter
        rep["llm_edit_rejections"] = dict(Counter(f"{x['type']}: {x['why']}" for x in rej))
    R2 = np.load(os.path.join(d, "reader_scores.npz"))
    rep["seconds_per_reply"] = {"reader": round(float(np.mean(R2["t_reader"])), 3),
                                "nli (doc replies)": round(float(np.nanmean(np.where(np.isnan(R2["nli"]), np.nan, R2["t_nli"]))), 3)}
    for j, z in J.items():
        rep["seconds_per_reply"][f"judge_{j}"] = round(float(np.mean(z["t"])), 3)
    return rep


# ---------------------------------------------------------------- local: rule pairs only

def build_local(a):
    rows = [json.loads(l) for l in open(a.corpus)]
    pairs, jobs = build_pairs(rows, a.smoke)
    from collections import Counter
    print("rule pairs", Counter((p["type"], p["format"]) for p in pairs))
    print("thirds", Counter((p["type"], p["third"]) for p in pairs))
    print("LLM slots", Counter((j["type"], j["format"]) for j in jobs))
    rng = random.Random(0)  # nosec B311
    for t in TARGET:
        ps = [p for p in pairs if p["type"] == t]
        for p in rng.sample(ps, min(a.show, len(ps))):
            print(f"--- {t} {p['domain']} pos={p['pos']} unit={p['edit_unit']}/{p['n_units']}: "
                  f"{p['before']!r} -> {p['after']!r}")
            s = p["response"]
            e0 = p["edit_span"][0]
            print("   ...", s[max(0, e0 - 120):e0 + 80].replace("\n", " "), "...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="pairs per type x format (and profile replies per domain)")
    ap.add_argument("--stage", choices=["reader", "judge"])
    ap.add_argument("--judge")
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--head", default="heads/omni_qwen3_8b.npz")
    ap.add_argument("--build-local", action="store_true")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--analyse")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.build_local:
        return build_local(a)
    if a.analyse:
        rep = analyse(a.analyse, a.head)
        if a.report:
            json.dump(rep, open(a.report, "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_pair"}, indent=1)[:20000])
        return
    if a.stage == "reader":
        return stage_reader(a)
    if a.stage == "judge":
        return stage_judge(a)
    kernel(a)


if __name__ == "__main__":
    sys.exit(main())
