"""Is each sentence of a reply grounded in the evidence? Read from the model's own neurons, not asked of it.

The served model reads the reply once, teacher-forced, with the evidence in its context (the user's documents
and messages, exact re-computations of arithmetic the reply states, passages a lookup found), and a linear
probe on its MLP neurons scores every sentence: how far that sentence's activations look like a sentence the
context does not ground. The probe is trained without human labels, on constructions in which grounding is
known by design (a sentence of a paragraph, the same with a name or number swapped, one taken out of it;
experiments/grounding_probe.py --export). Trained on Wikipedia-style text, the 4B's probe found RAGTruth's
human-labeled hallucinations in news, passages and reviews at AUROC 0.82 per response (docs/grounding_trace.md).

    g = Grounding(model, tok, "heads/grounding_probe.npz")
    g.check(messages, reply, documents=[...])
    # -> {"sentences": [{"text", "start", "end", "score", "z", "verdict"}], "total": 2.3, "verdict": ...}

With the scale of each signal over natural answer sentences stored with the probe (no labels), two more
readings join it from the model's own probabilities: how much the evidence raises the words written (the
evidence shift) and where it makes the model prefer another word (disagreement: a contradiction's mark). The
three, on one scale, are summed per sentence (0.77 per RAGTruth sentence, 0.82 per response, for the 4B), and a
sentence the disagreement dominates reads "contradicted".

No generation, no question asked of the model: two forward passes, one with the evidence (neurons and
probabilities at once) and one with the question alone. `raw()` gives each sentence's readings before any scale,
`score()` the verdicts; a probe calibrated on labeled replies (vt/verifier_pipeline.py) also carries the reply
threshold for a chosen false-alarm rate and the reply's probability of holding a hallucination.
"""
import re

import numpy as np

from vt.mathcheck import check_arithmetic


def sentence_spans(text):
    """[(start, end)] character ranges of the sentences of a text."""
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s)|\n+", text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if re.search(r"[A-Za-z0-9]", text[a:b])]


def select(documents, query, budget, chunk_words=150):
    """Documents that fit the budget as they are; otherwise their passages most relevant to the question and reply
    (BM25, overlapping chunks), in their original order, up to the budget: a long filing or paper is not simply
    cut at its end."""
    if sum(map(len, documents)) <= budget:
        return documents
    from vt.retrieval import Corpus, chunks
    pieces = [c for d in documents for c in chunks(d, chunk_words, chunk_words // 3)]
    if not pieces:
        return documents
    sc = Corpus(pieces).scores(query)
    keep, used = set(), 0
    for i in sorted(range(len(pieces)), key=lambda i: -sc[i]):
        if used + len(pieces[i]) > budget:
            continue
        keep.add(i)
        used += len(pieces[i]) + 2
    return [pieces[i] for i in sorted(keep)]


def exact_lines(text):
    """The reply's stated arithmetic, re-computed: evidence no model can doubt."""
    return [f"{m['expression']} = {m['computed']:g}" for m in check_arithmetic(text)]


class Grounding:
    def __init__(self, model, tok, path, retriever=None, flag_z=1.0, max_context_chars=8000):
        """flag_z: a sentence whose score is this many standard deviations (of the probe's scores on its
        training constructions) above their mean reads "not grounded"."""
        z = np.load(path)
        self.model, self.tok, self.retriever = model, tok, retriever
        self.layers = [int(i) for i in z["layers"]]
        self.mean, self.scale, self.coef = z["mean"], z["scale"], z["coef"]
        self.intercept, self.score_mean, self.score_std = float(z["intercept"]), float(z["score_mean"]), float(z["score_std"])
        self.flag_z, self.max_context_chars = flag_z, max_context_chars
        # the scale of each signal over natural answer sentences (no labels): probe, evidence shift, disagreement;
        # without them the probe alone, on the scale of its training constructions
        self.natural = {k: (float(z[f"nat_{k}_mean"]), float(z[f"nat_{k}_std"]))
                        for k in ("probe", "mean_shift", "max_disagreement", "copy_conflict", "uncopied",
                                  "self_reliance")
                        if f"nat_{k}_mean" in z.files}
        self.signals = all(k in self.natural for k in ("probe", "mean_shift", "max_disagreement"))
        # the reader's copying heads (vt/copyheads.py; `vt.verifier_pipeline copyheads`): where they point in the
        # evidence before each reply token, read in the same forward pass
        self.copy_heads = [(int(l), int(h)) for l, h in z["copy_heads"]] if "copy_heads" in z.files else []
        # source flow (vt/sourceflow.py) at these layers: how much each token draws on the reply's own earlier text
        # rather than parking on the sink; the same sign flags errors in answers, reasoning steps and long-form text
        self.flow_layers = [int(l) for l in z["flow_layers"]] if "flow_layers" in z.files else []
        # a calibration on labeled replies (vt/verifier_pipeline.py calibrate): the reply-total threshold at the
        # chosen false-alarm rate, and a logistic map from total to probability
        self.calibration = (float(z["cal_threshold"]), float(z["cal_a"]), float(z["cal_b"])) \
            if "cal_threshold" in z.files else None
        # a per-domain weighting of the signals (calibrate --fit-weights): the reply's risk is a weighted sum of
        # each signal's strongest sentence, and the calibration's threshold applies to it instead of the total
        self.weights = (np.asarray(z["cal_weights"], float), float(z["cal_weights_b"])) \
            if "cal_weights" in z.files else None

    def _prefix(self, user):
        msgs = [{"role": "user", "content": user}]
        try:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(p, add_special_tokens=False).input_ids

    def evidence(self, messages, reply, documents=()):
        """The question and everything the reply can be grounded in, as one user turn."""
        said = [m["content"] for m in messages if m.get("role") == "user"]
        question = said[-1] if said else ""
        parts = select(list(documents or []), f"{question} {reply}", self.max_context_chars)
        if self.retriever is not None and not parts:             # the question, and the question with the
            for q in (question[:300], f"{question[:200]} {reply[:300]}"):   # reply's own words: they steer the
                try:                                             # search to the pages that settle its claims
                    parts += [p for p in self.retriever(q) if p not in parts]
                except Exception as e:                           # no lookup: grounded in the conversation only
                    print(f"grounding lookup failed: {e!r}", flush=True)
        parts += said[:-1]
        sums = exact_lines(reply)
        if sums:
            parts.append("Exact arithmetic: " + "; ".join(sums))
        from vt.codecheck import check_code
        code = check_code(reply)
        if code:                                                 # parsed, never run
            parts.append("Code check: " + "; ".join(f"the {c['language']} parses" if c["ok"] else
                                                     f"the {c['language']} does not parse ({c['error']})" for c in code))
        ev = "\n\n".join(parts)[-self.max_context_chars:]
        return f"{question}\n\n{ev}".strip()

    def _read(self, prefix_ids, reply_ids, region=None):
        """One forward pass over evidence and reply: the probe's neurons at every position (on the CPU, float16)
        and the log-probability distributions over the reply's tokens."""
        import torch
        acts, hooks = {}, []
        blocks = self.model.model.layers
        for i in self.layers:
            def hook(mod, inp, i=i):
                acts[i] = inp[0][0].detach().to("cpu", torch.float16)
            hooks.append(blocks[i].mlp.down_proj.register_forward_pre_hook(hook))
        heads = None
        if self.copy_heads:                                      # the positions that predict each reply token
            from vt.copyheads import HeadAttention
            P = len(prefix_ids)
            heads = HeadAttention(self.model, self.copy_heads).at(range(P - 1, P + len(reply_ids) - 1))
            heads.__enter__()
        flow = None
        if self.flow_layers and region is not None:
            from vt.sourceflow import SourceFlow
            P = len(prefix_ids)
            flow = SourceFlow(self.model, self.flow_layers).at(range(P - 1, P + len(reply_ids) - 1), region)
            flow.__enter__()
        try:
            lw, dw = self._logprobs(prefix_ids, reply_ids)
        finally:
            for h in hooks:
                h.remove()
            if heads is not None:
                heads.__exit__()
            if flow is not None:
                flow.__exit__()
        copy = None
        if heads is not None:
            from vt.copyheads import token_copy
            copy = token_copy(heads.collect(), prefix_ids, reply_ids)
        return acts, lw, dw, copy, (flow.collect() if flow is not None else None)

    def raw(self, messages, reply, documents=(), signals=True):
        """Each sentence's raw readings: the probe's score, the evidence shift (mean over its tokens) and the
        disagreement (its largest). Two forward passes: with the evidence (neurons and probabilities at once)
        and with the question alone; signals=False: the probe's score only, one pass."""
        user = self.evidence(messages, reply, documents)
        enc = self.tok(reply, add_special_tokens=False, return_offsets_mapping=True)
        if not enc.input_ids:
            return []
        region = None
        if self.flow_layers:                                     # the same prefix, with each token's region
            from vt.sourceflow import regions_for
            pre, region = regions_for(self.tok, user, "", len(enc.input_ids))
        else:
            pre = self._prefix(user)
        acts, lw, dw, copy, flow = self._read(pre, enc.input_ids, region)
        if copy is not None:                                     # word tokens, the reply's first one aside
            wordy = np.array([j > 0 and bool(re.search(r"[A-Za-z0-9]", self.tok.decode([t])))
                              for j, t in enumerate(enc.input_ids)])
        if signals:
            question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            lo, do = self._logprobs(self._prefix(question.split("\n\n")[0][:500]), enc.input_ids)
            shift, dis = lw - lo, self._disagreement(dw, do, enc.input_ids)
            del do
        else:
            shift = dis = np.zeros(len(enc.input_ids))
        del dw
        rows = []
        for s, e in sentence_spans(reply):
            js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
            if not js:
                continue
            a, b = len(pre) + js[0], len(pre) + js[-1] + 1
            x = np.concatenate([acts[i][a:b].float().mean(0).numpy() for i in self.layers])
            row = {"text": reply[s:e].strip(), "start": s, "end": e,
                   "score": float(((x - self.mean) / self.scale) @ self.coef + self.intercept),
                   "mean_shift": float(shift[js].mean()), "max_disagreement": float(dis[js].max())}
            if copy is not None:
                match, conflict = copy
                w = [j for j in js if wordy[j]]
                row["copy_conflict"] = float(conflict[w].max()) if w else 0.0     # heads point at another token
                row["uncopied"] = float((match[w] < 0.05).mean()) if w else 0.0  # words not copied from it
            if flow is not None:                                 # draws on its own text rather than nothing
                row["self_reliance"] = float(flow[:, js, 3].mean() - flow[:, js, 0].mean())
            rows.append(row)
        return rows

    def score(self, rows):
        """Verdicts from raw readings: each sentence's signals on their natural scales, summed; the reply's total
        ungroundedness, and with a calibration its probability of holding a hallucination."""
        both = self.signals
        cut = max(self.flag_z, self.calibration[0]) if self.calibration is not None and getattr(self, "weights", None) \
            is None else self.flag_z
        out = []
        for r in rows:
            row = {"text": r["text"], "start": r["start"], "end": r["end"], "score": round(r["score"], 4)}
            if both:
                zp, zsh, zd = (self._z("probe", r["score"]), self._z("mean_shift", r["mean_shift"]),
                               self._z("max_disagreement", r["max_disagreement"]))
                terms = [zp, -zsh, zd]
                zc = None
                for k in ("copy_conflict", "uncopied", "self_reliance"):   # with their natural scales: more
                    if k in self.natural and k in r:
                        terms.append(self._z(k, r[k]))
                        row["z_" + k] = round(terms[-1], 3)
                        zc = terms[-1] if k == "copy_conflict" else zc
                zs = sum(terms) / len(terms) ** 0.5             # unit-scale signals, summed
                row.update(z_probe=round(zp, 3), z_shift=round(-zsh, 3), z_disagreement=round(zd, 3))
                kind = "contradicted" if max(zd, zc if zc is not None else zd) >= max(zp, -zsh) else "not grounded"
            else:
                zs, kind = (r["score"] - self.score_mean) / max(self.score_std, 1e-9), "not grounded"
            row.update(z=round(zs, 3), verdict=kind if zs >= cut else "consistent")
            out.append(row)
        total = round(float(sum(max(r["z"], 0.0) for r in out)), 3)
        flagged = [r["verdict"] for r in out if r["verdict"] != "consistent"]
        res = {"sentences": out, "total": total}
        value = total
        if self.weights is not None:                              # the domain's own weighting of the signals
            w, b = self.weights
            value = res["risk"] = round(float(self.reply_features(res) @ w + b), 4)
        if self.calibration is not None:                          # the reply's verdict from its calibrated total:
            thr, a_, b_ = self.calibration                        # sentences each under the cut can add up to it
            res["p_hallucinated"] = round(float(1 / (1 + np.exp(-(a_ * value + b_)))), 4)
            if value >= thr and not flagged and out:
                flagged = [max(out, key=lambda r: r["z"]).get("z_disagreement", 0) >= 1 and "contradicted"
                           or "not grounded"]
            elif value < thr:
                flagged = []
        res["verdict"] = None if not out else "contradicted" if "contradicted" in flagged else \
            "not grounded" if flagged else "consistent"
        return res

    FEATURES = ("total", "z_probe", "z_shift", "z_disagreement", "z_copy_conflict", "z_uncopied", "z_self_reliance")

    @classmethod
    def reply_features(cls, res):
        """The reply's total and each signal's strongest sentence (0 where a signal is absent)."""
        out = res["sentences"]
        return np.array([res["total"]] + [max((r.get(k, 0.0) for r in out), default=0.0) for k in cls.FEATURES[1:]])

    SHORT = 12                                                    # words: below this a reply is no sentence

    def check(self, messages, reply, documents=()):
        """The reply's verdicts. A fragment (under 12 words, no sentence end: "Rams", "$1.2 billion") is read as
        the statement it makes with its question, as a fact-checker would read it: alone it has no sentence for the probe to read (on
        HaluBench's one-word answers this took DROP from 0.40 to 0.53)."""
        question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = reply
        fragment = len(reply.split()) < self.SHORT and not re.search(r"[.!?]\s*$", reply.strip())
        if question and fragment:                                 # one statement: "Which team won: Rams"
            text = f"{question.split(chr(10) * 2)[0].strip()[:500].rstrip(' ?.!')}: {reply.strip()}"
        res = self.score(self.raw(messages, text, documents, signals=self.signals))
        if text is not reply:
            res["read_as"] = "question and reply"
        return res

    def _z(self, name, v):
        m, sd = self.natural[name]
        return (v - m) / max(sd, 1e-9)

    def _logprobs(self, prefix_ids, reply_ids):
        """log p of each reply token, and every position's whole distribution (one forward pass)."""
        import torch
        dev = next(self.model.parameters()).device
        x = torch.tensor([prefix_ids + reply_ids], device=dev)
        with torch.no_grad():
            try:                                                 # only the positions that predict the reply
                logits = self.model(input_ids=x, logits_to_keep=len(reply_ids) + 1).logits[0, :-1]
            except TypeError:
                logits = self.model(input_ids=x).logits[0, len(prefix_ids) - 1:-1]
        lp = torch.log_softmax(logits.float(), -1)
        return lp[torch.arange(len(reply_ids)), torch.tensor(reply_ids, device=dev)].cpu().numpy(), lp

    @staticmethod
    def _disagreement(lp_with, lp_without, reply_ids, min_boost=0.5):
        """Where the evidence makes the model prefer another token than the written one (and raised that token
        itself), by how much: log p(its favourite) - log p(written), 0 elsewhere."""
        import torch
        top = lp_with.argmax(-1)
        rows = torch.arange(len(reply_ids), device=lp_with.device)
        written = torch.tensor(reply_ids, device=lp_with.device)
        gap = lp_with[rows, top] - lp_with[rows, written]
        boost = lp_with[rows, top] - lp_without[rows, top]
        return torch.where((top != written) & (boost >= min_boost), gap, torch.zeros_like(gap)).cpu().numpy()
