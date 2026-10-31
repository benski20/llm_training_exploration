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

No generation, no question asked of the model: one pass with the evidence (two more, short, for the shift and
disagreement: with the evidence and with the question alone).
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
                        for k in ("probe", "mean_shift", "max_disagreement") if f"nat_{k}_mean" in z.files}

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
        parts = list(documents or [])
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

    def _read(self, ids):
        import torch
        acts, hooks = {}, []
        blocks = self.model.model.layers
        for i in self.layers:
            def hook(mod, inp, i=i):
                acts[i] = inp[0][0].float()
            hooks.append(blocks[i].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids], device=next(self.model.parameters()).device))
        finally:
            for h in hooks:
                h.remove()
        return acts

    def check(self, messages, reply, documents=()):
        pre = self._prefix(self.evidence(messages, reply, documents))
        enc = self.tok(reply, add_special_tokens=False, return_offsets_mapping=True)
        if not enc.input_ids:
            return {"sentences": [], "total": 0.0, "verdict": None}
        acts = self._read(pre + enc.input_ids)
        both = len(self.natural) == 3                            # the evidence's effect: a pass without it
        if both:
            question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            lw, dw = self._logprobs(pre, enc.input_ids)
            lo, do = self._logprobs(self._prefix(question.split("\n\n")[0][:500]), enc.input_ids)
            shift, dis = lw - lo, self._disagreement(dw, do, enc.input_ids)
            del dw, do
        rows = []
        for s, e in sentence_spans(reply):
            js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
            if not js:
                continue
            a, b = len(pre) + js[0], len(pre) + js[-1] + 1
            x = np.concatenate([acts[i][a:b].mean(0).cpu().numpy() for i in self.layers])
            score = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
            row = {"text": reply[s:e].strip(), "start": s, "end": e, "score": round(score, 4)}
            if both:
                zp, zsh, zd = (self._z("probe", score), self._z("mean_shift", float(shift[js].mean())),
                               self._z("max_disagreement", float(dis[js].max())))
                zs = (zp - zsh + zd) / 3 ** 0.5                  # three unit-scale signals, summed
                row.update(z_probe=round(zp, 3), z_shift=round(-zsh, 3), z_disagreement=round(zd, 3))
                kind = "contradicted" if zd >= max(zp, -zsh) else "not grounded"
            else:
                zs, kind = (score - self.score_mean) / max(self.score_std, 1e-9), "not grounded"
            row.update(z=round(zs, 3), verdict=kind if zs >= self.flag_z else "grounded")
            rows.append(row)
        total = round(float(sum(max(r["z"], 0.0) for r in rows)), 3)
        flagged = [r["verdict"] for r in rows if r["verdict"] != "grounded"]
        verdict = None if not rows else "contradicted" if "contradicted" in flagged else \
            "not grounded" if flagged else "grounded"
        return {"sentences": rows, "total": total, "verdict": verdict}

    def _z(self, name, v):
        m, sd = self.natural[name]
        return (v - m) / max(sd, 1e-9)

    def _logprobs(self, prefix_ids, reply_ids):
        """log p of each reply token, and every position's whole distribution (one forward pass)."""
        import torch
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([prefix_ids + reply_ids], device=dev)).logits
        lp = torch.log_softmax(logits[0, len(prefix_ids) - 1:-1].float(), -1)
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
