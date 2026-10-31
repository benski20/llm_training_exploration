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

No generation, no question asked of the model, one forward pass per reply.
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
        if self.retriever is not None and not parts:
            try:
                parts += list(self.retriever(question[:300]))
            except Exception as e:                               # no lookup: grounded in the conversation only
                print(f"grounding lookup failed: {e!r}", flush=True)
        parts += said[:-1]
        sums = exact_lines(reply)
        if sums:
            parts.append("Exact arithmetic: " + "; ".join(sums))
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
        rows = []
        for s, e in sentence_spans(reply):
            js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
            if not js:
                continue
            a, b = len(pre) + js[0], len(pre) + js[-1] + 1
            x = np.concatenate([acts[i][a:b].mean(0).cpu().numpy() for i in self.layers])
            score = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
            zs = (score - self.score_mean) / max(self.score_std, 1e-9)
            rows.append({"text": reply[s:e].strip(), "start": s, "end": e, "score": round(score, 4), "z": round(zs, 3),
                         "verdict": "not grounded" if zs >= self.flag_z else "grounded"})
        total = round(float(sum(max(r["z"], 0.0) for r in rows)), 3)
        flagged = [r for r in rows if r["verdict"] == "not grounded"]
        return {"sentences": rows, "total": total,
                "verdict": None if not rows else "not grounded" if flagged else "grounded"}
