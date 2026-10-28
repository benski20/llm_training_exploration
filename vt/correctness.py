"""Predict whether a pretrained chat model's answer is correct, from its internals.

A correctness estimator for an off-the-shelf model (Qwen2 / Llama-style, Hugging Face), built from
the signals that the experiments in VISION.md found to carry the most information:

  probe        a logistic probe on the hidden state at the answer's entity token (the first new
               name or number in the reply), trained on thousands of labelled answers
  output       the answer's token probabilities (entity token, first token, whole-answer min/mean)
  consistency  agreement of K sampled answers with the greedy one (optional: costs K generations)
  grounding    how strongly the question's own tokens drive the entity token, by exact attribution
               (`vt.qwen_glassbox.explain`, decisions fixed); readable as a reason

A stacker combines them into one calibrated probability. `CorrectnessHead.answer()` returns the
reply, P(correct) and the evidence behind it.
"""
import re
import string

import numpy as np
import torch
import torch.nn.functional as F

from vt.qwen_glassbox import explain

SYSTEM = "You are a helpful assistant. Answer the question with a short answer."
SKIP = {"the", "a", "an", "in", "it", "its", "this", "that", "yes", "no", "i"}


# ---------------------------------------------------------------------------------------- labels
def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def is_right(answer, aliases):
    a = norm(answer)
    return int(any(norm(al) and norm(al) in a for al in aliases))


def same_answer(x, y):
    """Two replies give the same answer: one contains the other, or token F1 >= 0.5."""
    x, y = norm(x), norm(y)
    if not x or not y:
        return False
    if x in y or y in x:
        return True
    a, b = x.split(), y.split()
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    return common > 0 and 2 * common / (len(a) + len(b)) >= 0.5


# ---------------------------------------------------------------------------------------- metrics
def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]                                   # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = labels.sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * (~labels).sum()))


def error_at(scores, y, coverage):
    keep = np.argsort(-np.asarray(scores, float), kind="mergesort")[: max(1, int(len(y) * coverage))]
    return float(1 - np.asarray(y)[keep].mean())


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def auroc_ci(scores, y, n=1000, seed=0):
    """95% bootstrap interval of the AUROC (resampling items)."""
    scores, y = np.asarray(scores, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(len(y), size=len(y))
        a = auroc(scores[i], y[i])
        if a is not None:
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def aurc(scores, y):
    """Area under the risk-coverage curve: mean error over all coverages, most confident first (lower is better)."""
    y = np.asarray(y, float)[np.argsort(-np.asarray(scores, float), kind="mergesort")]
    return float(np.mean(np.cumsum(1 - y) / np.arange(1, len(y) + 1)))


def report(scores, y, prob=None, ci=False):
    out = {"n": int(len(y)), "positives": int(np.sum(y)), "auroc": auroc(scores, y),
           "error_at_50pct": error_at(scores, y, 0.5), "error_at_25pct": error_at(scores, y, 0.25),
           "aurc": aurc(scores, y)}
    if ci:
        out["auroc_ci"] = auroc_ci(scores, y)
    if prob is not None:
        out["ece"] = ece(prob, y)
    return out


# ---------------------------------------------------------------------------------------- probe
class Probe:
    """Standardised logistic regression; L2 strength chosen on an inner split of the fitting data.

    With `prior` (a fitted Probe), fitting adapts it to new data: the prior's standardisation is kept,
    the weights start from the prior's, and the penalty pulls them toward it (l2 * |w - w_prior|^2), so a
    few hundred labels can move a probe trained on many thousands without overfitting it. The unchanged
    prior is one of the candidates, so the inner split only accepts an adaptation that helps."""

    def __init__(self, device="cpu", prior=None):
        self.device, self.prior = device, prior

    def _fit(self, X, y, l2, steps):
        if self.prior is not None:
            mu, sd = self.prior.mu.to(X.device), self.prior.sd.to(X.device)
            w0, b0 = self.prior.w.to(X.device), self.prior.b.to(X.device)
            if l2 == float("inf"):
                return mu, sd, w0.clone(), b0.clone()
        else:
            mu, sd = X.mean(0), X.std(0) + 1e-6
            w0 = torch.zeros(X.shape[1], device=self.device)
            b0 = torch.zeros(1, device=self.device)
        Z = (X - mu) / sd
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([w, b], lr=0.05 if self.prior is None else 0.01)
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(Z @ w + b, y) + l2 * (w - w0).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return mu, sd, w.detach(), b.detach()

    def fit(self, X, y, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0), steps=300, seed=1):
        if self.prior is not None and grid == (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            grid = (float("inf"), 10.0, 1.0, 0.1, 0.01)        # inf (first, so it wins ties): keep the prior
        try:
            return self._fit_grid(X, y, grid, steps, seed)
        except torch.OutOfMemoryError:                  # a GPU shared with the model: fall back to the CPU
            if self.device == "cpu":
                raise
            torch.cuda.empty_cache()
            self.device = "cpu"
            return self._fit_grid(X, y, grid, steps, seed)

    def _fit_grid(self, X, y, grid, steps, seed):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        idx = torch.as_tensor(np.random.default_rng(seed).permutation(len(y)), device=self.device)
        cut = int(0.8 * len(y))
        a, b = idx[:cut], idx[cut:]
        best = None
        for l2 in grid:
            mu, sd, w, c = self._fit(X[a], y[a], l2, steps)
            s = auroc((((X[b] - mu) / sd) @ w + c).cpu().numpy(), y[b].cpu().numpy())
            if s is not None and (best is None or s > best[0]):
                best = (s, l2)
        self.l2 = best[1] if best else 1.0
        self.mu, self.sd, self.w, self.b = self._fit(X, y, self.l2, steps)
        return self

    def logit(self, X, chunk=65536):
        out = []
        for i in range(0, max(len(X), 1), chunk):                # chunks: no full-size copy on the device
            x = torch.as_tensor(np.asarray(X[i:i + chunk]), dtype=torch.float32, device=self.device)
            out.append((((x - self.mu) / self.sd) @ self.w + self.b).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def state_dict(self):
        return {"mu": self.mu.cpu(), "sd": self.sd.cpu(), "w": self.w.cpu(), "b": self.b.cpu(), "l2": self.l2}

    @classmethod
    def from_state(cls, sd, device="cpu"):
        p = cls(device)
        p.mu, p.sd, p.w, p.b = (sd[k].float().to(device) for k in ("mu", "sd", "w", "b"))
        p.l2 = sd["l2"]
        return p


# ---------------------------------------------------------------------------------------- model I/O
def chat(tok, question, system=SYSTEM):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _stop_ids(tok):
    return {i for i in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None}


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=24, batch=32, n=1, sample=False, temperature=1.0):
    """Reply token ids for each prompt (n per prompt when sampling), stop tokens removed."""
    tok.padding_side = "left"
    stop = _stop_ids(tok)
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(dev)
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=50) if sample else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
        for row in g[:, enc.input_ids.shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in stop), len(row))
            out.append(row[:cut])
    return out


def entity_index(tok, ans_ids, question):
    """Index (into ans_ids) of the first token of the reply's first new capitalised or numeric word."""
    qwords = set(norm(question).split())
    starts = [j for j in range(len(ans_ids)) if j == 0 or tok.decode(ans_ids[j:j + 1]).startswith(" ")]
    for a, b in zip(starts, starts[1:] + [len(ans_ids)]):
        w = tok.decode(ans_ids[a:b]).strip().strip(string.punctuation + "\"'“”")
        if w and (w[0].isupper() or w[0].isdigit()) and norm(w) not in qwords and w.lower() not in SKIP:
            return a
    return 0


@torch.no_grad()
def features(model, tok, prompt_ids, ans_ids, questions, layers, batch=8):
    """Per item: hidden states (len(layers), d) at the entity-predicting position, and the answer's
    token probabilities. prompt_ids / ans_ids: lists of token-id lists."""
    dev = next(model.parameters()).device
    pad = tok.pad_token_id or tok.eos_token_id
    rows = []
    for i in range(0, len(prompt_ids), batch):
        seqs, info = [], []
        for p, a, q in zip(prompt_ids[i:i + batch], ans_ids[i:i + batch], questions[i:i + batch]):
            a = a or [int(tok.eos_token_id)]
            seqs.append(p + a)
            info.append((len(p), a, entity_index(tok, a, q)))
        M = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), M), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), M), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, M - len(s):] = torch.tensor(s)
            mask[j, M - len(s):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        hs = torch.stack([out.hidden_states[L + 1] for L in layers], 1)            # (B, nL, M, d)
        for j, (P, a, e) in enumerate(info):
            o = M - len(seqs[j])
            t = o + P - 1                                                         # predicts a[0]
            lp = F.log_softmax(out.logits[j, t:t + len(a)].float(), -1)
            tl = lp[torch.arange(len(a), device=dev), torch.tensor(a, device=dev)]
            rows.append({"h": hs[j, :, t + e].half().cpu(), "h_prompt": hs[j, :, t].half().cpu(), "e": e,
                         "first_p": float(tl[0].exp()), "entity_p": float(tl[e].exp()),
                         "min_lp": float(tl.min()), "mean_lp": float(tl.mean())})
    return rows


def question_groups(tok, prompt, question):
    enc = tok(prompt, return_offsets_mapping=True)
    a = prompt.rindex(question)
    b = a + len(question)
    return [f"q@{j}" if s < b and e > a else "template" for j, (s, e) in enumerate(enc["offset_mapping"])]


@torch.no_grad()
def grounding(model, tok, prompt, question, ans_ids):
    """Exact attribution of the entity token's logit: the question's signed contribution, its share of
    the positive mass, the question token that pushes hardest, and the attribution's exactness."""
    dev = next(model.parameters()).device
    p = tok(prompt).input_ids
    a = ans_ids or [int(tok.eos_token_id)]
    e = entity_index(tok, a, question)
    ids = torch.tensor([p + a[:e + 1]], device=dev)
    t = len(p) - 1 + e
    groups = question_groups(tok, prompt, question) + ["answer"] * e
    parts, logit = explain(model, ids[:, :t + 1], t, a[e], groups)
    pos = {n: v for n, v in parts.items() if v > 0}
    tot = sum(pos.values()) or 1.0
    qparts = {n: v for n, v in parts.items() if n.startswith("q@")}
    top = max(qparts, key=qparts.get) if qparts else None
    return {"question_signed": sum(qparts.values()),
            "question_share": sum(v for n, v in pos.items() if n.startswith("q@")) / tot,
            "template_signed": parts.get("template", 0.0), "logit": logit,
            "top_question_token": tok.decode([p[int(top[2:])]]) if top else "",
            "exact_err": abs(sum(parts.values()) - logit)}


def consistency(greedy, samples):
    return float(np.mean([same_answer(greedy, s) for s in samples])) if samples else 0.0


OUTPUT_KEYS = ["first_p", "entity_p", "min_lp", "mean_lp"]
GROUND_KEYS = ["question_signed", "question_share", "template_signed"]


def stack_features(probe_logit, row, use_consistency, use_grounding):
    x = [probe_logit] + [row[k] for k in OUTPUT_KEYS]
    if use_consistency:
        x.append(row["consistency"])
    if use_grounding:
        x += [row[k] for k in GROUND_KEYS]
    return np.array(x, dtype=np.float32)


class CorrectnessHead:
    """Trained correctness estimator for one model: probe on the entity hidden state (one layer or
    several), plus a stacker over probe, output probabilities, optional consistency and grounding."""

    def __init__(self, layers, probe, stacker, use_consistency, use_grounding, k_samples=6, calib=(1.0, 0.0)):
        self.layers, self.probe, self.stacker = layers, probe, stacker
        self.use_consistency, self.use_grounding, self.k = use_consistency, use_grounding, k_samples
        self.calib = tuple(calib)

    def state_dict(self):
        return {"layers": self.layers, "probe": self.probe.state_dict(), "stacker": self.stacker.state_dict(),
                "use_consistency": self.use_consistency, "use_grounding": self.use_grounding, "k": self.k,
                "calib": list(self.calib)}

    @classmethod
    def from_state(cls, sd):
        return cls(sd["layers"], Probe.from_state(sd["probe"]), Probe.from_state(sd["stacker"]),
                   sd["use_consistency"], sd["use_grounding"], sd.get("k", 6), sd.get("calib", (1.0, 0.0)))

    def calibrate(self, z, y, steps=500):
        """Refit the probability scale for a new domain from a small labelled sample: P = sigmoid(a z + b)
        on the stacker's raw logits z. Monotone, so the ranking (AUROC) is unchanged."""
        z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a * z + b, y)
            loss.backward()
            return loss
        opt.step(closure)
        self.calib = (float(a), float(b))
        return self

    @torch.no_grad()
    def answer(self, model, tok, question):
        prompt = chat(tok, question)
        a = generate(model, tok, [prompt], batch=1)[0]
        reply = tok.decode(a, skip_special_tokens=True).strip()
        row = features(model, tok, [tok(prompt).input_ids], [a], [question], self.layers, batch=1)[0]
        ev = {"entity": tok.decode([a[row["e"]]]) if a else "", **{k: row[k] for k in OUTPUT_KEYS}}
        if self.use_consistency:
            s = [tok.decode(x, skip_special_tokens=True) for x in
                 generate(model, tok, [prompt], batch=1, n=self.k, sample=True)]
            row["consistency"] = ev["consistency"] = consistency(reply, s)
        if self.use_grounding:
            g = grounding(model, tok, prompt, question, a)
            row.update(g)
            ev.update({k: g[k] for k in GROUND_KEYS + ["top_question_token"]})
        pl = float(self.probe.logit(row["h"].float().reshape(1, -1).numpy())[0])
        ev["probe_logit"] = pl
        z = float(self.stacker.logit(stack_features(pl, row, self.use_consistency, self.use_grounding)[None])[0])
        zc = self.calib[0] * z + self.calib[1]
        return {"reply": reply, "p_correct": float(1 / (1 + np.exp(-zc))), "logit": z, "evidence": ev}
