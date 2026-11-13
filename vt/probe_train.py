"""Training a grounding probe for any reader model, without human labels (the pipeline behind vt/grounding.py).

Labels come from construction: from any documents, a reply sentence is taken from a document (grounded), the
same sentence with a number or a name swapped (contradicted), or a sentence taken out of the document shown
(unsupported: on topic, not in the context), all worded like part of an answer so the form gives nothing away.
The reader reads each with its document in context; a linear probe on its MLP neurons (down_proj inputs,
every 4th layer, averaged over the reply's tokens) learns grounded from not.

What was measured (docs/grounding_trace.md): 600 constructions from general Wikipedia-style text gave the best
transfer to human-labeled hallucinations in other domains (RAGTruth 0.82 per response for Qwen3-4B); more
constructions (1,500) and constructions from the target domain's own documents both did worse, the probe
learning how the constructions were made. Keep the documents general and varied.

    from vt.probe_train import construct, train_probe
    examples = construct(documents, n=600)
    artifact, report = train_probe(model, tok, examples)
    save_probe(artifact, "heads/grounding_probe_mymodel.npz")
"""
import random
import re

import numpy as np

LEADS = ("", "", "According to the text, ", "The passage says that ", "It notes that ", "In short, ", "Notably, ")


def sent_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 8]


def corrupt(s, rng, names):
    """The sentence with one number or one capitalised name changed, or None if it has neither."""
    nums = list(re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", s))
    caps = [m for m in re.finditer(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{2,}\b", s)]
    if nums and (not caps or rng.random() < 0.5):
        m = rng.choice(nums)
        v = m.group(0).replace(",", "")
        try:
            x = float(v)
        except ValueError:
            return None
        new = str(int(x) + rng.choice([-3, -2, -1, 1, 2, 5, 10])) if x == int(x) else f"{x * rng.choice([0.5, 2, 1.5]):.1f}"
        return s[:m.start()] + new + s[m.end():]
    if caps:
        m = rng.choice(caps)
        other = [n for n in names if n != m.group(0)]
        return s[:m.start()] + rng.choice(other) + s[m.end():] if other else None
    return None


def phrase(s, rng):
    """The sentence worded like part of an answer: a lead-in, its parentheses dropped, its first letter lowered
    after a lead-in. Applied alike to every kind, so the form says nothing about the label."""
    s = re.sub(r"\s*\([^()]*\)", "", s).strip()
    lead = rng.choice(LEADS)
    return lead + (s[0].lower() + s[1:] if lead and s[:1].isupper() and not s[:2].isupper() else s)


def construct(documents, n=600, seed=0, phrasing=True):
    """n labeled examples {"context", "reply", "kind", "y"} from documents, a third of each kind."""
    rng = random.Random(seed)  # nosec B311: seeded sampling of evaluation data, not security
    docs = [d for d in documents if d and 60 <= len(d.split()) <= 600]
    rng.shuffle(docs)
    names = sorted({w for c in docs[:2000] for w in re.findall(r"\b[A-Z][a-z]{3,}\b", c)})
    out = []
    for c in docs:
        ss = sent_split(c)
        if not ss:
            continue
        s = rng.choice(ss)
        kind = ("grounded", "contradicted", "unsupported")[len(out) % 3]
        if kind == "contradicted":
            s2 = corrupt(s, rng, names)
            if s2 is None or s2 == s:
                continue
            s = s2
        elif kind == "unsupported":                          # on topic but not in the context: taken out of it
            if len(ss) < 3:
                continue
            c = c.replace(s, " ").strip()
        out.append({"context": c, "reply": phrase(s, rng) if phrasing else s, "kind": kind, "y": int(kind != "grounded")})
        if len(out) >= n:
            break
    return out


def general_documents():
    """The general documents the measured probes were built from: Dolly's reference paragraphs (CC-BY-SA)."""
    from datasets import load_dataset
    return [r["context"] for r in load_dataset("databricks/databricks-dolly-15k")["train"]  # nosec B615: public evaluation data, not served
            if r["context"] and 60 <= len(r["context"].split()) <= 400]


def chat_prefix(tok, user):
    msgs = [{"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(p, add_special_tokens=False).input_ids


class Reader:
    """MLP neuron activations (down_proj inputs) at every position, for every `every`-th layer; kept on the CPU
    in float16 (GPU memory is the model's)."""

    def __init__(self, model, every=4, layers=None):
        self.acts, self.model = {}, model
        self.layers = list(layers) if layers is not None else list(range(every - 1, len(model.model.layers), every))
        self.hooks = [model.model.layers[i].mlp.down_proj.register_forward_pre_hook(self._hook(i)) for i in self.layers]

    def _hook(self, i):
        def f(mod, inp):
            import torch
            self.acts[i] = inp[0][0].detach().to("cpu", torch.float16)
        return f

    def run(self, ids):
        import torch
        x = torch.tensor([ids], device=next(self.model.parameters()).device)
        with torch.no_grad():
            try:                                                # no logits needed: keep only the last one
                self.model(input_ids=x, logits_to_keep=1)
            except TypeError:
                self.model(input_ids=x)

    def mean(self, a, b):
        """Mean activation of positions [a, b) in every read layer, concatenated."""
        return np.concatenate([self.acts[i][a:b].float().mean(0).numpy() for i in self.layers]).astype(np.float32)

    def close(self):
        for h in self.hooks:
            h.remove()


def features(model, tok, examples, reader=None, progress=None):
    """Each example's reply read with its context in view: the probe's input vectors."""
    own = reader is None
    reader = reader or Reader(model)
    X = []
    try:
        for i, ex in enumerate(examples):
            pre = chat_prefix(tok, "Answer from this text.\n\n" + ex["context"])
            rep = tok(ex["reply"], add_special_tokens=False).input_ids
            reader.run(pre + rep)
            X.append(reader.mean(len(pre), len(pre) + len(rep)))
            if progress and (i + 1) % 100 == 0:
                progress(i + 1, len(examples))
    finally:
        if own:
            reader.close()
    return np.stack(X), reader.layers


def train_probe(model, tok, examples, C=0.01, model_name="", progress=None):
    """A probe fit to constructions: (artifact for save_probe, report with its held-out score on them)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from vt.correctness import auroc
    X, layers = features(model, tok, examples, progress=progress)
    y = np.array([ex["y"] for ex in examples])
    cut = int(0.8 * len(y))
    sc = StandardScaler().fit(X[:cut])
    held = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[:cut]), y[:cut])
    held_auroc = float(auroc(held.decision_function(sc.transform(X[cut:])), y[cut:].astype(float)))
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
    own = clf.decision_function(sc.transform(X))
    artifact = {"layers": np.array(layers), "mean": sc.mean_.astype(np.float32), "scale": sc.scale_.astype(np.float32),
                "coef": clf.coef_[0].astype(np.float32), "intercept": np.float32(clf.intercept_[0]),
                "score_mean": np.float32(own.mean()), "score_std": np.float32(own.std()), "model": np.array(model_name),
                "grounded_mean": np.float32(own[y == 0].mean()), "ungrounded_mean": np.float32(own[y == 1].mean())}
    report = {"constructions": len(y), "kinds": {k: sum(ex["kind"] == k for ex in examples)
                                                 for k in ("grounded", "contradicted", "unsupported")},
              "held_out_construction_auroc": round(held_auroc, 3), "layers": layers, "features": int(X.shape[1])}
    if held_auroc > 0.97:                                    # what preceded the probes that transferred worst
        report["warning"] = ("constructions this easy to tell apart taught earlier probes how they were made, "
                             "not grounding: check against real labeled replies before trusting it")
    return artifact, report


def save_probe(artifact, path):
    np.savez(path, **artifact)


def load_probe(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
