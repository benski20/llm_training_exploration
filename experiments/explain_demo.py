"""What an explanation looks like: ask a Glassbox model a chat question and print, for its answer,
the exact attribution of the key token to the conversation, the features that fired (labelled
by the tokens they fire on most), and the memory certificate.

    python experiments/explain_demo.py "User: Fact: Borita works for Rimeti. Where does Borita work?\\nAssistant:"
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import glassbox_chat as C  # noqa: E402
from chat_eval_large import MODELS  # noqa: E402
from selective_answering import DATA  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


@torch.no_grad()
def feature_labels(m, tok, stream, layer, windows=150, top=8):
    """Label each feature by the tokens it fires on most strongly in a sample of training text."""
    g = torch.Generator().manual_seed(3)
    ix = torch.randint(len(stream) - 129, (windows,), generator=g)
    best = defaultdict(list)
    for i in range(0, windows, 16):
        X = torch.stack([stream[j:j + 128] for j in ix[i:i + 16].tolist()])
        m(X)
        acts = m.trace["feat"][layer]["acts"]
        v, f = acts.topk(4, -1)
        for b, t, k in zip(*torch.nonzero(v > 0, as_tuple=True)):
            best[int(f[b, t, k])].append((float(v[b, t, k]), int(X[b, t])))
    labels = {}
    for feat, xs in best.items():
        xs.sort(reverse=True)
        c = Counter(tok.decode([t]).strip() or repr(tok.decode([t])) for _, t in xs[:40])
        labels[feat] = ", ".join(f"'{w}'" for w, _ in c.most_common(3))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="glassbox2_4")
    ap.add_argument("--answer-word", action="store_true", help="explain the reply's last word, not its least certain token")
    ap.add_argument("--alt", default=None, help="contrast: why the reply's token rather than this word?")
    a = ap.parse_args()
    torch.set_num_threads(2)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    name, make, ckpt = MODELS[a.model]
    model = make(tok.get_vocab_size())
    model.m.load_state_dict(torch.load(ckpt)["model"])
    model.m.eval()
    m = model.m
    prompt = a.prompt.replace("\\n", "\n")
    out, f = C.respond(model, tok, prompt, 256)
    ids = f["ids"][0].tolist()
    P = len(tok.encode(prompt).ids)
    print(f"Model: {name}\nPrompt: {prompt!r}\nReply:  {out!r}\n")
    with torch.no_grad():
        m(torch.tensor([ids]))
        k = torch.sigmoid(m.know(m.trace["z_f"]).squeeze(-1))[0]
        free = m.memory_free()[0]
    span = list(range(P - 1, len(ids) - 1))
    t = min(span, key=lambda i: float(k[i]))                      # the reply's least certain token
    if a.answer_word:                                             # first token of the reply's last word
        words = [i for i in span if tok.decode([ids[i + 1]]).startswith(" ")]
        t = words[-1] if words else t
    target = ids[t + 1]
    print(f"Key token: {tok.decode([target])!r} (confidence {float(k[t]):.2f}; "
          f"made without stored memory: {bool(free[t])}, proved by a closed gate)\n")
    parts = m.explain(torch.tensor([ids]), t, target)
    total = sum(parts.values())
    pos = sorted(((v, n) for n, v in parts.items() if v > 0), reverse=True)
    print(f"Exact decomposition of its score ({total:.3f}; the parts sum to it exactly):")
    for v, n in pos[:8]:
        kind, j = n.split("@")
        where = f"input token {tok.decode([ids[int(j)]])!r} (position {j})" if kind == "token" else f"{kind} {j}"
        print(f"  {v:+7.3f}  {where}")
    neg = sorted((v, n) for n, v in parts.items() if v < 0)[:3]
    for v, n in neg:
        kind, j = n.split("@")
        where = f"input token {tok.decode([ids[int(j)]])!r} (position {j})" if kind == "token" else f"{kind} {j}"
        print(f"  {v:+7.3f}  {where}  (argues against)")
    if a.alt:
        alt = tok.encode(" " + a.alt).ids[0]
        alt_parts = m.explain(torch.tensor([ids]), t, alt)
        diff = {n: parts[n] - alt_parts[n] for n in parts}
        print(f"\nWhy {tok.decode([target])!r} and not {tok.decode([alt])!r}? Exact decomposition of the score gap "
              f"({sum(diff.values()):+.3f}):")
        for n, v in sorted(diff.items(), key=lambda x: -abs(x[1]))[:8]:
            kind, j = n.split("@")
            where = f"input token {tok.decode([ids[int(j)]])!r} (position {j})" if kind == "token" else f"{kind} {j}"
            print(f"  {v:+7.3f}  {where}  ({'favours ' + repr(tok.decode([target])) if v > 0 else 'favours ' + repr(tok.decode([alt]))})")
    layer = m.config.n_layer - 1
    labels = feature_labels(m, tok, torch.load(f"{DATA}/stream.pt"), layer)
    m(torch.tensor([ids]))
    acts = m.trace["feat"][layer]["acts"][0, t]
    v, fi = acts.topk(6)
    print(f"\nStrongest active features at the last layer ({int((acts != 0).sum())} of {acts.numel()} active):")
    for val, feat in zip(v.tolist(), fi.tolist()):
        print(f"  feature {feat:4d}  activation {val:5.2f}  fires most on: {labels.get(feat, '(rare)')}")


if __name__ == "__main__":
    main()
