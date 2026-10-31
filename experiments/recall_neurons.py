"""Do clean neurons carry "I know this" in a way that transfers? If most neuron poles have one job (Goodfire,
"Neural Networks Are Already Clean"), the model's familiarity machinery should be a handful of neuron poles
that can be found without any hallucination labels, and should mean the same in every domain.

    pick   neuron poles (MLP down_proj inputs, positive and negative sides apart) that separate questions about
           real entities from questions about invented ones, read at the position where the answer starts:
           a contrast with no answers and no correctness labels in it
    test   closed-book answers of the same model to TriviaQA and Natural Questions questions, graded against
           their reference answers: does low familiarity mark the wrong ones?

next to: the answer's own log-probability, a dense probe on all neurons fit to the same contrast, and the
usual approach, a dense probe fit to one dataset's correctness labels and tested on the other (the transfer
that failed for this repo's trained heads). AUROC for "wrong" everywhere.

    python experiments/recall_neurons.py --model Qwen/Qwen3-4B --n 300 --out results/recall_neurons_qwen3_4b.json
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REAL = ["Marie Curie", "Albert Einstein", "Leonardo da Vinci", "Nelson Mandela", "William Shakespeare",
        "Cleopatra", "Isaac Newton", "Frida Kahlo", "Mahatma Gandhi", "Ludwig van Beethoven", "Abraham Lincoln",
        "Charles Darwin", "Ada Lovelace", "Napoleon Bonaparte", "Jane Austen", "Pablo Picasso", "Nikola Tesla",
        "Martin Luther King Jr.", "Wolfgang Amadeus Mozart", "Galileo Galilei", "Queen Victoria", "Confucius",
        "Vincent van Gogh", "Florence Nightingale", "Julius Caesar", "Genghis Khan", "Alan Turing",
        "the Eiffel Tower", "Mount Everest", "the Amazon River", "the Great Wall of China", "the Sahara",
        "Lake Baikal", "the Colosseum", "Machu Picchu", "the Taj Mahal", "Kyoto", "Buenos Aires", "Reykjavik",
        "the Danube", "Tasmania", "the Mona Lisa", "War and Peace", "Don Quixote", "the Odyssey", "Moby-Dick",
        "Hamlet", "the Beatles", "Microsoft", "Toyota", "the United Nations", "NASA", "the Roman Empire",
        "the French Revolution", "World War I", "the Renaissance", "penicillin", "photosynthesis", "the Higgs boson",
        "DNA", "the Pythagorean theorem", "the Magna Carta", "Wikipedia", "the Olympic Games", "Bitcoin",
        "the Statue of Liberty", "Stonehenge", "the Nile", "Antarctica", "Jupiter"]
FIRST = ["Velmira", "Toran", "Quessa", "Dalvik", "Orsella", "Branko", "Ilthea", "Marvo", "Seline", "Kastor",
         "Pellam", "Udrin", "Zaria", "Fennick", "Lorcan", "Ysolde", "Tibor", "Nerys", "Calder", "Omera"]
LAST = ["Ostrand", "Vekhart", "Lindqvorst", "Marabelle", "Tchenko", "Dravell", "Quillon", "Haskerby", "Ferrows",
        "Nadrasi", "Polgrave", "Esterhal", "Wintrop", "Korvacs", "Belladine", "Trussock", "Varenko", "Ashgrove"]
PLACES = ["the Velkar Basin", "Lake Ormundsee", "the Tressany Peaks", "the city of Quorvath", "the river Halsyne",
          "Mount Pellerin-Kost", "the island of Drevmoor", "the Brannick Steppe", "Port Ilvaran", "the Osterhide Gorge"]
WORKS = ["the novel The Glass Orchard of Venn", "the Treaty of Malvenhurst", "the Corvane Prize", "the Hessling Effect",
         "the opera Lurenza", "the Battle of Tarnwick", "Quessaline Industries", "the Dorvan Institute",
         "the Pellacott theorem", "the Order of the Ashen Gate"]
TEMPLATES = ["Who or what is {}?", "Tell me one fact about {}.", "When did {} become famous?", "Where is {} known from?"]


def fake_entities(n, rng):
    people = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    return list(dict.fromkeys(people + PLACES + WORKS))[:n]


def chat_prompt(tok, question):
    msgs = [{"role": "user", "content": question + " Answer in a few words."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Reader:
    """Every MLP neuron's activation (the input to down_proj) at the last position of each prompt."""

    def __init__(self, model):
        self.model, self.acts = model, {}
        self.layers = model.model.layers
        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        def f(mod, inp):
            self.acts[i] = inp[0][0, -1].float().cpu().numpy()
        return f

    def read(self, ids):
        import torch
        with torch.no_grad():
            self.model(input_ids=torch.tensor([ids]))
        return np.concatenate([self.acts[i] for i in range(len(self.layers))]).astype(np.float16)


def answer(model, tok, ids, max_new=24):
    import torch
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=max_new, do_sample=False, output_scores=True,
                             return_dict_in_generate=True)
    new = out.sequences[0, len(ids):]
    lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new)]
    return tok.decode(new, skip_special_tokens=True).strip(), float(np.mean(lps)) if lps else 0.0


def pole_auroc(X, y):
    """AUROC of each neuron pole for y: positive side (activation) and negative side (-activation), as one
    array of length 2 x neurons: positive poles first."""
    from vt.correctness import auroc
    Xp, Xn = np.maximum(X, 0), np.maximum(-X, 0)
    return np.array([auroc(Xp[:, j], y) for j in range(X.shape[1])] + [auroc(Xn[:, j], y) for j in range(X.shape[1])])


def fast_pole_auroc(X, y):
    """The same, by ranks (vectorised): AUROC = (mean rank of positives - (n1+1)/2) / n0."""
    from scipy.stats import rankdata
    P = np.concatenate([np.maximum(X, 0), np.maximum(-X, 0)], 1).astype(np.float32)
    r = np.apply_along_axis(rankdata, 0, P)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300, help="questions from each of TriviaQA and NQ")
    ap.add_argument("--ks", default="1,3,10,30")
    ap.add_argument("--out", default="recall_neurons.json")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from correction_rate import is_right
    from vt.correctness import auroc
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    reader = Reader(model)
    t0 = time.time()

    # the contrast: real vs invented entities, no answers, no labels of correctness
    fakes = fake_entities(len(REAL), rng)
    contrast = [(t.format(e), 1) for e in REAL for t in rng.sample(TEMPLATES, 2)] + \
               [(t.format(e), 0) for e in fakes for t in rng.sample(TEMPLATES, 2)]
    rng.shuffle(contrast)
    Xc = np.stack([reader.read(tok(chat_prompt(tok, q), add_special_tokens=False).input_ids) for q, _ in contrast])
    yc = np.array([lab for _, lab in contrast], float)
    print(f"contrast read: {len(contrast)} prompts, {time.time() - t0:.0f}s", flush=True)

    # the test: closed-book answers, graded
    sets = {}
    tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0).select(range(a.n))
    nq = load_dataset("google-research-datasets/nq_open", split="validation").shuffle(seed=0).select(range(a.n))
    for name, rows in (("triviaqa", [(r["question"], r["answer"]["aliases"]) for r in tq]),
                       ("nq", [(r["question"], r["answer"]) for r in nq])):
        X, right, lp, texts = [], [], [], []
        for i, (q, aliases) in enumerate(rows):
            ids = tok(chat_prompt(tok, q), add_special_tokens=False).input_ids
            X.append(reader.read(ids))
            text, mlp = answer(model, tok, ids)
            texts.append(text)
            right.append(bool(is_right(text, aliases)))
            lp.append(mlp)
            if (i + 1) % 50 == 0:
                print(f"{name} {i + 1}/{len(rows)} {time.time() - t0:.0f}s accuracy {np.mean(right):.2f}", flush=True)
        sets[name] = {"X": np.stack(X), "wrong": 1 - np.array(right, float), "logprob": np.array(lp),
                      "texts": texts}

    # neuron poles that mark the invented entities (low familiarity): AUROC for "invented" on the contrast
    fam = fast_pole_auroc(Xc.astype(np.float32), 1 - yc)          # high: fires for invented entities
    order = np.argsort(-np.abs(fam - 0.5))                         # strongest separators either way
    n_neurons = Xc.shape[1]

    def pole_values(X, idx):
        j, neg = idx % n_neurons, idx >= n_neurons
        v = np.where(neg, np.maximum(-X[:, j].astype(np.float32), 0), np.maximum(X[:, j].astype(np.float32), 0))
        return v * np.sign(fam[idx] - 0.5)                         # oriented: high = unfamiliar

    res = {"model": a.model, "contrast": {"n": len(contrast), "best_pole_auroc": round(float(np.abs(fam - 0.5).max() + 0.5), 3)},
           "accuracy": {k: round(float(1 - v["wrong"].mean()), 3) for k, v in sets.items()}}
    layers = len(reader.layers)
    per = n_neurons // layers
    res["top_poles"] = [{"layer": int((i % n_neurons) // per), "neuron": int((i % n_neurons) % per),
                         "pole": "negative" if i >= n_neurons else "positive",
                         "contrast_auroc_invented": round(float(fam[i]), 3)} for i in order[:10]]
    for k in [int(x) for x in a.ks.split(",")]:
        for name, s in sets.items():
            score = np.mean([pole_values(s["X"], i) / (np.abs(pole_values(Xc, i)).std() + 1e-6) for i in order[:k]], 0)
            res.setdefault("sparse_poles_from_contrast", {}).setdefault(f"k={k}", {})[name] = \
                round(float(auroc(score, s["wrong"])), 3)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)                # noqa: E731
    for name, s in sets.items():
        res.setdefault("answer_logprob", {})[name] = round(float(auroc(-s["logprob"], s["wrong"])), 3)
        k10 = np.mean([pole_values(s["X"], i) / (np.abs(pole_values(Xc, i)).std() + 1e-6) for i in order[:10]], 0)
        s["scores"] = {"logprob": s["logprob"].tolist(), "poles10": k10.tolist(), "wrong": s["wrong"].tolist()}
        res.setdefault("logprob_plus_poles10", {})[name] = round(float(auroc(z(-s["logprob"]) + z(k10), s["wrong"])), 3)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def dense(Xtr, ytr, C=0.05):
        sc = StandardScaler().fit(Xtr.astype(np.float32))
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr.astype(np.float32)), ytr)
        return lambda X: clf.decision_function(sc.transform(X.astype(np.float32)))
    f = dense(Xc, 1 - yc)
    res["dense_probe_from_contrast"] = {n: round(float(auroc(f(s["X"]), s["wrong"])), 3) for n, s in sets.items()}
    for src, dst in (("triviaqa", "nq"), ("nq", "triviaqa")):
        f = dense(sets[src]["X"], sets[src]["wrong"])
        res.setdefault("dense_probe_on_labels_transferred", {})[f"{src} -> {dst}"] = \
            round(float(auroc(f(sets[dst]["X"]), sets[dst]["wrong"])), 3)
    res["seconds"] = round(time.time() - t0)
    print(json.dumps(res, indent=1))
    json.dump(dict(res, answers={k: v["texts"] for k, v in sets.items()},
                   scores={k: v["scores"] for k, v in sets.items()}), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
