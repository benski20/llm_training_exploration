"""Which readout gets the most out of the same RAGTruth features? Run on the extraction shards of ragtruth_sota.py.

The learning curve of the linear sentence readout is flat (response F1 70.6 / 71.4 / 72.1 at 800 / 1,600 / 2,400
training answers, 600 test answers), so more of the same data will not reach the published 79. Variants, all on
the same frozen-reader features (the 4,096-dim projection of nine layers' MLP inputs, a sentence's mean):

    linear      class-balanced logistic regression per sentence; a reply's score is its worst sentence (shipped)
    context     the same, with the sentence's own vector minus its reply's mean vector appended, so the readout
                sees how a sentence departs from the rest of its reply
    mlp         one hidden layer (256, GELU, dropout), weight decay, early stopped on held-out answers
    mlp+context both
and, for the best sentence model, the reply score: worst sentence, or a small logistic readout on the reply's
sentence-score profile (max, top-2 mean, mean, share above 0.5, sentence count), fitted on out-of-fold scores.

Hyperparameters and thresholds come from 20% of the training answers; test answers are read once per variant.

    python experiments/readout_variants.py --out <shard dir> --report results/ragtruth_supervised/variants.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragtruth_sota import load, prf, response_scores  # noqa: E402


def with_context(d):
    X = d["X"].astype(np.float32)
    mean = np.zeros_like(X)
    for r in np.unique(d["resp"]):
        m = d["resp"] == r
        mean[m] = X[m].mean(0)
    return np.concatenate([X, X - mean], 1)


def linear(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]


def mlp(Xtr, ytr, wd, Xva=None, yva=None, hidden=256, epochs=40, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    t = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)          # noqa: E731
    X, y = t(Xtr), torch.tensor(ytr, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    pos = float(y.mean())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    best, state, bad = -1, None, 0
    Va = t(Xva) if Xva is not None else None
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(X)).split(256):
            opt.zero_grad()
            lossf(net(X[b]).squeeze(1), y[b]).backward()
            opt.step()
        if Va is not None:
            from vt.correctness import auroc
            net.eval()
            with torch.no_grad():
                s = auroc(net(Va).squeeze(1).numpy(), yva.astype(float))
            if s > best:
                best, state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 5:
                    break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    n_ep = ep + 1 - bad if Va is not None else epochs

    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(t(Z)).squeeze(1)).numpy()
    predict.epochs = n_ep
    return predict


def profile(p, resp, ids):
    rows = []
    for i in ids:
        s = np.sort(p[resp == i])[::-1] if (resp == i).any() else np.zeros(1)
        rows.append([s[0], s[:2].mean(), s.mean(), (s > 0.5).mean(), np.log1p(len(s))])
    return np.array(rows)


def best_threshold(rs, ry):
    ts = np.unique(rs)
    f1s = [prf(rs >= t, ry)["f1"] for t in ts]
    k = int(np.argmax(f1s))
    return float(ts[k]), f1s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from vt.correctness import auroc
    tr, te = load(a.out, "train"), load(a.out, "test")
    ids = np.unique(tr["resp_ids"])
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, size=len(ids) // 5, replace=False).tolist())
    is_val = np.array([r in val_ids for r in tr["resp"]])
    vids = np.array(sorted(val_ids))
    rv = {i: y for i, y in zip(tr["resp_ids"], tr["resp_y"])}
    ry_val = np.array([rv[i] for i in vids])
    feats = {"plain": (tr["X"].astype(np.float32), te["X"].astype(np.float32))}
    feats["context"] = (with_context(tr), with_context(te))
    res = {"train_answers": int(len(ids)), "test_answers": int(len(te["resp_ids"])), "variants": {}}

    def report(name, rs, thr, extra):
        flag = rs >= thr
        row = dict(prf(flag, te["resp_y"]), auroc=round(float(auroc(rs, te["resp_y"].astype(float))), 3), **extra)
        boot = []
        for _ in range(1000):
            i = rng.integers(0, len(rs), len(rs))
            boot.append(prf(flag[i], te["resp_y"][i])["f1"])
        row["f1_ci95"] = [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)]
        res["variants"][name] = row
        print(name, json.dumps(row), flush=True)
        json.dump(res, open(a.report, "w"), indent=1)

    models = {}
    for fname, (Xtr, Xte) in feats.items():
        for kind in ("linear", "mlp"):
            name = kind if fname == "plain" else f"{kind}+context"
            best = None
            for hp in ((0.001, 0.01) if kind == "linear" else (0.01, 0.1)):
                if kind == "linear":
                    f = linear(Xtr[~is_val], tr["y"][~is_val], hp)
                else:
                    f = mlp(Xtr[~is_val], tr["y"][~is_val], hp, Xtr[is_val], tr["y"][is_val])
                pv = f(Xtr[is_val])
                thr, vf1 = best_threshold(response_scores(pv, tr["resp"][is_val], vids), ry_val)
                if best is None or vf1 > best[0]:
                    best = (vf1, hp, thr, getattr(f, "epochs", None))
            vf1, hp, thr, ep = best
            if kind == "linear":
                f = linear(Xtr, tr["y"], hp)
            else:
                f = mlp(Xtr, tr["y"], hp, epochs=ep)                 # the epochs early stopping chose
            p = f(Xte)
            models[name] = (vf1, hp, ep, p)
            report(name, response_scores(p, te["resp"], te["resp_ids"]), thr,
                   {"hp": hp, "epochs": ep, "validation_f1": vf1,
                    "sentence_auroc": round(float(auroc(p, te["y"].astype(float))), 3)})

    # the reply score: a readout on the reply's sentence-score profile, for the sentence model best on validation
    name = max(models, key=lambda k: models[k][0])
    _, hp, ep, p_te = models[name]
    Xtr, Xte = feats["context" if "context" in name else "plain"]
    oof = np.zeros(len(tr["y"]))
    fold_of = {r: k for k, r in enumerate(rng.permutation(ids))}
    fold = np.array([fold_of[r] % 5 for r in tr["resp"]])
    for k in range(5):
        m = fold == k
        f = linear(Xtr[~m], tr["y"][~m], hp) if name.startswith("linear") else mlp(Xtr[~m], tr["y"][~m], hp, epochs=ep)
        oof[m] = f(Xtr[m])
    from sklearn.linear_model import LogisticRegression
    Ptr = profile(oof, tr["resp"], ids)
    ytr_r = np.array([rv[i] for i in ids])
    in_val = np.array([i in val_ids for i in ids])
    agg = LogisticRegression(class_weight="balanced").fit(Ptr[~in_val], ytr_r[~in_val])
    thr, vf1 = best_threshold(agg.predict_proba(Ptr[in_val])[:, 1], ytr_r[in_val])
    agg = LogisticRegression(class_weight="balanced").fit(Ptr, ytr_r)
    rs = agg.predict_proba(profile(p_te, te["resp"], te["resp_ids"]))[:, 1]
    report(f"{name} + profile", rs, thr, {"validation_f1": vf1, "profile_weights": agg.coef_[0].round(3).tolist()})


if __name__ == "__main__":
    main()
