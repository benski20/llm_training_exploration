"""Train and evaluate the large-scale grounding probe on features from kaggle/probe/extract.py.

Protocol (fixed before looking at any test result):
  - training features: every shard under --train (sources from experiments/probe_data.py; test sets excluded there);
    a 10% validation split by id hash is held out for early stopping, model selection and thresholds;
  - evaluation features: every shard under --eval (probe_evalsets.py: RAGTruth official test, HaluBench sample,
    business test). Nothing from them is used for fitting, selection or thresholds;
  - RAGTruth is scored as published: response-level precision / recall / F1 (threshold that maximises F1 on the
    RAGTruth answers of the validation split) and AUROC; the others by AUROC and balanced accuracy.

Models:
  ans_lr    L2 logistic on the answer vector [a_mean, a_judge, a_neu over all depths, a_sc, j_logit]
  unit_lr   L2 logistic on sentence vectors [u_res, u_neu, u_sc], trained on sentences with labels (span-labelled
            sources, and every sentence of a faithful answer); an answer scores its worst sentence
  ans_mlp   two-layer MLP on the answer vector (torch, CPU), early-stopped on validation
  set_attn  sentences -> shared MLP encoder -> attention pooling, concatenated with the answer vector -> logit;
            loss = answer BCE + 0.5 x sentence BCE where sentence labels exist
  stack     logistic on the validation scores of the four (fit on validation, so a slight optimism on val only)

    python experiments/probe_train.py --train results/probe/q17/train --eval results/probe/q17/eval \
        --out results/probe/q17/report.json --save heads/grounding_probe_q17.pt
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vt.probe_features import answer_vector, make_mlp, make_set_attn, unit_vector  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def load(dirs):
    A = {k: [] for k in ("ids", "label", "source", "task", "X", "j")}
    U = {k: [] for k in ("row", "lab", "X")}
    meta = None
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "shard*.npz"), recursive=True)):
            z = np.load(p, allow_pickle=True)
            mp = os.path.join(os.path.dirname(p), "meta.json")
            if meta is None and os.path.exists(mp):
                meta = json.load(open(mp))
            n0 = len(A["ids"])
            n = len(z["ids"])
            A["X"].append(answer_vector(z["a_mean"], z["a_judge"], z["a_neu"], z["a_sc"], z["j_logit"]))
            for k in ("ids", "label", "source", "task"):
                A[k] += z[k].tolist()
            m = len(z["u_row"])
            if m:
                U["X"].append(unit_vector(z["u_res"], z["u_neu"], z["u_sc"]))
                U["row"] += (z["u_row"] + n0).tolist()
                U["lab"] += z["u_lab"].tolist()
    if not A["X"]:
        return None
    A["X"] = np.concatenate(A["X"])
    U["X"] = np.concatenate(U["X"]) if U["X"] else np.zeros((0, 1), np.float16)
    for k in ("label",):
        A[k] = np.array(A[k])
    for k in ("ids", "source", "task"):
        A[k] = np.array(A[k])
    U["row"], U["lab"] = np.array(U["row"], int), np.array(U["lab"], int)
    A["meta"] = meta
    return A, U


def auroc(s, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def best_f1_threshold(s, y):
    order = np.argsort(-s)
    ys = np.asarray(y)[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(1, ys.sum())
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1))
    return float(s[order][k])


def prf(s, y, t):
    y = np.asarray(y)
    p = s >= t
    tp = float((p & (y == 1)).sum())
    P = tp / max(1, p.sum())
    R = tp / max(1, (y == 1).sum())
    return {"precision": float(100 * P), "recall": float(100 * R), "f1": float(100 * 2 * P * R / max(P + R, 1e-9))}


def worst_unit(unit_scores, rows, n, default):
    out = np.full(n, default, float)
    np.maximum.at(out, rows, unit_scores)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save")
    ap.add_argument("--models", default="ans_lr,unit_lr,ans_mlp,set_attn")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    from sklearn.linear_model import LogisticRegression
    tr = load(a.train)
    ev = load(a.eval)
    A, U = tr
    E, EU = ev
    val = np.array([int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0 for i in A["ids"]])
    log("train answers", int((~val).sum()), "val", int(val.sum()), "units", len(U["row"]), "eval answers", len(E["ids"]))
    log("train sources", {s: int((A["source"] == s).sum()) for s in np.unique(A["source"])})
    mu = A["X"][~val].astype(np.float32).mean(0)
    sd = A["X"][~val].astype(np.float32).std(0) + 1e-3
    umu = U["X"].astype(np.float32).mean(0)
    usd = U["X"].astype(np.float32).std(0) + 1e-3
    Z = lambda X: (X.astype(np.float32) - mu) / sd
    ZU = lambda X: (X.astype(np.float32) - umu) / usd
    y = A["label"]
    scores = {}                                              # model -> (val scores, eval scores)
    models = a.models.split(",")
    utr = (U["lab"] >= 0) & ~val[U["row"]]

    if "ans_lr" in models:
        m = LogisticRegression(C=0.02, class_weight="balanced", max_iter=300)
        m.fit(Z(A["X"][~val]), y[~val])
        scores["ans_lr"] = (m.decision_function(Z(A["X"][val])), m.decision_function(Z(E["X"])))
        log("ans_lr val AUROC", auroc(scores["ans_lr"][0], y[val]))
        ans_lr = m
    if "unit_lr" in models and utr.sum() > 100:
        m = LogisticRegression(C=0.02, class_weight="balanced", max_iter=300)
        m.fit(ZU(U["X"][utr]), U["lab"][utr])
        su = m.decision_function(ZU(U["X"]))
        sv = worst_unit(su, U["row"], len(y), -1e9)[val]
        seu = worst_unit(m.decision_function(ZU(EU["X"])), EU["row"], len(E["ids"]), -1e9)
        scores["unit_lr"] = (sv, seu)
        log("unit_lr val AUROC", auroc(sv, y[val]))
        unit_lr = m

    # torch models
    nn = torch.nn
    Xt = torch.tensor(Z(A["X"]))
    yt = torch.tensor(y, dtype=torch.float32)
    tri, vai = np.where(~val)[0], np.where(val)[0]
    pos_w = torch.tensor(float((y[~val] == 0).sum() / max(1, (y[~val] == 1).sum())))

    def run(model, batch_fn, n_train, params):
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
        best, best_state, bad = -1, None, 0
        for ep in range(a.epochs):
            model.train()
            perm = np.random.default_rng(ep).permutation(tri)
            for b in range(0, len(perm), 256):
                loss = batch_fn(perm[b:b + 256], train=True)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                sv = np.concatenate([batch_fn(vai[b:b + 1024], train=False) for b in range(0, len(vai), 1024)])
            auc = auroc(sv, y[val])
            log(f"  epoch {ep} val AUROC {auc:.4f}")
            if auc > best + 1e-4:
                best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 3:
                    break
        model.load_state_dict(best_state)
        return best

    if "ans_mlp" in models:
        mlp = make_mlp(Xt.shape[1])

        def bf(idx, train):
            o = mlp(Xt[idx])[:, 0]
            if train:
                return nn.functional.binary_cross_entropy_with_logits(o, yt[idx], pos_weight=pos_w)
            return o.detach().numpy()
        run(mlp, bf, len(tri), mlp.parameters())
        mlp.eval()
        with torch.no_grad():
            scores["ans_mlp"] = (mlp(Xt[vai])[:, 0].numpy(), mlp(torch.tensor(Z(E["X"])))[:, 0].numpy())
        log("ans_mlp val AUROC", auroc(scores["ans_mlp"][0], y[val]))

    if "set_attn" in models and len(U["row"]):
        UXt = torch.tensor(ZU(U["X"]))
        rows_of = [[] for _ in range(len(y))]
        for k, r in enumerate(U["row"]):
            rows_of[r].append(k)
        ulab = torch.tensor(U["lab"], dtype=torch.float32)

        sa = make_set_attn(UXt.shape[1], Xt.shape[1])

        def bf2(idx, train):
            o, us = sa(idx, Xt, UXt, rows_of)
            if not train:
                return o.detach().numpy()
            loss = nn.functional.binary_cross_entropy_with_logits(o, yt[idx], pos_weight=pos_w)
            ul, ut = [], []
            for i, u in zip(idx, us):
                if u is None:
                    continue
                lab = ulab[rows_of[i]]
                m = lab >= 0
                if m.any():
                    ul.append(u[m])
                    ut.append(lab[m])
            if ul:
                loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(torch.cat(ul), torch.cat(ut))
            return loss
        run(sa, bf2, len(tri), sa.parameters())
        sa.eval()
        EXt, EUXt = torch.tensor(Z(E["X"])), torch.tensor(ZU(EU["X"]))
        erows = [[] for _ in range(len(E["ids"]))]
        for k, r in enumerate(EU["row"]):
            erows[r].append(k)
        with torch.no_grad():
            sv = np.concatenate([sa(vai[b:b + 1024], Xt, UXt, rows_of)[0].numpy() for b in range(0, len(vai), 1024)])
            ei = np.arange(len(E["ids"]))
            se = np.concatenate([sa(ei[b:b + 1024], EXt, EUXt, erows)[0].numpy() for b in range(0, len(ei), 1024)])
        scores["set_attn"] = (sv, se)
        log("set_attn val AUROC", auroc(sv, y[val]))

    # stack on validation scores
    names = list(scores)
    Sv = np.stack([scores[k][0] for k in names], 1)
    Se = np.stack([scores[k][1] for k in names], 1)
    smu, ssd = Sv.mean(0), Sv.std(0) + 1e-9
    st = LogisticRegression(C=1.0, class_weight="balanced").fit((Sv - smu) / ssd, y[val])
    scores["stack"] = (st.decision_function((Sv - smu) / ssd), st.decision_function((Se - smu) / ssd))

    # report
    rep = {"n_train": int((~val).sum()), "n_val": int(val.sum()), "train_sources":
           {s: int((A["source"] == s).sum()) for s in np.unique(A["source"])}, "reader": (A["meta"] or {}).get("reader"),
           "models": {}}
    rt_val = val & np.char.startswith(A["source"].astype(str), "ragtruth")
    for k, (sv, se) in scores.items():
        r = {"val_auroc": auroc(sv, y[val])}
        for src in np.unique(E["source"]):
            m = E["source"] == src
            r[f"{src}_auroc"] = auroc(se[m], E["label"][m])
        rt = E["source"] == "ragtruth_test"
        if rt.any() and rt_val.any():
            t = best_f1_threshold(scores[k][0][rt_val[val]], y[rt_val])     # chosen on RAGTruth train-val answers
            r["ragtruth_test"] = dict(prf(se[rt], E["label"][rt], t), auroc=auroc(se[rt], E["label"][rt]))
            for task in np.unique(E["task"][rt]):
                mt = rt & (E["task"] == task)
                r[f"ragtruth_test_{task}"] = dict(prf(se[mt], E["label"][mt], t), auroc=auroc(se[mt], E["label"][mt]))
        hb = np.char.startswith(E["source"].astype(str), "halubench")
        if hb.any():
            r["halubench_mean_auroc"] = float(np.nanmean([auroc(se[E["source"] == s_], E["label"][E["source"] == s_])
                                                          for s_ in np.unique(E["source"][hb])]))
        rep["models"][k] = r
        log(k, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()
                           if not isinstance(v, dict)}), "| RAGTruth test", r.get("ragtruth_test"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_scores.npz", ids=E["ids"], source=E["source"],
                        label=E["label"], **{k: v[1] for k, v in scores.items()})
    if a.save:
        torch.save({"meta": A["meta"], "mu": mu, "sd": sd, "umu": umu, "usd": usd,
                    "ans_lr": (ans_lr.coef_[0], ans_lr.intercept_[0]) if "ans_lr" in models else None,
                    "unit_lr": (unit_lr.coef_[0], unit_lr.intercept_[0]) if "unit_lr" in scores else None,
                    "ans_mlp": mlp.state_dict() if "ans_mlp" in scores else None,
                    "set_attn": sa.state_dict() if "set_attn" in scores else None,
                    "stack": {"names": names, "mu": smu, "sd": ssd, "w": st.coef_[0], "b": st.intercept_[0]},
                    "threshold": best_f1_threshold(scores["stack"][0], y[val]),
                    "report": rep}, a.save)
        log("saved", a.save)


if __name__ == "__main__":
    main()
