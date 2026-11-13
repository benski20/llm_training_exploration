"""Does the cheap AI-text probe survive what a buyer will meet: a generator it never saw, and paraphrased text?

experiments/ai_text_detect.py found the hidden-state probe at mean AUROC 0.974 on held-out RAID domains, and a
false-accusation promise from 150 of a domain's human texts that holds. Two open questions decide whether that is a
product (results/ai_text/raid_qwen3_0.6b.json):

  1. unseen generator: there the probe had seen all 11 generators in training. Here every test is leave one
     domain AND one generator family out: the probe never saw the test domain or any model from the test family
     (OpenAI: GPT-2/3/4, ChatGPT; Meta: Llama-2-chat; Mistral: base, chat; MosaicML: MPT, chat; Cohere: base, chat).
  2. paraphrase: the probe caught 11-60% of paraphrased AI at the promise, Binoculars 61-82%. Two fixes, both
     still keeping the promise:
        probe+para   the probe trained with the other domains' paraphrased AI texts as well
        union        flag when the probe OR Binoculars passes its own threshold, each set from the 150 human texts
                     at half the error budget (2.5% false alarms, 97.5% confidence), so by the union bound the
                     pair keeps "5% false accusations with 95% confidence" without any fitting.

Reads the features ai_text_detect.py read wrote (AI_TEXT_SCRATCH); no model runs.

    AI_TEXT_SCRATCH=<dir> python experiments/ai_text_generalize.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_text_detect import DOMAINS, OUT, SCRATCH

FAMILIES = {"openai": ("gpt2", "gpt3", "gpt4", "chatgpt"), "meta": ("llama-chat",),
            "mistral": ("mistral", "mistral-chat"), "mosaicml": ("mpt", "mpt-chat"), "cohere": ("cohere", "cohere-chat")}


def fit_probe(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return lambda Z: clf.decision_function(sc.transform(Z))


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


def flagged(scores, thresholds, m):
    """Flagged when any score passes its own threshold."""
    return np.any([s[m] > t for s, t in zip(scores, thresholds)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--cal", type=int, default=150)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--report", default=f"{OUT}/raid_generalize.json")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    from vt.correctness import auroc
    from vt.verifier_pipeline import guaranteed_threshold

    with open(f"{SCRATCH}/sample.jsonl") as f:
        rows = [json.loads(line) for line in f]
    z = np.load(f"{SCRATCH}/features.npz")
    n = len(z["scalars"])
    rows = rows[:n]
    S, H = z["scalars"].astype(np.float64), z["hidden"].astype(np.float32)
    dom = np.array([r["domain"] for r in rows])
    y = np.array([r["y"] for r in rows])
    para = np.array([r["attack"] == "paraphrase" for r in rows])
    model = np.array([r["model"] for r in rows])
    fam = np.array([next((f for f, ms in FAMILIES.items() if m in ms), "human") for m in model])
    X = np.c_[H, S[:, :6], S[:, 1] / S[:, 2]]
    bino = -(S[:, 1] / S[:, 2])
    res = {"texts": n, "C": a.C}
    t0 = time.time()

    # 1. leave one domain and one generator family out
    fam_out = {f: {"auroc_probe": [], "auroc_binoculars": [], "caught_probe": [], "caught_binoculars": []}
               for f in FAMILIES}
    rng = np.random.default_rng(0)
    for d in DOMAINS:
        hum_d = np.where((dom == d) & (y == 0))[0]
        cals = [rng.choice(hum_d, size=a.cal, replace=False) for _ in range(a.splits)]
        for f in FAMILIES:
            tr = (dom != d) & ~para & (fam != f)
            score = fit_probe(X[tr], y[tr], a.C)
            te_ai = (dom == d) & ~para & (fam == f)
            sp = score(X[hum_d]), score(X[te_ai])
            for name, (sh, sa) in (("probe", sp), ("binoculars", (bino[hum_d], bino[te_ai]))):
                fam_out[f][f"auroc_{name}"].append(float(auroc(np.r_[sh, sa], np.r_[np.zeros(len(sh)), np.ones(len(sa))])))
                pos = {i: k for k, i in enumerate(hum_d)}
                caught = []
                for cal in cals:
                    thr = guaranteed_threshold(sh[[pos[i] for i in cal]], np.zeros(len(cal), int), a.fpr, 0.95)
                    caught.append(float((sa > thr).mean()))
                fam_out[f][f"caught_{name}"].append(float(np.mean(caught)))
            print(d, f, {k: round(v[-1], 3) for k, v in fam_out[f].items()}, f"{time.time() - t0:.0f}s", flush=True)
    res["unseen_family"] = {f: {k: round(float(np.mean(v)), 3) for k, v in d_.items()} for f, d_ in fam_out.items()}
    res["unseen_family"]["mean"] = {k: round(float(np.mean([res["unseen_family"][f][k] for f in FAMILIES])), 3)
                                    for k in fam_out["openai"]}
    save(res, a.report)

    # 2. paraphrase: probe trained with paraphrases, and the union of probe and Binoculars, at the promise
    probe, probe_para = np.zeros(n), np.zeros(n)
    for d in DOMAINS:
        tr = (dom != d) & ~para
        probe[dom == d] = fit_probe(X[tr], y[tr], a.C)(X[dom == d])
        tr = dom != d
        probe_para[dom == d] = fit_probe(X[tr], y[tr], a.C)(X[dom == d])
    detectors = {"binoculars": [bino], "probe": [probe], "probe+para": [probe_para],
                 "union(probe, binoculars)": [probe, bino], "union(probe+para, binoculars)": [probe_para, bino]}
    out = {}
    for name, ss in detectors.items():
        per = {}
        for d in DOMAINS:
            hum = np.where((dom == d) & (y == 0))[0]
            ai, pa = (dom == d) & (y == 1) & ~para, (dom == d) & para
            fa, ca, cp = [], [], []
            for _ in range(a.splits):
                cal = rng.choice(hum, size=a.cal, replace=False)
                test_h = np.setdiff1d(hum, cal)
                k = len(ss)                                    # union bound: split error and confidence budgets
                thrs = [guaranteed_threshold(s[cal], np.zeros(a.cal, int), a.fpr / k, 1 - 0.05 / k) for s in ss]
                fa.append(float(flagged(ss, thrs, test_h).mean()))
                ca.append(float(flagged(ss, thrs, ai).mean()))
                cp.append(float(flagged(ss, thrs, pa).mean()))
            per[d] = {"false_alarms_mean": round(float(np.mean(fa)), 3),
                      "splits_over_5pct": round(float(np.mean(np.array(fa) > a.fpr)), 3),
                      "caught": round(float(np.mean(ca)), 3), "caught_paraphrased": round(float(np.mean(cp)), 3)}
        per["mean"] = {k: round(float(np.mean([per[d][k] for d in DOMAINS])), 3) for k in per[DOMAINS[0]]}
        out[name] = per
        print(name, json.dumps(per["mean"]), flush=True)
    res["promise_5pct_from_150_human"] = out
    save(res, a.report)
    print(json.dumps(res["unseen_family"], indent=1))


if __name__ == "__main__":
    main()
