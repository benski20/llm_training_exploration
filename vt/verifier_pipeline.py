"""The verifier pipeline: train a grounding probe for a reader model, calibrate it on a customer's own labeled
replies, evaluate it, and write a model card. Each step is a command; each writes what the next one reads.

    python -m vt.verifier_pipeline train     --reader Qwen/Qwen3-4B --out probe.npz [--documents docs.jsonl] [--n 600]
    python -m vt.verifier_pipeline calibrate --reader Qwen/Qwen3-4B --probe probe.npz --labeled labeled.jsonl \\
                                             --out probe.calibrated.npz [--false-alarms 0.05] [--holdout 0.5]
    python -m vt.verifier_pipeline evaluate  --reader Qwen/Qwen3-4B --probe probe.calibrated.npz \\
                                             --labeled test.jsonl --card card.md
    python -m vt.verifier_pipeline halubench --n 300 --out labeled.jsonl        # a labeled set to try it with

Labeled replies are JSON lines: {"question", "reply", "documents": [...], "hallucinated": 0 or 1, "source"?}.

  train      constructions from general documents (or --documents: JSON lines with "text"), labels by design
             (vt/probe_train.py); no human labels. Keep the documents general and varied: probes from 1,500
             constructions, or from the target domain's own documents, transferred worse than from 600 general
             ones (docs/grounding_trace.md).
  calibrate  on the customer's labeled replies: each signal's scale over their sentences (labels not used), the
             reply-total threshold for the chosen false-alarm rate, a logistic map from total to probability; with
             --holdout, measured on the part not used to set them. About 100 replies with both kinds is enough
             to set a threshold; the error rate it reports is only as precise as their number.
  evaluate   error rates on labeled replies at the calibrated threshold (and AUROC), per source when given,
             seconds per reply; the model card says what the probe was trained on, how it was calibrated, what it
             measured, and what it is for.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np


def load_reader(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto").eval()
        if torch.cuda.get_device_capability(0)[0] < 8:          # T4: no bfloat16 matmuls
            for m in model.modules():
                if getattr(m, "compute_dtype", None) == torch.bfloat16:
                    m.compute_dtype = torch.float16
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
    return model, tok


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recall_at_fpr(score, y, fpr):
    """The threshold above which at most `fpr` of the clean replies score, and the hallucinated share above it."""
    neg = np.sort(score[y == 0])
    t = float(neg[int(np.ceil((1 - fpr) * len(neg))) - 1]) if len(neg) else float("inf")
    return t, float((score[y == 1] > t).mean()) if (y == 1).any() else float("nan")


def rates(score, y, threshold):
    flag = score > threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    return {"caught": round(tp / max(1, int((y == 1).sum())), 3), "false_alarms": round(fp / max(1, int((y == 0).sum())), 3),
            "precision": round(tp / max(1, tp + fp), 3), "hallucinated": int((y == 1).sum()), "clean": int((y == 0).sum())}


def read_all(g, rows, progress=True):
    """Every labeled reply's raw sentence readings (two passes each), and seconds per reply."""
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        out.append(g.raw([{"role": "user", "content": r.get("question", "")}], str(r["reply"]),
                         documents=r.get("documents") or (), signals=True))
        if progress and (i + 1) % 25 == 0:
            print(f"read {i + 1}/{len(rows)} {(time.time() - t0) / (i + 1):.1f}s a reply", flush=True)
    return out, (time.time() - t0) / max(1, len(rows))


def cmd_train(a):
    from vt.probe_train import construct, general_documents, save_probe, train_probe
    docs = [r["text"] for r in read_jsonl(a.documents)] if a.documents else general_documents()
    examples = construct(docs, n=a.n, seed=a.seed)
    model, tok = load_reader(a.reader)
    artifact, report = train_probe(model, tok, examples, model_name=a.reader,
                                   progress=lambda i, n: print(f"constructions {i}/{n}", flush=True))
    save_probe(artifact, a.out)
    report.update(reader=a.reader, documents=a.documents or "general (Dolly reference paragraphs)", probe=a.out,
                  sha256=sha256(a.out))
    json.dump(report, open(os.path.splitext(a.out)[0] + ".train.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


def cmd_calibrate(a):
    from sklearn.linear_model import LogisticRegression

    from vt.correctness import auroc
    from vt.grounding import Grounding
    from vt.probe_train import load_probe, save_probe
    rows = read_jsonl(a.labeled)
    rng = random.Random(a.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = len(idx) - int(a.holdout * len(idx))
    fit, held = [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]
    model, tok = load_reader(a.reader)
    g = Grounding(model, tok, a.probe)
    raw_fit, secs = read_all(g, fit)
    sent = [s for r in raw_fit for s in r]
    natural = {k: (float(np.mean([s[k] for s in sent])), float(np.std([s[k] for s in sent])))
               for k in ("score", "mean_shift", "max_disagreement")}            # labels not used
    g.natural = {"probe": natural["score"], "mean_shift": natural["mean_shift"],
                 "max_disagreement": natural["max_disagreement"]}
    g.calibration = None
    tot = np.array([g.score(r)["total"] for r in raw_fit])
    y = np.array([int(r["hallucinated"]) for r in fit])
    if y.min() == y.max():
        raise SystemExit("the labeled replies need both clean and hallucinated ones")
    thr, _ = recall_at_fpr(tot, y, a.false_alarms)
    lr = LogisticRegression().fit(tot[:, None], y)
    art = load_probe(a.probe)
    art.update({f"nat_{k}_{s}": np.float32(v) for k, (m, sd) in g.natural.items() for s, v in (("mean", m), ("std", sd))})
    art.update(cal_threshold=np.float32(thr), cal_a=np.float32(lr.coef_[0][0]), cal_b=np.float32(lr.intercept_[0]),
               cal_false_alarms=np.float32(a.false_alarms), cal_n=np.int32(len(fit)))
    save_probe(art, a.out)
    report = {"probe": a.out, "from": a.probe, "reader": a.reader, "labeled": a.labeled, "target_false_alarms": a.false_alarms,
              "fit": dict(n=len(fit), threshold=round(thr, 4), auroc=round(float(auroc(tot, y.astype(float))), 3),
                          **rates(tot, y, thr)), "seconds_per_reply": round(secs, 2), "sha256": sha256(a.out)}
    if held:                                                     # measured on replies not used to set anything
        g2 = Grounding(model, tok, a.out)
        raw_h, _ = read_all(g2, held)
        th = np.array([g2.score(r)["total"] for r in raw_h])
        yh = np.array([int(r["hallucinated"]) for r in held])
        report["held_out"] = dict(n=len(held), auroc=round(float(auroc(th, yh.astype(float))), 3) if yh.min() != yh.max()
                                  else None, **rates(th, yh, thr))
    json.dump(report, open(os.path.splitext(a.out)[0] + ".calibration.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


def cmd_evaluate(a):
    from vt.correctness import auroc
    from vt.grounding import Grounding
    rows = read_jsonl(a.labeled)
    model, tok = load_reader(a.reader)
    g = Grounding(model, tok, a.probe)
    if g.calibration is None:
        raise SystemExit("evaluate needs a calibrated probe (vt.verifier_pipeline calibrate)")
    raw, secs = read_all(g, rows)
    tot = np.array([g.score(r)["total"] for r in raw])
    y = np.array([int(r["hallucinated"]) for r in rows])
    thr = g.calibration[0]
    res = {"probe": a.probe, "sha256": sha256(a.probe), "reader": a.reader, "labeled": a.labeled, "n": len(rows),
           "auroc": round(float(auroc(tot, y.astype(float))), 3) if y.min() != y.max() else None,
           "at_threshold": rates(tot, y, thr), "threshold": round(thr, 4), "seconds_per_reply": round(secs, 2)}
    sources = sorted({r.get("source") for r in rows if r.get("source")})
    if sources:
        res["per_source"] = {}
        for src in sources:
            m = np.array([r.get("source") == src for r in rows])
            res["per_source"][src] = dict(auroc=round(float(auroc(tot[m], y[m].astype(float))), 3)
                                          if y[m].min() != y[m].max() else None, **rates(tot[m], y[m], thr))
    out = os.path.splitext(a.card)[0]
    json.dump(res, open(out + ".json", "w"), indent=1)
    open(a.card, "w").write(model_card(res, a.probe))
    print(json.dumps(res, indent=1))


def model_card(res, probe):
    z = np.load(probe)
    train_path = os.path.splitext(probe.replace(".calibrated", ""))[0] + ".train.json"
    train = json.load(open(train_path)) if os.path.exists(train_path) else {}
    t = res["at_threshold"]
    lines = [f"# Grounding verifier: {res['reader']}", "",
             f"Probe `{os.path.basename(probe)}` (sha256 `{res['sha256'][:16]}…`), reader `{res['reader']}`.", "",
             "## What it does", "",
             "Reads each sentence of a reply against the evidence it should rest on (the documents, the conversation, "
             "exact re-computation of stated arithmetic, parsed code) from the reader model's own neurons and "
             "probabilities, in two forward passes, and flags the sentences the evidence contradicts or does not "
             "hold. It checks grounding, not truth: a claim no evidence addresses is not marked wrong.", "",
             "## Training", "",
             f"- Constructions: {train.get('constructions', '?')} from {train.get('documents', 'general documents')}, "
             "labels by design (no human labels)",
             f"- Held-out construction AUROC: {train.get('held_out_construction_auroc', '?')}"
             + (f" ({train['warning']})" if train.get("warning") else ""), "",
             "## Calibration", "",
             f"- On {int(z['cal_n']) if 'cal_n' in z.files else '?'} labeled replies, threshold "
             f"{res['threshold']} for {float(z['cal_false_alarms']) if 'cal_false_alarms' in z.files else '?'} false alarms",
             "", "## Measured", "",
             f"On {res['n']} labeled replies (`{res['labeled']}`): AUROC {res['auroc']}; at the threshold "
             f"{t['caught']:.0%} of {t['hallucinated']} hallucinated replies caught, {t['false_alarms']:.0%} of "
             f"{t['clean']} clean ones flagged, precision {t['precision']:.0%}; {res['seconds_per_reply']} s a reply.", ""]
    if res.get("per_source"):
        lines += ["| source | AUROC | caught | false alarms |", "|---|---|---|---|"]
        lines += [f"| {s} | {v['auroc']} | {v['caught']:.0%} | {v['false_alarms']:.0%} |" for s, v in res["per_source"].items()]
        lines += [""]
    lines += ["## Intended use and limits", "",
              "- For replies written from evidence: RAG answers, summaries, document Q&A, support bots on a knowledge base.",
              "- Closed-book questions depend on the lookup finding the deciding fact; without it errors pass.",
              "- Calibrate on your own replies; the false-alarm rate holds for data like the calibration set.", ""]
    return "\n".join(lines)


def cmd_halubench(a):
    from datasets import load_dataset
    rows = list(load_dataset("PatronusAI/HaluBench", split="test"))
    random.Random(a.seed).shuffle(rows)
    per = {}
    out = []
    for r in rows:
        key = (r["source_ds"], r["label"])
        if per.get(key, 0) < a.n // 12:
            per[key] = per.get(key, 0) + 1
            out.append({"question": r["question"], "reply": str(r["answer"]), "documents": [r["passage"]],
                        "hallucinated": int(r["label"] == "FAIL"), "source": r["source_ds"]})
    write_jsonl(out, a.out)
    print(f"{len(out)} labeled replies in {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vt.verifier_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--reader", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--documents", help="JSON lines with a \"text\" field (default: general documents)")
    t.add_argument("--n", type=int, default=600)
    t.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate")
    for p in (c,):
        p.add_argument("--reader", required=True)
        p.add_argument("--probe", required=True)
        p.add_argument("--labeled", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--false-alarms", type=float, default=0.05)
        p.add_argument("--holdout", type=float, default=0.5)
        p.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--reader", required=True)
    e.add_argument("--probe", required=True)
    e.add_argument("--labeled", required=True)
    e.add_argument("--card", required=True, help="the model card (.md; its numbers also as .json)")
    h = sub.add_parser("halubench")
    h.add_argument("--n", type=int, default=300)
    h.add_argument("--out", required=True)
    h.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    {"train": cmd_train, "calibrate": cmd_calibrate, "evaluate": cmd_evaluate, "halubench": cmd_halubench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
