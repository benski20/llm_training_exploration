"""vt-omni: train, adapt, calibrate, evaluate and serve the omni verifier (vt/omni.py).

    # the universal heads, from stored corpus features (experiments/omni_corpus.py)
    vt-omni train --feats /data/omni/q17 --corpus corpus.jsonl --out heads/omni_qwen3_1p7b.npz

    # onboarding a domain: unlabelled replies (no labels) -> profile; labelled replies -> adapt + guarantee
    vt-omni onboard --head heads/omni_qwen3_1p7b.npz --name acme --unlabelled traffic.jsonl \
        --labelled labelled.jsonl --fpr 0.10 --confidence 0.95 --out profiles/acme.json

    # held-out measurement and a model card
    vt-omni evaluate --head ... --profile profiles/acme.json --labelled test.jsonl --card card.json

    # the API (same keys, limits, metering as vt-verify); "profile": "acme" in a request picks the domain
    vt-omni serve --head heads/omni_qwen3_1p7b.npz --profiles profiles/ --port 8000

JSONL rows: {"reply", "question"?, "documents"?: [...], "messages"?: [...], "label"?: 0 correct / 1 wrong}
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np


def _items(path):
    from vt.omni import make_item
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows, [make_item(r["reply"], r.get("question", ""), r.get("documents"), r.get("messages")) for r in rows]


CHECKERS = {"business_record": "vt.checkers.business_record"}


def _verifier(a):
    """the head with its reader, plus the optional NLI scorer and written checkers named on the command line."""
    import importlib

    from vt.omni import JudgeScorer, NLIScorer, OmniVerifier
    checkers = [importlib.import_module(CHECKERS[c]).check for c in (a.checkers or [])]
    judges = [JudgeScorer(m) for m in (a.judges or [])]
    return OmniVerifier.load(a.head, nli=NLIScorer() if a.nli else None, checkers=checkers,
                             judges=judges).attach_reader(threads=a.threads)


def _feats(v, items, label="reading"):
    out, t = [], time.time()
    for i, it in enumerate(items):
        out.append(v.features(it))
        if (i + 1) % 25 == 0:
            print(f"  {label} {i + 1}/{len(items)} ({(time.time() - t) / (i + 1):.2f} s each)", file=sys.stderr, flush=True)
    keep = [k for k, f in enumerate(out) if f is not None]
    return [out[k] for k in keep], keep


def cmd_train(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
    from omni_corpus import load_features

    from vt.omni import OmniVerifier, train_artifact
    F = load_features(a.feats)
    rows = {}
    for line in open(a.corpus):
        r = json.loads(line)
        rows[r["id"]] = r
    art = train_artifact(F, rows, parts=tuple(a.parts), pca=a.pca, l2=a.l2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    OmniVerifier.save(art, a.out)
    print(json.dumps(art["meta"], indent=1))


def cmd_onboard(a):
    from vt.omni import OmniVerifier
    v = _verifier(a)
    _, items = _items(a.unlabelled)
    fu, _ = _feats(v, items, "unlabelled")
    prof = v.profile(fu, a.name)
    if a.labelled:
        rows, items = _items(a.labelled)
        fl, keep = _feats(v, items, "labelled")
        y = np.array([int(rows[k]["label"]) for k in keep])
        if not a.no_adapt:
            prof = v.adapt(prof, fl, y)
        prof = v.calibrate(prof, fl, y, a.fpr, a.confidence)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    OmniVerifier.save_profile(prof, a.out)
    print(json.dumps(prof["card"], indent=1))


def cmd_evaluate(a):
    from vt.correctness import auroc
    from vt.omni import OmniVerifier
    v = _verifier(a)
    prof = OmniVerifier.load_profile(a.profile) if a.profile else v.universal_profile()
    rows, items = _items(a.labelled)
    t = time.time()
    f, keep = _feats(v, items, "test")
    sec = (time.time() - t) / max(len(items), 1)
    y = np.array([int(rows[k]["label"]) for k in keep])
    s = v.score(prof, f)
    card = {"profile": prof["name"], "reader": v.meta["reader"], "n": int(len(y)), "wrong": int(y.sum()),
            "auroc": auroc(s, y), "seconds_per_reply": round(sec, 3), "calibration": prof["card"]}
    if prof.get("threshold") is not None:
        flag = s >= prof["threshold"]
        card.update(false_alarms=float(flag[y == 0].mean()) if (y == 0).any() else None,
                    caught=float(flag[y == 1].mean()) if (y == 1).any() else None,
                    precision=float(y[flag].mean()) if flag.any() else None)
    if a.card:
        json.dump(card, open(a.card, "w"), indent=1)
    print(json.dumps(card, indent=1))


class OmniService:
    """vt.verify_service's verifier interface over the omni verifier: one model lock, named profiles."""

    def __init__(self, v, profiles, info):
        self.v, self.profiles, self.info, self.lock = v, profiles, info, threading.Lock()

    def verify(self, item):
        t = time.time()
        prof = self.profiles.get(item.get("profile") or "", None)
        with self.lock:
            res = self.v.verify(item["reply"], item["question"], item["documents"], prof=prof)
        tok = self.v.reader.tok
        res["usage"] = {"tokens": len(tok(item["reply"], add_special_tokens=False).input_ids) +
                        sum(len(tok(d, add_special_tokens=False).input_ids) for d in item["documents"])}
        res["ms"] = round(1000 * (time.time() - t), 1)
        return res


def cmd_serve(a):
    from vt.omni import OmniVerifier
    from vt.verifier_pipeline import sha256
    from vt.verify_service import keys_from_env, make_server
    v = _verifier(a)
    profiles = {}
    for p in sorted(glob.glob(os.path.join(a.profiles, "*.json"))) if a.profiles else []:
        prof = OmniVerifier.load_profile(p)
        profiles[prof["name"]] = prof
    info = {"verifier": "omni", "reader": v.meta["reader"], "head": os.path.basename(a.head),
            "head_sha256": sha256(a.head)[:16], "profiles": sorted(profiles)}
    srv = make_server(OmniService(v, profiles, info), a.host, a.port, keys_from_env(), a.rate_per_minute)
    print(f"omni verification API on http://{a.host}:{a.port} (profiles: {sorted(profiles) or 'universal only'})",
          flush=True)
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vt-omni")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--feats", required=True)
    t.add_argument("--corpus", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--parts", nargs="+", default=["mean"])
    t.add_argument("--pca", type=int, default=256)
    t.add_argument("--l2", type=float, default=0.05)
    for name in ("onboard", "evaluate", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--head", required=True)
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--nli", action="store_true", help="add the NLI cross-encoders for replies with documents")
        p.add_argument("--checkers", nargs="*", choices=sorted(CHECKERS), help="written domain checkers to apply")
        p.add_argument("--judges", nargs="*", help="external judge models read in one pass, e.g. Qwen/Qwen3-14B "
                       "microsoft/phi-4 (4-bit on a GPU)")
        if name == "onboard":
            p.add_argument("--name", required=True)
            p.add_argument("--unlabelled", required=True)
            p.add_argument("--labelled")
            p.add_argument("--no-adapt", action="store_true")
            p.add_argument("--fpr", type=float, default=0.10)
            p.add_argument("--confidence", type=float, default=0.95)
            p.add_argument("--out", required=True)
        if name == "evaluate":
            p.add_argument("--profile")
            p.add_argument("--labelled", required=True)
            p.add_argument("--card")
        if name == "serve":
            p.add_argument("--profiles")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--rate-per-minute", type=int, default=120)
    a = ap.parse_args(argv)
    {"train": cmd_train, "onboard": cmd_onboard, "evaluate": cmd_evaluate, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
