"""Command-line interface.

  vt prove   --spec dyck --length 16 --out runs/dyck16      # CEGIS until proved
  vt verify  runs/dyck16/model.npz --spec dyck              # global proof only
  vt check   runs/dyck16/model.npz runs/dyck16/certificate.json [--exhaustive]
  vt predict runs/dyck16/model.npz "(()())()(())((()))"
"""
import argparse
import json
import os
import sys

import numpy as np

from .artifact import load_artifact, make_certificate, save_artifact, sha256_file, write_json
from .global_verify import SPECS, exhaustive_check, verify_auto, verify_global
from .verify import np_forward

TOKENS = {"dyck": "()", "sorted": "abcd"}


def _encode(spec_name, text):
    alphabet = TOKENS[spec_name]
    return np.array([alphabet.index(ch) for ch in text])


def cmd_prove(a):
    import torch
    from .cegis import CegisConfig, cegis
    torch.set_num_threads(a.threads)
    spec = SPECS[a.spec]()
    V = len(TOKENS[a.spec])
    tuned = {"lr_decay": (0.92, 0.7), "epochs_first": (200, 60), "epochs_round": (30, 15)}
    for k, (tied, untied) in tuned.items():
        if getattr(a, k) is None:
            setattr(a, k, tied if (a.shared or a.uniform) else untied)
    if a.lr is None:
        a.lr = 3e-2 if a.uniform else 3e-3
    a.shared = a.shared or a.uniform
    cfg = CegisConfig(L=a.length, d_model=a.d_model, d_hidden=a.d_hidden, n_blocks=a.n_blocks,
                      ternary=not a.float, causal=a.causal, rounds=a.rounds,
                      ternary_from_round=a.qat_from_round, verify_workers=a.workers, bits=a.bits,
                      shared=a.shared, lr=a.lr, lr_decay=a.lr_decay,
                      epochs_first=a.epochs_first, epochs_round=a.epochs_round,
                      lengths=tuple(int(v) for v in a.also_lengths.split(",") if v), tau=a.tau,
                      n_train=a.n_train, enumerate_up_to=a.enumerate_up_to,
                      uniform=a.uniform, verifier=a.verifier, window=a.window,
                      verify_time=a.verify_time, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    progress = open(os.path.join(a.out, "progress.jsonl"), "a")

    def log(e):
        print(json.dumps(e), flush=True)
        progress.write(json.dumps(e) + "\n"); progress.flush()

    model, res = cegis(spec, cfg, V=V, log=log)
    write_json(res, os.path.join(a.out, "history.json"))
    lengths = res["lengths"]
    arts = {L: os.path.join(a.out, "model.npz" if L == a.length else f"model_L{L}.npz")
            for L in lengths}
    for L, art in arts.items():
        save_artifact(model.export(L), art)
    if res["status"] != "proved":
        print(f"not proved after {res['rounds']} rounds; artifact(s) saved without certificate")
        return 1
    # re-verify each saved artifact itself, so certificates bind to the files on disk
    ok = True
    for L, art in arts.items():
        P = load_artifact(art)
        if V ** L <= cfg.enumerate_up_to and not (a.uniform and a.verifier != "milp"):
            n_err, min_margin, _ = exhaustive_check(P, spec, a.tau)
            out = {"status": "proved" if n_err == 0 else "counterexample", "queries": [],
                   "method": "enumeration", "min_margin": min_margin, "time": 0.0}
        else:
            out = verify_auto(P, spec, method=a.verifier, tau=a.tau, time_limit=a.verify_time,
                              workers=a.workers)
        cert = make_certificate(art, a.spec, out, a.tau, {"length": L, "method": out["method"],
                                                          "training": res["config"],
                                                          "cegis_rounds": res["rounds"]})
        name = "certificate.json" if L == a.length else f"certificate_L{L}.json"
        write_json(cert, os.path.join(a.out, name))
        ok &= out["status"] == "proved"
        print(f"L={L}: {out['status']} ({out['method']}) {art} sha256={cert['artifact_sha256'][:16]}...")
    print(f"total {res['elapsed']:.0f}s")
    return 0 if ok else 1


def cmd_verify(a):
    P = load_artifact(a.artifact)
    out = verify_auto(P, SPECS[a.spec](), method=a.verifier, tau=a.tau,
                      time_limit=a.verify_time, workers=a.workers)
    print(json.dumps({k: out.get(k) for k in ("status", "method", "time", "counterexamples")}))
    if a.certificate:
        write_json(make_certificate(a.artifact, a.spec, out, a.tau,
                                    {"length": P["L"], "method": out["method"]}), a.certificate)
    return 0 if out["status"] == "proved" else 1


def cmd_check(a):
    with open(a.certificate) as f:
        cert = json.load(f)
    digest = sha256_file(a.artifact)
    ok = True
    if digest != cert["artifact_sha256"]:
        print("FAIL: artifact hash does not match certificate"); return 1
    if cert["status"] != "proved":
        print(f"FAIL: certificate status is {cert['status']}"); return 1
    P = load_artifact(a.artifact)
    spec = SPECS[cert["spec"]]()
    method = a.verifier if a.verifier != "auto" else (
        "exact" if cert.get("method") == "exact-dp" else "auto")
    out = verify_auto(P, spec, method=method, tau=cert["tau"], time_limit=a.verify_time,
                      workers=a.workers)
    print(f"re-verification: {out['status']} via {out['method']} ({out['time']:.1f}s)")
    ok &= out["status"] == "proved"
    if a.exhaustive:
        n_err, min_margin, first = exhaustive_check(P, spec, cert["tau"])
        print(f"exhaustive: {n_err} errors over {P['V'] ** P['L']} inputs, "
              f"min margin {min_margin:.4f}" + (f", first error {first}" if first else ""))
        ok &= n_err == 0
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


def cmd_predict(a):
    P = load_artifact(a.artifact)
    spec_name = a.spec
    for text in a.inputs:
        seq = _encode(spec_name, text)
        if len(seq) != P["L"]:
            print(f"{text}: length {len(seq)} != model length {P['L']}", file=sys.stderr)
            return 2
        logits = np_forward(P, np.eye(P["V"])[seq])
        print(f"{text}\t{int(np.argmax(logits))}\t(spec: {SPECS[spec_name]().label(seq)})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove", help="train with counterexample guidance until proved")
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--also-lengths", default="",
                   help="comma-separated shorter lengths to train and prove too (needs --shared)")
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=4)
    p.add_argument("--d-hidden", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=1)
    p.add_argument("--float", action="store_true", help="float weights instead of ternary")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--shared", action="store_true",
                   help="weight-tied across positions: Toeplitz causal mixing + pooled readout")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--enumerate-up-to", type=int, default=1 << 16,
                   help="lengths with V^L at most this are decided by enumeration, not MILP")
    p.add_argument("--lr", type=float, default=None, help="default 3e-2 with --uniform, else 3e-3")
    # schedule defaults depend on --shared (tuned settings for each family)
    p.add_argument("--lr-decay", type=float, default=None,
                   help="per-round lr multiplier (default 0.92 with --shared, else 0.7)")
    p.add_argument("--epochs-first", type=int, default=None, help="default 200 shared / 60")
    p.add_argument("--epochs-round", type=int, default=None, help="default 30 shared / 15")
    p.add_argument("--uniform", action="store_true",
                   help="uniform causal mixing: exactly verifiable at any length (implies --shared)")
    p.add_argument("--window", type=int, default=1,
                   help="with --uniform: free kernel taps for the last WINDOW offsets")
    p.add_argument("--bits", type=int, default=0, help="k-bit uniform weights (with --float)")
    p.add_argument("--qat-from-round", type=int, default=0,
                   help="train float weights for this many rounds before quantising")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--verify-time", type=float, default=300.0)
    p.add_argument("--tau", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_prove)
    p = sub.add_parser("verify", help="prove global correctness of an artifact")
    p.add_argument("artifact"); p.add_argument("--spec", default="dyck", choices=SPECS)
    p.add_argument("--tau", type=float, default=1e-4)
    p.add_argument("--verify-time", type=float, default=300.0)
    p.add_argument("--certificate", help="write a certificate JSON here")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("check", help="check a certificate against an artifact")
    p.add_argument("artifact"); p.add_argument("certificate")
    p.add_argument("--exhaustive", action="store_true", help="also enumerate all inputs (small L)")
    p.add_argument("--verify-time", type=float, default=300.0)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("predict", help="run the deployed artifact")
    p.add_argument("artifact"); p.add_argument("inputs", nargs="+")
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.set_defaults(fn=cmd_predict)
    for name, sp_ in sub.choices.items():
        if name != "predict":
            sp_.add_argument("--verifier", default="auto", choices=["auto", "exact", "milp"],
                             help="exact rational DP when the structure allows (auto), or MILP")
            sp_.add_argument("--workers", type=int, default=os.cpu_count(),
                             help="MILP cases solved in parallel")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
