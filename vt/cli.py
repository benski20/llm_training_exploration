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
from .global_verify import SPECS, exhaustive_check, verify_global
from .verify import np_forward

TOKENS = {"dyck": "()"}


def _encode(spec_name, text):
    alphabet = TOKENS[spec_name]
    return np.array([alphabet.index(ch) for ch in text])


def cmd_prove(a):
    import torch
    from .cegis import CegisConfig, cegis
    torch.set_num_threads(a.threads)
    spec = SPECS[a.spec]()
    cfg = CegisConfig(L=a.length, d_model=a.d_model, d_hidden=a.d_hidden, n_blocks=a.n_blocks,
                      ternary=not a.float, causal=a.causal, rounds=a.rounds,
                      ternary_from_round=a.qat_from_round,
                      verify_time=a.verify_time, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    model, res = cegis(spec, cfg, log=lambda e: print(json.dumps(e), flush=True))
    art = os.path.join(a.out, "model.npz")
    save_artifact(model.export(), art)
    write_json(res, os.path.join(a.out, "history.json"))
    if res["status"] != "proved":
        print(f"not proved after {res['rounds']} rounds; artifact saved without certificate")
        return 1
    # re-verify the saved artifact itself, so the certificate binds to the file on disk
    P = load_artifact(art)
    out = verify_global(P, spec, tau=a.tau, time_limit=a.verify_time)
    cert = make_certificate(art, a.spec, out, a.tau, {"training": res["config"],
                                                      "cegis_rounds": res["rounds"]})
    write_json(cert, os.path.join(a.out, "certificate.json"))
    print(f"{out['status']}: {art} sha256={cert['artifact_sha256'][:16]}... "
          f"({out['time']:.1f}s verify, {res['elapsed']:.0f}s total)")
    return 0 if out["status"] == "proved" else 1


def cmd_verify(a):
    P = load_artifact(a.artifact)
    out = verify_global(P, SPECS[a.spec](), tau=a.tau, time_limit=a.verify_time)
    print(json.dumps({k: out[k] for k in ("status", "time", "unstable", "counterexamples")}))
    if a.certificate:
        write_json(make_certificate(a.artifact, a.spec, out, a.tau), a.certificate)
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
    out = verify_global(P, spec, tau=cert["tau"], time_limit=a.verify_time)
    print(f"re-verification: {out['status']} ({out['time']:.1f}s)")
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
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=4)
    p.add_argument("--d-hidden", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=1)
    p.add_argument("--float", action="store_true", help="float weights instead of ternary")
    p.add_argument("--causal", action="store_true")
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
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
