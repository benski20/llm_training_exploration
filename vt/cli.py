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


def _spec(a):
    """(spec, name, alphabet, extra certificate fields) from --spec / --spec-file."""
    if getattr(a, "spec_file", None):
        from .counterspec import spec_from_def
        with open(a.spec_file) as f:
            d = json.load(f)
        return spec_from_def(d), "dfa:" + d.get("name", "dfa"), d["alphabet"], {"spec_dfa": d}
    return SPECS[a.spec](), a.spec, TOKENS[a.spec], {}


def _spec_from_cert(cert):
    if "spec_dfa" in cert:
        from .counterspec import spec_from_def
        return spec_from_def(cert["spec_dfa"]), cert["spec_dfa"]["alphabet"]
    return SPECS[cert["spec"]](), TOKENS[cert["spec"]]


def _encode(alphabet, text):
    return np.array([alphabet.index(ch) for ch in text])


def cmd_prove(a):
    import torch
    from .cegis import CegisConfig, cegis
    torch.set_num_threads(a.threads)
    spec, spec_name, alphabet, spec_extra = _spec(a)
    V = len(alphabet)
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
                      init=a.init,
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
        cert = make_certificate(art, spec_name, out, a.tau, {"length": L, "method": out["method"],
                                                             "training": res["config"],
                                                             "cegis_rounds": res["rounds"],
                                                             **spec_extra})
        name = "certificate.json" if L == a.length else f"certificate_L{L}.json"
        write_json(cert, os.path.join(a.out, name))
        ok &= out["status"] == "proved"
        print(f"L={L}: {out['status']} ({out['method']}) {art} sha256={cert['artifact_sha256'][:16]}...")
    print(f"total {res['elapsed']:.0f}s")
    return 0 if ok else 1


def cmd_verify(a):
    P = load_artifact(a.artifact)
    spec, spec_name, _, spec_extra = _spec(a)
    out = verify_auto(P, spec, method=a.verifier, tau=a.tau,
                      time_limit=a.verify_time, workers=a.workers)
    print(json.dumps({k: out.get(k) for k in ("status", "method", "time", "counterexamples")}))
    if a.certificate:
        write_json(make_certificate(a.artifact, spec_name, out, a.tau,
                                    {"length": P["L"], "method": out["method"], **spec_extra}),
                   a.certificate)
    return 0 if out["status"] == "proved" else 1


def _pinned(a):
    """The operator's spec, if they chose one explicitly (--spec or --spec-file)."""
    if getattr(a, "spec_file", None) or getattr(a, "spec", None):
        return _spec(a)[:3]
    return None


def cmd_check(a):
    from .artifact import ArtifactError, read_artifact
    from .serve import GateError, resolve_cert_spec
    with open(a.certificate) as f:
        cert = json.load(f)
    try:
        P, digest = read_artifact(a.artifact)          # hash and load the same bytes
    except ArtifactError as e:
        print(f"FAIL: {e}"); return 1
    if digest != cert.get("artifact_sha256"):
        print("FAIL: artifact hash does not match certificate"); return 1
    if cert.get("status") != "proved":
        print(f"FAIL: certificate status is {cert.get('status')}"); return 1
    try:
        spec, name, _ = resolve_cert_spec(cert, _pinned(a))
    except GateError as e:
        print(f"FAIL: {e}"); return 1
    ok = True
    method = a.verifier if a.verifier != "auto" else (
        "exact" if cert.get("method") == "exact-dp" else "auto")
    out = verify_auto(P, spec, method=method, tau=float(cert["tau"]), time_limit=a.verify_time,
                      workers=a.workers)
    print(f"re-verification against {name}: {out['status']} via {out['method']} ({out['time']:.1f}s)")
    ok &= out["status"] == "proved"
    if a.exhaustive:
        n_err, min_margin, first = exhaustive_check(P, spec, float(cert["tau"]))
        print(f"exhaustive: {n_err} errors over {P['V'] ** P['L']} inputs, "
              f"min margin {min_margin:.4f}" + (f", first error {first}" if first else ""))
        ok &= n_err == 0
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


def cmd_predict(a):
    P = load_artifact(a.artifact)
    spec, _, alphabet, _ = _spec(a)
    for text in a.inputs:
        if any(ch not in alphabet for ch in text):
            print(f"{text}: symbols must be from {alphabet!r}", file=sys.stderr)
            return 2
        seq = _encode(alphabet, text)
        if len(seq) != P["L"]:
            print(f"{text}: length {len(seq)} != model length {P['L']}", file=sys.stderr)
            return 2
        logits = np_forward(P, np.eye(P["V"])[seq])
        print(f"{text}\t{int(np.argmax(logits))}\t(spec: {spec.label(seq)})")
    return 0


def cmd_serve(a):
    from .serve import GateError, load_gated, make_server
    pinned = _pinned(a)
    try:
        P, digest, cert, gate = load_gated(a.artifact, a.certificate, a.allow_unverified, pinned,
                                           a.trusted_key, a.verify_time, a.workers)
    except GateError as e:
        print(f"refusing to serve: {e}", file=sys.stderr)
        return 1
    alphabet = gate.get("alphabet") or (pinned[2] if pinned else TOKENS["dyck"])
    srv = make_server(P, digest, cert, alphabet, a.host, a.port, gate=gate,
                      max_concurrency=a.max_concurrency, deadline=a.deadline)
    print(f"serving {a.artifact} (sha256 {digest[:16]}..., {gate['how']}) "
          f"on http://{a.host}:{srv.server_port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_keygen(a):
    from .artifact import generate_keypair
    generate_keypair(a.private, a.public)
    print(f"wrote {a.private} (keep secret) and {a.public}")
    return 0


def cmd_sign(a):
    """Sign a certificate after re-verifying it, so a signature always means
    'the signer's own verifier proved this artifact'."""
    from .artifact import sign_certificate
    if cmd_check(argparse.Namespace(artifact=a.artifact, certificate=a.certificate, spec=a.spec,
                                    spec_file=a.spec_file, verifier="auto", exhaustive=False,
                                    verify_time=a.verify_time, workers=a.workers)) != 0:
        print("not signing: re-verification failed", file=sys.stderr)
        return 1
    with open(a.certificate) as f:
        cert = json.load(f)
    write_json(sign_certificate(cert, a.key), a.out or a.certificate)
    print(f"signed {a.out or a.certificate}")
    return 0


def _gen_cert(art, spec_name, results, extra):
    from .artifact import sha256_file
    ok = all(r["status"] == "proved" for r in results.values())
    return {"format": "vt-certificate/1", "artifact_sha256": sha256_file(art), "spec": spec_name,
            "method": "exact-gen-dp", "status": "proved" if ok else "failed",
            "claim": make_certificate.__globals__["CLAIMS"]["exact-gen-dp"],
            "lengths": {str(L): r for L, r in results.items()}, **extra}


def cmd_gen_train(a):
    import torch
    from .gen_train import GenConfig, heldout_nll, train_generator
    from .gen_verify import read_generator, save_generator, verify_generator
    torch.set_num_threads(a.threads)
    spec, name, alphabet, extra = _spec(a)
    cfg = GenConfig(L=a.length, lengths=tuple(int(v) for v in a.also_lengths.split(",") if v),
                    d_model=a.d_model, d_hidden=a.d_hidden, window=a.window, steps=a.steps,
                    lr=a.lr, lam=a.lam, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    model, rep = train_generator(spec, cfg, len(alphabet),
                                 log=lambda e: print(json.dumps(e), flush=True))
    art = os.path.join(a.out, "generator.npz")
    save_generator(model.export(), art)
    G, _ = read_generator(art)                     # certify the bytes on disk
    results = {L: verify_generator(G, spec, L, exact_mass=False) for L in rep["lengths"]}
    cert = _gen_cert(art, name, results, {**extra, "training": rep["config"],
                     "heldout_nll_per_token": {str(L): heldout_nll(model, spec, L)
                                               for L in rep["lengths"]}})
    write_json(cert, os.path.join(a.out, "certificate.json"))
    for L, r in results.items():
        print(f"L={L}: {r['status']} (sound={r['sound']} complete={r['complete']} live={r['live']})")
    return 0 if cert["status"] == "proved" else 1


def cmd_gen_verify(a):
    from .gen_verify import read_generator, verify_generator
    spec, name, _, extra = _spec(a)
    G, _ = read_generator(a.artifact)
    results = {int(L): verify_generator(G, spec, int(L), exact_mass=a.exact_mass)
               for L in a.lengths.split(",")}
    for L, r in results.items():
        print(json.dumps({"L": L, **{k: r[k] for k in ("status", "sound", "complete", "live",
                                                      "p_violation", "time")}}))
    if a.certificate:
        write_json(_gen_cert(a.artifact, name, results, extra), a.certificate)
    return 0 if all(r["status"] == "proved" for r in results.values()) else 1


def cmd_gen_sample(a):
    """Sample only after re-proving, at the requested length, that the
    generator cannot emit a string outside the pinned spec."""
    import torch
    from .gen_verify import load_generator_model, read_generator, verify_generator
    spec, name, alphabet, _ = _spec(a)
    G, digest = read_generator(a.artifact)
    r = verify_generator(G, spec, a.length, exact_mass=False)
    if not (r["sound"] and r["live"]):
        print(f"refusing to sample: not proved sound and live for {name} at L={a.length}",
              file=sys.stderr)
        return 1
    m = load_generator_model(G)
    x = m.sample(a.n, a.length, eps=0.0, generator=torch.Generator().manual_seed(a.seed))
    for row in x.tolist():
        if spec.label(row) != 1:                     # cannot happen given the proof
            print("internal error: proved generator emitted an invalid string", file=sys.stderr)
            return 2
        print("".join(alphabet[v] for v in row))
    return 0


def cmd_respond(a):
    from .pcr import PlanError, respond
    try:
        with open(a.plan) as f:
            plan = json.load(f)
        with open(a.kb) as f:
            kb = json.load(f)
        with open(a.policy) as f:
            policy = json.load(f)
        out = respond(plan, kb, policy, seed=a.seed)
    except (PlanError, json.JSONDecodeError) as e:
        print(f"rejected plan: {e}", file=sys.stderr)
        return 1
    print(out["text"])
    print(json.dumps(out["guarantees"], indent=1, default=str))
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
    p.add_argument("--init", default="", help="warm-start a tied model from an artifact (.npz)")
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
    p = sub.add_parser("check", help="re-verify an artifact against its certificate")
    p.add_argument("artifact"); p.add_argument("certificate")
    p.add_argument("--spec", default=None, choices=SPECS, help="pin the expected spec")
    p.add_argument("--exhaustive", action="store_true", help="also enumerate all inputs (small L)")
    p.add_argument("--verify-time", type=float, default=300.0)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("serve", help="HTTP inference server; re-proves the artifact before serving")
    p.add_argument("artifact"); p.add_argument("certificate", nargs="?")
    p.add_argument("--spec", default=None, choices=SPECS, help="pin the expected spec")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--trusted-key", help="Ed25519 public key: accept its signature instead of re-proving")
    p.add_argument("--verify-time", type=float, default=600.0)
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--deadline", type=float, default=30.0, help="max seconds per connection")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("gen-train", help="train a count-state generator and certify its support")
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--also-lengths", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=8); p.add_argument("--d-hidden", type=int, default=32)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam", type=float, default=10.0, help="support-hinge weight")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--threads", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_gen_train)
    p = sub.add_parser("gen-verify", help="exactly verify a generator's support at given lengths")
    p.add_argument("artifact"); p.add_argument("--lengths", required=True)
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.add_argument("--exact-mass", action="store_true", help="also compute the exact P(violation)")
    p.add_argument("--certificate")
    p.set_defaults(fn=cmd_gen_verify)
    p = sub.add_parser("gen-sample", help="sample from a generator, after re-proving it at that length")
    p.add_argument("artifact"); p.add_argument("--length", type=int, required=True)
    p.add_argument("-n", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.set_defaults(fn=cmd_gen_sample)
    p = sub.add_parser("respond", help="plan -> check -> realize, with a guarantee report")
    p.add_argument("plan"); p.add_argument("--kb", required=True); p.add_argument("--policy", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_respond)
    p = sub.add_parser("keygen", help="create an Ed25519 key pair for signing certificates")
    p.add_argument("private"); p.add_argument("public")
    p.set_defaults(fn=cmd_keygen)
    p = sub.add_parser("sign", help="re-verify, then sign a certificate")
    p.add_argument("artifact"); p.add_argument("certificate"); p.add_argument("--key", required=True)
    p.add_argument("--out"); p.add_argument("--spec", default=None, choices=SPECS)
    p.add_argument("--verify-time", type=float, default=600.0)
    p.set_defaults(fn=cmd_sign)
    p = sub.add_parser("predict", help="run the deployed artifact")
    p.add_argument("artifact"); p.add_argument("inputs", nargs="+")
    p.add_argument("--spec", default="dyck", choices=SPECS)
    p.set_defaults(fn=cmd_predict)
    for name, sp_ in sub.choices.items():
        if name not in ("keygen", "respond"):
            sp_.add_argument("--spec-file", help="JSON DFA spec (any regular language); overrides --spec")
        if name not in ("predict", "keygen", "gen-train", "gen-verify", "gen-sample", "respond"):
            if name == "serve":
                sp_.add_argument("--workers", type=int, default=os.cpu_count())
                continue
            sp_.add_argument("--verifier", default="auto", choices=["auto", "exact", "milp"],
                             help="exact rational DP when the structure allows (auto), or MILP")
            sp_.add_argument("--workers", type=int, default=os.cpu_count(),
                             help="MILP cases solved in parallel")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
