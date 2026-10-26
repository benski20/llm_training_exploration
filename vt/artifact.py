"""Deployable model artifacts and proof certificates.

The artifact is the exported float64 weight dict (`.npz`). The certificate is
a JSON file that binds a verification result to the artifact's SHA-256, so
"this file was proved correct" is checkable: `vt check` re-hashes the
artifact and re-runs the verifier (and, for short sequences, exhaustive
enumeration as an independent second check). Inference (`vt predict`) runs
the same numpy forward pass the verifier encoded, so the proved artifact is
exactly what is deployed.
"""
import hashlib
import io
import json
import platform
import time

import numpy as np
import scipy

LIST_KEYS = ("mix", "W1", "b1", "W2", "b2")
SCALARS = ("L", "V", "d", "h", "C")


VERIFIER_DESCRIPTIONS = {
    "milp": "symbolic-bounds + exact big-M MILP (HiGHS, floating point)",
    "exact-dp": "dynamic program over (position, token counts, spec state) in exact rational arithmetic",
    "enumeration": "exhaustive enumeration of all inputs (float64 forward pass)",
}


def save_artifact(P, path):
    arrays = {k: np.asarray(P[k]) for k in ("W_in", "b_in", "W_out", "b_out")}
    for k in LIST_KEYS:
        for i, a in enumerate(P[k]):
            arrays[f"{k}.{i}"] = np.asarray(a)
    arrays["meta"] = np.array([P[k] for k in SCALARS] + [len(P["mix"])], dtype=np.int64)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return sha256_file(path)


def load_artifact(path):
    z = np.load(path)
    meta = z["meta"].tolist()
    P = dict(zip(SCALARS, meta[:5]))
    n = meta[5]
    for k in ("W_in", "b_in", "W_out", "b_out"):
        P[k] = z[k].astype(np.float64)
    for k in LIST_KEYS:
        P[k] = [z[f"{k}.{i}"].astype(np.float64) for i in range(n)]
    return P


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_certificate(artifact_path, spec_name, result, tau, extra=None):
    return {
        "format": "vt-certificate/1",
        "artifact_sha256": sha256_file(artifact_path),
        "spec": spec_name,
        "claim": ("for every input sequence in V^L, argmax of the model's logits equals "
                  "the spec label, with logit margin > tau"),
        "tau": tau,
        "status": result["status"],
        "queries": result.get("queries", []),
        "verifier": {"method": VERIFIER_DESCRIPTIONS.get(result.get("method", "milp"), result.get("method")),
                     "scipy": scipy.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **({"min_margin": result["min_margin"]} if "min_margin" in result else {}),
        **(extra or {}),
    }


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
