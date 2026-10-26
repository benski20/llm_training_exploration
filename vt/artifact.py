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
import os
import platform
import time
import zipfile

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


MAX_ARTIFACT_BYTES = 256 << 20   # on-disk and uncompressed size caps
MAX_DIM = 1 << 16


class ArtifactError(ValueError):
    """The artifact file is malformed, oversized or inconsistent."""


def read_artifact(path, max_bytes=MAX_ARTIFACT_BYTES):
    """Read an artifact ONCE. Returns (P, sha256 hex of exactly those bytes), so
    the hash and the loaded weights can never come from different file contents."""
    with open(path, "rb") as f:
        buf = f.read(max_bytes + 1)
    if len(buf) > max_bytes:
        raise ArtifactError(f"artifact larger than {max_bytes} bytes")
    return load_artifact_bytes(buf), hashlib.sha256(buf).hexdigest()


def load_artifact_bytes(buf, max_uncompressed=MAX_ARTIFACT_BYTES):
    try:
        zf = zipfile.ZipFile(io.BytesIO(buf))
        infos = zf.infolist()
    except zipfile.BadZipFile as e:
        raise ArtifactError(f"not an npz artifact: {e}") from None
    if any(i.compress_type != zipfile.ZIP_STORED for i in infos):
        raise ArtifactError("compressed members are not allowed (zip-bomb guard)")
    if sum(i.file_size for i in infos) > max_uncompressed:
        raise ArtifactError("artifact members too large")
    z = np.load(io.BytesIO(buf), allow_pickle=False)
    try:
        meta = [int(v) for v in z["meta"].tolist()]
        if len(meta) != 6 or any(v < 1 or v > MAX_DIM for v in meta):
            raise ArtifactError(f"bad meta {meta}")
        P = dict(zip(SCALARS, meta[:5]))
        n = meta[5]
        for k in ("W_in", "b_in", "W_out", "b_out"):
            P[k] = z[k].astype(np.float64)
        for k in LIST_KEYS:
            P[k] = [z[f"{k}.{i}"].astype(np.float64) for i in range(n)]
    except KeyError as e:
        raise ArtifactError(f"missing array {e}") from None
    L, V, d, h, C = (P[k] for k in SCALARS)
    want = {"W_in": (d, V), "b_in": (d,), "W_out": (C, L * d), "b_out": (C,)}
    for k, shp in want.items():
        if P[k].shape != shp:
            raise ArtifactError(f"{k} has shape {P[k].shape}, expected {shp}")
    for i in range(n):
        for k, shp in {"mix": (L, L), "W1": (h, d), "b1": (h,), "W2": (d, h), "b2": (d,)}.items():
            if P[k][i].shape != shp:
                raise ArtifactError(f"{k}.{i} has shape {P[k][i].shape}, expected {shp}")
    arrays = [P[k] for k in ("W_in", "b_in", "W_out", "b_out")] + [a for k in LIST_KEYS for a in P[k]]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ArtifactError("non-finite weights")
    return P


def load_artifact(path):
    return read_artifact(path)[0]


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


# --- signatures: optional fast path; re-verification remains the default -----

def _canonical(cert):
    body = {k: v for k, v in cert.items() if k not in ("signature", "signer")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def generate_keypair(private_path, public_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    with open(private_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    os.chmod(private_path, 0o600)
    with open(public_path, "wb") as f:
        f.write(key.public_key().public_bytes(serialization.Encoding.PEM,
                                              serialization.PublicFormat.SubjectPublicKeyInfo))


def sign_certificate(cert, private_path):
    import base64
    from cryptography.hazmat.primitives import serialization
    with open(private_path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    out = dict(cert)
    out["signer"] = hashlib.sha256(pub).hexdigest()
    out["signature"] = base64.b64encode(key.sign(_canonical(cert))).decode()
    return out


def signature_valid(cert, public_path):
    import base64
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    if "signature" not in cert:
        return False
    with open(public_path, "rb") as f:
        pub = serialization.load_pem_public_key(f.read())
    try:
        pub.verify(base64.b64decode(cert["signature"]), _canonical(cert))
        return True
    except (InvalidSignature, ValueError):
        return False
