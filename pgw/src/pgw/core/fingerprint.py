"""Model pack fingerprints.

The fingerprint binds every party (gateway, mask factory, compute service) to the exact same
public int8 weights, scales and architecture. Masks computed for one fingerprint are useless (and
must be refused) for another.

Definition (normative, SPEC.md section 8.3)::

    fingerprint = hex(SHA-256(b"pgw-modelpack-v1\\x00" || canonical_json(semantic_manifest)))

where ``semantic_manifest`` is the manifest dict without the keys in :data:`NON_SEMANTIC_KEYS`, and
``canonical_json`` is UTF-8 JSON with sorted keys, no whitespace, no NaN/Inf. The manifest lists the
SHA-256 of every tensor file, so the fingerprint transitively covers all weight bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, FingerprintMismatchError

__all__ = [
    "FINGERPRINT_DOMAIN",
    "NON_SEMANTIC_KEYS",
    "canonical_json",
    "compute_fingerprint",
    "require_same_fingerprint",
    "sha256_file",
    "short_fingerprint",
    "validate_fingerprint",
]

FINGERPRINT_DOMAIN = b"pgw-modelpack-v1\x00"
#: Manifest keys that do not affect computation and are excluded from the fingerprint.
NON_SEMANTIC_KEYS: frozenset[str] = frozenset({"fingerprint", "created_at", "tool", "notes"})
_FP_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, compact separators, UTF-8, finite floats only)."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except ValueError as exc:
        raise ConfigError(f"manifest is not canonical-JSON serialisable: {exc}") from exc


def compute_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Fingerprint (64 lowercase hex chars) of a manifest dict."""
    semantic = {k: v for k, v in manifest.items() if k not in NON_SEMANTIC_KEYS}
    return hashlib.sha256(FINGERPRINT_DOMAIN + canonical_json(semantic)).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def validate_fingerprint(fp: str) -> str:
    """Return ``fp`` if it is 64 lowercase hex chars, else raise :class:`ConfigError`."""
    if not isinstance(fp, str) or not _FP_RE.match(fp):
        raise ConfigError(f"invalid model fingerprint {fp!r}")
    return fp


def require_same_fingerprint(expected: str, actual: str, *, context: str = "") -> None:
    """Constant-time comparison; raise :class:`FingerprintMismatchError` on mismatch."""
    if not hmac.compare_digest(expected.encode(), actual.encode()):
        raise FingerprintMismatchError(
            f"model fingerprint mismatch{(' (' + context + ')') if context else ''}: "
            f"expected {short_fingerprint(expected)}, got {short_fingerprint(actual)}",
            details={"expected": expected, "actual": actual},
        )


def short_fingerprint(fp: str) -> str:
    """Display form: first 12 hex chars."""
    return fp[:12]
