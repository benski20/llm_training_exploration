"""Publishing the signed endpoint directory (verified by clients with :func:`pgw.sdk.directory.verify_directory`).

The signing key is an Ed25519 key held by the controller only (a file 0600, or better an HSM / KMS signing call
behind :class:`Signer`); its public half is pinned in SDK keys (``directory_key``) and sent in the enrollment
response over the CA-pinned channel. Pods never hold it. Every publish increments the version (persisted before the
document leaves, so a restarted controller can never reuse a version with other content), and each document expires
after ``validity_s`` (default 1 h): a controller that stops publishing makes clients fall back to the enrolled URL
within the hour, instead of steering them to pods that may be gone.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pgw.sdk.directory import DOMAIN, FORMAT, MAX_VALIDITY_S, directory_key_id

__all__ = ["DirectoryPublisher", "Signer", "generate_key", "load_signer"]


class Signer:
    """An Ed25519 signer (in-process key; subclass for an HSM / KMS)."""

    def __init__(self, private_key: Any):
        self._key = private_key

    @property
    def public_key(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def sign(self, data: bytes) -> bytes:
        return bytes(self._key.sign(data))

    def __repr__(self) -> str:
        return f"Signer(key_id={directory_key_id(self.public_key)})"


def generate_key(path: Path) -> Signer:
    """A new signing key written to ``path`` (PKCS#8 PEM, 0600; refuses to overwrite)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    k = Ed25519PrivateKey.generate()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()))
    return Signer(k)


def load_signer(path: Path) -> Signer:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    p = Path(path)
    if p.stat().st_mode & 0o077:
        raise PermissionError(f"{p} must not be readable by group or others (chmod 600)")
    k = serialization.load_pem_private_key(p.read_bytes(), password=None)
    if not isinstance(k, Ed25519PrivateKey):
        raise ValueError("the directory signing key must be Ed25519")
    return Signer(k)


class DirectoryPublisher:
    """Builds, signs and versions directories. ``state_file`` keeps the last version (written before signing)."""

    def __init__(self, signer: Signer, state_file: Path | None = None, *, validity_s: int = 3600,
                 clock: Callable[[], float] = time.time):
        if not 0 < validity_s <= MAX_VALIDITY_S:
            raise ValueError(f"validity_s must be in (0, {MAX_VALIDITY_S}]")
        self.signer, self.state_file, self.validity_s, self.clock = signer, state_file, int(validity_s), clock
        self._lock = threading.Lock()
        self._version = 0
        if state_file is not None and Path(state_file).exists():
            self._version = int(json.loads(Path(state_file).read_text())["version"])

    @property
    def version(self) -> int:
        return self._version

    def _bump(self) -> int:
        self._version += 1
        if self.state_file is not None:
            p = Path(self.state_file)
            tmp = p.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"version": self._version}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        return self._version

    def publish(self, endpoints: Iterable[Mapping[str, Any]]) -> bytes:
        """The signed wire form (JSON bytes) of a new version listing ``endpoints`` (each
        ``{region, url, models, priority?, lane_port?, dc?, state?}``)."""
        eps = []
        for e in endpoints:
            item = {k: e[k] for k in ("region", "url", "models", "priority", "lane_port", "dc", "state") if k in e}
            item.setdefault("models", [])
            eps.append(item)
        with self._lock:
            now = int(self.clock())
            payload = {"format": FORMAT, "v": 1, "version": self._bump(), "issued_at": now,
                       "expires_at": now + self.validity_s, "key_id": directory_key_id(self.signer.public_key),
                       "endpoints": eps}
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sig = self.signer.sign(DOMAIN + raw)
        return json.dumps({"payload": _b64(raw), "sig": _b64(sig)}).encode("ascii")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")
