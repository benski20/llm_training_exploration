"""Trained draft heads for ``speculation="draft_head"``: shipped with the library, pinned by SHA-256.

A draft head is a few-MB public artefact per model (``benchmarks/feature_head_distill.py``: distilled from the
model's own answers to public chat prompts; int8 weights, ``FeatureHeadDrafter`` format v2). It runs on the
client's CPU only, next to the gateway's plaintext state; it is never sent to the provider, and its proposals
only decide which rows a verify pass carries (the answer is the target model's, bit for bit).

Resolution of ``speculation="draft_head"`` without ``draft_head_path`` (:func:`resolve_draft_head`):

1. the file shipped in the package (``pgw/sdk/draft_heads/<file>``), checked against its pinned SHA-256 and
   size: a mismatch refuses (a modified installation);
2. else a copy in the client cache (``<cache>/models/draft-heads/<file>``), checked the same way (a bad copy is
   deleted and fetched again);
3. else, when ``PGW_DRAFT_HEAD_URL`` names a mirror (``<url>/<file>``), the file is downloaded, checked, and
   only then moved into the cache.

The pin also names the target checkpoint (repository and commit) the head was distilled for; the loader checks
the hidden size against the served model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pgw.core.errors import ConfigError, IntegrityError

log = logging.getLogger(__name__)

__all__ = ["DraftHeadPin", "PINNED_DRAFT_HEADS", "PACKAGE_DIR", "resolve_draft_head", "verify_draft_head"]

#: Where the shipped head files live (package data).
PACKAGE_DIR = Path(__file__).resolve().parent / "draft_heads"


@dataclass(frozen=True)
class DraftHeadPin:
    """A shipped draft head.

    Attributes:
        file: File name (in :data:`PACKAGE_DIR`, the cache, or under the mirror URL).
        sha256: SHA-256 of the file (hex).
        size: Size in bytes.
        hf_repo: Target checkpoint the head was distilled for.
        revision: Commit of that checkpoint (the SDK's model pin).
    """

    file: str
    sha256: str
    size: int
    hf_repo: str
    revision: str


#: The release's draft heads (format ``pgw-feature-head-v2``), from the Kaggle distillation runs of 2026-11-26
#: (``benchmarks/results/feature_head/``; acceptance and tokens per pass in docs/sdk/configuration.md).
PINNED_DRAFT_HEADS: dict[str, DraftHeadPin] = {
    # d = 256, V = 16,384, 5.3 MB; held-out greedy alpha 0.48 (chat) / 0.49 (code), tokens per pass at R = 3: 1.65 / 1.66
    "qwen2.5-0.5b-instruct": DraftHeadPin(
        "qwen2.5-0.5b-instruct-d256-v16384.pgwhead",
        "948b95e70608620b420707e5df5ed69ee8ecf9c07351ff4036c33a80f8242162", 5288953,
        "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    # d = 384, V = 16,384, 11.2 MB; held-out greedy alpha 0.47 (chat) / 0.51 (code), tokens per pass at R = 3 with
    # prompt lookup: 1.66 / 1.69 (replay of the 7B's own answers)
    "qwen2.5-7b-instruct": DraftHeadPin(
        "qwen2.5-7b-instruct-d384-v16384.pgwhead",
        "019397d17a8cc2313288d722c1a791c2e18e04d9a2a92eca2316b879538ad604", 11191161,
        "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_draft_head(path: Path, pin: DraftHeadPin) -> bool:
    """True when ``path`` is the pinned file (size, then SHA-256)."""
    try:
        if path.stat().st_size != pin.size:
            return False
    except OSError:
        return False
    return _sha256(path) == pin.sha256


def _download(url: str, dst: Path, pin: DraftHeadPin, timeout_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".draft-head-", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout_s) as r:  # noqa: S310
            n = 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                n += len(chunk)
                if n > pin.size:
                    raise IntegrityError(f"draft head download {url}: larger than the pinned {pin.size} bytes")
                out.write(chunk)
        if not verify_draft_head(Path(tmp), pin):
            raise IntegrityError(f"draft head download {url}: SHA-256 does not match the pin {pin.sha256[:16]}...")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def resolve_draft_head(model: str, *, cache_dir: Path | None = None, base_url: str | None = None,
                       pins: Mapping[str, DraftHeadPin] | None = None, package_dir: Path | None = None,
                       timeout_s: float = 120.0) -> Path:
    """The verified local path of ``model``'s pinned draft head (see the module docstring).

    Raises ``ConfigError`` when the model has no pinned head or the file cannot be found or fetched, and
    ``IntegrityError`` when the shipped file does not match its pin."""
    table = PINNED_DRAFT_HEADS if pins is None else pins
    pin = table.get(model)
    if pin is None:
        have = ", ".join(sorted(table)) or "none"
        raise ConfigError(f"speculation='draft_head': no trained draft head ships for {model!r} (heads: {have}); "
                          "pass draft_head_path, or use speculation='lookup'")
    shipped = (package_dir or PACKAGE_DIR) / pin.file
    if shipped.exists():
        if not verify_draft_head(shipped, pin):
            raise IntegrityError(f"draft head {shipped} does not match its pinned SHA-256 {pin.sha256[:16]}... "
                                 "(modified installation); reinstall pgw or pass draft_head_path")
        return shipped
    cached = None if cache_dir is None else Path(cache_dir) / "draft-heads" / pin.file
    if cached is not None and cached.exists():
        if verify_draft_head(cached, pin):
            return cached
        log.warning("draft head %s does not match its pin; fetching it again", cached)
        cached.unlink()
    url = base_url if base_url is not None else os.environ.get("PGW_DRAFT_HEAD_URL", "").strip() or None
    if url is None or cached is None:
        raise ConfigError(f"speculation='draft_head': the draft head {pin.file} for {model!r} is not installed "
                          "(set PGW_DRAFT_HEAD_URL to a mirror, or pass draft_head_path)")
    src = url.rstrip("/") + "/" + pin.file
    log.info("fetching draft head %s for %s from %s", pin.file, model, src)
    try:
        _download(src, cached, pin, timeout_s)
    except IntegrityError:
        raise
    except OSError as exc:
        raise ConfigError(f"speculation='draft_head': cannot fetch {src} ({exc})") from exc
    return cached
