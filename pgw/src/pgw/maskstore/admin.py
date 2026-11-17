"""Operator helpers: build a key provider / store from settings, and the ``pgw maskstore`` actions.

The CLI (``pgw.cli``) calls these; they hold no state of their own.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, PGWError
from pgw.maskstore.config import MaskStoreSettings
from pgw.maskstore.keys import (
    EnvKeyProvider,
    FileKeyProvider,
    KeyProvider,
    KmsKeyProvider,
    Pkcs11KeyProvider,
    write_key_file,
)
from pgw.maskstore.store import MaskStore

__all__ = ["gc_store", "import_files", "init_store", "key_provider_from_settings", "open_store", "store_status"]

log = logging.getLogger(__name__)


def key_provider_from_settings(settings: MaskStoreSettings) -> KeyProvider:
    """Instantiate the configured :class:`KeyProvider`."""
    kind = settings.key_provider
    if kind == "file":
        if settings.key_file is None:
            raise ConfigError("key_provider='file' needs key_file (PGW_MASKSTORE_KEY_FILE)")
        return FileKeyProvider(settings.key_file, previous=settings.key_previous_files)
    if kind == "env":
        return EnvKeyProvider(settings.key_env_var)
    if kind == "pkcs11":
        if not (settings.pkcs11_module and settings.pkcs11_token_label and settings.pkcs11_key_label):
            raise ConfigError("key_provider='pkcs11' needs pkcs11_module, pkcs11_token_label and pkcs11_key_label")
        pin = settings.pkcs11_pin.get_secret_value() if settings.pkcs11_pin else None
        return Pkcs11KeyProvider(settings.pkcs11_module, settings.pkcs11_token_label, settings.pkcs11_key_label, pin)
    if settings.kms_key_uri is None:
        raise ConfigError("key_provider='kms' needs kms_key_uri")
    mac = settings.kms_mac_key.get_secret_value() if settings.kms_mac_key else None
    return KmsKeyProvider(settings.kms_key_uri, mac_key_b64=mac)


def open_store(settings: MaskStoreSettings | None = None, **overrides: Any) -> MaskStore:
    """Open the store described by ``settings`` (default: from ``PGW_MASKSTORE_*`` env)."""
    s = settings if settings is not None else MaskStoreSettings(**overrides)
    return MaskStore(s, key_provider_from_settings(s))


def init_store(root: str | Path, key_file: str | Path, **overrides: Any) -> dict[str, Any]:
    """``pgw maskstore init``: create the KEK file (if absent, mode 0600) and an empty store."""
    kf = Path(key_file)
    if not kf.exists():
        write_key_file(kf)
    settings = MaskStoreSettings(root=Path(root), key_provider="file", key_file=kf, **overrides)
    with open_store(settings) as store:
        return store.status()


def store_status(settings: MaskStoreSettings | None = None, **overrides: Any) -> dict[str, Any]:
    """``pgw maskstore status``: inventory, generation and protection summary (no secrets)."""
    with open_store(settings, **overrides) as store:
        return store.status()


def import_files(
    files: Iterable[str | Path],
    settings: MaskStoreSettings | None = None,
    *,
    delete_source: bool = False,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """``pgw maskstore import``: import partner segments; one result entry per file."""
    results: list[dict[str, Any]] = []
    with open_store(settings, **overrides) as store:
        for f in files:
            p = Path(f)
            try:
                seg_id = store.import_segment(p)
            except PGWError as exc:
                log.error("import of %s failed: %s", p.name, exc.message)
                results.append({"file": str(p), "ok": False, "code": str(exc.code), "error": exc.message})
                continue
            if delete_source:
                p.unlink(missing_ok=True)
            results.append({"file": str(p), "ok": True, "segment_id": seg_id})
    return results


def gc_store(settings: MaskStoreSettings | None = None, **overrides: Any) -> int:
    """``pgw maskstore gc``: shred consumed segments; return bytes freed."""
    with open_store(settings, **overrides) as store:
        return store.gc()
