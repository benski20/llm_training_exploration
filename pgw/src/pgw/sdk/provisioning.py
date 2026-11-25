"""Provisioning adapter: what the in-process gateway needs from :func:`pgw.provision.client.provision`.

The provisioning function (built by the provisioning package) returns the non-secret model artifacts, checked
against the pack fingerprint, and a ready pks-v3 hint store. This module defines the small interface the
library consumes, so it can run against the real function (wired by name, imported lazily) or against a stub
that uses a model pack already on disk (tests, air-gapped installs, the existing local pack / hint code paths).

Contract (``pgw.provision.client.provision``)::

    provision(model: str, *, session: pgw.sdk.session.Session, cache_dir: Path,
              progress: Callable[[str], None] | None) -> ProvisionedModel

The returned object is read through :func:`adapt` and needs these attributes (aliases accepted):

* ``fingerprint`` (str, 64 hex): the pack fingerprint the provider serves (must equal the enrollment entry).
* ``pack_path`` (alias ``pack_dir``, ``path``): a directory :meth:`pgw.modelpack.ModelPack.open` accepts with
  ``verify="full"`` and ``expected_fingerprint=fingerprint`` (the gateway re-verifies it at every start).
* ``hint_dir`` (optional; alias ``hint_root``, or ``hint_store.root``): the pks-v3 hint root (the factory's
  ``he_hint_dir``: :meth:`pgw.hemask.pks_backend.PKSHintStore.open` finds ``<root>/<fp-dir>``). ``None``: the
  factory builds the hint locally on first use (``hint="local"``).
* ``hint_key_file`` (optional): the 32-byte key authenticating the hint store's ``meta.json``.
* ``name`` (optional): the model name.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pgw.core.errors import ConfigError

__all__ = [
    "PROVISION_FUNCTION",
    "LocalPackProvisioner",
    "Provisioned",
    "Provisioner",
    "adapt",
    "default_provisioner",
]

log = logging.getLogger(__name__)

#: The real provisioning function, imported by name on first use.
PROVISION_FUNCTION = "pgw.provision.client:provision"


@runtime_checkable
class Provisioner(Protocol):
    """``provision(model, *, session, cache_dir, progress) -> object`` (see the module docstring)."""

    def __call__(self, model: str, *, session: Any, cache_dir: Path,
                 progress: Callable[[str], None] | None = None) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class Provisioned:
    """The normalized result of a provisioning call."""

    name: str
    fingerprint: str
    pack_path: Path
    hint_dir: Path | None = None
    hint_key_file: Path | None = None
    source: str = "provider"


def _first(obj: Any, *names: str) -> Any:
    for n in names:
        v = obj.get(n) if isinstance(obj, Mapping) else getattr(obj, n, None)
        if v is not None:
            return v
    return None


def adapt(obj: Any, *, model: str) -> Provisioned:
    """Read a provisioning result (object or mapping) into :class:`Provisioned`.

    Raises:
        ConfigError: no fingerprint or no pack directory.
    """
    if isinstance(obj, Provisioned):
        return obj
    fp = _first(obj, "fingerprint", "pack_fingerprint")
    pack = _first(obj, "pack_path", "pack_dir", "path")
    if fp is None or pack is None:
        raise ConfigError(f"provisioning of {model!r} returned no fingerprint or pack directory")
    hint = _first(obj, "hint_dir", "hint_root")
    if hint is None:
        store = _first(obj, "hint_store", "hints")
        hint = getattr(store, "root", None) if store is not None else None
    key = _first(obj, "hint_key_file")
    fp = str(fp).strip().lower()
    if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
        raise ConfigError(f"provisioning of {model!r} returned an invalid fingerprint")
    return Provisioned(name=str(_first(obj, "name") or model), fingerprint=fp, pack_path=Path(pack),
                       hint_dir=Path(hint) if hint is not None else None,
                       hint_key_file=Path(key) if key is not None else None,
                       source=str(_first(obj, "source") or "provider"))


def _resolve(spec: str) -> Callable[..., Any]:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)  # type: ignore[no-any-return]


def default_provisioner() -> Provisioner:
    """The real provisioning function (:data:`PROVISION_FUNCTION`), resolved on first call.

    Raises (on call):
        ConfigError: the provisioning package is not installed in this build.
    """

    def provision(model: str, *, session: Any, cache_dir: Path,
                  progress: Callable[[str], None] | None = None) -> Any:
        try:
            fn = _resolve(PROVISION_FUNCTION)
        except (ImportError, AttributeError) as exc:
            raise ConfigError("this pgw build has no provisioning client (pgw.provision.client.provision); "
                              "pass provisioner=LocalPackProvisioner({...}) with local model packs") from exc
        return fn(model, session=session, cache_dir=cache_dir, progress=progress)

    return provision


class LocalPackProvisioner:
    """Provisioning from model packs already on this host (the existing local pack and hint code paths).

    Args:
        packs: ``{model name: pack directory}``. The pack is checked against the provider's fingerprint for the
            model (``session.model_entry(model)``) with a full verification.
        hint_dirs: Optional ``{model name: pks-v3 hint root}``; without one the factory builds the hint locally.
    """

    def __init__(self, packs: Mapping[str, Path | str], hint_dirs: Mapping[str, Path | str] | None = None):
        self.packs = {k.lower(): Path(v) for k, v in packs.items()}
        self.hint_dirs = {k.lower(): Path(v) for k, v in (hint_dirs or {}).items()}

    def __call__(self, model: str, *, session: Any, cache_dir: Path,
                 progress: Callable[[str], None] | None = None) -> Provisioned:
        from pgw.modelpack import ModelPack

        path = self.packs.get(model.lower())
        if path is None:
            raise ConfigError(f"no local pack for model {model!r}")
        entry = session.model_entry(model) if session is not None else None
        expected = str(entry["fingerprint"]) if entry else None
        if progress is not None:
            progress(f"verifying the local pack for {model}")
        with ModelPack.open(path, verify="full", expected_fingerprint=expected) as p:
            fp = p.fingerprint
        return Provisioned(name=model, fingerprint=fp, pack_path=path, hint_dir=self.hint_dirs.get(model.lower()),
                           source="local")
