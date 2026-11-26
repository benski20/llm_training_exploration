"""Provisioning: what the in-process gateway and its background mask factory need from the provider.

The default provisioner is :class:`ProviderProvisioner`, a thin layer over :func:`pgw.provision.client.provision`
(``docs/sdk/CONTRACT.md``, appendix A5) that adds what the client library decides:

* **Fingerprint pins.** The provider's catalog alone is never trusted. Every supported model has a pin
  (:data:`PINNED_MODELS`): the public Hugging Face checkpoint (repository and commit) and, where it was derived for
  this release, the expected pack fingerprint (``pgw pack <repo> --revision <commit> --name <model>`` with the default
  packing parameters). A provider serving another fingerprint is refused before anything is downloaded. For a model
  whose fingerprint is not baked in, the first provisioning re-derives the pack from the pinned public checkpoint
  (``source="hf"``) and requires the provider's fingerprint to match; the result is remembered in the cache
  (``public-check.json``) and the derived pack is reused. ``SDKOptions.model_fingerprints`` adds or replaces pins
  (custom packs); a replaced pin is the caller's explicit decision.
* **Hint.** ``hint="local"`` (default): the background mask factory builds the pks-v3 hint itself from the
  fingerprint-checked int8 weights (one-time: about 11 CPU-minutes at 7B with AVX-512 VNNI / AMX, then cached),
  unless :meth:`ProviderProvisioner.prepare` (``pgw sdk prepare``) built it ahead of time in the same pass as the
  factory pack; a prepared store is used when its ``meta.json`` verifies under the cache's hint key.
  ``hint="provider"`` (opt-in): the provider computes the expensive part for client-chosen seeds and the client
  accepts it only after a >= 40-bit check against the same weights (see :mod:`pgw.provision.client`); it streams
  about 44 GB at 7B.
* **Factory pack.** The gateway reads the weight-free pack (``pack_path``); the background factory needs the int8
  weights (its private check vectors ``v W8``, and the hint with ``hint="local"``), so they are kept once as a full
  pack next to it (``factory_pack_path``; about 6.5 GB at 7B, streamed from the provider and checked against the
  fingerprint).

A provisioner is any callable ``provision(model, *, session, cache_dir, progress) -> object`` whose result
:func:`adapt` reads (``fingerprint``, ``pack_path``/``pack_dir``/``path``, optional ``hint_dir``/``hint_root``,
``hint_key_file``, ``factory_pack_path``/``factory_pack_dir``, ``name``, ``source``). A provisioner that also
accepts ``masked_head=`` gets the engine's decision. :class:`LocalPackProvisioner` serves full packs already on disk
(tests, air-gapped installs).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pgw.core.errors import ConfigError, FingerprintMismatchError

__all__ = [
    "PINNED_MODELS",
    "LocalPackProvisioner",
    "ModelPin",
    "ProviderProvisioner",
    "Provisioned",
    "Provisioner",
    "adapt",
    "default_provisioner",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPin:
    """What a supported model must be: its public checkpoint and, when known, its pack fingerprint.

    Attributes:
        hf_repo: Hugging Face repository of the public checkpoint.
        revision: Commit of that repository (the pack records it; packing is deterministic for it).
        fingerprint: Expected pack fingerprint (``pgw pack hf_repo --revision revision --name <model>``, default
            parameters), or ``None``: derived from the public checkpoint on first use and cross-checked.
    """

    hf_repo: str
    revision: str
    fingerprint: str | None = None


#: The release's pins. 0.5B and 1.5B were derived with ``scripts/pin_model_fingerprints.py`` (the repository's pack
#: tool on the public checkpoints at these commits); 3B and 7B are pinned by checkpoint commit and verified by
#: re-deriving the pack from the public checkpoint on first provisioning (run the script on a host with those
#: checkpoints to bake their fingerprints in). The commits are the provider's (deploy/runpod/pod_boot.MODELS).
PINNED_MODELS: dict[str, ModelPin] = {
    "qwen2.5-0.5b-instruct": ModelPin("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775",
                                      "915e55c19364542c4fd8ab5c8f7154806e2511048755e0d10426374b5c8fe9c7"),
    "qwen2.5-1.5b-instruct": ModelPin("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                                      "81bb7adcea1524171297c0004c0efb1edad9f71dd0535462c9b32ab03ac99348"),
    "qwen2.5-3b-instruct": ModelPin("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen2.5-7b-instruct": ModelPin("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}


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
    #: full pack (int8 weights) for the background factory; ``None``: ``pack_path`` is a full pack
    factory_pack_path: Path | None = None


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
    fpack = _first(obj, "factory_pack_path", "factory_pack_dir")
    fp = str(fp).strip().lower()
    if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
        raise ConfigError(f"provisioning of {model!r} returned an invalid fingerprint")
    return Provisioned(name=str(_first(obj, "name") or model), fingerprint=fp, pack_path=Path(pack),
                       hint_dir=Path(hint) if hint is not None else None,
                       hint_key_file=Path(key) if key is not None else None,
                       source=str(_first(obj, "source") or "provider"),
                       factory_pack_path=Path(fpack) if fpack is not None else None)


def _progress_text(ev: Any) -> str:
    if not isinstance(ev, Mapping):
        return str(ev)
    stage, done, total, unit = ev.get("stage"), ev.get("done", 0), ev.get("total", 0), ev.get("unit", "")
    if unit == "bytes" and total:
        return f"{stage}: {done / 1e6:.1f} / {total / 1e6:.1f} MB"
    if total:
        return f"{stage}: {done} / {total} {unit}"
    return str(stage)


class ProviderProvisioner:
    """The default provisioner: :func:`pgw.provision.client.provision` with the library's pins and hint choice.

    Args:
        hint: ``"local"`` (default) or ``"provider"`` (see the module docstring).
        fingerprints: Extra or replacement pins ``{model: fingerprint}`` (``SDKOptions.model_fingerprints``).
        pins: The release pins (default :data:`PINNED_MODELS`).
        timeout_s: HTTP timeout of one provisioning request.
    """

    def __init__(self, *, hint: Literal["local", "provider"] = "local",
                 fingerprints: Mapping[str, str] | None = None, pins: Mapping[str, ModelPin] | None = None,
                 timeout_s: float = 600.0):
        if hint not in ("local", "provider"):
            raise ConfigError("hint must be 'local' or 'provider'")
        self.hint = hint
        self.fingerprints = {k.lower(): str(v).lower() for k, v in (fingerprints or {}).items()}
        self.pins = dict(PINNED_MODELS if pins is None else pins)
        self.timeout_s = timeout_s

    @classmethod
    def from_options(cls, options: Any) -> ProviderProvisioner:
        """The provisioner for :class:`pgw.sdk.options.SDKOptions` (``hint``, ``model_fingerprints``)."""
        return cls(hint=getattr(options, "hint", "local"),
                   fingerprints=getattr(options, "model_fingerprints", None) or None,
                   timeout_s=max(60.0, float(getattr(options, "timeout_s", 600.0))))

    def expected_fingerprint(self, model: str) -> str | None:
        """The pinned fingerprint of ``model`` (an explicit option first, then the release pin)."""
        m = model.lower()
        if m in self.fingerprints:
            return self.fingerprints[m]
        pin = self.pins.get(m)
        return pin.fingerprint if pin is not None else None

    def __call__(self, model: str, *, session: Any, cache_dir: Path,
                 progress: Callable[[str], None] | None = None, masked_head: bool = False) -> Provisioned:
        """Provision ``model`` (blocking; the engine runs it in a worker thread).

        With ``hint="local"`` a complete hint store prepared ahead of time (:meth:`prepare`, ``pgw sdk prepare``;
        here or on another host whose cache was copied) is handed to the factory when its ``meta.json`` verifies
        under the cache's hint key (:func:`pgw.provision.client.find_prepared_hint`); otherwise the factory builds the
        hint on first use, as before.

        Raises:
            FingerprintMismatchError: the provider serves another pack than the pin (or than the public checkpoint).
            IntegrityError: an artifact, a weight or a provider hint operand failed its check.
            ConfigError / ModelNotFoundError / ServiceUnavailableError (from :func:`pgw.provision.client.provision`).
        """
        return self._provision(model, session=session, cache_dir=cache_dir, progress=progress,
                               masked_head=masked_head)

    def prepare(self, model: str, *, session: Any, cache_dir: Path, progress: Callable[[str], None] | None = None,
                masked_head: bool = False, build_hint: bool = True) -> Provisioned:
        """The one-time setup of ``model`` ahead of the first request (``pgw sdk prepare``): artifacts, the factory
        pack and, with ``hint="local"`` and ``build_hint``, the pks-v3 hint, built in one pass over the weights on
        every core and resumable after an interruption (:func:`pgw.provision.client.provision` with ``resume``)."""
        return self._provision(model, session=session, cache_dir=cache_dir, progress=progress,
                               masked_head=masked_head, build_hint=build_hint, resume=True)

    def _provision(self, model: str, *, session: Any, cache_dir: Path, progress: Callable[[str], None] | None,
                   masked_head: bool, build_hint: bool = False, resume: bool = False) -> Provisioned:
        from pgw.core.fingerprint import validate_fingerprint
        from pgw.provision.client import provision
        from pgw.provision.session import ProvisionSession

        entry = session.model_entry(model) if session is not None else None
        if not entry:
            raise ConfigError(f"the provider's enrollment does not list model {model!r}")
        served = validate_fingerprint(str(entry["fingerprint"]).lower())
        expected = self.expected_fingerprint(model)
        pin = self.pins.get(model.lower())
        if expected is not None and served != expected:
            raise FingerprintMismatchError(
                f"the provider serves {model!r} as pack {served[:12]}, but this library pins {expected[:12]} "
                "(a different checkpoint or packing); refusing it", details={"model": model})
        if expected is None:
            if pin is None:
                raise ConfigError(f"no fingerprint pin for model {model!r}: pass model_fingerprints={{...}}")
            for k, want in (("hf_repo", pin.hf_repo), ("revision", pin.revision)):
                got = entry.get(k)
                if got is not None and str(got) != want:
                    raise FingerprintMismatchError(
                        f"the provider packs {model!r} from {k} {str(got)[:48]!r}, not the pinned {want[:48]!r}",
                        details={"model": model})
        root = Path(cache_dir)
        root = root.parent if root.name == "models" else root  # provision() keeps models/ and keys/ under it

        def say(ev: Any) -> None:
            if progress is not None:
                progress(_progress_text(ev))

        ps = ProvisionSession(base_url=session.service_url, token=session.token().get_secret_value(),
                              ca_file=session.ca_file, client_cert_file=session.client_cert_file,
                              client_key_file=session.client_key_file, timeout_s=self.timeout_s)
        hint: Literal["provider", "local", "none"] = "provider" if self.hint == "provider" else "none"
        if build_hint and self.hint == "local":
            hint = "local"  # prepare: the fused, resumable build now (else: the factory on first use)
        source = "provider"
        kw: dict[str, Any] = {"session": ps, "cache_dir": root, "progress": say, "expected_fingerprint": served,
                              "masked_head": masked_head, "factory_pack": True, "resume": resume}
        try:
            if expected is None and not self._public_check_ok(root, served, pin):
                assert pin is not None
                log.info("provisioning %s: deriving the pack from the public checkpoint %s@%s to check the "
                         "provider's fingerprint (first use only)", model, pin.hf_repo, pin.revision[:12])
                pm = provision(model, hint=hint, source="hf", hf_repo=pin.hf_repo, hf_revision=pin.revision,
                               hf_name=model, **kw)
                self._record_public_check(root, served, pin)
                source = "provider+public-check"
            else:
                pm = provision(model, hint=hint, **kw)
        finally:
            ps.close()
        try:
            has_hint = pm.hint_store is not None
            hint_dir, key_file = (pm.hint_root, pm.hint_key_file) if has_hint else (None, None)
            if not has_hint and self.hint == "local":
                from pgw.provision.client import find_prepared_hint

                found = find_prepared_hint(root, pm.fingerprint, masked_head=masked_head)
                if found is not None:
                    hint_dir, key_file = found
                    source += "+prepared-hint"
            return Provisioned(name=model, fingerprint=pm.fingerprint, pack_path=pm.pack_dir, hint_dir=hint_dir,
                               hint_key_file=key_file, source=source, factory_pack_path=pm.factory_pack_dir)
        finally:
            pm.close()  # the factory opens the hint store itself

    @staticmethod
    def _check_file(root: Path, fp: str) -> Path:
        return root / "models" / fp[:16] / "public-check.json"

    def _public_check_ok(self, root: Path, fp: str, pin: ModelPin | None) -> bool:
        if pin is None:
            return False
        try:
            d = json.loads(self._check_file(root, fp).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(d.get("fingerprint") == fp and d.get("hf_repo") == pin.hf_repo
                    and d.get("revision") == pin.revision)

    def _record_public_check(self, root: Path, fp: str, pin: ModelPin) -> None:
        from pgw.core.fileio import atomic_write_text

        with contextlib.suppress(OSError):
            atomic_write_text(self._check_file(root, fp), json.dumps(
                {"fingerprint": fp, "hf_repo": pin.hf_repo, "revision": pin.revision, "checked_at": time.time()})
                + "\n", mode=0o600)


def default_provisioner(options: Any = None) -> Provisioner:
    """The default provisioner for ``options`` (:class:`ProviderProvisioner`)."""
    return ProviderProvisioner.from_options(options) if options is not None else ProviderProvisioner()


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
