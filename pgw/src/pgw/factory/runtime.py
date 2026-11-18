"""Wiring helpers: settings -> packs, generators, sink and scheduler (used by the CLI and services)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError
from pgw.factory.config import FactorySettings
from pgw.factory.generator import MaskGenerator
from pgw.factory.governor import Throttle
from pgw.factory.scheduler import FactoryScheduler
from pgw.factory.sinks import ExportSink, StoreSink, load_recipient_public_key

__all__ = [
    "build_generators", "build_scheduler", "build_sink", "key_provider_from_settings", "open_packs", "open_store",
]

log = logging.getLogger(__name__)


def key_provider_from_settings(ms: Any) -> Any:
    """Construct the KEK provider configured in :class:`pgw.maskstore.MaskStoreSettings`."""
    from pgw.maskstore import keys

    kind = ms.key_provider
    if kind == "file":
        if ms.key_file is None:
            raise ConfigError("PGW_MASKSTORE_KEY_FILE is required for the 'file' key provider")
        return keys.FileKeyProvider(ms.key_file)
    if kind == "env":
        return keys.EnvKeyProvider(ms.key_env_var)
    if kind == "pkcs11":
        if not (ms.pkcs11_module and ms.pkcs11_token_label and ms.pkcs11_key_label):
            raise ConfigError("pkcs11 key provider needs module, token label and key label")
        return keys.Pkcs11KeyProvider(ms.pkcs11_module, ms.pkcs11_token_label, ms.pkcs11_key_label,
                                      pin=os.environ.get("PGW_MASKSTORE_PKCS11_PIN"))
    if kind == "kms":
        if not ms.kms_key_uri:
            raise ConfigError("kms key provider needs PGW_MASKSTORE_KMS_KEY_URI")
        return keys.KmsKeyProvider(ms.kms_key_uri)
    raise ConfigError(f"unknown key provider {kind!r}")  # pragma: no cover - pydantic validates


def open_store(store_settings: Any = None, **overrides: Any) -> Any:
    """Open the customer's :class:`pgw.maskstore.MaskStore` from ``PGW_MASKSTORE_*`` (+ overrides)."""
    from pgw.maskstore import MaskStore, MaskStoreSettings

    ms = store_settings if store_settings is not None else MaskStoreSettings(**overrides)
    if store_settings is not None and overrides:
        ms = ms.model_copy(update=overrides)
    return MaskStore(ms, key_provider_from_settings(ms))


def open_packs(settings: FactorySettings) -> list[Any]:
    """Open every configured pack with ``components=("linears",)`` and full verification."""
    from pgw.modelpack import ModelPack

    packs = []
    for p in settings.all_pack_paths:
        pack = ModelPack.open(p, verify="full", components=("linears",))
        log.info("factory loaded pack %s (%s) from %s", pack.name, pack.short_fingerprint, Path(p).name)
        packs.append(pack)
    return packs


def build_generators(settings: FactorySettings, packs: list[Any], throttle: Throttle | None = None
                     ) -> dict[tuple[str, str], MaskGenerator]:
    """One :class:`MaskGenerator` per ``(pack, ring)``. Rings a pack cannot decode are rejected."""
    gens: dict[tuple[str, str], MaskGenerator] = {}
    for pack in packs:
        for ring in settings.ring_names:
            remote = build_he_backend(settings, pack, ring) if settings.he_service_url else None
            g = MaskGenerator(pack, ring, device=settings.device, backend=settings.backend, throttle=throttle,
                              pipeline=settings.pipeline, memory_fraction=settings.gpu_memory_fraction,
                              self_check_rows=settings.self_check_rows, remote=remote)
            if g.key in gens:
                raise ConfigError(f"duplicate pack/ring {pack.name}/{ring}")
            gens[g.key] = g
    return gens


def build_he_backend(settings: FactorySettings, pack: Any, ring: str) -> Any:
    """Encrypted mask generation for ``pack``: W8.r from the provider on MLWE ciphertexts (pgw.hemask).

    Builds the one-time hint W8.A locally on first use (minutes for 7B on CPU), then reuses it.
    """
    from pydantic import SecretStr

    from pgw.core.ring import RingParams
    from pgw.hemask.backend import HEMaskBackend, HintStore
    from pgw.hemask.client import RemoteMulmod64
    from pgw.wire import ClientConfig, ComputeClient

    assert settings.he_service_url is not None
    token = None
    if settings.he_token_file is not None:
        token = SecretStr(Path(settings.he_token_file).read_text().strip())
    client = ComputeClient(ClientConfig(base_url=settings.he_service_url, token=token, ca_file=settings.he_ca_file,
                                        request_timeout_s=settings.he_timeout_s,
                                        max_request_bytes=1 << 30))
    hint_root = settings.he_hint_dir or Path(settings.state_dir or ".") / "hemask"
    from pgw.hemask.lwe import HEParams

    q_bits = HEParams(k=RingParams.from_name(ring).k, rank=settings.he_rank).q
    hints = HintStore.open_or_build(Path(hint_root), pack, settings.he_rank, q_bits)
    out = {s.id.key: s.out_features for s in pack.linear_specs()}
    remote = RemoteMulmod64(client, pack.fingerprint, out, timeout_s=settings.he_timeout_s, q_bits=q_bits)
    log.info("factory: encrypted mask generation for %s via %s (rank %d, hint %.2f GB)",
             pack.short_fingerprint, settings.he_service_url, settings.he_rank, hints.nbytes / 1e9)
    return HEMaskBackend(pack, RingParams.from_name(ring), hints, remote, groups_per_call=settings.he_groups_per_call)


def build_sink(settings: FactorySettings, store: Any = None) -> Any:
    """The configured output: :class:`StoreSink` (default) or :class:`ExportSink`."""
    if settings.output == "export":
        if settings.recipient_public_key_file is None:
            raise ConfigError("output='export' needs PGW_FACTORY_RECIPIENT_PUBLIC_KEY_FILE")
        assert settings.export_dir is not None
        signing_key = None
        if settings.partner_signing_key_file is not None:
            from pgw.maskstore.partner import PartnerSigningKey

            signing_key = PartnerSigningKey(settings.partner_signing_key_file)
        return ExportSink(settings.export_dir, load_recipient_public_key(settings.recipient_public_key_file),
                          signing_key=signing_key, destination_store_id=settings.destination_store_id)
    return StoreSink(store if store is not None else open_store())


def build_scheduler(settings: FactorySettings, store: Any = None, *, metrics: Any = None) -> FactoryScheduler:
    """Packs + generators + sink + scheduler with a shared throttle."""
    throttle = Throttle(settings.max_slots_per_hour, settings.duty_cycle)
    packs = open_packs(settings)
    gens = build_generators(settings, packs, throttle)
    return FactoryScheduler(settings, build_sink(settings, store), gens, throttle=throttle, metrics=metrics)
