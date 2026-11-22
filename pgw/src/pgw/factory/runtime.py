"""Wiring helpers: settings -> packs, generators, sink and scheduler (used by the CLI and services)."""

from __future__ import annotations

import logging
import os
import threading
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
    "watch_client_cert",
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


def build_generators(settings: FactorySettings, packs: list[Any], throttle: Throttle | None = None, *,
                     worker_index: int = 0, workers: int = 1) -> dict[tuple[str, str], MaskGenerator]:
    """One :class:`MaskGenerator` per ``(pack, ring)``. Rings a pack cannot decode are rejected.

    ``worker_index``/``workers``: position in a :mod:`pgw.factory.fleet` (which compute service of
    ``he_service_urls`` this worker starts on)."""
    gens: dict[tuple[str, str], MaskGenerator] = {}
    for pack in packs:
        for ring in settings.ring_names:
            remote = (build_he_backend(settings, pack, ring, worker_index=worker_index, workers=workers)
                      if settings.he_urls else None)
            g = MaskGenerator(pack, ring, device=settings.device, backend=settings.backend, throttle=throttle,
                              pipeline=settings.pipeline, memory_fraction=settings.gpu_memory_fraction,
                              self_check_rows=settings.self_check_rows, remote=remote,
                              gpu_prng=None if settings.gpu_prng == "auto" else bool(settings.gpu_prng),
                              skip_kinds=settings.skip_kinds)
            if g.key in gens:
                raise ConfigError(f"duplicate pack/ring {pack.name}/{ring}")
            gens[g.key] = g
    return gens


def build_he_backend(settings: FactorySettings, pack: Any, ring: str, *, worker_index: int = 0,
                     workers: int = 1) -> Any:
    """Encrypted mask generation for ``pack``: W8.r from the provider on MLWE ciphertexts (pgw.hemask).

    Builds the one-time hint W8.A locally on first use (minutes for 7B on CPU), then reuses it. With
    several ``he_service_urls`` the provider calls are sharded by segment
    (:class:`pgw.factory.fleet.SegmentShardedRemote`).
    """
    from pydantic import SecretStr

    from pgw.core.ring import RingParams
    from pgw.hemask.backend import HEMaskBackend, HintStore
    from pgw.hemask.client import RemoteMulmod64
    from pgw.wire import ClientConfig, ComputeClient

    urls = settings.he_urls
    assert urls
    token = None
    if settings.he_token_file is not None:
        token = SecretStr(Path(settings.he_token_file).read_text().strip())
    # The gateway's enrolled client certificate, when configured: a service that requires mTLS for compute
    # refuses the token alone.
    clients = [ComputeClient(ClientConfig(base_url=url, token=token, ca_file=settings.he_ca_file,
                                          client_cert_file=settings.he_client_cert_file,
                                          client_key_file=settings.he_client_key_file,
                                          request_timeout_s=settings.he_timeout_s,
                                          max_request_bytes=1 << 30)) for url in urls]
    if settings.he_client_cert_file is not None and settings.he_client_key_file is not None:
        watch_client_cert(settings.he_client_cert_file, settings.he_client_key_file)
    hint_root = settings.he_hint_dir or Path(settings.state_dir or ".") / "hemask"
    from pgw.hemask.lwe import HEParams

    q_bits = HEParams(k=RingParams.from_name(ring).k, rank=settings.he_rank).q
    hints = HintStore.open_or_build(Path(hint_root), pack, settings.he_rank, q_bits)
    out = {s.id.key: s.out_features for s in pack.linear_specs()}
    he_params = HEParams(k=RingParams.from_name(ring).k, rank=settings.he_rank)
    norms = (getattr(hints, "meta", None) or {}).get("row_norms") if settings.he_wire_v2 else None
    # the bulk channel is per service: only with a single service URL
    bulk = build_bulk_client(settings, token) if len(urls) == 1 else None
    remotes = [RemoteMulmod64(c, pack.fingerprint, out, timeout_s=settings.he_timeout_s, q_bits=q_bits,
                              params=he_params if norms else None, row_norms=norms, wire_v2=settings.he_wire_v2,
                              bulk=bulk, bulk_split=settings.he_bulk_split)
               for c in clients]
    remote: Any = remotes[0]
    if len(remotes) > 1:
        from pgw.factory.fleet import SegmentShardedRemote

        remote = SegmentShardedRemote(remotes, start=worker_index, rotate=workers < len(remotes), names=urls)
    log.info("factory: encrypted mask generation for %s via %s (rank %d, hint %.2f GB)",
             pack.short_fingerprint, ", ".join(urls), settings.he_rank, hints.nbytes / 1e9)
    return HEMaskBackend(pack, RingParams.from_name(ring), hints, remote, groups_per_call=settings.he_groups_per_call,
                         max_inflight=settings.he_max_inflight, encrypt=settings.he_encrypt,
                         pipeline_depth=settings.he_pipeline_depth, decrypt=settings.he_decrypt,
                         hint_ntt_bytes=int(settings.he_hint_ntt_gb * 1e9))


def build_bulk_client(settings: FactorySettings, token: Any = None) -> Any:
    """:class:`pgw.hemask.bulk.BulkMaskgenClient` for ``he_bulk_port`` (``None`` when unset)."""
    if settings.he_bulk_port is None or settings.he_service_url is None:
        return None
    from urllib.parse import urlsplit

    from pgw.hemask.bulk import BulkMaskgenClient
    from pgw.wire import ClientConfig
    from pgw.wire.client import build_ssl_context

    url = urlsplit(settings.he_service_url)
    host = url.hostname or "127.0.0.1"
    ctx = None
    if url.scheme == "https":
        cfg = ClientConfig(base_url=settings.he_service_url, ca_file=settings.he_ca_file,
                           client_cert_file=settings.he_client_cert_file, client_key_file=settings.he_client_key_file)
        ctx = build_ssl_context(cfg)  # registered: renewed client certificates reach new connections
        ctx.set_alpn_protocols(["pgw-maskgen-bulk"])
    elif host not in ("127.0.0.1", "::1", "localhost"):
        raise ConfigError("he_bulk_port with a plain-http service URL is only allowed on a loopback host")
    auth = f"Bearer {token.get_secret_value()}" if token is not None else None
    log.info("factory: maskgen bulk channel %s:%d (tls=%s, split %d)", host, settings.he_bulk_port, ctx is not None,
             settings.he_bulk_split)
    return BulkMaskgenClient(host, settings.he_bulk_port, ssl_context=ctx, server_hostname=host if ctx else None,
                             authorization=auth, timeout_s=settings.he_timeout_s,
                             max_conns=max(4, 4 * settings.he_bulk_split))


_WATCHED: dict[tuple[str, str], threading.Thread] = {}
_WATCH_LOCK = threading.Lock()


def watch_client_cert(cert_file: Path, key_file: Path, *, interval_s: float = 60.0,
                      stop: threading.Event | None = None) -> threading.Thread:
    """Reload a client certificate pair into this process's compute clients when the files change.

    The gateway renews its client certificate in place (same paths) in its own process; the factory is a
    separate process, so it polls the files (modification time and size) and calls
    :func:`pgw.wire.client.reload_client_cert` on a change. One watcher per pair per process.
    """
    key = (str(Path(cert_file).resolve()), str(Path(key_file).resolve()))
    with _WATCH_LOCK:
        t = _WATCHED.get(key)
        if t is not None and t.is_alive():
            return t
        ev = stop or threading.Event()

        def stamp() -> tuple[tuple[int, int], ...] | None:
            try:
                return tuple((os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in key)
            except OSError:
                return None

        def run() -> None:
            from pgw.wire.client import reload_client_cert

            last = stamp()
            while not ev.wait(interval_s):
                cur = stamp()
                if cur is None or cur == last:
                    continue
                try:
                    reload_client_cert(Path(key[0]), Path(key[1]))
                    last = cur
                    log.info("factory: loaded the renewed client certificate %s", Path(key[0]).name)
                except ConfigError as exc:  # e.g. between the key and the certificate rename: retry next time
                    log.warning("factory: cannot load the renewed client certificate yet: %s", exc.message)

        t = threading.Thread(target=run, name="pgw-factory-cert-watch", daemon=True)
        _WATCHED[key] = t
        t.start()
        return t


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
