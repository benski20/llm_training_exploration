"""``pgw doctor``: environment and deployment checks.

Every check yields a :class:`Check` with status ``ok``, ``warn``, ``fail`` or ``skip``. The command
exits 1 when any check fails (or, with ``--strict``, warns). Checks never print secrets: key files
are checked for permissions only, tokens are only used to authenticate the probe request.

Checks (run when relevant to ``--role`` and the given options):

* ``python`` / ``packages`` / ``extras``: interpreter version, core and per-role dependencies.
* ``entropy``: the OS CSPRNG and ChaCha20 (mask derivation, SPEC 7.3) are usable.
* ``selftest``: mask -> exact ring matmul -> unmask -> Freivalds on every ring preset and every
  available backend (:func:`pgw.core.selftest.run_selftest`).
* ``cpu_int8`` / ``cuda``: fast exact int8 paths and GPU presence.
* ``memory``: available RAM.
* ``settings.<role>``: the ``PGW_*`` settings for the role parse and pass the production policy.
* ``pack``: full SHA-256 verification of a model pack, ring headroom, mask bytes per slot; with
  ``--deep`` also the weight-row norms (manifest ``row_norm_sq_max`` and any ``wnorm`` sidecar) against
  the int8 weights.
* ``maskstore``: KEK file permissions, store integrity/inventory, free disk space.
* ``maskstore.read_rate``: writes a :data:`MASKSTORE_PROBE_BYTES` probe file next to the store, drops it
  from the page cache and reads it back in 1 MiB blocks in shuffled order (the gateway reads each slot's
  ``W r`` chunks); reports MB/s and, with ``--pack``, the tokens/s that rate can feed. Fails below the
  expected load (``expected_tokens_per_second`` x mask bytes per slot), warns within
  :data:`SUPPLY_MARGIN` of it, and without an expected load warns below :data:`MASKSTORE_MIN_READ_MB_S`
  (one chat-speed 7B stream; a gp3 volume at its 125 MB/s baseline is below it: masks need NVMe or RAM).
* ``service``: the compute service answers ``/readyz`` and ``/pgw/v1/models`` and serves the
  pack's fingerprint and ring.
* ``mask_supply``: times one mask chunk end to end into a temporary store (encrypted generation over
  the network when ``PGW_FACTORY_HE_SERVICE_URL`` is set and the hint is built, else local GEMMs),
  so the rate is bounded by the measured link throughput; compares it with the expected load
  (``expected_tokens_per_second`` / ``PGW_DOCTOR_EXPECTED_TOKENS_PER_S``) and checks that the store
  volume holds ``target_slots``. Opt-in: ``--mask-supply`` with ``--pack``.
* ``tls_certs``: certificate files from the role's settings and the given CA file; warns when one
  expires within :data:`CERT_WARN_DAYS` days, fails when one has expired.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import logging
import os
import platform
import shutil
import stat
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = ["ROLES", "ROLE_EXTRAS", "Check", "DoctorReport", "format_report", "run_doctor"]

log = logging.getLogger(__name__)

Status = Literal["ok", "warn", "fail", "skip"]
ROLES = ("all", "gateway", "service", "factory", "pack")

#: Python modules each deployment role needs beyond the core dependencies (import names).
ROLE_EXTRAS: dict[str, tuple[str, ...]] = {
    "gateway": ("fastapi", "uvicorn", "httpx", "h2", "transformers", "tokenizers", "prometheus_client"),
    "service": ("fastapi", "hypercorn", "h2", "httpx", "prometheus_client"),
    "factory": ("httpx", "prometheus_client"),
    "pack": ("transformers", "huggingface_hub", "safetensors"),
}
CORE_MODULES = ("numpy", "torch", "cryptography", "pydantic", "pydantic_settings", "msgpack", "safetensors", "click")
MIN_FREE_DISK_BYTES = 1 << 30
MIN_AVAILABLE_RAM_BYTES = 1 << 30
#: Certificates expiring within this many days are reported as a warning.
CERT_WARN_DAYS = 7
#: Env fallback for the expected mask consumption (tokens per second) of the mask-supply check.
EXPECTED_TPS_ENV = "PGW_DOCTOR_EXPECTED_TOKENS_PER_S"
#: Supply must exceed the expected load by this factor to pass without a warning.
SUPPLY_MARGIN = 1.2
#: Size of the mask-store read probe (``maskstore.read_rate``; written next to the store, then deleted).
MASKSTORE_PROBE_BYTES = 64 << 20
#: Block size of the probe's reads (the order of a slot's per-linear ``W r`` chunks).
MASKSTORE_PROBE_BLOCK = 1 << 20
#: Without an expected load, read rates below this warn: one chat-speed Qwen2.5-7B stream needs about
#: 5.56 MB/slot x 45 tokens/s = 250 MB/s (docs/operations/sizing.md, section 4).
MASKSTORE_MIN_READ_MB_S = 250.0


@dataclass
class Check:
    """Result of one doctor check."""

    name: str
    status: Status
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0


@dataclass
class DoctorReport:
    """All check results plus a summary."""

    checks: list[Check] = field(default_factory=list)
    strict: bool = False

    @property
    def ok(self) -> bool:
        """No failed checks (and, when strict, no warnings)."""
        bad: set[str] = {"fail", "warn"} if self.strict else {"fail"}
        return not any(c.status in bad for c in self.checks)

    def counts(self) -> dict[str, int]:
        """Number of checks per status."""
        out = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    def get(self, name: str) -> Check | None:
        """The check called ``name`` (or ``None``)."""
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        from pgw import __version__

        return {
            "ok": self.ok,
            "strict": self.strict,
            "pgw": __version__,
            "python": platform.python_version(),
            "summary": self.counts(),
            "checks": [asdict(c) for c in self.checks],
        }



def _settings_without_env(cls: Any, values: dict[str, Any]) -> Any:
    """``cls(**values)`` for a pydantic-settings class, ignoring environment variables and ``.env``."""

    class _NoEnv(cls):  # type: ignore[misc]
        @classmethod
        def settings_customise_sources(cls, settings_cls: Any, init_settings: Any, *args: Any, **kwargs: Any) -> Any:
            return (init_settings,)

    return _NoEnv(**values)

def _run(report: DoctorReport, name: str, fn: Callable[[], tuple[Status, str, dict[str, Any]]]) -> Check:
    """Run one check; unexpected exceptions become a ``fail`` with the error type and message."""
    from pgw.core.errors import PGWError

    t0 = time.perf_counter()
    try:
        status, detail, data = fn()
    except PGWError as exc:
        status, detail, data = "fail", f"[{exc.code}] {exc.message}", {}
    except Exception as exc:  # report, never crash the doctor
        from pydantic import ValidationError

        from pgw.cli._common import short_validation

        log.debug("doctor check %s raised", name, exc_info=True)
        if isinstance(exc, ValidationError):  # never echo input values (may be secrets)
            status, detail, data = "fail", f"invalid settings: {short_validation(exc)}", {}
        else:
            status, detail, data = "fail", f"{type(exc).__name__}: {exc}", {}
    c = Check(name, status, detail, data, round((time.perf_counter() - t0) * 1e3, 1))
    report.checks.append(c)
    return c


# ---- individual checks ---------------------------------------------------------------------------
def _check_python() -> tuple[Status, str, dict[str, Any]]:
    v = sys.version_info
    data = {"version": platform.python_version(), "implementation": platform.python_implementation()}
    if (v.major, v.minor) < (3, 11):
        return "fail", f"Python {data['version']} < 3.11", data
    return "ok", f"Python {data['version']}", data


def _version_of(module: str) -> str | None:
    dist = {"pydantic_settings": "pydantic-settings", "prometheus_client": "prometheus-client",
            "huggingface_hub": "huggingface-hub"}.get(module, module)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def _check_packages() -> tuple[Status, str, dict[str, Any]]:
    data: dict[str, Any] = {}
    missing = []
    for m in CORE_MODULES:
        if _importable(m):
            data[m] = _version_of(m) or "?"
        else:
            missing.append(m)
    if missing:
        return "fail", f"missing core modules: {', '.join(missing)}", data
    return "ok", f"torch {data.get('torch')}, numpy {data.get('numpy')}, cryptography {data.get('cryptography')}", data


def _check_extras(roles: Iterable[str], required: bool) -> tuple[Status, str, dict[str, Any]]:
    data: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for role in roles:
        miss = [m for m in ROLE_EXTRAS[role] if not _importable(m)]
        data[role] = "ok" if not miss else {"missing": miss}
        if miss:
            missing[role] = miss
    if not missing:
        return "ok", "all extras for " + ", ".join(roles), data
    msg = "; ".join(f"{r}: {', '.join(m)} (pip install 'pgw-sdk[{r}]')" for r, m in missing.items())
    return ("fail" if required else "warn"), msg, data


def _check_entropy() -> tuple[Status, str, dict[str, Any]]:
    from pgw.core.prng import SeedExpander

    a, b = os.urandom(32), os.urandom(32)
    if a == b or len(set(a)) < 8:
        return "fail", "os.urandom returned repeated/low-entropy output", {}
    sx = SeedExpander(a)
    r1 = sx.ring_uniform((4, 8), 40, "doctor")
    r2 = SeedExpander(a).ring_uniform((4, 8), 40, "doctor")
    if not bool((r1 == r2).all()) or int(r1.max()) >= (1 << 40) or int(r1.min()) < 0:
        return "fail", "ChaCha20 seed expansion is not deterministic or out of range", {}
    return "ok", "os.urandom and ChaCha20 seed expansion OK", {}


def _check_selftest(include_gpu: bool) -> tuple[Status, str, dict[str, Any]]:
    from pgw.core.selftest import run_selftest

    res = run_selftest(include_gpu=include_gpu)
    failures = sorted(k for k, v in res.checks.items() if not v)
    backends = sorted({k.split("/")[1] for k in res.checks})
    data = {"backends": backends, "failures": failures, "timings_ms": res.timings_ms}
    if not res.ok:
        return "fail", f"exactness/integrity failures: {', '.join(failures)}", data
    return "ok", f"{len(res.checks)} checks exact on {', '.join(backends)}", data


def _check_cpu_int8() -> tuple[Status, str, dict[str, Any]]:
    try:
        from pgw.service.kernels import cpu_int8_available
    except ImportError:
        try:
            from pgw.factory.kernels import cpu_int8_available
        except ImportError:
            return "skip", "no component with a cpu_int8 kernel is installed", {}
    if cpu_int8_available():
        return "ok", "exact torch._int_mm on CPU available (fast CPU path)", {"available": True}
    return "warn", "CPU int8 GEMM not exact/available; CPU falls back to float64 limbs (slower)", {"available": False}


def _check_cuda(role: str) -> tuple[Status, str, dict[str, Any]]:
    import torch

    from pgw.core.ringmm import cuda_int8_available

    if not torch.cuda.is_available():
        needs_gpu = role == "service"
        return ("warn" if needs_gpu else "ok"), "no CUDA device (CPU backends will be used)", {"available": False}
    devices: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                        "memory_gib": round(p.total_memory / 2**30, 1)})
    int8 = cuda_int8_available()
    data = {"available": True, "devices": devices, "cuda_int8": int8, "torch_cuda": torch.version.cuda}
    if not int8:
        return "warn", f"{len(devices)} CUDA device(s) but exact cuda_int8 path unavailable", data
    return "ok", f"{len(devices)} CUDA device(s): " + ", ".join(str(d["name"]) for d in devices), data


def _check_native() -> tuple[Status, str, dict[str, Any]]:
    """Which native extensions are loaded (prebuilt in the wheels; pure-Python fallback otherwise)."""
    from pgw.core import native as fast_mod
    from pgw.hemask import ntt, pks_native

    loaded = {"pgw_fast": fast_mod.fast is not None, "pgw_ntt": ntt.native is not None,
              "pgw_pks": pks_native.native() is not None}
    data: dict[str, Any] = dict(loaded)
    missing = [k for k, ok in loaded.items() if not ok]
    if not missing:
        return "ok", "pgw_fast, pgw_ntt, pgw_pks loaded", data
    off = os.environ.get("PGW_NATIVE", "1").strip().lower() in ("0", "false", "no", "off")
    why = "disabled by PGW_NATIVE=0" if off else "not installed (sdist without Rust, or unsupported platform)"
    return "warn", f"{', '.join(missing)} {why}: pure-Python fallback, same results, much slower", data


def _check_gateway_gpu() -> tuple[Status, str, dict[str, Any]]:
    """Which customer-side parts run on a gateway GPU (runs the encrypted factory's device self-test)."""
    try:
        from pgw.gateway.gpu_report import gpu_report
    except ModuleNotFoundError as exc:  # client install: the gateway server extra is not installed
        return "skip", f"gateway server extra not installed ({exc.name}); nothing to report", {}

    data = gpu_report(run_self_test=True)
    he = data["encrypted_mask_factory"]
    if not data["cuda"]:
        return "ok", "no gateway GPU: encrypted factory products on " + str(he["engine"]), data
    if he.get("cuda_disabled"):
        return "warn", f"gateway GPU present but its encrypted-factory self-test failed ({he['cuda_disabled']})", data
    where = ", ".join(data["on_gpu"]) or "nothing"
    return "ok", f"on the gateway GPU: {where}; decode float path on the CPU (reference bits)", data


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            parts = v.split()
            if parts and parts[0].isdigit():
                out[k] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def _check_memory() -> tuple[Status, str, dict[str, Any]]:
    import torch

    mi = _meminfo()
    data = {"cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "mem_total": mi.get("MemTotal"), "mem_available": mi.get("MemAvailable")}
    avail = mi.get("MemAvailable")
    if avail is None:
        return "skip", "cannot read /proc/meminfo", data
    detail = f"{data['cpus']} CPUs, {avail / 2**30:.1f} GiB RAM available"
    if avail < MIN_AVAILABLE_RAM_BYTES:
        return "warn", detail + " (< 1 GiB)", data
    return "ok", detail, data


def _check_role_settings(role: str) -> tuple[Status, str, dict[str, Any]]:
    """Parse ``PGW_<ROLE>_*`` settings and apply the production policy checks."""
    warnings: list[str] = []
    data: dict[str, Any] = {}
    if role == "service":
        from pgw.service.config import ServiceSettings

        s = ServiceSettings()
        data = {"packs": [str(p) for p in s.packs], "tls": s.tls.enabled, "port": s.port}
        if not s.packs:
            warnings.append("no packs configured (PGW_SERVICE_PACKS)")
        missing = [str(p) for p in s.packs if not Path(p).is_dir()]
        if missing:
            return "fail", f"pack directories not found: {', '.join(missing)}", data
        if s.allow_unauthenticated:
            warnings.append("allow_unauthenticated=true (dev only)")
        elif s.tokens_file is None and s.mtls_tenants_file is None:
            return "fail", "no tenant credentials: set PGW_SERVICE_TOKENS_FILE or PGW_SERVICE_MTLS_TENANTS_FILE", data
        if not s.tls.enabled:
            warnings.append("TLS disabled (terminate TLS in front of the service or set PGW_SERVICE_TLS__*)")
    elif role == "gateway":
        from pgw.gateway.config import GatewaySettings

        g = GatewaySettings()
        models = g.all_models()
        g.validate_policy(models)
        data = {"models": [m.name for m in models], "port": g.port, "maskstore_root": str(g.maskstore_root)}
        if not models:
            warnings.append("no models configured (PGW_GATEWAY_MODELS / PGW_GATEWAY_MODELS_FILE)")
        if g.admin_token is None:
            warnings.append("no admin token (PGW_GATEWAY_ADMIN_TOKEN or PGW_GATEWAY_ADMIN_TOKEN_FILE): /admin/* and "
                            "console sign-in disabled")
        vf = g.console_viewer_tokens_file
        if g.console_enabled and vf is not None and not Path(vf).is_file():
            return "fail", f"console viewer tokens file {vf} not found (gateway would not start)", data
        if g.api_keys_file is None and not g.allow_anonymous:
            warnings.append("no API key file (PGW_GATEWAY_API_KEYS_FILE)")
        if g.allow_dev_modes:
            warnings.append("allow_dev_modes=true (NOT a production configuration)")
        for m in models:
            if not Path(m.pack_path).is_dir():
                return "fail", f"model {m.name!r}: pack {m.pack_path} not found", data
    elif role == "factory":
        from pgw.factory.config import FactorySettings

        f = FactorySettings()  # pack_path comes from PGW_FACTORY_PACK_PATH
        data = {"pack_path": str(f.pack_path), "output": f.output, "device": f.device}
        if not Path(f.pack_path).is_dir():
            return "fail", f"pack {f.pack_path} not found", data
        if f.output == "export" and f.export_dir is None:
            return "fail", "output=export needs PGW_FACTORY_EXPORT_DIR", data
    elif role == "pack":
        return "ok", "no settings", {}
    else:  # pragma: no cover - validated by click
        raise ValueError(role)
    if warnings:
        return "warn", "; ".join(warnings), data
    return "ok", "settings valid", data


def _check_row_norm_sidecar(pack: Any) -> tuple[int, list[str]]:
    """Compare a weight-norm sidecar (older packs, see ``MaskedBackend.warm_weight_norms``) found in
    the pack directory or the default cache directory with norms recomputed from the weights.
    Returns ``(entries checked, linears whose value differs)``."""
    from pgw.engine.backends import _read_norm_sidecar
    from pgw.modelpack.loader import row_norm_sq_max

    found = _read_norm_sidecar(pack, None)
    if not found:
        return 0, []
    weights = {e.id: e.weight for e in pack.manifest.linears}
    bad = [key for key, v in sorted(found.items()) if row_norm_sq_max(pack.load_tensor(weights[key])) != v]
    return len(found), bad


def _check_pack(path: Path, ring: str | None, deep: bool = False) -> tuple[Status, str, dict[str, Any]]:
    from pgw.core.ring import RingParams
    from pgw.modelpack import ModelPack

    with ModelPack.open(path, verify="full") as pack:
        rings = pack.supported_rings()
        m_total = sum(s.out_features for s in pack.linear_specs())
        slot_bytes = {name: m_total * RingParams.from_name(name).k // 8 for name in rings}
        data: dict[str, Any] = {"name": pack.name, "fingerprint": pack.fingerprint, "arch": pack.arch.arch,
                                "layers": pack.num_layers, "supported_rings": rings, "mask_bytes_per_slot": slot_bytes}
        if ring is not None:
            pack.check_ring(RingParams.from_name(ring))
        if deep:
            # Install-time only (reads every weight): the capped-ring output bound relies on these values,
            # and the request path trusts them. Raises FingerprintMismatchError on a manifest mismatch.
            data["row_norms_checked"] = pack.verify_row_norms()
            n_side, bad = _check_row_norm_sidecar(pack)
            data["row_norm_sidecar_checked"] = n_side
            if bad:
                data["row_norm_sidecar_mismatch"] = bad
                return "fail", (f"weight-norm sidecar for fp {pack.short_fingerprint} disagrees with the weights "
                                f"({', '.join(bad[:5])}); delete it, it is recomputed at the next start"), data
        if not rings:
            return "fail", "no ring preset has enough headroom for this pack", data
        extra = "; row norms verified" if deep else ""
        return "ok", f"{pack.name} fp {pack.short_fingerprint} verified; rings {', '.join(rings)}{extra}", data


def _check_key_file(key_file: Path) -> tuple[Status, str, dict[str, Any]]:
    st = key_file.stat()
    mode = stat.S_IMODE(st.st_mode)
    data = {"mode": oct(mode), "size": st.st_size}
    if mode & 0o077:
        return "fail", f"KEK file {key_file} is accessible by group/others (mode {oct(mode)}; must be 0600)", data
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        return "warn", f"KEK file {key_file} is owned by uid {st.st_uid}, not the current user", data
    return "ok", f"KEK file permissions {oct(mode)}", data


def _check_maskstore(root: Path, key_file: Path | None) -> tuple[Status, str, dict[str, Any]]:
    from pgw.maskstore.admin import store_status
    from pgw.maskstore.config import MaskStoreSettings

    overrides: dict[str, Any] = {"root": root}
    if key_file is not None:
        overrides.update(key_file=key_file, key_provider="file")
    settings = MaskStoreSettings(**overrides)
    status = store_status(settings)
    usage = shutil.disk_usage(root)
    data = {"status": status, "disk_free": usage.free, "disk_total": usage.total}
    rb = status.get("rollback_protection") or {}
    if rb.get("counter", "none") == "none" and not settings.allow_no_external_counter:
        return "fail", (
            "no external monotonic counter (PGW_MASKSTORE_COUNTER): gateways refuse to hand out masks, since a "
            "whole-volume snapshot restore would go undetected and re-serve consumed masks"
        ), data
    dev = [
        name for name in (
            "allow_no_external_counter", "allow_unsigned_partner_segments", "allow_unbound_partner_segments",
            "allow_network_filesystem", "allow_no_rollback_witness",
        ) if getattr(settings, name)
    ]
    if dev:
        return "warn", "store opened and verified, but DEV-ONLY settings are enabled: " + ", ".join(dev), data
    if usage.free < MIN_FREE_DISK_BYTES:
        return "warn", f"store opened and verified, but only {usage.free / 2**30:.2f} GiB free", data
    return "ok", f"store opened and verified; {usage.free / 2**30:.1f} GiB free", data


def _measure_read_rate(directory: Path, nbytes: int, block: int = MASKSTORE_PROBE_BLOCK) -> dict[str, Any]:
    """Write ``nbytes`` of random data to a hidden probe file in ``directory``, drop it from the page cache
    and time reading it back in ``block``-sized reads in shuffled order. The file is always deleted.

    Random data, so a compressing or deduplicating filesystem cannot make the reads look faster. The
    store never lists the probe (it is a dot file at the store root, not under ``*/*/segments/``)."""
    import random
    import secrets

    nblocks = max(1, -(-int(nbytes) // block))
    path = Path(directory) / f".pgw-doctor-readprobe-{secrets.token_hex(6)}"
    cache_dropped = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            for _ in range(nblocks):
                os.write(fd, os.urandom(block))
            os.fsync(fd)
        finally:
            os.close(fd)
        fd = os.open(path, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                cache_dropped = True
            order = list(range(nblocks))
            random.Random(nblocks).shuffle(order)  # noqa: S311 - a fixed, non-sequential read order, not a secret
            buf = bytearray(block)
            preadv = getattr(os, "preadv", None)
            t0 = time.perf_counter()
            done = 0
            for i in order:
                if preadv is not None:
                    done += preadv(fd, [buf], i * block)
                else:  # pragma: no cover - platforms without preadv
                    done += len(os.pread(fd, block, i * block))
            seconds = time.perf_counter() - t0
        finally:
            os.close(fd)
    finally:
        path.unlink(missing_ok=True)
    return {"bytes": done, "seconds": seconds, "block_bytes": block, "cache_dropped": cache_dropped,
            "mb_per_s": done / seconds / 1e6 if seconds > 0 else float("inf")}


def _check_maskstore_read_rate(root: Path, bytes_per_slot: int | None, expected_tps: float | None,
                               probe_bytes: int = MASKSTORE_PROBE_BYTES) -> tuple[Status, str, dict[str, Any]]:
    """Read throughput of the mask-store volume against the mask bytes the expected load reads."""
    if not Path(root).is_dir():  # the ``maskstore`` check already reports it; no probe to write
        return "skip", f"store root missing ({root}); read rate not measured", {}
    free = shutil.disk_usage(root).free
    if free < 2 * probe_bytes + MIN_FREE_DISK_BYTES:
        return "warn", (f"not enough free space at {root} for a {probe_bytes / 2**20:.0f} MiB read probe; "
                        "read rate not measured"), {"disk_free": free}
    try:
        m = _measure_read_rate(root, probe_bytes)
    except OSError as exc:
        return "warn", f"cannot write a read probe next to the store ({type(exc).__name__}: {exc})", {}
    rate_mb = float(m["mb_per_s"])
    data: dict[str, Any] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                            "min_mb_per_s_without_load": MASKSTORE_MIN_READ_MB_S}
    parts = [f"{rate_mb:,.0f} MB/s ({m['bytes'] / 2**20:.0f} MiB in {m['block_bytes'] >> 10} KiB shuffled reads)"]
    notes: list[str] = []
    if not m["cache_dropped"]:
        notes.append("the page cache could not be dropped on this platform, so the rate may be optimistic")
    if bytes_per_slot:
        data["bytes_per_slot"] = int(bytes_per_slot)
        data["supported_tokens_per_second"] = round(rate_mb * 1e6 / bytes_per_slot, 1)
        parts.append(f"feeds {rate_mb * 1e6 / bytes_per_slot:,.1f} tokens/s at {bytes_per_slot / 1e6:.2f} MB/slot")
    exp = _expected_tps(expected_tps)
    if exp is not None and bytes_per_slot:
        need_mb = exp * bytes_per_slot / 1e6
        data.update(expected_tokens_per_second=exp, needed_mb_per_s=round(need_mb, 1))
        if rate_mb < need_mb:
            return "fail", "; ".join([*parts, f"below the {need_mb:,.0f} MB/s that {exp:g} tokens/s read: put the "
                                      "mask store on local NVMe or RAM (docs/operations/sizing.md, section 4)",
                                      *notes]), data
        if rate_mb < SUPPLY_MARGIN * need_mb:
            return "warn", "; ".join([*parts, f"within {round((SUPPLY_MARGIN - 1) * 100)}% of the {need_mb:,.0f} "
                                      f"MB/s that {exp:g} tokens/s read", *notes]), data
    elif exp is not None:
        notes.append("give --pack to compare with the expected load (mask bytes per slot)")
    elif rate_mb < MASKSTORE_MIN_READ_MB_S:
        return "warn", "; ".join([*parts, f"below {MASKSTORE_MIN_READ_MB_S:.0f} MB/s, about one chat-speed 7B stream: "
                                  "use local NVMe or RAM for the mask store (docs/operations/sizing.md, section 4)",
                                  *notes]), data
    return "ok", "; ".join([*parts, *notes]), data


def _check_service(url: str, token_file: Path | None, ca_file: Path | None, fingerprint: str | None,
                   ring: str | None) -> tuple[Status, str, dict[str, Any]]:
    from pydantic import SecretStr

    from pgw.wire.client import ClientConfig, ComputeClient

    token = None
    if token_file is not None:
        token = SecretStr(token_file.read_text(encoding="utf-8").strip())
    cfg = ClientConfig(base_url=url, token=token, ca_file=ca_file, http2=url.startswith("https://"),
                       max_retries=1, request_timeout_s=15.0)

    async def probe() -> tuple[bool, Any]:
        client = ComputeClient(cfg)
        try:
            ready = await client.ready()
            models = await client.models()
            return ready, models
        finally:
            await client.aclose()

    ready, models = asyncio.run(probe())
    served = [{"name": m.name, "fp": m.fingerprint, "rings": list(m.rings)} for m in models.models]
    data = {"ready": ready, "server": models.server, "versions": list(models.versions), "models": served}
    if not ready:
        return "fail", f"{url}/readyz is not ready", data
    if fingerprint is not None:
        match = [m for m in models.models if m.fingerprint == fingerprint]
        if not match:
            return "fail", f"service does not serve fingerprint {fingerprint[:12]}", data
        if ring is not None and ring.upper() not in match[0].rings:
            return "fail", f"service serves {fingerprint[:12]} but not ring {ring}", data
    return "ok", f"ready; serves {len(served)} model(s)", data


def _expected_tps(value: float | None) -> float | None:
    if value is not None:
        return float(value)
    raw = os.environ.get(EXPECTED_TPS_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; ignored", EXPECTED_TPS_ENV, raw)
        return None


def _factory_settings_for(pack_path: Path) -> Any:
    """``FactorySettings`` from ``PGW_FACTORY_*`` with the given pack (``None`` when they do not parse)."""
    from pgw.factory.config import FactorySettings

    try:
        return FactorySettings(pack_path=pack_path)
    except Exception as exc:  # invalid settings are reported by settings.factory
        log.debug("factory settings do not parse (%s); timing local generation", type(exc).__name__)
        return None


def _mask_supply_backend(fs: Any, pack: Any, ring: str) -> tuple[Any, str]:
    """``(remote backend or None, mode)``. Encrypted mode only when the factory uses it and its hint
    exists: building the hint is a one-time factory job, far too heavy for the doctor."""
    if fs is None or not (getattr(fs, "he_urls", None) or getattr(fs, "he_service_url", None)):
        return None, "local"
    from pgw.core.ring import RingParams
    from pgw.factory.runtime import build_he_backend
    from pgw.hemask.backend import HintStore
    from pgw.hemask.lwe import HEParams

    q_bits = HEParams(k=RingParams.from_name(ring).k, rank=fs.he_rank).q
    hint_root = Path(fs.he_hint_dir or Path(fs.state_dir or ".") / "hemask")
    if HintStore.open(hint_root, pack.fingerprint, fs.he_rank, q_bits) is None:
        return None, "encrypted (hint not built)"
    return build_he_backend(fs, pack, ring), "encrypted"


def _nearest_existing(path: Path) -> Path:
    """``path`` or its nearest existing ancestor (the volume a store about to be created would use)."""
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _check_mask_supply(pack_path: Path, ring: str | None, maskstore_root: Path | None,
                       expected_tps: float | None, slots: int | None = None) -> tuple[Status, str, dict[str, Any]]:
    """Time one mask chunk into a temporary store and compare the supply rate with the expected load."""
    import tempfile

    from pgw.factory.bench import _time_generation
    from pgw.factory.generator import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.config import MaskStoreSettings
    from pgw.modelpack import ModelPack

    fs = _factory_settings_for(pack_path)
    ring_name = ring or (fs.ring_names[0] if fs is not None else "K32A12C27")
    link = {"bytes": 0, "seconds": 0.0}
    with ModelPack.open(pack_path, verify="fingerprint", components=("linears",)) as pack:
        remote, mode = _mask_supply_backend(fs, pack, ring_name)
        if remote is None and mode != "local":
            return "warn", ("encrypted mask generation is configured but its one-time hint is not built yet; "
                            "start the factory once (it builds the hint), then re-run the doctor"), {"mode": mode}
        gen = MaskGenerator(pack, ring_name, device=fs.device if fs is not None else "auto",
                            backend=fs.backend if fs is not None else "auto", remote=remote)
        if remote is not None:
            inner = remote.remote

            def counted(linear: str, b: Any) -> Any:
                t0 = time.perf_counter()
                y = inner(linear, b)
                link["seconds"] += time.perf_counter() - t0
                link["bytes"] += int(getattr(b, "nbytes", 0)) + int(getattr(y, "nbytes", 0))
                return y

            remote.remote = counted
        # One chunk: 256 slots (one encrypted group) or 32 locally (keeps a CPU timing of 7B short).
        n = int(slots or (256 if remote is not None else 32))
        bps = int(gen.bytes_per_slot)
        tmp_root = Path(tempfile.gettempdir())
        need_tmp = n * bps + (64 << 20)
        if shutil.disk_usage(tmp_root).free < 2 * need_tmp:
            return "warn", (f"not enough free space in {tmp_root} for a {n}-slot test chunk "
                            f"({need_tmp / 1e9:.2f} GB); set TMPDIR to a larger volume"), {"mode": mode}
        with tempfile.TemporaryDirectory(prefix="pgw-doctor-") as tmp:
            # Never the production counter or free-space floor: a throw-away store in a temporary directory.
            # Explicit settings with the env and .env sources disabled, so no PGW_MASKSTORE_* override
            # (counter, KEK provider, network-filesystem policy, ...) reaches this store.
            tmp_settings = _settings_without_env(
                MaskStoreSettings,
                {"root": Path(tmp) / "store", "key_provider": "file", "key_file": Path(tmp) / "kek",
                 "counter": "none", "min_free_bytes": 0},
            )
            store = MaskStore(tmp_settings, FileKeyProvider(Path(tmp) / "kek", create=True))
            try:
                seconds, _gemm, _chk = _time_generation(gen, n, min(n, 256), 2, store, warmup=False)
            finally:
                store.close()
    rate = n / seconds if seconds > 0 else 0.0
    data: dict[str, Any] = {
        "mode": mode, "ring": ring_name, "slots_timed": n, "seconds": round(seconds, 3),
        "measured_slots_per_second": round(rate, 3), "bytes_per_slot": bps,
    }
    if link["seconds"] > 0:
        # The provider round trips alone bound the rate (pipelined: local work overlaps them).
        data["link_bytes"] = link["bytes"]
        data["link_mb_per_s"] = round(link["bytes"] / link["seconds"] / 1e6, 2)
        data["link_bound_slots_per_second"] = round(n / link["seconds"], 3)
        rate = min(rate, n / link["seconds"])
    data["slots_per_second"] = round(rate, 3)
    data["slots_per_hour"] = round(rate * 3600)
    problems: list[str] = []
    notes: list[str] = []
    seg = int(fs.segment_slots) if fs is not None else 4096
    first = int(fs.first_segment_slots or 0) if fs is not None else 0
    if rate > 0:
        data["seconds_per_segment"] = round(seg / rate, 1)
        if first:
            data["seconds_to_first_segment"] = round(min(first, seg) / rate, 1)
    exp = _expected_tps(expected_tps)
    if exp is not None:
        data["expected_tokens_per_second"] = exp
        if rate < exp:
            problems.append(f"supply {rate:.2f} slots/s is below the expected load {exp:.2f} tokens/s: the store "
                            "drains (add factory capacity or a faster link)")
        elif rate < SUPPLY_MARGIN * exp:
            notes.append(f"supply {rate:.2f} slots/s is within {round((SUPPLY_MARGIN - 1) * 100)}% of the "
                         f"expected load {exp:.2f} tokens/s")
    if maskstore_root is not None:
        try:
            ms = MaskStoreSettings(root=maskstore_root)
            target, min_free = int(ms.target_slots), int(ms.min_free_bytes)
        except Exception:  # invalid store settings are reported by the maskstore check
            target, min_free = 0, 0
        if target:
            total = shutil.disk_usage(_nearest_existing(maskstore_root)).total
            need = target * bps + min_free
            data.update(disk_total=total, disk_needed_for_target=need)
            if total < need:
                problems.append(f"the store volume ({total / 1e9:.1f} GB) cannot hold target_slots={target} x "
                                f"{bps / 1e6:.2f} MB + min_free_bytes ({need / 1e9:.1f} GB)")
    summary = f"{mode}: {rate:.2f} slots/s ({rate * 3600:,.0f}/h, {bps / 1e6:.2f} MB/slot, {ring_name})"
    if problems:
        return "fail", "; ".join([summary, *problems, *notes]), data
    if notes:
        return "warn", "; ".join([summary, *notes]), data
    return "ok", summary, data


def _cert_files_from_settings(role: str) -> list[Path]:
    """Certificate files named by the role's ``PGW_*`` settings (best effort; invalid settings give none)."""
    out: list[Any] = []
    if role in ("gateway", "all"):
        try:
            from pgw.gateway.config import GatewaySettings

            g = GatewaySettings()
            out += [g.tls.cert_file, g.tls.ca_file]
            for m in g.all_models():  # per model: CA of the compute service and the mTLS client certificate
                out += [m.service_ca_file, m.client_cert_file]
        except Exception:
            log.debug("gateway settings do not parse; no certificate files from them")
    if role in ("service", "all"):
        try:
            from pgw.service.config import ServiceSettings

            sv = ServiceSettings()
            out += [sv.tls.cert_file, sv.tls.ca_file, sv.client_issuing_ca_cert_file, sv.enrollment_ca_file]
        except Exception:
            log.debug("service settings do not parse; no certificate files from them")
    if role in ("factory", "all"):
        raw = os.environ.get("PGW_FACTORY_HE_CA_FILE", "").strip()
        if raw:
            out.append(Path(raw))
    return [Path(p) for p in out if p]


def _check_cert_files(paths: Iterable[Path], *, now: float | None = None,
                      warn_days: int = CERT_WARN_DAYS) -> tuple[Status, str, dict[str, Any]]:
    """Expiry of every certificate in the given PEM files (bundles included)."""
    import datetime as dt

    from cryptography import x509

    now_dt = dt.datetime.fromtimestamp(time.time() if now is None else now, tz=dt.UTC)
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p in seen:
            continue
        seen.add(p)
        try:
            certs = x509.load_pem_x509_certificates(p.read_bytes())
        except (OSError, ValueError) as exc:
            errors.append(f"{p.name} ({type(exc).__name__})")
            continue
        for c in certs:
            end = c.not_valid_after_utc
            rows.append({"file": str(p), "subject": c.subject.rfc4514_string()[:120], "not_after": end.isoformat(),
                         "days_left": round((end - now_dt).total_seconds() / 86400, 1)})
    data: dict[str, Any] = {"certificates": rows, "warn_days": warn_days}
    if errors:
        data["unreadable"] = errors
    if not rows and not errors:
        return "skip", "no certificate files configured", data
    expired = [r for r in rows if r["days_left"] <= 0]
    if expired:
        return "fail", "expired: " + ", ".join(f"{Path(r['file']).name} ({r['subject']})" for r in expired), data
    soon = [f"{Path(r['file']).name} expires in {r['days_left']:.1f} days" for r in rows if r["days_left"] < warn_days]
    if soon or errors:
        return "warn", "; ".join(soon + [f"unreadable: {e}" for e in errors]), data
    nxt = min(rows, key=lambda r: r["days_left"])
    detail = f"{len(rows)} certificate(s); next expiry in {nxt['days_left']:.0f} days ({Path(nxt['file']).name})"
    return "ok", detail, data


# ---- orchestration ------------------------------------------------------------------------------
def run_doctor(
    *,
    role: str = "all",
    pack: Path | None = None,
    ring: str | None = None,
    maskstore_root: Path | None = None,
    key_file: Path | None = None,
    service_url: str | None = None,
    service_token_file: Path | None = None,
    service_ca_file: Path | None = None,
    include_gpu: bool = True,
    check_settings: bool = False,
    strict: bool = False,
    mask_supply: bool = False,
    deep: bool = False,
    expected_tokens_per_second: float | None = None,
    cert_files: Iterable[Path] = (),
    maskstore_read_bytes: int | None = None,
) -> DoctorReport:
    """Run all checks relevant to ``role`` and the given options.

    Args:
        role: ``all`` (developer box: extras are warnings) or one deployment role
            (``gateway|service|factory|pack``: that role's extras are required).
        pack: Model pack to verify fully (and to match against the service).
        ring: Ring preset to check against the pack and the service.
        maskstore_root / key_file: Mask store to open read-only-ish (verify integrity, inventory).
        service_url / service_token_file / service_ca_file: Compute service to probe.
        include_gpu: Include ``cuda_int8`` in the arithmetic self test when available.
        check_settings: Also validate the role's ``PGW_*`` settings (implied for a specific role).
        strict: Treat warnings as failures.
        deep: With ``pack``: also recompute every linear's ``row_norm_sq_max`` from the int8 weights and
            compare it with the manifest and any weight-norm sidecar (reads every weight; install time).
        mask_supply: Time one mask chunk (opt-in, ``--mask-supply``; needs ``pack``). Off by default:
            it runs real mask generation, far too heavy for a gateway pod's start-up check.
        expected_tokens_per_second: Expected mask consumption for the supply check and the mask-store
            read-rate check (default from ``PGW_DOCTOR_EXPECTED_TOKENS_PER_S``; unset = report the rate only).
        maskstore_read_bytes: Size of the ``maskstore.read_rate`` probe (default
            :data:`MASKSTORE_PROBE_BYTES`; 0 skips the check). Runs whenever ``maskstore_root`` is given.
        cert_files: Extra PEM certificate files whose expiry is checked (plus those in the settings).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    report = DoctorReport(strict=strict)
    _run(report, "python", _check_python)
    _run(report, "packages", _check_packages)
    roles = [r for r in ROLE_EXTRAS if role in ("all", r)]
    _run(report, "extras", lambda: _check_extras(roles, required=role != "all"))
    _run(report, "entropy", _check_entropy)
    _run(report, "selftest", lambda: _check_selftest(include_gpu))
    _run(report, "native", _check_native)
    if role in ("all", "service", "factory"):
        _run(report, "cpu_int8", _check_cpu_int8)
    _run(report, "cuda", lambda: _check_cuda(role))
    if role in ("all", "gateway", "factory"):
        _run(report, "gateway_gpu", _check_gateway_gpu)
    _run(report, "memory", _check_memory)
    if role != "all" and (check_settings or role in ("gateway", "service", "factory")):
        _run(report, f"settings.{role}", lambda: _check_role_settings(role))
    fingerprint: str | None = None
    slot_bytes: int | None = None
    if pack is not None:
        c = _run(report, "pack", lambda: _check_pack(pack, ring, deep))
        fingerprint = c.data.get("fingerprint")
        per_ring = c.data.get("mask_bytes_per_slot") or {}
        rings = list(per_ring)
        pick = ring if ring in per_ring else ("K32A12C27" if "K32A12C27" in per_ring else (rings[0] if rings else None))
        slot_bytes = int(per_ring[pick]) if pick is not None else None
    if key_file is not None:
        _run(report, "maskstore.key_file", lambda: _check_key_file(key_file))
    if maskstore_root is not None:
        _run(report, "maskstore", lambda: _check_maskstore(maskstore_root, key_file))
        probe = MASKSTORE_PROBE_BYTES if maskstore_read_bytes is None else int(maskstore_read_bytes)
        if probe > 0:
            _run(report, "maskstore.read_rate", lambda: _check_maskstore_read_rate(
                maskstore_root, slot_bytes, expected_tokens_per_second, probe))
    if service_url is not None:
        _run(report, "service", lambda: _check_service(service_url, service_token_file, service_ca_file,
                                                        fingerprint, ring))
    if mask_supply and pack is not None:
        _run(report, "mask_supply", lambda: _check_mask_supply(pack, ring, maskstore_root,
                                                                expected_tokens_per_second))
    certs = [*cert_files, *([service_ca_file] if service_ca_file is not None else [])]
    if role != "pack":
        certs += _cert_files_from_settings(role)
    c = _run(report, "tls_certs", lambda: _check_cert_files(certs))
    if c.status == "skip":
        report.checks.remove(c)  # nothing configured: keep the report short
    return report


def format_report(report: DoctorReport) -> str:
    """Human-readable multi-line rendering."""
    tags = {"ok": " OK ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    width = max((len(c.name) for c in report.checks), default=10)
    lines = [f"[{tags[c.status]}] {c.name:<{width}}  {c.detail}" for c in report.checks]
    n = report.counts()
    verdict = "healthy" if report.ok else "PROBLEMS FOUND"
    lines.append(f"{verdict}: {n['ok']} ok, {n['warn']} warn, {n['fail']} fail, {n['skip']} skip")
    return "\n".join(lines)
