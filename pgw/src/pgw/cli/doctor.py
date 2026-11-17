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
* ``pack``: full SHA-256 verification of a model pack, ring headroom, mask bytes per slot.
* ``maskstore``: KEK file permissions, store integrity/inventory, free disk space.
* ``service``: the compute service answers ``/readyz`` and ``/pgw/v1/models`` and serves the
  pack's fingerprint and ring.
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
    if (v.major, v.minor) < (3, 12):
        return "fail", f"Python {data['version']} < 3.12", data
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
    msg = "; ".join(f"{r}: {', '.join(m)} (pip install 'pgw[{r}]')" for r, m in missing.items())
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
            warnings.append("no admin token (PGW_GATEWAY_ADMIN_TOKEN): /admin/* disabled")
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


def _check_pack(path: Path, ring: str | None) -> tuple[Status, str, dict[str, Any]]:
    from pgw.core.ring import RingParams
    from pgw.modelpack import ModelPack

    with ModelPack.open(path, verify="full") as pack:
        rings = pack.supported_rings()
        m_total = sum(s.out_features for s in pack.linear_specs())
        slot_bytes = {name: m_total * RingParams.from_name(name).k // 8 for name in rings}
        data = {"name": pack.name, "fingerprint": pack.fingerprint, "arch": pack.arch.arch,
                "layers": pack.num_layers, "supported_rings": rings, "mask_bytes_per_slot": slot_bytes}
        if ring is not None:
            pack.check_ring(RingParams.from_name(ring))
        if not rings:
            return "fail", "no ring preset has enough headroom for this pack", data
        return "ok", f"{pack.name} fp {pack.short_fingerprint} verified; rings {', '.join(rings)}", data


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

    overrides: dict[str, Any] = {"root": root}
    if key_file is not None:
        overrides.update(key_file=key_file, key_provider="file")
    status = store_status(**overrides)
    usage = shutil.disk_usage(root)
    data = {"status": status, "disk_free": usage.free, "disk_total": usage.total}
    if usage.free < MIN_FREE_DISK_BYTES:
        return "warn", f"store opened and verified, but only {usage.free / 2**30:.2f} GiB free", data
    return "ok", f"store opened and verified; {usage.free / 2**30:.1f} GiB free", data


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
    if role in ("all", "service", "factory"):
        _run(report, "cpu_int8", _check_cpu_int8)
    _run(report, "cuda", lambda: _check_cuda(role))
    _run(report, "memory", _check_memory)
    if role != "all" and (check_settings or role in ("gateway", "service", "factory")):
        _run(report, f"settings.{role}", lambda: _check_role_settings(role))
    fingerprint: str | None = None
    if pack is not None:
        c = _run(report, "pack", lambda: _check_pack(pack, ring))
        fingerprint = c.data.get("fingerprint")
    if key_file is not None:
        _run(report, "maskstore.key_file", lambda: _check_key_file(key_file))
    if maskstore_root is not None:
        _run(report, "maskstore", lambda: _check_maskstore(maskstore_root, key_file))
    if service_url is not None:
        _run(report, "service", lambda: _check_service(service_url, service_token_file, service_ca_file,
                                                        fingerprint, ring))
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
