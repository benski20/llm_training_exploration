"""GPU mask service: the plain ``W8 r`` factory on a customer-owned GPU (opt-in, ``PGW_FACTORY_MASK_SERVICE``).

Standard deployment for 32B+ models (``deploy/customer-cloud``): the customer runs, in its own cloud account and
in the provider's region, a CPU-only gateway VM and a small GPU VM running this mask service. The mask service
computes the one-time masks ``W8 r mod 2^K`` directly from the customer's own copy of the **public** model pack
(no encryption, no hint): the provider never receives ``r``, the segment seeds or the check vectors, and the
weights it would need to know are its own already.

This module adds no new mask arithmetic. Generation is :class:`pgw.factory.MaskGenerator` (``cuda_int8`` digit
GEMMs, :mod:`pgw.factory.gpu_prng` ChaCha20 on the device, per-chunk Freivalds output self-check), the segment
format is SPEC 7.2 (ring or ``p31`` check vectors), delivery is a local :class:`~pgw.factory.sinks.StoreSink`
(gateway on the same host) or the partner export channel (HPKE-sealed to the gateway store's recipient key,
Ed25519-signed, bound to the destination store; :class:`~pgw.factory.sinks.ExportSink`, pulled by the gateway
with :mod:`pgw.factory.courier`). What this module adds, as guards in front of that path:

* :func:`startup_check`: before the first segment, a throw-away segment (fresh random seed, never stored) is
  generated on the production device with the production chunk size, and sampled rows of ``W8 r`` and the full
  ``u W8`` of sampled linears are compared **bit for bit** with an independent CPU reference (Python/OpenSSL
  ChaCha20 for ``r`` and ``u``, exact integer products in numpy). Any difference refuses start-up
  (:class:`pgw.core.IntegrityError`); the per-chunk Freivalds self-check then guards every production chunk.
* :func:`scrub_device_memory`: after every segment (and after the start-up check) the device memory the torch
  caching allocator holds but no tensor uses is overwritten with zeros before it is released, so ``r`` digit
  planes, products or ``W r`` of a finished segment do not linger in freed GPU blocks. The generator already
  zeroes its own temporaries (digit planes, int32 products, device ``W r`` after the copy back, host ``r``).
* :func:`check_mask_service_settings`: refuses configurations that would weaken the deployment (encrypted
  generation mixed in, no output self-check, a CPU device unless explicitly allowed, partner export without a
  signing key or destination store).
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError
from pgw.core.prng import SeedExpander, new_seed

__all__ = [
    "MaskServiceCheck", "check_mask_service_settings", "prepare_mask_service", "scrub_device_memory",
    "startup_check",
]

log = logging.getLogger(__name__)

#: Rows of ``W`` per block of the exact CPU reference (bounds the int64 copy of the weight: 512 x n x 8 bytes).
_REF_BLOCK_ROWS = 512


@dataclass
class MaskServiceCheck:
    """Outcome of :func:`startup_check` (no secret material)."""

    ok: bool
    fingerprint: str
    ring: str
    check_field: str
    device: str
    backend: str
    gpu_prng: bool
    slots: int
    chunk_slots: int
    checks: int
    linears_checked: list[str] = field(default_factory=list)
    rows_per_linear: int = 0
    wr_elements_compared: int = 0
    uw_elements_compared: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    generate_seconds: float = 0.0
    reference_seconds: float = 0.0
    scrubbed_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (for ``pgw factory mask-service --check-only --json``)."""
        return dict(self.__dict__)


# ---- exact CPU reference (independent of the GPU kernels, torch._int_mm and pgw_fast) ---------------------------
def _w_as_u64(w8: torch.Tensor, r0: int, r1: int) -> np.ndarray:
    """Rows ``[r0, r1)`` of the int8 weight as uint64 two's complement (wraps mod 2^64, exact mod 2^K)."""
    return np.ascontiguousarray(w8[r0:r1].numpy().astype(np.int64)).view(np.uint64)


def reference_wr_rows(r_rows: torch.Tensor, w8: torch.Tensor, k: int) -> np.ndarray:
    """``r_rows @ W8^T mod 2^K`` (uint64 ``(t, m)``) with numpy integer arithmetic mod 2^64 (no BLAS, no floats)."""
    r = np.ascontiguousarray(r_rows.numpy()).view(np.uint64)
    m = int(w8.shape[0])
    out = np.empty((r.shape[0], m), dtype=np.uint64)
    for r0 in range(0, m, _REF_BLOCK_ROWS):
        r1 = min(m, r0 + _REF_BLOCK_ROWS)
        out[:, r0:r1] = r @ _w_as_u64(w8, r0, r1).T
    out &= np.uint64((1 << k) - 1)
    r.fill(0)
    return out


def reference_uw(u: torch.Tensor, w8: torch.Tensor, k: int, prime: int | None) -> np.ndarray:
    """``u @ W8`` mod ``2^K`` (ring format) or mod ``p`` (prime format) as uint64 ``(c, n)``, exactly.

    Prime format: ``u < 2^31`` and ``|W8| <= 128``, so every sum over ``m < 2^24`` rows fits int64 exactly."""
    m = int(w8.shape[0])
    c, n = int(u.shape[0]), int(w8.shape[1])
    if prime is None:
        uu = np.ascontiguousarray(u.numpy()).view(np.uint64)
        acc = np.zeros((c, n), dtype=np.uint64)
        for r0 in range(0, m, _REF_BLOCK_ROWS):
            r1 = min(m, r0 + _REF_BLOCK_ROWS)
            acc += uu[:, r0:r1] @ _w_as_u64(w8, r0, r1)
        acc &= np.uint64((1 << k) - 1)
        return acc
    if m >= 1 << 24:
        raise ConfigError("out_features too large for the exact prime-field reference")
    ui = np.ascontiguousarray(u.numpy()).astype(np.int64)
    acc_i = np.zeros((c, n), dtype=np.int64)
    for r0 in range(0, m, _REF_BLOCK_ROWS):
        r1 = min(m, r0 + _REF_BLOCK_ROWS)
        acc_i += ui[:, r0:r1] @ w8[r0:r1].numpy().astype(np.int64)
    ui.fill(0)
    return np.mod(acc_i, prime).astype(np.uint64)


def _pick_linears(specs: list[Any], limit: int) -> list[int]:
    """Indices of the linears the start-up check compares: all (``limit <= 0``), else the first and the last
    layer's linears (every kind) plus evenly spaced ones, ``limit`` in total (at least the first and last)."""
    n = len(specs)
    if limit <= 0 or limit >= n:
        return list(range(n))
    head = list(range(min(4, n)))
    tail = list(range(max(0, n - 4), n))
    chosen: list[int] = []
    for i in head[: max(1, limit // 2)] + tail[-max(1, limit - limit // 2):]:
        if i not in chosen:
            chosen.append(i)
    for x in np.linspace(0, n - 1, num=limit):
        if len(chosen) >= limit:
            break
        if round(x) not in chosen:
            chosen.append(round(x))
    return sorted(chosen[:limit])


def _pick_rows(slots: int, chunk: int, rows: int, rng: np.random.Generator) -> list[int]:
    """Sample rows: first, last, both sides of the first chunk boundary, then random ones."""
    want = [0, slots - 1, chunk - 1, chunk]
    out = sorted({x for x in want if 0 <= x < slots})[: max(rows, 2)]
    pool = [x for x in range(slots) if x not in out]
    extra = max(0, rows - len(out))
    if extra and pool:
        out += [int(x) for x in rng.choice(pool, size=min(extra, len(pool)), replace=False)]
    return sorted(out)


def startup_check(gen: Any, *, rows: int = 4, linears: int = 8, extra_slots: int = 3,
                  seed: bytes | None = None, chunk_slots: int | None = None) -> MaskServiceCheck:
    """Generate a throw-away segment on the generator's device and compare it bit for bit with a CPU reference.

    The segment has ``chunk_slots_for(chunk_slots) + extra_slots`` slots (one full production chunk and a partial
    one, so padding and chunk boundaries are exercised), the generator's check count and check format. For the linears
    chosen by ``linears`` (``0`` = every linear) the sampled rows of ``W8 r`` and the whole ``u W8`` are compared
    with :func:`reference_wr_rows` / :func:`reference_uw` on ``r`` and ``u`` re-derived by the Python expander.
    The seed is fresh (``os.urandom``) unless a test passes one; every buffer is zeroed afterwards and the device
    is scrubbed. Never writes a segment.

    Raises:
        IntegrityError: any element differs (the report is in ``details``); the mask service must not start.
        ConfigError: the generator uses encrypted generation (the mask service computes locally).
    """
    if getattr(gen, "remote", None) is not None:
        raise ConfigError("the GPU mask service computes W8 r locally; it cannot run with encrypted generation")
    if rows < 2:
        raise ConfigError("startup check needs at least 2 rows per linear")
    from pgw.core.freivalds import P31, prime_check_vectors

    chunk = int(gen.chunk_slots_for(chunk_slots))  # the production GEMM batch (settings.chunk_slots or auto)
    slots = max(2, min(chunk + max(0, int(extra_slots)), int(gen.max_segment_slots())))
    checks = int(gen.checks_for())
    prime = P31 if gen.check_field is not None else None
    k = gen.ring.k
    specs = list(gen.specs)
    picked = {specs[i].id.key: specs[i] for i in _pick_linears(specs, int(linears))}
    sample = _pick_rows(slots, min(chunk, slots), int(rows), np.random.default_rng())
    seed_b = bytearray(seed if seed is not None else new_seed())
    rep = MaskServiceCheck(ok=False, fingerprint=gen.fingerprint, ring=gen.ring.name,
                           check_field=gen.check_field or "ring", device=str(gen.device), backend=gen.backend,
                           gpu_prng=False, slots=slots, chunk_slots=chunk, checks=checks,
                           linears_checked=list(picked), rows_per_linear=len(sample))
    got_wr: dict[str, dict[int, torch.Tensor]] = {key: {} for key in picked}
    got_uw: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    try:
        it = gen.generate(bytes(seed_b), slots, checks=checks, chunk_slots=chunk)
        try:
            for item in it:
                try:
                    key = item.id.key
                    if key in picked:
                        if item.uw is not None:
                            got_uw[key] = item.uw.detach().to("cpu").clone()
                        lo, hi = item.slot_start, item.slot_start + int(item.wr.shape[0])
                        for s in sample:
                            if lo <= s < hi:
                                got_wr[key][s] = item.wr[s - lo].detach().to("cpu").clone()
                finally:
                    item.wr.zero_()
        finally:
            it.close()
        rep.gpu_prng = bool(gen.last_stats.extra.get("gpu_prng"))
        rep.generate_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        exp = SeedExpander(bytes(seed_b))
        for key, spec in picked.items():
            w8 = gen.pack.load_linear(spec.id, device="cpu").w8
            if w8.device.type != "cpu":  # pragma: no cover - packs load on the CPU
                w8 = w8.cpu()
            r_rows = torch.cat([exp.ring_uniform_rows(spec.in_features, k, s, 1, "r", key) for s in sample])
            try:
                ref = reference_wr_rows(r_rows, w8, k)
            finally:
                r_rows.zero_()
            mine = torch.stack([got_wr[key][s] for s in sample]).numpy().view(np.uint64) \
                if len(got_wr[key]) == len(sample) else None
            if mine is None:
                rep.mismatches.append({"linear": key, "what": "wr", "error": "rows missing from the output"})
            else:
                bad = int((mine != ref).any(axis=1).sum())
                rep.wr_elements_compared += int(ref.size)
                if bad:
                    rep.mismatches.append({"linear": key, "what": "wr", "bad_rows": bad, "rows": len(sample)})
            ref.fill(0)
            if checks:
                if prime is not None:
                    u = prime_check_vectors(bytes(seed_b), checks, spec.out_features, key)
                else:
                    u = exp.ring_uniform((checks, spec.out_features), k, "u", key)
                try:
                    ref_u = reference_uw(u, w8, k, prime)
                finally:
                    u.zero_()
                have = got_uw.get(key)
                if have is None or tuple(have.shape) != ref_u.shape:
                    rep.mismatches.append({"linear": key, "what": "uw", "error": "check vectors missing"})
                else:
                    bad_u = int((have.numpy().view(np.uint64) != ref_u).sum())
                    rep.uw_elements_compared += int(ref_u.size)
                    if bad_u:
                        rep.mismatches.append({"linear": key, "what": "uw", "bad_elements": bad_u})
        rep.reference_seconds = time.perf_counter() - t1
    finally:
        for d in got_wr.values():
            for t in d.values():
                t.zero_()
        for t in got_uw.values():
            t.zero_()
        seed_b[:] = bytes(len(seed_b))
        rep.scrubbed_bytes = scrub_device_memory(gen.device)
    rep.ok = not rep.mismatches
    if not rep.ok:
        log.error("mask service start-up check FAILED on %s/%s: %s", rep.device, rep.backend, rep.mismatches)
        raise IntegrityError(
            "GPU mask service start-up check: device masks differ from the CPU reference; refusing to start",
            details=rep.to_dict(),
        )
    log.info("mask service start-up check passed: %s %s on %s/%s (gpu prng %s), %d linears x %d rows of W r and "
             "%d u W elements bit-identical to the CPU reference (%.1fs device, %.1fs reference)",
             gen.pack.name if hasattr(gen.pack, "name") else "?", rep.ring, rep.device, rep.backend, rep.gpu_prng,
             len(picked), len(sample), rep.uw_elements_compared, rep.generate_seconds, rep.reference_seconds)
    return rep


# ---- device scrubbing -------------------------------------------------------------------------------------------
def scrub_device_memory(device: Any, *, block_bytes: int = 64 << 20, _cuda: Any = None,
                        _alloc: Any = None) -> int:
    """Overwrite the GPU memory torch's caching allocator holds but no tensor uses, then release it.

    Synchronizes the device, allocates zero-filled blocks up to the reserved-but-unallocated amount (the
    allocator hands out its cached free blocks first), frees them and empties the cache. Returns the bytes
    zeroed. Best effort (a fragmented cache may leave small blocks; allocation failures stop the loop); a CPU
    device returns 0. The generator itself zeroes every temporary it allocates, so this only covers blocks
    freed by library code (e.g. GEMM workspaces) between those wipes.
    """
    cuda = _cuda if _cuda is not None else torch.cuda
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    try:
        cuda.synchronize(dev)
        gap = int(cuda.memory_reserved(dev)) - int(cuda.memory_allocated(dev))
    except Exception as exc:  # pragma: no cover - driver dependent
        log.warning("device scrub skipped on %s: %r", dev, exc)
        return 0
    blocks: list[Any] = []
    done = 0
    size = max(1 << 20, int(block_bytes))
    try:
        while done < gap:
            n = min(size, gap - done)
            try:
                blocks.append(_alloc(n) if _alloc is not None else torch.zeros(n, dtype=torch.uint8, device=dev))
            except Exception:  # out of memory with a fragmented cache: smaller blocks, then give up
                if size <= 1 << 20:
                    break
                size //= 2
                continue
            done += n
        with contextlib.suppress(Exception):
            cuda.synchronize(dev)
    finally:
        blocks.clear()
        with contextlib.suppress(Exception):
            cuda.empty_cache()
    return done


# ---- settings and wiring ----------------------------------------------------------------------------------------
def check_mask_service_settings(settings: Any) -> None:
    """Refuse configurations that do not fit the customer-cloud GPU mask service (``mask_service = true``)."""
    if settings.he_urls:
        raise ConfigError("mask_service computes W8 r on the customer's GPU; unset he_service_url(s)")
    if settings.self_check_rows == 0:
        raise ConfigError("mask_service needs the per-chunk output self-check (self_check_rows != 0)")
    if settings.output == "export":
        if settings.partner_signing_key_file is None or not settings.destination_store_id:
            raise ConfigError(
                "mask_service export to a gateway on another VM needs partner_signing_key_file and "
                "destination_store_id (signed, store-bound segments; the gateway trusts that key)")
        if settings.recipient_public_key_file is None:
            raise ConfigError("mask_service export needs the gateway store's recipient_public_key_file")


def prepare_mask_service(settings: Any, gens: dict[Any, Any]) -> dict[Any, MaskServiceCheck]:
    """Validate the device, turn on per-segment scrubbing and run :func:`startup_check` for every generator.

    Raises before any segment is produced: :class:`ConfigError` (no CUDA device and ``mask_service_allow_cpu``
    unset) or :class:`IntegrityError` (start-up check mismatch)."""
    check_mask_service_settings(settings)
    reports: dict[Any, MaskServiceCheck] = {}
    for key, gen in gens.items():
        if gen.device.type != "cuda" and not settings.mask_service_allow_cpu:
            raise ConfigError(
                f"mask_service found no GPU (device {gen.device}); set mask_service_allow_cpu=true to run the "
                "same path on the CPU (slow; small models only) or use the CPU factory")
        gen.scrub_device = True
        if getattr(settings, "schedule", "standard") == "fused2":  # pgw.factory.fused2_gen: its own reference check
            t0 = time.perf_counter()
            rep = gen.startup_check(rows=settings.mask_service_check_rows,
                                    linears=settings.mask_service_check_linears)
            reports[key] = gen.mask_service_check = MaskServiceCheck(
                ok=bool(rep["ok"]), fingerprint=gen.fingerprint, ring=gen.ring.name, check_field="p61",
                device=str(gen.device), backend=str(gen.backend), gpu_prng=False, slots=0, chunk_slots=gen.chunk_rows,
                checks=1, linears_checked=list(rep["linears_checked"]),
                rows_per_linear=settings.mask_service_check_rows,
                wr_elements_compared=int(rep["wr_elements_compared"]),
                uw_elements_compared=int(rep["uw_elements_compared"]), generate_seconds=time.perf_counter() - t0)
            continue
        reports[key] = gen.mask_service_check = startup_check(
            gen, rows=settings.mask_service_check_rows, linears=settings.mask_service_check_linears,
            chunk_slots=settings.chunk_slots)
    return reports
