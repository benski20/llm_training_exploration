"""Cold-start time of a GPU pod, by component (estimates unless stated; the trial in RUNBOOK section 3.9 measures them).

Sequence (:mod:`pgw.placement.registry`): RunPod schedules the pod and starts the container -> the network volume
is mounted -> the pre-staged pack is verified against its pinned fingerprint (every byte hashed) -> the service
starts and loads it to the GPU -> device self-tests and CUDA-graph warm-up -> ``/readyz`` -> registration (CSR,
leaf certificate, the controller's TLS + health probe) -> the next signed directory -> clients pick it up at their
next refresh. The pod logs each step's duration (``boot_timings`` in its registration) so these figures get replaced
by measurements.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from pgw.placement.packs import PACK_GB

__all__ = ["estimate", "measure_hash_mb_s"]

#: Assumptions (MB/s and seconds; RunPod does not publish them; replace with the trial's measurements).
DEFAULTS: dict[str, float] = {
    "schedule_s": 15.0,  # API create -> host chosen, container created (Secure Cloud, GPU in stock)
    "image_gb": 9.0,  # provider image (CUDA runtime, torch, pgw)
    "pull_mb_s": 120.0,  # registry -> host when the image is not cached on that host
    "mount_s": 5.0,  # network volume attach
    "volume_mb_s": 500.0,  # network volume sequential read (first read; the second comes from the page cache)
    "hash_mb_s": 1500.0,  # SHA-256 per core x hashing threads (4), bounded by the volume read
    "hash_threads": 4.0,
    "load_mb_s": 3000.0,  # page cache -> host tensors -> GPU (pinned copies)
    "service_init_s": 20.0,  # interpreter, torch and CUDA initialisation, accounts, TLS listeners
    "selftest_s": 30.0,  # bit-exact device self-tests (NTT / pks kernels), CUDA graph capture, warm-up
    "register_s": 2.0,  # CSR, leaf, controller's TLS + readiness probe, directory signature
    "hf_mb_s": 150.0,  # Hugging Face download (no pre-staged pack)
    "pack_s_per_gb": 25.0,  # shard-wise packing, CPU (no pre-staged pack), per GB of pack written
}
CHECKPOINT_GB = {"qwen2.5-0.5b-instruct": 1.0, "qwen2.5-7b-instruct": 15.2, "qwen2.5-32b-instruct": 65.5,
                 "qwen2.5-72b-instruct": 145.4}


def measure_hash_mb_s(mb: int = 256) -> float:
    """SHA-256 throughput of one core here (MB/s), to calibrate ``hash_mb_s`` / ``hash_threads``."""
    buf = os.urandom(1 << 20)
    h = hashlib.sha256()
    t0 = time.perf_counter()
    for _ in range(mb):
        h.update(buf)
    return mb / (time.perf_counter() - t0)


def estimate(model: str, *, prestaged: bool = True, image_cached: bool = False, refresh_s: float = 300.0,
             **over: float) -> dict[str, Any]:
    """Seconds per component of one cold start, the total to ``/readyz`` and registration, and the expected time
    until clients use the pod (``+ refresh_s / 2``: clients refresh the directory every ``refresh_s``)."""
    a = {**DEFAULTS, **over}
    gb = PACK_GB[model]
    mb = gb * 1000.0
    c: dict[str, float] = {"schedule": a["schedule_s"],
                           "image_pull": 0.0 if image_cached else a["image_gb"] * 1000.0 / a["pull_mb_s"],
                           "mount": a["mount_s"] if prestaged else 0.0}
    if not prestaged:
        c["download"] = CHECKPOINT_GB.get(model, gb * 1.75) * 1000.0 / a["hf_mb_s"]
        c["pack"] = gb * a["pack_s_per_gb"]
    c["verify_pack"] = mb / min(a["volume_mb_s"] if prestaged else 2000.0, a["hash_mb_s"] * a["hash_threads"])
    c["load"] = mb / a["load_mb_s"]
    c["service_init"] = a["service_init_s"]
    c["selftests"] = a["selftest_s"]
    c["register"] = a["register_s"]
    ready = sum(c.values())
    return {"model": model, "prestaged": prestaged, "image_cached": image_cached, "components_s": c,
            "ready_s": ready, "in_use_s": ready + refresh_s / 2.0, "assumptions": a}
