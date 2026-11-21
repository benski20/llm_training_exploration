"""Which parts of the customer side run on a gateway GPU (``pgw doctor`` and ``/admin/status``).

A gateway GPU is used only where the result is exact integer arithmetic that a start-up self-check
found identical to the CPU's, so outputs stay the same bits:

* **Encrypted mask factory** (the ``pgw factory run`` process the gateway supervises): encryption and
  decryption products as fused device passes (:mod:`pgw.hemask.devmm`; ``PGW_HEMASK_ENGINE``,
  ``PGW_HEMASK_DEVICE_PASSES``, ``PGW_HEMASK_GPU_HINT_MB``). The provider sees the same ciphertexts.
* **Local mask factory** (no encryption): ``cuda_int8`` ring GEMMs and the GPU mask PRNG
  (``PGW_FACTORY_DEVICE``, ``PGW_FACTORY_GPU_PRNG``).
* **Local linears** (``local_linears_device``): the exact integer product of qkv/o.

The decode float path (norms, attention, rope, the LM head) stays on the CPU: it defines the reference
bits, and GPU float kernels do not reproduce them.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["gpu_report"]


def gpu_report(runtimes: dict[str, Any] | None = None, *, run_self_test: bool = False) -> dict[str, Any]:
    """Device inventory and where each customer-side part runs (public configuration only)."""
    import torch

    out: dict[str, Any] = {"cuda": bool(torch.cuda.is_available()), "torch": torch.__version__,
                           "torch_cuda": torch.version.cuda}
    if out["cuda"]:
        out["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    from pgw.hemask import devmm, fastmm

    engine = fastmm.engine()
    passes = engine == "cuda" and os.environ.get("PGW_HEMASK_DEVICE_PASSES", "1").strip().lower() not in (
        "0", "false", "no", "off")
    if run_self_test and passes:
        from pgw.hemask.lwe import HEParams

        passes = devmm.device_for(HEParams(k=32, rank=16)) is not None
        engine = fastmm.engine()
    out["encrypted_mask_factory"] = {
        "engine": engine, "device_passes": passes, "self_tests": devmm.status(),
        "cuda_disabled": fastmm.cuda_disabled(),
        "hint_cache_mb": os.environ.get("PGW_HEMASK_GPU_HINT_MB", "auto"),
    }
    out["local_mask_factory"] = {"device": os.environ.get("PGW_FACTORY_DEVICE", "auto"),
                                 "gpu_prng": os.environ.get("PGW_FACTORY_GPU_PRNG", "auto")}
    models: dict[str, Any] = {}
    for name, rt in (runtimes or {}).items():
        cfg = getattr(rt, "config", None)
        policy = dict(getattr(getattr(rt, "backend", None), "local_policy", {}) or {})
        draft = getattr(rt, "draft_model", None)
        models[name] = {
            "decode_floats": str(getattr(cfg, "device", "cpu")),
            "local_linears": policy.get("device", "cpu") if policy.get("kinds") else None,
            "draft": (str(getattr(cfg, "draft_device", None) or getattr(cfg, "device", "cpu"))
                      if draft is not None else None),
        }
    out["models"] = models
    on_gpu = []
    if passes:
        on_gpu.append("encrypted factory products")
    if any(str(m.get("local_linears") or "").startswith("cuda") for m in models.values()):
        on_gpu.append("local linears")
    if any(str(m.get("draft") or "").startswith("cuda") for m in models.values()):
        on_gpu.append("draft model")
    out["on_gpu"] = on_gpu
    return out
