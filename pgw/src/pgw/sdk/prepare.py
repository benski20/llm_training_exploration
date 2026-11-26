"""``pgw sdk prepare``: a model's one-time client setup ahead of the first request.

What the first request of a model otherwise waits for, done now, with progress, on every core, resumable:

1. enrollment of the API key (as the client does on first use);
2. the gateway's weight-free pack and the factory pack (the int8 weights, streamed from the provider and checked
   against the pinned fingerprint's manifest, SHA-256 per tensor);
3. with ``mask_scheme="pks-v3"`` and ``hint="local"`` (the defaults), the pks-v3 compact hint, built from the same
   weights while each linear is in memory (one pass over the weights; native kernels on every core:
   :mod:`pgw.hemask.hint_gemm`).

An interrupted run continues where it stopped: the factory pack keeps a partial directory with the objects written
(the final full verification hashes every byte against the manifest), the hint keeps ``<store>.partial`` whose state
(seeds, per-linear SHA-256 and row norm) is authenticated with HMAC-SHA256 under the cache's hint key
(:class:`pgw.hemask.pks_backend.HintBuild`). The store becomes usable only when complete. A client started later
with the same cache picks the prepared hint up (:func:`pgw.provision.client.find_prepared_hint`) instead of building
it in the background factory.

**Run setup on a big box once, copy the cache.** The hint depends only on the public weights and two public seeds the
customer draws, so a large host the customer controls can prepare a model and the cache can be copied to the
application hosts: ``<cache>/models/<fingerprint[:16]>/`` (``pack``, ``factory-pack``, ``hints``) and
``<cache>/keys/pks-hint.key``. On the target the client opens the store only if ``meta.json`` verifies under that key
(HMAC-SHA256 over the seeds, the row norms that drive the noise refusal and every hint file's SHA-256), and the
factory re-hashes every hint file against those digests when it opens the store; the packs are verified against the
pinned fingerprint as always. The hint key authenticates, it does not encrypt (nothing in the cache needs secrecy
from the provider: the seeds are public); copy it over a channel you trust and keep it 0600, like the cache. Without
the key on the target, a copied store is ignored and the factory builds its own hint.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError

__all__ = ["prepare"]


def prepare(model: str, *, api_key: str | None = None, options: Any = None, provisioner: Any = None,
            progress: Callable[[str], None] | None = None, enroll_transport: Any = None,
            build_hint: bool | None = None) -> dict[str, Any]:
    """Prepare ``model`` under the client's cache (see the module docstring); returns a summary (paths, sizes,
    seconds; no secrets).

    Args:
        model: A supported model name (``qwen2.5-7b-instruct``).
        api_key: The SDK key (default ``PGW_API_KEY``).
        options: :class:`pgw.sdk.options.SDKOptions` (default: from the environment).
        provisioner: A :class:`pgw.sdk.provisioning.ProviderProvisioner` (default: the options' one).
        progress: Called with short progress lines (stage and counts; no secrets).
        build_hint: Build the pks-v3 hint now (default: when the options use pks-v3 with ``hint="local"``).
    """
    from pgw.sdk.engine import Engine
    from pgw.sdk.keys import parse_key
    from pgw.sdk.options import SDKOptions, canonical_model
    from pgw.sdk.provisioning import ProviderProvisioner

    name = canonical_model(model)
    opts = options if options is not None else SDKOptions.from_env()
    key = parse_key(api_key if api_key is not None else os.environ.get("PGW_API_KEY"))
    prov = provisioner if provisioner is not None else ProviderProvisioner.from_options(opts)
    if not hasattr(prov, "prepare"):
        raise ConfigError("this provisioner has no prepare step (pgw sdk prepare needs the provider provisioner)")
    if build_hint is None:
        build_hint = opts.mask_scheme == "pks-v3" and getattr(prov, "hint", "local") == "local"
    eng = Engine(key, opts, provisioner=prov, enroll_transport=enroll_transport)
    t0 = time.perf_counter()

    async def run() -> tuple[Any, bool]:
        session = await eng.ensure_session()
        entry = session.model_entry(name)
        if entry is None:
            session = await eng.ensure_session(force=True)
            entry = session.model_entry(name)
        if entry is None:
            raise ConfigError(f"the provider does not serve model {name!r} to this API key")
        eng.layout.ensure()
        masked = await eng._decide_masked_head(session, str(entry["fingerprint"]).lower())
        res = await asyncio.to_thread(prov.prepare, name, session=session, cache_dir=eng.layout.models_dir,
                                      progress=progress, masked_head=masked, build_hint=bool(build_hint))
        return res, masked

    try:
        res, masked = eng.call(run())
    finally:
        eng.close()
    out: dict[str, Any] = {"model": name, "fingerprint": res.fingerprint, "masked_head": masked,
                           "pack": str(res.pack_path), "factory_pack": str(res.factory_pack_path or ""),
                           "hint_dir": str(res.hint_dir) if res.hint_dir else None, "source": res.source,
                           "seconds": round(time.perf_counter() - t0, 1)}
    for k in ("pack", "factory_pack", "hint_dir"):
        p = out[k]
        if p:
            out[k + "_bytes"] = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    return out
