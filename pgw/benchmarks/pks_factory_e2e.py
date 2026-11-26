"""End-to-end client CPU per mask slot of the pks-v3 factory: the real ``MaskGenerator`` with the real
``PKSMaskBackend`` and ``RemotePKS`` writing real segments into a ``MaskStore``, at the shapes of ONE decoder layer of
``--model`` (synthetic int8 weights and a random compact hint of the right shapes), with a stand-in provider.

Where ``benchmarks/pks_slot_breakdown.py`` times each part with the function production calls, this runs them as
production chains them (rng thread, encryption into the request body, the reply decode, decryption, every-row
self-check, segment writer thread, p31 check vectors per segment, buffer reuse), so Python overhead, threads and
allocation are all counted. The provider is not: the stand-in's own CPU (building a reply of random ``y`` with the
request's id, about one memcpy of the reply) is measured on its thread and subtracted. The replies being random, the
every-row self-check is computed in full and its verdict ignored.

Output: process CPU seconds per slot for the layer, and that times the layer count (the masked LM head is not in
the layer; ``pks_slot_breakdown.py`` has it, about 1-2 % of the total from 7B up), i.e. the factory's CPU per slot
of the whole model's decoder layers, and slots/s per client core.

    PYTHONPATH=src taskset -c 3 python benchmarks/pks_factory_e2e.py --model 7b --slots 1024 --segments 3
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

LAYERS = {"0.5b": 24, "7b": 28, "32b": 64, "72b": 80}


def _cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


class _Pack:
    """One decoder layer with real linear ids and random int8 weights (what MaskGenerator and the backend read)."""

    fingerprint = "e" * 64
    short_fingerprint = "e" * 12

    def __init__(self, model: str):
        import torch
        from benchmarks.pks_slot_breakdown import SHAPES

        from pgw.core.types import LinearId, LinearSpec

        _, shapes = SHAPES[model]
        rng = np.random.default_rng(0)
        self.specs = [LinearSpec(LinearId(0, k), m, n) for k, (m, n) in shapes.items() if k != "lm_head"]
        self.w = {s.id.key: torch.from_numpy(rng.integers(-127, 128, (s.out_features, s.in_features), dtype=np.int8))
                  for s in self.specs}

    def linear_specs(self) -> list[Any]:
        return list(self.specs)

    def load_linear(self, lid: Any, device: Any = "cpu", **kw: Any) -> Any:
        return SimpleNamespace(w8=self.w[lid.key])

    def check_ring(self, params: Any) -> None:
        pass


class _Hints:
    """A compact-hint store stand-in: random hints of the right shapes, the meta the backend reads."""

    def __init__(self, pack: _Pack):
        from pgw.core.prng import new_seed
        from pgw.hemask.pks import NR, PACK

        rng = np.random.default_rng(1)
        self.a_seed, self.ks_seed = new_seed(), new_seed()
        self._h = {}
        for s in pack.specs:
            h = rng.integers(0, 256, (s.out_features // PACK, 2, NR, 3), dtype=np.uint8)
            h[..., 2] &= 0x7F
            self._h[s.id.key] = h
        self.meta = {"row_norms": {s.id.key: 3000.0 for s in pack.specs}}
        self.nbytes = sum(h.nbytes for h in self._h.values())

    def hint(self, linear: str) -> np.ndarray:
        return self._h[linear]


class _Provider:
    """``post(body, rid, decode)``: a v3 reply of random ``y`` of the right size for the request in ``body``; its CPU
    time (the provider's, not the client's) is accumulated in ``self.cpu``."""

    def __init__(self, out: dict[str, int]):
        self.out = out
        self.cpu = 0.0
        # the largest reply: the widest linear, 16 groups (the backend's cap per request)
        self.y = np.random.default_rng(2).integers(0, 256, max(out.values()) * 16 * 256 * 36 // 8,
                                                    dtype=np.uint8).tobytes()

    def __call__(self, body: bytes, rid: str, decode: Any) -> Any:
        import msgpack

        from pgw.hemask.wire import MaskgenResponse, MaskgenResult, encode_maskgen_response_parts

        t0 = time.thread_time()
        up = msgpack.Unpacker(raw=False, max_bin_len=1 << 31)
        up.feed(body[:512])  # the header: id, items[0] linear and cols come before the payloads
        up.read_map_header()
        head: dict[str, Any] = {}
        while "items" not in head:
            k = up.unpack()
            if k == "items":
                up.read_array_header()
                up.read_map_header()
                head["items"] = {up.unpack(): up.unpack(), up.unpack(): up.unpack()}
            else:
                head[k] = up.unpack()
        it = head["items"]
        nbytes = self.out[it["linear"]] * it["cols"] * 36 // 8
        y = memoryview(self.y)[:nbytes]
        data = b"".join(encode_maskgen_response_parts(MaskgenResponse(
            rid, (MaskgenResult(it["linear"], it["cols"], y),), 3, (1, 2, 3))))
        self.cpu += time.thread_time() - t0
        return decode(data)


def run(model: str, slots: int, segments: int, tmp: Path) -> dict[str, Any]:
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.factory import generator as gen_mod

    # the stand-in replies are random, so the every-row self-check (fully computed, as in production) would reject
    # them: its flags are computed and then ignored (benchmark only)
    check = gen_mod.check_flags
    gen_mod.check_flags = lambda *a: np.zeros_like(check(*a))  # type: ignore[assignment]
    from pgw.hemask.pks_backend import PKSMaskBackend, RemotePKS
    from pgw.maskstore import FileKeyProvider, MaskStore

    pack = _Pack(model)
    hints = _Hints(pack)
    out = {s.id.key: s.out_features for s in pack.specs}
    prov = _Provider(out)
    remote = RemotePKS(prov, pack.fingerprint, out, hints.a_seed)
    ring = RingParams.from_name("K32A12C27")
    be = PKSMaskBackend(pack, ring, hints, remote, groups_per_call=4, threads=1)  # type: ignore[arg-type]
    gen = MaskGenerator(pack, ring, device="cpu", remote=be, check_field="p31", self_check_rows=-1)
    store = MaskStore.open(tmp / "store", FileKeyProvider(tmp / "kek", create=True))
    try:
        gen.generate_into_store(store, slots)  # warm-up: private self-check vectors, pools, first-touch memory
        cpu0, prov0, w0 = _cpu(), prov.cpu, time.perf_counter()
        for _ in range(segments):
            gen.generate_into_store(store, slots)
        cpu = _cpu() - cpu0 - (prov.cpu - prov0)
        wall = time.perf_counter() - w0
        stats = dict(gen.last_stats.extra)
    finally:
        store.close()
    per_layer = cpu / (segments * slots)
    layers = LAYERS[model]
    return {"model": model, "slots": slots, "segments": segments, "layer_cpu_ms_per_slot": round(per_layer * 1e3, 3),
            "layers_cpu_ms_per_slot": round(per_layer * layers * 1e3, 2),
            "slots_per_s_per_core": round(1.0 / (per_layer * layers), 2), "wall_s": round(wall, 2),
            "provider_cpu_s": round(prov.cpu - prov0, 2), "r_reused": stats.get("r_reused"),
            "wire_fused": getattr(PKSMaskBackend, "wire_fused", False), "threads": threading.active_count()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="7b", choices=sorted(LAYERS))
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--prefault-gb", type=float, default=6.0, help="as in pks_slot_breakdown.py")
    a = ap.parse_args()
    os.environ.setdefault("PGW_PKS_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    if a.prefault_gb > 0:
        warm = np.empty(int(a.prefault_gb * (1 << 30)), dtype=np.uint8)
        warm[::4096] = 1
        del warm
    with tempfile.TemporaryDirectory(dir=a.tmpdir) as td:
        print(json.dumps(run(a.model, a.slots, a.segments, Path(td))))


if __name__ == "__main__":
    main()
