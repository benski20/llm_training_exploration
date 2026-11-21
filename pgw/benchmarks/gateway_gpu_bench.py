"""Gateway-GPU benchmark: encrypted mask factory rate per engine, local GPU factory, local linears.

Modes (one per process, so the engine choice is clean):

* ``--mode he``: the real :class:`pgw.factory.MaskGenerator` with :class:`pgw.hemask.HEMaskBackend` on a
  synthetic pack of ``--layers`` layers of ``--preset`` (int8 weights shared across layers, hints
  hard-linked per shape). The provider is in process (``ComputeKernel.mulmod64`` on ``--provider-device``;
  on Kaggle the same T4, so it competes with the gateway work; Modal runs it on another GPU). The
  engine comes from the environment (``PGW_HEMASK_ENGINE``, ``PGW_HEMASK_DEVICE_PASSES``). Reports
  slots/s scaled to the full model and the per-stage seconds (encrypt, provider, decrypt, check).
* ``--mode local``: the plain local factory (``cuda_int8`` and the GPU PRNG when on CUDA).
* ``--mode linears``: per-token gateway time of the exact local qkv + o (one row, the gateway's
  quantize -> exact integer product -> range check -> dequantize), CPU vs CUDA, with a bit comparison.

    python benchmarks/gateway_gpu_bench.py --mode he --preset qwen2.5-0.5b --layers 4 --slots 4096
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgw.factory.bench import MODEL_PRESETS


class _Pack:
    def __init__(self, preset: str, layers: int, seed: int = 0):
        shape = MODEL_PRESETS[preset]
        self.name = f"{preset}-synthetic"
        self.fingerprint = f"{seed:064x}"
        self.specs = [s for i in range(layers) for s in shape.layer_specs(i)]
        gen = torch.Generator().manual_seed(seed)
        self.w: dict[tuple[int, int], torch.Tensor] = {}
        for s in shape.layer_specs(0):
            key = (s.out_features, s.in_features)
            self.w.setdefault(key, torch.randint(-127, 128, key, dtype=torch.int8, generator=gen))

    def linear_specs(self) -> list[Any]:
        return list(self.specs)

    def check_ring(self, params: Any) -> None:
        for s in self.specs:
            params.check_in_features(s.in_features)

    def load_linear(self, lid: Any, device: Any = "cpu") -> Any:
        from types import SimpleNamespace

        spec = next(s for s in self.specs if s.id.key == str(getattr(lid, "key", lid)))
        return SimpleNamespace(id=spec.id, w8=self.w[(spec.out_features, spec.in_features)].to(device))

    def iter_linears(self) -> Any:
        for s in self.specs:
            yield self.load_linear(s.id)


def _hints(root: Path, pack: _Pack, rank: int, q_bits: int) -> Any:
    """A hint store with one hint per weight shape, hard-linked for the other layers."""
    from pgw.hemask.backend import HintStore
    from pgw.hemask.fastmm import to_planes
    from pgw.hemask.lwe import HEParams, matmul_mod64, public_matrix

    params = HEParams(k=16, rank=rank, q_bits=q_bits)
    path = HintStore.dir_for(root, pack.fingerprint, rank, q_bits)
    path.mkdir(parents=True)
    seed = os.urandom(32)
    a = public_matrix(seed, max(s.in_features for s in pack.specs), params)
    first: dict[tuple[int, int], Path] = {}
    norms = {}
    t0 = time.perf_counter()
    for s in pack.specs:
        key = (s.out_features, s.in_features)
        w = pack.w[key].numpy()
        norms[s.id.key] = float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max())
        dst = path / f"{s.id.key}.npy"
        if key in first:
            os.link(first[key], dst)
            continue
        h = np.empty((w.shape[0], q_bits // 8 - 1, params.dim), dtype=np.int8)
        for c0 in range(0, params.dim, 512):
            h[:, :, c0:c0 + 512] = to_planes(matmul_mod64(w, a[:w.shape[1], c0:c0 + 512]))[:, 1:q_bits // 8]
        np.save(dst, h)
        first[key] = dst
    meta = {"version": 2, "fingerprint": pack.fingerprint, "rank": rank, "q_bits": q_bits, "a_seed": seed.hex(),
            "dim": params.dim, "row_norms": norms, "created": time.time()}
    (path / "meta.json").write_text(json.dumps(meta))
    print(f"hint build {time.perf_counter() - t0:.1f}s", flush=True)
    return HintStore(path, meta)


class _Provider:
    def __init__(self, pack: _Pack, device: str, nbytes: int):
        from pgw.service.kernels import ComputeKernel

        self.kern: dict[tuple[int, int], Any] = {}
        self.spec = {s.id.key: (s.out_features, s.in_features) for s in pack.specs}
        for s in pack.specs:
            key = (s.out_features, s.in_features)
            if key not in self.kern:
                self.kern[key] = ComputeKernel(s, pack.w[key], device=device)
        self.nbytes = nbytes
        self.lock = threading.Lock()
        self.seconds = 0.0

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        with self.lock:
            t0 = time.perf_counter()
            y: np.ndarray = self.kern[self.spec[linear]].mulmod64(b, self.nbytes)
            self.seconds += time.perf_counter() - t0
            return y


def bench_he(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core.prng import new_seed
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.hemask import fastmm
    from pgw.hemask.backend import HEMaskBackend
    from pgw.hemask.lwe import HEParams

    ring = RingParams.from_name(args.ring)
    p = HEParams(k=ring.k, rank=args.rank)
    pack = _Pack(args.preset, args.layers)
    with tempfile.TemporaryDirectory() as tmp:
        hints = _hints(Path(tmp), pack, args.rank, p.q)
        prov = _Provider(pack, args.provider_device, p.b_bytes)
        t0 = time.perf_counter()
        be = HEMaskBackend(pack, ring, hints, prov, groups_per_call=args.groups_per_call,
                           max_inflight=args.inflight)
        t_init = time.perf_counter() - t0
        gen = MaskGenerator(pack, ring, device=args.device, remote=be)
        chunk = gen.chunk_slots_for(args.chunk_slots)
        t0 = time.perf_counter()
        for _ in gen.generate(new_seed(), args.warmup_slots, checks=2):  # uploads hints, compiles
            pass
        t_warm = time.perf_counter() - t0
        prov.seconds = 0.0
        he0 = dict(be.stats)
        t0 = time.perf_counter()
        for _ in gen.generate(new_seed(), args.slots, checks=2, chunk_slots=chunk):
            pass
        wall = time.perf_counter() - t0
        st = gen.last_stats
        he = {k: round(be.stats[k] - he0[k], 3) for k in he0 if isinstance(he0[k], int | float)}
        scale = MODEL_PRESETS[args.preset].layers / args.layers
        gateway_s = wall - prov.seconds  # the provider shares the device here; Modal runs it elsewhere
        dev_stats = dict(be._dev.stats) if be._dev is not None else None
        return {
            "mode": "he", "engine": be.engine, "fastmm_engine": fastmm.engine(), "layers": args.layers,
            "slots": args.slots, "chunk_slots": chunk, "init_s": round(t_init, 2), "warmup_s": round(t_warm, 2),
            "wall_s": round(wall, 3), "provider_s": round(prov.seconds, 3),
            "slots_per_s_full_model": round(args.slots / wall / scale, 2),
            "slots_per_s_full_model_gateway_only": round(args.slots / max(gateway_s, 1e-9) / scale, 2),
            "stages": {"rng": round(st.rng_seconds, 3), "gemm": round(st.gemm_seconds, 3),
                       "check": round(st.check_seconds, 3)},
            "he": he, "device_stats": dev_stats,
        }


def bench_local(args: argparse.Namespace) -> dict[str, Any]:
    from dataclasses import asdict

    from pgw.factory.bench import benchmark_preset

    res = benchmark_preset(args.preset, ring=args.ring, device=args.device, slots=args.slots, chunk_slots=None,
                           layers=args.layers)
    return {"mode": "local", **{k: v for k, v in asdict(res).items() if not isinstance(v, dict | list)}}


def bench_linears(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core.ring import RingParams, dequantize_outputs, quantize_activations
    from pgw.engine.backends import _row_norm_max, exact_int_matmul

    ring = RingParams.from_name(args.ring)
    shape = MODEL_PRESETS[args.preset]
    specs = [s for s in shape.layer_specs(0) if s.id.kind.value in ("qkv", "o")]
    gen = torch.Generator().manual_seed(1)
    ws = {s.id.kind.value: torch.randint(-127, 128, (s.out_features, s.in_features), dtype=torch.int8, generator=gen)
          for s in specs}
    scales = {k: torch.rand(w.shape[0], generator=gen) * 1e-3 for k, w in ws.items()}
    xs = {k: torch.randn(args.rows, w.shape[1], generator=gen) for k, w in ws.items()}
    out: dict[str, Any] = {"mode": "linears", "rows": args.rows, "layers": shape.layers}

    def run(dev: str) -> tuple[float, dict[str, torch.Tensor]]:
        w8t = {k: (w.contiguous().to(dev).t() if dev == "cuda" else w.t().contiguous()) for k, w in ws.items()}
        norms = {k: _row_norm_max(w) for k, w in ws.items()}
        res: dict[str, torch.Tensor] = {}

        def one(k: str) -> torch.Tensor:
            xq, sx = quantize_activations(xs[k], ring, norms[k] if ring.cap else None)
            yq = exact_int_matmul(xq.to(dev), w8t[k], ring.a).cpu()
            bound = int(ring.output_bound(ws[k].shape[1]))
            assert int(yq.abs().max()) <= bound
            return dequantize_outputs(yq, sx, scales[k], None, torch.float32)

        for k in ws:
            for _ in range(5):
                res[k] = one(k)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            for _layer in range(shape.layers):
                for k in ws:
                    one(k)
            times.append(time.perf_counter() - t0)
        return sorted(times)[len(times) // 2], res

    for threads in (1, max(1, torch.get_num_threads())):
        torch.set_num_threads(threads)
        t, ref = run("cpu")
        out[f"cpu_{threads}t_ms_per_token"] = round(1e3 * t, 3)
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        t, got = run("cuda")
        out["cuda_ms_per_token"] = round(1e3 * t, 3)
        out["bit_identical"] = all(torch.equal(ref[k], got[k]) for k in ref)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("he", "local", "linears"), default="he")
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(MODEL_PRESETS))
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=4096)
    ap.add_argument("--warmup-slots", type=int, default=256)
    ap.add_argument("--chunk-slots", type=int, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--groups-per-call", type=int, default=4)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--provider-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fn = {"he": bench_he, "local": bench_local, "linears": bench_linears}[args.mode]
    res = fn(args)
    res["env"] = {k: v for k, v in os.environ.items() if k.startswith("PGW_HEMASK")}
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    res["cpus"] = os.cpu_count()
    print("RESULT " + json.dumps(res), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
