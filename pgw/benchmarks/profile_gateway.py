"""Profile the gateway side of masked inference at real model widths.

Builds a synthetic Qwen2 checkpoint with the hidden/intermediate sizes of a real model but only a
few layers, packs it, fills a real encrypted MaskStore with the real MaskGenerator, and runs decode
steps (T rows per linear) through ``MaskedBackend`` + an in-process ``LocalComputeClient``. Service
compute time is measured separately and subtracted, so the numbers are gateway CPU only.

Usage:
    python benchmarks/profile_gateway.py --preset 7b --layers 2 --steps 8 [--rows 1] [--profile]

Prints per-linear-group and per-token gateway CPU (extrapolated to the full layer count) and,
with ``--profile``, the top functions by cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PRESETS: dict[str, dict[str, int]] = {
    # hidden, intermediate, heads, kv heads, full layer count
    "0.5b": {"hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14,
             "num_key_value_heads": 2, "full_layers": 24},
    "7b": {"hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
           "num_key_value_heads": 4, "full_layers": 28},
    "8b": {"hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
           "num_key_value_heads": 8, "full_layers": 32},
}


def build_checkpoint(out: Path, preset: dict[str, int], layers: int) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    sys_path_tests = Path(__file__).resolve().parents[1] / "tests"
    import sys

    sys.path.insert(0, str(sys_path_tests.parent))
    from tests.conftest import _save_byte_tokenizer  # type: ignore[import-not-found]

    cfg = Qwen2Config(
        vocab_size=260, hidden_size=preset["hidden_size"], intermediate_size=preset["intermediate_size"],
        num_hidden_layers=layers, num_attention_heads=preset["num_attention_heads"],
        num_key_value_heads=preset["num_key_value_heads"], max_position_embeddings=4096,
        tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.eval().save_pretrained(out, safe_serialization=True)
    _save_byte_tokenizer(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="7b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1, help="rows per call (1 = decode, >1 = prefill/batch)")
    ap.add_argument("--ring", default="K40A12")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--noise", action="store_true", help="enable the outbound-noise monitor (gateway default)")
    args = ap.parse_args()

    from pgw.core import RingParams
    from pgw.core.types import LinearId
    from pgw.engine import LocalComputeClient
    from pgw.engine.backends import LinearCall, MaskedBackend
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.modelpack import ModelPack, pack_checkpoint

    preset = PRESETS[args.preset]
    ring = RingParams.from_name(args.ring)
    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-prof-"))
    work.mkdir(parents=True, exist_ok=True)
    ck, pk = work / "ckpt", work / "pack"
    t0 = time.perf_counter()
    if not (pk / "manifest.json").exists():
        if not (ck / "config.json").exists():
            build_checkpoint(ck, preset, args.layers)
        pack_checkpoint(ck, pk, name=f"synthetic-{args.preset}", overwrite=True)
    pack = ModelPack.open(pk)
    print(f"pack ready in {time.perf_counter() - t0:.1f}s")

    slots = args.steps * args.rows + 4
    counter = work / "counter"
    kp = FileKeyProvider(work / "kek", create=True)
    store_kw: dict[str, Any] = {"fsync": False}
    try:
        from pgw.maskstore.counters import FileMonotonicCounter

        store_kw["counter"] = FileMonotonicCounter(counter, fsync=False)
    except Exception:  # noqa: BLE001 - fall back to whatever the store's dev default is
        pass
    store = MaskStore.open(work / f"store-{time.time_ns()}", kp, **store_kw)
    t0 = time.perf_counter()
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="prof")
    print(f"factory: {slots} slots in {time.perf_counter() - t0:.1f}s")

    class TimedClient(LocalComputeClient):
        service_s = 0.0

        def compute_sync(self, request: Any) -> Any:
            t = time.perf_counter()
            try:
                return super().compute_sync(request)
            finally:
                TimedClient.service_s += time.perf_counter() - t

    client = TimedClient(pack)
    noise = None
    if args.noise:
        from pgw.observability.noise import OutboundNoiseMonitor

        noise = OutboundNoiseMonitor(ring)
    be = MaskedBackend(pack, None, client, ring, pack.fingerprint, min_soundness_bits=20, noise_monitor=noise)
    res = store.reserve(pack.fingerprint, ring, slots)
    a = pack.arch
    groups = ["qkv", "o", "gate_up", "down"]
    in_dim = {sp.id.kind.value: sp.in_features
              for sp in pack.linear_specs() if sp.id.layer == 0}

    # warm up (weight caches, chunk caches)
    for g in groups:
        be.forward(LinearCall(LinearId(0, g), torch.randn(args.rows, in_dim[g]), session=res))
    per_group = {g: 0.0 for g in groups}
    prof = cProfile.Profile() if args.profile else None
    TimedClient.service_s = 0.0
    total0 = time.perf_counter()
    for _ in range(args.steps - 1):
        for layer in range(args.layers):
            for g in groups:
                x = torch.randn(args.rows, in_dim[g])
                s0 = TimedClient.service_s
                t = time.perf_counter()
                if prof:
                    prof.enable()
                be.forward(LinearCall(LinearId(layer, g), x, session=res))
                if prof:
                    prof.disable()
                per_group[g] += (time.perf_counter() - t) - (TimedClient.service_s - s0)
    wall = time.perf_counter() - total0
    n = (args.steps - 1) * args.layers
    scale = preset["full_layers"] / 1
    out = {
        "preset": args.preset, "ring": args.ring, "rows": args.rows,
        "gateway_ms_per_call": {g: 1000 * v / n for g, v in per_group.items()},
        "gateway_ms_per_layer": 1000 * sum(per_group.values()) / n,
        "gateway_s_per_step_full_model": sum(per_group.values()) / n * scale,
        "service_s_per_step_full_model": TimedClient.service_s / n * scale,
        "wall_s": wall,
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(35)
        print(s.getvalue())
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(25)
        print(s.getvalue())
    res.close()
    store.close()


if __name__ == "__main__":
    main()
