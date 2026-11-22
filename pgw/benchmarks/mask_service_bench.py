"""GPU mask service: start-up exactness check and slots/s on synthetic layers of a public architecture.

For ``--preset`` (public shapes, ``pgw.factory.bench.MODEL_PRESETS``) and ``--layers`` synthetic layers with random
int8 weights, on ``--device``:

1. runs :func:`pgw.factory.mask_service.startup_check` over **every** linear (``linears=0``) in both check formats
   (ring and p31): device output vs the independent CPU reference, bit for bit;
2. times generation of ``--slots`` slots (generation only, then into a temporary encrypted mask store) and
   extrapolates to the full model (slots/s scales with ``layers / model layers``).

    python benchmarks/mask_service_bench.py --preset qwen2.5-32b --layers 2 --device cuda --slots 2048 --json out.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="qwen2.5-7b")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--chunk-slots", type=int, default=None)
    ap.add_argument("--check-rows", type=int, default=4)
    ap.add_argument("--no-store", action="store_true", help="Skip the timed run into an encrypted store")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from pgw.core.ring import RingParams
    from pgw.factory.bench import MODEL_PRESETS, _SyntheticPack
    from pgw.factory.generator import MaskGenerator, format_stats, slot_bytes
    from pgw.factory.mask_service import startup_check
    from pgw.factory.sinks import StoreSink

    shape = MODEL_PRESETS[a.preset]
    pack = _SyntheticPack(shape, a.layers)
    full_specs = [s for i in range(shape.layers) for s in shape.layer_specs(i)]
    out: dict[str, Any] = {"preset": a.preset, "layers": a.layers, "model_layers": shape.layers, "ring": a.ring,
                           "slot_mb_full_model": slot_bytes(full_specs, RingParams.from_name(a.ring)) / 1e6,
                           "torch": torch.__version__, "checks": {}, "rates": {}}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    for field in ("ring", "p31"):
        gen = MaskGenerator(pack, a.ring, device=a.device, backend=a.backend,
                            check_field=None if field == "ring" else field)
        gen.scrub_device = True
        t0 = time.perf_counter()
        rep = startup_check(gen, rows=a.check_rows, linears=0)
        d = rep.to_dict()
        d["wall_s"] = round(time.perf_counter() - t0, 2)
        d.pop("fingerprint", None)
        out["checks"][field] = d
        print(f"[{field}] start-up check OK on {rep.device}/{rep.backend} gpu_prng={rep.gpu_prng}: "
              f"{len(rep.linears_checked)} linears x {rep.rows_per_linear} rows, {rep.wr_elements_compared:,} W r + "
              f"{rep.uw_elements_compared:,} u W elements bit-identical ({d['wall_s']}s)", flush=True)
        # generation only (warm: the check above compiled kernels)
        if gen.device.type == "cuda":
            torch.cuda.synchronize(gen.device)
        t0 = time.perf_counter()
        for item in gen.generate(bytes(32 * [7]), a.slots, checks=gen.checks_for(), chunk_slots=a.chunk_slots):
            item.wr.zero_()
        if gen.device.type == "cuda":
            torch.cuda.synchronize(gen.device)
        dt = time.perf_counter() - t0
        r = {"generate_slots_per_s_measured": a.slots / dt, "generate_slots_per_s_full_model":
             a.slots / dt * a.layers / shape.layers, "generate_stats": format_stats(gen.last_stats)}
        if not a.no_store:
            from pgw.maskstore import FileKeyProvider, MaskStore

            with tempfile.TemporaryDirectory(prefix="pgw-ms-bench-") as tmp:
                st = MaskStore.open(Path(tmp) / "s", FileKeyProvider(Path(tmp) / "kek", create=True))
                try:
                    t0 = time.perf_counter()
                    gen.generate_into_store(StoreSink(st), a.slots, chunk_slots=a.chunk_slots)
                    dt = time.perf_counter() - t0
                finally:
                    st.close()
            r["store_slots_per_s_measured"] = a.slots / dt
            r["store_slots_per_s_full_model"] = a.slots / dt * a.layers / shape.layers
            r["store_stats"] = format_stats(gen.last_stats)
        out["rates"][field] = r
        print(f"[{field}] {json.dumps(r)}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
