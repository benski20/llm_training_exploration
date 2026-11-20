#!/usr/bin/env python3
"""Cost of one speculative verify pass with R rows, and of each extra row (chat-speed plan WP-A0).

Exact speculation pays off only if an extra verify row is cheap. The plan's gate for training an
EAGLE drafter (WP-A8) is ``c <= 45 ms`` per extra row at 7B on Kaggle after A1-A5. A verify pass of
one sequence with ``R`` rows (the real row + R-1 drafts) costs, per component:

* ``layers_ms``: gateway plaintext float work of all layers for the R rows, exactly as the engine
  verifies (``SequenceInput(last_logits=R, serial_tail=R-1)``: draft rows attend one at a time so
  they are bit-identical to plain decoding), over a stub linear backend (no masking, no service);
* ``head_ms``: final norm + exact LM head for the R rows (one shared pass with the native kernel);
* ``sample_ms``: the sampler on each of the R rows (greedy + logprobs, as the Kaggle validation);
* ``masking_ms`` (``--masking``): gateway mask/unmask of the 4 x layers calls at R rows, from
  ``profile_gateway.py --sessions 1 --rows R --null-service`` (builds a synthetic checkpoint);
* ``service_ms`` (``--service-url``): service time of one pass at R rows from
  ``python -m pgw.service.loadtest --sessions 1 --rows R`` against a running service (on Kaggle, the
  T4: rows >= 2 leave the fused 1-row Triton GEMV today).

Layer-proportional components are measured on ``--layers`` layers and scaled to the preset's full
layer count. Reported: ``pass_ms[R]``, ``c_ms[R]`` = (pass(R) - pass(1)) / (R - 1), and the projected
ms per token at draft acceptance rates alpha for a chain of R-1 drafts in strict fixed-R mode, where
every pass costs pass(R): ``(pass(R) + trips_ms) / E``, ``E = (1 - alpha^R) / (1 - alpha)`` tokens per
pass, against ``pass(1) + trips_ms`` without speculation. ``--trips-ms`` adds the per-pass round-trip
time that does not grow with rows (default 0: just the gateway terms measured here).

Usage::

    PYTHONPATH=src python benchmarks/spec_cost_bench.py [--preset 7b] [--layers 4] [--rows 1,2,4,8]
        [--context 48] [--steps 5] [--masking] [--service-url URL --token-file F] [--trips-ms 163]
        [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
GATE_C_MS = 45.0  # WP-A8 training gate (Kaggle, 7B, after A1-A5)


def load_bench(name: str) -> Any:
    """Import a sibling benchmark script as a module."""
    spec = importlib.util.spec_from_file_location(f"pgw_bench_{name}", HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _median_ms(fn: Any, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def gateway_float(args: argparse.Namespace, rows_list: list[int]) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """Layers, head and sampling ms per verify pass for each R (layer part scaled to the full model)."""
    pt = load_bench("profile_token")
    import torch

    from pgw.engine import model as model_mod
    from pgw.engine.model import SequenceInput, TransformerModel
    from pgw.engine.sampling import Sampler, SamplingParams

    if args.threads:
        torch.set_num_threads(args.threads)
    preset = dict(pt.PRESETS[args.preset])
    full_layers = preset["num_layers"]
    if args.vocab:
        preset["vocab_size"] = args.vocab
    pack = pt.SynthPack(preset, args.layers or None)
    a = pack.arch
    scale = full_layers / a.num_layers
    model = TransformerModel(pack, pt.make_backends(a, 1)[0])
    cache = model.new_cache(args.context + max(rows_list) + 8)
    model.forward([(17 * i + 3) % a.vocab_size for i in range(args.context)], cache)
    params = SamplingParams(max_tokens=16, temperature=0.0, logprobs=2, ignore_eos=True)

    def verify(rows: int) -> list[torch.Tensor]:
        cache.truncate(args.context)
        ids = [(11 + 7 * i) % a.vocab_size for i in range(rows)]
        (rl,) = model.forward_batch([SequenceInput(ids, cache, last_logits=rows, serial_tail=rows - 1)])
        return [rl[i] for i in range(rows)]  # every row's logits, as a fully accepted pass reads them

    def no_head(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.zeros((h.shape[0], a.vocab_size))

    out: dict[int, dict[str, float]] = {}
    for rows in rows_list:
        for _ in range(2):
            verify(rows)
        total = _median_ms(lambda r=rows: verify(r), args.steps)
        orig = model_mod.lm_head_logits
        model_mod.lm_head_logits = no_head
        try:
            layers = _median_ms(lambda r=rows: verify(r), args.steps)
        finally:
            model_mod.lm_head_logits = orig
        logits = verify(rows)
        samplers = [Sampler(params, [1, 2, 3]) for _ in range(rows)]

        def sample_all(lg: list[torch.Tensor] = logits, ss: list[Any] = samplers) -> None:
            for s, x in zip(ss, lg, strict=True):
                s.sample(x, [1, 2, 3])

        sample = _median_ms(sample_all, max(5, args.steps))
        out[rows] = {"layers_ms": round(layers * scale, 3), "head_ms": round(total - layers, 3),
                     "sample_ms": round(sample, 3)}
        print(json.dumps({"rows": rows, **out[rows]}), flush=True)
    info = {"layers_measured": a.num_layers, "layers_full": full_layers, "vocab": a.vocab_size,
            "lm_head_dtype": str(model._head.dtype), "batch_lm_head": bool(model.batch_lm_head),
            "torch_threads": torch.get_num_threads(), "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY")}
    model.close()
    return out, info


def masking(args: argparse.Namespace, rows: int) -> float:
    """Gateway masking ms per pass at ``rows`` (profile_gateway, null service, scaled to the full model)."""
    with_json = Path(args.workdir) / f"mask_r{rows}.json"
    cmd = [sys.executable, str(HERE / "profile_gateway.py"), "--preset", args.preset, "--layers", "1",
           "--steps", str(max(3, args.steps)), "--rows", str(rows), "--sessions", "1", "--null-service",
           "--ring", args.ring, "--workdir", str(Path(args.workdir) / "pg"), "--json", str(with_json)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=7200)  # noqa: S603
    res = json.loads(with_json.read_text())
    return float(res["gateway_s_per_step_full_model"]) * 1e3


def service(args: argparse.Namespace, rows: int) -> float:
    """Service ms per pass (all layers) at ``rows`` for one session, from the loadtest."""
    cmd = [sys.executable, "-m", "pgw.service.loadtest", "--url", args.service_url, "--ring", args.ring,
           "--sessions", "1", "--tokens", str(max(2, args.steps)), "--rows", str(rows), "--timeout", "600"]
    if args.token_file:
        cmd += ["--token-file", args.token_file]
    if args.fingerprint:
        cmd += ["--fingerprint", args.fingerprint]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)  # noqa: S603
    res = json.loads(r.stdout[r.stdout.index("{"):])
    return rows / float(res["tokens_per_s"]) * 1e3  # tokens_per_s counts token-rows


def projection(passes: dict[int, float], trips_ms: float) -> dict[str, Any]:
    base = passes[1] + trips_ms
    out: dict[str, Any] = {"no_spec_ms_per_token": round(base, 2)}
    for rows, p in passes.items():
        if rows == 1:
            continue
        row: dict[str, Any] = {}
        for alpha in ALPHAS:
            e = (1 - alpha**rows) / (1 - alpha)
            row[f"alpha_{alpha}"] = {"ms_per_token": round((p + trips_ms) / e, 2),
                                     "speedup": round(base * e / (p + trips_ms), 3)}
        out[f"R{rows}"] = row
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows_list = sorted({1, *(int(r) for r in str(args.rows).split(",") if r)})
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    comp, info = gateway_float(args, rows_list)
    for rows in rows_list:
        if args.masking:
            comp[rows]["masking_ms"] = round(masking(args, rows), 3)
        if args.service_url:
            comp[rows]["service_ms"] = round(service(args, rows), 3)
    passes = {r: sum(v.values()) for r, v in comp.items()}
    c = {r: round((passes[r] - passes[1]) / (r - 1), 3) for r in rows_list if r > 1}
    per_comp_c = {r: {k: round((comp[r][k] - comp[1][k]) / (r - 1), 3) for k in comp[r]} for r in rows_list if r > 1}
    return {
        "preset": args.preset, **info, "context": args.context,
        "components_ms": {str(r): v for r, v in comp.items()},
        "pass_ms": {str(r): round(p, 3) for r, p in passes.items()},
        "c_ms": {str(r): v for r, v in c.items()},
        "c_ms_by_component": {str(r): v for r, v in per_comp_c.items()},
        "first_extra_row_ms": round(passes[2] - passes[1], 3) if 2 in passes else None,
        "gate_c_ms": GATE_C_MS,
        "gate_note": "the WP-A8 gate applies to a Kaggle 7B run with masking and service included",
        "trips_ms": args.trips_ms,
        "projection": projection(passes, args.trips_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="7b", choices=["7b", "0.5b", "8b"])
    ap.add_argument("--layers", type=int, default=4, help="layers measured (scaled to the preset's; 0 = all)")
    ap.add_argument("--vocab", type=int, default=0, help="override the vocabulary (0 = the preset's; smoke tests)")
    ap.add_argument("--rows", default="1,2,4,8", help="verify rows R per pass, comma list (1 is always added)")
    ap.add_argument("--context", type=int, default=48)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: default)")
    ap.add_argument("--masking", action="store_true", help="add gateway masking (profile_gateway subprocess)")
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--service-url", default=None, help="add service time (pgw.service.loadtest) from this service")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--fingerprint", default=None)
    ap.add_argument("--trips-ms", type=float, default=0.0,
                    help="row-independent round-trip ms per pass added to the projection (Kaggle run 27: 163)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.workdir is None:
        import tempfile

        args.workdir = tempfile.mkdtemp(prefix="pgw-spec-cost-")
    res = run(args)
    text = json.dumps(res, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
