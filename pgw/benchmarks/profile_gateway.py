"""Profile the gateway side of masked inference at real model widths.

Builds a synthetic Qwen2 checkpoint with the hidden/intermediate sizes of a real model but only a
few layers, packs it, fills a real encrypted MaskStore with the real MaskGenerator, and runs decode
steps (T rows per linear) through ``MaskedBackend`` + an in-process ``LocalComputeClient``. Service
compute time is measured separately and subtracted, so the numbers are gateway CPU only.

Usage:
    python benchmarks/profile_gateway.py --preset 7b --layers 2 --steps 8 [--rows 1] [--sessions 4] [--profile]

Prints per-linear-group and per-token gateway CPU (extrapolated to the full layer count) and,
with ``--profile``, the top functions by cumulative time. ``--sessions N`` makes N reservations and
sends one call per session in each ``forward_many`` (as the batching engine does); the per-call
numbers are then per round trip (all sessions), and ``gateway_ms_per_session_call`` divides by N.

Speculation-shaped sweep (chat-speed plan WP-A0): ``--sessions 1 --rows 1,2,4,8`` measures one session's
verify pass at each row count R (the same masks, checks and unmasking a fixed-R pass does) and adds
``per_extra_row`` (gateway ms per full-model pass per extra row, against the first ``--rows`` value, given
as ``per_extra_row_baseline_rows``; start the list with 1 to measure against decode). With one ``--rows``
value the output is the single result as before. ``--profile`` tables print after the JSON, as before.
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


def build_checkpoint(out: Path, preset: dict[str, int], layers: int, zero_linears: bool = False) -> Path:
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
    if zero_linears:  # --null-service: W = 0, so the service result is all zeros (gateway work is unchanged)
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if "layers." in name and "proj" in name:
                    prm.zero_()
    model.eval().save_pretrained(out, safe_serialization=True)
    _save_byte_tokenizer(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="7b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rows", default="1", help="rows per call (1 = decode, >1 = prefill/batch/verify); a comma "
                    "list sweeps them (e.g. 1,2,4,8 with --sessions 1: speculative verify passes)")
    ap.add_argument("--sessions", type=int, default=1, help="reservations per round trip (one call each)")
    ap.add_argument("--ring", default="K40A12")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--noise", action="store_true", help="enable the outbound-noise monitor (gateway default)")
    ap.add_argument("--service-url", default=None, help="use a real compute service (ComputeClient) instead of "
                    "the in-process one; reports wall ms per call including the round trip")
    ap.add_argument("--null-service", action="store_true", help="zero linear weights and a service stub that "
                    "returns zeros: the gateway does exactly the same work (masking, checks, unmasking are "
                    "data-independent), but no CPU matmul runs between its steps and evicts its caches (closer to "
                    "a GPU service)")
    ap.add_argument("--token-file", default=None, help="service bearer token file (with --service-url)")
    args = ap.parse_args()

    from pgw.core import RingParams
    from pgw.engine import LocalComputeClient
    from pgw.engine.backends import MaskedBackend
    from pgw.factory import MaskGenerator
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.modelpack import ModelPack, pack_checkpoint

    preset = PRESETS[args.preset]
    rows_list = [int(r) for r in str(args.rows).split(",") if r]
    ring = RingParams.from_name(args.ring)
    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-prof-"))
    work.mkdir(parents=True, exist_ok=True)
    sfx = "-null" if args.null_service else ""
    ck, pk = work / f"ckpt{sfx}", work / f"pack{sfx}"
    t0 = time.perf_counter()
    if not (pk / "manifest.json").exists():
        if not (ck / "config.json").exists():
            build_checkpoint(ck, preset, args.layers, zero_linears=args.null_service)
        pack_checkpoint(ck, pk, name=f"synthetic-{args.preset}", overwrite=True)
    pack = ModelPack.open(pk)
    print(f"pack ready in {time.perf_counter() - t0:.1f}s")

    slots = args.steps * sum(rows_list) + 4
    nses = max(1, args.sessions)
    counter = work / "counter"
    kp = FileKeyProvider(work / "kek", create=True)
    store_kw: dict[str, Any] = {"fsync": False}
    try:
        from pgw.maskstore.counters import FileMonotonicCounter

        store_kw["counter"] = FileMonotonicCounter(counter, fsync=False)
    except Exception as exc:  # fall back to whatever the store's dev default is
        print(f"no file counter ({type(exc).__name__}); using the store default")
    store = MaskStore.open(work / f"store-{time.time_ns()}", kp, **store_kw)
    t0 = time.perf_counter()
    MaskGenerator(pack, ring).generate_into_store(store, slots * nses, chunk_slots=16, factory_id="prof")
    print(f"factory: {slots * nses} slots in {time.perf_counter() - t0:.1f}s")

    class TimedClient(LocalComputeClient):
        service_s = 0.0

        def compute_sync(self, request: Any) -> Any:
            t = time.perf_counter()
            try:
                return super().compute_sync(request)
            finally:
                TimedClient.service_s += time.perf_counter() - t

    class NullClient:
        """Service stub for zero weights: ``y_hat = W x_hat = 0``."""

        def __init__(self) -> None:
            self.out = {sp.id.key: sp.out_features for sp in pack.linear_specs()}

        def compute_sync(self, request: Any) -> Any:
            from pgw.wire.messages import ComputeResponse, ComputeResult, ServerTiming

            nb = ring.k // 8
            items = tuple(ComputeResult(it.linear, it.rows, bytes(it.rows * self.out[it.linear] * nb))
                          for it in request.items)
            return ComputeResponse(request.request_id, items, ServerTiming(0.0, 0.0, 0))

    client: Any = NullClient() if args.null_service else TimedClient(pack)
    if args.service_url:
        from pgw.wire.client import ClientConfig, ComputeClient

        tok = Path(args.token_file).read_text().strip() if args.token_file else None
        client = ComputeClient(ClientConfig(base_url=args.service_url, token=tok))
        client.connect_sync(pack.fingerprint, ring.name)
    noise = None
    if args.noise:
        from pgw.observability.noise import OutboundNoiseMonitor

        noise = OutboundNoiseMonitor(ring)
    be = MaskedBackend(pack, None, client, ring, pack.fingerprint, min_soundness_bits=20, noise_monitor=noise)
    sessions = [store.reserve(pack.fingerprint, ring, slots) for _ in range(nses)]
    in_dim = {sp.id.kind.value: sp.in_features
              for sp in pack.linear_specs() if sp.id.layer == 0}

    measured = [_measure(args, rows, be, client, sessions, in_dim, preset, nses, TimedClient) for rows in rows_list]
    results = [m[0] for m in measured]
    out = results[0] if len(results) == 1 else {
        "preset": args.preset, "ring": args.ring, "sessions": nses, "sweep": results,
        "per_extra_row_baseline_rows": results[0]["rows"],
        "per_extra_row": {
            str(r["rows"]): {
                "gateway_ms_per_step_full_model": round(
                    (r["gateway_s_per_step_full_model"] - results[0]["gateway_s_per_step_full_model"]) * 1e3
                    / max(1, r["rows"] - results[0]["rows"]), 4)}
            for r in results[1:]},
    }
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    for _, tables in measured:
        if tables:
            print(tables)
    for r in sessions:
        r.close()
    store.close()


def _measure(args: Any, rows: int, be: Any, client: Any, sessions: list[Any], in_dim: dict[str, int],
             preset: dict[str, int], nses: int, TimedClient: Any) -> tuple[dict[str, Any], str]:
    """Gateway ms per call at ``rows`` rows per session call (warm-up step + ``steps - 1`` measured), and the
    ``--profile`` tables as text (empty without ``--profile``); the caller prints them after the JSON."""
    from pgw.core.types import LinearId
    from pgw.engine.backends import LinearCall

    groups = ["qkv", "o", "gate_up", "down"]

    def calls_for(lid: Any) -> list[Any]:
        return [LinearCall(lid, torch.randn(rows, in_dim[lid.kind.value]), session=r) for r in sessions]

    # warm up every layer (weight norms, chunk and stream-key caches are per linear), i.e. one full step:
    # warming only layer 0 left the first measured step paying the other layers' first-call costs
    for layer in range(args.layers):
        for g in groups:
            be.forward_many(calls_for(LinearId(layer, g)))
    per_group = {g: 0.0 for g in groups}
    t_prep = t_fin = 0.0
    prof = cProfile.Profile() if args.profile else None
    TimedClient.service_s = 0.0
    total0 = time.perf_counter()
    for _ in range(args.steps - 1):
        for layer in range(args.layers):
            for g in groups:
                calls = calls_for(LinearId(layer, g))
                if args.service_url:  # real service: wall time of the whole round trip
                    s0 = TimedClient.service_s
                    t = time.perf_counter()
                    be.forward_many(calls)
                    per_group[g] += (time.perf_counter() - t) - (TimedClient.service_s - s0)
                    continue
                t = time.perf_counter()
                if prof:
                    prof.enable()
                prep = be.prepare_calls(calls)
                if prof:
                    prof.disable()
                t1 = time.perf_counter()
                resp = client.compute_sync(prep.request)
                t2 = time.perf_counter()
                if prof:
                    prof.enable()
                be.finish_prepared(prep, resp, t2 - t1)
                if prof:
                    prof.disable()
                t3 = time.perf_counter()
                t_prep += t1 - t
                t_fin += t3 - t2
                per_group[g] += (t1 - t) + (t3 - t2)
    wall = time.perf_counter() - total0
    n = (args.steps - 1) * args.layers
    scale = preset["full_layers"] / 1
    ncalls = max(1, n * len(groups))
    out = {
        "preset": args.preset, "ring": args.ring, "rows": rows, "sessions": nses,
        "gateway_ms_per_call": {g: 1000 * v / n for g, v in per_group.items()},
        "gateway_ms_per_call_avg": 1000 * sum(per_group.values()) / ncalls,
        "gateway_ms_per_session_call": 1000 * sum(per_group.values()) / ncalls / nses,
        "prepare_ms_per_call": 1000 * t_prep / ncalls,
        "finish_ms_per_call": 1000 * t_fin / ncalls,
        "gateway_ms_per_layer": 1000 * sum(per_group.values()) / n,
        "gateway_s_per_step_full_model": sum(per_group.values()) / n * scale,
        "service_s_per_step_full_model": TimedClient.service_s / n * scale,
        "wall_s": wall,
        "wall_ms_per_call": 1000 * wall / ncalls,
    }
    tables = ""
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(35)
        tables += s.getvalue() + "\n"
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(25)
        tables += s.getvalue()
    return out, tables


if __name__ == "__main__":
    main()
