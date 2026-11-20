"""End-to-end local throughput: ``Generator`` + ``MaskedBackend`` over the in-process service.

Builds (once, cached in ``--workdir``) a synthetic Qwen2 checkpoint, packs it, fills a real
encrypted MaskStore with the real MaskGenerator, and runs ``--streams`` concurrent greedy requests
through the continuous-batching engine (``Generator.agenerate``) with windowed reservations, as the
gateway does. The compute service is the in-process ``LocalComputeClient`` (its CPU time is
reported separately). Prints tokens/s (aggregate) and gateway seconds per token.

Usage:
    python benchmarks/e2e_local.py --streams 1 --tokens 48 --workdir /tmp/pgw-e2e
    python benchmarks/e2e_local.py --streams 4 --tokens 48 --workdir /tmp/pgw-e2e

With ``--check`` the outputs are compared with the plaintext reference backend (bit-identical
token ids and logprobs are expected).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PRESETS: dict[str, dict[str, int]] = {
    "small": {"hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14,
              "num_key_value_heads": 2, "layers": 4, "vocab": 2048},
    # 7B layer and vocabulary shapes with 4 of 28 layers: per-layer costs scale by 7, the LM head does not.
    "7b4": {"hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
            "num_key_value_heads": 4, "layers": 4, "vocab": 152064},
    "mid": {"hidden_size": 1536, "intermediate_size": 8960, "num_attention_heads": 12,
            "num_key_value_heads": 2, "layers": 2, "vocab": 4096},
}


def build_pack(work: Path, preset: str, null: bool = False) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint

    p = PRESETS[preset]
    sfx = "-null" if null else ""
    ck, pk = work / f"ckpt-{preset}{sfx}", work / f"pack-{preset}{sfx}"
    if (pk / "manifest.json").exists():
        return pk
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.conftest import _save_byte_tokenizer  # type: ignore[import-not-found]

    cfg = Qwen2Config(
        vocab_size=p["vocab"], hidden_size=p["hidden_size"], intermediate_size=p["intermediate_size"],
        num_hidden_layers=p["layers"], num_attention_heads=p["num_attention_heads"],
        num_key_value_heads=p["num_key_value_heads"], max_position_embeddings=2048,
        tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    if null:  # zero linear weights: the service result is all zeros (see --null-service)
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if "layers." in name and "proj" in name:
                    prm.zero_()
    model.eval().save_pretrained(ck, safe_serialization=True)
    _save_byte_tokenizer(ck)
    pack_checkpoint(ck, pk, name=f"synthetic-{preset}", overwrite=True)
    return pk


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core import RingParams
    from pgw.engine import Generator, LocalComputeClient, MaskedBackend, PlainBackend, SamplingParams, TransformerModel
    from pgw.factory import MaskGenerator
    from pgw.gateway.masks import WindowedReservation
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.modelpack import ModelPack

    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    pack = ModelPack.open(build_pack(work, args.preset, args.null_service))
    ring = RingParams.from_name(args.ring)
    prompt_len = 8
    per = prompt_len + args.tokens
    slots = args.streams * (per + args.window) * (args.repeat + 1) + 64
    kp = FileKeyProvider(work / "kek", create=True)
    store = MaskStore.open(work / f"store-{time.time_ns()}", kp, fsync=False,
                           counter=FileMonotonicCounter(work / f"counter-{time.time_ns()}", fsync=False))
    t0 = time.perf_counter()
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="e2e")
    print(f"factory: {slots} slots in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    class TimedClient(LocalComputeClient):
        service_s = 0.0

        def compute_sync(self, request: Any) -> Any:
            t = time.perf_counter()
            try:
                return super().compute_sync(request)
            finally:
                TimedClient.service_s += time.perf_counter() - t

    client: Any = TimedClient(pack, backend=args.service_backend)
    if args.null_service:
        from pgw.wire.messages import ComputeResponse, ComputeResult

        outm = {sp.id.key: sp.out_features for sp in pack.linear_specs()}
        nb = ring.k // 8

        class NullClient:
            """Service stub for zero weights (``W x_hat = 0``): no service CPU time at all."""

            async def compute(self, req: Any) -> Any:
                return ComputeResponse(req.request_id, tuple(
                    ComputeResult(i.linear, i.rows, bytes(i.rows * outm[i.linear] * nb)) for i in req.items))

        client = NullClient()
    backend = MaskedBackend(pack, None, client, ring, pack.fingerprint, min_soundness_bits=20)
    model = TransformerModel(pack, backend)
    fp = pack.fingerprint

    def factory(n: int) -> Any:
        first = min(n, prompt_len + args.window)
        res = store.reserve(fp, ring, first)
        if first >= n:
            return res
        return WindowedReservation(res, lambda s: store.reserve(fp, ring, s), cap=n, window=args.window)

    gen = Generator(model, None, max_batch=32, session_factory=factory)
    params = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True, logprobs=0)
    prompts = [[(7 * s + 3 * i) % (pack.arch.vocab_size - 1) + 1 for i in range(prompt_len)]
               for s in range(args.streams)]

    async def once() -> tuple[float, list[Any]]:
        t = time.perf_counter()
        outs = await asyncio.gather(*(gen.agenerate(p, params) for p in prompts))
        return time.perf_counter() - t, outs

    await once()  # warm-up (weight metadata, chunk caches, threads)
    best: dict[str, Any] | None = None
    outs: list[Any] = []
    for _ in range(args.repeat):
        st0 = backend.stats.snapshot()
        svc0 = TimedClient.service_s
        wall, outs = await once()
        st1 = backend.stats.snapshot()
        toks = sum(len(o.token_ids) for o in outs)
        decode_steps = args.tokens  # per stream
        row = {
            "wall_s": wall,
            "tokens": toks,
            "tokens_per_s": toks / wall,
            "gateway_local_s_per_token": (st1["local_seconds"] - st0["local_seconds"]) / toks,
            "service_s_per_token": (TimedClient.service_s - svc0) / toks,
            # everything but the (serial) service compute: gateway crypto, hops, model, LM head
            "gateway_wall_s_per_token": (wall - (TimedClient.service_s - svc0)) / toks,
            "wall_ms_per_step": 1000 * wall / decode_steps,
            "round_trips": st1["round_trips"] - st0["round_trips"],
        }
        if best is None or row["tokens_per_s"] > best["tokens_per_s"]:
            best = row
    await gen.aclose()
    assert best is not None
    result = {"preset": args.preset, "ring": args.ring, "streams": args.streams, **best}
    if args.check:
        plain = TransformerModel(pack, PlainBackend(pack, ring))
        pgen = Generator(plain, None, max_batch=32)
        ref = [await pgen.agenerate(p, params, session=None) for p in prompts]
        await pgen.aclose()
        ok = all(a.token_ids == b.token_ids and a.logprobs == b.logprobs for a, b in zip(outs, ref, strict=True))
        result["bit_identical_to_plain"] = ok
        plain.close()
    model.close()
    store.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--null-service", action="store_true", help="zero linear weights and an async service stub "
                    "returning zeros (no service CPU; isolates the gateway side: crypto, hops, model, sampling)")
    ap.add_argument("--service-backend", default="auto", help="ringmm backend of the in-process service")
    ap.add_argument("--check", action="store_true", help="compare with the plaintext reference")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
