"""Profile the gateway's plaintext work per decode token ("unaccounted" time) at full model shapes.

Kaggle run 26 left about 0.094 s per decode token that is neither service round trips nor gateway
masking: the LM head, RMSNorm, RoPE, attention over the KV cache, sampling (with logprobs/top-k),
per-layer Python, torch dispatch and thread hops. This script measures exactly that part.

It builds a :class:`pgw.engine.model.TransformerModel` with the real layer count and shapes
(Qwen2.5-7B by default: 28 layers, hidden 3584, 28/4 heads, vocab 152064) over a synthetic pack
and a **stub backend** that returns fixed float32 outputs for every linear (no masking, no service,
like ``e2e_local --null-service``). Only the gateway-side float tensors are synthesized: the LM head
(bfloat16, 1.1 GB at 7B), the norms and a broadcast embedding (one row viewed ``vocab`` times), so
the process stays at about 1.5 GB.

Reported per decode step (one new token, ``--context`` positions already in the cache):

- ``step_ms``: one forward pass through the synchronous path (``forward_batch``), un-instrumented;
- ``step_async_ms``: the same through the gateway's async split-hop path (``aforward_batch`` with a
  backend that has ``prepare_calls``/``send_prepared``/``finish_prepared``, like ``MaskedBackend``),
  so the event-loop and thread-hop overhead is ``step_async_ms - step_ms``;
- ``components_ms``: an instrumented pass split into the LM head, RMSNorm, RoPE (tables and
  application), KV-cache append, attention, and the rest of the layer body (``glue``: q/k/v views,
  residual adds, SiLU-gate, LinearCall lists, generator hand-offs);
- ``sample_ms``: :meth:`pgw.engine.sampling.Sampler.sample` on the step's logits (greedy with
  ``--logprobs`` alternatives, as the Kaggle validation requests);
- ``layers_only_ms``: the step with the LM head pass replaced by an empty tensor (all 28 layers,
  final norm and the generator machinery);
- ``micro_us``: single-op timings at these shapes (what each glue op costs on this CPU).

``--reference-paths`` swaps in the reference RoPE form and torch's argmax / top-k in the sampler (the
code before phase 2) for an A/B on one machine; ``PGW_LM_HEAD_ISA=avx`` does the same for the LM head
kernel's instruction set (the pre-phase-2 kernel also lacked register blocking across rows).

Usage:
    PYTHONPATH=src python benchmarks/profile_token.py [--preset 7b] [--context 48] [--steps 20]
        [--threads N] [--layers L] [--rows R] [--json out.json] [--cprofile]

``--rows R`` profiles a step of R concurrent sequences (one row each; the LM head then reads the
head once for all of them). The numbers are for this machine only; Kaggle runs are what count.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import os
import platform
import pstats
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

# As the ``pgw`` CLI does for the gateway (pgw/cli/__init__.py): idle OpenMP workers sleep instead of
# spinning. Spinning workers (the default) slow the native LM head that follows torch's per-layer
# attention by about a third on 4 vCPUs. Set OMP_WAIT_POLICY=ACTIVE to see that effect.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch  # after the OpenMP setting on purpose
import torch.nn.functional as F

PRESETS: dict[str, dict[str, Any]] = {
    "7b": {"hidden_size": 3584, "intermediate_size": 18944, "num_heads": 28, "num_kv_heads": 4, "num_layers": 28,
           "vocab_size": 152064, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6},
    "0.5b": {"hidden_size": 896, "intermediate_size": 4864, "num_heads": 14, "num_kv_heads": 2, "num_layers": 24,
             "vocab_size": 151936, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6},
    "1.5b": {"hidden_size": 1536, "intermediate_size": 8960, "num_heads": 12, "num_kv_heads": 2, "num_layers": 28,
             "vocab_size": 151936, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6},
    "8b": {"hidden_size": 4096, "intermediate_size": 14336, "num_heads": 32, "num_kv_heads": 8, "num_layers": 32,
           "vocab_size": 128256, "rope_theta": 500000.0, "rms_norm_eps": 1e-5},
}


def _bf16_exact(*shape: int, gen: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    """float32 values that are exactly bfloat16 (as an open checkpoint's)."""
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16).to(torch.float32)


class SynthPack:
    """Just enough of :class:`pgw.engine.pack.EnginePack` for :class:`TransformerModel` (plain tensors only)."""

    def __init__(self, preset: dict[str, Any], layers: int | None = None, seed: int = 0):
        from pgw.modelpack.manifest import ArchConfig

        p = dict(preset)
        if layers:
            p["num_layers"] = layers
        self.arch = ArchConfig(
            arch="qwen2", hidden_size=p["hidden_size"], intermediate_size=p["intermediate_size"],
            num_layers=p["num_layers"], num_heads=p["num_heads"], num_kv_heads=p["num_kv_heads"],
            head_dim=p["hidden_size"] // p["num_heads"], vocab_size=p["vocab_size"],
            rms_norm_eps=p["rms_norm_eps"], rope_theta=p["rope_theta"], max_position_embeddings=32768,
            tie_word_embeddings=False,
        )
        self._gen = torch.Generator().manual_seed(seed)
        self.name = "synthetic"
        self.fingerprint = "0" * 64

    def load_plain(self, name: str, dtype: torch.dtype = torch.float32, device: Any = "cpu") -> torch.Tensor:
        a, g = self.arch, self._gen
        if name == "embed_tokens":  # one row viewed vocab times: the lookup costs the same, no 2 GB tensor
            return _bf16_exact(1, a.hidden_size, gen=g, scale=0.02).expand(a.vocab_size, a.hidden_size)
        if name == "lm_head":
            # bfloat16 directly (what compact_head keeps for a bfloat16 checkpoint) without the float32 copy
            return (torch.randn((a.vocab_size, a.hidden_size), generator=g) * 0.02).to(torch.bfloat16)
        if name.endswith("norm"):
            return 1.0 + _bf16_exact(a.hidden_size, gen=g, scale=0.05)
        raise KeyError(name)


def _stub_outputs(arch: Any, rows: int, seed: int = 1) -> dict[str, torch.Tensor]:
    """Fixed float32 linear outputs per kind (magnitudes like a real layer's)."""
    from pgw.core.types import LinearKind
    from pgw.engine.pack import linear_split_sizes

    g = torch.Generator().manual_seed(seed)
    out = {}
    for kind in (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP, LinearKind.DOWN):
        m = sum(linear_split_sizes(arch, kind))
        out[kind.value] = torch.randn((rows, m), generator=g) * (0.5 if kind != LinearKind.GATE_UP else 2.0)
    return out


def make_backends(arch: Any, rows: int) -> tuple[Any, Any]:
    """(sync stub, async split-hop stub) returning the fixed outputs."""
    from pgw.core.ring import RingParams
    from pgw.engine.backends import LinearBackend, LinearCall

    outs = _stub_outputs(arch, 1)

    def result(c: LinearCall) -> torch.Tensor:
        y = outs[c.id.kind.value]
        n = int(c.x.shape[0])
        return y if n == 1 else y.expand(n, -1)

    class Stub(LinearBackend):
        ring = RingParams.from_name("K32A12C27")

        def forward(self, call: LinearCall) -> torch.Tensor:
            return result(call)

    class Prepared:
        def __init__(self, calls: Sequence[LinearCall]):
            self.calls = list(calls)

    class SplitStub(Stub):
        """Has the split-hop interface of MaskedBackend (prepare / send / finish), no crypto."""

        split_hop = True
        mask_source = None

        def prepare_calls(self, calls: Sequence[LinearCall]) -> Prepared:
            return Prepared(calls)

        async def send_prepared(self, prep: Prepared) -> tuple[Any, float]:
            return None, 0.0

        def finish_prepared(self, prep: Prepared, response: Any, remote_s: float) -> list[torch.Tensor]:
            return [result(c) for c in prep.calls]

        @staticmethod
        def discard(prep: Any) -> None:
            return None

    return Stub(), SplitStub()


class Timers:
    """Wraps module-level callables to accumulate their wall time per name."""

    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._undo: list[tuple[Any, str, Any]] = []

    def wrap(self, owner: Any, attr: str, name: str) -> None:
        fn = getattr(owner, attr)
        total, calls = self.total, self.calls

        def timed(*a: Any, **k: Any) -> Any:
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                total[name] += time.perf_counter() - t
                calls[name] += 1

        self._undo.append((owner, attr, fn))
        setattr(owner, attr, timed)

    def reset(self) -> None:
        self.total.clear()
        self.calls.clear()

    def restore(self) -> None:
        for owner, attr, fn in reversed(self._undo):
            setattr(owner, attr, fn)
        self._undo.clear()


def _median_ms(fn: Callable[[], Any], reps: int) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def micro(arch: Any, reps: int = 300) -> dict[str, float]:
    """Single-op timings (microseconds, median) at these shapes, one row."""
    from pgw.engine.model import rms_norm
    from pgw.engine.rope import apply_rope

    g = torch.Generator().manual_seed(5)
    hsz, inter = arch.hidden_size, arch.intermediate_size
    nh, hd = arch.num_heads, arch.head_dim
    h = torch.randn((1, hsz), generator=g)
    w = torch.randn(hsz, generator=g)
    gu = torch.randn((1, 2 * inter), generator=g)
    q = torch.randn((1, nh, hd), generator=g)
    cs = torch.randn((1, hd), generator=g)
    ops: dict[str, Callable[[], Any]] = {
        "rms_norm": lambda: rms_norm(h, w, 1e-6),
        "residual_add": lambda: h + h,
        "silu_gate": lambda: F.silu(gu[:, :inter]) * gu[:, inter:],
        "rope_q": lambda: apply_rope(q, cs, cs),
        "tensor_to_same_dtype": lambda: h.to(torch.float32),
    }
    out = {}
    for name, fn in ops.items():
        for _ in range(20):
            fn()
        out[name] = round(1000 * _median_ms(fn, reps), 2)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.engine import lm_head as lm_head_mod
    from pgw.engine import model as model_mod
    from pgw.engine.kv_cache import KVCache
    from pgw.engine.model import SequenceInput, TransformerModel
    from pgw.engine.rope import RotaryEmbedding
    from pgw.engine.sampling import Sampler, SamplingParams

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.reference_paths:  # the pre-phase-2 forms of RoPE and of the sampler's argmax / top-k
        from pgw.engine import rope as rope_mod
        from pgw.engine import sampling as sampling_mod

        model_mod.apply_rope = getattr(rope_mod, "apply_rope_reference", rope_mod.apply_rope)

        def _topk(x: torch.Tensor, k: int) -> list[tuple[int, float]]:
            top = torch.topk(x, k)
            return [(int(i), float(v)) for v, i in zip(top.values, top.indices, strict=True)]

        if hasattr(sampling_mod, "fast_argmax"):
            sampling_mod.fast_argmax = lambda x: int(torch.argmax(x))  # type: ignore[assignment]
            sampling_mod.fast_topk = _topk  # type: ignore[assignment]
    pack = SynthPack(PRESETS[args.preset], args.layers)
    a = pack.arch
    sync_be, async_be = make_backends(a, args.rows)
    t0 = time.perf_counter()
    model = TransformerModel(pack, sync_be)
    amodel = TransformerModel.__new__(TransformerModel)
    amodel.__dict__.update(model.__dict__)  # same tensors, own backend and plaintext thread
    amodel.backend = async_be
    amodel._plain_pool = None
    import threading

    amodel._plain_lock = threading.Lock()
    amodel._plain_pending = 0
    load_s = time.perf_counter() - t0
    caches = [model.new_cache(args.context + args.steps * 8 + 16) for _ in range(args.rows)]
    acaches = [model.new_cache(args.context + args.steps * 8 + 16) for _ in range(args.rows)]
    # fill the caches with a prefill of `context` positions (computed once)
    prompt = [(17 * i + 3) % a.vocab_size for i in range(args.context)]
    for c in caches + acaches:
        model.forward(prompt, c)

    def seqs(cs: list[KVCache]) -> list[SequenceInput]:
        return [SequenceInput([11 + i], c) for i, c in enumerate(cs)]

    def rewind(cs: list[KVCache]) -> None:
        for c in cs:
            c.truncate(args.context)

    def sync_step() -> list[torch.Tensor]:
        rewind(caches)
        return model.forward_batch(seqs(caches))

    loop = asyncio.new_event_loop()

    def async_step() -> list[torch.Tensor]:
        rewind(acaches)
        return loop.run_until_complete(amodel.aforward_batch(seqs(acaches)))

    for _ in range(2):  # warm-up (GQA self-check, thread pools, allocator)
        sync_step()
        async_step()
    step_ms = _median_ms(sync_step, args.steps)
    step_async_ms = _median_ms(async_step, args.steps)
    same_async = all(torch.equal(x, y) for x, y in zip(sync_step(), async_step(), strict=True))

    # instrumented pass: per-component wall time
    timers = Timers()
    timers.wrap(model_mod, "lm_head_logits", "lm_head")
    timers.wrap(model_mod, "rms_norm", "rms_norm")
    timers.wrap(model_mod, "apply_rope", "rope_apply")
    timers.wrap(RotaryEmbedding, "cos_sin", "rope_tables")
    timers.wrap(KVCache, "append", "kv_append")
    timers.wrap(model_mod, "_attend", "attention")
    timers.wrap(model_mod, "_norm_rows", "final_norm_rows")
    totals: dict[str, list[float]] = defaultdict(list)
    walls = []
    try:
        for _ in range(args.steps):
            timers.reset()
            t = time.perf_counter()
            sync_step()
            walls.append(time.perf_counter() - t)
            for k, v in timers.total.items():
                totals[k].append(v)
        calls = dict(timers.calls)  # of the last step
    finally:
        timers.restore()
    comp = {k: round(1000 * statistics.median(v), 3) for k, v in totals.items()}
    wall_instr = 1000 * statistics.median(walls)
    comp["glue"] = round(wall_instr - sum(comp.values()), 3)

    # the step without the LM head (its pass replaced by an empty logits tensor): layers + final norm
    def no_head(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.zeros((h.shape[0], a.vocab_size))

    orig_head = model_mod.lm_head_logits
    model_mod.lm_head_logits = no_head
    try:
        layers_ms = _median_ms(sync_step, args.steps)
    finally:
        model_mod.lm_head_logits = orig_head

    # sampling (greedy + logprobs alternatives) on this step's logits
    logits = sync_step()
    params = SamplingParams(max_tokens=16, temperature=0.0, logprobs=args.logprobs, ignore_eos=True)
    samplers = [Sampler(params, prompt) for _ in range(args.rows)]

    def sample_all() -> None:
        for s, lg in zip(samplers, logits, strict=True):
            s.sample(lg, [1, 2, 3])

    for _ in range(5):
        sample_all()
    sample_ms = _median_ms(sample_all, max(args.steps, 20))

    # LM head alone (one pass, all rows) for reference
    hn = torch.randn((args.rows, a.hidden_size), generator=torch.Generator().manual_seed(9))
    lm_ms = _median_ms(lambda: lm_head_mod.lm_head_logits(hn, model._head), max(3, args.steps // 2))

    out: dict[str, Any] = {
        "preset": args.preset,
        "layers": a.num_layers,
        "rows": args.rows,
        "context": args.context,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
        "native_lm_head": lm_head_mod.native_lm_head(),
        "lm_head_dtype": str(model._head.dtype),
        "cpu": platform.processor() or platform.machine(),
        "load_s": round(load_s, 2),
        "step_ms": round(step_ms, 3),
        "step_async_ms": round(step_async_ms, 3),
        "async_matches_sync": same_async,
        "hop_overhead_ms": round(step_async_ms - step_ms, 3),
        "instrumented_step_ms": round(wall_instr, 3),
        "components_ms": dict(sorted(comp.items(), key=lambda kv: -kv[1])),
        "component_calls_per_step": calls,
        "layers_only_ms": round(layers_ms, 3),
        "lm_head_alone_ms": round(lm_ms, 3),
        "sample_ms": round(sample_ms, 3),
        "token_ms": round(step_async_ms + sample_ms, 3),
        "micro_us": micro(a),
    }
    out["lm_head_isa"] = getattr(lm_head_mod, "lm_head_isa", lambda: None)()
    out["reference_paths"] = bool(args.reference_paths)
    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.steps):
            async_step()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
        out["cprofile_top"] = s.getvalue()
    loop.close()
    amodel.close()
    model.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="7b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=0, help="override the layer count (0: the preset's)")
    ap.add_argument("--context", type=int, default=48, help="positions already in the KV cache")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rows", type=int, default=1, help="concurrent sequences per step (one row each)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: default)")
    ap.add_argument("--logprobs", type=int, default=2, help="top logprobs per token (Kaggle validation: 2)")
    ap.add_argument("--json", default=None, help="also write the result here")
    ap.add_argument("--cprofile", action="store_true", help="add the top functions of the async path")
    ap.add_argument("--reference-paths", action="store_true",
                    help="use the reference (pre-phase-2) RoPE and sampler argmax/top-k; with "
                         "PGW_LM_HEAD_ISA=avx the LM head also uses the widest pre-phase-2 instruction set")
    args = ap.parse_args()
    res = run(args)
    prof = res.pop("cprofile_top", None)
    text = json.dumps(res, indent=2)
    print(text)
    if prof:
        print(prof)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
