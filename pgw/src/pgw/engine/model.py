"""Decoder-only transformer forward pass (Qwen2 and Llama) with pluggable linear backend.

Extension points (chat-speed WP-A5; each default is the engine's code before they existed, bit for
bit, and phase-B modes plug in here without editing this file):

* :class:`FloatOps` (``TransformerModel.float_ops``): the plaintext float operations of a layer
  (RMSNorm, RoPE, attention over the KV cache, SiLU gate). Default :class:`TorchFloatOps`.
* :class:`HeadMode` (``TransformerModel.head_mode``): how logits are made from the final hidden rows;
  may add round trips. Default: the gateway's exact float32 head (:meth:`TransformerModel._logits`).
* :class:`Schedule` (``TransformerModel.schedule``): the linears of a layer and the float work between
  them. Default :class:`StandardSchedule` (qkv, o, gate_up, down: four round trips).
* Backend hooks read with ``getattr`` (a backend without them behaves as before):
  ``local_linear(lid) -> bool`` / ``forward_local(calls)``: linears computed exactly on the gateway;
  the forward pass steps past them without a network hop. ``sync_trip_ok(rows) -> bool``: the
  backend's synchronous ``forward_many`` is a fast trip (e.g. the native lane), so a decode pass may
  run as one call on the plaintext thread (:attr:`TransformerModel.sync_decode`).
* :attr:`SequenceInput.capture_hidden`: hidden states of chosen layers (EAGLE-style drafters).

Speed settings that keep the bits (each guarded by a self-check that falls back to today's path):
decode float ops on fewer torch threads (:meth:`TransformerModel.enable_float_threads`, env
``PGW_FLOAT_THREADS``), ``is_causal`` SDPA for a whole prompt block (:attr:`TransformerModel.causal_sdpa`).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import functools
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from collections.abc import Generator as PyGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.nn.functional as F

from pgw.core import native
from pgw.core.errors import ConfigError, ShapeError
from pgw.core.types import LinearId, LinearKind, ModelArch
from pgw.engine.backends import LinearBackend, LinearCall
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.lm_head import compact_head, lm_head_logits, native_lm_head
from pgw.engine.pack import linear_split_sizes
from pgw.engine.rope import RotaryEmbedding, apply_rope

__all__ = [
    "FloatOps", "GatewayHead", "HeadMode", "PassState", "Schedule", "SequenceInput", "StandardSchedule",
    "TorchFloatOps", "TransformerModel",
    "causal_self_check", "float_threads_self_check", "rms_norm",
]

log = logging.getLogger(__name__)

_T = TypeVar("_T")

#: A step of the forward coroutine: the backend calls to evaluate next (results are sent back).
_Steps = PyGenerator[list[LinearCall], list[torch.Tensor], list[torch.Tensor]]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF-compatible RMSNorm (statistics in float32)."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return weight * xf.to(x.dtype)


@dataclass
class SequenceInput:
    """One sequence of a batched forward pass.

    Attributes:
        input_ids: 1-D new token ids (prompt chunk or the last sampled token).
        cache: That sequence's KV cache.
        session: Backend session handle (mask reservation) or ``None``.
        all_logits: Return logits for every new position instead of only the last one.
        last_logits: Return :class:`RowLogits` for the last ``N`` new positions (speculative
            verification); takes precedence over ``all_logits`` when > 0.
        serial_tail: The last ``serial_tail`` positions are computed exactly as if each had been fed
            alone in its own step (attention one row at a time; each row's logits a 1-row product),
            so speculative verification gives bit-identical results to plain decoding. The rows
            before them are processed as one block, as plain decoding would.
        capture_hidden: Layers whose output hidden states (the residual stream after that layer,
            ``(T, hidden)`` for the new positions, before the final norm) are kept for a drafter;
            ``-1`` is the embedding output. Empty (default): nothing is kept.
        hidden: Filled by the forward pass when ``capture_hidden`` is set: ``{layer: tensor}``.
    """

    input_ids: torch.Tensor | Sequence[int]
    cache: KVCache
    session: object | None = None
    all_logits: bool = False
    last_logits: int = 0
    serial_tail: int = 0
    capture_hidden: Sequence[int] = ()
    hidden: dict[int, torch.Tensor] | None = None


def _sdpa_gqa_supported() -> bool:
    """True when ``F.scaled_dot_product_attention`` takes ``enable_gqa`` (torch >= 2.5)."""
    try:
        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:  # pragma: no cover - unusual version strings: keep the proven path
        return False
    return (major, minor) >= (2, 5)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float, groups: int,
            gqa: bool, causal: bool = False) -> torch.Tensor:
    """Causal attention of ``q (T, H, hd)`` over ``k, v (L, KVH, hd)``; returns ``(T, H*hd)``.

    ``gqa``: let SDPA broadcast each KV head over its ``groups`` query heads (``enable_gqa``) instead
    of materializing ``repeat_interleave`` copies of the cache. Only used where
    :func:`gqa_self_check` showed both forms give the same bits.

    ``causal``: SDPA's ``is_causal`` instead of the explicit boolean mask (the flash kernel then skips
    key blocks that are wholly masked). Only valid for a whole block from position 0 (``start == 0``
    and ``L == T``: ``is_causal`` aligns the mask top-left) and only used where
    :func:`causal_self_check` showed both forms give the same bits.
    """
    t = q.shape[0]
    length = k.shape[0]
    if groups > 1 and not gqa:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # (heads, *, hd)
    mask = None
    if causal:
        if start != 0 or length != t:
            raise ValueError("is_causal attention needs a whole block from position 0")
        kw: dict[str, Any] = {"is_causal": True}
    else:
        if t > 1:
            qpos = torch.arange(start, start + t, device=q.device)[:, None]
            kpos = torch.arange(length, device=q.device)[None, :]
            mask = kpos <= qpos
        kw = {"attn_mask": mask}
    if groups > 1 and gqa:
        out = F.scaled_dot_product_attention(qh, kh, vh, scale=scale, enable_gqa=True, **kw)
    else:
        out = F.scaled_dot_product_attention(qh, kh, vh, scale=scale, **kw)
    return out.transpose(0, 1).reshape(t, q.shape[1] * q.shape[2])


#: Cache lengths the GQA self-check covers (crosses the CPU flash kernel's 512-key block).
GQA_CHECK_LENGTHS = (1, 37, 100, 513)
_gqa_verdicts: dict[tuple[Any, ...], bool] = {}
_gqa_lock = threading.Lock()


def gqa_self_check(heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device | str,
                   scale: float, lengths: Sequence[int] = GQA_CHECK_LENGTHS) -> bool:
    """True when the ``enable_gqa`` SDPA form is bit-identical to the ``repeat_interleave`` form here.

    Runs both forms at the current torch thread count over a decode row (T=1), a short chunk and a
    whole prefill (T>1, causal mask), a speculative serial tail (rows one at a time over growing
    views) and cache-like views (a slice of a ``(layers, cap, kvh, hd)`` buffer, as
    :meth:`KVCache.append` returns), for every length in ``lengths``. Any difference or error: False.
    """
    groups = heads // kv_heads
    if groups <= 1:
        return True
    if not _sdpa_gqa_supported():
        return False
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(0x6A7)
    cap = max(lengths) + 3

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen).to(dtype=dtype, device=dev)

    try:
        with torch.no_grad():
            for length in lengths:
                kbuf, vbuf = rand(2, cap, kv_heads, head_dim), rand(2, cap, kv_heads, head_dim)
                k, v = kbuf[1, :length], vbuf[1, :length]
                cases = [(1, length - 1)]
                if length > 1:
                    cases += [(length, 0), (min(length - 1, 5), length - min(length - 1, 5))]
                for t, start in cases:
                    q = rand(t, heads, head_dim) * 2.0
                    if not torch.equal(_attend(q, k, v, start, scale, groups, False),
                                       _attend(q, k, v, start, scale, groups, True)):
                        return False
                tail = min(length, 3)
                st0 = length - tail
                q = rand(tail, heads, head_dim)
                for j in range(tail):
                    kj, vj = k[: st0 + j + 1], v[: st0 + j + 1]
                    if not torch.equal(_attend(q[j : j + 1], kj, vj, st0 + j, scale, groups, False),
                                       _attend(q[j : j + 1], kj, vj, st0 + j, scale, groups, True)):
                        return False
    except Exception as exc:  # e.g. a backend without enable_gqa support for these inputs
        log.debug("GQA SDPA self-check raised: %r", exc)
        return False
    return True


def _gqa_verdict(heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device,
                 scale: float) -> bool:
    """Cached :func:`gqa_self_check` per (shape, dtype, device, thread count) and process."""
    key = (heads, kv_heads, head_dim, dtype, str(device), scale, torch.get_num_threads())
    hit = _gqa_verdicts.get(key)
    if hit is not None:
        return hit
    with _gqa_lock:
        hit = _gqa_verdicts.get(key)
        if hit is None:
            hit = _gqa_verdicts[key] = gqa_self_check(heads, kv_heads, head_dim, dtype, device, scale)
            if hit:
                log.info("attention: SDPA enable_gqa (bit-identical to repeat_interleave; %d threads)", key[-1])
            else:
                log.warning("attention: enable_gqa SDPA is not bit-identical here (torch %s, %s, %s, %d "
                            "threads); keeping repeat_interleave", torch.__version__, dtype, device, key[-1])
        return hit


#: Prompt lengths the ``is_causal`` self-check covers: both sides of the CPU flash kernel's query
#: split sizes (32 below 192 rows, 64 below 768, then 256) and of its 512-key block. ``is_causal`` is
#: used for whole blocks of any length (>= ``causal_min_rows``): above 768 rows the split sizes no
#: longer change, so longer prompts only add blocks of the shapes checked here (spot checks at 1600,
#: 2100 and 2600 rows with 7B head shapes, float32 and bfloat16, with and without GQA, were equal;
#: not run at load: the check's attention grows with the square of the length).
CAUSAL_CHECK_LENGTHS = (16, 31, 33, 100, 191, 192, 300, 513, 767, 768, 1030)
_causal_verdicts: dict[tuple[Any, ...], bool] = {}
_causal_lock = threading.Lock()


def causal_self_check(heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device | str,
                      scale: float, gqa: bool, lengths: Sequence[int] = CAUSAL_CHECK_LENGTHS) -> bool:
    """True when ``is_causal`` SDPA gives the same bits as the explicit causal mask here.

    Compares both forms of :func:`_attend` (with the given GQA form, at the current torch thread
    count) for a whole block from position 0 over cache-like views, for every length in ``lengths``.
    Why they agree on CPU: in an unmasked position the score is ``s*scale`` either way (adding the
    mask's 0.0 is exact), and a key block that is wholly masked only contributes ``exp(-inf) = 0`` to
    the running sums with an unchanged running maximum, so skipping it changes nothing. Any
    difference or error: False (the explicit mask stays).
    """
    groups = heads // kv_heads
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(0xCA5A)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen).to(dtype=dtype, device=dev)

    try:
        with torch.no_grad():
            for t in lengths:
                kbuf, vbuf = rand(2, t + 3, kv_heads, head_dim), rand(2, t + 3, kv_heads, head_dim)
                k, v = kbuf[1, :t], vbuf[1, :t]
                q = rand(t, heads, head_dim) * 2.0
                if not torch.equal(_attend(q, k, v, 0, scale, groups, gqa),
                                   _attend(q, k, v, 0, scale, groups, gqa, causal=True)):
                    return False
    except Exception as exc:
        log.debug("is_causal SDPA self-check raised: %r", exc)
        return False
    return True


def _causal_verdict(heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device,
                    scale: float, gqa: bool, max_len: int) -> bool:
    """Cached :func:`causal_self_check` per (shape, dtype, device, GQA form, thread count) and process."""
    key = (heads, kv_heads, head_dim, dtype, str(device), scale, gqa, max_len, torch.get_num_threads())
    hit = _causal_verdicts.get(key)
    if hit is not None:
        return hit
    with _causal_lock:
        hit = _causal_verdicts.get(key)
        if hit is None:
            lengths = [n for n in CAUSAL_CHECK_LENGTHS if n <= max_len] or [max(2, min(max_len, 16))]
            hit = _causal_verdicts[key] = causal_self_check(heads, kv_heads, head_dim, dtype, device, scale, gqa,
                                                           lengths)
            if hit:
                log.info("attention: is_causal SDPA for whole prompt blocks (bit-identical to the mask; %d threads)",
                         key[-1])
            else:
                log.warning("attention: is_causal SDPA is not bit-identical here (torch %s, %s, %s, %d threads); "
                            "keeping the explicit mask", torch.__version__, dtype, device, key[-1])
        return hit


# ---- torch thread scoping (P0) ---------------------------------------------------------------------
def _find_omp_set() -> Callable[[int], None] | None:
    """The OpenMP runtime's ``omp_set_num_threads`` that torch's intra-op parallelism reads, or None.

    It changes the calling thread's thread count only (``torch.set_num_threads`` would also change the
    process default that threads created meanwhile inherit, so other threads' float ops could change
    their thread count). Verified on a scratch thread: setting it moves ``torch.get_num_threads()``
    there and nowhere else. None when torch does not use OpenMP or the symbol is not visible.
    """
    try:
        if "parallel backend: OpenMP" not in torch.__config__.parallel_info():
            return None
        fn = ctypes.CDLL(None).omp_set_num_threads
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_int]
    fn.restype = None
    seen: list[bool] = []

    def probe() -> None:
        before = torch.get_num_threads()  # also runs torch's lazy per-thread thread-count init
        want = 1 if before != 1 else 2
        fn(want)
        seen.append(torch.get_num_threads() == want)
        fn(before)
        seen.append(torch.get_num_threads() == before)

    main_before = torch.get_num_threads()
    th = threading.Thread(target=probe, name="pgw-omp-probe")
    th.start()
    th.join()
    if seen != [True, True] or torch.get_num_threads() != main_before:
        return None
    return fn  # type: ignore[no-any-return,unused-ignore]


_omp_set_cache: list[Callable[[int], None] | None] = []
_omp_lock = threading.Lock()
#: Per thread: ``(model id, thread count before the scope)`` while a decode thread scope is active
#: (:meth:`TransformerModel._step`), else unset; the head restores that count (``_head_threads``).
_scope_tls = threading.local()


def _omp_set() -> Callable[[int], None] | None:
    if not _omp_set_cache:
        with _omp_lock:
            if not _omp_set_cache:
                _omp_set_cache.append(_find_omp_set())
    return _omp_set_cache[0]


#: Cache lengths the thread-scope self-check covers (both sides of the CPU flash kernel's 512-key block).
#: Longer caches are not checked at load (to keep it cheap); spot checks at 6000 and 12000 keys (1
#: and 4 rows, float32 and bfloat16, with and without GQA, 1 vs 4 threads) were equal too.
FLOAT_THREADS_CHECK_LENGTHS = (1, 37, 100, 513, 1030, 2100, 4100)


def float_threads_self_check(model: TransformerModel, threads: int, rows: int = 1, *,
                             lengths: Sequence[int] = FLOAT_THREADS_CHECK_LENGTHS) -> bool:
    """True when the model's float ops on ``rows``-row blocks give the same bits at ``threads`` torch
    threads as at the calling thread's current count (``torch.equal`` on every output).

    Runs the model's own :class:`FloatOps` at its shapes, dtype and norm weights, on inputs cast and
    sliced as the layer does it (float32 linear outputs ``.to(dtype)``, then column slices):
    input/post/final RMSNorm (block and row by row), RoPE of q and k at the positions, the SiLU gate,
    residual adds and attention over cache-like views (GQA form chosen per thread count, as the model
    does) for every cache length in ``lengths``, as a block and as a serial tail. The LM head is not
    covered: its kernel does not use torch threads (the native kernel has its own; the torch
    fallback's element-wise order does not depend on them).

    Element-wise ops are not always thread-count invariant: torch splits a large tensor (above its
    32768-element grain) into one range per thread, and a range edge that is not a multiple of the
    vector width moves elements between the vector and the scalar code (e.g. SiLU at 7B: 2+ rows of
    18944 differ in the last bit). Hence the check per row count.
    """
    setter = _omp_set()
    if setter is None or model.device.type != "cpu" or threads < 1:
        return False
    a = model.arch
    nh, kvh, hd = model.num_heads, model.num_kv_heads, model.head_dim
    inter = model.split_gate_up[0]
    dt = model.dtype
    ops = model.float_ops or _TORCH_OPS
    gen = torch.Generator().manual_seed(0xF7A5)
    max_pos = int(a.max_position_embeddings)
    lens = [n for n in lengths if n <= max_pos] or [1]

    def rand(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(shape, generator=gen) * scale).to(dt)

    def raw(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(shape, generator=gen) * scale  # float32, as the backends return linear outputs

    if rows < 1:
        return False
    nq, nk, _ = model.split_qkv
    # the layer's float work on linear outputs as the layer gets them (float32 (rows, m)), with the
    # same casts and slicing as :meth:`TransformerModel._layer_standard` (so the same memory layouts:
    # strided views for a float32 model, contiguous copies otherwise)
    h, gu, o = rand(rows, a.hidden_size, scale=0.5), raw(rows, 2 * inter, scale=2.0), raw(rows, a.hidden_size)
    caches = []
    for n in lens:
        kbuf, vbuf = rand(2, n + 3, kvh, hd), rand(2, n + 3, kvh, hd)
        caches.append((n, kbuf[1, :n], vbuf[1, :n], {rows: raw(rows, nq + 2 * nk, scale=2.0)} if rows <= n else {}))
    norm_w = (model.input_norms[0], model.post_norms[-1])

    def run() -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for w in norm_w:
                out.append(ops.rms_norm(h, w, model.eps))
            out.append(ops.norm_rows(h, model.final_norm, model.eps))
            out.append(ops.rms_norm(h[-1:], model.final_norm, model.eps).to(torch.float32))
            out.append(ops.silu_mul(gu[:, :inter].to(dt), gu[:, inter:].to(dt)))
            out.append(h + o.to(dt))
            out.append(h + gu[:, : a.hidden_size].to(dt))
            for n, k, v, qs in caches:
                for r, y_raw in qs.items():
                    start = n - r
                    y = y_raw.to(dt)
                    cos, sin = model.rope.cos_sin(start, r, dt)
                    qr = ops.rope(y[:, :nq].reshape(r, nh, hd), cos, sin)
                    out.append(qr)
                    out.append(ops.rope(y[:, nq : nq + nk].reshape(r, kvh, hd), cos, sin))
                    out.append(ops.attention(model, qr, k, v, start))
                    for j in range(r):
                        out.append(ops.attention(model, qr[j : j + 1], k[: start + j + 1], v[: start + j + 1],
                                                 start + j))
        return out

    base = torch.get_num_threads()
    try:
        ref = run()
        setter(threads)
        try:
            got = run()
        finally:
            setter(base)
    except Exception as exc:
        log.debug("thread-scope self-check raised: %r", exc)
        return False
    return len(ref) == len(got) and all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring %s=%r (not an integer)", name, raw)
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ---- hooks ---------------------------------------------------------------------------------------------
class FloatOps:
    """The gateway's plaintext float operations of a layer (WP-B1 plugs a native path in here).

    The default implementation is today's torch code; each method looks the module function up at
    call time (so profilers that wrap ``pgw.engine.model.rms_norm`` and friends still see it).
    ``attention`` gets the model (KV-head grouping, scale and the guarded SDPA forms).
    """

    name = "torch"

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return rms_norm(x, weight, eps)

    def norm_rows(self, h: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Final norm of each row on its own, float32 (rows that plain decoding computes alone)."""
        return _norm_rows(h, weight, eps)

    def rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return apply_rope(x, cos, sin)

    def attention(self, model: TransformerModel, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  start: int) -> torch.Tensor:
        return model._attention(q, k, v, start)

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


#: The default float path (``TransformerModel.float_ops = None`` uses it).
TorchFloatOps = FloatOps
_TORCH_OPS = FloatOps()


class HeadMode:
    """Turns a step's final hidden rows into logits (WP-B2 adds a provider int8 head as a round trip).

    :meth:`steps` is a generator like the forward pass: it may yield lists of
    :class:`pgw.engine.backends.LinearCall` (round trips) and returns the step's per-sequence outputs.
    :meth:`linears` lists those round trips' linears (the engine counts trips per pass with it). The
    default is the gateway's exact float32 head, :meth:`TransformerModel._logits`, with no round trip.
    """

    name = "gateway"

    def linears(self, model: TransformerModel) -> Sequence[LinearId]:
        return ()

    def steps(self, model: TransformerModel, seqs: Sequence[SequenceInput], hs: list[torch.Tensor]) -> _Steps:
        return model._logits(seqs, hs)
        yield  # pragma: no cover - makes this a generator


GatewayHead = HeadMode


class Schedule:
    """The linears of a layer and the float work between them (WP-B4's fused2 replaces it).

    :meth:`layer` is a generator over one layer of a forward pass: it yields lists of
    :class:`pgw.engine.backends.LinearCall`, receives their results, and replaces the items of
    ``st.hs`` (the residual stream per sequence, see :class:`PassState`). :meth:`linears` lists the
    layer's linears in round-trip order. The default is the standard four-trip layer,
    :meth:`TransformerModel._layer_standard`.
    """

    name = "standard"

    def linears(self, model: TransformerModel, layer: int) -> Sequence[LinearId]:
        return model._lids[layer]

    def layer(self, model: TransformerModel, layer: int,
              st: PassState) -> PyGenerator[list[LinearCall], list[torch.Tensor], None]:
        yield from model._layer_standard(layer, st)


StandardSchedule = Schedule


class PassState:
    """State of one forward pass, shared with the layer schedule (per sequence, in batch order).

    Attributes:
        seqs: The pass's sequences.
        ids: Their new token ids.
        caches: Their KV caches.
        starts: Cache lengths before the pass (position of each sequence's first new row).
        rope: ``(cos, sin)`` per sequence.
        hs: Residual stream ``(T, hidden)`` per sequence. A schedule replaces items; it never writes
            into the tensors (captured hidden states keep references to them).
        sessions: Backend session handles.
        ops: The pass's float ops.
    """

    __slots__ = ("caches", "hs", "ids", "ops", "rope", "seqs", "sessions", "starts")

    def __init__(self, seqs: Sequence[SequenceInput], ids: list[torch.Tensor], caches: list[KVCache],
                 starts: list[int], rope: list[tuple[torch.Tensor, torch.Tensor]], hs: list[torch.Tensor],
                 ops: FloatOps):
        self.seqs = seqs
        self.ids = ids
        self.caches = caches
        self.starts = starts
        self.rope = rope
        self.hs = hs
        self.sessions = [s.session for s in seqs]
        self.ops = ops


def _norm_rows(h: torch.Tensor, norm: torch.Tensor, eps: float) -> torch.Tensor:
    """Final norm of each row on its own (as a 1-row decode step would), float32."""
    return torch.cat([rms_norm(h[i : i + 1], norm, eps) for i in range(int(h.shape[0]))]).to(torch.float32)


class RowLogits:
    """Logits of a verify pass; row ``i`` is bit-identical to plain decoding of that position.

    Given ``logits`` (all rows computed in the step's shared LM-head pass, see
    :attr:`TransformerModel.batch_lm_head`) it only indexes them; otherwise each row is computed on
    first access (rows after a rejected draft are never computed). Either way a row is the
    deterministic product of :func:`pgw.engine.lm_head.lm_head_logits`, whose rows do not depend on
    how many rows share a pass.
    """

    def __init__(self, h: torch.Tensor, norm: torch.Tensor, eps: float, lm_head: torch.Tensor,
                 logits: torch.Tensor | None = None,
                 norm_rows: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None):
        self._h = h
        self._norm = norm
        self._eps = eps
        self._lm_head = lm_head
        self._logits = logits
        self._norm_rows = norm_rows  # None: the module's _norm_rows (looked up when a row is computed)
        self._rows: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return int(self._h.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        i = int(i) if i >= 0 else len(self) + int(i)
        if self._logits is not None:
            return self._logits[i]
        hit = self._rows.get(i)
        if hit is None:
            norm_rows = self._norm_rows if self._norm_rows is not None else _norm_rows
            hn = norm_rows(self._h[i : i + 1], self._norm, self._eps)
            hit = self._rows[i] = lm_head_logits(hn, self._lm_head)[0]
        return hit


class TransformerModel:
    """Plaintext transformer runtime.

    Per layer: ``h = rmsnorm(x)`` -> ``qkv`` (backend) -> split q/k/v (row order from manifest
    ``parts``) -> RoPE (incl. Llama-3 ``rope_scaling``) -> causal GQA attention over the KV cache ->
    ``o`` (backend) -> residual -> rmsnorm -> ``gate_up`` (backend) -> ``silu(gate) * up`` ->
    ``down`` (backend) -> residual. Then final norm and LM head (plaintext, float).

    Prefill sends all prompt tokens of a layer in one backend call (one round trip per linear).
    :meth:`forward_batch` runs several sequences in lockstep so that each linear group is one
    ``forward_many`` (one round trip) for all of them.

    Args:
        pack: Open :class:`pgw.modelpack.ModelPack` or :class:`pgw.engine.pack.EnginePack`
            (plain tensors are loaded; linears are not).
        backend: Linear backend.
        device: Device for plaintext ops.
        dtype: Compute dtype for plaintext ops (float32 default; logits always float32).
    """

    def __init__(self, pack: Any, backend: LinearBackend, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.pack: Any = pack
        self.backend = backend
        self.device = torch.device(device)
        self.dtype = dtype
        a = self.pack.arch
        self.arch = a
        if ModelArch(a.arch) not in (ModelArch.QWEN2, ModelArch.LLAMA):  # pragma: no cover - enum-limited
            raise ConfigError(f"unsupported architecture {a.arch}")
        if a.num_heads % a.num_kv_heads:
            raise ConfigError("num_heads must be a multiple of num_kv_heads")
        self.num_layers = a.num_layers
        self.num_heads = a.num_heads
        self.num_kv_heads = a.num_kv_heads
        self.head_dim = a.head_dim
        self.eps = a.rms_norm_eps
        self.vocab_size = a.vocab_size
        self.split_qkv = linear_split_sizes(a, LinearKind.QKV)
        self.split_gate_up = linear_split_sizes(a, LinearKind.GATE_UP)
        self._check_parts()

        load = self.pack.load_plain
        self.embed_tokens = load("embed_tokens", dtype=dtype, device=self.device)
        self.final_norm = load("final_norm", dtype=dtype, device=self.device)
        # On a GPU the deterministic head would be the per-chunk torch reference (about 17k kernel
        # launches per pass); the native CPU kernel gives the same bits much faster, so the head
        # stays in host memory there (normalized rows are copied over; logits come back on the CPU).
        head_dev = self.device
        if self.device.type != "cpu" and native_lm_head():
            head_dev = torch.device("cpu")
            log.warning("model on %s: the LM head stays on the CPU (deterministic native kernel; same "
                        "logits, bit for bit)", self.device)
        self._head: torch.Tensor
        if a.tie_word_embeddings and dtype == torch.float32 and head_dev == self.device:
            self._head = self.embed_tokens
        else:
            self._head = load("lm_head", dtype=torch.float32, device=head_dev)
        self._head = self._head.contiguous()  # no-op normally; the native LM head reads it in place
        # bfloat16 checkpoints: keep the head in bfloat16 when that is lossless (same logits, bit for
        # bit; half the bytes read per token). A tied head keeps its float32 embedding for lookups.
        self._head = compact_head(self._head)
        #: Optional replacement for ``h @ lm_head^T`` (draft models only: e.g. an int8 product).
        #: Without it the float32 head is :func:`pgw.engine.lm_head.lm_head_logits`: a fixed
        #: summation order, so each row's logits are bit-identical however rows are grouped.
        self.lm_head_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        #: One LM-head pass for all rows of a step (the last positions of every sequence and all
        #: rows of speculative verify passes): the float32 head is read once per step instead of
        #: once per row. Exact either way (the deterministic product's rows do not depend on the
        #: batch), so this is only a speed choice: on by default with the native kernel (one pass
        #: costs about one row); off with the torch fallback, whose cost grows with the rows (verify
        #: rows then stay lazy: rows after a rejected draft are never computed).
        self.batch_lm_head = native_lm_head()
        layers = range(a.num_layers)
        self.input_norms = [load(f"layers.{i}.input_norm", dtype=dtype, device=self.device) for i in layers]
        self.post_norms = [load(f"layers.{i}.post_attn_norm", dtype=dtype, device=self.device) for i in layers]
        if tuple(self.embed_tokens.shape) != (a.vocab_size, a.hidden_size):
            raise ShapeError(f"embed_tokens shape {tuple(self.embed_tokens.shape)} does not match the architecture")
        self.rope = RotaryEmbedding(a.head_dim, a.rope_theta, a.rope_scaling, a.max_position_embeddings, self.device)
        self.scale = a.head_dim**-0.5
        # (thread count, verdict): one tuple, replaced whole, so a thread never reads a verdict that
        # belongs to another thread's count (no lock needed; a lost update only re-reads the cache)
        self._gqa_state: tuple[int, bool] = (-1, False)
        self._use_gqa()  # the GQA self-check runs at load (cached per process; again if threads change)
        #: Single worker thread for the plaintext parts of async forward passes (see run_plain).
        self._plain_pool: ThreadPoolExecutor | None = None
        self._plain_lock = threading.Lock()
        #: run_plain calls submitted and not yet completed (as seen from the event loop).
        self._plain_pending = 0
        # LinearId objects per layer (hot path: four per layer per forward)
        self._lids = [tuple(LinearId(i, k) for k in (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP,
                                                       LinearKind.DOWN)) for i in range(a.num_layers)]
        # ---- hooks (None = today's code; see the module docstring) ----
        #: Plaintext float ops of a layer (None: :class:`TorchFloatOps`).
        self.float_ops: FloatOps | None = None
        #: Logits from the final hidden rows (None: the exact gateway head, :meth:`_logits`).
        self.head_mode: HeadMode | None = None
        #: Linears of a layer and the float work between them (None: :meth:`_layer_standard`).
        self.schedule: Schedule | None = None
        # ---- speed settings that keep the bits ----
        #: Decode passes run with this many torch threads; set only through
        #: :meth:`enable_float_threads` (self-checked). None: the caller's thread count, as before.
        self._float_threads: int | None = None
        self._float_rows = 0  # passes of at most this many rows per sequence are scoped (self-checked)
        self._float_base = torch.get_num_threads()  # thread count the self-check compared against
        self._causal_state: tuple[int, bool, bool] = (-1, False, False)  # (threads, GQA form, verdict)
        if self.causal_sdpa:
            # the is_causal self-check runs at load, draft models included (cached per process, shape and
            # thread count: about 1 s for 7B shapes on 4 vCPUs); a pass at another thread count (e.g. a
            # prefill on a thread with a different count) runs it once more, mid-pass, then it is cached
            self._use_causal()
        self.sync_decode = _env_bool("PGW_SYNC_DECODE", type(self).sync_decode)
        env_threads = _env_int("PGW_FLOAT_THREADS")
        if env_threads is not None and env_threads > 0:
            self.enable_float_threads(env_threads)
        log.info(
            "engine model ready: arch=%s layers=%d hidden=%d heads=%d/%d vocab=%d backend=%s",
            a.arch, a.num_layers, a.hidden_size, a.num_heads, a.num_kv_heads, a.vocab_size, type(backend).__name__,
        )

    @property
    def lm_head(self) -> torch.Tensor:
        """The LM head ``(vocab, hidden)``.

        Normally the tensor the model uses. When the model runs on a GPU and the head is kept on the
        CPU (see ``__init__``), this returns a float32 copy on the model's device instead, so callers
        that build their own product on that device (e.g. a float16 GPU draft head) keep working.
        """
        if self._head.device.type == "cpu" and self.device.type != "cpu":
            return self._head.to(device=self.device, dtype=torch.float32)
        return self._head

    @lm_head.setter
    def lm_head(self, value: torch.Tensor) -> None:
        self._head = value

    def _check_parts(self) -> None:
        """Cross-check derived split sizes against manifest ``parts`` when the pack has a manifest."""
        manifest = getattr(self.pack, "manifest", None)
        if manifest is None or not hasattr(manifest, "linear"):
            return
        for kind, sizes in ((LinearKind.QKV, self.split_qkv), (LinearKind.GATE_UP, self.split_gate_up)):
            try:
                entry = manifest.linear(LinearId(0, kind))
            except KeyError as exc:
                raise ConfigError(f"manifest lacks linear 0.{kind.value}") from exc
            got = tuple(int(p.rows) for p in entry.parts)
            starts = tuple(int(p.row_start) for p in entry.parts)
            exp_starts = tuple(sum(sizes[:i]) for i in range(len(sizes)))
            if got != sizes or starts != exp_starts:
                raise ConfigError(f"manifest parts of {kind.value} {got}@{starts} do not match architecture {sizes}")

    # ---- caches --------------------------------------------------------------------------------
    def new_cache(self, max_len: int | None = None) -> KVCache:
        """Empty KV cache sized for this model (``max_len`` defaults to the model context length)."""
        cap = max_len if max_len is not None else self.arch.max_position_embeddings
        return KVCache(self.num_layers, self.num_kv_heads, self.head_dim, cap, self.dtype, self.device,
                       initial_len=min(cap, 256))

    # ---- forward -------------------------------------------------------------------------------
    def _ids(self, ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        t = torch.as_tensor(ids, dtype=torch.long).reshape(-1).to(self.device)
        if t.numel() == 0:
            raise ShapeError("forward needs at least one token")
        if int(t.min()) < 0 or int(t.max()) >= self.vocab_size:
            raise ShapeError(f"token id out of range [0, {self.vocab_size})")
        return t

    #: Use SDPA's ``enable_gqa`` (no ``repeat_interleave`` copy of the cache per layer) where a
    #: per-process self-check shows it gives the same bits; False always keeps the old form.
    gqa_sdpa = True

    def _use_gqa(self) -> bool:
        threads = torch.get_num_threads()
        state = self._gqa_state
        if threads != state[0]:  # verdicts are per thread count (kernels may split by it)
            state = (threads, bool(self.num_heads != self.num_kv_heads and _gqa_verdict(
                self.num_heads, self.num_kv_heads, self.head_dim, self.dtype, self.device, self.scale)))
            self._gqa_state = state
        return state[1] and self.gqa_sdpa

    #: SDPA's ``is_causal`` for a whole prompt block from position 0 of at least ``causal_min_rows``
    #: rows (the CPU flash kernel then skips masked key blocks: about a third less attention time at
    #: 500 rows) where a per-process self-check shows it gives the same bits as the explicit mask.
    #: False always keeps the mask. ``PGW_CAUSAL_SDPA=0`` turns it off.
    causal_sdpa = _env_bool("PGW_CAUSAL_SDPA", True)
    causal_min_rows = 16

    def _use_causal(self) -> bool:
        threads = torch.get_num_threads()
        groups = self.num_heads // self.num_kv_heads
        gqa = groups > 1 and self._use_gqa()
        state = self._causal_state
        if threads != state[0] or gqa != state[1]:
            state = (threads, gqa, _causal_verdict(self.num_heads, self.num_kv_heads, self.head_dim, self.dtype,
                                                   self.device, self.scale, gqa,
                                                   int(self.arch.max_position_embeddings)))
            self._causal_state = state
        return state[2]

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int) -> torch.Tensor:
        """Causal GQA attention of ``q (T, H, hd)`` over ``k, v (L, KVH, hd)``; returns ``(T, H*hd)``."""
        groups = self.num_heads // self.num_kv_heads
        gqa = groups > 1 and self._use_gqa()
        t = int(q.shape[0])
        if (start == 0 and t >= self.causal_min_rows and int(k.shape[0]) == t and self.causal_sdpa
                and self._use_causal()):
            return _attend(q, k, v, start, self.scale, groups, gqa, causal=True)
        return _attend(q, k, v, start, self.scale, groups, gqa)

    # ---- torch thread scoping --------------------------------------------------------------------
    #: Largest pass (new rows per sequence: 1-row decode, speculative verify passes) that
    #: :meth:`enable_float_threads` checks for scoping; it scopes the row counts 1..R that all pass.
    float_threads_max_rows = 8

    @property
    def float_threads(self) -> int | None:
        """Torch threads of decode float ops (None: the caller's thread count, as before)."""
        return self._float_threads

    @property
    def float_threads_rows(self) -> int:
        """Passes of at most this many new rows per sequence use :attr:`float_threads` (0: none)."""
        return self._float_rows

    def enable_float_threads(self, threads: int | None) -> bool:
        """Run decode float ops with ``threads`` torch threads, if that gives the same bits.

        Runs :func:`float_threads_self_check` (this model's float ops at its shapes, at ``threads`` and
        at the calling thread's count, ``torch.equal``) for 1, 2, ... rows up to
        :attr:`float_threads_max_rows`; passes of up to the last row count that passed (and all below
        it) are scoped (:attr:`float_threads_rows`). If even 1 row fails, without a thread-local
        OpenMP setter, or on a non-CPU device it keeps today's thread count and returns False.
        ``None`` or 0 turns it off. Only the running thread's OpenMP count changes, and only for the
        float segments of those passes (the masking path's finish/prepare and the LM head kernel are
        outside them).
        """
        self._float_threads, self._float_rows = None, 0
        if threads is None or threads <= 0:
            return True
        threads = int(threads)
        base = torch.get_num_threads()
        if base == threads:
            log.info("float threads: already %d torch threads; nothing to scope", threads)
            return True
        ok_rows = 0
        for r in range(1, max(0, int(self.float_threads_max_rows)) + 1):
            if not float_threads_self_check(self, threads, r):
                break
            ok_rows = r
        if ok_rows:
            self._float_threads, self._float_rows, self._float_base = threads, ok_rows, base
            log.info("float threads: passes of up to %d rows per sequence run their float ops on %d torch threads "
                     "(bit-identical to %d; self-checked)", ok_rows, threads, base)
            return True
        log.warning("float threads: %d torch threads unavailable or not bit-identical to %d here; keeping %d",
                    threads, base, base)
        return False

    def _step(self, gen: _Steps, value: list[torch.Tensor] | None, rows: int) -> tuple[bool, Any]:
        """:meth:`_advance` with the decode thread count applied (``rows``: new rows per sequence).

        Only on a thread that runs at the count the self-check compared against (``_float_base``):
        a thread at any other count (e.g. ``torch.set_num_threads`` called after load) keeps its own
        count, since the scoped bits were never compared with that count's.
        """
        n = self._float_threads
        if n is None or rows > self._float_rows:
            return self._advance(gen, value)
        prev = torch.get_num_threads()
        setter = _omp_set()
        if prev != self._float_base or prev == n or setter is None:
            return self._advance(gen, value)
        setter(n)
        outer = getattr(_scope_tls, "scope", None)
        _scope_tls.scope = (id(self), prev)
        try:
            return self._advance(gen, value)
        finally:
            _scope_tls.scope = outer
            setter(prev)

    @staticmethod
    def _pass_rows(seqs: Sequence[SequenceInput]) -> int:
        """Largest number of new rows of one sequence in a pass."""
        return max((len(s.input_ids) for s in seqs), default=0)

    # ---- local linears (exact, on the gateway) ---------------------------------------------------
    def _local_results(self, calls: list[LinearCall]) -> list[torch.Tensor] | None:
        """Results of a step's calls when the backend computes that linear locally, else None.

        A step's calls are one linear for every sequence, so they are local or remote together.
        """
        is_local = getattr(self.backend, "local_linear", None)
        if is_local is None or not calls or not is_local(calls[0].id):
            return None
        if any(c.id != calls[0].id for c in calls):  # pragma: no cover - schedules yield one linear per step
            raise ShapeError("a step mixes linears; local linears need one linear per step")
        rows_ok = getattr(self.backend, "local_rows_ok", None)
        if rows_ok is not None and not rows_ok(sum(int(c.x.shape[0]) for c in calls)):
            return None  # a large (prefill) pass sends every linear (R1: local_max_rows)
        return self._check_results(calls, self.backend.forward_local(calls))  # type: ignore[attr-defined]

    def _advance_local(self, gen: _Steps, value: list[torch.Tensor] | None, rows: int) -> tuple[bool, Any]:
        """:meth:`_step`, then step past calls the backend computes locally (no network hop)."""
        finished, out = self._step(gen, value, rows)
        while not finished:
            res = self._local_results(out)
            if res is None:
                break
            finished, out = self._step(gen, res, rows)
        return finished, out

    def _trip_plan(self) -> tuple[int, bool]:
        """``(remote round trips per pass, any local linear)`` for the current hooks and backend."""
        is_local = getattr(self.backend, "local_linear", None)
        lids: list[LinearId] = []
        for layer in range(self.num_layers):
            lids.extend(self.schedule.linears(self, layer) if self.schedule is not None else self._lids[layer])
        if self.head_mode is not None:
            lids.extend(self.head_mode.linears(self))
        local = 0 if is_local is None else sum(1 for lid in lids if is_local(lid))
        return len(lids) - local, local > 0

    def eagle_capture_layers(self) -> tuple[int, int, int]:
        """Low, middle and high layer whose outputs an EAGLE-3-style drafter reads (``capture_hidden``)."""
        n = self.num_layers
        return (min(2, n - 1), n // 2, max(n - 3, 0))

    def _steps(self, seqs: Sequence[SequenceInput]) -> _Steps:
        """Forward coroutine: yields backend call lists, receives their results, returns logits."""
        if not seqs:
            return []
        ids = [self._ids(s.input_ids) for s in seqs]
        caches = [s.cache for s in seqs]
        if len({id(c) for c in caches}) != len(caches):
            raise ConfigError("each sequence in a batch needs its own KV cache")
        starts = [c.length for c in caches]
        for t, c in zip(ids, caches, strict=True):
            if c.remaining < t.numel():
                raise ContextOverflowError(
                    f"sequence would exceed the context capacity ({c.length} + {t.numel()} > {c.max_len})",
                    details={"length": c.length, "new": int(t.numel()), "max_len": c.max_len},
                )
            c.reserve(c.length + t.numel())
        rope = [self.rope.cos_sin(st, t.numel(), self.dtype) for st, t in zip(starts, ids, strict=True)]
        st = PassState(seqs, ids, caches, starts, rope, [self.embed_tokens[t] for t in ids],
                       self.float_ops or _TORCH_OPS)
        # hidden states for drafters: {sequence index: (wanted layers, captured)}
        capture: dict[int, tuple[frozenset[int], dict[int, torch.Tensor]]] = {
            si: (frozenset(int(x) for x in s.capture_hidden), {}) for si, s in enumerate(seqs) if s.capture_hidden}
        if capture:
            self._capture(capture, st, -1)
        sched = self.schedule
        for layer in range(self.num_layers):
            if sched is None:
                yield from self._layer_standard(layer, st)
            else:
                yield from sched.layer(self, layer, st)
            if capture:
                self._capture(capture, st, layer)
        for si, (_want, got) in capture.items():
            seqs[si].hidden = got
        for t, c in zip(ids, caches, strict=True):
            c.advance(t.numel())
        if self.head_mode is None:
            return self._logits(seqs, st.hs)
        return (yield from self.head_mode.steps(self, seqs, st.hs))

    @staticmethod
    def _capture(capture: dict[int, tuple[frozenset[int], dict[int, torch.Tensor]]], st: PassState,
                 layer: int) -> None:
        for si, (want, got) in capture.items():
            if layer in want:
                got[layer] = st.hs[si]  # never written into later (layers replace the list items)

    def _layer_standard(self, layer: int, st: PassState) -> PyGenerator[list[LinearCall], list[torch.Tensor], None]:
        """One layer, four round trips: qkv, o, gate_up, down (the default :class:`Schedule`)."""
        ops = st.ops
        hs, sessions, caches, starts, seqs = st.hs, st.sessions, st.caches, st.starts, st.seqs
        nh, kvh, hd = self.num_heads, self.num_kv_heads, self.head_dim
        nq, nk, _ = self.split_qkv
        lid_qkv, lid_o, lid_gu, lid_down = self._lids[layer]
        normed = [ops.rms_norm(h, self.input_norms[layer], self.eps) for h in hs]
        qkv = yield [LinearCall(lid_qkv, x, sess) for x, sess in zip(normed, sessions, strict=True)]
        attn_out = []
        for si, y_raw in enumerate(qkv):
            y = y_raw.to(self.dtype)
            rows = int(y.shape[0])
            q = y[:, :nq].reshape(rows, nh, hd)
            k = y[:, nq : nq + nk].reshape(rows, kvh, hd)
            v = y[:, nq + nk :].reshape(rows, kvh, hd)
            cos, sin = st.rope[si]
            q = ops.rope(q, cos, sin)
            k = ops.rope(k, cos, sin)
            k_all, v_all = caches[si].append(layer, k, v)
            tail = min(int(seqs[si].serial_tail), rows)
            if tail <= 0:
                attn_out.append(ops.attention(self, q, k_all, v_all, starts[si]))
            else:  # the head as one block, then each tail row alone (as plain decoding would)
                st0, head = starts[si], rows - tail
                parts = [ops.attention(self, q[:head], k_all[: st0 + head], v_all[: st0 + head], st0)] if head else []
                for j in range(head, rows):
                    parts.append(ops.attention(self, q[j : j + 1], k_all[: st0 + j + 1], v_all[: st0 + j + 1],
                                               st0 + j))
                attn_out.append(torch.cat(parts, dim=0))
        o = yield [LinearCall(lid_o, x, sess) for x, sess in zip(attn_out, sessions, strict=True)]
        hs = [h + y.to(self.dtype) for h, y in zip(hs, o, strict=True)]
        normed = [ops.rms_norm(h, self.post_norms[layer], self.eps) for h in hs]
        gu = yield [LinearCall(lid_gu, x, sess) for x, sess in zip(normed, sessions, strict=True)]
        inter = self.split_gate_up[0]
        acts = [ops.silu_mul(y[:, :inter].to(self.dtype), y[:, inter:].to(self.dtype)) for y in gu]
        down = yield [LinearCall(lid_down, x, sess) for x, sess in zip(acts, sessions, strict=True)]
        st.hs = [h + y.to(self.dtype) for h, y in zip(hs, down, strict=True)]

    def _logits(self, seqs: Sequence[SequenceInput], hs: list[torch.Tensor]) -> list[Any]:
        """Final norm and LM head of a step: per sequence the last row's logits ``(vocab,)``, all rows'
        ``(T, vocab)`` (``all_logits``) or :class:`RowLogits` (``last_logits``)."""
        ops = self.float_ops or _TORCH_OPS
        norm_rows = None if self.float_ops is None else ops.norm_rows  # RowLogits: None = module default
        out: list[Any] = []
        if self.lm_head_fn is not None:  # approximate head (draft models): one product per sequence
            for s, h in zip(seqs, hs, strict=True):
                if s.last_logits > 0:
                    out.append(RowLogits(h[-s.last_logits :], self.final_norm, self.eps, self._head,
                                         norm_rows=norm_rows))
                    continue
                hn = ops.rms_norm(h if s.all_logits else h[-1:], self.final_norm, self.eps).to(torch.float32)
                with self._head_threads():
                    logits = self.lm_head_fn(hn)
                out.append(logits if s.all_logits else logits[0])
            return out
        # Normalized rows per sequence (rows that plain decoding computes alone are normed alone);
        # None: verify rows left to RowLogits (computed lazily when the pass is not shared).
        shared = self.batch_lm_head
        hns: list[torch.Tensor | None] = []
        for s, h in zip(seqs, hs, strict=True):
            if s.last_logits > 0:
                hns.append(ops.norm_rows(h[-s.last_logits :], self.final_norm, self.eps) if shared else None)
            else:
                hns.append(ops.rms_norm(h if s.all_logits else h[-1:], self.final_norm, self.eps).to(torch.float32))
        given = [x for x in hns if x is not None]
        logits_of: list[torch.Tensor | None] = []
        with self._head_threads():
            if shared and len(given) > 1:  # one pass over the head for every row of the step
                flat = lm_head_logits(torch.cat(given), self._head)
                at = 0
                for x in hns:
                    n = 0 if x is None else int(x.shape[0])
                    logits_of.append(None if x is None else flat[at : at + n])
                    at += n
            else:
                logits_of = [None if x is None else lm_head_logits(x, self._head) for x in hns]
        for s, h, lg in zip(seqs, hs, logits_of, strict=True):
            if s.last_logits > 0:
                out.append(RowLogits(h[-s.last_logits :], self.final_norm, self.eps, self._head, lg,
                                     norm_rows=norm_rows))
            elif lg is None:  # pragma: no cover - every other sequence has normalized rows
                raise AssertionError("missing logits")
            else:
                out.append(lg if s.all_logits else lg[0])
        return out

    def _head_native(self) -> bool:
        """Whether :func:`lm_head_logits` computes this model's head with the native kernel (which
        does not use torch threads); mirrors its dispatch (a bfloat16 head on a build without
        ``lm_head_bf16`` falls back to the torch reference)."""
        if self.lm_head_fn is not None or not native_lm_head() or self._head.device.type != "cpu":
            return False
        if self._head.dtype == torch.float32:
            return True
        return self._head.dtype == torch.bfloat16 and native.native_available("lm_head_bf16")

    @contextlib.contextmanager
    def _head_threads(self) -> Iterator[None]:
        """Inside this model's decode thread scope (:meth:`_step`): the thread's count from before the
        scope back for a torch-computed head (the native head kernel needs nothing)."""
        scope = getattr(_scope_tls, "scope", None) if self._float_threads is not None else None
        setter = _omp_set() if scope is not None and scope[0] == id(self) else None
        if setter is None or scope is None or self._head_native():
            yield
            return
        cur = torch.get_num_threads()
        setter(scope[1])
        try:
            yield
        finally:
            setter(cur)

    @staticmethod
    def _check_results(calls: list[LinearCall], results: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(results) != len(calls):
            raise ShapeError(f"backend returned {len(results)} results for {len(calls)} calls")
        for c, r in zip(calls, results, strict=True):
            if r.ndim != 2 or r.shape[0] != c.x.shape[0]:
                raise ShapeError(f"{c.id}: backend returned shape {tuple(r.shape)} for {c.x.shape[0]} rows")
        return results

    def forward_batch(self, seqs: Sequence[SequenceInput]) -> list[torch.Tensor]:
        """Run several sequences in lockstep; one ``forward_many`` per linear group for all of them."""
        gen = self._steps(seqs)
        rows = self._pass_rows(seqs)
        try:
            finished, out = self._advance_local(gen, None, rows)
            while not finished:
                res = self._check_results(out, self.backend.forward_many(out))
                finished, out = self._advance_local(gen, res, rows)
            return out  # type: ignore[no-any-return]
        finally:
            gen.close()  # drop references to activations promptly on failure

    # ---- off-loop execution ----------------------------------------------------------------------
    def _plain_executor(self) -> ThreadPoolExecutor:
        with self._plain_lock:
            if self._plain_pool is None:
                self._plain_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-engine-plain")
            return self._plain_pool

    async def run_plain(self, fn: Callable[..., _T], *args: Any) -> _T:
        """Run ``fn(*args)`` on this model's dedicated plaintext worker thread.

        Embedding, norms, RoPE, attention over the KV cache, the LM head GEMM and sampling are
        CPU/GPU-heavy (the LM head alone is ``hidden x vocab`` per sequence); running them on the
        event loop would stall every stream and the health probes. torch releases the GIL, so the
        loop stays responsive. One thread per model serializes access to that model's KV caches.

        If the awaiting task is cancelled, the call still waits for the running function to finish
        (it mutates KV caches and sequence state that the canceller may release next).
        """
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._plain_executor(), functools.partial(fn, *args))
        self._plain_pending += 1
        fut.add_done_callback(self._plain_done)
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            if not fut.done():
                try:
                    await asyncio.wait({fut})
                finally:
                    if fut.done() and not fut.cancelled():
                        fut.exception()  # mark retrieved
            raise

    def _plain_done(self, _fut: Any) -> None:
        self._plain_pending -= 1

    def close(self) -> None:
        """Stop the plaintext worker thread (idempotent; it is recreated on next use)."""
        with self._plain_lock:
            pool, self._plain_pool = self._plain_pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    @staticmethod
    def _advance(gen: _Steps, value: list[torch.Tensor] | None) -> tuple[bool, Any]:
        """One step of the forward coroutine; ``(finished, calls_or_logits)`` (no StopIteration leaks)."""
        try:
            return False, (next(gen) if value is None else gen.send(value))
        except StopIteration as stop:
            return True, stop.value

    #: Run a decode pass (at most ``sync_decode_max_rows`` new rows per sequence) as one call on the
    #: plaintext thread whose round trips are the backend's synchronous ``forward_many`` (no event-loop
    #: hop per trip), when the backend says such a trip is fast (``backend.sync_trip_ok(rows)``, e.g.
    #: over the native trip lane). Same calls in the same order as the split-hop path, so the same masks
    #: and bits. Off by default (``PGW_SYNC_DECODE=1`` turns it on).
    #: Contract for a backend that offers ``sync_trip_ok`` (WP-A4b): its ``forward_many`` is then called
    #: from the plaintext worker thread while the event loop keeps serving other sessions, so it must
    #: not touch the event loop (no ``asyncio.run``, no loop re-entry or loop-bound objects) and must
    #: be safe alongside the backend's async calls of other passes.
    #: Not thread-safe for several plaintext threads: the GQA and ``is_causal`` verdicts are cached per
    #: model (one tuple, replaced whole); a per-session plaintext pool would make that per thread.
    sync_decode = False
    sync_decode_max_rows = 8

    def _sync_pass(self, seqs: Sequence[SequenceInput], backend: Any) -> bool:
        if not self.sync_decode or not seqs:
            return False
        ok = getattr(backend, "sync_trip_ok", None)
        if ok is None or self._pass_rows(seqs) > self.sync_decode_max_rows:
            return False
        return bool(ok(sum(len(s.input_ids) for s in seqs)))

    async def aforward_batch(self, seqs: Sequence[SequenceInput]) -> list[torch.Tensor]:
        """Async :meth:`forward_batch` using :meth:`LinearBackend.aforward_many`.

        The plaintext segments between backend calls run on :meth:`run_plain`'s worker thread, so
        the event loop only awaits I/O and never executes model math. With :attr:`sync_decode` and a
        backend offering fast synchronous trips, a decode pass runs wholly on that thread.
        """
        backend = self.backend
        if self._sync_pass(seqs, backend):
            return await self.run_plain(self.forward_batch, seqs)
        gen = self._steps(seqs)
        rows = self._pass_rows(seqs)
        if getattr(backend, "split_hop", False):
            try:
                new_rows = sum(len(s.input_ids) for s in seqs)
                if not self._inline_safe(seqs, backend):
                    new_rows = -1  # no inline hops in this pass
                return await self._aforward_split(gen, backend, new_rows, rows)
            finally:
                gen.close()
        try:
            finished, out = await self.run_plain(self._advance_local, gen, None, rows)
            while not finished:
                calls = out
                res = self._check_results(calls, await self.backend.aforward_many(calls))
                finished, out = await self.run_plain(self._advance_local, gen, res, rows)
            return out  # type: ignore[no-any-return]
        finally:
            gen.close()  # drop references to activations promptly on failure

    #: Round trips with at least this many rows in calls of 2+ rows (prefill) are split in
    #: ``pipeline_blocks`` blocks that overlap: the gateway masks block j+1 while the service computes
    #: block j, and verifies block j while the service computes the next. Rows are independent, so
    #: results are identical. 1-row calls in the same round trip (decode steps of other streams) go
    #: wholly into the first block. Each session's masks are taken in row order (block 0 first).
    pipeline_min_rows = 16
    #: Blocks per pipelined round trip (2: halves, as before; 4: quarters for long prompts).
    pipeline_blocks = 2

    #: Round trips of at most this many rows (decode) run their local half (finish the previous
    #: calls, the plaintext step, prepare the next calls) directly on the event loop when the
    #: plaintext thread is idle, instead of handing it to that thread: two thread switches less per
    #: round trip. The step that computes the LM head always runs on the thread. 0 disables.
    inline_max_rows = 8

    def _inline_safe(self, seqs: Sequence[SequenceInput], backend: Any) -> bool:
        """True when no mask operation of this pass can block for long, so its small hops may run on
        the event loop: every reservation already served a pass (its current extent's check vectors
        and keys are cached) and has the rows of this pass in its current extent (no window extension,
        which is a durable store transaction, and no first use of another segment)."""
        if not self._lids:
            return False
        # the first linear that takes masks (with local qkv / o that is layer 0's gate_up: a local
        # linear's rows are never taken, so it would never count as "already served a pass")
        is_local = getattr(backend, "local_linear", None)
        first = next((lid for lid in self._lids[0] if not (callable(is_local) and is_local(lid))), self._lids[0][0])
        for s in seqs:
            res = s.session if s.session is not None else getattr(backend, "mask_source", None)
            if res is None:
                continue
            until = getattr(res, "rows_until_boundary", None)
            taken = getattr(res, "taken", None)
            if until is None or taken is None:
                return False
            try:
                if int(taken(first)) <= 0 or int(until(first)) < len(s.input_ids):
                    return False
            except Exception:  # any doubt: use the worker thread
                return False
        return True

    @staticmethod
    def _split_blocks(calls: Sequence[LinearCall], blocks: int) -> tuple[list[list[LinearCall]], list[list[int]]]:
        """Calls of 2+ rows split in ``blocks`` row ranges (``n*j//blocks``; 2 blocks: ``n//2``, as the
        halves always were); 1-row calls wholly in block 0. Returns the non-empty blocks and, per block,
        the position in ``calls`` of each of its items."""
        parts: list[list[LinearCall]] = [[] for _ in range(blocks)]
        where: list[list[int]] = [[] for _ in range(blocks)]
        for i, c in enumerate(calls):
            n = int(c.x.shape[0])
            if n < 2:
                parts[0].append(c)
                where[0].append(i)
                continue
            for j in range(blocks):
                a, b = n * j // blocks, n * (j + 1) // blocks
                if b > a:
                    parts[j].append(LinearCall(c.id, c.x[a:b], c.session, c.meta))
                    where[j].append(i)
        keep = [j for j in range(blocks) if parts[j]]
        return [parts[j] for j in keep], [where[j] for j in keep]

    async def _aforward_split(self, gen: _Steps, backend: Any, new_rows: int = 1 << 30,
                              pass_rows: int = 1 << 30) -> list[torch.Tensor]:
        """One plaintext-thread hop per round trip: finish(previous) + model step + prepare(next).

        The plaintext segment between two linears only needs the previous results, so verifying and
        unmasking them, running the step and masking the next calls happen in one worker call; the
        event loop only awaits the network. Large (prefill) round trips are pipelined in blocks.
        Linears the backend computes locally are stepped past inside the hop (no network hop).
        Any prepared-but-unsent request is wiped on failure.
        """
        owned: list[Any] = []  # prepared requests owned by this coroutine (wiped on failure)
        big = self.pipeline_min_rows
        nblocks = max(2, int(self.pipeline_blocks))

        def rows(calls: Sequence[LinearCall]) -> int:
            return sum(int(c.x.shape[0]) for c in calls)

        def finish(prev: Any, response: Any, remote_s: float) -> list[torch.Tensor]:
            owned.remove(prev)  # finish_prepared always wipes prev
            return self._check_results(list(prev.calls), backend.finish_prepared(prev, response, remote_s))

        def prepare(calls: Sequence[LinearCall]) -> Any:
            prep = backend.prepare_calls(calls)
            owned.append(prep)
            return prep

        def hop(prev: Any, response: Any, remote_s: float,
                extra: tuple[list[list[torch.Tensor]], list[int]] | None) -> tuple[bool, Any]:
            res = None
            if prev is not None:
                res = finish(prev, response, remote_s)
                if extra is not None:  # last block of a pipelined round trip: rejoin by call position
                    got, last = extra
                    if len(res) != len(last):  # pragma: no cover - finish() checked the block's calls
                        raise ShapeError(f"last block returned {len(res)} results for {len(last)} calls")
                    for j, i in enumerate(last):
                        got[i].append(res[j])
                    res = [g[0] if len(g) == 1 else torch.cat(g, dim=0) for g in got]
            finished, out = self._advance_local(gen, res, pass_rows)
            if finished:
                return True, out
            # Calls of 2+ rows are split in blocks; 1-row calls (decode next to prefills) go wholly
            # into the first block. Pipelined only when the split rows are many enough to pay.
            if big and sum(n for n in (int(c.x.shape[0]) for c in out) if n >= 2) >= big:
                return False, ("split", out)
            return False, prepare(out)

        last_rt, has_local = self._trip_plan()  # the hop after `last_rt` round trips runs the LM head
        # local linears are computed inside a hop: never on the event loop
        inline_rows = int(self.inline_max_rows) if new_rows >= 0 and not has_local else 0
        rt = 0
        try:
            if inline_rows and new_rows <= inline_rows and last_rt > 0 and self._plain_pending == 0:
                finished, out = hop(None, None, 0.0, None)  # embedding + first norm + mask: small
            else:
                finished, out = await self.run_plain(hop, None, None, 0.0, None)
            while not finished:
                if isinstance(out, tuple):  # pipelined blocks
                    calls = out[1]
                    parts, where = self._split_blocks(calls, nblocks)
                    preps: list[Any] = []
                    tasks: list[asyncio.Future[Any]] = []
                    try:
                        for blk in parts:  # mask block j+1 while the service computes block j
                            p = await self.run_plain(prepare, blk)
                            preps.append(p)
                            tasks.append(asyncio.ensure_future(backend.send_prepared(p)))
                        got: list[list[torch.Tensor]] = [[] for _ in calls]
                        for j in range(len(parts) - 1):  # verify block j while later blocks compute
                            r, s = await tasks[j]
                            y = await self.run_plain(finish, preps[j], r, s)
                            for pos, part in zip(where[j], y, strict=True):
                                got[pos].append(part)
                        rl, sl = await tasks[-1]
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        raise
                    rt += 1
                    finished, out = await self.run_plain(hop, preps[-1], rl, sl, (got, where[-1]))
                    continue
                prep = out
                response, remote_s = await backend.send_prepared(prep)
                rt += 1
                if (inline_rows and rt < last_rt and self._plain_pending == 0
                        and rows(getattr(prep, "calls", ())) <= inline_rows):
                    finished, out = hop(prep, response, remote_s, None)  # small: no thread hand-off
                else:
                    finished, out = await self.run_plain(hop, prep, response, remote_s, None)
            return out  # type: ignore[no-any-return]
        finally:
            for prep in owned:
                backend.discard(prep)
            owned.clear()

    def forward(self, input_ids: torch.Tensor | Sequence[int], cache: KVCache, *, session: object | None = None,
                all_logits: bool = False) -> torch.Tensor:
        """Run ``input_ids`` (1-D, new tokens only) through the model, appending to ``cache``.

        Returns:
            float32 logits ``(vocab,)`` for the last position, or ``(T, vocab)`` if ``all_logits``.
        """
        return self.forward_batch([SequenceInput(input_ids, cache, session, all_logits)])[0]

    async def aforward(self, input_ids: torch.Tensor | Sequence[int], cache: KVCache, *, session: object | None = None,
                       all_logits: bool = False) -> torch.Tensor:
        """Async variant using :meth:`LinearBackend.aforward` (lets the gateway interleave sessions)."""
        return (await self.aforward_batch([SequenceInput(input_ids, cache, session, all_logits)]))[0]
