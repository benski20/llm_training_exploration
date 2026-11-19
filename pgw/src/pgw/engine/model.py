"""Decoder-only transformer forward pass (Qwen2 and Llama) with pluggable linear backend."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable, Sequence
from collections.abc import Generator as PyGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.nn.functional as F

from pgw.core.errors import ConfigError, ShapeError
from pgw.core.types import LinearId, LinearKind, ModelArch
from pgw.engine.backends import LinearBackend, LinearCall
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.pack import linear_split_sizes
from pgw.engine.rope import RotaryEmbedding, apply_rope

__all__ = ["SequenceInput", "TransformerModel", "rms_norm"]

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
    """

    input_ids: torch.Tensor | Sequence[int]
    cache: KVCache
    session: object | None = None
    all_logits: bool = False
    last_logits: int = 0
    serial_tail: int = 0


class RowLogits:
    """Logits of a verify pass, computed per row on first access (each a 1-row ``lm_head`` product,
    the same operation as plain decoding). Rows after a rejected draft are never computed, so the
    ``lm_head`` costs one product per emitted token, as without speculation."""

    def __init__(self, h: torch.Tensor, norm: torch.Tensor, eps: float, lm_head: torch.Tensor):
        self._h = h
        self._norm = norm
        self._eps = eps
        self._lm_head = lm_head
        self._rows: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return int(self._h.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        i = int(i) if i >= 0 else len(self) + int(i)
        hit = self._rows.get(i)
        if hit is None:
            hn = rms_norm(self._h[i : i + 1], self._norm, self._eps).to(torch.float32)
            hit = self._rows[i] = (hn @ self._lm_head.t())[0]
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
        if a.tie_word_embeddings:
            self.lm_head = self.embed_tokens if dtype == torch.float32 else load("lm_head", torch.float32, self.device)
        else:
            self.lm_head = load("lm_head", dtype=torch.float32, device=self.device)
        #: Optional replacement for ``h @ lm_head^T`` (draft models only: e.g. an int8 product).
        self.lm_head_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        layers = range(a.num_layers)
        self.input_norms = [load(f"layers.{i}.input_norm", dtype=dtype, device=self.device) for i in layers]
        self.post_norms = [load(f"layers.{i}.post_attn_norm", dtype=dtype, device=self.device) for i in layers]
        if tuple(self.embed_tokens.shape) != (a.vocab_size, a.hidden_size):
            raise ShapeError(f"embed_tokens shape {tuple(self.embed_tokens.shape)} does not match the architecture")
        self.rope = RotaryEmbedding(a.head_dim, a.rope_theta, a.rope_scaling, a.max_position_embeddings, self.device)
        self.scale = a.head_dim**-0.5
        #: Single worker thread for the plaintext parts of async forward passes (see run_plain).
        self._plain_pool: ThreadPoolExecutor | None = None
        self._plain_lock = threading.Lock()
        log.info(
            "engine model ready: arch=%s layers=%d hidden=%d heads=%d/%d vocab=%d backend=%s",
            a.arch, a.num_layers, a.hidden_size, a.num_heads, a.num_kv_heads, a.vocab_size, type(backend).__name__,
        )

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

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int) -> torch.Tensor:
        """Causal GQA attention of ``q (T, H, hd)`` over ``k, v (L, KVH, hd)``; returns ``(T, H*hd)``."""
        t = q.shape[0]
        length = k.shape[0]
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # (H, *, hd)
        mask = None
        if t > 1:
            qpos = torch.arange(start, start + t, device=q.device)[:, None]
            kpos = torch.arange(length, device=q.device)[None, :]
            mask = kpos <= qpos
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=self.scale)
        return out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim)

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
        hs = [self.embed_tokens[t] for t in ids]
        sessions = [s.session for s in seqs]
        nh, kvh, hd = self.num_heads, self.num_kv_heads, self.head_dim
        nq, nk, _ = self.split_qkv
        for layer in range(self.num_layers):
            normed = [rms_norm(h, self.input_norms[layer], self.eps) for h in hs]
            lid = LinearId(layer, LinearKind.QKV)
            qkv = yield [LinearCall(lid, x, sess) for x, sess in zip(normed, sessions, strict=True)]
            attn_out = []
            for si, y_raw in enumerate(qkv):
                y = y_raw.to(self.dtype)
                rows = int(y.shape[0])
                q = y[:, :nq].reshape(rows, nh, hd)
                k = y[:, nq : nq + nk].reshape(rows, kvh, hd)
                v = y[:, nq + nk :].reshape(rows, kvh, hd)
                cos, sin = rope[si]
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
                k_all, v_all = caches[si].append(layer, k, v)
                tail = min(int(seqs[si].serial_tail), rows)
                if tail <= 0:
                    attn_out.append(self._attention(q, k_all, v_all, starts[si]))
                else:  # the head as one block, then each tail row alone (as plain decoding would)
                    st0, head = starts[si], rows - tail
                    parts = [self._attention(q[:head], k_all[: st0 + head], v_all[: st0 + head], st0)] if head else []
                    for j in range(head, rows):
                        parts.append(self._attention(q[j : j + 1], k_all[: st0 + j + 1], v_all[: st0 + j + 1], st0 + j))
                    attn_out.append(torch.cat(parts, dim=0))
            lid = LinearId(layer, LinearKind.O)
            o = yield [LinearCall(lid, x, sess) for x, sess in zip(attn_out, sessions, strict=True)]
            hs = [h + y.to(self.dtype) for h, y in zip(hs, o, strict=True)]
            normed = [rms_norm(h, self.post_norms[layer], self.eps) for h in hs]
            lid = LinearId(layer, LinearKind.GATE_UP)
            gu = yield [LinearCall(lid, x, sess) for x, sess in zip(normed, sessions, strict=True)]
            inter = self.split_gate_up[0]
            acts = [F.silu(y[:, :inter].to(self.dtype)) * y[:, inter:].to(self.dtype) for y in gu]
            lid = LinearId(layer, LinearKind.DOWN)
            down = yield [LinearCall(lid, x, sess) for x, sess in zip(acts, sessions, strict=True)]
            hs = [h + y.to(self.dtype) for h, y in zip(hs, down, strict=True)]
        for t, c in zip(ids, caches, strict=True):
            c.advance(t.numel())
        out: list[Any] = []
        for s, h in zip(seqs, hs, strict=True):
            if s.last_logits > 0:
                out.append(RowLogits(h[-s.last_logits :], self.final_norm, self.eps, self.lm_head))
                continue
            many = s.all_logits
            hsel = h if s.all_logits else h[-1:]
            hn = rms_norm(hsel, self.final_norm, self.eps).to(torch.float32)
            logits = hn @ self.lm_head.t() if self.lm_head_fn is None else self.lm_head_fn(hn)
            out.append(logits if many else logits[0])
        return out

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
        try:
            calls = next(gen)
            while True:
                res = self._check_results(calls, self.backend.forward_many(calls))
                calls = gen.send(res)
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]
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

    async def aforward_batch(self, seqs: Sequence[SequenceInput]) -> list[torch.Tensor]:
        """Async :meth:`forward_batch` using :meth:`LinearBackend.aforward_many`.

        The plaintext segments between backend calls run on :meth:`run_plain`'s worker thread, so
        the event loop only awaits I/O and never executes model math.
        """
        gen = self._steps(seqs)
        backend = self.backend
        if getattr(backend, "split_hop", False):
            try:
                return await self._aforward_split(gen, backend)
            finally:
                gen.close()
        try:
            finished, out = await self.run_plain(self._advance, gen, None)
            while not finished:
                calls = out
                res = self._check_results(calls, await self.backend.aforward_many(calls))
                finished, out = await self.run_plain(self._advance, gen, res)
            return out  # type: ignore[no-any-return]
        finally:
            gen.close()  # drop references to activations promptly on failure

    async def _aforward_split(self, gen: _Steps, backend: Any) -> list[torch.Tensor]:
        """One plaintext-thread hop per round trip: finish(previous) + model step + prepare(next).

        The plaintext segment between two linears only needs the previous results, so verifying and
        unmasking them, running the step and masking the next calls happen in one worker call; the
        event loop only awaits the network. Any prepared-but-unsent request is wiped on failure.
        """
        pending: list[Any] = [None]  # prepared request owned by this coroutine (wiped on failure)

        def hop(prev: Any, response: Any, remote_s: float) -> tuple[bool, Any]:
            res = None
            if prev is not None:
                pending[0] = None  # finish_prepared always wipes prev
                calls = prev.calls
                res = self._check_results(list(calls), backend.finish_prepared(prev, response, remote_s))
            finished, out = self._advance(gen, res)
            if finished:
                return True, out
            pending[0] = backend.prepare_calls(out)
            return False, pending[0]

        try:
            finished, out = await self.run_plain(hop, None, None, 0.0)
            while not finished:
                prep = out
                response, remote_s = await backend.send_prepared(prep)
                finished, out = await self.run_plain(hop, prep, response, remote_s)
            return out  # type: ignore[no-any-return]
        finally:
            backend.discard(pending[0])
            pending[0] = None

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
