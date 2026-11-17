"""Decoder-only transformer forward pass (Qwen2 and Llama) with pluggable linear backend."""

from __future__ import annotations

import logging
from collections.abc import Generator as PyGenerator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    """

    input_ids: torch.Tensor | Sequence[int]
    cache: KVCache
    session: object | None = None
    all_logits: bool = False


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

    def __init__(self, pack: object, backend: LinearBackend, device: str = "cpu", dtype: torch.dtype = torch.float32):
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
        self.input_norms = [load(f"layers.{i}.input_norm", dtype=dtype, device=self.device) for i in range(a.num_layers)]
        self.post_norms = [load(f"layers.{i}.post_attn_norm", dtype=dtype, device=self.device) for i in range(a.num_layers)]
        if tuple(self.embed_tokens.shape) != (a.vocab_size, a.hidden_size):
            raise ShapeError(f"embed_tokens shape {tuple(self.embed_tokens.shape)} does not match the architecture")
        self.rope = RotaryEmbedding(a.head_dim, a.rope_theta, a.rope_scaling, a.max_position_embeddings, self.device)
        self.scale = a.head_dim**-0.5
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
            for si, y in enumerate(qkv):
                y = y.to(self.dtype)
                t = y.shape[0]
                q = y[:, :nq].reshape(t, nh, hd)
                k = y[:, nq : nq + nk].reshape(t, kvh, hd)
                v = y[:, nq + nk :].reshape(t, kvh, hd)
                cos, sin = rope[si]
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
                k_all, v_all = caches[si].append(layer, k, v)
                attn_out.append(self._attention(q, k_all, v_all, starts[si]))
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
        out = []
        for s, h in zip(seqs, hs, strict=True):
            hsel = h if s.all_logits else h[-1:]
            hn = rms_norm(hsel, self.final_norm, self.eps).to(torch.float32)
            logits = hn @ self.lm_head.t()
            out.append(logits if s.all_logits else logits[0])
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

    async def aforward_batch(self, seqs: Sequence[SequenceInput]) -> list[torch.Tensor]:
        """Async :meth:`forward_batch` using :meth:`LinearBackend.aforward_many`."""
        gen = self._steps(seqs)
        try:
            calls = next(gen)
            while True:
                res = self._check_results(calls, await self.backend.aforward_many(calls))
                calls = gen.send(res)
        except StopIteration as stop:
            return stop.value  # type: ignore[no-any-return]

    def forward(self, input_ids: torch.Tensor, cache: KVCache, *, session: object | None = None,
                all_logits: bool = False) -> torch.Tensor:
        """Run ``input_ids`` (1-D, new tokens only) through the model, appending to ``cache``.

        Returns:
            float32 logits ``(vocab,)`` for the last position, or ``(T, vocab)`` if ``all_logits``.
        """
        return self.forward_batch([SequenceInput(input_ids, cache, session, all_logits)])[0]

    async def aforward(self, input_ids: torch.Tensor, cache: KVCache, *, session: object | None = None,
                       all_logits: bool = False) -> torch.Tensor:
        """Async variant using :meth:`LinearBackend.aforward` (lets the gateway interleave sessions)."""
        return (await self.aforward_batch([SequenceInput(input_ids, cache, session, all_logits)]))[0]
