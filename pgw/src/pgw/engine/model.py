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
from pgw.engine.lm_head import compact_head, lm_head_logits, native_lm_head
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


def _sdpa_gqa_supported() -> bool:
    """True when ``F.scaled_dot_product_attention`` takes ``enable_gqa`` (torch >= 2.5)."""
    try:
        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:  # pragma: no cover - unusual version strings: keep the proven path
        return False
    return (major, minor) >= (2, 5)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float, groups: int,
            gqa: bool) -> torch.Tensor:
    """Causal attention of ``q (T, H, hd)`` over ``k, v (L, KVH, hd)``; returns ``(T, H*hd)``.

    ``gqa``: let SDPA broadcast each KV head over its ``groups`` query heads (``enable_gqa``) instead
    of materializing ``repeat_interleave`` copies of the cache. Only used where
    :func:`gqa_self_check` showed both forms give the same bits.
    """
    t = q.shape[0]
    length = k.shape[0]
    if groups > 1 and not gqa:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # (heads, *, hd)
    mask = None
    if t > 1:
        qpos = torch.arange(start, start + t, device=q.device)[:, None]
        kpos = torch.arange(length, device=q.device)[None, :]
        mask = kpos <= qpos
    if groups > 1 and gqa:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale, enable_gqa=True)
    else:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
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
                 logits: torch.Tensor | None = None):
        self._h = h
        self._norm = norm
        self._eps = eps
        self._lm_head = lm_head
        self._logits = logits
        self._rows: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return int(self._h.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        i = int(i) if i >= 0 else len(self) + int(i)
        if self._logits is not None:
            return self._logits[i]
        hit = self._rows.get(i)
        if hit is None:
            hn = _norm_rows(self._h[i : i + 1], self._norm, self._eps)
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
        self._gqa = False
        self._gqa_threads = -1
        self._use_gqa()  # the GQA self-check runs at load (cached per process; again if threads change)
        #: Single worker thread for the plaintext parts of async forward passes (see run_plain).
        self._plain_pool: ThreadPoolExecutor | None = None
        self._plain_lock = threading.Lock()
        #: run_plain calls submitted and not yet completed (as seen from the event loop).
        self._plain_pending = 0
        # LinearId objects per layer (hot path: four per layer per forward)
        self._lids = [tuple(LinearId(i, k) for k in (LinearKind.QKV, LinearKind.O, LinearKind.GATE_UP,
                                                       LinearKind.DOWN)) for i in range(a.num_layers)]
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
        if threads != self._gqa_threads:  # verdicts are per thread count (kernels may split by it)
            self._gqa = bool(self.num_heads != self.num_kv_heads and _gqa_verdict(
                self.num_heads, self.num_kv_heads, self.head_dim, self.dtype, self.device, self.scale))
            self._gqa_threads = threads
        return self._gqa and self.gqa_sdpa

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int) -> torch.Tensor:
        """Causal GQA attention of ``q (T, H, hd)`` over ``k, v (L, KVH, hd)``; returns ``(T, H*hd)``."""
        groups = self.num_heads // self.num_kv_heads
        return _attend(q, k, v, start, self.scale, groups, groups > 1 and self._use_gqa())

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
            lid_qkv, lid_o, lid_gu, lid_down = self._lids[layer]
            normed = [rms_norm(h, self.input_norms[layer], self.eps) for h in hs]
            lid = lid_qkv
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
            lid = lid_o
            o = yield [LinearCall(lid, x, sess) for x, sess in zip(attn_out, sessions, strict=True)]
            hs = [h + y.to(self.dtype) for h, y in zip(hs, o, strict=True)]
            normed = [rms_norm(h, self.post_norms[layer], self.eps) for h in hs]
            lid = lid_gu
            gu = yield [LinearCall(lid, x, sess) for x, sess in zip(normed, sessions, strict=True)]
            inter = self.split_gate_up[0]
            acts = [F.silu(y[:, :inter].to(self.dtype)) * y[:, inter:].to(self.dtype) for y in gu]
            lid = lid_down
            down = yield [LinearCall(lid, x, sess) for x, sess in zip(acts, sessions, strict=True)]
            hs = [h + y.to(self.dtype) for h, y in zip(hs, down, strict=True)]
        for t, c in zip(ids, caches, strict=True):
            c.advance(t.numel())
        return self._logits(seqs, hs)

    def _logits(self, seqs: Sequence[SequenceInput], hs: list[torch.Tensor]) -> list[Any]:
        """Final norm and LM head of a step: per sequence the last row's logits ``(vocab,)``, all rows'
        ``(T, vocab)`` (``all_logits``) or :class:`RowLogits` (``last_logits``)."""
        out: list[Any] = []
        if self.lm_head_fn is not None:  # approximate head (draft models): one product per sequence
            for s, h in zip(seqs, hs, strict=True):
                if s.last_logits > 0:
                    out.append(RowLogits(h[-s.last_logits :], self.final_norm, self.eps, self._head))
                    continue
                hn = rms_norm(h if s.all_logits else h[-1:], self.final_norm, self.eps).to(torch.float32)
                logits = self.lm_head_fn(hn)
                out.append(logits if s.all_logits else logits[0])
            return out
        # Normalized rows per sequence (rows that plain decoding computes alone are normed alone);
        # None: verify rows left to RowLogits (computed lazily when the pass is not shared).
        shared = self.batch_lm_head
        hns: list[torch.Tensor | None] = []
        for s, h in zip(seqs, hs, strict=True):
            if s.last_logits > 0:
                hns.append(_norm_rows(h[-s.last_logits :], self.final_norm, self.eps) if shared else None)
            else:
                hns.append(rms_norm(h if s.all_logits else h[-1:], self.final_norm, self.eps).to(torch.float32))
        given = [x for x in hns if x is not None]
        logits_of: list[torch.Tensor | None] = []
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
                out.append(RowLogits(h[-s.last_logits :], self.final_norm, self.eps, self._head, lg))
            elif lg is None:  # pragma: no cover - every other sequence has normalized rows
                raise AssertionError("missing logits")
            else:
                out.append(lg if s.all_logits else lg[0])
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

    async def aforward_batch(self, seqs: Sequence[SequenceInput]) -> list[torch.Tensor]:
        """Async :meth:`forward_batch` using :meth:`LinearBackend.aforward_many`.

        The plaintext segments between backend calls run on :meth:`run_plain`'s worker thread, so
        the event loop only awaits I/O and never executes model math.
        """
        gen = self._steps(seqs)
        backend = self.backend
        if getattr(backend, "split_hop", False):
            try:
                new_rows = sum(len(s.input_ids) for s in seqs)
                if not self._inline_safe(seqs, backend):
                    new_rows = -1  # no inline hops in this pass
                return await self._aforward_split(gen, backend, new_rows)
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

    #: Round trips with at least this many rows in calls of 2+ rows (prefill) are split in two halves
    #: that overlap: the gateway masks the second half while the service computes the first, and
    #: verifies the first while the service computes the second. Rows are independent, so results
    #: are identical. 1-row calls in the same round trip (decode steps of other streams) go wholly
    #: into the first half. Each session's masks are taken in row order (the first half first).
    pipeline_min_rows = 16

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
        first = self._lids[0][0]
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

    async def _aforward_split(self, gen: _Steps, backend: Any, new_rows: int = 1 << 30) -> list[torch.Tensor]:
        """One plaintext-thread hop per round trip: finish(previous) + model step + prepare(next).

        The plaintext segment between two linears only needs the previous results, so verifying and
        unmasking them, running the step and masking the next calls happen in one worker call; the
        event loop only awaits the network. Large (prefill) round trips are pipelined in two halves.
        Any prepared-but-unsent request is wiped on failure.
        """
        owned: list[Any] = []  # prepared requests owned by this coroutine (wiped on failure)
        big = self.pipeline_min_rows

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
                extra: tuple[list[torch.Tensor], list[int]] | None) -> tuple[bool, Any]:
            res = None
            if prev is not None:
                res = finish(prev, response, remote_s)
                if extra is not None:  # second half of a pipelined round trip: rejoin by call position
                    first, in_b = extra
                    if len(res) != len(in_b):  # pragma: no cover - finish() checked the half's calls
                        raise ShapeError(f"second half returned {len(res)} results for {len(in_b)} calls")
                    joined = list(first)
                    for j, i in enumerate(in_b):
                        joined[i] = torch.cat([first[i], res[j]], dim=0)
                    res = joined
            finished, out = self._advance(gen, res)
            if finished:
                return True, out
            # Calls of 2+ rows are split in halves; 1-row calls (decode next to prefills) go wholly
            # into the first half. Pipelined only when the split rows are many enough to pay.
            if big and sum(n for n in (int(c.x.shape[0]) for c in out) if n >= 2) >= big:
                return False, ("split", out)
            return False, prepare(out)

        inline_rows = int(self.inline_max_rows) if new_rows >= 0 else 0
        last_rt = 4 * self.num_layers  # the hop after this many round trips runs the LM head
        rt = 0
        try:
            if inline_rows and new_rows <= inline_rows and last_rt > 0 and self._plain_pending == 0:
                finished, out = hop(None, None, 0.0, None)  # embedding + first norm + mask: small
            else:
                finished, out = await self.run_plain(hop, None, None, 0.0, None)
            while not finished:
                if isinstance(out, tuple):  # pipelined halves
                    calls = out[1]
                    halves = [[], []]  # type: list[list[LinearCall]]
                    in_b: list[int] = []  # positions of the calls that have rows in the second half
                    for i, c in enumerate(calls):
                        n = int(c.x.shape[0])
                        if n < 2:  # whole call in half A (half B has no item for it)
                            halves[0].append(c)
                            continue
                        h = n // 2
                        halves[0].append(LinearCall(c.id, c.x[:h], c.session, c.meta))
                        halves[1].append(LinearCall(c.id, c.x[h:], c.session, c.meta))
                        in_b.append(i)
                    pa = await self.run_plain(prepare, halves[0])
                    ta = asyncio.ensure_future(backend.send_prepared(pa))
                    try:
                        pb = await self.run_plain(prepare, halves[1])
                        tb = asyncio.ensure_future(backend.send_prepared(pb))
                    except BaseException:
                        ta.cancel()
                        raise
                    try:
                        ra, sa = await ta
                        ya = await self.run_plain(finish, pa, ra, sa)
                        rb, sb = await tb
                    except BaseException:
                        tb.cancel()
                        raise
                    rt += 1
                    finished, out = await self.run_plain(hop, pb, rb, sb, (ya, in_b))
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
