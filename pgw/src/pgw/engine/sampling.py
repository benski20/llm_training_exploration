"""Sampling parameters and token sampler (OpenAI-compatible semantics)."""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pgw.core import native

__all__ = ["Sampler", "SamplingParams", "fast_argmax", "fast_topk"]


def fast_argmax(x: torch.Tensor) -> int:
    """``int(torch.argmax(x))`` for a 1-D tensor, cheaper on the CPU.

    ``torch.argmax`` over a 152k vocabulary costs about 0.4 ms on the CPU; numpy's ``argmax`` has the
    same contract (the index of the *first* maximal element; ``-0.0 == +0.0``) and takes about 10 us.
    A NaN anywhere makes numpy return a NaN's index; then torch decides (same result as before).
    """
    if x.ndim == 1 and x.numel() > 0 and x.dtype == torch.float32 and x.device.type == "cpu":
        a = x.detach().numpy()
        i = int(a.argmax())
        if not math.isnan(float(a[i])):
            return i
    return int(torch.argmax(x))


def fast_topk(x: torch.Tensor, k: int) -> list[tuple[int, float]]:
    """``[(index, value)]`` of ``torch.topk(x, k)`` (descending) for a 1-D tensor, cheaper on the CPU.

    The native helper answers only when the result is unique (no NaN; the ``k + 1`` largest values
    strictly decreasing), so it cannot differ from ``torch.topk``, whose order among equal values is
    implementation-defined; otherwise (ties, NaN, no native module) this is ``torch.topk`` itself.
    """
    if (x.ndim == 1 and x.dtype == torch.float32 and x.device.type == "cpu" and 0 < k <= x.numel()
            and native.native_available("top_k_strict")):
        got = native.fast.top_k_strict(x.detach().numpy(), int(k))
        if got is not None:
            idx, vals = got
            return [(int(i), float(v)) for i, v in zip(idx, vals, strict=True)]
    top = torch.topk(x, k)
    return [(int(i), float(v)) for v, i in zip(top.values, top.indices, strict=True)]


_MN_LOCK = threading.Lock()
_MN_OK: bool | None = None


def _native_multinomial_ok() -> bool:
    """Whether ``pgw_fast.torch_multinomial1`` reproduces this torch build's ``torch.multinomial(p, 1, g)``:
    checked once per process on public test distributions (index and generator state afterwards, byte for
    byte); ``PGW_NATIVE_MULTINOMIAL=0`` turns it off."""
    global _MN_OK
    ok = _MN_OK
    if ok is not None:
        return ok
    with _MN_LOCK:
        if _MN_OK is not None:
            return _MN_OK
        ok = False
        if (os.environ.get("PGW_NATIVE_MULTINOMIAL", "1").strip().lower() not in ("0", "false", "no", "off")
                and native.native_available("torch_multinomial1")):
            try:
                ok = True
                for case, (n, temp) in enumerate(((7, 1.0), (1000, 0.5), (151936, 0.7), (151936, 1.6))):
                    logits = torch.randn(n, generator=torch.Generator().manual_seed(1000 + case)) * 4
                    logits[::5] = float("-inf")
                    probs = torch.softmax(logits / temp, dim=-1)
                    g = torch.Generator().manual_seed(77 + case)
                    torch.rand(case + 1, generator=g)
                    st = g.get_state()
                    got = native.fast.torch_multinomial1(st.numpy(), probs.numpy())
                    ref = int(torch.multinomial(probs, 1, generator=g))
                    if got is None or int(got) != ref or not torch.equal(st, g.get_state()):
                        ok = False
                        break
            except Exception:  # any surprise: torch's own draw
                ok = False
            if not ok:
                import logging

                logging.getLogger(__name__).warning("native multinomial disagrees with torch.multinomial: not used")
        _MN_OK = ok
        return ok


def multinomial1(probs: torch.Tensor, gen: torch.Generator) -> int:
    """``int(torch.multinomial(probs, 1, generator=gen))`` for a 1-D float32 CPU tensor: the same index and the
    same generator state afterwards (``pgw_fast.torch_multinomial1``: torch's exponential race, ``log1p`` only
    for the elements that can win; checked against torch once per process), else torch itself."""
    if (probs.ndim == 1 and probs.dtype == torch.float32 and probs.device.type == "cpu" and probs.is_contiguous()
            and gen.device.type == "cpu" and _native_multinomial_ok()):
        st = gen.get_state()
        got = native.fast.torch_multinomial1(st.numpy(), probs.detach().numpy())
        if got is not None:
            gen.set_state(st)
            return int(got)
    return int(torch.multinomial(probs, 1, generator=gen))


def _top_p_reference(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """The top-p (nucleus) filter: keep the smallest prefix of the descending order with mass >= ``top_p``."""
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = (cum - probs) >= top_p  # keep the smallest prefix with mass >= top_p
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(0, order, sorted_logits)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """:func:`_top_p_reference` with numpy's sort (about 5x cheaper than ``torch.sort`` over a 152k vocabulary on
    one core, far more with busy intra-op threads), bit-identical.

    A descending sort of a 1-D float32 array has one sequence of values, whatever the sort; only the order of equal
    values (and of ``-0.0`` / ``+0.0``) can differ, and each element keeps its own value through the scatter. The
    softmax, cumulative sum and removal mask run on that same value sequence, so they are identical; the kept set
    could differ only where the mask changes inside a run of equal values, which is checked (then, as with NaN, the
    reference runs)."""
    import numpy as np

    if logits.ndim != 1 or logits.dtype != torch.float32 or logits.device.type != "cpu" or logits.numel() < 2:
        return _top_p_reference(logits, top_p)
    a = logits.detach().numpy()
    if np.isnan(a).any():
        return _top_p_reference(logits, top_p)
    order = np.argsort(-a)  # ascending order of -a: the descending order of a, ties in any order
    sv = a[order]
    sorted_logits = torch.from_numpy(sv)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = (cum - probs) >= top_p
    remove[0] = False
    r = remove.numpy()
    at = np.flatnonzero(r[1:] != r[:-1]) + 1
    if at.size and bool((sv[at] == sv[at - 1]).any()):
        return _top_p_reference(logits, top_p)  # the boundary splits equal values: torch's tie order decides
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(0, torch.from_numpy(order), sorted_logits)


class SamplingParams(BaseModel):
    """Decoding controls. ``temperature == 0`` means greedy.

    Penalties follow common serving semantics: ``repetition_penalty`` (HF style, divides positive
    and multiplies negative logits) applies to prompt and generated tokens; ``presence_penalty``
    and ``frequency_penalty`` (OpenAI style) apply to generated tokens only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0, description="0 disables top-k")
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, gt=0.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    logprobs: int | None = Field(None, ge=0, le=20)
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False

    @field_validator("stop", mode="before")
    @classmethod
    def _stop_tuple(cls, v: object) -> object:
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,)
        return v

    @field_validator("stop")
    @classmethod
    def _stop_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(s == "" for s in v):
            raise ValueError("stop strings must be non-empty")
        if len(v) > 16:
            raise ValueError("at most 16 stop strings")
        return v

    @property
    def greedy(self) -> bool:
        """True when decoding is deterministic argmax."""
        return self.temperature == 0.0 or self.top_k == 1


class Sampler:
    """Stateful sampler for one sequence (holds the RNG and token counts for penalties).

    Args:
        params: Sampling parameters.
        prompt_ids: Prompt tokens (for ``repetition_penalty``).
    """

    def __init__(self, params: SamplingParams, prompt_ids: Sequence[int] = ()):
        self.params = params
        self._gen = torch.Generator(device="cpu")
        seed = params.seed if params.seed is not None else int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
        self._gen.manual_seed(seed)
        self._prompt = set(int(t) for t in prompt_ids)
        self.last_top_logprobs: list[tuple[int, float]] | None = None

    def fork(self) -> Sampler:
        """A copy with the same parameters and the RNG at the same state (for coupled drafting: a draft
        sampled with the fork draws the very noise the target's next samples will draw)."""
        other = Sampler.__new__(Sampler)
        other.params = self.params
        other._gen = torch.Generator(device="cpu")
        other._gen.set_state(self._gen.get_state())
        other._prompt = self._prompt
        other.last_top_logprobs = None
        return other

    def propose(self, logits: torch.Tensor, generated: list[int]) -> int:
        """Like :meth:`sample` without logprobs (draft proposals). Advances this sampler's RNG exactly as
        :meth:`sample` would."""
        work = self._penalize(logits.detach().to(torch.float32).cpu().clone(), generated)
        p = self.params
        if p.greedy:
            return fast_argmax(work)
        probs = torch.softmax(self._filter(work / p.temperature), dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
            return fast_argmax(work)
        return multinomial1(probs, self._gen)

    def _penalize(self, logits: torch.Tensor, generated: Sequence[int]) -> torch.Tensor:
        p = self.params
        if p.repetition_penalty != 1.0:
            seen = self._prompt | set(int(t) for t in generated)
            if seen:
                idx = torch.tensor(sorted(seen), dtype=torch.long)
                idx = idx[idx < logits.numel()]
                vals = logits[idx]
                logits[idx] = torch.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
        if (p.presence_penalty or p.frequency_penalty) and generated:
            counts = Counter(int(t) for t in generated)
            idx = torch.tensor(list(counts.keys()), dtype=torch.long)
            cnt = torch.tensor(list(counts.values()), dtype=torch.float32)
            keep = idx < logits.numel()
            idx, cnt = idx[keep], cnt[keep]
            logits[idx] -= p.frequency_penalty * cnt + p.presence_penalty
        return logits

    def _filter(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k, top-p and min-p (on temperature-scaled logits)."""
        p = self.params
        if p.top_k > 0 and p.top_k < logits.numel():
            kth = torch.topk(logits, p.top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if p.min_p > 0.0:
            probs = torch.softmax(logits, dim=-1)
            logits = logits.masked_fill(probs < p.min_p * probs.max(), float("-inf"))
        if p.top_p < 1.0:
            logits = top_p_filter(logits, p.top_p)  # keep the smallest prefix with mass >= top_p
        return logits

    def sample(self, logits: torch.Tensor, generated: list[int]) -> tuple[int, float | None]:
        """Pick the next token from float32 ``(vocab,)`` logits; return ``(token_id, logprob)``.

        ``logprob`` is the log-probability of the chosen token under the model distribution
        (before penalties and filters) when ``params.logprobs`` is set, else ``None``. The top
        ``params.logprobs`` alternatives are left in :attr:`last_top_logprobs`.
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
        raw = logits.detach().to(torch.float32).cpu()
        p = self.params
        # penalties write into a copy; without them nothing below writes into ``raw`` (no 0.6 MB clone)
        penalized = p.repetition_penalty != 1.0 or bool((p.presence_penalty or p.frequency_penalty) and generated)
        work = self._penalize(raw.clone(), generated) if penalized else raw
        if p.greedy:
            token = fast_argmax(work)
        else:
            scaled = self._filter(work / p.temperature)
            probs = torch.softmax(scaled, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                token = fast_argmax(work)
            else:
                token = multinomial1(probs, self._gen)
        logprob: float | None = None
        self.last_top_logprobs = None
        if p.logprobs is not None:
            lsm = torch.log_softmax(raw, dim=-1)
            logprob = float(lsm[token])
            if p.logprobs > 0:
                self.last_top_logprobs = fast_topk(lsm, min(p.logprobs, lsm.numel()))
        return token, logprob
