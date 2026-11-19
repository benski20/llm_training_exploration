"""Sampling parameters and token sampler (OpenAI-compatible semantics)."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Sampler", "SamplingParams"]


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
            return int(torch.argmax(work))
        probs = torch.softmax(self._filter(work / p.temperature), dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
            return int(torch.argmax(work))
        return int(torch.multinomial(probs, 1, generator=self._gen))

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
            sorted_logits, order = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            remove = (cum - probs) >= p.top_p  # keep the smallest prefix with mass >= top_p
            remove[0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, order, sorted_logits)
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
        work = self._penalize(raw.clone(), generated)
        p = self.params
        if p.greedy:
            token = int(torch.argmax(work))
        else:
            scaled = self._filter(work / p.temperature)
            probs = torch.softmax(scaled, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                token = int(torch.argmax(work))
            else:
                token = int(torch.multinomial(probs, 1, generator=self._gen))
        logprob: float | None = None
        self.last_top_logprobs = None
        if p.logprobs is not None:
            lsm = torch.log_softmax(raw, dim=-1)
            logprob = float(lsm[token])
            if p.logprobs > 0:
                top = torch.topk(lsm, min(p.logprobs, lsm.numel()))
                self.last_top_logprobs = [(int(i), float(v)) for v, i in zip(top.values, top.indices, strict=True)]
        return token, logprob
