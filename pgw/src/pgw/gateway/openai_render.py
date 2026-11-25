"""OpenAI request -> engine parameters and engine events -> OpenAI fields, shared by the gateway's HTTP API
(:mod:`pgw.gateway.app`) and the in-process client library (:mod:`pgw.sdk`).

Framework-free on purpose (no FastAPI import): the client library runs the same conversion code as the
gateway, so a request gives the same prompt ids, sampling parameters, visible tokens and logprobs on both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from pgw.engine import SamplingParams, TokenEvent
from pgw.gateway.openai_schemas import ChatCompletionRequest, CompletionRequest, message_text

if TYPE_CHECKING:  # pragma: no cover
    from pgw.gateway.runtime import ModelRuntime

__all__ = ["TokenText", "chat_logprobs", "chat_messages", "sampling_params", "stop_ids", "visible"]


def sampling_params(req: ChatCompletionRequest | CompletionRequest, max_tokens: int,
                    logprobs: int | None) -> SamplingParams:
    """The engine's :class:`SamplingParams` of a validated request.

    Raises:
        InvalidRequestError: parameters the engine refuses.
    """
    from pgw.gateway.runtime import InvalidRequestError

    try:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            stop=req.stop,
            stop_token_ids=tuple(req.stop_token_ids or ()),
            seed=req.seed,
            logprobs=logprobs,
            ignore_eos=req.ignore_eos,
        )
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid"}
        raise InvalidRequestError(f"invalid sampling parameters: {first['msg']}") from exc


def chat_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    """The chat template's messages (text content; ``developer`` is a ``system`` message)."""
    from pgw.gateway.runtime import InvalidRequestError

    out: list[dict[str, Any]] = []
    for m in req.messages:
        try:
            text = message_text(m.content)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        role = "system" if m.role == "developer" else m.role
        msg: dict[str, Any] = {"role": role, "content": text}
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


class TokenText:
    """Token id -> (display string, raw bytes) for logprobs (cached per request).

    ``bytes`` are the token's raw bytes, which can be part of a multi-byte UTF-8 character (OpenAI
    semantics: clients join them to rebuild split characters). The display string is those bytes
    decoded with replacement characters.
    """

    def __init__(self, rt: ModelRuntime):
        self.rt = rt
        self._cache: dict[int, tuple[str, bytes]] = {}

    def entry(self, tid: int) -> tuple[str, bytes]:
        """``(token, raw_bytes)`` of one token id."""
        hit = self._cache.get(tid)
        if hit is None:
            tok = self.rt.tokenizer
            assert tok is not None
            raw = tok.token_bytes(tid)
            hit = (raw.decode("utf-8", "replace"), raw)
            self._cache[tid] = hit
        return hit

    def __call__(self, tid: int) -> str:
        return self.entry(tid)[0]


def stop_ids(rt: ModelRuntime, params: SamplingParams) -> frozenset[int]:
    """Token ids that end the answer without being part of it (EOS unless ``ignore_eos``, plus ``stop_token_ids``)."""
    ids = set(params.stop_token_ids)
    if not params.ignore_eos:
        if rt.tokenizer is not None:
            ids.update(rt.tokenizer.eos_token_ids)
        if rt.model is not None:
            ids.update(getattr(rt.model.arch, "eos_token_ids", ()) or ())
    return frozenset(ids)


def visible(ev: TokenEvent, stop: frozenset[int]) -> bool:
    """Whether a token event belongs to the output (stop tokens are not part of it)."""
    return not (ev.finish_reason == "stop" and ev.token_id in stop)


def chat_logprobs(events: list[TokenEvent], tt: TokenText) -> dict[str, Any]:
    """OpenAI ``choices[].logprobs`` of a chat completion: ``{"content": [...]}``."""
    content = []
    for ev in events:
        tok, raw = tt.entry(ev.token_id)
        tops = []
        for i, lp in ev.top_logprobs or []:
            ttok, traw = tt.entry(i)
            tops.append({"token": ttok, "logprob": lp, "bytes": list(traw)})
        content.append({"token": tok, "logprob": ev.logprob, "bytes": list(raw), "top_logprobs": tops})
    return {"content": content}
