"""Response objects shaped like the ``openai`` package's (field for field; pydantic models).

``model_dump()`` gives the JSON an OpenAI-compatible server would send, so code written for ``openai`` reads
these unchanged (``resp.choices[0].message.content``, ``chunk.choices[0].delta.content``, ``resp.usage``).
With the ``openai`` package installed, :meth:`_Base.to_openai` converts to its own classes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "ChatCompletionTokenLogprob",
    "Choice",
    "ChoiceDelta",
    "ChoiceLogprobs",
    "ChunkChoice",
    "CompletionUsage",
    "Model",
    "ModelList",
    "PromptTokensDetails",
    "TopLogprob",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (as the HTTP API would send it)."""
        return self.model_dump(mode="json")

    def to_json(self, **kw: Any) -> str:
        """JSON string (as the HTTP API would send it)."""
        return self.model_dump_json(**kw)

    def to_openai(self) -> Any:
        """The same object as the ``openai`` package's class (needs ``openai`` installed)."""
        import importlib

        target = _OPENAI_TYPES.get(type(self).__name__)
        if target is None:
            raise TypeError(f"{type(self).__name__} has no openai counterpart")
        mod, name = target
        cls = getattr(importlib.import_module(mod), name)
        return cls.model_validate(self.to_dict())


class TopLogprob(_Base):
    token: str
    logprob: float
    bytes: list[int] | None = None


class ChatCompletionTokenLogprob(_Base):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = []


class ChoiceLogprobs(_Base):
    content: list[ChatCompletionTokenLogprob] | None = None
    refusal: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionMessage(_Base):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[Any] | None = None
    function_call: Any = None


class Choice(_Base):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    logprobs: ChoiceLogprobs | None = None


class PromptTokensDetails(_Base):
    cached_tokens: int = 0


class CompletionUsage(_Base):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletion(_Base):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class ChoiceDelta(_Base):
    role: Literal["assistant"] | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[Any] | None = None
    function_call: Any = None


class ChunkChoice(_Base):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(_Base):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class Model(_Base):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "pgw"


class ModelList(_Base):
    """``client.models.list()``: iterable like the ``openai`` package's page object."""

    object: Literal["list"] = "list"
    data: list[Model]

    def __iter__(self) -> Iterator[Model]:  # type: ignore[override]
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


_OPENAI_TYPES: dict[str, tuple[str, str]] = {
    "ChatCompletion": ("openai.types.chat", "ChatCompletion"),
    "ChatCompletionChunk": ("openai.types.chat", "ChatCompletionChunk"),
    "Model": ("openai.types", "Model"),
    "CompletionUsage": ("openai.types", "CompletionUsage"),
}
