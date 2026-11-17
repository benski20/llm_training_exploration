"""OpenAI-compatible request/response schemas (subset). Unknown request fields are ignored.

Extensions (vLLM-style, optional): ``top_k``, ``min_p``, ``repetition_penalty``, ``ignore_eos``,
``stop_token_ids``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ChatChoice",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "ErrorBody",
    "ModelCard",
    "ModelList",
    "Usage",
    "message_text",
]

_MAX_STOP = 16


class ChatMessage(BaseModel):
    """A chat message (text content; content parts are joined)."""

    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


def message_text(content: str | list[dict[str, Any]] | None) -> str:
    """Text of a message: strings as-is, content parts of type ``text`` joined, others rejected."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out: list[str] = []
    for part in content:
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("text content part needs a string 'text'")
            out.append(text)
        else:
            raise ValueError(f"unsupported content part type {ptype!r} (text only)")
    return "".join(out)


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(min_length=1, max_length=256)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = Field(None, ge=-(1 << 63), lt=1 << 64, description="Sampling seed (64-bit)")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None
    n: int = Field(1, ge=1, le=1, description="Only n=1 is supported")
    # extensions
    top_k: int = Field(0, ge=0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, gt=0.0, le=10.0)
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = Field(None, max_length=4096)

    @field_validator("stop")
    @classmethod
    def _stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list):
            if len(v) > _MAX_STOP:
                raise ValueError(f"at most {_MAX_STOP} stop sequences")
            if any(not s for s in v):
                raise ValueError("stop sequences must be non-empty")
        elif v == "":
            raise ValueError("stop sequences must be non-empty")
        return v

    @property
    def include_usage(self) -> bool:
        """``stream_options.include_usage``."""
        return bool((self.stream_options or {}).get("include_usage", False))

    @property
    def requested_max_tokens(self) -> int | None:
        """``max_completion_tokens`` (preferred) or ``max_tokens``."""
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens


class ChatCompletionRequest(_SamplingFields):
    """``POST /v1/chat/completions``."""

    messages: list[ChatMessage] = Field(min_length=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)


class CompletionRequest(_SamplingFields):
    """``POST /v1/completions``."""

    prompt: str | list[str]
    echo: bool = False
    logprobs: int | None = Field(None, ge=0, le=5)
    suffix: str | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list) and not v:
            raise ValueError("prompt list must not be empty")
        return v


class Usage(BaseModel):
    """Token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    """One choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "error", "cancelled"] | None
    logprobs: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """Non-streaming response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
    system_fingerprint: str | None = Field(None, description="PGW model fingerprint (short)")


class ChatCompletionChunk(BaseModel):
    """SSE chunk (``data: {...}`` lines, terminated by ``data: [DONE]``)."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: Usage | None = None
    system_fingerprint: str | None = None


class CompletionChoice(BaseModel):
    """One ``/v1/completions`` choice."""

    index: int = 0
    text: str
    logprobs: dict[str, Any] | None = None
    finish_reason: Literal["stop", "length", "error", "cancelled"] | None


class CompletionResponse(BaseModel):
    """``/v1/completions`` response (non-streaming; streaming chunks use the same object type)."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None


class ModelCard(BaseModel):
    """Entry of ``GET /v1/models``."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "pgw"


class ModelList(BaseModel):
    """``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    """OpenAI-style error: ``{"error": {"message", "type", "code", "param"}}``."""

    error: dict[str, Any]
