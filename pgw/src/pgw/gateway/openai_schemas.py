"""OpenAI-compatible request/response schemas (subset). Unknown request fields are ignored."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatChoice",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "CompletionRequest",
    "ErrorBody",
    "ModelCard",
    "ModelList",
    "Usage",
]


class ChatMessage(BaseModel):
    """A chat message (text content; content parts are joined)."""

    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None
    n: int = Field(1, ge=1, le=1, description="Only n=1 is supported")


class ChatCompletionRequest(_SamplingFields):
    """``POST /v1/chat/completions``."""

    messages: list[ChatMessage]
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)


class CompletionRequest(_SamplingFields):
    """``POST /v1/completions``."""

    prompt: str | list[str]
    echo: bool = False
    logprobs: int | None = Field(None, ge=0, le=5)


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
