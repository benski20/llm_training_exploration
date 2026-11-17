"""OpenAI-compatible request schemas (subset).

Unknown request fields are ignored, and optional fields sent as JSON ``null`` take their defaults
(as with the OpenAI API). Features the gateway cannot honour are rejected with 400 ``protocol``
instead of being dropped: ``tools``/``functions``, a ``tool_choice``/``function_call`` other than
``"none"``, a ``response_format`` other than ``{"type": "text"}`` and assistant messages carrying
``tool_calls``/``function_call``. Responses are rendered in :mod:`pgw.gateway.app`; their
``finish_reason`` is one of :data:`FINISH_REASONS`.

Extensions (vLLM-style, optional): ``top_k``, ``min_p``, ``repetition_penalty``, ``ignore_eos``,
``stop_token_ids``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "FINISH_REASONS",
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "ModelCard",
    "ModelList",
    "message_text",
]

_MAX_STOP = 16

#: ``finish_reason`` values the API emits (OpenAI's ``stop`` and ``length``; errors are error bodies).
FINISH_REASONS: tuple[str, ...] = ("stop", "length")


def _drop_nulls(data: Any) -> Any:
    """Treat explicit JSON ``null`` like an absent optional field (OpenAI semantics)."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    return data


class ChatMessage(BaseModel):
    """A chat message (text content; content parts are joined)."""

    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    function_call: Any = None

    @field_validator("tool_calls")
    @classmethod
    def _no_tool_calls(cls, v: list[Any] | None) -> list[Any] | None:
        if v:
            raise ValueError("tool calls are not supported (messages carrying tool_calls)")
        return v

    @field_validator("function_call")
    @classmethod
    def _no_function_call(cls, v: Any) -> Any:
        if v is not None:
            raise ValueError("function calling is not supported (messages carrying function_call)")
        return v


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

    @model_validator(mode="before")
    @classmethod
    def _nulls_are_defaults(cls, data: Any) -> Any:
        return _drop_nulls(data)

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
    # unsupported OpenAI features: declared only to reject them (never silently ignored)
    tools: list[Any] | None = None
    functions: list[Any] | None = None
    tool_choice: Any = None
    function_call: Any = None
    response_format: Any = None

    @field_validator("tools", "functions")
    @classmethod
    def _no_tools(cls, v: list[Any] | None) -> list[Any] | None:
        if v:
            raise ValueError("tools and function calling are not supported")
        return v

    @field_validator("tool_choice", "function_call")
    @classmethod
    def _no_tool_choice(cls, v: Any) -> Any:
        if v is not None and v != "none":
            raise ValueError("tool calling is not supported (only 'none' is accepted)")
        return v

    @field_validator("response_format")
    @classmethod
    def _text_only(cls, v: Any) -> Any:
        if v is None:
            return v
        rtype = v.get("type") if isinstance(v, dict) else v
        if rtype != "text":
            raise ValueError(f"response_format {rtype!r} is not supported (only 'text')")
        return v

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

    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        description="Text, list of texts, token ids, or list of token-id lists (OpenAI semantics)"
    )
    echo: bool = False
    logprobs: int | None = Field(None, ge=0, le=5)
    suffix: str | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str | list[str] | list[int] | list[list[int]]) -> str | list[str] | list[int] | list[list[int]]:
        if isinstance(v, list):
            if not v:
                raise ValueError("prompt list must not be empty")
            if isinstance(v[0], list) and any(not ids for ids in v):
                raise ValueError("token-id prompts must not be empty")
        return v

    def prompt_items(self) -> list[str | list[int]]:
        """The prompts as a list: each a text or a token-id list."""
        v = self.prompt
        if isinstance(v, str):
            return [v]
        if v and isinstance(v[0], int):
            return [[int(t) for t in v]]
        return list(v)  # type: ignore[arg-type]


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
