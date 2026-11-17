"""Wires engine + mask store + wire client per model."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from pgw.gateway.config import GatewayModelConfig, GatewaySettings

__all__ = ["InferenceRuntime", "ModelRuntime"]


class ModelRuntime:
    """Everything needed to serve one model: pack, tokenizer, :class:`pgw.engine.TransformerModel`,
    backend (masked/plain/float), mask store handle, compute client, admission control."""

    def __init__(self, config: GatewayModelConfig, settings: GatewaySettings):
        raise NotImplementedError("gateway team: ModelRuntime")

    async def start(self) -> None:
        """Open pack (verify fingerprint), connect to the service (fingerprint/ring check), check masks."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Drain sessions and close connections."""
        raise NotImplementedError

    async def chat(self, messages: Sequence[dict[str, Any]], params: Any) -> Any:
        """Non-streaming chat completion -> :class:`pgw.engine.GenerationResult`."""
        raise NotImplementedError

    def chat_stream(self, messages: Sequence[dict[str, Any]], params: Any) -> AsyncIterator[Any]:
        """Streaming chat completion -> async iterator of :class:`pgw.engine.TokenEvent`."""
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        """Admin status: fingerprint, ring, mask inventory, sessions, service health."""
        raise NotImplementedError


class InferenceRuntime:
    """All model runtimes of the gateway, keyed by served name."""

    def __init__(self, settings: GatewaySettings):
        raise NotImplementedError("gateway team: InferenceRuntime")

    def get(self, model: str) -> ModelRuntime:
        """Runtime by name; raises :class:`pgw.core.ModelNotFoundError`."""
        raise NotImplementedError

    async def start(self) -> None:
        """Start all model runtimes."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop all model runtimes."""
        raise NotImplementedError
