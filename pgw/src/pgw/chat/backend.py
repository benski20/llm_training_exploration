"""The chat server's view of the client library: :class:`SDKBackend` wraps one :class:`pgw.OpenAI` in this process.

Everything the browser asks for goes through here and nowhere else: requests go to the SDK (which masks every linear
layer's input before it leaves the process and checks every result), status comes from ``client.status()``. The
backend adds no network access of its own.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator
from typing import Any, Protocol

from pgw.chat.status import error_view, summarize

__all__ = ["MODEL_INFO", "Backend", "ChatStream", "SDKBackend", "device_profile", "recommended_model"]

#: what the model picker shows (planning figures from docs/sdk/models.md; ``stream_disk_gb``: with
#: ``factory_weights="stream"``, which keeps no int8 weights on disk)
MODEL_INFO: dict[str, dict[str, Any]] = {
    "qwen2.5-0.5b-instruct": {"label": "Qwen2.5 0.5B", "params": "0.5B", "trips_per_token": 97, "download_gb": 1,
                              "disk_gb": 10, "stream_disk_gb": 9, "memory_gb": 8, "cpus": 4,
                              "note": "Fastest to set up; short, simple answers"},
    "qwen2.5-1.5b-instruct": {"label": "Qwen2.5 1.5B", "params": "1.5B", "trips_per_token": 113, "download_gb": 3,
                              "disk_gb": 13, "stream_disk_gb": 11, "memory_gb": 8, "cpus": 4,
                              "note": "Good default for most laptops"},
    "qwen2.5-3b-instruct": {"label": "Qwen2.5 3B", "params": "3B", "trips_per_token": 145, "download_gb": 6,
                            "disk_gb": 17, "stream_disk_gb": 14, "memory_gb": 16, "cpus": 8,
                            "note": "Better answers; Qwen research licence"},
    "qwen2.5-7b-instruct": {"label": "Qwen2.5 7B", "params": "7B", "trips_per_token": 113, "download_gb": 11,
                            "disk_gb": 26, "stream_disk_gb": 20, "memory_gb": 32, "cpus": 16,
                            "note": "Best answers; needs a strong machine"},
}


def device_profile() -> dict[str, Any]:
    """CPUs, memory and free disk of this machine (what the model picker recommends from; never sent anywhere)."""
    cpus = os.cpu_count() or 1
    mem_gb = None
    with contextlib.suppress(Exception):
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {"cpus": cpus, "memory_gb": None if mem_gb is None else round(mem_gb, 1)}


def recommended_model(profile: dict[str, Any], free_disk_gb: float | None = None) -> str:
    """The largest model the machine meets the planning figures for, capped at 3B (7B is opt-in)."""
    best = "qwen2.5-0.5b-instruct"
    for m in ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct"):
        i = MODEL_INFO[m]
        mem = profile.get("memory_gb")
        if profile.get("cpus", 1) >= i["cpus"] and (mem is None or mem >= i["memory_gb"]) and (
                free_disk_gb is None or free_disk_gb >= i["stream_disk_gb"]):
            best = m
    return best


class ChatStream(Protocol):
    """One streamed answer: iterate for text deltas; ``finish_reason`` and ``usage`` are set at the end."""

    finish_reason: str | None
    usage: dict[str, Any] | None

    def __iter__(self) -> Iterator[str]: ...

    def close(self) -> None: ...


class Backend(Protocol):
    def settings(self) -> dict[str, Any]: ...

    def models(self) -> list[dict[str, Any]]: ...

    def status(self, model: str) -> dict[str, Any]: ...

    def warmup(self, model: str) -> None: ...

    def stream(self, model: str, messages: list[dict[str, str]], params: dict[str, Any],
               conversation_id: str | None) -> ChatStream: ...

    def close(self) -> None: ...


class _SDKStream:
    def __init__(self, it: Any):
        self._it = it
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[str]:
        for chunk in self._it:
            if getattr(chunk, "usage", None) is not None:
                u = chunk.usage
                details = getattr(u, "prompt_tokens_details", None)
                self.usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                              "completion_tokens": getattr(u, "completion_tokens", None),
                              "cached_tokens": getattr(details, "cached_tokens", None) if details else None}
            for ch in chunk.choices or ():
                if ch.finish_reason:
                    self.finish_reason = ch.finish_reason
                text = getattr(ch.delta, "content", None)
                if text:
                    yield text

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._it.close()


class SDKBackend:
    """The client library in this process.

    Args:
        client: A :class:`pgw.OpenAI`.
        request_overrides: Extra keyword arguments for every ``create`` call (tests).
    """

    def __init__(self, client: Any, *, request_overrides: dict[str, Any] | None = None):
        self.client = client
        self._overrides = dict(request_overrides or {})
        self._mu = threading.Lock()
        self._warming: dict[str, threading.Thread] = {}
        self._warm_error: dict[str, dict[str, Any]] = {}
        self._status: tuple[float, dict[str, Any]] | None = None

    def settings(self) -> dict[str, Any]:
        o = self.client.options
        pc = getattr(o, "prefix_cache", "auto")
        return {"mask_scheme": getattr(o, "mask_scheme", None), "masked_head": str(getattr(o, "masked_head", "auto")),
                "prefix_cache": pc if isinstance(pc, str) else ("on" if pc else "off"),
                "speculation": getattr(o, "speculation", "off"),
                "schedule": getattr(o, "schedule", "standard"),
                "factory_weights": getattr(o, "factory_weights", "pack"),
                "usage_reports": bool(getattr(o, "usage_reports", True)),
                "service_url": getattr(getattr(self.client, "_key", None), "service_url", None)}

    def models(self) -> list[dict[str, Any]]:
        from pgw.sdk.options import SUPPORTED_MODELS

        return [{"id": m, **MODEL_INFO.get(m, {"label": m})} for m in SUPPORTED_MODELS]

    def _raw_status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._mu:
            cached = self._status
        if cached is None or now - cached[0] > 0.5:
            cached = (now, dict(self.client.status()))
            with self._mu:
                self._status = cached
        return cached[1]

    def status(self, model: str) -> dict[str, Any]:
        with self._mu:
            t = self._warming.get(model)
            warming = t is not None and t.is_alive()
            werr = self._warm_error.get(model)
        try:
            st = self._raw_status()
        except Exception as exc:  # status must not fail the page
            return {"model": model, "phase": "failed", "title": "Status unavailable", "detail": None,
                    "error": error_view(exc)}
        return summarize(st, model, warming=warming, warm_error=werr)

    def warmup(self, model: str) -> None:
        """Start enrolling / downloading / connecting ``model`` in the background (idempotent)."""
        with self._mu:
            t = self._warming.get(model)
            if t is not None and t.is_alive():
                return
            self._warm_error.pop(model, None)

            def run() -> None:
                try:
                    self.client.warmup(model)
                except Exception as exc:
                    with self._mu:
                        self._warm_error[model] = error_view(exc)

            t = threading.Thread(target=run, name=f"pgw-chat-warmup-{model}", daemon=True)
            self._warming[model] = t
            t.start()

    def stream(self, model: str, messages: list[dict[str, str]], params: dict[str, Any],
               conversation_id: str | None) -> ChatStream:
        kw: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                              "stream_options": {"include_usage": True}, **params}
        extra = dict(self._overrides.get("extra_body") or {})
        if conversation_id:
            extra["conversation_id"] = conversation_id  # exact prefix reuse, when the prefix cache is on
        kw.update({k: v for k, v in self._overrides.items() if k != "extra_body"})
        if extra:
            kw["extra_body"] = extra
        with self._mu:
            self._warm_error.pop(model, None)
        return _SDKStream(self.client.chat.completions.create(**kw))

    def close(self) -> None:
        self.client.close()
