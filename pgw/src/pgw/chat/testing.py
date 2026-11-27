"""A fake backend for tests, demos and screenshots: canned streamed answers, no model, no provider.

Used by ``pgw chat --demo`` (the page then shows a "demo: canned answers" banner) and by the tests. Special
messages drive the states the UI must show: ``/error <kind>`` raises the library's typed error of that kind
(``integrity``, ``stalled``, ``connection``, ``not_ready``, ``auth``), ``/slow`` streams slowly (for the stop button
and slow-link behaviour), ``/long`` streams a long markdown answer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from pgw.chat.backend import MODEL_INFO

__all__ = ["FakeBackend"]

CANNED = ("Here is a short answer, streamed word by word.\n\n"
          "* Every linear layer ran on **masked** data on the GPU.\n"
          "* Every result was checked on this computer before use.\n\n"
          "```python\nprint('hello from a canned answer')\n```\n")

LONG = ("## Masking, in brief\n\nA one-time pad hides each activation: the provider computes on `x + r` and this "
        "computer removes `W r` afterwards.\n\n1. Draw a fresh mask.\n2. Send the masked row.\n3. Check the result "
        "with Freivalds' test.\n\n> Nothing in plaintext leaves this computer.\n\n| step | where |\n|---|---|\n"
        "| mask | this computer |\n| matrix product | GPU |\n| check | this computer |\n")


class _Stream:
    def __init__(self, text: str, delay: float, exc: BaseException | None):
        self._text, self._delay, self._exc = text, delay, exc
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[str]:
        if self._exc is not None:
            time.sleep(self._delay)
            raise self._exc
        words = self._text.split(" ")
        for i, w in enumerate(words):
            if self.closed.wait(self._delay):
                return
            yield w + (" " if i < len(words) - 1 else "")
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 20, "completion_tokens": len(words), "cached_tokens": 0}

    def close(self) -> None:
        self.closed.set()


def _error(kind: str) -> BaseException:
    from pgw.sdk import errors as E

    return {"integrity": E.IntegrityError("result failed its Freivalds check"),
            "stalled": E.ProviderStalledError("the provider's mask generation made no progress for 31s"),
            "connection": E.APIConnectionError("connection failed (ConnectError)"),
            "not_ready": E.NotReadyError("no masks in stock after 120s"),
            "auth": E.AuthenticationError("invalid, revoked or expired SDK key")}.get(
        kind, E.InternalServerError("internal error (RuntimeError)"))


class FakeBackend:
    def __init__(self, *, delay: float = 0.02, phase: str = "ready"):
        self.delay = delay
        self.phase = phase
        self.requests: list[dict[str, Any]] = []
        self.verified = 0
        self.streams: list[_Stream] = []
        self.warmed: list[str] = []

    def settings(self) -> dict[str, Any]:
        return {"mask_scheme": "pks-v3", "masked_head": "auto", "prefix_cache": "auto", "speculation": "off",
                "factory_weights": "stream", "usage_reports": True, "service_url": "https://gpu.example.com"}

    def models(self) -> list[dict[str, Any]]:
        from pgw.sdk.options import SUPPORTED_MODELS

        return [{"id": m, **MODEL_INFO.get(m, {"label": m})} for m in SUPPORTED_MODELS]

    def status(self, model: str) -> dict[str, Any]:
        titles = {"ready": "Ready", "setup": "First-time setup on this machine", "stalled": "Provider unreachable",
                  "provisioning": "Downloading the public model files"}
        setup = self.phase == "setup"
        return {"model": model, "phase": self.phase, "title": titles.get(self.phase, self.phase),
                "detail": "Building the one-time pks-v3 hint, 42% done (about 6 min left), then the first mask "
                          "segment" if setup else None,
                "percent": 42.0 if setup else None, "eta_s": 360 if setup else None,
                "masks_available": 5000.0, "requests": self.verified, "verified_requests": self.verified,
                "integrity_failures": 0, "last_verified_at": time.time() if self.verified else None,
                "lane": "open", "masked_head": True, "prefix_cache": False, "mask_scheme": "pks-v3", "error": None}

    def warmup(self, model: str) -> None:
        self.warmed.append(model)

    def stream(self, model: str, messages: list[dict[str, str]], params: dict[str, Any],
               conversation_id: str | None) -> _Stream:
        self.requests.append({"model": model, "n_messages": len(messages), "params": dict(params),
                              "conversation_id": conversation_id})
        last = messages[-1]["content"].strip()
        exc, delay, text = None, self.delay, CANNED
        if last.startswith("/error"):
            exc = _error(last.split(" ", 1)[1].strip() if " " in last else "")
        elif last.startswith("/slow"):
            delay = max(delay, 0.4)
        elif last.startswith("/long"):
            text = LONG
        st = _Stream(text, delay, exc)
        self.streams.append(st)
        if exc is None:
            self.verified += 1
        return st

    def close(self) -> None:
        return None
