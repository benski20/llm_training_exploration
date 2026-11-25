"""``pgw.OpenAI`` / ``pgw.AsyncOpenAI``: drop-in clients whose gateway runs in this process.

.. code-block:: python

    from pgw import OpenAI

    client = OpenAI()                                   # PGW_API_KEY from the environment
    r = client.chat.completions.create(model="qwen2.5-7b-instruct",
                                       messages=[{"role": "user", "content": "Hello"}])
    print(r.choices[0].message.content)
    for chunk in client.chat.completions.create(model="qwen2.5-7b-instruct", messages=[...], stream=True):
        print(chunk.choices[0].delta.content or "", end="")

Construction parses the key (format errors are raised at once) and does nothing else; the first request
enrolls this installation, provisions the model and connects. Clients are thread-safe; clients with the same
key and directories in one process share one engine. Close a client (``close()``, ``with``) or let the
interpreter exit (``atexit``) to stop its background factories cleanly.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from pgw.sdk import types as T
from pgw.sdk.errors import BadRequestError, PGWAPIError, from_pgw_error
from pgw.sdk.keys import SdkKey, parse_key
from pgw.sdk.options import SDKOptions

__all__ = ["AsyncOpenAI", "OpenAI"]

_UNSUPPORTED = ("tools", "functions", "response_format", "n", "audio", "modalities", "prediction")


# ---- shared engines -------------------------------------------------------------------------------------
class _Registry:
    """Process-wide engines keyed by (key id, provider, cache dir, state dir), reference counted."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._engines: dict[tuple[str, ...], list[Any]] = {}

    def acquire(self, key: SdkKey, options: SDKOptions, provisioner: Any, enroll_transport: Any) -> Any:
        from pgw.sdk.cache import CacheLayout
        from pgw.sdk.engine import Engine

        lay = CacheLayout(options.cache_dir, options.state_dir)
        k = (key.key_id_hex, key.service_url, str(lay.root.resolve()), str(lay.state.resolve()))
        with self._mu:
            hit = self._engines.get(k)
            if hit is not None and not hit[0]._closed and hit[0].pid == os.getpid():
                hit[1] += 1
                return hit[0]
            eng = Engine(key, options, provisioner=provisioner, enroll_transport=enroll_transport)
            self._engines[k] = [eng, 1]
            return eng

    def release(self, engine: Any) -> None:
        with self._mu:
            for k, v in list(self._engines.items()):
                if v[0] is engine:
                    v[1] -= 1
                    if v[1] > 0:
                        return
                    del self._engines[k]
                    break
        engine.close()

    def close_all(self) -> None:
        with self._mu:
            engines = [v[0] for v in self._engines.values()]
            self._engines.clear()
        for e in engines:
            with contextlib.suppress(Exception):
                e.close(timeout_s=10.0)

    def after_fork(self) -> None:
        self._mu = threading.Lock()
        self._engines = {}


_REGISTRY = _Registry()
atexit.register(_REGISTRY.close_all)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_REGISTRY.after_fork)


def _configure_logging(level: str | None) -> None:
    import logging

    root = logging.getLogger("pgw")
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())  # silent by default: no lastResort output on stderr
    if level:
        from pgw.core.log import JsonFormatter

        if not any(getattr(h, "_pgw_sdk", False) for h in root.handlers):
            h = logging.StreamHandler()
            h.setFormatter(JsonFormatter())
            h._pgw_sdk = True  # type: ignore[attr-defined]
            root.addHandler(h)
        root.setLevel(level.upper())


def _build_request(kwargs: dict[str, Any]) -> Any:
    """Validate ``create(**kwargs)`` into the gateway's :class:`ChatCompletionRequest` (same rules as the API)."""
    from pydantic import ValidationError

    from pgw.gateway.openai_schemas import ChatCompletionRequest

    body = {k: v for k, v in kwargs.items() if v is not None}
    extra_body = body.pop("extra_body", None) or {}
    body.pop("extra_headers", None)
    body.pop("extra_query", None)
    body.pop("timeout", None)
    if isinstance(extra_body, dict):
        body.update(extra_body)
    msgs = body.get("messages")
    if msgs is not None:
        body["messages"] = [m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else dict(m))
                            for m in msgs]
    if "stop" in body and isinstance(body["stop"], tuple):
        body["stop"] = list(body["stop"])
    if body.get("n", 1) != 1:
        raise BadRequestError("only n=1 is supported")
    try:
        return ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request", "loc": ()}
        loc = ".".join(str(x) for x in first.get("loc", ()))
        raise BadRequestError(f"invalid request{(' (' + loc + ')') if loc else ''}: {first['msg']}") from exc


def _conversation(kwargs: dict[str, Any]) -> str | None:
    """Conversation id for exact prefix reuse: ``extra_body={"conversation_id": ...}`` or the gateway's header."""
    eb = kwargs.get("extra_body") or {}
    conv = eb.get("conversation_id") if isinstance(eb, dict) else None
    if conv is None:
        hdrs = {str(k).lower(): v for k, v in (kwargs.get("extra_headers") or {}).items()}
        conv = hdrs.get("x-pgw-conversation-id")
    if conv is not None and (not isinstance(conv, str) or len(conv) > 128):
        raise BadRequestError("conversation_id must be a string of at most 128 characters")
    return conv


def _strip_conversation(kwargs: dict[str, Any]) -> dict[str, Any]:
    eb = kwargs.get("extra_body")
    if isinstance(eb, dict) and "conversation_id" in eb:
        kwargs = dict(kwargs)
        kwargs["extra_body"] = {k: v for k, v in eb.items() if k != "conversation_id"}
    return kwargs


class _Base:
    def __init__(self, api_key: str | None, *, options: SDKOptions | None, provisioner: Any,
                 enroll_transport: Any, timeout: float | None, max_retries: int | None, kw: dict[str, Any]):
        for ignored in ("base_url", "organization", "project", "default_headers", "default_query", "http_client",
                        "websocket_base_url", "_strict_response_validation"):
            kw.pop(ignored, None)  # accepted for drop-in compatibility: the provider comes from the key
        opts = options or SDKOptions.from_env(**kw)
        if options is not None and kw:
            opts = opts.with_(**kw)
        if timeout is not None:
            opts = opts.with_(timeout_s=float(timeout))
        if max_retries is not None:
            opts = opts.with_(max_retries=int(max_retries))
        opts.validate()
        _configure_logging(opts.log_level)
        try:
            self._key = parse_key(api_key if api_key is not None else os.environ.get("PGW_API_KEY"))
        except PGWAPIError:
            raise
        except Exception as exc:
            raise from_pgw_error(exc) from None
        self.options = opts
        self._engine = _REGISTRY.acquire(self._key, opts, provisioner, enroll_transport)
        self._closed = False

    @property
    def api_key_id(self) -> str:
        """The public id of the API key (safe to log)."""
        return self._key.key_id_hex

    def status(self) -> dict[str, Any]:
        """Session, models (state, lane, native stack, mask stock, factory, last integrity check), cache paths."""
        return dict(self._engine.status())


class _SyncStream:
    """Iterator over the chunks of a streaming chat completion (``close()`` or leaving the ``with`` block cancels
    generation and burns the request's unused masks)."""

    _END = object()

    def __init__(self, engine: Any, req: Any, conversation: str | None, timeout_s: float):
        self._q: queue.Queue[Any] = queue.Queue()
        self._timeout_s = timeout_s
        self._done = False

        async def pump() -> None:
            try:
                async for c in engine.chat_stream(req, conversation=conversation):
                    self._q.put(c)
            except BaseException as exc:
                self._q.put(exc if isinstance(exc, Exception) else from_pgw_error(exc))
                if not isinstance(exc, Exception):
                    raise
            finally:
                self._q.put(self._END)

        self._fut = engine.submit(pump())

    def __iter__(self) -> Iterator[T.ChatCompletionChunk]:
        return self

    def __next__(self) -> T.ChatCompletionChunk:
        if self._done:
            raise StopIteration
        try:
            item = self._q.get(timeout=self._timeout_s)
        except queue.Empty:
            self.close()
            from pgw.sdk.errors import APITimeoutError

            raise APITimeoutError("the stream produced nothing within the timeout") from None
        if item is self._END:
            self._done = True
            raise StopIteration
        if isinstance(item, BaseException):
            self._done = True
            raise from_pgw_error(item) if not isinstance(item, PGWAPIError) else item
        return item  # type: ignore[no-any-return]

    def close(self) -> None:
        """Stop generation (idempotent)."""
        self._done = True
        if not self._fut.done():
            self._fut.cancel()

    def __enter__(self) -> _SyncStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class _Completions:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(self, **kwargs: Any) -> T.ChatCompletion | _SyncStream:
        """``chat.completions.create(model=..., messages=..., stream=False, temperature, top_p, max_tokens,
        max_completion_tokens, seed, stop, logprobs, top_logprobs, user, stream_options, extra_body)``.

        Returns:
            :class:`ChatCompletion`, or with ``stream=True`` an iterator of :class:`ChatCompletionChunk`.
        """
        o = self._owner
        o._check_open()
        conv = _conversation(kwargs)
        req = _build_request(_strip_conversation(kwargs))
        eng = o._engine
        if req.stream:
            return _SyncStream(eng, req, conv, o.options.timeout_s + o.options.ready_timeout_s)
        try:
            return eng.call(eng.chat(req, conversation=conv))  # type: ignore[no-any-return]
        except PGWAPIError:
            raise
        except Exception as exc:
            raise from_pgw_error(exc) from exc


class _Chat:
    def __init__(self, owner: Any):
        self.completions = _Completions(owner) if isinstance(owner, OpenAI) else _AsyncCompletions(owner)


class _Models:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def list(self) -> T.ModelList:
        """Models the provider serves to this key and this release supports (enrolls on first use)."""
        o = self._owner
        o._check_open()
        names = o._engine.call(o._engine.list_models(), timeout_s=o.options.timeout_s)
        return T.ModelList(data=[T.Model(id=n) for n in names])

    def retrieve(self, model: str) -> T.Model:
        """One model (raises :class:`NotFoundError` if not served)."""
        from pgw.sdk.errors import NotFoundError
        from pgw.sdk.options import canonical_model

        name = canonical_model(model)
        if name not in {m.id for m in self.list()}:
            raise NotFoundError(f"model {model!r} is not served to this API key")
        return T.Model(id=name)


class OpenAI(_Base):
    """Synchronous client (thread-safe).

    Args:
        api_key: The SDK key (default ``PGW_API_KEY``).
        timeout: Per-request budget in seconds (default 600).
        max_retries: Retries after transport failures before the first token (default 2).
        options: :class:`SDKOptions`; keyword arguments of the same names override fields (``cache_dir=...``,
            ``mask_workers=...``, ``mask_scheme=...``, ...).
        provisioner: Provisioning function (default :func:`pgw.provision.client.provision`).
        base_url, organization, project, default_headers, http_client: accepted for drop-in compatibility and
            ignored (the provider is named by the key).
    """

    def __init__(self, api_key: str | None = None, *, timeout: float | None = None, max_retries: int | None = None,
                 options: SDKOptions | None = None, provisioner: Any = None, _enroll_transport: Any = None,
                 **kw: Any):
        super().__init__(api_key, options=options, provisioner=provisioner, enroll_transport=_enroll_transport,
                         timeout=timeout, max_retries=max_retries, kw=kw)
        self.chat = _Chat(self)
        self.models = _Models(self)

    def _check_open(self) -> None:
        if self._closed:
            raise PGWAPIError("the pgw client is closed")

    def warmup(self, model: str) -> None:
        """Enroll, provision and connect ``model`` now (otherwise the first request does it)."""
        self._check_open()
        self._engine.call(self._engine.ensure_model(model), timeout_s=self.options.ready_timeout_s)

    def reset_integrity_breaker(self, model: str) -> None:
        """Re-open ``model`` after an integrity failure blocked it (do this only after investigating)."""
        self._engine.call(self._engine.reset_breaker(model), timeout_s=30)

    def close(self) -> None:
        """Release this client (the shared engine stops when its last client closes)."""
        if not self._closed:
            self._closed = True
            _REGISTRY.release(self._engine)

    def __enter__(self) -> OpenAI:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---- async -----------------------------------------------------------------------------------------------
class _AsyncStream:
    """Async iterator over the chunks of a streaming chat completion (``aclose()`` cancels generation)."""

    _END = object()

    def __init__(self, engine: Any, req: Any, conversation: str | None, timeout_s: float):
        self._engine = engine
        self._req = req
        self._conv = conversation
        self._timeout_s = timeout_s
        self._q: asyncio.Queue[Any] | None = None
        self._fut: Any = None
        self._done = False

    def _start(self) -> None:
        caller = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._q = q
        end = self._END

        def put(item: Any) -> None:
            with contextlib.suppress(RuntimeError):  # the caller's loop closed
                caller.call_soon_threadsafe(q.put_nowait, item)

        async def pump() -> None:
            try:
                async for c in self._engine.chat_stream(self._req, conversation=self._conv):
                    put(c)
            except BaseException as exc:
                put(exc if isinstance(exc, Exception) else from_pgw_error(exc))
                if not isinstance(exc, Exception):
                    raise
            finally:
                put(end)

        self._fut = self._engine.submit(pump())

    def __aiter__(self) -> AsyncIterator[T.ChatCompletionChunk]:
        return self

    async def __anext__(self) -> T.ChatCompletionChunk:
        if self._done:
            raise StopAsyncIteration
        if self._q is None:
            self._start()
        assert self._q is not None
        try:
            item = await asyncio.wait_for(self._q.get(), timeout=self._timeout_s)
        except TimeoutError:
            await self.aclose()
            from pgw.sdk.errors import APITimeoutError

            raise APITimeoutError("the stream produced nothing within the timeout") from None
        if item is self._END:
            self._done = True
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._done = True
            raise from_pgw_error(item) if not isinstance(item, PGWAPIError) else item
        return item  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        """Stop generation (idempotent)."""
        self._done = True
        if self._fut is not None and not self._fut.done():
            self._fut.cancel()

    async def __aenter__(self) -> _AsyncStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class _AsyncCompletions:
    def __init__(self, owner: AsyncOpenAI):
        self._owner = owner

    async def create(self, **kwargs: Any) -> T.ChatCompletion | _AsyncStream:
        """As :meth:`OpenAI.chat.completions.create`; with ``stream=True`` an async iterator of chunks."""
        o = self._owner
        o._check_open()
        conv = _conversation(kwargs)
        req = _build_request(_strip_conversation(kwargs))
        eng = o._engine
        if req.stream:
            return _AsyncStream(eng, req, conv, o.options.timeout_s + o.options.ready_timeout_s)
        fut = eng.submit(eng.chat(req, conversation=conv))
        try:
            return await asyncio.wrap_future(fut)
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except PGWAPIError:
            raise
        except Exception as exc:
            raise from_pgw_error(exc) from exc


class _AsyncModels:
    def __init__(self, owner: AsyncOpenAI):
        self._owner = owner

    async def list(self) -> T.ModelList:
        """Models the provider serves to this key and this release supports."""
        o = self._owner
        o._check_open()
        names = await asyncio.wrap_future(o._engine.submit(o._engine.list_models()))
        return T.ModelList(data=[T.Model(id=n) for n in names])

    async def retrieve(self, model: str) -> T.Model:
        """One model (raises :class:`NotFoundError` if not served)."""
        from pgw.sdk.errors import NotFoundError
        from pgw.sdk.options import canonical_model

        name = canonical_model(model)
        if name not in {m.id for m in await self.list()}:
            raise NotFoundError(f"model {model!r} is not served to this API key")
        return T.Model(id=name)


class AsyncOpenAI(_Base):
    """Asyncio client: the same as :class:`OpenAI` with ``await``-able methods (any event loop; the gateway runs
    on its own loop thread, so a slow request never blocks the caller's loop)."""

    def __init__(self, api_key: str | None = None, *, timeout: float | None = None, max_retries: int | None = None,
                 options: SDKOptions | None = None, provisioner: Any = None, _enroll_transport: Any = None,
                 **kw: Any):
        super().__init__(api_key, options=options, provisioner=provisioner, enroll_transport=_enroll_transport,
                         timeout=timeout, max_retries=max_retries, kw=kw)
        self.chat = _Chat(self)
        self.models = _AsyncModels(self)

    def _check_open(self) -> None:
        if self._closed:
            raise PGWAPIError("the pgw client is closed")

    async def warmup(self, model: str) -> None:
        """Enroll, provision and connect ``model`` now."""
        self._check_open()
        await asyncio.wrap_future(self._engine.submit(self._engine.ensure_model(model)))

    async def reset_integrity_breaker(self, model: str) -> None:
        """Re-open ``model`` after an integrity failure blocked it (only after investigating)."""
        await asyncio.wrap_future(self._engine.submit(self._engine.reset_breaker(model)))

    async def close(self) -> None:
        """Release this client (the shared engine stops when its last client closes)."""
        if not self._closed:
            self._closed = True
            await asyncio.to_thread(_REGISTRY.release, self._engine)

    async def __aenter__(self) -> AsyncOpenAI:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
