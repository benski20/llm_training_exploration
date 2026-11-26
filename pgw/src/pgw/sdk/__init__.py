"""PGW client library: an OpenAI-compatible client whose private gateway runs in this process.

.. code-block:: python

    from pgw import OpenAI                       # or: from pgw.sdk import OpenAI, AsyncOpenAI
    client = OpenAI()                            # reads PGW_API_KEY
    client.chat.completions.create(model="qwen2.5-7b-instruct", messages=[...], stream=True)

The provider (the GPU compute service) only ever sees masked data; every result it returns is checked
(Freivalds / prime-field / range checks) before it is used, and a failed check is a hard
:class:`IntegrityError`. Modules:

========================  ========================================================================
``pgw.sdk.client``        :class:`OpenAI`, :class:`AsyncOpenAI` (public API)
``pgw.sdk.keys``          SDK key format, :func:`parse_key`
``pgw.sdk.errors``        typed exceptions
``pgw.sdk.types``         response objects shaped like the ``openai`` package's
``pgw.sdk.options``       :class:`SDKOptions`, supported models
``pgw.sdk.session``       enrollment (``POST /pgw/v1/sdk/enroll``) and the cached session files
``pgw.sdk.provisioning``  the provisioning adapter (:func:`pgw.provision.client.provision`, local packs)
``pgw.sdk.engine``        the in-process gateway (existing runtime, native stack, back-pressure)
``pgw.sdk.factory``       the background mask factory processes
``pgw.sdk.cache``         cache / state directories and file locks
``pgw.sdk.cli``           ``pgw sdk status`` / ``pgw sdk doctor``
========================  ========================================================================

Imports are lazy: ``import pgw.sdk`` loads nothing heavy (no torch) until a client is created.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AsyncOpenAI",
    "AuthenticationError",
    "BadRequestError",
    "IntegrityError",
    "InternalServerError",
    "LocalPackProvisioner",
    "NotFoundError",
    "NotReadyError",
    "OpenAI",
    "PGWAPIError",
    "PermissionDeniedError",
    "RateLimitError",
    "SDKOptions",
    "SdkKey",
    "UnsupportedModelError",
    "parse_key",
    "prepare",
]

_ERRORS = ("APIConnectionError", "APIError", "APITimeoutError", "AuthenticationError", "BadRequestError",
           "IntegrityError", "InternalServerError", "NotFoundError", "NotReadyError", "PGWAPIError",
           "PermissionDeniedError", "RateLimitError", "UnsupportedModelError")

_LAZY: dict[str, str] = {
    "OpenAI": "pgw.sdk.client",
    "AsyncOpenAI": "pgw.sdk.client",
    "SDKOptions": "pgw.sdk.options",
    "SdkKey": "pgw.sdk.keys",
    "parse_key": "pgw.sdk.keys",
    "LocalPackProvisioner": "pgw.sdk.provisioning",
    "prepare": "pgw.sdk.prepare",
    **dict.fromkeys(_ERRORS, "pgw.sdk.errors"),
}


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'pgw.sdk' has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
