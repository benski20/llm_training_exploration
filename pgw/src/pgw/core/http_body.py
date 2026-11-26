"""Bounded reading of small HTTP request bodies (JSON and form routes).

``await request.body()`` buffers the whole body before a route can look at its size. A ``Content-Length`` check in
front of it does not help against a chunked request, which has no ``Content-Length``: the body is read to the end
and only then compared with the limit, so one unauthenticated client can make the process buffer gigabytes
(security review 2026-11, S1). :func:`read_limited` reads the stream and stops as soon as the limit is passed.
"""

from __future__ import annotations

from typing import Any

from pgw.core.errors import PayloadTooLargeError

__all__ = ["read_limited"]


async def read_limited(request: Any, limit: int, *, what: str = "request body") -> bytes:
    """The request body, or :class:`PayloadTooLargeError` once more than ``limit`` bytes arrived.

    A declared ``Content-Length`` above the limit (or not a number) is refused before anything is read.

    Raises:
        PayloadTooLargeError: the body is (or declares to be) larger than ``limit`` bytes.
    """
    length = request.headers.get("content-length")
    if length is not None and (not length.strip().isdigit() or int(length) > limit):
        raise PayloadTooLargeError(f"{what} too large")
    buf = bytearray()
    async for chunk in request.stream():
        if chunk:
            buf += chunk
            if len(buf) > limit:
                raise PayloadTooLargeError(f"{what} too large")
    return bytes(buf)
