"""msgpack encoding/decoding of :mod:`pgw.wire.messages` with strict validation."""

from __future__ import annotations

from typing import TypeVar

__all__ = ["CONTENT_TYPE", "decode_message", "encode_message"]

#: Media type of every PGW body. The protocol major version is also in the ``PGW-Protocol`` header.
CONTENT_TYPE = "application/vnd.pgw+msgpack"

T = TypeVar("T")


def encode_message(msg: object) -> bytes:
    """Serialize any message dataclass to msgpack bytes (``use_bin_type=True``)."""
    raise NotImplementedError("wire team: encode_message")


def decode_message(data: bytes, expected: type[T], *, max_bytes: int | None = None) -> T:
    """Parse and validate a message of type ``expected``.

    Raises:
        ProtocolError: malformed msgpack, wrong ``type``, missing keys, bad field types.
        VersionMismatchError: ``v`` not in :data:`SUPPORTED_VERSIONS`.
        PayloadTooLargeError: ``len(data) > max_bytes``.
        ErrorResponse payloads are raised as the typed :class:`pgw.core.PGWError` when ``expected``
        is not :class:`ErrorResponse`.
    """
    raise NotImplementedError("wire team: decode_message")
