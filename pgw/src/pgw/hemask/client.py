"""Gateway/factory side of ``POST /pgw/v1/maskgen`` on top of :class:`pgw.wire.ComputeClient`."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from pgw.hemask.wire import (
    MASKGEN_PATH,
    MaskgenItem,
    MaskgenRequest,
    decode_maskgen_response,
    encode_maskgen_request,
)

__all__ = ["RemoteMulmod64"]


class RemoteMulmod64:
    """``W8 b mod 2^64`` computed by the compute service for one model pack.

    Args:
        client: A connected :class:`pgw.wire.ComputeClient` (TLS, token, retries come from it).
        fingerprint: Pack fingerprint the service serves.
        out_features: ``{linear: m}`` used to validate response sizes.
        timeout_s: Per-call budget (large GEMMs on CPU services can take seconds).
    """

    def __init__(self, client: Any, fingerprint: str, out_features: dict[str, int], *, timeout_s: float = 600.0):
        self.client = client
        self.fingerprint = fingerprint
        self.out_features = dict(out_features)
        self.timeout_s = timeout_s
        self.bytes_up = 0
        self.bytes_down = 0
        self.calls = 0

    def _request(self, linear: str, b: np.ndarray) -> tuple[MaskgenRequest, bytes]:
        if b.dtype != np.uint64 or b.ndim != 2:
            raise ValueError("ciphertext must be a 2-D uint64 array")
        req = MaskgenRequest(uuid.uuid4().hex, self.fingerprint,
                             (MaskgenItem(linear, int(b.shape[1]), np.ascontiguousarray(b, dtype="<u8").tobytes()),))
        return req, encode_maskgen_request(req)

    def _decode(self, req: MaskgenRequest, linear: str, cols: int) -> Any:
        def decode(data: bytes) -> np.ndarray:
            resp = decode_maskgen_response(data, req, self.out_features)
            self.bytes_down += len(data)
            return np.frombuffer(resp.items[0].y, dtype="<u8").reshape(self.out_features[linear], cols).astype(
                np.uint64)
        return decode

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        """Blocking call (from a worker thread)."""
        req, body = self._request(linear, b)
        self.calls += 1
        self.bytes_up += len(body)
        out: np.ndarray = self.client.post_raw_sync(MASKGEN_PATH, body, req.request_id,
                                                     self._decode(req, linear, int(b.shape[1])),
                                                     timeout_s=self.timeout_s)
        return out

    async def acall(self, linear: str, b: np.ndarray) -> np.ndarray:
        req, body = self._request(linear, b)
        self.calls += 1
        self.bytes_up += len(body)
        out: np.ndarray = await self.client.post_raw(MASKGEN_PATH, body, req.request_id,
                                                      self._decode(req, linear, int(b.shape[1])),
                                                      timeout_s=self.timeout_s)
        return out
