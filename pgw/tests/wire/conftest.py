"""Shared helpers for the wire tests: a fake compute handler doing real exact ring matmuls."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core import (
    ConfigError,
    FingerprintMismatchError,
    RingParams,
    pack_ring,
    ringmm_reference,
    unpack_ring,
)
from pgw.wire import (
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    LinearInfo,
    ModelInfo,
    ModelsResponse,
    RequestContext,
    ServerTiming,
    WireLimits,
)

FP = "ab" * 32
OTHER_FP = "cd" * 32
DIMS: dict[str, tuple[int, int]] = {"0.qkv": (24, 16), "0.o": (16, 16), "0.gate_up": (40, 16), "0.down": (16, 20)}


def make_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randint(-127, 128, (m, n), generator=g, dtype=torch.int64) for k, (m, n) in DIMS.items()}


def random_x_hat(rows: int, n: int, ring: RingParams, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64))


def make_request(
    ring: RingParams, spec: list[tuple[str, int]], *, fp: str = FP, rid: str = "req-1", seed: int = 0, deadline_ms=None
) -> tuple[ComputeRequest, list[torch.Tensor]]:
    xs = []
    items = []
    for i, (lid, rows) in enumerate(spec):
        x = random_x_hat(rows, DIMS[lid][1], ring, seed + i)
        xs.append(x)
        items.append(ComputeItem(lid, rows, pack_ring(x, ring)))
    return ComputeRequest(rid, fp, ring.name, tuple(items), deadline_ms=deadline_ms), xs


def expected_y(ring: RingParams, lid: str, x: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.from_numpy(ringmm_reference(x, weights[lid], ring.k))


class FakeHandler:
    """In-memory ComputeHandler: exact ``W8 x_hat mod 2^K`` with the reference matmul."""

    def __init__(self, *, rings=("K32A8", "K40A12", "K48A20"), delay_s: float = 0.0, max_rows: int = 8192,
                 max_bytes: int = 256 << 20, tokens: Mapping[str, str] | None = None):
        self.weights = make_weights()
        self.rings = tuple(rings)
        self.delay_s = delay_s
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.calls: list[ComputeRequest] = []
        self.contexts: list[RequestContext] = []
        self.done: list[tuple[int, str | None]] = []
        self.fail_with: BaseException | None = None
        self.corrupt = False
        self.tokens = tokens

    def linear_dims(self, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
        if fingerprint != FP:
            raise FingerprintMismatchError("unknown model fingerprint")
        if ring not in self.rings:
            raise ConfigError(f"ring {ring} not served")
        return DIMS

    async def compute(self, request: ComputeRequest, ctx: RequestContext) -> ComputeResponse:
        self.calls.append(request)
        self.contexts.append(ctx)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        ring = RingParams.from_name(request.ring)
        out = []
        for it in request.items:
            m, n = DIMS[it.linear]
            x = unpack_ring(it.x_hat, (it.rows, n), ring)
            y = expected_y(ring, it.linear, x, self.weights)
            data = pack_ring(y, ring)
            if self.corrupt:
                data = data[:-1]
            out.append(ComputeResult(it.linear, it.rows, data))
        return ComputeResponse(request.request_id, tuple(out), ServerTiming(0.1, 1.0, request.total_rows))

    async def models(self) -> ModelsResponse:
        linears = tuple(LinearInfo(k, m, n) for k, (m, n) in DIMS.items())
        return ModelsResponse(
            (ModelInfo("tiny", FP, self.rings, linears, self.max_rows, self.max_bytes),), server="fake"
        )

    def on_request_done(self, ctx: RequestContext, status: int, code: str | None) -> None:
        self.done.append((status, code))


class AuthHandler(FakeHandler):
    """FakeHandler with bearer-token authentication."""

    def authenticate(self, authorization: str | None, client_cert_subject: str | None) -> Any:
        from pgw.core import AuthError

        if authorization != "Bearer good-token":
            raise AuthError("invalid or missing bearer token")
        return {"tenant": "t1"}


def make_app(handler: Any, limits: WireLimits | None = None, **kw: Any) -> Any:
    from fastapi import FastAPI

    from pgw.wire import create_router

    app = FastAPI()
    app.include_router(create_router(handler, limits=limits, **kw))
    return app


@pytest.fixture
def handler() -> FakeHandler:
    return FakeHandler()


@pytest.fixture(params=["K32A8", "K40A12", "K48A20"])
def any_ring(request: pytest.FixtureRequest) -> RingParams:
    return RingParams.from_name(request.param)
