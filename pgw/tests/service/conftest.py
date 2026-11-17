"""Fixtures for the compute service tests (CPU only, tiny pack)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import torch

from pgw.core import RingParams, ringmm_reference
from pgw.service import ModelRegistry, ServiceSettings, create_app, hash_token
from pgw.wire import ClientConfig, ComputeClient, ComputeItem, ComputeRequest

TOKEN_A = "tok-tenant-a-0123456789"  # noqa: S105 - test credential
TOKEN_B = "tok-tenant-b-9876543210"  # noqa: S105 - test credential


@pytest.fixture(scope="session")
def pack_weights(tiny_pack: Any) -> dict[str, torch.Tensor]:
    """``{linear: int8 W8}`` of the tiny pack (reference for exactness checks)."""
    return {lw.id.key: lw.w8.clone() for lw in tiny_pack.iter_linears()}


@pytest.fixture(scope="session")
def pack_fp(tiny_pack: Any) -> str:
    return str(tiny_pack.fingerprint)


@pytest.fixture(scope="session")
def shared_registry(tiny_pack_dir: Path) -> ModelRegistry:
    """Registry with the tiny pack loaded on CPU (shared, read-only use)."""
    reg = ModelRegistry("cpu", rings=("K40A12", "K48A20", "K32A8"))
    reg.load(tiny_pack_dir)
    return reg


@pytest.fixture
def tokens_file(tmp_path: Path) -> Path:
    p = tmp_path / "tenants.txt"
    p.write_text(f"# tenants\ntenant-a:{hash_token(TOKEN_A)}\n\ntenant-b:{hash_token(TOKEN_B)}\n")
    return p


def rand_x(rows: int, n: int, ring: RingParams, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, ring.modulus, size=(rows, n), dtype=np.int64))


def expected(x: torch.Tensor, w8: torch.Tensor, ring: RingParams) -> torch.Tensor:
    return torch.from_numpy(ringmm_reference(x, w8, ring.k))


def make_req(fp: str, ring: RingParams, items: list[tuple[str, torch.Tensor]], rid: str = "r1",
             deadline_ms: int | None = None) -> ComputeRequest:
    from pgw.core import pack_ring

    return ComputeRequest(
        rid, fp, ring.name, tuple(ComputeItem(lid, int(x.shape[0]), pack_ring(x, ring)) for lid, x in items),
        deadline_ms=deadline_ms,
    )


@contextlib.asynccontextmanager
async def running_app(settings: ServiceSettings, **kw: Any) -> AsyncIterator[Any]:
    """Create the app and run its lifespan (startup/shutdown) around the block."""
    app = create_app(settings, **kw)
    async with app.router.lifespan_context(app):
        yield app


def asgi_client(app: Any, token: str | None = None, **cfg: Any) -> ComputeClient:
    from pydantic import SecretStr

    conf = ClientConfig(base_url="http://service.test", token=SecretStr(token) if token else None, **cfg)
    return ComputeClient(conf, transport=httpx.ASGITransport(app=app))


def http_client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://service.test")
