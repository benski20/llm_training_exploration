"""Gateway test fixtures.

* ``gw_pack`` - the shared tiny Qwen2 pack (``tests/conftest.py::tiny_pack_dir``) opened with
  :class:`pgw.modelpack.ModelPack`.
* ``InProcessService`` - a real ``pgw.wire`` compute router (msgpack over HTTP, validation, error
  mapping) on top of :class:`pgw.engine.LocalComputeClient`, reachable through an
  ``httpx.ASGITransport``. Optional ``tamper`` simulates a malicious service.
* ``make_store`` - a real encrypted :class:`pgw.maskstore.MaskStore` filled by the real
  :class:`pgw.factory.MaskGenerator`.
* ``gateway`` - async context manager running the app lifespan and yielding an ``httpx.AsyncClient``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch

from pgw.core import ConfigError, FingerprintMismatchError, RingParams
from pgw.engine import LocalComputeClient

SERVICE_URL = "http://127.0.0.1:18443"
ADMIN_TOKEN = "test-admin-token-0123456789"  # noqa: S105 - test fixture
MODEL = "tiny"


class LocalHandler:
    """``pgw.wire.ComputeHandler`` doing exact masked compute with a :class:`LocalComputeClient`."""

    def __init__(self, pack: Any, *, tamper: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
                 fingerprint: str | None = None, rings: tuple[str, ...] = ("K32A8", "K40A12", "K48A20")):
        self.pack = pack
        self.local = LocalComputeClient(pack, tamper=tamper, record=True, rings=rings)
        self.fp = fingerprint or pack.fingerprint
        self.rings = rings
        self.requests = 0

    def linear_dims(self, fingerprint: str, ring: str) -> Mapping[str, tuple[int, int]]:
        if fingerprint != self.fp:
            raise FingerprintMismatchError("unknown model fingerprint")
        if ring not in self.rings:
            raise ConfigError(f"ring {ring} not served")
        return {s.id.key: (s.out_features, s.in_features) for s in self.pack.linear_specs()}

    async def compute(self, request: Any, ctx: Any) -> Any:
        self.requests += 1
        return await self.local.compute(request)

    async def models(self) -> Any:
        from pgw.wire.messages import ModelsResponse

        m = await self.local.models()
        if self.fp != self.pack.fingerprint:
            info = m.models[0]
            info = type(info)(info.name, self.fp, info.rings, info.linears, info.max_rows_per_item,
                              info.max_request_bytes)
            return ModelsResponse((info,), server="test-service")
        return m


class InProcessService:
    """In-process compute service reachable via ``self.transport``."""

    def __init__(self, pack: Any, **kw: Any):
        from fastapi import FastAPI

        from pgw.wire.server import create_router

        self.handler = LocalHandler(pack, **kw)
        self.app = FastAPI()
        self.app.include_router(create_router(self.handler))
        self.transport = httpx.ASGITransport(app=self.app)

    @property
    def received(self) -> list[tuple[str, torch.Tensor]]:
        return self.handler.local.received

    def set_tamper(self, fn: Callable[[str, torch.Tensor], torch.Tensor] | None) -> None:
        self.handler.local.tamper = fn


@pytest.fixture(scope="session")
def gw_pack(tiny_pack_dir: Path) -> Any:
    from pgw.modelpack import ModelPack

    return ModelPack.open(tiny_pack_dir)


def fill_store(store: Any, pack: Any, ring: RingParams, slots: int, *, segment_slots: int = 512) -> None:
    from pgw.factory import MaskGenerator

    gen = MaskGenerator(pack, ring)
    left = slots
    while left > 0:
        n = min(left, segment_slots)
        gen.generate_into_store(store, n, checks=2, chunk_slots=16, factory_id="test")
        left -= n


@pytest.fixture
def make_store(tmp_path: Path, gw_pack: Any) -> Any:
    from pgw.maskstore import FileKeyProvider, MaskStore

    stores: list[Any] = []

    def make(slots: int, ring: RingParams | str = "K40A12", name: str = "store", **overrides: Any) -> Any:
        rp = RingParams.from_name(ring) if isinstance(ring, str) else ring
        kp = FileKeyProvider(tmp_path / f"{name}.kek", create=True)
        store = MaskStore.open(tmp_path / name, kp, fsync=False, low_water_slots=overrides.pop("low_water_slots", 64),
                               **overrides)
        if slots:
            fill_store(store, gw_pack, rp, slots)
        stores.append(store)
        return store

    yield make
    for s in stores:
        with contextlib.suppress(Exception):
            s.close()


def masked_settings(tiny_pack_dir: Path, tmp_path: Path, **kw: Any) -> Any:
    from pgw.gateway import GatewayModelConfig, GatewaySettings

    model_kw = kw.pop("model_kw", {})
    model = GatewayModelConfig(name=MODEL, pack_path=tiny_pack_dir, mode="masked", service_url=SERVICE_URL,
                               **model_kw)
    base: dict[str, Any] = {
        "models": [model], "admin_token": ADMIN_TOKEN, "maskstore_root": tmp_path / "unused-store",
        "audit_log_file": tmp_path / "audit.jsonl", "default_max_tokens": 8,
    }
    base.update(kw)
    return GatewaySettings(**base)


def plain_settings(tiny_pack_dir: Path, tmp_path: Path, mode: str = "plain", **kw: Any) -> Any:
    from pgw.gateway import GatewayModelConfig, GatewaySettings

    model_kw = kw.pop("model_kw", {})
    model = GatewayModelConfig(name=MODEL, pack_path=tiny_pack_dir, mode=mode, **model_kw)
    base: dict[str, Any] = {
        "models": [model], "admin_token": ADMIN_TOKEN, "allow_dev_modes": True,
        "audit_log_file": tmp_path / "audit.jsonl", "default_max_tokens": 8,
    }
    base.update(kw)
    return GatewaySettings(**base)


@contextlib.asynccontextmanager
async def gateway(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app lifespan and yield a client talking to it in-process."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test", timeout=120) as client:
            yield client


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def parse_sse(body: str) -> list[Any]:
    """``data:`` payloads of an SSE body (JSON decoded; ``[DONE]`` kept as the string)."""
    import json

    out: list[Any] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), block
        data = block[len("data: "):]
        out.append(data if data == "[DONE]" else json.loads(data))
    return out
