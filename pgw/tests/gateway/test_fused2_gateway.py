"""The gateway service with ``schedule = "fused2"`` (opt-in): masked == plain bit for bit through the store.

A real gateway app (lifespan, admission, reservations, the engine's generator, the OpenAI API) with a fused2 model:
the compute service is the in-process wire router (models, authentication) plus ``POST /pgw/v1/lane/open`` and a
real native ``LaneService`` + ``LaneManager`` serving the fused2 lane entry derived from the provider's own pack
(the Triton pair op on CPU). Masks: the fused2 factory (:class:`pgw.factory.fused2_gen.Fused2MaskGenerator`) fills
a real encrypted mask store under the fused2 lane fingerprint. The plain dev mode with the same setting is the
fused2 plaintext reference; logprobs (and text) must be identical. Also: admin status, native driver on/off,
window and incremental reservations, the noise monitor on fused2 payloads, a provider serving other weights
refused, tampering caught (fail closed), config validation, defaults off.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from pgw.core.ring import RingParams
from pgw.gateway import ApiKeyStore, GatewayModelConfig, create_app
from tests.gateway.conftest import (
    ADMIN,
    MODEL,
    SERVICE_URL,
    InProcessService,
    auth,
    gateway,
    masked_settings,
    plain_settings,
)

fast = pytest.importorskip("pgw_fast")
if not hasattr(fast, "TripChannel"):
    pytest.skip("pgw_fast without the lane client", allow_module_level=True)

RING = RingParams(56, 12, 53)
F2 = {"ring": RING.name, "schedule": "fused2", "float_ops": "native"}
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 6, "ignore_eos": True, "logprobs": True, "top_logprobs": 3}


@pytest.fixture(scope="module")
def f2w(tiny_pack_dir: Path) -> Any:
    from pgw.engine.fused2 import Fused2Weights, migration_multipliers
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(tiny_pack_dir)
    return Fused2Weights(pack, None, multipliers=migration_multipliers(pack, 8))


@pytest.fixture(scope="module")
def entry(f2w: Any) -> Any:
    from pgw.service.fused2_lane import Fused2LaneEntry

    return Fused2LaneEntry.from_weights(f2w, RING, device="cpu", max_rows=8)


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    import shutil
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="pgwf2g"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


class F2Service(InProcessService):
    """The in-process compute service plus the trip lane (native ``LaneService``) serving a fused2 entry."""

    def __init__(self, pack: Any, entry: Any, sock_dir: Path, *, evil: bool = False):
        from pgw.service import ModelRegistry, ServiceMetrics, ServiceSettings
        from pgw.service.lane import LaneManager, add_lane_routes

        super().__init__(pack)
        h = self.handler
        h.authenticate = lambda authz, subject: None  # type: ignore[attr-defined]
        h.precheck = lambda: None  # type: ignore[attr-defined]
        h.draining = False  # type: ignore[attr-defined]
        h.lane_admission = None  # type: ignore[attr-defined]
        self.calls: list[tuple[int, int]] = []
        state = SimpleNamespace(registry=ModelRegistry("cpu"), handler=h, metrics=ServiceMetrics(enabled=False),
                                fused2={entry.fingerprint: entry})
        s = ServiceSettings(device="cpu", allow_unauthenticated=True, lane="uds", lane_socket=sock_dir / "lane.sock")
        self.mgr = LaneManager(s, state)
        real = self.mgr._py_execute

        def spy(slot: int, linear: int, ring: int, rows: int, body: bytes, tenant: str, left: float) -> Any:
            self.calls.append((linear, rows))
            out = real(slot, linear, ring, rows, body, tenant, left)
            if evil and isinstance(out, bytes) and linear == 3:  # a dishonest provider: one wrong output element
                b = bytearray(out)
                b[9] ^= 1
                return bytes(b)
            return out

        self.mgr._py_execute = spy  # type: ignore[method-assign]
        self._loop = asyncio.new_event_loop()
        assert self.mgr._start_sync(self._loop)
        state.lane = self.mgr
        add_lane_routes(self.app, state)

    def close(self) -> None:
        self.mgr.svc.stop(2.0)
        self._loop.close()


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def _fill(store: Any, f2w: Any, slots: int, segments: int = 1) -> str:
    from pgw.factory.fused2_gen import Fused2MaskGenerator

    gen = Fused2MaskGenerator(f2w, RING, chunk_rows=16)
    for _ in range(segments):
        gen.generate_into_store(store, slots)
    return gen.fingerprint


def _masked_app(tiny_pack_dir: Path, tmp_path: Path, store: Any, svc: InProcessService, keys: ApiKeyStore,
                **model_kw: Any) -> Any:
    reservation = model_kw.pop("_reservation", "window")
    st = masked_settings(tiny_pack_dir, tmp_path, model_kw={**F2, "lane": "auto", **model_kw},
                         mask_reservation=reservation)
    return create_app(st, mask_store=store, transports={MODEL: svc.transport}, key_store=keys)


async def _plain(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str, req: dict[str, Any]) -> Any:
    app = create_app(plain_settings(tiny_pack_dir, tmp_path, model_kw=dict(F2)), key_store=keys)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=req, headers=auth(key))
        assert r.status_code == 200, r.text
        return r.json()


@pytest.mark.parametrize(("native", "reservation"), [("auto", "window"), ("off", "incremental"),
                                                     ("auto", "incremental")])
async def test_fused2_gateway_masked_equals_plain(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                                  keys: ApiKeyStore, key: str, f2w: Any, entry: Any, sock_dir: Path,
                                                  native: str, reservation: str) -> None:
    store = make_store(0)
    fp = _fill(store, f2w, 64, segments=3)
    assert fp == entry.fingerprint
    svc = F2Service(gw_pack, entry, sock_dir)
    try:
        app = _masked_app(tiny_pack_dir, tmp_path, store, svc, keys, fused2_decode_native=native,
                          _reservation=reservation)
        async with gateway(app) as c:
            before = store.available_slots(fp, RING)
            r1 = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
            assert r1.status_code == 200, r1.text
            req2 = {**CHAT, "messages": [{"role": "user", "content": "What is two plus two?"}], "max_tokens": 4}
            r2 = await c.post("/v1/chat/completions", json=req2, headers=auth(key))
            assert r2.status_code == 200, r2.text
            st = (await c.get("/admin/status", headers=ADMIN)).json()
            used = before - store.available_slots(fp, RING)
        p1 = await _plain(tiny_pack_dir, tmp_path, keys, key, CHAT)
        p2 = await _plain(tiny_pack_dir, tmp_path, keys, key, req2)
    finally:
        svc.close()
    for m, p in ((r1.json(), p1), (r2.json(), p2)):
        a, b = m["choices"][0], p["choices"][0]
        assert a["message"] == b["message"]
        assert a["logprobs"] == b["logprobs"]  # exact float equality, top alternatives included
    model = next(x for x in st["models"] if x["name"] == MODEL) if "models" in st else st[MODEL]
    assert model["schedule"] == "fused2" and model["ring"] == RING.name
    f2 = model["fused2"]
    assert f2["lane_fingerprint"] == entry.fingerprint == model["fingerprint"]
    assert f2["base_fingerprint"] == gw_pack.fingerprint
    assert f2["native_decode"] is (native == "auto")
    if native == "auto":
        assert f2["native_passes"] > 0  # single-sequence decode passes ran in the native driver
    L = int(gw_pack.arch.num_layers)
    assert f2["trips_per_token"] == 2 * L + 1 and f2["counters"]["checks"] > 0
    assert model["noise_monitor"]["state"] == "active"
    assert svc.calls and {li for li, _ in svc.calls} == set(range(2 * L + 1))
    # one mask slot per token row (prompt + generated - 1 per request), unused rows burned on close
    assert used >= sum(x["usage"]["prompt_tokens"] + x["usage"]["completion_tokens"] - 1 for x in (r1.json(), r2.json()))


async def test_fused2_gateway_tamper_fails_closed(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                  make_store: Any, keys: ApiKeyStore, key: str, f2w: Any,
                                                  entry: Any, sock_dir: Path) -> None:
    store = make_store(0)
    fp = _fill(store, f2w, 128)
    svc = F2Service(gw_pack, entry, sock_dir, evil=True)
    try:
        app = _masked_app(tiny_pack_dir, tmp_path, store, svc, keys)
        async with gateway(app) as c:
            before = store.available_slots(fp, RING)
            r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
            assert r.status_code >= 500 and "integrity" in r.text.lower(), r.text
            st = (await c.get("/admin/status", headers=ADMIN)).json()
            r2 = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
            assert r2.status_code == 503  # breaker open: no further masked traffic
            left = store.available_slots(fp, RING)
    finally:
        svc.close()
    model = next(x for x in st["models"] if x["name"] == MODEL) if "models" in st else st[MODEL]
    assert model["breaker"]["open"] is True and model["breaker"]["trips"] >= 1
    assert left < before  # the failed request's rows stay consumed (never handed out again)


async def test_fused2_gateway_refuses_other_weights(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                    make_store: Any, keys: ApiKeyStore, entry: Any,
                                                    sock_dir: Path) -> None:
    """The service serves the fused2 entry of migration 8; a gateway deriving migration 6 computes another lane
    fingerprint: the ticket is refused (no entry) and the model never serves."""
    store = make_store(0)
    svc = F2Service(gw_pack, entry, sock_dir)
    try:
        app = _masked_app(tiny_pack_dir, tmp_path, store, svc, keys, fused2_migrate_bits=6)
        async with gateway(app) as c:
            st = (await c.get("/admin/status", headers=ADMIN)).json()
            model = next(x for x in st["models"] if x["name"] == MODEL) if "models" in st else st[MODEL]
            # the service has no entry for the gateway's lane fingerprint: never ready, retried in the background
            assert model["ready"] is False and model["state"] == "connecting"
            assert "not served" in str(model["last_error"])
            assert model["fingerprint"] != entry.fingerprint
    finally:
        svc.close()
    assert not svc.calls


def _cfg(**kw: Any) -> GatewayModelConfig:
    base: dict[str, Any] = {"name": "m", "pack_path": Path("/x"), "service_url": SERVICE_URL, "fingerprint": "ab" * 32}
    return GatewayModelConfig(**{**base, **kw})


def test_fused2_config_validation_and_defaults_off() -> None:
    c = _cfg()
    assert c.schedule == "standard" and c.fused2_decode_native == "auto" and c.fused2_migrate_bits == 8
    ok = _cfg(**F2, lane="auto")
    assert ok.schedule == "fused2"
    assert _cfg(**F2, lane="tls").lane == "tls" and _cfg(**F2, lane="required").lane == "required"
    for bad, match in (
        ({"ring": "K32A12C27"}, "K56"), ({"ring": "K56A12"}, "K56"), ({"float_ops": "torch"}, "native"),
        ({"lane": "off"}, "lane"), ({"lane": "stream"}, "lane"), ({"freivalds_rate": 0.5}, "freivalds_rate"),
        ({"head_mode": "provider_int8"}, "head"), ({"spec_lookup": 3}, "speculate"),
        ({"local_linears": "attention"}, "local_linears"), ({"prefix_cache": "conversation"}, "prefix_cache"),
        ({"prefill_chunk_rows": 64}, "prefill_chunk_rows"), ({"mode": "float"}, "masked"),
    ):
        with pytest.raises(ValidationError, match=match):
            _cfg(**{**F2, "lane": "auto", **bad})
    # the plain dev mode needs no lane (it is the plaintext reference)
    assert _cfg(**F2, mode="plain").mode == "plain"


def test_fused2_refuses_the_mask_ring(tmp_path: Path) -> None:
    from pgw.core.errors import ConfigError
    from pgw.gateway import GatewaySettings

    m = _cfg(**F2, lane="auto")
    s = GatewaySettings(models=[m], admin_token="x" * 32, mask_ring={"enabled": True, "capacity_bytes": 1 << 20},
                        maskstore_root=tmp_path)
    with pytest.raises(ConfigError, match="mask_ring"):
        s.validate_policy()
