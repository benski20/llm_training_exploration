"""W1: ``lane = "stream"`` (the blocking stream lane) wired into the gateway: over a real socket the answers
stay bit-identical to the plaintext quantized reference, decode trips take the lane (no fallback), and the
service receives only regular ``ComputeRequest`` messages, each once."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pgw.gateway import ApiKeyStore, create_app
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings
from tests.gateway.test_speed_wiring import CHAT, _masked, key, keys, reference  # noqa: F401 - fixtures


class _SocketService:
    """An ``InProcessService`` app on a real loopback socket (uvicorn in a thread)."""

    def __init__(self, app: Any):
        import uvicorn

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = int(s.getsockname()[1])
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, ws="websockets-sansio",
                                                    log_config=None, lifespan="off", ws_ping_interval=None,
                                                    ws_per_message_deflate=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(250):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)


@pytest.mark.parametrize("local", ["off", "attention"])
async def test_lane_stream_over_a_real_socket_is_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                               make_store: Any, keys: ApiKeyStore,  # noqa: F811
                                                               key: str, reference: dict[str, Any],  # noqa: F811
                                                               local: str) -> None:
    from pgw.wire.codec import encode_message

    svc = InProcessService(gw_pack)
    sent: list[bytes] = []
    real = svc.handler.compute

    async def record(req: Any, *a: Any, **k: Any) -> Any:
        sent.append(encode_message(req))
        return await real(req, *a, **k)

    svc.handler.compute = record  # type: ignore[method-assign]
    sock = _SocketService(svc.app)
    try:
        app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={
            "lane": "stream", "local_linears": local, "float_threads": 1, "service_url": sock.url}),
            mask_store=make_store(200), key_store=keys)
        async with gateway(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
            assert r.status_code == 200, r.text
            rt = app.state.pgw.runtime.get(MODEL)
            await rt.generator.wait_drained()
            status = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
            pending = rt.backend.pending_checks
    finally:
        sock.close()
    out = r.json()["choices"][0]
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert sp["lane"] == "stream" and sp["lane_state"] == "stream" and sp["sync_decode"] and sp["trip_engine"]
    st = sp["stream_lane_stats"]
    assert st["calls"] > 0 and st["fallbacks"] == 0 and st["opens"] == 1
    assert status["backend"]["integrity_failures"] == 0 and status["backend"]["freivalds_checked"] > 0
    assert pending == 0
    # the service received regular msgpack ComputeRequests only, each exactly once
    assert len(sent) == len(set(sent)) and len(sent) >= st["calls"]


async def test_lane_stream_with_an_in_process_service_uses_the_regular_path(
        tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any, keys: ApiKeyStore,  # noqa: F811
        key: str, reference: dict[str, Any]) -> None:  # noqa: F811
    svc = InProcessService(gw_pack)
    out, status, _rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key,
                                     {"lane": "stream"})
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    st = status["speed"]["stream_lane_stats"]
    assert st["calls"] == 0 and st["opens"] == 0 and st["open_failures"] == 0
