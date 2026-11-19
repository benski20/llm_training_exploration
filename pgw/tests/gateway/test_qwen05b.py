"""Real model through the whole masked gateway: Qwen2.5-0.5B-Instruct (network, slow).

Packs the checkpoint, fills an encrypted mask store with the real factory, and checks that the
masked chat completion is token- and logprob-identical to the local exact-integer reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING
from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app
from tests.gateway.conftest import ADMIN_TOKEN, SERVICE_URL, InProcessService, auth, fill_store, gateway

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def q05_pack_dir(qwen05b_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from pgw.modelpack import pack_checkpoint

    out = tmp_path_factory.mktemp("gw_q05") / "pack"
    pack_checkpoint(qwen05b_dir, out)
    return out


async def test_qwen05b_masked_equals_plain(q05_pack_dir: Path, tmp_path: Path) -> None:
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(q05_pack_dir)
    store = MaskStore.open(tmp_path / "store", FileKeyProvider(tmp_path / "kek", create=True), fsync=False,
                           low_water_slots=0)
    fill_store(store, pack, DEFAULT_RING, 48)
    svc = InProcessService(pack)
    keys = ApiKeyStore(None)
    _, key = keys.create("app")
    req: dict[str, Any] = {"model": "q05", "messages": [{"role": "user", "content": "What is 2+2?"}],
                           "temperature": 0, "max_tokens": 4, "logprobs": True}
    masked = GatewaySettings(
        models=[GatewayModelConfig(name="q05", pack_path=q05_pack_dir, service_url=SERVICE_URL,
                                   fingerprint=pack.fingerprint)],
        admin_token=ADMIN_TOKEN, default_max_tokens=4,
    )
    async with gateway(create_app(masked, mask_store=store, transports={"q05": svc.transport}, key_store=keys)) as c:
        rm = await c.post("/v1/chat/completions", json=req, headers=auth(key))
    assert rm.status_code == 200, rm.text
    plain = GatewaySettings(
        models=[GatewayModelConfig(name="q05", pack_path=q05_pack_dir, mode="plain")],
        admin_token=ADMIN_TOKEN, allow_dev_modes=True,
    )
    async with gateway(create_app(plain, key_store=keys)) as c:
        rp = await c.post("/v1/chat/completions", json=req, headers=auth(key))
    a, b = rm.json()["choices"][0], rp.json()["choices"][0]
    assert a["message"]["content"] and a["message"] == b["message"]
    assert a["logprobs"] == b["logprobs"]
    store.close()
