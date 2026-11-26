"""Pack format 2 (bfloat16 embeddings / LM head, widened at load) through the whole gateway: the masked path (real
encrypted mask store, in-process compute service), the plaintext reference and the int8 masked-head reference give
bit-identical tokens and logprobs from a format-1 (float32) and a format-2 pack of the same bfloat16 checkpoint.
Tiny model always; Qwen2.5-0.5B-Instruct when available (network/HF cache, slow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING
from pgw.gateway import ApiKeyStore, GatewayModelConfig, GatewaySettings, create_app
from pgw.modelpack import ModelPack, pack_checkpoint
from tests.gateway.conftest import (
    ADMIN_TOKEN,
    MODEL,
    SERVICE_URL,
    InProcessService,
    auth,
    fill_store,
    gateway,
    masked_settings,
    plain_settings,
)
from tests.modelpack.test_source_dtype import make_bf16_ckpt

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 8, "ignore_eos": True, "logprobs": True, "top_logprobs": 5}


def _store(tmp: Path, pack: Any, slots: int) -> Any:
    from pgw.maskstore import FileKeyProvider, MaskStore

    store = MaskStore.open(tmp / "store", FileKeyProvider(tmp / "kek", create=True), fsync=False, low_water_slots=0)
    fill_store(store, pack, DEFAULT_RING, slots)
    return store


async def _post(app: Any, key: str, req: dict[str, Any]) -> dict[str, Any]:
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=req, headers=auth(key))
    assert r.status_code == 200, r.text
    return dict(r.json()["choices"][0])


@pytest.fixture(scope="module")
def tiny_packs(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    d = tmp_path_factory.mktemp("gw_fmt")
    ckpt = make_bf16_ckpt(tiny_hf_dir, d / "ckpt")
    pack_checkpoint(ckpt, d / "v1", name="tiny-bf16", plain_dtype="float32")
    pack_checkpoint(ckpt, d / "v2", name="tiny-bf16")
    return {1: d / "v1", 2: d / "v2"}


async def test_tiny_masked_and_plain_bit_identical_across_formats(tiny_packs: dict[int, Path],
                                                                  tmp_path: Path) -> None:
    keys = ApiKeyStore(None)
    _, key = keys.create("app")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for v, pdir in tiny_packs.items():
        pack = ModelPack.open(pdir, verify="full")
        assert pack.manifest.version == v
        assert pack.manifest.tensors["lm_head"].dtype == ("float32" if v == 1 else "bfloat16")
        work = tmp_path / f"v{v}"
        work.mkdir()
        store = _store(work, pack, 160)
        svc = InProcessService(pack)
        masked = create_app(masked_settings(pdir, work), mask_store=store, transports={MODEL: svc.transport},
                            key_store=keys)
        out[(v, "masked")] = await _post(masked, key, CHAT)
        out[(v, "plain")] = await _post(create_app(plain_settings(pdir, work), key_store=keys), key, CHAT)
        out[(v, "plain_int8_head")] = await _post(
            create_app(plain_settings(pdir, work, model_kw={"head_mode": "provider_int8"}), key_store=keys), key, CHAT)
        store.close()
        assert out[(v, "masked")]["message"] == out[(v, "plain")]["message"]
        assert out[(v, "masked")]["logprobs"] == out[(v, "plain")]["logprobs"]
    for kind in ("masked", "plain", "plain_int8_head"):
        a, b = out[(1, kind)], out[(2, kind)]
        assert a["message"]["content"] and a["message"] == b["message"], kind
        assert a["logprobs"] == b["logprobs"], kind  # exact float equality, top-5 alternatives included


@pytest.mark.network
@pytest.mark.slow
async def test_qwen05b_masked_bit_identical_across_formats(qwen05b_dir: Path, tmp_path: Path) -> None:
    keys = ApiKeyStore(None)
    _, key = keys.create("app")
    req: dict[str, Any] = {"model": "q05", "messages": [{"role": "user", "content": "What is 2+2?"}],
                           "temperature": 0, "max_tokens": 4, "logprobs": True, "top_logprobs": 3}
    got: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for v, plain_dtype in ((1, "float32"), (2, "source")):
        pdir = tmp_path / f"p{v}"
        pack_checkpoint(qwen05b_dir, pdir, plain_dtype=plain_dtype)  # type: ignore[arg-type]
        pack = ModelPack.open(pdir, verify="full")
        assert pack.manifest.version == v
        assert pack.manifest.tensors["embed_tokens"].dtype == ("float32" if v == 1 else "bfloat16")
        work = tmp_path / f"w{v}"
        work.mkdir()
        store = _store(work, pack, 48)
        svc = InProcessService(pack)
        masked = GatewaySettings(
            models=[GatewayModelConfig(name="q05", pack_path=pdir, service_url=SERVICE_URL,
                                       fingerprint=pack.fingerprint)],
            admin_token=ADMIN_TOKEN, default_max_tokens=4)
        rm = await _post(create_app(masked, mask_store=store, transports={"q05": svc.transport}, key_store=keys),
                         key, req)
        plain = GatewaySettings(models=[GatewayModelConfig(name="q05", pack_path=pdir, mode="plain")],
                                admin_token=ADMIN_TOKEN, allow_dev_modes=True)
        rp = await _post(create_app(plain, key_store=keys), key, req)
        store.close()
        pack.close()
        assert rm["message"]["content"] and rm["message"] == rp["message"] and rm["logprobs"] == rp["logprobs"]
        got[v] = (rm, rp)
        if v == 1:
            import shutil

            shutil.rmtree(pdir)  # one 0.5B pack on disk at a time
    for i in range(2):
        assert got[1][i]["message"] == got[2][i]["message"]
        assert got[1][i]["logprobs"] == got[2][i]["logprobs"]
