"""Chunked prefill through the gateway (``prefill_chunk_rows``): a multi-turn chat (with and without conversation
prefix reuse, plain and strict speculation) gives the bits of unchunked prompt passes, the provider sees the same
rows per linear in total (prompt passes in items of at most the chunk), and the masks consumed are unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pgw.core import DEFAULT_RING
from pgw.engine.model import native_float_available
from pgw.gateway import ApiKeyStore, GatewayModelConfig
from tests.gateway.conftest import MODEL, InProcessService, gateway
from tests.gateway.test_prefix_cache import _app, _converse

pytestmark = pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")

BASE = {"local_linears": "off", "float_ops": "native-blocked"}


def _slots(store: Any, fp: str) -> int:
    return sum(i.slots_available for i in store.inventory(fp, DEFAULT_RING))


@pytest.mark.parametrize("variant", ["plain", "reuse", "strict"])
async def test_chunked_prompt_passes_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                   make_store: Any, variant: str) -> None:
    keys = ApiKeyStore(None)
    key = keys.create("app")[1]
    extra: dict[str, Any] = {}
    if variant == "reuse":
        extra = {"float_ops": "native", "prefix_cache": "conversation"}
    if variant == "strict":
        extra = {"draft_pack_path": str(tiny_pack_dir), "spec_rows": 3, "spec_strict_passes": True}
    chunk = 7
    runs = []
    for name, kw in (("ref", {}), ("chunked", {"prefill_chunk_rows": chunk})):
        svc = InProcessService(gw_pack)
        store = make_store(1500, name=f"{variant}-{name}")
        before = _slots(store, gw_pack.fingerprint)
        app = _app(tiny_pack_dir, tmp_path, store, svc, keys, {**BASE, **extra, **kw})
        async with gateway(app) as c:
            rt = app.state.pgw.runtime.get(MODEL)
            assert rt.model.prefill_chunk_rows == kw.get("prefill_chunk_rows", 0)
            out = await _converse(c, key, 3)
            await rt.generator.wait_drained()
        runs.append((out, svc, before - _slots(store, gw_pack.fingerprint)))
    (ref, svc0, used0), (got, svc, used) = runs
    for a, b in zip(got, ref, strict=True):
        assert a["choices"][0]["message"] == b["choices"][0]["message"]
        assert a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"]
        assert a["usage"] == b["usage"]
    assert used == used0
    tot: dict[str, list[int]] = {}
    for s in (svc0, svc):
        per: dict[str, int] = {}
        for lid, x in s.received:
            per[lid] = per.get(lid, 0) + int(x.shape[0])
        tot.setdefault("rows", []).append(sum(per.values()))
    assert tot["rows"][0] == tot["rows"][1]
    assert max(int(x.shape[0]) for _, x in svc.received) <= chunk
    assert max(int(x.shape[0]) for _, x in svc0.received) > chunk


def test_prefill_chunk_config_validation() -> None:
    def cfg(**kw: Any) -> GatewayModelConfig:
        return GatewayModelConfig(name="m", pack_path=Path("/x"), service_url="https://compute.example.net:8443",
                                  fingerprint="ab" * 32, **kw)

    assert cfg().prefill_chunk_rows is None
    with pytest.raises(ValidationError, match="native float_ops"):
        cfg(prefill_chunk_rows=1024)
    assert cfg(prefill_chunk_rows=1024, float_ops="native-blocked-bf16kv").prefill_chunk_rows == 1024
    assert cfg(prefill_chunk_rows=0).prefill_chunk_rows == 0
