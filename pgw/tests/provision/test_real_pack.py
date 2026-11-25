"""Opt-in: provisioning a real pack end to end on CPU (``PGW_PROVISION_REAL_PACK=<pack dir>``, e.g. Qwen2.5-0.5B).

A compute service on loopback serves the pack with ``provision`` on; the client provisions artifacts and a
provider-built, client-verified hint (masked head included), then the hint files of a few linears are compared with
a local build for the same seeds. Prints the measured sizes and times (``-s``)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pgw.hemask.pks_backend import PKSHintStore
from pgw.modelpack import ModelPack
from pgw.modelpack.masked_head import MaskedHeadPack
from pgw.provision.client import provision
from pgw.service.auth import hash_token
from pgw.service.config import ServiceSettings
from tests.provision.test_provision_e2e import TOKEN, _free_port, _Server, _session

REAL = os.environ.get("PGW_PROVISION_REAL_PACK")
pytestmark = pytest.mark.skipif(not REAL, reason="set PGW_PROVISION_REAL_PACK=<pack dir> (minutes of CPU)")


def test_real_pack_provider_hint(tmp_path: Path) -> None:
    pack_dir = Path(str(REAL))
    tok = tmp_path / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN)}\n")
    port = _free_port()
    s = ServiceSettings(host="127.0.0.1", port=port, metrics_port=None, packs=[pack_dir], device="cpu",
                        tokens_file=tok, batch_window_ms=1.0, rings=["K32A12C27"], drain_timeout_s=2.0,
                        provision=True, masked_head="also", warmup=False)
    srv = _Server(s)
    url = f"http://127.0.0.1:{port}"
    srv.start(url)
    try:
        full = ModelPack.open(pack_dir, components=("linears", "plain", "tokenizer"))
        pm = provision(full.name, session=_session({"url": url}), cache_dir=tmp_path / "cache", hint="provider",
                       masked_head=True)
        try:
            assert pm.hint_source == "provider-verified"
            st = pm.hint_store
            hp = MaskedHeadPack(full)
            keys = {"0.qkv", f"{full.arch.num_layers - 1}.down", hp.head_id.key}
            ref = PKSHintStore.build(tmp_path / "ref", hp, only=keys, seeds=(st.a_seed, st.ks_seed))
            for k in keys:
                a = hashlib.sha256((Path(st.path) / f"{k}.npy").read_bytes()).hexdigest()
                assert a == ref.meta["digests"][k] == st.meta["digests"][k]
                assert st.meta["row_norms"][k] == ref.meta["row_norms"][k]
            print(json.dumps({"sizes_GB": {k: round(v / 1e9, 3) for k, v in pm.sizes.items()},
                              "timings_s": {k: round(v, 1) for k, v in pm.timings.items()}}, indent=1))
            gp = pm.open_pack()
            assert gp.fingerprint == full.fingerprint
        finally:
            pm.close()
    finally:
        srv.stop()
