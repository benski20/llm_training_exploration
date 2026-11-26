"""Provisioning a pack-format-2 model (bfloat16 embeddings / untied head): the provider serves the bfloat16 bytes,
the client's gateway pack holds them as they are (half the plain bytes), verifies them in full against the pinned
fingerprint, and loads float32 values bit-identical to a format-1 pack of the same checkpoint. The public-source
path re-derives the same format-2 pin and hard-links the plain file into the factory pack."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core.errors import FingerprintMismatchError, PGWError
from pgw.modelpack import ModelPack, pack_checkpoint
from pgw.modelpack.masked_head import MaskedHeadPack
from pgw.provision.client import FACTORY_PACK_DIR, provision
from pgw.provision.pack import GatewayPack
from pgw.provision.session import ProvisionSession
from pgw.service.auth import hash_token
from pgw.service.config import ServiceSettings
from tests.modelpack.test_source_dtype import make_bf16_ckpt
from tests.provision.test_provision_e2e import TOKEN, _free_port, _Server


@pytest.fixture(scope="module")
def bf16(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    d = tmp_path_factory.mktemp("prov_bf16")
    ckpt = make_bf16_ckpt(tiny_hf_dir, d / "ckpt")
    pack_checkpoint(ckpt, d / "v1", name="tiny-bf16", plain_dtype="float32")
    pack_checkpoint(ckpt, d / "v2", name="tiny-bf16")
    tok = d / "tenants.txt"
    tok.write_text(f"tenant-a:{hash_token(TOKEN)}\n")
    port = _free_port()
    s = ServiceSettings(host="127.0.0.1", port=port, metrics_port=None, packs=[d / "v2"], device="cpu",
                        tokens_file=tok, batch_window_ms=1.0, rings=["K32A12C27"], drain_timeout_s=2.0,
                        provision=True, provision_chunk_mb=1)
    srv = _Server(s)
    url = f"http://127.0.0.1:{port}"
    srv.start(url)
    try:
        yield {"url": url, "ckpt": ckpt, "v1": d / "v1", "v2": d / "v2",
               "fp": ModelPack.open(d / "v2").fingerprint, "fp1": ModelPack.open(d / "v1").fingerprint}
    finally:
        srv.stop()


def _session(b: dict[str, Any]) -> ProvisionSession:
    return ProvisionSession(b["url"], token=TOKEN, retries=0)


def _same_values(gp: Any, ref: Any) -> None:
    for n in ref.manifest.plain_tensor_names():
        a, b = gp.load_plain(n), ref.load_plain(n)
        assert a.dtype == b.dtype == torch.float32 and a.view(torch.int32).equal(b.view(torch.int32)), n
    hg, hr = MaskedHeadPack(gp, float_head=False), MaskedHeadPack(ref)
    mg, mr = hg.load_linear_meta(hg.head_id), hr.load_linear_meta(hr.head_id)
    assert mg.scale.equal(mr.scale) and mg.row_norm_sq_max == mr.row_norm_sq_max


def test_provider_path_bf16_gateway_pack(bf16: dict[str, Any], tmp_path: Path) -> None:
    pm = provision(bf16["fp"], session=_session(bf16), cache_dir=tmp_path, hint="none")
    gp = pm.open_pack(verify="full")
    assert gp.manifest.version == 2
    assert gp.manifest.tensors["embed_tokens"].dtype == gp.manifest.tensors["lm_head"].dtype == "bfloat16"
    with ModelPack.open(bf16["v1"], verify="full") as v1:
        _same_values(gp, v1)  # widened at load: bit-identical to the float32 (format-1) pack
        big = sum(v1.manifest.tensors[n].nbytes for n in ("embed_tokens", "lm_head"))
        assert v1.size_bytes("plain") - gp.size_bytes("plain") == big // 2
    gp.close()
    # a format-2 pack is not the format-1 pin, and vice versa
    with pytest.raises(FingerprintMismatchError):
        provision("tiny-bf16", session=_session(bf16), cache_dir=tmp_path / "x", expected_fingerprint=bf16["fp1"],
                  hint="none")
    # one flipped bit of a bfloat16 head value in the installed pack: rejected by the full verification
    from pgw.modelpack.st_io import read_safetensors_header

    f = pm.pack_dir / gp.manifest.tensors["lm_head"].file
    entries, _ = read_safetensors_header(f)
    raw = bytearray(f.read_bytes())
    raw[entries["lm_head"].begin + 3] ^= 0x40
    f.write_bytes(bytes(raw))
    with pytest.raises(PGWError):
        GatewayPack.open_gateway(pm.pack_dir, expected_fingerprint=pm.fingerprint, verify="full")
    pm.close()


def test_public_source_rederives_format2_pin_and_links(bf16: dict[str, Any], tmp_path: Path) -> None:
    man = ModelPack.open(bf16["v2"]).manifest
    # with the provider's manifest as packing parameters ("source"), and with none at all (the default is format 2)
    pm = provision(bf16["fp"], session=_session(bf16), cache_dir=tmp_path / "a", hint="none", masked_head=False,
                   source="hf", hf_repo=str(bf16["ckpt"]))
    assert pm.fingerprint == bf16["fp"]
    pm.close()
    pm = provision(bf16["fp"], session=None, cache_dir=tmp_path / "b", hint="none", masked_head=False,
                   source="hf", hf_repo=str(bf16["ckpt"]), hf_name=man.name, factory_pack=True)
    assert pm.fingerprint == bf16["fp"]
    for f in set(man.plain_files()) - set(man.linear_files()):
        assert os.path.samefile(pm.pack_dir / f, pm.factory_pack_dir / f)  # type: ignore[operator]
    assert pm.factory_pack_dir == pm.pack_dir.parent / FACTORY_PACK_DIR
    gp = pm.open_pack(verify="full")
    with ModelPack.open(bf16["v1"]) as v1:
        _same_values(gp, v1)
    gp.close()
    ModelPack.open(pm.factory_pack_dir, verify="full", expected_fingerprint=bf16["fp"]).close()  # type: ignore[arg-type]
    pm.close()
    # the format-1 parameters still reproduce the format-1 pin from the same checkpoint
    pm1 = provision(bf16["fp1"], session=None, cache_dir=tmp_path / "c", hint="none", masked_head=False,
                    source="hf", hf_repo=str(bf16["ckpt"]), hf_name=man.name, hf_plain_dtype="float32")
    assert pm1.fingerprint == bf16["fp1"] and pm1.manifest.version == 1
    pm1.close()
    # but the default (format 2) does not land on a format-1 pin
    with pytest.raises(FingerprintMismatchError):
        provision(bf16["fp1"], session=None, cache_dir=tmp_path / "d", hint="none", masked_head=False,
                  source="hf", hf_repo=str(bf16["ckpt"]), hf_name=man.name)
