"""``pgw.provision.client.weight_source``: one pass over ``W8`` from an untrusted source for the weight-free factory
(``factory_weights="stream"``): the provider's endpoint, a full pack directory or the public checkpoint, every
tensor checked against the pinned manifest; and a check-stock pass over the real endpoint."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import IntegrityError, PGWError
from pgw.modelpack import ModelPack
from pgw.provision.client import provision, weight_source
from tests.provision.test_provision_e2e import _session, _Tamper, service  # noqa: F401 - the module fixture


def _gateway(service: dict[str, Any], tmp: Path) -> Any:  # noqa: F811
    pm = provision(service["fp"], session=_session(service), cache_dir=tmp, hint="none", masked_head=False)
    pm.close()
    return pm.open_pack(verify="full")


def test_provider_source_streams_checked_weights(service: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    gp = _gateway(service, tmp_path / "c")
    src = weight_source(gp, session=_session(service))
    with ModelPack.open(service["pack_dir"]) as full:
        total = 0
        for e in full.manifest.linears:
            w = src.get(e.id)
            assert np.array_equal(w, full.load_linear(e.id).w8.numpy()), e.id
            total += w.nbytes
    assert src.bytes == total and src.network_bytes >= total
    src.close()
    assert not list((tmp_path / "c").rglob("factory-pack*"))  # nothing written


def test_provider_source_rejects_a_tampered_weight_chunk(service: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    gp = _gateway(service, tmp_path / "c")
    key = gp.manifest.linears[3].id

    def flip(path: str, params: dict[str, Any], data: bytes) -> bytes:
        if params.get("id") == f"weight/{key}":
            return bytes([data[0] ^ 1]) + data[1:]
        return data

    src = weight_source(gp, session=_Tamper(_session(service), flip))
    src.get(gp.manifest.linears[0].id)
    with pytest.raises(IntegrityError):
        src.get(key)


def test_masked_head_comes_from_the_gateway_pack(service: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    from pgw.modelpack.masked_head import MaskedHeadPack

    gp = _gateway(service, tmp_path / "c")
    src = weight_source(gp, session=_session(service))
    with ModelPack.open(service["pack_dir"]) as full:
        hp = MaskedHeadPack(full)
        want = hp.load_linear(hp.head_id).w8.numpy()
        got = src.get(hp.head_id.key)
    assert np.array_equal(got, want)


def test_directory_source_checks_every_tensor(service: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    gp = _gateway(service, tmp_path / "c")
    nas = tmp_path / "nas"
    shutil.copytree(service["pack_dir"], nas)
    src = weight_source(gp, full_dir=nas)
    key = gp.manifest.linears[0].id
    good = src.get(key)
    src.close()
    with ModelPack.open(nas) as full:
        e = full.manifest.tensors[full.manifest.linears[0].weight]
        f = nas / e.file
    from pgw.modelpack.st_io import read_safetensors_header

    hdr, _ = read_safetensors_header(f)
    b = bytearray(f.read_bytes())
    b[hdr[full.manifest.linears[0].weight].begin] ^= 0x01
    f.write_bytes(bytes(b))
    src = weight_source(gp, full_dir=nas)
    with pytest.raises(PGWError):
        src.get(key)
    assert good.shape[0] > 0


def test_public_checkpoint_source(service: dict[str, Any], tiny_hf_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    gp = _gateway(service, tmp_path / "c")
    work = tmp_path / "work"
    src = weight_source(gp, hf_repo=str(tiny_hf_dir), work_dir=work)
    with ModelPack.open(service["pack_dir"]) as full:
        key = full.manifest.linears[1].id
        assert np.array_equal(src.get(key), full.load_linear(key).w8.numpy())
    src.close()
    assert not (work / "full").exists()  # the locally packed checkpoint is deleted after the pass


def test_check_stock_pass_over_the_endpoint(service: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    """A real pass (provider endpoint, pinned manifest) gives the uW an on-demand draw computes from the full pack."""
    from pgw.core.freivalds import prime_check_vectors
    from pgw.core.ring import RingParams
    from pgw.factory.checkstock import CheckStock, StockLinear
    from pgw.factory.kernels import prime_vecmat
    from pgw.maskstore import FileKeyProvider

    gp = _gateway(service, tmp_path / "c")
    lins = [StockLinear(e.id, e.out_features, e.in_features, 14) for e in gp.manifest.linears]
    stock = CheckStock(tmp_path / "stock", FileKeyProvider(tmp_path / "kek", create=True), fingerprint=gp.fingerprint,
                       ring=RingParams.from_name("K32A12C27"), check_field="p31", checks=1, linears=lins,
                       source=lambda: weight_source(gp, session=_session(service)), target_segments=3,
                       low_water_segments=1, target_private=2, low_water_private=1, background=False)
    seed = os.urandom(32)
    out = stock.refill(seeds=[seed, os.urandom(32)])
    assert out["weights_bytes"] == sum(e.out_features * e.in_features for e in gp.manifest.linears)
    item = stock.take_segment()
    assert bytes(item.seed) == seed
    with ModelPack.open(service["pack_dir"]) as full:
        for e in full.manifest.linears:
            u = prime_check_vectors(seed, 1, e.out_features, e.id)
            want = prime_vecmat(u, full.load_linear(e.id).w8)
            assert torch.equal(item.uw_tensor(e.id), want), e.id
