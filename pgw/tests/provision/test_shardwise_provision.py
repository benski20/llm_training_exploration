"""The public-checkpoint path (``source="hf"``, ``weight_source(hf_repo=...)``) packs shard by shard: one shard of
the checkpoint on disk at a time, never the whole checkpoint, the pinned fingerprint re-derived, verified in full,
and a checkpoint that does not reproduce the pin refused with nothing installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from pgw.core.errors import FingerprintMismatchError
from pgw.modelpack import ModelPack, pack_checkpoint
from pgw.provision.client import FACTORY_PACK_DIR, provision, weight_source
from pgw.provision.pack import GatewayPack
from tests.modelpack.test_shardwise import hf_like, install_fake_hub, reshard
from tests.modelpack.test_source_dtype import make_bf16_ckpt

REPO = "org/tiny-bf16"
COMMIT = "cd" * 20


@pytest.fixture()
def hub(tiny_hf_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    bf = make_bf16_ckpt(tiny_hf_dir, tmp_path / "bf", untie=True)
    d = reshard(bf, tmp_path / "hubroot" / COMMIT, hf_like(bf, 3))
    log = install_fake_hub(monkeypatch, d, REPO, COMMIT)
    # the pin: pack_checkpoint of the hub repo at the commit (the whole snapshot, as the pin script did)
    man = pack_checkpoint(REPO, tmp_path / "pin", name="tiny-bf16", revision=COMMIT)
    log["downloads"].clear()
    return {"dir": d, "log": log, "fp": man.fingerprint, "pin": tmp_path / "pin"}


def test_public_check_streams_one_shard_at_a_time(hub: dict[str, object], tmp_path: Path) -> None:
    log = hub["log"]
    pm = provision(str(hub["fp"]), session=None, cache_dir=tmp_path / "c", hint="none", masked_head=False,
                   source="hf", hf_repo=REPO, hf_revision=COMMIT, hf_name="tiny-bf16", factory_pack=True)
    assert pm.fingerprint == hub["fp"]
    assert log["downloads"] == sorted(set(p.name for p in Path(hub["dir"]).glob("model-*.safetensors")))  # type: ignore[arg-type]
    assert max(log["present"]) == 1  # each shard deleted before the next was fetched
    mdir = pm.pack_dir.parent
    assert not (mdir / "hf-stream").exists() and not list(mdir.rglob("model-*.safetensors"))
    ModelPack.open(pm.factory_pack_dir, verify="full", expected_fingerprint=pm.fingerprint).close()  # type: ignore[arg-type]
    assert pm.factory_pack_dir == mdir / FACTORY_PACK_DIR
    pm.close()


def test_weight_source_hf_streams_and_deletes(hub: dict[str, object], tmp_path: Path) -> None:
    pm = provision(str(hub["fp"]), session=None, cache_dir=tmp_path / "c", hint="none", masked_head=False,
                   source="hf", hf_repo=REPO, hf_revision=COMMIT, hf_name="tiny-bf16")
    gp = GatewayPack.open_gateway(pm.pack_dir, expected_fingerprint=pm.fingerprint, verify="full")
    hub["log"]["downloads"].clear()  # type: ignore[index]
    work = tmp_path / "pass"
    src = weight_source(gp, hf_repo=REPO, hf_revision=COMMIT, work_dir=work)
    assert max(hub["log"]["present"]) == 1  # type: ignore[index]
    with ModelPack.open(hub["pin"]) as ref:  # type: ignore[arg-type]
        for e in ref.manifest.linears:
            assert np.array_equal(src.get(e.id), ref.load_linear(e.id).w8.numpy())
    src.close()
    assert not (work / "full").exists() and not list(work.rglob("*.safetensors"))
    gp.close()
    pm.close()


def test_a_changed_checkpoint_is_refused(hub: dict[str, object], tmp_path: Path) -> None:
    d = Path(hub["dir"])  # type: ignore[arg-type]
    fn = "model-00002-of-00003.safetensors"
    sd = load_file(d / fn)
    k = next(n for n in sd if n.endswith("mlp.down_proj.weight"))
    sd[k] = sd[k].clone()
    sd[k][0, 0] = sd[k][0, 0] * 2 + torch.tensor(0.01, dtype=sd[k].dtype)
    save_file(sd, d / fn, metadata={"format": "pt"})
    with pytest.raises(FingerprintMismatchError):
        provision(str(hub["fp"]), session=None, cache_dir=tmp_path / "c", hint="none", masked_head=False,
                  source="hf", hf_repo=REPO, hf_revision=COMMIT, hf_name="tiny-bf16")
    assert not list((tmp_path / "c").rglob("manifest.json"))
