"""Shard-wise streaming packer (:mod:`pgw.modelpack.shardwise`): byte-identical to ``pack_checkpoint``.

Exactness: every pack file (linear shards, ``plain.safetensors``, tokenizer) is byte-identical to the whole-checkpoint
packer's, and the manifest equals it except ``created_at``; the hash-only mode gives the same manifest. Checked on
the tiny Qwen2 (tied head, q/k/v biases), a bfloat16 untied variant, the tiny Llama, and the real Qwen2.5-0.5B /
1.5B checkpoints from the local Hugging Face cache (the release pins and the format-1 pins). Checkpoints are
resharded so that linear groups and plain tensors straddle shard boundaries (spill files), and the shard source
deletes each shard before the next is fetched. Interrupted runs resume; an edited work dir cannot yield a pack that
verifies.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pgw.core.errors import ConfigError, FingerprintMismatchError, IntegrityError
from pgw.modelpack import Manifest, ModelPack, pack_checkpoint
from pgw.modelpack.masked_head import MaskedHeadPack
from pgw.modelpack.shardwise import HubShards, LocalShards, VirtualShards, stream_pack_checkpoint

from .test_source_dtype import make_bf16_ckpt

PIN_05B = "d773a502fb976d45070332cbd4222ee30bc971d14c7f9c56a317805c230c0b44"
PIN_15B = "3d198de49e61c32b43150c237c0fad0f1ec062f0a355df75d64706bbac8d70d9"
PIN1_05B = "915e55c19364542c4fd8ab5c8f7154806e2511048755e0d10426374b5c8fe9c7"
PIN1_15B = "81bb7adcea1524171297c0004c0efb1edad9f71dd0535462c9b32ab03ac99348"


# ---- helpers ---------------------------------------------------------------------------------
def _file_digests(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _manifest_body(m: Manifest) -> dict[str, Any]:
    d = json.loads(m.to_json())
    d.pop("created_at", None)
    return d


def assert_same_pack(ref: Path, new: Path) -> None:
    assert _file_digests(ref) == _file_digests(new)
    assert _manifest_body(Manifest.from_file(ref / "manifest.json")) == _manifest_body(
        Manifest.from_file(new / "manifest.json"))
    # manifest.json bytes equal once created_at is equalized
    a = (ref / "manifest.json").read_text()
    b = (new / "manifest.json").read_text()
    ca, cb = (json.loads(x)["created_at"] for x in (a, b))
    assert a == b.replace(cb, ca)


def round_robin(ckpt: Path, k: int) -> dict[str, str]:
    """Every tensor to virtual shard ``i % k`` (sorted names): nearly every group straddles shards."""
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        names = sorted(f.keys())
    return {n: f"model-{i % k + 1:05d}-of-{k:05d}.safetensors" for i, n in enumerate(names)}


def hf_like(ckpt: Path, k: int) -> dict[str, str]:
    """Like ``save_pretrained`` sharding: names in layer order cut into ``k`` contiguous runs by size."""
    import re

    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        names = list(f.keys())
        sizes = {n: f.get_slice(n).get_shape() for n in names}

    def order(n: str) -> tuple[int, str]:
        m = re.search(r"layers\.(\d+)\.", n)
        return (int(m.group(1)) + 1 if m else (0 if "embed" in n else 10**6), n)

    names.sort(key=order)
    nb = [int(torch.tensor(sizes[n]).prod()) for n in names]
    per = sum(nb) / k
    out, acc, s = {}, 0.0, 0
    for n, b in zip(names, nb, strict=True):
        if acc > per * (s + 1) and s < k - 1:
            s += 1
        out[n] = f"model-{s + 1:05d}-of-{k:05d}.safetensors"
        acc += b
    return out


def reshard(ckpt: Path, out: Path, assignment: dict[str, str]) -> Path:
    """A real sharded copy of ``ckpt`` (files + ``model.safetensors.index.json``)."""
    shutil.copytree(ckpt, out, ignore=shutil.ignore_patterns("model.safetensors"))
    sd = load_file(ckpt / "model.safetensors")
    for fn in sorted(set(assignment.values())):
        save_file({k: v for k, v in sd.items() if assignment[k] == fn}, out / fn, metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": assignment}))
    return out


class CountingShards(LocalShards):
    """Copies each shard into ``copy_to`` on fetch (a download) and records how many were present at once."""

    def __init__(self, ckpt: Path, copy_to: Path):
        super().__init__(ckpt, copy_to=copy_to)
        self.fetched: list[str] = []
        self.max_present = 0

    def fetch(self, file: str) -> Path:
        p = super().fetch(file)
        self.fetched.append(file)
        assert self.copy_to is not None
        self.max_present = max(self.max_present, len(list(self.copy_to.glob("*.safetensors"))))
        return p


# ---- tiny models -----------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 7])
def test_tiny_virtual_shards_byte_identical(tiny_hf_dir: Path, tmp_path: Path, k: int) -> None:
    ref = tmp_path / "ref"
    pack_checkpoint(tiny_hf_dir, ref, name="tiny", shard_bytes=40_000)
    src = VirtualShards(tiny_hf_dir, round_robin(tiny_hf_dir, k))
    r = stream_pack_checkpoint(src, tmp_path / "new", work_dir=tmp_path / "w", name="tiny", shard_bytes=40_000)
    assert len(Manifest.from_file(ref / "manifest.json").linear_files()) > 1
    assert_same_pack(ref, tmp_path / "new")
    assert r.stats["shards"] == k
    if k > 1:
        assert r.stats["spill_peak_bytes"] > 0  # groups straddled shards
    assert not (tmp_path / "w").exists()  # work dir cleaned up
    # hash only: same manifest, nothing written
    h = stream_pack_checkpoint(src, None, work_dir=tmp_path / "w2", name="tiny", shard_bytes=40_000)
    assert h.out_dir is None and h.fingerprint == r.fingerprint
    assert _manifest_body(h.manifest) == _manifest_body(r.manifest)


@pytest.mark.parametrize("untie", [True, False])
@pytest.mark.parametrize("plain_dtype", ["source", "float32"])
def test_bf16_resharded_download_delete(tiny_hf_dir: Path, tmp_path: Path, untie: bool, plain_dtype: str) -> None:
    """bfloat16 checkpoint (as the public ones), tied or untied head, format 2 or 1, as real shard files fetched
    one by one (copied = downloaded) and deleted after use."""
    bf = make_bf16_ckpt(tiny_hf_dir, tmp_path / "bf", untie=untie)
    sh = reshard(bf, tmp_path / "sharded", hf_like(bf, 4))
    ref = tmp_path / "ref"
    pack_checkpoint(bf, ref, name="tiny-bf16", plain_dtype=plain_dtype)  # type: ignore[arg-type]
    src = CountingShards(sh, tmp_path / "w" / "shards")
    stream_pack_checkpoint(src, tmp_path / "new", work_dir=tmp_path / "w", name="tiny-bf16",
                           plain_dtype=plain_dtype)  # type: ignore[arg-type]
    m_ref, m_new = (Manifest.from_file(p / "manifest.json") for p in (ref, tmp_path / "new"))
    # the source dir name differs (bf vs sharded): everything else is the same
    assert m_ref.source.repo == "bf" and m_new.source.repo == "sharded"
    assert _file_digests(ref) == _file_digests(tmp_path / "new")
    a, b = _manifest_body(m_ref), _manifest_body(m_new)
    a.pop("source"), b.pop("source"), a.pop("fingerprint"), b.pop("fingerprint")
    assert a == b
    assert src.fetched == sorted(set(hf_like(bf, 4).values())) and src.max_present == 1
    with ModelPack.open(tmp_path / "new", verify="full") as p:
        assert p.manifest.arch.tie_word_embeddings == (not untie)
        assert ("lm_head" in p.manifest.tensors) == untie


def test_masked_head_identical(tiny_hf_dir: Path, tmp_path: Path) -> None:
    """The masked head (derived from the plain head) is the same from both packers, tied and untied."""
    for untie in (True, False):
        bf = make_bf16_ckpt(tiny_hf_dir, tmp_path / f"bf{untie}", untie=untie)
        pack_checkpoint(bf, tmp_path / f"ref{untie}", name="t")
        stream_pack_checkpoint(VirtualShards(bf, round_robin(bf, 3)), tmp_path / f"new{untie}",
                               work_dir=tmp_path / f"w{untie}", name="t")
        with ModelPack.open(tmp_path / f"ref{untie}", verify="full") as a, \
                ModelPack.open(tmp_path / f"new{untie}", verify="full") as b:
            ha, hb = MaskedHeadPack(a), MaskedHeadPack(b)
            assert ha.fingerprint == hb.fingerprint
            la, lb = ha.load_linear(ha.head_id), hb.load_linear(hb.head_id)
            assert la.w8.equal(lb.w8) and la.scale.view(torch.int32).equal(lb.scale.view(torch.int32))


def test_llama_byte_identical(tiny_llama_hf_dir: Path, tmp_path: Path) -> None:
    pack_checkpoint(tiny_llama_hf_dir, tmp_path / "ref", name="tl")
    stream_pack_checkpoint(VirtualShards(tiny_llama_hf_dir, round_robin(tiny_llama_hf_dir, 4)), tmp_path / "new",
                           work_dir=tmp_path / "w", name="tl")
    assert_same_pack(tmp_path / "ref", tmp_path / "new")


def test_qwen3_byte_identical(tmp_path: Path) -> None:
    """Qwen3 (per-head q/k norms as plain tensors): the shard-wise packer writes the same pack as pack_checkpoint."""
    from tests.engine.test_qwen3 import _save_qwen3

    d = _save_qwen3(tmp_path / "q3", seed=13)
    pack_checkpoint(d, tmp_path / "ref", name="tq3")
    stream_pack_checkpoint(VirtualShards(d, round_robin(d, 3)), tmp_path / "new", work_dir=tmp_path / "w", name="tq3")
    assert_same_pack(tmp_path / "ref", tmp_path / "new")
    with ModelPack.open(tmp_path / "new", verify="full") as p:
        assert "layers.0.q_norm" in p.manifest.tensors


def test_quantize_blocks_equal_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row blocks (here 3 rows) give the whole-matrix quantization bit for bit, across part boundaries."""
    from pgw.core.ring import quantize_weight_rows
    from pgw.modelpack import shardwise

    g = torch.Generator().manual_seed(3)
    parts = [(torch.randn(r, 37, generator=g) * s).to(torch.bfloat16) for r, s in ((10, 1.0), (4, 1e-3), (7, 50.0))]
    parts[1][2] = 0  # a zero row: scale 1.0
    monkeypatch.setattr(shardwise, "_BLOCK_ELEMS", 3 * 37)
    w8, sw = shardwise._quantize_blocks(parts, 21, 37)
    r8, rs = quantize_weight_rows(torch.cat([p.to(torch.float64) for p in parts]))
    assert w8.equal(r8) and sw.view(torch.int32).equal(rs.view(torch.int32))


# ---- resume and integrity --------------------------------------------------------------------
class _StopError(Exception):
    pass


@pytest.mark.parametrize("write", [True, False])
def test_resume_after_interruption(tiny_hf_dir: Path, tmp_path: Path, write: bool) -> None:
    ref = tmp_path / "ref"
    pack_checkpoint(tiny_hf_dir, ref, name="tiny", shard_bytes=40_000)
    src = VirtualShards(tiny_hf_dir, round_robin(tiny_hf_dir, 5))
    seen: list[str] = []

    def stop_after_3(lid: str) -> None:
        seen.append(lid)
        if len(seen) == 3:
            raise _StopError

    out = tmp_path / "new" if write else None
    with pytest.raises(_StopError):
        stream_pack_checkpoint(src, out, work_dir=tmp_path / "w", name="tiny", shard_bytes=40_000,
                               progress=stop_after_3)
    assert (tmp_path / "w" / "state.json").is_file() and not (tmp_path / "new").exists()
    r = stream_pack_checkpoint(src, out, work_dir=tmp_path / "w", name="tiny", shard_bytes=40_000)
    assert r.stats["resumed_items"] >= 3
    assert r.fingerprint == Manifest.from_file(ref / "manifest.json").fingerprint
    if write:
        assert_same_pack(ref, tmp_path / "new")
    # resume=False starts over and gives the same result
    r2 = stream_pack_checkpoint(src, None, work_dir=tmp_path / "w3", name="tiny", shard_bytes=40_000, resume=False)
    assert r2.fingerprint == r.fingerprint and r2.stats["resumed_items"] == 0


def test_edited_work_dir_never_verifies(tiny_hf_dir: Path, tmp_path: Path) -> None:
    """A byte changed in the partial output between runs: the pack is assembled from the recorded digests, so its
    full verification fails (callers verify against their pin; nothing is trusted from the work dir)."""
    src = VirtualShards(tiny_hf_dir, round_robin(tiny_hf_dir, 3))
    n = [0]

    def stop(lid: str) -> None:
        n[0] += 1
        if n[0] == 5:
            raise _StopError

    with pytest.raises(_StopError):
        stream_pack_checkpoint(src, tmp_path / "new", work_dir=tmp_path / "w", name="tiny", progress=stop)
    f = next((tmp_path / "w" / "out").glob("linears-*.safetensors"))
    st = json.loads((tmp_path / "w" / "state.json").read_text())
    from pgw.modelpack.st_io import read_safetensors_header

    ents, _ = read_safetensors_header(f)
    done = next(n for k, v in st["done"].items() for n in v["sha256"] if n in ents and n.endswith(".weight"))
    with open(f, "r+b") as fh:
        fh.seek(ents[done].begin + 1)
        b = fh.read(1)
        fh.seek(ents[done].begin + 1)
        fh.write(bytes([(b[0] + 1) % 127]))
    r = stream_pack_checkpoint(src, tmp_path / "new", work_dir=tmp_path / "w", name="tiny")
    pack_checkpoint(tiny_hf_dir, tmp_path / "ref", name="tiny")
    assert r.fingerprint == Manifest.from_file(tmp_path / "ref" / "manifest.json").fingerprint  # digests recorded
    with pytest.raises((FingerprintMismatchError, IntegrityError)):
        ModelPack.open(tmp_path / "new", verify="full")


def test_changed_plan_starts_over(tiny_hf_dir: Path, tmp_path: Path) -> None:
    src = VirtualShards(tiny_hf_dir, round_robin(tiny_hf_dir, 3))

    def stop(lid: str) -> None:
        raise _StopError

    with pytest.raises(_StopError):
        stream_pack_checkpoint(src, None, work_dir=tmp_path / "w", name="tiny", progress=stop)
    r = stream_pack_checkpoint(src, None, work_dir=tmp_path / "w", name="tiny", plain_dtype="float32")
    assert r.stats["resumed_items"] == 0 and r.manifest.version == 1


def test_rejects_like_pack_checkpoint(tiny_hf_dir: Path, tmp_path: Path) -> None:
    bad = reshard(tiny_hf_dir, tmp_path / "bad", hf_like(tiny_hf_dir, 2))
    idx = json.loads((bad / "model.safetensors.index.json").read_text())
    k = next(iter(idx["weight_map"]))
    idx["weight_map"][k] = "model-00002-of-00002.safetensors" if idx["weight_map"][k].startswith(
        "model-00001") else "model-00001-of-00002.safetensors"
    (bad / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(ConfigError, match="index says"):
        stream_pack_checkpoint(bad, None, work_dir=tmp_path / "w")
    # a NaN in a shard is caught when that shard is processed, as pack_checkpoint does
    nan = reshard(tiny_hf_dir, tmp_path / "nan", hf_like(tiny_hf_dir, 2))
    fn = "model-00002-of-00002.safetensors"
    sd = load_file(nan / fn)
    name = next(n for n in sd if n.endswith("down_proj.weight"))
    sd[name][0, 0] = float("nan")
    save_file(sd, nan / fn, metadata={"format": "pt"})
    with pytest.raises(ConfigError, match="NaN"):
        stream_pack_checkpoint(nan, tmp_path / "o", work_dir=tmp_path / "w2")
    assert not (tmp_path / "o").exists()
    with pytest.raises(ConfigError, match="already exists"):
        (tmp_path / "exists").mkdir()
        (tmp_path / "exists" / "x").write_text("x")
        stream_pack_checkpoint(tiny_hf_dir, tmp_path / "exists", work_dir=tmp_path / "w3")


# ---- the hub source (huggingface_hub replaced by a local "hub") -------------------------------
def install_fake_hub(monkeypatch: pytest.MonkeyPatch, hub: Path, repo: str, commit: str) -> dict[str, list[int]]:
    """Serve the checkpoint directory ``hub`` as hub repo ``repo`` at ``commit``: model_info, per-file downloads into
    ``local_dir``, shard headers (range requests in the real library) and ``snapshot_download`` (whole checkpoint;
    nothing is in the local cache). Returns the log: shards downloaded, shards present after each download."""
    import huggingface_hub
    from huggingface_hub.utils._safetensors import SafetensorsFileMetadata, SafetensorsRepoMetadata, TensorInfo

    from pgw.modelpack.shardwise import read_shard_header

    log: dict[str, list[Any]] = {"downloads": [], "present": []}

    class Info:
        sha = commit
        siblings: ClassVar[list[Any]] = [type("S", (), {"rfilename": p.name})() for p in hub.iterdir() if p.is_file()]

    def model_info(self: Any, r: str, revision: str | None = None, files_metadata: bool = False) -> Info:
        assert r == repo and revision in (None, "main", commit)
        return Info()

    def hf_hub_download(r: str, fn: str, *, revision: str, local_dir: str, **kw: Any) -> str:
        assert r == repo and revision == commit
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hub / fn, d / fn)
        if fn.endswith(".safetensors"):
            log["downloads"].append(fn)
            log["present"].append(len(list(d.glob("*.safetensors"))))
        return str(d / fn)

    def get_safetensors_metadata(r: str, *, revision: str, **kw: Any) -> SafetensorsRepoMetadata:
        wm = json.loads((hub / "model.safetensors.index.json").read_text())["weight_map"]
        files = {}
        for fn in sorted(set(wm.values())):
            ts = read_shard_header(hub / fn)
            files[fn] = SafetensorsFileMetadata(metadata={}, tensors={
                k: TensorInfo(dtype=t.st_dtype, shape=list(t.shape), data_offsets=(0, 0)) for k, t in ts.items()})
        return SafetensorsRepoMetadata(metadata=None, sharded=True, weight_map=wm, files_metadata=files)

    def snapshot_download(r: str, *, revision: str | None = None, local_files_only: bool = False,
                          **kw: Any) -> str:
        if local_files_only:
            raise FileNotFoundError("not cached")
        assert r == repo
        return str(hub)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", get_safetensors_metadata)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return log


def test_hub_source_downloads_one_shard_at_a_time(tiny_hf_dir: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    bf = make_bf16_ckpt(tiny_hf_dir, tmp_path / "bf", untie=True)
    commit = "ab" * 20
    hub = reshard(bf, tmp_path / commit, hf_like(bf, 3))  # named like a snapshot dir: pack_checkpoint records it
    log = install_fake_hub(monkeypatch, hub, "org/tiny", commit)
    r = stream_pack_checkpoint("org/tiny", tmp_path / "new", work_dir=tmp_path / "w", revision="main")
    assert r.manifest.source.repo == "org/tiny" and r.manifest.source.revision == commit
    assert r.manifest.name == "tiny"
    assert log["downloads"] == sorted(set(hf_like(bf, 3).values())) and max(log["present"]) == 1
    assert not (tmp_path / "w").exists()
    # pack_checkpoint of the same hub id (whole snapshot): the same pack, manifest included
    ref = pack_checkpoint("org/tiny", tmp_path / "ref", revision="main")
    assert ref.fingerprint == r.fingerprint
    assert_same_pack(tmp_path / "ref", tmp_path / "new")
    # record_revision="given" (the provisioning path) keeps the revision as passed
    g = stream_pack_checkpoint("org/tiny", None, work_dir=tmp_path / "w2", revision="main", record_revision="given")
    assert g.manifest.source.revision == "main"
    assert isinstance(HubShards, type)


# ---- real checkpoints from the local HF cache --------------------------------------------------
def _cached(repo: str, rev: str) -> Path:
    try:
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(repo, revision=rev, local_files_only=True))
    except Exception:
        pytest.skip(f"{repo}@{rev[:8]} is not in the local Hugging Face cache")
    if not (p / "model.safetensors").exists():
        pytest.skip(f"{repo} cache incomplete")
    return p


def _pack_ref_digests(ckpt: Path, out: Path, **kw: Any) -> tuple[dict[str, str], dict[str, Any]]:
    pack_checkpoint(ckpt, out, **kw)
    d, m = _file_digests(out), _manifest_body(Manifest.from_file(out / "manifest.json"))
    shutil.rmtree(out)
    return d, m


@pytest.mark.slow
def test_qwen05b_release_pins_and_bytes(tmp_path: Path) -> None:
    """Qwen2.5-0.5B at the pinned commit: written pack byte-identical to pack_checkpoint's (sharded like the hub:
    groups straddle shards), and the release pins of format 2 and format 1 reproduced in hash-only mode."""
    rev = "7ae557604adf67be50417f59c2c2f167def9a775"
    ck = _cached("Qwen/Qwen2.5-0.5B-Instruct", rev)
    name = "qwen2.5-0.5b-instruct"
    ref_d, ref_m = _pack_ref_digests(ck, tmp_path / "ref", name=name)
    assert ref_m["fingerprint"] == PIN_05B
    r = stream_pack_checkpoint(VirtualShards(ck, hf_like(ck, 4)), tmp_path / "new", work_dir=tmp_path / "w",
                               name=name)
    assert r.fingerprint == PIN_05B and r.stats["spill_peak_bytes"] > 0
    assert _file_digests(tmp_path / "new") == ref_d and _manifest_body(r.manifest) == ref_m
    shutil.rmtree(tmp_path / "new")
    # the hub-id path (cached checkpoint read in place), hash only: both formats' pins
    h2 = stream_pack_checkpoint("Qwen/Qwen2.5-0.5B-Instruct", None, work_dir=tmp_path / "w2", name=name,
                                revision=rev)
    h1 = stream_pack_checkpoint("Qwen/Qwen2.5-0.5B-Instruct", None, work_dir=tmp_path / "w3", name=name,
                                revision=rev, plain_dtype="float32")
    assert (h2.fingerprint, h1.fingerprint) == (PIN_05B, PIN1_05B)


@pytest.mark.slow
def test_qwen15b_release_pins_and_bytes(tmp_path: Path) -> None:
    rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ck = _cached("Qwen/Qwen2.5-1.5B-Instruct", rev)
    name = "qwen2.5-1.5b-instruct"
    free = shutil.disk_usage(tmp_path).free
    if free < 4e9:
        pytest.skip("needs about 2 GB of scratch disk for the packs")
    ref_d, ref_m = _pack_ref_digests(ck, tmp_path / "ref", name=name)
    assert ref_m["fingerprint"] == PIN_15B
    r = stream_pack_checkpoint(VirtualShards(ck, round_robin(ck, 5)), tmp_path / "new", work_dir=tmp_path / "w",
                               name=name)
    assert r.fingerprint == PIN_15B
    assert _file_digests(tmp_path / "new") == ref_d and _manifest_body(r.manifest) == ref_m
    shutil.rmtree(tmp_path / "new")
    h1 = stream_pack_checkpoint(VirtualShards(ck, hf_like(ck, 3)), None, work_dir=tmp_path / "w2", name=name,
                                revision=rev, plain_dtype="float32")
    assert h1.fingerprint == PIN1_15B

