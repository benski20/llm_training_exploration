"""Pack format 2: embeddings / LM head stored in the checkpoint's dtype (bfloat16), widened to float32 at load.

Exactness: every float32 value a loader returns is bit-identical to a format-1 (float32) pack of the same checkpoint.
Integrity: the fingerprint covers the format version and each tensor's dtype and bytes; old (format-1) packs still
load; an unknown format version gets a clear "re-provision" error.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pgw.core.errors import ConfigError, FingerprintMismatchError
from pgw.modelpack import PACK_VERSION, PACK_VERSIONS, Manifest, ModelPack, pack_checkpoint
from pgw.modelpack.masked_head import MaskedHeadPack


def make_bf16_ckpt(src: Path, out: Path, *, untie: bool = True, head_dtype: torch.dtype = torch.bfloat16) -> Path:
    """A copy of the tiny checkpoint converted to bfloat16 (as the public Qwen2.5 checkpoints are), optionally
    with an untied LM head (stored in ``head_dtype``)."""
    shutil.copytree(src, out)
    sd = load_file(out / "model.safetensors")
    new = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    if untie:
        g = torch.Generator().manual_seed(7)
        emb = sd["model.embed_tokens.weight"]
        new["lm_head.weight"] = (torch.randn(emb.shape, generator=g) * 0.05).to(head_dtype)
        cfg = json.loads((out / "config.json").read_text())
        cfg["tie_word_embeddings"] = False
        (out / "config.json").write_text(json.dumps(cfg))
    save_file(new, out / "model.safetensors", metadata={"format": "pt"})
    return out


@pytest.fixture(scope="module")
def bf16_ckpt(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_bf16_ckpt(tiny_hf_dir, tmp_path_factory.mktemp("bf16") / "ckpt")


@pytest.fixture(scope="module")
def packs(bf16_ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(format-1 float32 pack, format-2 source-dtype pack) of the same bfloat16 checkpoint."""
    d = tmp_path_factory.mktemp("packs")
    pack_checkpoint(bf16_ckpt, d / "v1", name="tiny-bf16", plain_dtype="float32")
    pack_checkpoint(bf16_ckpt, d / "v2", name="tiny-bf16")
    return d / "v1", d / "v2"


def test_format2_is_default_and_records_dtype_per_tensor(packs: tuple[Path, Path], bf16_ckpt: Path) -> None:
    v1, v2 = (Manifest.from_file(p) for p in packs)
    assert PACK_VERSION == 2 and PACK_VERSIONS == (1, 2)
    assert (v1.version, v1.quant.plain_dtype) == (1, "float32")
    assert (v2.version, v2.quant.plain_dtype) == (2, "source")
    assert v2.tensors["embed_tokens"].dtype == v2.tensors["lm_head"].dtype == "bfloat16"
    assert v1.tensors["embed_tokens"].dtype == v1.tensors["lm_head"].dtype == "float32"
    for n in v2.plain_tensor_names():  # norms stay float32
        if n not in ("embed_tokens", "lm_head"):
            assert v2.tensors[n].dtype == "float32"
    # format 2 stores the checkpoint's bytes unconverted
    with safe_open(str(bf16_ckpt / "model.safetensors"), "pt") as f, \
            safe_open(str(packs[1] / "plain.safetensors"), "pt") as g:
        for hf, name in (("model.embed_tokens.weight", "embed_tokens"), ("lm_head.weight", "lm_head")):
            a, b = f.get_tensor(hf), g.get_tensor(name)
            assert a.dtype == b.dtype == torch.bfloat16
            assert a.view(torch.int16).equal(b.view(torch.int16))
    # the linears are untouched by the format change
    for e in v1.linears:
        for t in (e.weight, e.scale, e.bias):
            if t is not None:
                assert v1.tensors[t] == v2.tensors[t]
    assert v1.fingerprint != v2.fingerprint


def test_plain_halved_on_disk(packs: tuple[Path, Path]) -> None:
    s1 = (packs[0] / "plain.safetensors").stat().st_size
    s2 = (packs[1] / "plain.safetensors").stat().st_size
    with ModelPack.open(packs[0]) as p1, ModelPack.open(packs[1]) as p2:
        big = sum(p1.manifest.tensors[n].nbytes for n in ("embed_tokens", "lm_head"))
        assert p1.size_bytes("plain") - p2.size_bytes("plain") == big // 2
    assert abs((s1 - s2) - big // 2) < 4096  # the safetensors headers differ by a few bytes


def test_loaded_values_bit_identical(packs: tuple[Path, Path]) -> None:
    """Every float32 tensor the gateway / factory / provider get is the same, bit for bit."""
    with ModelPack.open(packs[0], verify="full") as p1, ModelPack.open(packs[1], verify="full") as p2:
        a, b = p1.load_plain_tensors(), p2.load_plain_tensors()
        assert set(a) == set(b)
        for n in a:
            assert a[n].dtype == b[n].dtype == torch.float32
            assert a[n].view(torch.int32).equal(b[n].view(torch.int32)), n
        for n in p1.manifest.plain_tensor_names():
            assert p1.load_plain(n).view(torch.int32).equal(p2.load_plain(n).view(torch.int32))
        for e in p1.manifest.linears:
            l1, l2 = p1.load_linear(e.id), p2.load_linear(e.id)
            assert l1.w8.equal(l2.w8) and l1.scale.equal(l2.scale)
        # the masked LM head (int8 head, scales, row norm) derived on every side
        h1, h2 = MaskedHeadPack(p1), MaskedHeadPack(p2)
        w1, w2 = h1.load_linear(h1.head_id), h2.load_linear(h2.head_id)
        assert w1.w8.equal(w2.w8) and w1.scale.equal(w2.scale)
        m1 = MaskedHeadPack(p1, float_head=False).load_linear_meta(h1.head_id)
        m2 = MaskedHeadPack(p2, float_head=False).load_linear_meta(h2.head_id)
        assert m1.scale.equal(m2.scale) and m1.row_norm_sq_max == m2.row_norm_sq_max


def test_float32_source_keeps_float32_per_tensor(tiny_hf_dir: Path, bf16_ckpt: Path, tmp_path: Path) -> None:
    """A float32 checkpoint is never narrowed: format 2 keeps float32 for it (the choice is per tensor)."""
    m = pack_checkpoint(tiny_hf_dir, tmp_path / "f32")
    assert m.version == 2 and m.tensors["embed_tokens"].dtype == "float32"
    sd = load_file(tiny_hf_dir / "model.safetensors")
    with ModelPack.open(tmp_path / "f32", verify="full") as p:
        assert p.load_plain("embed_tokens").equal(sd["model.embed_tokens.weight"])
    # mixed checkpoint: bfloat16 embeddings, float32 untied head -> one of each
    mixed = make_bf16_ckpt(tiny_hf_dir, tmp_path / "mixed", head_dtype=torch.float32)
    mm = pack_checkpoint(mixed, tmp_path / "mp")
    assert (mm.tensors["embed_tokens"].dtype, mm.tensors["lm_head"].dtype) == ("bfloat16", "float32")
    ref = load_file(mixed / "model.safetensors")
    with ModelPack.open(tmp_path / "mp", verify="full") as p:
        assert p.load_plain("lm_head").equal(ref["lm_head.weight"])
        assert p.load_plain("embed_tokens").equal(ref["model.embed_tokens.weight"].float())


def test_format1_still_written_and_loaded(bf16_ckpt: Path, packs: tuple[Path, Path], tmp_path: Path) -> None:
    """plain_dtype='float32' writes format 1 exactly as before (no new manifest fields: old pins reproduce) and
    format-1 packs load and verify."""
    raw = json.loads((packs[0] / "manifest.json").read_text())
    assert raw["version"] == 1 and raw["quant"] == {"weights": "int8-per-row-symmetric", "plain_dtype": "float32"}
    again = pack_checkpoint(bf16_ckpt, tmp_path / "again", name="tiny-bf16", plain_dtype="float32")
    assert again.fingerprint == Manifest.from_file(packs[0]).fingerprint
    with ModelPack.open(packs[0], verify="full", expected_fingerprint=again.fingerprint) as p:
        assert p.manifest.version == 1


def _rewrite(pack: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    p = pack / "manifest.json"
    doc = json.loads(p.read_text())
    mutate(doc)
    p.write_text(json.dumps(doc))


def test_fingerprint_binds_format_and_dtype(packs: tuple[Path, Path], tmp_path: Path) -> None:
    v1fp = Manifest.from_file(packs[0]).fingerprint
    v2fp = Manifest.from_file(packs[1]).fingerprint
    # a format-2 pack is not accepted under the format-1 pin, nor the other way round
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(packs[1], expected_fingerprint=v1fp)
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(packs[0], expected_fingerprint=v2fp)
    # relabelling the stored dtype (or the version) without re-fingerprinting: content mismatch
    for mutate in (lambda d: d["tensors"]["embed_tokens"].update(dtype="float32"),
                   lambda d: d.update(version=1, quant={**d["quant"], "plain_dtype": "bfloat16"})):
        work = tmp_path / f"w{id(mutate)}"
        shutil.copytree(packs[1], work)
        _rewrite(work, mutate)
        with pytest.raises(FingerprintMismatchError):
            ModelPack.open(work, verify="fingerprint")
    # re-fingerprinted relabel: self-consistent, but not the pinned pack
    work = tmp_path / "refp"
    shutil.copytree(packs[1], work)
    m = Manifest.from_file(work)
    t = dict(m.tensors)
    t["embed_tokens"] = t["embed_tokens"].model_copy(update={"dtype": "float32"})
    forged = m.model_copy(update={"tensors": t, "fingerprint": None}).with_fingerprint()
    (work / "manifest.json").write_text(forged.to_json())
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(work, expected_fingerprint=v2fp)
    with pytest.raises(ConfigError):  # and the file does not hold what it now claims
        ModelPack.open(work, verify="full")


def test_tampered_bf16_bytes_rejected(packs: tuple[Path, Path], tmp_path: Path) -> None:
    from pgw.modelpack.st_io import read_safetensors_header

    work = tmp_path / "t"
    shutil.copytree(packs[1], work)
    p = work / "plain.safetensors"
    entries, _ = read_safetensors_header(p)
    raw = bytearray(p.read_bytes())
    raw[entries["lm_head"].begin + 5] ^= 0x01  # one bit of one bfloat16 head value
    p.write_bytes(bytes(raw))
    fp = Manifest.from_file(packs[1]).fingerprint
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(work, verify="full", expected_fingerprint=fp)
    # the masked head's source is hash-checked on load too (provider / factory / gateway side)
    with ModelPack.open(work, verify="fingerprint", expected_fingerprint=fp) as pk:
        h = MaskedHeadPack(pk)
        with pytest.raises(FingerprintMismatchError):
            h.load_linear(h.head_id)


def test_inconsistent_version_rejected(packs: tuple[Path, Path], tmp_path: Path) -> None:
    for i, mutate in enumerate((lambda d: d.update(version=1),  # format 1 cannot say 'source'
                                lambda d: d["quant"].update(plain_dtype="float32"))):  # format 2 must say 'source'
        work = tmp_path / f"v{i}"
        shutil.copytree(packs[1], work)
        _rewrite(work, mutate)
        with pytest.raises(ConfigError, match="version"):
            ModelPack.open(work, verify="fingerprint")
    # format 2 keeps the norms float32 (checked even with a matching, re-computed fingerprint)
    work = tmp_path / "norm"
    shutil.copytree(packs[1], work)
    m = Manifest.from_file(work)
    t = dict(m.tensors)
    t["final_norm"] = t["final_norm"].model_copy(update={"dtype": "bfloat16"})
    (work / "manifest.json").write_text(m.model_copy(update={"tensors": t, "fingerprint": None})
                                        .with_fingerprint().to_json())
    with pytest.raises(ConfigError, match="final_norm"):
        ModelPack.open(work, verify="fingerprint")


def test_unknown_version_asks_to_reprovision(packs: tuple[Path, Path], tmp_path: Path) -> None:
    work = tmp_path / "v3"
    shutil.copytree(packs[1], work)
    _rewrite(work, lambda d: d.update(version=3))
    with pytest.raises(ConfigError, match=r"pack format version 3.*re-provision"):
        ModelPack.open(work)


def test_invalid_plain_dtype(bf16_ckpt: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        pack_checkpoint(bf16_ckpt, tmp_path / "x", plain_dtype="int8")  # type: ignore[arg-type]
