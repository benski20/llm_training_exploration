"""Hardening tests: malformed inputs, invalid values and unsafe file operations are rejected."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from pgw.core.errors import ConfigError, ShapeError
from pgw.modelpack import MANIFEST_FILE, ArchConfig, Manifest, ModelPack, SourceInfo, pack_checkpoint, pack_state_dict
from pgw.modelpack.packer import _AtomicDir
from pgw.modelpack.st_io import SafetensorsStreamWriter, TensorPlan, hash_tensor_ranges, read_safetensors_header


# ---- helpers ---------------------------------------------------------------------------------
@pytest.fixture
def pack_copy(tiny_pack_dir: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "pack"
    shutil.copytree(tiny_pack_dir, dst)
    return dst


@pytest.fixture
def ckpt_copy(tiny_hf_dir: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "ckpt"
    shutil.copytree(tiny_hf_dir, dst)
    return dst


def _edit_ckpt(ckpt: Path, **updates: torch.Tensor) -> None:
    sd = load_file(ckpt / "model.safetensors")
    sd.update(updates)
    save_file(sd, ckpt / "model.safetensors")


def _write_raw_st(path: Path, header: dict, data: bytes) -> None:
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)


def _overwrite_tensor(root: Path, name: str, t: torch.Tensor) -> Manifest:
    """Replace a tensor's bytes in place AND update manifest hash + fingerprint (a self-consistent
    but invalid pack, as a buggy or malicious packer would produce)."""
    m = Manifest.from_file(root)
    e = m.tensors[name]
    entries, _ = read_safetensors_header(root / e.file)
    he = entries[name]
    raw = bytearray((root / e.file).read_bytes())
    data = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    assert len(data) == he.nbytes
    raw[he.begin : he.end] = data
    (root / e.file).write_bytes(bytes(raw))
    import hashlib

    tensors = dict(m.tensors)
    tensors[name] = e.model_copy(update={"sha256": hashlib.sha256(data).hexdigest()})
    m2 = m.model_copy(update={"tensors": tensors}).with_fingerprint()
    (root / MANIFEST_FILE).write_text(m2.to_json())
    return m2


# ---- st_io -----------------------------------------------------------------------------------
def test_header_rejects_negative_dims_and_bad_metadata(tmp_path):
    p = tmp_path / "neg.safetensors"
    _write_raw_st(p, {"x": {"dtype": "F32", "shape": [-1, -1], "data_offsets": [0, 4]}}, b"\0" * 4)
    with pytest.raises(ConfigError, match="negative"):
        read_safetensors_header(p)
    _write_raw_st(p, {"__metadata__": ["not", "a", "map"]}, b"")
    with pytest.raises(ConfigError, match="__metadata__"):
        read_safetensors_header(p)
    _write_raw_st(p, {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, b"\0" * 4)
    with pytest.raises(ConfigError, match="bad header entry"):
        read_safetensors_header(p)


def test_hash_ranges_forbids_int8_min(tmp_path):
    p = tmp_path / "w.safetensors"
    w = torch.tensor([[1, -127], [127, -128]], dtype=torch.int8)
    with SafetensorsStreamWriter(p, [TensorPlan("w", "int8", (2, 2))]) as wr:
        wr.write("w", w)
    entries, _ = read_safetensors_header(p)
    hash_tensor_ranges(p, entries.values())  # not requested: fine
    with pytest.raises(ConfigError, match="-128"):
        hash_tensor_ranges(p, entries.values(), forbid_int8_min={"w"})


# ---- manifest schema / consistency ------------------------------------------------------------
def _arch_kwargs(**over):
    kw = dict(arch="qwen2", hidden_size=64, intermediate_size=128, num_layers=1, num_heads=4, num_kv_heads=2,
              head_dim=16, vocab_size=260, rms_norm_eps=1e-6, rope_theta=1e4, max_position_embeddings=256,
              tie_word_embeddings=True)
    kw.update(over)
    return kw


@pytest.mark.parametrize(
    "over",
    [{"hidden_size": 0}, {"num_layers": -1}, {"num_kv_heads": 3}, {"rms_norm_eps": float("nan")},
     {"rope_theta": 0.0}, {"eos_token_ids": (-1,)}, {"bos_token_id": -2}],
)
def test_arch_config_rejects_invalid_values(over):
    with pytest.raises(ValueError):
        ArchConfig(**_arch_kwargs(**over))


def test_tensor_entry_rejects_bad_names_and_shapes(pack_copy):
    raw = json.loads((pack_copy / MANIFEST_FILE).read_text())
    raw["tensors"]["final_norm"]["shape"] = [-64]
    (pack_copy / MANIFEST_FILE).write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        ModelPack.open(pack_copy, verify="none")
    raw["tensors"]["final_norm"]["shape"] = [64]
    raw["tensors"]["final_norm"]["file"] = MANIFEST_FILE
    (pack_copy / MANIFEST_FILE).write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        ModelPack.open(pack_copy, verify="none")


def test_missing_qwen_bias_rejected(pack_copy):
    m = Manifest.from_file(pack_copy)
    lin = list(m.linears)
    lin[0] = lin[0].model_copy(update={"bias": None})
    tensors = {k: v for k, v in m.tensors.items() if k != "linears.0.qkv.bias"}
    m2 = m.model_copy(update={"linears": tuple(lin), "tensors": tensors}).with_fingerprint()
    with pytest.raises(ConfigError, match="missing bias"):
        m2.check_consistency()


def test_unreferenced_and_non_canonical_tensors_rejected(pack_copy):
    m = Manifest.from_file(pack_copy)
    tensors = dict(m.tensors)
    tensors["extra"] = tensors["final_norm"]
    with pytest.raises(ConfigError, match="unreferenced"):
        m.model_copy(update={"tensors": tensors}).check_consistency()
    lin = list(m.linears)
    lin[1] = lin[1].model_copy(update={"weight": "linears.0.qkv.weight"})
    with pytest.raises(ConfigError, match="non-canonical"):
        m.model_copy(update={"linears": tuple(lin)}).check_consistency()


# ---- packer ----------------------------------------------------------------------------------
def test_non_finite_weights_rejected(ckpt_copy, tmp_path):
    w = load_file(ckpt_copy / "model.safetensors")["model.layers.0.mlp.up_proj.weight"].clone()
    w[3, 5] = float("nan")
    _edit_ckpt(ckpt_copy, **{"model.layers.0.mlp.up_proj.weight": w})
    with pytest.raises(ConfigError, match="NaN or Inf"):
        pack_checkpoint(ckpt_copy, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_non_finite_plain_and_bias_rejected(ckpt_copy, tmp_path):
    sd = load_file(ckpt_copy / "model.safetensors")
    b = sd["model.layers.1.self_attn.k_proj.bias"].clone()
    b[0] = float("inf")
    _edit_ckpt(ckpt_copy, **{"model.layers.1.self_attn.k_proj.bias": b})
    with pytest.raises(ConfigError, match=r"k_proj\.bias"):
        pack_checkpoint(ckpt_copy, tmp_path / "out")
    _edit_ckpt(ckpt_copy, **{"model.layers.1.self_attn.k_proj.bias": sd["model.layers.1.self_attn.k_proj.bias"]})
    n = sd["model.norm.weight"].clone()
    n[1] = float("nan")
    _edit_ckpt(ckpt_copy, **{"model.norm.weight": n})
    with pytest.raises(ConfigError, match=r"model\.norm\.weight"):
        pack_checkpoint(ckpt_copy, tmp_path / "out")


def test_plain_dtype_overflow_rejected(ckpt_copy, tmp_path):
    e = load_file(ckpt_copy / "model.safetensors")["model.embed_tokens.weight"].clone()
    e[0, 0] = 1e6  # beyond float16 range
    _edit_ckpt(ckpt_copy, **{"model.embed_tokens.weight": e})
    with pytest.raises(ConfigError, match="overflow"):
        pack_checkpoint(ckpt_copy, tmp_path / "out", plain_dtype="float16")


def test_integer_checkpoint_weights_rejected(tiny_hf_dir, tmp_path):
    sd = {k: v.clone() for k, v in load_file(tiny_hf_dir / "model.safetensors").items()}
    sd["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(64, 64, dtype=torch.int32)
    arch_json = json.loads((tiny_hf_dir / "config.json").read_text())
    from pgw.modelpack import arch_config_from_hf

    with pytest.raises(ConfigError, match="not a float type"):
        pack_state_dict(sd, arch_config_from_hf(arch_json), tmp_path / "o", name="x", source=SourceInfo(repo="u"))


def test_plain_shape_mismatch_detected_early(tiny_hf_dir, tmp_path):
    sd = {k: v.clone() for k, v in load_file(tiny_hf_dir / "model.safetensors").items()}
    sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"][:-4]
    from pgw.modelpack import arch_config_from_hf

    arch = arch_config_from_hf(json.loads((tiny_hf_dir / "config.json").read_text()))
    with pytest.raises(ShapeError, match="embed_tokens"):
        pack_state_dict(sd, arch, tmp_path / "o", name="x", source=SourceInfo(repo="u"))


def test_bad_index_file_rejected(ckpt_copy, tmp_path):
    idx = ckpt_copy / "model.safetensors.index.json"
    idx.write_text("{broken")
    with pytest.raises(ConfigError, match="cannot parse"):
        pack_checkpoint(ckpt_copy, tmp_path / "out")
    idx.write_text(json.dumps({"weight_map": {"model.norm.weight": "../outside/model.safetensors"}}))
    with pytest.raises(ConfigError, match="outside the checkpoint"):
        pack_checkpoint(ckpt_copy, tmp_path / "out")


def test_atomic_commit_does_not_clobber_concurrent_writer(tmp_path):
    out = tmp_path / "pack"
    ad = _AtomicDir(out, overwrite=False)
    (ad.tmp / "mine").write_text("a")
    out.mkdir()
    (out / "theirs").write_text("b")  # another packer finished first
    with pytest.raises(ConfigError, match="appeared"):
        ad.commit()
    ad.abort()
    assert (out / "theirs").exists() and not (out / "mine").exists()
    ad2 = _AtomicDir(out, overwrite=True)
    (ad2.tmp / "mine").write_text("a")
    ad2.commit()
    assert (out / "mine").exists() and not (out / "theirs").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# ---- loader ----------------------------------------------------------------------------------
def test_int8_min_weight_rejected_by_full_verify(pack_copy):
    m = Manifest.from_file(pack_copy)
    lw = ModelPack.open(pack_copy).load_linear("0.o")
    w = lw.w8.clone()
    w[0, 0] = -128
    m2 = _overwrite_tensor(pack_copy, "linears.0.o.weight", w)
    ModelPack.open(pack_copy, verify="fingerprint")  # cheap mode cannot see it
    with pytest.raises(ConfigError, match="-128"):
        ModelPack.open(pack_copy, verify="full", components=("linears",))
    with pytest.raises(ConfigError, match="-128"):
        ModelPack.open(pack_copy).load_linear("0.o", check_hash=True)
    assert m2.fingerprint != m.fingerprint


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_scale_rejected(pack_copy, bad):
    sw = ModelPack.open(pack_copy).load_linear("1.down").scale.clone()
    sw[2] = bad
    _overwrite_tensor(pack_copy, "linears.1.down.scale", sw)
    pack = ModelPack.open(pack_copy)
    with pytest.raises(ConfigError, match="scales must be finite and positive"):
        pack.load_linear("1.down")
    with pytest.raises(ConfigError, match="scales"):
        ModelPack.open(pack_copy, verify="full")


def test_export_overlap_rejected_and_atomic(tiny_pack_dir, pack_copy, tmp_path):
    pack = ModelPack.open(pack_copy)
    for target in (pack_copy, pack_copy / "sub", tmp_path):
        with pytest.raises(ConfigError, match="overlaps"):
            pack.export(target, ["linears"], overwrite=True)
    assert ModelPack.open(pack_copy, verify="full").fingerprint == pack.fingerprint
    out = tmp_path / "svc"
    pack.export(out, ["linears"])
    with pytest.raises(ConfigError, match="already exists"):
        pack.export(out, ["linears"])
    pack.export(out, ["linears", "plain"], overwrite=True)
    p2 = ModelPack.open(out, verify="full", components=("linears", "plain"))
    assert p2.fingerprint == pack.fingerprint
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
