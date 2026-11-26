"""Packing HF checkpoints (tiny Qwen2 / Llama) into model packs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pgw.core import linear_ids, quantize_weight_rows
from pgw.core.errors import ConfigError, ShapeError
from pgw.core.fingerprint import compute_fingerprint, sha256_file
from pgw.modelpack import (
    MANIFEST_FILE,
    PLAIN_FILE,
    Manifest,
    ModelPack,
    SourceInfo,
    arch_config_from_hf,
    hf_linear_parts,
    pack_checkpoint,
    pack_state_dict,
)


def _hf_state(ckpt: Path) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        sd.update(load_file(f))
    return sd


def test_tiny_qwen_pack_matches_direct_quantization(tiny_pack_dir, tiny_linear_weights, tiny_hf_dir):
    pack = ModelPack.open(tiny_pack_dir, verify="full")
    assert pack.name == "tiny-qwen2"
    assert len(pack.manifest.linears) == 4 * 2
    for lid, ref in tiny_linear_weights.items():
        got = pack.load_linear(lid)
        assert got.id == ref.id
        assert got.w8.dtype == torch.int8 and torch.equal(got.w8, ref.w8)
        assert got.scale.dtype == torch.float32 and torch.equal(got.scale, ref.scale)
        if ref.bias is None:
            assert got.bias is None
        else:
            assert torch.equal(got.bias, ref.bias)
    # only qkv groups have biases for Qwen2
    assert {e.id for e in pack.manifest.linears if e.bias} == {"0.qkv", "1.qkv"}


def test_tiny_qwen_manifest_contents(tiny_pack_dir, tiny_hf_dir):
    m = Manifest.from_file(tiny_pack_dir)
    assert m.fingerprint == m.compute_fingerprint()
    assert m.source.repo == tiny_hf_dir.name and "/" not in m.source.repo
    assert m.tool and m.created_at
    a = m.arch
    assert (a.hidden_size, a.num_layers, a.num_heads, a.num_kv_heads, a.head_dim, a.vocab_size) == (64, 2, 4, 2, 16, 260)
    assert a.tie_word_embeddings and a.bos_token_id == 256 and 258 in a.eos_token_ids
    qkv = m.linear("0.qkv")
    assert [(p.name, p.row_start, p.rows) for p in qkv.parts] == [("q_proj", 0, 64), ("k_proj", 64, 32), ("v_proj", 96, 32)]
    gu = m.linear("1.gate_up")
    assert [(p.name, p.row_start, p.rows) for p in gu.parts] == [("gate_proj", 0, 128), ("up_proj", 128, 128)]
    assert "lm_head" not in m.tensors  # tied
    files = {t.file for t in m.tokenizer}
    assert {"tokenizer.json", "tokenizer_config.json"} <= files
    for t in m.tokenizer:
        assert sha256_file(tiny_pack_dir / "tokenizer" / t.file) == t.sha256
    # on-disk JSON fingerprint equals the normative definition over the raw dict
    raw = json.loads((tiny_pack_dir / MANIFEST_FILE).read_text())
    assert compute_fingerprint(raw) == m.fingerprint
    m.check_consistency()


def test_plain_tensors_match_hf(tiny_pack_dir, tiny_hf_dir):
    sd = _hf_state(tiny_hf_dir)
    pack = ModelPack.open(tiny_pack_dir)
    assert torch.equal(pack.load_plain("embed_tokens"), sd["model.embed_tokens.weight"].float())
    assert torch.equal(pack.load_plain("lm_head"), sd["model.embed_tokens.weight"].float())  # tied
    assert torch.equal(pack.load_plain("final_norm"), sd["model.norm.weight"].float())
    assert torch.equal(pack.load_plain("layers.1.input_norm"), sd["model.layers.1.input_layernorm.weight"].float())
    assert torch.equal(pack.load_plain("layers.0.post_attn_norm"),
                      sd["model.layers.0.post_attention_layernorm.weight"].float())


def test_dequantized_pack_reproduces_hf_logits(tiny_pack_dir, tiny_hf_model):
    """Swapping the HF linears for dequantized pack weights keeps logits close (quantization only)."""
    import copy

    model = copy.deepcopy(tiny_hf_model)
    pack = ModelPack.open(tiny_pack_dir)
    with torch.no_grad():
        for lw in pack.iter_linears():
            layer = model.model.layers[lw.id.layer]
            w = lw.dequantize()
            for p in pack.linear_entry(lw.id).parts:
                mod = layer.self_attn if p.name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else layer.mlp
                getattr(mod, p.name).weight.copy_(w[p.row_start:p.row_start + p.rows])
        ids = torch.tensor([[256, 72, 101, 108, 108, 111, 33, 258]])
        ref = tiny_hf_model(ids).logits
        got = model(ids).logits
    assert (got - ref).abs().max().item() < 0.05 * ref.abs().max().item()
    assert not torch.equal(got, ref)  # it really was replaced


def test_tiny_llama_pack(tiny_llama_hf_dir, tmp_path):
    out = tmp_path / "llama"
    m = pack_checkpoint(tiny_llama_hf_dir, out, name="tiny-llama")
    assert m.arch.arch.value == "llama"
    assert all(e.bias is None for e in m.linears)
    sd = _hf_state(tiny_llama_hf_dir)
    pack = ModelPack.open(out, verify="full")
    if not m.arch.tie_word_embeddings:
        assert torch.equal(pack.load_plain("lm_head"), sd["lm_head.weight"].float())
    lw = pack.load_linear("1.down")
    w8, sw = quantize_weight_rows(sd["model.layers.1.mlp.down_proj.weight"])
    assert torch.equal(lw.w8, w8) and torch.equal(lw.scale, sw)


def test_untied_llama_writes_lm_head(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=1, vocab_size=50, tie_word_embeddings=False,
                      rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
                      max_position_embeddings=256, rope_theta=500000.0)
    torch.manual_seed(3)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path / "ckpt")
    m = pack_checkpoint(tmp_path / "ckpt", tmp_path / "pack")
    assert "lm_head" in m.tensors
    assert m.arch.rope_scaling is not None and m.arch.rope_scaling["rope_type"] == "llama3"
    assert m.arch.rope_theta == 500000.0
    pack = ModelPack.open(tmp_path / "pack", verify="full")
    sd = _hf_state(tmp_path / "ckpt")
    assert torch.equal(pack.load_plain("lm_head"), sd["lm_head.weight"])
    assert not torch.equal(pack.load_plain("lm_head"), pack.load_plain("embed_tokens"))


def test_fingerprint_is_deterministic(tiny_hf_dir, tiny_pack_dir, tmp_path):
    m2 = pack_checkpoint(tiny_hf_dir, tmp_path / "again", name="tiny-qwen2")
    assert m2.fingerprint == Manifest.from_file(tiny_pack_dir).fingerprint
    m3 = pack_checkpoint(tiny_hf_dir, tmp_path / "other", name="other-name")
    assert m3.fingerprint != m2.fingerprint  # name is semantic


def test_sharding(tiny_hf_dir, tiny_linear_weights, tmp_path):
    m = pack_checkpoint(tiny_hf_dir, tmp_path / "sharded", shard_bytes=20_000)
    files = m.linear_files()
    assert len(files) > 2
    n = len(files)
    assert files == [f"linears-{i:05d}-of-{n:05d}.safetensors" for i in range(n)]
    # a group's tensors are never split across shards
    for e in m.linears:
        assert len({m.tensors[t].file for t in (e.weight, e.scale, e.bias) if t}) == 1
    pack = ModelPack.open(tmp_path / "sharded", verify="full")
    for lid, ref in tiny_linear_weights.items():
        assert torch.equal(pack.load_linear(lid).w8, ref.w8)
    # shards hold standard safetensors with the format metadata HF tools expect
    with safe_open(str(tmp_path / "sharded" / files[0]), framework="pt") as f:
        assert f.metadata()["format"] == "pt"
    # tensor file names are part of the manifest, so the shard layout is covered by the fingerprint
    assert m.fingerprint != pack_checkpoint(tiny_hf_dir, tmp_path / "one").fingerprint


def test_plain_dtype_bf16(tiny_hf_dir, tmp_path):
    m = pack_checkpoint(tiny_hf_dir, tmp_path / "bf16", plain_dtype="bfloat16")
    assert m.quant.plain_dtype == "bfloat16"
    assert m.tensors["embed_tokens"].dtype == "bfloat16"
    assert m.tensors["final_norm"].dtype == "float32"  # norms stay float32
    pack = ModelPack.open(tmp_path / "bf16", verify="full")
    e = pack.load_plain("embed_tokens")
    assert e.dtype == torch.float32
    assert torch.equal(e, _hf_state(tiny_hf_dir)["model.embed_tokens.weight"].to(torch.bfloat16).float())
    with pytest.raises(ConfigError):
        pack_checkpoint(tiny_hf_dir, tmp_path / "bad", plain_dtype="int8")  # type: ignore[arg-type]


def test_sharded_hf_checkpoint(tiny_hf_model, tiny_linear_weights, tmp_path):
    ckpt = tmp_path / "sharded_ckpt"
    tiny_hf_model.save_pretrained(ckpt, max_shard_size="40KB")
    assert (ckpt / "model.safetensors.index.json").exists()
    assert len(list(ckpt.glob("*.safetensors"))) > 1
    pack_checkpoint(ckpt, tmp_path / "p")
    pack = ModelPack.open(tmp_path / "p", components=("linears", "plain"))
    for lid, ref in tiny_linear_weights.items():
        assert torch.equal(pack.load_linear(lid).w8, ref.w8)


def test_bf16_checkpoint(tiny_hf_dir, tmp_path):
    ckpt = tmp_path / "bf16ckpt"
    shutil.copytree(tiny_hf_dir, ckpt)
    sd = load_file(ckpt / "model.safetensors")
    save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, ckpt / "model.safetensors", metadata={"format": "pt"})
    m = pack_checkpoint(ckpt, tmp_path / "p")
    pack = ModelPack.open(tmp_path / "p", verify="full")
    w = torch.cat([sd[n].to(torch.bfloat16) for _, n, _ in hf_linear_parts("qwen2", 0, "qkv")])
    w8, sw = quantize_weight_rows(w)
    assert torch.equal(pack.load_linear("0.qkv").w8, w8) and torch.equal(pack.load_linear("0.qkv").scale, sw)
    assert m.version == 2 and m.tensors["embed_tokens"].dtype == "bfloat16"  # format 2: the source dtype
    assert torch.equal(pack.load_plain("embed_tokens"), sd["model.embed_tokens.weight"].to(torch.bfloat16).float())


def test_existing_output_requires_overwrite(tiny_hf_dir, tmp_path):
    out = tmp_path / "p"
    m1 = pack_checkpoint(tiny_hf_dir, out)
    with pytest.raises(ConfigError, match="already exists"):
        pack_checkpoint(tiny_hf_dir, out)
    (out / "stale.txt").write_text("x")
    m2 = pack_checkpoint(tiny_hf_dir, out, overwrite=True)
    assert m2.fingerprint == m1.fingerprint
    assert not (out / "stale.txt").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]  # temp/old dirs cleaned up
    empty = tmp_path / "empty"
    empty.mkdir()
    pack_checkpoint(tiny_hf_dir, empty)  # an empty dir is fine
    assert (empty / MANIFEST_FILE).exists()


def test_failed_pack_leaves_nothing(tiny_hf_dir, tmp_path):
    ckpt = tmp_path / "broken"
    shutil.copytree(tiny_hf_dir, ckpt)
    sd = load_file(ckpt / "model.safetensors")
    del sd["model.layers.1.mlp.down_proj.weight"]
    save_file(sd, ckpt / "model.safetensors")
    with pytest.raises(ConfigError, match="down_proj"):
        pack_checkpoint(ckpt, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out")]


def test_progress_callback_and_failure_midway(tiny_hf_dir, tmp_path):
    seen: list[str] = []
    pack_checkpoint(tiny_hf_dir, tmp_path / "ok", progress=seen.append)
    assert seen == [lid.key for lid in linear_ids(2)]

    def boom(lid: str) -> None:
        if lid == "1.o":
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        pack_checkpoint(tiny_hf_dir, tmp_path / "ok", overwrite=True, progress=boom)
    ModelPack.open(tmp_path / "ok", verify="full")  # the previous pack is untouched
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_shape_mismatch_with_config(tiny_hf_dir, tmp_path):
    ckpt = tmp_path / "bad"
    shutil.copytree(tiny_hf_dir, ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg["intermediate_size"] = 96
    (ckpt / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ShapeError, match="gate_proj"):
        pack_checkpoint(ckpt, tmp_path / "out")


def test_unexpected_bias_rejected(tiny_hf_dir, tmp_path):
    ckpt = tmp_path / "bias"
    shutil.copytree(tiny_hf_dir, ckpt)
    sd = load_file(ckpt / "model.safetensors")
    sd["model.layers.0.self_attn.o_proj.bias"] = torch.zeros(64)
    save_file(sd, ckpt / "model.safetensors")
    with pytest.raises(ConfigError, match="unexpected bias"):
        pack_checkpoint(ckpt, tmp_path / "out")


def test_bad_sources(tmp_path, tiny_hf_dir):
    with pytest.raises(ConfigError, match="neither"):
        pack_checkpoint(str(tmp_path / "does-not-exist"), tmp_path / "o")
    d = tmp_path / "noconfig"
    d.mkdir()
    with pytest.raises(ConfigError, match=r"config\.json"):
        pack_checkpoint(d, tmp_path / "o")
    shutil.copy(tiny_hf_dir / "config.json", d / "config.json")
    (d / "pytorch_model.bin").write_bytes(b"")
    with pytest.raises(ConfigError, match=r"\.bin"):
        pack_checkpoint(d, tmp_path / "o")
    cfg = json.loads((tiny_hf_dir / "config.json").read_text())
    cfg["model_type"] = "gpt2"
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match="unsupported model_type"):
        pack_checkpoint(d, tmp_path / "o")


def test_hf_cache_snapshot_provenance(tiny_hf_dir, tmp_path):
    snap = tmp_path / "hub" / "models--acme--Tiny-Qwen" / "snapshots" / ("ab" * 20)
    shutil.copytree(tiny_hf_dir, snap)
    m = pack_checkpoint(snap, tmp_path / "out")
    assert m.source.repo == "acme/Tiny-Qwen" and m.source.revision == "ab" * 20
    assert m.name == "tiny-qwen"


def test_pack_state_dict(tiny_hf_dir, tiny_linear_weights, tmp_path):
    sd = _hf_state(tiny_hf_dir)
    arch = arch_config_from_hf(json.loads((tiny_hf_dir / "config.json").read_text()))
    m = pack_state_dict(sd, arch, tmp_path / "sd", name="sd", source=SourceInfo(repo="unit"),
                        tokenizer_dir=tiny_hf_dir)
    pack = ModelPack.open(tmp_path / "sd", verify="full")
    assert pack.fingerprint == m.fingerprint
    for lid, ref in tiny_linear_weights.items():
        assert torch.equal(pack.load_linear(lid).w8, ref.w8)
    # without tokenizer
    m2 = pack_state_dict(sd, arch, tmp_path / "sd2", name="sd", source=SourceInfo(repo="unit"))
    assert m2.tokenizer == ()
    ModelPack.open(tmp_path / "sd2", verify="full")
    # missing plain tensor
    sd.pop("model.norm.weight")
    with pytest.raises(ConfigError, match=r"model\.norm"):
        pack_state_dict(sd, arch, tmp_path / "sd3", name="sd", source=SourceInfo(repo="unit"))
    assert not (tmp_path / "sd3").exists()


def test_pack_files_layout(tiny_pack_dir):
    names = sorted(p.name for p in tiny_pack_dir.iterdir())
    assert names == ["linears-00000-of-00001.safetensors", MANIFEST_FILE, PLAIN_FILE, "tokenizer"]
