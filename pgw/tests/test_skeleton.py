"""Smoke tests for the project skeleton: imports, shared fixtures, interface data models."""

import importlib

import pytest
import torch
from click.testing import CliRunner

from pgw.core import (
    RING_K40_A12,
    PreparedWeight,
    linear_ids,
    mask_activations,
    quantize_activations,
    unmask_outputs,
)

SUBPACKAGES = ["core", "modelpack", "engine", "factory", "maskstore", "wire", "service", "gateway", "observability", "cli"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    mod = importlib.import_module(f"pgw.{name}")
    for sym in getattr(mod, "__all__", []):
        assert hasattr(mod, sym), f"pgw.{name}.{sym} missing"


def test_tiny_checkpoint_and_tokenizer(tiny_hf_dir, tiny_hf_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_hf_dir)
    ids = tok.apply_chat_template([{"role": "user", "content": "héllo"}], add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    assert max(ids) < 260 and tok.decode(ids, skip_special_tokens=True).strip().endswith("assistant")
    assert tok.decode(tok("héllo").input_ids) == "héllo"
    out = tiny_hf_model(torch.tensor([ids]))
    assert out.logits.shape == (1, len(ids), 260)


def test_tiny_linear_weights_reproduce_hf_projections(tiny_hf_model, tiny_linear_weights):
    """Quantized linear groups (masked path) approximate the HF float projections."""
    cfg = tiny_hf_model.config
    assert set(tiny_linear_weights) == {i.key for i in linear_ids(cfg.num_hidden_layers)}
    qkv = tiny_linear_weights["0.qkv"]
    hd = cfg.hidden_size // cfg.num_attention_heads
    assert qkv.w8.shape == (cfg.hidden_size + 2 * cfg.num_key_value_heads * hd, cfg.hidden_size)
    assert qkv.bias is not None and tiny_linear_weights["0.o"].bias is None
    x = torch.randn(5, cfg.hidden_size)
    attn = tiny_hf_model.model.layers[0].self_attn
    ref = torch.cat([attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)], dim=1)
    p = RING_K40_A12
    xq, sx = quantize_activations(x, p)
    r = torch.randint(0, p.modulus, xq.shape)
    pw = PreparedWeight(qkv.w8)
    yq = unmask_outputs(pw(mask_activations(xq, r, p), p), pw(r, p), p, in_features=x.shape[1])
    y = yq.double() * sx[:, None] * qkv.scale.double()[None] + qkv.bias.double()[None]
    assert torch.allclose(y.float(), ref, atol=2e-2 * ref.abs().max().item())


def test_llama_fixture(tiny_llama_hf_dir):
    assert (tiny_llama_hf_dir / "config.json").exists()
    assert (tiny_llama_hf_dir / "tokenizer.json").exists()


def test_manifest_schema_and_fingerprint():
    from pgw.modelpack import ArchConfig, LinearEntry, LinearPart, Manifest, SourceInfo, TensorEntry

    arch = ArchConfig(arch="qwen2", hidden_size=64, intermediate_size=128, num_layers=1, num_heads=4,
                      num_kv_heads=2, head_dim=16, vocab_size=260, rms_norm_eps=1e-6, rope_theta=1e4,
                      max_position_embeddings=256, tie_word_embeddings=True)
    lin = LinearEntry(id="0.o", out_features=64, in_features=64, weight="linears.0.o.weight",
                      scale="linears.0.o.scale", parts=(LinearPart(name="o_proj", row_start=0, rows=64),))
    t = TensorEntry(file="linears-00000-of-00001.safetensors", dtype="int8", shape=(64, 64), sha256="0" * 64)
    m = Manifest(name="t", source=SourceInfo(repo="tiny"), arch=arch, linears=(lin,), tensors={"linears.0.o.weight": t})
    m2 = Manifest.model_validate_json(m.with_fingerprint().to_json())
    assert m2.verify_fingerprint() == m.compute_fingerprint()
    assert m2.model_copy(update={"created_at": "x"}).verify_fingerprint() == m2.fingerprint
    assert m2.linear("0.o").spec().in_features == 64


def test_settings_classes_construct(tmp_path, monkeypatch):
    from pgw.factory import FactorySettings
    from pgw.gateway import GatewaySettings
    from pgw.maskstore import MaskStoreSettings
    from pgw.service import ServiceSettings

    monkeypatch.setenv("PGW_SERVICE_PORT", "9999")
    assert ServiceSettings().port == 9999
    assert GatewaySettings().port == 8080
    assert MaskStoreSettings(root=tmp_path).low_water_slots == 4096
    assert FactorySettings(pack_path=tmp_path).checks == 2


def test_wire_messages_and_metric_names():
    from pgw.observability import METRIC_NAMES
    from pgw.wire import ComputeItem, ComputeRequest

    req = ComputeRequest("id", "0" * 64, "K40A12", (ComputeItem("0.qkv", 1, b"\x00" * 5),))
    assert req.version == 1 and req.type == "compute"
    assert all(n.startswith("pgw_") for n in METRIC_NAMES)


def test_cli_version_and_doctor():
    from pgw.cli import cli

    r = CliRunner().invoke(cli, ["version"])
    assert r.exit_code == 0 and "protocol v1" in r.output
    r = CliRunner().invoke(cli, ["doctor", "--json"])
    assert r.exit_code == 0, r.output


def test_dependency_rules():
    """SPEC section 4: service must not import maskstore/factory; core imports nothing internal."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "pgw"
    forbidden = {"service": {"maskstore", "factory", "gateway"}, "core": set(SUBPACKAGES) - {"core"},
                 "wire": {"engine", "gateway", "service", "maskstore", "factory"}}
    for pkg, bad in forbidden.items():
        for f in (root / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for n in names:
                    parts = n.split(".")
                    if parts[0] == "pgw" and len(parts) > 1:
                        assert parts[1] not in bad, f"{f.relative_to(root)} imports {n}"
