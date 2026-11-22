"""ModelPack loading, partial loading, verification and tamper detection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from pgw.core import LINEAR_ORDER, RING_K32_A8, RING_K40_A12, LinearKind, RingParams
from pgw.core.errors import ConfigError, FingerprintMismatchError, RingOverflowError, ShapeError
from pgw.modelpack import MANIFEST_FILE, PLAIN_FILE, Manifest, ModelPack
from pgw.modelpack.st_io import read_safetensors_header


@pytest.fixture
def pack_copy(tiny_pack_dir: Path, tmp_path: Path) -> Path:
    """A private, mutable copy of the tiny pack."""
    dst = tmp_path / "pack"
    shutil.copytree(tiny_pack_dir, dst)
    return dst


def _flip_byte(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


def _rewrite_manifest(root: Path, refingerprint: bool = True, **updates) -> Manifest:
    m = Manifest.from_file(root)
    m = m.model_copy(update=updates)
    if refingerprint:
        m = m.with_fingerprint()
    (root / MANIFEST_FILE).write_text(m.to_json())
    return m


def test_metadata(tiny_pack):
    assert tiny_pack.name == "tiny-qwen2"
    assert len(tiny_pack.fingerprint) == 64 and tiny_pack.short_fingerprint == tiny_pack.fingerprint[:12]
    assert tiny_pack.arch.hidden_size == 64 and tiny_pack.num_layers == 2
    specs = tiny_pack.linear_specs()
    assert [s.id.key for s in specs][:4] == ["0.qkv", "0.o", "0.gate_up", "0.down"]
    assert specs[0].has_bias and not specs[1].has_bias
    assert tiny_pack.tokenizer_dir.is_dir()
    assert tiny_pack.max_in_features() == 128
    assert "tiny-qwen2" in repr(tiny_pack)


def test_verify_modes_on_clean_pack(tiny_pack_dir):
    for mode in ("none", "fingerprint", "full"):
        p = ModelPack.open(tiny_pack_dir, verify=mode)
        assert p.fingerprint == Manifest.from_file(tiny_pack_dir).fingerprint
    with pytest.raises(ConfigError, match="verify mode"):
        ModelPack.open(tiny_pack_dir, verify="paranoid")  # type: ignore[arg-type]


def test_expected_fingerprint(tiny_pack_dir):
    fp = Manifest.from_file(tiny_pack_dir).fingerprint
    assert ModelPack.open(tiny_pack_dir, expected_fingerprint=fp).fingerprint == fp
    other = "f" * 64
    for mode in ("none", "fingerprint", "full"):
        with pytest.raises(FingerprintMismatchError):
            ModelPack.open(tiny_pack_dir, verify=mode, expected_fingerprint=other)
    with pytest.raises(ConfigError, match="invalid model fingerprint"):
        ModelPack.open(tiny_pack_dir, expected_fingerprint="xyz")


def test_tampered_weight_detected_by_full_only(pack_copy):
    entries, _ = read_safetensors_header(pack_copy / "linears-00000-of-00001.safetensors")
    e = entries["linears.1.down.weight"]
    _flip_byte(pack_copy / "linears-00000-of-00001.safetensors", e.begin + 5)
    ModelPack.open(pack_copy, verify="fingerprint")  # cheap mode does not hash
    with pytest.raises(FingerprintMismatchError, match="1 tensor") as ei:
        ModelPack.open(pack_copy, verify="full")
    assert ei.value.details["mismatched"] == ["linears.1.down.weight"]
    # per-load hash checks catch it too
    p = ModelPack.open(pack_copy, verify="none")
    p.load_linear("0.qkv", check_hash=True)
    with pytest.raises(FingerprintMismatchError):
        p.load_linear("1.down", check_hash=True)


def test_tampered_plain_and_tokenizer(pack_copy):
    entries, _ = read_safetensors_header(pack_copy / PLAIN_FILE)
    _flip_byte(pack_copy / PLAIN_FILE, entries["final_norm"].begin)
    with pytest.raises(FingerprintMismatchError, match="final_norm"):
        ModelPack.open(pack_copy, verify="full")
    ModelPack.open(pack_copy, verify="full", components=("linears", "tokenizer"))  # not checked
    tok = pack_copy / "tokenizer" / "tokenizer.json"
    tok.write_text(tok.read_text() + " ")
    with pytest.raises(FingerprintMismatchError, match=r"tokenizer\.json"):
        ModelPack.open(pack_copy, verify="full", components=("tokenizer",))


def test_manifest_edit_breaks_fingerprint(pack_copy):
    raw = json.loads((pack_copy / MANIFEST_FILE).read_text())
    raw["arch"]["rope_theta"] = 1.0
    (pack_copy / MANIFEST_FILE).write_text(json.dumps(raw))
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(pack_copy)
    ModelPack.open(pack_copy, verify="none")  # explicit opt-out


def test_non_semantic_manifest_fields_do_not_matter(pack_copy):
    fp = Manifest.from_file(pack_copy).fingerprint
    _rewrite_manifest(pack_copy, refingerprint=False, notes="hello", created_at="2020-01-01T00:00:00Z", tool="x")
    assert ModelPack.open(pack_copy, verify="full").fingerprint == fp


def test_manifest_hash_swap_detected(pack_copy):
    """An attacker who edits a tensor AND updates its hash in the manifest changes the fingerprint."""
    m = Manifest.from_file(pack_copy)
    fp = m.fingerprint
    tensors = dict(m.tensors)
    tensors["final_norm"] = tensors["final_norm"].model_copy(update={"sha256": "0" * 64})
    m2 = _rewrite_manifest(pack_copy, tensors=tensors)
    assert m2.fingerprint != fp
    with pytest.raises(FingerprintMismatchError):
        ModelPack.open(pack_copy, expected_fingerprint=fp)


def test_structural_inconsistency_rejected(pack_copy):
    m = Manifest.from_file(pack_copy)
    arch = m.arch.model_copy(update={"intermediate_size": 96})
    _rewrite_manifest(pack_copy, arch=arch)
    with pytest.raises(ConfigError, match="shape"):
        ModelPack.open(pack_copy)


def test_linear_order_and_parts_checked(pack_copy):
    m = Manifest.from_file(pack_copy)
    _rewrite_manifest(pack_copy, linears=tuple(reversed(m.linears)))
    with pytest.raises(ConfigError, match="canonical order"):
        ModelPack.open(pack_copy)
    lin = list(m.linears)
    lin[0] = lin[0].model_copy(update={"parts": tuple(reversed(lin[0].parts))})
    _rewrite_manifest(pack_copy, linears=tuple(lin))
    with pytest.raises(ConfigError, match="parts"):
        ModelPack.open(pack_copy)


def test_missing_and_invalid_manifest(tmp_path, pack_copy):
    with pytest.raises(ConfigError, match="does not exist"):
        ModelPack.open(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="no model pack manifest"):
        ModelPack.open(empty)
    (pack_copy / MANIFEST_FILE).write_text("{not json")
    with pytest.raises(ConfigError, match="invalid model pack manifest"):
        ModelPack.open(pack_copy)
    (pack_copy / MANIFEST_FILE).write_text(json.dumps({"format": "other"}))
    with pytest.raises(ConfigError):
        ModelPack.open(pack_copy, verify="none")


def test_path_traversal_rejected_in_manifest(pack_copy):
    raw = json.loads((pack_copy / MANIFEST_FILE).read_text())
    raw["tensors"]["final_norm"]["file"] = "../../etc/passwd"
    (pack_copy / MANIFEST_FILE).write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        ModelPack.open(pack_copy, verify="none")
    raw = json.loads((pack_copy / MANIFEST_FILE).read_text())
    raw["tensors"]["final_norm"]["file"] = PLAIN_FILE
    raw["tokenizer"][0]["file"] = "../manifest.json"
    (pack_copy / MANIFEST_FILE).write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        ModelPack.open(pack_copy, verify="none")


def test_missing_files_per_component(pack_copy):
    (pack_copy / PLAIN_FILE).unlink()
    with pytest.raises(ConfigError, match="missing 1 file"):
        ModelPack.open(pack_copy)
    p = ModelPack.open(pack_copy, verify="full", components=("linears",))  # service/factory view
    assert p.load_linear("0.o").w8.shape == (64, 64)
    with pytest.raises(ConfigError, match="missing"):
        p.load_plain("final_norm")
    with pytest.raises(ConfigError, match="unknown pack component"):
        ModelPack.open(pack_copy, components=("weights",))  # type: ignore[arg-type]


def test_header_layout_mismatch_detected(pack_copy, tiny_pack_dir):
    """A shard swapped for a different file with the same name fails full verification."""
    shutil.copy(pack_copy / PLAIN_FILE, pack_copy / "linears-00000-of-00001.safetensors")
    with pytest.raises(ConfigError, match="tensor set differs"):
        ModelPack.open(pack_copy, verify="full", components=("linears",))
    p = ModelPack.open(pack_copy, verify="fingerprint", components=("linears",))
    with pytest.raises(ConfigError):
        p.load_linear("0.qkv")


def test_load_tensor_shape_check(pack_copy):
    m = Manifest.from_file(pack_copy)
    tensors = dict(m.tensors)
    # point final_norm at a tensor of the right name in a file where its dtype differs
    tensors["final_norm"] = tensors["final_norm"].model_copy(update={"dtype": "float16"})
    _rewrite_manifest(pack_copy, tensors=tensors)
    p = ModelPack.open(pack_copy, verify="fingerprint", components=("linears",))
    with pytest.raises(ShapeError, match="final_norm"):
        p.load_plain("final_norm")


def test_iterators_and_layers(tiny_pack):
    all_ids = [lw.id.key for lw in tiny_pack.iter_linears()]
    assert all_ids == [e.id for e in tiny_pack.manifest.linears]
    assert [lw.id.key for lw in tiny_pack.iter_linears(layers=[1], kinds=["down", LinearKind.QKV])] == ["1.qkv", "1.down"]
    layers = list(tiny_pack.iter_layers())
    assert [i for i, _ in layers] == [0, 1]
    l0 = layers[0][1]
    assert set(l0) == set(LINEAR_ORDER)  # the four per-layer kinds (LM_HEAD is never a layer linear)
    assert l0[LinearKind.GATE_UP].w8.shape == (256, 64)
    assert l0[LinearKind.DOWN].spec.in_features == 128
    assert [i for i, _ in tiny_pack.iter_layers(layers=[1])] == [1]
    with pytest.raises(ConfigError, match="out of range"):
        tiny_pack.load_layer(2)
    with pytest.raises(ConfigError, match="no linear"):
        tiny_pack.load_linear("7.qkv")


def test_plain_loading(tiny_pack):
    plain = tiny_pack.load_plain_tensors()
    assert set(plain) == {"embed_tokens", "final_norm", "lm_head"} | {
        f"layers.{i}.{n}" for i in range(2) for n in ("input_norm", "post_attn_norm")}
    assert plain["lm_head"] is plain["embed_tokens"]  # tied: no second copy
    assert plain["embed_tokens"].shape == (260, 64)
    assert tiny_pack.load_plain("final_norm", dtype=torch.float64).dtype == torch.float64
    with pytest.raises(ConfigError, match="unknown plain tensor"):
        tiny_pack.load_plain("layers.9.input_norm")
    with pytest.raises(ConfigError, match="no tensor"):
        tiny_pack.load_tensor("nope")


def test_rings_and_sizes(tiny_pack):
    assert tiny_pack.supported_rings() == ["K32A8", "K40A12", "K48A20"]
    tiny_pack.check_ring(RING_K40_A12)
    small = RingParams(16, 8)  # 16-bit ring: 127*127*128 overflows 2^15
    assert tiny_pack.supported_rings([small, RING_K32_A8]) == ["K32A8"]
    with pytest.raises(RingOverflowError):
        tiny_pack.check_ring(small)
    lin = tiny_pack.size_bytes("linears")
    plain = tiny_pack.size_bytes("plain")
    assert lin > 0 and plain > 0 and tiny_pack.size_bytes() == lin + plain
    assert plain == 4 * (260 * 64 + 64 + 4 * 64)
    with pytest.raises(ConfigError):
        tiny_pack.size_bytes("bogus")  # type: ignore[arg-type]


def test_device_argument_cpu(tiny_pack):
    lw = tiny_pack.load_linear("0.o", device=torch.device("cpu"))
    assert lw.w8.device.type == "cpu"


@pytest.mark.gpu
def test_load_to_cuda(tiny_pack):  # pragma: no cover - needs CUDA
    lw = tiny_pack.load_linear("0.o", device="cuda")
    assert lw.w8.is_cuda and lw.scale.is_cuda


def test_export_subsets(tiny_pack, tmp_path):
    svc = tiny_pack.export(tmp_path / "svc", ["linears"])
    assert sorted(p.name for p in svc.iterdir()) == ["linears-00000-of-00001.safetensors", MANIFEST_FILE]
    p = ModelPack.open(svc, verify="full", components=["linears"])
    assert p.fingerprint == tiny_pack.fingerprint
    with pytest.raises(ConfigError):
        ModelPack.open(svc)  # plain/tokenizer missing
    gw = tiny_pack.export(tmp_path / "gw", ["plain", "tokenizer"], link=False)
    g = ModelPack.open(gw, verify="full", components=["plain", "tokenizer"])
    assert g.fingerprint == tiny_pack.fingerprint
    assert (gw / "tokenizer" / "tokenizer.json").exists()
    with pytest.raises(ConfigError, match="already exists"):
        tiny_pack.export(tmp_path / "gw", ["plain"])
    tiny_pack.export(tmp_path / "gw", ["plain"], overwrite=True)
    with pytest.raises(ConfigError, match="opened without"):
        p.export(tmp_path / "x", ["plain"])


def test_tokenizer_loads_from_pack(tiny_pack):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_pack.tokenizer_dir)
    ids = tok("hi")["input_ids"]
    assert ids[-2:] == [ord("h"), ord("i")]
    assert tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False).startswith("<|im_start|>user")


def test_concurrent_loads_and_close(tiny_pack_dir):
    from concurrent.futures import ThreadPoolExecutor

    with ModelPack.open(tiny_pack_dir) as p:
        ref = {lw.id.key: lw.w8 for lw in p.iter_linears()}
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda k: (k, p.load_linear(k).w8), list(ref) * 4))
        assert all(torch.equal(ref[k], w) for k, w in got)
    p.close()
    assert torch.equal(p.load_linear("0.o").w8, ref["0.o"])  # reopens lazily


def test_full_verify_single_thread(tiny_pack_dir):
    ModelPack.open(tiny_pack_dir, verify="full", workers=1)
    p = ModelPack.open(tiny_pack_dir, verify="none")
    p.verify_files(["linears"], workers=3)


def test_content_digest(tiny_pack, tiny_pack_dir):
    import hashlib

    assert tiny_pack.content_digest() == hashlib.sha256((tiny_pack_dir / MANIFEST_FILE).read_bytes()).hexdigest()
