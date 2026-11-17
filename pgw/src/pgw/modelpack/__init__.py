"""Model packs: public int8 weights + manifest + fingerprint, shared by gateway, factory and service.

Owner: modelpack team. Contract: SPEC.md section 8.

Typical use::

    manifest = pack_checkpoint("Qwen/Qwen2.5-0.5B-Instruct", "packs/qwen2.5-0.5b")
    pack = ModelPack.open("packs/qwen2.5-0.5b", verify="full")
    w = pack.load_linear(LinearId.parse("0.qkv"))     # LinearWeights(w8, scale, bias)
"""

from pgw.modelpack.arch import (
    HF_LINEAR_PARTS,
    HF_PLAIN_TENSORS,
    arch_config_from_hf,
    expected_linear_shape,
    hf_linear_parts,
    hf_plain_tensor_names,
)
from pgw.modelpack.loader import COMPONENTS, Component, LinearMeta, LinearWeights, ModelPack, VerifyMode
from pgw.modelpack.manifest import (
    MANIFEST_FILE,
    PACK_FORMAT,
    PACK_VERSION,
    PLAIN_FILE,
    TOKENIZER_DIR,
    ArchConfig,
    LinearEntry,
    LinearPart,
    Manifest,
    QuantConfig,
    SourceInfo,
    TensorEntry,
    TokenizerFile,
)
from pgw.modelpack.packer import TOKENIZER_FILES, CheckpointReader, pack_checkpoint, pack_state_dict

__all__ = [
    "COMPONENTS", "HF_LINEAR_PARTS", "HF_PLAIN_TENSORS", "MANIFEST_FILE", "PACK_FORMAT", "PACK_VERSION",
    "PLAIN_FILE", "TOKENIZER_DIR", "TOKENIZER_FILES",
    "ArchConfig", "CheckpointReader", "Component", "LinearEntry", "LinearMeta", "LinearPart", "LinearWeights",
    "Manifest",
    "ModelPack", "QuantConfig", "SourceInfo", "TensorEntry", "TokenizerFile", "VerifyMode",
    "arch_config_from_hf", "expected_linear_shape", "hf_linear_parts", "hf_plain_tensor_names",
    "pack_checkpoint", "pack_state_dict",
]
