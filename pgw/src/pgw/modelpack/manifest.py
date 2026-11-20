"""Model pack manifest schema (``manifest.json``) and fingerprint binding.

Pack directory layout (SPEC.md section 8.1)::

    <pack>/manifest.json                 this schema (canonical JSON not required on disk)
    <pack>/linears-00000-of-0000N.safetensors   int8 weights, float32 scales, float32 biases
    <pack>/plain.safetensors             embeddings, norms, lm_head (plain_dtype)
    <pack>/tokenizer/...                 tokenizer files copied from the checkpoint

Tensor naming inside safetensors files:

* ``linears.<id>.weight``  int8 ``(m, n)``     e.g. ``linears.3.qkv.weight``
* ``linears.<id>.scale``   float32 ``(m,)``    per-row weight scale ``sw``
* ``linears.<id>.bias``    float32 ``(m,)``    only if the group has a bias
* ``embed_tokens``, ``final_norm``, ``lm_head`` (absent if tied), ``layers.<i>.input_norm``,
  ``layers.<i>.post_attn_norm``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pgw.core.errors import ConfigError
from pgw.core.fingerprint import compute_fingerprint, require_same_fingerprint, validate_fingerprint
from pgw.core.types import LinearId, LinearKind, LinearSpec, ModelArch, linear_ids

__all__ = [
    "MANIFEST_FILE",
    "PACK_FORMAT",
    "PACK_VERSION",
    "PLAIN_FILE",
    "TOKENIZER_DIR",
    "ArchConfig",
    "LinearEntry",
    "LinearPart",
    "Manifest",
    "QuantConfig",
    "SourceInfo",
    "TensorEntry",
    "TokenizerFile",
]

MANIFEST_FILE = "manifest.json"
PLAIN_FILE = "plain.safetensors"
TOKENIZER_DIR = "tokenizer"
_HEX64 = r"^[0-9a-f]{64}$"
_SAFE_REL = r"^[A-Za-z0-9._-][A-Za-z0-9._/-]*$"
PACK_FORMAT: Literal["pgw-modelpack"] = "pgw-modelpack"
PACK_VERSION: Literal[1] = 1

_STRICT = ConfigDict(extra="forbid", frozen=True)


class TensorEntry(BaseModel):
    """Location and integrity data of one tensor."""

    model_config = _STRICT
    file: str = Field(description="File name relative to the pack root", pattern=_SAFE_REL)
    dtype: Literal["int8", "float32", "float16", "bfloat16"]
    shape: tuple[int, ...]
    sha256: str = Field(description="SHA-256 (hex) of the tensor's raw little-endian bytes", pattern=_HEX64)

    @field_validator("file")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if "/" in v or v in (".", "..") or v == MANIFEST_FILE:
            raise ValueError(f"tensor file must be a plain file name in the pack root, got {v!r}")
        return v

    @field_validator("shape")
    @classmethod
    def _non_negative(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(d < 0 for d in v):
            raise ValueError(f"tensor dimensions must be non-negative, got {list(v)}")
        return v

    @property
    def numel(self) -> int:
        """Number of elements."""
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.numel * {"int8": 1, "float32": 4, "float16": 2, "bfloat16": 2}[self.dtype]


class LinearPart(BaseModel):
    """One HF projection inside a concatenated group, e.g. ``k_proj`` rows 896..1023 of ``0.qkv``."""

    model_config = _STRICT
    name: str = Field(min_length=1)
    row_start: int = Field(ge=0)
    rows: int = Field(gt=0)


class LinearEntry(BaseModel):
    """One masked linear group."""

    model_config = _STRICT
    id: str = Field(description="Canonical linear id, e.g. '3.qkv'")
    out_features: int = Field(gt=0)
    in_features: int = Field(gt=0)
    weight: str = Field(description="Tensor name of the int8 weight")
    scale: str = Field(description="Tensor name of the float32 per-row scale")
    bias: str | None = None
    parts: tuple[LinearPart, ...]
    row_norm_sq_max: int | None = Field(
        default=None,
        ge=0,
        description="max_j sum_i W8[j, i]^2 (exact integer): the largest squared Euclidean norm of a weight "
        "row. Capped rings bound each call's outputs with its square root, so the gateway never reads W8. "
        "Absent in packs written before it existed (they keep their fingerprint).",
    )

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        LinearId.parse(v)
        return v

    @model_validator(mode="after")
    def _norm_in_range(self) -> LinearEntry:
        if self.row_norm_sq_max is not None and self.row_norm_sq_max > 127 * 127 * self.in_features:
            raise ValueError(
                f"linear {self.id}: row_norm_sq_max {self.row_norm_sq_max} exceeds 127^2 * in_features"
            )
        return self

    @property
    def linear_id(self) -> LinearId:
        """Parsed id."""
        return LinearId.parse(self.id)

    def spec(self) -> LinearSpec:
        """As a :class:`~pgw.core.types.LinearSpec`."""
        return LinearSpec(self.linear_id, self.out_features, self.in_features, self.bias is not None)


class ArchConfig(BaseModel):
    """Architecture hyper-parameters the engine needs (copied from the HF config)."""

    model_config = _STRICT
    arch: ModelArch
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    rms_norm_eps: float = Field(gt=0, allow_inf_nan=False)
    rope_theta: float = Field(gt=0, allow_inf_nan=False)
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = Field(gt=0)
    tie_word_embeddings: bool
    bos_token_id: int | None = Field(default=None, ge=0)
    eos_token_ids: tuple[int, ...] = ()
    hidden_act: Literal["silu"] = "silu"

    @model_validator(mode="after")
    def _check_heads(self) -> ArchConfig:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads={self.num_heads} is not a multiple of num_kv_heads={self.num_kv_heads}"
            )
        if any(e < 0 for e in self.eos_token_ids):
            raise ValueError(f"eos_token_ids must be non-negative, got {list(self.eos_token_ids)}")
        return self


class QuantConfig(BaseModel):
    """Quantization scheme of the pack."""

    model_config = _STRICT
    weights: Literal["int8-per-row-symmetric"] = "int8-per-row-symmetric"
    plain_dtype: Literal["float32", "float16", "bfloat16"] = "float32"


class SourceInfo(BaseModel):
    """Provenance of the checkpoint."""

    model_config = _STRICT
    repo: str = Field(description="HF repo id or local path name (no absolute paths)")
    revision: str | None = None
    license: str | None = None


class TokenizerFile(BaseModel):
    """A tokenizer file copied into ``<pack>/tokenizer/``."""

    model_config = _STRICT
    file: str = Field(description="File name relative to <pack>/tokenizer", pattern=_SAFE_REL)
    sha256: str = Field(pattern=_HEX64)

    @field_validator("file")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v.split("/") or v.startswith("/"):
            raise ValueError(f"tokenizer file must stay inside the tokenizer dir, got {v!r}")
        return v


class Manifest(BaseModel):
    """``manifest.json``. Every field except ``fingerprint``, ``created_at``, ``tool`` and ``notes``
    is covered by the fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["pgw-modelpack"] = PACK_FORMAT
    version: Literal[1] = PACK_VERSION
    name: str = Field(min_length=1, description="Served model name, e.g. 'qwen2.5-0.5b-instruct'")
    source: SourceInfo
    arch: ArchConfig
    quant: QuantConfig = QuantConfig()
    linears: tuple[LinearEntry, ...] = Field(description="All 4*num_layers groups in canonical order")
    tensors: dict[str, TensorEntry]
    tokenizer: tuple[TokenizerFile, ...] = ()
    fingerprint: str | None = None
    created_at: str | None = None
    tool: str | None = None
    notes: str | None = None

    # ---- fingerprint ---------------------------------------------------------------------
    def semantic_dict(self) -> dict[str, Any]:
        """JSON-mode dict used for fingerprinting (enums as values, tuples as lists).

        Optional linear fields that are unset (``row_norm_sq_max`` of older packs) are left out, so a
        pack written before the field existed keeps its fingerprint; when set, the value is covered.
        """
        return _drop_unset_linear_fields(self.model_dump(mode="json"))

    def compute_fingerprint(self) -> str:
        """Fingerprint of this manifest's semantic content."""
        return compute_fingerprint(self.semantic_dict())

    def with_fingerprint(self) -> Manifest:
        """Copy with ``fingerprint`` set."""
        return self.model_copy(update={"fingerprint": self.compute_fingerprint()})

    def verify_fingerprint(self, expected: str | None = None) -> str:
        """Check the stored fingerprint against the content (and ``expected`` if given); return it."""
        actual = self.compute_fingerprint()
        stored = validate_fingerprint(self.fingerprint or "")
        require_same_fingerprint(stored, actual, context="manifest content")
        if expected is not None:
            require_same_fingerprint(expected, actual, context="expected pack")
        return actual

    # ---- helpers ---------------------------------------------------------------------------
    def linear(self, lid: LinearId | str) -> LinearEntry:
        """Entry for ``lid``; raises ``KeyError`` if absent."""
        key = str(lid)
        for e in self.linears:
            if e.id == key:
                return e
        raise KeyError(key)

    def linear_specs(self) -> list[LinearSpec]:
        """Specs of all linears in canonical order."""
        return [e.spec() for e in self.linears]

    def plain_tensor_names(self) -> list[str]:
        """Names of the plain tensors this manifest must contain."""
        names = ["embed_tokens", "final_norm"]
        if not self.arch.tie_word_embeddings:
            names.append("lm_head")
        for i in range(self.arch.num_layers):
            names += [f"layers.{i}.input_norm", f"layers.{i}.post_attn_norm"]
        return names

    def linear_files(self) -> list[str]:
        """Sorted distinct files holding linear tensors."""
        out: set[str] = set()
        for e in self.linears:
            for t in (e.weight, e.scale, e.bias):
                if t is not None:
                    out.add(self.tensors[t].file)
        return sorted(out)

    def plain_files(self) -> list[str]:
        """Sorted distinct files holding plain tensors."""
        return sorted({self.tensors[n].file for n in self.plain_tensor_names() if n in self.tensors})

    def check_consistency(self) -> None:
        """Structural self-consistency: canonical linear order, shapes implied by ``arch``, dtypes,
        part layout, canonical tensor names, biases as the architecture implies (SPEC 8.2: Qwen2
        ``qkv`` only, Llama none), presence of every referenced tensor and of all plain tensors,
        and no unreferenced tensors.

        Raises:
            ConfigError: describing the first problem found.
        """
        from pgw.modelpack.arch import expected_linear_shape

        a = self.arch
        ids = [e.id for e in self.linears]
        want = [lid.key for lid in linear_ids(a.num_layers)]
        if ids != want:
            raise ConfigError(f"manifest linears are not the {len(want)} groups in canonical order")

        def tensor(name: str, dtype: str | tuple[str, ...], shape: tuple[int, ...]) -> None:
            t = self.tensors.get(name)
            if t is None:
                raise ConfigError(f"manifest references missing tensor {name!r}")
            dts = (dtype,) if isinstance(dtype, str) else dtype
            if t.dtype not in dts or tuple(t.shape) != shape:
                raise ConfigError(
                    f"tensor {name!r}: expected {'|'.join(dts)}{list(shape)}, got {t.dtype}{list(t.shape)}"
                )

        for e in self.linears:
            m, n, parts = expected_linear_shape(a, e.linear_id)
            k = f"linears.{e.id}"
            names = (e.weight, e.scale, e.bias)
            if names != (f"{k}.weight", f"{k}.scale", f"{k}.bias" if e.bias is not None else None):
                raise ConfigError(f"linear {e.id}: non-canonical tensor names {[x for x in names if x]}")
            want_bias = a.arch == ModelArch.QWEN2 and e.linear_id.kind == LinearKind.QKV
            if (e.bias is not None) != want_bias:
                raise ConfigError(
                    f"linear {e.id}: {'missing' if want_bias else 'unexpected'} bias for {a.arch.value}"
                )
            if (e.out_features, e.in_features) != (m, n):
                raise ConfigError(f"linear {e.id}: shape ({e.out_features},{e.in_features}) != expected ({m},{n})")
            got_parts = [(p.name, p.rows) for p in e.parts]
            if got_parts != parts:
                raise ConfigError(f"linear {e.id}: parts {got_parts} != expected {parts}")
            row = 0
            for p in e.parts:
                if p.row_start != row:
                    raise ConfigError(f"linear {e.id}: part {p.name} starts at {p.row_start}, expected {row}")
                row += p.rows
            tensor(e.weight, "int8", (m, n))
            tensor(e.scale, "float32", (m,))
            if e.bias is not None:
                tensor(e.bias, "float32", (m,))
        floats = ("float32", "float16", "bfloat16")
        h = a.hidden_size
        tensor("embed_tokens", floats, (a.vocab_size, h))
        tensor("final_norm", floats, (h,))
        if not a.tie_word_embeddings:
            tensor("lm_head", floats, (a.vocab_size, h))
        for i in range(a.num_layers):
            tensor(f"layers.{i}.input_norm", floats, (h,))
            tensor(f"layers.{i}.post_attn_norm", floats, (h,))
        referenced = set(self.plain_tensor_names())
        for e in self.linears:
            referenced.update(t for t in (e.weight, e.scale, e.bias) if t is not None)
        extra = sorted(set(self.tensors) - referenced)
        if extra:
            raise ConfigError(f"manifest lists {len(extra)} unreferenced tensor(s): {extra[:3]}")
        files = [f.file for f in self.tokenizer]
        if len(set(files)) != len(files):
            raise ConfigError("duplicate tokenizer file entries in manifest")

    def to_json(self) -> str:
        """Pretty JSON for writing ``manifest.json``."""
        return json.dumps(_drop_unset_linear_fields(self.model_dump(mode="json")), indent=1, sort_keys=True)

    @classmethod
    def from_file(cls, path: str | Path) -> Manifest:
        """Read ``manifest.json`` (or a pack directory containing it).

        Raises:
            ConfigError: missing, unreadable or invalid manifest.
        """
        from pydantic import ValidationError

        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILE
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"no model pack manifest at {p}", details={"path": str(p)}) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"cannot read model pack manifest {p}: {exc}") from exc
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            raise ConfigError(
                f"invalid model pack manifest {p}: {exc.error_count()} error(s): {exc.errors()[0]['msg']}",
                details={"path": str(p)},
            ) from exc


#: Optional :class:`LinearEntry` fields added after format version 1 shipped. When unset they are not
#: written and not fingerprinted, so older manifests read, round-trip and fingerprint as before.
_OPTIONAL_LINEAR_FIELDS = ("row_norm_sq_max",)


def _drop_unset_linear_fields(d: dict[str, Any]) -> dict[str, Any]:
    for e in d.get("linears", ()):
        for f in _OPTIONAL_LINEAR_FIELDS:
            if f in e and e[f] is None:
                del e[f]
    return d
