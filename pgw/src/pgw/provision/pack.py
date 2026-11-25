"""The weight-free gateway pack: exactly what the masked gateway (masked LM head included) reads, nothing else.

What the gateway loads (determined from the engine and gateway code):

* ``manifest.json`` (architecture, linear specs and ``row_norm_sq_max``, per-tensor SHA-256; the fingerprint);
* every plain tensor (:meth:`pgw.modelpack.manifest.Manifest.plain_tensor_names`: ``embed_tokens``,
  ``final_norm``, the per-layer norms and, when untied, ``lm_head``): the engine's embeddings and norms
  (:class:`pgw.engine.model.TransformerModel`), and the source of the int8 head's scales and row norm on the
  gateway side of the masked head (:meth:`pgw.modelpack.masked_head.MaskedHeadPack.load_linear_meta`);
* every linear's ``scale`` (and ``bias`` for Qwen2 ``qkv``): :meth:`pgw.modelpack.ModelPack.load_linear_meta`, the
  only per-linear read of the masked backend (:class:`pgw.engine.backends.MaskedBackend`);
* the tokenizer files listed in the manifest.

Not the int8 linear weights (``linears.<id>.weight``): in a pack they share the ``linears-*`` shards with the
scales, so this pack stores the scales and biases in ``linear-meta.safetensors`` instead and :class:`GatewayPack`
maps the shard names onto it. All bytes are checked against the manifest's SHA-256 values, which the fingerprint
covers.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError, FingerprintMismatchError
from pgw.modelpack.loader import ModelPack, _check_scale_bias
from pgw.modelpack.manifest import MANIFEST_FILE, TOKENIZER_DIR, Manifest
from pgw.modelpack.st_io import _DTYPES, hash_tensor_ranges, read_safetensors_header

__all__ = [
    "LINEAR_META_FILE", "GatewayPack", "PackObject", "gateway_objects", "weight_objects", "write_safetensors_layout",
]

LINEAR_META_FILE = "linear-meta.safetensors"


@dataclass(frozen=True)
class PackObject:
    """One artifact the client needs, as the fingerprint-bound manifest describes it (``sha256``/``size`` are the
    manifest's, never the provider's)."""

    id: str  # "manifest.json" | "tensor/<name>" | "tokenizer/<file>" | "weight/<linear>"
    kind: str  # "manifest" | "tensor" | "tokenizer" | "weight"
    name: str
    size: int
    sha256: str | None  # None for manifest.json (checked by fingerprint instead)
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    file: str | None = None  # destination file in the gateway pack (tensors), relative to the pack root


def _tensor_obj(man: Manifest, name: str, file: str) -> PackObject:
    t = man.tensors[name]
    return PackObject(f"tensor/{name}", "tensor", name, t.nbytes, t.sha256, t.dtype, tuple(t.shape), file)


def gateway_objects(man: Manifest) -> list[PackObject]:
    """The objects of the weight-free gateway pack, derived from the manifest alone (the provider cannot add or
    drop any): plain tensors in their manifest files, scales/biases in :data:`LINEAR_META_FILE`, tokenizer files."""
    out: list[PackObject] = []
    for n in man.plain_tensor_names():
        out.append(_tensor_obj(man, n, man.tensors[n].file))
    for e in man.linears:
        for n in (e.scale, e.bias):
            if n is not None:
                out.append(_tensor_obj(man, n, LINEAR_META_FILE))
    for t in man.tokenizer:
        out.append(PackObject(f"tokenizer/{t.file}", "tokenizer", t.file, -1, t.sha256,
                              file=f"{TOKENIZER_DIR}/{t.file}"))
    return out


def weight_objects(man: Manifest) -> list[PackObject]:
    """The int8 weights (streamed for hint verification / local builds / factory check vectors; never cached)."""
    out = []
    for e in man.linears:
        t = man.tensors[e.weight]
        out.append(PackObject(f"weight/{e.id}", "weight", e.weight, t.nbytes, t.sha256, t.dtype, tuple(t.shape)))
    return out


def write_safetensors_layout(path: Path, objs: list[PackObject]) -> dict[str, tuple[int, int]]:
    """Create ``path`` with a safetensors header for ``objs`` (in order) and a zero-filled data section; return
    ``{object id: (begin, end)}`` absolute file offsets where each tensor's raw bytes go."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "writer": "pgw-provision"}}
    off = 0
    rel: dict[str, tuple[int, int]] = {}
    for o in objs:
        if o.dtype not in _DTYPES or o.shape is None:
            raise ConfigError(f"{o.id}: unsupported tensor dtype {o.dtype!r}")
        header[o.name] = {"dtype": _DTYPES[o.dtype][0], "shape": list(o.shape), "data_offsets": [off, off + o.size]}
        rel[o.id] = (off, off + o.size)
        off += o.size
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    base = 8 + len(hb)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.truncate(base + off)
    return {k: (base + b, base + e) for k, (b, e) in rel.items()}


class GatewayPack(ModelPack):
    """A :class:`ModelPack` over a weight-free gateway pack (see the module docstring).

    Same fingerprint, architecture, specs, plain tensors, tokenizer and ``load_linear_meta`` as the full pack;
    ``load_linear`` / ``iter_linears`` raise (no int8 weights here)."""

    @classmethod
    def open_gateway(cls, path: str | Path, *, expected_fingerprint: str | None = None,
                     verify: str = "full") -> GatewayPack:
        root = Path(path)
        man = Manifest.from_file(root / MANIFEST_FILE)
        man.verify_fingerprint(expected_fingerprint)
        man.check_consistency()
        pack = cls(root, man, ("plain", "tokenizer"))
        pack._check_present()
        if not (root / LINEAR_META_FILE).is_file():
            raise ConfigError(f"gateway pack {root} has no {LINEAR_META_FILE}")
        if verify == "full":
            pack.verify_files()
            pack.verify_linear_meta()
        return pack

    def _handle(self, file: str) -> Any:
        if file in self.manifest.linear_files() and file not in self.manifest.plain_files():
            file = LINEAR_META_FILE
        return super()._handle(file)

    def load_linear(self, lid: Any, device: Any = "cpu", *, check_hash: bool = False) -> Any:
        raise ConfigError(f"the gateway pack holds no int8 weights (linear {lid}); the masked gateway never "
                          "needs them (pgw.provision.client.stream_weights streams them when a caller does)")

    def verify_linear_meta(self) -> None:
        """Hash every scale/bias tensor in :data:`LINEAR_META_FILE` against the manifest and check their values."""
        entries, _ = read_safetensors_header(self.root / LINEAR_META_FILE)
        want = {n for e in self.manifest.linears for n in (e.scale, e.bias) if n is not None}
        if set(entries) != want:
            raise ConfigError(f"{LINEAR_META_FILE}: tensor set differs from the manifest's scales and biases")
        for n, he in entries.items():
            me = self.manifest.tensors[n]
            if he.dtype != me.dtype or he.shape != tuple(me.shape):
                raise ConfigError(f"{LINEAR_META_FILE}: {n} is {he.dtype}{list(he.shape)}, manifest says "
                                  f"{me.dtype}{list(me.shape)}")
        digests = hash_tensor_ranges(self.root / LINEAR_META_FILE, list(entries.values()))
        bad = [n for n in sorted(want) if digests[n] != self.manifest.tensors[n].sha256]
        if bad:
            raise FingerprintMismatchError(f"gateway pack {self.name}: {len(bad)} scale/bias tensor(s) do not match "
                                           f"the manifest: {bad[:5]}")
        for e in self.manifest.linears:
            sw = self.load_tensor(e.scale)
            b = self.load_tensor(e.bias) if e.bias is not None else None
            _check_scale_bias(e.id, sw, b)
