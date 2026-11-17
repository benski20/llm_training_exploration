"""Building model packs from Hugging Face checkpoints (``pgw pack``).

The packer streams: tensor shapes are read from the checkpoint headers first, the whole pack
layout (shards, offsets) is planned, and then each linear group is loaded, concatenated, quantized
and written before the next one is touched. Peak memory is a few copies of the largest linear group
(float64 during quantization), independent of model size, so a 70B checkpoint packs on a small host.

Output is written into a temporary sibling directory and renamed into place atomically, so a
crashed or interrupted run never leaves a half-written pack under the final name.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from pgw.core.errors import ConfigError, ShapeError
from pgw.core.fingerprint import sha256_file
from pgw.core.ring import quantize_weight_rows
from pgw.core.types import LinearId, linear_ids
from pgw.modelpack.arch import arch_config_from_hf, expected_linear_shape, hf_linear_parts, hf_plain_tensor_names
from pgw.modelpack.manifest import (
    MANIFEST_FILE,
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
from pgw.modelpack.st_io import SafetensorsStreamWriter, TensorPlan

__all__ = ["TOKENIZER_FILES", "CheckpointReader", "pack_checkpoint", "pack_state_dict"]

log = logging.getLogger(__name__)

PlainDType = Literal["float32", "float16", "bfloat16"]

#: Files copied from the checkpoint into ``<pack>/tokenizer`` when present.
TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)
_HUB_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
_PLAIN_FLOAT_NAMES = ("embed_tokens", "lm_head")  # stored in plain_dtype; norms always float32
_TOOL = "pgw-modelpack/1"


# ---- tensor sources --------------------------------------------------------------------------
class TensorSource(Protocol):
    """Read-only access to HF-named checkpoint tensors."""

    def keys(self) -> set[str]:
        """All tensor names."""
        ...

    def shape(self, name: str) -> tuple[int, ...]:
        """Shape without loading the data."""
        ...

    def get(self, name: str) -> torch.Tensor:
        """Load one tensor (CPU)."""
        ...


class _StateDictSource:
    def __init__(self, sd: Mapping[str, torch.Tensor]):
        self._sd = sd

    def keys(self) -> set[str]:
        return set(self._sd.keys())

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._sd[name].shape)

    def get(self, name: str) -> torch.Tensor:
        return self._sd[name].detach().to("cpu")


class CheckpointReader:
    """Lazy, memory-mapped reader over the ``*.safetensors`` files of an HF checkpoint directory.

    Supports single-file checkpoints and sharded ones (``model.safetensors.index.json``).
    """

    def __init__(self, ckpt_dir: str | Path):
        from safetensors import safe_open

        self.dir = Path(ckpt_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(p.name for p in self.dir.glob("*.safetensors"))
            weight_map = {}
        if not files:
            bins = list(self.dir.glob("*.bin")) + list(self.dir.glob("*.pt"))
            hint = " (only PyTorch .bin weights found; convert to safetensors first)" if bins else ""
            raise ConfigError(f"no *.safetensors weights in checkpoint {self.dir.name}{hint}")
        self._stack = ExitStack()
        self._handles: dict[str, Any] = {}
        self._where: dict[str, str] = {}
        for fn in files:
            path = self.dir / fn
            if not path.exists():
                raise ConfigError(f"checkpoint shard {fn} listed in the index is missing")
            h = self._stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            self._handles[fn] = h
            for k in h.keys():
                if k in self._where:
                    raise ConfigError(f"tensor {k} appears in several checkpoint shards")
                self._where[k] = fn
        for k, fn in weight_map.items():
            if self._where.get(k) != fn:
                raise ConfigError(f"index says {k} is in {fn}, but it is not there")

    def keys(self) -> set[str]:
        """All tensor names."""
        return set(self._where)

    def shape(self, name: str) -> tuple[int, ...]:
        """Shape from the header (no data read)."""
        return tuple(self._handles[self._where[name]].get_slice(name).get_shape())

    def get(self, name: str) -> torch.Tensor:
        """Load one tensor (memory-mapped, CPU)."""
        try:
            fn = self._where[name]
        except KeyError as exc:
            raise ConfigError(f"checkpoint has no tensor {name!r}") from exc
        t: torch.Tensor = self._handles[fn].get_tensor(name)
        return t

    def close(self) -> None:
        """Release file handles / mmaps."""
        self._stack.close()

    def __enter__(self) -> CheckpointReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---- planning --------------------------------------------------------------------------------
@dataclass
class _LinearPlan:
    lid: LinearId
    m: int
    n: int
    parts: list[tuple[str, str, str | None, int]]  # (part name, hf weight, hf bias, rows)
    has_bias: bool
    shard: int = 0

    @property
    def nbytes(self) -> int:
        return self.m * self.n + 4 * self.m * (2 if self.has_bias else 1)

    def tensor_plans(self) -> list[TensorPlan]:
        k = f"linears.{self.lid.key}"
        out = [TensorPlan(f"{k}.weight", "int8", (self.m, self.n)), TensorPlan(f"{k}.scale", "float32", (self.m,))]
        if self.has_bias:
            out.append(TensorPlan(f"{k}.bias", "float32", (self.m,)))
        return out


def _plan_linears(src: TensorSource, arch: ArchConfig, shard_bytes: int) -> list[_LinearPlan]:
    keys = src.keys()
    plans: list[_LinearPlan] = []
    for lid in linear_ids(arch.num_layers):
        m_exp, n_exp, parts_exp = expected_linear_shape(arch, lid)
        parts: list[tuple[str, str, str | None, int]] = []
        for (pname, wname, bname), (_, rows_exp) in zip(
            hf_linear_parts(arch.arch, lid.layer, lid.kind), parts_exp, strict=True
        ):
            if wname not in keys:
                raise ConfigError(f"checkpoint is missing {wname}", details={"tensor": wname})
            shp = src.shape(wname)
            if shp != (rows_exp, n_exp):
                raise ShapeError(f"{wname}: shape {list(shp)} != expected {[rows_exp, n_exp]} from config.json")
            if bname is not None:
                if bname not in keys:
                    raise ConfigError(f"checkpoint is missing {bname}", details={"tensor": bname})
                if src.shape(bname) != (rows_exp,):
                    raise ShapeError(f"{bname}: shape {list(src.shape(bname))} != expected [{rows_exp}]")
            else:
                unexpected = wname[: -len(".weight")] + ".bias"
                if unexpected in keys:
                    raise ConfigError(f"checkpoint has unexpected bias {unexpected} for {arch.arch.value}")
            parts.append((pname, wname, bname, rows_exp))
        plans.append(_LinearPlan(lid, m_exp, n_exp, parts, any(p[2] is not None for p in parts)))
    shard, used = 0, 0
    for p in plans:
        if used > 0 and used + p.nbytes > shard_bytes:
            shard, used = shard + 1, 0
        p.shard = shard
        used += p.nbytes
    return plans


def _shard_name(i: int, total: int) -> str:
    return f"linears-{i:05d}-of-{total:05d}.safetensors"


# ---- tokenizer -------------------------------------------------------------------------------
def _copy_tokenizer(tok_src: Path | None, out: Path) -> tuple[TokenizerFile, ...]:
    if tok_src is None:
        return ()
    dst = out / TOKENIZER_DIR
    entries: list[TokenizerFile] = []
    for fn in TOKENIZER_FILES:
        sp = tok_src / fn
        if sp.is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sp, dst / fn)  # follows HF-cache symlinks
            entries.append(TokenizerFile(file=fn, sha256=sha256_file(dst / fn)))
    if not any(e.file in ("tokenizer.json", "tokenizer.model", "vocab.json") for e in entries):
        log.warning("no tokenizer files found in %s; the gateway will need a tokenizer from elsewhere", tok_src.name)
    return tuple(entries)


# ---- core writer ------------------------------------------------------------------------------
def _write_pack(
    src: TensorSource,
    arch: ArchConfig,
    out: Path,
    *,
    name: str,
    source: SourceInfo,
    tokenizer_dir: Path | None,
    plain_dtype: PlainDType,
    shard_bytes: int,
    progress: Callable[[str], None] | None = None,
) -> Manifest:
    """Write a complete pack into the (empty, existing) directory ``out``."""
    t0 = time.monotonic()
    if shard_bytes <= 0:
        raise ConfigError("shard_bytes must be positive")
    plans = _plan_linears(src, arch, shard_bytes)
    n_shards = plans[-1].shard + 1
    tensors: dict[str, TensorEntry] = {}
    entries: list[LinearEntry] = []
    meta = {"format": "pt", "pgw": "modelpack-v1"}
    for s in range(n_shards):
        fn = _shard_name(s, n_shards)
        shard_plans = [p for p in plans if p.shard == s]
        tplans = [tp for p in shard_plans for tp in p.tensor_plans()]
        with SafetensorsStreamWriter(out / fn, tplans, metadata=meta) as w:
            for p in shard_plans:
                ws = [src.get(wn) for _, wn, _, _ in p.parts]
                w_full = torch.cat([x.to(torch.float64) for x in ws], dim=0) if len(ws) > 1 else ws[0]
                del ws
                w8, sw = quantize_weight_rows(w_full)
                del w_full
                bias = (
                    torch.cat([src.get(bn).to(torch.float32) for _, _, bn, _ in p.parts if bn is not None])
                    if p.has_bias
                    else None
                )
                for tname, t in zip(
                    (tp.name for tp in p.tensor_plans()), (w8, sw, bias) if bias is not None else (w8, sw), strict=True
                ):
                    digest = w.write(tname, t)
                    tensors[tname] = TensorEntry(file=fn, dtype=_dt_name(t), shape=tuple(t.shape), sha256=digest)
                k = f"linears.{p.lid.key}"
                row = 0
                lparts = []
                for pname, _, _, rows in p.parts:
                    lparts.append(LinearPart(name=pname, row_start=row, rows=rows))
                    row += rows
                entries.append(
                    LinearEntry(
                        id=p.lid.key,
                        out_features=p.m,
                        in_features=p.n,
                        weight=f"{k}.weight",
                        scale=f"{k}.scale",
                        bias=f"{k}.bias" if p.has_bias else None,
                        parts=tuple(lparts),
                    )
                )
                if progress:
                    progress(p.lid.key)
        log.info("wrote %s (%d linear groups)", fn, len(shard_plans))

    # plain tensors
    keys = src.keys()
    hf_plain = hf_plain_tensor_names(arch.num_layers, arch.tie_word_embeddings)
    if not arch.tie_word_embeddings and "lm_head.weight" not in keys:
        raise ConfigError("tie_word_embeddings is false but the checkpoint has no lm_head.weight")
    pplans: list[TensorPlan] = []
    for pname, hf in hf_plain.items():
        if hf not in keys:
            raise ConfigError(f"checkpoint is missing {hf}", details={"tensor": hf})
        dt = plain_dtype if pname in _PLAIN_FLOAT_NAMES else "float32"
        pplans.append(TensorPlan(pname, dt, src.shape(hf)))
    with SafetensorsStreamWriter(out / PLAIN_FILE, pplans, metadata=meta) as w:
        for tp in pplans:
            t = src.get(hf_plain[tp.name]).to(_torch_dtype(tp.dtype))
            digest = w.write(tp.name, t)
            tensors[tp.name] = TensorEntry(file=PLAIN_FILE, dtype=tp.dtype, shape=tuple(t.shape), sha256=digest)

    tok = _copy_tokenizer(tokenizer_dir, out)
    manifest = Manifest(
        name=name,
        source=source,
        arch=arch,
        quant=QuantConfig(plain_dtype=plain_dtype),
        linears=tuple(entries),
        tensors=tensors,
        tokenizer=tok,
        created_at=_dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
        tool=_TOOL,
    ).with_fingerprint()
    manifest.check_consistency()
    mp = out / MANIFEST_FILE
    mp.write_text(manifest.to_json() + "\n", encoding="utf-8")
    _fsync_path(mp)
    log.info(
        "packed %s: %d layers, %d linear shards, fingerprint %s (%.1fs)",
        name, arch.num_layers, n_shards, (manifest.fingerprint or "")[:12], time.monotonic() - t0,
    )
    return manifest


def _dt_name(t: torch.Tensor) -> Literal["int8", "float32", "float16", "bfloat16"]:
    return {torch.int8: "int8", torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}[
        t.dtype
    ]  # type: ignore[return-value]


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "int8": torch.int8}[name]


def _fsync_path(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _AtomicDir:
    """Create ``out`` atomically: build in a temp sibling, then rename into place."""

    def __init__(self, out: Path, overwrite: bool):
        self.out = out
        self.overwrite = overwrite
        if out.exists() and not overwrite:
            if out.is_dir() and not any(out.iterdir()):
                pass  # empty directory is fine to replace
            else:
                raise ConfigError(f"output {out} already exists (use overwrite=True)", details={"path": str(out)})
        out.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))

    def commit(self) -> None:
        old: Path | None = None
        if self.out.exists():
            old = self.out.parent / f".{self.out.name}.old-{os.getpid()}-{time.monotonic_ns()}"
            os.replace(self.out, old)
        os.replace(self.tmp, self.out)
        dfd = os.open(self.out.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)

    def abort(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def _write_atomic(out_dir: str | Path, overwrite: bool, fn: Callable[[Path], Manifest]) -> Manifest:
    ad = _AtomicDir(Path(out_dir), overwrite)
    try:
        m = fn(ad.tmp)
        ad.commit()
        return m
    except BaseException:
        ad.abort()
        raise


# ---- public API -------------------------------------------------------------------------------
def _default_name(source: str) -> str:
    base = source.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-") or "model"


def _local_source_info(p: Path, revision: str | None) -> SourceInfo:
    """Provenance of a local directory; recognizes HF cache snapshots
    (``.../models--org--name/snapshots/<commit>``) and records ``org/name`` + commit.
    Never records absolute paths."""
    rp = p.resolve()
    if rp.parent.name == "snapshots" and rp.parent.parent.name.startswith("models--"):
        repo = rp.parent.parent.name[len("models--"):].replace("--", "/")
        rev = revision or (rp.name if re.fullmatch(r"[0-9a-f]{40}", rp.name) else None)
        return SourceInfo(repo=repo, revision=rev)
    return SourceInfo(repo=rp.name, revision=revision)


def _resolve_source(source: str | Path, revision: str | None) -> tuple[Path, SourceInfo]:
    p = Path(source).expanduser()
    if p.is_dir():
        return p, _local_source_info(p, revision)
    s = str(source)
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", s):
        raise ConfigError(f"checkpoint {s!r} is neither a local directory nor a hub repo id", details={"source": s})
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency of the pack extra
        raise ConfigError("huggingface_hub is required to pack from the hub (pip install 'pgw[pack]')") from exc
    try:
        local = Path(snapshot_download(s, revision=revision, allow_patterns=_HUB_PATTERNS))
    except Exception as exc:
        raise ConfigError(f"cannot download {s}: {type(exc).__name__}: {exc}", details={"source": s}) from exc
    # snapshots/<commit sha> -> record the resolved commit
    rev = local.name if re.fullmatch(r"[0-9a-f]{40}", local.name) else revision
    return local, SourceInfo(repo=s, revision=rev)


def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot parse {p.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p.name} is not a JSON object")
    return data


def pack_checkpoint(
    source: str | Path,
    out_dir: str | Path,
    *,
    name: str | None = None,
    revision: str | None = None,
    plain_dtype: PlainDType = "float32",
    shard_bytes: int = 2 << 30,
    overwrite: bool = False,
    license: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Manifest:
    """Quantize an HF checkpoint (hub repo id or local directory) into a PGW model pack.

    Steps: read ``config.json`` -> :class:`ArchConfig` (qwen2/llama only); for each layer and
    linear kind concatenate the HF projections (order from :data:`pgw.modelpack.arch.HF_LINEAR_PARTS`),
    quantize with :func:`pgw.core.ring.quantize_weight_rows`, write sharded safetensors; copy plain
    tensors and tokenizer files; write ``manifest.json`` with per-tensor SHA-256 and the fingerprint.
    Streams tensors so packing a 70B model needs O(largest tensor) RAM. Writes to a temp dir and
    renames atomically.

    Args:
        source: Local checkpoint directory or HF hub repo id (``org/name``).
        out_dir: Pack directory to create.
        name: Served model name (default: derived from ``source``).
        revision: Hub revision (branch, tag or commit); recorded in the manifest.
        plain_dtype: Storage dtype of ``embed_tokens``/``lm_head`` (norms are always float32).
        shard_bytes: Target maximum size of a ``linears-*`` shard (a group is never split).
        overwrite: Replace an existing ``out_dir``.
        license: Optional license identifier recorded in the manifest.
        progress: Optional callback called with each linear id after it is written.

    Returns:
        The written manifest (with fingerprint).

    Raises:
        ConfigError: unsupported architecture, missing tensors or existing ``out_dir`` without ``overwrite``.
        ShapeError: checkpoint tensor shapes disagree with ``config.json``.
    """
    if plain_dtype not in ("float32", "float16", "bfloat16"):
        raise ConfigError(f"invalid plain_dtype {plain_dtype!r}")
    ckpt, info = _resolve_source(source, revision)
    if license is not None:
        info = info.model_copy(update={"license": license})
    cfg = _read_json(ckpt / "config.json")
    if cfg is None:
        raise ConfigError(f"checkpoint {ckpt.name} has no config.json")
    arch = arch_config_from_hf(cfg, _read_json(ckpt / "generation_config.json"))
    model_name = name or _default_name(info.repo)
    log.info("packing %s (%s, %d layers) -> %s", info.repo, arch.arch.value, arch.num_layers, Path(out_dir).name)
    with CheckpointReader(ckpt) as reader:
        return _write_atomic(
            out_dir,
            overwrite,
            lambda tmp: _write_pack(
                reader, arch, tmp, name=model_name, source=info, tokenizer_dir=ckpt,
                plain_dtype=plain_dtype, shard_bytes=shard_bytes, progress=progress,
            ),
        )


def pack_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    arch: ArchConfig,
    out_dir: str | Path,
    *,
    name: str,
    source: SourceInfo,
    tokenizer_dir: str | Path | None = None,
    plain_dtype: PlainDType = "float32",
    shard_bytes: int = 2 << 30,
    overwrite: bool = False,
) -> Manifest:
    """Same as :func:`pack_checkpoint` but from an in-memory HF-named state dict (used by tests)."""
    src = _StateDictSource(state_dict)
    tok = Path(tokenizer_dir) if tokenizer_dir is not None else None
    return _write_atomic(
        out_dir,
        overwrite,
        lambda tmp: _write_pack(
            src, arch, tmp, name=name, source=source, tokenizer_dir=tok, plain_dtype=plain_dtype,
            shard_bytes=shard_bytes,
        ),
    )

