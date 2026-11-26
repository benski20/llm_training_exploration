"""Shard-wise, streaming, resumable packing (:func:`stream_pack_checkpoint`): the same pack, one checkpoint shard
on disk at a time.

:func:`pgw.modelpack.pack_checkpoint` needs the whole checkpoint on disk (``snapshot_download``), which is
15 / 29 / 65 / 145 GB of bfloat16 at 7B / 14B / 32B / 72B. This packer plans everything from the safetensors
*headers* (fetched with range requests from the hub, or read from local files), then walks the checkpoint's
``*.safetensors`` shards in order: fetch one shard, quantize and write (or only hash) every linear group and plain
tensor whose inputs are now all present, copy the few tensors still needed by a group that continues in a later
shard into a small spill file, delete the shard, next. Peak disk is one shard + the spill (at most the parts of
the groups that straddle the boundary) + the output; with ``out_dir=None`` (hash only, e.g. to derive a pin) no
output is written at all. Peak RAM is one shard's largest input tensors plus the int8 output of one group:
quantization runs over row blocks (per-row arithmetic, so the result is the same bits as the whole-matrix call).

The output is byte-identical to :func:`pgw.modelpack.pack_checkpoint` (same planned layout: shard names,
header bytes, tensor order and offsets; same arithmetic per tensor; same manifest and fingerprint; only
``created_at`` differs, as between any two runs). ``tests/modelpack/test_shardwise.py`` checks this at a tiny model,
Qwen2.5-0.5B and 1.5B, including the release pins.

Resumable: progress (each finished item's SHA-256 digests and ``row_norm_sq_max``, finished shards, spill files)
is recorded in ``<work_dir>/state.json`` after the item's bytes are fsynced, keyed by a digest of the whole plan
(source, headers, parameters). A rerun with the same ``work_dir`` continues where the last one stopped. The
partial output lives in ``<work_dir>/out`` and is renamed to ``out_dir`` only when complete; callers that need
integrity open the result with ``ModelPack.open(..., verify="full")`` against their pinned fingerprint, which
re-hashes every byte (a stale or edited work dir can therefore never pass as a good pack).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import resource
import shutil
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from pgw.core.errors import ConfigError, ShapeError
from pgw.core.fingerprint import sha256_file
from pgw.core.ring import quantize_weight_rows
from pgw.modelpack.arch import arch_config_from_hf
from pgw.modelpack.loader import row_norm_sq_max
from pgw.modelpack.manifest import (
    MANIFEST_FILE,
    PLAIN_FILE,
    TOKENIZER_DIR,
    ArchConfig,
    LinearEntry,
    Manifest,
    QuantConfig,
    SourceInfo,
    TensorEntry,
    TokenizerFile,
)
from pgw.modelpack.packer import (
    _HUB_PATTERNS,
    _ST_TORCH,
    _TOOL,
    PLAIN_DTYPES,
    TOKENIZER_FILES,
    PlainDType,
    _AtomicDir,
    _check_float,
    _copy_tokenizer,
    _default_name,
    _dt_name,
    _fsync_path,
    _linear_entry,
    _LinearPlan,
    _local_source_info,
    _plan_linears,
    _plan_plain,
    _read_json,
    _shard_name,
    _torch_dtype,
)
from pgw.modelpack.st_io import SafetensorsRandomWriter, TensorPlan, tensor_sha256

__all__ = ["HubShards", "LocalShards", "ShardSource", "StreamPackResult", "VirtualShards", "plan_summary",
           "stream_pack_checkpoint"]

log = logging.getLogger(__name__)

#: Rows per quantization block: about 64 M elements (512 MB of float64 for the largest block), whatever the width.
_BLOCK_ELEMS = 64 << 20
_STATE_VERSION = 1
_META_FILES = ("config.json", *TOKENIZER_FILES)


# ---- shard sources ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardTensor:
    """One checkpoint tensor as a shard header describes it (no data)."""

    name: str
    file: str
    st_dtype: str  # safetensors code, e.g. "BF16"
    shape: tuple[int, ...]


class ShardSource(Protocol):
    """Where the checkpoint's shards come from. Headers first (:meth:`tensors`), data one shard at a time."""

    #: Stable description of the source (part of the plan digest: a resumed run must use the same checkpoint).
    label: str

    def meta_dir(self) -> Path:
        """Directory holding ``config.json``, ``generation_config.json`` and the tokenizer files."""
        ...

    def tensors(self) -> dict[str, ShardTensor]:
        """Every tensor of the checkpoint, from the shard headers (consistent with the index, one shard each)."""
        ...

    def fetch(self, file: str) -> Path:
        """Make shard ``file`` available locally; returns its path (may block on a download)."""
        ...

    def release(self, file: str, path: Path) -> None:
        """The shard is no longer needed: delete it if it was downloaded for this run."""
        ...

    def source_info(self, revision: str | None, record_revision: Literal["resolved", "given"]) -> SourceInfo:
        """Provenance recorded in the manifest (exactly as :func:`pgw.modelpack.pack_checkpoint` records it)."""
        ...


def _parse_header_bytes(raw: bytes, file: str) -> dict[str, ShardTensor]:
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{file}: malformed safetensors header: {exc}") from exc
    if not isinstance(hdr, dict):
        raise ConfigError(f"{file}: malformed safetensors header")
    hdr.pop("__metadata__", None)
    out: dict[str, ShardTensor] = {}
    for k, e in hdr.items():
        try:
            out[k] = ShardTensor(k, file, str(e["dtype"]), tuple(int(d) for d in e["shape"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{file}: bad header entry for {k!r}") from exc
    return out


def read_shard_header(path: Path, file: str | None = None) -> dict[str, ShardTensor]:
    """Tensors of one local safetensors file, from its header."""
    name = file or path.name
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ConfigError(f"{name}: truncated safetensors file")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen > (256 << 20):
                raise ConfigError(f"{name}: invalid safetensors header length {hlen}")
            return _parse_header_bytes(f.read(hlen), name)
    except FileNotFoundError as exc:
        raise ConfigError(f"checkpoint shard {name} is missing") from exc


def _check_index(files: list[str], per_file: Mapping[str, Mapping[str, ShardTensor]],
                 weight_map: Mapping[str, str] | None) -> dict[str, ShardTensor]:
    """Same rules as :class:`pgw.modelpack.packer.CheckpointReader`: shard names stay in the checkpoint, every
    tensor is in exactly one shard, and the index (when present) says where each tensor is."""
    bad = [f for f in files if "/" in f or "\\" in f or f in (".", "..") or not f.endswith(".safetensors")]
    if bad:
        raise ConfigError(f"model.safetensors.index.json references shard files outside the checkpoint: {bad[:3]}")
    where: dict[str, ShardTensor] = {}
    for fn in files:
        for k, t in per_file[fn].items():
            if k in where:
                raise ConfigError(f"tensor {k} appears in several checkpoint shards")
            where[k] = t
    for k, fn in (weight_map or {}).items():
        if k not in where or where[k].file != fn:
            raise ConfigError(f"index says {k} is in {fn}, but it is not there")
    return where


def _read_index(meta: Path) -> tuple[list[str] | None, dict[str, str] | None]:
    index = meta / "model.safetensors.index.json"
    if not index.exists():
        return None, None
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weight_map, dict):
            raise TypeError("weight_map is not an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigError(f"cannot parse {index.name}: {type(exc).__name__}: {exc}") from exc
    return sorted({str(v) for v in weight_map.values()}), {str(k): str(v) for k, v in weight_map.items()}


class LocalShards:
    """The shards of a checkpoint directory on this host, read in place (never deleted).

    ``copy_to`` (tests, measurements): each shard is copied there when fetched and deleted when released, i.e. a
    download simulated with local files.
    """

    def __init__(self, ckpt_dir: str | Path, *, copy_to: str | Path | None = None, info: SourceInfo | None = None):
        self.dir = Path(ckpt_dir)
        self.copy_to = Path(copy_to) if copy_to is not None else None
        self._info = info
        self.label = f"local:{self.dir.resolve().name}"
        files, self._weight_map = _read_index(self.dir)
        if files is None:
            files = sorted(p.name for p in self.dir.glob("*.safetensors"))
        if not files:
            bins = list(self.dir.glob("*.bin")) + list(self.dir.glob("*.pt"))
            hint = " (only PyTorch .bin weights found; convert to safetensors first)" if bins else ""
            raise ConfigError(f"no *.safetensors weights in checkpoint {self.dir.name}{hint}")
        self.files = files

    def meta_dir(self) -> Path:
        return self.dir

    def tensors(self) -> dict[str, ShardTensor]:
        per = {}
        for fn in self.files:
            if "/" in fn or "\\" in fn or fn in (".", ".."):
                raise ConfigError(
                    f"model.safetensors.index.json references shard files outside the checkpoint: {[fn]}")
            per[fn] = read_shard_header(self.dir / fn, fn)
        return _check_index(self.files, per, self._weight_map)

    def fetch(self, file: str) -> Path:
        src = self.dir / file
        if self.copy_to is None:
            return src
        self.copy_to.mkdir(parents=True, exist_ok=True)
        dst = self.copy_to / file
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return dst

    def release(self, file: str, path: Path) -> None:
        if self.copy_to is not None and path.parent == self.copy_to:
            path.unlink(missing_ok=True)

    def source_info(self, revision: str | None, record_revision: Literal["resolved", "given"]) -> SourceInfo:
        return self._info if self._info is not None else _local_source_info(self.dir, revision)


class VirtualShards(LocalShards):
    """A local checkpoint presented as differently sharded (tests): ``assignment`` maps each tensor to a virtual
    shard name; every virtual shard reads the real file in place. Exercises groups and plain tensors split across
    shards without copying the checkpoint."""

    def __init__(self, ckpt_dir: str | Path, assignment: Mapping[str, str]):
        super().__init__(ckpt_dir)
        self.assignment = dict(assignment)
        self.label = f"virtual:{self.dir.resolve().name}:" + hashlib.sha256(
            json.dumps(sorted(self.assignment.items())).encode()).hexdigest()[:16]
        self._real: dict[str, str] = {}

    def tensors(self) -> dict[str, ShardTensor]:
        real = super().tensors()
        if set(real) != set(self.assignment):
            raise ConfigError("virtual shard assignment does not cover the checkpoint's tensors exactly")
        self._real = {k: t.file for k, t in real.items()}
        return {k: ShardTensor(k, self.assignment[k], t.st_dtype, t.shape) for k, t in real.items()}

    def fetch(self, file: str) -> Path:
        reals = {self._real[k] for k, v in self.assignment.items() if v == file}
        if len(reals) != 1:
            raise ConfigError("a virtual shard must map to one real file")
        return self.dir / reals.pop()

    def release(self, file: str, path: Path) -> None:
        return None


class HubShards:
    """A public Hugging Face checkpoint, one shard at a time: the revision is resolved to a commit once, the small
    files (config, tokenizer, index) are downloaded into ``<work>/meta``, the shard headers are read with range
    requests (``huggingface_hub.get_safetensors_metadata``), and each shard is downloaded into ``<work>/shards``
    when needed and deleted when released. A shard already downloaded (complete) by an interrupted run is reused.
    """

    def __init__(self, repo: str, revision: str | None, work: str | Path):
        from huggingface_hub import HfApi

        self.repo = repo
        self.work = Path(work)
        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=False)
        except Exception as exc:
            raise ConfigError(f"cannot resolve {repo}@{revision or 'main'}: {type(exc).__name__}: {exc}",
                              details={"source": repo}) from exc
        self.commit = str(info.sha)
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ConfigError(f"{repo}@{revision}: the hub returned no commit")
        self.revision = revision
        self.siblings = sorted(s.rfilename for s in (info.siblings or []))
        self.label = f"hub:{repo}@{self.commit}"
        self._meta = self.work / "meta"
        self._shards = self.work / "shards"

    def _download(self, fn: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        try:
            return Path(hf_hub_download(self.repo, fn, revision=self.commit, local_dir=str(local_dir)))
        except Exception as exc:
            raise ConfigError(f"cannot download {self.repo}/{fn}: {type(exc).__name__}: {exc}",
                              details={"source": self.repo}) from exc

    def meta_dir(self) -> Path:
        # the files pack_checkpoint's snapshot would have that the pack reads (config, tokenizer, index)
        want = [f for f in self.siblings if "/" not in f and (f in _META_FILES or f == "model.safetensors.index.json")]
        for fn in want:
            if not (self._meta / fn).is_file():
                self._download(fn, self._meta)
        return self._meta

    def tensors(self) -> dict[str, ShardTensor]:
        from huggingface_hub import get_safetensors_metadata

        self.meta_dir()
        files, weight_map = _read_index(self._meta)
        if files is None:
            files = sorted(f for f in self.siblings if "/" not in f and f.endswith(".safetensors"))
        if not files:
            raise ConfigError(f"no *.safetensors weights in checkpoint {self.repo}")
        try:
            md = get_safetensors_metadata(self.repo, revision=self.commit)
        except Exception as exc:
            raise ConfigError(f"cannot read the shard headers of {self.repo}: {type(exc).__name__}: {exc}") from exc
        per: dict[str, dict[str, ShardTensor]] = {}
        for fn in files:
            fm = md.files_metadata.get(fn)
            if fm is None:
                raise ConfigError(f"checkpoint shard {fn} listed in the index is missing")
            per[fn] = {k: ShardTensor(k, fn, str(t.dtype), tuple(int(d) for d in t.shape))
                       for k, t in fm.tensors.items()}
        return _check_index(files, per, weight_map)

    def fetch(self, file: str) -> Path:
        return self._download(file, self._shards)

    def release(self, file: str, path: Path) -> None:
        path.unlink(missing_ok=True)
        # huggingface_hub's per-file local_dir metadata
        for p in (self._shards / ".cache" / "huggingface" / "download").glob(f"{file}*"):
            p.unlink(missing_ok=True)

    def source_info(self, revision: str | None, record_revision: Literal["resolved", "given"]) -> SourceInfo:
        rev = revision if (record_revision == "given" and revision is not None) else self.commit
        return SourceInfo(repo=self.repo, revision=rev)


def _cached_snapshot(repo: str, revision: str | None) -> Path | None:
    """The complete checkpoint in the local Hugging Face cache, if it is there (no network)."""
    try:
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(repo, revision=revision, allow_patterns=_HUB_PATTERNS, local_files_only=True))
    except Exception:
        return None
    if not (p / "config.json").is_file():
        return None
    try:
        files, _ = _read_index(p)
    except ConfigError:
        return None
    if files is None:
        files = ["model.safetensors"]
    return p if all((p / f).is_file() for f in files) else None  # complete, else stream from the hub


# ---- the plan ----------------------------------------------------------------------------------
class _HeaderSource:
    """:class:`pgw.modelpack.packer.TensorSource` over headers only (planning)."""

    def __init__(self, tensors: Mapping[str, ShardTensor]):
        self._t = tensors

    def keys(self) -> set[str]:
        return set(self._t)

    def shape(self, name: str) -> tuple[int, ...]:
        try:
            return self._t[name].shape
        except KeyError as exc:
            raise ConfigError(f"checkpoint has no tensor {name!r}") from exc

    def dtype(self, name: str) -> torch.dtype:
        return _ST_TORCH.get(self._t[name].st_dtype.upper(), torch.float32)

    def get(self, name: str) -> torch.Tensor:  # pragma: no cover - planning never reads data
        raise AssertionError("header-only source")


@dataclass
class _Item:
    key: str  # "linear:<id>" or "plain:<name>"
    needs: tuple[str, ...]  # HF tensor names
    linear: _LinearPlan | None = None
    plain: TensorPlan | None = None
    hf: str | None = None
    out_file: str = ""


@dataclass
class StreamPackResult:
    """The manifest written (or, with ``out_dir=None``, the one that would be) and what the run used."""

    manifest: Manifest
    out_dir: Path | None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return str(self.manifest.fingerprint)


def _du(root: Path) -> int:
    """Allocated bytes under ``root`` (sparse files count what is written)."""
    total = 0
    if not root.exists():
        return 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, fn)).st_blocks * 512
    return total


def _rss_peak() -> int:
    """Peak resident set (``ru_maxrss``; counts the memory-mapped shard pages read, which the kernel can drop)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _rss_anon() -> int:
    """Current anonymous resident memory (``RssAnon``: the process's own allocations, no file pages); 0 if unknown."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class _State:
    """``state.json``: what is finished, under a plan digest. Saved atomically (fsync) after each step."""

    def __init__(self, path: Path, plan: str, resume: bool):
        self.path, self.plan = path, plan
        self.done: dict[str, dict[str, Any]] = {}
        self.shards_done: list[str] = []
        self.spill: dict[str, list[str]] = {}
        if resume and path.is_file():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if d.get("v") == _STATE_VERSION and d.get("plan") == plan:
                    self.done, self.shards_done, self.spill = d["done"], d["shards_done"], d["spill"]
                    log.info("resuming a shard-wise pack: %d items, %d shards done", len(self.done),
                             len(self.shards_done))
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("unreadable %s; starting over", path.name)

    @property
    def resumed(self) -> bool:
        return bool(self.done or self.shards_done)

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"v": _STATE_VERSION, "plan": self.plan, "done": self.done,
                                   "shards_done": self.shards_done, "spill": self.spill}), encoding="utf-8")
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.path)


# ---- the work ----------------------------------------------------------------------------------
def _quantize_blocks(ws: list[torch.Tensor], m: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize_weight_rows`` of the row-concatenation of ``ws``, over row blocks.

    ``quantize_weight_rows`` is per row (row max, per-element float64 division and round-half-even), so each row's
    bits depend on that row only: blocks of rows give exactly the whole-matrix result (and the concatenation of the
    float64 parts is the float64 of their concatenation)."""
    w8 = torch.empty((m, n), dtype=torch.int8)
    sw = torch.empty((m,), dtype=torch.float32)
    step = max(1, _BLOCK_ELEMS // max(1, n))
    row = 0
    for w in ws:
        r = int(w.shape[0])
        for r0 in range(0, r, step):
            q8, qs = quantize_weight_rows(w[r0 : r0 + step])
            w8[row + r0 : row + r0 + q8.shape[0]] = q8
            sw[row + r0 : row + r0 + q8.shape[0]] = qs
            del q8, qs
        row += r
    if row != m:
        raise ShapeError(f"group rows {row} != planned {m}")
    return w8, sw


class _Tensors:
    """Tensors available now: the open shard and the spill files."""

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()
        self._where: dict[str, Any] = {}

    def add(self, path: Path, names: set[str]) -> None:
        from safetensors import safe_open

        h = self._stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        have = set(h.keys())
        for k in names:
            if k not in have:
                raise ConfigError(f"{path.name}: tensor {k} is not in the shard its header listed")
            self._where[k] = h

    def has(self, name: str) -> bool:
        return name in self._where

    def get(self, name: str) -> torch.Tensor:
        t: torch.Tensor = self._where[name].get_tensor(name)
        return t

    def close(self) -> None:
        self._stack.close()
        self._where.clear()


def _check_fetched(path: Path, file: str, want: Mapping[str, ShardTensor]) -> None:
    got = read_shard_header(path, file)
    for k, t in want.items():
        g = got.get(k)
        if g is None or g.st_dtype != t.st_dtype or g.shape != t.shape:
            raise ConfigError(f"{file}: downloaded shard disagrees with its header for {k} "
                              f"({g.st_dtype if g else None}{list(g.shape) if g else ''} vs "
                              f"{t.st_dtype}{list(t.shape)})")


def _open_source(source: str | Path | ShardSource, revision: str | None, work: Path, *, use_hf_cache: bool,
                 record_revision: Literal["resolved", "given"]) -> ShardSource:
    if isinstance(source, str | Path) and Path(source).expanduser().is_dir():
        return LocalShards(Path(source).expanduser())
    if not isinstance(source, str | Path):
        return source
    s = str(source)
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", s):
        raise ConfigError(f"checkpoint {s!r} is neither a local directory nor a hub repo id", details={"source": s})
    cached = _cached_snapshot(s, revision) if use_hf_cache else None
    if cached is not None:  # recorded as HubShards would record it (the commit is the snapshot's name)
        commit = cached.name if re.fullmatch(r"[0-9a-f]{40}", cached.name) else revision
        rev = revision if (record_revision == "given" and revision is not None) else commit
        return LocalShards(cached, info=SourceInfo(repo=s, revision=rev))
    return HubShards(s, revision, work)


def stream_pack_checkpoint(
    source: str | Path | ShardSource,
    out_dir: str | Path | None,
    *,
    work_dir: str | Path,
    name: str | None = None,
    revision: str | None = None,
    plain_dtype: PlainDType = "source",
    shard_bytes: int = 2 << 30,
    overwrite: bool = False,
    license: str | None = None,
    progress: Callable[[str], None] | None = None,
    allow_unsupported_rope: bool = False,
    resume: bool = True,
    prefetch: int = 0,
    record_revision: Literal["resolved", "given"] = "resolved",
    use_hf_cache: bool = True,
) -> StreamPackResult:
    """Pack a checkpoint shard by shard (see the module docstring); same pack as :func:`pack_checkpoint`.

    Args:
        source: Hub repo id (``org/name``), local checkpoint directory, or a :class:`ShardSource`.
        out_dir: Pack directory to create, or ``None`` to only compute the manifest (fingerprint) without writing
            any tensor (``StreamPackResult.out_dir`` is then ``None``).
        work_dir: Scratch directory for the shard being processed, spill files, the state and the partial output.
            Reusing it resumes an interrupted run. Removed after a successful run.
        name, revision, plain_dtype, shard_bytes, overwrite, license, progress, allow_unsupported_rope: as for
            :func:`pgw.modelpack.pack_checkpoint`.
        resume: Continue from ``work_dir/state.json`` when its plan matches (default); ``False`` starts over.
        prefetch: Shards downloaded ahead while the current one is processed (0: one shard on disk at a time).
        record_revision: Hub source: record the resolved commit (``"resolved"``, as ``pack_checkpoint`` does) or the
            ``revision`` as given when one is given (``"given"``, as the provisioning path did).
        use_hf_cache: Hub source: when the whole checkpoint is already in the local Hugging Face cache, read it there
            in place (nothing downloaded or deleted).

    Returns:
        :class:`StreamPackResult` (manifest with fingerprint, output directory, stats: peak work-dir bytes, peak RSS,
        bytes fetched, seconds).
    """
    t_start = time.monotonic()
    if plain_dtype not in PLAIN_DTYPES:
        raise ConfigError(f"invalid plain_dtype {plain_dtype!r}")
    if shard_bytes <= 0:
        raise ConfigError("shard_bytes must be positive")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(work / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        src = _open_source(source, revision, work, use_hf_cache=use_hf_cache, record_revision=record_revision)
        res = _run(src, None if out_dir is None else Path(out_dir), work, name=name, revision=revision,
                   plain_dtype=plain_dtype, shard_bytes=shard_bytes, overwrite=overwrite, license=license,
                   progress=progress, allow_unsupported_rope=allow_unsupported_rope, resume=resume,
                   prefetch=max(0, int(prefetch)), record_revision=record_revision, t_start=t_start)
        for d in ("meta", "shards", "spill", "out"):
            shutil.rmtree(work / d, ignore_errors=True)
        with contextlib.suppress(OSError):
            (work / ".lock").unlink()
        with contextlib.suppress(OSError):
            work.rmdir()  # only when empty (the caller's other files stay)
        return res
    finally:
        os.close(lock_fd)


@dataclass
class _Prepared:
    meta: Path
    info: SourceInfo
    arch: ArchConfig
    model_name: str
    tensors: dict[str, ShardTensor]
    plans: list[_LinearPlan]
    n_shards: int
    pplans: list[TensorPlan]
    version: int
    items: list[_Item]
    out_plans: dict[str, list[TensorPlan]]
    tok_files: list[str]
    plan_digest: str


def _prepare(src: ShardSource, *, name: str | None, revision: str | None, plain_dtype: PlainDType, shard_bytes: int,
             license: str | None, allow_unsupported_rope: bool, record_revision: Literal["resolved", "given"],
             write: bool) -> _Prepared:
    """Everything decided from config and headers before any shard is fetched (all shape / dtype checks)."""
    meta = src.meta_dir()
    info = src.source_info(revision, record_revision)
    if license is not None:
        info = info.model_copy(update={"license": license})
    cfg = _read_json(meta / "config.json")
    if cfg is None:
        raise ConfigError(f"checkpoint {info.repo} has no config.json")
    arch: ArchConfig = arch_config_from_hf(cfg, _read_json(meta / "generation_config.json"),
                                           allow_unsupported_rope=allow_unsupported_rope)
    model_name = name or _default_name(info.repo)
    tensors = src.tensors()
    for t in tensors.values():
        if t.st_dtype.upper() not in _ST_TORCH:
            raise ConfigError(f"{t.name}: dtype {t.st_dtype} is not a float type (quantized checkpoints such as "
                              "GPTQ/AWQ are not supported)", details={"tensor": t.name})
    hs = _HeaderSource(tensors)
    plans = _plan_linears(hs, arch, shard_bytes)
    n_shards = plans[-1].shard + 1
    pplans, hf_plain = _plan_plain(hs, arch, plain_dtype)
    version = 2 if plain_dtype == "source" else 1
    items: list[_Item] = []
    for p in plans:
        needs = tuple(x for _, wn, bn, _ in p.parts for x in (wn, bn) if x is not None)
        items.append(_Item(f"linear:{p.lid.key}", needs, linear=p, out_file=_shard_name(p.shard, n_shards)))
    for tp in pplans:
        items.append(_Item(f"plain:{tp.name}", (hf_plain[tp.name],), plain=tp, hf=hf_plain[tp.name],
                           out_file=PLAIN_FILE))
    out_plans: dict[str, list[TensorPlan]] = {}
    for it in items:
        out_plans.setdefault(it.out_file, []).extend(it.linear.tensor_plans() if it.linear else [it.plain])  # type: ignore[list-item]
    tok_files = [fn for fn in TOKENIZER_FILES if (meta / fn).is_file()]
    plan_digest = hashlib.sha256(json.dumps({
        "source": src.label, "info": info.model_dump(mode="json"), "name": model_name,
        "arch": arch.model_dump(mode="json"), "plain_dtype": plain_dtype, "shard_bytes": shard_bytes,
        "write": write,
        "tensors": sorted((t.name, t.file, t.st_dtype, list(t.shape)) for t in tensors.values()),
        "tokenizer": {fn: sha256_file(meta / fn) for fn in tok_files},
    }, sort_keys=True).encode()).hexdigest()
    return _Prepared(meta, info, arch, model_name, tensors, plans, n_shards, pplans, version, items, out_plans,
                     tok_files, plan_digest)


def plan_summary(source: str | Path | ShardSource, *, work_dir: str | Path, revision: str | None = None,
                 plain_dtype: PlainDType = "source", shard_bytes: int = 2 << 30, prefetch: int = 0) -> dict[str, Any]:
    """What a shard-wise run of ``source`` needs, from the headers alone (no shard is downloaded): shard sizes, the
    spill after each shard, the peak scratch disk (``prefetch + 1`` shards + spill), the output size, and the
    largest group's input bytes (the RAM driver). For sizing a host (e.g. a Kaggle kernel) before a long run."""
    src = _open_source(source, revision, Path(work_dir), use_hf_cache=True, record_revision="resolved")
    pr = _prepare(src, name=None, revision=revision, plain_dtype=plain_dtype, shard_bytes=shard_bytes, license=None,
                  allow_unsupported_rope=True, record_revision="resolved", write=False)
    elem = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
    nbytes = {k: elem[t.st_dtype.upper()] * int(torch.Size(t.shape).numel()) for k, t in pr.tensors.items()}
    by_file: dict[str, list[str]] = {}
    for t in pr.tensors.values():
        by_file.setdefault(t.file, []).append(t.name)
    order = sorted(by_file)
    shard_bytes_of = {f: sum(nbytes[k] for k in by_file[f]) for f in order}
    pending = list(pr.items)
    seen: set[str] = set()
    rows = []
    peak = 0
    for i, f in enumerate(order):
        seen.update(by_file[f])
        pending = [it for it in pending if not all(k in seen for k in it.needs)]
        spill = sum(nbytes[k] for k in {k for it in pending for k in it.needs if k in seen})
        ahead = sum(shard_bytes_of[g] for g in order[i + 1 : i + 1 + prefetch])
        peak = max(peak, shard_bytes_of[f] + ahead + spill)
        rows.append({"file": f, "bytes": shard_bytes_of[f], "spill_after": spill})
    out_bytes = sum(sum(tp.nbytes for tp in tps) for tps in pr.out_plans.values())
    largest = max(pr.items, key=lambda it: sum(nbytes[k] for k in it.needs))
    return {"repo": pr.info.repo, "revision": pr.info.revision, "layers": pr.arch.num_layers,
            "tie_word_embeddings": pr.arch.tie_word_embeddings, "shards": rows,
            "checkpoint_bytes": sum(shard_bytes_of.values()), "max_shard_bytes": max(shard_bytes_of.values()),
            "max_spill_bytes": max(r["spill_after"] for r in rows), "peak_scratch_bytes": peak,
            "output_bytes": out_bytes, "largest_item": largest.key,
            "largest_item_input_bytes": sum(nbytes[k] for k in largest.needs)}


def _run(src: ShardSource, out_dir: Path | None, work: Path, *, name: str | None, revision: str | None,
         plain_dtype: PlainDType, shard_bytes: int, overwrite: bool, license: str | None,
         progress: Callable[[str], None] | None, allow_unsupported_rope: bool, resume: bool, prefetch: int,
         record_revision: Literal["resolved", "given"], t_start: float) -> StreamPackResult:
    if out_dir is not None and out_dir.exists() and not overwrite:
        if not (out_dir.is_dir() and not any(out_dir.iterdir())):
            raise ConfigError(f"output {out_dir} already exists (use overwrite=True)", details={"path": str(out_dir)})
    pr = _prepare(src, name=name, revision=revision, plain_dtype=plain_dtype, shard_bytes=shard_bytes,
                  license=license, allow_unsupported_rope=allow_unsupported_rope, record_revision=record_revision,
                  write=out_dir is not None)
    meta, info, arch, model_name, tensors = pr.meta, pr.info, pr.arch, pr.model_name, pr.tensors
    plans, n_shards, pplans, version, items = pr.plans, pr.n_shards, pr.pplans, pr.version, pr.items
    out_plans, tok_files, plan_digest = pr.out_plans, pr.tok_files, pr.plan_digest
    st_meta = {"format": "pt", "pgw": f"modelpack-v{version}"}

    state = _State(work / "state.json", plan_digest, resume)
    if not state.resumed:  # stale partial output or spill of another plan
        for d in ("out", "spill"):
            shutil.rmtree(work / d, ignore_errors=True)
    part_out = work / "out"
    writers: dict[str, SafetensorsRandomWriter] = {}
    if out_dir is not None:
        part_out.mkdir(parents=True, exist_ok=True)
        for fn, tps in out_plans.items():
            writers[fn] = SafetensorsRandomWriter(part_out / fn, tps, metadata=st_meta).open(reuse=state.resumed)
    spill_dir = work / "spill"
    spill_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"shards": 0, "fetched_bytes": 0, "fetch_s": 0.0, "compute_s": 0.0,
                             "peak_work_bytes": 0, "peak_rss_bytes": 0, "peak_rss_anon_bytes": 0,
                             "spill_peak_bytes": 0,
                             "resumed_items": len(state.done), "hash_only": out_dir is None}
    lock = threading.Lock()

    def sample() -> None:
        with lock:
            stats["peak_work_bytes"] = max(stats["peak_work_bytes"], _du(work))
            stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], _rss_peak())
            stats["peak_rss_anon_bytes"] = max(stats["peak_rss_anon_bytes"], _rss_anon())

    pending = [it for it in items if it.key not in state.done]
    by_file: dict[str, dict[str, ShardTensor]] = {}
    for t in tensors.values():
        by_file.setdefault(t.file, {})[t.name] = t
    shard_order = sorted(by_file)

    def needed(fn: str) -> bool:
        if fn in state.shards_done:
            return False
        names = set(by_file[fn])
        return any(names.intersection(it.needs) for it in pending)

    todo = [fn for fn in shard_order if needed(fn)]

    def fetch(fn: str) -> tuple[Path, float]:
        t0 = time.monotonic()
        p = src.fetch(fn)
        dt = time.monotonic() - t0
        sample()
        return p, dt

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pgw-shard-fetch") if prefetch else None
    futures: dict[str, Future[tuple[Path, float]]] = {}
    stop = threading.Event()

    def sampler() -> None:  # peaks between the explicit samples (inside a group's quantization, mid-download)
        while not stop.wait(0.5):
            sample()

    sampler_t = threading.Thread(target=sampler, name="pgw-shard-sampler", daemon=True)
    sampler_t.start()

    def schedule(i: int) -> None:
        if pool is None:
            return
        for fn in todo[i : i + 1 + prefetch]:
            if fn not in futures:
                futures[fn] = pool.submit(fetch, fn)

    try:
        for i, fn in enumerate(todo):
            schedule(i)
            path, dt = futures.pop(fn).result() if pool is not None else fetch(fn)
            stats["fetch_s"] += dt
            stats["shards"] += 1
            _check_fetched(path, fn, by_file[fn])
            stats["fetched_bytes"] += path.stat().st_size
            t0 = time.monotonic()
            avail = _Tensors()
            try:
                avail.add(path, set(by_file[fn]))
                for sfn, names in state.spill.items():
                    avail.add(spill_dir / sfn, set(names))
                still: list[_Item] = []
                for it in pending:
                    if all(avail.has(k) for k in it.needs):
                        _do_item(it, avail, writers, state)
                        state.save()
                        if progress is not None and it.linear is not None:
                            progress(it.linear.lid.key)
                        sample()
                    else:
                        still.append(it)
                pending = still
                # tensors of this shard that a later shard's item still needs: copy them aside
                keep = sorted({k for it in pending for k in it.needs if k in by_file[fn]})
                if keep:
                    from safetensors.torch import save_file

                    sfn = f"spill-{len(state.spill):05d}-{fn}"
                    save_file({k: avail.get(k).contiguous() for k in keep}, str(spill_dir / sfn))
                    _fsync_path(spill_dir / sfn)
                    state.spill[sfn] = keep
            finally:
                avail.close()
            # spill files whose tensors no pending item needs any more
            live = {k for it in pending for k in it.needs}
            for sfn in [s for s, names in state.spill.items() if not live.intersection(names)]:
                del state.spill[sfn]
            state.shards_done.append(fn)
            state.save()
            sample()
            with lock:
                stats["spill_peak_bytes"] = max(stats["spill_peak_bytes"], _du(spill_dir))
            for p in spill_dir.iterdir():
                if p.name not in state.spill:
                    p.unlink(missing_ok=True)
            src.release(fn, path)
            stats["compute_s"] += time.monotonic() - t0
            log.info("shard %s done (%d/%d); %d items left", fn, i + 1, len(todo), len(pending))
    finally:
        stop.set()
        sampler_t.join()
        if pool is not None:
            for f in futures.values():
                f.cancel()
            pool.shutdown(wait=True)
            for fn, f in futures.items():  # downloaded ahead, not processed (an error stopped the run)
                if f.done() and not f.cancelled() and f.exception() is None:
                    with contextlib.suppress(Exception):
                        src.release(fn, f.result()[0])
        for w in writers.values():
            w.close()
    if pending:
        raise ConfigError(f"shard-wise pack incomplete: {len(pending)} items never had all inputs "
                          f"(e.g. {pending[0].key})")

    # ---- manifest, in the order pack_checkpoint builds it
    tens: dict[str, TensorEntry] = {}
    entries: list[LinearEntry] = []
    for p in plans:
        d = state.done[f"linear:{p.lid.key}"]
        for tp in p.tensor_plans():
            tens[tp.name] = TensorEntry(file=_shard_name(p.shard, n_shards), dtype=tp.dtype,  # type: ignore[arg-type]
                                        shape=tp.shape, sha256=d["sha256"][tp.name])
        entries.append(_linear_entry(p, int(d["norm"])))
    for tp in pplans:
        d = state.done[f"plain:{tp.name}"]
        tens[tp.name] = TensorEntry(file=PLAIN_FILE, dtype=tp.dtype, shape=tp.shape,  # type: ignore[arg-type]
                                    sha256=d["sha256"][tp.name])
    if out_dir is not None:
        shutil.rmtree(part_out / TOKENIZER_DIR, ignore_errors=True)
        tok = _copy_tokenizer(meta, part_out)
    else:
        tok = tuple(TokenizerFile(file=fn, sha256=sha256_file(meta / fn)) for fn in tok_files)
    manifest = Manifest(
        version=version, name=model_name, source=info, arch=arch, quant=QuantConfig(plain_dtype=plain_dtype),
        linears=tuple(entries), tensors=tens, tokenizer=tok,
        created_at=_dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(), tool=_TOOL,
    ).with_fingerprint()
    manifest.check_consistency()
    final: Path | None = None
    if out_dir is not None:
        mp = part_out / MANIFEST_FILE
        mp.write_text(manifest.to_json() + "\n", encoding="utf-8")
        _fsync_path(mp)
        if (part_out / TOKENIZER_DIR).is_dir():
            _fsync_path(part_out / TOKENIZER_DIR)
        _fsync_path(part_out)
        sample()
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        ad = _AtomicDir(out_dir, overwrite)  # pack_checkpoint's commit: temp sibling, rename into place
        try:
            shutil.rmtree(ad.tmp)
            try:
                os.replace(part_out, ad.tmp)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copytree(part_out, ad.tmp)  # work dir on another filesystem
                shutil.rmtree(part_out, ignore_errors=True)
            ad.commit()
        except BaseException:
            ad.abort()
            raise
        final = out_dir
    stats["seconds"] = time.monotonic() - t_start
    stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], _rss_peak())
    shutil.rmtree(work / "spill", ignore_errors=True)
    shutil.rmtree(work / "shards", ignore_errors=True)
    with contextlib.suppress(OSError):
        (work / "state.json").unlink()
    log.info("shard-wise pack of %s: fingerprint %s, %d shards, peak work dir %.2f GB, peak RSS %.2f GB, %.0f s",
             model_name, (manifest.fingerprint or "")[:12], stats["shards"], stats["peak_work_bytes"] / 1e9,
             stats["peak_rss_bytes"] / 1e9, stats["seconds"])
    return StreamPackResult(manifest=manifest, out_dir=final, stats=stats)


def _do_item(it: _Item, avail: _Tensors, writers: Mapping[str, SafetensorsRandomWriter], state: _State) -> None:
    """Compute one item's tensors exactly as ``pgw.modelpack.packer._write_pack`` does, write (or hash) them."""
    out: list[tuple[str, torch.Tensor]] = []
    norm: int | None = None
    if it.linear is not None:
        p = it.linear
        ws = [_check_float(wn, avail.get(wn)) for _, wn, _, _ in p.parts]
        w8, sw = _quantize_blocks(ws, p.m, p.n)
        del ws
        norm = row_norm_sq_max(w8)
        bias = (torch.cat([_check_float(bn, avail.get(bn)).to(torch.float32) for _, _, bn, _ in p.parts
                           if bn is not None]) if p.has_bias else None)
        tps = p.tensor_plans()
        for tp, t in zip(tps, (w8, sw, bias) if bias is not None else (w8, sw), strict=True):
            if _dt_name(t) != tp.dtype or tuple(t.shape) != tp.shape:
                raise ShapeError(f"{tp.name}: planned {tp.dtype}{list(tp.shape)}, got {_dt_name(t)}{list(t.shape)}")
            out.append((tp.name, t))
    else:
        tp, hf = it.plain, it.hf
        assert tp is not None and hf is not None
        t = _check_float(hf, avail.get(hf)).to(_torch_dtype(tp.dtype))
        if not bool(torch.isfinite(t).all()):
            raise ConfigError(f"{hf}: values overflow plain_dtype {tp.dtype}", details={"tensor": hf})
        out.append((tp.name, t))
    w = writers.get(it.out_file)
    digests = {n: (w.write(n, t) if w is not None else tensor_sha256(t)) for n, t in out}
    if w is not None:
        w.sync()  # the bytes are durable before the state says the item is done
    rec: dict[str, Any] = {"sha256": digests}
    if norm is not None:
        rec["norm"] = norm
    state.done[it.key] = rec


def iter_shards(src: ShardSource) -> Iterator[str]:  # pragma: no cover - convenience for scripts
    """Shard file names of ``src`` in processing order."""
    yield from sorted({t.file for t in src.tensors().values()})
