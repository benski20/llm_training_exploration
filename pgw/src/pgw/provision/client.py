"""Client provisioning: the gateway's non-secret artifacts and a ready pks-v3 hint store, from the provider.

``provision(model, session=..., cache_dir=...) -> ProvisionedModel`` (``docs/sdk/CONTRACT.md``, appendix A3):

1. **Artifacts.** ``manifest.json`` is fetched first and checked against the pinned fingerprint; the list of
   objects the gateway needs is derived from it alone (:func:`pgw.provision.pack.gateway_objects`), and every
   object's bytes must hash to the manifest's SHA-256. Objects are fetched in chunks (provider-listed per-chunk
   SHA-256: early reject and resume), written in place into a partial pack directory (``pack.partial``, chunk state
   in ``state.json``) and the directory is renamed to ``pack`` only after a full verification. ``source="hf"``
   instead downloads the public checkpoint, packs it locally and requires the same fingerprint.
2. **Hint** (``hint="provider"``, the SDK default): the client draws ``a_seed`` and ``ks_seed`` (CSPRNG), sends
   only ``a_seed``, and receives ``H = W8 A mod 2^56`` per linear. Each linear's ``H`` is accepted only if the
   2-adic Freivalds check of :mod:`pgw.provision.hintcheck` passes against ``W8`` streamed once from the provider
   (or the public source) and checked against the fingerprint (soundness error ``<= 2^-41`` against any ``H`` the
   provider picks); the switch and key switch are computed locally and the store is written by the same code as a
   local build (:meth:`pgw.hemask.pks_backend.PKSHintStore.write`): for the same seeds the hint files are
   bit-identical to a local build's. ``hint="local"`` computes ``H`` itself (tens of minutes at 7B).
   Argument and costs: ``research/reviews/pks-v3-provider-hint.md``.

Concurrency: one ``flock`` per model directory serialises ``provision`` calls; the hint store keeps its own lock
(a rebuild never replaces seeds under a running factory). Nothing secret is written or logged here: the seeds are
public, the hint key that MACs the store's metadata is created 0600 and never leaves the cache directory.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pgw.core.errors import ConfigError, FingerprintMismatchError, IntegrityError, ModelNotFoundError
from pgw.core.fingerprint import validate_fingerprint
from pgw.modelpack.manifest import MANIFEST_FILE, TOKENIZER_DIR, Manifest
from pgw.provision.pack import (
    LINEAR_META_FILE,
    GatewayPack,
    PackObject,
    _tensor_obj,
    gateway_objects,
    weight_objects,
    write_safetensors_layout,
)

__all__ = ["ProvisionedModel", "find_prepared_hint", "provision", "stream_weights"]

log = logging.getLogger(__name__)

HintMode = Literal["provider", "local", "none"]
Progress = Callable[[dict[str, Any]], None] | None
_STATE = "state.json"


# ------------------------------------------------------------------------------------------------
# Result
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvisionedModel:
    """Everything the in-process gateway needs that is not secret (fields: ``docs/sdk/CONTRACT.md``, A3)."""

    name: str
    fingerprint: str
    factory_fingerprint: str
    pack_dir: Path
    tokenizer_dir: Path
    manifest: Manifest
    hint_root: Path | None
    hint_store: Any
    hint_key_file: Path | None
    hint_source: str | None
    masked_head: bool
    sizes: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    #: full pack (int8 weights included) for the background mask factory (``factory_pack=True``), else ``None``
    factory_pack_dir: Path | None = None

    def open_pack(self, verify: Literal["full", "fingerprint"] = "full") -> GatewayPack:
        """The weight-free gateway pack (``verify="full"`` re-hashes every byte against the manifest)."""
        return GatewayPack.open_gateway(self.pack_dir, expected_fingerprint=self.fingerprint, verify=verify)

    def close(self) -> None:
        if self.hint_store is not None:
            self.hint_store.close()


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

def _emit(progress: Progress, stage: str, done: int = 0, total: int = 0, unit: str = "bytes", item: str = "") -> None:
    if progress is not None:
        with contextlib.suppress(Exception):  # a progress callback never breaks provisioning
            progress({"stage": stage, "done": int(done), "total": int(total), "unit": unit, "item": item})


@contextlib.contextmanager
def _flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _is_fp(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _hint_key(cache_dir: Path) -> tuple[Path, bytes]:
    from pgw.maskstore.keys import _read_key_file, write_key_file

    p = cache_dir / "keys" / "pks-hint.key"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(ConfigError):  # a concurrent creator won the race
            write_key_file(p)
    return p, _read_key_file(p)


class _Remote:
    """The provider's provisioning manifest for one fingerprint and chunk downloads with hash checks."""

    def __init__(self, session: Any, fp: str):
        self.session = session
        self.fp = fp
        doc = session.get_json(f"/{fp}/manifest")
        if not isinstance(doc, dict) or doc.get("format") != "pgw-provision" or doc.get("fingerprint") != fp:
            raise IntegrityError("the provider's provisioning manifest is malformed or for another model")
        self.chunk_bytes = int(doc["chunk_bytes"])
        if not 1 <= self.chunk_bytes <= 1 << 30:
            raise IntegrityError("bad provisioning chunk size")
        self.objects: dict[str, dict[str, Any]] = {o["id"]: o for o in doc["objects"]}
        self.bytes = 0

    def entry(self, obj: PackObject) -> dict[str, Any]:
        e = self.objects.get(obj.id)
        if e is None:
            raise ConfigError(f"the provider does not serve {obj.id} for this model")
        if obj.sha256 is not None and e.get("sha256") != obj.sha256:
            raise IntegrityError(f"{obj.id}: the provider lists a different SHA-256 than the pinned manifest")
        size = obj.size if obj.size >= 0 else int(e["size"])
        if int(e["size"]) != size or len(e["chunks"]) != max(1, -(-size // self.chunk_bytes)):
            raise IntegrityError(f"{obj.id}: size or chunk list does not match the pinned manifest")
        return e

    def chunk(self, oid: str, i: int, want: str) -> bytes:
        data = self.session.get_bytes(f"/{self.fp}/object", id=oid, chunk=i)
        if hashlib.sha256(data).hexdigest() != want:
            raise IntegrityError(f"{oid}: chunk {i} does not match its SHA-256 (tampered or corrupted)")
        self.bytes += len(data)
        return data

    def object_bytes(self, obj: PackObject) -> bytes:
        """A whole object in memory (manifest, weights), checked chunk by chunk and as a whole."""
        e = self.entry(obj)
        buf = bytearray()
        for i, want in enumerate(e["chunks"]):
            buf += self.chunk(obj.id, i, want)
        if len(buf) != int(e["size"]):
            raise IntegrityError(f"{obj.id}: wrong size")
        if obj.sha256 is not None and hashlib.sha256(buf).hexdigest() != obj.sha256:
            raise IntegrityError(f"{obj.id}: bytes do not match the pinned manifest")
        return bytes(buf)


def _fetch_manifest(remote: _Remote, fp: str) -> tuple[Manifest, bytes]:
    """``manifest.json`` (its exact bytes too): parsed, its content fingerprint equal to the pinned one."""
    mobj = PackObject("manifest.json", "manifest", MANIFEST_FILE, -1, None)
    raw = remote.object_bytes(mobj)
    try:
        man = Manifest.model_validate_json(raw)
    except Exception as exc:
        raise IntegrityError("the provider's manifest.json is not a valid pack manifest") from exc
    man.verify_fingerprint(fp)  # content == fingerprint == the pinned one
    man.check_consistency()
    return man, raw


# ------------------------------------------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------------------------------------------

def _install_from_provider(remote: _Remote, man: Manifest, raw_manifest: bytes, mdir: Path,
                           progress: Progress) -> Path:
    """Download the gateway objects into ``mdir/pack.partial`` (resumable) and rename to ``mdir/pack``."""
    part = mdir / "pack.partial"
    objs = gateway_objects(man)
    state_p = part / _STATE
    state: dict[str, Any] | None = None
    if state_p.is_file():
        with contextlib.suppress(ValueError, OSError):
            state = json.loads(state_p.read_text())
        if not state or state.get("fingerprint") != man.fingerprint or state.get("chunk_bytes") != remote.chunk_bytes:
            state = None
    if state is None:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True, mode=0o700)
        (part / MANIFEST_FILE).write_bytes(raw_manifest)
        offsets: dict[str, list[int]] = {}
        by_file: dict[str, list[PackObject]] = {}
        for o in objs:
            if o.kind == "tensor":
                by_file.setdefault(str(o.file), []).append(o)
        for file, fobjs in by_file.items():
            for k, (b, e) in write_safetensors_layout(part / file, fobjs).items():
                offsets[k] = [b, e]
        (part / TOKENIZER_DIR).mkdir(exist_ok=True)
        state = {"fingerprint": man.fingerprint, "chunk_bytes": remote.chunk_bytes, "offsets": offsets, "done": {}}
        _write_json_atomic(state_p, state)
    entries = {o.id: remote.entry(o) for o in objs}
    total = sum(int(e["size"]) for e in entries.values())
    done_b = 0
    for o in objs:
        e = entries[o.id]
        size = int(e["size"])
        done = set(state["done"].get(o.id, []))
        if o.kind == "tokenizer":
            dest = part / TOKENIZER_DIR / o.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != size:
                with open(dest, "wb") as f:
                    f.truncate(size)
                done = set()
            base = 0
        else:
            dest = part / str(o.file)
            base = int(state["offsets"][o.id][0])
        fd = os.open(dest, os.O_RDWR)
        try:
            for i, want in enumerate(e["chunks"]):
                if i in done:
                    continue
                data = remote.chunk(o.id, i, want)
                os.pwrite(fd, data, base + i * remote.chunk_bytes)
                done.add(i)
                if len(done) % 16 == 0 or len(done) == len(e["chunks"]):
                    os.fsync(fd)
                    state["done"][o.id] = sorted(done)
                    _write_json_atomic(state_p, state)
                _emit(progress, "artifacts", done_b + min(size, (i + 1) * remote.chunk_bytes), total, item=o.id)
        finally:
            os.close(fd)
        done_b += size
    # whole-object verification against the pinned manifest, then the atomic rename
    try:
        GatewayPack.open_gateway(part, expected_fingerprint=man.fingerprint, verify="full").close()
    except Exception:
        shutil.rmtree(part, ignore_errors=True)  # never resume from bytes that failed verification
        raise
    state_p.unlink()
    final = mdir / "pack"
    if final.exists():
        shutil.rmtree(final)
    part.rename(final)
    return final


def _install_from_full_pack(full: Any, mdir: Path) -> Path:
    """The weight-free gateway pack from a verified full pack on disk (the public-source fallback)."""
    from pgw.modelpack.st_io import read_safetensors_header

    man = full.manifest
    part = mdir / "pack.partial"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True, mode=0o700)
    shutil.copyfile(full.root / MANIFEST_FILE, part / MANIFEST_FILE)
    objs = gateway_objects(man)
    by_file: dict[str, list[PackObject]] = {}
    for o in objs:
        if o.kind == "tensor":
            by_file.setdefault(str(o.file), []).append(o)
    headers: dict[str, Any] = {}
    for file, fobjs in by_file.items():
        offs = write_safetensors_layout(part / file, fobjs)
        with open(part / file, "r+b") as out:
            for o in fobjs:
                src = man.tensors[o.name].file
                if src not in headers:
                    headers[src], _ = read_safetensors_header(full.root / src)
                he = headers[src][o.name]
                with open(full.root / src, "rb") as f:
                    f.seek(he.begin)
                    out.seek(offs[o.id][0])
                    left = he.nbytes
                    while left:
                        buf = f.read(min(left, 8 << 20))
                        out.write(buf)
                        left -= len(buf)
    for t in man.tokenizer:
        dst = part / TOKENIZER_DIR / t.file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(full.root / TOKENIZER_DIR / t.file, dst)
    GatewayPack.open_gateway(part, expected_fingerprint=man.fingerprint, verify="full").close()
    final = mdir / "pack"
    if final.exists():
        shutil.rmtree(final)
    part.rename(final)
    return final


# ------------------------------------------------------------------------------------------------
# Weights (W8): streamed, checked against the manifest, never written
# ------------------------------------------------------------------------------------------------

class _Weights:
    """``get(key) -> W8`` (``(m, n)`` int8), each checked against the fingerprint-bound manifest: from the provider
    (``remote``) or a verified full pack on disk (``full``); the masked head is derived from the gateway pack's
    verified plain head exactly as every party derives it (:func:`pgw.modelpack.masked_head.quantize_head`)."""

    def __init__(self, man: Manifest, gpack: GatewayPack, *, remote: _Remote | None = None, full: Any = None):
        self.man, self.gpack, self.remote, self.full = man, gpack, remote, full
        self.objs = {o.id: o for o in weight_objects(man)}
        self.bytes = 0
        #: ``row_norm_sq_max`` of every linear fetched (for packs without it in the manifest: the gateway's sidecar)
        self.record_norms = any(e.row_norm_sq_max is None for e in man.linears)
        self.norms: dict[str, int] = {}

    def get(self, key: str) -> np.ndarray:
        from pgw.core.types import LinearKind

        if key.endswith("." + LinearKind.LM_HEAD.value):
            from pgw.modelpack.masked_head import MaskedHeadPack

            hp = MaskedHeadPack(self.gpack)
            w8 = hp.load_linear(hp.head_id).w8.numpy()
            hp.release_head()
            return w8
        o = self.objs.get(f"weight/{key}")
        if o is None or o.shape is None:
            raise ConfigError(f"unknown linear {key!r}")
        if self.full is not None:
            w = self.full.load_linear(key, check_hash=True).w8.numpy()
        else:
            assert self.remote is not None
            raw = self.remote.object_bytes(o)
            w = np.frombuffer(raw, dtype=np.int8).reshape(o.shape)
            if (w == -128).any():
                raise IntegrityError(f"{key}: int8 weight contains -128")
        self.bytes += int(w.nbytes)
        if self.record_norms:
            import torch

            from pgw.modelpack.loader import row_norm_sq_max

            self.norms[key] = row_norm_sq_max(torch.from_numpy(np.array(w)))
        return w


def stream_weights(pm: ProvisionedModel, session: Any, *, linears: list[str] | None = None) -> Iterator[Any]:
    """The pack's int8 linears one at a time as :class:`pgw.modelpack.loader.LinearWeights` (``W8`` from the
    provider, checked against the pinned manifest; scales and biases from the gateway pack). For callers that need
    ``W8`` transiently, e.g. the factory's private check vectors ``v W8``; nothing is written to disk."""
    import torch

    from pgw.core.types import LinearId
    from pgw.modelpack.loader import LinearWeights

    gp = pm.open_pack(verify="fingerprint")
    w = _Weights(pm.manifest, gp, remote=_Remote(session, pm.fingerprint))
    for e in pm.manifest.linears:
        if linears is not None and e.id not in linears:
            continue
        meta = gp.load_linear_meta(e.id)
        yield LinearWeights(LinearId.parse(e.id), torch.from_numpy(w.get(e.id).copy()), meta.scale, meta.bias)


def _ensure_norm_sidecar(gpack: GatewayPack, weights: _Weights) -> None:
    """Packs written before ``row_norm_sq_max`` existed: the capped-ring norms from the checked weights, written as
    the gateway's sidecar next to the gateway pack (so the gateway never reads weights)."""
    import torch

    from pgw.engine.backends import _read_norm_sidecar, _write_norm_sidecar
    from pgw.modelpack.loader import row_norm_sq_max

    if all(e.row_norm_sq_max is not None for e in gpack.manifest.linears):
        return
    if _read_norm_sidecar(gpack, gpack.root) is not None:
        return
    norms = {e.id: (weights.norms[e.id] if e.id in weights.norms
                    else row_norm_sq_max(torch.from_numpy(weights.get(e.id)))) for e in gpack.manifest.linears}
    _write_norm_sidecar(gpack, norms, gpack.root)


# ------------------------------------------------------------------------------------------------
# Factory pack: the full pack (int8 weights included) for the client's background mask factory
# ------------------------------------------------------------------------------------------------

FACTORY_PACK_DIR = "factory-pack"


def _install_factory_pack(weights: _Weights, gpack: GatewayPack, man: Manifest, mdir: Path,
                          progress: Progress, *, on_weight: Callable[[str, np.ndarray], None] | None = None,
                          need: Callable[[str], bool] | None = None, resume: bool = False) -> Path:
    """``mdir/factory-pack``: a full :class:`pgw.modelpack.ModelPack` with the same fingerprint as the gateway pack.

    The in-process gateway never reads ``W8``; the client's background mask factory does (its private check vectors
    ``v W8`` once per process and after every rejected segment, and the pks-v3 hint when it is built locally). ``W8``
    comes from ``weights`` (each linear checked against the fingerprint-bound manifest), scales and biases and the
    plain tensors from the verified gateway pack (plain-only files and the tokenizer are hard links where the file
    system allows). The directory is built under a temporary name, verified in full against the pinned fingerprint
    and renamed.

    ``on_weight(key, W8)`` sees every linear's checked ``W8`` while it is in memory (the fused hint build of
    ``pgw sdk prepare``: one pass over the weights instead of two); ``need(key)`` says whether it still wants a linear
    the resumed directory already holds (it is then read back and checked against the manifest's SHA-256 again).
    ``resume``: keep ``factory-pack.partial`` across runs (``state.json`` lists the objects written and synced); the
    final full verification hashes every byte against the manifest as for a fresh build."""
    from pgw.modelpack.loader import ModelPack
    from pgw.modelpack.st_io import read_safetensors_header

    part = mdir / (FACTORY_PACK_DIR + ".partial")
    state_p = part / _STATE
    state: dict[str, Any] | None = None
    if resume and state_p.is_file():
        with contextlib.suppress(ValueError, OSError):
            state = json.loads(state_p.read_text())
        if not isinstance(state, dict) or state.get("fingerprint") != man.fingerprint or \
                not isinstance(state.get("done"), list):
            state = None
    fresh = state is None
    if fresh:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True, mode=0o700)
        state = {"fingerprint": man.fingerprint, "done": []}
    assert state is not None
    done_ids: set[str] = set(state["done"])

    def link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():  # a resumed directory (the final verification checks every byte)
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)

    link_or_copy(gpack.root / MANIFEST_FILE, part / MANIFEST_FILE)
    for t in man.tokenizer:
        link_or_copy(gpack.root / TOKENIZER_DIR / t.file, part / TOKENIZER_DIR / t.file)
    weight_of = {e.weight: e.id for e in man.linears}
    lin_files = set(man.linear_files())
    for f in man.plain_files():
        if f not in lin_files:
            link_or_copy(gpack.root / f, part / f)
    headers: dict[str, Any] = {}

    def save_state() -> None:
        if resume:
            state["done"] = sorted(done_ids)
            _write_json_atomic(state_p, state)

    def gateway_bytes(name: str) -> bytes:
        src = man.tensors[name].file
        src = LINEAR_META_FILE if (src in lin_files and name not in man.plain_tensor_names()) else src
        if src not in headers:
            headers[src], _ = read_safetensors_header(gpack.root / src)
        he = headers[src][name]
        with open(gpack.root / src, "rb") as fh:
            fh.seek(he.begin)
            return fh.read(he.nbytes)

    total = sum(o.size for o in weight_objects(man))
    done = 0
    for f in sorted(lin_files):
        names = sorted(n for n, t in man.tensors.items() if t.file == f)
        objs = [_tensor_obj(man, n, f) for n in names]
        if fresh or not (part / f).is_file():
            for o in objs:
                done_ids.discard(o.id)
            offs = write_safetensors_layout(part / f, objs)
        else:  # same deterministic layout as when it was created
            hdr, _ = read_safetensors_header(part / f)
            offs = {o.id: (hdr[o.name].begin, hdr[o.name].begin + hdr[o.name].nbytes) for o in objs}
        fd = os.open(part / f, os.O_RDWR)
        try:
            for o in objs:
                key = weight_of.get(o.name)
                if o.id in done_ids:
                    if key is not None:
                        done += o.size
                        if on_weight is not None and (need is None or need(key)):
                            data = os.pread(fd, o.size, offs[o.id][0])
                            if len(data) == o.size and hashlib.sha256(data).hexdigest() == o.sha256:
                                on_weight(key, np.frombuffer(data, dtype=np.int8).reshape(o.shape))
                                continue
                            done_ids.discard(o.id)  # damaged since: fetch it again below
                            done -= o.size
                        else:
                            continue
                    else:
                        continue
                w: np.ndarray | None = None
                if key is not None:
                    w = weights.get(key)
                    data = w.tobytes()
                    done += len(data)
                    _emit(progress, "weights", done, total, item=key)
                else:
                    data = gateway_bytes(o.name)
                if len(data) != o.size:
                    raise IntegrityError(f"{o.name}: wrong size for the factory pack")
                os.pwrite(fd, data, offs[o.id][0])
                if resume and key is not None:
                    os.fsync(fd)
                    done_ids.add(o.id)
                    save_state()
                if w is not None and on_weight is not None:
                    on_weight(key, w)  # type: ignore[arg-type]
            os.fsync(fd)
            done_ids.update(o.id for o in objs)
            save_state()
        finally:
            os.close(fd)
    state_p.unlink(missing_ok=True)
    try:
        ModelPack.open(part, verify="full", expected_fingerprint=man.fingerprint).close()
    except Exception:
        shutil.rmtree(part, ignore_errors=True)
        raise
    final = mdir / FACTORY_PACK_DIR
    if final.exists():
        shutil.rmtree(final)
    part.rename(final)
    return final


def _open_factory_pack(mdir: Path, fp: str) -> Any:
    """The cached factory pack if it is complete and has fingerprint ``fp`` (manifest check only; the factory
    verifies every byte when it opens it), else ``None``."""
    from pgw.modelpack.loader import ModelPack

    d = mdir / FACTORY_PACK_DIR
    if not d.is_dir():
        return None
    try:
        return ModelPack.open(d, verify="fingerprint", expected_fingerprint=fp)
    except Exception as exc:
        log.warning("cached factory pack for %s is unusable (%s); rebuilding", fp[:12], exc)
        shutil.rmtree(d, ignore_errors=True)
        return None


# ------------------------------------------------------------------------------------------------
# Hint
# ------------------------------------------------------------------------------------------------

def _hint_specs(man: Manifest, masked_head: bool) -> dict[str, tuple[int, int]]:
    from pgw.modelpack.masked_head import head_linear_id

    specs = {e.id: (int(e.out_features), int(e.in_features)) for e in man.linears}
    if masked_head:
        a = man.arch
        specs[head_linear_id(a.num_layers).key] = (int(a.vocab_size), int(a.hidden_size))
    return specs


def _local_items(specs: dict[str, tuple[int, int]], weights: _Weights, a_seed: bytes, progress: Progress
                 ) -> Iterator[Any]:
    for idx, (key, (m, n)) in enumerate(specs.items()):
        _emit(progress, "hint", idx, len(specs), "linears", key)
        w = weights.get(key)
        if w.shape != (m, n):
            raise ConfigError(f"{key}: weight shape {w.shape} != ({m}, {n})")
        yield _local_linear(key, w, a_seed)


def _local_linear(key: str, w: np.ndarray, a_seed: bytes) -> Any:
    """One linear of a local build: ``H = W8 A mod 2^56`` in 1024-row blocks (``A``'s byte planes once)."""
    from pgw.hemask.hint_gemm import HintGemm
    from pgw.hemask.pks_backend import HintLinear, row_norm

    m, n = w.shape

    def blocks() -> Iterator[tuple[int, np.ndarray]]:
        gemm = HintGemm(a_seed, n, key)
        try:
            for r0 in range(0, m, 1024):
                yield r0, gemm(w[r0:r0 + 1024])
        finally:
            gemm.close()

    return HintLinear(key, m, n, row_norm(w), blocks())


class _ProviderH:
    """The provider's ``H`` for one job: chunks fetched one or two ahead of their use (network next to CPU)."""

    def __init__(self, session: Any, fp: str, a_seed: bytes, masked_head: bool, keys: list[str]):
        self.session = session
        self.job = session.request("POST", f"/{fp}/hint", json={"a_seed": a_seed.hex(), "masked_head": masked_head,
                                                                "linears": keys}).json()
        if self.job.get("form") != "h56" or self.job.get("scheme") != "pks-v3":
            raise ConfigError(f"the provider offers hint form {self.job.get('form')!r}, not h56")
        self.rows = int(self.job["rows_per_chunk"])
        if self.rows <= 0 or self.rows % 32:
            raise IntegrityError("bad rows_per_chunk from the provider")
        self.info = {d["key"]: d for d in self.job["linears"]}
        self.bytes = 0
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pgw-provision-h")

    def chunks(self, key: str, m: int) -> Iterator[tuple[int, bytes]]:
        n_chunks = -(-m // self.rows)
        jid = self.job["job"]

        def fetch(i: int) -> bytes:
            return self.session.get_bytes(f"/hint/{jid}/h", linear=key, chunk=i)

        futs = {i: self.pool.submit(fetch, i) for i in range(min(2, n_chunks))}
        for i in range(n_chunks):
            data = futs.pop(i).result()
            if i + 2 < n_chunks:
                futs[i + 2] = self.pool.submit(fetch, i + 2)
            self.bytes += len(data)
            yield i * self.rows, data

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(Exception):
            self.session.request("DELETE", f"/hint/{self.job['job']}")


def _provider_items(specs: dict[str, tuple[int, int]], weights: _Weights, a_seed: bytes, ph: _ProviderH,
                    progress: Progress, stats: dict[str, float]) -> Iterator[Any]:
    from pgw.hemask.pks_backend import HintLinear, row_norm
    from pgw.provision.hintcheck import SOUNDNESS_BITS, LinearHCheck, decode_h56

    for idx, (key, (m, n)) in enumerate(specs.items()):
        _emit(progress, "hint", idx, len(specs), "linears", key)
        info = ph.info.get(key)
        if info is None or (int(info["m"]), int(info["n"])) != (m, n):
            raise IntegrityError(f"{key}: the provider's job does not match the pinned shapes")
        w = weights.get(key)
        if w.shape != (m, n):
            raise ConfigError(f"{key}: weight shape {w.shape} != ({m}, {n})")
        t0 = time.perf_counter()
        ck = LinearHCheck(a_seed, key, m, n)
        ck.add_weights(0, w)
        norm = row_norm(w)
        del w
        stats["verify_s"] += time.perf_counter() - t0

        def blocks(key: str = key, m: int = m, ck: LinearHCheck = ck) -> Iterator[tuple[int, np.ndarray]]:
            for r0, data in ph.chunks(key, m):
                rows = min(ph.rows, m - r0)
                t1 = time.perf_counter()
                h = decode_h56(data, rows)
                ck.add_h(r0, h)
                stats["verify_s"] += time.perf_counter() - t1
                yield r0, h
            t2 = time.perf_counter()
            ok = ck.passed()
            stats["verify_s"] += time.perf_counter() - t2
            if not ok:
                raise IntegrityError(f"provider hint operand for {key} failed the {SOUNDNESS_BITS}-bit check against "
                                     "the pinned weights; nothing was stored (use hint='local' or retry)")

        yield HintLinear(key, m, n, norm, blocks())


def _open_store(hint_root: Path, ffp: str, mac_key: bytes, keys: set[str]) -> Any:
    from pgw.hemask.pks_backend import PKSHintStore

    lock = PKSHintStore.lock_path(hint_root, ffp)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        st = PKSHintStore.open(hint_root, ffp, mac_key=mac_key)
        if st is None or not keys <= set(st.meta["row_norms"]):
            os.close(fd)
            return None
        st._lock_fd = fd
        return st
    except BaseException:
        os.close(fd)
        raise


def find_prepared_hint(cache_dir: str | Path, fingerprint: str, *, masked_head: bool) -> tuple[Path, Path] | None:
    """``(hint_root, hint_key_file)`` of a complete hint store that :func:`provision` built under ``cache_dir`` for
    ``fingerprint`` (e.g. by ``pgw sdk prepare``, here or on another host whose cache was copied), else ``None``.

    The store's ``meta.json`` must verify under the cache's hint key (HMAC-SHA256) and cover every linear the masked
    gateway uses; without the key file nothing is trusted (``None``: the factory then builds its own hint). A store
    whose MAC does not verify under the key raises :class:`IntegrityError` (never used, never silently replaced).
    The per-file SHA-256 digests are checked by the factory when it opens the store."""
    from pgw.hemask.pks_backend import PKSHintStore
    from pgw.maskstore.keys import _read_key_file
    from pgw.modelpack.masked_head import head_fingerprint

    cache = Path(cache_dir).expanduser()
    key_file = cache / "keys" / "pks-hint.key"
    mdir = cache / "models" / fingerprint[:16]
    hint_root = mdir / "hints"
    if not key_file.is_file() or not hint_root.is_dir() or not (mdir / "pack" / MANIFEST_FILE).is_file():
        return None
    man = Manifest.from_file(mdir / "pack" / MANIFEST_FILE)
    man.verify_fingerprint(fingerprint)
    ffp = head_fingerprint(fingerprint) if masked_head else fingerprint
    st = PKSHintStore.open(hint_root, ffp, verify=False, mac_key=_read_key_file(key_file))
    if st is None:
        return None
    try:
        if "mac" not in st.meta or not set(_hint_specs(man, masked_head)) <= set(st.meta["row_norms"]):
            return None
    finally:
        st.close()
    return hint_root, key_file


def _build_store(kind: str, hint_root: Path, ffp: str, mac_key: bytes, specs: dict[str, tuple[int, int]],
                 weights: _Weights, session: Any, fp: str, masked_head: bool, progress: Progress,
                 stats: dict[str, float]) -> Any:
    from pgw.core.prng import new_seed
    from pgw.hemask.pks_backend import PKSHintStore
    from pgw.provision.hintcheck import CHALLENGE_COLS, SOUNDNESS_BITS

    hint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = PKSHintStore.lock_path(hint_root, ffp)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigError(f"pks-v3 hint store under {hint_root} is in use by another process; stop it first") \
                from None
        a_seed, ks_seed = new_seed(), new_seed()  # client-chosen (CSPRNG); only a_seed is ever sent
        if kind == "provider":
            ph = _ProviderH(session, fp, a_seed, masked_head, list(specs))
            try:
                st = PKSHintStore.write(
                    hint_root, ffp, a_seed, ks_seed, _provider_items(specs, weights, a_seed, ph, progress, stats),
                    mac_key=mac_key, extra_meta={"source": "provider-verified", "check": {
                        "form": "h56", "challenge_cols": CHALLENGE_COLS, "soundness_bits": SOUNDNESS_BITS}})
            finally:
                stats["h_bytes"] += ph.bytes
                ph.close()
        else:
            st = PKSHintStore.write(hint_root, ffp, a_seed, ks_seed, _local_items(specs, weights, a_seed, progress),
                                    mac_key=mac_key, extra_meta={"source": "local"})
        fcntl.flock(fd, fcntl.LOCK_SH)
        st._lock_fd = fd
        return st
    except BaseException:
        os.close(fd)
        raise


class _LocalHint:
    """A local hint build that takes ``W8`` linear by linear as it arrives (fused with the factory-pack download:
    one pass over the weights) under the store's exclusive lock; ``resume``: a resumable
    :class:`pgw.hemask.pks_backend.HintBuild` (HMAC-authenticated partial state under the hint key)."""

    def __init__(self, hint_root: Path, ffp: str, mac_key: bytes, specs: dict[str, tuple[int, int]], *,
                 resume: bool, progress: Progress):
        from pgw.hemask.pks_backend import HintBuild, PKSHintStore

        hint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.fd = os.open(PKSHintStore.lock_path(hint_root, ffp), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise ConfigError(f"pks-v3 hint store under {hint_root} is in use by another process; stop it first") \
                from None
        try:
            self.hb = HintBuild(hint_root, ffp, mac_key=mac_key, resumable=resume)
        except BaseException:
            os.close(self.fd)
            raise
        self.specs, self.progress = specs, progress
        self.built = 0
        self.seconds = 0.0

    def need(self, key: str) -> bool:
        return key in self.specs and not self.hb.done(key)

    def on_weight(self, key: str, w: np.ndarray) -> None:
        if not self.need(key):
            return
        m, n = self.specs[key]
        if w.shape != (m, n):
            raise ConfigError(f"{key}: weight shape {w.shape} != ({m}, {n})")
        t0 = time.perf_counter()
        self.hb.add(_local_linear(key, w, self.hb.a_seed))
        self.seconds += time.perf_counter() - t0
        self.built += 1
        _emit(self.progress, "hint", sum(self.hb.done(k) for k in self.specs), len(self.specs), "linears", key)

    def finish(self, weights: _Weights) -> Any:
        """The linears not seen yet (the masked head, or all of them when the factory pack already existed), then
        ``meta.json`` and the rename; the store keeps a shared lock."""
        try:
            for key in self.specs:
                if self.need(key):
                    self.on_weight(key, weights.get(key))
            st = self.hb.finish(keys=list(self.specs), extra_meta={"source": "local"})
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            st._lock_fd = self.fd
            return st
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        fd, self.fd = self.fd, -1
        if fd >= 0:
            self.hb.abort()
            os.close(fd)


# ------------------------------------------------------------------------------------------------
# Public-source fallback
# ------------------------------------------------------------------------------------------------

def _full_from_hf(hf_repo: str, man_hint: Manifest | None, fp: str | None, work: Path, *,
                  name: str | None, revision: str | None, plain_dtype: str | None) -> Any:
    """Pack the public checkpoint locally (``hf_repo``: hub id or local directory) with the pinned manifest's
    packing parameters, and require the pinned fingerprint."""
    from pgw.modelpack import ModelPack, pack_checkpoint

    if man_hint is not None:
        name = name or man_hint.name
        revision = revision or man_hint.source.revision
        plain_dtype = plain_dtype or man_hint.quant.plain_dtype
    src: str | Path = hf_repo
    if not Path(hf_repo).is_dir():
        from huggingface_hub import snapshot_download

        src = snapshot_download(hf_repo, revision=revision,
                                allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])
    out = work / "full"
    shutil.rmtree(out, ignore_errors=True)
    pack_checkpoint(src, out, name=name, revision=revision, plain_dtype=plain_dtype or "float32",  # type: ignore[arg-type]
                    license=man_hint.source.license if man_hint is not None else None)
    full = ModelPack.open(out, verify="full")
    if fp is not None and full.fingerprint != fp:
        raise FingerprintMismatchError(
            f"the public checkpoint {hf_repo} packs to fingerprint {full.fingerprint[:12]}, not the pinned "
            f"{fp[:12]} (different revision or packing parameters)")
    return full


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------

def provision(model: str, *, session: Any = None, cache_dir: str | Path, progress: Progress = None,
              hint: HintMode = "provider", expected_fingerprint: str | None = None, masked_head: bool = True,
              source: Literal["provider", "hf"] = "provider", hf_repo: str | None = None,
              hf_revision: str | None = None, hf_name: str | None = None, hf_plain_dtype: str | None = None,
              keep_full_pack: bool = False, factory_pack: bool = False, resume: bool = False) -> ProvisionedModel:
    """Provision ``model`` (name or fingerprint) under ``cache_dir`` (see the module docstring).

    ``factory_pack=True`` also keeps a full pack with the int8 weights (``factory_pack_dir``) for the client's
    background mask factory: built from the provider's fingerprint-checked ``W8`` (or kept from ``source="hf"``),
    once; the hint check and a local hint build then read ``W8`` from it instead of streaming it again. With
    ``hint="local"`` the hint of each linear is built while its ``W8`` is in memory for the factory pack (one pass
    over the weights; same files as a separate build). ``resume=True`` keeps a partial factory pack and a partial
    hint build (HMAC-authenticated state) across interruptions, so a long one-time setup continues where it stopped
    (``pgw sdk prepare``)."""
    if hint not in ("provider", "local", "none"):
        raise ConfigError(f"hint must be 'provider', 'local' or 'none', not {hint!r}")
    if source not in ("provider", "hf"):
        raise ConfigError(f"source must be 'provider' or 'hf', not {source!r}")
    if session is None and (source == "provider" or hint == "provider"):
        raise ConfigError("a provider session is needed (source='hf' with hint='local' or 'none' works without)")
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    stats: dict[str, float] = {"verify_s": 0.0, "h_bytes": 0}

    # ---- the fingerprint to pin
    fp = expected_fingerprint
    if fp is not None:
        validate_fingerprint(fp)
    if _is_fp(model):
        if fp is not None and fp != model:
            raise FingerprintMismatchError("model fingerprint and expected_fingerprint differ")
        fp = model
    if not _is_fp(model) and session is not None:
        _emit(progress, "catalog")
        cat = session.get_json("/models")
        hits = [m for m in cat.get("models", []) if m.get("name") == model]
        if len(hits) != 1:
            raise ModelNotFoundError(f"the provider serves no (unique) model named {model!r}")
        served = validate_fingerprint(str(hits[0]["fingerprint"]))
        if fp is not None and served != fp:
            raise FingerprintMismatchError(f"the provider serves {model!r} as {served[:12]}, not the pinned {fp[:12]}")
        fp = served
    if fp is None:
        raise ConfigError("without a provider session pass the model's fingerprint (expected_fingerprint)")
    from pgw.modelpack.masked_head import head_fingerprint

    ffp = head_fingerprint(fp) if masked_head else fp
    mdir = cache / "models" / fp[:16]
    mdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _flock(mdir / ".lock"):
        # ---- artifacts
        t0 = time.perf_counter()
        remote: _Remote | None = None
        full: Any = None
        man: Manifest | None = None
        pack_dir = mdir / "pack"
        gpack: GatewayPack | None = None
        if pack_dir.is_dir():
            try:
                gpack = GatewayPack.open_gateway(pack_dir, expected_fingerprint=fp, verify="fingerprint")
                man = gpack.manifest
            except Exception as exc:
                log.warning("cached gateway pack for %s is unusable (%s); re-provisioning", fp[:12], exc)
                shutil.rmtree(pack_dir, ignore_errors=True)
                gpack = None
        if gpack is None:
            if source == "provider":
                remote = _Remote(session, fp)
                man, raw_b = _fetch_manifest(remote, fp)
                _install_from_provider(remote, man, raw_b, mdir, progress)
            else:
                man_hint = None
                if session is not None:
                    with contextlib.suppress(Exception):
                        remote = _Remote(session, fp)
                        man_hint = _fetch_manifest(remote, fp)[0]
                if not hf_repo:
                    raise ConfigError("source='hf' needs hf_repo (hub id or local checkpoint directory)")
                full = _full_from_hf(hf_repo, man_hint, fp, mdir, name=hf_name, revision=hf_revision,
                                     plain_dtype=hf_plain_dtype)
                _install_from_full_pack(full, mdir)
            gpack = GatewayPack.open_gateway(pack_dir, expected_fingerprint=fp, verify="fingerprint")
            man = gpack.manifest
        assert man is not None and gpack is not None
        timings["artifacts"] = time.perf_counter() - t0

        def weights() -> _Weights:
            nonlocal remote, full
            if full is None and (session is None or source == "hf"):
                if not hf_repo:
                    raise ConfigError("the weights are needed (hint build) but there is no provider session and no "
                                      "hf_repo")
                full = _full_from_hf(hf_repo, man, fp, mdir, name=hf_name, revision=hf_revision,
                                     plain_dtype=hf_plain_dtype)
            if full is not None:
                return _Weights(man, gpack, full=full)
            if remote is None:
                remote = _Remote(session, fp)
            return _Weights(man, gpack, remote=remote)

        wsrc: _Weights | None = None

        # ---- factory pack (full pack for the client's background mask factory)
        fpack_dir: Path | None = None
        streamed_w = 0
        local_hint: _LocalHint | None = None
        key_file: Path | None = None
        hint_root: Path | None = None
        store: Any = None
        specs: dict[str, tuple[int, int]] = {}
        mac_key = b""
        if hint != "none":
            key_file, mac_key = _hint_key(cache)
            hint_root = mdir / "hints"
            specs = _hint_specs(man, masked_head)
            store = _open_store(hint_root, ffp, mac_key, set(specs)) if hint_root.is_dir() else None
        if factory_pack:
            t2 = time.perf_counter()
            fpk = _open_factory_pack(mdir, fp)
            if fpk is None:
                if full is not None:  # source="hf": the verified full pack is the factory pack
                    full.close()
                    dst = mdir / FACTORY_PACK_DIR
                    shutil.rmtree(dst, ignore_errors=True)
                    Path(full.root).rename(dst)
                    full = None
                else:
                    wsrc = wsrc or weights()
                    if hint == "local" and store is None:  # fused: each linear's hint while its W8 is in memory
                        assert hint_root is not None
                        local_hint = _LocalHint(hint_root, ffp, mac_key, specs, resume=resume, progress=progress)
                    try:
                        _install_factory_pack(wsrc, gpack, man, mdir, progress, resume=resume,
                                              on_weight=local_hint.on_weight if local_hint is not None else None,
                                              need=local_hint.need if local_hint is not None else None)
                    except BaseException:
                        if local_hint is not None:
                            local_hint.abort()
                        raise
                fpk = _open_factory_pack(mdir, fp)
                if fpk is None:
                    raise IntegrityError("the factory pack could not be built")
            full = full or fpk
            streamed_w = wsrc.bytes if wsrc is not None and wsrc.remote is not None else 0
            wsrc = _Weights(man, gpack, full=full)  # W8 from disk from here on (checked again per linear)
            fpack_dir = mdir / FACTORY_PACK_DIR
            timings["factory_pack"] = time.perf_counter() - t2

        # ---- hint
        t1 = time.perf_counter()
        hint_source: str | None = None
        if hint != "none":
            assert hint_root is not None
            if store is None:
                wsrc = wsrc or weights()
                if local_hint is None and hint == "local" and resume:
                    local_hint = _LocalHint(hint_root, ffp, mac_key, specs, resume=True, progress=progress)
                if local_hint is not None:
                    store = local_hint.finish(wsrc)
                    timings["hint_fused"] = local_hint.seconds
                else:
                    store = _build_store(hint, hint_root, ffp, mac_key, specs, wsrc, session, fp, masked_head,
                                         progress, stats)
            hint_source = str(store.meta.get("source", "local"))
        timings["hint"] = time.perf_counter() - t1 + timings.get("hint_fused", 0.0)
        timings["verify"] = stats["verify_s"]
        if any(e.row_norm_sq_max is None for e in man.linears):  # older packs: norms from the checked weights
            wsrc = wsrc or weights()
            _ensure_norm_sidecar(gpack, wsrc)
        if full is not None and fpack_dir is not None and Path(full.root) == fpack_dir:
            full.close()  # the factory pack stays
        elif full is not None and not keep_full_pack:
            full.close()
            shutil.rmtree(full.root, ignore_errors=True)
        gpack.close()
    sizes = {
        "artifacts": sum(p.stat().st_size for p in pack_dir.rglob("*") if p.is_file()),
        "hint": int(store.nbytes) if store is not None else 0,
        "downloaded": int((remote.bytes if remote is not None else 0) + stats["h_bytes"]),
        "weights_streamed": int(streamed_w + (wsrc.bytes if wsrc is not None and wsrc.remote is not None else 0)),
        "h_streamed": int(stats["h_bytes"]),
    }
    timings["total"] = time.perf_counter() - t_start
    _emit(progress, "done", 1, 1, "models", fp[:12])
    log.info("provisioned %s (fp %s): artifacts %.2f GB, hint %.2f GB (%s), %.0f s", man.name, fp[:12],
             sizes["artifacts"] / 1e9, sizes["hint"] / 1e9, hint_source, timings["total"])
    return ProvisionedModel(name=man.name, fingerprint=fp, factory_fingerprint=ffp, pack_dir=pack_dir,
                            tokenizer_dir=pack_dir / TOKENIZER_DIR, manifest=man, hint_root=hint_root,
                            hint_store=store, hint_key_file=key_file, hint_source=hint_source,
                            masked_head=masked_head, sizes=sizes, timings=timings, factory_pack_dir=fpack_dir)

