"""Provider side of client provisioning: ``/pgw/v1/provision/...`` (opt-in, ``ServiceSettings.provision``).

Everything served here is public: the served packs' manifests, plain tensors, linear scales/biases, tokenizer
files and int8 weights (the gateway's non-secret artifacts, :mod:`pgw.provision.pack`), and pks-v3 hint
operands ``H = W8 A(a_seed) mod 2^56`` for a seed the customer chose. The client checks every byte against the
pack fingerprint it pins and every ``H`` with a 2-adic Freivalds test before using it (:mod:`pgw.provision`); this
module is therefore not trusted for integrity, only for availability. Payloads: ``docs/sdk/CONTRACT.md`` (A3).

Authentication is the compute path's (``PGW-Protocol`` header, bearer token / client certificate); hint jobs are
bound to the tenant that created them. ``H`` is computed on the service's GPU when one is configured and passes
the exact self-check of :class:`pgw.hemask.hint_gpu.GpuHintPlanes`, else on the CPU (int8 byte-plane GEMMs).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pgw.core.errors import ConfigError, ModelNotFoundError, PGWError, ProtocolError, ServiceUnavailableError
from pgw.core.fingerprint import short_fingerprint

__all__ = ["PROVISION_PREFIX", "ProvisionService", "add_provision_routes"]

log = logging.getLogger(__name__)

PROVISION_PREFIX = "/pgw/v1/provision"
#: Rows of ``H`` per chunk (a multiple of 32: the client key-switches whole 32-row blocks).
H_ROWS_PER_CHUNK = 256
#: Rows of ``H`` computed per device call (cached, then served in chunks).
_H_BLOCK_ROWS = 2048
_M56 = np.uint64((1 << 56) - 1)
_HEX = set("0123456789abcdef")


class NotFoundError(ModelNotFoundError):
    """Unknown model, object, chunk or job (HTTP 404)."""


@dataclass
class _ObjEntry:
    id: str
    kind: str
    name: str
    path: Path
    begin: int
    size: int
    sha256: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    chunks: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "dtype": self.dtype,
                "shape": list(self.shape) if self.shape is not None else None, "size": self.size,
                "sha256": self.sha256, "chunks": self.chunks}


@dataclass
class _HintJob:
    id: str
    tenant: str
    fingerprint: str
    a_seed: bytes
    linears: dict[str, tuple[int, int]]  # key -> (m, n)
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    a_cache: tuple[str, np.ndarray] | None = None
    blocks: OrderedDict[tuple[str, int], bytes] = field(default_factory=OrderedDict)


class ProvisionService:
    """Catalog, artifact chunks and hint jobs for the packs of a :class:`pgw.service.registry.ModelRegistry`."""

    def __init__(self, registry: Any, *, chunk_bytes: int = 8 << 20, max_jobs: int = 4, job_ttl_s: float = 6 * 3600.0,
                 h_device: str | None = None):
        self.registry = registry
        self.chunk_bytes = int(chunk_bytes)
        self.max_jobs = int(max_jobs)
        self.job_ttl_s = float(job_ttl_s)
        self._lock = threading.Lock()
        self._packs: dict[str, Any] = {}
        self._heads: dict[str, Any] = {}
        self._objects: dict[str, dict[str, _ObjEntry]] = {}
        self._jobs: dict[str, _HintJob] = {}
        self._compute = threading.Semaphore(1)  # one H block at a time (the device also serves compute)
        self._h_device = h_device
        self._gpu: Any = None
        self._gpu_checked = False
        self.h_blocks_computed = 0

    # ---- packs -------------------------------------------------------------------------------
    def _base_entries(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m in self.registry.models():
            fp = m.base_fingerprint or m.fingerprint
            e = out.setdefault(fp, {"name": m.name, "fingerprint": fp, "masked_head": False, "pack_path": m.pack_path,
                                    "devices": set()})
            e["masked_head"] = e["masked_head"] or bool(m.base_fingerprint)
            e["devices"].update(m.devices())
        return out

    def _pack(self, fp: str) -> Any:
        with self._lock:
            p = self._packs.get(fp)
            if p is not None:
                return p
        e = self._base_entries().get(fp)
        if e is None or e["pack_path"] is None:
            raise NotFoundError(f"no served model with fingerprint {short_fingerprint(fp)}")
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(e["pack_path"], verify="fingerprint", expected_fingerprint=fp, components=("linears",))
        with self._lock:
            return self._packs.setdefault(fp, pack)

    def device(self, fp: str | None = None) -> str:
        devs: set[str] = set()
        for e in self._base_entries().values():
            if fp is None or e["fingerprint"] == fp:
                devs |= e["devices"]
        want = self._h_device or next((d for d in sorted(devs) if d.startswith("cuda")), "cpu")
        return want if want.startswith("cuda") and self._gpu_ok(want) else "cpu"

    def catalog(self) -> dict[str, Any]:
        out = []
        for fp, e in sorted(self._base_entries().items(), key=lambda kv: kv[1]["name"]):
            out.append({"name": e["name"], "fingerprint": fp, "masked_head": e["masked_head"], "hint_build": True,
                        "device": "cuda" if any(d.startswith("cuda") for d in e["devices"]) else "cpu"})
        return {"models": out}

    # ---- artifacts ---------------------------------------------------------------------------
    def _object_table(self, fp: str) -> dict[str, _ObjEntry]:
        with self._lock:
            t = self._objects.get(fp)
        if t is not None:
            return t
        from pgw.modelpack.manifest import MANIFEST_FILE, TOKENIZER_DIR
        from pgw.modelpack.st_io import read_safetensors_header
        from pgw.provision.pack import gateway_objects, weight_objects

        pack = self._pack(fp)
        man = pack.manifest
        root = Path(pack.root)
        mpath = root / MANIFEST_FILE
        raw = mpath.read_bytes()
        table: dict[str, _ObjEntry] = {"manifest.json": _ObjEntry("manifest.json", "manifest", MANIFEST_FILE, mpath, 0,
                                                                  len(raw), hashlib.sha256(raw).hexdigest())}
        headers: dict[str, Any] = {}
        for o in [*gateway_objects(man), *weight_objects(man)]:
            if o.kind == "tokenizer":
                p = root / TOKENIZER_DIR / o.name
                if not p.is_file():
                    continue  # reported as missing when requested
                table[o.id] = _ObjEntry(o.id, o.kind, o.name, p, 0, p.stat().st_size, str(o.sha256))
                continue
            file = man.tensors[o.name].file
            if not (root / file).is_file():
                continue
            if file not in headers:
                headers[file], _ = read_safetensors_header(root / file)
            he = headers[file].get(o.name)
            if he is None or he.nbytes != o.size:
                raise ConfigError(f"pack {pack.name}: {file} does not hold {o.name} as the manifest says")
            table[o.id] = _ObjEntry(o.id, o.kind, o.name, root / file, he.begin, o.size, str(o.sha256), o.dtype,
                                    o.shape)
        for e in table.values():  # per-chunk SHA-256 (the client's early reject / resume; not its trust anchor)
            e.chunks = [hashlib.sha256(self._read(e, i)).hexdigest() for i in range(self._nchunks(e))]
        with self._lock:
            return self._objects.setdefault(fp, table)

    def _nchunks(self, e: _ObjEntry) -> int:
        return max(1, -(-e.size // self.chunk_bytes))

    def _read(self, e: _ObjEntry, i: int) -> bytes:
        if not 0 <= i < self._nchunks(e):
            raise NotFoundError(f"{e.id}: no chunk {i}")
        off = i * self.chunk_bytes
        n = min(self.chunk_bytes, e.size - off)
        with open(e.path, "rb") as f:
            f.seek(e.begin + off)
            data = f.read(n)
        if len(data) != n:
            raise ConfigError(f"{e.id}: short read")
        return data

    def manifest(self, fp: str) -> dict[str, Any]:
        t = self._object_table(fp)
        pack = self._pack(fp)
        return {"format": "pgw-provision", "version": 1, "fingerprint": fp, "name": pack.name,
                "chunk_bytes": self.chunk_bytes, "objects": [e.public() for e in t.values()]}

    def chunk(self, fp: str, oid: str, i: int) -> tuple[bytes, str]:
        e = self._object_table(fp).get(oid)
        if e is None:
            raise NotFoundError(f"no provisioning object {oid!r} for {short_fingerprint(fp)}")
        data = self._read(e, i)
        return data, e.chunks[i]

    # ---- hint operand H ----------------------------------------------------------------------
    def _gpu_ok(self, dev: str) -> bool:
        if self._gpu_checked:
            return self._gpu is not None
        self._gpu_checked = True
        try:
            from pgw.hemask.hint_gpu import GpuHintPlanes

            g = GpuHintPlanes(dev)
            if g.self_check([(256, 896), (256, 4864)]):
                self._gpu = g
                log.info("provision: hint operands on %s (self-check passed)", dev)
        except Exception as exc:  # no CUDA, no torch._int_mm, ...: the CPU path
            log.info("provision: hint operands on the CPU (%s)", type(exc).__name__)
        return self._gpu is not None

    def _weight(self, fp: str, key: str) -> np.ndarray:
        from pgw.core.types import LinearKind

        pack = self._pack(fp)
        if key.endswith("." + LinearKind.LM_HEAD.value):
            with self._lock:
                w = self._heads.get(fp)
            if w is None:
                from pgw.modelpack import ModelPack
                from pgw.modelpack.masked_head import MaskedHeadPack

                full = ModelPack.open(pack.root, verify="fingerprint", expected_fingerprint=fp,
                                      components=("plain",))
                hp = MaskedHeadPack(full)
                w = hp.load_linear(hp.head_id).w8.numpy()
                with self._lock:
                    self._heads[fp] = w
            return w
        return pack.load_linear(key).w8.numpy()

    def create_job(self, tenant: str, fp: str, a_seed: bytes, *, masked_head: bool,
                   linears: list[str] | None) -> dict[str, Any]:
        from pgw.hemask.pks import PACK, RANK
        from pgw.modelpack.masked_head import head_linear_id

        pack = self._pack(fp)
        specs = {s.id.key: (int(s.out_features), int(s.in_features)) for s in pack.linear_specs()}
        if masked_head:
            a = pack.arch
            specs[head_linear_id(a.num_layers).key] = (int(a.vocab_size), int(a.hidden_size))
        if linears is not None:
            unknown = [k for k in linears if k not in specs]
            if unknown:
                raise ProtocolError(f"unknown linear(s) {unknown[:3]}")
            specs = {k: specs[k] for k in linears}
        bad = [k for k, (m, n) in specs.items() if m % PACK or n % RANK]
        if bad:
            raise ProtocolError(f"linear(s) {bad[:3]} have no pks-v3 shape (out % 32, in % 16)")
        self._expire()
        with self._lock:
            if len(self._jobs) >= self.max_jobs or sum(j.tenant == tenant for j in self._jobs.values()) >= 2:
                raise ServiceUnavailableError("too many provisioning hint jobs; retry later",
                                              details={"reason": "provision_jobs"})
            job = _HintJob(secrets.token_hex(16), tenant, fp, bytes(a_seed), specs)
            self._jobs[job.id] = job
        dev = self.device(fp)
        log.info("provision: hint job for %s (%d linears, tenant %s, %s)", short_fingerprint(fp), len(specs), tenant,
                 dev)
        return {"job": job.id, "scheme": "pks-v3", "form": "h56", "rows_per_chunk": H_ROWS_PER_CHUNK, "device": dev,
                "linears": [{"key": k, "m": m, "n": n, "chunks": -(-m // H_ROWS_PER_CHUNK)}
                            for k, (m, n) in specs.items()]}

    def _expire(self) -> None:
        now = time.monotonic()
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if now - j.last_used > self.job_ttl_s]:
                self._jobs.pop(jid, None)

    def _job(self, tenant: str, jid: str) -> _HintJob:
        self._expire()
        with self._lock:
            job = self._jobs.get(jid)
        if job is None or not secrets.compare_digest(job.tenant, tenant):
            raise NotFoundError("no such provisioning hint job")
        job.last_used = time.monotonic()
        return job

    def delete_job(self, tenant: str, jid: str) -> None:
        job = self._job(tenant, jid)
        with self._lock:
            self._jobs.pop(job.id, None)

    def _h_rows(self, w: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self._gpu is not None:
            from pgw.hemask.fastmm import planes_to_u64

            try:
                return planes_to_u64(self._gpu(w, a)) & _M56
            except Exception as exc:  # pragma: no cover - GPU only
                log.warning("provision: GPU hint operand failed (%s); using the CPU", exc)
                self._gpu = None
        from pgw.hemask.lwe import matmul_mod64

        return matmul_mod64(np.ascontiguousarray(w), a) & _M56

    def h_chunk(self, tenant: str, jid: str, linear: str, i: int) -> bytes:
        from pgw.hemask import pks
        from pgw.provision.hintcheck import encode_h56

        job = self._job(tenant, jid)
        if linear not in job.linears:
            raise NotFoundError(f"linear {linear!r} is not part of this job")
        m, n = job.linears[linear]
        r0 = i * H_ROWS_PER_CHUNK
        if not 0 <= r0 < m:
            raise NotFoundError(f"{linear}: no chunk {i}")
        b0 = r0 // _H_BLOCK_ROWS * _H_BLOCK_ROWS
        with job.lock:
            blk = job.blocks.get((linear, b0))
            if blk is None:
                if job.a_cache is None or job.a_cache[0] != linear:
                    job.a_cache = None
                    job.a_cache = (linear, pks.a_mlwe_rows(pks.public_ring_rows(job.a_seed, n, linear)))
                w = self._weight(job.fingerprint, linear)
                if w.shape != (m, n):
                    raise ConfigError(f"{linear}: weight shape {w.shape} != ({m}, {n})")
                with self._compute:
                    if not self._gpu_checked:
                        self.device(job.fingerprint)
                    h = self._h_rows(w[b0:b0 + _H_BLOCK_ROWS], job.a_cache[1])
                self.h_blocks_computed += 1
                blk = encode_h56(h)
                del h
                job.blocks[(linear, b0)] = blk
                while len(job.blocks) > 2:
                    job.blocks.popitem(last=False)
        per_row = pks.NU * 7
        lo = (r0 - b0) * per_row
        rows = min(H_ROWS_PER_CHUNK, m - r0)
        return blk[lo:lo + rows * per_row]


def add_provision_routes(app: Any, state: Any) -> None:
    """Mount ``/pgw/v1/provision/...`` (see the module docstring and ``docs/sdk/CONTRACT.md``, A3)."""
    import json

    from starlette.requests import Request
    from starlette.responses import Response

    from pgw.wire.messages import PROTOCOL_HEADER
    from pgw.wire.server import (
        _client_cert_subject,
        _maybe_await,
        check_protocol_header,
        error_headers,
        error_response_bytes,
    )

    s = state.settings
    handler = state.handler
    svc = ProvisionService(state.registry, chunk_bytes=int(s.provision_chunk_mb) << 20,
                           max_jobs=s.provision_max_jobs, job_ttl_s=s.provision_job_ttl_s)
    state.provision = svc

    async def auth(request: Request) -> Any:
        check_protocol_header(request.headers.get(PROTOCOL_HEADER))
        identity = await _maybe_await(handler.authenticate(request.headers.get("authorization"),
                                                           _client_cert_subject(request.scope)))
        if not s.provision:
            raise ServiceUnavailableError("provisioning is not enabled on this service",
                                          details={"reason": "provision = false"})
        return identity

    def fail(exc: BaseException) -> Response:
        if not isinstance(exc, PGWError):
            log.error("provision internal error", exc_info=exc)
        status, body = error_response_bytes(exc, None, json_body=True)
        return Response(content=body, status_code=status, media_type="application/json", headers=error_headers(exc))

    def js(obj: Any) -> Response:
        return Response(content=json.dumps(obj), media_type="application/json", headers={"Cache-Control": "no-store"})

    def fp_of(request: Request) -> str:
        fp = str(request.path_params["fp"])
        if len(fp) != 64 or not set(fp) <= _HEX:
            raise ProtocolError("fingerprint must be 64 lowercase hex characters")
        return fp

    def int_param(request: Request, name: str) -> int:
        try:
            v = int(request.query_params[name])
        except (KeyError, ValueError) as exc:
            raise ProtocolError(f"query parameter {name!r} must be an integer") from exc
        if v < 0:
            raise ProtocolError(f"query parameter {name!r} must be >= 0")
        return v

    async def models(request: Request) -> Response:
        try:
            await auth(request)
            return js(await asyncio.to_thread(svc.catalog))
        except Exception as exc:
            return fail(exc)

    async def manifest(request: Request) -> Response:
        try:
            await auth(request)
            return js(await asyncio.to_thread(svc.manifest, fp_of(request)))
        except Exception as exc:
            return fail(exc)

    async def obj(request: Request) -> Response:
        try:
            await auth(request)
            oid = request.query_params.get("id")
            if not oid:
                raise ProtocolError("query parameter 'id' is required")
            data, digest = await asyncio.to_thread(svc.chunk, fp_of(request), oid, int_param(request, "chunk"))
            return Response(content=data, media_type="application/octet-stream",
                            headers={"PGW-Chunk-Sha256": digest, "Cache-Control": "no-store"})
        except Exception as exc:
            return fail(exc)

    async def hint_create(request: Request) -> Response:
        try:
            ident = await auth(request)
            from pgw.core.http_body import read_limited

            raw = await read_limited(request, 1 << 16, what="provision/hint body")
            try:
                req = json.loads(raw or b"{}")
            except ValueError as exc:
                raise ProtocolError("provision/hint body must be JSON") from exc
            if not isinstance(req, dict):
                raise ProtocolError("provision/hint body must be a JSON object")
            seed = req.get("a_seed")
            if not isinstance(seed, str) or len(seed) != 64 or not set(seed) <= _HEX:
                raise ProtocolError("a_seed must be 64 lowercase hex characters")
            lin = req.get("linears")
            if lin is not None and (not isinstance(lin, list) or not all(isinstance(x, str) for x in lin)):
                raise ProtocolError("linears must be a list of linear keys or null")
            out = await asyncio.to_thread(svc.create_job, ident.tenant_id, fp_of(request), bytes.fromhex(seed),
                                          masked_head=bool(req.get("masked_head", False)), linears=lin)
            return js(out)
        except Exception as exc:
            return fail(exc)

    async def hint_h(request: Request) -> Response:
        try:
            ident = await auth(request)
            lin = request.query_params.get("linear")
            if not lin:
                raise ProtocolError("query parameter 'linear' is required")
            data = await asyncio.to_thread(svc.h_chunk, ident.tenant_id, str(request.path_params["job"]), lin,
                                           int_param(request, "chunk"))
            return Response(content=data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return fail(exc)

    async def hint_delete(request: Request) -> Response:
        try:
            ident = await auth(request)
            await asyncio.to_thread(svc.delete_job, ident.tenant_id, str(request.path_params["job"]))
            return js({"ok": True})
        except Exception as exc:
            return fail(exc)

    p = PROVISION_PREFIX
    app.add_route(f"{p}/models", models, methods=["GET"], include_in_schema=False)
    app.add_route(p + "/hint/{job}/h", hint_h, methods=["GET"], include_in_schema=False)
    app.add_route(p + "/hint/{job}", hint_delete, methods=["DELETE"], include_in_schema=False)
    app.add_route(p + "/{fp}/manifest", manifest, methods=["GET"], include_in_schema=False)
    app.add_route(p + "/{fp}/object", obj, methods=["GET"], include_in_schema=False)
    app.add_route(p + "/{fp}/hint", hint_create, methods=["POST"], include_in_schema=False)
