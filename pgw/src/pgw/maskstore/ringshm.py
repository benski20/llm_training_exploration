"""Mask ring v2: the factory runs as the gateway's supervised child and fills shared-memory slabs.

Design ``research/plans/mask-ringbuffer-design.md`` §9 with the design-review fixes (§15):

* **Transport.** A ``SOCK_SEQPACKET`` :func:`socket.socketpair` created *before* the child is
  spawned; the child's end is passed with ``pass_fds``. There is no listening socket, no path in a
  shared directory and nothing another process of the same uid could connect to.
* **One memfd per grant.** For every ``GRANT`` the gateway creates a fresh memfd of one slab
  (``MFD_ALLOW_SEALING``, size sealed), maps it read-only, ``mlock``-ed, ``DONTDUMP``,
  ``DONTFORK``, and sends the fd. The whole arena is never handed out, and a memfd is never reused,
  so a zombie writer can never write into a re-granted slab.
* **Publish = seal.** On ``PUBLISH`` the gateway adds ``F_SEAL_WRITE``: the kernel refuses while any
  writable mapping exists, so a child that kept its mapping gets the slab refused, and a published
  slab can never be written again (integrity of ``W r`` without a checksum). Then the seed is
  checked (non-zero, unique in the shared registry), the soundness floor applied and the slab
  becomes READY. Reads use the same R4-guarded copy-out as v1.
* **Both processes** are non-dumpable (``PR_SET_DUMPABLE=0``, checked by the gateway in ``READY``;
  a dumpable child is killed). The child ``mlock``s its mappings (fail closed) and marks them
  ``DONTDUMP``/``DONTFORK``; ``PR_SET_PDEATHSIG`` kills it when the gateway dies.
* **Crash handling.** Child EOF does not mean the child is dead: the gateway kills it through its
  pidfd and reaps it before dropping the slabs it had granted. READY slabs keep serving (sealed).
  Gateway EOF: the child zeroes its open mapping and exits.

Messages are JSON and never carry seeds, rows or addresses. This transport serves a child of the
gateway only (no Kubernetes sidecar layout: that would need a listening socket).
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import torch

from pgw.core.errors import ConfigError, MaskReuseError, MaskStoreError, PGWError, ShapeError
from pgw.core.prng import SEED_BYTES
from pgw.core.ring import RingParams
from pgw.core.settings import ring_from_setting
from pgw.core.types import LinearId, LinearSpec
from pgw.maskstore.ringbuf import (
    FdMapping,
    MaskRing,
    RingFull,
    _all_zero,
    _as_ring_array,
    _lib,
    _pack_into,
    _zero,
    harden_process,
)

__all__ = ["RemoteRingSink", "RingBroker", "child_env", "serve"]

log = logging.getLogger(__name__)

_MAX_MSG = 1 << 16
_PR_SET_PDEATHSIG = 1
_PR_GET_DUMPABLE = 3
_RPC_TIMEOUT_S = 60.0


def _send(sock: socket.socket, lock: threading.Lock, msg: dict[str, Any], fds: Sequence[int] = (), *,
          nowait: bool = False) -> bool:
    data = json.dumps(msg, separators=(",", ":")).encode()
    if len(data) > _MAX_MSG:
        raise MaskStoreError("ring control message too large")
    with lock:
        try:
            if fds:
                socket.send_fds(sock, [data], list(fds), socket.MSG_DONTWAIT if nowait else 0)
            else:
                sock.send(data, socket.MSG_DONTWAIT if nowait else 0)
        except (BlockingIOError, BrokenPipeError, ConnectionResetError, OSError):
            if nowait:
                return False
            raise
    return True


def _recv(sock: socket.socket) -> tuple[dict[str, Any] | None, list[int]]:
    data, fds, _flags, _addr = socket.recv_fds(sock, _MAX_MSG, 4)
    if not data:
        for fd in fds:
            os.close(fd)
        return None, []
    try:
        msg = json.loads(data)
        if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
            raise ValueError("bad message")
    except ValueError:
        for fd in fds:
            os.close(fd)
        raise MaskStoreError("malformed ring control message") from None
    return msg, list(fds)


def _spec_sig(specs: Sequence[LinearSpec]) -> list[list[Any]]:
    return [[s.id.key, int(s.out_features), int(s.in_features)] for s in specs]


# =================================================================================================
# gateway side
# =================================================================================================
class RingBroker:
    """Runs the factory child and hands it shared slabs of the gateway's rings.

    Args:
        source: The gateway's :class:`pgw.maskstore.tiered.TieredMaskSource` (rings built with
            ``shared=True``).
        argv: Child command (default: ``python -m pgw.maskstore.ringshm``); ``--ring-fd N`` is added.
        env: Child environment (``PGW_FACTORY_*``).
    """

    def __init__(self, source: Any, argv: Sequence[str] | None = None, *, env: dict[str, str] | None = None,
                 device_lock: Any = None):
        self.source = source
        #: Host-wide per-device lock (``pgw.factory.embedded.acquire_device_lock``), released by :meth:`stop`.
        self.device_lock = device_lock
        self.argv = list(argv) if argv is not None else [sys.executable, "-m", "pgw.maskstore.ringshm"]
        self.env = env
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._pidfd: int | None = None
        self._grants: dict[tuple[str, str, int, int], MaskRing] = {}
        self._grants_open = True  #: False once the child was reaped: a late grant is aborted at once
        self._glock = threading.Lock()
        self._reap_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stopping = False
        self.ready = threading.Event()
        self.child_dumpable: int | None = None
        self.exit_code: int | None = None
        self.errors = 0
        self.published = 0

    # ---- lifecycle -------------------------------------------------------------------------------
    def start(self) -> None:
        gw, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            child.set_inheritable(True)
            self._proc = subprocess.Popen(  # noqa: S603 - our own interpreter and module
                [*self.argv, "--ring-fd", str(child.fileno())], pass_fds=(child.fileno(),), env=self.env,
                close_fds=True,
            )
        except BaseException:
            gw.close()
            raise
        finally:
            child.close()
        self._pidfd = os.pidfd_open(self._proc.pid)
        self.attach(gw)

    def attach(self, gw: socket.socket) -> None:
        """Serve the gateway end of a channel (``start`` uses it; tests attach an in-process peer)."""
        self._sock = gw
        models = [{"fp": fp, "ring": rg, "linears": [[lin.id, lin.out_features, lin.in_features] for lin in r.linears],
                   "segment_slots": r.segment_slots, "slab_bytes": r.geometry.slab_bytes, "wr_base": r.geometry.wr_base}
                  for (fp, rg), r in self.source.rings.items() if r.shared]
        _send(gw, self._send_lock, {"t": "HELLO", "incarnation": uuid.uuid4().hex, "models": models})
        add = getattr(self.source, "add_ring_listener", None)
        if add is not None:
            add(lambda r: r.add_free_listener(lambda: self._notify_free(r)))
        self._reader = threading.Thread(target=self._read_loop, name="pgw-maskring-broker", daemon=True)
        self._reader.start()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _notify_free(self, ring: MaskRing) -> None:
        sock = self._sock
        if sock is not None and not self._stopping:  # from the wiper thread: never block it
            _send(sock, self._send_lock, {"t": "FREE", "fp": ring.fingerprint, "ring": ring.ring.name}, nowait=True)

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the child to stop, then kill and reap it; drop every slab it was still filling.

        Order: BYE, wait for the child, shut the channel down and join the reader (no grant or
        publish can be in flight after that), kill and reap, release the device lock. Slabs the
        reader was still copying are protected by their slab lock when the rings close later.
        """
        self._stopping = True
        sock = self._sock
        if sock is not None:
            with contextlib.suppress(OSError, MaskStoreError):
                _send(sock, self._send_lock, {"t": "BYE"}, nowait=True)
        proc = self._proc
        if proc is not None:
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                pass
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout)
            if self._reader.is_alive():  # pragma: no cover - a handler stuck in a syscall
                log.error("ring broker reader did not stop within %.0fs", timeout)
        self._kill_and_reap()
        if sock is not None:
            sock.close()
        lock, self.device_lock = self.device_lock, None
        if lock is not None:
            lock.release()

    def _kill_and_reap(self) -> None:
        with self._reap_lock:
            self._kill_and_reap_locked()

    def _kill_and_reap_locked(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            if self._pidfd is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    signal.pidfd_send_signal(self._pidfd, signal.SIGKILL)
            else:  # pragma: no cover
                proc.kill()
        if proc is not None:
            self.exit_code = proc.wait()
        if self._pidfd is not None:
            with contextlib.suppress(OSError):
                os.close(self._pidfd)
            self._pidfd = None
        # Only now (child reaped: no writer can remain) drop the slabs it had been granted.
        with self._glock:
            grants, self._grants = self._grants, {}
            self._grants_open = False
        for (_, _, idx, gen), ring in grants.items():
            ring.abort_shared(idx, gen)

    # ---- protocol --------------------------------------------------------------------------------
    def _read_loop(self) -> None:
        sock = self._sock
        assert sock is not None
        while True:
            try:
                msg, fds = _recv(sock)
            except MaskStoreError:
                self.errors += 1
                log.error("factory child sent a malformed message; stopping it")
                break
            except OSError:
                break
            for fd in fds:  # the child never sends fds
                os.close(fd)
            if msg is None:
                break
            try:
                self._handle(sock, msg)
            except Exception:  # pragma: no cover - keep serving; the child gets an error reply
                log.exception("ring broker failed to handle %s", msg.get("t"))
        if not self._stopping:
            log.error("mask factory child closed its channel; killing and reaping it")
        self._kill_and_reap()

    def _ring(self, msg: dict[str, Any]) -> MaskRing:
        r = self.source.ring_for(str(msg.get("fp", "")), str(msg.get("ring", "")))
        if r is None or not r.shared:
            raise ConfigError("no shared mask ring for this model")
        return r  # type: ignore[no-any-return]

    def _reply(self, sock: socket.socket, msg: dict[str, Any], fds: Sequence[int] = ()) -> None:
        with contextlib.suppress(OSError):
            _send(sock, self._send_lock, msg, fds)

    def _handle(self, sock: socket.socket, msg: dict[str, Any]) -> None:
        t, rid = msg["t"], msg.get("rid")
        if t == "READY":
            self.child_dumpable = int(msg.get("dumpable", 1))
            if self.child_dumpable != 0:
                log.error("factory child is dumpable (PR_SET_DUMPABLE failed); killing it")
                self._stopping = True
                self._kill_and_reap()
                return
            self.ready.set()
            return
        try:
            if t == "LEVELS":
                r = self.source.ring_for(str(msg.get("fp", "")), str(msg.get("ring", "")))
                lv = None
                if r is not None and r.shared:
                    c = r.counts()
                    lv = [c["available_slots"], r.capacity_slots, c["free_slabs"], r.segment_slots]
                self._reply(sock, {"t": "LEVELS", "rid": rid, "levels": lv})
            elif t == "GRANT_REQ":
                r = self._ring(msg)
                idx, gen, fd = r.grant_shared(r.fingerprint, r.ring, msg["specs"], slots=int(msg["slots"]),
                                              checks=int(msg["checks"]), chunk_slots=int(msg["chunk_slots"]))
                with self._glock:
                    late = not self._grants_open
                    if not late:
                        self._grants[(r.fingerprint, r.ring.name, idx, gen)] = r
                if late:  # the child was reaped while this grant was made: never leave it FILLING
                    r.abort_shared(idx, gen)
                    raise MaskStoreError("factory child is gone")
                self._reply(sock, {"t": "GRANT", "rid": rid, "slab": idx, "gen": gen,
                                   "slab_bytes": r.geometry.slab_bytes, "wr_base": r.geometry.wr_base}, [fd])
            elif t == "PUBLISH":
                r = self._ring(msg)
                idx, gen = int(msg["slab"]), int(msg["gen"])
                with self._glock:
                    self._grants.pop((r.fingerprint, r.ring.name, idx, gen), None)
                sid = r.publish_shared(idx, gen, slots=int(msg["slots"]), checks=int(msg["checks"]),
                                       chunk_slots=int(msg["chunk_slots"]), factory_id=str(msg.get("factory_id", "")))
                self.published += 1
                self._reply(sock, {"t": "OK", "rid": rid, "segment_id": sid})
            elif t == "ABORT":
                r = self._ring(msg)
                idx, gen = int(msg["slab"]), int(msg["gen"])
                with self._glock:
                    self._grants.pop((r.fingerprint, r.ring.name, idx, gen), None)
                r.abort_shared(idx, gen)
            else:
                raise MaskStoreError(f"unknown ring control message {t!r}")
        except RingFull:
            self._reply(sock, {"t": "ERR", "rid": rid, "kind": "full", "msg": "no free slab"})
        except PGWError as exc:
            self.errors += 1
            kind = "reuse" if isinstance(exc, MaskReuseError) else "config" if isinstance(exc, ConfigError) else "store"
            self._reply(sock, {"t": "ERR", "rid": rid, "kind": kind, "msg": exc.message[:300]})
        except (KeyError, TypeError, ValueError) as exc:
            self.errors += 1
            self._reply(sock, {"t": "ERR", "rid": rid, "kind": "config", "msg": f"bad request: {type(exc).__name__}"})
        except Exception as exc:  # never leave the child waiting
            self.errors += 1
            log.exception("ring broker failed to handle %s", t)
            self._reply(sock, {"t": "ERR", "rid": rid, "kind": "store", "msg": type(exc).__name__})


#: Environment the factory child inherits besides ``PGW_FACTORY_*`` (runtime only, never secrets).
CHILD_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH",
                  "PGW_LOG_LEVEL", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
CHILD_ENV_PREFIXES = ("PGW_FACTORY_", "CUDA_", "NVIDIA_", "PYTORCH_")


def child_env(base: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """The factory child's environment: ``PGW_FACTORY_*``, GPU runtime variables and the basic
    process variables of ``base`` (default ``os.environ``) only, so gateway-only secrets (admin
    token, API keys, KEKs) never reach it. ``extra`` adds defaults that ``base`` does not set."""
    src = dict(os.environ if base is None else base)
    env = {k: v for k, v in src.items() if k in CHILD_ENV_KEYS or k.startswith(CHILD_ENV_PREFIXES)}
    for k, v in extra.items():
        env.setdefault(k, v)
    return env


# =================================================================================================
# child side
# =================================================================================================
class _Channel:
    """Child end: RPCs (one at a time per caller thread) and FREE / BYE notifications."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.lock = threading.Lock()
        self.hello: dict[str, Any] | None = None
        self._waiting: dict[str, queue.SimpleQueue[tuple[dict[str, Any], list[int]]]] = {}
        self._wlock = threading.Lock()
        self.free_listeners: list[Callable[[], None]] = []
        self.closed = threading.Event()
        self.on_close: list[Callable[[], None]] = []
        self._hello = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pgw-ring-child", daemon=True)
        self._thread.start()

    def wait_hello(self, timeout: float = 30.0) -> dict[str, Any]:
        if not self._hello.wait(timeout) or self.hello is None:
            raise MaskStoreError("no HELLO from the gateway")
        return self.hello

    def _loop(self) -> None:
        try:
            while True:
                msg, fds = _recv(self.sock)
                if msg is None or msg["t"] == "BYE":
                    break
                if msg["t"] == "HELLO":
                    self.hello = msg
                    self._hello.set()
                elif msg["t"] == "FREE":
                    for fn in list(self.free_listeners):
                        fn()
                else:
                    with self._wlock:
                        q = self._waiting.pop(str(msg.get("rid")), None)
                    if q is None:
                        for fd in fds:
                            os.close(fd)
                    else:
                        q.put((msg, fds))
        except (OSError, MaskStoreError):
            pass
        finally:
            self.closed.set()
            self._hello.set()
            for fn in list(self.on_close):
                with contextlib.suppress(Exception):
                    fn()
            with self._wlock:
                waiting, self._waiting = self._waiting, {}
            for q in waiting.values():
                q.put(({"t": "ERR", "kind": "closed", "msg": "gateway closed the channel"}, []))

    def send(self, msg: dict[str, Any]) -> None:
        _send(self.sock, self.lock, msg)

    def rpc(self, msg: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
        if self.closed.is_set():
            raise MaskStoreError("gateway closed the ring channel")
        rid = uuid.uuid4().hex
        q: queue.SimpleQueue[tuple[dict[str, Any], list[int]]] = queue.SimpleQueue()
        with self._wlock:
            self._waiting[rid] = q
        self.send({**msg, "rid": rid})
        try:
            return q.get(timeout=_RPC_TIMEOUT_S)
        except queue.Empty:
            with self._wlock:
                self._waiting.pop(rid, None)
            raise MaskStoreError("gateway did not answer a ring request") from None


def _raise_err(msg: dict[str, Any]) -> None:
    kind, text = msg.get("kind"), str(msg.get("msg", "ring request refused"))
    if kind == "full":
        raise RingFull(text)
    if kind == "reuse":
        raise MaskReuseError(text)
    if kind == "config":
        raise ConfigError(text)
    raise MaskStoreError(text)


class RemoteRingSink:
    """The :class:`pgw.factory.sinks.RingSink` interface over the channel to the gateway."""

    kind = "ring"

    def __init__(self, chan: _Channel, *, allow_unlocked: bool = False):
        self.chan = chan
        self.allow_unlocked = allow_unlocked
        self.open_writers: set[RemoteSegmentWriter] = set()
        chan.on_close.append(self._wipe_open)

    def _wipe_open(self) -> None:
        for w in list(self.open_writers):
            w._drop(wipe=True)

    def levels(self, fingerprint: str, ring: RingParams) -> tuple[int, int, int, int] | None:
        msg, _ = self.chan.rpc({"t": "LEVELS", "fp": fingerprint, "ring": ring.name})
        lv = msg.get("levels")
        return None if lv is None else (int(lv[0]), int(lv[1]), int(lv[2]), int(lv[3]))

    def available_slots(self, fingerprint: str, ring: RingParams) -> int:
        lv = self.levels(fingerprint, ring)
        return 0 if lv is None else lv[0]

    def add_free_listener(self, fn: Callable[[], None]) -> None:
        self.chan.free_listeners.append(fn)

    def default_levels(self) -> tuple[int, int] | None:
        return None

    def writer(self, fingerprint: str, ring: RingParams, specs: Sequence[LinearSpec], *, slots: int, seed: bytes,
               checks: int = 2, chunk_slots: int = 256, factory_id: str = "local",
               destination: str | None = None) -> RemoteSegmentWriter:
        if destination is not None:
            raise ConfigError("ring segments are never exported (destination is not allowed)")
        if not isinstance(seed, bytes | bytearray) or len(seed) != SEED_BYTES or _all_zero(seed):
            raise MaskReuseError("refusing a missing or all-zero segment seed")
        rp = ring_from_setting(ring)
        msg, fds = self.chan.rpc({"t": "GRANT_REQ", "fp": fingerprint, "ring": rp.name, "specs": _spec_sig(specs),
                                  "slots": int(slots), "checks": int(checks), "chunk_slots": int(chunk_slots)})
        if msg["t"] != "GRANT":
            for fd in fds:
                os.close(fd)
            _raise_err(msg)
        if len(fds) != 1:
            for fd in fds:
                os.close(fd)
            raise MaskStoreError("ring grant without a slab")
        try:
            mapping = FdMapping(fds[0], int(msg["slab_bytes"]), writable=True, allow_unlocked=self.allow_unlocked)
        except BaseException:
            os.close(fds[0])
            self.chan.send({"t": "ABORT", "fp": fingerprint, "ring": rp.name, "slab": msg["slab"], "gen": msg["gen"]})
            raise
        os.close(fds[0])  # the mapping is enough; the fd is never kept
        w = RemoteSegmentWriter(self, mapping, fingerprint=fingerprint, ring=rp, specs=list(specs), slots=int(slots),
                                seed=seed, checks=int(checks), chunk_slots=min(int(chunk_slots), int(slots)),
                                factory_id=factory_id, slab=int(msg["slab"]), gen=int(msg["gen"]),
                                wr_base=int(msg["wr_base"]))
        self.open_writers.add(w)
        return w


class RemoteSegmentWriter:
    """``SegmentWriter`` contract into a granted shared slab (same layout as v1)."""

    def __init__(self, sink: RemoteRingSink, mapping: FdMapping, *, fingerprint: str, ring: RingParams,
                 specs: list[LinearSpec], slots: int, seed: bytes, checks: int, chunk_slots: int, factory_id: str,
                 slab: int, gen: int, wr_base: int):
        self._sink = sink
        self._map: FdMapping | None = mapping
        self.fingerprint = fingerprint
        self.ring = ring
        self.slots = slots
        self.checks = checks
        self.chunk_slots = chunk_slots
        self.factory_id = factory_id
        self.slab, self.gen = slab, gen
        self._seed = bytearray(seed)
        self._lin = {s.id.key: (i, s) for i, s in enumerate(specs)}
        bpe = ring.bytes_per_element
        self._wr_off, off = [], wr_base
        for s in specs:
            self._wr_off.append(off)
            off += slots * s.out_features * bpe
        self._uw: dict[str, bytearray] = {}
        self._done = dict.fromkeys(self._lin, 0)
        self._state = "open"
        self._lock = threading.Lock()

    def write_linear(self, lid: LinearId | str, wr: torch.Tensor, uw: torch.Tensor) -> None:
        key = str(lid)
        with self._lock:
            if self._state != "open" or self._map is None:
                raise MaskStoreError(f"ring segment writer is {self._state}")
            if key not in self._lin:
                raise ConfigError(f"linear {key} is not part of this segment")
            li, s = self._lin[key]
            rows = int(wr.shape[0]) if wr.ndim == 2 else 0
            if rows <= 0 or self._done[key] + rows > self.slots:
                raise ShapeError(f"bad W r chunk for {key}")
            a = _as_ring_array(wr, (rows, s.out_features), self.ring, f"W r of {key}")
            ua = _as_ring_array(uw, (self.checks, s.in_features), self.ring, f"uW of {key}")
            packed = bytearray(self.checks * s.in_features * self.ring.bytes_per_element)
            if len(packed):
                _pack_into(memoryview(packed), ua, self.ring)
            prev = self._uw.get(key)
            if prev is None:
                self._uw[key] = packed
            elif not hmac.compare_digest(prev, packed):
                raise MaskStoreError(f"uW of {key} changed between chunks")
            bpe = self.ring.bytes_per_element
            mv = self._map.view(self._wr_off[li] + self._done[key] * s.out_features * bpe, rows * s.out_features * bpe)
            try:
                _pack_into(mv, a, self.ring)
            finally:
                mv.release()
            self._done[key] += rows

    def commit(self) -> str:
        with self._lock:
            if self._state != "open" or self._map is None:
                raise MaskStoreError(f"ring segment writer is {self._state}")
            if any(n != self.slots for n in self._done.values()):
                self._abort_locked()
                raise MaskStoreError("segment incomplete")
            secrets = bytearray(self._seed)
            for key in self._lin:
                secrets += self._uw[key]
            mv = self._map.view(0, len(secrets))
            mv[:] = secrets
            mv.release()
            _zero(secrets)
            self._state = "published"
            self._drop(wipe=False)  # unmap before PUBLISH: the gateway's write seal needs no writable map
        msg, _ = self._sink.chan.rpc({"t": "PUBLISH", "fp": self.fingerprint, "ring": self.ring.name,
                                      "slab": self.slab, "gen": self.gen, "slots": self.slots, "checks": self.checks,
                                      "chunk_slots": self.chunk_slots, "factory_id": self.factory_id})
        if msg["t"] != "OK":
            _raise_err(msg)
        return str(msg["segment_id"])

    def _abort_locked(self) -> None:
        self._state = "aborted"
        self._drop(wipe=True)
        with contextlib.suppress(OSError, MaskStoreError):
            self._sink.chan.send({"t": "ABORT", "fp": self.fingerprint, "ring": self.ring.name, "slab": self.slab,
                                  "gen": self.gen})

    def abort(self) -> None:
        with self._lock:
            if self._state == "open":
                self._abort_locked()

    def _drop(self, *, wipe: bool) -> None:
        m, self._map = self._map, None
        if m is not None:
            m.close(wipe=wipe)
        _zero(self._seed)
        for b in self._uw.values():
            _zero(b)
        self._uw.clear()
        self._sink.open_writers.discard(self)


def _harden_child() -> int:
    harden_process()
    lib = _lib()
    lib.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    return int(lib.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0))


def serve(fd: int, build: Callable[[dict[str, Any]], tuple[Any, dict[tuple[str, str], Any]]], *,
          ring_low_fraction: float = 0.5, allow_unlocked: bool = False) -> int:
    """Child main loop: harden, read HELLO, build ``(settings, generators)``, run the scheduler.

    ``build(hello)`` returns the factory settings and generators (tests inject fakes)."""
    from pgw.factory.scheduler import FactoryScheduler

    dumpable = _harden_child()
    sock = socket.socket(fileno=fd)
    chan = _Channel(sock)
    hello = chan.wait_hello()
    settings, gens = build(hello)
    sink = RemoteRingSink(chan, allow_unlocked=allow_unlocked)
    sch = FactoryScheduler(settings, None, gens, ring_sink=sink, ring_low_fraction=ring_low_fraction, metrics=False)
    stop = threading.Event()
    chan.on_close.append(stop.set)
    chan.send({"t": "READY", "dumpable": dumpable})
    if chan.closed.is_set():
        return 0
    try:
        sch.run_forever(stop)
    finally:
        sink._wipe_open()
    return 0


def _build_from_env(hello: dict[str, Any]) -> tuple[Any, dict[tuple[str, str], Any]]:
    from pgw.factory.config import FactorySettings
    from pgw.factory.governor import Throttle
    from pgw.factory.runtime import build_generators, open_packs

    settings = FactorySettings()
    gens = build_generators(settings, open_packs(settings), Throttle(settings.max_slots_per_hour, settings.duty_cycle))
    wanted = {(m["fp"], m["ring"]) for m in hello.get("models", [])}
    return settings, {k: g for k, g in gens.items() if k in wanted}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PGW mask factory child filling the gateway's shared mask rings")
    ap.add_argument("--ring-fd", type=int, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("PGW_LOG_LEVEL", "INFO"))
    return serve(args.ring_fd, _build_from_env)


if __name__ == "__main__":  # pragma: no cover - child process entry
    sys.exit(main())

