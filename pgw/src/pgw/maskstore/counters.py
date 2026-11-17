"""Rollback witnesses for the mask store (SPEC.md section 7.6).

Restoring an old ``state.sqlite`` (backup restore, VM snapshot revert, a malicious admin) would make
already-consumed slots look fresh, which means mask reuse. The store therefore mirrors its global
``generation`` and ``consumed_total`` into witnesses that live outside the database:

``GenerationJournal``
    Append-only, HMAC-chained journal (``<root>/generation.journal`` by default; put it on a
    different volume for extra safety). Every record is
    ``HMAC(prev_mac || store_id || generation || consumed_total)`` under a key derived from the KEK,
    so it cannot be forged without the KEK. It detects restoring the database alone. Restoring the
    database *and* the journal from the same snapshot is only detectable with a monotonic counter.

``MonotonicCounter`` implementations
    ``FileMonotonicCounter`` (a counter file, ideally on separate storage), ``TpmNvCounter`` (TPM 2.0
    NV counter via ``tpm2-tools``; stub-quality), ``ProviderCounter`` (the key provider's hardware
    counter, e.g. an HSM). The store increments the counter once per generation bump.

Ordering (normative): the database transaction commits first, then witnesses are advanced, then
masks are handed out, all while the store's inter-process lock is held. A crash in between leaves
the database at most one generation *ahead* of the witnesses, which is safe (ahead means more
consumed) and is resynchronised at the next start. A database *behind* its witnesses means
rollback: the store refuses to serve.
"""

from __future__ import annotations

import abc
import hmac
import logging
import os
import shutil
import struct
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pgw.core.errors import ConfigError, MaskStoreError

__all__ = [
    "FileMonotonicCounter",
    "GenerationJournal",
    "JournalRecord",
    "MonotonicCounter",
    "ProviderCounter",
    "TpmNvCounter",
]

log = logging.getLogger(__name__)

MacFn = Callable[[bytes], bytes | None]


def fsync_dir(path: Path) -> None:
    """fsync a directory so renames/creations inside it are durable."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes, *, fsync: bool = True, mode: int = 0o600) -> None:
    """Write ``data`` to ``path`` atomically (temp file + fsync + rename + dir fsync)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, data)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if fsync:
        fsync_dir(path.parent)


# ---- monotonic counters -----------------------------------------------------------------------------
class MonotonicCounter(abc.ABC):
    """A counter that can only go up and lives outside the state database."""

    #: Short name for logs and status output.
    kind: str = "counter"

    @abc.abstractmethod
    def read(self) -> int:
        """Current value."""

    @abc.abstractmethod
    def increment(self) -> int:
        """Increment by one and return the new value."""

    @property
    def hardware(self) -> bool:
        """True if rollback of the counter itself requires tampering with hardware."""
        return False


class FileMonotonicCounter(MonotonicCounter):
    """Counter in a small file (u64 value + CRC-free magic), written atomically.

    Place it on storage that is not snapshotted together with the store (e.g. a different volume,
    or a host path when the store is in a container volume). Not tamper-proof against an attacker
    who can write that file, but it defeats accidental restores of the store directory.
    """

    kind = "file"
    _MAGIC = b"PGWCTR1\0"

    def __init__(self, path: str | Path, *, fsync: bool = True):
        self.path = Path(path)
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> int:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise MaskStoreError(f"cannot read monotonic counter file: {exc.strerror}") from exc
        if len(data) != 16 or data[:8] != self._MAGIC:
            raise MaskStoreError("monotonic counter file is corrupt", details={"path": str(self.path)})
        return int(struct.unpack(">Q", data[8:])[0])

    def increment(self) -> int:
        v = self.read() + 1
        atomic_write(self.path, self._MAGIC + struct.pack(">Q", v), fsync=self.fsync)
        return v


class ProviderCounter(MonotonicCounter):
    """Uses the key provider's hardware counter (:meth:`KeyProvider.monotonic_counter`)."""

    kind = "provider"

    def __init__(self, provider: object):
        self._p = provider
        if getattr(provider, "monotonic_counter", lambda: None)() is None:
            raise ConfigError(f"{type(provider).__name__} has no hardware monotonic counter")

    @property
    def hardware(self) -> bool:
        return True

    def read(self) -> int:
        v = self._p.monotonic_counter()  # type: ignore[attr-defined]
        if v is None:
            raise MaskStoreError("key provider counter became unavailable")
        return int(v)

    def increment(self) -> int:
        v = self._p.increment_counter()  # type: ignore[attr-defined]
        if v is None:
            raise MaskStoreError("key provider counter became unavailable")
        return int(v)


class TpmNvCounter(MonotonicCounter):
    """TPM 2.0 NV counter (``TPM_NT_COUNTER``) via ``tpm2-tools`` (stub-quality).

    The NV index must be defined beforehand, e.g.
    ``tpm2_nvdefine 0x1500016 -C o -s 8 -a "ownerread|ownerwrite|nt=counter"``.

    Args:
        nv_index: NV index (e.g. ``0x1500016``).
        tcti: Optional TCTI string (``device:/dev/tpmrm0``, ``swtpm:port=2321``).
        runner: Injectable command runner (tests); defaults to :func:`subprocess.run`.
    """

    kind = "tpm"

    def __init__(
        self,
        nv_index: int,
        *,
        tcti: str | None = None,
        runner: Callable[[list[str]], bytes] | None = None,
    ):
        self.nv_index = int(nv_index)
        self.tcti = tcti
        if runner is None:
            if shutil.which("tpm2_nvread") is None or shutil.which("tpm2_nvincrement") is None:
                raise ConfigError("TPM counter needs tpm2-tools (tpm2_nvread, tpm2_nvincrement) on PATH")
            runner = self._run
        self._runner = runner

    @property
    def hardware(self) -> bool:
        return True

    def _run(self, cmd: list[str]) -> bytes:  # pragma: no cover - needs a TPM
        env = dict(os.environ)
        if self.tcti:
            env["TPM2TOOLS_TCTI"] = self.tcti
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, timeout=30, env=env)  # noqa: S603
        except (OSError, subprocess.SubprocessError) as exc:
            raise MaskStoreError(f"TPM command {cmd[0]} failed: {type(exc).__name__}") from exc
        return res.stdout

    def read(self) -> int:
        out = self._runner(["tpm2_nvread", hex(self.nv_index), "-C", "o", "-s", "8"])
        if len(out) != 8:
            raise MaskStoreError("unexpected TPM NV counter read length")
        return int.from_bytes(out, "big")

    def increment(self) -> int:
        self._runner(["tpm2_nvincrement", hex(self.nv_index), "-C", "o"])
        return self.read()


# ---- HMAC-chained journal ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class JournalRecord:
    """One authenticated journal entry."""

    generation: int
    consumed: int
    mac: bytes


class GenerationJournal:
    """Append-only HMAC-chained record of ``(generation, consumed_total)``.

    Line format (ASCII): ``<kind> <generation> <consumed> <prev_mac_hex> <mac_hex>\\n`` with
    ``kind`` in ``{"g", "c"}`` (``c`` = checkpoint written by compaction, its ``prev_mac`` is the
    head of the compacted chain).

    Args:
        path: Journal file.
        store_id: Binds records to one store (a journal cannot be swapped between stores).
        mac: Keyed MAC function (the key provider's :meth:`mac`).
        fsync: fsync after each append.
        max_bytes: Compact (rewrite as one checkpoint) when the file grows beyond this.
    """

    _GENESIS = b"\0" * 32

    def __init__(self, path: str | Path, store_id: str, mac: MacFn, *, fsync: bool = True, max_bytes: int = 8 << 20):
        self.path = Path(path)
        self.store_id = store_id
        self._mac = mac
        self.fsync = fsync
        self.max_bytes = max_bytes

    def _compute(self, prev: bytes, gen: int, consumed: int) -> bytes:
        msg = b"pgw/v1/journal\0" + prev + self.store_id.encode() + struct.pack(">QQ", gen, consumed)
        m = self._mac(msg)
        if m is None:
            raise MaskStoreError("key provider cannot MAC; generation journal unavailable")
        return m

    def exists(self) -> bool:
        """Whether the journal file exists."""
        return self.path.exists()

    def _parse(self, line: str, lineno: int) -> tuple[str, int, int, bytes, bytes]:
        parts = line.split()
        try:
            kind, g, c, prev, mac = parts
            if kind not in ("g", "c"):
                raise ValueError(kind)
            return kind, int(g), int(c), bytes.fromhex(prev), bytes.fromhex(mac)
        except ValueError as exc:
            raise MaskStoreError(f"generation journal is corrupt at line {lineno}") from exc

    @property
    def pending_path(self) -> Path:
        """Staged journal written during KEK rotation (promoted once the state DB is re-MACed)."""
        return self.path.with_name(self.path.name + ".next")

    def stage_checkpoint(self, generation: int, consumed: int, prev: bytes) -> None:
        """Write a single-checkpoint journal to :attr:`pending_path` under the *current* MAC key.

        Used by KEK rotation: the staged journal is promoted with :meth:`promote_pending` after
        the state database (re-MACed under the new key) has committed.
        """
        mac = self._compute(prev, generation, consumed)
        line = f"c {generation} {consumed} {prev.hex()} {mac.hex()}\n".encode("ascii")
        atomic_write(self.pending_path, line, fsync=self.fsync)

    def promote_pending(self) -> None:
        """Atomically replace the journal with the staged one."""
        os.replace(self.pending_path, self.path)
        if self.fsync:
            fsync_dir(self.path.parent)

    def recover_pending(self) -> None:
        """Finish or discard a KEK rotation that was interrupted.

        If the main journal verifies under the current key, a staged journal is stale (the rotation
        never committed) and is deleted. Otherwise a staged journal that verifies is promoted (the
        rotation committed but crashed before promotion).
        """
        if not self.pending_path.exists():
            return
        try:
            self.verify()
        except MaskStoreError:
            try:
                self.verify(self.pending_path)
            except MaskStoreError:
                return  # neither verifies: leave both for the operator; verify() will refuse
            log.warning("completing an interrupted KEK rotation: promoting the staged generation journal")
            self.promote_pending()
            return
        log.warning("discarding a staged generation journal from an interrupted KEK rotation")
        self.pending_path.unlink(missing_ok=True)

    def verify(self, path: Path | None = None) -> JournalRecord | None:
        """Verify the full chain; return the head record (``None`` if the journal is empty/missing).

        Args:
            path: Journal file to verify (default: :attr:`path`).

        Raises:
            MaskStoreError: broken chain, forged or reordered records.
        """
        try:
            text = (path or self.path).read_text("ascii")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise MaskStoreError("cannot read generation journal") from exc
        head: JournalRecord | None = None
        prev_mac: bytes | None = None
        last_gen = -1
        last_consumed = -1
        lines = text.splitlines()
        if text and not text.endswith("\n"):
            # torn last write (crash mid-append): ignore the partial line
            log.warning("generation journal has a torn final line; ignoring it")
            lines = lines[:-1]
        for i, line in enumerate(lines, 1):
            kind, g, c, prev, mac = self._parse(line, i)
            if i == 1:
                if kind == "g" and prev != self._GENESIS:
                    raise MaskStoreError("generation journal does not start at genesis")
            else:
                if kind != "g":
                    raise MaskStoreError(f"unexpected checkpoint at journal line {i}")
                if prev != prev_mac:
                    raise MaskStoreError(f"generation journal chain broken at line {i}")
            if not hmac.compare_digest(mac, self._compute(prev, g, c)):
                raise MaskStoreError(f"generation journal record {i} failed authentication (forged or foreign)")
            if g < last_gen or c < last_consumed:
                raise MaskStoreError(f"generation journal goes backwards at line {i}")
            last_gen, last_consumed, prev_mac = g, c, mac
            head = JournalRecord(g, c, mac)
        return head

    def head(self) -> JournalRecord | None:
        """Last record, authenticated individually (fast path; :meth:`verify` checks the chain)."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return None
        window = 4096
        with self.path.open("rb") as f:
            f.seek(max(0, size - window))
            tail = f.read()
        try:
            parts = tail.decode("ascii").split("\n")
        except UnicodeDecodeError as exc:
            raise MaskStoreError("generation journal is corrupt (non-ASCII)") from exc
        complete = parts[:-1]  # the last element is "" (clean end) or a torn partial line
        if size > window:
            complete = complete[1:]  # the first line of the window may be partial
        complete = [ln for ln in complete if ln.strip()]
        if not complete:
            return self.verify()
        _, g, c, prev, mac = self._parse(complete[-1], -1)
        if not hmac.compare_digest(mac, self._compute(prev, g, c)):
            raise MaskStoreError("generation journal head failed authentication")
        return JournalRecord(g, c, mac)

    def _repair_torn_tail(self) -> None:
        """Truncate a torn final line (crash mid-append) so the next append starts cleanly."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size == 0:
            return
        with self.path.open("r+b") as f:
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            f.seek(max(0, size - 4096))
            tail = f.read()
            cut = tail.rfind(b"\n")
            new_size = (max(0, size - 4096) + cut + 1) if cut >= 0 else (0 if size <= 4096 else size)
            f.truncate(new_size)
            os.fsync(f.fileno())
        log.warning("truncated a torn generation journal record")

    def append(self, generation: int, consumed: int) -> JournalRecord:
        """Append a record after the current head."""
        self._repair_torn_tail()
        head = self.head()
        prev = head.mac if head is not None else self._GENESIS
        if head is not None and (generation < head.generation or consumed < head.consumed):
            raise MaskStoreError("refusing to append a journal record that goes backwards")
        mac = self._compute(prev, generation, consumed)
        line = f"g {generation} {consumed} {prev.hex()} {mac.hex()}\n".encode("ascii")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        if head is None and self.fsync:
            fsync_dir(self.path.parent)
        rec = JournalRecord(generation, consumed, mac)
        if self.path.stat().st_size > self.max_bytes:
            self.compact()
        return rec

    def compact(self) -> None:
        """Rewrite the journal as a single checkpoint record chained to the current head."""
        head = self.verify()
        if head is None:
            return
        mac = self._compute(head.mac, head.generation, head.consumed)
        line = f"c {head.generation} {head.consumed} {head.mac.hex()} {mac.hex()}\n".encode("ascii")
        atomic_write(self.path, line, fsync=self.fsync)
        log.info("compacted generation journal at generation %d", head.generation)
