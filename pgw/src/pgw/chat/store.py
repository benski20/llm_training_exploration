"""Local conversation history of ``pgw chat``.

One JSON file per conversation under ``<state_dir>/chat/conversations/`` (the SDK's state directory), the directory
0700 and every file 0600, written atomically. Nothing here is ever sent anywhere: the history exists so the user
can come back to a conversation on this machine. Files are **not encrypted** beyond the operating system's
permissions and the disk's own encryption (FileVault, BitLocker, LUKS): a key kept next to them would add nothing
against someone who can read the user's files. :class:`MemoryStore` keeps conversations in RAM only
(``pgw chat --no-history``).

Conversation ids are ``c_`` + 24 hex digits drawn here; any other id is refused before a path is built from it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from pgw.core.fileio import atomic_write_text

__all__ = ["ConversationStore", "DiskStore", "MemoryStore", "new_id", "valid_id"]

_ID_RE = re.compile(r"c_[0-9a-f]{24}")
MAX_TITLE = 80
#: the largest conversation file read back (a bigger file is ignored, never parsed)
MAX_FILE_BYTES = 16 << 20


def new_id() -> str:
    return "c_" + secrets.token_hex(12)


def new_message_id() -> str:
    return "m_" + secrets.token_hex(8)


def valid_id(cid: object) -> bool:
    return isinstance(cid, str) and _ID_RE.fullmatch(cid) is not None


def title_from(text: str) -> str:
    """A short title from the first user message (local only)."""
    line = " ".join(text.split())
    return line if len(line) <= MAX_TITLE else line[: MAX_TITLE - 1].rstrip() + "…"


def _summary(conv: dict[str, Any]) -> dict[str, Any]:
    return {"id": conv["id"], "title": conv.get("title") or "New chat", "model": conv.get("model"),
            "created_at": conv.get("created_at"), "updated_at": conv.get("updated_at"),
            "messages": len(conv.get("messages") or [])}


class ConversationStore:
    """Conversations as plain dicts: ``{"id", "title", "model", "created_at", "updated_at", "messages": [
    {"id", "role": "user"|"assistant", "content", "created_at", ...}]}``. Thread-safe."""

    persistent = False

    def __init__(self) -> None:
        self._mu = threading.RLock()

    def _load(self, cid: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def _save(self, conv: dict[str, Any]) -> None:
        raise NotImplementedError

    def _remove(self, cid: str) -> bool:
        raise NotImplementedError

    def _all(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def summaries(self) -> list[dict[str, Any]]:
        with self._mu:
            convs = [_summary(c) for c in self._all()]
        return sorted(convs, key=lambda c: float(c.get("updated_at") or 0.0), reverse=True)

    def get(self, cid: str) -> dict[str, Any] | None:
        if not valid_id(cid):
            return None
        with self._mu:
            return self._load(cid)

    def create(self, model: str) -> dict[str, Any]:
        now = time.time()
        conv = {"id": new_id(), "title": "", "model": model, "created_at": now, "updated_at": now, "messages": []}
        with self._mu:
            self._save(conv)
        return conv

    def append(self, cid: str, message: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
        """Append ``message`` (``id`` and ``created_at`` are filled in) and return it."""
        with self._mu:
            conv = self._load(cid)
            if conv is None:
                raise KeyError(cid)
            msg = {"id": new_message_id(), "created_at": time.time(), **message}
            conv["messages"].append(msg)
            if not conv.get("title") and msg.get("role") == "user":
                conv["title"] = title_from(str(msg.get("content") or ""))
            if model:
                conv["model"] = model
            conv["updated_at"] = msg["created_at"]
            self._save(conv)
            return msg

    def truncate_to_last_user(self, cid: str) -> None:
        """Drop the answers after the last user message (to answer it again)."""
        with self._mu:
            conv = self._load(cid)
            if conv is None:
                raise KeyError(cid)
            msgs = conv["messages"]
            while msgs and msgs[-1].get("role") != "user":
                msgs.pop()
            self._save(conv)

    def delete(self, cid: str) -> bool:
        if not valid_id(cid):
            return False
        with self._mu:
            return self._remove(cid)

    def delete_all(self) -> int:
        with self._mu:
            return sum(int(self._remove(c["id"])) for c in self._all())


class MemoryStore(ConversationStore):
    """Conversations in this process's memory only (gone when ``pgw chat`` exits)."""

    def __init__(self) -> None:
        super().__init__()
        self._convs: dict[str, dict[str, Any]] = {}

    def _load(self, cid: str) -> dict[str, Any] | None:
        c = self._convs.get(cid)
        return json.loads(json.dumps(c)) if c is not None else None

    def _save(self, conv: dict[str, Any]) -> None:
        self._convs[conv["id"]] = json.loads(json.dumps(conv))

    def _remove(self, cid: str) -> bool:
        return self._convs.pop(cid, None) is not None

    def _all(self) -> list[dict[str, Any]]:
        return list(self._convs.values())


class DiskStore(ConversationStore):
    """Conversations as 0600 JSON files in a 0700 directory (``<state_dir>/chat/conversations``)."""

    persistent = True

    def __init__(self, root: Path):
        super().__init__()
        from pgw.sdk.cache import ensure_private_dir

        self.root = ensure_private_dir(Path(root))

    def _path(self, cid: str) -> Path:
        if not valid_id(cid):  # never build a path from anything else
            raise ValueError("invalid conversation id")
        return self.root / f"{cid}.json"

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            st = os.lstat(path)
            if not os.path.isfile(path) or os.path.islink(path) or st.st_size > MAX_FILE_BYTES:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not valid_id(data.get("id")) or not isinstance(data.get("messages"), list):
            return None
        return data

    def _load(self, cid: str) -> dict[str, Any] | None:
        data = self._read(self._path(cid))
        return data if data is not None and data["id"] == cid else None

    def _save(self, conv: dict[str, Any]) -> None:
        atomic_write_text(self._path(conv["id"]), json.dumps(conv, ensure_ascii=False) + "\n", mode=0o600)

    def _remove(self, cid: str) -> bool:
        try:
            self._path(cid).unlink()
            return True
        except FileNotFoundError:
            return False

    def _all(self) -> list[dict[str, Any]]:
        out = []
        with contextlib.suppress(OSError):
            for f in self.root.iterdir():
                if f.suffix == ".json" and valid_id(f.stem):
                    c = self._read(f)
                    if c is not None and c["id"] == f.stem:
                        out.append(c)
        return out
