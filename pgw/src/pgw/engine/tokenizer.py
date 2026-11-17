"""Tokenizer wrapper with chat templates and incremental detokenization."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError

__all__ = ["ChatTokenizer", "IncrementalDecoder"]

log = logging.getLogger(__name__)


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode`` (byte-level BPE token characters -> raw bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_BYTE_DECODER = _byte_decoder()


def _is_byte_level(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` is a byte-level BPE (GPT-2 / Llama-3 / Qwen style) fast tokenizer."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return False
    try:
        spec = json.loads(backend.to_str())
    except Exception:
        return False
    return "ByteLevel" in json.dumps(spec.get("decoder")) or "ByteLevel" in json.dumps(spec.get("pre_tokenizer"))


def _ids_of(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


class ChatTokenizer:
    """Wraps the HF tokenizer shipped in the model pack (``<pack>/tokenizer``).

    Args:
        tokenizer_dir: Directory with ``tokenizer.json`` / ``tokenizer_config.json`` (and optionally
            ``generation_config.json`` / ``config.json`` for extra EOS ids).
        extra_eos_token_ids: Additional EOS ids (e.g. from the pack manifest ``arch.eos_token_ids``).
        tokenizer: Use an already loaded HF tokenizer instead of loading from ``tokenizer_dir``.
    """

    def __init__(
        self,
        tokenizer_dir: str | Path | None,
        *,
        extra_eos_token_ids: Sequence[int] = (),
        tokenizer: Any = None,
    ):
        if tokenizer is None:
            if tokenizer_dir is None:
                raise ConfigError("ChatTokenizer needs a tokenizer directory or a tokenizer object")
            path = Path(tokenizer_dir)
            if not path.is_dir():
                raise ConfigError(f"tokenizer directory {path} does not exist")
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency
                raise ConfigError("the 'transformers' package is required for tokenization") from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(path))
            except Exception as exc:
                raise ConfigError(f"cannot load tokenizer from {path.name}: {type(exc).__name__}") from exc
        self._tok = tokenizer
        eos: list[int] = []
        eos += _ids_of(getattr(tokenizer, "eos_token_id", None))
        if tokenizer_dir is not None:
            for fname in ("generation_config.json", "config.json"):
                f = Path(tokenizer_dir) / fname
                if f.is_file():
                    try:
                        eos += _ids_of(json.loads(f.read_text(encoding="utf-8")).get("eos_token_id"))
                    except (ValueError, OSError):
                        log.warning("ignoring unreadable %s", fname)
        eos += [int(e) for e in extra_eos_token_ids]
        self._eos = tuple(dict.fromkeys(eos))
        self._special = frozenset(int(i) for i in getattr(tokenizer, "all_special_ids", []) or [])
        self._byte_level: bool | None = None

    @classmethod
    def from_pack(cls, pack: Any) -> ChatTokenizer:
        """Tokenizer of a model pack (adds the manifest EOS ids)."""
        tdir = getattr(pack, "tokenizer_dir", None)
        if tdir is None:
            raise ConfigError("pack has no tokenizer")
        return cls(tdir, extra_eos_token_ids=tuple(getattr(pack.arch, "eos_token_ids", ())))

    @property
    def hf(self) -> Any:
        """The underlying HF tokenizer."""
        return self._tok

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """All end-of-sequence ids (pack manifest + tokenizer config)."""
        return self._eos

    @property
    def special_ids(self) -> frozenset[int]:
        """Ids of special tokens."""
        return self._special

    def __len__(self) -> int:
        return len(self._tok)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Text -> ids."""
        return [int(i) for i in self._tok.encode(text, add_special_tokens=add_special_tokens)]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Ids -> text."""
        return str(self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens))

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of one token (may be part of a multi-byte UTF-8 character).

        Byte-level BPE vocabularies map through GPT-2's byte alphabet, SentencePiece byte-fallback
        tokens (``<0xNN>``) give their byte and ``▁`` a space. Special/added tokens and other
        vocabularies use the token's decoded text.
        """
        tid = int(token_id)
        convert = getattr(self._tok, "convert_ids_to_tokens", None)
        piece = convert(tid) if convert is not None and tid not in self._special else None
        if isinstance(piece, str) and piece:
            if self._byte_level is None:
                self._byte_level = _is_byte_level(self._tok)
            if self._byte_level and all(ch in _BYTE_DECODER for ch in piece):
                return bytes(_BYTE_DECODER[ch] for ch in piece)
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    return bytes([int(piece[3:5], 16)])
                except ValueError:
                    pass
            if not self._byte_level and "▁" in piece:
                return piece.replace("▁", " ").encode("utf-8")
        return self.decode([tid], skip_special_tokens=False).encode("utf-8")

    def render_chat(self, messages: Sequence[dict[str, Any]], add_generation_prompt: bool = True, **kwargs: Any) -> str:
        """Chat template as text."""
        if not getattr(self._tok, "chat_template", None):
            raise ConfigError("tokenizer has no chat template")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m:
                raise ConfigError("each chat message needs a 'role'")
        try:
            out = self._tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
            )
        except Exception as exc:
            raise ConfigError(f"chat template failed: {type(exc).__name__}: {exc}") from exc
        return str(out)

    def apply_chat_template(
        self, messages: Sequence[dict[str, Any]], add_generation_prompt: bool = True, **kwargs: Any
    ) -> list[int]:
        """OpenAI-style ``[{"role", "content"}]`` -> prompt ids using the model's chat template."""
        return self.encode(self.render_chat(messages, add_generation_prompt, **kwargs), add_special_tokens=False)

    def incremental_decoder(self, skip_special_tokens: bool = True) -> IncrementalDecoder:
        """New streaming detokenizer (handles multi-byte UTF-8 split across tokens)."""
        return IncrementalDecoder(self, skip_special_tokens=skip_special_tokens)


class IncrementalDecoder:
    """Emits only text that is final (no partial UTF-8 sequences).

    Decodes a sliding window of tokens and emits the difference, so that tokenizers whose decoding
    depends on neighbours (SentencePiece spaces, byte-level BPE split characters) stream correctly.
    """

    def __init__(self, tokenizer: ChatTokenizer, *, skip_special_tokens: bool = True):
        self._tok = tokenizer
        self._skip = skip_special_tokens
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def _dec(self, ids: Sequence[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=self._skip)

    def push(self, token_id: int) -> str:
        """Add a token, return newly finalized text (possibly empty)."""
        self._ids.append(int(token_id))
        prefix_text = self._dec(self._ids[self._prefix : self._read])
        new_text = self._dec(self._ids[self._prefix :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            delta = new_text[len(prefix_text) :]
            self._prefix = self._read
            self._read = len(self._ids)
            return delta
        return ""

    def flush(self) -> str:
        """Return any remaining text (incomplete UTF-8 becomes U+FFFD)."""
        prefix_text = self._dec(self._ids[self._prefix : self._read])
        new_text = self._dec(self._ids[self._prefix :])
        self._prefix = self._read = len(self._ids)
        return new_text[len(prefix_text) :] if len(new_text) > len(prefix_text) else ""
