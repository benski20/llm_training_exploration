"""The LM head as a masked remote linear (opt-in; off by default).

A weight-free CPU gateway cannot afford the float LM head (Qwen2.5-7B: 152064 x 3584, 1.09 GB of
bfloat16 read per token, about 25-45 ms on one core). With :class:`MaskedHeadPack` and
:class:`MaskedHead` the head is one more masked linear, ``"<num_layers>.lm_head"`` of shape
``(vocab, hidden)``, computed by the provider exactly like the layers' linears: the gateway sends
``x_hat = xq + r`` of the final-normed hidden row, unmasks ``y_hat - W r``, checks the range and
Freivalds on every row, and rescales. The provider sees one more masked call per pass of public shape.

Exactness: the head's weight is the pack's plain ``lm_head`` (``embed_tokens`` when tied) quantized per
output row to int8 by :func:`pgw.core.ring.quantize_weight_rows` (SPEC 5), a deterministic function of
a tensor the pack fingerprint already covers. The logits are therefore those of the int8 head, not of
the float head: a **new pinned reference** (like ``float_ops = "native"``). The plaintext reference is the
same wrapped pack with :class:`pgw.engine.backends.PlainBackend` and the same :class:`MaskedHead`, and the
masked gateway's logits equal it bit for bit (``tests/engine/test_masked_head.py``).

Binding: :attr:`MaskedHeadPack.fingerprint` is domain-separated from the base pack's
(``SHA-256("pgw/v1/masked-lm-head\\0" || base fingerprint)``, :func:`head_fingerprint`), so masks, segments
and the provider's pack for a head-masking deployment can never be mixed with those of the base pack.

Wiring (all opt-in, default off): the gateway model setting ``head_mode = "provider_int8"``
(:class:`pgw.gateway.config.GatewayModelConfig`; also in the ``plain`` dev mode, which is then the plaintext
reference), the compute service setting ``masked_head`` (``PGW_SERVICE_MASKED_HEAD``: every pack is served a
second time under its derived fingerprint, the layers' kernels shared, the int8 head derived by the provider
from its own verified pack) and the mask factory setting ``masked_head`` (``PGW_FACTORY_MASKED_HEAD``: masks
and check vectors for the head pack; the encrypted factory's hint is extended from the base pack's hint with
the same customer-chosen public seed). Each party derives the fingerprint from the base pack it verified
itself; a service that does not serve the derived fingerprint is refused at connect.

Speculation: verify rows (``last_logits``) are one head call of those rows (each row's logits equal a 1-row
decode's: activations are quantized per row and every row is normed alone), and padded strict passes
(``pad_rows`` / ``fixed_item``) send the head call through the same padding as the layers' calls
(:meth:`pgw.engine.model.TransformerModel._padded`). Not supported: draft heads (``lm_head_fn``).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.ring import quantize_weight_rows
from pgw.core.types import LinearId, LinearKind, LinearSpec

__all__ = ["MaskedHead", "MaskedHeadPack", "enable_masked_head", "head_fingerprint", "head_linear_id",
           "quantize_head"]

_DOMAIN = b"pgw/v1/masked-lm-head\x00"
#: Rows quantized per block (the float64 copy of a 7B head would be 4.4 GB; rows are independent).
_BLOCK_ROWS = 4096


def head_linear_id(num_layers: int) -> LinearId:
    """``LinearId(num_layers, LM_HEAD)``: the masked head's id (``"<num_layers>.lm_head"``)."""
    return LinearId(int(num_layers), LinearKind.LM_HEAD)


def head_fingerprint(base_fingerprint: str) -> str:
    """The masked-head pack's fingerprint, ``SHA-256("pgw/v1/masked-lm-head\\0" || base fingerprint)`` (hex).

    Every party derives it from the base fingerprint it verified itself; it is never taken from the wire."""
    return hashlib.sha256(_DOMAIN + str(base_fingerprint).encode()).hexdigest()


def quantize_head(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`quantize_weight_rows` of a ``(vocab, hidden)`` head in row blocks (bit-identical: every row's
    scale and int8 values depend only on that row). ``w`` may be float32 or bfloat16 (each block is widened
    to float64 exactly either way), so a memory-mapped bfloat16 head is never copied whole."""
    m = int(w.shape[0])
    w8 = torch.empty(tuple(w.shape), dtype=torch.int8)
    sw = torch.empty(m, dtype=torch.float32)
    for a in range(0, m, _BLOCK_ROWS):
        b = min(m, a + _BLOCK_ROWS)
        w8[a:b], sw[a:b] = quantize_weight_rows(w[a:b])
    return w8, sw


class MaskedHeadPack:
    """A pack (``ModelPack`` or ``InMemoryPack``) plus the masked LM head as its last linear.

    Everything not overridden is the base pack's. ``linear_specs()`` appends the head's spec, so the mask
    factory, the store, the masked backend and a pack-driven compute service treat it like any linear.

    Args:
        base: The verified base pack.
        float_head: ``False`` on a gateway whose model uses the masked head: ``load_plain("lm_head")`` of an
            untied head returns an empty ``(0, hidden)`` tensor, so the model never loads the float head
            (2.18 GB of float32 at 7B). The int8 head's scales and row norm are still derived from the pack.
    """

    def __init__(self, base: Any, *, float_head: bool = True):
        if isinstance(base, MaskedHeadPack):
            raise ConfigError("pack already has a masked LM head")
        self._base = base
        arch = base.arch
        self._lid = head_linear_id(arch.num_layers)
        self._spec = LinearSpec(self._lid, int(arch.vocab_size), int(arch.hidden_size), False)
        self._fp = head_fingerprint(base.fingerprint)
        self._lock = threading.Lock()
        self._w: Any = None  # LinearWeights, built on first load_linear (the provider's / factory's side)
        self._meta: Any = None  # LinearMeta (scales, row norm) without the int8 weight (the gateway's side)
        self.float_head = bool(float_head)

    def __getattr__(self, name: str) -> Any:  # name, arch, tokenizer_dir, manifest, close, ...
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def __enter__(self) -> MaskedHeadPack:
        return self

    def __exit__(self, *exc: object) -> None:
        close = getattr(self._base, "close", None)
        if callable(close):
            close()

    def __repr__(self) -> str:
        return f"MaskedHeadPack({self._base!r}, fp={self._fp[:12]})"

    @property
    def base(self) -> Any:
        """The wrapped pack."""
        return self._base

    @property
    def fingerprint(self) -> str:
        return self._fp

    @property
    def base_fingerprint(self) -> str:
        """The wrapped pack's fingerprint (what the manifest and ``config.fingerprint`` pin)."""
        return str(self._base.fingerprint)

    @property
    def short_fingerprint(self) -> str:
        return self._fp[:12]

    @property
    def head_id(self) -> LinearId:
        return self._lid

    def linear_specs(self) -> list[LinearSpec]:
        return [*self._base.linear_specs(), self._spec]

    def load_plain(self, name: str, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> Any:
        if name == "lm_head" and not self.float_head and not self._base.arch.tie_word_embeddings:
            return torch.empty((0, int(self._base.arch.hidden_size)), dtype=dtype, device=device)
        return self._base.load_plain(name, dtype=dtype, device=device)

    def iter_linears(self, device: torch.device | str = "cpu", *, layers: Any = None,
                     kinds: Any = None) -> Iterator[Any]:
        """Every linear in canonical order, the head last (in layer ``num_layers``, kind ``lm_head``)."""
        lay = set(layers) if layers is not None else None
        kin = {LinearKind(k) for k in kinds} if kinds is not None else None
        for spec in self.linear_specs():
            lid = spec.id
            if (lay is None or lid.layer in lay) and (kin is None or lid.kind in kin):
                yield self.load_linear(lid, device)

    def _source(self) -> torch.Tensor:
        """The plain head the int8 head is derived from: a ``ModelPack``'s stored tensor (``lm_head``, or
        ``embed_tokens`` when tied) memory-mapped in its stored dtype and hash-checked against the manifest
        (which the fingerprint covers), else the pack's float32 ``load_plain``."""
        base = self._base
        load_tensor = getattr(base, "load_tensor", None)
        if callable(load_tensor) and getattr(base, "manifest", None) is not None:
            name = "embed_tokens" if base.arch.tie_word_embeddings else "lm_head"
            w = load_tensor(name, check_hash=True)
        else:
            w = base.load_plain("lm_head", dtype=torch.float32, device="cpu")
        if tuple(w.shape) != (self._spec.out_features, self._spec.in_features):
            raise ConfigError(f"lm_head has shape {tuple(w.shape)}, expected "
                              f"({self._spec.out_features}, {self._spec.in_features})")
        if w.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            w = w.to(torch.float32)
        return w

    def _head(self) -> Any:
        from pgw.modelpack.loader import LinearWeights

        with self._lock:
            if self._w is None:
                w = self._source()
                w8, sw = quantize_head(w)
                del w
                self._w = LinearWeights(self._lid, w8, sw, None)
            return self._w

    def release_head(self) -> None:
        """Drop the cached int8 head weight (e.g. once a service kernel holds its own copy)."""
        with self._lock:
            self._w = None

    def _is_head(self, lid: Any) -> bool:
        return str(lid) == self._lid.key

    def load_linear(self, lid: Any, device: torch.device | str = "cpu", **kw: Any) -> Any:
        if not self._is_head(lid):
            return self._base.load_linear(lid, device=device, **kw)
        from pgw.modelpack.loader import LinearWeights

        lw = self._head()
        if torch.device(device) == lw.w8.device:
            return lw
        return LinearWeights(lw.id, lw.w8.to(device), lw.scale.to(device), None)

    def load_linear_meta(self, lid: Any, **kw: Any) -> Any:
        if not self._is_head(lid):
            meta = getattr(self._base, "load_linear_meta", None)
            return meta(lid, **kw) if callable(meta) else self._base.load_linear(lid)
        from pgw.modelpack.loader import LinearMeta, row_norm_sq_max

        with self._lock:
            if self._meta is None:
                lw = self._w
                if lw is not None:
                    w8, sw = lw.w8, lw.scale
                else:  # the gateway's side: derive the scales and the row norm, keep no int8 weight
                    w = self._source()
                    w8, sw = quantize_head(w)
                    del w
                self._meta = LinearMeta(self._lid, sw, None, self._spec.out_features, self._spec.in_features,
                                        row_norm_sq_max(w8))
                del w8
            return self._meta

    def max_in_features(self) -> int:
        return max(s.in_features for s in self.linear_specs())

    def check_ring(self, params: Any) -> None:
        params.check_in_features(self.max_in_features())


class MaskedHead:
    """:class:`pgw.engine.model.HeadMode` with the head as one masked call per sequence.

    Each sequence's rows are normed exactly as :meth:`TransformerModel._logits` norms them (the last row
    alone, all rows for ``all_logits``, each of the last ``N`` rows alone for ``last_logits``), then sent as
    one ``LinearCall`` of the head linear in that sequence's session. The returned float32 rows are the
    logits (a :class:`pgw.engine.model.RowLogits` over them for ``last_logits``).
    """

    name = "masked"
    #: The head call takes the pass's padding like the layers' calls (``TransformerModel._padded``).
    supports_pad_rows = True

    def __init__(self, num_layers: int):
        self.lid = head_linear_id(num_layers)

    def linears(self, model: Any) -> Sequence[LinearId]:
        return (self.lid,)

    def steps(self, model: Any, seqs: Sequence[Any], hs: list[torch.Tensor]) -> Any:
        from pgw.engine.backends import LinearCall
        from pgw.engine.model import _TORCH_OPS

        if model.lm_head_fn is not None:
            raise ConfigError("the masked LM head does not combine with a draft head (lm_head_fn)")
        ops = model.float_ops or _TORCH_OPS
        calls = []
        for s, h in zip(seqs, hs, strict=True):
            if s.last_logits > 0:  # verify rows: each normed alone, as a 1-row decode step norms it
                hn = ops.norm_rows(h[-s.last_logits :], model.final_norm, model.eps).to(torch.float32)
            else:
                hn = ops.rms_norm(h if s.all_logits else h[-1:], model.final_norm, model.eps).to(torch.float32)
            calls.append(LinearCall(self.lid, hn.contiguous(), s.session))
        ys = yield calls
        out: list[Any] = []
        for s, h, y in zip(seqs, hs, ys, strict=True):
            if s.last_logits > 0:
                from pgw.engine.model import RowLogits

                out.append(RowLogits(h[-s.last_logits :], model.final_norm, model.eps, model._head, y))
            else:
                out.append(y if s.all_logits else y[0])
        return out


def enable_masked_head(model: Any, *, drop_float_head: bool = True) -> MaskedHead:
    """Install :class:`MaskedHead` on ``model`` (whose pack must be a :class:`MaskedHeadPack`).

    ``drop_float_head``: release the model's float head (1.09 GB of bfloat16 at 7B), which the masked head
    never reads (a tied head that is the embedding table itself stays, as the table).
    """
    pack = getattr(model, "pack", None)
    if pack is not None and not isinstance(pack, MaskedHeadPack):
        raise ConfigError("the masked LM head needs the model's pack wrapped in MaskedHeadPack")
    if getattr(model, "lm_head_fn", None) is not None:
        raise ConfigError("the masked LM head does not combine with a draft head (lm_head_fn)")
    head = MaskedHead(model.num_layers)
    model.head_mode = head
    if drop_float_head and model._head is not model.embed_tokens:
        model._head = model._head.new_empty((0, int(model._head.shape[1])))
    return head
