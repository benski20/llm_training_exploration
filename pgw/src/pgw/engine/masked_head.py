"""The LM head as a masked remote linear (opt-in; off by default).

A weight-free CPU gateway cannot afford the float LM head (Qwen2.5-7B: 152064 x 3584, 1.09 GB of
bfloat16 read per token, about 25-45 ms on one core). With :class:`MaskedHeadPack` and
:class:`MaskedHead` the head is one more masked linear, ``"<num_layers>.lm_head"`` of shape
``(vocab, hidden)``, computed by the provider exactly like the layers' linears: the gateway sends
``x_hat = xq + r`` of the final-normed hidden row, unmasks ``y_hat - W r``, checks the range and
Freivalds on every row, and rescales. The provider sees one more masked call per token of public shape.

Exactness: the head's weight is the pack's plain ``lm_head`` (``embed_tokens`` when tied) quantized per
output row to int8 by :func:`pgw.core.ring.quantize_weight_rows` (SPEC 5), a deterministic function of
a tensor the pack fingerprint already covers. The logits are therefore those of the int8 head, not of
the float head: a **new pinned reference** (like ``float_ops = "native"``). The plaintext reference is the
same wrapped pack with :class:`pgw.engine.backends.PlainBackend` and the same :class:`MaskedHead`, and the
masked gateway's logits equal it bit for bit (``tests/engine/test_masked_head.py``).

Binding: :attr:`MaskedHeadPack.fingerprint` is domain-separated from the base pack's
(``SHA-256("pgw/v1/masked-lm-head\\0" || base fingerprint)``), so masks, segments and the provider's pack
for a head-masking deployment can never be mixed with those of the base pack. Both parties wrap the
same pack. Library API only: the service registry, the factory CLI and the gateway runtime do not
wire it yet.

Not supported with it: speculative verify rows (``last_logits``), draft heads (``lm_head_fn``) and
padding-row passes (:attr:`TransformerModel.supports_pad_rows` is false with any head-mode hook).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.ring import quantize_weight_rows
from pgw.core.types import LinearId, LinearKind, LinearSpec

__all__ = ["MaskedHead", "MaskedHeadPack", "head_linear_id", "quantize_head"]

_DOMAIN = b"pgw/v1/masked-lm-head\x00"
#: Rows quantized per block (the float64 copy of a 7B head would be 4.4 GB; rows are independent).
_BLOCK_ROWS = 4096


def head_linear_id(num_layers: int) -> LinearId:
    """``LinearId(num_layers, LM_HEAD)``: the masked head's id (``"<num_layers>.lm_head"``)."""
    return LinearId(int(num_layers), LinearKind.LM_HEAD)


def quantize_head(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`quantize_weight_rows` of a ``(vocab, hidden)`` head in row blocks (bit-identical: every row's
    scale and int8 values depend only on that row)."""
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
    """

    def __init__(self, base: Any):
        if isinstance(base, MaskedHeadPack):
            raise ConfigError("pack already has a masked LM head")
        self._base = base
        arch = base.arch
        self._lid = head_linear_id(arch.num_layers)
        self._spec = LinearSpec(self._lid, int(arch.vocab_size), int(arch.hidden_size), False)
        self._fp = hashlib.sha256(_DOMAIN + str(base.fingerprint).encode()).hexdigest()
        self._lock = threading.Lock()
        self._w: Any = None  # LinearWeights, built on first use (the provider's / factory's side only)

    def __getattr__(self, name: str) -> Any:  # name, arch, tokenizer_dir, load_plain, manifest, ...
        return getattr(self._base, name)

    @property
    def base(self) -> Any:
        """The wrapped pack."""
        return self._base

    @property
    def fingerprint(self) -> str:
        return self._fp

    @property
    def head_id(self) -> LinearId:
        return self._lid

    def linear_specs(self) -> list[LinearSpec]:
        return [*self._base.linear_specs(), self._spec]

    def _head(self) -> Any:
        from pgw.modelpack.loader import LinearWeights

        with self._lock:
            if self._w is None:
                w = self._base.load_plain("lm_head", dtype=torch.float32, device="cpu")
                if tuple(w.shape) != (self._spec.out_features, self._spec.in_features):
                    raise ConfigError(f"lm_head has shape {tuple(w.shape)}, expected "
                                      f"({self._spec.out_features}, {self._spec.in_features})")
                w8, sw = quantize_head(w)
                del w
                self._w = LinearWeights(self._lid, w8, sw, None)
            return self._w

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

        lw = self._head()
        return LinearMeta(self._lid, lw.scale, None, self._spec.out_features, self._spec.in_features,
                          row_norm_sq_max(lw.w8))

    def max_in_features(self) -> int:
        return max(s.in_features for s in self.linear_specs())

    def check_ring(self, params: Any) -> None:
        params.check_in_features(self.max_in_features())


class MaskedHead:
    """:class:`pgw.engine.model.HeadMode` with the head as one masked call per sequence.

    Each sequence's rows are normed exactly as :meth:`TransformerModel._logits` norms them (the last row
    alone, or all rows for ``all_logits``), then sent as one ``LinearCall`` of the head linear in that
    sequence's session. The returned float32 rows are the logits.
    """

    name = "masked"

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
            if s.last_logits > 0:
                raise ConfigError("the masked LM head does not support speculative verify rows (last_logits)")
            hn = ops.rms_norm(h if s.all_logits else h[-1:], model.final_norm, model.eps).to(torch.float32)
            calls.append(LinearCall(self.lid, hn.contiguous(), s.session))
        ys = yield calls
        return [y if s.all_logits else y[0] for s, y in zip(seqs, ys, strict=True)]


def enable_masked_head(model: Any, *, drop_float_head: bool = True) -> MaskedHead:
    """Install :class:`MaskedHead` on ``model`` (whose pack must be a :class:`MaskedHeadPack`).

    ``drop_float_head``: release the model's float head (1.09 GB of bfloat16 at 7B), which the masked head
    never reads (a tied head that is the embedding table itself stays, as the table).
    """
    pack = getattr(model, "pack", None)
    if pack is not None and not isinstance(pack, MaskedHeadPack):
        raise ConfigError("the masked LM head needs the model's pack wrapped in MaskedHeadPack")
    head = MaskedHead(model.num_layers)
    model.head_mode = head
    if drop_float_head and model._head is not model.embed_tokens:
        model._head = model._head.new_empty((0, int(model._head.shape[1])))
    return head
