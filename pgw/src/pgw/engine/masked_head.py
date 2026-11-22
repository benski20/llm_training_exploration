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

from collections.abc import Sequence
from typing import Any

import torch

from pgw.core.errors import ConfigError
from pgw.core.types import LinearId

__all__ = ["MaskedHead", "MaskedHeadPack", "enable_masked_head", "head_fingerprint", "head_linear_id",
           "quantize_head"]

from pgw.modelpack.masked_head import (  # noqa: F401  (re-exported)
    _BLOCK_ROWS,
    _DOMAIN,
    MaskedHeadPack,
    head_fingerprint,
    head_linear_id,
    quantize_head,
)


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
