"""Epistemic channels: interpretability designed in, not reverse-engineered.

A channel is a named, reserved slice of the residual stream at a fixed layer,
for example "grounded" (the fact being stated is supported by the context or
memory) or "checked" (the current step passed a solver). Training gives each
channel:

  * supervision: an auxiliary loss ties the channel's readout to a signal
    computable at training time (a fact present in context, a step that
    checks, a span copied from a source);
  * causal role: intervention training writes values into the channel and
    trains the model to behave accordingly (e.g. abstain when "grounded" is
    forced low), so the readout reflects what the model uses rather than a
    correlate it can route around.

At inference the channels are monitors (read per token), control knobs (write
to steer behaviour), and inputs to calibrated release gates.

    em = EpistemicModel(gpt, {"grounded": ChannelSpec(layer=1, dims=(120, 128))})
    logits, loss = gpt(x, y)
    aux = em.aux_loss("grounded", positions, targets)          # supervision
    with em.intervene("grounded", mask, value=-4.0):           # causal training / steering
        logits, _ = gpt(x, y_abstain)
    em.read("grounded", positions)                             # monitoring
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ChannelSpec:
    layer: int
    dims: tuple            # (start, stop) slice of the residual stream
    meaning: str = ""
    high: float = 4.0      # canonical "true" readout
    low: float = -4.0      # canonical "false" readout


class EpistemicModel:
    def __init__(self, model, channels):
        self.model, self.channels = model, dict(channels)
        n = model.config.n_embd
        spans = []
        for name, c in self.channels.items():
            a, b = c.dims
            if not (0 <= a < b <= n) or not (0 <= c.layer < len(model.blocks)):
                raise ValueError(f"channel {name}: bad layer/dims")
            spans.append((c.layer, a, b, name))
        for (l1, a1, b1, n1), (l2, a2, b2, n2) in zip(spans, spans[1:]):
            if l1 == l2 and a2 < b1 and a1 < b2:
                raise ValueError(f"channels {n1} and {n2} overlap")
        self._acts = {}
        self._writes = {}
        self._hooks = [model.blocks[layer].register_forward_hook(self._hook(layer))
                       for layer in sorted({c.layer for c in self.channels.values()})]

    def _hook(self, layer):
        def fn(mod, inp, out):
            mine = {n: c for n, c in self.channels.items() if c.layer == layer}
            if any(n in self._writes for n in mine):
                out = out.clone()
                for n, c in mine.items():
                    if n in self._writes:
                        mask, value = self._writes[n]
                        a, b = c.dims
                        v = value if torch.is_tensor(value) else torch.tensor(float(value))
                        out[..., a:b] = torch.where(mask[..., None], v.to(out.dtype), out[..., a:b])
            self._acts[layer] = out
            return out
        return fn

    def read(self, name, positions=None):
        """Channel readout (mean over its dims) from the last forward pass:
        (B, T), or (B,) at the given per-sequence positions."""
        c = self.channels[name]
        a, b = c.dims
        r = self._acts[c.layer][..., a:b].mean(-1)
        return r if positions is None else r[torch.arange(r.shape[0]), positions]

    def aux_loss(self, name, positions, targets):
        """Smooth-L1 between the readout at `positions` and high/low targets (bool tensor)."""
        c = self.channels[name]
        t = torch.where(targets.bool(), torch.tensor(c.high), torch.tensor(c.low))
        return F.smooth_l1_loss(self.read(name, positions), t)

    @contextmanager
    def intervene(self, name, mask, value):
        """Write `value` into the channel wherever mask (B, T) is True, for the
        forward passes inside the block."""
        self._writes[name] = (mask, value)
        try:
            yield
        finally:
            self._writes.pop(name, None)

    def remove(self):
        for h in self._hooks:
            h.remove()
