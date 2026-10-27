"""Glassbox-3: Glassbox-2 plus epistemic uncertainty that shapes what the model says.

On top of Glassbox-2 (exact sparse features, exact attribution, logged hard-gated memory):

* Epinet (Osband et al., "Epistemic Neural Networks"). A small network reads the final
  features (gradient stopped, so the base model and its exact attribution are unchanged)
  and an index z ~ N(0, I), and adds a fixed random prior network of the same shape. It is
  trained to predict the next token for random z; where training data constrains the
  prediction, the learnt part cancels the prior and every z agrees. Where it does not, the
  prior survives and predictions disagree. u = mutual information across z (the epistemic
  part of the uncertainty) is high on what the model has not learnt.
* Uncertainty in the output. u raises the logit of a dedicated <unsure> token by an explicit
  term softplus(c) * u, and lowers the commitment k that sets the sharpness of every
  next-token distribution. A confident token therefore requires low epistemic uncertainty.
* Feature novelty. During training the model counts which last-layer features fire together
  (a hashed pair table). At inference, rare co-activation means unfamiliar territory: a
  label-free signal that names the feature pairs involved.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glassbox2 import G2Config, Glassbox2


@dataclass
class G3Config(G2Config):
    unsure_id: int = 1
    epi_index: int = 8
    epi_hidden: int = 64
    prior_scale: float = 1.0
    epi_samples: int = 4
    pair_buckets: int = 1 << 16


def _mlp(d_in, h, d_out):
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out))


class Glassbox3(Glassbox2):
    def __init__(self, c: G3Config):
        super().__init__(c)
        d = c.n_embd
        self.epi = _mlp(d + c.epi_index, c.epi_hidden, d)
        self.prior = _mlp(d + c.epi_index, c.epi_hidden, d)
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.c_unsure = nn.Parameter(torch.tensor(1.0))
        self.c_k = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("feat_count", torch.zeros(c.n_features))
        self.register_buffer("pair_count", torch.zeros(c.pair_buckets))
        self.register_buffer("n_seen", torch.zeros(()))
        # fixed evaluation indices: u is a deterministic function of the input
        self.register_buffer("epi_z", torch.randn(c.epi_samples, c.epi_index))

    def _epi_head(self, phi, h):
        delta = self.epi(h) + self.config.prior_scale * self.prior(h)
        return F.linear(phi.detach() + delta, self.head.weight.detach())   # base head gets no gradient

    def epi_logits(self, phi, z):
        """Index-conditioned next-token logits from stop-gradient features."""
        h = torch.cat([phi.detach(), z.expand(*phi.shape[:-1], z.shape[-1])], -1)
        return self._epi_head(phi, h)

    def uncertainty(self, phi, n=None):
        """Epistemic uncertainty per position: mutual information between the next token and z,
        over the fixed indices epi_z (n > epi_samples draws extra random indices)."""
        zs = list(self.epi_z)
        if n and n > len(zs):
            zs += list(torch.randn(n - len(zs), self.config.epi_index, device=phi.device))
        ps = [F.softmax(self.epi_logits(phi, z).float(), -1) for z in zs[:n or len(zs)]]
        P = torch.stack(ps)                                            # (n, B, T, V)
        ent = lambda q: -(q * torch.log(q.clamp_min(1e-12))).sum(-1)   # noqa: E731
        return ent(P.mean(0)) - ent(P).mean(0)

    def forward(self, idx, u_samples=None):
        raw_scaled, k = super().forward(idx)
        tr = self.trace
        phi, raw = tr["z_f"], tr["raw_logits"]
        u = self.uncertainty(phi, u_samples).detach()
        k_eff = k - F.softplus(self.c_k) * u
        logits = raw * self.temperature(k_eff)[..., None]
        bump = torch.zeros_like(logits)
        bump[..., self.config.unsure_id] = F.softplus(self.c_unsure) * u
        tr.update(u=u, k=k, k_eff=k_eff)
        return logits + bump, k_eff

    def epi_loss(self, y):
        """Epinet training: next-token NLL for one random index per sequence (base frozen)."""
        phi = self.trace["z_f"]
        z = torch.randn(phi.shape[0], 1, self.config.epi_index, device=phi.device)
        h = torch.cat([phi.detach(), z.expand(-1, phi.shape[1], -1)], -1)
        logits = self._epi_head(phi, h)
        return F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())

    def _pair_ids(self, idx):
        """Hash every pair of active last-layer features at each position: (B, T, k*(k-1)/2)."""
        i = idx.sort(-1).values
        a, b = torch.triu_indices(i.shape[-1], i.shape[-1], 1, device=i.device)
        return (i[..., a] * 1000003 + i[..., b]) % self.config.pair_buckets

    @torch.no_grad()
    def record_features(self, mask=None):
        """Count active features and feature pairs at the last layer (mask: positions to count)."""
        acts = self.trace["feat"][-1]["acts"]
        active = acts != 0
        idx = acts.abs().topk(self.config.k_active, -1).indices
        keep = torch.ones(acts.shape[:2], dtype=torch.bool, device=acts.device) if mask is None else mask
        self.feat_count += active[keep].float().sum(0)
        pairs = self._pair_ids(idx)[keep].flatten()
        self.pair_count.index_add_(0, pairs, torch.ones_like(pairs, dtype=torch.float))
        self.n_seen += keep.sum()

    @torch.no_grad()
    def novelty(self):
        """Per position: mean -log frequency of its active feature pairs (from the last forward)."""
        acts = self.trace["feat"][-1]["acts"]
        idx = acts.abs().topk(self.config.k_active, -1).indices
        freq = (self.pair_count[self._pair_ids(idx)] + 1) / (self.n_seen + 1)
        return -torch.log(freq).mean(-1)
