"""Source flow: where each written token's information comes from, read in the same forward pass.

For every token of any output (a grounded answer, a step of a reasoning chain, a sentence of a long essay) and at
a few layers, the attention output is split by the region it reads from:

    sink      the first tokens and the chat template: a head parked there moves nothing (memory alone)
    question  the user's request
    evidence  documents in context (empty when there are none)
    reply     the output's own earlier text (earlier steps of a derivation, earlier sentences)

Each head's contribution from key j is its attention weight times the size of what it moves from j,
||v_j W_O^h|| (the norm-based reading of attention, Kobayashi et al. 2020: attention to a token whose value is
near zero moves nothing). Shares per region, per layer, per token. Copying (vt/copyheads.py) is the special case
of lifting text from the evidence; a derivation step that does not draw on the steps before it, or a biography
sentence that draws on neither the question nor any source, is the general one (docs/source_flow.md).

Recomputed from each layer's own projections in a forward pre-hook, only at the reply's positions: the forward
pass and its attention kernel are unchanged.
"""
import sys

import numpy as np

REGIONS = ("sink", "question", "evidence", "reply")


class SourceFlow:
    def __init__(self, model, layers):
        self.model, self.layers = model, [int(l) for l in layers]
        self.hooks, self.queries, self.region, self.out, self._gram = [], None, None, {}, {}

    def __enter__(self):
        for l in self.layers:
            attn = self.model.model.layers[l].self_attn
            self.hooks.append(attn.register_forward_pre_hook(
                lambda mod, args, kwargs, l=l: self._capture(mod, args, kwargs, l), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def at(self, queries, region):
        """Capture these query positions in the next forward pass; region: one id (index into REGIONS) per token
        position of the whole input."""
        self.queries, self.region, self.out = list(queries), np.asarray(region), {}
        return self

    def gram(self, attn, l, H, d):
        """W_O^h^T W_O^h per head (d x d): ||v W_O^h||^2 = v G_h v^T without forming v W_O^h."""
        if l not in self._gram:
            import torch
            w = attn.o_proj.weight.float().view(-1, H, d)                         # hidden x H x d
            self._gram[l] = torch.einsum("khd,khe->hde", w, w)
        return self._gram[l]

    def _capture(self, attn, args, kwargs, l):
        import torch
        x = kwargs.get("hidden_states", args[0] if args else None)[0]
        cos, sin = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        cfg = attn.config
        H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
        d = getattr(attn, "head_dim", cfg.hidden_size // H)
        with torch.no_grad():
            T = x.shape[0]
            q = attn.q_proj(x).view(T, H, d)
            k = attn.k_proj(x).view(T, KV, d)
            if getattr(attn, "q_norm", None) is not None:
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = q.transpose(0, 1)[None], k.transpose(0, 1)[None]
            rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
            q, _ = rope(q, q, cos, sin)
            k, _ = rope(k, k, cos, sin)
            v = attn.v_proj(x).view(T, KV, d).float()
            g = H // KV
            qpos = torch.as_tensor(self.queries)
            qq = q[0][:, qpos].float()                                               # H x Q x d
            kk = k[0].float().repeat_interleave(g, 0)                                 # H x T x d
            s = torch.einsum("hqd,htd->hqt", qq, kk) * attn.scaling
            s = s.masked_fill(torch.arange(T)[None, None, :] > qpos[None, :, None], float("-inf"))
            a = torch.softmax(s, -1)                                                 # H x Q x T
            vv = v.transpose(0, 1).repeat_interleave(g, 0)                           # H x T x d
            n = torch.einsum("htd,hde,hte->ht", vv, self.gram(attn, l, H, d), vv).clamp_min(0).sqrt()  # H x T
            f = (a * n[:, None, :]).sum(0)                                           # Q x T, summed over heads
            reg = torch.as_tensor(self.region[:T])
            shares = torch.stack([f[:, reg == r].sum(1) for r in range(len(REGIONS))], 1)
            self.out[l] = (shares / shares.sum(1, keepdim=True).clamp_min(1e-9)).numpy()   # Q x regions

    def collect(self):
        """layers x queries x regions."""
        return np.stack([self.out[l] for l in self.layers])


def regions_for(tok, question, evidence, reply_len):
    """The templated prefix's token ids, and one region id per token of prefix + reply: tokens whose characters
    fall in the question are "question", in the evidence "evidence", the reply "reply", the rest "sink"."""
    user = f"{question}\n\n{evidence}".strip() if evidence else question
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    u0 = text.find(user)
    q_end = u0 + len(question)
    e0 = u0 + len(user) - len(evidence) if evidence else -1
    reg = []
    for s, e in enc.offset_mapping:
        if u0 <= s and e <= q_end:
            reg.append(1)
        elif evidence and e0 <= s and e <= u0 + len(user):
            reg.append(2)
        else:
            reg.append(0)
    return enc.input_ids, np.array(reg + [3] * reply_len)
