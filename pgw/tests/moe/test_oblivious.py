"""Routing-oblivious MoE dispatch: exactness against the plaintext quantized reference, and what the
compute service can see (shapes, row distribution, abort behaviour)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError, ShapeError
from pgw.core.ring import DEFAULT_RING, RingParams, quantize_activations, quantize_weight_rows
from pgw.moe import (
    LockstepRefill,
    assemble_expert_input,
    capacity_rows,
    plan_dispatch,
    provider_view,
    simulate_refill,
    unmask_expert_output,
)

RING: RingParams = DEFAULT_RING  # K32A12C27


def _ring_mm(x: np.ndarray, w8: np.ndarray, params: RingParams) -> np.ndarray:
    """Provider: ``x @ W8^T mod 2^K`` (uint64 products wrap mod 2^64; 2^K divides 2^64)."""
    return (x.astype(np.uint64) @ w8.astype(np.int64).view(np.uint64).T) & np.uint64(params.mask)


class TinyMoE:
    """One MoE FFN block (SwiGLU experts, softmax-over-top-k gating) with int8 expert weights."""

    def __init__(self, seed: int = 0, hidden: int = 48, ffn: int = 40, experts: int = 4, k: int = 2) -> None:
        g = torch.Generator().manual_seed(seed)
        self.hidden, self.ffn, self.e, self.k = hidden, ffn, experts, k
        self.router = torch.randn(experts, hidden, generator=g, dtype=torch.float64)
        self.w = {}
        for e in range(experts):
            for name, shape in (("gate_up", (2 * ffn, hidden)), ("down", (hidden, ffn))):
                w8, sw = quantize_weight_rows(torch.randn(*shape, generator=g) * 0.05)
                w8n = w8.numpy().astype(np.int64)
                wnorm = float(np.sqrt((w8n.astype(np.float64) ** 2).sum(axis=1)).max())
                self.w[(e, name)] = (w8n, sw.numpy().astype(np.float64), wnorm)

    def route(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        logits = x.to(torch.float64) @ self.router.T
        top = torch.topk(logits, self.k, dim=1)
        return top.indices.numpy(), torch.softmax(top.values, dim=1).numpy()

    def forward(self, x: torch.Tensor, int_mm) -> torch.Tensor:
        """``int_mm(name, {expert: (tokens, xq)}) -> {expert: yq}`` does every expert linear of one trip."""
        topk, gates = self.route(x)
        b = x.shape[0]
        per_expert = {e: np.flatnonzero((topk == e).any(axis=1)) for e in range(self.e)}

        def quant(rows: np.ndarray, e: int, name: str) -> tuple[np.ndarray, np.ndarray]:
            xq, sx = quantize_activations(torch.from_numpy(rows), RING, self.w[(e, name)][2])
            return xq.numpy(), sx.numpy()

        xs = x.to(torch.float64).numpy()
        q1 = {e: quant(xs[t], e, "gate_up") for e, t in per_expert.items()}
        y1 = int_mm("gate_up", {e: (per_expert[e], q1[e][0]) for e in range(self.e)})
        h = {}
        for e in range(self.e):
            _, sw, _ = self.w[(e, "gate_up")]
            gu = y1[e].astype(np.float64) * q1[e][1][:, None] * sw[None, :]
            gate, up = gu[:, : self.ffn], gu[:, self.ffn :]
            h[e] = gate / (1.0 + np.exp(-gate)) * up
        q2 = {e: quant(h[e], e, "down") for e in range(self.e)}
        y2 = int_mm("down", {e: (per_expert[e], q2[e][0]) for e in range(self.e)})
        out = np.zeros((b, self.hidden), dtype=np.float64)
        for i in range(b):
            for j in range(self.k):  # fixed accumulation order: choice 0, 1, ...
                e = int(topk[i, j])
                row = int(np.flatnonzero(per_expert[e] == i)[0])
                _, sw, _ = self.w[(e, "down")]
                out[i] += gates[i, j] * (y2[e][row].astype(np.float64) * q2[e][1][row] * sw)
        return torch.from_numpy(out)


def _plain_int_mm(moe: TinyMoE):
    def mm(name: str, rows: dict) -> dict:
        return {e: xq @ moe.w[(e, name)][0].T for e, (_, xq) in rows.items()}
    return mm


class MaskedRunner:
    """Gateway + simulated factory and provider for the oblivious dispatch; records the provider's view."""

    def __init__(self, moe: TinyMoE, topk: np.ndarray, tamper: tuple[str, int, str] | None = None) -> None:
        self.moe, self.topk, self.tamper = moe, topk, tamper
        self.views: list[tuple] = []
        self.sent: list[tuple[int, np.ndarray]] = []
        self.slots_used = 0
        self.plans = []

    def __call__(self, name: str, rows: dict) -> dict:
        moe = self.moe
        plan = plan_dispatch(self.topk, moe.e)
        self.plans.append(plan)
        n = next(iter(rows.values()))[1].shape[1]
        self.views.append(provider_view(plan, n))
        out = {}
        for e in range(moe.e):
            tokens, xq = rows[e]
            w8 = moe.w[(e, name)][0]
            pos = plan.real(e)
            assert np.array_equal(plan.source[e, pos], tokens)  # plan order == token order
            r = np.frombuffer(np.random.bytes(xq.size * 8), dtype=np.uint64).reshape(xq.shape) & np.uint64(RING.mask)
            wr = _ring_mm(r, w8, RING)  # factory: W r (never leaves the gateway)
            self.slots_used += pos.size
            x_hat = assemble_expert_input(plan, e, xq, r, RING)
            self.sent.append((e, x_hat.copy()))
            y_hat = _ring_mm(x_hat, w8, RING)  # provider
            if self.tamper and self.tamper[0] == name and self.tamper[1] == e:
                kind = self.tamper[2]
                dummy = np.flatnonzero(plan.source[e] < 0)
                idx = int(pos[0]) if kind == "real" else int(dummy[0])
                y_hat[idx, 0] = (y_hat[idx, 0] + np.uint64(1 << 20)) & np.uint64(RING.mask)
            u = np.frombuffer(np.random.bytes(2 * w8.shape[0] * 8), dtype=np.uint64).reshape(2, -1) & np.uint64(RING.mask)
            uw = (u @ w8.astype(np.int64).view(np.uint64)) & np.uint64(RING.mask)
            out[e] = unmask_expert_output(plan, e, x_hat, y_hat, wr.view(np.int64), (u, uw), RING, n)
        return out


def _skewed_topk(b: int, e: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(e, size=k, replace=False) for _ in range(b)])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_moe_is_bit_identical_to_plaintext_reference(seed: int) -> None:
    moe = TinyMoE(seed)
    x = torch.randn(7, moe.hidden, generator=torch.Generator().manual_seed(100 + seed), dtype=torch.float64)
    topk, _ = moe.route(x)
    ref = moe.forward(x, _plain_int_mm(moe))
    runner = MaskedRunner(moe, topk)
    got = moe.forward(x, runner)
    assert torch.equal(ref, got)  # bit-identical, not just close
    # masks: exactly the routed rows (B * k per expert linear), dummies take none
    assert runner.slots_used == 2 * x.shape[0] * moe.k
    for plan in runner.plans:
        assert int(plan.masks_needed().sum()) == x.shape[0] * moe.k


def test_provider_view_does_not_depend_on_routing() -> None:
    b, e, k = 9, 6, 2
    all_same = np.tile(np.array([[0, 1]]), (b, 1))  # every token to experts 0 and 1
    spread = _skewed_topk(b, e, k, 5)
    v1 = provider_view(plan_dispatch(all_same, e), 32)
    v2 = provider_view(plan_dispatch(spread, e), 32)
    assert v1 == v2 == tuple((i, b, 32) for i in range(e))


def test_rows_on_the_wire_are_uniform_for_any_routing() -> None:
    # Extreme routing: all tokens to experts 0 and 1; experts 2..5 get only dummies. Every expert's rows must
    # look the same: uniform bytes (chi-square on byte values, loose bound) and identical shapes.
    moe = TinyMoE(3, experts=6, k=2)
    b = 64
    x = torch.randn(b, moe.hidden, generator=torch.Generator().manual_seed(9), dtype=torch.float64) * 0.0
    x[:, 0] = 1.0  # all tokens equal -> same routing for all
    topk, _ = moe.route(x)
    assert len(set(map(tuple, np.sort(topk, axis=1)))) == 1
    runner = MaskedRunner(moe, topk)
    moe.forward(x, runner)
    for _, xh in runner.sent:
        assert xh.shape == (b, moe.hidden) or xh.shape == (b, moe.ffn)
        by = (xh.astype("<u8").view(np.uint8).reshape(xh.size, 8))[:, :4].ravel()  # the K=32 bytes
        counts = np.bincount(by, minlength=256).astype(np.float64)
        exp = by.size / 256
        chi2 = float(((counts - exp) ** 2 / exp).sum())
        assert chi2 < 400, chi2  # 255 dof: mean 255, sd ~22.6
        assert not (xh >> np.uint64(32)).any()  # values reduced mod 2^K


@pytest.mark.parametrize("kind", ["real", "dummy"])
def test_tampering_is_caught_identically_on_real_and_dummy_rows(kind: str) -> None:
    # Selective-failure defence: an abort must not reveal whether the tampered row was real.
    moe = TinyMoE(4)
    x = torch.randn(5, moe.hidden, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    topk, _ = moe.route(x)
    e = int(topk[0, 0])
    plan_probe = plan_dispatch(topk, moe.e, rows=capacity_rows(5))
    if kind == "dummy" and not (plan_probe.source[e] < 0).any():
        pytest.skip("expert has no dummy row in this routing")
    with pytest.raises(IntegrityError, match="Freivalds"):
        moe.forward(x, MaskedRunner(moe, topk, tamper=("gate_up", e, kind)))


def test_plan_validation() -> None:
    t = np.array([[0, 1], [1, 2]])
    with pytest.raises(ConfigError):
        plan_dispatch(t, 3, rows=1)  # rows < tokens could overflow
    with pytest.raises(ValueError):
        plan_dispatch(np.array([[0, 0]]), 3)
    with pytest.raises(ValueError):
        plan_dispatch(np.array([[0, 3]]), 3)
    with pytest.raises(ShapeError):
        plan_dispatch(np.array([0, 1]), 3)
    p = plan_dispatch(t, 3, rows=capacity_rows(2, bucket=8), shuffle=False)
    assert p.rows == 8 and p.masks_needed().tolist() == [1, 2, 1]
    assert p.source[1, :2].tolist() == [0, 1] and p.choice[1, :2].tolist() == [1, 0]
    dummy = plan_dispatch(np.empty((0, 2), dtype=np.int64), 3, rows=4)  # all-dummy trip of 4 rows
    assert dummy.masks_needed().sum() == 0 and provider_view(dummy, 8) == provider_view(p, 8)[:0] + tuple(
        (i, 4, 8) for i in range(3))


def test_lockstep_refill_is_routing_independent_and_simulates() -> None:
    pol = LockstepRefill(num_experts=8, top_k=2, overprovision=1.5, reserve_slots=512, group_slots=256)
    assert pol.target_per_expert(0) == 512
    assert pol.target_per_expert(1000) == 512 + 256 * int(np.ceil(1.5 * 2 * 1000 / 8 / 256))
    assert pol.due(1000, pol.target_per_expert(1000)) == 0
    rng = np.random.default_rng(0)
    balanced = [np.stack([rng.choice(8, 2, replace=False) for _ in range(32)]) for _ in range(200)]
    sim = simulate_refill(pol, balanced, lag_trips=2)
    assert sim.rows_served == 200 * 32 and sim.stall_trips == 0
    assert sim.slots_consumed == 200 * 32 * 2
    # a hot pair of experts drains its pools: stalls appear with rho = 1 and no reserve
    hot = [np.tile(np.array([[0, 1]]), (32, 1)) for _ in range(50)]
    tight = LockstepRefill(num_experts=8, top_k=2, overprovision=1.0, reserve_slots=0, group_slots=256)
    sim2 = simulate_refill(tight, hot, lag_trips=1)
    assert sim2.rows_served == 50 * 32 and sim2.stall_trips > 0 and sim2.waste > 0.5
    with pytest.raises(ConfigError):
        LockstepRefill(num_experts=4, top_k=5)
