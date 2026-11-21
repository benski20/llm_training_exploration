"""Routing-oblivious expert dispatch for masked MoE layers (opt-in; not wired into the engine).

Problem. In a mixture-of-experts layer each token is routed to ``k`` of ``E`` experts. If the gateway sent
each expert only the rows routed to it, the compute service would see the per-expert row counts of every
trip, i.e. the routing histogram of the batch. Routing depends on the text (topic, language, code vs
prose), so that is a data-dependent leak the dense protocol does not have.

Construction (fixed-capacity padding with uniform dummies). For a decode trip with ``B`` token rows:

1. every expert gets exactly ``C`` rows, ``C >= B`` a public number (default ``C = B``). A token picks
   ``k`` *distinct* experts, so no expert ever has more than ``B`` real rows: **no overflow is possible**,
   so there is no overflow path, no extra trip and no data-dependent shape;
2. a real row is ``x_hat = xq + r mod 2^K`` with a fresh one-time mask ``r`` for that expert's linear
   (one mask slot of that expert, exactly as in the dense protocol);
3. a dummy row is drawn **uniformly** from ``Z_{2^K}^n`` by the gateway CSPRNG (ChaCha20, fresh key).
   It consumes no mask slot: the gateway never unmasks it;
4. real rows occupy uniformly random positions among the ``C`` (defence in depth; positions carry no
   information anyway because every row is uniform);
5. every one of the ``C`` rows of every expert, real or dummy, gets the same Freivalds check, and any
   failure aborts the whole trip. Only the range check after unmasking is real-row only.

Security argument (sketch; the full text with the selective-failure discussion is in
``research/plans/scale-north-star.md``). The service's view of a trip is the tuple of shapes
``(E, C, n)`` - a function of the public ``B`` (already visible in the dense protocol) - and the row
values. Real rows are one-time-pad encryptions (uniform and independent of everything else because each
``r`` is fresh and uniform); dummy rows are uniform by construction. So the joint distribution of all
``E * C`` rows is uniform on ``Z_{2^K}^{E C n}`` for **every** routing: the view is perfectly simulatable
without the routing. Abort behaviour: Freivalds runs identically on every row, so whether a tampered row
is detected does not depend on whether it is real. The range check can only add an abort for a real row
whose error already passed every Freivalds vector, which happens with probability ``<= 2^-soundness``
(the same per-call bound as the dense protocol); that is the only routing-dependent event and it is
negligible. Mask-pool refills must be routing-independent too: see :mod:`pgw.moe.refill`.

Exactness. Real rows go through the unchanged one-time-mask arithmetic (:mod:`pgw.core.masking`), so every
real output is the exact integer product, bit-identical to the plaintext quantized reference.

Cost. Provider compute and wire per expert linear are ``E * C`` rows per trip (``E / k`` times the
routed rows: the same as running every expert on every token), while mask slots, mask storage and
mask-factory work stay at the routed ``B * k`` rows.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from pgw.core.errors import ConfigError, IntegrityError, ShapeError
from pgw.core.freivalds import freivalds_check_u64
from pgw.core.masking import unmask_outputs_u64
from pgw.core.ring import RingParams

__all__ = [
    "DispatchPlan",
    "assemble_expert_input",
    "capacity_rows",
    "plan_dispatch",
    "provider_view",
    "uniform_rows",
    "unmask_expert_output",
]


def capacity_rows(tokens: int, *, bucket: int = 0) -> int:
    """Public rows per expert for a trip of ``tokens`` rows: ``tokens``, or rounded up to a multiple of
    ``bucket`` (to also hide the exact batch size within the bucket). Never data-dependent."""
    if tokens < 0 or bucket < 0:
        raise ConfigError("tokens and bucket must be non-negative")
    if bucket <= 1 or tokens == 0:
        return tokens
    return -(-tokens // bucket) * bucket


@dataclass(frozen=True)
class DispatchPlan:
    """Where each token's routed rows go. ``source[e, j]`` is the token index of row ``j`` of expert ``e``
    (``-1`` for a dummy row) and ``choice[e, j]`` which of that token's ``k`` choices it is (``-1`` for a
    dummy). Gateway-private: only :func:`provider_view` may leave the gateway."""

    num_experts: int
    top_k: int
    tokens: int
    rows: int
    source: np.ndarray
    choice: np.ndarray

    def real(self, expert: int) -> np.ndarray:
        """Positions (ascending) of the real rows of ``expert``."""
        return np.flatnonzero(self.source[expert] >= 0)

    def masks_needed(self) -> np.ndarray:
        """Mask slots consumed per expert (``(E,)``, sums to ``tokens * top_k``)."""
        return (self.source >= 0).sum(axis=1)


def plan_dispatch(
    topk: np.ndarray,
    num_experts: int,
    *,
    rows: int | None = None,
    shuffle: bool = True,
) -> DispatchPlan:
    """Plan an oblivious dispatch of one trip.

    Args:
        topk: ``(B, k)`` integer expert ids per token (distinct within a row, as every top-k router gives).
        num_experts: ``E``.
        rows: Public rows per expert ``C``; default :func:`capacity_rows` ``(B)``. Must be ``>= B``: any
            smaller value could overflow, and an overflow path would make the trip shape routing-dependent.
        shuffle: Place real rows at CSPRNG-chosen positions (default; positions are not secret, this is
            defence in depth). ``False`` packs them first in token order (tests).

    Raises:
        ConfigError: ``rows < B`` or bad expert counts. ShapeError / ValueError: malformed ``topk``.
    """
    t = np.asarray(topk)
    if t.ndim != 2:
        raise ShapeError(f"topk must be (tokens, k), got shape {t.shape}")
    if not np.issubdtype(t.dtype, np.integer):
        raise ShapeError(f"topk must hold integer expert ids, got {t.dtype}")
    b, k = int(t.shape[0]), int(t.shape[1])
    if num_experts < 1 or not 1 <= k <= num_experts:
        raise ConfigError(f"need 1 <= k <= E, got k={k} E={num_experts}")
    c = capacity_rows(b) if rows is None else int(rows)
    if c < b:
        raise ConfigError(f"rows per expert ({c}) < tokens ({b}): a routing-dependent overflow would be possible")
    if t.size and (int(t.min()) < 0 or int(t.max()) >= num_experts):
        raise ValueError(f"expert id outside [0, {num_experts})")
    if k > 1 and t.size:
        srt = np.sort(t, axis=1)
        if bool((srt[:, 1:] == srt[:, :-1]).any()):
            raise ValueError("a token routes to the same expert twice (top-k ids must be distinct)")
    source = np.full((num_experts, c), -1, dtype=np.int64)
    choice = np.full((num_experts, c), -1, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(secrets.randbits(128))) if shuffle else None
    for e in range(num_experts):
        tok, ch = np.nonzero(t == e)  # tokens in ascending order, each at most once
        if tok.size == 0:
            continue
        pos = np.sort(rng.choice(c, size=tok.size, replace=False)) if rng is not None else np.arange(tok.size)
        source[e, pos] = tok
        choice[e, pos] = ch
    return DispatchPlan(num_experts, k, b, c, source, choice)


def provider_view(plan: DispatchPlan, in_features: int) -> tuple[tuple[int, int, int], ...]:
    """Everything about a trip's shape the service sees: one ``(expert, rows, in_features)`` per expert.
    A function of ``(E, C, n)`` only, never of the routing."""
    return tuple((e, plan.rows, int(in_features)) for e in range(plan.num_experts))


def uniform_rows(rows: int, cols: int, params: RingParams) -> np.ndarray:
    """``(rows, cols)`` uint64 uniform in ``[0, 2^K)`` (dummy rows).

    ChaCha20 keyed with 32 fresh bytes from the OS CSPRNG, one key per call; ``K <= 32`` draws only
    4 bytes per element, so the gateway's dummy-row cost is ``E * C * n * 4`` keystream bytes per linear.
    """
    count = rows * cols
    if count == 0:
        return np.zeros((rows, cols), dtype=np.uint64)
    width = 4 if params.k <= 32 else 8
    enc = Cipher(algorithms.ChaCha20(os.urandom(32), bytes(16)), mode=None).encryptor()
    raw = np.frombuffer(enc.update(bytes(width * count)), dtype="<u4" if width == 4 else "<u8")
    out = raw.astype(np.uint64).reshape(rows, cols)
    out &= np.uint64(params.mask)
    return out


def assemble_expert_input(
    plan: DispatchPlan,
    expert: int,
    xq: np.ndarray,
    r: np.ndarray,
    params: RingParams,
) -> np.ndarray:
    """Masked input ``(C, n)`` uint64 of ``expert`` for the service.

    Args:
        plan: The trip's plan.
        expert: Expert id.
        xq: Signed quantized activations ``(count, n)`` int64 of the expert's real rows, in the order of
            :meth:`DispatchPlan.real` (``count = masks_needed()[expert]``).
        r: The fresh one-time masks ``(count, n)`` for those rows (one mask slot of this expert each).
        params: Ring.
    """
    pos = plan.real(expert)
    xq = np.asarray(xq)
    r = np.asarray(r)
    if xq.ndim != 2 or xq.shape != r.shape or xq.shape[0] != pos.size:
        raise ShapeError(f"expert {expert}: need {pos.size} real rows of xq and r with equal shapes, "
                         f"got {xq.shape} and {r.shape}")
    if xq.size and (int(xq.min()) < -params.qmax or int(xq.max()) > params.qmax):
        from pgw.core.errors import RingOverflowError

        raise RingOverflowError(f"quantized activations exceed |xq| <= qmax={params.qmax}")
    n = xq.shape[1] if xq.ndim == 2 else 0
    out = uniform_rows(plan.rows, n, params)  # dummies; real rows overwritten below
    if pos.size:
        out[pos] = (xq.astype(np.int64).view(np.uint64) + r.astype(np.uint64)) & np.uint64(params.mask)
    return out


def unmask_expert_output(
    plan: DispatchPlan,
    expert: int,
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    wr: np.ndarray,
    checks: tuple[np.ndarray, np.ndarray],
    params: RingParams,
    in_features: int,
) -> np.ndarray:
    """Verify all ``C`` rows of ``expert`` and return the exact ``yq`` ``(count, m)`` of its real rows.

    Every row (dummy or real) gets the same Freivalds check first, so whether a tampered row is caught
    never depends on the routing; then the real rows are unmasked with their ``W r`` (order of
    :meth:`DispatchPlan.real`) and range-checked.

    Raises:
        IntegrityError: any row fails Freivalds, or a real output is out of range.
    """
    u64, uw64 = checks
    x_hat = np.asarray(x_hat, dtype=np.uint64)
    y_hat = np.asarray(y_hat, dtype=np.uint64)
    if x_hat.shape[0] != plan.rows or y_hat.shape[0] != plan.rows:
        raise ShapeError(f"expert {expert}: expected {plan.rows} rows, got x_hat {x_hat.shape} y_hat {y_hat.shape}")
    ok = freivalds_check_u64(x_hat, y_hat & np.uint64(params.mask), u64, uw64, params)
    if not bool(ok.all()):
        raise IntegrityError("compute service result failed the Freivalds check",
                             details={"linear": f"expert{expert}", "bad_rows": int((~ok).sum()), "rows": int(ok.size)})
    pos = plan.real(expert)
    wr = np.asarray(wr)
    if wr.shape != (pos.size, y_hat.shape[1]):
        raise ShapeError(f"expert {expert}: W r shape {wr.shape} != ({pos.size}, {y_hat.shape[1]})")
    return unmask_outputs_u64(y_hat[pos], wr.astype(np.uint64), params, in_features)
