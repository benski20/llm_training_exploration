"""Mixture-of-experts building blocks for masked serving (opt-in, nothing here is used by default).

* :mod:`pgw.moe.oblivious`: routing-oblivious expert dispatch. Every expert receives the same public
  number of rows per trip; rows of tokens not routed to it are uniform dummies that need no mask slot.
* :mod:`pgw.moe.refill`: a routing-independent refill schedule for per-expert mask pools.

Design and security argument: ``research/plans/scale-north-star.md`` (section "MoE").
"""

from pgw.moe.oblivious import (
    DispatchPlan,
    assemble_expert_input,
    capacity_rows,
    plan_dispatch,
    provider_view,
    unmask_expert_output,
)
from pgw.moe.refill import LockstepRefill, RefillSimulation, simulate_refill

__all__ = [
    "DispatchPlan",
    "LockstepRefill",
    "RefillSimulation",
    "assemble_expert_input",
    "capacity_rows",
    "plan_dispatch",
    "provider_view",
    "simulate_refill",
    "unmask_expert_output",
]
