"""Checks adopted from the Rust core (INTERFACE_NOTES: rust-core hardening review)."""

from __future__ import annotations

import pytest
import torch

from pgw.core import RING_K40_A12, ConfigError, RingOverflowError, mask_activations, quantize_weight_rows
from pgw.core.freivalds import freivalds_check


def test_freivalds_refuses_zero_check_vectors() -> None:
    p = RING_K40_A12
    x = torch.zeros((2, 8), dtype=torch.int64)
    y = torch.zeros((2, 4), dtype=torch.int64)
    with pytest.raises(ConfigError, match="checks=0"):
        freivalds_check(x, y, torch.zeros((0, 4), dtype=torch.int64), torch.zeros((0, 8), dtype=torch.int64), p)


def test_mask_activations_enforces_qmax() -> None:
    p = RING_K40_A12
    r = torch.zeros((1, 4), dtype=torch.int64)
    ok = torch.tensor([[p.qmax, -p.qmax, 0, 1]])
    assert mask_activations(ok, r, p).shape == (1, 4)
    with pytest.raises(RingOverflowError):
        mask_activations(torch.tensor([[p.qmax + 1, 0, 0, 0]]), r, p)
    with pytest.raises(RingOverflowError):
        mask_activations(torch.tensor([[-p.qmax - 1, 0, 0, 0]]), r, p)
    empty = torch.zeros((0, 4), dtype=torch.int64)
    assert mask_activations(empty, empty, p).shape == (0, 4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_quantize_weight_rows_rejects_non_finite(bad: float) -> None:
    w = torch.ones((2, 3))
    w[1, 2] = bad
    with pytest.raises(RingOverflowError):
        quantize_weight_rows(w)
