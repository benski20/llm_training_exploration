"""Default Freivalds checks meet the soundness floor for 8B-class layers (quality review finding)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pgw.core import RING_K40_A12, RING_K48_A20, LinearId, LinearSpec, soundness_bits
from pgw.factory.config import FactorySettings
from pgw.factory.generator import MaskGenerator


def _specs(hidden: int, inter: int) -> list[LinearSpec]:
    return [LinearSpec(LinearId.parse("0.qkv"), 3 * hidden, hidden, True),
            LinearSpec(LinearId.parse("0.down"), hidden, inter, False)]


@pytest.mark.parametrize(("hidden", "inter"), [(4096, 14336), (3584, 18944), (896, 4864)])
@pytest.mark.parametrize("ring", [RING_K40_A12, RING_K48_A20])
def test_auto_checks_reach_the_floor_for_flagship_layers(hidden: int, inter: int, ring: object) -> None:
    gen = SimpleNamespace(specs=_specs(hidden, inter), ring=ring, MIN_SOUNDNESS_BITS=MaskGenerator.MIN_SOUNDNESS_BITS)
    c = MaskGenerator.checks_for(gen)  # type: ignore[arg-type]
    worst = min(soundness_bits(ring, s.in_features, c) for s in gen.specs)
    assert worst >= MaskGenerator.MIN_SOUNDNESS_BITS >= 20
    # the old fixed default (2) was below the floor for Llama-3-8B / Qwen-7B down_proj
    if inter >= 14336:
        assert soundness_bits(ring, inter, 2) < 16 and c >= 4


def test_factory_default_is_automatic_checks() -> None:
    assert FactorySettings.model_fields["checks"].default is None
