"""Guards: a proved guard never accepts an invalid string; unproved guards are refused by default."""
import itertools

import numpy as np
import pytest

from test_global import hand_built_dyck
from vt.artifact import save_artifact
from vt.cli import main
from vt.global_verify import DyckSpec
from vt.guard import Guard
from vt.serve import GateError


def test_proved_guard_equals_spec_on_all_inputs(tmp_path):
    art, cert = str(tmp_path / "m.npz"), str(tmp_path / "c.json")
    save_artifact(hand_built_dyck(10), art)
    assert main(["verify", art, "--certificate", cert, "--workers", "1"]) == 0
    g = Guard(art, cert)
    strings = ["".join(s) for s in itertools.product("()", repeat=10)]
    labels = [DyckSpec().label([c == ")" for c in s]) for s in strings]
    assert list(g.accepts(strings)) == [bool(y) for y in labels]
    assert not g.accepts(["(())", "(()))(()()x", ""]).any()     # wrong length / alphabet


def test_unproved_guard_refused_by_default(tmp_path):
    art = str(tmp_path / "m.npz")
    save_artifact(hand_built_dyck(6), art)
    with pytest.raises(GateError):
        Guard(art, None)
    assert Guard(art, None, alphabet="()", allow_unverified=True).proved is False
