"""The provenance check releases a year only if it was copied from a fact about the asked entity."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from provenance_by_design import provenance_ok  # noqa: E402

P = "P3 born 1801; P7 born 1777; P12 born 1843. Q P7 born? A "


def src(year_start):
    return [year_start + i for i in range(4)]


def test_accepts_copy_from_the_asked_entity():
    assert provenance_ok(P, "P7", src(P.index("1777")))


def test_rejects_wrong_entity_noncontiguous_or_partial_copies():
    assert not provenance_ok(P, "P7", src(P.index("1843")))            # another person's year
    assert not provenance_ok(P, "P1", src(P.index("1801")))            # prefix of P12/P1 confusion
    s = src(P.index("1777")); s[2] += 1
    assert not provenance_ok(P, "P7", s)                               # not contiguous
    assert not provenance_ok(P, "P7", src(P.index("1777"))[:3])        # partial copy
    assert not provenance_ok(P, "P7", [None] * 4)                      # generated, not copied
    q = P.index("Q P7")
    assert not provenance_ok(P, "P7", src(q))                          # copied from the question
