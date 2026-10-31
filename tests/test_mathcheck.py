"""vt/mathcheck.py: arithmetic written in a reply, re-computed exactly."""
import pytest

from vt.mathcheck import check_arithmetic, evaluate


def test_finds_and_checks_equations():
    got = check_arithmetic("First, 17 × 23 = 401. Then 401 + 9 = 410, and (4 + 6) / 2 = 5. Finally 2^10 = 1024.")
    assert [(g["expression"], g["ok"]) for g in got] == [("17 × 23", False), ("401 + 9", True),
                                                       ("(4 + 6) / 2", True), ("2^10", True)]
    assert got[0]["computed"] == 391


def test_percentages_thousands_and_rounding():
    got = check_arithmetic("15% of 80 is 12, 1,200 * 3 = 3,600 and 10 / 3 = 3.33")
    assert all(g["ok"] for g in got) and len(got) == 3
    assert not check_arithmetic("20% of 50 is 15")[0]["ok"]


def test_prose_without_arithmetic_and_plain_numbers_are_ignored():
    assert check_arithmetic("The tower is 330 metres tall and was built in 1889.") == []
    assert check_arithmetic("Version 2.0 is out.") == []


@pytest.mark.parametrize("bad", ["__import__('os')", "2 ** 99999", "a + 1", "[1] * 3"])
def test_the_evaluator_refuses_anything_but_arithmetic(bad):
    with pytest.raises((ValueError, SyntaxError)):
        evaluate(bad)


def test_latex_arithmetic_is_read():
    got = check_arithmetic(r"\( 17 \times 23 = 391 \) and \frac{10}{4} = 2.5, so $3 \cdot 4 = 13$")
    assert [(g["ok"]) for g in got] == [True, True, False]


def test_explicit_mode_reads_prose_as_prose():
    from vt.mathcheck import check_arithmetic
    prose = "The war (1914-1918) is 4 years long. In 1914 - 1918 = 4 years of fighting."
    assert check_arithmetic(prose, explicit=True) == []             # "is" is prose, a range of years no sum
    assert not check_arithmetic("so 12 * 4 = 50 dollars", explicit=True)[0]["ok"]
    assert check_arithmetic("12 * 4 is 50", explicit=True) == []


def test_percent_in_everyday_wording():
    from vt.mathcheck import check_arithmetic
    assert check_arithmetic("A 20% tip on $45 would be $9.")[0]["ok"]
    assert not check_arithmetic("15% of 80 comes to 14")[0]["ok"]


def test_operations_in_words():
    from vt.mathcheck import check_arithmetic
    assert check_arithmetic("17 times 23 is 391.")[0]["ok"]
    assert not check_arithmetic("100 divided by 4 equals 20")[0]["ok"]
    assert check_arithmetic("I visited 3 times in 2019.") == []
