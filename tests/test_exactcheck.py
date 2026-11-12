from vt.exactcheck import check, numbers


def test_numbers_with_scales():
    assert [round(v) for v, _, _ in numbers("Revenue was $1.2 billion, up 15%.")] == [1200000000, 15]


def test_number_in_evidence_or_one_step_away():
    ev = "Sales were 400 units in March and 100 units in April."
    assert check("They sold 400 units in March.", ev)["numbers"] == []
    assert check("Sales fell by 300 units.", ev)["numbers"] == []          # 400 - 100
    assert check("Sales fell 75%.", ev)["numbers"] == []                   # (100 - 400) / 400
    assert check("They sold 950 units.", ev)["numbers"] == ["950"]


def test_years_and_small_counts_are_not_checked():
    assert check("In 2019 three of 5 teams won.", "Nothing relevant.")["numbers"] == []


def test_quotes():
    ev = "The minister said the plan was 'a careful and fair compromise for everyone'."
    assert check('He called it "a careful and fair compromise".', ev)["quotes"] == []
    assert check('He called it "a reckless and unfair giveaway".', ev)["quotes"] == ["a reckless and unfair giveaway"]
