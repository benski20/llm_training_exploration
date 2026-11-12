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


def test_cleared_threshold_and_three_way():
    import numpy as np

    from vt.verifier_pipeline import cleared_threshold, guaranteed_threshold, three_way
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    s = np.r_[rng.normal(0, 1, 400), rng.normal(3, 1, 400)]
    lo, hi = cleared_threshold(s, y, 0.05, 0.95), guaranteed_threshold(s, y, 0.1, 0.95)
    assert lo is not None and lo < hi
    v = three_way(s, lo, hi)
    assert y[v == "supported"].mean() <= 0.05 and (v[y == 0] == "unsupported").mean() <= 0.1
    assert cleared_threshold(s, np.ones_like(y), 0.05, 0.95) is None      # nothing can be cleared
