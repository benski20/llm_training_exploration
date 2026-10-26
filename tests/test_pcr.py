"""Plan -> Check -> Realize: exact checking, safe evaluation, independent output
verification, conformal guarantee, and fuzzing with hostile plans."""
import json
import os
import random
from fractions import Fraction

import numpy as np
import pytest

from vt.pcr import (ConformalJudge, KnowledgeBase, Plan, PlanError, allowed_responses, check_plan,
                    respond, safe_eval, verify_response)
from vt.pcr.check import EvalError

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "pcr")
KB = json.load(open(os.path.join(EX, "kb.json")))
POLICY = json.load(open(os.path.join(EX, "policy.json")))


def plan(name):
    return json.load(open(os.path.join(EX, f"plan_{name}.json")))


def test_safe_eval_is_exact():
    assert safe_eval("1/3 + 1/6", {}) == Fraction(1, 2)
    assert safe_eval("0.1 + 0.2 == 0.3", {}) is True          # exact, unlike floats
    assert safe_eval("x * 2 ** 10", {"x": Fraction(3)}) == 3072
    assert safe_eval("1 < 2 < 3 and not 2 > 3", {}) is True


@pytest.mark.parametrize("expr", [
    "__import__('os').system('true')", "(lambda: 1)()", "x.__class__", "[1][0]", "open('f')",
    "2 ** 10 ** 10", "2 ** 0.5", "1 / 0", "y + 1", "1 if 1 else 2", "{1: 2}", "'a' * 10",
])
def test_safe_eval_rejects(expr):
    with pytest.raises(EvalError):
        safe_eval(expr, {"x": Fraction(1)})


def test_number_growth_is_bounded():
    env = {"a": safe_eval("10 ** 64", {})}
    env["b"] = safe_eval("a ** 30", env)                  # 1920 digits: allowed
    for expr in ("a ** 64", "b * b", "b ** 2", "(b * 10) ** 64"):
        with pytest.raises(EvalError):
            safe_eval(expr, env)


def test_plan_parsing_is_strict():
    good = plan("good")
    Plan.from_json(good)
    for bad in [dict(good, extra=1), dict(good, claims=[{"subject": "x", "relation": "y"}]),
                dict(good, claims=[{"subject": "x\x00", "relation": "y", "object": "z"}] ),
                dict(good, steps=[{"name": "1bad", "expr": "1"}]),
                dict(good, steps=[{"name": "a", "expr": "1"}, {"name": "a", "expr": "2"}]),
                dict(good, claims=[{"subject": "s", "relation": "r", "object": "o"}] * 65)]:
        with pytest.raises(PlanError):
            Plan.from_json(bad)


def test_good_flawed_harmful():
    g = respond(plan("good"), KB, POLICY, seed=1)
    assert "1815" in g["text"] and "28" in g["text"]
    assert g["guarantees"]["reasoning"]["answer"] == "28"
    f = respond(plan("flawed"), KB, POLICY)
    assert "1816" not in f["text"] and "telephone" not in f["text"] and "29" not in f["text"]
    assert f["guarantees"]["truthfulness"]["omitted"] == {"contradicted": 1, "unsupported": 1}
    assert set(f["guarantees"]["reasoning"]["rejected_steps"]) == {"years_until_1843", "trick"}
    assert f["guarantees"]["reasoning"]["answer"] is None
    h = respond(plan("harmful"), KB, POLICY)
    assert h["text"] == "I can't help with that request."
    assert h["guarantees"]["harmlessness"]["intent"] == "refused"


def test_verifier_rejects_any_edit():
    p = Plan.from_json(plan("good"))
    rep = check_plan(p, KnowledgeBase(KB), POLICY)
    text = allowed_responses(rep, POLICY)[0]
    assert verify_response(text, rep, POLICY)["ok"]
    for bad in [text.replace("1815", "1816"), text + " Also, the Moon is cheese.",
                text.replace("Ada", "Obviously Ada"), text.replace(".", "!", 1), text.upper()]:
        assert not verify_response(bad, rep, POLICY)["ok"], bad


def test_tone_constraints_on_lexicon():
    pol = dict(POLICY, banned_words=["babbage"])
    p = Plan.from_json(plan("good"))
    rep = check_plan(p, KnowledgeBase(KB), pol)
    for t in allowed_responses(rep, pol):
        assert not verify_response(t, rep, pol)["ok"]       # every licensed text names Babbage
    with pytest.raises(RuntimeError):
        respond(plan("good"), KB, pol)                       # never releases a violating text


def test_conformal_judge_bound_holds_empirically():
    rng = np.random.default_rng(0)
    alpha, misses = 0.1, []
    for _ in range(300):
        cal = rng.normal(2.0, 1.0, 199)                      # scores of known-bad outputs
        j = ConformalJudge(lambda x: x)
        j.calibrate(cal, alpha)
        fresh = rng.normal(2.0, 1.0, 1000)
        misses.append(np.mean([j.passes(x) for x in fresh]))
    assert np.mean(misses) <= alpha + 0.01
    with pytest.raises(ValueError):
        ConformalJudge(lambda x: x).calibrate([1.0, 2.0], 0.1)    # too few to certify alpha


def test_fuzz_hostile_plans_never_release_unverified_text():
    rnd = random.Random(0)
    atoms = ["1", "2.5", "x", "a", "__import__", "(", ")", "+", "*", "**", "/", "-", "==", "and",
             "lambda", ".", "[", "]", "'s'", "99999"]
    rels = ["born_in", "designed_by", "field", "made_up"]
    for i in range(400):
        steps = [{"name": f"s{k}", "expr": " ".join(rnd.choice(atoms) for _ in range(rnd.randint(1, 7)))}
                 for k in range(rnd.randint(0, 4))]
        claims = [{"subject": rnd.choice(["Ada Lovelace", "Analytical Engine", "Bob"]),
                   "relation": rnd.choice(rels), "object": rnd.choice(["1815", "Charles Babbage", "x"])}
                  for _ in range(rnd.randint(0, 3))]
        pj = {"intent": {"topic": rnd.choice(["arithmetic", "chemistry"]),
                         "purpose": rnd.choice(["answer_question", "weapons"])},
              "tone": rnd.choice(["neutral", "warm"]), "claims": claims, "steps": steps}
        if steps and rnd.random() < 0.5:
            pj["answer"] = {"expr": steps[-1]["name"]}
        out = respond(pj, KB, POLICY, seed=i)
        rep = out["report"]
        assert verify_response(out["text"], rep, POLICY)["ok"]
        for c in rep["claims"]:                                       # unverified objects never leak
            if c["status"] != "verified" and c["claim"].object not in {"1815", "Charles Babbage"}:
                assert c["claim"].object not in out["text"].split()
