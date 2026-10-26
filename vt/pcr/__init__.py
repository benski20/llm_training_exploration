"""Plan -> Check -> Realize: responses whose truthfulness, reasoning, harmlessness
and tone come with stated guarantees.

    from vt.pcr import respond
    out = respond(plan_json, kb_triples, policy)
    out["text"], out["guarantees"]

Guarantee levels, from strongest:
  proved                  exact evaluation / exact language membership / lexicon checks
  checked_against_kb      a claim matches the trusted knowledge base
  checked_against_policy  the declared intent is allowed by the explicit policy
  statistical             a conformally calibrated judge: P(bad passes) <= alpha
  unverified              no check applied (reported, never hidden)
"""
from .check import KnowledgeBase, check_plan, fmt, safe_eval  # noqa: F401
from .judge import ConformalJudge  # noqa: F401
from .plan import Plan, PlanError  # noqa: F401
from .realize import allowed_responses, realize, verify_response  # noqa: F401


def respond(plan_json, kb_triples, policy, judges=(), seed=0):
    """Check an untrusted plan, realize only what passed, verify the text, and
    report what is guaranteed and how. Judges: (property, ConformalJudge) pairs
    applied to the final text."""
    plan = Plan.from_json(plan_json)
    report = check_plan(plan, KnowledgeBase(kb_triples), policy)
    text = realize(report, policy, seed)
    v = verify_response(text, report, policy)
    if not v["ok"]:                             # never release unverified text
        raise RuntimeError(f"realized text failed verification: {v}")
    claims = report["claims"]
    steps = report["steps"]
    refused = report["intent"]["status"] != "allowed"
    g = {
        "truthfulness": {
            "level": "checked_against_kb",
            "stated": 0 if refused else sum(c["status"] == "verified" for c in claims),
            "omitted": {s: sum(c["status"] == s for c in claims) for s in ("contradicted", "unsupported")},
            "statement": "every factual claim in the text matches the knowledge base; "
                         "unverified claims were omitted",
        },
        "reasoning": {
            "level": "proved",
            "stated_steps": 0 if refused else sum(s["status"] == "verified" for s in steps),
            "rejected_steps": [s["step"].name for s in steps if s["status"] != "verified"],
            "answer": None if (refused or not report["answer"] or report["answer"]["status"] != "verified")
            else fmt(report["answer"]["value"]),
            "statement": "every arithmetic/logic step and the answer were evaluated exactly; "
                         "the translation of the question into steps is not verified",
        },
        "harmlessness": {
            "level": "checked_against_policy",
            "intent": report["intent"]["status"],
            "banned_lexicon": "proved absent",
        },
        "tone": {"level": "proved", "register": plan.tone,
                 "statement": "text is a member of the licensed language for this register; "
                              "lexicon and length constraints hold"},
        "realization": {"level": "proved", "statement": "text expresses only checked items "
                        "(independent membership check)"},
    }
    for prop, judge in judges:
        ok = judge.passes(text)
        g.setdefault(prop, {})["statistical"] = {**judge.guarantee(), "passed": bool(ok)}
        if not ok:
            text = "I can't share that response."
            g[prop]["action"] = "withheld"
    return {"text": text, "guarantees": g, "report": report}
