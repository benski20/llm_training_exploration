"""Check a plan: claims against a knowledge base, reasoning steps by exact
evaluation, intent against an explicit policy.

Nothing in the plan is executed as code. Expressions are parsed with `ast` and
evaluated by a whitelist interpreter over exact rationals (Fraction), with
limits on size, exponents and magnitudes, so a hostile planner cannot run
code or exhaust memory.
"""
import ast
import re
from fractions import Fraction

from .plan import Plan

MAX_EXPR = 200
MAX_EXP = 64
MAX_BITS = 6644          # about 2000 decimal digits


class EvalError(ValueError):
    pass


def _num(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return Fraction(str(v))
    raise EvalError(f"unsupported literal {v!r}")


def safe_eval(expr, env):
    """Exact evaluation of + - * / ** (small integer exponents), comparisons,
    and/or/not, numbers and previously defined step names."""
    if len(expr) > MAX_EXPR:
        raise EvalError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise EvalError(f"syntax error: {e.msg}") from None

    def big(x):
        if isinstance(x, Fraction) and (x.numerator.bit_length() > MAX_BITS or
                                        x.denominator.bit_length() > MAX_BITS):
            raise EvalError("number too large")
        return x

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return _num(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise EvalError(f"unknown name {n.id!r}")
            return env[n.id]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return not v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(a, bool) or isinstance(b, bool):
                raise EvalError("arithmetic on booleans")
            if isinstance(n.op, ast.Add):
                return big(a + b)
            if isinstance(n.op, ast.Sub):
                return big(a - b)
            if isinstance(n.op, ast.Mult):
                return big(a * b)
            if isinstance(n.op, ast.Div):
                if b == 0:
                    raise EvalError("division by zero")
                return big(a / b)
            if isinstance(n.op, ast.Pow):
                if b.denominator != 1 or abs(b) > MAX_EXP:
                    raise EvalError("exponent must be an integer with |e| <= 64")
                if a == 0 and b < 0:
                    raise EvalError("division by zero")
                size = max(a.numerator.bit_length(), a.denominator.bit_length()) * abs(int(b))
                if size > MAX_BITS:                       # reject before computing
                    raise EvalError("number too large")
                return big(a ** int(b))
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right_node in zip(n.ops, n.comparators):
                right = ev(right_node)
                ok = {ast.Eq: left == right, ast.NotEq: left != right, ast.Lt: left < right,
                      ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right}.get(type(op))
                if ok is None:
                    raise EvalError("unsupported comparison")
                if not ok:
                    return False
                left = right
            return True
        raise EvalError(f"unsupported syntax: {type(n).__name__}")
    return ev(tree)


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().casefold())


class KnowledgeBase:
    """Trusted facts: (subject, relation) -> object."""

    def __init__(self, triples):
        self.facts = {}
        for t in triples:
            self.facts[(_norm(t["subject"]), _norm(t["relation"]))] = t["object"]

    def check(self, claim):
        obj = self.facts.get((_norm(claim.subject), _norm(claim.relation)))
        if obj is None:
            return "unsupported", None
        return ("verified" if _norm(obj) == _norm(claim.object) else "contradicted"), obj


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


def check_plan(plan: Plan, kb: KnowledgeBase, policy: dict):
    """Returns a report: every item with a status and the kind of guarantee."""
    rep = {"intent": None, "claims": [], "steps": [], "answer": None}
    topic_ok = plan.topic in policy.get("allowed_topics", [])
    purpose_ok = plan.purpose not in policy.get("denied_purposes", [])
    tone_ok = plan.tone in policy.get("tones", ["neutral"])
    rep["intent"] = {"topic": plan.topic, "purpose": plan.purpose, "tone": plan.tone,
                     "status": "allowed" if (topic_ok and purpose_ok and tone_ok) else "refused",
                     "guarantee": "checked against explicit policy"}
    for c in plan.claims:
        status, kb_obj = kb.check(c)
        rep["claims"].append({"claim": c, "status": status, "kb_object": kb_obj,
                              "guarantee": "checked against knowledge base"})
    env, failed = {}, set()
    for s in plan.steps:
        try:
            v = safe_eval(s.expr, env)
            if s.value is not None:
                claimed = safe_eval(s.value, {})
                if claimed != v:
                    failed.add(s.name)
                    rep["steps"].append({"step": s, "status": "wrong_value", "value": v,
                                         "claimed": claimed, "guarantee": "exact evaluation"})
                    continue
            env[s.name] = v
            rep["steps"].append({"step": s, "status": "verified", "value": v,
                                 "guarantee": "exact evaluation"})
        except EvalError as e:
            failed.add(s.name)
            rep["steps"].append({"step": s, "status": "invalid", "error": str(e),
                                 "guarantee": "exact evaluation"})
    if plan.answer is not None:
        try:
            v = safe_eval(plan.answer, env)
            rep["answer"] = {"expr": plan.answer, "status": "verified", "value": v,
                             "guarantee": "exact evaluation"}
        except EvalError as e:
            rep["answer"] = {"expr": plan.answer, "status": "invalid", "error": str(e),
                             "guarantee": "exact evaluation"}
    return rep
