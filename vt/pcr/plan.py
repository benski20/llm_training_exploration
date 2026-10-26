"""The plan: a planner's structured, checkable intermediate representation.

A planner (any model, untrusted) does not write prose. It writes a plan:

    {"intent": {"topic": "arithmetic", "purpose": "answer_question"},
     "tone": "warm",
     "claims": [{"subject": "Ada Lovelace", "relation": "born_in", "object": "1815"}],
     "steps": [{"name": "total", "expr": "3 * 4 + 2"}],
     "answer": {"expr": "total"}}

Everything here is data from an untrusted source, so parsing is strict:
unknown keys are rejected, sizes are bounded, and strings are length-limited
and restricted to printable characters.
"""
from dataclasses import dataclass, field

MAX_ITEMS = 64
MAX_STR = 200
NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz_0123456789")


class PlanError(ValueError):
    pass


def _str(v, what, max_len=MAX_STR):
    if not isinstance(v, str) or not v.strip():
        raise PlanError(f"{what} must be a non-empty string")
    if len(v) > max_len or not v.isprintable():
        raise PlanError(f"{what} too long or not printable")
    return v.strip()


def _keys(d, allowed, required, what):
    if not isinstance(d, dict):
        raise PlanError(f"{what} must be an object")
    extra = set(d) - set(allowed)
    missing = set(required) - set(d)
    if extra or missing:
        raise PlanError(f"{what}: unexpected {sorted(extra)} / missing {sorted(missing)}")


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Step:
    name: str
    expr: str
    value: str = None        # optional value the planner claims for expr


@dataclass
class Plan:
    topic: str
    purpose: str
    tone: str = "neutral"
    claims: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    answer: str = None

    @classmethod
    def from_json(cls, d):
        _keys(d, {"intent", "tone", "claims", "steps", "answer"}, {"intent"}, "plan")
        _keys(d["intent"], {"topic", "purpose"}, {"topic", "purpose"}, "intent")
        claims, steps = d.get("claims", []), d.get("steps", [])
        if not isinstance(claims, list) or not isinstance(steps, list) or \
                len(claims) > MAX_ITEMS or len(steps) > MAX_ITEMS:
            raise PlanError(f"claims/steps must be lists of at most {MAX_ITEMS}")
        cs = []
        for c in claims:
            _keys(c, {"subject", "relation", "object"}, {"subject", "relation", "object"}, "claim")
            cs.append(Claim(_str(c["subject"], "subject"), _str(c["relation"], "relation"),
                            _str(c["object"], "object")))
        ss, names = [], set()
        for s in steps:
            _keys(s, {"name", "expr", "value"}, {"name", "expr"}, "step")
            name = _str(s["name"], "step name", 40)
            if set(name) - NAME_CHARS or name[0].isdigit() or name in names:
                raise PlanError(f"bad or duplicate step name {name!r}")
            names.add(name)
            val = _str(str(s["value"]), "step value", 60) if "value" in s else None
            ss.append(Step(name, _str(s["expr"], "step expr"), val))
        ans = None
        if "answer" in d:
            _keys(d["answer"], {"expr"}, {"expr"}, "answer")
            ans = _str(d["answer"]["expr"], "answer expr")
        tone = _str(d.get("tone", "neutral"), "tone", 20)
        return cls(_str(d["intent"]["topic"], "topic", 40), _str(d["intent"]["purpose"], "purpose", 40),
                   tone, cs, ss, ans)
