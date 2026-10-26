"""Realize a checked plan as prose, and verify the prose independently.

The realizer may only express items the checker passed. Each item maps to a
fixed sentence template, and tone adds optional openers and closers from a
policy lexicon. The set of responses it can produce for a report is a small
finite language, so `verify_response` re-derives that language from the
report and policy and checks the text is exactly a member. That proves the
response states only checked content, in the approved register, and does
not depend on trusting the realizer.
"""
import itertools
import random
import re

from .check import fmt

DEFAULT_TONES = {
    "neutral": {"openers": [], "closers": []},
    "warm": {"openers": ["Happy to help!", "Good question!"], "closers": ["Hope that helps."]},
}
OMITTED = "Some statements were left out because they could not be verified."
REFUSAL = "I can't help with that request."


def _pretty(name):
    return name.replace("_", " ")


def content_sentences(report, policy):
    """The sentences the checked items license, in plan order."""
    if report["intent"]["status"] != "allowed":
        return [REFUSAL], False
    templates = policy.get("relation_templates", {})
    out, omitted = [], False
    for c in report["claims"]:
        if c["status"] != "verified":
            omitted = True
            continue
        cl = c["claim"]
        t = templates.get(cl.relation, "{subject}'s " + _pretty(cl.relation) + " is {object}.")
        out.append(t.format(subject=cl.subject, object=c["kb_object"]))
    for s in report["steps"]:
        if s["status"] != "verified":
            omitted = True
            continue
        out.append(f"{_pretty(s['step'].name).capitalize()} is {s['step'].expr}, which is {fmt(s['value'])}.")
    a = report.get("answer")
    if a is not None:
        if a["status"] == "verified":
            out.append(f"So the answer is {fmt(a['value'])}.")
        else:
            omitted = True
    return out, omitted


def allowed_responses(report, policy):
    """The finite language of responses licensed by this report and policy."""
    body, omitted = content_sentences(report, policy)
    if body == [REFUSAL]:
        return [REFUSAL]
    tones = policy.get("tone_lexicon", DEFAULT_TONES)
    tone = tones[report["intent"]["tone"]]
    tail = [OMITTED] if omitted else []
    out = []
    for op, cl in itertools.product([None] + tone["openers"], [None] + tone["closers"]):
        parts = ([op] if op else []) + body + tail + ([cl] if cl else [])
        out.append(" ".join(parts))
    if tone.get("require_marker"):
        out = [r for r in out if any(m in r for m in tone["require_marker"])]
    return out


def realize(report, policy, seed=0):
    options = allowed_responses(report, policy)
    if not options:
        raise ValueError("no response satisfies the tone constraints")
    return random.Random(seed).choice(options)


def tone_violations(text, policy):
    """Formal register constraints, checked on the final string."""
    issues = []
    for w in policy.get("banned_words", []):
        if re.search(r"\b" + re.escape(w.casefold()) + r"\b", text.casefold()):
            issues.append(f"banned word {w!r}")
    max_words = policy.get("max_sentence_words")
    if max_words:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if len(sent.split()) > max_words:
                issues.append(f"sentence longer than {max_words} words")
    if policy.get("no_shouting") and re.search(r"\b[A-Z]{4,}\b", text):
        issues.append("all-caps word")
    return issues


def verify_response(text, report, policy):
    """Independent check of a realized response."""
    member = text in set(allowed_responses(report, policy))
    issues = tone_violations(text, policy)
    return {"member_of_licensed_language": member, "tone_violations": issues,
            "ok": member and not issues}
