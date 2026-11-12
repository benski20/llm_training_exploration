"""Exact checks of a reply against its evidence, for the claims that can be checked mechanically.

Numbers: every number a sentence states should be in the evidence (any thousand / million / billion scaling,
1% tolerance for rounding), or follow from two of the evidence's numbers by one operation (sum, difference,
product, ratio, percent or percent change). A year or a small count (0-12) is not checked: they are too often
written as words, implied or inferred.

Quotes: text in double quotes (three words or more) should appear in the evidence: verbatim after normalising
case, punctuation and spacing, or with at least 90% of a same-length window matching.

These never consult a model, so a flag is a fact about the text ("this figure is not in the source and does
not follow from it by one step"), not a judgement; whether that makes the sentence wrong is the probe's call.
"""
import difflib
import itertools
import re

NUM = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?\s*(?:%|percent|per cent|billion|million|thousand|bn|k)?",
                 re.I)
SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
WORDS = {"billion": 1e9, "bn": 1e9, "million": 1e6, "thousand": 1e3, "k": 1e3}


def numbers(text):
    """(value, span text, is_percent) for each number in the text."""
    out = []
    for m in NUM.finditer(text):
        raw = m.group(0).strip()
        digits = re.match(r"-?\$?([\d,]*\d(?:\.\d+)?)", raw)
        if not digits:
            continue
        try:
            v = float(digits.group(1).replace(",", ""))
        except ValueError:
            continue
        if raw.startswith("-"):
            v = -v
        unit = re.sub(r"[\d.,$\s-]", "", raw).lower()
        v *= WORDS.get(unit, 1)
        out.append((v, raw, "%" in raw or "percent" in unit or "cent" in unit))
    return out


def _close(a, b, tol=0.01):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def number_supported(v, pct, ev_vals, max_pairs=4000):
    if any(_close(abs(v) * s, abs(e)) for e in ev_vals for s in SCALES):
        return True
    vals = list(dict.fromkeys(abs(e) for e in ev_vals if e))[:90]
    for a, b in itertools.islice(itertools.permutations(vals, 2), max_pairs):
        derived = (a + b, a - b, a * b, a / b, 100 * a / b, 100 * (a - b) / b)
        if any(_close(abs(v), abs(d), 0.01) or _close(abs(v), abs(d) / 1e3, 0.01) or _close(abs(v), abs(d) * 1e3, 0.01)
               for d in derived):
            return True
    return False


def unsupported_numbers(sentence, evidence):
    ev = [v for v, _, _ in numbers(evidence)]
    out = []
    for v, raw, pct in numbers(sentence):
        if not pct and float(v).is_integer() and (0 <= v <= 12 or 1800 <= v <= 2100):
            continue                                            # counts and years: not checked
        if not number_supported(v, pct, ev):
            out.append(raw)
    return out


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def unsupported_quotes(sentence, evidence):
    ev = _norm(evidence)
    out = []
    for q in re.findall(r"[\"“]([^\"”]{8,}?)[\"”]", sentence):
        nq = _norm(q)
        if len(nq.split()) < 3 or nq in ev:
            continue
        words, evw = nq.split(), ev.split()
        n = len(words)
        best = max((difflib.SequenceMatcher(None, words, evw[i:i + n]).ratio()
                    for i in range(0, max(1, len(evw) - n + 1))), default=0.0)
        if best < 0.9:
            out.append(q)
    return out


def check(sentence, evidence):
    """{'numbers': [...], 'quotes': [...]}: the sentence's figures and quotations the evidence does not back."""
    return {"numbers": unsupported_numbers(sentence, evidence), "quotes": unsupported_quotes(sentence, evidence)}
