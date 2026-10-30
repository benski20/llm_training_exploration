"""Exact checks of the arithmetic written in a reply: "17 × 23 = 391", "(4 + 6) / 2 = 5", "15% of 80 is 12".
Each equation is re-computed with a safe evaluator (numbers and + - * / ^ ( ) only: no names, no calls, small
exponents), and a mismatch is reported with the right value. It costs nothing, needs no model or lookup, and
covers a frequent class of reasoning errors: a wrong step in a calculation.

    check_arithmetic("So 17 × 23 = 401, and 401 + 9 = 410.")
    # -> [{"expression": "17 × 23", "stated": 401.0, "computed": 391.0, "ok": False}, {... "ok": True}]
"""
import ast
import operator
import re

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
# an arithmetic expression with at least one operator, then "=" or "is"/"equals", then a number
EQUATION = re.compile(rf"((?:\(?\s*-?{NUM}\s*\)?\s*[-+*/×÷x^·]\s*)+\(?\s*-?{NUM}\s*\)?)\s*(?:=|equals|is)\s*\$?\s*(-?{NUM})(%?)")
PERCENT = re.compile(rf"({NUM})\s*%\s*(?:\w+\s+)?(?:of|on)\s*\$?\s*({NUM})\s*(?:\w+\s+)?"
                     rf"(?:=|equals|is|would be|comes to|will be)\s*\$?\s*(-?{NUM})")    # "20% tip on $45 would be $9"


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("not arithmetic")


def evaluate(expr):
    """The value of a plain arithmetic expression, or ValueError (anything but numbers and operators)."""
    e = expr.replace(",", "").replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")
    e = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", e)            # "3 x 4"
    if len(e) > 200:
        raise ValueError("too long")
    return float(_eval(ast.parse(e, mode="eval")))


def _num(s):
    return float(s.replace(",", ""))


def close(a, b):
    return abs(a - b) <= max(1e-6, 0.005 * max(abs(a), abs(b)))  # stated values are often rounded


def plain(text):
    """LaTeX and markdown math as plain arithmetic: \\( 17 \\times 23 = 391 \\) -> 17 × 23 = 391."""
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    for a, b in (("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"), ("\\left", ""), ("\\right", ""),
                 ("\\(", " "), ("\\)", " "), ("\\[", " "), ("\\]", " "), ("$", " "), ("{", ""), ("}", "")):
        t = t.replace(a, b)
    return re.sub(r"\\[a-zA-Z]+", " ", t)


YEARS = re.compile(r"^\(?\s*\d{4}\s*\)?\s*[-–]\s*\(?\s*\d{4}\s*\)?$")      # 1914-1918: a range, not a sum


def check_arithmetic(text, explicit=False):
    """Every equation written in the text, re-computed: [{"expression", "stated", "computed", "ok"}].
    explicit: only equations written with "=" or "equals" (for replies that are not about a calculation,
    where "1914-1918 is 4 years" is prose), and never a range of years."""
    text = plain(text)
    out = []
    for m in [] if explicit else PERCENT.finditer(text):
        computed, stated = _num(m.group(1)) / 100 * _num(m.group(2)), _num(m.group(3))
        out.append({"expression": f"{m.group(1)}% of {m.group(2)}", "stated": stated, "computed": round(computed, 6),
                    "ok": close(computed, stated)})
    for m in EQUATION.finditer(text):
        expr = m.group(1).strip()
        if explicit and (re.search(r"\bis\b", text[m.end(1):m.start(2)]) or YEARS.match(expr)):
            continue
        try:
            computed = evaluate(expr)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        stated = _num(m.group(2)) / (100 if m.group(3) else 1)
        out.append({"expression": expr, "stated": stated, "computed": round(computed, 6), "ok": close(computed, stated)})
    return out
