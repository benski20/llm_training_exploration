"""Static checks of the code in a reply: Python and JSON blocks are parsed, never run (running a model's code on
a public server is the risk; parsing is not). A block that does not parse is a definite, checkable error;
one that parses says nothing about whether it is right, so it earns no verdict of its own.

    check_code("```python\\ndef f(x)\\n    return x\\n```")
    # -> [{"language": "python", "ok": False, "error": "expected ':' (line 1)"}]
"""
import ast
import json
import re
import warnings

FENCE = re.compile(r"```[ \t]*([\w+#-]*)[^\n]*\n(.*?)```", re.S)
PYTHON = {"python", "py", "python3"}


def check_code(text, max_blocks=10, max_chars=50_000):
    """[{"language", "ok", "error"?}] for each Python or JSON block in the text (others are skipped)."""
    out = []
    for m in list(FENCE.finditer(text))[:max_blocks]:
        lang, code = m.group(1).lower(), m.group(2)[:max_chars]
        if lang in PYTHON:
            try:
                with warnings.catch_warnings():              # invalid escapes etc. are warnings, not errors
                    warnings.simplefilter("ignore")
                    ast.parse(code)
                out.append({"language": "python", "ok": True})
            except SyntaxError as e:
                out.append({"language": "python", "ok": False, "error": f"{e.msg} (line {e.lineno})"})
            except (ValueError, RecursionError, MemoryError) as e:   # null bytes, absurd nesting
                out.append({"language": "python", "ok": False, "error": type(e).__name__})
        elif lang == "json":
            try:
                json.loads(code)
                out.append({"language": "json", "ok": True})
            except ValueError as e:
                out.append({"language": "json", "ok": False, "error": str(e)[:120]})
    return out
