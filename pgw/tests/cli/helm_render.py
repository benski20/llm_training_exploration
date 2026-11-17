"""A small, strict renderer for the Go-template/Sprig subset used by PGW's Helm charts.

``helm`` is not available in every environment where the tests run, so the chart tests render the
templates with this module and then validate the YAML and the Kubernetes objects. CI additionally
runs the real ``helm lint`` / ``helm template`` (see ``.github/workflows/ci.yml``).

Supported: actions with ``{{-``/``-}}`` trimming and comments; ``if/else if/else``, ``range``
(lists and maps, with ``$i, $v :=``), ``with``, ``define``, ``template``, ``include``; variables
(``:=`` and ``=``); pipelines and parenthesized sub-pipelines; field access on ``.``/``$``/variables;
and the functions in :data:`FUNCS`. The renderer is deliberately *stricter* than Helm: unknown
functions, field access on nil and malformed syntax raise :class:`RenderError`.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["RenderError", "deep_merge", "render_chart", "render_chart_objects"]


class RenderError(Exception):
    """Template syntax or evaluation error (or a template ``fail``/``required``)."""


# ---- parsing -------------------------------------------------------------------------------------
@dataclass
class Text:
    text: str


@dataclass
class Action:
    expr: str


@dataclass
class Block:
    kind: str  # if | range | with | define
    branches: list[tuple[str | None, list[Any]]] = field(default_factory=list)  # (expr, body); else -> None


_ACTION_RE = re.compile(r"\{\{(-\s)?(.*?)(\s-)?\}\}", re.S)


def _lex(src: str) -> list[Text | Action]:
    out: list[Text | Action] = []
    pos = 0
    trim_next = False
    for m in _ACTION_RE.finditer(src):
        text = src[pos:m.start()]
        if trim_next:
            text = text.lstrip()
        if m.group(1):
            text = text.rstrip()
        if text:
            out.append(Text(text))
        body = m.group(2).strip()
        if not (body.startswith("/*") and body.endswith("*/")):
            out.append(Action(body))
        trim_next = bool(m.group(3))
        pos = m.end()
    tail = src[pos:]
    if trim_next:
        tail = tail.lstrip()
    if "{{" in tail or "}}" in tail:
        raise RenderError("unbalanced template delimiters")
    if tail:
        out.append(Text(tail))
    return out


def _parse(tokens: list[Text | Action]) -> tuple[list[Any], dict[str, list[Any]]]:
    defines: dict[str, list[Any]] = {}
    root: list[Any] = []
    stack: list[tuple[Block, list[Any]]] = []
    cur = root
    for t in tokens:
        if isinstance(t, Text):
            cur.append(t)
            continue
        word, _, rest = t.expr.partition(" ")
        rest = rest.strip()
        if word in ("if", "range", "with", "define"):
            blk = Block(word, [(rest, [])])
            cur.append(blk)
            stack.append((blk, cur))
            cur = blk.branches[-1][1]
        elif word == "else":
            if not stack:
                raise RenderError("else without if/range/with")
            blk = stack[-1][0]
            if rest.startswith("if "):
                if blk.kind != "if":
                    raise RenderError("else if outside if")
                blk.branches.append((rest[3:].strip(), []))
            else:
                blk.branches.append((None, []))
            cur = blk.branches[-1][1]
        elif word == "end":
            if not stack:
                raise RenderError("end without block")
            blk, cur = stack.pop()
            if blk.kind == "define":
                name = json.loads(blk.branches[0][0] or '""')
                defines[name] = blk.branches[0][1]
                cur.remove(blk)
        else:
            cur.append(t)
    if stack:
        raise RenderError(f"unclosed {stack[-1][0].kind} block")
    return root, defines


# ---- expressions ---------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"""
    \s*(?:
      (?P<str>"(?:\\.|[^"\\])*"|`[^`]*`)
    | (?P<num>-?\d+(?:\.\d+)?)
    | (?P<decl>:=)
    | (?P<assign>=)
    | (?P<pipe>\|)
    | (?P<lp>\()
    | (?P<rp>\))
    | (?P<comma>,)
    | (?P<var>\$[A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)
    | (?P<field>(?:\.[A-Za-z0-9_]+)+|\.)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    )""", re.X)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    out = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m or m.end() == pos:
            raise RenderError(f"cannot tokenize {expr[pos:]!r} in {{{{ {expr} }}}}")
        kind = m.lastgroup
        assert kind is not None
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _Scope:
    def __init__(self, parent: _Scope | None = None) -> None:
        self.vars: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        s: _Scope | None = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        raise RenderError(f"undefined variable {name}")

    def assign(self, name: str, value: Any) -> None:
        s: _Scope | None = self
        while s is not None:
            if name in s.vars:
                s.vars[name] = value
                return
            s = s.parent
        raise RenderError(f"assignment to undeclared variable {name}")


def _truthy(v: Any) -> bool:
    if v is None or v is False:
        return False
    if isinstance(v, int | float | str | list | dict | tuple):
        return bool(v)
    return True


def _field(value: Any, path: list[str], expr: str) -> Any:
    for p in path:
        if value is None:
            raise RenderError(f"nil pointer evaluating .{p} in {expr!r}")
        if isinstance(value, dict):
            value = value.get(p)
        else:
            raise RenderError(f"cannot access .{p} on {type(value).__name__} in {expr!r}")
    return value


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e6:
        return str(int(v))  # Go %v prints 2.0 as 2
    if isinstance(v, dict | list):
        raise RenderError(f"refusing to print a {type(v).__name__} without toYaml/toJson: {v!r}")
    return str(v)


def _go_printf(fmt: str, *args: Any) -> str:
    it = iter(args)

    def repl(m: re.Match[str]) -> str:
        verb = m.group(1)
        if verb == "%":
            return "%"
        try:
            a = next(it)
        except StopIteration as exc:
            raise RenderError(f"printf: missing argument for %{verb}") from exc
        if verb == "q":
            return json.dumps(_to_text(a))
        if verb == "d":
            return str(int(a))
        return _to_text(a)

    return re.sub(r"%([sdvq%])", repl, fmt)


def _indent(n: int, s: str) -> str:
    pad = " " * int(n)
    return "\n".join(pad + line if line else line for line in str(s).split("\n"))


def _to_yaml(v: Any) -> str:
    if v is None:
        return "null"
    text: str = yaml.safe_dump(v, default_flow_style=False, sort_keys=True, width=1 << 20)
    return text.rstrip("\n").removesuffix("\n...")


def _required(msg: str, v: Any) -> Any:
    if v is None or v == "":
        raise RenderError(f"required: {msg}")
    return v


def _fail(msg: str) -> Any:
    raise RenderError(f"fail: {msg}")


def _dict(*kv: Any) -> dict[str, Any]:
    if len(kv) % 2:
        raise RenderError("dict needs an even number of arguments")
    return {str(kv[i]): kv[i + 1] for i in range(0, len(kv), 2)}


def _set(d: dict[str, Any], k: str, v: Any) -> dict[str, Any]:
    d[k] = v
    return d


def _cmp(op: Callable[[Any, Any], bool]) -> Callable[..., bool]:
    def f(a: Any, b: Any) -> bool:
        return op(a, b)

    return f


FUNCS: dict[str, Callable[..., Any]] = {
    "default": lambda d, v=None: v if _truthy(v) else d,
    "required": _required,
    "fail": _fail,
    "quote": lambda *v: " ".join(json.dumps(_to_text(x)) for x in v),
    "squote": lambda *v: " ".join("'" + _to_text(x) + "'" for x in v),
    "toYaml": _to_yaml,
    "toJson": lambda v: json.dumps(v, separators=(",", ":"), sort_keys=True),
    "nindent": lambda n, s: "\n" + _indent(n, s),
    "indent": _indent,
    "trunc": lambda n, s: str(s)[: int(n)] if int(n) >= 0 else str(s)[int(n):],
    "trimSuffix": lambda suf, s: str(s).removesuffix(suf),
    "trimPrefix": lambda pre, s: str(s).removeprefix(pre),
    "replace": lambda old, new, s: str(s).replace(old, new),
    "contains": lambda sub, s: sub in str(s),
    "hasPrefix": lambda pre, s: str(s).startswith(pre),
    "printf": _go_printf,
    "print": lambda *a: "".join(_to_text(x) for x in a),
    "lower": lambda s: str(s).lower(),
    "upper": lambda s: str(s).upper(),
    "toString": _to_text,
    "int": lambda v: int(v or 0),
    "int64": lambda v: int(v or 0),
    "add": lambda *a: sum(int(x) for x in a),
    "mul": lambda *a: __import__("math").prod(int(x) for x in a),
    "hasSuffix": lambda suf, s: str(s).endswith(suf),
    "len": lambda v: len(v),
    "eq": _cmp(lambda a, b: a == b),
    "ne": _cmp(lambda a, b: a != b),
    "lt": _cmp(lambda a, b: a < b),
    "gt": _cmp(lambda a, b: a > b),
    "le": _cmp(lambda a, b: a <= b),
    "ge": _cmp(lambda a, b: a >= b),
    "not": lambda v: not _truthy(v),
    "empty": lambda v: not _truthy(v),
    "and": lambda *a: next((x for x in a if not _truthy(x)), a[-1]),
    "or": lambda *a: next((x for x in a if _truthy(x)), a[-1]),
    "dict": _dict,
    "set": _set,
    "list": lambda *a: list(a),
    "append": lambda lst, v: [*(lst or []), v],
    "hasKey": lambda d, k: k in d,
    "ternary": lambda a, b, c: a if _truthy(c) else b,
    "sha256sum": lambda s: hashlib.sha256(str(s).encode()).hexdigest(),
    "b64enc": lambda s: base64.b64encode(str(s).encode()).decode(),
    "deepCopy": copy.deepcopy,
    "join": lambda sep, lst: str(sep).join(_to_text(x) for x in (lst or [])),
    "index": lambda coll, *keys: _index(coll, keys),
}


def _index(coll: Any, keys: tuple[Any, ...]) -> Any:
    for k in keys:
        try:
            coll = coll[k]
        except (KeyError, IndexError, TypeError) as exc:
            raise RenderError(f"index {k!r} out of range") from exc
    return coll


class _Renderer:
    def __init__(self, defines: dict[str, list[Any]], files: dict[str, list[Any]]) -> None:
        self.defines = defines
        self.files = files
        self.depth = 0

    # -- evaluation --
    def _operand(self, tokens: list[tuple[str, str]], i: int, dot: Any, scope: _Scope,
                 expr: str) -> tuple[Any, int, bool]:
        """Return (value_or_funcname, next_index, is_function)."""
        kind, tok = tokens[i]
        if kind == "str":
            return (json.loads(tok) if tok.startswith('"') else tok[1:-1]), i + 1, False
        if kind == "num":
            return (float(tok) if "." in tok else int(tok)), i + 1, False
        if kind == "var":
            name, *path = tok.split(".")
            base = scope.get(name)
            return _field(base, path, expr), i + 1, False
        if kind == "field":
            if tok == ".":
                return dot, i + 1, False
            return _field(dot, tok[1:].split("."), expr), i + 1, False
        if kind == "lp":
            depth, j = 1, i + 1
            while j < len(tokens) and depth:
                if tokens[j][0] == "lp":
                    depth += 1
                elif tokens[j][0] == "rp":
                    depth -= 1
                j += 1
            if depth:
                raise RenderError(f"unbalanced parentheses in {expr!r}")
            return self._pipeline(tokens[i + 1:j - 1], dot, scope, expr), j, False
        if kind == "ident":
            if tok in ("true", "false"):
                return tok == "true", i + 1, False
            if tok == "nil":
                return None, i + 1, False
            if tok in ("include", "template") or tok in FUNCS:
                return tok, i + 1, True
            raise RenderError(f"unknown function {tok!r} in {expr!r}")
        raise RenderError(f"unexpected token {tok!r} in {expr!r}")

    def _command(self, tokens: list[tuple[str, str]], dot: Any, scope: _Scope, expr: str,
                 piped: tuple[Any] | None) -> Any:
        if not tokens:
            raise RenderError(f"empty command in {expr!r}")
        first, i, is_fn = self._operand(tokens, 0, dot, scope, expr)
        args = []
        while i < len(tokens):
            v, i, f = self._operand(tokens, i, dot, scope, expr)
            if f:
                raise RenderError(f"function {v!r} used as an argument without parentheses in {expr!r}")
            args.append(v)
        if piped is not None:
            args.append(piped[0])
        if not is_fn:
            if args:
                raise RenderError(f"{expr!r}: cannot give arguments to a non-function")
            return first
        if first == "include":
            if len(args) != 2:
                raise RenderError(f"include takes 2 arguments in {expr!r}")
            return self.include(args[0], args[1])
        return FUNCS[first](*args)

    def _pipeline(self, tokens: list[tuple[str, str]], dot: Any, scope: _Scope, expr: str) -> Any:
        cmds: list[list[tuple[str, str]]] = [[]]
        depth = 0
        for t in tokens:
            if t[0] == "lp":
                depth += 1
            elif t[0] == "rp":
                depth -= 1
            if t[0] == "pipe" and depth == 0:
                cmds.append([])
            else:
                cmds[-1].append(t)
        val: tuple[Any] | None = None
        for c in cmds:
            val = (self._command(c, dot, scope, expr, val),)
        assert val is not None
        return val[0]

    def eval(self, expr: str, dot: Any, scope: _Scope) -> tuple[Any, list[str], str | None]:
        """Evaluate a pipeline with an optional declaration; return (value, var names, decl kind)."""
        tokens = _tokenize(expr)
        names: list[str] = []
        decl = None
        for k, kind in enumerate(tokens):
            if kind[0] in ("decl", "assign"):
                head = tokens[:k]
                if all(t[0] in ("var", "comma") for t in head) and head:
                    names = [t[1] for t in head if t[0] == "var"]
                    decl = kind[0]
                    tokens = tokens[k + 1:]
                break
        return self._pipeline(tokens, dot, scope, expr), names, decl

    # -- execution --
    def include(self, name: str, data: Any) -> str:
        self.depth += 1
        if self.depth > 50:
            raise RenderError("include recursion too deep")
        try:
            if name in self.defines:
                body = self.defines[name]
            elif name in self.files:
                body = self.files[name]
            else:
                raise RenderError(f"no template {name!r}")
            scope = _Scope()
            scope.vars["$"] = data
            return self.exec(body, data, scope)
        finally:
            self.depth -= 1

    def exec(self, nodes: list[Any], dot: Any, scope: _Scope) -> str:
        out: list[str] = []
        for n in nodes:
            if isinstance(n, Text):
                out.append(n.text)
            elif isinstance(n, Action):
                word, _, rest = n.expr.partition(" ")
                if word == "template":
                    toks = _tokenize(rest)
                    name = json.loads(toks[0][1])
                    data = self._pipeline(toks[1:], dot, scope, n.expr) if len(toks) > 1 else None
                    out.append(self.include(name, data))
                    continue
                val, names, decl = self.eval(n.expr, dot, scope)
                if decl == "decl":
                    scope.vars[names[0]] = val
                elif decl == "assign":
                    scope.assign(names[0], val)
                else:
                    out.append(_to_text(val))
            else:
                out.append(self._block(n, dot, scope))
        return "".join(out)

    def _block(self, b: Block, dot: Any, scope: _Scope) -> str:
        if b.kind == "if":
            for cond, body in b.branches:
                if cond is None or _truthy(self.eval(cond, dot, scope)[0]):
                    return self.exec(body, dot, _Scope(scope))
            return ""
        if b.kind == "with":
            val = self.eval(b.branches[0][0] or "", dot, scope)[0]
            if _truthy(val):
                return self.exec(b.branches[0][1], val, _Scope(scope))
            if len(b.branches) > 1:
                return self.exec(b.branches[1][1], dot, _Scope(scope))
            return ""
        if b.kind == "range":
            val, names, _ = self.eval(b.branches[0][0] or "", dot, scope)
            if isinstance(val, dict):
                items = [(k, val[k]) for k in sorted(val)]
            elif isinstance(val, list | tuple):
                items = list(enumerate(val))
            elif val is None:
                items = []
            else:
                raise RenderError(f"range over {type(val).__name__}")
            if not items:
                return self.exec(b.branches[1][1], dot, _Scope(scope)) if len(b.branches) > 1 else ""
            parts = []
            for k, v in items:
                s = _Scope(scope)
                if len(names) == 1:
                    s.vars[names[0]] = v
                elif len(names) == 2:
                    s.vars[names[0]], s.vars[names[1]] = k, v
                parts.append(self.exec(b.branches[0][1], v, s))
            return "".join(parts)
        raise RenderError(f"unexpected block {b.kind}")


# ---- charts --------------------------------------------------------------------------------------
def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Helm-style values merge (maps merge recursively, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def render_chart(chart_dir: Path, values: dict[str, Any] | None = None, *, release: str = "rel",
                 namespace: str = "pgw") -> dict[str, str]:
    """Render every template of ``chart_dir``; returns ``{template path: rendered text}``."""
    chart = yaml.safe_load((chart_dir / "Chart.yaml").read_text())
    vals = deep_merge(yaml.safe_load((chart_dir / "values.yaml").read_text()) or {}, values or {})
    name = chart["name"]
    files: dict[str, list[Any]] = {}
    defines: dict[str, list[Any]] = {}
    for f in sorted((chart_dir / "templates").iterdir()):
        nodes, defs = _parse(_lex(f.read_text()))
        dup = set(defs) & set(defines)
        if dup:
            raise RenderError(f"duplicate define {sorted(dup)}")
        defines.update(defs)
        files[f"{name}/templates/{f.name}"] = nodes
    r = _Renderer(defines, files)
    root = {
        "Values": vals,
        "Release": {"Name": release, "Namespace": namespace, "Service": "Helm", "IsInstall": True},
        "Chart": {"Name": name, "Version": chart["version"], "AppVersion": chart.get("appVersion", "")},
        "Template": {"BasePath": f"{name}/templates", "Name": ""},
        "Capabilities": {"KubeVersion": {"Version": "v1.30.0"}},
    }
    out: dict[str, str] = {}
    for path, nodes in files.items():
        if path.rsplit("/", 1)[-1].startswith("_"):
            continue
        scope = _Scope()
        scope.vars["$"] = root
        tmpl_root = dict(root, Template={"BasePath": f"{name}/templates", "Name": path})
        out[path] = r.exec(nodes, tmpl_root, scope)
    return out


def render_chart_objects(chart_dir: Path, values: dict[str, Any] | None = None, **kw: Any) -> list[dict[str, Any]]:
    """Render and parse all YAML manifests (NOTES.txt excluded); empty documents are dropped."""
    objs: list[dict[str, Any]] = []
    for path, text in render_chart(chart_dir, values, **kw).items():
        if not path.endswith((".yaml", ".yml")):
            continue
        try:
            docs = list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            raise RenderError(f"{path}: rendered output is not valid YAML: {exc}\n---\n{text}") from exc
        for d in docs:
            if d is None:
                continue
            if not isinstance(d, dict):
                raise RenderError(f"{path}: document is not a mapping:\n{text}")
            d.setdefault("__source__", path)
            objs.append(d)
    return objs
