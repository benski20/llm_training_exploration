# HTML templates: long markup lines and typographic dashes are intended.
# ruff: noqa: E501, RUF001
"""HTML building blocks of the developer platform (server-rendered, no template engine).

Every dynamic value goes through :func:`e` (HTML escaping). Pages run under a strict CSP
(``script-src 'self'; style-src 'self'``): no inline scripts, no ``style`` attributes; charts are SVG
with geometry in attributes only. Class names are those of the shared design system
(``pgw/design/pgw.css``), so the platform and the gateway console look like one product.
"""

from __future__ import annotations

import html
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

__all__ = ["ICONS", "Nav", "badge", "code_block", "compact", "e", "fmt_ms", "icon", "integer", "pct", "rel_time"]


def e(v: object) -> str:
    """HTML-escape any value (``None`` renders empty)."""
    return "" if v is None else html.escape(str(v), quote=True)


# ---- formatting ----------------------------------------------------------------------------------------
def integer(n: float | None) -> str:
    return "–" if n is None else f"{round(n):,}"


def compact(n: float | None) -> str:
    """``8.66k``, ``5.87M``: three significant digits (a dash for an unknown value)."""
    if n is None:
        return "–"
    n = float(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            v = n / div
            s = f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"
            return s.rstrip("0").rstrip(".") + suffix if "." in s else s + suffix
    return integer(n)


def pct(num: float, den: float, digits: int = 2) -> str:
    if den <= 0:
        return "–"
    v = 100.0 * num / den
    if v >= 100.0:
        return "100%"
    # never round a rate that is not perfect up to 100%
    q = math.floor(v * 10**digits) / 10**digits
    return f"{q:.{digits}f}%"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "–"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def utc(ts: float | None, fmt: str = "%b %d, %Y %H:%M UTC") -> str:
    if ts is None:
        return "–"
    return datetime.fromtimestamp(ts, UTC).strftime(fmt).replace(" 0", " ")


def rel_time(ts: float | None, now: float | None = None) -> str:
    if ts is None:
        return "Never"
    d = (now if now is not None else time.time()) - ts
    if d < 45:
        return "Just now"
    for limit, div, unit in ((3600, 60, "min"), (86_400, 3600, "h"), (86_400 * 45, 86_400, "d")):
        if d < limit:
            return f"{max(1, int(d // div))} {unit} ago"
    return utc(ts, "%b %d, %Y")


def time_tag(ts: float | None, now: float | None = None) -> str:
    """Relative time with the exact UTC time in a tooltip and machine-readable attribute."""
    if ts is None:
        return '<span class="muted">Never</span>'
    iso = datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")
    return f'<time datetime="{e(iso)}" title="{e(utc(ts, "%Y-%m-%d %H:%M:%S UTC"))}">{e(rel_time(ts, now))}</time>'


# ---- icons (the console's 16px stroke set) -----------------------------------------------------------
ICONS: dict[str, str] = {
    "overview": '<rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/>'
                '<rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/>',
    "key": '<circle cx="5.5" cy="10.5" r="3"/><path d="M7.7 8.3 13.5 2.5M11.5 4.5l1.5 1.5M10 6l1.2 1.2"/>',
    "cube": '<path d="M8 1.8 13.5 5v6L8 14.2 2.5 11V5z"/><path d="M2.5 5 8 8.2 13.5 5M8 8.2v6"/>',
    "list": '<path d="M5.5 4h8M5.5 8h8M5.5 12h8"/><circle cx="2.6" cy="4" r=".6" fill="currentColor"/>'
            '<circle cx="2.6" cy="8" r=".6" fill="currentColor"/><circle cx="2.6" cy="12" r=".6" fill="currentColor"/>',
    "chart": '<path d="M2.5 13.5h11"/><path d="M4.5 11V8M8 11V4.5M11.5 11V6.5"/>',
    "terminal": '<rect x="1.8" y="2.8" width="12.4" height="10.4" rx="1.6"/><path d="m4.6 6.2 2 1.8-2 1.8M8.4 10h3"/>',
    "settings": '<circle cx="8" cy="8" r="2"/><path d="M8 1.8v1.6M8 12.6v1.6M1.8 8h1.6M12.6 8h1.6M3.6 3.6l1.1 1.1'
                'M11.3 11.3l1.1 1.1M3.6 12.4l1.1-1.1M11.3 4.7l1.1-1.1"/>',
    "shield": '<path d="M8 1.8 13 3.6v4.1c0 3-2.1 5.4-5 6.5-2.9-1.1-5-3.5-5-6.5V3.6z"/><path d="m5.8 8 1.6 1.6L10.4 6.6"/>',
    "lock": '<rect x="3" y="7" width="10" height="7" rx="1.5"/><path d="M5.2 7V5a2.8 2.8 0 0 1 5.6 0v2"/>',
    "check": '<path d="m3 8.5 3.2 3L13 4.5"/>',
    "copy": '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/>'
            '<path d="M10.5 5.5V3.5a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2"/>',
    "plus": '<path d="M8 3v10M3 8h10"/>',
    "sun": '<circle cx="8" cy="8" r="2.8"/><path d="M8 1.5v1.3M8 13.2v1.3M1.5 8h1.3M13.2 8h1.3M3.4 3.4l.9.9'
           'M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9"/>',
    "moon": '<path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z"/>',
    "logout": '<path d="M6 2.5H3.5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1H6M10.5 11 13.5 8l-3-3M13.5 8H6"/>',
    "alert": '<path d="M8 2 14.5 13.5h-13z"/><path d="M8 6.5v3"/><circle cx="8" cy="11.5" r=".5" fill="currentColor"/>',
    "info": '<circle cx="8" cy="8" r="6"/><path d="M8 7.2V11"/><circle cx="8" cy="5" r=".5" fill="currentColor"/>',
    "download": '<path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M2.5 13.5h11"/>',
    "arrow-right": '<path d="M3 8h10M9 4l4 4-4 4"/>',
    "arrow-left": '<path d="M13 8H3M7 4 3 8l4 4"/>',
    "x": '<path d="m4 4 8 8M12 4l-8 8"/>',
    "book": '<path d="M2.5 3.2c2-.8 3.8-.7 5.5.6v9.4c-1.7-1.3-3.5-1.4-5.5-.6zM13.5 3.2c-2-.8-3.8-.7-5.5.6v9.4'
            'c1.7-1.3 3.5-1.4 5.5-.6z"/>',
    "bolt": '<path d="M9 1.8 3.5 9h4l-.5 5.2L12.5 7h-4z"/>',
}


def icon(name: str, size: int = 16, cls: str = "") -> str:
    c = f' class="{e(cls)}"' if cls else ""
    return (f'<svg{c} width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
            f'focusable="false">{ICONS[name]}</svg>')


LOGO = ('<svg class="logo-mark" width="22" height="22" viewBox="0 0 32 32" aria-hidden="true" focusable="false">'
        '<rect width="32" height="32" rx="8" class="logo-bg"/><path d="M10 23V9.5h6.6a4.6 4.6 0 0 1 0 9.2H13.4" '
        'fill="none" class="logo-fg" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="22.6" cy="22.4" r="2.3" class="logo-dot"/></svg>')


# ---- components --------------------------------------------------------------------------------------
def badge(text: str, tone: str = "neutral", *, dot: bool = False, title: str | None = None) -> str:
    cls = "badge" if tone == "neutral" else f"badge badge-{tone}"
    t = f' title="{e(title)}"' if title else ""
    d = f'<span class="dot dot-{tone}" aria-hidden="true"></span>' if dot and tone != "neutral" else (
        '<span class="dot" aria-hidden="true"></span>' if dot else "")
    return f'<span class="{cls}"{t}>{d}{e(text)}</span>'


def private_pill(verified: bool, private: bool = True) -> str:
    """The request log's privacy column: ``Private · verified`` (or what is not true about the request)."""
    if private and verified:
        return (f'<span class="pill pill-ok" title="Masked in your process; every result passed its integrity '
                f'check">{icon("lock", 12)}Private<span class="pill-sep" aria-hidden="true">·</span>verified</span>')
    if private:
        return (f'<span class="pill" title="Masked in your process; the request ended before a verified result">'
                f'{icon("lock", 12)}Private<span class="pill-sep" aria-hidden="true">·</span>not completed</span>')
    return f'<span class="pill pill-warn">{icon("alert", 12)}Not private</span>'


def copy_button(target_id: str, label: str = "Copy", *, text: bool = False) -> str:
    """A button that copies the text content (or value) of ``#target_id`` (platform.js)."""
    inner = f'{icon("copy", 14)}<span class="copy-label">{e(label)}</span>' if text else icon("copy", 14)
    cls = "btn btn-ghost btn-sm copy-btn" + ("" if text else " btn-icon")
    return (f'<button type="button" class="{cls}" data-copy="#{e(target_id)}" aria-label="{e(label)}" '
            f'title="{e(label)}" hidden>{inner}</button>')


def code_block(code: str, *, lang: str = "", block_id: str, title: str | None = None) -> str:
    head = (f'<div class="code-bar"><span class="code-lang">{e(title or lang)}</span>'
            f'{copy_button(block_id, "Copy", text=True)}</div>')
    return (f'<div class="code">{head}<pre class="code-pre" tabindex="0"><code id="{e(block_id)}">{e(code)}</code>'
            f'</pre></div>')


def page_head(title: str, desc: str | None = None, actions: str = "") -> str:
    d = f'<p class="page-desc">{desc}</p>' if desc else ""
    a = f'<div class="page-actions">{actions}</div>' if actions else ""
    return f'<div class="page-head"><div><h1 class="page-title">{e(title)}</h1>{d}</div>{a}</div>'


def card(body: str, *, title: str | None = None, desc: str | None = None, aside: str = "", flush: bool = False,
         cls: str = "") -> str:
    head = ""
    if title:
        dd = f'<p class="card-desc">{desc}</p>' if desc else ""
        head = f'<div class="card-head"><div><h2 class="card-title">{e(title)}</h2>{dd}</div>{aside}</div>'
    body_cls = "card-body-flush" if flush else "card-body"
    return f'<section class="card {cls}">{head}<div class="{body_cls}">{body}</div></section>'


def stat(label: str, value: str, sub: str = "", *, tone: str = "") -> str:
    v = f'<div class="stat-value num{" text-" + tone if tone else ""}">{value}</div>'
    s = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f'<div class="card stat"><div class="stat-label">{e(label)}</div>{v}{s}</div>'


def delta(cur: float, prev: float) -> str:
    """``+12.4%`` vs the previous period (neutral styling: volume is not good or bad)."""
    if prev <= 0:
        return ""
    d = 100.0 * (cur - prev) / prev
    arrow = "↑" if d >= 0 else "↓"
    return f'<span class="delta" title="Compared with the previous period">{arrow} {abs(d):.1f}%</span>'


def segmented(items: Sequence[tuple[str, str, bool]], label: str) -> str:
    """Link-based segmented control: ``(text, href, selected)``."""
    links = "".join(
        f'<a class="tab{" is-active" if sel else ""}" href="{e(href)}"{" aria-current=" + chr(34) + "page" + chr(34) if sel else ""}>'
        f"{e(text)}</a>" for text, href, sel in items)
    return f'<nav class="tabs tabs-segmented" aria-label="{e(label)}">{links}</nav>'


def empty_state(title: str, text: str, action: str = "", icon_name: str = "info") -> str:
    a = f'<div class="state-action">{action}</div>' if action else ""
    return (f'<div class="state"><div class="state-icon">{icon(icon_name, 18)}</div>'
            f'<div class="state-title">{e(title)}</div><div class="state-text">{e(text)}</div>{a}</div>')


def callout(text: str, *, tone: str = "info", title: str | None = None) -> str:
    ic = {"info": "info", "warn": "alert", "bad": "alert", "ok": "check"}.get(tone, "info")
    t = f'<div class="callout-title">{e(title)}</div>' if title else ""
    role = ' role="alert"' if tone == "bad" else ""
    return (f'<div class="callout callout-{tone}"{role}><span class="callout-icon">{icon(ic)}</span>'
            f'<div>{t}<div class="callout-text">{text}</div></div></div>')


# ---- charts ------------------------------------------------------------------------------------------
def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = float(10 ** math.floor(math.log10(v)))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


#: Series opacities of the single-accent palette (stacked bars), strongest first.
SERIES_OPACITY = (0.9, 0.62, 0.42, 0.26, 0.16)


def bar_chart(days: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], *, label: str,
              unit: str = "requests", height: int = 200) -> str:
    """Stacked daily bars, one accent at decreasing opacities per series; labels in HTML around the SVG."""
    n = len(days)
    totals = [sum(s[1][i] for s in series) for i in range(n)]
    top = _nice_max(max(totals, default=0))
    w = 1000.0
    gap = 0.5 if n <= 31 else 0.3
    bw = w / max(n, 1)
    rects = []
    for i, day in enumerate(days):
        y = float(height)
        tip = [f"{_fmt_day(day)}: {integer(totals[i])} {unit}"]
        for si, (name, vals) in enumerate(series):
            v = vals[i]
            if v <= 0:
                continue
            h = v / top * height
            y -= h
            op = SERIES_OPACITY[min(si, len(SERIES_OPACITY) - 1)]
            rects.append(f'<rect class="bar" x="{i * bw + bw * gap / 2:.2f}" y="{y:.2f}" width="{bw * (1 - gap):.2f}" '
                         f'height="{h:.2f}" fill-opacity="{op}"><title>{e(name)}: {integer(v)}</title></rect>')
            if len(series) > 1:
                tip.append(f"{name}: {integer(v)}")
        rects.append(f'<rect class="bar-hit" x="{i * bw:.2f}" y="0" width="{bw:.2f}" height="{height}">'
                     f'<title>{e(chr(10).join(tip))}</title></rect>')
    grid = "".join(f'<line class="chart-grid" x1="0" x2="{w:.0f}" y1="{height * f:.2f}" y2="{height * f:.2f}" '
                   f'vector-effect="non-scaling-stroke"/>' for f in (0.0, 0.5))
    base = (f'<line class="chart-base" x1="0" x2="{w:.0f}" y1="{height}" y2="{height}" '
            f'vector-effect="non-scaling-stroke"/>')
    svg = (f'<svg class="bars-svg" viewBox="0 0 {w:.0f} {height}" preserveAspectRatio="none" role="img" '
           f'aria-label="{e(label)}">{grid}{"".join(rects)}{base}</svg>')
    yax = "".join(f'<span>{e(compact(top * f))}</span>' for f in (1.0, 0.5, 0.0))
    picks = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    xax = "".join(f"<span>{e(_fmt_day(days[i]))}</span>" for i in picks)
    legend = ""
    if len(series) > 1:
        items = "".join(
            f'<span class="chart-legend-item"><span class="legend-swatch swatch-{min(si, 4)}" aria-hidden="true">'
            f'</span>{e(name)}</span>' for si, (name, _v) in enumerate(series))
        legend = f'<div class="chart-legend">{items}</div>'
    return (f'<figure class="bars">{legend}<div class="bars-plot"><div class="bars-y" aria-hidden="true">{yax}</div>'
            f'<div class="bars-area">{svg}<div class="bars-x" aria-hidden="true">{xax}</div></div></div></figure>')


def share_bar(fraction: float) -> str:
    f = max(0.0, min(1.0, fraction))
    return (f'<svg class="share" viewBox="0 0 100 4" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect class="share-track" width="100" height="4" rx="2"/>'
            f'<rect class="share-fill" width="{f * 100:.2f}" height="4" rx="2"/></svg>')


def _fmt_day(day: str) -> str:
    try:
        return datetime.strptime(day, "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")
    except ValueError:
        return day


# ---- shell -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Nav:
    key: str
    label: str
    path: str
    icon: str


NAV_GROUPS: list[tuple[str, list[Nav]]] = [
    ("Build", [Nav("overview", "Overview", "/", "overview"), Nav("quickstart", "Quickstart", "/quickstart", "terminal"),
               Nav("keys", "API keys", "/keys", "key"), Nav("models", "Models", "/models", "cube")]),
    ("Observe", [Nav("usage", "Usage", "/usage", "chart"), Nav("requests", "Requests", "/requests", "list")]),
    ("Organisation", [Nav("settings", "Settings", "/settings", "settings")]),
]


def document(*, title: str, body: str, prefix: str, body_class: str = "") -> str:
    a = f"{prefix}/assets"
    cls = f' class="{e(body_class)}"' if body_class else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="robots" content="noindex"><meta name="color-scheme" content="light dark">'
        f'<title>{e(title)} · PGW</title>'
        f'<link rel="icon" type="image/svg+xml" href="{a}/favicon.svg">'
        f'<script src="{a}/theme.js"></script>'
        f'<link rel="stylesheet" href="{a}/pgw.css"><link rel="stylesheet" href="{a}/platform.css">'
        f'<script src="{a}/platform.js" defer></script>'
        f'</head><body{cls}>{body}</body></html>'
    )


def shell(*, prefix: str, active: str, title: str, content: str, org_name: str, plan: str | None,
          user: str, role: str, csrf: str) -> str:
    groups = []
    for group, items in NAV_GROUPS:
        lis = []
        for it in items:
            href = prefix + it.path if it.path != "/" else prefix + "/"
            cur = it.key == active
            lis.append(f'<li><a class="nav-link{" is-active" if cur else ""}" href="{e(href)}"'
                       f'{" aria-current=" + chr(34) + "page" + chr(34) if cur else ""} title="{e(it.label)}">'
                       f'<span class="nav-icon">{icon(it.icon)}</span><span class="nav-label">{e(it.label)}</span>'
                       f'</a></li>')
        groups.append(f'<div class="nav-group"><div class="nav-group-label">{e(group)}</div><ul>{"".join(lis)}</ul></div>')
    initials = "".join(w[0] for w in org_name.split()[:2]).upper() or "O"
    plan_html = f'<span class="org-plan">{e(plan)}</span>' if plan else ""
    role_badge = badge("Admin", "accent") if role == "admin" else badge("Viewer", title="Read-only access")
    sidebar = (
        '<aside class="sidebar" aria-label="Primary">'
        f'<a class="brand" href="{e(prefix)}/">{LOGO}<span class="brand-name">PGW <span class="brand-sub">Platform'
        '</span></span></a>'
        f'<div class="org" title="{e(org_name)}"><span class="org-avatar" aria-hidden="true">{e(initials)}</span>'
        f'<span class="org-text"><span class="org-name">{e(org_name)}</span>{plan_html}</span></div>'
        f'<nav class="nav" aria-label="Platform">{"".join(groups)}</nav>'
        '<div class="sidebar-foot"><div class="privacy-note">'
        f'{icon("shield", 14)}<span>Prompts and completions are masked in your process and never stored.</span>'
        '</div></div></aside>'
    )
    topbar = (
        '<header class="topbar">'
        f'<div class="topbar-title"><span class="crumb">{e(org_name)}</span><span class="crumb-sep" aria-hidden="true">'
        f'/</span><span class="crumb-cur">{e(title)}</span></div>'
        '<div class="topbar-right">'
        f'<span class="topbar-user hide-sm">{e(user)}</span>{role_badge}'
        '<span class="topbar-sep" aria-hidden="true"></span>'
        '<button type="button" class="btn btn-ghost btn-sm btn-icon theme-toggle" aria-label="Toggle dark theme" '
        f'title="Toggle theme" hidden><span class="theme-icon-dark">{icon("moon")}</span>'
        f'<span class="theme-icon-light">{icon("sun")}</span></button>'
        f'<form method="post" action="{e(prefix)}/logout" class="inline-form">'
        f'<input type="hidden" name="csrf" value="{e(csrf)}">'
        f'<button type="submit" class="btn btn-ghost btn-sm btn-icon" aria-label="Sign out" title="Sign out">'
        f'{icon("logout")}</button></form>'
        '</div></header>'
    )
    return (f'<div class="shell"><a class="skip-link" href="#main">Skip to content</a>{sidebar}'
            f'<div class="main-col">{topbar}<main id="main" class="content" tabindex="-1">'
            f'<div class="page">{content}</div></main></div></div>')


def table(head: Iterable[tuple[str, str]], rows: Iterable[str], *, caption: str, cls: str = "") -> str:
    """``head``: ``(label, extra class)``; ``rows``: rendered ``<tr>`` strings."""
    th = "".join(f'<th scope="col"{" class=" + chr(34) + c + chr(34) if c else ""}>{e(lbl)}</th>' for lbl, c in head)
    return (f'<div class="table-wrap"><table class="table {e(cls)}"><caption class="sr-only">{e(caption)}</caption>'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
