# HTML templates: long markup lines and typographic dashes are intended.
# ruff: noqa: E501, RUF001
"""Page bodies of the developer platform (pure functions of already-fetched, content-free data)."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlencode

from pgw.platform.client import AuditEvent, CreatedKey, ModelInfo, Org, Principal, RequestPage, SdkKey, UsageRow
from pgw.platform.ui import (
    badge,
    bar_chart,
    callout,
    card,
    code_block,
    compact,
    copy_button,
    delta,
    e,
    empty_state,
    fmt_ms,
    icon,
    integer,
    page_head,
    pct,
    private_pill,
    segmented,
    share_bar,
    stat,
    table,
    time_tag,
    utc,
)

__all__ = ["RANGES", "Ctx", "day_span"]

#: Usage ranges (days).
RANGES = {"7d": 7, "30d": 30, "90d": 90}
_MODEL_STATUS = {"available": ("Available", "ok"), "degraded": ("Degraded", "warn"),
                 "unavailable": ("Unavailable", "bad"), "preview": ("Preview", "accent")}


@dataclass(frozen=True)
class Ctx:
    prefix: str
    principal: Principal
    org: Org
    csrf: str
    now: float

    @property
    def admin(self) -> bool:
        return self.principal.role == "admin"

    def url(self, path: str, **q: str | None) -> str:
        qs = urlencode({k: v for k, v in q.items() if v})
        return f"{self.prefix}{path}" + (f"?{qs}" if qs else "")


def day_span(now: float, days: int, *, offset: int = 0) -> list[str]:
    """The last ``days`` UTC days ending today (shifted back ``offset`` days), oldest first."""
    today = datetime.fromtimestamp(now, UTC).date() - timedelta(days=offset)
    return [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


@dataclass
class _Totals:
    """Sums over usage rows; ``None`` where no row carries the metric (the provider does not meter it)."""

    requests: int | None = None
    prompt: int | None = None
    completion: int | None = None
    verified: int | None = None
    tokens: int = 0
    errors: int = 0

    def add(self, r: UsageRow) -> None:
        def plus(a: int | None, b: int | None) -> int | None:
            return a if b is None else (a or 0) + b

        self.requests = plus(self.requests, r.requests)
        self.prompt = plus(self.prompt, r.prompt_tokens)
        self.completion = plus(self.completion, r.completion_tokens)
        self.verified = plus(self.verified, r.verified)
        self.tokens += r.tokens
        self.errors += r.errors

    @property
    def verified_rate(self) -> str:
        return pct(self.verified, self.requests) if self.verified is not None and self.requests else "–"


def _sum(rows: Sequence[UsageRow]) -> _Totals:
    t = _Totals()
    for r in rows:
        t.add(r)
    return t


def _n(v: int | None, fmt: Callable[[float], str] = integer) -> str:
    return "–" if v is None else fmt(v)


def _key_label(keys: dict[str, SdkKey], key_id: str) -> str:
    k = keys.get(key_id)
    return k.name if k else key_id


def _model_chips(k: SdkKey) -> str:
    if not k.models:
        return '<span class="muted">All models</span>'
    return '<span class="badges">' + "".join(f'<code class="chip">{e(m)}</code>' for m in k.models) + "</span>"


def _hidden_csrf(ctx: Ctx) -> str:
    return f'<input type="hidden" name="csrf" value="{e(ctx.csrf)}">'


# =====================================================================================================
# Overview
# =====================================================================================================
def overview(ctx: Ctx, *, rng: str, usage: Sequence[UsageRow], prev: Sequence[UsageRow], keys: Sequence[SdkKey],
             recent: RequestPage | None, models: Sequence[ModelInfo]) -> str:
    days = day_span(ctx.now, RANGES[rng])
    cur, before = _sum(usage), _sum(prev)
    metered_requests = cur.requests is not None
    active = [k for k in keys if k.active]
    by_key = {k.id: k for k in keys}
    head = page_head(
        "Overview", f"Private inference for {e(ctx.org.name)}, last {RANGES[rng]} days.",
        segmented([(r, ctx.url("/", range=r), r == rng) for r in RANGES], "Time range"))

    verified_rate = cur.verified_rate
    verified_head = (f"{verified_rate} checked client-side" if cur.verified is not None
                     else "Checked in your process")
    posture = (
        '<section class="posture-strip" aria-label="Privacy posture">'
        f'<div class="posture-item">{icon("lock", 14)}<span><strong>Masked in your process</strong>'
        '<span class="muted"> Our GPUs only see masked data</span></span></div>'
        f'<div class="posture-item">{icon("shield", 14)}<span><strong>{e(verified_head)}</strong>'
        '<span class="muted"> Every result is integrity-checked by your client</span></span></div>'
        f'<div class="posture-item">{icon("check", 14)}<span><strong>No content stored</strong>'
        '<span class="muted"> Only metadata is logged</span></span></div>'
        '</section>'
    )
    stats = (
        '<div class="stats-grid stats-4">'
        + (stat("Requests", compact(cur.requests), f'{integer(cur.requests)} {delta(cur.requests, before.requests or 0)}')
           if cur.requests is not None else stat("Requests", "–", "Counted by the client library"))
        + stat("Tokens", compact(cur.tokens),
               f'{compact(cur.prompt)} in · {compact(cur.completion)} out' if cur.prompt is not None
               else f'{integer(cur.tokens)} {delta(cur.tokens, before.tokens)}')
        + (stat("Checked results", e(verified_rate),
                f'{integer(cur.verified)} of {integer(cur.requests or 0)} requests, reported by your client',
                tone="ok" if cur.requests else "")
           if cur.verified is not None else stat("Checked results", "–", "Checked in your process, per result"))
        + stat("Active keys", str(len(active)), f"{len(keys) - len(active)} revoked" if len(keys) != len(active)
               else "none revoked")
        + "</div>"
    )
    per_day: dict[str, int] = defaultdict(int)
    for r in usage:
        per_day[r.day] += r.requests if metered_requests and r.requests is not None else r.tokens
    what = "Requests" if metered_requests else "Tokens"
    chart = card(bar_chart(days, [(what, [per_day.get(d, 0) for d in days])], label=f"{what} per day",
                           unit=what.lower()),
                 title=what, desc="Per UTC day", aside=f'<a class="card-link" href="{e(ctx.url("/usage", range=rng))}">'
                 f'Usage {icon("arrow-right", 12)}</a>')

    if recent is None:
        recent_html = empty_state("Request log not available", "This provider does not keep a per-request log; "
                                  "usage is metered per key, model and day.")
    elif recent.records:
        rows = "".join(
            f'<tr><td>{time_tag(r.ts, ctx.now)}</td><td class="hide-sm">{e(_key_label(by_key, r.key_id))}</td>'
            f'<td class="hide-sm"><code class="cell-model">{e(r.model)}</code></td><td>{private_pill(r.verified, r.private)}</td>'
            f'<td class="al-right num">{e(fmt_ms(r.latency_ms))}</td></tr>' for r in recent.records)
        recent_html = table([("Time", ""), ("Key", "hide-sm"), ("Model", "hide-sm"), ("Privacy", ""), ("Latency", "al-right")],
                            [rows], caption="Most recent requests")
    else:
        recent_html = empty_state("No requests yet", "Requests made with your keys appear here within seconds.")
    recent_card = card(recent_html, title="Recent requests", flush=True,
                       aside=f'<a class="card-link" href="{e(ctx.url("/requests"))}">View all {icon("arrow-right", 12)}</a>')

    key_req: dict[str, int] = defaultdict(int)
    for r in usage:
        key_req[r.key_id] += r.requests if metered_requests and r.requests is not None else r.tokens
    top = sorted(key_req.items(), key=lambda kv: -kv[1])[:5]
    if top:
        total = sum(key_req.values()) or 1
        items = "".join(
            f'<li class="rank-row"><div class="rank-head"><span class="rank-name">{e(_key_label(by_key, kid))}</span>'
            f'<span class="num muted">{compact(n)}</span></div>{share_bar(n / total)}</li>' for kid, n in top)
        top_html = f'<ul class="rank">{items}</ul>'
    else:
        top_html = empty_state("No usage yet", "Create a key and make your first request.", icon_name="key")
    top_card = card(top_html, title="Top keys", desc=f"By {what.lower()}",
                    aside=f'<a class="card-link" href="{e(ctx.url("/keys"))}">Keys {icon("arrow-right", 12)}</a>')

    notice = ""
    down = [m for m in models if m.status in ("degraded", "unavailable")]
    if down:
        notice = callout(", ".join(f"<strong>{e(m.name)}</strong> is {e(m.status)}" for m in down) + ".", tone="warn")
    start = ""
    if not keys:
        start = callout(f'Create an API key and follow the <a href="{e(ctx.url("/quickstart"))}">quickstart</a> to '
                        "make your first private request.", tone="info", title="Get started")
    return (head + notice + start + posture + stats + chart
            + f'<div class="grid-2 grid-wide-left">{recent_card}{top_card}</div>')


# =====================================================================================================
# API keys
# =====================================================================================================
def keys_page(ctx: Ctx, *, keys: Sequence[SdkKey], created: CreatedKey | None = None,
              flash: str | None = None) -> str:
    action = (f'<a class="btn btn-primary" href="{e(ctx.url("/keys/new"))}">{icon("plus", 14)}Create key</a>'
              if ctx.admin else "")
    head = page_head("API keys", "Keys authenticate the PGW client library. The full key is shown once, when it is "
                     "created; PGW stores only a hash.", action)
    reveal = ""
    if created is not None:
        snippet = f'export PGW_API_KEY="{created.secret}"'
        reveal = (
            '<section class="card reveal" aria-labelledby="reveal-title">'
            f'<div class="reveal-head">{icon("check", 16, "text-ok")}<div><h2 class="card-title" id="reveal-title">'
            f'Key “{e(created.key.name)}” created</h2><p class="card-desc">Copy it now. For your security it will '
            'not be shown again.</p></div></div>'
            '<div class="secret"><code class="secret-value mono" id="new-key">'
            f'{e(created.secret)}</code>{copy_button("new-key", "Copy key", text=True)}</div>'
            + code_block(snippet, lang="shell", block_id="new-key-env", title="Use it")
            + f'<div class="reveal-foot"><a class="btn btn-secondary btn-sm" href="{e(ctx.url("/keys"))}">Done</a>'
            f'<a class="btn btn-ghost btn-sm" href="{e(ctx.url("/quickstart"))}">Quickstart {icon("arrow-right", 12)}</a>'
            '</div></section>'
        )
    msg = callout(e(flash), tone="ok") if flash else ""
    if not keys:
        body = card(empty_state("No API keys yet", "Create a key to start making private requests.",
                                action, icon_name="key"))
        return head + reveal + msg + body
    rows = []
    for k in keys:
        status = badge("Active", "ok", dot=True) if k.active else badge("Revoked", title=f"Revoked {utc(k.revoked_at)}")
        env = badge("Live", "accent") if k.env == "live" else badge(k.env.capitalize())
        revoke = ""
        if ctx.admin and k.active:
            revoke = (f'<a class="btn btn-ghost btn-sm btn-danger-ghost" href="{e(ctx.url(f"/keys/{k.id}/revoke"))}" '
                      f'aria-label="Revoke {e(k.name)}">Revoke</a>')
        by = f'<span class="cell-sub">{e(k.created_by)}</span>' if k.created_by else ""
        rows.append(
            f'<tr class="{"" if k.active else "is-revoked"}"><td><div class="cell-stack"><span class="cell-strong">'
            f'{e(k.name)}</span>{by}</div></td><td class="hide-sm"><code class="key-prefix">{e(k.prefix)}</code></td>'
            f'<td class="hide-sm">{env}</td><td class="wrap hide-sm">{_model_chips(k)}</td>'
            f'<td class="hide-sm">{time_tag(k.last_used_at, ctx.now)}</td>'
            f'<td class="hide-md">{time_tag(k.created_at, ctx.now)}</td><td>{status}</td>'
            f'<td class="al-right">{revoke}</td></tr>')
    tbl = table([("Name", ""), ("Key", "hide-sm"), ("Env", "hide-sm"), ("Models", "hide-sm"),
                 ("Last used", "hide-sm"), ("Created", "hide-md"),
                 ("Status", ""), ("", "al-right")], rows, caption="API keys")
    return head + reveal + msg + card(tbl, flush=True)


def key_new_page(ctx: Ctx, *, models: Sequence[ModelInfo], error: str | None = None, name: str = "",
                 env: str = "live", scoped: Sequence[str] = ()) -> str:
    head = page_head("Create API key", "Give the key a name you will recognise in the usage and request logs.")
    err = callout(e(error), tone="bad") if error else ""
    model_opts = "".join(
        f'<label class="check"><input type="checkbox" name="models" value="{e(m.id)}"'
        f'{" checked" if m.id in scoped else ""}><span class="check-box" aria-hidden="true">{icon("check", 12)}</span>'
        f'<span class="check-text"><span class="check-label mono">{e(m.id)}</span>'
        f'<span class="check-hint">{e(m.params or "")} · {e(_MODEL_STATUS.get(m.status, (m.status, ""))[0])}</span>'
        f'</span></label>' for m in models)
    env_opts = "".join(
        f'<label class="radio-card"><input type="radio" name="env" value="{v}"{" checked" if env == v else ""}>'
        f'<span class="radio-card-body"><span class="radio-card-title">{t}</span><span class="radio-card-text">{d}'
        f'</span></span></label>'
        for v, t, d in (("live", "Live", "Production traffic."), ("test", "Test", "Development and CI; same privacy.")))
    form = (
        f'<form class="form" method="post" action="{e(ctx.url("/keys"))}">{_hidden_csrf(ctx)}'
        '<div class="field"><label class="field-label" for="key-name">Name</label>'
        f'<input class="input" id="key-name" name="name" required maxlength="64" autocomplete="off" '
        f'placeholder="e.g. Production backend" value="{e(name)}" autofocus></div>'
        f'<fieldset class="fieldset"><legend class="field-label">Environment</legend><div class="radio-cards">{env_opts}'
        '</div></fieldset>'
        '<fieldset class="fieldset"><legend class="field-label">Models</legend>'
        '<p class="field-hint">Leave all unchecked to allow every model, including ones added later.</p>'
        f'<div class="check-list">{model_opts}</div></fieldset>'
        f'<div class="form-actions"><a class="btn btn-ghost" href="{e(ctx.url("/keys"))}">Cancel</a>'
        '<button class="btn btn-primary" type="submit">Create key</button></div></form>'
    )
    return head + err + f'<div class="narrow">{card(form)}</div>'


def key_revoke_page(ctx: Ctx, key: SdkKey) -> str:
    head = page_head("Revoke key")
    body = (
        f'<p class="confirm-text">Revoke <strong>{e(key.name)}</strong> (<code>{e(key.prefix)}</code>)? Applications '
        'using it stop working within a minute. This cannot be undone.</p>'
        f'<form class="form-actions" method="post" action="{e(ctx.url(f"/keys/{key.id}/revoke"))}">{_hidden_csrf(ctx)}'
        f'<a class="btn btn-ghost" href="{e(ctx.url("/keys"))}">Cancel</a>'
        '<button class="btn btn-danger" type="submit">Revoke key</button></form>'
    )
    return head + f'<div class="narrow">{card(body)}</div>'


# =====================================================================================================
# Usage
# =====================================================================================================
def usage_page(ctx: Ctx, *, rng: str, by: str, usage: Sequence[UsageRow], keys: Sequence[SdkKey],
               models: Sequence[ModelInfo]) -> str:
    days = day_span(ctx.now, RANGES[rng])
    t = _sum(usage)
    metered_requests = t.requests is not None
    split = t.prompt is not None
    by_key = {k.id: k for k in keys}
    controls = (segmented([(r, ctx.url("/usage", range=r, by=by), r == rng) for r in RANGES], "Time range")
                + f'<a class="btn btn-secondary btn-sm" href="{e(ctx.url("/usage.csv", range=rng))}" download>'
                f'{icon("download", 14)}Export CSV</a>')
    head = page_head("Usage", "Usage per UTC day. Counts are final about a minute after a request ends.", controls)
    if split:
        tok_stats = (stat("Input tokens", compact(t.prompt), integer(t.prompt))
                     + stat("Output tokens", compact(t.completion), _n(t.completion)))
    else:
        trips = sum(r.trips or 0 for r in usage)
        tok_stats = (stat("Tokens", compact(t.tokens), "Input and output positions")
                     + stat("Compute round trips", compact(trips), integer(trips)))
    stats = (
        '<div class="stats-grid stats-4">'
        + stat("Requests", compact(t.requests) if metered_requests else "–",
               integer(t.requests) if t.requests is not None else "Counted by the client library")
        + tok_stats
        + (stat("Checked results", e(t.verified_rate), f'{integer(t.errors)} failed requests (client-reported)',
                tone="ok" if t.requests else "")
           if t.verified is not None else stat("Checked results", "–", "Checked in your process, per result"))
        + "</div>"
    )
    what = "Requests" if metered_requests else "Tokens"
    model_ids = [m.id for m in models]
    seen = {r.model for r in usage}
    ordered = [m for m in model_ids if m in seen] + sorted(seen - set(model_ids))
    grid: dict[tuple[str, str], int] = defaultdict(int)
    for r in usage:
        grid[(r.model, r.day)] += (r.requests or 0) if metered_requests else r.tokens
    series = [(m, [grid.get((m, d), 0) for d in days]) for m in ordered] or [(what, [0] * len(days))]
    chart = card(bar_chart(days, series, label=f"{what} per day by model", unit=what.lower()),
                 title=f"{what} by model")

    tabs = segmented([(lbl, ctx.url("/usage", range=rng, by=v), v == by)
                      for v, lbl in (("day", "By day"), ("key", "By key"), ("model", "By model"))], "Group usage")
    agg: dict[str, _Totals] = defaultdict(_Totals)
    for r in usage:
        agg[{"day": r.day, "key": r.key_id, "model": r.model}[by]].add(r)

    def measure(a: _Totals) -> int:
        return (a.requests or 0) if metered_requests else a.tokens

    if by == "day":
        order = sorted(agg, reverse=True)
        first = ("Date", "")
    else:
        order = sorted(agg, key=lambda g: -measure(agg[g]))
        first = ("Key", "") if by == "key" else ("Model", "")
    total = measure(t) or 1
    rows = []
    for g in order:
        a = agg[g]
        if by == "day":
            label = e(datetime.strptime(g, "%Y-%m-%d").strftime("%a, %b %d").replace(" 0", " "))
        elif by == "key":
            k = by_key.get(g)
            label = (f'<div class="cell-stack"><span class="cell-strong">{e(_key_label(by_key, g))}</span>'
                     f'<code class="cell-sub">{e(k.prefix if k else g)}</code></div>')
        else:
            label = f'<code class="cell-model">{e(g)}</code>'
        share = "" if by == "day" else (
            f'<td class="share-cell hide-sm"><div class="share-wrap">{share_bar(measure(a) / total)}'
            f'<span class="num muted">{pct(measure(a), total, 1)}</span></div></td>')
        cells = [f'<td class="al-right num">{_n(a.requests)}</td>']
        if split:
            cells += [f'<td class="al-right num">{_n(a.prompt)}</td>', f'<td class="al-right num">{_n(a.completion)}</td>']
        else:
            cells += [f'<td class="al-right num">{integer(a.tokens)}</td>']
        cells += [f'<td class="al-right num hide-sm">{e(a.verified_rate)}</td>']
        rows.append(f'<tr><td>{label}</td>{"".join(cells)}{share}</tr>')
    cols = [first, ("Requests", "al-right")]
    cols += [("Input tokens", "al-right"), ("Output tokens", "al-right")] if split else [("Tokens", "al-right")]
    cols += [("Checked", "al-right hide-sm")] + ([] if by == "day" else [("Share", "hide-sm")])
    if rows:
        tbl = table(cols, rows, caption="Usage breakdown")
    else:
        tbl = empty_state("No usage in this period", "Requests made with your keys are counted here.")
    breakdown = (f'<section class="card"><div class="card-head card-head-bar">{tabs}</div>'
                 f'<div class="card-body-flush">{tbl}</div></section>')
    return head + stats + chart + breakdown


def usage_csv(usage: Sequence[UsageRow], keys: Sequence[SdkKey]) -> str:
    names = {k.id: k.name for k in keys}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["day", "key_id", "key_name", "model", "requests", "input_tokens", "output_tokens", "tokens",
                "verified", "errors", "trips"])
    for r in sorted(usage, key=lambda r: (r.day, r.key_id, r.model)):
        name = names.get(r.key_id, "")
        if name[:1] in ("=", "+", "-", "@"):  # spreadsheet formula injection
            name = "'" + name
        w.writerow([r.day, r.key_id, name, r.model, *("" if v is None else v for v in (
            r.requests, r.prompt_tokens, r.completion_tokens)), r.tokens, "" if r.verified is None else r.verified,
            r.errors, "" if r.trips is None else r.trips])
    return buf.getvalue()


# =====================================================================================================
# Request log
# =====================================================================================================
def requests_page(ctx: Ctx, *, page: RequestPage | None, keys: Sequence[SdkKey], models: Sequence[ModelInfo],
                  key_id: str | None, model: str | None, status: str | None, before: str | None) -> str:
    by_key = {k.id: k for k in keys}
    head = page_head("Requests", "Metadata of every request made with your keys. Prompts and completions never reach "
                     "PGW in readable form, so there is nothing to show or store.")

    def opts(values: Sequence[tuple[str, str]], cur: str | None, all_label: str) -> str:
        o = [f'<option value="">{e(all_label)}</option>']
        o += [f'<option value="{e(v)}"{" selected" if v == cur else ""}>{e(t)}</option>' for v, t in values]
        return "".join(o)

    def sel(name: str, label: str, options: str) -> str:
        return (f'<label class="filter"><span class="sr-only">{e(label)}</span><span class="select-wrap">'
                f'<select class="input select" name="{name}" data-autosubmit>{options}</select>'
                f'<span class="select-chevron" aria-hidden="true"><svg width="12" height="12" viewBox="0 0 16 16" '
                f'fill="none" stroke="currentColor" stroke-width="1.5"><path d="m4 6 4 4 4-4"/></svg></span></span></label>')

    filters = (
        f'<form class="filters" method="get" action="{e(ctx.url("/requests"))}" role="search">'
        + sel("key", "Key", opts([(k.id, k.name) for k in keys], key_id, "All keys"))
        + sel("model", "Model", opts([(m.id, m.id) for m in models], model, "All models"))
        + sel("status", "Status", opts([("ok", "Succeeded"), ("error", "Failed")], status, "All statuses"))
        + '<button class="btn btn-secondary btn-sm filter-apply" type="submit">Apply</button>'
        + (f'<a class="btn btn-ghost btn-sm" href="{e(ctx.url("/requests"))}">Clear</a>' if (key_id or model or status)
           else "")
        + "</form>"
    )
    if page is None:
        return head + card(empty_state(
            "Request log not available", "This provider meters usage per key, model and day but does not keep a "
            "per-request log. Usage has the daily figures.", f'<a class="btn btn-secondary btn-sm" '
            f'href="{e(ctx.url("/usage"))}">Open usage</a>', icon_name="list"))
    if page.records:
        rows = []
        for r in page.records:
            st = ("" if r.status == "ok"
                  else " " + badge(r.error_code.replace("_", " ") if r.error_code else "failed", "bad"))
            toks = (f'{integer(r.prompt_tokens)}<span class="muted"> → </span>{integer(r.completion_tokens)}')
            rows.append(
                f'<tr><td>{time_tag(r.ts, ctx.now)}</td><td class="hide-md"><code class="req-id">{e(r.id)}</code></td>'
                f'<td class="hide-sm">{e(_key_label(by_key, r.key_id))}</td><td class="hide-sm"><code class="cell-model">{e(r.model)}'
                f'</code></td><td class="wrap"><div class="privacy-cell">{private_pill(r.verified, r.private)}{st}</div></td>'
                f'<td class="al-right num hide-sm">{toks}</td>'
                f'<td class="al-right num hide-sm">{e(fmt_ms(r.ttft_ms))}</td><td class="al-right num">'
                f'{e(fmt_ms(r.latency_ms))}</td></tr>')
        tbl = table([("Time", ""), ("Request", "hide-md"), ("Key", "hide-sm"), ("Model", "hide-sm"), ("Privacy", ""),
                     ("Tokens in → out", "al-right hide-sm"), ("TTFT", "al-right hide-sm"), ("Latency", "al-right")], rows,
                    caption="Request log")
    else:
        tbl = empty_state("No matching requests", "Try a different filter, or make a request with one of your keys.")
    pager = ""
    if before or page.next_before:
        newest = (f'<a class="btn btn-secondary btn-sm" href="{e(ctx.url("/requests", key=key_id, model=model, status=status))}">'
                  f'{icon("arrow-left", 14)}Newest</a>' if before else "")
        older = (f'<a class="btn btn-secondary btn-sm" href="{e(ctx.url("/requests", key=key_id, model=model, status=status, before=page.next_before))}">'
                 f'Older{icon("arrow-right", 14)}</a>' if page.next_before else "")
        pager = f'<div class="pager"><span class="muted small">{len(page.records)} requests</span><div class="pager-btns">{newest}{older}</div></div>'
    return head + f'<section class="card">{filters}<div class="card-body-flush">{tbl}</div>{pager}</section>'


# =====================================================================================================
# Models
# =====================================================================================================
def models_page(ctx: Ctx, *, models: Sequence[ModelInfo]) -> str:
    head = page_head("Models", "Open-weight models served privately: the weights are public, your data stays masked. "
                     "Outputs are identical to the reference model.")
    if not models:
        return head + card(empty_state("No models available", "Your organisation has no models enabled yet."))
    cards = []
    for i, m in enumerate(models):
        label, tone = _MODEL_STATUS.get(m.status, (m.status.capitalize(), "neutral"))
        facts = "".join(
            f'<div class="fact"><dt>{e(k)}</dt><dd class="num">{e(v)}</dd></div>'
            for k, v in (("Parameters", m.params or "–"),
                         ("Context", f"{m.context_length // 1024}k tokens" if m.context_length else "–"),
                         ("Time to first token", fmt_ms(m.ttft_ms) if m.ttft_ms else "–"),
                         ("Throughput", f"{m.tokens_per_s:.0f} tok/s" if m.tokens_per_s else "–")))
        cards.append(
            '<section class="card model-card"><div class="card-body">'
            f'<div class="model-card-head"><h2 class="card-title">{e(m.name)}</h2>{badge(label, tone, dot=True)}</div>'
            f'<div class="model-id"><code id="model-{i}">{e(m.id)}</code>{copy_button(f"model-{i}", "Copy model ID")}</div>'
            + (f'<p class="model-desc">{e(m.description)}</p>' if m.description else "")
            + f'<dl class="facts">{facts}</dl></div></section>')
    return head + f'<div class="model-grid">{"".join(cards)}</div>'


# =====================================================================================================
# Quickstart
# =====================================================================================================
def quickstart_page(ctx: Ctx, *, models: Sequence[ModelInfo], has_keys: bool) -> str:
    model = models[0].id if models else "qwen2.5-7b-instruct"
    head = page_head("Quickstart", "Three lines from zero to private inference. The PGW client is a drop-in for the "
                     "OpenAI SDK; the private gateway runs inside your process.")
    key_hint = (f'<a href="{e(ctx.url("/keys/new"))}">Create a key</a>' if ctx.admin
                else "Ask an admin of your organisation for a key")
    steps = [
        ("Install the client", "Python 3.12 or newer. Runs on CPU: no GPU and no model setup on your side.",
         code_block("pip install pgw", lang="shell", block_id="qs-install", title="Terminal")),
        ("Import it", "Replace the OpenAI import; everything else in your code stays the same.",
         code_block("from pgw import OpenAI", lang="python", block_id="qs-import", title="Python")),
        ("Add your key", f"{key_hint}, then pass it in or set <code>PGW_API_KEY</code>.",
         code_block('client = OpenAI(api_key="pgw_live_…")', lang="python", block_id="qs-key", title="Python")),
    ]
    step_html = "".join(
        f'<li class="qs-step"><span class="step-n" aria-hidden="true">{i}</span><div class="qs-step-body">'
        f'<h2 class="qs-step-title">{e(t)}</h2><p class="qs-step-text">{d}</p>{c}</div></li>'
        for i, (t, d, c) in enumerate(steps, 1))
    full = (
        "from pgw import OpenAI\n\n"
        "client = OpenAI()  # reads PGW_API_KEY\n\n"
        "stream = client.chat.completions.create(\n"
        f'    model="{model}",\n'
        '    messages=[{"role": "user", "content": "Summarise this contract in three bullet points."}],\n'
        "    stream=True,\n"
        ")\n"
        "for chunk in stream:\n"
        '    print(chunk.choices[0].delta.content or "", end="")\n'
    )
    how = (
        '<ul class="how">'
        f'<li>{icon("lock", 14)}<span><strong>Masked before it leaves.</strong> Prompts, activations and outputs are '
        'masked in your process with one-time masks; our GPUs compute on data that is random to them.</span></li>'
        f'<li>{icon("shield", 14)}<span><strong>Every result verified.</strong> Each masked result is integrity-checked '
        'before it is used, so a wrong answer from the server is caught, not returned.</span></li>'
        f'<li>{icon("check", 14)}<span><strong>Nothing stored.</strong> The platform logs metadata only: time, key, '
        'model, token counts and timing.</span></li></ul>'
    )
    keys_note = "" if has_keys else callout(f"You have no API keys yet. {key_hint} to run this example.", tone="info")
    return (head + keys_note + f'<div class="qs-layout"><ol class="qs-steps">{step_html}</ol>'
            f'<aside class="qs-side">{card(how, title="How it works")}</aside></div>'
            + card(code_block(full, lang="python", block_id="qs-full", title="example.py"),
                   title="A complete example", desc="Streaming works as with the OpenAI SDK; so do async clients "
                   "(<code>from pgw import AsyncOpenAI</code>)."))


# =====================================================================================================
# Settings
# =====================================================================================================
def settings_page(ctx: Ctx, *, auth: str, expires_at: float | None, audit: Sequence[AuditEvent] | None = None,
                  keys: Sequence[SdkKey] = ()) -> str:
    o, p = ctx.org, ctx.principal
    head = page_head("Settings", "Your organisation, your session and how the platform looks.")

    def rows(items: Sequence[tuple[str, str]]) -> str:
        return '<dl class="desc">' + "".join(f'<div class="desc-row"><dt>{e(k)}</dt><dd>{v}</dd></div>'
                                             for k, v in items) + "</dl>"

    org = card(rows([
        ("Name", e(o.name)),
        ("Organisation ID", f'<span class="mono-val"><code id="org-id">{e(o.id)}</code>{copy_button("org-id", "Copy ID")}</span>'),
        ("Plan", e(o.plan or "–")),
        ("Region", e(o.region or "–")),
        ("Created", e(utc(o.created_at, "%b %d, %Y"))),
        ("Request metadata kept", e(f"{o.retention_days} days" if o.retention_days else "–")),
        ("Prompt and completion content", '<span class="text-ok">Never stored</span>'),
    ]), title="Organisation")
    you = card(rows([
        ("Name", e(p.name or "–")),
        ("Email", e(p.email or "–")),
        ("Role", "Admin" if p.role == "admin" else "Viewer (read-only)"),
        ("Signed in with", "Single sign-on" if auth == "oidc" else "Access token"),
        ("Session expires", e(utc(expires_at, "%b %d, %H:%M UTC")) if expires_at else "–"),
    ]), title="Your session", aside=(
        f'<form method="post" action="{e(ctx.url("/logout"))}">{_hidden_csrf(ctx)}'
        f'<button class="btn btn-secondary btn-sm" type="submit">{icon("logout", 14)}Sign out</button></form>'))
    theme_opts = "".join(
        f'<label class="radio-card"><input type="radio" name="theme" value="{v}" data-theme-choice>'
        f'<span class="radio-card-body"><span class="theme-swatch theme-swatch-{v}" aria-hidden="true"></span>'
        f'<span class="radio-card-title">{t}</span></span></label>'
        for v, t in (("system", "System"), ("light", "Light"), ("dark", "Dark")))
    appearance = card(
        f'<fieldset class="fieldset"><legend class="sr-only">Theme</legend><div class="radio-cards radio-cards-3">'
        f'{theme_opts}</div></fieldset><noscript><p class="field-hint">Theme selection needs JavaScript; the platform '
        'follows your system setting.</p></noscript>',
        title="Appearance", desc="Saved in this browser.")
    activity = ""
    if audit is not None:
        names = {k.id: k.name for k in keys}
        if audit:
            audit_rows = [f'<tr><td>{time_tag(ev.ts, ctx.now)}</td><td>{e(_event_label(ev.event))}'
                    + (f' <span class="muted">({e(ev.reason.replace("_", " "))})</span>' if ev.reason else "")
                    + f'</td><td class="hide-sm">{e(names.get(ev.key_id or "", ev.key_id or ""))}</td>'
                    f'<td class="hide-sm muted">{e(ev.actor or "")}</td></tr>' for ev in audit]
            body = table([("Time", ""), ("Event", ""), ("Key", "hide-sm"), ("By", "hide-sm")], audit_rows,
                         caption="Audit log")
        else:
            body = empty_state("No activity yet", "Key changes and sign-ins of your organisation appear here.")
        activity = card(body, title="Audit log", desc="Key, enrollment and quota events of your organisation.",
                        flush=True)
    return head + f'<div class="settings-grid">{org}{you}{appearance}</div>' + activity


_EVENTS = {"key.created": "Key created", "key.revoked": "Key revoked", "sdk.enroll.ok": "Client enrolled",
           "sdk.enroll.failed": "Enrollment refused", "quota.exceeded": "Quota exceeded", "org.created":
           "Organisation created", "org.disabled": "Organisation disabled", "admin.denied": "Admin access denied",
           "admin_token.created": "Admin token created", "admin_token.revoked": "Admin token revoked"}


def _event_label(event: str) -> str:
    return _EVENTS.get(event, event.replace("_", " ").replace(".", " "))


# =====================================================================================================
# Sign-in
# =====================================================================================================
def login_page(*, prefix: str, token_enabled: bool, sso_label: str | None, error: str | None = None,
               unavailable: str | None = None) -> str:
    from pgw.platform.ui import LOGO

    err = callout(e(error), tone="bad") if error else ""
    body = ""
    if unavailable:
        body = callout(e(unavailable), tone="warn", title="Sign-in unavailable")
    else:
        if sso_label:
            body += (f'<div class="login-sso"><a class="btn btn-secondary login-submit" href="{e(prefix)}/sso/start">'
                     f'Continue with {e(sso_label)}</a></div>')
        if sso_label and token_enabled:
            body += '<div class="login-divider">or</div>'
        if token_enabled:
            body += (
                f'<form class="login-form" method="post" action="{e(prefix)}/login">'
                '<div class="field"><label class="field-label" for="token">Access token</label>'
                '<input class="input input-lg" id="token" name="token" type="password" required autocomplete="off" '
                'spellcheck="false" autofocus></div>'
                '<button class="btn btn-primary login-submit" type="submit">Sign in</button></form>'
            )
    return (
        '<main class="login" id="main">'
        '<div class="login-card">'
        f'<div class="login-brand">{LOGO}</div>'
        '<h1 class="login-title">Sign in to PGW</h1>'
        '<p class="login-sub">Keys, usage and request logs for your organisation.</p>'
        f'<div class="login-callout">{err}</div>{body}'
        f'<div class="login-foot">{icon("lock", 14)}<span>This console shows metadata only. Prompts and completions '
        'are masked in your process and never stored by PGW.</span></div>'
        '</div></main>'
    )


def error_page(*, prefix: str, title: str, message: str) -> str:
    return (f'<main class="login" id="main"><div class="login-card"><h1 class="login-title">{e(title)}</h1>'
            f'<p class="login-sub">{e(message)}</p><div class="login-foot"><a href="{e(prefix)}/">Back to the platform'
            '</a></div></div></main>')


def models_scope_valid(scoped: Sequence[str], models: Sequence[ModelInfo]) -> bool:
    known = {m.id for m in models}
    return all(m in known for m in scoped)


def today(now: float) -> date:
    return datetime.fromtimestamp(now, UTC).date()
