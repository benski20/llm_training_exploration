import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Overview, OverviewModel, UsageWindow } from "../api/types";
import { fmtAxisTime, fmtCompact, fmtDuration, fmtInt, fmtMs, fmtPercent } from "../lib/format";
import { explainReason } from "../lib/readiness";
import { useResource } from "../lib/useResource";
import { usageColumns } from "../lib/usage";
import {
  AreaChart,
  Badge,
  Card,
  Callout,
  ErrorState,
  IconAlert,
  IconInfo,
  IconCheck,
  Meter,
  PageHeader,
  Sparkline,
  Stat,
  StatusDot,
  Table,
  Tabs,
  modelTone,
  type Column,
} from "../ui";

const WINDOWS: { value: UsageWindow; label: string }[] = [
  { value: "1h", label: "1h" },
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
];

export function runwayTone(s: number | null | undefined): "ok" | "warn" | "bad" | undefined {
  if (s === null || s === undefined) return undefined;
  if (s < 6 * 3600) return "bad";
  if (s < 48 * 3600) return "warn";
  return "ok";
}

function Alerts({ o }: { o: Overview }) {
  const notReady = o.models.filter((m) => !m.ready);
  const noise = o.privacy.noise_monitor;
  const items: { tone: "info" | "warn" | "bad"; text: string }[] = [];
  if (o.integrity_failures_total > 0)
    items.push({ tone: "bad", text: `${fmtInt(o.integrity_failures_total)} integrity failure(s) recorded since start. Investigate before resetting the breaker.` });
  if (noise?.alarm) items.push({ tone: "bad", text: "The outbound-noise monitor is alarming: masked traffic is blocked (fail-closed)." });
  for (const m of notReady) {
    const r = explainReason(m.reason);
    items.push({ tone: r.tone, text: `${m.name} ${r.detail}` });
  }
  if (!items.length) return null;
  return (
    <div className="stack-sm">
      {items.map((it, i) => (
        <Callout key={i} tone={it.tone === "info" ? "accent" : it.tone} icon={it.tone === "info" ? <IconInfo /> : <IconAlert />}>
          {it.text}
        </Callout>
      ))}
    </div>
  );
}

function PostureRow({ ok, label, value }: { ok: boolean | null; label: string; value: string }) {
  return (
    <li className="posture-row">
      <span className={ok === null ? "posture-icon is-unknown" : ok ? "posture-icon is-ok" : "posture-icon is-bad"} aria-hidden="true">
        {ok ? <IconCheck size={12} /> : <IconAlert size={12} />}
      </span>
      <span className="posture-label">{label}</span>
      <span className="posture-value">{value}</span>
    </li>
  );
}

export function OverviewPage() {
  const [win, setWin] = useState<UsageWindow>("24h");
  const ov = useResource("overview", () => api.overview(), 10_000);
  const usage = useResource(`usage:${win}`, () => api.usage(win), 30_000);
  const o = ov.data;
  const c = usageColumns(usage.data);
  const bucket = usage.data?.bucket_s ?? 3600;
  const fx = (t: number) => fmtAxisTime(t, bucket);
  const errRate = o && o.requests_24h ? o.errors_24h / o.requests_24h : null;
  const noise = o?.privacy.noise_monitor;
  const noiseState = noise?.alarm ? "alarm" : noise?.state ?? "unknown";

  const modelCols: Column<OverviewModel>[] = [
    {
      key: "name",
      header: "Model",
      render: (m) => (
        <span className="cell-model">
          <StatusDot tone={modelTone(m.ready, m.state)} label={m.ready ? "ready" : "not ready"} />
          <Link to="/models" className="link-quiet">
            {m.name}
          </Link>
        </span>
      ),
    },
    {
      key: "state",
      header: "State",
      render: (m) => (
        <span className="cell-stack">
          <span>{m.state}</span>
          {!m.ready && m.reason ? <span className="cell-sub">{explainReason(m.reason).short}</span> : null}
        </span>
      ),
    },
    { key: "ring", header: "Ring", mono: true, render: (m) => m.ring, className: "hide-sm" },
    { key: "masks", header: "Masks", align: "right", render: (m) => <span className="num">{fmtCompact(m.masks_available)}</span> },
    {
      key: "runway",
      header: "Runway",
      align: "right",
      render: (m) => {
        const t = runwayTone(m.masks_runway_s);
        return <span className={t ? `num text-${t}` : "num"}>{m.masks_runway_s === null || m.masks_runway_s === undefined ? "idle" : fmtDuration(m.masks_runway_s)}</span>;
      },
    },
  ];

  if (ov.error && !o) {
    return (
      <>
        <PageHeader title="Overview" />
        <ErrorState error={ov.error} onRetry={ov.reload} />
      </>
    );
  }

  const maxMasks = Math.max(1, ...(o?.models.map((m) => m.masks_available ?? 0) ?? [1]));

  return (
    <div className="page">
      <PageHeader
        title="Overview"
        description="Health, traffic and privacy posture of this gateway."
        actions={<Tabs label="Usage window" variant="segmented" items={WINDOWS} value={win} onChange={setWin} />}
      />
      {o ? <Alerts o={o} /> : null}

      <div className="stats-grid">
        <Stat label="Requests · 24h" loading={!o} value={fmtCompact(o?.requests_24h)} sub={o ? `${fmtInt(o.requests_24h)} total` : null} chart={<Sparkline values={c.requests} label="Requests trend" />} />
        <Stat label="Tokens · 24h" loading={!o} value={fmtCompact(o?.tokens_24h)} sub="prompt + completion" chart={<Sparkline values={c.tokens} label="Tokens trend" />} />
        <Stat
          label="Errors · 24h"
          loading={!o}
          value={fmtInt(o?.errors_24h)}
          tone={errRate !== null && errRate > 0.05 ? "warn" : undefined}
          sub={errRate === null ? "no traffic" : `${fmtPercent(errRate, 2)} of requests`}
          chart={<Sparkline values={c.errors} label="Errors trend" />}
        />
        <Stat
          label="Integrity failures"
          loading={!o}
          value={fmtInt(o?.integrity_failures_total)}
          tone={o ? (o.integrity_failures_total > 0 ? "bad" : "ok") : undefined}
          sub={o && o.integrity_failures_total === 0 ? "None since start" : "Since start · alert on any"}
        />
        <Stat label="Freivalds-checked calls" loading={!o} value={fmtCompact(o?.freivalds_checked_total)} sub={o?.privacy.freivalds_rate != null ? `check rate ${fmtPercent(o.privacy.freivalds_rate)}` : "since start"} />
        <Stat label="Streaming requests" loading={!o} value={fmtCompact(o?.stream_requests_total)} sub="since start" />
      </div>

      <div className="grid-2">
        <Card title="Requests" description={`Per ${fmtDuration(bucket)} bucket · ${win}`}>
          <AreaChart
            label="Requests and errors over time"
            x={c.t}
            series={[
              { name: "Requests", values: c.requests, tone: 1, area: true },
              { name: "Errors", values: c.errors, tone: 2 },
            ]}
            format={(v) => fmtCompact(v)}
            formatX={fx}
          />
        </Card>
        <Card title="Tokens" description="Prompt and completion tokens">
          <AreaChart
            label="Prompt and completion tokens over time"
            x={c.t}
            series={[
              { name: "Completion", values: c.completion, tone: 1, area: true },
              { name: "Prompt", values: c.prompt, tone: 2 },
            ]}
            format={(v) => fmtCompact(v)}
            formatX={fx}
          />
        </Card>
        <Card title="Latency" description="Request duration, p50 and p95">
          <AreaChart
            label="Request latency p50 and p95"
            x={c.t}
            series={[
              { name: "p95", values: c.p95, tone: 1, area: true },
              { name: "p50", values: c.p50, tone: 2 },
            ]}
            format={fmtMs}
            formatX={fx}
          />
        </Card>
        <Card title="Time to first token" description="Median TTFT per bucket">
          <AreaChart label="Time to first token, median" x={c.t} series={[{ name: "TTFT p50", values: c.ttft, tone: 1, area: true }]} format={fmtMs} formatX={fx} />
        </Card>
      </div>

      <div className="grid-3">
        <Card title="Models" description="Readiness and mask runway" flush className="span-2">
          <Table columns={modelCols} rows={o?.models} loading={!o} rowKey={(m) => m.name} caption="Model health" empty="No models configured" />
        </Card>
        <Card title="Privacy posture" description="What protects every request right now">
          <ul className="posture">
            <PostureRow ok={o ? o.privacy.masked : null} label="Masked compute" value={o ? (o.privacy.masked ? "On" : "Off") : "—"} />
            <PostureRow ok={o ? Boolean(o.privacy.ring) : null} label="Ring" value={o?.privacy.ring ?? "—"} />
            <PostureRow
              ok={o ? (o.privacy.freivalds_rate ?? 0) > 0 : null}
              label="Freivalds verification"
              value={o?.privacy.freivalds_rate != null ? fmtPercent(o.privacy.freivalds_rate) : "—"}
            />
            <PostureRow ok={o ? !noise?.alarm && noiseState !== "unavailable" : null} label="Noise monitor" value={noiseState} />
            <PostureRow ok={o ? o.integrity_failures_total === 0 : null} label="Integrity failures" value={fmtInt(o?.integrity_failures_total)} />
          </ul>
          <Link to="/security" className="card-link">
            What the compute provider can see →
          </Link>
        </Card>
      </div>

      <Card title="Mask inventory" description="One-time pads available per model; each token consumes one slot." actions={<Link to="/masks" className="card-link">Details →</Link>}>
        <div className="inventory">
          {o?.models.map((m) => {
            const t = runwayTone(m.masks_runway_s);
            return (
              <div className="inventory-row" key={m.name}>
                <div className="inventory-name">{m.name}</div>
                <Meter value={m.masks_available ?? 0} max={maxMasks} tone={t === "bad" ? "bad" : t === "warn" ? "warn" : "accent"} label={`${m.name} masks available`} />
                <div className="inventory-num num">{fmtCompact(m.masks_available)}</div>
                <div className="inventory-runway">
                  {t ? (
                    <Badge tone={t} dot>
                      {fmtDuration(m.masks_runway_s)}
                    </Badge>
                  ) : (
                    <span className="muted">idle</span>
                  )}
                </div>
              </div>
            );
          })}
          {!o ? <div className="muted">Loading…</div> : null}
        </div>
      </Card>
    </div>
  );
}
