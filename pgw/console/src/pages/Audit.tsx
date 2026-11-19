import { useState } from "react";
import { api } from "../api/client";
import type { AuditEvent } from "../api/types";
import { useGateway } from "../app/gateway";
import { fmtDateTime, fmtInt, fmtMs, fmtRelative } from "../lib/format";
import { useResource } from "../lib/useResource";
import {
  Badge,
  Button,
  Card,
  Callout,
  DescList,
  Dialog,
  EmptyState,
  ErrorState,
  Field,
  IconList,
  IconLock,
  IconRefresh,
  Mono,
  PageHeader,
  Select,
  Table,
  statusTone,
  type Column,
} from "../ui";

const PAGE = 50;

const STATUS_OPTIONS = [
  ["", "Any status"],
  ["2xx", "Success (2xx)"],
  ["4xx", "Client errors (4xx)"],
  ["5xx", "Server errors (5xx)"],
  ["401", "401 Unauthenticated"],
  ["429", "429 Rate limited"],
  ["502", "502 Integrity failed"],
  ["503", "503 Unavailable"],
] as const;

/** Labels for audit fields in the detail drawer, in display order. */
const FIELDS: [keyof AuditEvent, string][] = [
  ["event", "Event"],
  ["request_id", "Request id"],
  ["key_id", "Key id"],
  ["key_name", "Key name"],
  ["model", "Model"],
  ["endpoint", "Endpoint"],
  ["status", "Status"],
  ["code", "Error code"],
  ["stream", "Streamed"],
  ["prompt_tokens", "Prompt tokens"],
  ["completion_tokens", "Completion tokens"],
  ["prompts", "Prompts"],
  ["masks_reserved", "Masks reserved"],
  ["masks_consumed", "Masks consumed"],
  ["masks_burned", "Masks burned"],
  ["queue_ms", "Queue wait"],
  ["ttft_ms", "Time to first token"],
  ["duration_ms", "Duration"],
  ["finish_reason", "Finish reason"],
  ["mode", "Mode"],
  ["ring", "Ring"],
  ["fingerprint", "Fingerprint"],
  ["action", "Action"],
  ["target", "Target"],
  ["scopes", "Scopes"],
  ["reason", "Reason"],
  ["client", "Client"],
  ["retry_after_s", "Retry after"],
];

function fieldValue(k: keyof AuditEvent, v: unknown) {
  if (v === undefined || v === null || v === "") return null;
  if (k.endsWith("_ms")) return <span className="num">{fmtMs(v as number)}</span>;
  if (k === "retry_after_s") return <span className="num">{String(v)} s</span>;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return <span className="num">{fmtInt(v)}</span>;
  if (["request_id", "key_id", "fingerprint", "ring", "client"].includes(k)) return <Mono copy={k === "request_id" || k === "fingerprint" ? String(v) : undefined}>{String(v)}</Mono>;
  return String(v);
}

export function AuditPage() {
  const gw = useGateway();
  const keys = useResource("keys", () => api.keys());
  const [keyId, setKeyId] = useState("");
  const [model, setModel] = useState("");
  const [status, setStatus] = useState("");
  const [cursors, setCursors] = useState<(string | number | null)[]>([null]);
  const [open, setOpen] = useState<AuditEvent | null>(null);
  const before = cursors[cursors.length - 1];
  const q = { limit: PAGE, before, key_id: keyId || undefined, model: model || undefined, status: status || undefined };
  const page = useResource(JSON.stringify(q), () => api.audit(q));
  const events = page.data?.events;

  const resetPaging = () => setCursors([null]);
  const filtered = keyId || model || status;

  const cols: Column<AuditEvent>[] = [
    { key: "ts", header: "Time", render: (e) => <span title={e.ts} className="num nowrap">{fmtDateTime(e.ts)}</span> },
    { key: "event", header: "Event", render: (e) => <span className="cell-stack"><span>{e.event}</span>{e.action ? <span className="cell-sub">{e.action}</span> : null}</span> },
    { key: "key", header: "Key", render: (e) => (e.key_name || e.key_id ? <span className="cell-stack"><span>{e.key_name ?? ""}</span><span className="cell-sub mono">{e.key_id}</span></span> : <span className="muted">—</span>) },
    { key: "model", header: "Model", className: "hide-sm", render: (e) => e.model ?? e.target ?? <span className="muted">—</span> },
    {
      key: "status",
      header: "Status",
      render: (e) =>
        e.status ? (
          <Badge tone={statusTone(e.status)} title={e.code}>
            <span className="num">{e.status}</span>
            {e.code ? <span className="badge-sub">{e.code}</span> : null}
          </Badge>
        ) : (
          <span className="muted">—</span>
        ),
    },
    { key: "tokens", header: "Tokens", align: "right", className: "hide-md", render: (e) => (e.prompt_tokens !== undefined ? <span className="num">{fmtInt(e.prompt_tokens)} → {fmtInt(e.completion_tokens)}</span> : <span className="muted">—</span>) },
    { key: "dur", header: "Duration", align: "right", render: (e) => <span className="num">{fmtMs(e.duration_ms)}</span> },
    { key: "rid", header: "Request", className: "hide-lg", mono: true, render: (e) => (e.request_id ? <span className="truncate">{e.request_id}</span> : <span className="muted">—</span>) },
  ];

  return (
    <div className="page">
      <PageHeader
        title="Audit log"
        description="Who did what, when, and with which result. Records carry ids, counts, durations and codes; never prompts or completions."
        actions={
          <Button icon={<IconRefresh />} onClick={() => (cursors.length > 1 ? resetPaging() : void page.reload())} loading={page.loading && Boolean(events)}>
            Refresh
          </Button>
        }
      />
      <div className="filters" role="search" aria-label="Filter audit events">
        <Field label="API key">
          {({ id }) => (
            <Select id={id} value={keyId} onChange={(e) => { setKeyId(e.target.value); resetPaging(); }}>
              <option value="">All keys</option>
              <option value="admin-token">admin-token</option>
              {keys.data?.keys.map((k) => (
                <option key={k.key_id} value={k.key_id}>
                  {k.name} ({k.key_id})
                </option>
              ))}
            </Select>
          )}
        </Field>
        <Field label="Model">
          {({ id }) => (
            <Select id={id} value={model} onChange={(e) => { setModel(e.target.value); resetPaging(); }}>
              <option value="">All models</option>
              {gw.models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </Select>
          )}
        </Field>
        <Field label="Status">
          {({ id }) => (
            <Select id={id} value={status} onChange={(e) => { setStatus(e.target.value); resetPaging(); }}>
              {STATUS_OPTIONS.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </Select>
          )}
        </Field>
        {filtered ? (
          <Button variant="ghost" className="filters-clear" onClick={() => { setKeyId(""); setModel(""); setStatus(""); resetPaging(); }}>
            Clear filters
          </Button>
        ) : null}
      </div>

      {page.error && !events ? (
        <ErrorState error={page.error} onRetry={page.reload} />
      ) : (
        <Card
          flush
          footer={
            <div className="pager">
              <span className="muted small">
                {events ? `${events.length} event${events.length === 1 ? "" : "s"}` : ""}
                {cursors.length > 1 ? ` · page ${cursors.length}` : " · newest first"}
              </span>
              <div className="pager-btns">
                <Button size="sm" disabled={cursors.length <= 1} onClick={() => setCursors((c) => c.slice(0, -1))}>
                  Newer
                </Button>
                <Button size="sm" disabled={!page.data?.next_before} onClick={() => setCursors((c) => [...c, page.data?.next_before ?? null])}>
                  Older
                </Button>
              </div>
            </div>
          }
        >
          <Table
            columns={cols}
            rows={page.loading ? undefined : events}
            loading={page.loading}
            rowKey={(e) => `${e.ts}|${e.request_id ?? ""}|${e.event}|${e.target ?? ""}`}
            onRowClick={setOpen}
            rowLabel={(e) => `${e.event} at ${fmtDateTime(e.ts)}, open details`}
            caption="Audit events"
            empty={<EmptyState icon={<IconList size={20} />} title={filtered ? "No events match these filters" : "No audit events yet"} />}
          />
        </Card>
      )}

      <Dialog open={open !== null} onClose={() => setOpen(null)} variant="drawer" title={open?.event ?? ""} description={open ? `${fmtDateTime(open.ts)} · ${fmtRelative(open.ts)}` : undefined}>
        {open ? (
          <>
            <DescList items={FIELDS.map(([k, label]) => [label, fieldValue(k, open[k])] as [string, ReturnType<typeof fieldValue>]).filter(([, v]) => v !== null)} />
            <Callout tone="neutral" icon={<IconLock />} className="mt-4">
              Prompts and completions are never written to the audit log. Use the request id to correlate with application logs.
            </Callout>
          </>
        ) : null}
      </Dialog>
    </div>
  );
}
