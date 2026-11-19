import { useState } from "react";
import { api } from "../api/client";
import type { ModelStatus } from "../api/types";
import { useGateway } from "../app/gateway";
import { useSession } from "../app/session";
import { fmtBytes, fmtCompact, fmtDuration, fmtInt, fmtMs, fmtPercent, shortHash } from "../lib/format";
import {
  Badge,
  Button,
  Callout,
  Card,
  Checkbox,
  ConfirmDialog,
  DescList,
  EmptyState,
  ErrorState,
  IconAlert,
  IconCube,
  IconRefresh,
  Loading,
  Meter,
  Mono,
  PageHeader,
  Stat,
  StatusDot,
  modelTone,
  useToast,
} from "../ui";

function SplitBar({ a, b, labelA, labelB }: { a: number; b: number; labelA: string; labelB: string }) {
  const total = a + b;
  return (
    <div className="split">
      <Meter value={a} max={total || 1} label={`${labelA} share`} height={8} />
      <div className="split-legend">
        <span>
          <span className="legend-swatch tone-1" aria-hidden="true" /> {labelA} <span className="num">{total ? fmtPercent(a / total, 0) : "—"}</span>
        </span>
        <span>
          <span className="legend-swatch tone-track" aria-hidden="true" /> {labelB} <span className="num">{total ? fmtPercent(b / total, 0) : "—"}</span>
        </span>
      </div>
    </div>
  );
}

function ModelDetail({ m, onChanged }: { m: ModelStatus; onChanged: () => void }) {
  const { isAdmin } = useSession();
  const toast = useToast();
  const [confirm, setConfirm] = useState<"reload" | "reset" | null>(null);
  const [resetFlags, setResetFlags] = useState({ breaker: true, noise: true });
  const b = m.backend ?? {};
  const s = m.sessions;
  const br = m.breaker;
  const nm = m.noise_monitor;
  const cl = m.client ?? {};
  const checked = b.freivalds_checked ?? 0;
  const skipped = b.freivalds_skipped ?? 0;
  const masked = m.mode === "masked";

  return (
    <div className="stack">
      <Card>
        <div className="model-head">
          <div className="model-title">
            <StatusDot tone={modelTone(m.ready, m.state)} pulse={!m.ready} />
            <h2>{m.name}</h2>
            <Badge tone={m.ready ? "ok" : "warn"}>{m.ready ? "Ready" : m.state}</Badge>
            <Badge tone={masked ? "accent" : "warn"}>{masked ? "masked" : m.mode}</Badge>
          </div>
          {isAdmin ? (
            <div className="model-actions">
              <Button icon={<IconRefresh />} onClick={() => setConfirm("reload")}>
                Reload
              </Button>
              <Button onClick={() => setConfirm("reset")} disabled={!br?.open && !nm?.alarm} title={!br?.open && !nm?.alarm ? "Nothing to reset: breaker closed, no noise alarm" : undefined}>
                Reset safety
              </Button>
            </div>
          ) : null}
        </div>
        {!m.ready && m.reason ? (
          <Callout tone="warn" icon={<IconAlert />} className="mt-3">
            Not ready: {m.reason}
            {m.last_error ? <span className="muted"> · last error: {m.last_error}</span> : null}
          </Callout>
        ) : null}
        <DescList
          cols={2}
          className="mt-3"
          items={[
            ["Fingerprint", m.fingerprint ? <Mono copy={m.fingerprint} title={m.fingerprint}>{m.fingerprint_short ?? shortHash(m.fingerprint)}</Mono> : "—"],
            ["Ring", <span className="mono">{m.ring}</span>],
            ["Freivalds rate", m.freivalds_rate != null ? fmtPercent(m.freivalds_rate) : "—"],
            ["Max context", fmtInt(m.max_context)],
            ["Service", m.service?.origin ? <span className="mono">{m.service.origin}</span> : "—"],
            ["Masks available", fmtInt(m.masks_available)],
          ]}
        />
      </Card>

      <div className="stats-grid stats-4">
        <Stat label="Active sessions" value={s ? `${s.active} / ${s.max_active}` : "—"} sub={s?.draining ? "draining" : "generating now"} />
        <Stat label="Waiting" value={s ? `${s.waiting} / ${s.max_queue}` : "—"} sub={s ? `${fmtInt(s.rejected)} rejected since start` : undefined} tone={s && s.waiting > s.max_queue * 0.5 ? "warn" : undefined} />
        <Stat
          label="Integrity breaker"
          value={br ? (br.open ? "Open" : "Closed") : "—"}
          tone={br ? (br.open ? "bad" : "ok") : undefined}
          sub={br ? (br.open ? `open ${fmtDuration(br.open_for_s)} · ${br.last_code ?? ""}` : `${fmtInt(br.trips)} trips since start`) : undefined}
        />
        <Stat label="Integrity failures" value={fmtInt(b.integrity_failures)} tone={b.integrity_failures ? "bad" : "ok"} sub="Freivalds or range check" />
      </div>

      <div className="grid-2">
        <Card title="Compute backend" description="Cumulative counters since the model started">
          <DescList
            items={[
              ["Round trips", <span className="num">{fmtInt(b.round_trips)}</span>],
              ["Linear calls · rows", <span className="num">{fmtCompact(b.calls)} · {fmtCompact(b.rows)}</span>],
              ["Freivalds checked", <span className="num">{fmtInt(checked)}</span>],
              ["Freivalds skipped", <span className="num">{fmtInt(skipped)}</span>],
              ["Check coverage", <span className="num">{checked + skipped ? fmtPercent(checked / (checked + skipped)) : "—"}</span>],
              ["Bytes up / down", <span className="num">{fmtBytes(b.bytes_up)} / {fmtBytes(b.bytes_down)}</span>],
            ]}
          />
          <div className="subhead">Time split</div>
          <SplitBar a={b.remote_seconds ?? 0} b={b.local_seconds ?? 0} labelA={`Remote ${fmtDuration(b.remote_seconds)}`} labelB={`Local ${fmtDuration(b.local_seconds)}`} />
        </Card>
        <div className="stack">
          <Card title="Service client">
            <DescList
              items={[
                ["Requests · attempts", <span className="num">{fmtInt(cl.requests)} · {fmtInt(cl.attempts)}</span>],
                ["Retries", <span className={cl.retries ? "num text-warn" : "num"}>{fmtInt(cl.retries)}</span>],
                ["Failures", <span className={cl.failures ? "num text-bad" : "num"}>{fmtInt(cl.failures)}</span>],
                ["Last RTT", <span className="num">{cl.last_rtt_s != null ? fmtMs(cl.last_rtt_s * 1000) : "—"}</span>],
                ["Server", m.service?.server ?? "—"],
              ]}
            />
          </Card>
          <Card title="Outbound-noise monitor" description="Checks that what leaves the gateway is statistically indistinguishable from noise.">
            <DescList
              items={[
                ["State", nm ? <Badge tone={nm.alarm ? "bad" : nm.state === "ok" || nm.state === "active" ? "ok" : "neutral"} dot>{nm.alarm ? "alarm" : nm.state ?? "—"}</Badge> : "—"],
                ["Samples · windows", <span className="num">{fmtCompact(nm?.samples)} · {fmtInt(nm?.windows)}</span>],
                ["Duplicate rows", <span className={nm?.duplicate_rows ? "num text-bad" : "num"}>{fmtInt(nm?.duplicate_rows)}</span>],
                ["Reasons", nm?.reasons?.length ? nm.reasons.join(", ") : <span className="muted">none</span>],
              ]}
            />
          </Card>
        </div>
      </div>

      <ConfirmDialog
        open={confirm === "reload"}
        onClose={() => setConfirm(null)}
        tone="primary"
        title={`Reload ${m.name}?`}
        confirmLabel="Reload model"
        onConfirm={async () => {
          const r = await api.reloadModel(m.name);
          toast({ title: `${m.name} reloaded`, description: r.ready ? "The model is ready." : "The model is starting.", tone: "ok" });
          onChanged();
        }}
      >
        <p>The model drains, re-verifies its pack and the compute service, then starts again. In-flight requests finish first; new requests queue or fail with 503 meanwhile.</p>
      </ConfirmDialog>
      <ConfirmDialog
        open={confirm === "reset"}
        onClose={() => setConfirm(null)}
        title={`Reset safety state of ${m.name}?`}
        confirmLabel="Reset"
        onConfirm={async () => {
          await api.resetModel(m.name, resetFlags);
          toast({ title: "Safety state reset", tone: "ok" });
          onChanged();
        }}
      >
        <p>Only reset after the cause has been investigated. A reset re-enables masked traffic.</p>
        <div className="stack-sm mt-3">
          <Checkbox label="Integrity circuit breaker" checked={resetFlags.breaker} onChange={(e) => setResetFlags((f) => ({ ...f, breaker: e.target.checked }))} />
          <Checkbox label="Outbound-noise alarm" checked={resetFlags.noise} onChange={(e) => setResetFlags((f) => ({ ...f, noise: e.target.checked }))} />
        </div>
      </ConfirmDialog>
    </div>
  );
}

export function ModelsPage() {
  const gw = useGateway();
  const st = gw.status;
  return (
    <div className="page">
      <PageHeader title="Models" description="Per-model state, verification counters and operator actions." />
      {st.error && !st.data ? (
        <ErrorState error={st.error} onRetry={st.reload} />
      ) : !st.data ? (
        <Loading />
      ) : !gw.models.length ? (
        <EmptyState icon={<IconCube size={20} />} title="No models configured">
          Add models to the gateway configuration (PGW_GATEWAY_MODELS_FILE).
        </EmptyState>
      ) : (
        <div className="models-layout">
          <nav className="model-list" aria-label="Models">
            {gw.models.map((m) => (
              <button key={m.name} type="button" className={m.name === gw.model ? "model-item is-active" : "model-item"} aria-current={m.name === gw.model ? "true" : undefined} onClick={() => gw.setModel(m.name)}>
                <StatusDot tone={modelTone(m.ready, m.state)} />
                <span className="model-item-text">
                  <span className="model-item-name">{m.name}</span>
                  <span className="model-item-sub">
                    {m.ready ? "ready" : m.reason || m.state} · {m.ring}
                  </span>
                </span>
              </button>
            ))}
          </nav>
          {gw.selected ? <ModelDetail key={gw.selected.name} m={gw.selected} onChanged={() => void st.reload()} /> : null}
        </div>
      )}
    </div>
  );
}
