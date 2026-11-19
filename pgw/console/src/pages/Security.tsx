import { api } from "../api/client";
import { useGateway } from "../app/gateway";
import { fmtPercent } from "../lib/format";
import { useResource } from "../lib/useResource";
import { Badge, Card, DescList, IconCheck, IconEye, IconEyeOff, IconFile, IconX, PageHeader, Table, type Column } from "../ui";
import type { ModelStatus } from "../api/types";

const SEES = [
  ["Masked activations", "x + r for every linear layer, where r is a fresh one-time pad. To the provider this is uniformly random noise."],
  ["Public model weights", "The open-weight model it multiplies with. The fingerprint proves which weights were used."],
  ["Shapes and timing", "Number of rows per call, request sizes and timing. These reveal roughly how long prompts and answers are."],
] as const;

const NEVER = [
  ["Prompts and completions", "Text, token ids and logits stay on this gateway."],
  ["Unmasked activations", "Masks are subtracted only here, after verifying the result."],
  ["The masks themselves", "Pads come from your encrypted mask store and are used exactly once."],
  ["API keys and identities", "Application keys and user identities are not sent; telemetry is disabled."],
] as const;

const DOCS = [
  ["docs/security/whitepaper.md", "How masked inference works and why it is exact"],
  ["docs/security/threat-model.md", "Who is trusted for what, and residual risks"],
  ["docs/operations/encrypted-masks.md", "Mask store encryption, rollback protection, refills"],
  ["docs/compliance/audit-readiness.md", "Audit log fields and evidence for reviews"],
  ["docs/api/gateway-api.md", "API reference, errors and limits"],
] as const;

export function SecurityPage() {
  const gw = useGateway();
  const ov = useResource("overview", () => api.overview(), 30_000);
  const st = gw.status.data;
  const tls = st?.tls;
  const httpsEgress = st?.egress_allowlist?.length ? st.egress_allowlist.every((o) => o.startsWith("https://")) : undefined;

  const cols: Column<ModelStatus>[] = [
    { key: "name", header: "Model", render: (m) => m.name },
    { key: "mode", header: "Compute", render: (m) => (m.mode === "masked" ? <Badge tone="accent">masked</Badge> : <Badge tone="warn">{m.mode}</Badge>) },
    { key: "ring", header: "Ring", mono: true, render: (m) => m.ring },
    { key: "fr", header: "Freivalds rate", align: "right", render: (m) => <span className="num">{m.freivalds_rate != null ? fmtPercent(m.freivalds_rate) : "—"}</span> },
    {
      key: "noise",
      header: "Noise monitor",
      render: (m) => (m.noise_monitor ? <Badge tone={m.noise_monitor.alarm ? "bad" : "neutral"} dot>{m.noise_monitor.alarm ? "alarm" : m.noise_monitor.state ?? "—"}</Badge> : <span className="muted">—</span>),
    },
    { key: "breaker", header: "Breaker", render: (m) => (m.breaker ? <Badge tone={m.breaker.open ? "bad" : "ok"} dot>{m.breaker.open ? "open" : "closed"}</Badge> : "—") },
  ];

  const yes = (v: boolean | undefined, on = "On", off = "Off") => (v === undefined ? <span className="muted">not reported</span> : v ? <Badge tone="ok" dot>{on}</Badge> : <Badge tone="warn" dot>{off}</Badge>);

  return (
    <div className="page">
      <PageHeader title="Security" description="What the GPU compute provider can and cannot see, and the controls currently in force on this gateway." />

      <div className="grid-2">
        <Card title="What the compute provider sees" className="trust-card">
          <ul className="trust-list">
            {SEES.map(([t, d]) => (
              <li key={t}>
                <span className="trust-icon is-sees" aria-hidden="true">
                  <IconEye size={14} />
                </span>
                <div>
                  <div className="trust-title">{t}</div>
                  <div className="trust-text">{d}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>
        <Card title="What never leaves your gateway" className="trust-card">
          <ul className="trust-list">
            {NEVER.map(([t, d]) => (
              <li key={t}>
                <span className="trust-icon is-never" aria-hidden="true">
                  <IconEyeOff size={14} />
                </span>
                <div>
                  <div className="trust-title">{t}</div>
                  <div className="trust-text">{d}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card title="How every result is checked">
        <ol className="steps">
          <li>
            <span className="step-n">1</span>
            <div>
              <div className="trust-title">Mask</div>
              <div className="trust-text">The gateway adds a one-time pad r to each activation x and sends only x + r.</div>
            </div>
          </li>
          <li>
            <span className="step-n">2</span>
            <div>
              <div className="trust-title">Compute</div>
              <div className="trust-text">The provider returns W(x + r) using exact integer ring arithmetic.</div>
            </div>
          </li>
          <li>
            <span className="step-n">3</span>
            <div>
              <div className="trust-title">Verify</div>
              <div className="trust-text">A Freivalds check confirms the product cheaply; failures trip the integrity breaker and fail closed.</div>
            </div>
          </li>
          <li>
            <span className="step-n">4</span>
            <div>
              <div className="trust-title">Unmask</div>
              <div className="trust-text">The gateway subtracts the precomputed W r and continues the forward pass locally.</div>
            </div>
          </li>
        </ol>
      </Card>

      <div className="grid-3">
        <Card title="Current settings" className="span-2" flush>
          <Table columns={cols} rows={gw.models} rowKey={(m) => m.name} caption="Per-model security settings" empty="No models" />
          <div className="card-pad">
            <DescList
              cols={2}
              items={[
                ["Masked compute", yes(ov.data?.privacy.masked)],
                ["Production policy", yes(st?.production_policy, "Enforced", "Dev modes allowed")],
                ["TLS to compute service", tls?.enabled !== undefined ? yes(tls.enabled) : yes(httpsEgress, "HTTPS only", "Plain HTTP present")],
                ["Mutual TLS", yes(tls?.mtls)],
                ["Telemetry", st?.telemetry ? <Badge tone={st.telemetry === "disabled" ? "ok" : "warn"} dot>{st.telemetry}</Badge> : <span className="muted">—</span>],
                ["Audit log", st?.audit_log ? <span className="mono small">{st.audit_log}</span> : <span className="muted">logger only</span>],
              ]}
            />
            {st?.egress_allowlist?.length ? (
              <div className="egress">
                <div className="subhead">Egress allowlist</div>
                <ul className="egress-list">
                  {st.egress_allowlist.map((o) => (
                    <li key={o} className="mono small">
                      {o.startsWith("https://") ? <IconCheck size={12} className="text-ok" /> : <IconX size={12} className="text-warn" />}
                      {o}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </Card>
        <Card title="Documentation" description="In the PGW distribution">
          <ul className="doc-list">
            {DOCS.map(([p, d]) => (
              <li key={p}>
                <IconFile size={14} />
                <div>
                  <div className="mono small doc-path">{p}</div>
                  <div className="trust-text">{d}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
