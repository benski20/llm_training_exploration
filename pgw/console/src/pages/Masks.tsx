import { useState } from "react";
import { api } from "../api/client";
import type { MaskInventoryRow, UsageWindow } from "../api/types";
import { useGateway } from "../app/gateway";
import { fmtAxisTime, fmtBytes, fmtCompact, fmtDuration, fmtInt, shortHash } from "../lib/format";
import { useResource } from "../lib/useResource";
import { consumptionRate, runwaySeconds, sum, usageColumns } from "../lib/usage";
import {
  AreaChart,
  Badge,
  Callout,
  Card,
  DescList,
  EmptyState,
  ErrorState,
  IconAlert,
  IconLayers,
  Loading,
  Meter,
  PageHeader,
  Stat,
  Table,
  Tabs,
  type Column,
} from "../ui";
import { runwayTone } from "./Overview";

export function MasksPage() {
  const gw = useGateway();
  const [win, setWin] = useState<UsageWindow>("24h");
  const masks = useResource("masks", () => api.masks(), 30_000);
  const ov = useResource("overview", () => api.overview(), 30_000);
  const usage = useResource(`usage:${win}`, () => api.usage(win), 60_000);
  const m = masks.data;
  const store = m?.store;
  const c = usageColumns(usage.data);
  const rate = consumptionRate(usage.data);
  const models = Object.entries(m?.models ?? {});
  const low = models.filter(([, x]) => x.available_slots !== null && x.low_water_slots !== null && x.available_slots < x.low_water_slots);

  const invCols: Column<MaskInventoryRow>[] = [
    { key: "fp", header: "Fingerprint", mono: true, render: (r) => <span title={r.fingerprint}>{shortHash(r.fingerprint)}</span> },
    {
      key: "model",
      header: "Model",
      render: (r) => models.find(([, x]) => x.fingerprint === r.fingerprint)?.[0] ?? <span className="muted">unassigned</span>,
    },
    { key: "ring", header: "Ring", mono: true, render: (r) => r.ring },
    { key: "seg", header: "Segments", align: "right", render: (r) => <span className="num">{fmtInt(r.segments)}</span> },
    { key: "avail", header: "Available", align: "right", render: (r) => <span className="num">{fmtInt(r.slots_available)}</span> },
    { key: "total", header: "Total slots", align: "right", className: "hide-sm", render: (r) => <span className="num">{fmtInt(r.slots_total)}</span> },
    { key: "bytes", header: "On disk", align: "right", render: (r) => <span className="num">{fmtBytes(r.bytes_on_disk)}</span> },
  ];

  if (masks.error && !m) {
    return (
      <div className="page">
        <PageHeader title="Masks" />
        <ErrorState error={masks.error} onRetry={masks.reload} />
      </div>
    );
  }

  return (
    <div className="page">
      <PageHeader
        title="Masks"
        description="One-time pads in the encrypted mask store. Every generated token consumes one slot per model; slots are never reused."
        actions={<Tabs label="Trend window" variant="segmented" items={[{ value: "1h", label: "1h" }, { value: "24h", label: "24h" }, { value: "7d", label: "7d" }]} value={win} onChange={setWin} />}
      />
      {!m ? <Loading /> : null}
      {store?.tainted ? (
        <Callout tone="bad" icon={<IconAlert />} title="Mask store tainted">
          The store detected a possible rollback or tampering and refuses to serve masks. See docs/operations/encrypted-masks.md.
        </Callout>
      ) : null}
      {low.length ? (
        <Callout tone="warn" icon={<IconAlert />} title="Below low water">
          {low.map(([n]) => n).join(", ")} {low.length === 1 ? "is" : "are"} below the low-water mark. Generate and import more masks with the mask factory.
        </Callout>
      ) : null}
      {m && !store ? (
        <EmptyState icon={<IconLayers size={20} />} title="No mask store configured">
          This gateway runs without masked models.
        </EmptyState>
      ) : null}

      {m && store ? (
        <>
          <div className="stats-grid stats-4">
            <Stat label="Consumed since init" value={fmtCompact(store.consumed_total)} sub={`generation ${fmtInt(store.generation)}`} />
            <Stat label={`Consumed · ${win}`} value={fmtCompact(sum(c.masks))} sub={rate ? `${fmtCompact(rate * 3600)} slots / hour` : "idle"} />
            <Stat label="Open reservations" value={fmtInt(store.open_reservations)} sub="in-flight requests" />
            <Stat label="Low-water mark" value={fmtCompact(store.low_water_slots)} sub="slots per model" />
          </div>

          <div className="mask-models">
            {models.map(([name, x]) => {
              const ovm = ov.data?.models.find((o) => o.name === name);
              const byModel = usage.data?.by_model?.[name]?.masks_consumed;
              const modelRate = typeof byModel === "number" && usage.data ? byModel / (usage.data.series.length * usage.data.bucket_s || 1) : null;
              const runway = ovm?.masks_runway_s ?? runwaySeconds(x.available_slots, modelRate);
              const t = runwayTone(runway);
              const isLow = x.available_slots !== null && x.low_water_slots !== null && x.available_slots < x.low_water_slots;
              const cap = Math.max(x.available_slots ?? 0, (x.low_water_slots ?? 0) * 4, 1);
              return (
                <Card key={name} className={name === gw.model ? "is-selected" : undefined}>
                  <div className="mask-card-head">
                    <div>
                      <div className="mask-card-name">{name}</div>
                      <div className="mask-card-sub mono">
                        {shortHash(x.fingerprint)} · {x.ring}
                      </div>
                    </div>
                    {x.available_slots === null ? (
                      <Badge tone="neutral" dot>
                        Not started
                      </Badge>
                    ) : isLow ? (
                      <Badge tone="warn" dot>
                        Below low water
                      </Badge>
                    ) : (
                      <Badge tone="ok" dot>
                        Healthy
                      </Badge>
                    )}
                  </div>
                  <div className="mask-card-num">
                    <span className="num big">{fmtCompact(x.available_slots)}</span>
                    <span className="muted">slots available</span>
                  </div>
                  <Meter value={x.available_slots ?? 0} max={cap} threshold={x.available_slots === null ? null : x.low_water_slots} tone={isLow ? "warn" : "accent"} label={`${name} masks available relative to low-water mark`} height={6} />
                  <DescList
                    className="mt-3"
                    items={[
                      ["Runway", <span className={t ? `num text-${t}` : "num"}>{runway === null || runway === undefined ? "idle (no recent use)" : `≈ ${fmtDuration(runway)}`}</span>],
                      ["Low-water mark", <span className="num">{fmtInt(x.low_water_slots)}</span>],
                    ]}
                  />
                </Card>
              );
            })}
          </div>

          <Card title="Consumption" description={`Mask slots consumed per ${fmtDuration(usage.data?.bucket_s ?? 3600)} bucket, all models`}>
            <AreaChart label="Mask slots consumed over time" x={c.t} series={[{ name: "Slots consumed", values: c.masks, tone: 1, area: true }]} format={fmtCompact} formatX={(t) => fmtAxisTime(t, usage.data?.bucket_s ?? 3600)} />
          </Card>

          <div className="grid-3">
            <Card title="Inventory" description="Per fingerprint and ring" flush className="span-2">
              <Table columns={invCols} rows={store.inventory ?? []} rowKey={(r) => `${r.fingerprint}:${r.ring}`} caption="Mask inventory" empty="No segments on disk" />
            </Card>
            <Card title="Store">
              <DescList
                items={[
                  ["Store id", <span className="mono">{shortHash(store.store_id, 16)}</span>],
                  ["Segments", store.segments_by_status ? Object.entries(store.segments_by_status).map(([k, v]) => `${v} ${k}`).join(" · ") : "—"],
                  ["Key provider", <span className="mono">{String(store.key_provider?.kind ?? store.key_provider?.type ?? "—")}</span>],
                  ["Journal", store.rollback_protection?.journal ? "on" : "off"],
                  ["Counter", `${store.rollback_protection?.counter ?? "—"}${store.rollback_protection?.hardware_counter ? " (hardware)" : ""}`],
                  ["Row MACs", store.rollback_protection?.row_macs ? "verified" : "—"],
                ]}
              />
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}
