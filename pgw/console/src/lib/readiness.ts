// Plain-language explanations of a model's "not ready" reason (gateway `ready()` reasons).

export interface Readiness {
  /** One short phrase for tables and badges. */
  short: string;
  /** A sentence for alerts: what is happening and what to do. */
  detail: string;
  /** "info" when the gateway is still getting ready by itself; "warn" when someone should act. */
  tone: "info" | "warn";
}

export function explainReason(reason: string | null | undefined): Readiness {
  const r = (reason || "").trim();
  if (r === "no mask slots")
    return {
      short: "preparing privacy masks",
      detail:
        "is preparing its first privacy masks. Every request uses fresh one-time masks; after setup the mask factory makes a first batch (a few minutes for a small model, longer for large ones). It becomes ready by itself; see Masks for progress.",
      tone: "info",
    };
  if (r === "waiting for the compute service")
    return { short: "connecting to provider", detail: "is connecting to the compute service. It retries by itself; check the network if this persists.", tone: "info" };
  if (r === "not started") return { short: "starting", detail: "is starting.", tone: "info" };
  if (r === "integrity breaker open")
    return {
      short: "integrity check failed",
      detail: "was stopped after a failed integrity check on the provider's results. Investigate, then reset it on the Models page.",
      tone: "warn",
    };
  if (r === "noise alarm")
    return { short: "noise alarm", detail: "is blocked: the outbound-noise monitor is alarming (fail-closed).", tone: "warn" };
  if (r === "noise monitor unavailable")
    return { short: "noise monitor unavailable", detail: "is blocked: the required outbound-noise monitor is unavailable.", tone: "warn" };
  if (r.startsWith("failed"))
    return { short: "failed", detail: `could not start (${r.replace(/^failed:\s*/, "")}).`, tone: "warn" };
  return { short: r || "not ready", detail: r ? `is not ready: ${r}.` : "is not ready.", tone: "warn" };
}
