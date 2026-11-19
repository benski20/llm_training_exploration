import type { Usage } from "../api/types";

/** Column-oriented view of a usage series for charts. */
export function usageColumns(u: Usage | undefined) {
  const s = u?.series ?? [];
  return {
    t: s.map((p) => p.t),
    requests: s.map((p) => p.requests),
    errors: s.map((p) => p.errors),
    prompt: s.map((p) => p.prompt_tokens),
    completion: s.map((p) => p.completion_tokens),
    tokens: s.map((p) => p.prompt_tokens + p.completion_tokens),
    p50: s.map((p) => p.p50_ms),
    p95: s.map((p) => p.p95_ms),
    ttft: s.map((p) => p.ttft_p50_ms),
    masks: s.map((p) => p.masks_consumed),
  };
}

/** Slots consumed per second over the trailing part of the series (for runway estimates). */
export function consumptionRate(u: Usage | undefined, trailingBuckets = 12): number | null {
  if (!u || !u.series.length || !u.bucket_s) return null;
  const tail = u.series.slice(-trailingBuckets);
  const total = tail.reduce((a, p) => a + (p.masks_consumed || 0), 0);
  return total / (tail.length * u.bucket_s);
}

/** Seconds until `available` slots run out at `ratePerS`; null when idle or unknown. */
export function runwaySeconds(available: number | null | undefined, ratePerS: number | null): number | null {
  if (available === null || available === undefined || !ratePerS || ratePerS <= 0) return null;
  return available / ratePerS;
}

export function sum(xs: (number | null | undefined)[]): number {
  return xs.reduce<number>((a, b) => a + (b ?? 0), 0);
}
