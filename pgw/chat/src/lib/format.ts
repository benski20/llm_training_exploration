export function relativeTime(ts: number, now = Date.now() / 1000): string {
  const d = Math.max(0, now - ts);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  if (d < 7 * 86400) return `${Math.floor(d / 86400)} d ago`;
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Sidebar groups: "Today", "Previous 7 days", "Older". */
export function groupOf(ts: number, now = Date.now() / 1000): string {
  const d = now - ts;
  if (d < 86400) return "Today";
  if (d < 7 * 86400) return "Previous 7 days";
  return "Older";
}

export function duration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 90) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function tokensPerSecond(tps: number | null | undefined): string {
  if (tps === null || tps === undefined || !Number.isFinite(tps) || tps <= 0) return "";
  return `${tps < 10 ? tps.toFixed(1) : Math.round(tps)} tok/s`;
}

/** Live speed while streaming: tokens after the first over the time since the first. */
export function liveRate(tokens: number, firstAt: number | null, now: number): number | null {
  if (firstAt === null || tokens < 2) return null;
  const s = (now - firstAt) / 1000;
  return s > 0.2 ? (tokens - 1) / s : null;
}

export function gb(n: number | null | undefined): string {
  return n === null || n === undefined ? "?" : `${n} GB`;
}
