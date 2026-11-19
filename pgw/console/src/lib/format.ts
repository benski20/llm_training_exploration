// Formatting helpers. Every function accepts null/undefined and returns an em dash for "no data".

export const DASH = "—";

type Num = number | null | undefined;

const isNum = (n: Num): n is number => typeof n === "number" && Number.isFinite(n);

export function fmtInt(n: Num): string {
  if (!isNum(n)) return DASH;
  return Math.round(n).toLocaleString("en-US");
}

/** 950, 1.2k, 34k, 1.2M, 3.4B. */
export function fmtCompact(n: Num): string {
  if (!isNum(n)) return DASH;
  const a = Math.abs(n);
  const units: [number, string][] = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [v, s] of units) {
    if (a >= v) {
      const x = n / v;
      const digits = Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? 1 : 2;
      return `${trimZeros(x.toFixed(digits))}${s}`;
    }
  }
  return Number.isInteger(n) ? String(n) : trimZeros(n.toFixed(a < 10 ? 2 : 1));
}

function trimZeros(s: string): string {
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

export function fmtBytes(n: Num): string {
  if (!isNum(n)) return DASH;
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return i === 0 ? `${Math.round(v)} B` : `${trimZeros(v.toFixed(v >= 100 ? 0 : 1))} ${units[i]}`;
}

/** Milliseconds: "84 ms", "1.24 s". */
export function fmtMs(ms: Num): string {
  if (!isNum(ms)) return DASH;
  if (ms < 1) return `${trimZeros(ms.toFixed(2))} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${trimZeros((ms / 1000).toFixed(ms < 10_000 ? 2 : 1))} s`;
}

/** Seconds as a coarse human duration: "45s", "12m", "3h 20m", "6d 4h". */
export function fmtDuration(s: Num): string {
  if (!isNum(s)) return DASH;
  if (s < 0) s = 0;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return m % 60 ? `${h}h ${m % 60}m` : `${h}h`;
  const d = Math.floor(h / 24);
  return h % 24 ? `${d}d ${h % 24}h` : `${d}d`;
}

/** 0.0625 -> "6.25%", 1 -> "100%". */
export function fmtPercent(r: Num, digits = 1): string {
  if (!isNum(r)) return DASH;
  const p = r * 100;
  if (p === 0 || p === 100) return `${p}%`;
  return `${trimZeros(p.toFixed(p < 1 ? Math.max(digits, 2) : digits))}%`;
}

/** Audit timestamps are ISO strings; usage series uses unix seconds. Returns epoch ms or null. */
export function toEpochMs(ts: string | number | null | undefined): number | null {
  if (ts === null || ts === undefined || ts === "") return null;
  if (typeof ts === "number") return ts < 1e12 ? ts * 1000 : ts;
  const n = Number(ts);
  if (Number.isFinite(n)) return n < 1e12 ? n * 1000 : n;
  const d = Date.parse(ts);
  return Number.isNaN(d) ? null : d;
}

export function fmtRelative(ts: string | number | null | undefined, now: number = Date.now()): string {
  const t = toEpochMs(ts);
  if (t === null) return DASH;
  const diff = Math.round((now - t) / 1000);
  const future = diff < 0;
  const a = Math.abs(diff);
  let s: string;
  if (a < 45) return future ? "in a few seconds" : "just now";
  if (a < 3600) s = `${Math.round(a / 60)} min`;
  else if (a < 86400) s = `${Math.round(a / 3600)} h`;
  else if (a < 86400 * 45) s = `${Math.round(a / 86400)} d`;
  else return fmtDate(t);
  return future ? `in ${s}` : `${s} ago`;
}

const pad = (n: number) => String(n).padStart(2, "0");

/** "2026-11-19 14:03:22" in local time. */
export function fmtDateTime(ts: string | number | null | undefined): string {
  const t = toEpochMs(ts);
  if (t === null) return DASH;
  const d = new Date(t);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function fmtDate(ts: string | number | null | undefined): string {
  const t = toEpochMs(ts);
  if (t === null) return DASH;
  const d = new Date(t);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Axis label for a bucket start, granular enough for the window. */
export function fmtAxisTime(tSec: number, bucketS: number): string {
  const d = new Date(tSec * 1000);
  if (bucketS >= 3600 * 6) return `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** First `n` characters of a hash/fingerprint. */
export function shortHash(s: string | null | undefined, n = 12): string {
  if (!s) return DASH;
  return s.length > n ? s.slice(0, n) : s;
}

/** Key id prefix as shown in tables: "pgw-3fa9c2d1e07b". */
export function keyPrefix(keyId: string): string {
  return `pgw-${keyId}`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${fmtInt(n)} ${n === 1 ? one : many}`;
}
