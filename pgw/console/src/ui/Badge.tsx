import type { ReactNode } from "react";
import { cx } from "../lib/cx";

export type Tone = "neutral" | "accent" | "ok" | "warn" | "bad";

export function Badge({ tone = "neutral", children, dot, className, title }: { tone?: Tone; children: ReactNode; dot?: boolean; className?: string; title?: string }) {
  return (
    <span className={cx("badge", `badge-${tone}`, className)} title={title}>
      {dot ? <span className={cx("dot", `dot-${tone}`)} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}

export function StatusDot({ tone, pulse, label }: { tone: Tone; pulse?: boolean; label?: string }) {
  return <span className={cx("dot", `dot-${tone}`, pulse && "dot-pulse")} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />;
}

/** Map a model state / readiness to a tone and label. */
export function modelTone(ready: boolean | undefined, state: string | undefined): Tone {
  if (ready) return "ok";
  const s = (state || "").toLowerCase();
  if (["failed", "error", "stopped", "breaker_open"].some((x) => s.includes(x))) return "bad";
  return "warn";
}

export function statusTone(code: number | undefined | null): Tone {
  if (!code) return "neutral";
  if (code < 300) return "ok";
  if (code === 429 || code === 503 || code === 504 || code < 500) return "warn";
  return "bad";
}
