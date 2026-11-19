import type { ReactNode } from "react";
import { cx } from "../lib/cx";
import { IconAlert } from "./Icons";
import { Button } from "./Button";

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return (
    <svg className="spinner" width={size} height={size} viewBox="0 0 16 16" role={label ? "status" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <circle cx="8" cy="8" r="6.25" fill="none" strokeWidth="1.5" className="spinner-track" />
      <path d="M14.25 8A6.25 6.25 0 0 0 8 1.75" fill="none" strokeWidth="1.5" strokeLinecap="round" className="spinner-head" />
    </svg>
  );
}

export function Skeleton({ className, lines = 1 }: { className?: string; lines?: number }) {
  return (
    <span className={cx("skeleton-group", className)} aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="skeleton" />
      ))}
    </span>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="state state-loading" role="status" aria-live="polite">
      <Spinner size={18} />
      <span>{label}…</span>
    </div>
  );
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="state state-empty">
      {icon ? <div className="state-icon">{icon}</div> : null}
      <div className="state-title">{title}</div>
      {children ? <div className="state-text">{children}</div> : null}
      {action ? <div className="state-action">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const msg = error instanceof Error ? error.message : String(error);
  const status = (error as { status?: number })?.status;
  return (
    <div className={cx("state state-error", compact && "state-compact")} role="alert">
      <div className="state-icon state-icon-bad">
        <IconAlert />
      </div>
      <div className="state-title">{status === 403 ? "Not permitted" : "Could not load this data"}</div>
      <div className="state-text">{msg}</div>
      {onRetry ? (
        <div className="state-action">
          <Button size="sm" onClick={onRetry}>
            Try again
          </Button>
        </div>
      ) : null}
    </div>
  );
}

export function Callout({ tone = "neutral", title, children, icon, className }: { tone?: "neutral" | "warn" | "bad" | "ok" | "accent"; title?: ReactNode; children?: ReactNode; icon?: ReactNode; className?: string }) {
  return (
    <div className={cx("callout", `callout-${tone}`, className)} role={tone === "bad" || tone === "warn" ? "alert" : undefined}>
      {icon ? <span className="callout-icon">{icon}</span> : null}
      <div className="callout-body">
        {title ? <div className="callout-title">{title}</div> : null}
        {children ? <div className="callout-text">{children}</div> : null}
      </div>
    </div>
  );
}
