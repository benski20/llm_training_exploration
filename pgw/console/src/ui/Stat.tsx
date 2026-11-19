import type { ReactNode } from "react";
import { cx } from "../lib/cx";
import { Skeleton } from "./States";

export function Stat({
  label,
  value,
  sub,
  tone,
  chart,
  loading,
  icon,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "ok" | "warn" | "bad";
  chart?: ReactNode;
  loading?: boolean;
  icon?: ReactNode;
}) {
  return (
    <div className={cx("stat", tone && `stat-${tone}`)}>
      <div className="stat-label">
        {icon ? <span className="stat-icon">{icon}</span> : null}
        {label}
      </div>
      <div className="stat-value">{loading ? <Skeleton className="sk-value" /> : value}</div>
      {sub ? <div className="stat-sub">{loading ? <Skeleton className="sk-sub" /> : sub}</div> : null}
      {chart ? <div className="stat-chart">{chart}</div> : null}
    </div>
  );
}
