// Small inline-SVG charts. All geometry is expressed as SVG attributes (CSP-safe: no style attrs),
// colors come from CSS classes bound to design tokens so both themes work.

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";
import { cx } from "../lib/cx";

/** Track an element's content width (ResizeObserver; falls back to a fixed width in tests). */
export function useWidth<T extends HTMLElement>(fallback = 600): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const cw = Math.floor(entries[0].contentRect.width);
      if (cw > 0) setW(cw);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

type Pt = [number, number];

function linePath(pts: Pt[]): string {
  let d = "";
  let pen = false;
  for (const [x, y] of pts) {
    if (!Number.isFinite(y)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

function areaPath(pts: Pt[], base: number): string {
  const ok = pts.filter(([, y]) => Number.isFinite(y));
  if (ok.length < 2) return "";
  return `${linePath(ok)}L${ok[ok.length - 1][0].toFixed(1)},${base}L${ok[0][0].toFixed(1)},${base}Z`;
}

/** "Nice" upper bound and tick step for a max value. */
export function niceScale(max: number, ticks = 4): { max: number; step: number } {
  if (!(max > 0)) return { max: 1, step: 0.25 };
  const raw = max / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  return { max: Math.ceil(max / step - 1e-9) * step, step };
}

export function Sparkline({ values, height = 32, className, label }: { values: (number | null)[]; height?: number; className?: string; label?: string }) {
  const [ref, width] = useWidth<HTMLDivElement>(160);
  const { line, area } = useMemo(() => {
    const vs = values.map((v) => (v === null ? NaN : v));
    const max = Math.max(1e-9, ...vs.filter(Number.isFinite));
    const n = Math.max(1, vs.length - 1);
    const pts: Pt[] = vs.map((v, i) => [(i / n) * width, height - 2 - (v / max) * (height - 4)]);
    return { line: linePath(pts), area: areaPath(pts, height) };
  }, [values, width, height]);
  const gid = useId();
  return (
    <div ref={ref} className={cx("spark", className)}>
      <svg width={width} height={height} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
        <defs>
          <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" className="grad-stop-a" />
            <stop offset="1" className="grad-stop-b" />
          </linearGradient>
        </defs>
        {area ? <path d={area} fill={`url(#${gid})`} /> : null}
        <path d={line} className="spark-line" fill="none" />
      </svg>
    </div>
  );
}

export interface Series {
  name: string;
  values: (number | null)[];
  /** 1 = accent, 2 = neutral gray. */
  tone?: 1 | 2;
  /** Fill under the line (first series only, by convention). */
  area?: boolean;
}

export function AreaChart({
  x,
  series,
  height = 180,
  format,
  formatX,
  label,
  emptyText = "No data in this window",
}: {
  x: number[];
  series: Series[];
  height?: number;
  format: (v: number | null) => string;
  formatX: (x: number) => string;
  label: string;
  emptyText?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(560);
  const [hover, setHover] = useState<number | null>(null);
  const gid = useId();
  const padL = 44;
  const padR = 8;
  const padT = 8;
  const padB = 22;
  const iw = Math.max(10, width - padL - padR);
  const ih = height - padT - padB;
  const n = x.length;

  const geo = useMemo(() => {
    const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null && Number.isFinite(v));
    const { max, step } = niceScale(all.length ? Math.max(...all) : 0);
    const xi = (i: number) => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
    const yv = (v: number) => padT + ih - (v / max) * ih;
    const paths = series.map((s) => {
      const pts: Pt[] = s.values.map((v, i) => [xi(i), v === null ? NaN : yv(v)]);
      return { line: linePath(pts), area: s.area ? areaPath(pts, padT + ih) : "" };
    });
    const ticks: number[] = [];
    for (let v = 0; v <= max + 1e-9; v += step) ticks.push(v);
    const labelEvery = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 72))));
    return { max, ticks, xi, yv, paths, labelEvery, hasData: all.length > 0 };
  }, [series, n, iw, ih]);

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    const i = n <= 1 ? 0 : Math.round((px / r.width) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };
  const onKey = (e: KeyboardEvent) => {
    if (!n) return;
    if (e.key === "ArrowRight") setHover((h) => Math.min(n - 1, (h ?? n - 1) + 1));
    else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? n) - 1));
    else if (e.key === "Home") setHover(0);
    else if (e.key === "End") setHover(n - 1);
    else if (e.key === "Escape") setHover(null);
    else return;
    e.preventDefault();
  };

  const idx = hover ?? n - 1;
  const legend: ReactNode = (
    <div className="chart-legend" aria-live="polite">
      <span className="chart-legend-x">{n ? (hover === null ? "Latest" : formatX(x[idx])) : ""}</span>
      {series.map((s) => (
        <span className="chart-legend-item" key={s.name}>
          <span className={cx("legend-swatch", `tone-${s.tone ?? 1}`)} aria-hidden="true" />
          <span className="chart-legend-name">{s.name}</span>
          <span className="chart-legend-val">{n ? format(s.values[idx] ?? null) : "—"}</span>
        </span>
      ))}
    </div>
  );

  return (
    <div className="chart">
      {legend}
      <div ref={ref} className="chart-plot">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${label}. Use left and right arrow keys to inspect values.`}
          tabIndex={0}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
          className="chart-svg"
        >
          <defs>
            <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" className="grad-stop-a" />
              <stop offset="1" className="grad-stop-b" />
            </linearGradient>
          </defs>
          {geo.ticks.map((t) => (
            <g key={t}>
              <line x1={padL} x2={padL + iw} y1={geo.yv(t)} y2={geo.yv(t)} className={cx("chart-grid", t === 0 && "chart-base")} />
              <text x={padL - 8} y={geo.yv(t)} dy="0.32em" textAnchor="end" className="chart-tick">
                {format(t)}
              </text>
            </g>
          ))}
          {x.map((t, i) =>
            i % geo.labelEvery === 0 ? (
              <text key={t} x={geo.xi(i)} y={height - 6} textAnchor={i === 0 ? "start" : "middle"} className="chart-tick">
                {formatX(t)}
              </text>
            ) : null,
          )}
          {geo.paths.map((p, si) => (
            <g key={series[si].name}>
              {p.area ? <path d={p.area} fill={`url(#${gid})`} /> : null}
              <path d={p.line} fill="none" className={cx("chart-line", `tone-${series[si].tone ?? 1}`)} />
            </g>
          ))}
          {!geo.hasData ? (
            <text x={padL + iw / 2} y={padT + ih / 2} textAnchor="middle" className="chart-empty">
              {emptyText}
            </text>
          ) : null}
          {hover !== null && n ? (
            <g className="chart-hover">
              <line x1={geo.xi(hover)} x2={geo.xi(hover)} y1={padT} y2={padT + ih} className="chart-crosshair" />
              {series.map((s) => {
                const v = s.values[hover];
                return v === null || v === undefined ? null : (
                  <circle key={s.name} cx={geo.xi(hover)} cy={geo.yv(v)} r={4} className={cx("chart-dot", `tone-${s.tone ?? 1}`)} />
                );
              })}
            </g>
          ) : null}
          <rect x={padL} y={padT} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
      </div>
    </div>
  );
}

/** Horizontal meter (e.g. mask inventory vs. capacity) with an optional threshold marker. */
export function Meter({ value, max, threshold, tone = "accent", label, height = 6 }: { value: number; max: number; threshold?: number | null; tone?: "accent" | "ok" | "warn" | "bad"; label: string; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(200);
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const tx = threshold && max > 0 ? Math.max(0, Math.min(1, threshold / max)) * width : null;
  return (
    <div ref={ref} className="meter">
      <svg width={width} height={height + 4} role="meter" aria-label={label} aria-valuenow={value} aria-valuemin={0} aria-valuemax={max}>
        <rect x={0} y={2} width={width} height={height} rx={height / 2} className="meter-track" />
        <rect x={0} y={2} width={Math.max(frac > 0 ? height : 0, frac * width)} height={height} rx={height / 2} className={cx("meter-fill", `meter-${tone}`)} />
        {tx !== null ? <line x1={tx} x2={tx} y1={0} y2={height + 4} className="meter-threshold" /> : null}
      </svg>
    </div>
  );
}
