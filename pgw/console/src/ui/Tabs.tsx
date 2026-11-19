import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { cx } from "../lib/cx";

export interface TabItem<V extends string> {
  value: V;
  label: ReactNode;
}

/**
 * Tab list with roving focus (Arrow keys / Home / End). `variant="segmented"` renders a compact
 * segmented control for filters such as a time window; panels are rendered by the caller with
 * `tabPanelProps`.
 */
export function Tabs<V extends string>({
  items,
  value,
  onChange,
  label,
  variant = "underline",
  className,
  idBase,
}: {
  items: TabItem<V>[];
  value: V;
  onChange: (v: V) => void;
  label: string;
  variant?: "underline" | "segmented";
  className?: string;
  idBase?: string;
}) {
  const auto = useId();
  const base = idBase ?? auto;
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    let n = -1;
    if (e.key === "ArrowRight") n = (i + 1) % items.length;
    else if (e.key === "ArrowLeft") n = (i - 1 + items.length) % items.length;
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = items.length - 1;
    if (n >= 0) {
      e.preventDefault();
      refs.current[n]?.focus();
      onChange(items[n].value);
    }
  };
  return (
    <div role="tablist" aria-label={label} className={cx("tabs", `tabs-${variant}`, className)}>
      {items.map((it, i) => {
        const selected = it.value === value;
        return (
          <button
            key={it.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`${base}-tab-${it.value}`}
            aria-selected={selected}
            aria-controls={`${base}-panel-${it.value}`}
            tabIndex={selected ? 0 : -1}
            className={cx("tab", selected && "is-active")}
            onClick={() => onChange(it.value)}
            onKeyDown={(e) => onKey(e, i)}
          >
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

export function tabPanelProps(idBase: string, value: string) {
  return { role: "tabpanel", id: `${idBase}-panel-${value}`, "aria-labelledby": `${idBase}-tab-${value}`, tabIndex: 0 } as const;
}
