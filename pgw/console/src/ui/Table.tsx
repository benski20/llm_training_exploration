import type { KeyboardEvent, ReactNode } from "react";
import { cx } from "../lib/cx";
import { Skeleton } from "./States";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  align?: "left" | "right" | "center";
  mono?: boolean;
  /** Extra class on th/td (width helpers, hide-on-narrow). */
  className?: string;
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  rowLabel,
  loading,
  empty,
  caption,
  className,
  selectedKey,
}: {
  columns: Column<T>[];
  rows: T[] | undefined;
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  /** Accessible label for clickable rows. */
  rowLabel?: (row: T) => string;
  loading?: boolean;
  empty?: ReactNode;
  caption?: string;
  className?: string;
  selectedKey?: string | null;
}) {
  const onKey = (e: KeyboardEvent<HTMLTableRowElement>, row: T) => {
    if (onRowClick && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      onRowClick(row);
    }
  };
  return (
    <div className={cx("table-wrap", className)}>
      <table className="table">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={cx(c.align && `al-${c.align}`, c.className)}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading && !rows
            ? Array.from({ length: 5 }, (_, i) => (
                <tr key={`sk${i}`} className="row-skeleton">
                  {columns.map((c) => (
                    <td key={c.key} className={c.className}>
                      <Skeleton />
                    </td>
                  ))}
                </tr>
              ))
            : null}
          {rows && rows.length === 0 ? (
            <tr className="row-empty">
              <td colSpan={columns.length}>{empty ?? "No rows"}</td>
            </tr>
          ) : null}
          {rows?.map((row) => {
            const k = rowKey(row);
            return (
              <tr
                key={k}
                className={cx(onRowClick && "row-click", selectedKey === k && "is-selected")}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                onKeyDown={onRowClick ? (e) => onKey(e, row) : undefined}
                tabIndex={onRowClick ? 0 : undefined}
                aria-label={onRowClick && rowLabel ? rowLabel(row) : undefined}
              >
                {columns.map((c) => (
                  <td key={c.key} className={cx(c.align && `al-${c.align}`, c.mono && "mono", c.className)}>
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
