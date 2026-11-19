import type { ReactNode } from "react";
import { cx } from "../lib/cx";

export function Card({
  title,
  description,
  actions,
  children,
  className,
  flush,
  footer,
  as: Tag = "section",
  labelledBy,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  /** No body padding (tables, lists). */
  flush?: boolean;
  footer?: ReactNode;
  as?: "section" | "div" | "article";
  labelledBy?: string;
}) {
  return (
    <Tag className={cx("card", className)} aria-labelledby={labelledBy}>
      {title || actions ? (
        <header className="card-head">
          <div className="card-titles">
            {title ? (
              <h2 className="card-title" id={labelledBy}>
                {title}
              </h2>
            ) : null}
            {description ? <p className="card-desc">{description}</p> : null}
          </div>
          {actions ? <div className="card-actions">{actions}</div> : null}
        </header>
      ) : null}
      <div className={cx("card-body", flush && "card-body-flush")}>{children}</div>
      {footer ? <footer className="card-foot">{footer}</footer> : null}
    </Tag>
  );
}

export function PageHeader({ title, description, actions, meta }: { title: ReactNode; description?: ReactNode; actions?: ReactNode; meta?: ReactNode }) {
  return (
    <div className="page-head">
      <div className="page-titles">
        <h1 className="page-title">{title}</h1>
        {description ? <p className="page-desc">{description}</p> : null}
        {meta ? <div className="page-meta">{meta}</div> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}

/** Two-column key/value list. */
export function DescList({ items, className, cols = 1 }: { items: [ReactNode, ReactNode, string?][]; className?: string; cols?: 1 | 2 }) {
  return (
    <dl className={cx("desc", cols === 2 && "desc-2", className)}>
      {items.map(([k, v, cls], i) => (
        <div className={cx("desc-row", cls)} key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
