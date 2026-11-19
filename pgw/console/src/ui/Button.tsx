import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cx } from "../lib/cx";
import { Spinner } from "./States";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md";
  icon?: ReactNode;
  loading?: boolean;
  /** Icon-only button: children are hidden, `label` is the accessible name. */
  iconOnly?: boolean;
  label?: string;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, loading, iconOnly, label, className, children, disabled, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx("btn", `btn-${variant}`, `btn-${size}`, iconOnly && "btn-icon", loading && "is-loading", className)}
      disabled={disabled || loading}
      aria-label={iconOnly ? label : undefined}
      title={iconOnly ? label : rest.title}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size={14} /> : icon}
      {!iconOnly && children ? <span className="btn-label">{children}</span> : null}
    </button>
  );
});
