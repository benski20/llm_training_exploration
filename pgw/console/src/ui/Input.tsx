import { forwardRef, useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { cx } from "../lib/cx";
import { IconChevronDown } from "./Icons";

export interface FieldProps {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  /** Render-prop receiving the generated id and aria-describedby for the control. */
  children: (ids: { id: string; describedBy: string | undefined; invalid: boolean }) => ReactNode;
  className?: string;
  aside?: ReactNode;
}

export function Field({ label, hint, error, children, className, aside }: FieldProps) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const errId = error ? `${id}-err` : undefined;
  const describedBy = [hintId, errId].filter(Boolean).join(" ") || undefined;
  return (
    <div className={cx("field", className)}>
      <div className="field-head">
        <label className="field-label" htmlFor={id}>
          {label}
        </label>
        {aside ? <span className="field-aside">{aside}</span> : null}
      </div>
      {children({ id, describedBy, invalid: Boolean(error) })}
      {hint && !error ? (
        <div className="field-hint" id={hintId}>
          {hint}
        </div>
      ) : null}
      {error ? (
        <div className="field-error" id={errId} role="alert">
          {error}
        </div>
      ) : null}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement> & { mono?: boolean }>(function Input(
  { className, mono, ...rest },
  ref,
) {
  return <input ref={ref} className={cx("input", mono && "mono", className)} {...rest} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  return <textarea ref={ref} className={cx("input textarea", className)} {...rest} />;
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement> & { size_?: "sm" | "md" }>(function Select(
  { className, children, size_ = "md", ...rest },
  ref,
) {
  return (
    <span className={cx("select-wrap", `select-${size_}`, className)}>
      <select ref={ref} className="input select" {...rest}>
        {children}
      </select>
      <IconChevronDown className="select-chevron" size={14} />
    </span>
  );
});

export function Checkbox({ label, hint, className, ...rest }: InputHTMLAttributes<HTMLInputElement> & { label: ReactNode; hint?: ReactNode }) {
  return (
    <label className={cx("check", className)}>
      <input type="checkbox" {...rest} />
      <span className="check-box" aria-hidden="true">
        <svg viewBox="0 0 12 12" width="10" height="10">
          <path d="m2.5 6.2 2.2 2.2 4.8-4.8" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
      <span className="check-text">
        <span className="check-label">{label}</span>
        {hint ? <span className="check-hint">{hint}</span> : null}
      </span>
    </label>
  );
}

export function Switch({ label, checked, onChange, disabled, id }: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; id?: string }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={cx("switch", checked && "is-on")}
      onClick={() => onChange(!checked)}
      disabled={disabled}
    >
      <span className="switch-thumb" />
    </button>
  );
}
