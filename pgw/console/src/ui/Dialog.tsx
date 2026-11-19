import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { cx } from "../lib/cx";
import { Button } from "./Button";
import { IconX } from "./Icons";

/**
 * Modal dialog on the native <dialog> element: focus trapping, Esc to close, inert background and
 * top-layer rendering come from the browser (no portals, no injected styles).
 */
export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  variant = "modal",
  size = "md",
  dismissable = true,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  variant?: "modal" | "drawer";
  size?: "sm" | "md" | "lg";
  /** When false, Esc and backdrop clicks do nothing (e.g. while showing a one-time secret). */
  dismissable?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descId = useId();

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      if (typeof d.showModal === "function") d.showModal();
      else d.setAttribute("open", "");
    } else if (!open && d.open) {
      if (typeof d.close === "function") d.close();
      else d.removeAttribute("open");
    }
  }, [open]);

  return (
    <dialog
      ref={ref}
      className={cx("dialog", `dialog-${variant}`, `dialog-${size}`)}
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      onCancel={(e) => {
        e.preventDefault();
        if (dismissable) onClose();
      }}
      onMouseDown={(e) => {
        // Click on the backdrop (the dialog element itself, outside the panel).
        if (dismissable && e.target === ref.current) onClose();
      }}
    >
      {open ? (
        <div className="dialog-panel">
          <header className="dialog-head">
            <div>
              <h2 className="dialog-title" id={titleId}>
                {title}
              </h2>
              {description ? (
                <p className="dialog-desc" id={descId}>
                  {description}
                </p>
              ) : null}
            </div>
            {dismissable ? <Button variant="ghost" size="sm" iconOnly label="Close" icon={<IconX />} onClick={onClose} /> : null}
          </header>
          <div className="dialog-body">{children}</div>
          {footer ? <footer className="dialog-foot">{footer}</footer> : null}
        </div>
      ) : null}
    </dialog>
  );
}

/** Confirmation dialog for destructive or impactful admin actions. */
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  children,
  confirmLabel = "Confirm",
  tone = "danger",
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<unknown> | void;
  title: ReactNode;
  children?: ReactNode;
  confirmLabel?: string;
  tone?: "danger" | "primary";
}) {
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try {
      await onConfirm();
      onClose();
    } catch {
      // Errors are surfaced by the caller (toast); keep the dialog open.
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open={open}
      onClose={busy ? () => undefined : onClose}
      title={title}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant={tone} onClick={run} loading={busy} autoFocus>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="confirm-text">{children}</div>
    </Dialog>
  );
}
