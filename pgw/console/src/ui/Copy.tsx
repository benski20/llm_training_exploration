import { useEffect, useState, type ReactNode } from "react";
import { cx } from "../lib/cx";
import { Button } from "./Button";
import { IconCheck, IconCopy } from "./Icons";

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for non-secure contexts: a temporary textarea + execCommand.
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.className = "sr-only";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

export function CopyButton({ text, label = "Copy", size = "sm", variant = "ghost", showLabel }: { text: string; label?: string; size?: "sm" | "md"; variant?: "ghost" | "secondary" | "primary"; showLabel?: boolean }) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setDone(false), 1600);
    return () => clearTimeout(t);
  }, [done]);
  return (
    <Button
      size={size}
      variant={variant}
      iconOnly={!showLabel}
      label={done ? "Copied" : label}
      icon={done ? <IconCheck /> : <IconCopy />}
      onClick={async () => setDone(await copyText(text))}
    >
      {showLabel ? (done ? "Copied" : label) : null}
    </Button>
  );
}

/** Monospace inline value with a copy affordance (IDs, fingerprints). */
export function Mono({ children, copy, className, title }: { children: ReactNode; copy?: string; className?: string; title?: string }) {
  return (
    <span className={cx("mono-val", className)} title={title}>
      <span className="mono">{children}</span>
      {copy ? <CopyButton text={copy} label="Copy to clipboard" /> : null}
    </span>
  );
}

export function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="code">
      <div className="code-bar">
        <span className="code-label">{label}</span>
        <CopyButton text={code} label={`Copy ${label}`} />
      </div>
      <pre className="code-pre" tabIndex={0} aria-label={label}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
