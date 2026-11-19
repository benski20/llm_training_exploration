import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { cx } from "../lib/cx";
import { IconAlert, IconCheck, IconInfo, IconX } from "./Icons";

export interface ToastInput {
  title: string;
  description?: string;
  tone?: "neutral" | "ok" | "warn" | "bad";
  /** Toasts with the same key replace each other instead of stacking. */
  key?: string;
  durationMs?: number;
}

interface ToastItem extends Required<Pick<ToastInput, "title" | "tone">> {
  id: number;
  key: string;
  description?: string;
}

const Ctx = createContext<((t: ToastInput) => void) | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const seq = useRef(0);

  const dismiss = useCallback((id: number) => {
    setItems((xs) => xs.filter((x) => x.id !== id));
    const t = timers.current.get(id);
    if (t) clearTimeout(t);
    timers.current.delete(id);
  }, []);

  const push = useCallback(
    (t: ToastInput) => {
      const id = ++seq.current;
      const key = t.key ?? `${t.title}|${t.description ?? ""}`;
      setItems((xs) => [...xs.filter((x) => x.key !== key), { id, key, title: t.title, description: t.description, tone: t.tone ?? "neutral" }].slice(-4));
      const ms = t.durationMs ?? (t.tone === "bad" || t.tone === "warn" ? 7000 : 4000);
      timers.current.set(
        id,
        setTimeout(() => dismiss(id), ms),
      );
    },
    [dismiss],
  );

  const value = useMemo(() => push, [push]);

  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="region" aria-label="Notifications">
        <ol aria-live="polite" aria-relevant="additions">
          {items.map((t) => (
            <li key={t.id} className={cx("toast", `toast-${t.tone}`)} role={t.tone === "bad" ? "alert" : "status"}>
              <span className="toast-icon" aria-hidden="true">
                {t.tone === "ok" ? <IconCheck /> : t.tone === "neutral" ? <IconInfo /> : <IconAlert />}
              </span>
              <div className="toast-body">
                <div className="toast-title">{t.title}</div>
                {t.description ? <div className="toast-desc">{t.description}</div> : null}
              </div>
              <button type="button" className="toast-close" aria-label="Dismiss notification" onClick={() => dismiss(t.id)}>
                <IconX size={14} />
              </button>
            </li>
          ))}
        </ol>
      </div>
    </Ctx.Provider>
  );
}

export function useToast(): (t: ToastInput) => void {
  const v = useContext(Ctx);
  if (!v) throw new Error("useToast outside ToastProvider");
  return v;
}
