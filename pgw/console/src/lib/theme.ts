// Theme: follows prefers-color-scheme until the user picks one; the choice is stored in localStorage
// and applied as <html data-theme="light|dark"> (CSS tokens switch on that attribute).

import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "pgw-console-theme";

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function system(): Theme {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Apply the stored theme before React renders (called from main.tsx, no inline script needed). */
export function applyInitialTheme(): void {
  const t = stored();
  if (t) document.documentElement.dataset.theme = t;
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? system());
  const [explicit, setExplicit] = useState(() => stored() !== null);

  useEffect(() => {
    if (explicit) return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const on = () => setTheme(mq.matches ? "dark" : "light");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [explicit]);

  const toggle = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem(KEY, next);
      } catch {
        /* storage unavailable: theme still applies for this page */
      }
      return next;
    });
    setExplicit(true);
  }, []);

  return [theme, toggle];
}
