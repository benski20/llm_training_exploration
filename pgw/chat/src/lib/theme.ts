export type ThemePref = "system" | "light" | "dark";

const KEY = "pgw-chat-theme";

export function loadTheme(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    // storage blocked
  }
  return "system";
}

export function applyTheme(p: ThemePref): void {
  const root = document.documentElement;
  if (p === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", p);
  try {
    localStorage.setItem(KEY, p);
  } catch {
    // storage blocked
  }
}
