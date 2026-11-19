import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { Role } from "../api/types";

export interface SessionState {
  status: "loading" | "authed" | "anon";
  role: Role | null;
  version: string | null;
  expiresAt: string | number | null;
  /** Set when the session ended because the server answered 401. */
  expired: boolean;
}

interface SessionApi extends SessionState {
  login: (token: string) => Promise<void>;
  logout: () => Promise<void>;
  isAdmin: boolean;
}

const Ctx = createContext<SessionApi | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<SessionState>({ status: "loading", role: null, version: null, expiresAt: null, expired: false });

  useEffect(() => {
    let alive = true;
    api
      .me()
      .then((m) => alive && setS({ status: "authed", role: m.role, version: m.version ?? null, expiresAt: m.expires_at ?? null, expired: false }))
      .catch(() => alive && setS((x) => ({ ...x, status: "anon", role: null })));
    const unsub = api.subscribe((e) => {
      if (e.kind === "unauthorized") setS((x) => ({ ...x, status: "anon", role: null, expired: x.status === "authed" }));
    });
    return () => {
      alive = false;
      unsub();
    };
  }, []);

  const login = useCallback(async (token: string) => {
    const r = await api.login(token);
    let version: string | null = null;
    let expiresAt: string | number | null = null;
    try {
      const m = await api.me();
      version = m.version ?? null;
      expiresAt = m.expires_at ?? null;
    } catch {
      /* /me is informational here; login already succeeded */
    }
    setS({ status: "authed", role: r.role, version, expiresAt, expired: false });
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setS({ status: "anon", role: null, version: null, expiresAt: null, expired: false });
    }
  }, []);

  const value = useMemo<SessionApi>(() => ({ ...s, login, logout, isAdmin: s.role === "admin" }), [s, login, logout]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession(): SessionApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSession outside SessionProvider");
  return v;
}
