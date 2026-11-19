import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { AuthKind, MeResponse, Role, SessionUser } from "../api/types";

export interface SessionState {
  status: "loading" | "authed" | "anon";
  role: Role | null;
  version: string | null;
  expiresAt: string | number | null;
  /** How the user signed in (token or single sign-on). */
  auth: AuthKind | null;
  /** Single sign-on user (null for token sign-in). */
  user: SessionUser | null;
  /** Set when the session ended because the server answered 401. */
  expired: boolean;
}

interface SessionApi extends SessionState {
  login: (token: string) => Promise<void>;
  logout: () => Promise<void>;
  isAdmin: boolean;
}

const Ctx = createContext<SessionApi | null>(null);

const ANON = { role: null, version: null, expiresAt: null, auth: null, user: null, expired: false } as const;

function fromMe(m: MeResponse): SessionState {
  return {
    status: "authed",
    role: m.role,
    version: m.version ?? null,
    expiresAt: m.expires_at ?? null,
    auth: m.auth ?? "token",
    user: m.user ?? null,
    expired: false,
  };
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<SessionState>({ status: "loading", ...ANON });

  useEffect(() => {
    let alive = true;
    api
      .me()
      .then((m) => alive && setS(fromMe(m)))
      .catch(() => alive && setS((x) => ({ ...x, status: "anon", role: null, auth: null, user: null })));
    const unsub = api.subscribe((e) => {
      if (e.kind === "unauthorized") setS((x) => ({ ...x, status: "anon", role: null, auth: null, user: null, expired: x.status === "authed" }));
    });
    return () => {
      alive = false;
      unsub();
    };
  }, []);

  const login = useCallback(async (token: string) => {
    const r = await api.login(token);
    let next: SessionState = { status: "authed", ...ANON, role: r.role, auth: "token" };
    try {
      next = fromMe(await api.me());
    } catch {
      /* /me is informational here; login already succeeded */
    }
    setS(next);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setS({ status: "anon", ...ANON });
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
