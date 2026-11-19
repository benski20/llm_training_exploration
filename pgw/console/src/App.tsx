import { useEffect, type ReactNode } from "react";
import { BrowserRouter, HashRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api/client";
import { GatewayProvider } from "./app/gateway";
import { SessionProvider, useSession } from "./app/session";
import { Shell } from "./layout/Shell";
import { describeError } from "./lib/errors";
import { AuditPage } from "./pages/Audit";
import { KeysPage } from "./pages/Keys";
import { LoginPage } from "./pages/Login";
import { MasksPage } from "./pages/Masks";
import { ModelsPage } from "./pages/Models";
import { OverviewPage } from "./pages/Overview";
import { PlaygroundPage } from "./pages/Playground";
import { SecurityPage } from "./pages/Security";
import { Loading, ToastProvider, useToast } from "./ui";

/** Global API error -> toast (403 viewer, 429/503 retry hints, network). Polling repeats are de-duplicated by key. */
function ApiToasts() {
  const toast = useToast();
  useEffect(
    () =>
      api.subscribe((e) => {
        if (e.kind !== "error") return;
        const err = e.error;
        const mutating = e.method !== "GET";
        if (err.status === 403 || err.retryable || mutating) {
          const d = describeError(err);
          toast({ title: d.title, description: d.detail, tone: err.retryable ? "warn" : "bad", key: `${err.status}:${err.code}:${e.method}` });
        }
      }),
    [toast],
  );
  return null;
}

function RequireAuth({ children }: { children: ReactNode }) {
  const s = useSession();
  const loc = useLocation();
  if (s.status === "loading") return <div className="boot"><Loading label="Connecting to gateway" /></div>;
  if (s.status === "anon") return <Navigate to="/login" replace state={{ from: loc.pathname + loc.search }} />;
  return <>{children}</>;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <GatewayProvider>
              <Shell />
            </GatewayProvider>
          </RequireAuth>
        }
      >
        <Route index element={<OverviewPage />} />
        <Route path="playground" element={<PlaygroundPage />} />
        <Route path="keys" element={<KeysPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="masks" element={<MasksPage />} />
        <Route path="audit" element={<AuditPage />} />
        <Route path="security" element={<SecurityPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <Router>
      <ToastProvider>
        <SessionProvider>
          <ApiToasts />
          <AppRoutes />
        </SessionProvider>
      </ToastProvider>
    </Router>
  );
}

// The gateway serves the console at /console/ (history routing); the standalone demo build uses hash
// routing so it works from any URL.
function Router({ children }: { children: ReactNode }) {
  return __DEMO__ ? <HashRouter>{children}</HashRouter> : <BrowserRouter basename="/console">{children}</BrowserRouter>;
}
