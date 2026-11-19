// Shared gateway status (polled /admin/status) and the global model selection from the top bar.

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { ModelStatus, Status } from "../api/types";
import { useResource, type Resource } from "../lib/useResource";

const MODEL_KEY = "pgw-console-model";

interface GatewayApi {
  status: Resource<Status>;
  models: ModelStatus[];
  /** Selected model name, or null when the gateway has none yet. */
  model: string | null;
  setModel: (name: string) => void;
  selected: ModelStatus | null;
}

const Ctx = createContext<GatewayApi | null>(null);

function readStored(): string | null {
  try {
    return localStorage.getItem(MODEL_KEY);
  } catch {
    return null;
  }
}

export function GatewayProvider({ children }: { children: ReactNode }) {
  const status = useResource("status", () => api.status(), 15_000);
  const [pref, setPref] = useState<string | null>(readStored);
  const models = useMemo(() => status.data?.models ?? [], [status.data]);
  const model = models.find((m) => m.name === pref)?.name ?? models[0]?.name ?? null;
  const selected = models.find((m) => m.name === model) ?? null;

  const setModel = useCallback((name: string) => {
    setPref(name);
    try {
      localStorage.setItem(MODEL_KEY, name);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(() => ({ status, models, model, setModel, selected }), [status, models, model, setModel, selected]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useGateway(): GatewayApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useGateway outside GatewayProvider");
  return v;
}
