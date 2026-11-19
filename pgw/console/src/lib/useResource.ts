import { useCallback, useEffect, useRef, useState } from "react";

export interface Resource<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  /** Epoch ms of the last successful load. */
  updatedAt: number | null;
  reload: () => Promise<void>;
}

/**
 * Load `fetcher()` when `key` changes, optionally polling every `intervalMs` while the tab is
 * visible. Keeps the last good data on refresh errors so polling hiccups do not blank the page.
 */
export function useResource<T>(key: string | null, fetcher: () => Promise<T>, intervalMs?: number): Resource<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState<boolean>(key !== null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const fetchRef = useRef(fetcher);
  fetchRef.current = fetcher;
  const gen = useRef(0);

  const load = useCallback(async () => {
    const g = ++gen.current;
    setLoading(true);
    try {
      const d = await fetchRef.current();
      if (g !== gen.current) return;
      setData(d);
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      if (g !== gen.current) return;
      setError(e);
    } finally {
      if (g === gen.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (key === null) return;
    setData(undefined);
    setError(null);
    void load();
  }, [key, load]);

  useEffect(() => {
    if (key === null || !intervalMs) return;
    const t = setInterval(() => {
      if (typeof document === "undefined" || document.visibilityState === "visible") void load();
    }, intervalMs);
    return () => clearInterval(t);
  }, [key, intervalMs, load]);

  return { data, error, loading, updatedAt, reload: load };
}
