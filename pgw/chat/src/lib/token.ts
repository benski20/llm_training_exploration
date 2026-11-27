// The per-launch API token. `pgw chat` opens `http://127.0.0.1:<port>/#token=<token>`: the fragment is never sent
// to a server (not in requests, logs or Referer). We move it to sessionStorage (this tab only) and strip it from the
// address bar so it is not bookmarked or shared by accident.

const KEY = "pgw-chat-token";

export function takeTokenFromLocation(loc: Pick<Location, "hash" | "pathname" | "search">, hist?: History): string | null {
  const m = /(?:^#|&)token=([A-Za-z0-9_-]{32,128})(?:&|$)/.exec(loc.hash);
  if (m) {
    try {
      sessionStorage.setItem(KEY, m[1]);
    } catch {
      // storage blocked: the token still works for this page view
    }
    hist?.replaceState(null, "", loc.pathname + loc.search);
    return m[1];
  }
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function forgetToken(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
