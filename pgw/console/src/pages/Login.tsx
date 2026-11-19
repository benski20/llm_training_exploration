import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { useSession } from "../app/session";
import { useTheme } from "../lib/theme";
import { Button, Callout, Field, IconEye, IconEyeOff, IconLock, IconMoon, IconSun, Input, Loading, Logo } from "../ui";

export function LoginPage() {
  const s = useSession();
  const nav = useNavigate();
  const loc = useLocation();
  const [theme, toggleTheme] = useTheme();
  const [token, setToken] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const from = (loc.state as { from?: string } | null)?.from ?? "/";

  if (s.status === "loading") return <div className="boot"><Loading label="Connecting to gateway" /></div>;
  if (s.status === "authed") return <Navigate to={from} replace />;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!token.trim()) {
      setError("Enter an admin or viewer token.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await s.login(token.trim());
      setToken("");
      nav(from, { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setError("That token was not accepted.");
      else if (err instanceof ApiError && err.status === 429) setError(`Too many sign-in attempts.${err.retryAfter ? ` Try again in ${err.retryAfter}s.` : " Try again shortly."}`);
      else setError(err instanceof Error ? err.message : "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <div className="login-theme">
        <Button variant="ghost" size="sm" iconOnly label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} icon={theme === "dark" ? <IconSun /> : <IconMoon />} onClick={toggleTheme} />
      </div>
      <main className="login-card">
        <div className="login-brand">
          <Logo size={36} />
        </div>
        <h1 className="login-title">Sign in to PGW Console</h1>
        <p className="login-sub">Operate your private inference gateway.</p>
        {s.expired ? (
          <Callout tone="warn" className="login-callout">
            Your session expired. Sign in again to continue.
          </Callout>
        ) : null}
        <form onSubmit={submit} className="login-form" noValidate>
          <Field label="Access token" error={error} hint="The gateway admin token, or a viewer token for read-only access.">
            {({ id, describedBy, invalid }) => (
              <div className="input-affix">
                <Input
                  id={id}
                  type={show ? "text" : "password"}
                  autoComplete="current-password"
                  spellCheck={false}
                  mono
                  autoFocus
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  aria-describedby={describedBy}
                  aria-invalid={invalid || undefined}
                  placeholder="••••••••••••••••"
                />
                <Button variant="ghost" size="sm" iconOnly label={show ? "Hide token" : "Show token"} icon={show ? <IconEyeOff /> : <IconEye />} onClick={() => setShow((v) => !v)} className="input-affix-btn" />
              </div>
            )}
          </Field>
          <Button type="submit" variant="primary" loading={busy} className="login-submit">
            Sign in
          </Button>
        </form>
        <p className="login-foot">
          <IconLock size={13} />
          The token is exchanged for an HttpOnly session cookie and is never stored by the browser.
        </p>
      </main>
    </div>
  );
}
