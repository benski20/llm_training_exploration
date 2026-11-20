import { useCallback, useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { ApiError } from "../api/client";
import { codeProblem, setupApi, type SetupApi, type SetupCheck, type SetupState, type SSOInput } from "../api/setup";
import { cx } from "../lib/cx";
import { fmtBytes, fmtInt } from "../lib/format";
import { Badge, Button, Callout, Card, Checkbox, CodeBlock, Field, IconAlert, IconCheck, IconInfo, IconLock, Input, Loading, Logo, Meter, Spinner, Textarea } from "../ui";

type StepId = "enroll" | "model" | "maskstore" | "admin" | "verify" | "finish";

const STEPS: { id: StepId; title: string; blurb: string }[] = [
  { id: "enroll", title: "Connect to your provider", blurb: "Paste your enrollment code" },
  { id: "model", title: "Choose a model", blurb: "Download and prepare it" },
  { id: "maskstore", title: "Protect the masks", blurb: "Storage and encryption key" },
  { id: "admin", title: "Admin access", blurb: "Your sign-in for the console" },
  { id: "verify", title: "Check everything", blurb: "Run the safety checks" },
  { id: "finish", title: "Finish", blurb: "Start serving" },
];

function message(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return err instanceof Error ? err.message : "Something went wrong.";
}

function stepDone(st: SetupState, id: StepId): boolean {
  switch (id) {
    case "enroll":
      return Boolean(st.enroll);
    case "model":
      return st.model?.pack_status === "ready";
    case "maskstore":
      return Boolean(st.maskstore);
    case "admin":
      return Boolean(st.admin);
    case "verify":
      return Boolean(st.verify?.ok);
    case "finish":
      return st.finished;
  }
}

/** First-run setup: sign in with the setup token, then the guided wizard. */
export function SetupPage({ client = setupApi }: { client?: SetupApi }) {
  const [phase, setPhase] = useState<"loading" | "anon" | "authed">("loading");

  useEffect(() => {
    let alive = true;
    void client.resume().then((ok) => alive && setPhase(ok ? "authed" : "anon"));
    return () => {
      alive = false;
    };
  }, [client]);

  if (phase === "loading") return <div className="boot"><Loading label="Connecting to gateway" /></div>;
  if (phase === "anon") return <SetupLogin client={client} onDone={() => setPhase("authed")} />;
  return <SetupWizard client={client} />;
}

function SetupLogin({ client, onDone }: { client: SetupApi; onDone: () => void }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!token.trim()) {
      setError("Enter the setup token from the gateway's log.");
      return;
    }
    if (/^\s*PGW1/i.test(token)) {
      setError("That is the enrollment code from your provider; you paste it in the next step. Here, enter the setup token the gateway printed when it started.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await client.login(token.trim());
      onDone();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setError("That setup token was not accepted. Copy it again from the gateway's log.");
      else if (err instanceof ApiError && err.status === 429) setError("Too many attempts. Wait a minute and try again.");
      else setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <main className="login-card">
        <div className="login-brand">
          <Logo size={36} />
        </div>
        <h1 className="login-title">Set up your PGW gateway</h1>
        <p className="login-sub">This takes about ten minutes plus the model download. You will need the enrollment code from your provider.</p>
        <form onSubmit={submit} className="login-form" noValidate>
          <Field label="Setup token" error={error} hint="Printed by the gateway when it started (look for “setup token” in its log or terminal).">
            {({ id, describedBy, invalid }) => (
              <Input id={id} type="password" mono autoFocus autoComplete="off" spellCheck={false} value={token} onChange={(e) => setToken(e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} placeholder="pgw-setup-…" />
            )}
          </Field>
          <Button type="submit" variant="primary" loading={busy} className="login-submit">
            Start setup
          </Button>
        </form>
        <p className="login-foot">
          <IconLock size={13} />
          The token proves you have access to the gateway machine. It is exchanged for a private session cookie.
        </p>
      </main>
    </div>
  );
}

function SetupWizard({ client }: { client: SetupApi }) {
  const [st, setSt] = useState<SetupState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [current, setCurrent] = useState<StepId | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await client.state();
      setSt(s);
      setLoadError(null);
      return s;
    } catch (err) {
      setLoadError(message(err));
      return null;
    }
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const busy = Boolean(
    st && ((st.model?.pack_status === "building") || (st.jobs.hint && !["done", "failed"].includes(st.jobs.hint.phase)) || (st.jobs.factory?.running && !st.masks)),
  );
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => void refresh(), 1500);
    return () => clearInterval(t);
  }, [busy, refresh]);

  const firstOpen = useMemo<StepId>(() => (st ? (STEPS.find((s) => !stepDone(st, s.id))?.id ?? "finish") : "enroll"), [st]);
  // Stay on the step the admin is looking at while background work updates the state; move on only
  // when they continue (the first open step is chosen once, when the wizard loads).
  useEffect(() => {
    if (st && current === null) setCurrent(firstOpen);
  }, [st, current, firstOpen]);
  const active = current ?? firstOpen;

  if (!st) return <div className="boot">{loadError ? <Callout tone="bad" title="Could not load the setup state">{loadError}</Callout> : <Loading label="Loading setup" />}</div>;

  const reachable = (id: StepId) => {
    const i = STEPS.findIndex((s) => s.id === id);
    return STEPS.slice(0, i).every((s) => stepDone(st, s.id));
  };
  const next = () => {
    const i = STEPS.findIndex((s) => s.id === active);
    setCurrent(STEPS[Math.min(i + 1, STEPS.length - 1)].id);
  };

  return (
    <div className="setup">
      <header className="setup-head">
        <Logo size={28} />
        <div>
          <h1 className="setup-title">Gateway setup</h1>
          <p className="setup-sub">Private inference on open models. Your prompts and answers never leave this gateway unprotected.</p>
        </div>
      </header>
      <div className="setup-body">
        <nav className="setup-steps" aria-label="Setup steps">
          <ol>
            {STEPS.map((s, i) => {
              const done = stepDone(st, s.id);
              const can = reachable(s.id);
              return (
                <li key={s.id}>
                  <button type="button" className={cx("setup-step", active === s.id && "is-active", done && "is-done")} onClick={() => setCurrent(s.id)} disabled={!can} aria-current={active === s.id ? "step" : undefined}>
                    <span className="step-n" aria-hidden="true">{done ? <IconCheck size={12} /> : i + 1}</span>
                    <span className="setup-step-text">
                      <span className="setup-step-title">{s.title}</span>
                      <span className="setup-step-blurb">{s.blurb}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
        <main className="setup-main">
          {active === "enroll" ? <EnrollStep st={st} client={client} onDone={async () => { await refresh(); next(); }} /> : null}
          {active === "model" ? <ModelStep st={st} client={client} refresh={refresh} onNext={next} /> : null}
          {active === "maskstore" ? <MaskStoreStep st={st} client={client} onDone={async () => { await refresh(); next(); }} /> : null}
          {active === "admin" ? <AdminStep st={st} client={client} onDone={async () => { await refresh(); next(); }} /> : null}
          {active === "verify" ? <VerifyStep st={st} client={client} refresh={refresh} onNext={next} /> : null}
          {active === "finish" ? <FinishStep st={st} client={client} /> : null}
        </main>
      </div>
    </div>
  );
}

function StepCard({ title, description, children, footer }: { title: string; description: ReactNode; children: ReactNode; footer?: ReactNode }) {
  return (
    <Card title={title} description={description} footer={footer} className="setup-card">
      {children}
    </Card>
  );
}

// ---- 1. enrollment ---------------------------------------------------------------------------------
function EnrollStep({ st, client, onDone }: { st: SetupState; client: SetupApi; onDone: () => Promise<void> }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const p = codeProblem(code);
    if (p) {
      setError(p);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await client.enroll(code);
      setCode("");
      await onDone();
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <StepCard title="Connect to your provider" description="Your provider sent you a one-time enrollment code. It tells this gateway where the compute service is and how to recognise it.">
      {st.enroll ? (
        <Callout tone="ok" title="Connected" icon={<IconCheck />} className="setup-gap">
          This gateway belongs to <strong>{st.enroll.tenant}</strong> at <span className="mono">{st.enroll.service_url}</span>.
          {st.enroll.ca_fingerprint ? " The service's identity was checked against the code before anything was sent." : null}
          {" "}You can paste a new code below to replace this connection.
        </Callout>
      ) : null}
      <form onSubmit={submit} className="setup-form" noValidate>
        <Field label="Enrollment code" error={error} hint="It starts with PGW1-. Paste it exactly as you received it; spaces and line breaks are fine.">
          {({ id, describedBy, invalid }) => (
            <Textarea id={id} className="mono setup-code" rows={4} spellCheck={false} autoComplete="off" value={code} onChange={(e) => setCode(e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} placeholder="PGW1-AEXD4-HHS2J-…" />
          )}
        </Field>
        <div className="setup-actions">
          <Button type="submit" variant="primary" loading={busy}>
            Connect
          </Button>
        </div>
      </form>
    </StepCard>
  );
}

// ---- 2. model -----------------------------------------------------------------------------------------
const PHASE_TEXT: Record<string, string> = {
  queued: "Waiting to start",
  downloading: "Downloading the open-weight model",
  packing: "Preparing the model for private inference",
  verifying: "Checking that it is exactly the model the service runs",
  building: "Building",
  done: "Ready",
  failed: "Failed",
};

function ModelStep({ st, client, refresh, onNext }: { st: SetupState; client: SetupApi; refresh: () => Promise<SetupState | null>; onNext: () => void }) {
  const models = st.enroll?.models ?? [];
  const [choice, setChoice] = useState<string>(st.model?.name ?? models[0]?.name ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const job = st.jobs.pack;
  const status = st.model?.pack_status;

  const start = async () => {
    if (!choice) {
      setError("Choose a model first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await client.chooseModel(choice);
      await refresh();
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <StepCard
      title="Choose a model"
      description="The gateway downloads the public model weights and prepares them locally. Only models your provider runs are listed."
      footer={status === "ready" ? <Button variant="primary" onClick={onNext}>Continue</Button> : null}
    >
      <fieldset className="setup-choices" disabled={status === "building"}>
        <legend className="sr-only">Model</legend>
        {models.map((m) => (
          <label key={m.name} className={cx("setup-choice", choice === m.name && "is-selected")}>
            <input type="radio" name="model" value={m.name} checked={choice === m.name} onChange={() => setChoice(m.name)} />
            <span className="setup-choice-text">
              <span className="setup-choice-title">{m.name}</span>
              <span className="setup-choice-hint">{m.hf_repo ? `From ${m.hf_repo}` : "Source not stated by the provider"}</span>
            </span>
          </label>
        ))}
      </fieldset>
      {status === "building" || (job && job.phase !== "done" && job.phase !== "failed") ? (
        <div className="setup-progress" aria-live="polite">
          <div className="setup-progress-head">
            <Spinner size={14} />
            <span>{PHASE_TEXT[job?.phase ?? "queued"]}</span>
            {job && job.total > 0 && job.phase === "packing" ? <span className="setup-muted">{fmtInt(job.done)} of {fmtInt(job.total)} parts</span> : null}
          </div>
          <Meter value={job && job.total > 0 && job.phase === "packing" ? job.done : 0} max={job && job.total > 0 ? job.total : 1} label="Model preparation progress" />
          <p className="setup-muted">Large models take a while to download (about 15 GB for a 7B model). You can leave this page open; the gateway keeps working.</p>
        </div>
      ) : null}
      {status === "ready" ? (
        <Callout tone="ok" title={`${st.model?.name} is ready`} icon={<IconCheck />} className="setup-gap">
          Its fingerprint matches the one your provider serves, so both sides compute with exactly the same weights.
        </Callout>
      ) : null}
      {status === "failed" || status === "interrupted" ? (
        <Callout tone="bad" title="The model could not be prepared" icon={<IconAlert />} className="setup-gap">
          {st.model?.error || job?.error || "The gateway was restarted while preparing the model."} Try again, or contact your provider if it keeps failing.
        </Callout>
      ) : null}
      {error ? <Callout tone="bad" className="setup-gap">{error}</Callout> : null}
      {status !== "ready" && status !== "building" ? (
        <div className="setup-actions">
          <Button variant="primary" loading={busy} onClick={start} disabled={!models.length}>
            {status === "failed" || status === "interrupted" ? "Try again" : "Download and prepare"}
          </Button>
        </div>
      ) : null}
    </StepCard>
  );
}

// ---- 3. mask store --------------------------------------------------------------------------------------
function MaskStoreStep({ st, client, onDone }: { st: SetupState; client: SetupApi; onDone: () => Promise<void> }) {
  const [root, setRoot] = useState(st.maskstore?.root ?? `${st.paths.data_dir}/masks`);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const est = st.estimate;
  const total = est.total_bytes ?? est.hint_bytes;
  const lowDisk = est.disk_free_bytes !== null && total !== null && est.disk_free_bytes < total;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!root.trim().startsWith("/")) {
      setError("Enter a full folder path, for example /var/lib/pgw/masks.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await client.maskstore(root.trim());
      await onDone();
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <StepCard title="Protect the masks" description="Every request is hidden with one-time random masks. The gateway makes them ahead of time and keeps them encrypted on this machine.">
      <div className="setup-estimate">
        <div>
          <div className="setup-estimate-n">{fmtBytes(total)}</div>
          <div className="setup-muted">{est.basis === "model" ? "disk space needed for this model" : "typical for a 7B model (about 34 GB)"}</div>
        </div>
        <ul className="setup-list">
          <li>One-time encryption hint: {fmtBytes(est.hint_bytes)}</li>
          {est.buffer_bytes !== null ? <li>Ready-made masks for {fmtInt(est.buffer_slots)} tokens: {fmtBytes(est.buffer_bytes)}</li> : null}
          <li>Free space on the data disk: {fmtBytes(est.disk_free_bytes)}</li>
        </ul>
      </div>
      {lowDisk ? <Callout tone="warn" className="setup-gap">There may not be enough free space. Choose a larger disk below or free some space first.</Callout> : null}
      {st.maskstore ? (
        <Callout tone="ok" title="Mask store created" icon={<IconCheck />} className="setup-gap">
          Masks are kept in <span className="mono">{st.maskstore.root}</span>, encrypted with a key that only this gateway can read.
        </Callout>
      ) : null}
      <form onSubmit={submit} className="setup-form" noValidate>
        <Field label="Where to keep the masks" error={error} hint="A local disk of this machine. Never use a network share, and never restore this folder from a backup.">
          {({ id, describedBy, invalid }) => <Input id={id} mono value={root} onChange={(e) => setRoot(e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
        </Field>
        <fieldset className="setup-choices">
          <legend className="field-label">Encryption key</legend>
          <label className="setup-choice is-selected">
            <input type="radio" name="kms" value="file" checked readOnly />
            <span className="setup-choice-text">
              <span className="setup-choice-title">Key file on this machine</span>
              <span className="setup-choice-hint">Created now, readable only by the gateway's user, kept apart from the masks.</span>
            </span>
          </label>
          {st.kms_options.map((k) => (
            <label key={k.id} className="setup-choice is-disabled">
              <input type="radio" name="kms" value={k.id} disabled />
              <span className="setup-choice-text">
                <span className="setup-choice-title">
                  {k.label} <Badge>Coming soon</Badge>
                </span>
                <span className="setup-choice-hint">Keep the key in your cloud key service.</span>
              </span>
            </label>
          ))}
        </fieldset>
        <div className="setup-actions">
          <Button type="submit" variant="primary" loading={busy}>
            {st.maskstore ? "Create again" : "Create mask store"}
          </Button>
        </div>
      </form>
    </StepCard>
  );
}

// ---- 4. admin access ------------------------------------------------------------------------------------
const httpsUrl = (v: string) => /^https:\/\/[^\s/?#]+[^\s?#]*$/.test(v.trim());

function AdminStep({ st, client, onDone }: { st: SetupState; client: SetupApi; onDone: () => Promise<void> }) {
  const [token, setToken] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [useSSO, setUseSSO] = useState(false);
  const [sso, setSSO] = useState<SSOInput>({ issuer: "", client_id: "", client_secret: "", redirect_url: `${window.location.origin}/console/api/oidc/callback`, admin_values: [], viewer_values: [], label: "SSO" });
  const [adminGroups, setAdminGroups] = useState("");
  const [viewerGroups, setViewerGroups] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);

  const validate = (): boolean => {
    const e: Record<string, string> = {};
    if (useSSO) {
      if (!httpsUrl(sso.issuer)) e.issuer = "Enter the https address of your identity provider.";
      if (!sso.client_id.trim()) e.client_id = "Enter the client ID from your identity provider.";
      if (!sso.client_secret.trim()) e.client_secret = "Enter the client secret.";
      if (!/\/console\/api\/oidc\/callback$/.test(sso.redirect_url.trim())) e.redirect_url = "Must end with /console/api/oidc/callback.";
      if (!adminGroups.trim()) e.admin = "Name at least one group whose members become admins.";
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const split = (v: string) => v.split(",").map((x) => x.trim()).filter(Boolean);

  const generate = async () => {
    if (!validate()) return;
    setBusy(true);
    try {
      const r = await client.admin(useSSO ? { ...sso, admin_values: split(adminGroups), viewer_values: split(viewerGroups) } : null);
      setToken(r.admin_token);
      setSaved(false);
    } catch (err) {
      setErrors({ form: message(err) });
    } finally {
      setBusy(false);
    }
  };

  const set = (k: keyof SSOInput) => (e: { target: { value: string } }) => setSSO((s) => ({ ...s, [k]: e.target.value }));

  return (
    <StepCard
      title="Admin access"
      description="You sign in to this console with an admin token. Optionally, your colleagues can sign in with your company's single sign-on."
      footer={token ? <Button variant="primary" disabled={!saved} onClick={() => void onDone()}>Continue</Button> : null}
    >
      {token ? (
        <div className="setup-form">
          <Callout tone="warn" title="Save this token now" icon={<IconAlert />}>
            It is shown only once. Store it in your password manager. You need it to sign in after setup.
          </Callout>
          <CodeBlock code={token} label="Admin token" />
          <Checkbox label="I have saved the admin token in a safe place" checked={saved} onChange={(e) => setSaved(e.target.checked)} />
        </div>
      ) : (
        <div className="setup-form">
          {st.admin ? <Callout tone="ok" icon={<IconCheck />} title="An admin token exists">Generate a new one if you lost it; the old one stops working.</Callout> : null}
          <Checkbox label="Also set up single sign-on (optional)" hint="OpenID Connect: Microsoft Entra ID, Okta, Keycloak and others." checked={useSSO} onChange={(e) => setUseSSO(e.target.checked)} />
          {useSSO ? (
            <div className="setup-grid">
              <Field label="Identity provider address" error={errors.issuer} hint="The issuer URL, e.g. https://login.example.com/realms/main">
                {({ id, describedBy, invalid }) => <Input id={id} value={sso.issuer} onChange={set("issuer")} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
              </Field>
              <Field label="Button label" hint="Shown as “Sign in with …”">
                {({ id, describedBy }) => <Input id={id} value={sso.label} maxLength={40} onChange={set("label")} aria-describedby={describedBy} />}
              </Field>
              <Field label="Client ID" error={errors.client_id}>
                {({ id, describedBy, invalid }) => <Input id={id} value={sso.client_id} onChange={set("client_id")} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
              </Field>
              <Field label="Client secret" error={errors.client_secret} hint="Stored in a file only the gateway can read.">
                {({ id, describedBy, invalid }) => <Input id={id} type="password" autoComplete="off" value={sso.client_secret} onChange={set("client_secret")} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
              </Field>
              <Field label="Redirect address" error={errors.redirect_url} hint="Register exactly this address at your identity provider." className="setup-span">
                {({ id, describedBy, invalid }) => <Input id={id} mono value={sso.redirect_url} onChange={set("redirect_url")} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
              </Field>
              <Field label="Admin groups" error={errors.admin} hint="Comma-separated group names that get full access.">
                {({ id, describedBy, invalid }) => <Input id={id} value={adminGroups} onChange={(e) => setAdminGroups(e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} />}
              </Field>
              <Field label="Read-only groups" hint="Optional: comma-separated groups that can only look.">
                {({ id, describedBy }) => <Input id={id} value={viewerGroups} onChange={(e) => setViewerGroups(e.target.value)} aria-describedby={describedBy} />}
              </Field>
            </div>
          ) : null}
          {errors.form ? <Callout tone="bad">{errors.form}</Callout> : null}
          <div className="setup-actions">
            <Button variant="primary" loading={busy} onClick={() => void generate()}>
              {st.admin ? "Generate a new admin token" : "Generate admin token"}
            </Button>
          </div>
        </div>
      )}
    </StepCard>
  );
}

// ---- 5. verify -------------------------------------------------------------------------------------------
const CHECK_TONE: Record<string, "ok" | "warn" | "bad" | "neutral"> = { ok: "ok", warn: "warn", fail: "bad", info: "neutral" };

function CheckRow({ c }: { c: SetupCheck }) {
  const icon = c.status === "ok" ? <IconCheck size={14} /> : c.status === "info" ? <IconInfo size={14} /> : <IconAlert size={14} />;
  return (
    <li className={cx("setup-check", `setup-check-${c.status}`)} data-check={c.id}>
      <span className="setup-check-icon" aria-hidden="true">{icon}</span>
      <span className="setup-check-text">
        <span className="setup-check-label">
          {c.label} <Badge tone={CHECK_TONE[c.status]}>{c.status === "fail" ? "problem" : c.status === "warn" ? "in progress" : c.status === "info" ? "advice" : "ok"}</Badge>
        </span>
        <span className="setup-muted">{c.detail}</span>
      </span>
    </li>
  );
}

function VerifyStep({ st, client, refresh, onNext }: { st: SetupState; client: SetupApi; refresh: () => Promise<SetupState | null>; onNext: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hint = st.jobs.hint;

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await client.verify();
      await refresh();
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <StepCard
      title="Check everything"
      description="The gateway checks the connection, the model, the mask store and its safety features, and starts making masks."
      footer={st.verify?.ok ? <Button variant="primary" onClick={onNext}>Continue</Button> : null}
    >
      {hint && hint.phase !== "done" ? (
        <div className="setup-progress" aria-live="polite">
          <div className="setup-progress-head">
            {hint.phase === "failed" ? <IconAlert size={14} /> : <Spinner size={14} />}
            <span>{hint.phase === "failed" ? `Encryption hint failed: ${hint.error}` : "Building the one-time encryption hint (a few minutes for large models)"}</span>
            {hint.total > 0 ? <span className="setup-muted">{fmtInt(hint.done)} of {fmtInt(hint.total)}</span> : null}
          </div>
          <Meter value={hint.done} max={hint.total || 1} label="Encryption hint progress" />
        </div>
      ) : null}
      {st.masks !== null ? <p className="setup-muted setup-gap">Masks ready: {fmtInt(st.masks)} tokens{st.jobs.factory?.running ? " (the mask factory is running)" : ""}.</p> : null}
      {st.verify ? (
        <ul className="setup-checks">
          {st.verify.checks.map((c) => (
            <CheckRow key={c.id} c={c} />
          ))}
        </ul>
      ) : (
        <p className="setup-muted">Nothing checked yet.</p>
      )}
      {st.verify && !st.verify.ok ? <Callout tone="bad" className="setup-gap">Fix the problems above, then check again.</Callout> : null}
      {error ? <Callout tone="bad" className="setup-gap">{error}</Callout> : null}
      <div className="setup-actions">
        <Button variant={st.verify?.ok ? "secondary" : "primary"} loading={busy} onClick={() => void run()}>
          {st.verify ? "Check again" : "Run the checks"}
        </Button>
      </div>
    </StepCard>
  );
}

// ---- 6. finish -------------------------------------------------------------------------------------------
function FinishStep({ st, client }: { st: SetupState; client: SetupApi }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ mode: string; message: string; notes?: string[] } | null>(null);

  const finish = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await client.finish());
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  if (result) {
    return (
      <StepCard title={result.mode === "normal" ? "Setup complete" : "Almost done"} description="">
        <Callout tone={result.mode === "normal" ? "ok" : "warn"} icon={result.mode === "normal" ? <IconCheck /> : <IconInfo />}>
          {result.message}
        </Callout>
        {result.notes?.map((n) => (
          <Callout key={n} tone="warn" className="setup-gap">
            {n}
          </Callout>
        ))}
        {result.mode === "normal" ? (
          <div className="setup-actions">
            <a className="btn btn-primary btn-md" href={__DEMO__ ? "./" : "/console/login"}>
              <span className="btn-label">Go to sign-in</span>
            </a>
          </div>
        ) : null}
      </StepCard>
    );
  }

  return (
    <StepCard title="Finish" description="Save the configuration and start serving your model.">
      <ul className="setup-list">
        <li>Provider: {st.enroll?.service_url} (tenant {st.enroll?.tenant})</li>
        <li>Model: {st.model?.name}</li>
        <li>Masks: {st.maskstore?.root}</li>
        <li>Settings are written to <span className="mono">{st.paths.config_dir}/gateway.env</span></li>
      </ul>
      {error ? <Callout tone="bad" className="setup-gap">{error}</Callout> : null}
      <div className="setup-actions">
        <Button variant="primary" loading={busy} onClick={() => void finish()} disabled={!st.verify?.ok}>
          Finish setup
        </Button>
      </div>
    </StepCard>
  );
}
