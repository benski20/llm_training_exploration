import { useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { ApiKeyRecord, CreateKeyRequest, CreateKeyResponse } from "../api/types";
import { useGateway } from "../app/gateway";
import { useSession } from "../app/session";
import { fmtCompact, fmtDate, fmtRelative, keyPrefix } from "../lib/format";
import { useResource } from "../lib/useResource";
import {
  Badge,
  Button,
  Callout,
  Card,
  Checkbox,
  ConfirmDialog,
  CopyButton,
  Dialog,
  EmptyState,
  ErrorState,
  Field,
  IconAlert,
  IconKey,
  IconLock,
  IconPlus,
  Input,
  Meter,
  PageHeader,
  Table,
  useToast,
  type Column,
} from "../ui";

function limitText(k: ApiKeyRecord): string[] {
  const out: string[] = [];
  if (k.requests_per_minute) out.push(`${fmtCompact(k.requests_per_minute)} rpm`);
  if (k.tokens_per_minute) out.push(`${fmtCompact(k.tokens_per_minute)} tpm`);
  if (k.max_concurrent_requests) out.push(`${k.max_concurrent_requests} concurrent`);
  return out;
}

interface Form {
  name: string;
  inference: boolean;
  admin: boolean;
  models: string[];
  rpm: string;
  tpm: string;
  daily: string;
  concurrent: string;
}

const EMPTY: Form = { name: "", inference: true, admin: false, models: [], rpm: "", tpm: "", daily: "", concurrent: "" };

function optInt(s: string): number | null | "bad" {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isInteger(n) && n > 0 ? n : "bad";
}

function CreateKeyDialog({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const gw = useGateway();
  const [f, setF] = useState<Form>(EMPTY);
  const [errors, setErrors] = useState<Partial<Record<keyof Form | "form", string>>>({});
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState<CreateKeyResponse | null>(null);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((x) => ({ ...x, [k]: v }));

  const close = () => {
    setF(EMPTY);
    setErrors({});
    setCreated(null);
    onClose();
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const errs: typeof errors = {};
    if (!f.name.trim()) errs.name = "Give the key a name.";
    if (!f.inference && !f.admin) errs.form = "Select at least one scope.";
    const nums = { rpm: optInt(f.rpm), tpm: optInt(f.tpm), daily: optInt(f.daily), concurrent: optInt(f.concurrent) };
    for (const [k, v] of Object.entries(nums)) if (v === "bad") errs[k as keyof Form] = "Positive whole number, or empty for the default.";
    setErrors(errs);
    if (Object.keys(errs).length) return;
    const req: CreateKeyRequest = {
      name: f.name.trim(),
      scopes: [f.inference && "inference", f.admin && "admin"].filter(Boolean) as string[],
      models: f.models,
      requests_per_minute: nums.rpm as number | null,
      tokens_per_minute: nums.tpm as number | null,
      daily_token_quota: nums.daily as number | null,
      max_concurrent_requests: nums.concurrent as number | null,
    };
    setBusy(true);
    try {
      setCreated(await api.createKey(req));
      onCreated();
    } catch (err) {
      setErrors({ form: err instanceof ApiError ? err.message : "Could not create the key." });
    } finally {
      setBusy(false);
    }
  };

  if (created) {
    return (
      <Dialog
        open={open}
        onClose={close}
        dismissable={false}
        title="API key created"
        description={`${created.record.name} · ${keyPrefix(created.record.key_id)}`}
        footer={
          <Button variant="primary" onClick={close}>
            I have stored the key
          </Button>
        }
      >
        <Callout tone="warn" icon={<IconAlert />} title="Copy this key now">
          It is shown only once. The gateway keeps only a SHA-256 hash and cannot show it again.
        </Callout>
        <div className="secret">
          <code className="secret-value mono" aria-label="New API key">
            {created.key}
          </code>
          <CopyButton text={created.key} label="Copy key" variant="secondary" size="md" showLabel />
        </div>
      </Dialog>
    );
  }

  return (
    <Dialog
      open={open}
      onClose={close}
      title="Create API key"
      description="Keys authenticate applications against the OpenAI-compatible API."
      footer={
        <>
          <Button variant="ghost" onClick={close} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" form="create-key" loading={busy}>
            Create key
          </Button>
        </>
      }
    >
      <form id="create-key" className="form" onSubmit={submit} noValidate>
        {errors.form ? <Callout tone="bad">{errors.form}</Callout> : null}
        <Field label="Name" error={errors.name} hint="For your reference, e.g. the application using it.">
          {({ id, describedBy, invalid }) => (
            <Input id={id} value={f.name} onChange={(e) => set("name", e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} placeholder="billing-app" autoFocus maxLength={128} />
          )}
        </Field>
        <fieldset className="fieldset">
          <legend className="field-label">Scopes</legend>
          <Checkbox label="inference" hint="Chat and completions" checked={f.inference} onChange={(e) => set("inference", e.target.checked)} />
          <Checkbox label="admin" hint="Admin API: status, keys, model actions" checked={f.admin} onChange={(e) => set("admin", e.target.checked)} />
        </fieldset>
        <fieldset className="fieldset">
          <legend className="field-label">Models</legend>
          <p className="field-hint">Leave all unchecked to allow every model. Other model names are not disclosed to restricted keys.</p>
          <div className="check-grid">
            {gw.models.map((m) => (
              <Checkbox
                key={m.name}
                label={<span className="mono">{m.name}</span>}
                checked={f.models.includes(m.name)}
                onChange={(e) => set("models", e.target.checked ? [...f.models, m.name] : f.models.filter((x) => x !== m.name))}
              />
            ))}
          </div>
        </fieldset>
        <div className="field-grid">
          {(
            [
              ["rpm", "Requests / minute"],
              ["tpm", "Tokens / minute"],
              ["daily", "Daily token quota"],
              ["concurrent", "Max concurrent"],
            ] as const
          ).map(([k, label]) => (
            <Field key={k} label={label} error={errors[k]}>
              {({ id, describedBy, invalid }) => (
                <Input id={id} inputMode="numeric" className="num" value={f[k]} onChange={(e) => set(k, e.target.value)} aria-describedby={describedBy} aria-invalid={invalid || undefined} placeholder="default" />
              )}
            </Field>
          ))}
        </div>
      </form>
    </Dialog>
  );
}

export function KeysPage() {
  const { isAdmin } = useSession();
  const toast = useToast();
  const keys = useResource("keys", () => api.keys());
  const [creating, setCreating] = useState(false);
  const [revoke, setRevoke] = useState<ApiKeyRecord | null>(null);
  const rows = keys.data?.keys;

  const cols: Column<ApiKeyRecord>[] = [
    {
      key: "name",
      header: "Name",
      render: (k) => (
        <span className="cell-stack">
          <span className="cell-strong">{k.name}</span>
          <span className="cell-sub mono">{keyPrefix(k.key_id)}-••••</span>
        </span>
      ),
    },
    {
      key: "scopes",
      header: "Scopes",
      render: (k) => (
        <span className="badges">
          {k.disabled ? <Badge tone="bad">revoked</Badge> : null}
          {k.scopes.map((s) => (
            <Badge key={s} tone={s === "admin" ? "accent" : "neutral"}>
              {s}
            </Badge>
          ))}
        </span>
      ),
    },
    { key: "models", header: "Models", className: "hide-md", render: (k) => (k.models.length ? <span className="mono small">{k.models.join(", ")}</span> : <span className="muted">all</span>) },
    {
      key: "limits",
      header: "Limits",
      className: "hide-md",
      render: (k) => {
        const l = limitText(k);
        return l.length ? <span className="small num">{l.join(" · ")}</span> : <span className="muted">defaults</span>;
      },
    },
    {
      key: "today",
      header: "Tokens today",
      render: (k) =>
        k.daily_token_quota ? (
          <div className="quota">
            <span className="small num">
              {fmtCompact(k.tokens_today ?? 0)} / {fmtCompact(k.daily_token_quota)}
            </span>
            <Meter value={k.tokens_today ?? 0} max={k.daily_token_quota} tone={(k.tokens_today ?? 0) / k.daily_token_quota > 0.9 ? "warn" : "accent"} label={`${k.name} daily quota used`} height={4} />
          </div>
        ) : (
          <span className="num">{fmtCompact(k.tokens_today ?? 0)}</span>
        ),
    },
    { key: "created", header: "Created", className: "hide-sm", render: (k) => <span title={k.created_at}>{fmtDate(k.created_at)}</span> },
    { key: "used", header: "Last used", className: "hide-sm", render: (k) => (k.last_used_at ? <span title={k.last_used_at}>{fmtRelative(k.last_used_at)}</span> : <span className="muted">—</span>) },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "right",
      render: (k) =>
        isAdmin && !k.disabled && k.meta?.builtin !== "true" ? (
          <Button size="sm" variant="ghost" className="btn-danger-ghost" onClick={() => setRevoke(k)}>
            Revoke
          </Button>
        ) : null,
    },
  ];

  return (
    <div className="page">
      <PageHeader
        title="API keys"
        description="Application credentials for the OpenAI-compatible API. Secrets are shown once at creation and stored only as hashes."
        actions={
          isAdmin ? (
            <Button variant="primary" icon={<IconPlus />} onClick={() => setCreating(true)}>
              Create key
            </Button>
          ) : (
            <Badge tone="neutral">
              <IconLock size={12} /> Read-only
            </Badge>
          )
        }
      />
      {keys.error && !rows ? (
        <ErrorState error={keys.error} onRetry={keys.reload} />
      ) : (
        <Card flush>
          <Table
            columns={cols}
            rows={rows}
            loading={keys.loading}
            rowKey={(k) => k.key_id}
            caption="API keys"
            empty={
              <EmptyState icon={<IconKey size={20} />} title="No API keys yet" action={isAdmin ? <Button onClick={() => setCreating(true)}>Create the first key</Button> : null}>
                Applications need a key to call the gateway.
              </EmptyState>
            }
          />
        </Card>
      )}
      <CreateKeyDialog open={creating} onClose={() => setCreating(false)} onCreated={() => void keys.reload()} />
      <ConfirmDialog
        open={revoke !== null}
        onClose={() => setRevoke(null)}
        title="Revoke API key?"
        confirmLabel="Revoke key"
        onConfirm={async () => {
          if (!revoke) return;
          await api.revokeKey(revoke.key_id);
          toast({ title: "Key revoked", description: `${revoke.name} can no longer authenticate.`, tone: "ok" });
          await keys.reload();
        }}
      >
        <p>
          <strong>{revoke?.name}</strong> (<span className="mono">{revoke ? keyPrefix(revoke.key_id) : ""}</span>) stops working immediately. Applications using it will receive
          401 errors. This cannot be undone.
        </p>
      </ConfirmDialog>
    </div>
  );
}
