// Types for the console API contract (pgw/docs/api/console-api.md) and the existing admin API
// (pgw/docs/api/gateway-api.md, src/pgw/gateway/app.py). Fields the server may omit are optional;
// the UI renders "—" for anything missing rather than failing.

export type Role = "admin" | "viewer";

export interface LoginResponse {
  role: Role;
  csrf: string;
}

export type AuthKind = "token" | "oidc";

/** The signed-in single sign-on user (null for token sign-in). */
export interface SessionUser {
  sub: string;
  email?: string | null;
  name?: string | null;
}

export interface MeResponse {
  role: Role;
  csrf: string;
  expires_at?: string | number | null;
  version?: string | null;
  auth?: AuthKind;
  user?: SessionUser | null;
}

/** GET /console/api/auth-methods: which sign-in methods the gateway offers. */
export interface AuthMethods {
  token: boolean;
  oidc: { enabled: boolean; label: string };
}

export interface ApiErrorBody {
  error: { message: string; type?: string; code?: string; param?: string | null };
}

// ---- /admin/overview ----------------------------------------------------------------------------

export interface NoiseMonitorState {
  state?: string; // "ok" | "warming" | "alarm" | "unavailable" | "disabled" ...
  alarm?: boolean;
  samples?: number;
  windows?: number;
  duplicate_rows?: number;
  reasons?: string[];
  [k: string]: unknown;
}

export interface OverviewModel {
  name: string;
  state: string;
  ring: string;
  ready: boolean;
  reason?: string | null;
  masks_available?: number | null;
  masks_runway_s?: number | null;
}

export interface Overview {
  version?: string | number;
  uptime_s: number;
  models: OverviewModel[];
  requests_24h: number;
  tokens_24h: number;
  errors_24h: number;
  integrity_failures_total: number;
  freivalds_checked_total: number;
  stream_requests_total: number;
  privacy: {
    masked: boolean;
    ring?: string | null;
    freivalds_rate?: number | null;
    noise_monitor?: NoiseMonitorState | null;
  };
}

// ---- /admin/usage -------------------------------------------------------------------------------

export type UsageWindow = "1h" | "24h" | "7d";

export interface UsagePoint {
  t: number;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  errors: number;
  p50_ms: number | null;
  p95_ms: number | null;
  ttft_p50_ms: number | null;
  masks_consumed: number;
}

/** Aggregate per model or per key over the window (shape not fixed by the contract; read defensively). */
export interface UsageTotals {
  requests?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  errors?: number;
  masks_consumed?: number;
  [k: string]: unknown;
}

export interface Usage {
  window: UsageWindow;
  bucket_s: number;
  series: UsagePoint[];
  by_model?: Record<string, UsageTotals>;
  by_key?: Record<string, UsageTotals>;
}

// ---- /admin/status ------------------------------------------------------------------------------

export interface Sessions {
  active: number;
  waiting: number;
  max_active: number;
  max_queue: number;
  rejected: number;
  draining: boolean;
  rejected_by_reason?: Record<string, number>;
}

export interface Breaker {
  open: boolean;
  trips: number;
  cooldown_s?: number | null;
  open_for_s?: number | null;
  last_code?: string | null;
}

export interface BackendStats {
  calls?: number;
  rows?: number;
  round_trips?: number;
  bytes_up?: number;
  bytes_down?: number;
  freivalds_checked?: number;
  freivalds_skipped?: number;
  integrity_failures?: number;
  remote_seconds?: number;
  local_seconds?: number;
}

export interface ClientStats {
  requests?: number;
  attempts?: number;
  retries?: number;
  failures?: number;
  bytes_sent?: number;
  bytes_received?: number;
  last_rtt_s?: number;
  stream_requests?: number;
  [k: string]: number | undefined;
}

export interface ModelStatus {
  name: string;
  mode: string; // "masked" | "local" | ...
  started?: boolean;
  state: string;
  last_error?: string | null;
  connect_attempts?: number;
  ready: boolean;
  reason?: string | null;
  fingerprint?: string | null;
  fingerprint_short?: string | null;
  ring: string;
  max_context?: number | null;
  sessions?: Sessions;
  breaker?: Breaker;
  freivalds_rate?: number | null;
  service?: { server?: string; origin?: string; max_rows_per_item?: number };
  masks_available?: number | null;
  noise_monitor?: NoiseMonitorState;
  client?: ClientStats;
  backend?: BackendStats;
}

export interface Status {
  version?: number;
  uptime_s: number;
  production_policy: boolean;
  models: ModelStatus[];
  api_keys?: number;
  egress_allowlist?: string[];
  audit_log?: string | null;
  telemetry?: string;
  /** Optional transport facts (not in /admin/status today; shown when present). */
  tls?: { enabled?: boolean; mtls?: boolean; [k: string]: unknown };
}

// ---- /admin/masks -------------------------------------------------------------------------------

export interface MaskInventoryRow {
  fingerprint: string;
  ring: string;
  segments: number;
  slots_total: number;
  slots_available: number;
  bytes_on_disk: number;
}

export interface MaskStoreStatus {
  root?: string;
  store_id?: string;
  generation?: number;
  consumed_total?: number;
  open_reservations?: number;
  segments_by_status?: Record<string, number>;
  key_provider?: Record<string, unknown>;
  rollback_protection?: { journal?: boolean; counter?: string; hardware_counter?: boolean; row_macs?: boolean };
  low_water_slots?: number | null;
  inventory?: MaskInventoryRow[];
  tainted?: boolean;
}

export interface MaskModel {
  fingerprint: string;
  ring: string;
  available_slots: number | null;
  low_water_slots: number | null;
}

export interface Masks {
  store: MaskStoreStatus | null;
  models: Record<string, MaskModel>;
}

// ---- /admin/keys --------------------------------------------------------------------------------

export interface ApiKeyRecord {
  key_id: string;
  name: string;
  scopes: string[];
  models: string[];
  requests_per_minute?: number | null;
  tokens_per_minute?: number | null;
  daily_token_quota?: number | null;
  max_concurrent_requests?: number | null;
  created_at?: string;
  disabled?: boolean;
  meta?: Record<string, string>;
  tokens_today?: number;
  /** Optional: not in the v1 record; rendered when the backend adds it. */
  last_used_at?: string | null;
}

export interface CreateKeyRequest {
  name: string;
  scopes?: string[];
  models?: string[];
  requests_per_minute?: number | null;
  tokens_per_minute?: number | null;
  daily_token_quota?: number | null;
  max_concurrent_requests?: number | null;
}

export interface CreateKeyResponse {
  key: string;
  record: ApiKeyRecord;
}

// ---- /admin/audit -------------------------------------------------------------------------------

/** One JSONL audit record (allowlisted fields only, see pgw/gateway/audit.py). Never prompt text. */
export interface AuditEvent {
  ts: string;
  event: string;
  request_id?: string;
  key_id?: string;
  key_name?: string;
  model?: string;
  endpoint?: string;
  status?: number;
  code?: string;
  stream?: boolean;
  prompt_tokens?: number;
  completion_tokens?: number;
  masks_reserved?: number;
  masks_consumed?: number;
  masks_burned?: number;
  duration_ms?: number;
  ttft_ms?: number;
  queue_ms?: number;
  finish_reason?: string;
  client?: string;
  fingerprint?: string;
  ring?: string;
  mode?: string;
  action?: string;
  target?: string;
  scopes?: string;
  reason?: string;
  prompts?: number;
  retry_after_s?: number;
}

export interface AuditPage {
  events: AuditEvent[];
  next_before: string | number | null;
}

export interface AuditQuery {
  limit?: number;
  before?: string | number | null;
  key_id?: string;
  model?: string;
  status?: string;
}

// ---- chat ---------------------------------------------------------------------------------------

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  seed?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  system_fingerprint?: string | null;
  choices: { index: number; message: { role: string; content: string }; finish_reason: string | null }[];
  usage?: ChatUsage;
}

export interface ChatChunk {
  id: string;
  object: "chat.completion.chunk";
  created?: number;
  model?: string;
  system_fingerprint?: string | null;
  choices: { index: number; delta: { role?: string; content?: string }; finish_reason: string | null }[];
  usage?: ChatUsage | null;
}
