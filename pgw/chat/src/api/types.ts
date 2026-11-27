// Shapes of the `pgw chat` localhost API (src/pgw/chat/server.py).

export interface ModelInfo {
  id: string;
  label: string;
  params?: string;
  note?: string;
  trips_per_token?: number;
  download_gb?: number;
  disk_gb?: number;
  stream_disk_gb?: number;
  memory_gb?: number;
  cpus?: number;
}

export interface Settings {
  mask_scheme: string | null;
  masked_head: string;
  prefix_cache: string; // "auto" | "on" | "off"
  speculation: string;
  factory_weights: string; // "pack" | "stream"
  usage_reports: boolean;
  service_url?: string | null;
}

export interface Info {
  app: string;
  version: string;
  deployment: "local";
  demo: boolean;
  default_model: string;
  recommended_model: string;
  models: ModelInfo[];
  device: { cpus: number; memory_gb: number | null; free_disk_gb: number | null };
  settings: Settings;
  history: { persistent: boolean; location: string | null };
  limits: { max_message_chars: number; max_tokens: number };
}

export type Phase =
  | "idle"
  | "starting"
  | "enrolling"
  | "provisioning"
  | "connecting"
  | "setup"
  | "preparing"
  | "ready"
  | "stalled"
  | "blocked"
  | "failed";

export interface ErrorView {
  kind: string;
  title: string;
  message: string;
  code?: string;
  retryable: boolean;
}

export interface Status {
  model: string;
  phase: Phase;
  title: string;
  detail: string | null;
  percent: number | null;
  eta_s: number | null;
  masks_available: number | null;
  requests: number;
  verified_requests: number;
  integrity_failures: number;
  last_verified_at: number | null;
  lane: string | null;
  masked_head: boolean;
  prefix_cache: boolean;
  mask_scheme: string | null;
  error: ErrorView | null;
}

export interface Usage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cached_tokens: number | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: number;
  model?: string;
  status?: "done" | "stopped" | "error";
  verified?: boolean;
  finish_reason?: string | null;
  usage?: Usage | null;
  error?: ErrorView | null;
  elapsed_s?: number;
  ttft_s?: number | null;
  tokens_per_s?: number | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  model: string | null;
  created_at: number;
  updated_at: number;
  messages: number;
}

export interface Conversation {
  id: string;
  title: string;
  model: string | null;
  created_at: number;
  updated_at: number;
  messages: Message[];
}

export type ChatEvent =
  | {
      type: "start";
      conversation_id: string;
      generation_id: string;
      model: string;
      title: string;
      user_message: Message | null;
    }
  | { type: "delta"; text: string }
  | { type: "done"; message: Message; error: null }
  | { type: "error"; message: Message; error: ErrorView };
