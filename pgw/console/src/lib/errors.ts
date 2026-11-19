import { ApiError } from "../api/client";

/** Human wording for an API error, with a retry hint for 429/503. */
export function describeError(e: unknown): { title: string; detail: string } {
  if (!(e instanceof ApiError)) return { title: "Something went wrong", detail: e instanceof Error ? e.message : String(e) };
  const wait = e.retryAfter !== null ? ` Retry in ${e.retryAfter}s.` : " Retry shortly.";
  switch (true) {
    case e.status === 0:
      return { title: "Gateway unreachable", detail: "Check the network connection and that the gateway is running." };
    case e.status === 403:
      return { title: "Admin role required", detail: "This action is not available to viewer sessions." };
    case e.status === 429:
      return { title: "Rate limited", detail: `${e.message}.${wait}`.replace("..", ".") };
    case e.status === 503 && e.code === "mask_exhausted":
      return { title: "Mask store exhausted", detail: `${e.message}. Refill masks or lower max_tokens.` };
    case e.status === 503 || e.status === 504:
      return { title: "Temporarily unavailable", detail: `${e.message}.${wait}`.replace("..", ".") };
    case e.code === "integrity_check_failed":
      return { title: "Integrity check failed", detail: "A compute result failed verification. Do not retry; investigate before resetting." };
    default:
      return { title: e.status >= 500 ? "Server error" : "Request failed", detail: e.requestId ? `${e.message} (request ${e.requestId})` : e.message };
  }
}
