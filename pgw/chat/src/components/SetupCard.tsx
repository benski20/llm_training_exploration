import type { Info, Status } from "../api/types";
import { duration } from "../lib/format";
import { IconWarn } from "./Icons";

const BUSY = new Set(["starting", "enrolling", "provisioning", "connecting", "setup", "preparing", "idle"]);

export function isSettingUp(st: Status | null): boolean {
  return !!st && BUSY.has(st.phase) && st.phase !== "preparing";
}

/** First-run setup progress and the states where the model cannot answer now. `null` when ready. */
export function SetupCard({ info, status, onRetry }: { info: Info; status: Status | null; onRetry(): void }) {
  if (!status || status.phase === "ready") return null;
  const m = info.models.find((x) => x.id === status.model);
  const stream = info.settings.factory_weights === "stream";
  if (status.phase === "stalled" || status.phase === "blocked" || status.phase === "failed") {
    const bad = status.phase === "blocked";
    return (
      <section className={`setup-card is-${bad ? "bad" : "warn"}`} role="status" aria-live="polite">
        <div className="setup-head">
          <IconWarn />
          <h2>{status.title}</h2>
        </div>
        {status.detail ? <p className="setup-detail">{status.detail}</p> : null}
        {status.phase === "stalled" ? (
          <p className="setup-note">
            Your conversations are safe on this computer. The app keeps retrying and resumes by itself when the GPU
            service answers again.
          </p>
        ) : null}
        {status.phase === "failed" && (status.error?.retryable ?? true) ? (
          <button type="button" className="btn" onClick={onRetry}>
            Try again
          </button>
        ) : null}
      </section>
    );
  }
  const pct = status.percent;
  return (
    <section className="setup-card" role="status" aria-live="polite">
      <div className="setup-head">
        <span className="spinner" aria-hidden="true" />
        <h2>{status.phase === "preparing" ? "Preparing masks" : "Setting up private chat"}</h2>
      </div>
      <p className="setup-detail">
        <strong>{status.title}.</strong> {status.detail ?? ""}
      </p>
      {pct !== null ? (
        <progress className="setup-progress" max={100} value={pct} aria-label="Setup progress" />
      ) : (
        <progress className="setup-progress" aria-label="Setup in progress" />
      )}
      <div className="setup-row">
        <span>{pct !== null ? `${Math.round(pct)}%` : "Working"}</span>
        <span>{status.eta_s ? `about ${duration(status.eta_s)} left` : ""}</span>
      </div>
      {status.phase !== "preparing" ? (
        <p className="setup-note">
          Once per model and computer: about {m?.download_gb ?? "?"} GB of public model files are downloaded and
          checked against a pinned fingerprint, then the one-time mask setup is built on your CPU (in the background,
          at low priority). It needs about {stream ? m?.stream_disk_gb : m?.disk_gb} GB of disk. You can close this
          tab: setup continues while pgw chat runs, and resumes where it stopped if interrupted.
        </p>
      ) : null}
    </section>
  );
}
