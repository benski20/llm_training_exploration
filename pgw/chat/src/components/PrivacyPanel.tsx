import { useEffect, useRef } from "react";
import type { Info, Status } from "../api/types";
import { facts } from "../lib/privacy";
import { IconCheck, IconClose, IconWarn } from "./Icons";

export function PrivacyPanel({ info, status, onClose }: { info: Info; status: Status | null; onClose(): void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) {
      if (typeof d.showModal === "function") d.showModal();
      else d.setAttribute("open", "");
    }
  }, []);
  return (
    <dialog ref={ref} className="privacy" aria-labelledby="privacy-title" onClose={onClose}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}>
      <div className="privacy-inner">
        <div className="privacy-head">
          <h2 id="privacy-title">What is true right now</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <IconClose />
          </button>
        </div>
        <ul className="facts">
          {facts(info, status).map((f) => (
            <li key={f.id} className={`fact is-${f.tone}`} data-fact={f.id}>
              <span className="fact-icon">{f.tone === "ok" ? <IconCheck /> : f.tone === "info" ? <span className="fact-dot" /> : <IconWarn />}</span>
              <span>
                <span className="fact-title">{f.title}</span>
                <span className="fact-detail">{f.detail}</span>
              </span>
            </li>
          ))}
        </ul>
        <p className="privacy-foot">
          PGW Chat {info.version}. The masking and checks are the PGW client library's; its design and review status
          are in the security documentation. No external audit has been completed yet.
        </p>
      </div>
    </dialog>
  );
}
