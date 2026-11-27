import { useEffect, useRef, useState } from "react";
import type { Info, ModelInfo } from "../api/types";
import { IconCheck, IconChevron } from "./Icons";

interface Props {
  info: Info;
  model: string;
  onSelect(id: string): void;
  disabled?: boolean;
}

export function fitsDevice(m: ModelInfo, info: Info): boolean {
  const d = info.device;
  return (d.memory_gb === null || (m.memory_gb ?? 0) <= d.memory_gb) && (m.cpus ?? 0) <= d.cpus;
}

export function ModelPicker({ info, model, onSelect, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);
  const cur = info.models.find((m) => m.id === model);
  const stream = info.settings.factory_weights === "stream";
  return (
    <div className="picker" ref={ref}>
      <button type="button" className="picker-btn" aria-haspopup="listbox" aria-expanded={open} disabled={disabled}
        onClick={() => setOpen((o) => !o)}>
        {cur?.label ?? model}
        <IconChevron />
      </button>
      {open ? (
        <div className="picker-menu" role="listbox" aria-label="Model">
          {info.models.map((m) => {
            const fits = fitsDevice(m, info);
            const disk = stream ? m.stream_disk_gb : m.disk_gb;
            return (
              <button key={m.id} type="button" role="option" aria-selected={m.id === model}
                className={`picker-item${m.id === model ? " is-selected" : ""}`}
                onClick={() => {
                  setOpen(false);
                  if (m.id !== model) onSelect(m.id);
                }}>
                <span className="picker-main">
                  <span className="picker-label">
                    {m.label}
                    {m.id === info.recommended_model ? <span className="tag">Recommended here</span> : null}
                  </span>
                  <span className="picker-note">{m.note}</span>
                  <span className={`picker-res${fits ? "" : " is-warn"}`}>
                    {fits ? "" : "Above this computer's size · "}
                    {m.memory_gb} GB memory · about {disk} GB disk · {m.download_gb} GB first download
                  </span>
                </span>
                {m.id === model ? <IconCheck /> : null}
              </button>
            );
          })}
          <p className="picker-foot">
            Models run their non-linear steps on this computer and their large matrix products, masked, on our GPUs.
            Bigger models answer better but take more disk, memory and setup time.
          </p>
        </div>
      ) : null}
    </div>
  );
}
