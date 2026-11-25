import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyInitialTheme } from "./lib/theme";
// The PGW design system (tokens, fonts, components, shell), shared with the developer platform.
import "../../src/pgw/design/pgw.css";
import "./styles/layout.css";
import "./styles/pages.css";
import "./styles/setup.css";

applyInitialTheme();

let root: ReturnType<typeof createRoot> | null = null;

function render() {
  root?.unmount();
  root = createRoot(document.getElementById("root")!);
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

// `__MOCK__` is a compile-time constant (vite.config.ts `define`): in the production build the
// branch and the mock module are removed entirely.
if (__MOCK__) {
  void import("./mock").then((m) => {
    m.installMock();
    if (__DEMO__) {
      const tag = document.createElement("div");
      tag.className = "demo-banner";
      document.body.prepend(tag);
      void import("./mock/setup").then((setup) => {
        const fill = () => {
          tag.replaceChildren();
          if (setup.setupActive()) {
            tag.classList.add("is-inline");
            const row = (label: string, value: string) => {
              const line = document.createElement("div");
              line.className = "demo-banner-row";
              const l = document.createElement("span");
              l.textContent = label;
              const code = document.createElement("code");
              code.textContent = value;
              const copy = document.createElement("button");
              copy.type = "button";
              copy.className = "demo-banner-link";
              copy.textContent = "Copy";
              copy.onclick = () => {
                navigator.clipboard?.writeText(value).then(
                  () => (copy.textContent = "Copied"),
                  () => window.getSelection()?.selectAllChildren(code),
                );
              };
              line.append(l, code, copy);
              return line;
            };
            tag.append("Demo of first-run setup (sample data). ");
            tag.append(row("Step 1, setup token:", "setup"), row("Step 2, enrollment code:", setup.SAMPLE_CODE));
          } else {
            tag.classList.remove("is-inline");
            tag.append(`Demo with sample data. Sign in with the token "admin" (or "viewer"). `);
            const link = document.createElement("button");
            link.type = "button";
            link.className = "demo-banner-link";
            link.textContent = "Try the first-run setup wizard";
            link.onclick = () => {
              if (!setup.startSetupDemo()) {
                window.location.hash = "";
                fill();
                render();
              }
            };
            tag.append(link);
          }
        };
        fill();
        setup.onExit(() => {
          fill();
        });
      });
    }
    render();
  });
} else {
  render();
}
