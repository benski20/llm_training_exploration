import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyInitialTheme } from "./lib/theme";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/layout.css";
import "./styles/pages.css";
import "./styles/setup.css";

applyInitialTheme();

function render() {
  createRoot(document.getElementById("root")!).render(
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
      void import("./mock/setup").then((setup) => {
        if (setup.setupActive()) {
          tag.classList.add("is-inline");
          tag.append(`Demo of first-run setup. Setup token: "setup". Enrollment code: `);
          const code = document.createElement("code");
          code.textContent = setup.SAMPLE_CODE;
          tag.append(code);
        } else {
          tag.append(`Demo with sample data. Sign in with the token "admin" (or "viewer"). `);
          const link = document.createElement("button");
          link.type = "button";
          link.className = "demo-banner-link";
          link.textContent = "Try the first-run setup wizard";
          link.onclick = () => setup.startSetupDemo();
          tag.append(link);
        }
      });
      document.body.prepend(tag);
    }
    render();
  });
} else {
  render();
}
