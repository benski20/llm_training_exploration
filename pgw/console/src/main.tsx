import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyInitialTheme } from "./lib/theme";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/layout.css";
import "./styles/pages.css";

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
      tag.textContent = "Demo with sample data. Sign in with the token \"admin\" (or \"viewer\").";
      document.body.prepend(tag);
    }
    render();
  });
} else {
  render();
}
