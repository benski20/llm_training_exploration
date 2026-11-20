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
    render();
  });
} else {
  render();
}
