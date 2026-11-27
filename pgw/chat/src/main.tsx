import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyTheme, loadTheme } from "./lib/theme";
import "./styles/fonts.css";
import "./styles/chat.css";

applyTheme(loadTheme());

function render() {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

// `__MOCK__` is a compile-time constant (vite.config.ts `define`): in the production build the branch and the mock
// module are removed entirely.
if (__MOCK__) {
  void import("./mock").then((m) => {
    m.installMock();
    if (!window.location.hash.includes("token=")) {
      try {
        sessionStorage.setItem("pgw-chat-token", "mock-token-0123456789abcdef0123456789abcdef");
      } catch {
        // storage blocked
      }
    }
    render();
  });
} else {
  render();
}
