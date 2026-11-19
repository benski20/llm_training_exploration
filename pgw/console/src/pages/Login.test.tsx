import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SessionProvider } from "../app/session";
import { LoginPage } from "./Login";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const anon = () => json({ error: { message: "not signed in", type: "authentication_error", code: "unauthenticated", param: null } }, 401);

function mount(methods: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url === "/console/api/auth-methods") return methods === null ? json({ error: { message: "unknown" } }, 404) : json(methods);
      return anon();
    }),
  );
  render(
    <MemoryRouter initialEntries={["/login"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <SessionProvider>
        <LoginPage />
      </SessionProvider>
    </MemoryRouter>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("LoginPage", () => {
  it("offers SSO with the configured label next to the token form", async () => {
    mount({ token: true, oidc: { enabled: true, label: "Okta" } });
    const sso = await screen.findByRole("link", { name: "Sign in with Okta" });
    expect(sso.getAttribute("href")).toBe("/console/api/oidc/login");
    expect(screen.getByLabelText("Access token")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign in with token" })).toBeTruthy();
  });

  it("hides the token form when token sign-in is disabled", async () => {
    mount({ token: false, oidc: { enabled: true, label: "SSO" } });
    await screen.findByRole("link", { name: "Sign in with SSO" });
    expect(screen.queryByLabelText("Access token")).toBeNull();
    expect(screen.getByText(/Single sign-on is required/)).toBeTruthy();
  });

  it("shows only the token form without SSO (or on an older gateway)", async () => {
    mount(null);
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /Sign in with/ })).toBeNull();
  });
});
