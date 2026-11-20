import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { codeProblem, createSetupApi, type SetupState } from "../api/setup";
import { SetupPage } from "./Setup";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const apiErr = (status: number, message: string, code = "protocol") => json({ error: { message, type: "setup_error", code, param: null } }, status);

const BASE: SetupState = {
  setup_mode: true,
  finished: false,
  steps: ["enroll", "model", "maskstore", "admin", "verify", "finish"],
  estimate: { basis: "typical_7b", hint_bytes: 34_200_000_000, buffer_bytes: null, total_bytes: null, buffer_slots: 4000, disk_free_bytes: 500_000_000_000 },
  jobs: { pack: null, hint: null, factory: null },
  masks: null,
  paths: { config_dir: "/etc/pgw", data_dir: "/var/lib/pgw" },
  kms_options: [{ id: "aws-kms", label: "AWS KMS", available: false }],
};

const VALID_CODE = "PGW1-" + "A".repeat(140);

afterEach(() => vi.unstubAllGlobals());

describe("codeProblem", () => {
  it("accepts a well-formed code with spaces and line breaks", () => {
    expect(codeProblem(VALID_CODE.replace(/-/g, " ").replace("AAAAA", "AAAAA\n"))).toBeNull();
  });
  it("explains what is wrong in plain words", () => {
    expect(codeProblem("")).toMatch(/Paste/);
    expect(codeProblem("hello")).toMatch(/starts with PGW1/);
    expect(codeProblem("PGW2-AAAA")).toMatch(/different version/);
    expect(codeProblem("PGW1-AAAA!")).toMatch(/characters/);
    expect(codeProblem("PGW1-AAAAA")).toMatch(/incomplete/);
  });
});

function mount(handler: (url: string, init?: RequestInit) => Response) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetchImpl = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    return handler(url, init);
  });
  render(<SetupPage client={createSetupApi(fetchImpl)} />);
  return calls;
}

describe("SetupPage", () => {
  it("asks for the setup token and explains a wrong one", async () => {
    mount((url) => (url === "/console/api/setup/login" ? apiErr(401, "no", "unauthenticated") : apiErr(401, "not signed in", "unauthenticated")));
    const input = await screen.findByLabelText("Setup token");
    fireEvent.change(input, { target: { value: "pgw-setup-wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Start setup" }));
    expect(await screen.findByText(/was not accepted/)).toBeTruthy();
  });

  it("validates the enrollment code before sending it and sends the CSRF header", async () => {
    const calls = mount((url) => {
      if (url === "/console/api/me") return json({ role: "admin", csrf: "csrf-1", setup: true });
      if (url === "/console/api/setup/state") return json(BASE);
      if (url === "/console/api/setup/enroll") return apiErr(400, "the enrollment code has a typo (its checksum does not match)");
      return apiErr(404, "unknown");
    });
    const box = await screen.findByLabelText("Enrollment code");
    fireEvent.change(box, { target: { value: "not a code" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(await screen.findByText(/starts with PGW1-/)).toBeTruthy();
    expect(calls.some((c) => c.url === "/console/api/setup/enroll")).toBe(false);

    fireEvent.change(box, { target: { value: VALID_CODE } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(await screen.findByText(/has a typo/)).toBeTruthy();
    const post = calls.find((c) => c.url === "/console/api/setup/enroll");
    expect((post?.init?.headers as Record<string, string>)["X-PGW-CSRF"]).toBe("csrf-1");
    // later steps are locked until this one is done
    expect((screen.getByRole("button", { name: /Choose a model/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the admin token once and requires confirming it was saved", async () => {
    const st: SetupState = {
      ...BASE,
      enroll: { tenant: "acme", service_url: "https://compute.example.net:8443", models: [], client_cert: true, ca_fingerprint: "ab" },
      model: { name: "qwen7b", fingerprint: "f".repeat(64), ring: "K32A12C27", hf_repo: "Qwen/Qwen2.5-7B-Instruct", pack_status: "ready" },
      maskstore: { root: "/var/lib/pgw/masks", key_file: "/etc/pgw/secrets/maskstore.kek", counter_file: "/etc/pgw/secrets/maskstore.counter" },
    };
    mount((url) => {
      if (url === "/console/api/me") return json({ role: "admin", csrf: "c", setup: true });
      if (url === "/console/api/setup/state") return json(st);
      if (url === "/console/api/setup/admin") return json({ admin_token: "pgw-admin-secret-value", token_file: "/etc/pgw/secrets/admin_token", sso: false });
      return apiErr(404, "unknown");
    });
    fireEvent.click(await screen.findByRole("button", { name: "Generate admin token" }));
    expect(await screen.findByText("pgw-admin-secret-value")).toBeTruthy();
    const cont = screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement;
    expect(cont.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText(/I have saved the admin token/));
    expect(cont.disabled).toBe(false);
  });

  it("checks single sign-on fields before generating", async () => {
    const st: SetupState = {
      ...BASE,
      enroll: { tenant: "acme", service_url: "https://compute.example.net:8443", models: [], client_cert: false, ca_fingerprint: "ab" },
      model: { name: "m", fingerprint: "f".repeat(64), ring: null, hf_repo: null, pack_status: "ready" },
      maskstore: { root: "/m", key_file: "/k", counter_file: "/c" },
    };
    const calls = mount((url) => {
      if (url === "/console/api/me") return json({ role: "admin", csrf: "c", setup: true });
      if (url === "/console/api/setup/state") return json(st);
      return apiErr(404, "unknown");
    });
    fireEvent.click(await screen.findByLabelText(/Also set up single sign-on/));
    fireEvent.change(screen.getByLabelText("Identity provider address"), { target: { value: "http://idp.example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate admin token" }));
    expect(await screen.findByText(/https address/)).toBeTruthy();
    expect(screen.getByText(/at least one group/)).toBeTruthy();
    expect(calls.some((c) => c.url === "/console/api/setup/admin")).toBe(false);
  });
});
