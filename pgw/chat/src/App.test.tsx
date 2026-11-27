import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { App } from "./App";
import { installMock } from "./mock";

const TOKEN = "mock-token-0123456789abcdef0123456789abcdef";

describe("App (mock server)", () => {
  beforeEach(() => {
    sessionStorage.clear();
    window.history.replaceState(null, "", "/");
  });

  it("asks for the terminal link when the tab has no token", () => {
    installMock(globalThis as unknown as Window & typeof globalThis);
    render(<App />);
    expect(screen.getByText("Open PGW Chat from your terminal")).toBeTruthy();
  });

  it("sends a message, streams a private, verified answer and lists the chat", async () => {
    installMock(globalThis as unknown as Window & typeof globalThis);
    window.history.replaceState(null, "", `/#token=${TOKEN}`);
    render(<App />);
    expect(window.location.hash).toBe(""); // the token left the address bar
    await screen.findByText("What’s on your mind?");
    await screen.findByText("Private");
    const box = screen.getByLabelText("Message") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "How private is this?" } });
    fireEvent.keyDown(box, { key: "Enter" });
    await screen.findByText("How private is this?");
    await waitFor(() => expect(screen.getAllByText(/Masking keeps your words/).length).toBeGreaterThan(0), { timeout: 8000 });
    await waitFor(() => expect(screen.getByText("Computed privately")).toBeTruthy(), { timeout: 15000 });
    expect(screen.getByText(/tok\/s/)).toBeTruthy();
    expect(window.location.pathname).toMatch(/^\/c\/c_/);
    const nav = screen.getByRole("navigation");
    await waitFor(() => expect(within(nav).getByText("How private is this?")).toBeTruthy());
  }, 30000);

  it("shows first-run setup progress and holds sending until it is ready", async () => {
    window.history.replaceState(null, "", "/?state=setup");
    installMock(globalThis as unknown as Window & typeof globalThis);
    sessionStorage.setItem("pgw-chat-token", TOKEN);
    render(<App />);
    await screen.findByText("Setting up private chat");
    expect(screen.getByText("42%")).toBeTruthy();
    expect(screen.getByText("about 6 min left")).toBeTruthy();
    const box = screen.getByLabelText("Message") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "hello" } });
    expect((screen.getByLabelText("Send message") as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows the stalled provider state", async () => {
    window.history.replaceState(null, "", "/?state=stalled");
    installMock(globalThis as unknown as Window & typeof globalThis);
    sessionStorage.setItem("pgw-chat-token", TOKEN);
    render(<App />);
    await screen.findByText("Provider unreachable");
    expect(screen.getByText(/resumes by itself/)).toBeTruthy();
  });

  it("opens the privacy screen with what is true now", async () => {
    installMock(globalThis as unknown as Window & typeof globalThis);
    sessionStorage.setItem("pgw-chat-token", TOKEN);
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Show privacy details/ }));
    await screen.findByText("What is true right now");
    expect(screen.getByText("Your words stay on this computer")).toBeTruthy();
    expect(screen.getByText("The GPU sees only masked numbers")).toBeTruthy();
    expect(screen.getByText("What the GPU service can see")).toBeTruthy();
  });

  it("deletes a conversation after confirmation", async () => {
    installMock(globalThis as unknown as Window & typeof globalThis);
    sessionStorage.setItem("pgw-chat-token", TOKEN);
    render(<App />);
    const nav = await screen.findByRole("navigation");
    await within(nav).findByText("Explain Freivalds' algorithm simply");
    fireEvent.click(within(nav).getByLabelText("Delete Explain Freivalds' algorithm simply"));
    fireEvent.click(within(nav).getByText("Delete"));
    await waitFor(() => expect(within(nav).queryByText("Explain Freivalds' algorithm simply")).toBeNull());
  });
});
