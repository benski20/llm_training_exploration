import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { Button } from "./Button";
import { Badge, statusTone } from "./Badge";
import { niceScale } from "./Chart";
import { Dialog } from "./Dialog";
import { Field, Input } from "./Input";
import { Stat } from "./Stat";
import { Table } from "./Table";
import { Tabs } from "./Tabs";

describe("Button", () => {
  it("renders an accessible icon-only button and blocks clicks while loading", () => {
    const onClick = vi.fn();
    const { rerender } = render(<Button iconOnly label="Sign out" icon={<svg />} onClick={onClick} />);
    const b = screen.getByRole("button", { name: "Sign out" });
    fireEvent.click(b);
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<Button iconOnly label="Sign out" icon={<svg />} onClick={onClick} loading />);
    expect((screen.getByRole("button", { name: "Sign out" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Field", () => {
  it("links the label, hint and error to the control", () => {
    render(
      <Field label="Name" hint="For reference" error="Required">
        {({ id, describedBy, invalid }) => <Input id={id} aria-describedby={describedBy} aria-invalid={invalid} />}
      </Field>,
    );
    const input = screen.getByLabelText("Name");
    expect(input.getAttribute("aria-invalid")).toBe("true");
    expect(input.getAttribute("aria-describedby")).toContain(screen.getByRole("alert").id);
  });
});

describe("Table", () => {
  const rows = [
    { id: "a", name: "alpha", n: 1 },
    { id: "b", name: "beta", n: 2 },
  ];
  const cols = [
    { key: "name", header: "Name", render: (r: (typeof rows)[0]) => r.name },
    { key: "n", header: "N", align: "right" as const, render: (r: (typeof rows)[0]) => r.n },
  ];

  it("renders rows and opens a row with the keyboard", () => {
    const onRow = vi.fn();
    render(<Table columns={cols} rows={rows} rowKey={(r) => r.id} onRowClick={onRow} rowLabel={(r) => `open ${r.name}`} />);
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Name", "N"]);
    fireEvent.keyDown(screen.getByRole("row", { name: "open beta" }), { key: "Enter" });
    expect(onRow).toHaveBeenCalledWith(rows[1]);
  });

  it("shows the empty state", () => {
    render(<Table columns={cols} rows={[]} rowKey={(r) => r.id} empty="Nothing here" />);
    expect(screen.getByText("Nothing here")).toBeTruthy();
  });
});

describe("Tabs", () => {
  function Harness() {
    const [v, setV] = useState<"1h" | "24h" | "7d">("24h");
    return (
      <Tabs
        label="Window"
        value={v}
        onChange={setV}
        items={[
          { value: "1h", label: "1h" },
          { value: "24h", label: "24h" },
          { value: "7d", label: "7d" },
        ]}
      />
    );
  }
  it("supports arrow-key navigation with roving tabindex", () => {
    render(<Harness />);
    const t24 = screen.getByRole("tab", { name: "24h" });
    expect(t24.getAttribute("aria-selected")).toBe("true");
    expect(t24.getAttribute("tabindex")).toBe("0");
    fireEvent.keyDown(t24, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "7d" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.keyDown(screen.getByRole("tab", { name: "7d" }), { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "1h" }).getAttribute("aria-selected")).toBe("true");
  });
});

describe("Dialog", () => {
  it("renders content when open and closes from the close button", () => {
    const onClose = vi.fn();
    render(
      <Dialog open onClose={onClose} title="Revoke key?">
        body text
      </Dialog>,
    );
    expect(screen.getByText("body text")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Close", hidden: true }));
    expect(onClose).toHaveBeenCalled();
  });
});

describe("Stat, Badge, chart scale", () => {
  it("renders a stat with tone", () => {
    const { container } = render(<Stat label="Integrity failures" value="0" tone="ok" sub="None since start" />);
    expect(container.querySelector(".stat-ok")).not.toBeNull();
    expect(screen.getByText("None since start")).toBeTruthy();
  });
  it("maps HTTP status to badge tones", () => {
    expect(statusTone(200)).toBe("ok");
    expect(statusTone(429)).toBe("warn");
    expect(statusTone(502)).toBe("bad");
    render(<Badge tone="bad">revoked</Badge>);
    expect(screen.getByText("revoked").classList.contains("badge-bad")).toBe(true);
  });
  it("computes nice axis bounds", () => {
    expect(niceScale(287)).toEqual({ max: 300, step: 100 });
    expect(niceScale(0)).toEqual({ max: 1, step: 0.25 });
    expect(niceScale(1450).max).toBeGreaterThanOrEqual(1450);
  });
});
