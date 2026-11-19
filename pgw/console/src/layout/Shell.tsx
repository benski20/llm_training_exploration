import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useGateway } from "../app/gateway";
import { useSession } from "../app/session";
import { fmtDuration } from "../lib/format";
import { useTheme } from "../lib/theme";
import {
  Badge,
  Button,
  IconChat,
  IconCube,
  IconKey,
  IconLayers,
  IconList,
  IconLogout,
  IconMoon,
  IconOverview,
  IconShield,
  IconSun,
  Logo,
  Select,
  StatusDot,
  modelTone,
} from "../ui";

const NAV = [
  {
    group: "Monitor",
    items: [
      { to: "/", label: "Overview", icon: <IconOverview />, end: true },
      { to: "/models", label: "Models", icon: <IconCube /> },
      { to: "/masks", label: "Masks", icon: <IconLayers /> },
      { to: "/audit", label: "Audit log", icon: <IconList /> },
    ],
  },
  {
    group: "Build",
    items: [
      { to: "/playground", label: "Playground", icon: <IconChat /> },
      { to: "/keys", label: "API keys", icon: <IconKey /> },
    ],
  },
  {
    group: "Trust",
    items: [{ to: "/security", label: "Security", icon: <IconShield /> }],
  },
];

const TITLES: Record<string, string> = {
  "/": "Overview",
  "/models": "Models",
  "/masks": "Masks",
  "/audit": "Audit log",
  "/playground": "Playground",
  "/keys": "API keys",
  "/security": "Security",
};

export function Shell() {
  const session = useSession();
  const gw = useGateway();
  const [theme, toggleTheme] = useTheme();
  const loc = useLocation();
  const st = gw.status.data;
  const production = st?.production_policy;

  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <Logo />
          <span className="brand-name">
            PGW <span className="brand-sub">Console</span>
          </span>
        </div>
        <nav className="nav">
          {NAV.map((g) => (
            <div className="nav-group" key={g.group}>
              <div className="nav-group-label">{g.group}</div>
              <ul>
                {g.items.map((it) => (
                  <li key={it.to}>
                    <NavLink to={it.to} end={it.end} className={({ isActive }) => (isActive ? "nav-link is-active" : "nav-link")} title={it.label}>
                      <span className="nav-icon">{it.icon}</span>
                      <span className="nav-label">{it.label}</span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="sidebar-models" aria-label="Model health">
            {gw.models.map((m) => (
              <div className="sidebar-model" key={m.name} title={`${m.name}: ${m.ready ? "ready" : m.reason || m.state}`}>
                <StatusDot tone={modelTone(m.ready, m.state)} />
                <span className="sidebar-model-name">{m.name}</span>
              </div>
            ))}
          </div>
          <div className="sidebar-meta">
            <span>{session.version ? `v${session.version}` : "PGW"}</span>
            {st ? <span>up {fmtDuration(st.uptime_s)}</span> : null}
          </div>
        </div>
      </aside>

      <div className="main-col">
        <header className="topbar">
          <div className="topbar-title">{TITLES[loc.pathname] ?? ""}</div>
          <div className="topbar-right">
            {gw.models.length ? (
              <label className="model-picker">
                <span className="sr-only">Model</span>
                <StatusDot tone={modelTone(gw.selected?.ready, gw.selected?.state)} />
                <Select size_="sm" value={gw.model ?? ""} onChange={(e) => gw.setModel(e.target.value)} aria-label="Selected model">
                  {gw.models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))}
                </Select>
              </label>
            ) : null}
            {production === undefined ? null : production ? (
              <Badge tone="neutral" dot className="env-badge" title="Production policy: dev modes are refused">
                Production
              </Badge>
            ) : (
              <Badge tone="warn" dot className="env-badge" title="Dev modes are allowed on this gateway">
                Development
              </Badge>
            )}
            <span className="topbar-sep" aria-hidden="true" />
            {session.user ? (
              <span className="topbar-user hide-sm" title={session.user.email ? `Signed in with SSO as ${session.user.email}` : "Signed in with SSO"}>
                {session.user.name || session.user.email || session.user.sub}
              </span>
            ) : null}
            <Badge tone={session.isAdmin ? "accent" : "neutral"} title={session.isAdmin ? "Full access" : "Read-only access"}>
              {session.isAdmin ? "Admin" : "Viewer"}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              icon={theme === "dark" ? <IconSun /> : <IconMoon />}
              onClick={toggleTheme}
            />
            <Button variant="ghost" size="sm" iconOnly label="Sign out" icon={<IconLogout />} onClick={() => void session.logout()} />
          </div>
        </header>
        <main id="main" className="content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
