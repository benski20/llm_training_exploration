// Inline SVG icons (stroke-based, 16px grid). Decorative by default (aria-hidden).

import type { ReactNode, SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...rest }: P & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconOverview = (p: P) => (
  <Svg {...p}>
    <rect x="2" y="2" width="5" height="5" rx="1" />
    <rect x="9" y="2" width="5" height="5" rx="1" />
    <rect x="2" y="9" width="5" height="5" rx="1" />
    <rect x="9" y="9" width="5" height="5" rx="1" />
  </Svg>
);
export const IconChat = (p: P) => (
  <Svg {...p}>
    <path d="M2.5 4.5a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v4.5a2 2 0 0 1-2 2H7l-3 2.5V11h0a2 2 0 0 1-1.5-2z" />
  </Svg>
);
export const IconKey = (p: P) => (
  <Svg {...p}>
    <circle cx="5.5" cy="10.5" r="3" />
    <path d="M7.7 8.3 13.5 2.5M11.5 4.5l1.5 1.5M10 6l1.2 1.2" />
  </Svg>
);
export const IconCube = (p: P) => (
  <Svg {...p}>
    <path d="M8 1.8 13.5 5v6L8 14.2 2.5 11V5z" />
    <path d="M2.5 5 8 8.2 13.5 5M8 8.2v6" />
  </Svg>
);
export const IconLayers = (p: P) => (
  <Svg {...p}>
    <path d="M8 2 14 5.2 8 8.4 2 5.2z" />
    <path d="m2 8 6 3.2L14 8M2 10.8 8 14l6-3.2" />
  </Svg>
);
export const IconList = (p: P) => (
  <Svg {...p}>
    <path d="M5.5 4h8M5.5 8h8M5.5 12h8" />
    <circle cx="2.6" cy="4" r=".6" fill="currentColor" />
    <circle cx="2.6" cy="8" r=".6" fill="currentColor" />
    <circle cx="2.6" cy="12" r=".6" fill="currentColor" />
  </Svg>
);
export const IconShield = (p: P) => (
  <Svg {...p}>
    <path d="M8 1.8 13 3.6v4.1c0 3-2.1 5.4-5 6.5-2.9-1.1-5-3.5-5-6.5V3.6z" />
    <path d="m5.8 8 1.6 1.6L10.4 6.6" />
  </Svg>
);
export const IconSun = (p: P) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="2.8" />
    <path d="M8 1.5v1.3M8 13.2v1.3M1.5 8h1.3M13.2 8h1.3M3.4 3.4l.9.9M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9" />
  </Svg>
);
export const IconMoon = (p: P) => (
  <Svg {...p}>
    <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z" />
  </Svg>
);
export const IconLogout = (p: P) => (
  <Svg {...p}>
    <path d="M6 2.5H3.5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1H6M10.5 11 13.5 8l-3-3M13.5 8H6" />
  </Svg>
);
export const IconCopy = (p: P) => (
  <Svg {...p}>
    <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
    <path d="M10.5 5.5V3.5a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" />
  </Svg>
);
export const IconCheck = (p: P) => (
  <Svg {...p}>
    <path d="m3 8.5 3.2 3L13 4.5" />
  </Svg>
);
export const IconX = (p: P) => (
  <Svg {...p}>
    <path d="m4 4 8 8M12 4l-8 8" />
  </Svg>
);
export const IconPlus = (p: P) => (
  <Svg {...p}>
    <path d="M8 3v10M3 8h10" />
  </Svg>
);
export const IconRefresh = (p: P) => (
  <Svg {...p}>
    <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3" />
  </Svg>
);
export const IconAlert = (p: P) => (
  <Svg {...p}>
    <path d="M8 2 14.5 13.5h-13z" />
    <path d="M8 6.5v3" />
    <circle cx="8" cy="11.5" r=".5" fill="currentColor" />
  </Svg>
);
export const IconInfo = (p: P) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M8 7.2V11" />
    <circle cx="8" cy="5" r=".5" fill="currentColor" />
  </Svg>
);
export const IconLock = (p: P) => (
  <Svg {...p}>
    <rect x="3" y="7" width="10" height="7" rx="1.5" />
    <path d="M5.2 7V5a2.8 2.8 0 0 1 5.6 0v2" />
  </Svg>
);
export const IconEye = (p: P) => (
  <Svg {...p}>
    <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
    <circle cx="8" cy="8" r="2" />
  </Svg>
);
export const IconEyeOff = (p: P) => (
  <Svg {...p}>
    <path d="M6.6 3.7A6 6 0 0 1 8 3.5c4 0 6.5 4.5 6.5 4.5a11 11 0 0 1-1.7 2.2M10.3 10.3A3 3 0 0 1 5.7 5.7M3.6 4.9A11 11 0 0 0 1.5 8S4 12.5 8 12.5a6.6 6.6 0 0 0 3.1-.8M2 2l12 12" />
  </Svg>
);
export const IconSend = (p: P) => (
  <Svg {...p}>
    <path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" />
  </Svg>
);
export const IconStop = (p: P) => (
  <Svg {...p}>
    <rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" stroke="none" />
  </Svg>
);
export const IconChevronDown = (p: P) => (
  <Svg {...p}>
    <path d="m4 6 4 4 4-4" />
  </Svg>
);
export const IconChevronRight = (p: P) => (
  <Svg {...p}>
    <path d="m6 4 4 4-4 4" />
  </Svg>
);
export const IconTrash = (p: P) => (
  <Svg {...p}>
    <path d="M2.5 4h11M6 4V2.8h4V4M4 4l.7 9.2h6.6L12 4" />
  </Svg>
);
export const IconServer = (p: P) => (
  <Svg {...p}>
    <rect x="2" y="2.5" width="12" height="4.5" rx="1" />
    <rect x="2" y="9" width="12" height="4.5" rx="1" />
    <circle cx="4.6" cy="4.75" r=".5" fill="currentColor" />
    <circle cx="4.6" cy="11.25" r=".5" fill="currentColor" />
  </Svg>
);
export const IconFile = (p: P) => (
  <Svg {...p}>
    <path d="M9 1.8H4.5a1 1 0 0 0-1 1v10.4a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V5.3z" />
    <path d="M9 1.8v3.5h3.5" />
  </Svg>
);

export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false" className="logo-mark">
      <rect width="32" height="32" rx="8" className="logo-bg" />
      <path d="M10 23V9.5h6.6a4.6 4.6 0 0 1 0 9.2H13.4" fill="none" className="logo-fg" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="22.6" cy="22.4" r="2.3" className="logo-dot" />
    </svg>
  );
}
