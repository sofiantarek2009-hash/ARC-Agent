/* ═══════════════════════════════════════════════
   NORTHSTAR — SHARED CONSTANTS
   Animation configs, breakpoints, z-index scale
   ═══════════════════════════════════════════════ */

// ── Spring Physics (Emil Kowalski Protocol) ───
// GSAP handles WebGL scroll. Framer Motion handles DOM.

export const SPRING = {
  /** Default UI spring — snappy but not jarring */
  default: { type: "spring" as const, stiffness: 350, damping: 30 },
  /** Soft spring for large layout shifts */
  soft: { type: "spring" as const, stiffness: 200, damping: 25 },
  /** Tight spring for hover indicators */
  tight: { type: "spring" as const, stiffness: 400, damping: 28 },
  /** Heavy spring for large-mass elements */
  heavy: { type: "spring" as const, stiffness: 150, damping: 20 },
} as const;

// ── GSAP Easing ───────────────────────────────
// Used for WebGL camera paths & scroll-driven 3D
export const GSAP_EASE = {
  smooth: "power2.inOut",
  sharp: "power3.out",
  elastic: "elastic.out(1, 0.5)",
  expo: "expo.out",
} as const;

// ── Breakpoints (matches Tailwind defaults) ───
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
} as const;

// ── Z-Index Scale ─────────────────────────────
export const Z = {
  canvas: 0,          // WebGL canvases
  background: 1,      // Background patterns (DotPattern)
  content: 10,        // Main page content
  overlay: 20,        // Glassmorphic HUD overlays
  navbar: 50,         // EliteNavbar
  modal: 100,         // Modals & full-screen menus
  tooltip: 200,       // Tooltips
} as const;

// ── Color Constants (hex for Three.js) ────────
export const COLORS = {
  petroleum: "#040D14",
  charcoal: "#0D1117",
  neonCyan: "#00F2FE",
  neonCyanDim: "#00B4BD",
  hazardOrange: "#FF5722",
  hazardOrangeDim: "#BF360C",
  steel: "#8B949E",
  white: "#FFFFFF",
} as const;

// ── Scroll Section IDs ────────────────────────
export const SECTIONS = {
  hero: "hero",
  vision: "vision",
  showcase: "showcase",
  simulator: "simulator",
  services: "services",
  offshore: "offshore",
  chemicals: "chemicals",
  ecosystem: "ecosystem",
  contact: "contact",
} as const;
