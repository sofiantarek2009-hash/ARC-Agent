/* ═══════════════════════════════════════════════
   NORTHSTAR ENERGY SOLUTIONS — SHARED TYPES
   ═══════════════════════════════════════════════ */

// ── Company Data ──────────────────────────────

export interface CompanyStat {
  value: number;
  suffix: string;
  label: string;
}

export interface ValueProp {
  title: string;
  desc: string;
}

export interface Hub {
  name: string;
  desc: string;
}

export interface ServiceSubcategory {
  name: string;
  desc: string;
}

export interface ServiceCategory {
  id: string;
  name: string;
  subcategories: ServiceSubcategory[];
}

export interface ChemicalCategory {
  name: string;
  products: string[];
}

export interface Certification {
  name: string;
  scope: string;
  domain: string;
}

export interface Client {
  name: string;
  domain: string;
}

// ── 3D / WebGL ────────────────────────────────

export interface GLSLUniform {
  type: "float" | "vec2" | "vec3" | "vec4" | "mat4";
  value: number | number[];
}

export interface CameraWaypoint {
  position: [number, number, number];
  lookAt: [number, number, number];
  scrollProgress: number; // 0.0 – 1.0
}

// ── Telemetry / HUD ───────────────────────────

export interface TelemetryMetric {
  key: string;
  value: string;
  unit?: string;
  color?: string;
}

export interface InspectionSpec {
  standard: string; // e.g. "API 4G Cat III"
  title: string;
  description: string;
  depth?: string; // scroll depth trigger
}

// ── Showcase ──────────────────────────────────

export interface ShowcasePillar {
  id: string;
  title: string;
  subtitle: string;
  image: string;
  color: string;
  metrics: Record<string, string>;
  description: string;
}
