# NorthStar Energy Solutions: Enterprise Overhaul Implementation Plan

## Phase 1: Metadata & "Rookie Site" Eradication
- **Objective:** Ensure the site looks like a billion-dollar enterprise to scrapers, search engines, and users.
- **Actions:**
  - Create `/app/robots.txt` and `/app/sitemap.ts` for native SEO routing.
  - Create `/app/manifest.ts` for PWA capabilities.
  - Overhaul `layout.tsx` metadata (OpenGraph images, Twitter cards, canonical URLs, precise description tags).
  - Enforce clean HTML hierarchies and remove any residual console warnings or unoptimized layout shifts.

## Phase 2: Content Merger (Well Experts & Adriatic)
- **Objective:** Integrate newly discovered milestones and service capabilities from the legacy domains.
- **Actions:**
  - Update `company.ts` to include recent 2025 milestones: IWCF Primary Accredited Training Company, IADC Member, and IADC WellSharp® Provider.
  - Highlight the Oman Expansion (RDS-100 deployment at Ryboa Hima).
  - Explicitly add Machine Learning, AI, and Data Analytics to the Digital Solutions pillar.
  - Add missing Adriatic capabilities: Asset Integrity Management, Pressure Testing, Advanced vs. Conventional NDT, and Rig Audits.
  - Refine tone to be strictly B2B procurement-focused (no fluff, pure capability assertion).

## Phase 3: 3D Asset & Visual Upgrade
- **Objective:** Make the 3D WebGL scenes look cinematic, expensive, and photo-real.
- **Actions:**
  - Inject `@react-three/postprocessing` into `ServiceDiorama.tsx` for high-end Bloom, Ambient Occlusion, and Vignette effects.
  - Upgrade environment lighting to use HDRI studio/city presets.
  - Refine material shaders (PBR metallic/roughness tweaking) so assets look like solid, polished industrial equipment rather than raw CAD imports.

## Phase 4: UI/UX & Polish
- **Objective:** Ensure interaction design matches top-tier corporate sites.
- **Actions:**
  - Clean up responsive breakpoints (padding, font scaling).
  - Refine the Client Marquee and Impact Metrics spacing.
  - Conduct a full structural audit to guarantee zero horizontal scroll leaks and perfect symmetry.
