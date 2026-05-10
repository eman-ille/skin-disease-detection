---
name: Peach Glow Design System
colors:
  surface: '#fff8f6'
  surface-dim: '#ecd6cb'
  surface-bright: '#fff8f6'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#fff1eb'
  surface-container: '#ffeae0'
  surface-container-high: '#fae4d9'
  surface-container-highest: '#f5ded3'
  on-surface: '#251913'
  on-surface-variant: '#584237'
  inverse-surface: '#3b2e27'
  inverse-on-surface: '#ffede5'
  outline: '#8b7265'
  outline-variant: '#dfc0b1'
  surface-tint: '#9b4500'
  primary: '#9b4500'
  on-primary: '#ffffff'
  primary-container: '#ec7113'
  on-primary-container: '#4e1f00'
  inverse-primary: '#ffb68d'
  secondary: '#7b5739'
  on-secondary: '#ffffff'
  secondary-container: '#ffcda8'
  on-secondary-container: '#7a5538'
  tertiary: '#6c5b4c'
  on-tertiary: '#ffffff'
  tertiary-container: '#a59180'
  on-tertiary-container: '#382b1e'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdbc9'
  primary-fixed-dim: '#ffb68d'
  on-primary-fixed: '#331200'
  on-primary-fixed-variant: '#763300'
  secondary-fixed: '#ffdcc3'
  secondary-fixed-dim: '#edbd98'
  on-secondary-fixed: '#2e1501'
  on-secondary-fixed-variant: '#613f24'
  tertiary-fixed: '#f6decb'
  tertiary-fixed-dim: '#d9c2b0'
  on-tertiary-fixed: '#25190e'
  on-tertiary-fixed-variant: '#534436'
  background: '#fff8f6'
  on-background: '#251913'
  surface-variant: '#f5ded3'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  display-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: '0'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: '0'
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
  data-viz:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '500'
    lineHeight: '1'
    letterSpacing: -0.01em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
  container-margin: 32px
  gutter: 20px
---

## Brand & Style

This design system is engineered for a medical AI dashboard that prioritizes clarity, warmth, and high-trust interactions. The aesthetic blends **Minimalism** with a **Corporate Modern** foundation, moving away from sterile clinical whites toward a sophisticated, organic palette that reduces cognitive load for healthcare professionals.

The emotional response is one of "calm precision." By using soft tones and ample whitespace, the UI eliminates the anxiety typically associated with medical data environments, replacing it with a supportive, high-end editorial feel that emphasizes AI-driven insights without overwhelming the user.

## Colors

The color palette centers on a "warm-neutral" spectrum to ensure long-term legibility and comfort. 

- **Primary Accent (#EC7113):** Reserved for high-priority actions, critical data points, and active states. 
- **Secondary Accent (#F7C6A1):** Used for decorative elements, progress bars, and subtle highlights.
- **Warning/Disclaimer (#FBE3D0):** Employed as a background fill for notices or non-critical alerts to maintain the soft aesthetic while indicating importance.
- **Background & Surface:** The contrast between the cream background and pure white cards creates a gentle "lift" that organizes the dashboard without the need for harsh borders.

## Typography

The design system utilizes **Inter** for its exceptional legibility in data-dense environments. Hierarchy is established through weight and color rather than drastic size shifts.

- **Headlines:** Use `#2F1704` with semi-bold weights and slight negative letter-spacing to ground the page.
- **Body Text:** Use `#5E2D08` for secondary information to reduce visual harshness while maintaining WCAG AA accessibility.
- **Data Points:** Numbers within the AI dashboard should utilize slightly tighter tracking and medium weights to appear "clinical" and precise.

## Layout & Spacing

This design system uses a **Fixed Grid** layout for the main content area (max-width 1440px) to ensure consistency in medical data visualization. 

The spacing rhythm is based on a **4px baseline grid**. 
- **Margins:** 32px outer margins provide a breathable frame for the dashboard.
- **Gutter:** 20px gutters between cards allow the soft shadows to breathe without overlapping.
- **Padding:** Internal card padding should be a minimum of 24px (`lg`) to maintain a sophisticated, minimalist feel.

## Elevation & Depth

Visual hierarchy is managed through **Ambient Shadows** and tonal layering. 

- **Level 0 (Background):** #FDF1E7 (Flat).
- **Level 1 (Cards/Surface):** #FFFFFF with a soft shadow: `0 4px 12px rgba(94, 45, 8, 0.08)`. The shadow uses a brown-tinted alpha rather than pure black to keep the "Peach Glow" warmth intact.
- **Level 2 (Active/Hover):** Modest elevation increase; shadow shifts to `0 8px 20px rgba(94, 45, 8, 0.12)` to simulate a physical lift when a user interacts with a module.
- **Interaction:** No heavy borders are used; depth is created entirely through the interplay of the cream background and the shadowed white surfaces.

## Shapes

The shape language is consistently **Rounded**, reflecting a modern and approachable medical aesthetic. 

- **Base Radius:** 12px for cards, large containers, and primary dashboard modules.
- **Small Radius:** 8px for buttons, input fields, and checkboxes.
- **Pill:** Used exclusively for tags, status indicators (e.g., "Active," "Pending"), and toggle switches.

Avoid sharp corners entirely to maintain the soft, human-centric focus of the design system.

## Components

### Buttons
- **Primary:** Solid `#EC7113` with white text. 8px radius. High-contrast for clear CTAs.
- **Secondary:** Transparent background with a 1px border of `#F7C6A1`.
- **Ghost:** No border, text color `#5E2D08`. Used for tertiary actions.

### Cards
- Pure white background with 12px radius. Must include the system's signature soft shadow. Used for grouping patient data, AI insights, and lab results.

### Chips & Tags
- **Informational:** Background `#FBE3D0`, text `#5E2D08`, pill-shaped.
- **Status (Positive):** Pale sage background with dark green text (used sparingly for health indicators).

### Input Fields
- White background, 1px border in `#F7C6A1`, 8px radius. On focus, border thickens to 2px in `#EC7113`.

### Medical-Specific Components
- **Trend Indicators:** Small sparklines using `#EC7113` for data visualization.
- **AI Insight Callouts:** Cards with a left-accent border (4px) in `#EC7113` to denote AI-generated suggestions.
- **Disclaimer Banner:** A subtle `#FBE3D0` top-bar with `label-caps` typography for legal or medical cautionary notes.