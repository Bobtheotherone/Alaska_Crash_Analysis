/**
 * Design Tokens (TypeScript Export)
 *
 * This file mirrors the CSS variables defined in tokens.css for use in TypeScript contexts
 * (e.g., if using CSS-in-JS solutions like styled-components or Emotion).
 *
 * Note: These tokens were derived based on the visual mockups in the provided PDFs,
 * as explicit specifications were not available.
 */

export const Colors = {
  // Primary (Derived professional blue)
  primary: '#00529b',
  primaryDark: '#003366',
  secondary: '#f0a800',

  // Neutrals & Surfaces
  background: '#f5f7fa',
  surface1: '#ffffff',
  surface2: '#f0f2f5',
  border: '#e0e0e0',
  disabled: '#b0bec5',

  // Typography
  textPrimary: '#212121',
  textSecondary: '#616161',
  textOnPrimary: '#ffffff',

  // Feedback/Status
  success: '#388e3c',
  error: '#d32f2f',
  warning: '#f57c00',
  info: '#0288d1',
};

export const Typography = {
  familyBase:
    "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif",
  familyMono: "'Roboto Mono', monospace",

  size: {
    sm: '0.875rem', // 14px
    base: '1rem', // 16px
    md: '1.125rem', // 18px
    lg: '1.25rem', // 20px
    xl: '1.5rem', // 24px
    xxl: '2rem', // 32px
  },

  weight: {
    regular: 400,
    medium: 500,
    bold: 700,
  },

  lineHeight: {
    base: 1.5,
    tight: 1.25,
  },
};

// 8px Grid System
export const Spacing = {
  s1: '0.25rem', // 4px
  s2: '0.5rem', // 8px
  s3: '0.75rem', // 12px
  s4: '1rem', // 16px
  s5: '1.5rem', // 24px
  s6: '2rem', // 32px
  s7: '3rem', // 48px
  s8: '4rem', // 64px
};

export const BorderRadius = {
  sm: '4px',
  md: '8px',
  lg: '16px',
};

export const Elevation = {
  e1: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
  e2: '0 3px 6px rgba(0, 0, 0, 0.16), 0 3px 6px rgba(0, 0, 0, 0.23)',
  e3: '0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23)',
};

export const DesignTokens = {
  Colors,
  Typography,
  Spacing,
  BorderRadius,
  Elevation,
};