// Spacing scale following design system
export const spacing = {
  0: 0,
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  8: 32,
  10: 40,
  12: 48,
  16: 64,
  20: 80,
  24: 96,
  32: 128,
  40: 160,
  48: 192,
  56: 224,
  64: 256,
  // Semantic names
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

// Border radius values
export const borderRadius = {
  none: 0,
  sm: 4, // --radius-sm: calc(var(--radius) - 4px) = 6px
  md: 8, // --radius-md: calc(var(--radius) - 2px) = 8px
  lg: 10, // --radius-lg: var(--radius) = 10px (0.625rem = 10px)
  xl: 14, // --radius-xl: calc(var(--radius) + 4px) = 14px
  "2xl": 16,
  "3xl": 24,
  full: 9999,
} as const;

// Layout dimensions
export const layout = {
  // Common component heights
  buttonHeight: {
    sm: 32,
    md: 40,
    lg: 48,
    xl: 56,
  },

  // Input heights
  inputHeight: {
    sm: 32,
    md: 40,
    lg: 48,
  },

  // Icon sizes
  iconSize: {
    xs: 12,
    sm: 16,
    md: 20,
    lg: 24,
    xl: 32,
    "2xl": 40,
  },

  // Common widths
  maxWidth: {
    xs: 320,
    sm: 384,
    md: 448,
    lg: 512,
    xl: 576,
    "2xl": 672,
    "3xl": 768,
    "4xl": 896,
    "5xl": 1024,
    "6xl": 1152,
  },
} as const;

export type Spacing = typeof spacing;
export type BorderRadius = typeof borderRadius;
export type Layout = typeof layout;