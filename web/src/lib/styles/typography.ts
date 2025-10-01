// Font families with fallbacks for different platforms
export const fontFamilies = {
  default: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
  mono: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
} as const;

// Font weights
export const fontWeights = {
  light: "300" as const,
  normal: "400" as const,
  medium: "500" as const,
  semiBold: "600" as const,
  bold: "700" as const,
};

// Font sizes following design system
export const fontSizes = {
  xs: 12,
  sm: 14,
  base: 16,
  lg: 18,
  xl: 20,
  "2xl": 24,
  "3xl": 30,
  "4xl": 36,
  "5xl": 48,
} as const;

// Line heights - using pixel values for React Native compatibility
export const lineHeights = {
  none: undefined, // Let CSS handle it
  tight: 20,
  snug: 22,
  normal: 24,
  relaxed: 26,
  loose: 32,
} as const;

// Typography variants based on design system
export const typography = {
  // Headings
  h1: {
    fontSize: fontSizes["4xl"],
    fontWeight: fontWeights.medium,
    lineHeight: 40,
    fontFamily: fontFamilies.default,
  },
  h2: {
    fontSize: fontSizes["3xl"],
    fontWeight: fontWeights.medium,
    lineHeight: 36,
    fontFamily: fontFamilies.default,
  },
  h3: {
    fontSize: fontSizes["2xl"],
    fontWeight: fontWeights.medium,
    lineHeight: 28,
    fontFamily: fontFamilies.default,
  },
  h4: {
    fontSize: fontSizes.xl,
    fontWeight: fontWeights.medium,
    lineHeight: 24,
    fontFamily: fontFamilies.default,
  },
  h5: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.medium,
    lineHeight: 22,
    fontFamily: fontFamilies.default,
  },
  h6: {
    fontSize: fontSizes.base,
    fontWeight: fontWeights.medium,
    lineHeight: 20,
    fontFamily: fontFamilies.default,
  },

  // Body text
  body: {
    fontSize: fontSizes.base,
    fontWeight: fontWeights.normal,
    lineHeight: 20,
    fontFamily: fontFamilies.default,
  },
  bodyLarge: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.normal,
    lineHeight: 22,
    fontFamily: fontFamilies.default,
  },
  bodySmall: {
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.normal,
    lineHeight: 18,
    fontFamily: fontFamilies.default,
  },

  // Labels and captions
  label: {
    fontSize: fontSizes.base,
    fontWeight: fontWeights.medium,
    lineHeight: 20,
    fontFamily: fontFamilies.default,
  },
  labelSmall: {
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.medium,
    lineHeight: 18,
    fontFamily: fontFamilies.default,
  },
  caption: {
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.normal,
    lineHeight: 18,
    fontFamily: fontFamilies.default,
  },
  small: {
    fontSize: fontSizes.xs,
    fontWeight: fontWeights.normal,
    lineHeight: 16,
    fontFamily: fontFamilies.default,
  },

  // Button text
  button: {
    fontSize: fontSizes.base,
    fontWeight: fontWeights.medium,
    lineHeight: 20,
    fontFamily: fontFamilies.default,
  },
  buttonLarge: {
    fontSize: fontSizes.lg,
    fontWeight: fontWeights.medium,
    lineHeight: 22,
    fontFamily: fontFamilies.default,
  },
  buttonSmall: {
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.medium,
    lineHeight: 18,
    fontFamily: fontFamilies.default,
  },

  // Monospace text
  code: {
    fontSize: fontSizes.sm,
    fontWeight: fontWeights.normal,
    lineHeight: 18,
    fontFamily: fontFamilies.mono,
  },
} as const;

export type Typography = typeof typography;