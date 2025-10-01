// Design tokens extracted from Figma design
export const colors = {
  // Primary colors
  background: "#ffffff",
  foreground: "#1a1a1a", // Converted from oklch(0.145 0 0)

  // Card colors
  card: "#ffffff",
  cardForeground: "#1a1a1a",

  // Primary brand colors
  primary: "#030213", // Dark navy from design
  primaryForeground: "#ffffff",

  // Secondary colors
  secondary: "#f1f2f6", // Converted from oklch(0.95 0.0058 264.53)
  secondaryForeground: "#030213",

  // Muted colors
  muted: "#ececf0",
  mutedForeground: "#717182",

  // Accent colors
  accent: "#e9ebef",
  accentForeground: "#030213",

  // Destructive/error colors
  destructive: "#d4183d",
  destructiveForeground: "#ffffff",
  error: "#d4183d",
  success: "#84cc16",
  warning: "#f59e0b",

  // Border and input colors
  border: "rgba(0, 0, 0, 0.1)",
  inputBackground: "#f3f3f5",
  switchBackground: "#cbced4",
  surface: "#f8fafc",

  // Text colors
  text: "#1a1a1a",
  textSecondary: "#717182",

  // Slate palette (from the design)
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
  },

  // Blue palette
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
  },

  // Emerald palette
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
  },

  // Purple palette
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7c3aed",
    800: "#6b21a8",
    900: "#581c87",
  },

  // Amber palette
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
  },

  // Red palette
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
  },

  // Chart colors
  chart: {
    1: "#f97316", // Converted from oklch values
    2: "#06b6d4",
    3: "#3b82f6",
    4: "#84cc16",
    5: "#f59e0b",
  },

  // Dark theme colors
  dark: {
    background: "#1a1a1a", // Converted from oklch(0.145 0 0)
    foreground: "#fafafa", // Converted from oklch(0.985 0 0)
    card: "#1a1a1a",
    cardForeground: "#fafafa",
    primary: "#fafafa",
    primaryForeground: "#2a2a2a",
    secondary: "#404040", // Converted from oklch(0.269 0 0)
    secondaryForeground: "#fafafa",
    muted: "#404040",
    mutedForeground: "#a1a1aa",
    accent: "#404040",
    accentForeground: "#fafafa",
    border: "#404040",
    input: "#404040",
  },
} as const;

export type Colors = typeof colors;