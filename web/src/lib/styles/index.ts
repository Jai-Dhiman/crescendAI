import { type Colors, colors } from "./colors";
import {
  type BorderRadius,
  borderRadius,
  type Layout,
  layout,
  type Spacing,
  spacing,
} from "./spacing";
import {
  fontFamilies,
  fontSizes,
  fontWeights,
  type Typography,
  typography,
} from "./typography";

// Main theme object
export const theme = {
  colors,
  typography,
  fontSizes,
  fontWeights,
  fontFamilies,
  spacing,
  borderRadius,
  layout,
} as const;

// Theme type
export type Theme = {
  colors: Colors;
  typography: Typography;
  spacing: Spacing;
  borderRadius: BorderRadius;
  layout: Layout;
};

// Export individual theme parts
export {
  colors,
  typography,
  fontSizes,
  fontWeights,
  fontFamilies,
  spacing,
  borderRadius,
  layout,
};
export type { Colors, Typography, Spacing, BorderRadius, Layout };

// CSS box shadow styles for web
export const shadows = {
  sm: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
  md: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
  lg: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
  xl: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
} as const;

export type Shadows = typeof shadows;