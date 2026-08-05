import type { Dimension } from "./mock-session";

/**
 * One dimension-to-color mapping, used everywhere a score dimension needs a
 * swatch. Each entry is a `var()` reference into app.css's `--dim-*` custom
 * properties, not a literal hex — inline `style={{ backgroundColor }}`
 * consumers therefore repaint automatically on a theme change, the same as
 * any Tailwind utility class would.
 */
export const DIMENSION_COLOR_VAR: Record<Dimension, string> = {
	dynamics: "var(--dim-dynamics)",
	timing: "var(--dim-timing)",
	pedaling: "var(--dim-pedaling)",
	articulation: "var(--dim-articulation)",
	phrasing: "var(--dim-phrasing)",
	interpretation: "var(--dim-interpretation)",
};
