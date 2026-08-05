// WCAG 2.x relative luminance + contrast ratio.
// https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html

function srgbToLinear(channel: number): number {
	const c = channel / 255;
	return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function hexToRgb(hex: string): [number, number, number] {
	const clean = hex.replace("#", "");
	const r = Number.parseInt(clean.slice(0, 2), 16);
	const g = Number.parseInt(clean.slice(2, 4), 16);
	const b = Number.parseInt(clean.slice(4, 6), 16);
	return [r, g, b];
}

function relativeLuminance(hex: string): number {
	const [r, g, b] = hexToRgb(hex);
	const [rl, gl, bl] = [r, g, b].map(srgbToLinear);
	return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}

/** WCAG contrast ratio between two hex colors, from 1 (no contrast) to 21 (max). */
export function contrastRatio(fg: string, bg: string): number {
	const l1 = relativeLuminance(fg);
	const l2 = relativeLuminance(bg);
	const lighter = Math.max(l1, l2);
	const darker = Math.min(l1, l2);
	return (lighter + 0.05) / (darker + 0.05);
}
