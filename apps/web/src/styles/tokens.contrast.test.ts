import { describe, expect, it } from "vitest";
import { contrastRatio } from "../test-utils/contrast";
import { readTokenTable } from "../test-utils/read-tokens";

// [fg, bg] — 4.5 for text, 3.0 for non-text UI (borders on surfaces).
const TEXT_PAIRS: Array<[string, string]> = [
	["color-ink-primary", "color-surface-page"],
	["color-ink-primary", "color-surface-raised"],
	["color-ink-secondary", "color-surface-page"],
	["color-ink-secondary", "color-surface-raised"],
	["color-ink-tertiary", "color-surface-page"],
	["color-ink-tertiary", "color-surface-raised"],
	["color-on-accent", "color-accent"],
	["color-danger", "color-surface-page"],
	["color-warn", "color-surface-page"],
];

// color-border-subtle is deliberately absent. WCAG 1.4.11 requires 3:1 only
// for boundaries needed to IDENTIFY a component or its state; card edges and
// dividers are decorative and exempt. A divider cannot be both subtle and
// 3:1, so asserting it here would be asserting a contradiction.
const UI_PAIRS: Array<[string, string]> = [
	["color-border-strong", "color-surface-page"],
];

describe.each(["light", "dark"] as const)("token contrast (%s)", (theme) => {
	const table = readTokenTable(theme);

	it.each(TEXT_PAIRS)("%s on %s clears 4.5:1", (fgKey, bgKey) => {
		const fg = table[fgKey];
		const bg = table[bgKey];
		expect(fg, `${fgKey} is not declared for ${theme}`).toBeDefined();
		expect(bg, `${bgKey} is not declared for ${theme}`).toBeDefined();
		expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);
	});

	it.each(UI_PAIRS)("%s on %s clears 3:1", (fgKey, bgKey) => {
		const fg = table[fgKey];
		const bg = table[bgKey];
		expect(fg, `${fgKey} is not declared for ${theme}`).toBeDefined();
		expect(bg, `${bgKey} is not declared for ${theme}`).toBeDefined();
		expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(3);
	});
});
