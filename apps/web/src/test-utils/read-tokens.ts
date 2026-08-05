import { readFileSync } from "node:fs";
import { fileURLToPath, URL as NodeURL } from "node:url";

const APP_CSS_PATH = fileURLToPath(
	new NodeURL("../styles/app.css", import.meta.url),
);

/** Pulls `--name: value;` declarations out of one `{ ... }` block of raw CSS text. */
function parseDeclarationBlock(blockBody: string): Record<string, string> {
	const table: Record<string, string> = {};
	const re = /--([a-zA-Z0-9-]+):\s*([^;]+);/g;
	let match: RegExpExecArray | null;
	// biome-ignore lint/suspicious/noAssignInExpressions: standard regex exec loop
	while ((match = re.exec(blockBody)) !== null) {
		table[match[1]] = match[2].trim();
	}
	return table;
}

/** Extracts the body of the first top-level `selector { ... }` block matching `selectorRe`. */
function extractBlock(css: string, selectorRe: RegExp): string | null {
	const match = selectorRe.exec(css);
	if (!match) return null;
	const start = match.index + match[0].length;
	let depth = 1;
	let i = start;
	while (i < css.length && depth > 0) {
		if (css[i] === "{") depth++;
		if (css[i] === "}") depth--;
		i++;
	}
	return css.slice(start, i - 1);
}

/**
 * Reads the two-column token table from app.css. Light is the `@theme` base
 * block; dark overlays `html[data-theme="dark"]` on top of it. Returns a flat
 * map of bare variable name (no `--`) to its declared value.
 */
export function readTokenTable(
	theme: "light" | "dark",
): Record<string, string> {
	const css = readFileSync(APP_CSS_PATH, "utf-8");

	const themeBlock = extractBlock(css, /@theme\s*\{/);
	const base = themeBlock ? parseDeclarationBlock(themeBlock) : {};

	if (theme === "light") return base;

	const darkBlock = extractBlock(css, /html\[data-theme=["']dark["']\]\s*\{/);
	const darkOverrides = darkBlock ? parseDeclarationBlock(darkBlock) : {};

	return { ...base, ...darkOverrides };
}
