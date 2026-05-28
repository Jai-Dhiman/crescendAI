// apps/api/scripts/patch-wasm-shims.ts
//
// Replaces the wasm-pack --target bundler shim JS with the pattern documented
// by Cloudflare for Workers compatibility:
//   https://developers.cloudflare.com/workers/languages/rust
//
// Problem: wasm-pack --target bundler emits:
//   import * as wasm from "./score_analysis_bg.wasm";
//   __wbg_set_wasm(wasm);
//
// Under workerd, importing a .wasm file yields a WebAssembly.Module, NOT a
// WebAssembly.Instance with callable exports. So wasm.malloc, wasm.ngram_recall,
// etc. are all undefined — they are not properties of a Module object. The
// original __wbindgen_start boot error was also from this: wasm.__wbindgen_start
// is undefined because wasm is a Module, not an Instance.
//
// Fix: import the .wasm as a Module, call new WebAssembly.Instance(mod, imports)
// ourselves, then pass instance.exports to __wbg_set_wasm. This is the exact
// pattern from the Cloudflare official docs. Runs as the final step of
// `bun run build:wasm`. Idempotent.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const shims: Array<{ js: string; bgJs: string; bgWasm: string }> = [
	{
		js: "src/wasm/score-analysis/pkg/score_analysis.js",
		bgJs: "./score_analysis_bg.js",
		bgWasm: "./score_analysis_bg.wasm",
	},
	{
		js: "src/wasm/piece-identify/pkg/piece_identify.js",
		bgJs: "./piece_identify_bg.js",
		bgWasm: "./piece_identify_bg.wasm",
	},
];

// Sentinel to detect already-patched files so this script is idempotent.
const PATCHED_MARKER = "// @cloudflare-workers-patched";

for (const { js, bgJs, bgWasm } of shims) {
	const path = resolve(js);
	const before = readFileSync(path, "utf8");

	if (before.includes(PATCHED_MARKER)) {
		console.log(`already patched: ${js}`);
		continue;
	}

	// The re-exports line wasm-pack emits (all exports from bg.js):
	//   export { foo, bar } from "./xxx_bg.js";
	// We keep that — only the instantiation preamble changes.
	const exportsMatch = before.match(/^export \{[^}]+\} from "[^"]+";$/m);
	if (!exportsMatch) {
		console.error(`unexpected shim shape in ${js} — could not find re-exports line`);
		process.exit(1);
	}
	const reExportsLine = exportsMatch[0];

	// Build the CF-documented replacement for the entire shim.
	const after = `${PATCHED_MARKER}
// Cloudflare Workers compatible wasm-bindgen shim.
// Workers import .wasm as WebAssembly.Module; we must instantiate ourselves.
// See: https://developers.cloudflare.com/workers/languages/rust

import * as imports from "${bgJs}";
import wkmod from "${bgWasm}";

const instance = new WebAssembly.Instance(wkmod, { "${bgJs}": imports });
imports.__wbg_set_wasm(instance.exports);

${reExportsLine}
`;

	writeFileSync(path, after);
	console.log(`patched: ${js}`);
}
