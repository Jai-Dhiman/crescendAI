// Integration test: loads real Verovio against a real MusicXML fixture and
// verifies the worker's clip pipeline actually crops the rendered layout.
//
// This test exists because the unit tests in score-worker.test.ts mock Verovio
// entirely — they cannot detect when a Verovio API call (e.g. select()) returns
// success but produces an uncropped layout. Phase 1 shipped with that mismatch
// in prod: every clip rendered as the full piece. See investigation report.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const FIXTURE_PATH = resolve(
	dirname(fileURLToPath(import.meta.url)),
	"../../public/scores/chopin-nocturne-op9-no2.mxl",
);

function countMeasures(svg: string): number {
	const matches = svg.match(/<g [^>]*class="[^"]*\bmeasure\b/g) ?? [];
	return matches.length;
}

function firstMeasureId(svg: string): string | null {
	// Find the first <g ...> opening tag that has class="...measure..." and
	// extract its id attribute regardless of attribute order.
	for (const m of svg.matchAll(/<g\s+([^>]*)>/g)) {
		const attrs = m[1];
		if (!/class="[^"]*\bmeasure\b/.test(attrs)) continue;
		const id = attrs.match(/id="([^"]+)"/);
		return id?.[1] ?? null;
	}
	return null;
}

describe("processRenderClipRequest — real Verovio integration", () => {
	it("returns an SVG cropped to the requested bar range, not the full piece", async () => {
		// biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
		const esm = (await import("verovio/esm")) as any;
		// biome-ignore lint/suspicious/noExplicitAny: dynamic ESM
		const wasm = (await import("verovio/wasm")) as any;
		const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
		const VerovioModule = wasm.default ?? wasm;
		const mod = await VerovioModule();

		const bytes = readFileSync(FIXTURE_PATH);
		// Copy into a fresh ArrayBuffer so Verovio's WASM bindings get exactly
		// the type they expect (jsdom + Node Buffer interop is finicky).
		const arrayBuf = new ArrayBuffer(bytes.byteLength);
		new Uint8Array(arrayBuf).set(bytes);

		const { loadPiece, processRenderClipRequest } = await import(
			"./score-worker"
		);

		const entry = await loadPiece(
			arrayBuf,
			{
				module: mod,
				// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio class
				ToolkitClass: VerovioToolkit as any,
			},
			"nocturne-fixture",
		);

		expect(entry).not.toBe("failed");
		if (entry === "failed") return; // narrow for TS

		const totalMeasures = entry.measures.length;
		// Sanity: this fixture is a Chopin nocturne with > 20 measures so an
		// uncropped render would dramatically exceed any small request range.
		expect(totalMeasures).toBeGreaterThan(20);

		const startBar = 5;
		const endBar = 10;
		const requestedSpan = endBar - startBar + 1; // 6

		const svg = processRenderClipRequest(
			entry.tk,
			entry.measures,
			startBar,
			endBar,
		);

		const renderedMeasures = countMeasures(svg);

		// The rendered SVG must actually START at the requested bar. This is the
		// behaviour-level invariant: a working crop returns SVG whose first
		// measure element's xml:id matches entry.measures[startBar - 1].measureOn.
		// If the crop is broken (e.g. select() silently no-ops and renderToSVG(1)
		// returns page 1 of the narrow layout), the first measure will be bar 1.
		const expectedStartId = entry.measures[startBar - 1]?.measureOn;
		expect(expectedStartId).toBeTruthy();
		const actualStartId = firstMeasureId(svg);
		expect(actualStartId).toBe(expectedStartId);

		// And the clip must be roughly the requested length — at most 3x the
		// requested measures (tolerating any context measures Verovio adds).
		expect(renderedMeasures).toBeLessThanOrEqual(requestedSpan * 3);

		// Sanity: must not return the full piece.
		expect(renderedMeasures).toBeLessThan(totalMeasures);
	}, 30_000);
});
