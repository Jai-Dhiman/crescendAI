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

const BALLADE_FIXTURE_PATH = resolve(
	dirname(fileURLToPath(import.meta.url)),
	"../../public/scores/chopin-ballade-op23-no1.mxl",
);

describe("load() wall-clock spike — Ballade fixture", () => {
  it("completes loadPiece for the Ballade within 200ms", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(BALLADE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");

    const t0 = Date.now();
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-ballade-op23-no1",
    );
    const elapsed = Date.now() - t0;

    expect(entry).not.toBe("failed");
    // GATE (a): entry must have a populated ir field — the current loadPiece has no ir field,
    // so this assertion fails with "entry.ir is undefined" until Task 3 attaches IR build.
    // This gives the build agent a genuine red->green signal.
    if (entry === "failed") return;
    expect(entry.ir).toBeDefined();
    expect(entry.ir.pages.length).toBeGreaterThan(0);
    // GATE (b): once IR build is wired in (Task 3), verify total load+IR time stays under 200ms.
    // GATE: if this assertion fails, the eager-IR contract is not viable.
    // Halt the build and revise the spec toward lazy IR before proceeding.
    expect(elapsed).toBeLessThan(200);
  }, 30_000);
});

const NOCTURNE_FIXTURE_PATH = resolve(
	dirname(fileURLToPath(import.meta.url)),
	"../../public/scores/chopin-nocturne-op9-no2.mxl",
);

describe("loadPiece — IR and pageSvgs in returned CacheEntry", () => {
  it("returns a CacheEntry with ir and pageSvgs after loading the Nocturne", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(NOCTURNE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-nocturne-op9-no2",
    );

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    // pageSvgs must be present and match page count
    expect(Array.isArray(entry.pageSvgs)).toBe(true);
    expect(entry.pageSvgs!.length).toBeGreaterThan(0);

    // IR must be built
    expect(entry.ir).toBeDefined();
    expect(entry.ir!.bars.length).toBe(entry.measures.length);
    expect(entry.ir!.pages.length).toBe(entry.pageSvgs!.length);

    // Every note must have finite x/y
    for (const note of Object.values(entry.ir!.notes)) {
      expect(Number.isFinite(note.bbox.x)).toBe(true);
      expect(Number.isFinite(note.bbox.y)).toBe(true);
    }

    // At least some notes must have been found
    expect(Object.keys(entry.ir!.notes).length).toBeGreaterThan(0);

    // Responsive-width regression: re-rendering page 1 with a different pageWidth must produce
    // an SVG whose <svg> width attribute reflects the new width, not the cached pageWidth.
    // This assertion catches any missing tk.redoLayout({}) between setOptions and renderToSVG.
    const originalPageWidth = entry.ir!.pageWidth;
    const altWidth = originalPageWidth - 200;
    entry.tk.setOptions({ pageWidth: altWidth });
    entry.tk.redoLayout({});
    const altSvg = entry.tk.renderToSVG(1) as string;
    entry.tk.setOptions({ pageWidth: originalPageWidth });
    entry.tk.redoLayout({});
    const cachedWidth = entry.pageSvgs![0]?.match(/width="(\d+)"/)?.[1];
    const altSvgWidth = altSvg.match(/width="(\d+)"/)?.[1];
    expect(altSvgWidth).toBeDefined();
    expect(cachedWidth).toBeDefined();
    // The re-rendered SVG's width must differ from the cached page's width.
    // If redoLayout is missing, Verovio returns stale layout and widths will incorrectly match.
    expect(altSvgWidth).not.toBe(cachedWidth);
  }, 30_000);
});

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
