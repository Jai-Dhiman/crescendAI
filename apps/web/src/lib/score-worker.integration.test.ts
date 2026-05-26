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
  it("IR walk cost (parseScoreIR alone) is under 100ms", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(BALLADE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");
    const { parseScoreIR } = await import("./score-ir");

    // Load the piece fully so Verovio has rendered all pages and we have a warm tk.
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-ballade-op23-no1-irwalk",
    );

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    // Rebuild noteQstampMap from the already-warm tk (same logic as loadPiece).
    const noteQstampMap = new Map<string, number>();
    const timemap = entry.tk.renderToTimemap({ includeMeasures: true }) as Array<{
      qstamp: number;
      on?: string[];
    }>;
    for (const tmEntry of timemap) {
      if (Array.isArray(tmEntry.on)) {
        for (const noteId of tmEntry.on) {
          noteQstampMap.set(noteId, tmEntry.qstamp);
        }
      }
    }

    // Time parseScoreIR alone — this is the IR walk with no Verovio rendering noise.
    // Expected: ~15-25ms for the Ballade. Gate at 100ms to absorb machine variance.
    const t0 = Date.now();
    const ir = parseScoreIR(
      "chopin-ballade-op23-no1-irwalk",
      entry.pageSvgs,
      entry.measures,
      noteQstampMap,
      entry.tk.getVersion() as string,
      2400,
    );
    const irWalkMs = Date.now() - t0;

    expect(ir).toBeDefined();
    expect(ir.pages.length).toBeGreaterThan(0);
    expect(irWalkMs).toBeLessThan(100);
  }, 90_000);
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
    // Verovio SVG uses width="2400px" format (with px suffix). Match that.
    const cachedWidth = entry.pageSvgs![0]?.match(/width="(\d+)px"/)?.[1];
    const altSvgWidth = altSvg.match(/width="(\d+)px"/)?.[1];
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

const CZERNY_FIXTURE_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../public/scores/czerny-op299-no1.mxl",
);

async function makeBindings() {
  const esm = (await import("verovio/esm")) as any;
  const wasm = (await import("verovio/wasm")) as any;
  const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
  const VerovioModule = wasm.default ?? wasm;
  const mod = await VerovioModule();
  return { module: mod, ToolkitClass: VerovioToolkit as any };
}

describe("IR invariants — all 3 fixtures", () => {
  const fixtures = [
    { name: "czerny-op299-no1", path: CZERNY_FIXTURE_PATH },
    { name: "chopin-nocturne-op9-no2", path: FIXTURE_PATH },
    { name: "chopin-ballade-op23-no1", path: BALLADE_FIXTURE_PATH },
  ];

  for (const { name, path } of fixtures) {
    it(`${name}: bars.length === measures.length and all notes have finite x/y`, async () => {
      const bindings = await makeBindings();
      const bytes = readFileSync(path);
      const arrayBuf = new ArrayBuffer(bytes.byteLength);
      new Uint8Array(arrayBuf).set(bytes);

      const { loadPiece } = await import("./score-worker");
      const entry = await loadPiece(arrayBuf, bindings, name);

      expect(entry).not.toBe("failed");
      if (entry === "failed") return;

      expect(entry.ir.bars.length).toBe(entry.measures.length);
      expect(entry.ir.pages.length).toBeGreaterThan(0);
      expect(entry.ir.pages.length).toBe(entry.pageSvgs.length);

      for (const note of Object.values(entry.ir.notes)) {
        expect(Number.isFinite(note.bbox.x)).toBe(true);
        expect(Number.isFinite(note.bbox.y)).toBe(true);
      }

      // Per-note qstamp regression: verify at least one bar has >= 2 distinct qstamps.
      // This catches any regression where all notes collapse to qstampStart.
      const barsWithMultipleOnsets = entry.ir.bars.filter((bar) => {
        if (bar.noteIds.length < 2) return false;
        const qstamps = new Set(bar.noteIds.map((id) => entry.ir.notes[id]?.qstamp ?? -1));
        return qstamps.size >= 2;
      });
      expect(barsWithMultipleOnsets.length).toBeGreaterThan(0);
    }, 60_000);
  }
});

describe("IR/Clip correlation", () => {
  it("getClip first measure id matches ir.bars[startBar-1].measureOn for the Nocturne", async () => {
    const bindings = await makeBindings();
    const bytes = readFileSync(FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece, processRenderClipRequest } = await import("./score-worker");
    const entry = await loadPiece(arrayBuf, bindings, "nocturne-correlation");

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    const startBar = 5;
    const endBar = 10;
    const svg = processRenderClipRequest(entry.tk, entry.measures, startBar, endBar);

    const expectedMeasureOn = entry.ir.bars[startBar - 1]?.measureOn;
    expect(expectedMeasureOn).toBeTruthy();

    const actualId = firstMeasureId(svg);
    expect(actualId).toBe(expectedMeasureOn);
  }, 30_000);
});

describe("Cache eviction — reloading pieceId with different bytes yields disjoint note keys", () => {
  it("a second loadPiece call for the same pieceId produces a different ir.notes keyset", async () => {
    const bindings = await makeBindings();

    const nocturneBytes = readFileSync(FIXTURE_PATH);
    const nocturneBuf = new ArrayBuffer(nocturneBytes.byteLength);
    new Uint8Array(nocturneBuf).set(nocturneBytes);

    const czernyBytes = readFileSync(CZERNY_FIXTURE_PATH);
    const czernyBuf = new ArrayBuffer(czernyBytes.byteLength);
    new Uint8Array(czernyBuf).set(czernyBytes);

    const { loadPiece } = await import("./score-worker");

    const entry1 = await loadPiece(nocturneBuf, bindings, "eviction-test");
    expect(entry1).not.toBe("failed");
    if (entry1 === "failed") return;
    const firstKeys = new Set(Object.keys(entry1.ir.notes));

    const entry2 = await loadPiece(czernyBuf, bindings, "eviction-test");
    expect(entry2).not.toBe("failed");
    if (entry2 === "failed") return;
    const secondKeys = new Set(Object.keys(entry2.ir.notes));

    // Verovio randomizes element ids on every loadData call — the two keysets
    // should be disjoint (different pieces loaded into the same pieceId slot).
    const intersection = [...firstKeys].filter((k) => secondKeys.has(k));
    // Allow at most 5% overlap to account for any coincidental id collision.
    expect(intersection.length).toBeLessThan(firstKeys.size * 0.05);
  }, 60_000);
});

describe("Ballade load() with full IR build — production-scale perf gate", () => {
  it("IR walk cost (parseScoreIR alone) is under 100ms and IR invariants hold at Ballade scale", async () => {
    const esm = (await import("verovio/esm")) as any;
    const wasm = (await import("verovio/wasm")) as any;
    const VerovioToolkit = esm.VerovioToolkit ?? esm.default?.VerovioToolkit;
    const VerovioModule = wasm.default ?? wasm;
    const mod = await VerovioModule();

    const bytes = readFileSync(BALLADE_FIXTURE_PATH);
    const arrayBuf = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuf).set(bytes);

    const { loadPiece } = await import("./score-worker");
    const { parseScoreIR } = await import("./score-ir");

    // Load the piece so Verovio has rendered all pages and we have a warm tk.
    const entry = await loadPiece(
      arrayBuf,
      { module: mod, ToolkitClass: VerovioToolkit as any },
      "chopin-ballade-op23-no1-perf2",
    );

    expect(entry).not.toBe("failed");
    if (entry === "failed") return;

    // IR structural invariants at Ballade scale.
    expect(entry.ir.bars.length).toBe(entry.measures.length);
    expect(entry.ir.pages.length).toBeGreaterThan(0);

    // Rebuild noteQstampMap from the already-warm tk (same logic as loadPiece).
    const noteQstampMap = new Map<string, number>();
    const timemap = entry.tk.renderToTimemap({ includeMeasures: true }) as Array<{
      qstamp: number;
      on?: string[];
    }>;
    for (const tmEntry of timemap) {
      if (Array.isArray(tmEntry.on)) {
        for (const noteId of tmEntry.on) {
          noteQstampMap.set(noteId, tmEntry.qstamp);
        }
      }
    }

    // Time parseScoreIR alone — IR walk with no Verovio rendering noise.
    // Expected: ~15-25ms for the Ballade. Gate at 100ms to absorb machine variance.
    const t0 = Date.now();
    const ir = parseScoreIR(
      "chopin-ballade-op23-no1-perf2",
      entry.pageSvgs,
      entry.measures,
      noteQstampMap,
      entry.tk.getVersion() as string,
      2400,
    );
    const irWalkMs = Date.now() - t0;

    expect(ir).toBeDefined();
    expect(ir.pages.length).toBeGreaterThan(0);
    expect(ir.bars.length).toBe(entry.measures.length);
    expect(irWalkMs).toBeLessThan(100);
  }, 90_000);
});
