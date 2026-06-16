// apps/web/src/lib/score-worker.ts

// biome-ignore lint/suspicious/noExplicitAny: Verovio has no exported TS types
type VerovioTk = any;

interface MeasureEntry {
	qstamp: number;
	measureOn: string;
}

interface CacheEntry {
	tk: VerovioTk;
	measures: MeasureEntry[];
	ir: import("./score-ir").ScoreIR;
	pageSvgs: string[];
}

type LoadResult = CacheEntry | "failed";

const VEROVIO_OPTS = {
	pageWidth: 2400,
	adjustPageHeight: true,
	breaks: "smart",
	footer: "none",
	header: "none",
	scale: 40,
} as const;

// Narrower options for clip rendering.
// pageWidth 1600 at scale 40 fits ~2-4 bars of dense piano music cleanly.
const CLIP_RENDER_OPTS = {
	pageWidth: 1600,
	adjustPageHeight: true,
	breaks: "smart",
	footer: "none",
	header: "none",
	scale: 40,
} as const;

// Build a deduplicated, qstamp-sorted measure index from the timemap.
// Piano scores produce two measureOn entries per bar (one per staff), so
// we deduplicate by qstamp to get one entry per bar.
function buildMeasureIndex(tk: VerovioTk): MeasureEntry[] {
	// biome-ignore lint/suspicious/noExplicitAny: Verovio timemap has no types
	const timemap: any[] = tk.renderToTimemap({ includeMeasures: true });
	const seen = new Set<number>();
	return timemap
		.filter((e) => e.measureOn !== undefined)
		.filter((e) => {
			if (seen.has(e.qstamp)) return false;
			seen.add(e.qstamp);
			return true;
		})
		.sort((a, b) => a.qstamp - b.qstamp)
		.map((e) => ({ qstamp: e.qstamp, measureOn: e.measureOn }));
}

export function processGetPageRequest(pageSvgs: string[], pageN: number): string | "failed" {
	const svg = pageSvgs[pageN - 1];
	if (svg === undefined) return "failed";
	return svg;
}

// Verovio select() + redoLayout() crops the engraved output to the requested
// bar range. Per Verovio docs: "the selection will be applied only when some
// data is loaded or the layout is redone." Calling select() without
// redoLayout() (as Phase 1 originally did) leaves the layout untouched and
// renderToSVG returns the full piece.
function renderClipSvgSelect(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): string {
	const startEntry = measures[startBar - 1];
	if (!startEntry) return tk.renderToSVG(1) as string;

	tk.setOptions(CLIP_RENDER_OPTS);
	tk.select({ measureRange: `${startBar}-${endBar}` });
	tk.redoLayout({});
	const svg = tk.renderToSVG(1) as string;
	// Restore the full-piece layout for any subsequent render against this toolkit.
	tk.select({});
	tk.setOptions(VEROVIO_OPTS);
	tk.redoLayout({});
	return svg;
}

// Canonical render_clip dispatch. Restricts engraving to the requested bars.
export function processRenderClipRequest(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): string {
	return renderClipSvgSelect(tk, measures, startBar, endBar);
}

export interface ClipNote {
	midi: number;
	startQ: number;
	endQ: number;
}

export interface ClipPlaybackResult {
	svg: string;
	ir: import("./score-ir").ScoreIR;
	notes: ClipNote[];
}

export async function processGetClipPlaybackRequest(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): Promise<ClipPlaybackResult | "failed"> {
	const startEntry = measures[startBar - 1];
	if (!startEntry) return "failed";

	tk.setOptions(CLIP_RENDER_OPTS);
	tk.select({ measureRange: `${startBar}-${endBar}` });
	tk.redoLayout({});

	let svg = "";
	let ir: import("./score-ir").ScoreIR | undefined;
	const notes: ClipNote[] = [];

	try {
		svg = tk.renderToSVG(1) as string;

		const timemap: Array<{ qstamp: number; on?: string[]; off?: string[]; measureOn?: string }> =
			tk.renderToTimemap({ includeMeasures: true });

		const noteQstampMap = new Map<string, number>();
		const noteOffMap = new Map<string, number>();
		for (const entry of timemap) {
			if (Array.isArray(entry.on)) {
				for (const id of entry.on) {
					noteQstampMap.set(id, entry.qstamp);
				}
			}
			if (Array.isArray(entry.off)) {
				for (const id of entry.off) {
					noteOffMap.set(id, entry.qstamp);
				}
			}
		}

		const clipMeasures: MeasureEntry[] = [];
		const seenQstamps = new Set<number>();
		for (const entry of timemap) {
			if (entry.measureOn !== undefined && !seenQstamps.has(entry.qstamp)) {
				seenQstamps.add(entry.qstamp);
				clipMeasures.push({ qstamp: entry.qstamp, measureOn: entry.measureOn });
			}
		}
		clipMeasures.sort((a, b) => a.qstamp - b.qstamp);

		const { parseScoreIR } = await import("./score-ir");

		ir = parseScoreIR(
			"",
			[svg],
			clipMeasures,
			noteQstampMap,
			tk.getVersion() as string,
			CLIP_RENDER_OPTS.pageWidth,
		);

		// Restrict to the clip's own notes. renderToTimemap returns the WHOLE piece
		// even after select() (select restricts engraving, not the timemap), so
		// noteQstampMap covers every note in the piece (~thousands). parseScoreIR
		// found exactly the note ids present in the clip SVG, so use those — this
		// keeps the note events in the clip IR's qstamp coordinate system (what the
		// cursor and LoopClock use). Without this the array would carry the entire
		// piece with out-of-clip qstamps.
		const clipNoteIds = new Set<string>();
		for (const bar of ir.bars) {
			for (const nid of bar.noteIds) clipNoteIds.add(nid);
		}
		// Build notes BEFORE restoring full-piece scope, so getMIDIValuesForElement
		// reads from the clip's MIDI context (not the full-piece context).
		for (const [id, startQ] of noteQstampMap) {
			if (!clipNoteIds.has(id)) continue;
			// Verovio's getMIDIValuesForElement returns a single object { pitch, ... }
			// for a note, NOT an array. Some elements (e.g. grace/tied notes or ids
			// absent from the MIDI map) return undefined/null. Handle both shapes and
			// skip anything without a numeric pitch rather than crashing on `.pitch`.
			const raw = tk.getMIDIValuesForElement(id) as unknown;
			const v = (Array.isArray(raw) ? raw[0] : raw) as { pitch?: unknown } | undefined;
			if (!v || typeof v.pitch !== "number") continue;
			const midi = v.pitch;
			const endQ = noteOffMap.get(id) ?? startQ + 1;
			notes.push({ midi, startQ, endQ });
		}
		notes.sort((a, b) => a.startQ - b.startQ);
	} finally {
		// Restore full-piece layout regardless of whether parseScoreIR threw.
		tk.select({});
		tk.setOptions(VEROVIO_OPTS);
		tk.redoLayout({});
	}

	if (ir === undefined) return "failed";
	return { svg, ir, notes };
}

export interface VerovioBindings {
	module: unknown;
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	ToolkitClass: new (mod: unknown) => any;
}

export async function loadPiece(
	bytes: ArrayBuffer,
	bindings: VerovioBindings,
	pieceId?: string,
	transpose?: number,
): Promise<LoadResult> {
	const { module, ToolkitClass } = bindings;
	const ZIP_MAGIC = 0x04034b50;
	const isZip =
		bytes.byteLength >= 4 &&
		new DataView(bytes).getUint32(0, true) === ZIP_MAGIC;

	const applyOpts = (t: VerovioTk) => {
		t.setOptions(VEROVIO_OPTS);
		// Verovio applies `transpose` at loadData time. A bare semitone count is
		// auto-accidental-minimized. transpose:0 / undefined is a no-op, keeping
		// real pieces byte-identical to the pre-transpose code path.
		if (transpose !== undefined && transpose !== 0) {
			t.setOptions({ transpose: String(transpose) });
		}
	};

	let tk = new ToolkitClass(module);
	applyOpts(tk);
	let loaded = false;

	if (isZip) {
		try {
			const clone = bytes.slice(0);
			loaded = tk.loadZipDataBuffer(clone) as boolean;
		} catch {
			tk = new ToolkitClass(module);
			applyOpts(tk);
		}
	}

	if (!loaded) {
		let fallbackXml: string | null = null;
		if (isZip) {
			try {
				fallbackXml = await extractXmlFromMxl(bytes);
			} catch (e) {
				console.error("[score-worker] extractXmlFromMxl failed:", e);
			}
		} else {
			try {
				fallbackXml = new TextDecoder().decode(bytes);
			} catch (e) {
				console.error("[score-worker] TextDecoder failed:", e);
			}
		}
		if (fallbackXml !== null) {
			tk = new ToolkitClass(module);
			applyOpts(tk);
			try {
				const clean = fallbackXml.replace(
					/<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>/g,
					"",
				);
				loaded = tk.loadData(clean) as boolean;
			} catch (e) {
				console.error("[score-worker] loadData fallback failed:", e);
			}
		}
	}

	if (!loaded) return "failed";

	let measures: MeasureEntry[];
	try {
		measures = buildMeasureIndex(tk);
	} catch (e) {
		console.error("[score-worker] buildMeasureIndex failed for", pieceId ?? "?", e);
		return "failed";
	}

	// Render all pages eagerly and cache their SVGs.
	const pageCount = tk.getPageCount() as number;
	if (pageCount === 0) return "failed";

	const pageSvgs: string[] = [];
	for (let n = 1; n <= pageCount; n++) {
		pageSvgs.push(tk.renderToSVG(n) as string);
	}

	// Build a noteId -> onset qstamp map from the Verovio timemap.
	// tk.renderToTimemap({ includeMeasures: true }) returns entries like:
	//   { on: string[], off: string[], qstamp: number, measureOn?: string, ... }
	// where `on` is an array of note element ids starting at that onset tick.
	// We reuse the same timemap call that buildMeasureIndex already uses.
	const noteQstampMap = new Map<string, number>();
	const timemap2 = tk.renderToTimemap({ includeMeasures: true }) as Array<{
		qstamp: number;
		on?: string[];
	}>;
	for (const tmEntry of timemap2) {
		if (Array.isArray(tmEntry.on)) {
			for (const noteId of tmEntry.on) {
				noteQstampMap.set(noteId, tmEntry.qstamp);
			}
		}
	}

	const { parseScoreIR } = await import("./score-ir");
	const ir = parseScoreIR(
		pieceId ?? "",
		pageSvgs,
		measures,
		noteQstampMap,
		tk.getVersion() as string,
		VEROVIO_OPTS.pageWidth,
	);

	return { tk, measures, ir, pageSvgs };
}

// Expected R2 object format (scores/v1/{pieceId}.mxl):
//   ZIP archive (PK\x03\x04 magic), two entries in order:
//     META-INF/container.xml  — rootfiles declaration
//     {pieceId}.xml           — DOCTYPE-stripped MusicXML content
//   Both entries use DEFLATE (method 8). Local file headers carry the
//   correct compressedSize at offset 18 with flag bit 3 = 0 (no data
//   descriptor), because Python's zipfile.writestr() compresses in memory
//   before writing the header. This parser depends on that invariant.
//   See model/src/score_library/upload.py: wrap_as_mxl_zip().

// Extract the main XML file from an MXL ZIP using the browser's native
// DecompressionStream API, so we can fall back to loadData() if
// loadZipDataBuffer() throws a WASM exception on a particular file.
async function extractXmlFromMxl(bytes: ArrayBuffer): Promise<string | null> {
	const view = new DataView(bytes);
	let offset = 0;

	while (offset + 30 <= bytes.byteLength) {
		// Local file header signature: PK\x03\x04
		if (view.getUint32(offset, true) !== 0x04034b50) break;

		const flags = view.getUint16(offset + 6, true);
		const method = view.getUint16(offset + 8, true);
		const compressedSize = view.getUint32(offset + 18, true);
		const fileNameLen = view.getUint16(offset + 26, true);
		const extraLen = view.getUint16(offset + 28, true);
		const fileName = new TextDecoder().decode(
			new Uint8Array(bytes, offset + 30, fileNameLen),
		);
		const dataStart = offset + 30 + fileNameLen + extraLen;

		if (!fileName.startsWith("META-INF") && fileName.endsWith(".xml")) {
			const compressed = new Uint8Array(bytes, dataStart, compressedSize);
			if (method === 0) {
				// Stored — no compression
				return new TextDecoder().decode(compressed);
			}
			if (method === 8) {
				// DEFLATE
				const ds = new DecompressionStream("deflate-raw");
				const writer = ds.writable.getWriter();
				await writer.write(compressed);
				await writer.close();
				const reader = ds.readable.getReader();
				const chunks: Uint8Array[] = [];
				for (;;) {
					const { done, value } = await reader.read();
					if (done) break;
					chunks.push(value);
				}
				const total = chunks.reduce((n, c) => n + c.length, 0);
				const out = new Uint8Array(total);
				let pos = 0;
				for (const c of chunks) {
					out.set(c, pos);
					pos += c.length;
				}
				return new TextDecoder().decode(out);
			}
		}

		offset = dataStart + compressedSize;
		// Skip optional data descriptor that follows compressed data
		if (flags & 8) {
			offset += view.getUint32(offset, true) === 0x08074b50 ? 16 : 12;
		}
	}
	return null;
}

type WorkerInMsg =
	| { type: "load";             requestId: string; pieceId: string; bytes: ArrayBuffer; transpose?: number }
	| { type: "get_page";         requestId: string; pieceId: string; pageN: number; pageWidth?: number; transpose?: number }
	| { type: "get_clip";         requestId: string; pieceId: string; startBar: number; endBar: number; transpose?: number }
	| { type: "get_clip_playback"; requestId: string; pieceId: string; startBar: number; endBar: number; transpose?: number }
	| { type: "get_ir";           requestId: string; pieceId: string; transpose?: number };

// Worker message handler — only registers when loaded as a Web Worker (window is undefined)
if (typeof window === "undefined") {
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio module
	let verovioModule: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	let VerovioToolkitClass: new (mod: unknown) => VerovioTk = null as any;

	const ready = (async () => {
		console.log("[score-worker] WASM init start");
		const t0 = Date.now();
		const [wasm, esm] = await Promise.all([
			import("verovio/wasm") as Promise<{ default: () => Promise<unknown> }>,
			import("verovio/esm") as Promise<{
				VerovioToolkit: new (mod: unknown) => VerovioTk;
			}>,
		]);
		console.log(`[score-worker] imports resolved in ${Date.now() - t0}ms, creating module`);
		verovioModule = await wasm.default();
		VerovioToolkitClass = esm.VerovioToolkit;
		console.log(`[score-worker] WASM ready in ${Date.now() - t0}ms`);
	})();

	// Cache maps pieceId -> settled result or a Promise for the in-flight load.
	// Using Promise values lets concurrent handlers await the same load operation
	// instead of all racing to "bytes required on first request".
	const toolkitCache = new Map<string, LoadResult | Promise<LoadResult>>();

	(self as unknown as Worker).onmessage = async (
		event: MessageEvent<WorkerInMsg>,
	) => {
		const msg = event.data;
		try {
			await ready;

			// Synchronous cache lookup — no await between here and toolkitCache.set
			// so that concurrent handlers see the in-flight Promise rather than
			// undefined and all independently returning "bytes required".
			// ALL message types use the same composite key `${pieceId}:${transpose ?? 0}`.
			// load() always stores under the transpose-suffixed key, so every
			// subsequent get_clip/get_page/get_clip_playback/get_ir must look up the
			// same key — keying non-load messages by bare pieceId was a cache MISS for
			// every piece (even untransposed ones at `:0`).
			const cacheKey = `${msg.pieceId}:${msg.transpose ?? 0}`;
			const cached = toolkitCache.get(cacheKey);
			let result: LoadResult | undefined;

			if (cached instanceof Promise) {
				// Another handler already started loading — wait for it.
				result = await cached;
			} else {
				result = cached; // CacheEntry | "failed" | undefined
			}

			if (
				result === undefined ||
				(result === "failed" && msg.type === "load")
			) {
				if (msg.type !== "load") {
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						error:
							result === "failed"
								? `score previously failed to load for ${msg.pieceId}`
								: "bytes required on first request — send a 'load' message first",
					});
					return;
				}

				// Start loading and store the Promise synchronously before any await,
				// so concurrent handlers that run while we are loading will await the
				// same Promise instead of hitting "bytes required".
				const loadPromise = loadPiece(
					msg.bytes,
					{ module: verovioModule, ToolkitClass: VerovioToolkitClass },
					msg.pieceId,
					msg.transpose,
				);
				toolkitCache.set(cacheKey, loadPromise);
				result = await loadPromise;
				// Replace the in-flight Promise with the settled result.
				if (toolkitCache.get(cacheKey) === loadPromise) {
					toolkitCache.set(cacheKey, result);
				}
			}

			if (result === "failed") {
				(self as unknown as Worker).postMessage({
					requestId: msg.requestId,
					error: `score failed to load for ${msg.pieceId} — MXL data may be corrupt or incompatible`,
				});
				return;
			}

			const { tk, measures, ir, pageSvgs } = result;
			if (msg.type === "get_clip") {
				const svg = processRenderClipRequest(tk, measures, msg.startBar, msg.endBar);
				(self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: svg });
			} else if (msg.type === "get_page") {
				// If a custom pageWidth is supplied (e.g. from the sandbox's responsive-width logic),
				// re-render that page with the adjusted width; otherwise serve from the pre-rendered cache.
				let svg: string | "failed";
				if (msg.pageWidth !== undefined && msg.pageWidth !== ir.pageWidth) {
					tk.setOptions({ pageWidth: msg.pageWidth });
					// redoLayout is required after every layout-changing setOptions call before renderToSVG.
					// Without it, Verovio renders with the old layout geometry (silently wrong-sized).
					tk.redoLayout({});
					const rendered = tk.renderToSVG(msg.pageN) as string;
					// Restore original pageWidth so future pre-rendered cache reads remain consistent.
					tk.setOptions({ pageWidth: ir.pageWidth });
					tk.redoLayout({});
					svg = rendered || "failed";
				} else {
					svg = processGetPageRequest(pageSvgs, msg.pageN);
				}
				if (svg === "failed") {
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						error: `page ${msg.pageN} not found for ${msg.pieceId}`,
					});
				} else {
					(self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: svg });
				}
			} else if (msg.type === "get_ir") {
				(self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: ir });
			} else if (msg.type === "get_clip_playback") {
				const playback = await processGetClipPlaybackRequest(tk, measures, msg.startBar, msg.endBar);
				if (playback === "failed") {
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						error: `get_clip_playback failed for ${msg.pieceId} bars ${msg.startBar}-${msg.endBar}`,
					});
				} else {
					(self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: playback });
				}
			} else if (msg.type === "load") {
				// load was already handled above (bytes were used to call loadPiece).
				// Return the ir and pageSvgs to the renderer.
				(self as unknown as Worker).postMessage({
					requestId: msg.requestId,
					payload: { ir, pageSvgs },
				});
			}
		} catch (err) {
			const errorMsg =
				typeof WebAssembly !== "undefined" &&
				err instanceof WebAssembly.Exception
					? `Verovio WASM exception (${msg.type} for ${msg.pieceId}) — MXL data may be corrupt or incompatible`
					: String(err);
			(self as unknown as Worker).postMessage({
				requestId: msg.requestId,
				error: errorMsg,
			});
		}
	};
}
