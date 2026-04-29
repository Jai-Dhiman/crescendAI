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
	xmlContent: string | null;
}

export interface ClipSvgResult {
	svg: string;
	startMeasureId: string | null;
	endMeasureId: string | null;
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

// Narrower options for clip rendering (select/mei/mxl methods).
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

function getPageForBar(
	tk: VerovioTk,
	measures: MeasureEntry[],
	barNumber: number,
): number {
	const entry = measures[barNumber - 1];
	if (!entry) return 1;
	const page = tk.getPageWithElement(entry.measureOn) as number;
	return page > 0 ? page : 1;
}

// Approach A/B (default): render the page containing startBar, return full-page SVG + measure IDs.
// Client crops client-side (SvgClip.tsx or SvgClipBBox.tsx).
export function renderClipSvg(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): ClipSvgResult {
	const startEntry = measures[startBar - 1];
	const endEntry = measures[endBar - 1];
	const page = getPageForBar(tk, measures, startBar);
	const svg = tk.renderToSVG(page) as string;
	return {
		svg,
		startMeasureId: startEntry?.measureOn ?? null,
		endMeasureId: endEntry?.measureOn ?? null,
	};
}

export function renderFullSvg(tk: VerovioTk): string {
	return tk.renderToSVG(1) as string;
}

// Approach C: Verovio select() API — tells Verovio to render only the target
// measures as page 1, preserving musical context (clef, key/time sig).
// Uses CLIP_RENDER_OPTS (narrower pageWidth) so the bars fill the container.
export function renderClipSvgSelect(
	tk: VerovioTk,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): string {
	const startEntry = measures[startBar - 1];
	const endEntry = measures[endBar - 1];
	if (!startEntry) return tk.renderToSVG(1) as string;

	const endId = endEntry?.measureOn ?? startEntry.measureOn;
	tk.setOptions(CLIP_RENDER_OPTS);
	tk.select({ start: startEntry.measureOn, end: endId });
	const svg = tk.renderToSVG(1) as string;
	tk.select({});
	tk.setOptions(VEROVIO_OPTS);
	return svg;
}

// Approach D: MEI round-trip — export the loaded score as Verovio's native MEI
// format, reload into a fresh toolkit, then select and render.
// DOMParser not used — getMEI round-trip demonstrates format translation without DOM.
export function renderClipSvgMei(
	tk: VerovioTk,
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	VerovioToolkitClass: new (mod: unknown) => VerovioTk,
	verovioModule: unknown,
	measures: MeasureEntry[],
	startBar: number,
	endBar: number,
): string {
	const startEntry = measures[startBar - 1];
	const endEntry = measures[endBar - 1];
	if (!startEntry) return tk.renderToSVG(1) as string;

	const startId = startEntry.measureOn;
	const endId = endEntry?.measureOn ?? startEntry.measureOn;

	const mei = tk.getMEI() as string;

	const newTk = new VerovioToolkitClass(verovioModule);
	newTk.setOptions(CLIP_RENDER_OPTS);
	const loaded = newTk.loadData(mei) as boolean;
	if (!loaded) return tk.renderToSVG(1) as string;

	newTk.select({ start: startId, end: endId });
	const svg = newTk.renderToSVG(1) as string;
	newTk.select({});
	return svg;
}

// Approach E: MusicXML filter — strip measures outside the target range from the
// original MusicXML source using string operations (no DOMParser required).
// Carries forward the last <attributes> element so clef/key/time sig are correct.
export function renderClipSvgMxl(
	xmlContent: string,
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	VerovioToolkitClass: new (mod: unknown) => VerovioTk,
	verovioModule: unknown,
	startBar: number,
	endBar: number,
): string | null {
	const partPat = /(<part\b[^>]*>)([\s\S]*?)(<\/part>)/g;
	let parts = "";
	let pm: RegExpExecArray | null;

	while (true) {
		pm = partPat.exec(xmlContent);
		if (!pm) break;
		const [, partOpen, partBody, partClose] = pm;
		let lastAttrs = "";
		const kept: string[] = [];

		const measurePat = /<measure\b[^>]*>[\s\S]*?<\/measure>/g;
		let mm: RegExpExecArray | null;

		while (true) {
			mm = measurePat.exec(partBody);
			if (!mm) break;
			const measureXml = mm[0];
			const numM = /\bnumber="(\d+)"/.exec(measureXml);
			if (!numM) continue;
			const num = parseInt(numM[1], 10);

			if (num < startBar) {
				const a = /<attributes>[\s\S]*?<\/attributes>/.exec(measureXml);
				if (a) lastAttrs = a[0];
			} else if (num >= startBar && num <= endBar) {
				if (
					num === startBar &&
					lastAttrs &&
					!measureXml.includes("<attributes>")
				) {
					const insertAt = measureXml.indexOf(">") + 1;
					kept.push(
						measureXml.slice(0, insertAt) +
							lastAttrs +
							measureXml.slice(insertAt),
					);
				} else {
					kept.push(measureXml);
				}
			}
		}

		parts += partOpen + kept.join("") + partClose;
	}

	if (!parts) return null;

	const firstPart = xmlContent.indexOf("<part ");
	if (firstPart === -1) return null;
	const header = xmlContent.slice(0, firstPart);
	const clean = (header + parts + "</score-partwise>").replace(
		/<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>/g,
		"",
	);

	const newTk = new VerovioToolkitClass(verovioModule);
	newTk.setOptions(CLIP_RENDER_OPTS);
	const loaded = newTk.loadData(clean) as boolean;
	if (!loaded) return null;

	return newTk.renderToSVG(1) as string;
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
	| {
			type: "render_clip";
			requestId: string;
			pieceId: string;
			startBar: number;
			endBar: number;
			method?: "default" | "select" | "mei" | "mxl";
			bytes?: ArrayBuffer;
	  }
	| {
			type: "render_full";
			requestId: string;
			pieceId: string;
			bytes?: ArrayBuffer;
			pageWidth?: number;
	  };

// Worker message handler — only registers when loaded as a Web Worker (window is undefined)
if (typeof window === "undefined") {
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio module
	let verovioModule: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	let VerovioToolkitClass: new (mod: unknown) => VerovioTk = null as any;

	const ready = (async () => {
		const [wasm, esm] = await Promise.all([
			import("verovio/wasm") as Promise<{ default: () => Promise<unknown> }>,
			import("verovio/esm") as Promise<{
				VerovioToolkit: new (mod: unknown) => VerovioTk;
			}>,
		]);
		verovioModule = await wasm.default();
		VerovioToolkitClass = esm.VerovioToolkit;
	})();

	// Cache maps pieceId -> settled result or a Promise for the in-flight load.
	// Using Promise values lets concurrent handlers await the same load operation
	// instead of all racing to "bytes required on first request".
	const toolkitCache = new Map<string, LoadResult | Promise<LoadResult>>();

	async function loadPiece(bytes: ArrayBuffer): Promise<LoadResult> {
		const ZIP_MAGIC = 0x04034b50;
		const isZip =
			bytes.byteLength >= 4 &&
			new DataView(bytes).getUint32(0, true) === ZIP_MAGIC;

		// For ZIP: extract XML first so the buffer is readable before Verovio
		// corrupts it as WASM scratch space.
		// For plain XML: decode directly — the file is a .musicxml stored as .mxl.
		let xmlContent: string | null = null;
		if (isZip) {
			xmlContent = await extractXmlFromMxl(bytes);
		} else {
			try {
				xmlContent = new TextDecoder().decode(bytes);
			} catch {
				// non-text binary — will fail below
			}
		}

		let tk = new VerovioToolkitClass(verovioModule);
		tk.setOptions(VEROVIO_OPTS);
		let loaded = false;

		if (isZip) {
			try {
				loaded = tk.loadZipDataBuffer(bytes);
			} catch {
				// WASM exception — toolkit state is corrupt, create a fresh one below.
			}
		}

		// loadZipDataBuffer either threw or returned false, or file was plain XML.
		// Try loadData with pre-extracted (and DOCTYPE-stripped) XML string.
		if (!loaded && xmlContent !== null) {
			tk = new VerovioToolkitClass(verovioModule);
			tk.setOptions(VEROVIO_OPTS);
			try {
				const clean = xmlContent.replace(
					/<!DOCTYPE\s[^>[]*(\[[^\]]*\])?\s*>/g,
					"",
				);
				loaded = tk.loadData(clean) as boolean;
			} catch (fallbackErr) {
				console.error("[score-worker] loadData fallback failed:", fallbackErr);
			}
		}

		if (!loaded) return "failed";

		const entry: CacheEntry = { tk, measures: [], xmlContent };
		// renderToTimemap triggers the deprecated WASM 'try' instruction warning.
		// If it throws, clips fall back to rendering page 1.
		try {
			entry.measures = buildMeasureIndex(tk);
		} catch {
			// measures stays empty; getPageForBar returns 1 for all bars
		}
		return entry;
	}

	(self as unknown as Worker).onmessage = async (
		event: MessageEvent<WorkerInMsg>,
	) => {
		const msg = event.data;
		try {
			await ready;

			// Synchronous cache lookup — no await between here and toolkitCache.set
			// so that concurrent handlers see the in-flight Promise rather than
			// undefined and all independently returning "bytes required".
			const cached = toolkitCache.get(msg.pieceId);
			let result: LoadResult | undefined;

			if (cached instanceof Promise) {
				// Another handler already started loading — wait for it.
				result = await cached;
			} else {
				result = cached; // CacheEntry | "failed" | undefined
			}

			if (
				result === undefined ||
				(result === "failed" && msg.bytes !== undefined)
			) {
				if (!msg.bytes) {
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						error:
							result === "failed"
								? `score previously failed to load for ${msg.pieceId}`
								: "bytes required on first request",
					});
					return;
				}

				// Start loading and store the Promise synchronously before any await,
				// so concurrent handlers that run while we are loading will await the
				// same Promise instead of hitting "bytes required".
				const loadPromise = loadPiece(msg.bytes);
				toolkitCache.set(msg.pieceId, loadPromise);
				result = await loadPromise;
				// Replace the in-flight Promise with the settled result.
				if (toolkitCache.get(msg.pieceId) === loadPromise) {
					toolkitCache.set(msg.pieceId, result);
				}
			}

			if (result === "failed") {
				(self as unknown as Worker).postMessage({
					requestId: msg.requestId,
					error: `score failed to load for ${msg.pieceId} — MXL data may be corrupt or incompatible`,
				});
				return;
			}

			const { tk, measures, xmlContent } = result;
			if (msg.type === "render_clip") {
				const method = msg.method ?? "default";

				if (method === "select") {
					const svg = renderClipSvgSelect(
						tk,
						measures,
						msg.startBar,
						msg.endBar,
					);
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						svg,
					});
				} else if (method === "mei") {
					const svg = renderClipSvgMei(
						tk,
						VerovioToolkitClass,
						verovioModule,
						measures,
						msg.startBar,
						msg.endBar,
					);
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						svg,
					});
				} else if (method === "mxl") {
					if (!xmlContent) {
						(self as unknown as Worker).postMessage({
							requestId: msg.requestId,
							error: "xmlContent not available for mxl method",
						});
						return;
					}
					const svg = renderClipSvgMxl(
						xmlContent,
						VerovioToolkitClass,
						verovioModule,
						msg.startBar,
						msg.endBar,
					);
					if (!svg) {
						(self as unknown as Worker).postMessage({
							requestId: msg.requestId,
							error: "MXL filter render produced no output",
						});
						return;
					}
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						svg,
					});
				} else {
					// default: full page + measure IDs for client-side crop
					const clip = renderClipSvg(tk, measures, msg.startBar, msg.endBar);
					(self as unknown as Worker).postMessage({
						requestId: msg.requestId,
						svg: clip.svg,
						startMeasureId: clip.startMeasureId ?? undefined,
						endMeasureId: clip.endMeasureId ?? undefined,
					});
				}
			} else {
				let svg: string;
				if (msg.pageWidth !== undefined) {
					tk.setOptions({ pageWidth: msg.pageWidth });
					svg = renderFullSvg(tk);
					tk.setOptions({ pageWidth: VEROVIO_OPTS.pageWidth });
				} else {
					svg = renderFullSvg(tk);
				}
				(self as unknown as Worker).postMessage({
					requestId: msg.requestId,
					svg,
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
