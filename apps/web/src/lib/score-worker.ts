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

export function renderFullSvg(tk: VerovioTk): string {
	return tk.renderToSVG(1) as string;
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

export interface VerovioBindings {
	module: unknown;
	// biome-ignore lint/suspicious/noExplicitAny: dynamic Verovio ESM class
	ToolkitClass: new (mod: unknown) => any;
}

export async function loadPiece(
	bytes: ArrayBuffer,
	bindings: VerovioBindings,
	pieceId?: string,
): Promise<LoadResult> {
	const { module, ToolkitClass } = bindings;
	const ZIP_MAGIC = 0x04034b50;
	const isZip =
		bytes.byteLength >= 4 &&
		new DataView(bytes).getUint32(0, true) === ZIP_MAGIC;

	let tk = new ToolkitClass(module);
	tk.setOptions(VEROVIO_OPTS);
	let loaded = false;

	if (isZip) {
		try {
			const clone = bytes.slice(0);
			loaded = tk.loadZipDataBuffer(clone) as boolean;
		} catch {
			tk = new ToolkitClass(module);
			tk.setOptions(VEROVIO_OPTS);
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
			tk.setOptions(VEROVIO_OPTS);
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

	const entry: CacheEntry = { tk, measures: [] };
	try {
		entry.measures = buildMeasureIndex(tk);
	} catch (e) {
		console.error("[score-worker] buildMeasureIndex failed for", pieceId ?? "?", e);
		return "failed";
	}
	return entry;
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
				const loadPromise = loadPiece(
					msg.bytes,
					{ module: verovioModule, ToolkitClass: VerovioToolkitClass },
					msg.pieceId,
				);
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

			const { tk, measures } = result;
			if (msg.type === "render_clip") {
				const svg = processRenderClipRequest(
					tk,
					measures,
					msg.startBar,
					msg.endBar,
				);
				(self as unknown as Worker).postMessage({
					requestId: msg.requestId,
					svg,
				});
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
