import { api } from "./api";

interface CachedScore {
	// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type
	osmd: any;
	container: HTMLDivElement;
}

const cache = new Map<string, CachedScore>();
const pending = new Map<string, Promise<void>>();

async function doRender(pieceId: string): Promise<void> {
	const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");

	const container = document.createElement("div");
	container.style.position = "absolute";
	container.style.left = "-9999px";
	container.style.top = "-9999px";
	container.style.width = "1200px";
	document.body.appendChild(container);

	const osmd = new OpenSheetMusicDisplay(container, {
		backend: "svg",
		drawTitle: false,
		drawSubtitle: false,
		drawComposer: false,
		drawLyricist: false,
		drawPartNames: false,
		drawPartAbbreviations: false,
		drawMeasureNumbers: true,
		drawCredits: false,
	});

	const data = await api.scores.getData(pieceId);
	const blob = new Blob([data], {
		type: "application/vnd.recordare.musicxml+zip",
	});
	const url = URL.createObjectURL(blob);

	try {
		await osmd.load(url);
		osmd.render();
		cache.set(pieceId, { osmd, container });
	} catch (err) {
		// Clean up the leaked container on failure
		container.remove();
		throw err;
	} finally {
		URL.revokeObjectURL(url);
	}
}

async function ensureRendered(pieceId: string): Promise<void> {
	if (cache.has(pieceId)) return;

	// If another call is already rendering this piece, await the same promise
	const inflight = pending.get(pieceId);
	if (inflight) return inflight;

	const promise = doRender(pieceId);
	pending.set(pieceId, promise);
	try {
		await promise;
	} finally {
		pending.delete(pieceId);
	}
}

function getOsmdInstance(pieceId: string): CachedScore | null {
	return cache.get(pieceId) ?? null;
}

function clipBars(
	pieceId: string,
	startBar: number,
	endBar: number,
): SVGElement | null {
	const cached = cache.get(pieceId);
	if (!cached) return null;

	const { osmd, container } = cached;
	const measureList = osmd.graphic?.measureList;
	if (!measureList) return null;

	// Bars are 1-indexed; measureList is 0-indexed
	const startIdx = startBar - 1;
	const endIdx = Math.min(endBar - 1, measureList.length - 1);

	if (startIdx < 0 || startIdx > endIdx) return null;

	// Use OSMD's BoundingBox system: absolutePosition is in abstract OSMD units.
	// Multiplying by unitInPixels gives SVG pixel coordinates, which map directly
	// to the viewBox coordinate space.
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;

	for (let i = startIdx; i <= endIdx; i++) {
		const measure = measureList[i]?.[0];
		if (!measure?.boundingBox) continue;

		const pos = measure.boundingBox.absolutePosition;
		const size = measure.boundingBox.size;
		if (!pos || !size) continue;

		minX = Math.min(minX, pos.x);
		minY = Math.min(minY, pos.y);
		maxX = Math.max(maxX, pos.x + size.width);
		maxY = Math.max(maxY, pos.y + size.height);
	}

	if (minX === Infinity) return null;

	const sourceSvg = container.querySelector("svg");
	if (!sourceSvg) return null;

	// Convert OSMD abstract units → SVG pixel coordinates
	const unitInPixels: number =
		// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type
		(osmd.EngravingRules as any)?.unitInPixels ??
		// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type
		(osmd.rules as any)?.unitInPixels ??
		10;

	const pad = unitInPixels * 1.5;
	const vbX = Math.max(0, minX * unitInPixels - pad);
	const vbY = Math.max(0, minY * unitInPixels - pad);
	const vbW = (maxX - minX) * unitInPixels + pad * 2;
	const vbH = (maxY - minY) * unitInPixels + pad * 2;

	const cloned = sourceSvg.cloneNode(true) as SVGElement;
	cloned.setAttribute("viewBox", `${vbX} ${vbY} ${vbW} ${vbH}`);
	cloned.setAttribute("width", "100%");
	cloned.removeAttribute("height");
	cloned.style.maxHeight = "180px";

	return cloned;
}

function reset(): void {
	for (const entry of cache.values()) {
		entry.container.remove();
	}
	cache.clear();
	pending.clear();
}

export const osmdManager = {
	ensureRendered,
	clipBars,
	getOsmdInstance,
	reset,
};
