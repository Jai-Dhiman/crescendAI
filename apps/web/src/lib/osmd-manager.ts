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
		type: "application/vnd.recordare.musicxml",
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
	const endIdx = endBar - 1;

	if (startIdx < 0 || endIdx >= measureList.length) return null;

	// Find bounding box spanning all target measures
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;

	const containerRect = container.getBoundingClientRect();

	for (let i = startIdx; i <= endIdx; i++) {
		const measure = measureList[i]?.[0];
		if (!measure?.stave?.SVGElement) continue;

		const svgEl = measure.stave.SVGElement as SVGElement;
		const rect = svgEl.getBoundingClientRect();

		minX = Math.min(minX, rect.left - containerRect.left);
		minY = Math.min(minY, rect.top - containerRect.top);
		maxX = Math.max(maxX, rect.right - containerRect.left);
		maxY = Math.max(maxY, rect.bottom - containerRect.top);
	}

	if (minX === Infinity) return null;

	// Add padding
	const pad = 10;
	minX = Math.max(0, minX - pad);
	minY = Math.max(0, minY - pad);
	maxX += pad;
	maxY += pad;

	// Clone the container's SVG and set viewBox to the cropped region
	const sourceSvg = container.querySelector("svg");
	if (!sourceSvg) return null;

	const cloned = sourceSvg.cloneNode(true) as SVGElement;
	cloned.setAttribute("viewBox", `${minX} ${minY} ${maxX - minX} ${maxY - minY}`);
	cloned.setAttribute("width", "100%");
	cloned.setAttribute("height", "auto");
	cloned.style.maxHeight = "200px";

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
