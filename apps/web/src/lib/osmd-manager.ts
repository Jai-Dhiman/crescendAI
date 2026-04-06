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

function reset(): void {
	for (const entry of cache.values()) {
		entry.container.remove();
	}
	cache.clear();
	pending.clear();
}

export const osmdManager = {
	ensureRendered,
	getOsmdInstance,
	reset,
};
