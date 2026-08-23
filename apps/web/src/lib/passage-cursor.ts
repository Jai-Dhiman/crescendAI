// apps/web/src/lib/passage-cursor.ts
import type { BarIR } from "./score-ir";

export interface BarTimelineEntry {
	readonly bar: number;
	readonly tSec: number;
}

/**
 * Bridges PassagePlayer's real-time playback clock (tSec) to ScoreCursor's
 * qstamp clock. Nothing else in the codebase does this: useLoopPlayer's
 * qstampSource comes from a synthesized ClipNote clock with an exact qstamp,
 * not real recorded audio's coarse per-bar barTimeline.
 *
 * Returns null -- never a guessed position -- when playback hasn't started,
 * is past the last known bar boundary, or the resolved bar has no entry in
 * `bars` (a clip IR that doesn't cover the bar the manifest names). This
 * mirrors mark.ts's resolveAnchor: the system falls silent rather than
 * drawing a cursor it isn't sure about.
 */
export function createPassageQstampSource(
	barTimeline: readonly BarTimelineEntry[],
	bars: readonly BarIR[],
	getTSec: () => number | null,
): () => number | null {
	return () => {
		const tSec = getTSec();
		if (tSec === null) return null;

		let lower: BarTimelineEntry | undefined;
		let upper: BarTimelineEntry | undefined;
		for (let i = 0; i < barTimeline.length - 1; i++) {
			const a = barTimeline[i];
			const b = barTimeline[i + 1];
			if (tSec >= a.tSec && tSec <= b.tSec) {
				lower = a;
				upper = b;
				break;
			}
		}
		if (!lower || !upper) return null;

		const span = upper.tSec - lower.tSec;
		const fraction = span > 0 ? (tSec - lower.tSec) / span : 0;

		const barIR = bars.find((b) => b.barNumber === lower.bar);
		if (!barIR) return null;

		return barIR.qstampStart + fraction * (barIR.qstampEnd - barIR.qstampStart);
	};
}
