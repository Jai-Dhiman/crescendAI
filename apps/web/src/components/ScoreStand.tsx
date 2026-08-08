import { useEffect, useMemo, useRef, useState } from "react";
import { useMetronome } from "../hooks/useMetronome";
import type { Mark } from "../lib/mark";
import { formatElapsed } from "../lib/mark";
import type { BarIR } from "../lib/score-ir";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreMarkLayer } from "./ScoreMarkLayer";

interface ScoreStandProps {
	pieceId: string;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isRecording: boolean;
}

/**
 * The digital music stand (docs/apps/05-ui-system.md#2): a static, manually
 * paged score. Deliberately does not import ScoreCursor — that class exists
 * to drive a moving highlight from a live qstamp source, which is exactly
 * the "live following" this surface forbids. Page turns are the only way
 * the rendered page changes.
 */
export function ScoreStand({
	pieceId,
	marks,
	elapsedSeconds,
	isRecording,
}: ScoreStandProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const svgHostRef = useRef<HTMLDivElement>(null);
	const [pageCount, setPageCount] = useState(0);
	const [currentPage, setCurrentPage] = useState(1);
	const [pageSvg, setPageSvg] = useState<string | null>(null);
	// Full BarIR, not the narrower BarLocator: pageN is what lets this
	// component scope ScoreMarkLayer (Canvas A, lossy by design) to only the
	// bars actually on screen -- a bar on another page has no rect to place a
	// mark against here regardless of what mark-placement.ts does with it.
	const [allBars, setAllBars] = useState<readonly BarIR[]>([]);
	const [error, setError] = useState<string | null>(null);
	const metronome = useMetronome();

	useEffect(() => {
		let cancelled = false;
		async function load() {
			const result = await scoreRenderer.load(pieceId);
			if (cancelled) return;
			if (result === "failed") {
				setError("Score failed to load");
				return;
			}
			setPageCount(result.ir.pages.length);
			setAllBars(result.ir.bars);
		}
		load();
		return () => {
			cancelled = true;
		};
	}, [pieceId]);

	const barsForCurrentPage = useMemo(
		() =>
			allBars
				.filter((b) => b.pageN === currentPage)
				.map((b) => ({ barNumber: b.barNumber, measureOn: b.measureOn })),
		[allBars, currentPage],
	);

	useEffect(() => {
		let cancelled = false;
		async function loadPage() {
			const svg = await scoreRenderer.getPage(pieceId, currentPage);
			if (cancelled) return;
			// Injected imperatively into a dedicated child node, matching
			// src/scorehost/score-host.ts:382: this keeps the SVG in a sibling of
			// ScoreMarkLayer so React never owns or re-reconciles Verovio's DOM,
			// and ScoreMarkLayer's own ResizeObserver-driven measurement effect
			// isn't racing a React commit of the same subtree.
			if (svgHostRef.current) svgHostRef.current.innerHTML = svg;
			setPageSvg(svg);
		}
		if (pageCount > 0) loadPage();
		return () => {
			cancelled = true;
		};
	}, [pieceId, currentPage, pageCount]);

	if (error) {
		return <p className="text-danger">{error}</p>;
	}

	return (
		<div className="flex h-full flex-col">
			<div className="flex shrink-0 items-center justify-between border-b border-border-subtle px-4 py-2">
				<div className="flex items-center gap-2">
					{isRecording && (
						<span
							className="h-2 w-2 rounded-full bg-danger"
							aria-hidden="true"
						/>
					)}
					<span className="text-body-sm tabular-nums text-ink-secondary">
						{formatElapsed(elapsedSeconds)}
					</span>
				</div>
				<button
					type="button"
					onClick={metronome.toggle}
					className="text-label-sm text-ink-tertiary underline"
				>
					{metronome.isPlaying ? `Metronome ${metronome.bpm}` : "Metronome"}
				</button>
			</div>

			<div
				ref={containerRef}
				data-testid="score-stand-page"
				data-current-page={currentPage}
				className="score-container relative flex-1 overflow-auto"
			>
				<div ref={svgHostRef} />
				{pageSvg && (
					// ScoreMarkLayer renders absolute inset-0, so it must be a child of
					// this relative container, not a sibling after it -- a sibling would
					// anchor against the next positioned ancestor up the tree instead
					// (the flex column above), placing every mark at the wrong origin.
					<ScoreMarkLayer
						containerRef={containerRef}
						bars={barsForCurrentPage}
						marks={marks}
					/>
				)}
			</div>

			<div className="flex shrink-0 items-center justify-center gap-4 border-t border-border-subtle px-4 py-2">
				<button
					type="button"
					onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
					disabled={currentPage <= 1}
					className="rounded-full px-3 py-1 text-body-sm text-ink-secondary disabled:opacity-40"
					aria-label="Previous page"
				>
					Prev
				</button>
				<span className="text-body-xs text-ink-tertiary tabular-nums">
					{currentPage} / {pageCount || 1}
				</span>
				<button
					type="button"
					onClick={() => setCurrentPage((p) => Math.min(pageCount, p + 1))}
					disabled={currentPage >= pageCount}
					className="rounded-full px-3 py-1 text-body-sm text-ink-secondary disabled:opacity-40"
					aria-label="Next page"
				>
					Next
				</button>
			</div>
		</div>
	);
}
