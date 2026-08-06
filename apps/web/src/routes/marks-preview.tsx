import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { ScoreMarkLayer } from "../components/ScoreMarkLayer";
import { SessionTimelineStrip } from "../components/SessionTimelineStrip";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { BarLocator } from "../lib/mark-placement";
import { scoreRenderer } from "../lib/score-renderer";
import {
	FIXTURE_BARS,
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";

export const Route = createFileRoute("/marks-preview")({
	component: MarksPreview,
});

// A real piece that ships in public/scores/. The Nocturne rather than the
// Ballade: it is a third the size, and this gate cares that the measureOn chain
// survives a real engraving, not that it survives a long one.
const REAL_PIECE_ID = "chopin-nocturne-op9-no2";

/**
 * The only place in #157 where marks meet a real Verovio engraving. Everything
 * else uses stand-in divs, which verify the resolution logic against
 * correctly-shaped ids but cannot verify that Verovio emits those ids at all.
 */
function RealScoreSection() {
	const containerRef = useRef<HTMLDivElement>(null);
	const svgHostRef = useRef<HTMLDivElement>(null);
	const [svg, setSvg] = useState<string | null>(null);
	const [bars, setBars] = useState<readonly BarLocator[]>([]);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		async function load() {
			try {
				const result = await scoreRenderer.load(REAL_PIECE_ID);
				if (cancelled) return;
				if (result === "failed") {
					setError("Score failed to load");
					return;
				}
				const page = await scoreRenderer.getPage(REAL_PIECE_ID, 1);
				if (cancelled) return;
				// Page 1 only: bars on later pages exercise the unplaced path,
				// which the synthetic section already covers deterministically.
				setBars(
					result.ir.bars
						.filter((b) => b.pageN === 1)
						.map((b) => ({ barNumber: b.barNumber, measureOn: b.measureOn })),
				);
				// Inject BEFORE setSvg, not from a later effect. React runs child
				// effects before parent effects, so a parent-effect injection would
				// let ScoreMarkLayer measure an empty container and place every
				// mark against stale rects. It only looked correct because
				// injecting resized the container and the ResizeObserver happened
				// to fire a re-measure — placement must not depend on that.
				if (svgHostRef.current) svgHostRef.current.innerHTML = page;
				setSvg(page);
			} catch (e) {
				if (!cancelled) setError(String(e));
			}
		}
		load();
		return () => {
			cancelled = true;
		};
	}, []);

	// Anchor a mark to the first bar the IR actually reports, so this never
	// depends on a hardcoded bar number surviving a re-engraving.
	const marks = useMemo<readonly Mark[]>(() => {
		const first = bars[0];
		if (!first) return [];
		return [
			{
				id: "real-1",
				anchor: resolveAnchor({
					atSeconds: 30,
					bars: [first.barNumber, first.barNumber],
					alignmentQuality: 1,
				}),
				taxonomy: "needs_work",
				dimension: "pedaling",
				evidence: "pedal held through the bass change",
				lifecycle: "active",
				confidence: "established",
			},
		];
	}, [bars]);

	// Injected imperatively into a dedicated child node, matching the
	// established pattern at src/scorehost/score-host.ts:382. Two reasons this
	// is not React's dangerouslySetInnerHTML: it follows the code already in
	// the repo, and it keeps the SVG in a sibling of the mark layer so React
	// never owns or re-reconciles Verovio's DOM.
	//
	// Trust boundary: the markup is Verovio's own output, produced by our
	// worker from copyright-cleared score bytes we fetch. No user-supplied
	// content reaches this string. That is the same boundary score-host.ts
	// already accepts.
	if (error) {
		return <p className="text-danger">{error}</p>;
	}

	return (
		<div
			data-testid="real-score"
			ref={containerRef}
			className="score-container relative mb-12 min-h-64 border border-border-subtle"
		>
			<div ref={svgHostRef} />
			{!svg && (
				<p className="inline-block rounded bg-surface-raised px-2 py-0.5 text-ink-tertiary">
					Loading score...
				</p>
			)}
			{svg && (
				<ScoreMarkLayer containerRef={containerRef} bars={bars} marks={marks} />
			)}
		</div>
	);
}

/**
 * Dev preview surface for #157. Deliberately a top-level route rather than a
 * child of /app: /app redirects to /signin when VITE_AUTH_MODE=live, and the
 * a11y run needs to reach this page in a preview build. Removed when the real
 * surfaces (#158/#159/#162) consume the canvases.
 *
 * The measure stand-ins below carry the same ids score-ir emits as
 * BarIR.measureOn, so ScoreMarkLayer's resolution path is exercised for real.
 * Bar 88 is intentionally omitted to exercise the unplaced disclosure.
 */
export function MarksPreview() {
	const scoreRef = useRef<HTMLDivElement>(null);
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);

	return (
		<main className="mx-auto max-w-3xl px-6 py-12">
			<h1 className="mb-8 text-display-sm text-ink-primary">
				Mark system preview
			</h1>

			<h2 className="mb-2 text-label-md text-ink-secondary">Score overlay</h2>
			<div
				ref={scoreRef}
				className="score-container relative mb-12 h-64 border border-border-subtle"
			>
				{onPage.map((b, i) => (
					<div
						key={b.measureOn}
						id={b.measureOn}
						className="absolute h-24 w-24 border border-border-subtle"
						style={{ top: 80, left: 24 + i * 140 }}
					/>
				))}
				<ScoreMarkLayer
					containerRef={scoreRef}
					bars={FIXTURE_BARS}
					marks={FIXTURE_MARKS}
				/>
			</div>

			<h2 className="mb-2 text-label-md text-ink-secondary">
				Score overlay (real engraving)
			</h2>
			<RealScoreSection />

			<h2 className="mb-2 text-label-md text-ink-secondary">
				Session timeline
			</h2>
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>
		</main>
	);
}
