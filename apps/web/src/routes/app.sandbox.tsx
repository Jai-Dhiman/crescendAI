import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Artifact } from "../components/Artifact";
import { ClipSvg } from "../components/ClipSvg";
import { ArtifactOverlay } from "../components/ArtifactOverlay";
import { SegmentLoopArtifactCard } from "../components/cards/SegmentLoopArtifact";
import { PlayPassageCard } from "../components/cards/PlayPassageCard";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";
import { ScoreCursor } from "../lib/score-cursor";
import { scoreRenderer } from "../lib/score-renderer";
import type {
	ExerciseSetConfig,
	KeyboardGuideConfig,
	PassageManifest,
	PlayPassageConfig,
	ScoreHighlightConfig,
	SegmentLoopConfig,
} from "../lib/types";
import { useArtifactStore } from "../stores/artifact";

// --- Fixtures ---

// Sandbox UUIDs: valid format so Zod passes, but not in DB — tests the error state path.
// In production, exerciseIds are server-assigned UUIDs from the create_exercise tool call.
const exerciseSetWithId: ExerciseSetConfig = {
	sourcePassage: "Chopin Op. 10 No. 3, bars 1-8",
	targetSkill: "Cantabile tone production",
	scoreClip: { pieceId: "chopin.ballades.1", bars: [1, 8] },
	exercises: [
		{
			title: "Slow legato melody",
			instruction:
				"Play the RH melody at quarter-note pace, connecting each note with a singing tone. Focus on finger weight rather than pressure.",
			focusDimension: "articulation",
			hands: "right",
			exerciseId: "00000000-0000-0000-0000-000000000001",
		},
		{
			title: "Voicing the top note",
			instruction:
				"In each chord, bring out only the top note while suppressing the inner voices. Repeat 5 times.",
			focusDimension: "dynamics",
			hands: "both",
			exerciseId: "00000000-0000-0000-0000-000000000002",
		},
	],
};

const exerciseSetNoId: ExerciseSetConfig = {
	sourcePassage: "Beethoven Sonata Op. 13 (Pathetique), bars 5-12",
	targetSkill: "LH octave evenness",
	scoreClip: { pieceId: "chopin.ballades.1", bars: [36, 43] },
	exercises: [
		{
			title: "Octave isolation",
			instruction:
				"Practice LH octaves staccato at 60 BPM, ensuring each octave lands simultaneously.",
			focusDimension: "timing",
			hands: "left",
		},
		{
			title: "Dynamic shaping",
			instruction:
				"Replay the passage with a clear crescendo from p to f over 8 bars.",
			focusDimension: "dynamics",
			hands: "left",
		},
	],
};

// Score highlight fixtures — spread across the piece to test clip rendering at
// different measure index positions (opening, mid, climax, near-end).
const scoreHighlightOpening: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{
			bars: [1, 4],
			dimension: "dynamics",
			annotation: "Establish hushed, questioning p throughout",
		},
		{
			bars: [5, 8],
			dimension: "phrasing",
			annotation: "Lift at the phrase peak on beat 3",
		},
	],
};

const scoreHighlightWide: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{
			bars: [1, 8],
			dimension: "dynamics",
			annotation: "Opening statement — hushed and questioning",
		},
		{
			bars: [9, 16],
			dimension: "phrasing",
			annotation: "Secondary theme entry — shape the long arc",
		},
		{
			bars: [17, 24],
			dimension: "timing",
			annotation: "Subtle rubato — breathe at bar 20",
		},
	],
};

const scoreHighlightMid: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{
			bars: [94, 97],
			dimension: "timing",
			annotation: "First climax — push through to the peak",
		},
		{
			bars: [106, 112],
			dimension: "dynamics",
			annotation: "Sudden ff — commit to the outburst",
		},
	],
};

const scoreHighlightLate: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{
			bars: [200, 207],
			dimension: "phrasing",
			annotation: "Coda approach — broaden the phrase",
		},
		{
			bars: [250, 264],
			dimension: "interpretation",
			annotation: "Presto coda — controlled abandon",
		},
	],
};

const keyboardGuide: KeyboardGuideConfig = {};

// ---------------------------------------------------------------------------
// PlayPassage sandbox fixtures
// ---------------------------------------------------------------------------

// Minimal staff SVG — used as the mock score clip so the ready state renders
// without needing a real score from the API.
const MOCK_SCORE_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 72" width="400" height="72">
  <line x1="12" y1="16" x2="388" y2="16" stroke="#444" stroke-width="0.8"/>
  <line x1="12" y1="26" x2="388" y2="26" stroke="#444" stroke-width="0.8"/>
  <line x1="12" y1="36" x2="388" y2="36" stroke="#444" stroke-width="0.8"/>
  <line x1="12" y1="46" x2="388" y2="46" stroke="#444" stroke-width="0.8"/>
  <line x1="12" y1="56" x2="388" y2="56" stroke="#444" stroke-width="0.8"/>
  <line x1="12"  y1="16" x2="12"  y2="56" stroke="#444" stroke-width="1.2"/>
  <line x1="200" y1="16" x2="200" y2="56" stroke="#444" stroke-width="0.8"/>
  <line x1="388" y1="16" x2="388" y2="56" stroke="#444" stroke-width="1.8"/>
  <line x1="384" y1="16" x2="384" y2="56" stroke="#444" stroke-width="0.8"/>
  <ellipse cx="55"  cy="31" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 55 31)"/>
  <line x1="61"  y1="31" x2="61"  y2="10" stroke="#222" stroke-width="1.2"/>
  <ellipse cx="95"  cy="41" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 95 41)"/>
  <line x1="101" y1="41" x2="101" y2="20" stroke="#222" stroke-width="1.2"/>
  <ellipse cx="135" cy="26" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 135 26)"/>
  <line x1="141" y1="26" x2="141" y2="5"  stroke="#222" stroke-width="1.2"/>
  <ellipse cx="168" cy="36" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 168 36)"/>
  <line x1="174" y1="36" x2="174" y2="15" stroke="#222" stroke-width="1.2"/>
  <ellipse cx="240" cy="46" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 240 46)"/>
  <line x1="246" y1="46" x2="246" y2="25" stroke="#222" stroke-width="1.2"/>
  <ellipse cx="280" cy="31" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 280 31)"/>
  <line x1="286" y1="31" x2="286" y2="10" stroke="#222" stroke-width="1.2"/>
  <ellipse cx="320" cy="21" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 320 21)"/>
  <line x1="326" y1="21" x2="326" y2="0"  stroke="#222" stroke-width="1.2"/>
  <ellipse cx="358" cy="41" rx="6.5" ry="4.5" fill="#222" transform="rotate(-18 358 41)"/>
  <line x1="364" y1="41" x2="364" y2="20" stroke="#222" stroke-width="1.2"/>
</svg>`;

const MOCK_CLIP_SVG: string = MOCK_SCORE_SVG;

// Fake manifest — chunk URLs will 404, triggering the audio_error state
// (score + annotation still render; play button shows "Audio unavailable").
const MOCK_MANIFEST: PassageManifest = {
	source: { kind: "session", sessionId: "00000000-0000-0000-0000-0000000000ff" },
	pieceId: "chopin.ballades.1",
	bars: [5, 8],
	chunks: [
		{
			url: "/api/practice/chunk?sessionId=00000000-0000-0000-0000-0000000000ff&chunkIndex=0",
			chunkIndex: 0,
			durationSec: 15,
		},
	],
	startOffsetSec: 1.0,
	endOffsetSec: 13.0,
	barTimeline: [
		{ bar: 5, tSec: 0 },
		{ bar: 6, tSec: 3 },
		{ bar: 7, tSec: 6 },
		{ bar: 8, tSec: 9 },
	],
};

const MOCK_MANIFEST_MULTI_CHUNK: PassageManifest = {
	...MOCK_MANIFEST,
	bars: [1, 8],
	chunks: [
		{
			url: "/api/practice/chunk?sessionId=00000000-0000-0000-0000-0000000000ff&chunkIndex=0",
			chunkIndex: 0,
			durationSec: 15,
		},
		{
			url: "/api/practice/chunk?sessionId=00000000-0000-0000-0000-0000000000ff&chunkIndex=1",
			chunkIndex: 1,
			durationSec: 15,
		},
	],
	startOffsetSec: 0,
	endOffsetSec: 13.0,
	barTimeline: [
		{ bar: 1, tSec: 0 },
		{ bar: 2, tSec: 3 },
		{ bar: 3, tSec: 6 },
		{ bar: 4, tSec: 9 },
		{ bar: 5, tSec: 15 },
		{ bar: 6, tSec: 18 },
		{ bar: 7, tSec: 21 },
		{ bar: 8, tSec: 24 },
	],
};

// --- PlayPassage scenario configs ---

const ppTiming: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-0000000000ff",
	bars: [5, 8],
	focusBars: [6, 7],
	dimension: "timing",
	annotation: "You rushed through the triplets in bar 6 — try holding each beat a hair longer to let the phrase breathe.",
};

const ppDynamics: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-0000000000ff",
	bars: [5, 8],
	dimension: "dynamics",
	annotation: "The crescendo in bars 5–8 barely registers. Aim for a two-level jump in arm weight from bar 6 to bar 8.",
};

const ppPedaling: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-0000000000ff",
	bars: [5, 8],
	focusBars: [5, 6],
	dimension: "pedaling",
	annotation: "Harmonic change on beat 3 of bar 5 — the pedal needs to lift and re-engage there, not carry over.",
};

const ppPhrasing: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-0000000000ff",
	bars: [1, 8],
	focusBars: [5, 8],
	dimension: "phrasing",
	annotation: "The phrase peak should land on the downbeat of bar 7, not bar 5. Right now the energy peaks too early and the second half collapses.",
};

const ppLongAnnotation: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-0000000000ff",
	bars: [5, 8],
	focusBars: [6, 7],
	dimension: "interpretation",
	annotation: "There is a subtle but important distinction between rubato as an ornament and rubato as the structural shaping of a phrase. What you are doing in bars 6 and 7 is ornamental — small fluctuations that feel decorative. The phrase wants structural rubato: a broad elastic pull across all four bars where the time stretches through bar 6 and snaps back cleanly at bar 8.",
};

const ppFetchError: PlayPassageConfig = {
	sessionId: "00000000-0000-0000-0000-000000000000",
	bars: [5, 8],
	focusBars: [6, 7],
	dimension: "timing",
	annotation: "This session has no alignment data — card should show the fetch-error state.",
};

// ---------------------------------------------------------------------------
// ScoreHighlight edge cases
// ---------------------------------------------------------------------------

// Error state: invalid pieceId → getClip throws → annotation list renders without score clips
const scoreHighlightError: ScoreHighlightConfig = {
	pieceId: "invalid.piece.id",
	highlights: [
		{
			bars: [1, 4],
			dimension: "dynamics",
			annotation: "Score failed to load — annotation still renders below (no clip above)",
		},
	],
};

// No annotation: dimension label + bar range only
const scoreHighlightNoAnnotation: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{ bars: [1, 4], dimension: "dynamics" },
		{ bars: [5, 8], dimension: "timing" },
	],
};

// Single-bar clip (startBar === endBar): edge case for the viewBox crop logic
const scoreHighlightSingleBar: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{ bars: [8, 8], dimension: "phrasing", annotation: "Isolate bar 8 only — single-measure crop edge case" },
	],
};

// Out-of-range bars (piece ends at bar 264): documents actual behavior
const scoreHighlightOutOfRange: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{ bars: [300, 310], dimension: "interpretation", annotation: "Bars beyond piece end — expect error or empty clip" },
	],
};

// ---------------------------------------------------------------------------
// ExerciseSet edge cases
// ---------------------------------------------------------------------------

// Single exercise with exerciseId — minimal layout, "Add to practice" button present
// Clicking will show "Try again" (fake UUID → 404 from API)
const exerciseSingle: ExerciseSetConfig = {
	sourcePassage: "Chopin Op. 10 No. 3, bar 1",
	targetSkill: "Opening phrase shape",
	exercises: [
		{
			title: "Shape the melody peak",
			instruction:
				"Play bar 1 five times, crescendo to beat 3, diminuendo to the end. Keep the LH softer.",
			focusDimension: "dynamics",
			hands: "right",
			exerciseId: "00000000-0000-0000-0000-000000000003",
		},
	],
};

// 3 exercises, all with IDs — tests list layout; clicking any button shows "Try again"
const exerciseThreeWithIds: ExerciseSetConfig = {
	sourcePassage: "Beethoven Op. 13, bars 1-16",
	targetSkill: "LH / RH coordination",
	exercises: [
		{
			title: "LH alone, slow",
			instruction:
				"LH only at 50 BPM, no pedal. Focus on even weight between fingers 4 and 5.",
			focusDimension: "articulation",
			hands: "left",
			exerciseId: "00000000-0000-0000-0000-000000000004",
		},
		{
			title: "RH alone, shape the line",
			instruction:
				"RH melody at 60 BPM. Bring out every phrase peak by lifting into the note.",
			focusDimension: "phrasing",
			hands: "right",
			exerciseId: "00000000-0000-0000-0000-000000000005",
		},
		{
			title: "Hands together, block chords",
			instruction:
				"Reduce RH to block chords, align each chord's attack exactly with the LH beat. Then restore the melody.",
			focusDimension: "timing",
			hands: "both",
			exerciseId: "00000000-0000-0000-0000-000000000006",
		},
	],
};

// Mixed: first exercise has no exerciseId (no button); second does (button present)
const exerciseMixed: ExerciseSetConfig = {
	sourcePassage: "Schumann Kinderszenen No. 7, bars 1-8",
	targetSkill: "Singing tone in slow melody",
	exercises: [
		{
			title: "Listen first (no button)",
			instruction:
				"Listen to a recording of bars 1-8. Notice where the phrase peaks and breathes.",
			focusDimension: "interpretation",
			hands: "both",
		},
		{
			title: "Replicate the shape (has button)",
			instruction:
				"Play bars 1-8 aiming to match the phrasing shape you heard. Let the melody lead; keep the LH harmonic support quiet.",
			focusDimension: "phrasing",
			hands: "both",
			exerciseId: "00000000-0000-0000-0000-000000000007",
		},
	],
};

// Long text — tests title truncation and instruction wrapping
const exerciseLongText: ExerciseSetConfig = {
	sourcePassage:
		"Chopin Nocturne Op. 9 No. 2 in E-flat major, bars 1-12 (opening statement through first return)",
	targetSkill: "Ornamental right-hand phrasing in the Chopin nocturne style",
	exercises: [
		{
			title: "Slow-practice the ornament chain in bar 4",
			instruction:
				"The turn-into-trill ornament on beat 3 of bar 4 is the single most common breakdown point for intermediate players. Practice it as: hold the main note for a full beat, then execute the turn at exactly half the intended speed. The trill should feel like a continuation, not a surprise. Do not rush into it. Repeat at least ten times before returning to tempo. The goal is proprioceptive familiarity — your hand should know exactly how far to travel without looking.",
			focusDimension: "articulation",
			hands: "right",
			exerciseId: "00000000-0000-0000-0000-000000000008",
		},
	],
};

// ---------------------------------------------------------------------------
// SegmentLoop fixtures — one per status value
// Button clicks call real endpoints with a fake UUID → network/404 → shows error text
// ---------------------------------------------------------------------------

const slBase: Omit<SegmentLoopConfig, "status" | "attemptsCompleted"> = {
	id: "00000000-0000-0000-0000-000000000010",
	pieceId: "chopin.ballades.1",
	barsStart: 5,
	barsEnd: 8,
	requiredCorrect: 3,
	dimension: "timing",
};

const slPending: SegmentLoopConfig = { ...slBase, status: "pending", attemptsCompleted: 0 };
const slActive: SegmentLoopConfig = { ...slBase, status: "active", attemptsCompleted: 1 };
const slCompleted: SegmentLoopConfig = { ...slBase, status: "completed", attemptsCompleted: 3 };
const slDismissed: SegmentLoopConfig = { ...slBase, status: "dismissed", attemptsCompleted: 1 };
const slSuperseded: SegmentLoopConfig = { ...slBase, status: "superseded", attemptsCompleted: 0 };

// --- Artifact IDs ---

const SANDBOX_IDS = {
	exerciseWithId: "sandbox-exercise-set-1",
	exerciseNoId: "sandbox-exercise-set-2",
	exerciseSingle: "sandbox-exercise-single",
	exerciseThreeWithIds: "sandbox-exercise-three-with-ids",
	exerciseMixed: "sandbox-exercise-mixed",
	exerciseLongText: "sandbox-exercise-long-text",
	scoreHighlightOpening: "sandbox-score-highlight-opening",
	scoreHighlightWide: "sandbox-score-highlight-wide",
	scoreHighlightMid: "sandbox-score-highlight-mid",
	scoreHighlightLate: "sandbox-score-highlight-late",
	scoreHighlightError: "sandbox-score-highlight-error",
	scoreHighlightNoAnnotation: "sandbox-score-highlight-no-annotation",
	scoreHighlightSingleBar: "sandbox-score-highlight-single-bar",
	scoreHighlightOutOfRange: "sandbox-score-highlight-out-of-range",
	keyboardGuide: "sandbox-keyboard-guide-1",
	ppTiming: "sandbox-pp-timing",
	ppDynamics: "sandbox-pp-dynamics",
	ppPedaling: "sandbox-pp-pedaling",
	ppPhrasing: "sandbox-pp-phrasing",
	ppLongAnnotation: "sandbox-pp-long-annotation",
	ppMultiChunk: "sandbox-pp-multi-chunk",
	ppFetchError: "sandbox-pp-fetch-error",
	ppPlayable: "sandbox-pp-playable",
	ppAudioError: "sandbox-pp-audio-error",
	slPending: "sandbox-sl-pending",
	slActive: "sandbox-sl-active",
	slCompleted: "sandbox-sl-completed",
	slDismissed: "sandbox-sl-dismissed",
	slSuperseded: "sandbox-sl-superseded",
} as const;

// --- Section wrapper ---

interface SandboxSectionProps {
	title: string;
	artifactId: string;
	children: React.ReactNode;
}

function SandboxSection({ title, artifactId, children }: SandboxSectionProps) {
	const expand = useArtifactStore((s) => s.expand);
	const unregister = useArtifactStore((s) => s.unregister);
	const [key, setKey] = useState(0);

	function handleReset() {
		unregister(artifactId);
		setKey((k) => k + 1);
	}

	return (
		<section className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
			<div className="flex items-center justify-between">
				<h2 className="font-display text-display-xs text-cream">{title}</h2>
				<div className="flex items-center gap-2">
					<button
						type="button"
						onClick={() => expand(artifactId)}
						className="px-3 py-1.5 rounded-lg bg-surface border border-border text-body-sm text-text-secondary hover:text-cream hover:border-accent transition"
					>
						Expand
					</button>
					<button
						type="button"
						onClick={handleReset}
						className="px-3 py-1.5 rounded-lg bg-surface border border-border text-body-sm text-text-secondary hover:text-cream hover:border-accent transition"
					>
						Reset
					</button>
				</div>
			</div>
			<div key={key}>{children}</div>
		</section>
	);
}

// --- Approach comparison (bars 135–136, Chopin Ballade No. 1) ---

// --- Score renderer primitives ---

function SvgPanel({ svgMarkup }: { svgMarkup: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		ref.current.insertAdjacentHTML("afterbegin", svgMarkup);
	}, [svgMarkup]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

// Full-score panel with drag-to-resize. Bars reflow on drag-end (approach A).
function ScoreResizePanel({ pieceId }: { pieceId: string }) {
	const [svg, setSvg] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [width, setWidth] = useState(480);
	const panelRef = useRef<HTMLDivElement>(null);
	const cleanupRef = useRef<(() => void) | null>(null);
	// Memoize loaded pieceIds to avoid redundant load() calls on re-render.
	const loadedPieceRef = useRef<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		async function loadAndGetPage() {
			try {
				if (loadedPieceRef.current !== pieceId) {
					const result = await scoreRenderer.load(pieceId);
					if (cancelled) return;
					if (result === "failed") {
						setError("Score failed to load");
						return;
					}
					loadedPieceRef.current = pieceId;
				}
				const s = await scoreRenderer.getPage(pieceId, 1);
				if (!cancelled) setSvg(s);
			} catch (e) {
				if (!cancelled) setError(String(e));
			}
		}
		loadAndGetPage();
		return () => {
			cancelled = true;
			cleanupRef.current?.();
		};
	}, [pieceId]);

	function handleDragStart(e: React.MouseEvent) {
		e.preventDefault();
		const startX = e.clientX;
		const startWidth = width;

		function onMove(ev: MouseEvent) {
			const w = Math.max(
				200,
				Math.min(700, startWidth + (ev.clientX - startX)),
			);
			if (panelRef.current) panelRef.current.style.width = `${w}px`;
		}

		function cleanup() {
			document.removeEventListener("mousemove", onMove);
			document.removeEventListener("mouseup", onUp);
			document.body.style.cursor = "";
			cleanupRef.current = null;
		}

		function onUp(ev: MouseEvent) {
			const w = Math.max(
				200,
				Math.min(700, startWidth + (ev.clientX - startX)),
			);
			setWidth(w);
			// load() is already guaranteed by the mount effect; getPage() is safe here.
			scoreRenderer
				.getPage(pieceId, 1, Math.round(w / 0.4))
				.then(setSvg)
				.catch(() => {});
			cleanup();
		}

		document.addEventListener("mousemove", onMove);
		document.addEventListener("mouseup", onUp);
		document.body.style.cursor = "col-resize";
		cleanupRef.current = cleanup;
	}

	return (
		<div className="flex flex-col gap-2">
			<div
				ref={panelRef}
				className="relative border border-border rounded-lg overflow-hidden bg-white"
				style={{ width }}
			>
				{error && <p className="text-body-xs text-red-400 p-2">{error}</p>}
				{!svg && !error && (
					<p className="text-body-xs text-text-tertiary p-2">Loading…</p>
				)}
				{svg && <SvgPanel svgMarkup={svg} />}
				<div
					onMouseDown={handleDragStart}
					className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize bg-border hover:bg-accent transition-colors"
				/>
			</div>
			<span className="text-body-xs text-text-tertiary">
				Width: {width}px — drag right edge to reflow bars
			</span>
		</div>
	);
}

interface ClipTest {
	label: string;
	startBar: number;
	endBar: number;
}

function ScoreClipPanel({
	label,
	pieceId,
	startBar,
	endBar,
}: ClipTest & { pieceId: string }) {
	const [svg, setSvg] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	// Memoize loaded pieceIds to avoid redundant load() calls across clip re-renders.
	const loadedPieceRef = useRef<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		async function loadAndGetClip() {
			try {
				if (loadedPieceRef.current !== pieceId) {
					const result = await scoreRenderer.load(pieceId);
					if (cancelled) return;
					if (result === "failed") {
						setError("Score failed to load");
						return;
					}
					loadedPieceRef.current = pieceId;
				}
				const s = await scoreRenderer.getClip(pieceId, startBar, endBar);
				if (!cancelled) setSvg(s);
			} catch (e) {
				if (!cancelled) setError(String(e));
			}
		}
		loadAndGetClip();
		return () => {
			cancelled = true;
		};
	}, [pieceId, startBar, endBar]);

	return (
		<div className="flex flex-col gap-1">
			<span className="text-label-sm text-accent font-mono">{label}</span>
			<div className="border border-border rounded-lg overflow-hidden bg-white">
				{error && <p className="text-body-xs text-red-400 p-2">{error}</p>}
				{!svg && !error && (
					<p className="text-body-xs text-text-tertiary p-2">Loading…</p>
				)}
				{svg && <ClipSvg svg={svg} />}
			</div>
		</div>
	);
}

// Playback cursor demo. Loads a piece, renders page 1, instantiates ScoreCursor
// with a synthetic qstampSource that sweeps wall-clock time -> qstamp. Pause
// hides the cursor (qstampSource returns null). The sweep is clamped to page-1
// bars so the visible cursor stays aligned with the rendered SVG; if it crossed
// to page 2 in this single-page demo container, the overlay would visually jump
// back to top-left (each overlay is absolutely positioned at 0,0 by design).
function ScoreCursorPanel({ pieceId }: { pieceId: string }) {
	const [error, setError] = useState<string | null>(null);
	const [isPlaying, setIsPlaying] = useState(false);
	const [ready, setReady] = useState(false);
	const [maxQstamp, setMaxQstamp] = useState(0);
	const [debug, setDebug] = useState({
		q: 0,
		lineX: 0,
		lineVis: "?",
		bar: 0,
		overlayVB: "?",
		verovioVB: "?",
	});
	const containerRef = useRef<HTMLDivElement>(null);
	const cursorRef = useRef<ScoreCursor | null>(null);
	const playStartedAtRef = useRef<number | null>(null);
	const pausedOffsetRef = useRef<number>(0);
	const maxQstampRef = useRef<number>(0);
	const isPlayingRef = useRef(false);
	isPlayingRef.current = isPlaying;
	const lastQstampRef = useRef<number | null>(null);
	const lastBarRef = useRef<number>(0);

	// qstamp units per second of wall-clock; tuned for visible motion.
	const QSTAMP_PER_SEC = 4;

	useEffect(() => {
		let cancelled = false;
		async function setup() {
			try {
				const result = await scoreRenderer.load(pieceId);
				if (cancelled) return;
				if (result === "failed") {
					setError("Score failed to load");
					return;
				}
				const ir = scoreRenderer.getIR(pieceId);
				if (!ir) {
					setError("IR not available after load");
					return;
				}
				const svg = await scoreRenderer.getPage(pieceId, 1);
				if (cancelled || !containerRef.current) return;
				// SVG is produced by our trusted Verovio worker over MXL files we control;
				// matches the SvgPanel pattern used elsewhere in this sandbox.
				containerRef.current.textContent = "";
				containerRef.current.insertAdjacentHTML("afterbegin", svg);

				// Clamp the sweep to bars whose pageN === 1 so the cursor stays on
				// the rendered page. Falls back to first bar's end if no page-1 bars.
				const page1End =
					ir.bars
						.filter((b) => b.pageN === 1)
						.reduce((m, b) => Math.max(m, b.qstampEnd), 0) ||
					ir.bars[0]?.qstampEnd ||
					0;
				maxQstampRef.current = page1End;
				setMaxQstamp(page1End);

				const cursor = new ScoreCursor({
					pieceId,
					container: containerRef.current,
					ir,
					qstampSource: () => {
						if (!isPlayingRef.current || playStartedAtRef.current === null) {
							lastQstampRef.current = null;
							return null;
						}
						const elapsedSec =
							(performance.now() - playStartedAtRef.current) / 1000;
						const raw = pausedOffsetRef.current + elapsedSec * QSTAMP_PER_SEC;
						const q = maxQstampRef.current > 0
							? raw % maxQstampRef.current
							: 0;
						lastQstampRef.current = q;
						// Find current bar for debug readout.
						const bar = ir.bars.find(
							(b) => b.qstampStart <= q && q < b.qstampEnd,
						);
						lastBarRef.current = bar?.barNumber ?? 0;
						return q;
					},
				});
				cursor.start();
				cursorRef.current = cursor;
				// Make the cursor's line bright + thick for the demo so it's
				// obvious whether it's positioned and visible. Cursor module
				// hardcodes #2563eb / width 2 — we override after start().
				for (const overlay of Array.from(
					containerRef.current.querySelectorAll("svg.score-cursor-overlay"),
				)) {
					const line = overlay.querySelector("line");
					if (line) {
						line.setAttribute("stroke", "#ef4444");
						line.setAttribute("stroke-width", "6");
					}
				}

				// Sample the cursor state ~10x/sec for a visible debug readout.
				const debugInterval = window.setInterval(() => {
					const containerEl = containerRef.current;
					if (!containerEl) return;
					const overlay = containerEl.querySelector(
						"svg.score-cursor-overlay",
					);
					const line = overlay?.querySelector("line");
					// The Verovio SVG is the first <svg> child that isn't our overlay.
					const verovioSvg = Array.from(
						containerEl.querySelectorAll("svg"),
					).find((s) => !s.classList.contains("score-cursor-overlay"));
					setDebug({
						q: lastQstampRef.current ?? Number.NaN,
						lineX: line ? Number(line.getAttribute("x1") ?? "0") : -1,
						lineVis:
							(line?.getAttribute("visibility") as string | null) ?? "?",
						bar: lastBarRef.current,
						overlayVB: overlay?.getAttribute("viewBox") ?? "?",
						verovioVB: verovioSvg?.getAttribute("viewBox") ?? "?",
					});
				}, 100);
				(cursor as unknown as { __debugInterval: number }).__debugInterval =
					debugInterval;

				setReady(true);
			} catch (e) {
				if (!cancelled) setError(String(e));
			}
		}
		setup();
		return () => {
			cancelled = true;
			const c = cursorRef.current as
				| (ScoreCursor & { __debugInterval?: number })
				| null;
			if (c?.__debugInterval !== undefined) {
				window.clearInterval(c.__debugInterval);
			}
			cursorRef.current?.stop();
			cursorRef.current = null;
		};
	}, [pieceId]);

	function handlePlayPause() {
		if (isPlaying) {
			if (playStartedAtRef.current !== null) {
				const elapsedSec =
					(performance.now() - playStartedAtRef.current) / 1000;
				pausedOffsetRef.current += elapsedSec * QSTAMP_PER_SEC;
			}
			playStartedAtRef.current = null;
			setIsPlaying(false);
		} else {
			playStartedAtRef.current = performance.now();
			setIsPlaying(true);
		}
	}

	function handleReset() {
		pausedOffsetRef.current = 0;
		if (isPlayingRef.current) {
			playStartedAtRef.current = performance.now();
		}
	}

	return (
		<div className="flex flex-col gap-2">
			<div className="flex gap-2 items-center">
				<button
					type="button"
					onClick={handlePlayPause}
					disabled={!ready}
					className="px-3 py-1 bg-accent text-cream rounded text-body-sm disabled:opacity-50"
				>
					{isPlaying ? "Pause" : "Play"}
				</button>
				<button
					type="button"
					onClick={handleReset}
					disabled={!ready}
					className="px-3 py-1 border border-border text-text-secondary rounded text-body-sm disabled:opacity-50"
				>
					Reset
				</button>
				<span className="text-body-xs text-text-tertiary">
					page-1 qstamp: 0 → {maxQstamp.toFixed(2)} (≈ {QSTAMP_PER_SEC}/sec, loops)
				</span>
			</div>
			<div className="font-mono text-body-xs text-text-tertiary whitespace-pre">
				{`DEBUG: q=${Number.isNaN(debug.q) ? "null" : debug.q.toFixed(2)} bar=${debug.bar} lineX=${debug.lineX.toFixed(1)} vis=${debug.lineVis}\n  overlayVB="${debug.overlayVB}"\n  verovioVB="${debug.verovioVB}"`}
			</div>
			{/* Status sits OUTSIDE the cursor container: ScoreCursor owns its container's
			    DOM imperatively (appendChild overlays, textContent wipes), and React must
			    not also be managing children inside it — otherwise the reconciler crashes
			    with "node to be removed is not a child of this node" when React tries to
			    unmount a child we already wiped. */}
			{error && <p className="text-body-xs text-red-400">{error}</p>}
			{!ready && !error && (
				<p className="text-body-xs text-text-tertiary">Loading…</p>
			)}
			<div
				ref={containerRef}
				className="relative border border-border rounded-lg overflow-hidden bg-white [&>svg]:w-full [&>svg]:block"
			/>
		</div>
	);
}

// Chopin Ballade No. 1 is ~264 bars. Tests span: opening, single-bar isolation,
// first theme, likely page-boundary zone, climax, and final bars.
const CLIP_TESTS: ClipTest[] = [
	{
		label: "Bars 1-4 — opening (sparse, start of measure index)",
		startBar: 1,
		endBar: 4,
	},
	{ label: "Bar 8 — single bar isolation", startBar: 8, endBar: 8 },
	{
		label: "Bars 36-43 — first theme (denser notation)",
		startBar: 36,
		endBar: 43,
	},
	{
		label: "Bars 68-80 — transition (likely page boundary at scale 40%)",
		startBar: 68,
		endBar: 80,
	},
	{ label: "Bars 166-173 — climax region", startBar: 166, endBar: 173 },
	{
		label: "Bars 250-260 — near end (high measure index)",
		startBar: 250,
		endBar: 260,
	},
	{
		label: "Bars 261-264 — final bars (end-of-piece edge case)",
		startBar: 261,
		endBar: 264,
	},
];

// ---------------------------------------------------------------------------
// PlayablePassageSection — generates a WAV blob URL on mount so PassagePlayer
// can actually load and decode audio, making the Play button functional.
// Must be a component because URL.createObjectURL requires browser context (no SSR).
// ---------------------------------------------------------------------------

function makeWavBlobUrl(durationSec: number, freqs: number[]): string {
	const sr = 22050;
	const n = Math.floor(sr * durationSec);
	const buf = new ArrayBuffer(44 + n * 2);
	const view = new DataView(buf);
	const write4 = (pos: number, val: string) =>
		[...val].forEach((c, i) => view.setUint8(pos + i, c.charCodeAt(0)));
	write4(0, "RIFF"); view.setUint32(4, 36 + n * 2, true);
	write4(8, "WAVE"); write4(12, "fmt ");
	view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
	view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
	view.setUint16(32, 2, true); view.setUint16(34, 16, true);
	write4(36, "data"); view.setUint32(40, n * 2, true);
	for (let i = 0; i < n; i++) {
		const t = i / sr;
		const decay = Math.exp(-t * 2.5);
		const amp = freqs.reduce((s, f) => s + Math.sin(2 * Math.PI * f * t), 0) / freqs.length;
		view.setInt16(44 + i * 2, Math.round(amp * decay * 28000), true);
	}
	return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

interface PlayablePassageSectionProps {
	artifactId: string;
	title: string;
	config: PlayPassageConfig;
	audioError?: boolean;
}

function PlayablePassageSection({ artifactId, title, config, audioError }: PlayablePassageSectionProps) {
	const [manifest, setManifest] = useState<PassageManifest | null>(null);

	useEffect(() => {
		if (audioError) {
			// Use fake chunk URLs → PassagePlayer will fail to decode → audio_error state
			setManifest(MOCK_MANIFEST);
			return;
		}
		// C major triad (C4, E4, G4) with exponential decay — ~piano-like chord
		const url = makeWavBlobUrl(15, [261.63, 329.63, 392.0]);
		setManifest({
			source: { kind: "session", sessionId: config.sessionId },
			pieceId: "chopin.ballades.1",
			bars: config.bars,
			chunks: [{ url, chunkIndex: 0, durationSec: 15 }],
			startOffsetSec: 1.0,
			endOffsetSec: 13.0,
			barTimeline: [
				{ bar: config.bars[0], tSec: 0 },
				{ bar: config.bars[0] + 1, tSec: 3 },
				{ bar: config.bars[0] + 2, tSec: 6 },
				{ bar: config.bars[1], tSec: 9 },
			],
		});
	}, [config.sessionId, config.bars, audioError]);

	return (
		<SandboxSection title={title} artifactId={artifactId}>
			{manifest ? (
				<PlayPassageCard
					config={config}
					_mockManifest={manifest}
					_mockClip={MOCK_CLIP_SVG}
					_playable
				/>
			) : (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
		</SandboxSection>
	);
}

// --- Page ---

function ArtifactSandbox() {
	const scrollRef = useRef<HTMLDivElement>(null);

	return (
		<ArtifactScrollContext.Provider value={scrollRef}>
			<div
				ref={scrollRef}
				className="overflow-y-auto h-full bg-espresso text-cream"
			>
				<div className="mx-auto max-w-2xl px-4 py-10 flex flex-col gap-8">
					{/* Header */}
					<div>
						<h1 className="font-display text-display-md text-cream">
							Artifact Sandbox
						</h1>
						<p className="text-body-sm text-text-secondary mt-1">
							Dev-only. Test each artifact type in isolation.
						</p>
					</div>

					{/* ── SCORE ─────────────────────────────────────────────────── */}
					<h2 className="font-display text-display-sm text-text-tertiary tracking-wide uppercase text-xs">Score</h2>

					{/* Full score — resizable, bars reflow on drag-end */}
					<section className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
						<div>
							<h2 className="font-display text-display-xs text-cream">
								Full Score — Resizable
							</h2>
							<p className="text-body-sm text-text-secondary mt-1">
								Page 1, Verovio re-renders on drag-end. Drag right edge wide and
								narrow to see bar reflow.
							</p>
						</div>
						<ScoreResizePanel pieceId="chopin.ballades.1" />
					</section>

					{/* Clip grid — static renders at various positions in the piece */}
					<section className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
						<div>
							<h2 className="font-display text-display-xs text-cream">
								Score Clips
							</h2>
							<p className="text-body-sm text-text-secondary mt-1">
								Each clip renders the page containing startBar, then crops the
								viewBox to show only the requested bar range. Tests measure
								index accuracy, page boundaries, single-bar, and end-of-piece
								edge cases.
							</p>
						</div>
						<div className="flex flex-col gap-5">
							{CLIP_TESTS.map((t) => (
								<ScoreClipPanel
									key={`${t.startBar}-${t.endBar}`}
									pieceId="chopin.ballades.1"
									{...t}
								/>
							))}
						</div>
					</section>

					{/* Playback cursor — synthetic qstampSource sweep over page 1 */}
					<section className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
						<div>
							<h2 className="font-display text-display-xs text-cream">
								Playback Cursor
							</h2>
							<p className="text-body-sm text-text-secondary mt-1">
								Loads the piece, calls scoreRenderer.getIR(), and instantiates
								a ScoreCursor with a synthetic qstampSource that sweeps page-1
								qstamps. Play/Pause toggles whether qstampSource returns a
								number or null (null hides the cursor). Reset restarts the
								sweep at qstamp 0.
							</p>
						</div>
						<ScoreCursorPanel pieceId="chopin.ballades.1" />
					</section>

					{/* ScoreHighlight artifacts — tests the full artifact rendering pipeline */}
					<SandboxSection
						title="ScoreHighlight — Opening (bars 1-8)"
						artifactId={SANDBOX_IDS.scoreHighlightOpening}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightOpening}
							component={{
								type: "score_highlight",
								config: scoreHighlightOpening,
							}}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — Wide opening (bars 1-24, 3 highlights)"
						artifactId={SANDBOX_IDS.scoreHighlightWide}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightWide}
							component={{
								type: "score_highlight",
								config: scoreHighlightWide,
							}}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — Mid-piece (bars 94-112)"
						artifactId={SANDBOX_IDS.scoreHighlightMid}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightMid}
							component={{ type: "score_highlight", config: scoreHighlightMid }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — Late piece / final bars (bars 200-264)"
						artifactId={SANDBOX_IDS.scoreHighlightLate}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightLate}
							component={{
								type: "score_highlight",
								config: scoreHighlightLate,
							}}
						/>
					</SandboxSection>

					{/* ScoreHighlight edge cases */}
					<SandboxSection
						title="ScoreHighlight — error (invalid pieceId, annotation still renders)"
						artifactId={SANDBOX_IDS.scoreHighlightError}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightError}
							component={{ type: "score_highlight", config: scoreHighlightError }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — no annotation (dimension + bar range only)"
						artifactId={SANDBOX_IDS.scoreHighlightNoAnnotation}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightNoAnnotation}
							component={{ type: "score_highlight", config: scoreHighlightNoAnnotation }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — single bar (8,8) crop edge case"
						artifactId={SANDBOX_IDS.scoreHighlightSingleBar}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightSingleBar}
							component={{ type: "score_highlight", config: scoreHighlightSingleBar }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ScoreHighlight — out-of-range bars (300-310, piece ends at 264)"
						artifactId={SANDBOX_IDS.scoreHighlightOutOfRange}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightOutOfRange}
							component={{ type: "score_highlight", config: scoreHighlightOutOfRange }}
						/>
					</SandboxSection>

					{/* ── EXERCISES ─────────────────────────────────────────────── */}
					<h2 className="font-display text-display-sm text-text-tertiary tracking-wide uppercase text-xs">Exercises</h2>

					<SandboxSection
						title="ExerciseSet — with scoreClip + exerciseId"
						artifactId={SANDBOX_IDS.exerciseWithId}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseWithId}
							component={{ type: "exercise_set", config: exerciseSetWithId }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ExerciseSet — with scoreClip, no exerciseId"
						artifactId={SANDBOX_IDS.exerciseNoId}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseNoId}
							component={{ type: "exercise_set", config: exerciseSetNoId }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ExerciseSet — single exercise (with exerciseId)"
						artifactId={SANDBOX_IDS.exerciseSingle}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseSingle}
							component={{ type: "exercise_set", config: exerciseSingle }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ExerciseSet — 3 exercises with IDs (click 'Add to practice' → error state)"
						artifactId={SANDBOX_IDS.exerciseThreeWithIds}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseThreeWithIds}
							component={{ type: "exercise_set", config: exerciseThreeWithIds }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ExerciseSet — mixed (first: no button, second: has button)"
						artifactId={SANDBOX_IDS.exerciseMixed}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseMixed}
							component={{ type: "exercise_set", config: exerciseMixed }}
						/>
					</SandboxSection>

					<SandboxSection
						title="ExerciseSet — long text (title truncation + instruction wrapping)"
						artifactId={SANDBOX_IDS.exerciseLongText}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseLongText}
							component={{ type: "exercise_set", config: exerciseLongText }}
						/>
					</SandboxSection>

					{/* ── PLAYBACK ──────────────────────────────────────────────── */}
					<h2 className="font-display text-display-sm text-text-tertiary tracking-wide uppercase text-xs">Playback</h2>

					<SandboxSection
						title="PlayPassage — timing, focusBars [6,7]"
						artifactId={SANDBOX_IDS.ppTiming}
					>
						<PlayPassageCard
							config={ppTiming}
							_mockManifest={MOCK_MANIFEST}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — dynamics, no focusBars"
						artifactId={SANDBOX_IDS.ppDynamics}
					>
						<PlayPassageCard
							config={ppDynamics}
							_mockManifest={MOCK_MANIFEST}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — pedaling, focusBars [5,6]"
						artifactId={SANDBOX_IDS.ppPedaling}
					>
						<PlayPassageCard
							config={ppPedaling}
							_mockManifest={MOCK_MANIFEST}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — phrasing, wide passage bars 1-8, focusBars [5,8]"
						artifactId={SANDBOX_IDS.ppPhrasing}
					>
						<PlayPassageCard
							config={ppPhrasing}
							_mockManifest={MOCK_MANIFEST_MULTI_CHUNK}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — long annotation (overflow edge case)"
						artifactId={SANDBOX_IDS.ppLongAnnotation}
					>
						<PlayPassageCard
							config={ppLongAnnotation}
							_mockManifest={MOCK_MANIFEST}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — multi-chunk manifest (bars 1-8, 2 chunks)"
						artifactId={SANDBOX_IDS.ppMultiChunk}
					>
						<PlayPassageCard
							config={{ ...ppPhrasing, bars: [1, 8] }}
							_mockManifest={MOCK_MANIFEST_MULTI_CHUNK}
							_mockClip={MOCK_CLIP_SVG}
						/>
					</SandboxSection>

					<SandboxSection
						title="PlayPassage — fetch error state (session has no alignment)"
						artifactId={SANDBOX_IDS.ppFetchError}
					>
						<PlayPassageCard config={ppFetchError} />
					</SandboxSection>

					<PlayablePassageSection
						artifactId={SANDBOX_IDS.ppPlayable}
						title="PlayPassage — playable (real audio, click Play to hear)"
						config={ppTiming}
					/>

					<PlayablePassageSection
						artifactId={SANDBOX_IDS.ppAudioError}
						title="PlayPassage — audio_error (score renders, 'Audio unavailable')"
						config={ppDynamics}
						audioError
					/>

					<SandboxSection
						title="SegmentLoop — pending (Accept / Skip buttons; clicking shows API error)"
						artifactId={SANDBOX_IDS.slPending}
					>
						<SegmentLoopArtifactCard config={slPending} />
					</SandboxSection>

					<SandboxSection
						title="SegmentLoop — active (1/3 attempts; Dismiss → API error)"
						artifactId={SANDBOX_IDS.slActive}
					>
						<SegmentLoopArtifactCard config={slActive} />
					</SandboxSection>

					<SandboxSection
						title="SegmentLoop — completed (green badge, no buttons)"
						artifactId={SANDBOX_IDS.slCompleted}
					>
						<SegmentLoopArtifactCard config={slCompleted} />
					</SandboxSection>

					<SandboxSection
						title="SegmentLoop — dismissed (faded, no buttons)"
						artifactId={SANDBOX_IDS.slDismissed}
					>
						<SegmentLoopArtifactCard config={slDismissed} />
					</SandboxSection>

					<SandboxSection
						title="SegmentLoop — superseded (faded, no buttons)"
						artifactId={SANDBOX_IDS.slSuperseded}
					>
						<SegmentLoopArtifactCard config={slSuperseded} />
					</SandboxSection>

					{/* ── KEYBOARD ──────────────────────────────────────────────── */}
					<h2 className="font-display text-display-sm text-text-tertiary tracking-wide uppercase text-xs">Keyboard</h2>

					<SandboxSection
						title="KeyboardGuide (placeholder)"
						artifactId={SANDBOX_IDS.keyboardGuide}
					>
						<Artifact
							artifactId={SANDBOX_IDS.keyboardGuide}
							component={{ type: "keyboard_guide", config: keyboardGuide }}
						/>
					</SandboxSection>
				</div>
			</div>

			<ArtifactOverlay />
		</ArtifactScrollContext.Provider>
	);
}

export const Route = createFileRoute("/app/sandbox")({
	component: ArtifactSandbox,
});
