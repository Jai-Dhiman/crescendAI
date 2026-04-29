import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Artifact } from "../components/Artifact";
import { ArtifactOverlay } from "../components/ArtifactOverlay";
import { SvgClip } from "../components/SvgClip";
import { SvgClipBBox } from "../components/SvgClipBBox";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";
import type { ClipResult } from "../lib/score-renderer";
import { scoreRenderer } from "../lib/score-renderer";
import type {
	ExerciseSetConfig,
	KeyboardGuideConfig,
	ReferenceBrowserConfig,
	ScoreHighlightConfig,
} from "../lib/types";
import { useArtifactStore } from "../stores/artifact";

// --- Fixtures ---

// Sandbox UUIDs: valid format so Zod passes, but not in DB — tests the error state path.
// In production, exerciseIds are server-assigned UUIDs from the create_exercise tool call.
const exerciseSetWithId: ExerciseSetConfig = {
	sourcePassage: "Chopin Op. 10 No. 3, bars 1-8",
	targetSkill: "Cantabile tone production",
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
const referenceBrowser: ReferenceBrowserConfig = {};

// --- Artifact IDs ---

const SANDBOX_IDS = {
	exerciseWithId: "sandbox-exercise-set-1",
	exerciseNoId: "sandbox-exercise-set-2",
	scoreHighlightOpening: "sandbox-score-highlight-opening",
	scoreHighlightWide: "sandbox-score-highlight-wide",
	scoreHighlightMid: "sandbox-score-highlight-mid",
	scoreHighlightLate: "sandbox-score-highlight-late",
	keyboardGuide: "sandbox-keyboard-guide-1",
	referenceBrowser: "sandbox-reference-browser-1",
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

// Reusable display component for pre-cropped SVGs (approaches C/D/E).
// No viewBox manipulation needed — Verovio already sized the output to the selection.
function SvgDisplay({ svgMarkup }: { svgMarkup: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM
		ref.current.insertAdjacentHTML("afterbegin", svgMarkup);
		const svgEl = ref.current.querySelector("svg") as SVGSVGElement | null;
		if (!svgEl) return;
		svgEl.setAttribute("width", "100%");
		svgEl.removeAttribute("height");
		svgEl.style.display = "block";
	}, [svgMarkup]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

interface ApproachRowProps {
	label: string;
	description: string;
	children: React.ReactNode;
}

function ApproachRow({ label, description, children }: ApproachRowProps) {
	return (
		<div className="flex flex-col gap-2">
			<div>
				<span className="text-label-sm text-accent font-mono">{label}</span>
				<span className="text-body-xs text-text-tertiary ml-2">
					{description}
				</span>
			</div>
			<div className="border border-border rounded-lg overflow-hidden bg-white min-h-[48px]">
				{children}
			</div>
		</div>
	);
}

// Approach A: current implementation — full page SVG + getBoundingClientRect crop
function ApproachA({
	pieceId,
	startBar,
	endBar,
}: {
	pieceId: string;
	startBar: number;
	endBar: number;
}) {
	const [clip, setClip] = useState<ClipResult | null>(null);
	const [error, setError] = useState<string | null>(null);
	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.getClip(pieceId, startBar, endBar)
			.then((r) => {
				if (!cancelled) setClip(r);
			})
			.catch((e) => {
				if (!cancelled) setError(String(e));
			});
		return () => {
			cancelled = true;
		};
	}, [pieceId, startBar, endBar]);
	if (error) return <p className="text-body-xs text-red-400 p-2">{error}</p>;
	if (!clip)
		return <p className="text-body-xs text-text-tertiary p-2">Loading…</p>;
	return (
		<SvgClip
			svgMarkup={clip.svg}
			startMeasureId={clip.startMeasureId}
			endMeasureId={clip.endMeasureId}
		/>
	);
}

// Approach B: same worker output, crop via getBBox() — browser-native SVG coordinates
function ApproachB({
	pieceId,
	startBar,
	endBar,
}: {
	pieceId: string;
	startBar: number;
	endBar: number;
}) {
	const [clip, setClip] = useState<ClipResult | null>(null);
	const [error, setError] = useState<string | null>(null);
	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.getClip(pieceId, startBar, endBar)
			.then((r) => {
				if (!cancelled) setClip(r);
			})
			.catch((e) => {
				if (!cancelled) setError(String(e));
			});
		return () => {
			cancelled = true;
		};
	}, [pieceId, startBar, endBar]);
	if (error) return <p className="text-body-xs text-red-400 p-2">{error}</p>;
	if (!clip)
		return <p className="text-body-xs text-text-tertiary p-2">Loading…</p>;
	return (
		<SvgClipBBox
			svgMarkup={clip.svg}
			startMeasureId={clip.startMeasureId}
			endMeasureId={clip.endMeasureId}
		/>
	);
}

// Approaches C/D/E: worker returns pre-cropped SVG, client just displays it
function ApproachWorkerMethod({
	pieceId,
	startBar,
	endBar,
	method,
}: {
	pieceId: string;
	startBar: number;
	endBar: number;
	method: "select" | "mei" | "mxl";
}) {
	const [svg, setSvg] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.getClipMethod(pieceId, startBar, endBar, method)
			.then((s) => {
				if (!cancelled) setSvg(s);
			})
			.catch((e) => {
				if (!cancelled) setError(String(e));
			});
		return () => {
			cancelled = true;
		};
	}, [pieceId, startBar, endBar, method]);
	if (error) return <p className="text-body-xs text-red-400 p-2">{error}</p>;
	if (!svg)
		return <p className="text-body-xs text-text-tertiary p-2">Loading…</p>;
	return <SvgDisplay svgMarkup={svg} />;
}

function ApproachesComparison({
	pieceId,
	startBar,
	endBar,
}: {
	pieceId: string;
	startBar: number;
	endBar: number;
}) {
	return (
		<section className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
			<div>
				<h2 className="font-display text-display-xs text-cream">
					Rendering Approaches — Bars {startBar}–{endBar}
				</h2>
				<p className="text-body-sm text-text-secondary mt-1">
					Same bars, five different approaches from Verovio to screen. Compare
					output quality, layout context, and whitespace.
				</p>
			</div>
			<div className="flex flex-col gap-5">
				<ApproachRow
					label="A — Current (getBoundingClientRect)"
					description="Full page SVG → client measures screen coords → scaleY → viewBox. Broken in Firefox; dead space in Chrome."
				>
					<ApproachA pieceId={pieceId} startBar={startBar} endBar={endBar} />
				</ApproachRow>

				<ApproachRow
					label="B — getBBox crop"
					description="Same full page SVG, crop using SVG-native viewBox coords via getBBox(). No screen measurements, no scaleY."
				>
					<ApproachB pieceId={pieceId} startBar={startBar} endBar={endBar} />
				</ApproachRow>

				<ApproachRow
					label="C — tk.select() (Verovio selection API)"
					description="Worker calls tk.select({start, end}) before renderToSVG(1). Verovio produces a self-contained SVG for only those bars."
				>
					<ApproachWorkerMethod
						pieceId={pieceId}
						startBar={startBar}
						endBar={endBar}
						method="select"
					/>
				</ApproachRow>

				<ApproachRow
					label="D — MEI round-trip (export → filter → reload)"
					description="Worker exports full MEI, removes measures outside range, reloads into fresh Verovio toolkit, renders page 1."
				>
					<ApproachWorkerMethod
						pieceId={pieceId}
						startBar={startBar}
						endBar={endBar}
						method="mei"
					/>
				</ApproachRow>

				<ApproachRow
					label="E — MusicXML filter (strip source → reload)"
					description="Worker parses original MusicXML, keeps only target measures with attributes carry-forward, reloads, renders page 1."
				>
					<ApproachWorkerMethod
						pieceId={pieceId}
						startBar={startBar}
						endBar={endBar}
						method="mxl"
					/>
				</ApproachRow>
			</div>
		</section>
	);
}

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

	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.getFull(pieceId)
			.then((s) => {
				if (!cancelled) setSvg(s);
			})
			.catch((e) => {
				if (!cancelled) setError(String(e));
			});
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
			scoreRenderer
				.getFull(pieceId, Math.round(w / 0.4))
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
	const [clip, setClip] = useState<ClipResult | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.getClip(pieceId, startBar, endBar)
			.then((r) => {
				if (!cancelled) setClip(r);
			})
			.catch((e) => {
				if (!cancelled) setError(String(e));
			});
		return () => {
			cancelled = true;
		};
	}, [pieceId, startBar, endBar]);

	return (
		<div className="flex flex-col gap-1">
			<span className="text-label-sm text-accent font-mono">{label}</span>
			<div className="border border-border rounded-lg overflow-hidden bg-white">
				{error && <p className="text-body-xs text-red-400 p-2">{error}</p>}
				{!clip && !error && (
					<p className="text-body-xs text-text-tertiary p-2">Loading…</p>
				)}
				{clip && (
					<SvgClip
						svgMarkup={clip.svg}
						startMeasureId={clip.startMeasureId}
						endMeasureId={clip.endMeasureId}
					/>
				)}
			</div>
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

					{/* Approaches comparison — 5 ways to render bars 135-136 */}
					<ApproachesComparison
						pieceId="chopin.ballades.1"
						startBar={135}
						endBar={136}
					/>

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

					{/* ExerciseSet — with exerciseId (Start/Complete buttons) */}
					<SandboxSection
						title="ExerciseSet (with exerciseId)"
						artifactId={SANDBOX_IDS.exerciseWithId}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseWithId}
							component={{ type: "exercise_set", config: exerciseSetWithId }}
						/>
					</SandboxSection>

					{/* ExerciseSet — without exerciseId (Not yet saved path) */}
					<SandboxSection
						title="ExerciseSet (no exerciseId)"
						artifactId={SANDBOX_IDS.exerciseNoId}
					>
						<Artifact
							artifactId={SANDBOX_IDS.exerciseNoId}
							component={{ type: "exercise_set", config: exerciseSetNoId }}
						/>
					</SandboxSection>

					{/* KeyboardGuide */}
					<SandboxSection
						title="KeyboardGuide (placeholder)"
						artifactId={SANDBOX_IDS.keyboardGuide}
					>
						<Artifact
							artifactId={SANDBOX_IDS.keyboardGuide}
							component={{ type: "keyboard_guide", config: keyboardGuide }}
						/>
					</SandboxSection>

					{/* ReferenceBrowser */}
					<SandboxSection
						title="ReferenceBrowser (placeholder)"
						artifactId={SANDBOX_IDS.referenceBrowser}
					>
						<Artifact
							artifactId={SANDBOX_IDS.referenceBrowser}
							component={{
								type: "reference_browser",
								config: referenceBrowser,
							}}
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
