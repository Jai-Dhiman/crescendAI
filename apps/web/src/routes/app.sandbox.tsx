import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Artifact } from "../components/Artifact";
import { ArtifactOverlay } from "../components/ArtifactOverlay";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";
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

const scoreHighlight: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{ bars: [1, 4], dimension: "dynamics", annotation: "Establish hushed, questioning p throughout" },
		{ bars: [5, 8], dimension: "phrasing", annotation: "Lift at the phrase peak on beat 3" },
	],
};

// Tests full score rendering — wide bar range to exercise scroll/pagination and measure layout at scale.
const scoreHighlightFull: ScoreHighlightConfig = {
	pieceId: "chopin.ballades.1",
	highlights: [
		{ bars: [1, 8], dimension: "dynamics", annotation: "Opening statement — hushed and questioning" },
		{ bars: [9, 16], dimension: "phrasing", annotation: "Secondary theme entry — shape the long arc" },
		{ bars: [17, 24], dimension: "timing", annotation: "Subtle rubato — breathe at bar 20" },
	],
};

const keyboardGuide: KeyboardGuideConfig = {};
const referenceBrowser: ReferenceBrowserConfig = {};

// --- Artifact IDs ---

const SANDBOX_IDS = {
	exerciseWithId: "sandbox-exercise-set-1",
	exerciseNoId: "sandbox-exercise-set-2",
	scoreHighlightFull: "sandbox-score-highlight-full",
	scoreHighlight: "sandbox-score-highlight-1",
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

// --- Resize behavior sandbox ---

const PLACEHOLDER_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="120" viewBox="0 0 600 120">
  <rect width="600" height="120" fill="#fff"/>
  <line x1="0" y1="60" x2="600" y2="60" stroke="#333" stroke-width="1"/>
  <text x="10" y="20" font-size="11" fill="#666">Placeholder score — resize panel to compare variants</text>
  <rect x="20" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="80" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="140" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
  <rect x="200" y="40" width="40" height="40" rx="2" fill="none" stroke="#444" stroke-width="1"/>
</svg>`;

function SvgPanel({ svgMarkup }: { svgMarkup: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		ref.current.insertAdjacentHTML("afterbegin", svgMarkup);
	}, [svgMarkup]);
	return (
		<div
			ref={ref}
			// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio, not user input
			className="[&>svg]:w-full [&>svg]:block"
		/>
	);
}

function ResizeVariantPanel({
	label,
	description,
	svgMarkup,
	onResize,
}: {
	label: string;
	description: string;
	svgMarkup: string;
	onResize: (newWidth: number) => void;
}) {
	const [width, setWidth] = useState(300);
	const panelRef = useRef<HTMLDivElement>(null);

	function handleDragStart(e: React.MouseEvent) {
		e.preventDefault();
		const startX = e.clientX;
		const startWidth = width;

		function onMove(ev: MouseEvent) {
			const newWidth = Math.max(160, Math.min(520, startWidth + (ev.clientX - startX)));
			if (panelRef.current) panelRef.current.style.width = `${newWidth}px`;
		}

		function onUp(ev: MouseEvent) {
			const newWidth = Math.max(160, Math.min(520, startWidth + (ev.clientX - startX)));
			setWidth(newWidth);
			onResize(newWidth);
			document.removeEventListener("mousemove", onMove);
			document.removeEventListener("mouseup", onUp);
			document.body.style.cursor = "";
		}

		document.addEventListener("mousemove", onMove);
		document.addEventListener("mouseup", onUp);
		document.body.style.cursor = "col-resize";
	}

	return (
		<div className="flex flex-col gap-2">
			<div className="flex items-center gap-2">
				<span className="text-label-sm text-accent font-mono">{label}</span>
				<span className="text-body-xs text-text-tertiary">{description}</span>
			</div>
			<div
				ref={panelRef}
				className="relative border border-border rounded-lg overflow-hidden bg-white"
				style={{ width }}
			>
				<SvgPanel svgMarkup={svgMarkup} />
				<div
					onMouseDown={handleDragStart}
					className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize bg-border hover:bg-accent transition-colors"
				/>
			</div>
			<span className="text-body-xs text-text-tertiary">Width: {width}px</span>
		</div>
	);
}

function ResizeSandbox() {
	const [variantASvg, setVariantASvg] = useState(PLACEHOLDER_SVG);

	const [variantBSvg, setVariantBSvg] = useState(PLACEHOLDER_SVG);
	const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	function handleVariantAResize(newWidth: number) {
		const updatedSvg = PLACEHOLDER_SVG.replace('width="600"', `width="${newWidth * 2}"`);
		setVariantASvg(updatedSvg);
	}

	function handleVariantBResize(newWidth: number) {
		if (debounceRef.current) clearTimeout(debounceRef.current);
		debounceRef.current = setTimeout(() => {
			const updatedSvg = PLACEHOLDER_SVG.replace('width="600"', `width="${newWidth * 2}"`);
			setVariantBSvg(updatedSvg);
		}, 200);
	}

	return (
		<div className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-6">
			<div>
				<h2 className="font-display text-display-xs text-cream">Panel Resize Behavior</h2>
				<p className="text-body-sm text-text-secondary mt-1">
					Drag the right edge of each panel. Pick the behavior that feels right.
				</p>
			</div>
			<div className="flex flex-col gap-6">
				<ResizeVariantPanel
					label="A: Reflow on drag-end"
					description="Score re-renders once after you release the handle"
					svgMarkup={variantASvg}
					onResize={handleVariantAResize}
				/>
				<ResizeVariantPanel
					label="B: Debounced 200ms"
					description="Score re-renders 200ms after drag stops moving"
					svgMarkup={variantBSvg}
					onResize={handleVariantBResize}
				/>
				<ResizeVariantPanel
					label="C: Fixed-width CSS scale"
					description="Score rendered once at fixed width; container scales via CSS"
					svgMarkup={PLACEHOLDER_SVG}
					onResize={() => {}}
				/>
			</div>
		</div>
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

					{/* Resize behavior sandbox — pick A, B, or C before Task 8 executes */}
					<ResizeSandbox />

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

					{/* ScoreHighlight — Full Score: tests rendering across a large bar range (bars 1–24) */}
					<SandboxSection
						title="ScoreHighlight — Full Score"
						artifactId={SANDBOX_IDS.scoreHighlightFull}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlightFull}
							component={{ type: "score_highlight", config: scoreHighlightFull }}
						/>
					</SandboxSection>

					{/* ScoreHighlight */}
					<SandboxSection
						title="ScoreHighlight"
						artifactId={SANDBOX_IDS.scoreHighlight}
					>
						<Artifact
							artifactId={SANDBOX_IDS.scoreHighlight}
							component={{ type: "score_highlight", config: scoreHighlight }}
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
