import { createFileRoute } from "@tanstack/react-router";
import { useRef, useState } from "react";
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
