import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Artifact } from "../components/Artifact";
import { ArtifactOverlay } from "../components/ArtifactOverlay";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";
import { API_BASE } from "../lib/config";
import { osmdManager } from "../lib/osmd-manager";
import type {
	ExerciseSetConfig,
	KeyboardGuideConfig,
	ReferenceBrowserConfig,
	ScoreHighlightConfig,
} from "../lib/types";
import { useArtifactStore } from "../stores/artifact";

// --- Score debug loader ---

type DebugPhase =
	| { status: "idle" }
	| { status: "fetching" }
	| { status: "fetch_ok"; bytes: number }
	| { status: "rendering" }
	| { status: "clipping" }
	| { status: "done"; svgWidth: number; svgHeight: number }
	| { status: "error"; phase: string; message: string };

function ScoreDebugLoader() {
	const [phase, setPhase] = useState<DebugPhase>({ status: "idle" });
	const svgRef = useRef<HTMLDivElement>(null);

	async function run() {
		const pieceId = "chopin.ballades.1";
		setPhase({ status: "fetching" });

		// Step 1: raw fetch — confirm bytes arrive and check content-type
		let bytes: ArrayBuffer;
		try {
			const res = await fetch(`${API_BASE}/api/scores/${pieceId}/data`, {
				credentials: "include",
			});
			const ct = res.headers.get("content-type") ?? "(none)";
			if (!res.ok) {
				setPhase({ status: "error", phase: "fetch", message: `HTTP ${res.status} — content-type: ${ct}` });
				return;
			}
			bytes = await res.arrayBuffer();
			setPhase({ status: "fetch_ok", bytes: bytes.byteLength });
			console.log("[score-debug] fetch ok", { bytes: bytes.byteLength, contentType: ct });
		} catch (err) {
			setPhase({ status: "error", phase: "fetch", message: String(err) });
			return;
		}

		// Step 2: hand to osmdManager
		setPhase({ status: "rendering" });
		try {
			await osmdManager.ensureRendered(pieceId);
			console.log("[score-debug] osmd render complete");
		} catch (err) {
			setPhase({ status: "error", phase: "osmd.render", message: String(err) });
			return;
		}

		// Step 3: clip bars 1-4
		setPhase({ status: "clipping" });
		const svg = osmdManager.clipBars(pieceId, 1, 4);
		if (!svg) {
			setPhase({ status: "error", phase: "clipBars", message: "clipBars returned null — measureList may be empty or bounding boxes missing" });
			return;
		}

		// Step 4: inject into DOM
		if (svgRef.current) {
			while (svgRef.current.firstChild) svgRef.current.removeChild(svgRef.current.firstChild);
			svgRef.current.appendChild(svg);
		}
		setPhase({ status: "done", svgWidth: svg.getBoundingClientRect().width, svgHeight: svg.getBoundingClientRect().height });
		console.log("[score-debug] done", svg);
	}

	// Auto-run on mount
	// biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount
	useEffect(() => { run(); }, []);

	return (
		<div className="border border-border rounded-xl bg-surface-card p-5 flex flex-col gap-4">
			<div className="flex items-center justify-between">
				<h2 className="font-display text-display-xs text-cream">Score Debug Loader</h2>
				<button
					type="button"
					onClick={() => { osmdManager.reset(); run(); }}
					className="px-3 py-1.5 rounded-lg bg-surface border border-border text-body-sm text-text-secondary hover:text-cream hover:border-accent transition"
				>
					Retry
				</button>
			</div>

			{/* Phase badge */}
			<div className="flex items-center gap-2">
				<span className={`text-label-sm px-2 py-0.5 rounded ${
					phase.status === "error" ? "bg-red-900/40 text-red-300" :
					phase.status === "done" ? "bg-green-900/40 text-green-300" :
					"bg-surface text-text-tertiary"
				}`}>
					{phase.status}
				</span>
				{phase.status === "fetch_ok" && (
					<span className="text-body-xs text-text-tertiary">{(phase.bytes / 1024).toFixed(0)} KB received</span>
				)}
				{phase.status === "done" && (
					<span className="text-body-xs text-text-tertiary">bars 1–4 rendered</span>
				)}
				{phase.status === "error" && (
					<span className="text-body-xs text-red-300">phase: {phase.phase} — {phase.message}</span>
				)}
			</div>

			{/* SVG output */}
			<div
				ref={svgRef}
				className="bg-white rounded-md min-h-[40px] [&>svg]:block [&>svg]:w-full"
			/>
		</div>
	);
}

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

const keyboardGuide: KeyboardGuideConfig = {};
const referenceBrowser: ReferenceBrowserConfig = {};

// --- Artifact IDs ---

const SANDBOX_IDS = {
	exerciseWithId: "sandbox-exercise-set-1",
	exerciseNoId: "sandbox-exercise-set-2",
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

					{/* Score Debug Loader — raw OSMD render, no artifact machinery */}
					<ScoreDebugLoader />

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
