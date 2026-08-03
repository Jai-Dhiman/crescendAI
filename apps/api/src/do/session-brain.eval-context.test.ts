import { describe, expect, it } from "vitest";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";
import { buildEvalContext } from "./session-brain";

function artifact(): SynthesisArtifact {
	return {
		session_id: "sess_1",
		synthesis_scope: "session",
		strengths: [
			{ dimension: "dynamics", one_liner: "bars 3-6 shaped the phrase" },
		],
		focus_areas: [
			{
				dimension: "timing",
				one_liner: "rushed the left hand",
				severity: "moderate",
			},
		],
		prescribed_exercise: null,
		dominant_dimension: "timing",
		recurring_pattern: null,
		next_session_focus: "steady pulse in the left hand",
		diagnosis_refs: ["diag_1"],
		headline: "H".repeat(320),
		assigned_loops: [],
	};
}

const snapshot = { scored_chunks: [{ scores: [0.5] }], teaching_moments: [] };

describe("buildEvalContext", () => {
	it("attaches the full artifact so the judge sees more than the headline", () => {
		const ctx = buildEvalContext(snapshot, artifact());

		// The whole artifact must survive, not just the fields the DO happens to
		// need for delivery. #28: the judge grades the artifact, not the headline.
		expect(ctx["artifact"]).toEqual(artifact());
		const art = ctx["artifact"] as SynthesisArtifact;
		expect(art.focus_areas[0].one_liner).toBe("rushed the left hand");
		expect(art.next_session_focus).toBe("steady pulse in the left hand");
	});

	it("preserves the accumulator snapshot fields alongside the artifact", () => {
		const ctx = buildEvalContext(snapshot, artifact());

		expect(ctx["scored_chunks"]).toEqual([{ scores: [0.5] }]);
		expect(ctx["teaching_moments"]).toEqual([]);
	});

	it("keeps prescribed_exercise at the top level for existing routing consumers", () => {
		const withExercise = {
			...artifact(),
			prescribed_exercise: {
				kind: "own_passage_loop" as const,
				bar_range: [3, 6] as [number, number],
				target_dimension: "timing",
				tempo_factor: 0.8,
			},
		} as SynthesisArtifact;

		const ctx = buildEvalContext(snapshot, withExercise);

		expect(ctx["prescribed_exercise"]).toEqual(
			withExercise.prescribed_exercise,
		);
	});
});
