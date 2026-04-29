import type { ToolDefinition } from "../../loop/types";
import { DIMENSIONS } from "../../artifacts/diagnosis";

export type PastObservation = {
	artifact_id: string;
	session_id: string;
	days_ago: number;
	similarity_score: number;
};

function barOverlapFraction(
	a: [number, number],
	b: [number, number] | null,
): number {
	if (!b) return 0;
	const overlapStart = Math.max(a[0], b[0]);
	const overlapEnd = Math.min(a[1], b[1]);
	if (overlapEnd < overlapStart) return 0;
	return (
		(overlapEnd - overlapStart + 1) /
		(Math.max(a[1], b[1]) - Math.min(a[0], b[0]) + 1)
	);
}

export const fetchSimilarPastObservation: ToolDefinition = {
	name: "fetch-similar-past-observation",
	description:
		"Finds the most similar past DiagnosisArtifact for this student and dimension using similarity = 0.5 * piece_id_match + 0.5 * bar_range_overlap_fraction. Returns null when no candidate scores >= 0.5.",
	input_schema: {
		type: "object",
		properties: {
			dimension: { type: "string", enum: DIMENSIONS },
			piece_id: { type: "string" },
			bar_range: {
				type: "array",
				items: { type: "integer" },
				minItems: 2,
				maxItems: 2,
			},
			past_diagnoses: {
				type: "array",
				items: {
					type: "object",
					properties: {
						artifact_id: { type: "string" },
						session_id: { type: "string" },
						created_at: { type: "number" },
						primary_dimension: { type: "string" },
						piece_id: { type: ["string", "null"] },
						bar_range: { type: ["array", "null"] },
					},
					required: [
						"artifact_id",
						"session_id",
						"created_at",
						"primary_dimension",
					],
				},
				description:
					"Pre-materialized diagnosis records for this student from digest.",
			},
			now_ms: { type: "number" },
		},
		required: [
			"dimension",
			"piece_id",
			"bar_range",
			"past_diagnoses",
			"now_ms",
		],
	},
	invoke: async (input: unknown): Promise<PastObservation | null> => {
		const { dimension, piece_id, bar_range, past_diagnoses, now_ms } =
			input as {
				dimension: string;
				piece_id: string;
				bar_range: [number, number];
				past_diagnoses: {
					artifact_id: string;
					session_id: string;
					created_at: number;
					primary_dimension: string;
					piece_id?: string | null;
					bar_range?: [number, number] | null;
				}[];
				now_ms: number;
			};
		let best: PastObservation | null = null;
		for (const d of past_diagnoses) {
			if (d.primary_dimension !== dimension) continue;
			const pieceSim = d.piece_id === piece_id ? 0.5 : 0;
			const barSim = 0.5 * barOverlapFraction(bar_range, d.bar_range ?? null);
			const score = pieceSim + barSim;
			if (score >= 0.5 && (best === null || score > best.similarity_score)) {
				best = {
					artifact_id: d.artifact_id,
					session_id: d.session_id,
					days_ago: (now_ms - d.created_at) / 86_400_000,
					similarity_score: score,
				};
			}
		}
		return best;
	},
};
