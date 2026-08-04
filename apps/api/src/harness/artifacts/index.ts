import type { ZodTypeAny } from "zod";
import { DiagnosisArtifactSchema } from "./diagnosis";
import { SegmentLoopArtifactSchema } from "./segment-loop";
import { SynthesisArtifactSchema } from "./synthesis";

export { type DiagnosisArtifact, DiagnosisArtifactSchema } from "./diagnosis";
export {
	type ExerciseRoutingDecision,
	ExerciseRoutingDecisionSchema,
} from "./exercise-routing";
export {
	type SegmentLoopArtifact,
	SegmentLoopArtifactSchema,
	type SegmentLoopRef,
} from "./segment-loop";
export { type SynthesisArtifact, SynthesisArtifactSchema } from "./synthesis";

export const ARTIFACT_NAMES = [
	"DiagnosisArtifact",
	"SynthesisArtifact",
	"SegmentLoopArtifact",
] as const;
export type ArtifactName = (typeof ARTIFACT_NAMES)[number];

export const artifactSchemas: Record<ArtifactName, ZodTypeAny> = {
	DiagnosisArtifact: DiagnosisArtifactSchema,
	SynthesisArtifact: SynthesisArtifactSchema,
	SegmentLoopArtifact: SegmentLoopArtifactSchema,
};
