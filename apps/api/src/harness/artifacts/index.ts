import type { ZodTypeAny } from "zod";
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
import { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
import { SegmentLoopArtifactSchema, type SegmentLoopArtifact } from "./segment-loop";

export { DiagnosisArtifactSchema, type DiagnosisArtifact } from "./diagnosis";
export { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";
export { SegmentLoopArtifactSchema, type SegmentLoopArtifact, type SegmentLoopRef } from "./segment-loop";
export { ExerciseRoutingDecisionSchema, type ExerciseRoutingDecision } from "./exercise-routing";

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
