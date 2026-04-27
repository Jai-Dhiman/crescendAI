import { z } from "zod";
import { ArtifactRowSchema } from "./artifact";
import { ObservationSchema } from "./observation";
import { SignalSchemaName, signalSchemas } from "./signal";

export * from "./signal";
export * from "./observation";
export * from "./artifact";

const signalRef = z.object({
	kind: z.literal("signal"),
	chunk_id: z.string().min(1),
	schema_name: z.enum([
		SignalSchemaName.MuQQuality,
		SignalSchemaName.AMTTranscription,
		SignalSchemaName.StopMoment,
		SignalSchemaName.ScoreAlignment,
	]),
	row_id: z.string().min(1),
});

const observationRef = z.object({
	kind: z.literal("observation"),
	observation_id: z.string().uuid(),
});

const artifactRef = z.object({
	kind: z.literal("artifact"),
	artifact_id: z.string().uuid(),
	schema_name: z.string().min(1),
});

export const evidenceRefSchema = z.discriminatedUnion("kind", [
	signalRef,
	observationRef,
	artifactRef,
]);

export type EvidenceRef = z.infer<typeof evidenceRefSchema>;

export const contentSchemas = {
	...signalSchemas,
	Observation: ObservationSchema,
	ArtifactRow: ArtifactRowSchema,
} as const;

export type ContentSchemaName = keyof typeof contentSchemas;
