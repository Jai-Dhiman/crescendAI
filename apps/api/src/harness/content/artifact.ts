import { z } from "zod";

export const ArtifactRowSchema = z.object({
	artifact_id: z.string().uuid(),
	schema_name: z.string().min(1),
	schema_version: z.number().int().positive(),
	producer: z.string().min(1),
	created_at: z.string(),
	payload: z.unknown(),
});

export type ArtifactRow = z.infer<typeof ArtifactRowSchema>;
