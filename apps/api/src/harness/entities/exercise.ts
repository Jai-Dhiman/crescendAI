import { z } from "zod";

export const ExerciseSchema = z.object({
	id: z.string().uuid(),
	title: z.string().min(1),
	description: z.string(),
	instructions: z.string(),
	difficulty: z.string(),
	category: z.string(),
	repertoireTags: z.unknown().nullable().optional(),
	notationContent: z.string().nullable().optional(),
	notationFormat: z.string().nullable().optional(),
	midiContent: z.string().nullable().optional(),
	source: z.string().min(1),
	variantsJson: z.unknown().nullable().optional(),
	createdAt: z.string(),
});

export type Exercise = z.infer<typeof ExerciseSchema>;

export function exerciseDedupKey(input: {
	title: string;
	source: string;
}): string {
	const t = input.title.trim().toLowerCase();
	const s = input.source.trim().toLowerCase();
	return `${t}|${s}`;
}
