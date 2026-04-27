import { z } from "zod";

export const StudentSchema = z.object({
	studentId: z.string().min(1),
	inferredLevel: z.string().nullable().optional(),
	baselineDynamics: z.number().nullable().optional(),
	baselineTiming: z.number().nullable().optional(),
	baselinePedaling: z.number().nullable().optional(),
	baselineArticulation: z.number().nullable().optional(),
	baselinePhrasing: z.number().nullable().optional(),
	baselineInterpretation: z.number().nullable().optional(),
	baselineSessionCount: z.number().int().nonnegative(),
	explicitGoals: z.string().nullable().optional(),
	createdAt: z.string().datetime(),
	updatedAt: z.string().datetime(),
});

export type Student = z.infer<typeof StudentSchema>;

export function resolveStudent(input: { appleUserId: string }): {
	studentId: string;
} {
	return { studentId: input.appleUserId };
}
