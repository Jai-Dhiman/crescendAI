import { z } from "zod";

export * from "./student";
export * from "./piece";
export * from "./session";
export * from "./exercise";

const studentRef = z.object({
	kind: z.literal("student"),
	studentId: z.string().min(1),
});

const pieceRef = z.object({
	kind: z.literal("piece"),
	pieceId: z.string().min(1),
});

const movementRef = z.object({
	kind: z.literal("movement"),
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
});

const barRef = z.object({
	kind: z.literal("bar"),
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
	barNumber: z.number().int().nonnegative(),
});

const sessionRef = z.object({
	kind: z.literal("session"),
	sessionId: z.string().uuid(),
});

const exerciseRef = z.object({
	kind: z.literal("exercise"),
	exerciseId: z.string().uuid(),
});

export const entityRefSchema = z.discriminatedUnion("kind", [
	studentRef,
	pieceRef,
	movementRef,
	barRef,
	sessionRef,
	exerciseRef,
]);

export type EntityRef = z.infer<typeof entityRefSchema>;
export type EntityKind = EntityRef["kind"];

export const entityRefSchemas = {
	student: studentRef,
	piece: pieceRef,
	movement: movementRef,
	bar: barRef,
	session: sessionRef,
	exercise: exerciseRef,
} as const;
