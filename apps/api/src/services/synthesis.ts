import { eq, sql } from "drizzle-orm";
import { messages } from "../db/schema/conversations";
import { observations } from "../db/schema/observations";
import { sessions } from "../db/schema/sessions";
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";
import type { Db } from "../lib/types";
import type { SessionAccumulator } from "./accumulator";
import type { InlineComponent } from "./tool-processor";

// SCALER_MEAN defaults from the Rust stop classifier (same values)
const SCALER_MEAN: Record<Dimension, number> = {
	dynamics: 0.545,
	timing: 0.4848,
	pedaling: 0.4594,
	articulation: 0.5369,
	phrasing: 0.5188,
	interpretation: 0.5064,
};

export async function persistSynthesisMessage(
	db: Db,
	conversationId: string,
	text: string,
	sessionId: string,
	componentsJson?: InlineComponent[],
): Promise<void> {
	await db.insert(messages).values({
		conversationId,
		role: "assistant",
		content: text,
		messageType: "synthesis",
		sessionId,
		componentsJson:
			componentsJson && componentsJson.length > 0 ? componentsJson : undefined,
	});
}

export async function persistAccumulatedMoments(
	db: Db,
	studentId: string,
	sessionId: string,
	conversationId: string | null,
	moments: SessionAccumulator["teachingMoments"],
): Promise<void> {
	for (const moment of moments) {
		const framing = moment.isPositive ? "recognition" : "correction";
		const reasoningTrace = moment.llmAnalysis ?? moment.reasoning;

		await db
			.insert(observations)
			.values({
				studentId,
				sessionId,
				chunkIndex: moment.chunkIndex,
				dimension: moment.dimension,
				observationText: moment.reasoning,
				reasoningTrace,
				framing,
				dimensionScore: moment.score,
				studentBaseline: moment.baseline,
				isFallback: false,
				conversationId,
			})
			.onConflictDoNothing();
	}
}

export async function clearNeedsSynthesis(
	db: Db,
	sessionId: string,
): Promise<void> {
	await db
		.update(sessions)
		.set({ needsSynthesis: false })
		.where(eq(sessions.id, sessionId));
}

export async function loadBaselinesFromDb(
	db: Db,
	studentId: string,
): Promise<Record<Dimension, number> | null> {
	const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

	const rows = await db
		.select({
			dimension: observations.dimension,
			avgScore: sql<number>`AVG(${observations.dimensionScore})`,
		})
		.from(observations)
		.where(
			sql`${observations.studentId} = ${studentId} AND ${observations.createdAt} > ${thirtyDaysAgo}`,
		)
		.groupBy(observations.dimension);

	if (rows.length === 0) {
		return null;
	}

	const dimMap: Partial<Record<Dimension, number>> = {};
	for (const row of rows) {
		const dim = row.dimension as Dimension;
		if (DIMS_6.includes(dim) && row.avgScore !== null) {
			dimMap[dim] = row.avgScore;
		}
	}

	const result = {} as Record<Dimension, number>;
	for (const dim of DIMS_6) {
		result[dim] = dimMap[dim] ?? SCALER_MEAN[dim];
	}

	return result;
}
