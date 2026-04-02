import { eq } from "drizzle-orm";
import { sessions } from "../db/schema/sessions";
import { studentProfiles } from "../db/schema/students";
import type { ServiceContext } from "../lib/types";

interface SyncRequest {
	student: {
		inferredLevel?: string;
		baselineDynamics?: number;
		baselineTiming?: number;
		baselinePedaling?: number;
		baselineArticulation?: number;
		baselinePhrasing?: number;
		baselineInterpretation?: number;
		baselineSessionCount?: number;
	};
	newSessions: Array<{
		id: string;
		startedAt: string;
		endedAt?: string;
		avgDynamics?: number;
		avgTiming?: number;
		avgPedaling?: number;
		avgArticulation?: number;
		avgPhrasing?: number;
		avgInterpretation?: number;
		observationsJson?: unknown;
		chunksSummaryJson?: unknown;
	}>;
	lastSyncTimestamp?: string;
}

export async function handleSync(
	ctx: ServiceContext,
	studentId: string,
	data: SyncRequest,
) {
	const studentUpdate: Record<string, unknown> = { updatedAt: new Date() };

	if (data.student.inferredLevel !== undefined) {
		studentUpdate.inferredLevel = data.student.inferredLevel;
	}
	if (data.student.baselineDynamics !== undefined) {
		studentUpdate.baselineDynamics = data.student.baselineDynamics;
	}
	if (data.student.baselineTiming !== undefined) {
		studentUpdate.baselineTiming = data.student.baselineTiming;
	}
	if (data.student.baselinePedaling !== undefined) {
		studentUpdate.baselinePedaling = data.student.baselinePedaling;
	}
	if (data.student.baselineArticulation !== undefined) {
		studentUpdate.baselineArticulation = data.student.baselineArticulation;
	}
	if (data.student.baselinePhrasing !== undefined) {
		studentUpdate.baselinePhrasing = data.student.baselinePhrasing;
	}
	if (data.student.baselineInterpretation !== undefined) {
		studentUpdate.baselineInterpretation = data.student.baselineInterpretation;
	}
	if (data.student.baselineSessionCount !== undefined) {
		studentUpdate.baselineSessionCount = data.student.baselineSessionCount;
	}

	await ctx.db
		.update(studentProfiles)
		.set(studentUpdate)
		.where(eq(studentProfiles.studentId, studentId));

	for (const session of data.newSessions) {
		await ctx.db
			.insert(sessions)
			.values({
				id: session.id,
				studentId,
				startedAt: new Date(session.startedAt),
				endedAt:
					session.endedAt !== undefined ? new Date(session.endedAt) : null,
				avgDynamics: session.avgDynamics ?? null,
				avgTiming: session.avgTiming ?? null,
				avgPedaling: session.avgPedaling ?? null,
				avgArticulation: session.avgArticulation ?? null,
				avgPhrasing: session.avgPhrasing ?? null,
				avgInterpretation: session.avgInterpretation ?? null,
				observationsJson: session.observationsJson ?? null,
				chunksSummaryJson: session.chunksSummaryJson ?? null,
			})
			.onConflictDoNothing({ target: sessions.id });
	}

	return {
		syncTimestamp: new Date().toISOString(),
		exerciseUpdates: [],
	};
}
