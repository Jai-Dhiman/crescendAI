import { studentProfiles } from "../db/schema/students";
import { ValidationError } from "../lib/errors";
import type { Db } from "../lib/types";
import { type BaselineState, BaselineStateSchema } from "./student-baseline";

export async function loadBaselineState(
	db: Db,
	studentId: string,
): Promise<BaselineState | null> {
	const row = await db.query.studentProfiles.findFirst({
		where: (s, { eq: eqFn }) => eqFn(s.studentId, studentId),
	});
	if (!row || row.baselineState === null || row.baselineState === undefined) {
		return null;
	}
	const parsed = BaselineStateSchema.safeParse(row.baselineState);
	if (!parsed.success) {
		throw new ValidationError(
			`loadBaselineState: corrupt baseline_state for student ${studentId}: ${parsed.error.message}`,
		);
	}
	return parsed.data;
}

export async function saveBaselineState(
	db: Db,
	studentId: string,
	state: BaselineState,
): Promise<void> {
	await db
		.insert(studentProfiles)
		.values({ studentId, baselineState: state })
		.onConflictDoUpdate({
			target: studentProfiles.studentId,
			set: { baselineState: state, updatedAt: new Date() },
		});
}
