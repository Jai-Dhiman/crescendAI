import { describe, expect, it, vi } from "vitest";
import { NotFoundError } from "../lib/errors";
import { assignPendingExercise } from "./exercises";

const STUDENT_ID = "student-abc";
const SESSION_ID = "00000000-0000-0000-0000-000000000010";
const EXERCISE_ID = "00000000-0000-0000-0000-000000000001";
const FOREIGN_EXERCISE_ID = "00000000-0000-0000-0000-000000000099";

const PENDING_ROW = {
	id: "00000000-0000-0000-0000-000000000020",
	studentId: STUDENT_ID,
	sessionId: SESSION_ID,
	exerciseId: EXERCISE_ID,
	focusDimension: "pedaling",
	previewTitle: "Pedal Separation Drill",
	consumed: false,
	createdAt: new Date(),
};

const EXERCISE_ROW = {
	id: EXERCISE_ID,
	title: "Pedal Separation Drill",
	description: "Running passage bars 3-6",
	instructions: "Play each phrase with clean pedal lifts.",
	difficulty: "intermediate",
	category: "technique",
	repertoireTags: null,
	notationContent: null,
	notationFormat: null,
	midiContent: null,
	source: "generated",
	variantsJson: null,
	createdAt: new Date(),
};

const DIMENSION_ROWS = [{ exerciseId: EXERCISE_ID, dimension: "pedaling" }];

function makeCtx({
	pendingRow,
	exerciseRow,
	dimensionRows,
}: {
	pendingRow: typeof PENDING_ROW | null;
	exerciseRow: typeof EXERCISE_ROW | null;
	dimensionRows: typeof DIMENSION_ROWS;
}) {
	const mockUpdate = vi.fn().mockReturnValue({
		set: vi.fn().mockReturnValue({
			where: vi.fn().mockResolvedValue(undefined),
		}),
	});

	const mockInsert = vi.fn().mockReturnValue({
		values: vi.fn().mockReturnValue({
			onConflictDoUpdate: vi.fn().mockReturnValue({
				returning: vi
					.fn()
					.mockResolvedValue([{ id: "se-1", studentId: STUDENT_ID }]),
			}),
		}),
	});

	let selectCallCount = 0;
	const mockSelect = vi.fn().mockImplementation(() => {
		selectCallCount++;
		const callIndex = selectCallCount;
		return {
			from: vi.fn().mockReturnValue({
				where: vi
					.fn()
					.mockResolvedValue(
						callIndex === 1 ? (pendingRow ? [pendingRow] : []) : dimensionRows,
					),
			}),
		};
	});

	const mockFindFirst = vi.fn().mockResolvedValue(exerciseRow);

	const db = {
		select: mockSelect,
		update: mockUpdate,
		insert: mockInsert,
		query: {
			exercises: { findFirst: mockFindFirst },
		},
	};

	return { db: db as never, env: {} as never };
}

describe("assignPendingExercise", () => {
	it("returns ExerciseSetPayload and marks pending row consumed for a valid owned row", async () => {
		const ctx = makeCtx({
			pendingRow: PENDING_ROW,
			exerciseRow: EXERCISE_ROW,
			dimensionRows: DIMENSION_ROWS,
		});
		const payload = await assignPendingExercise(ctx, {
			studentId: STUDENT_ID,
			sessionId: SESSION_ID,
			exerciseId: EXERCISE_ID,
		});
		expect(payload).toMatchObject({
			sourcePassage: expect.any(String),
			targetSkill: "pedaling",
			exercises: expect.arrayContaining([
				expect.objectContaining({
					title: EXERCISE_ROW.title,
					instruction: EXERCISE_ROW.instructions,
					focusDimension: "pedaling",
					exerciseId: EXERCISE_ID,
				}),
			]),
		});
		expect(ctx.db.update).toHaveBeenCalled();
	});

	it("throws NotFoundError when no unconsumed pending row matches (IDOR: foreign exerciseId)", async () => {
		const ctx = makeCtx({
			pendingRow: null,
			exerciseRow: EXERCISE_ROW,
			dimensionRows: [],
		});
		await expect(
			assignPendingExercise(ctx, {
				studentId: STUDENT_ID,
				sessionId: SESSION_ID,
				exerciseId: FOREIGN_EXERCISE_ID,
			}),
		).rejects.toThrow(NotFoundError);
	});
});
