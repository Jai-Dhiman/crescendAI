import { describe, expect, it, test, vi } from "vitest";
import { NotFoundError } from "../lib/errors";
import { ExerciseRoutingDecisionSchema } from "../harness/artifacts/exercise-routing";
import { assignPendingExercise } from "./exercises";

const STUDENT_ID = "student-abc";
const SESSION_ID = "00000000-0000-0000-0000-000000000010";
const EXERCISE_ID = "00000000-0000-0000-0000-000000000001";
const FOREIGN_EXERCISE_ID = "00000000-0000-0000-0000-000000000099";

const PENDING_ROW = {
	id: "00000000-0000-0000-0000-000000000020",
	studentId: STUDENT_ID,
	sessionId: SESSION_ID,
	focusDimension: "pedaling",
	previewTitle: "Pedal Separation Drill",
	title: "Pedal Separation Drill",
	instruction: "Play each phrase with clean pedal lifts.",
	routingJson: {
		kind: "own_passage_loop",
		target_dimension: "pedaling",
		bar_range: [3, 6],
		tempo_factor: 0.8,
	},
	pieceId: null,
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

const PENDING_ROW_ID = "00000000-0000-0000-0000-000000000020";

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
			exerciseId: PENDING_ROW_ID,
		});
		expect(payload).toMatchObject({
			sourcePassage: expect.any(String),
			targetSkill: "pedaling",
			exercises: expect.arrayContaining([
				expect.objectContaining({
					title: PENDING_ROW.title,
					instruction: PENDING_ROW.instruction,
					focusDimension: "pedaling",
					exerciseId: PENDING_ROW_ID,
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

describe("assignPendingExercise — routing_json path", () => {
	const OWN_PASSAGE_ROUTING = ExerciseRoutingDecisionSchema.parse({
		kind: "own_passage_loop",
		target_dimension: "pedaling",
		bar_range: [12, 16],
		tempo_factor: 0.75,
	});

	const CORPUS_DRILL_ROUTING = ExerciseRoutingDecisionSchema.parse({
		kind: "corpus_drill",
		target_dimension: "timing",
		bar_range: [1, 8],
		tempo_factor: 0.8,
		primitive_id: null,
	});

	test("own_passage_loop with pieceId in row produces scoreClip in ExerciseSetPayload", async () => {
		const mockCtx = buildMockCtxWithPendingRow({
			routingJson: OWN_PASSAGE_ROUTING,
			focusDimension: "pedaling",
			previewTitle: "Pedaling drill",
			title: "Own passage loop",
			instruction: "Loop bars 12-16",
			pieceId: "chopin.ballade.1",
		});
		const payload = await assignPendingExercise(mockCtx, {
			studentId: "stu-1",
			sessionId: "sess-1",
			exerciseId: "pending-row-id",
		});
		expect(payload.scoreClip).toEqual({
			pieceId: "chopin.ballade.1",
			bars: [12, 16],
		});
		expect(payload.exercises[0].focusDimension).toBe("pedaling");
	});

	test("own_passage_loop with null pieceId in row produces no scoreClip (text-only)", async () => {
		const mockCtx = buildMockCtxWithPendingRow({
			routingJson: OWN_PASSAGE_ROUTING,
			focusDimension: "pedaling",
			previewTitle: "Pedaling drill",
			title: "Own passage loop",
			instruction: "Loop bars 12-16",
			pieceId: null,
		});
		const payload = await assignPendingExercise(mockCtx, {
			studentId: "stu-1",
			sessionId: "sess-1",
			exerciseId: "pending-row-id",
		});
		expect(payload.scoreClip).toBeUndefined();
		expect(payload.exercises[0].instruction).toContain("Loop bars 12-16");
	});

	test("own_passage_loop with null instruction AND null pieceId produces no scoreClip and falls back to bar-range string", async () => {
		const mockCtx = buildMockCtxWithPendingRow({
			routingJson: OWN_PASSAGE_ROUTING,
			focusDimension: "pedaling",
			previewTitle: "Pedaling drill",
			title: "Own passage loop",
			instruction: null,
			pieceId: null,
		});
		const payload = await assignPendingExercise(mockCtx, {
			studentId: "stu-1",
			sessionId: "sess-1",
			exerciseId: "pending-row-id",
		});
		expect(payload.scoreClip).toBeUndefined();
		expect(payload.exercises[0].instruction).toContain("12");
		expect(payload.exercises[0].instruction).toContain("16");
	});

	test("corpus_drill produces text stub, no scoreClip", async () => {
		const mockCtx = buildMockCtxWithPendingRow({
			routingJson: CORPUS_DRILL_ROUTING,
			focusDimension: "timing",
			previewTitle: "Timing drill",
			title: "Timing corpus drill",
			instruction: "Timing drill — bars 1-8",
			pieceId: "chopin.ballade.1",
		});
		const payload = await assignPendingExercise(mockCtx, {
			studentId: "stu-1",
			sessionId: "sess-1",
			exerciseId: "pending-row-id",
		});
		expect(payload.scoreClip).toBeUndefined();
		expect(payload.exercises[0].instruction).toContain("coming soon");
	});
});

function buildMockCtxWithPendingRow(row: {
	routingJson: unknown;
	focusDimension: string;
	previewTitle: string;
	title: string | null;
	instruction: string | null;
	pieceId: string | null;
}) {
	const pendingRow = {
		id: "pending-row-id",
		studentId: "stu-1",
		sessionId: "sess-1",
		focusDimension: row.focusDimension,
		previewTitle: row.previewTitle,
		title: row.title,
		instruction: row.instruction,
		routingJson: row.routingJson,
		pieceId: row.pieceId,
		consumed: false,
		createdAt: new Date(),
	};

	const mockDb = {
		select: () => ({
			from: () => ({
				where: () => Promise.resolve([pendingRow]),
			}),
		}),
		update: () => ({
			set: () => ({
				where: () => Promise.resolve([]),
			}),
		}),
		insert: () => ({
			values: () => ({
				onConflictDoUpdate: () => ({
					returning: () => Promise.resolve([{ id: "se-id" }]),
				}),
			}),
		}),
		query: {
			exercises: {
				findFirst: () => Promise.resolve(null),
			},
		},
	};

	return {
		db: mockDb as unknown as import("../lib/types").ServiceContext["db"],
		env: {},
	} as unknown as import("../lib/types").ServiceContext;
}
