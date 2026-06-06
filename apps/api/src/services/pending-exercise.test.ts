import { describe, expect, it, vi } from "vitest";
import { stageDominantExercise } from "./pending-exercise";

describe("stageDominantExercise", () => {
	function makeMockDb(exerciseId = "ex-uuid-1") {
		const insertedValues: unknown[] = [];
		let callCount = 0;
		const mockDb = {
			insert: vi.fn().mockImplementation(() => {
				callCount++;
				const call = callCount;
				return {
					values: vi.fn().mockImplementation((v: unknown) => {
						insertedValues.push({ call, values: v });
						if (call === 1) {
							return { returning: vi.fn().mockResolvedValue([{ id: exerciseId }]) };
						}
						return Promise.resolve(undefined);
					}),
				};
			}),
		};
		return { mockDb, insertedValues };
	}

	it("inserts exercises, exerciseDimensions, pendingExercises and returns ref", async () => {
		const { mockDb, insertedValues } = makeMockDb("ex-uuid-1");
		const result = await stageDominantExercise(mockDb as never, {
			studentId: "stu-1", sessionId: "sess-1", dominantDimension: "pedaling",
			proposedExercise: "Practice slow legato pedaling in bars 1-4.",
			pieceMetadata: { title: "Nocturne", composer: "Chopin" },
		});
		expect(mockDb.insert).toHaveBeenCalledTimes(3);
		const ex = insertedValues[0] as { values: Record<string, unknown> };
		expect(ex.values).toMatchObject({
			description: "Staged from session synthesis",
			instructions: "Practice slow legato pedaling in bars 1-4.",
			difficulty: "intermediate", category: "generated", source: "teacher_llm",
		});
		expect((ex.values.title as string).length).toBeGreaterThan(0);
		const dim = insertedValues[1] as { values: Record<string, unknown> };
		expect(dim.values).toMatchObject({ exerciseId: "ex-uuid-1", dimension: "pedaling" });
		const pend = insertedValues[2] as { values: Record<string, unknown> };
		expect(pend.values).toMatchObject({
			studentId: "stu-1", sessionId: "sess-1", exerciseId: "ex-uuid-1",
			focusDimension: "pedaling", consumed: false,
		});
		expect(typeof pend.values.previewTitle).toBe("string");
		expect(result).toEqual({ exerciseId: "ex-uuid-1", focusDimension: "pedaling", previewTitle: expect.any(String) });
	});

	it("derives previewTitle from first 60 chars of proposedExercise", async () => {
		const { mockDb } = makeMockDb();
		const result = await stageDominantExercise(mockDb as never, {
			studentId: "s", sessionId: "sess", dominantDimension: "dynamics",
			proposedExercise: "A".repeat(100), pieceMetadata: null,
		});
		expect(result.previewTitle).toBe("A".repeat(60));
	});

	it("falls back to dimension focus drill when proposedExercise is blank", async () => {
		const { mockDb } = makeMockDb();
		const result = await stageDominantExercise(mockDb as never, {
			studentId: "s", sessionId: "sess", dominantDimension: "timing",
			proposedExercise: "   ", pieceMetadata: null,
		});
		expect(result.previewTitle).toBe("timing focus drill");
	});

	it("throws InferenceError when exercises insert returns empty array", async () => {
		const mockDb = { insert: vi.fn().mockReturnValue({ values: vi.fn().mockReturnValue({ returning: vi.fn().mockResolvedValue([]) }) }) };
		await expect(stageDominantExercise(mockDb as never, {
			studentId: "s", sessionId: "sess", dominantDimension: "pedaling",
			proposedExercise: "drill", pieceMetadata: null,
		})).rejects.toThrow("Failed to insert staged exercise");
	});
});
