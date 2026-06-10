// apps/api/src/services/pending-exercise.test.ts
import { describe, test, expect, vi } from "vitest";
import {
  stageDominantExercise,
  buildPendingExerciseComponent,
} from "./pending-exercise";
import { pendingExercises } from "../db/schema/exercises";
import type { ExerciseRoutingDecision } from "../harness/artifacts/exercise-routing";

const ROUTING: ExerciseRoutingDecision = {
  kind: "own_passage_loop",
  target_dimension: "pedaling",
  bar_range: [12, 16],
  tempo_factor: 0.75,
};

describe("stageDominantExercise", () => {
  test("inserts exactly once into pending_exercises with routing_json, title, instruction, piece_id; never inserts into exercises", async () => {
    const insertSpy = vi.fn();
    let capturedTable: unknown = null;
    let capturedRow: unknown = null;

    const db = {
      insert: (table: unknown) => {
        capturedTable = table;
        insertSpy(table);
        return {
          values: (row: unknown) => {
            capturedRow = row;
            return {
              returning: () => Promise.resolve([{ id: "pending-row-id-1" }]),
            };
          },
        };
      },
    };

    const result = await stageDominantExercise(db as never, {
      studentId: "stu-1",
      sessionId: "sess-1",
      dominantDimension: "pedaling",
      routing: ROUTING,
      pieceCtx: { pieceId: "chopin.ballade.1" },
    });

    // insert called exactly once — never touches exercises or exerciseDimensions
    expect(insertSpy).toHaveBeenCalledTimes(1);
    expect(capturedTable).toBe(pendingExercises);

    // persisted fields
    const row = capturedRow as Record<string, unknown>;
    expect(row.routingJson).toEqual(ROUTING);
    expect(typeof row.title).toBe("string");
    expect((row.title as string).length).toBeGreaterThan(0);
    expect(typeof row.instruction).toBe("string");
    expect(row.pieceId).toBe("chopin.ballade.1");

    // return value uses the pending row id as exerciseId
    expect(result.exerciseId).toBe("pending-row-id-1");
    expect(result.focusDimension).toBe("pedaling");
  });

  test("pieceId is null when pieceCtx is null", async () => {
    let capturedRow: unknown = null;
    const db = {
      insert: (_table: unknown) => ({
        values: (row: unknown) => {
          capturedRow = row;
          return { returning: () => Promise.resolve([{ id: "pending-row-id-2" }]) };
        },
      }),
    };

    await stageDominantExercise(db as never, {
      studentId: "stu-1",
      sessionId: "sess-1",
      dominantDimension: "pedaling",
      routing: ROUTING,
      pieceCtx: null,
    });

    expect((capturedRow as Record<string, unknown>).pieceId).toBeNull();
  });
});

describe("buildPendingExerciseComponent", () => {
  test("sets exerciseId to the pending row id returned by stageDominantExercise", () => {
    const staged = {
      exerciseId: "pending-row-id-1",
      focusDimension: "pedaling",
      previewTitle: "Own passage loop: pedaling (bars 12-16)",
    };
    const component = buildPendingExerciseComponent(staged);
    expect(component.type).toBe("pending_exercise");
    expect(component.config.exerciseId).toBe("pending-row-id-1");
    expect(component.config.focusDimension).toBe("pedaling");
    expect(component.config.previewTitle).toBe("Own passage loop: pedaling (bars 12-16)");
  });
});
