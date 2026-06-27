// apps/api/src/services/tool-processor.corpus-drill.test.ts
import { describe, expect, test } from "vitest";
import { processToolUse } from "./tool-processor";

// A ServiceContext whose SCORES.get always 404s, so resolveTranspose -> 0.
function ctxNoScores() {
  return { db: {} as never, env: { SCORES: { get: async () => null } } as never };
}

describe("prescribe_exercise — corpus_drill", () => {
  test("corpus_drill returns an exercise_set with a primitive scoreClip", async () => {
    const result = await processToolUse(ctxNoScores(), "stu-1", "prescribe_exercise", {
      kind: "corpus_drill",
      target_dimension: "phrasing",
      bar_range: [4, 8],
      tempo_factor: 0.8,
      primitive_id: null,
      piece_id: null,
    });
    expect(result.isError).toBe(false);
    expect(result.componentsJson).toHaveLength(1);
    expect(result.componentsJson[0].type).toBe("exercise_set");
    const cfg = result.componentsJson[0].config as {
      scoreClip?: { pieceId: string; bars: [number, number]; transpose?: number };
    };
    expect(cfg.scoreClip?.pieceId).toBe("bach_001"); // phrasing stable-first in the 154-drill corpus
    expect(cfg.scoreClip?.bars).toEqual([1, 22]);
    expect(cfg.scoreClip).toHaveProperty("transpose");
  });
});
