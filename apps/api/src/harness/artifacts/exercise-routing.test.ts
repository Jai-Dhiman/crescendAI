import { describe, test, expect } from "vitest";
import { ExerciseRoutingDecisionSchema } from "./exercise-routing";

describe("ExerciseRoutingDecisionSchema — own_passage_loop", () => {
  const validLoop = {
    kind: "own_passage_loop",
    target_dimension: "pedaling",
    bar_range: [12, 16],
    tempo_factor: 0.75,
  };

  test("accepts a valid own_passage_loop", () => {
    expect(() => ExerciseRoutingDecisionSchema.parse(validLoop)).not.toThrow();
  });

  test("rejects own_passage_loop with bar_range start > end", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      bar_range: [16, 12],
    });
    expect(result.success).toBe(false);
    expect(result.error?.issues.some((i) => i.path.includes("bar_range"))).toBe(true);
  });

  test("rejects own_passage_loop with tempo_factor below 0.25", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      tempo_factor: 0.1,
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with tempo_factor above 1.0", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      tempo_factor: 1.1,
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with invalid target_dimension", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      target_dimension: "vibrato",
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with bar_range containing non-positive numbers", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validLoop,
      bar_range: [0, 4],
    });
    expect(result.success).toBe(false);
  });

  test("rejects own_passage_loop with extraneous primitive_id field", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      kind: "own_passage_loop",
      target_dimension: "pedaling",
      bar_range: [12, 16],
      tempo_factor: 0.75,
      primitive_id: "some-drill",
    });
    expect(result.success).toBe(false);
  });
});

describe("ExerciseRoutingDecisionSchema — corpus_drill", () => {
  const validDrill = {
    kind: "corpus_drill",
    target_dimension: "timing",
    bar_range: [1, 8],
    tempo_factor: 0.8,
    primitive_id: null,
  };

  test("accepts a valid corpus_drill with primitive_id null", () => {
    expect(() => ExerciseRoutingDecisionSchema.parse(validDrill)).not.toThrow();
  });

  test("accepts a valid corpus_drill with non-null primitive_id", () => {
    expect(() =>
      ExerciseRoutingDecisionSchema.parse({ ...validDrill, primitive_id: "drill-abc" })
    ).not.toThrow();
  });

  test("rejects corpus_drill with bar_range start > end", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      ...validDrill,
      bar_range: [8, 1],
    });
    expect(result.success).toBe(false);
  });
});

describe("ExerciseRoutingDecisionSchema — discriminant", () => {
  test("rejects an unknown kind", () => {
    const result = ExerciseRoutingDecisionSchema.safeParse({
      kind: "free_improv",
      target_dimension: "phrasing",
      bar_range: [1, 4],
      tempo_factor: 0.5,
    });
    expect(result.success).toBe(false);
  });
});
