// apps/api/src/services/teacher_style.test.ts
import { describe, expect, it } from "vitest";
import { evaluate } from "./teacher_style";

const SIGNALS = {
  max_neg_dev: 0.2, max_pos_dev: 0.0, n_significant: 2,
  drilling_present: false, drilling_improved: false,
  duration_min: 15.0, mode_count: 1, has_piece: true,
};

describe("teacher_style.evaluate", () => {
  it("evaluates arithmetic over signals", () => {
    expect(evaluate("1.5 * max_neg_dev + 0.3 * n_significant", SIGNALS)).toBeCloseTo(0.9, 6);
  });

  it("looks up a single signal", () => {
    expect(evaluate("max_neg_dev", SIGNALS)).toBeCloseTo(0.2, 6);
  });

  it("rejects unknown signal names", () => {
    expect(() => evaluate("max_neg_dev + bogus", SIGNALS)).toThrow(/unknown signal/);
  });
});
