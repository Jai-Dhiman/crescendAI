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

describe("teacher_style.evaluate conditionals", () => {
  it("returns then-branch when condition is true", () => {
    const sig = { ...SIGNALS, drilling_improved: true };
    expect(evaluate("1.5 if drilling_improved else 0", sig)).toBeCloseTo(1.5, 6);
  });

  it("returns else-branch when condition is false", () => {
    const sig = { ...SIGNALS, drilling_improved: false };
    expect(evaluate("1.5 if drilling_improved else 0.5", sig)).toBeCloseTo(0.5, 6);
  });

  it("evaluates the technical-corrective formula", () => {
    const sig = { ...SIGNALS, max_neg_dev: 0.2, n_significant: 2, drilling_improved: false };
    const formula = "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)";
    expect(evaluate(formula, sig)).toBeCloseTo(0.9, 6);
  });

  it("supports compound boolean: a < x and b < x", () => {
    const sig = { ...SIGNALS, max_neg_dev: 0.05, max_pos_dev: 0.05 };
    expect(evaluate("1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0", sig)).toBeCloseTo(1, 6);
  });
});
