// apps/api/src/services/teacher_style.test.ts
import { describe, expect, it } from "vitest";
import { evaluate, selectClusters, formatTeacherVoiceBlocks, deriveSignals } from "./teacher_style";
import fixtures from "../../../shared/teacher-style/test_fixtures.json";

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

  it("evaluate handles unary minus", () => {
    expect(evaluate("-1.0", SIGNALS)).toBeCloseTo(-1.0);
    expect(evaluate("max_neg_dev - -0.5", { ...SIGNALS, max_neg_dev: 1.0 })).toBeCloseTo(1.5);
  });
});

describe("teacher_style.selectClusters parity fixtures", () => {
  for (const f of fixtures) {
    it(`fixture ${f.name}: primary contains ${f.expected_primary_substring}`, () => {
      const sel = selectClusters(f.signals);
      expect(sel.primary.name.toLowerCase()).toContain(f.expected_primary_substring.toLowerCase());
      expect(sel.secondary.name.toLowerCase()).toContain(f.expected_secondary_substring.toLowerCase());
    });
  }

  it("fallback when all scores low", () => {
    // duration_min=25 avoids duration_min<20 bonus on Artifact, all deviation signals below meaningful threshold
    const sel = selectClusters({
      max_neg_dev: 0.12, max_pos_dev: 0, n_significant: 0,
      drilling_present: false, drilling_improved: false,
      duration_min: 25, mode_count: 1, has_piece: false,
    });
    expect(sel.primary.name.toLowerCase()).toContain("technical");
    expect(sel.secondary.name.toLowerCase()).toMatch(/(positive|praise)/);
  });
});

describe("teacher_style.formatTeacherVoiceBlocks", () => {
  const sig = {
    max_neg_dev: 0.25, max_pos_dev: 0, n_significant: 3,
    drilling_present: false, drilling_improved: false,
    duration_min: 15, mode_count: 1, has_piece: true,
  };

  it("emits both teacher_voice and also_consider blocks", () => {
    const out = formatTeacherVoiceBlocks(selectClusters(sig));
    expect(out).toContain("<teacher_voice");
    expect(out).toContain("<also_consider");
    expect(out).toContain("Register:");
    expect(out).toContain("Tone:");
  });

  it("includes a normalized cluster id in the attribute", () => {
    const out = formatTeacherVoiceBlocks(selectClusters(sig));
    expect(out).toMatch(/cluster="[a-z][a-z0-9-]+"/);
  });
});

describe("teacher_style.deriveSignals", () => {
  it("produces the documented signal vector", () => {
    const sig = deriveSignals(
      [
        { dimension: "dynamics", score: 0.8, deviation_from_mean: 0.25, direction: "above_average" },
        { dimension: "timing", score: 0.3, deviation_from_mean: -0.18, direction: "below_average" },
      ],
      [],
      900_000,
      { title: "Prelude", composer: "Bach", skill_level: 3 },
      "continuous_play",
    );
    expect(sig.max_neg_dev).toBeCloseTo(0.18, 3);
    expect(sig.max_pos_dev).toBeCloseTo(0.25, 3);
    expect(sig.n_significant).toBe(2);
    expect(sig.drilling_present).toBe(false);
    expect(sig.drilling_improved).toBe(false);
    expect(sig.duration_min).toBeCloseTo(15, 1);
    expect(sig.mode_count).toBe(1);
    expect(sig.has_piece).toBe(true);
  });

  it("derives drilling_improved from first vs final score", () => {
    const sig = deriveSignals(
      [],
      [{ first_score: 0.5, final_score: 0.8 }],
      600_000,
      { title: "X", composer: "Bach", skill_level: 1 },
      "continuous_play",
    );
    expect(sig.drilling_present).toBe(true);
    expect(sig.drilling_improved).toBe(true);
  });

  it("deriveSignals drilling_improved with multiple records", () => {
    const moments = [{ score: 0.7, dimension: "dynamics" }];
    const drillingRecords = [
      { first_score: 0.4, final_score: 0.5 },
      { first_score: 0.5, final_score: 0.75 },
    ];
    const sig = deriveSignals(moments, drillingRecords, 1800000, null, "drilling");
    expect(sig.drilling_improved).toBe(true);
  });
});
