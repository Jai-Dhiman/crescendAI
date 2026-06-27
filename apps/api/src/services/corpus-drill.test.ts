// apps/api/src/services/corpus-drill.test.ts
import { describe, expect, test, vi } from "vitest";
import type { CorpusDrillDecision } from "../harness/artifacts/exercise-routing";
import { buildCorpusDrillClip } from "./corpus-drill";

// A ServiceContext whose SCORES.get always 404s (returns null), so resolveTranspose
// degrades to transpose=0. pieceId is null here, so SCORES is never even read.
function ctxNoScores() {
  return {
    db: {} as never,
    env: { SCORES: { get: async () => null } } as never,
  };
}

function corpusDrill(over: Partial<CorpusDrillDecision> = {}): CorpusDrillDecision {
  return {
    kind: "corpus_drill",
    target_dimension: "timing",
    bar_range: [4, 8],
    tempo_factor: 0.8,
    primitive_id: null,
    ...over,
  } as CorpusDrillDecision;
}

describe("buildCorpusDrillClip — selection + assembly", () => {
  test("explicit primitive_id in the manifest is selected and clipped whole", async () => {
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: "czerny_001", target_dimension: "timing" }),
      null,
    );
    expect(payload.scoreClip).toEqual({
      pieceId: "czerny_001",
      bars: [1, 22], // whole primitive, NOT the student bar_range [4,8]
      tempoFactor: 0.8,
      transpose: 0, // null pieceId -> best-effort 0
    });
    // truly matched (explicit + dimension match) -> dimension-specific wording.
    expect(payload.exercises[0].instruction).toContain("timing");
    expect(payload.exercises[0].instruction).not.toContain("general warm-up");
  });

  test("dimension-matched selection picks the FAITHFUL stable-first primitive (timing -> chopin_etude_001)", async () => {
    // The sort is FAITHFUL to the model's match_by_dimension:
    // (source_exercise_number, primitive_id) ascending == (suffixNum(id), id). Over the
    // full 154-drill corpus the timing stable-first is chopin_etude_001 (suffix 1; the
    // lower-id suffix-1 ids bach_001/chopin_001 don't carry "timing"). bars == 79.
    const timing = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "timing" }),
      null,
    );
    expect(timing.scoreClip?.pieceId).toBe("chopin_etude_001");
    expect(timing.scoreClip?.bars).toEqual([1, 79]);
    expect(timing.exercises[0].instruction).toContain("timing");
    expect(timing.exercises[0].instruction).not.toContain("general warm-up");

    // "phrasing" stable-first is bach_001 (suffix-1, lowest id carrying phrasing).
    // bach_001.totalBars == 22.
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "phrasing" }),
      null,
    );
    expect(payload.scoreClip?.pieceId).toBe("bach_001");
    expect(payload.scoreClip?.bars).toEqual([1, 22]);
    expect(payload.exercises[0].instruction).toContain("phrasing");
    expect(payload.exercises[0].instruction).not.toContain("general warm-up");
  });

  test("pedaling now resolves to a REAL pedaling drill (corpus closed the pedaling=0 gap)", async () => {
    // The old 22-primitive corpus had ZERO pedaling drills, so "pedaling" widened to
    // the hanon_001 warm-up. The 154-drill corpus carries 37 pedaling drills; the
    // stable-first is chopin_001 (bars 34). This is a real dimension match -- NOT the
    // widen path -- so the wording is dimension-specific, not "general warm-up".
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "pedaling" }),
      null,
    );
    expect(payload.scoreClip?.pieceId).toBe("chopin_001");
    expect(payload.scoreClip?.bars).toEqual([1, 34]);
    expect(payload.exercises[0].instruction).toContain("pedaling");
    expect(payload.exercises[0].instruction).not.toContain("general warm-up");
  });

  test("explicit primitive_id NOT in the manifest falls through to dimension match", async () => {
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: "nonexistent_999", target_dimension: "phrasing" }),
      null,
    );
    // nonexistent_999 is not in the manifest -> ignored -> dimension match on "phrasing".
    expect(payload.scoreClip?.pieceId).toBe("bach_001");
  });
});

// Fake R2 whose .get(key) returns an object with .text() resolving to the given
// JSON string, or null to simulate a 404.
function ctxWithScoreJson(jsonByKey: Record<string, string | null>) {
  return {
    db: {} as never,
    env: {
      SCORES: {
        get: async (key: string) => {
          const v = jsonByKey[key];
          if (v == null) return null;
          return { text: async () => v };
        },
      },
    } as never,
  };
}

describe("buildCorpusDrillClip — resolveTranspose", () => {
  test("transposes from primitive key to passage key (czerny_001 'c'=0 -> 'D'=2 => +2)", async () => {
    const ctx = ctxWithScoreJson({
      "scores/v1/student.piece.json": JSON.stringify({ key_signature: "D" }),
    });
    const payload = await buildCorpusDrillClip(
      ctx,
      corpusDrill({ primitive_id: "czerny_001", target_dimension: "timing" }),
      "student.piece",
    );
    expect(payload.scoreClip?.transpose).toBe(2); // C(0) -> D(2)
  });

  test("nearest-octave: 'C'=0 -> 'A'=9 resolves to -3", async () => {
    const ctx = ctxWithScoreJson({
      "scores/v1/student.piece.json": JSON.stringify({ key_signature: "A" }),
    });
    const payload = await buildCorpusDrillClip(
      ctx,
      corpusDrill({ primitive_id: "hanon_001", target_dimension: "timing" }), // hanon_001 key "C"
      "student.piece",
    );
    expect(payload.scoreClip?.transpose).toBe(-3);
  });

  test("tritone 'C'=0 -> 'F#'=6 resolves to +6", async () => {
    const ctx = ctxWithScoreJson({
      "scores/v1/student.piece.json": JSON.stringify({ key_signature: "F#" }),
    });
    const payload = await buildCorpusDrillClip(
      ctx,
      corpusDrill({ primitive_id: "hanon_001", target_dimension: "timing" }),
      "student.piece",
    );
    expect(payload.scoreClip?.transpose).toBe(6);
  });

  test("404 (no score JSON) degrades to transpose=0 with a structured warn", async () => {
    const warn = vi.spyOn(console, "log").mockImplementation(() => {});
    const ctx = ctxWithScoreJson({}); // every key -> null (404)
    const payload = await buildCorpusDrillClip(
      ctx,
      corpusDrill({ primitive_id: "hanon_001", target_dimension: "timing" }),
      "missing.piece",
    );
    expect(payload.scoreClip?.transpose).toBe(0);
    expect(warn).toHaveBeenCalled();
    const logged = (warn.mock.calls[0]?.[0] ?? "") as string;
    expect(logged).toContain("resolveTranspose");
    warn.mockRestore();
  });

  test("null key_signature degrades to transpose=0 with a warn", async () => {
    const warn = vi.spyOn(console, "log").mockImplementation(() => {});
    const ctx = ctxWithScoreJson({
      "scores/v1/student.piece.json": JSON.stringify({ key_signature: null }),
    });
    const payload = await buildCorpusDrillClip(
      ctx,
      corpusDrill({ primitive_id: "hanon_001", target_dimension: "timing" }),
      "student.piece",
    );
    expect(payload.scoreClip?.transpose).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
