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

  test("dimension-matched selection picks the FAITHFUL stable-first primitive (timing -> czerny_001)", async () => {
    // "timing" matches hanon_001..020 + czerny_001. The sort is FAITHFUL to the
    // model's match_by_dimension: (source_exercise_number, primitive_id) ascending,
    // i.e. (suffixNum(id), id). All three sources have a suffix-1 member, so the
    // id tiebreak decides and "czerny_001" < "hanon_001": the timing stable-first
    // is czerny_001, NOT hanon_001. czerny_001.totalBars == 22.
    const timing = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "timing" }),
      null,
    );
    expect(timing.scoreClip?.pieceId).toBe("czerny_001");
    expect(timing.scoreClip?.bars).toEqual([1, 22]);
    expect(timing.exercises[0].instruction).toContain("timing");
    expect(timing.exercises[0].instruction).not.toContain("general warm-up");

    // "phrasing" has a single match (only burgmuller_001 carries it among the 22),
    // so it resolves unambiguously regardless of tiebreak. burgmuller_001.totalBars == 23.
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "phrasing" }),
      null,
    );
    expect(payload.scoreClip?.pieceId).toBe("burgmuller_001");
    expect(payload.scoreClip?.bars).toEqual([1, 23]);
    expect(payload.exercises[0].instruction).toContain("phrasing");
    expect(payload.exercises[0].instruction).not.toContain("general warm-up");
  });

  test("no dimension match widens to the explicit hanon_001 constant with honest general-warm-up wording", async () => {
    // "pedaling" has NO built primitive among the 22 (chopin/satie are unbuilt).
    // The widen fallback is the EXPLICIT WIDEN_DEFAULT_PRIMITIVE constant ("hanon_001").
    // hanon_001.totalBars == 29.
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: null, target_dimension: "pedaling" }),
      null,
    );
    expect(payload.scoreClip?.pieceId).toBe("hanon_001");
    expect(payload.scoreClip?.bars).toEqual([1, 29]);
    expect(payload.exercises[0].instruction).toContain("general warm-up");
    expect(payload.exercises[0].instruction).toContain("pedaling"); // honest: names the missing dim
  });

  test("explicit primitive_id NOT in the manifest falls through to dimension match", async () => {
    const payload = await buildCorpusDrillClip(
      ctxNoScores(),
      corpusDrill({ primitive_id: "chopin_009", target_dimension: "phrasing" }),
      null,
    );
    // chopin_009 is not a built asset -> ignored -> dimension match on "phrasing".
    expect(payload.scoreClip?.pieceId).toBe("burgmuller_001");
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
