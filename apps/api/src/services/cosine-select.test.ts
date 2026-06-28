import { describe, expect, test } from "vitest";
import { cosineSelectWithinDimension } from "./cosine-select";
import embeddingsAsset from "./exercise_embeddings.json";
import manifest from "./exercise_primitives_manifest.json";

const ASSET = embeddingsAsset as { dim: number; ids: string[]; vectors: number[][] };
const MANIFEST = manifest as Record<string, { dimensions: string[] }>;

function vectorOf(id: string): number[] {
  const i = ASSET.ids.indexOf(id);
  if (i < 0) throw new Error(`no vector for ${id}`);
  return ASSET.vectors[i];
}

function firstIdForDimension(dim: string): string {
  const id = ASSET.ids.find((x) => MANIFEST[x]?.dimensions.includes(dim));
  if (!id) throw new Error(`no drill for ${dim}`);
  return id;
}

describe("cosineSelectWithinDimension", () => {
  test("a drill's own vector selects that drill (cosine ~1) within its dimension", () => {
    const id = firstIdForDimension("timing");
    const match = cosineSelectWithinDimension(vectorOf(id), "timing");
    expect(match).not.toBeNull();
    expect(match?.primitiveId).toBe(id);
    expect(match?.score).toBeCloseTo(1, 5);
  });

  test("respects the dimension FILTER (never returns an off-dimension drill)", () => {
    // Query is a pedaling drill's vector, but we rank within timing: result must
    // be a timing-tagged drill, not the pedaling one.
    const pedalId = firstIdForDimension("pedaling");
    const match = cosineSelectWithinDimension(vectorOf(pedalId), "timing");
    expect(match).not.toBeNull();
    expect(MANIFEST[match!.primitiveId].dimensions).toContain("timing");
  });

  test("returns the MOST similar drill within the bucket, not just the first", () => {
    // Build a query closest to a specific timing drill (its own vector). The
    // selector must return it even if it is not the lowest-id timing drill.
    const timingIds = ASSET.ids.filter((x) => MANIFEST[x]?.dimensions.includes("timing"));
    const target = timingIds[timingIds.length - 1]; // last, not first
    const match = cosineSelectWithinDimension(vectorOf(target), "timing");
    expect(match?.primitiveId).toBe(target);
  });

  test("returns null when no drill is tagged for the dimension", () => {
    const q = ASSET.vectors[0];
    expect(cosineSelectWithinDimension(q, "no_such_dimension")).toBeNull();
  });

  test("throws on a dimension/query dimensionality mismatch", () => {
    expect(() => cosineSelectWithinDimension([1, 2, 3], "timing")).toThrow(/dim/);
  });

  test("throws on a zero-magnitude query", () => {
    const zero = new Array(ASSET.dim).fill(0);
    expect(() => cosineSelectWithinDimension(zero, "timing")).toThrow(/zero magnitude/);
  });

  test("an unnormalized query (scaled) selects the same drill as normalized", () => {
    const id = firstIdForDimension("phrasing");
    const v = vectorOf(id);
    const scaled = v.map((x) => x * 7.5);
    expect(cosineSelectWithinDimension(scaled, "phrasing")?.primitiveId).toBe(id);
  });
});
