import embeddingsAsset from "./exercise_embeddings.json";
import manifest from "./exercise_primitives_manifest.json";

// Cosine drill selection: rank catalog drills by similarity of their Aria
// embedding to a query embedding of the student's weak passage, WITHIN the
// teacher-diagnosed dimension bucket (FILTER then RANK). This replaces the
// constant-per-dimension top-1 selector when a query embedding is available;
// when it is not (no AMT MIDI for the weak bars), callers fall back to the
// deterministic selectPrimitive path.
//
// The catalog vectors in exercise_embeddings.json are L2-normalized at export,
// so cosine reduces to a dot product once the query is normalized here.

type EmbeddingsAsset = { dim: number; ids: string[]; vectors: number[][] };
type ManifestEntry = { dimensions: string[] };

const ASSET = embeddingsAsset as EmbeddingsAsset;
const MANIFEST = manifest as Record<string, ManifestEntry>;

// id -> normalized vector, built once at module load.
const VECTORS: Map<string, number[]> = new Map(
  ASSET.ids.map((id, i) => [id, ASSET.vectors[i]]),
);

export type CosineMatch = { primitiveId: string; score: number };

function l2normalize(v: number[]): number[] {
  let sumsq = 0;
  for (const x of v) sumsq += x * x;
  const norm = Math.sqrt(sumsq);
  if (norm === 0) {
    // A zero query has no direction; refuse it rather than dividing by zero.
    throw new Error("cosine-select: query embedding has zero magnitude");
  }
  return v.map((x) => x / norm);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Rank catalog drills tagged for `targetDimension` by cosine similarity to
 * `query`, returning the best match. Returns null when no catalog drill is
 * tagged for the dimension (the caller widens to the neutral default), mirroring
 * the empty-bucket branch of selectPrimitive.
 *
 * Ties (equal cosine) break by primitive_id ascending, matching the model-side
 * match_exercises determinism.
 */
export function cosineSelectWithinDimension(
  query: number[],
  targetDimension: string,
): CosineMatch | null {
  if (query.length !== ASSET.dim) {
    throw new Error(
      `cosine-select: query dim ${query.length} != catalog dim ${ASSET.dim}`,
    );
  }
  const q = l2normalize(query);

  let best: CosineMatch | null = null;
  for (const id of ASSET.ids) {
    const entry = MANIFEST[id];
    if (entry === undefined || !entry.dimensions.includes(targetDimension)) {
      continue;
    }
    const vec = VECTORS.get(id);
    if (vec === undefined) continue;
    const score = dot(q, vec);
    if (
      best === null ||
      score > best.score ||
      (score === best.score && id < best.primitiveId)
    ) {
      best = { primitiveId: id, score };
    }
  }
  return best;
}
