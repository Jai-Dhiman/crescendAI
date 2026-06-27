import type { CorpusDrillDecision } from "../harness/artifacts/exercise-routing";
import type { ServiceContext } from "../lib/types";
import manifest from "./exercise_primitives_manifest.json";
import type { ExerciseSetPayload } from "./exercises";
import { parseKeyToPc, transposeInterval } from "./keys";

type ManifestEntry = { dimensions: string[]; key: string; totalBars: number };
const MANIFEST: Record<string, ManifestEntry> = manifest as Record<string, ManifestEntry>;

// The neutral generic fallback drill is an EXPLICIT product choice (a Hanon finger
// warm-up), NOT whatever happens to sort first globally. If this id is ever absent
// from the manifest that is a build error — we raise rather than silently fall back
// to the sort-first primitive (which would surface e.g. a Burgmuller character piece
// as a "general warm-up").
const WIDEN_DEFAULT_PRIMITIVE = "hanon_001";

// FAITHFUL match sort, mirroring the model's match_by_dimension ordering:
// (source_exercise_number, primitive_id) ascending == (suffixNum(id), id). All ids
// are "<source>_NNN"; the numeric suffix orders within a source, the id breaks ties
// ACROSS sources. We do NOT invent a source-priority tiebreak — the model has none.
function suffixNum(id: string): number {
  const m = id.match(/_(\d+)$/);
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
}

function stableSorted(ids: string[]): string[] {
  return [...ids].sort((a, b) => {
    const sa = suffixNum(a);
    const sb = suffixNum(b);
    if (sa !== sb) return sa - sb;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

type Selection = { primitiveId: string; widened: boolean };

function selectPrimitive(decision: CorpusDrillDecision): Selection {
  if (decision.primitive_id && decision.primitive_id in MANIFEST) {
    return { primitiveId: decision.primitive_id, widened: false };
  }
  const matches = stableSorted(
    Object.keys(MANIFEST).filter((id) =>
      MANIFEST[id].dimensions.includes(decision.target_dimension),
    ),
  );
  if (matches.length > 0) {
    return { primitiveId: matches[0], widened: false };
  }
  // WIDEN: the explicit neutral warm-up default (NOT sort-derived global-first).
  if (!(WIDEN_DEFAULT_PRIMITIVE in MANIFEST)) {
    throw new Error(
      `corpus-drill: WIDEN_DEFAULT_PRIMITIVE "${WIDEN_DEFAULT_PRIMITIVE}" absent from manifest`,
    );
  }
  return { primitiveId: WIDEN_DEFAULT_PRIMITIVE, widened: true };
}

async function resolveTranspose(
  ctx: ServiceContext,
  pieceId: string | null,
  primitiveKey: string,
): Promise<number> {
  // Best-effort: any unresolvable input -> 0 + a structured warn (NOT a silent
  // catch{return null}). The drill still renders, untransposed.
  const warn = (reason: string) =>
    console.log(
      JSON.stringify({
        level: "warn",
        message: "resolveTranspose: falling back to transpose=0",
        reason,
        pieceId,
        primitiveKey,
      }),
    );

  if (pieceId === null) {
    warn("no pieceId");
    return 0;
  }

  let raw: string;
  try {
    const obj = await ctx.env.SCORES.get(`scores/v1/${pieceId}.json`);
    if (obj === null) {
      warn("score JSON 404");
      return 0;
    }
    raw = await obj.text();
  } catch (e) {
    warn(`R2 read failed: ${String(e)}`);
    return 0;
  }

  let passageKey: unknown;
  try {
    passageKey = (JSON.parse(raw) as { key_signature?: unknown }).key_signature;
  } catch (e) {
    warn(`score JSON parse failed: ${String(e)}`);
    return 0;
  }
  if (typeof passageKey !== "string") {
    warn("key_signature null or non-string");
    return 0;
  }

  const fromPc = parseKeyToPc(primitiveKey);
  const toPc = parseKeyToPc(passageKey);
  if (fromPc === null || toPc === null) {
    warn(`unparseable key (primitive=${primitiveKey}, passage=${passageKey})`);
    return 0;
  }
  return transposeInterval(fromPc, toPc);
}

export async function buildCorpusDrillClip(
  ctx: ServiceContext,
  decision: CorpusDrillDecision,
  pieceId: string | null,
): Promise<ExerciseSetPayload> {
  const { primitiveId, widened } = selectPrimitive(decision);
  const entry = MANIFEST[primitiveId];
  const transpose = await resolveTranspose(ctx, pieceId, entry.key);

  const dim = decision.target_dimension;
  const instruction = widened
    ? `This is a general warm-up drill (no ${dim}-specific drill in corpus yet). ` +
      `Play this primitive at ${Math.round(decision.tempo_factor * 100)}% tempo.`
    : `${dim} drill. Play this primitive at ${Math.round(decision.tempo_factor * 100)}% tempo, focusing on ${dim}.`;

  return {
    sourcePassage: `bars ${decision.bar_range[0]}-${decision.bar_range[1]}`,
    targetSkill: dim,
    scoreClip: {
      pieceId: primitiveId,
      bars: [1, entry.totalBars],
      tempoFactor: decision.tempo_factor,
      transpose,
    },
    exercises: [
      {
        title: widened ? `General warm-up: ${dim}` : `${dim} corpus drill`,
        instruction,
        focusDimension: dim,
      },
    ],
  };
}
