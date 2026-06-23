# Corpus-Drill Runtime Selection + Transposed/Excerpted Display Design

**Goal:** A pianist who gets a `corpus_drill` prescription sees a real, playable score clip of a matched practice primitive — transposed into their own piece's key and playable at the prescribed tempo — instead of a "coming soon" text stub.
**Issue:** #47 (epic #44 S4)

**Not in scope:**
- Within-bucket RANK / difficulty ordering among matched primitives (#42 / #43 / S6). Selection is a deterministic stable sort only.
- Growing the renderable corpus beyond the 22 primitives that have R2 assets on `main` (the 443/242 corpus on `issue-49` is UNMERGED and its R2 uploads are deferred).
- Any production deploy or prod R2 upload (local-first, pre-beta; "shipped" = local merge).
- Changing the Python `keys.py` oracle or back-porting normalization to Python.
- Cropping the primitive to the student's `bar_range` — corpus_drill renders the WHOLE primitive (`[1, totalBars]`). `bar_range` refers to the student passage and is ignored for primitive cropping.

## Problem

Today a `corpus_drill` prescription is a dead end. Both call sites emit a text-only stub:

- `apps/api/src/services/tool-processor.ts` `processPrescribeExercise` (~lines 116-137): `"<dim> drill coming soon (drill: <primitive_id>). Practice bars N-M ..."` with no `scoreClip`.
- `apps/api/src/services/exercises.ts` `assignPendingExercise` (~lines 254-272): `"<dim> drill coming soon (drill: <primitive_id>). In the meantime: ..."` with no `scoreClip`.

The pianist never sees the actual drill notation, cannot hear it, and the matched-primitive concept (the whole point of the retrieval-based exercise system, slices A-C) is invisible at runtime. Meanwhile the rendering pipeline already exists: `own_passage_loop` renders a playable, looped, tempo-controlled clip through `ExerciseSetCard` -> `scoreRenderer` -> `score-worker`, and the worker already supports client-side Verovio transpose + crop. The only missing piece is a runtime decision layer that (a) picks WHICH primitive to show, (b) computes the transpose interval from the primitive's notated key to the student passage's key, and (c) assembles a `scoreClip` the existing renderer can consume.

The primitive-selection and transpose logic exists in Python (`model/src/exercise_corpus/briefing.py` + `keys.py`) but runs offline at corpus-build time, not in the live Worker request path. The Worker has no port of it.

## Solution (from the user's perspective)

When the teacher prescribes a corpus drill (in post-session synthesis via `assignPendingExercise`, or mid-chat via `processPrescribeExercise`), the pianist sees an `ExerciseSetCard` with a real engraved score of a matched practice primitive (e.g. a Hanon or Czerny figure), already transposed into the key of the piece they were just playing, with the existing play / stop / tempo-slider transport and moving cursor. The drill plays the transposed pitches that match the displayed notation. When no dimension-specific drill exists yet, they get an honest general warm-up drill (hanon_001) with wording that says so, rather than a silent or misleading match.

## Design

A single deep module `apps/api/src/services/corpus-drill.ts` owns the entire runtime corpus_drill decision. Its public interface is one async function, `buildCorpusDrillClip(ctx, decision, pieceId)`, that returns a fully-assembled `ExerciseSetPayload`. Both call sites delegate to it, replacing their divergent text stubs with one shared path.

The module hides three substantial sub-decisions behind that one interface:

1. **Primitive selection** (`selectPrimitive`, internal): Given the routing decision, pick a primitive id from the committed manifest. Priority: explicit `primitive_id` if present and in the manifest; else the stable-sorted first primitive whose `dimensions` include `target_dimension`; else WIDEN to the global stable-first renderable (hanon_001) flagged `widened: true`. Stable sort key is `(numeric suffix of primitive_id, primitive_id)`.

2. **Transpose resolution** (`resolveTranspose`, internal): A best-effort TS port of `keys.py`. Loads the student passage's `key_signature` from R2 (`scores/v1/{pieceId}.json`), parses both the primitive key and passage key to pitch classes, and computes the nearest-octave interval FROM primitive TO passage. Genuinely-unresolvable inputs (no pieceId, 404, null/unparseable key) degrade to `transpose = 0` with a structured `console.log` warn — explicit, not a swallowed `catch {}`.

3. **Clip assembly**: Build `scoreClip = { pieceId: primitiveId, bars: [1, totalBars], tempoFactor, transpose }` and honest instruction text (general-warm-up wording when widened, dimension-specific wording when truly matched).

### Manifest as single source of truth

The manifest `apps/api/src/services/exercise_primitives_manifest.json` is generated (not hand-edited) by extending `model/src/exercise_corpus/build_render_assets.py`. For ONLY the primitives that actually have a built `.mxl` (the 22 renderable on `main`: hanon_001-020, czerny_001, burgmuller_001), it emits `{ "<id>": { dimensions, key, totalBars } }`. `dimensions` and `key` come from `technique_tags.toml`; `totalBars` from the partitura-parsed score. Restricting to built assets is the FILTER that keeps the runtime from selecting a primitive whose `.mxl` is not in R2 (which would 404 the renderer).

`key` is emitted VERBATIM from `technique_tags.toml` (so czerny_001 stays `"c"` lowercase); normalization happens at runtime in `resolveTranspose`, not in the manifest.

### The transpose-port decision (A')

`keys.py`'s `_PC` dict is uppercase-tonic-only and **raises** on lowercase-minor keys. The corpus itself uses lowercase-minor tags (czerny_001 = `"c"`), so the Python oracle is latently broken on its own corpus. The TS port is a deliberate strict **superset**:

1. Replicate `_PC` verbatim (uppercase entries only, exactly as in `keys.py`).
2. Replicate mode-stripping exactly: strip trailing `" major"`/`" minor"` (case-insensitive), then strip a trailing `"m"` when length > 1.
3. ADD one normalization the oracle lacks: before the `_PC` lookup, capitalize ONLY the first character of the remaining tonic (`"c"`->`"C"`, `"eb"`->`"Eb"`, `"f#"`->`"F#"`, `"C#"`->`"C#"`).

Result: identical to the oracle for every input the oracle accepts (all uppercase); musically-correct tonic extraction for lowercase tags the oracle rejects. Mode is irrelevant to semitone transposition; only tonic pitch class matters, so this is correct, not a hack. The `transpose=0`+warn branch is reserved for genuinely-unresolvable cases (no pieceId, 404, null key, or a tonic STILL unparseable after first-char capitalization, e.g. garbage).

Trade-off chosen: superset-port over (a) inheriting the oracle bug (czerny_001 would always degrade to transpose=0) and (b) back-porting to Python (out of scope, would force a Python change + re-validation of `briefing.py`). The parity test asserts byte-for-byte equality with the oracle on the oracle's accepted (uppercase) domain, then adds TS-specific superset cases documented as intentional divergence.

### Contract change: `scoreClip.transpose`

`scoreClip` gains an optional `transpose?: number` in BOTH the API (`ExerciseSetPayload` in `exercises.ts`, and the `exercise_set` config built in `tool-processor.ts`) and the web type (`ExerciseSetConfig.scoreClip` in `apps/web/src/lib/types.ts`). When `transpose` is omitted or `0`, behavior is byte-identical to today's `own_passage` path (regression-safe): the web renderer already keys its cache by `${pieceId}:${transpose ?? 0}` and the worker's `transpose:0`/undefined is a documented Verovio no-op.

### Web threading

The web `ExerciseSetCard` path threads `config.scoreClip.transpose` into `scoreRenderer.load(pieceId, transpose)` and `getClip/getClipPlayback(pieceId, b0, b1, transpose)`. The renderer + worker already accept and apply `transpose` (client-side Verovio `tk.setOptions({ transpose: String(n) })` at load). corpus_drill carries `tempoFactor`, so the existing `LoopTransport` + `ScoreCursor` path renders unchanged (playable at reduced tempo).

CRITICAL CORRECTNESS: `getClipPlayback` must extract the playback notes AFTER the transposed load, so smplr plays the transposed pitches that match the displayed notation. The worker already does this (it reads `getMIDIValuesForElement` from the toolkit loaded WITH transpose). A web verification asserts that a transposed load yields shifted playback MIDI relative to an untransposed load of the same piece.

## Modules

### `corpus-drill.ts` (deep, NEW)
- **Interface:** `export async function buildCorpusDrillClip(ctx: ServiceContext, decision: CorpusDrillDecision, pieceId: string | null): Promise<ExerciseSetPayload>`
- **Hides:** manifest lookup; the three-tier primitive selection (explicit -> dimension-matched -> widened) and its stable sort; loading + parsing the passage key from R2; the `keys.py` superset port (`parse_key_to_pc` + `transpose_interval`); best-effort failure handling with structured warns; honest instruction-text construction.
- **Tested through:** the public `buildCorpusDrillClip` only (selection, transpose, widening, and assembly are all observable in the returned `ExerciseSetPayload`). The R2 read is exercised through a fake `ctx.env.SCORES` (an `R2Bucket`-shaped object), not a mocked internal collaborator.
- **Depth verdict:** DEEP — one async function hides selection + key-resolution + transpose-math + assembly.

### `exercise_primitives_manifest.json` (generated data, NEW)
- **Interface:** `Record<primitiveId, { dimensions: string[]; key: string; totalBars: number }>`, imported by `corpus-drill.ts`.
- **Hides:** which primitives are actually renderable (the 22-asset filter) and their per-primitive facts; consumers never read `technique_tags.toml` or parse scores at runtime.
- **Depth verdict:** DEEP (as a data interface) — collapses build-time corpus knowledge into one runtime-cheap lookup.

### `build_render_assets.py` `build()` (MODIFY, existing deep module)
- **Interface unchanged for assets**; ALSO writes the manifest JSON next to the assets, keyed by built primitive id.
- **Hides:** TOML parsing + partitura bar-counting + the built-asset filter, behind the same `just build-exercise-assets` entry point.
- **Depth verdict:** DEEP — manifest generation is a hidden side-effect of the existing deterministic build.

## Verification Architecture

- **Canonical success state:**
  1. `apps/api/src/services/exercise_primitives_manifest.json` exists, contains EXACTLY the 22 built primitives, each with `dimensions` (array), `key` (verbatim string), `totalBars` (positive int).
  2. `buildCorpusDrillClip` returns an `ExerciseSetPayload` whose `scoreClip` has `pieceId = <selected primitive>`, `bars = [1, totalBars]`, `tempoFactor = decision.tempo_factor`, and `transpose = interval(primitiveKey -> passageKey)` (or `0`+warn on unresolvable input).
  3. The TS `parse_key_to_pc` + `transpose_interval` match the Python oracle byte-for-byte on the uppercase domain; the documented superset cases resolve.
  4. The flipped `exercises.test.ts` corpus_drill assertion: corpus_drill now HAS a `scoreClip` (pieceId = primitive, bars = `[1, N]`, transpose present).
  5. Web: a transposed `getClipPlayback` yields playback MIDI shifted by `transpose` relative to the untransposed load.
- **Automated check:** `cd apps/api && bun run test src/services/corpus-drill.test.ts src/services/exercises.test.ts` (Vitest) for the API; `cd model && uv run python -m pytest tests/exercise_corpus/test_build_render_assets.py` for the manifest; `cd apps/web && bunx vitest run src/lib/score-worker.transpose.test.ts` for the web playback-correctness check.
- **Harness (Task Group 0):** A parity-fixture harness is buildable BEFORE the feature. `model/src/exercise_corpus/keys.py` is the reference implementation; a committed JSON fixture of `(from_key, to_key) -> expected_interval` pairs (generated from the live oracle for the uppercase domain) is the golden the TS port must match. This is Task 1 (it produces the fixture by running the real oracle, then the TS parity test consumes it). It is shaped as a normal vertical-slice task: one test (TS parity) + one impl (the TS port + committed fixture) + one commit.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/exercise_corpus/build_render_assets.py` | `build()` also emits `exercise_primitives_manifest.json` (22 built primitives: dimensions, key, totalBars) | Modify |
| `model/tests/exercise_corpus/test_build_render_assets.py` | Test asserting manifest shape + 22-entry filter | New |
| `apps/api/src/services/exercise_primitives_manifest.json` | Generated manifest, committed | New |
| `apps/api/src/services/corpus-drill.ts` | `buildCorpusDrillClip` deep module (selection + transpose port + assembly) | New |
| `apps/api/src/services/corpus-drill.test.ts` | Behavior tests for selection, transpose, widening, assembly, parity | New |
| `apps/api/src/services/exercises.ts` | `ExerciseSetPayload.scoreClip` gains `transpose?: number`; corpus_drill branch calls `buildCorpusDrillClip` | Modify |
| `apps/api/src/services/exercises.test.ts` | FLIP corpus_drill assertion (now HAS scoreClip) | Modify |
| `apps/api/src/services/tool-processor.ts` | exercise_set config `scoreClip` gains `transpose`; corpus_drill branch calls `buildCorpusDrillClip` | Modify |
| `apps/web/src/lib/types.ts` | `ExerciseSetConfig.scoreClip` gains `transpose?: number` | Modify |
| `apps/web/src/components/.../ExerciseSetCard` (clip-loading call site) | Thread `scoreClip.transpose` into `load`/`getClip`/`getClipPlayback` | Modify |
| `apps/web/src/lib/score-worker.transpose.test.ts` | Transposed playback-MIDI correctness test | New |

## Open Questions

- Q: Does the web `ExerciseSetCard` clip-loading call site already destructure `scoreClip`, or does it read fields individually? Default: the build agent reads the actual component file first (named in its task) and threads `transpose` through whatever call shape exists, without restructuring unrelated code.
- Q: Does `model/tests/exercise_corpus/` already exist as a pytest dir? Default: if not, the build agent creates it alongside the new test file; the test is anchored to `__file__`, not CWD.
