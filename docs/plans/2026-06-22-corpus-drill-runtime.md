# Corpus-Drill Runtime Selection + Transposed/Excerpted Display Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.
> Use model "Sonnet 4.6" for all subagents (search, review, implementation).

**Goal:** A pianist who gets a `corpus_drill` prescription sees a real, playable score clip of a matched practice primitive — transposed into their own piece's key and playable at the prescribed tempo — instead of a "coming soon" text stub.
**Issue:** #47 (epic #44 S4)
**Branch:** `issue-47-corpus-drill-runtime` (worktree already created at `.worktrees/issue-47-corpus-drill-runtime`)
**Spec:** docs/specs/2026-06-22-corpus-drill-runtime-design.md
**Style:** Follow `apps/api/TS_STYLE.md` for all `apps/api/` code (never destructure `c.env`, domain errors not HTTPException, `console.log(JSON.stringify({...}))` logging). Python: `uv`, explicit exceptions, anchor paths to `__file__`. No emojis.

---

## Task Groups

- **Group A (parallel):** Task 1 (manifest generation, Python), Task 2 (TS key-port + parity fixture). Non-overlapping files.
- **Group B (sequential, depends on A):** Task 3 (`buildCorpusDrillClip` selection + assembly), then Task 4 (transpose wiring into `buildCorpusDrillClip`). Both touch `corpus-drill.ts` / its test, so they are sequential within B and depend on Task 1's committed manifest + Task 2's committed port.
- **Group C (sequential, depends on B):** Task 5 (`scoreClip.transpose` contract + `exercises.ts` wiring + FLIP `exercises.test.ts`), Task 6 (`tool-processor.ts` wiring). Task 5 and Task 6 both touch the API contract type lineage; run sequentially.
- **Group D (depends on C for the type, otherwise independent files):** Task 7 (web type + `ExerciseSetCard` threading), Task 8 (web transposed-playback correctness test).

**Decouple check:**
- Group A `[SHIPS INDEPENDENTLY]`: the committed manifest + TS key-port are usable building blocks; after A the manifest exists and the parity port is verified, even before the runtime module exists. No user-facing value alone.
- Group B + C together are the first point of user-facing value: corpus_drill stops being a text stub server-side. Web (Group D) is required for the pianist to actually SEE the transposed clip, so the full user story needs A-D.

---

### Task 1: Generate the exercise-primitives manifest from the build script
**Group:** A (parallel with Task 2)

**Behavior being verified:** `just build-exercise-assets` (via `build()`) emits a manifest JSON containing exactly the 22 built primitives, each with `dimensions`, verbatim `key`, and partitura-counted `totalBars`.
**Interface under test:** `build_render_assets.build()` public return + the manifest file it writes.

**Files:**
- Modify: `model/src/exercise_corpus/build_render_assets.py`
- Test: `model/tests/exercise_corpus/test_build_render_assets.py` (APPEND — this file ALREADY EXISTS with 3 passing regression tests; preserve them verbatim, do NOT recreate the file, do NOT recreate `__init__.py`)

**IMPORTANT — do NOT overwrite the existing test file.** `model/tests/exercise_corpus/test_build_render_assets.py` already exists with three passing regression tests against the current `build()`:
- `test_build_emits_valid_mxl_with_matching_note_count`
- `test_build_is_idempotent`
- `test_build_raises_naming_bad_xml`

and `model/tests/exercise_corpus/__init__.py` already exists. These existing tests call `build(xml_dir=_XML_DIR, out_dir=out_dir)` with NO `manifest_path`, so they MUST stay manifest-side-effect-free (see Task 1's opt-in manifest contract in Step 3). The build agent APPENDS the new manifest test below the existing three; it does not touch them.

- [ ] **Step 1: Write the failing test (APPEND to the existing file)**

Append this test (and the two module-level constants it needs) to the END of the existing `model/tests/exercise_corpus/test_build_render_assets.py`. Add a `tmp_path` so the manifest and the rebuilt assets land in a temp dir — NEVER write into the committed `apps/api` manifest or the real `model/data/.../mxl` dir from a test:

```python
# ---- appended: manifest-emission behavior (Task 1) -------------------------
# build() emits the manifest ONLY when an explicit manifest_path is provided
# (default None => no write), so the three regression tests above stay
# manifest-side-effect-free. Bar counts (29/22/23) were verified empirically
# via partitura; assert against the manifest build() actually wrote.

# The 22 primitives that have committed .mxl assets on main.
_EXPECTED_IDS = {
    *(f"hanon_{i:03d}" for i in range(1, 21)),
    "czerny_001",
    "burgmuller_001",
}


def test_build_emits_manifest_for_exactly_the_built_primitives(tmp_path: Path):
    import json

    manifest_path = tmp_path / "manifest.json"
    # Explicit manifest_path + tmp out_dir: writes ONLY to tmp, never to the
    # committed apps/api manifest or the real model/data mxl dir.
    build(xml_dir=_XML_DIR, out_dir=tmp_path / "mxl", manifest_path=manifest_path)

    assert manifest_path.exists(), f"manifest not written at {manifest_path}"
    manifest = json.loads(manifest_path.read_text())

    assert set(manifest.keys()) == _EXPECTED_IDS

    # Spot-check the three distinct sources with verbatim keys and real bar counts.
    assert manifest["hanon_001"] == {
        "dimensions": ["articulation", "timing"],
        "key": "C",
        "totalBars": 29,
    }
    assert manifest["czerny_001"] == {
        "dimensions": ["timing", "articulation"],
        "key": "c",  # VERBATIM lowercase from technique_tags.toml (not normalized)
        "totalBars": 22,
    }
    assert manifest["burgmuller_001"] == {
        "dimensions": ["phrasing", "dynamics", "interpretation"],
        "key": "C",
        "totalBars": 23,
    }
```

The existing file already imports `Path`, `glob`, and `build`, and defines `_XML_DIR = Path(__file__).resolve().parents[2] / "data" / "scores" / "exercise_primitives"`. Reuse `_XML_DIR`; do NOT redefine it. The `dimensions`/`key`/`totalBars` literals above are the build agent's expectations — if `technique_tags.toml` orders `dimensions` differently or partitura reports a different `totalBars`, fix the ASSERTION to the manifest's actual emitted values (the bar counts 29/22/23 were verified empirically with partitura at plan time, but confirm at implementation time and pin the manifest's real values).

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run python -m pytest tests/exercise_corpus/test_build_render_assets.py -q
```
Expected: FAIL — the new test fails because `build()` does not yet accept a `manifest_path` argument (`TypeError: build() got an unexpected keyword argument 'manifest_path'`), OR (once the param is added but unwired) the manifest file is not written (`AssertionError: manifest not written at .../manifest.json`). The three pre-existing regression tests MUST still pass.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add manifest emission to `build()` in `model/src/exercise_corpus/build_render_assets.py`. Add these imports at the top (after the existing imports):

```python
import json
import tomllib
```

Add a module-level anchor after the existing `_DEFAULT_OUT_DIR`. Do NOT add a `_DEFAULT_MANIFEST` default — the manifest write is OPT-IN (default `manifest_path=None` => no write) so existing callers stay side-effect-free. The canonical committed-manifest path lives in the `just build-exercise-assets` recipe, not in the function default:

```python
_TECHNIQUE_TAGS = Path(__file__).resolve().parent / "technique_tags.toml"
```

Add a helper above `build()`:

```python
def _count_bars(xml_path: Path) -> int:
    """Number of measures in the primitive's first part (partitura-parsed)."""
    score = partitura.load_score(str(xml_path))
    part = score.parts[0]
    return len(list(part.iter_all(partitura.score.Measure)))
```

In `build()`, accept a `manifest_path` parameter and accumulate manifest entries inside the existing loop, then write the manifest after the loop. Change the signature and body:

```python
def build(
    xml_dir: Path = _DEFAULT_XML_DIR,
    out_dir: Path = _DEFAULT_OUT_DIR,
    manifest_path: Path | None = None,
) -> list[Path]:
    xml_dir = Path(xml_dir)
    out_dir = Path(out_dir)
    xml_files = sorted(glob.glob(str(xml_dir / "*.xml")))
    if not xml_files:
        raise FileNotFoundError(f"No primitive .xml files found in {xml_dir}")

    with open(_TECHNIQUE_TAGS, "rb") as f:
        tags = tomllib.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    manifest: dict[str, dict] = {}

    for xml_path_str in xml_files:
        xml_path = Path(xml_path_str)
        primitive_id = xml_path.stem

        try:
            partitura.load_score(str(xml_path))
        except Exception as e:  # noqa: BLE001 — re-raise with the offending file named
            raise ValueError(f"primitive .xml failed partitura load: {xml_path} ({e})") from e

        raw_xml = xml_path.read_bytes()
        mxl_path = out_dir / f"{primitive_id}.mxl"

        if _existing_inner_xml(mxl_path) == _strip_doctype(raw_xml):
            produced.append(mxl_path)
        else:
            mxl_bytes = wrap_as_mxl_zip(raw_xml, primitive_id)
            mxl_path.write_bytes(mxl_bytes)
            produced.append(mxl_path)

        # Manifest entry for this built primitive (restricted to built assets by
        # construction: we only loop over committed .xml that produce an .mxl).
        if primitive_id not in tags:
            raise ValueError(
                f"primitive {primitive_id!r} has a built .xml but no technique_tags.toml entry"
            )
        entry = tags[primitive_id]
        manifest[primitive_id] = {
            "dimensions": entry["dimensions"],
            "key": entry["key"],
            "totalBars": _count_bars(xml_path),
        }

    # OPT-IN manifest write: only when an explicit manifest_path is supplied.
    # Default (None) => no write, so the existing regression tests that call
    # build(xml_dir=..., out_dir=tmp) NEVER touch the committed apps/api manifest.
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return produced
```

(Keep the existing module docstring; only the import block, the new `_TECHNIQUE_TAGS` anchor, `_count_bars`, and `build()` change. The manifest dict is always built, but written only when `manifest_path` is provided.)

Also update the `just build-exercise-assets` recipe to pass an explicit `manifest_path` so it (and ONLY it) regenerates the committed runtime manifest. Replace the recipe body in `justfile` (recipe `build-exercise-assets`, currently `cd model && uv run python -c "from exercise_corpus.build_render_assets import build; print(f'built {len(build())} assets')"`) with one that passes the real path:

```make
build-exercise-assets:
    cd model && uv run python -c "from pathlib import Path; from exercise_corpus.build_render_assets import build; mp = Path('..') / 'apps' / 'api' / 'src' / 'services' / 'exercise_primitives_manifest.json'; print(f'built {len(build(manifest_path=mp))} assets + manifest at {mp}')"
```

This is the canonical regeneration of the committed `apps/api/src/services/exercise_primitives_manifest.json`. No test invocation writes into that committed path (the manifest test passes an explicit tmp `manifest_path`; the three regression tests pass no `manifest_path` at all).

- [ ] **Step 4: Run the test suite AND regenerate the committed manifest**

```bash
# Test suite: all 4 tests pass; NONE writes into committed paths (manifest test
# is tmp-scoped, the 3 regression tests pass no manifest_path).
cd model && uv run python -m pytest tests/exercise_corpus/test_build_render_assets.py -q && cd ..
# Regenerate the committed runtime manifest (the ONLY callsite with an explicit
# manifest_path). Task 3 imports this committed file.
just build-exercise-assets
# Confirm the committed manifest now exists and has exactly 22 entries.
cd model && uv run python -c "import json; m=json.load(open('../apps/api/src/services/exercise_primitives_manifest.json')); assert len(m)==22, len(m); print('manifest OK:', len(m), 'entries')" && cd ..
```
Expected: PASS (4 tests) and a committed 22-entry manifest at `apps/api/src/services/exercise_primitives_manifest.json`.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add model/src/exercise_corpus/build_render_assets.py model/tests/exercise_corpus/test_build_render_assets.py justfile apps/api/src/services/exercise_primitives_manifest.json
# build-exercise-assets is deterministic+idempotent, so it should NOT have changed
# any committed .mxl bytes; if `git status` shows mxl churn, investigate before adding.
git commit -m "feat(exercise-corpus): emit exercise_primitives_manifest (opt-in) from build script"
```

---

### Task 2: TS port of keys.py (parse_key_to_pc + transpose_interval) with parity fixture
**Group:** A (parallel with Task 1)

**Behavior being verified:** The TS key-port matches the Python oracle byte-for-byte on the oracle's accepted (uppercase) domain, AND resolves lowercase-minor tags the oracle rejects (documented superset).
**Interface under test:** exported `parseKeyToPc(key: string): number | null` and `transposeInterval(fromPc: number, toPc: number): number`.

**Files:**
- Create: `apps/api/src/services/keys.ts`
- Create: `apps/api/src/services/keys-parity-fixture.json` (golden, generated from the live oracle)
- Test: `apps/api/src/services/keys.test.ts`

- [ ] **Step 1: Write the failing test**

First generate the committed parity fixture by running the REAL Python oracle (this is the golden the TS port must match — generate it, do not hand-write it):

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime/model
uv run python -c "
import sys, json; sys.path.insert(0,'src')
from exercise_corpus.keys import parse_key_to_pc, transpose_interval
# Oracle-accepted (uppercase) domain only — the fixture is the parity ground truth.
keys = ['C','C#','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B','C major','Am','C#m','F#']
pc = {k: parse_key_to_pc(k) for k in keys}
intervals = [
    {'from':'C','to':'C','expected':transpose_interval(0,0)},
    {'from':'C','to':'D','expected':transpose_interval(0,2)},
    {'from':'C','to':'A','expected':transpose_interval(0,9)},
    {'from':'C','to':'F#','expected':transpose_interval(0,6)},
    {'from':'Bb','to':'C','expected':transpose_interval(10,0)},
    {'from':'E','to':'Eb','expected':transpose_interval(4,3)},
]
print(json.dumps({'pc':pc,'intervals':intervals}, indent=2, sort_keys=True))
" > ../apps/api/src/services/keys-parity-fixture.json
echo "fixture written"
```

Then the test:

```typescript
// apps/api/src/services/keys.test.ts
import { describe, expect, it, test } from "vitest";
import fixture from "./keys-parity-fixture.json";
import { parseKeyToPc, transposeInterval } from "./keys";

describe("keys TS port — parity with the Python oracle (uppercase domain)", () => {
  // The fixture is generated from model/src/exercise_corpus/keys.py. The TS port
  // must match it byte-for-byte for every input the oracle accepts.
  for (const [key, pc] of Object.entries(fixture.pc as Record<string, number>)) {
    it(`parseKeyToPc(${JSON.stringify(key)}) === ${pc} (oracle parity)`, () => {
      expect(parseKeyToPc(key)).toBe(pc);
    });
  }

  for (const { from, to, expected } of fixture.intervals as Array<{
    from: string;
    to: string;
    expected: number;
  }>) {
    it(`transposeInterval(${from} -> ${to}) === ${expected} (oracle parity)`, () => {
      const f = parseKeyToPc(from);
      const t = parseKeyToPc(to);
      expect(f).not.toBeNull();
      expect(t).not.toBeNull();
      expect(transposeInterval(f as number, t as number)).toBe(expected);
    });
  }
});

describe("keys TS port — intentional supersets (oracle RAISES, TS resolves)", () => {
  // These diverge from the Python oracle BY DESIGN: keys.py._PC is uppercase-only
  // and raises on lowercase-minor tags, which is latently broken for its own corpus
  // (e.g. czerny_001 = "c"). The TS port capitalizes the first tonic char before
  // lookup, so it resolves the correct tonic pitch class. Mode is irrelevant to
  // semitone transposition.
  test('parseKeyToPc("c") === 0 (lowercase minor, oracle would raise)', () => {
    expect(parseKeyToPc("c")).toBe(0);
  });
  test('parseKeyToPc("eb") === 3 (oracle would raise)', () => {
    expect(parseKeyToPc("eb")).toBe(3);
  });
  test('parseKeyToPc("f#") === 6 (oracle would raise)', () => {
    expect(parseKeyToPc("f#")).toBe(6);
  });
  test("parseKeyToPc returns null on genuine garbage", () => {
    expect(parseKeyToPc("zzz")).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/keys.test.ts
```
Expected: FAIL — `Cannot find module './keys'` / `parseKeyToPc is not a function` (the module does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/keys.ts
// Best-effort TS port of model/src/exercise_corpus/keys.py.
//
// parseKeyToPc is a strict SUPERSET of the Python parse_key_to_pc: identical
// results for every input the oracle accepts (uppercase tonics), plus correct
// tonic extraction for lowercase-minor tags the oracle rejects (it raises). The
// added normalization is the single first-char capitalization in step 3 below;
// _PC, the mode-stripping, and transposeInterval are byte-for-byte the oracle.
// Returns null (never throws) so the caller can degrade to transpose=0 + warn.

// _PC replicated verbatim from keys.py (uppercase tonic entries only).
const PC: Record<string, number> = {
  C: 0,
  "C#": 1, Db: 1,
  D: 2,
  "D#": 3, Eb: 3,
  E: 4,
  F: 5,
  "F#": 6, Gb: 6,
  G: 7,
  "G#": 8, Ab: 8,
  A: 9,
  "A#": 10, Bb: 10,
  B: 11,
};

export function parseKeyToPc(keySignature: string): number | null {
  let s = keySignature.trim();
  // Strip trailing " major" / " minor" (case-insensitive), like the oracle.
  for (const suffix of [" major", " minor"]) {
    if (s.toLowerCase().endsWith(suffix)) {
      s = s.slice(0, -suffix.length).trim();
      break;
    }
  }
  // Strip trailing "m" (minor shorthand) — only if not the entire string.
  if (s.endsWith("m") && s.length > 1) {
    s = s.slice(0, -1);
  }
  // Superset normalization the oracle lacks: capitalize ONLY the first char of
  // the remaining tonic so "c"->"C", "eb"->"Eb", "f#"->"F#", "C#"->"C#".
  if (s.length > 0) {
    s = s[0].toUpperCase() + s.slice(1);
  }
  return s in PC ? PC[s] : null;
}

export function transposeInterval(fromPc: number, toPc: number): number {
  // Nearest-octave semitone shift, range [-5, +6], tritone resolves to +6.
  let d = ((toPc - fromPc) % 12 + 12) % 12;
  if (d > 6) d -= 12;
  return d;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/keys.test.ts
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/api/src/services/keys.ts apps/api/src/services/keys.test.ts apps/api/src/services/keys-parity-fixture.json
git commit -m "feat(exercise-corpus): TS keys port (parse_key_to_pc superset + transpose_interval) with oracle parity fixture"
```

---

### Task 3: buildCorpusDrillClip — primitive selection, widening, and clip assembly (transpose=0)
**Group:** B (sequential; depends on Group A's committed manifest + keys port)

**Behavior being verified:** Given a corpus_drill decision, `buildCorpusDrillClip` selects the right primitive (explicit id -> dimension-matched stable-first -> widened hanon_001), assembles a `scoreClip` of the WHOLE primitive (`[1, totalBars]`) carrying `tempoFactor`, and writes honest instruction text. (Transpose is added in Task 4; here it is 0 because `pieceId` is null.)
**Interface under test:** `buildCorpusDrillClip(ctx, decision, pieceId)`.

**Files:**
- Create: `apps/api/src/services/corpus-drill.ts`
- Test: `apps/api/src/services/corpus-drill.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/corpus-drill.test.ts
import { describe, expect, test } from "vitest";
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
    // is czerny_001, NOT hanon_001. (No invented source-priority tiebreak; we
    // mirror the model, which has none.) czerny_001.totalBars == 22 (empirically
    // verified; build agent asserts against the manifest's actual totalBars).
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
    // The widen fallback is the EXPLICIT WIDEN_DEFAULT_PRIMITIVE constant ("hanon_001"),
    // a deliberate product choice (a neutral Hanon finger warm-up) — NOT whatever
    // sorts first globally (global stable-first is actually burgmuller_001).
    // hanon_001.totalBars == 29 (empirically verified; assert against the manifest).
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/corpus-drill.test.ts
```
Expected: FAIL — `Cannot find module './corpus-drill'` / `buildCorpusDrillClip is not a function`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/corpus-drill.ts
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
  // Implemented in Task 4. For now (Task 3) every reachable test passes a null
  // pieceId, so the best-effort branch returns 0. Task 4 adds the R2 read.
  if (pieceId === null) return 0;
  return 0;
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
    ? `General warm-up drill (no ${dim}-specific drill in corpus yet). ` +
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
```

NOTE: this requires `ExerciseSetPayload.scoreClip` to allow `transpose?: number`. Task 5 adds that field to the type. To keep Task 3 green standalone, add the field to `ExerciseSetPayload` in `apps/api/src/services/exercises.ts` AS PART OF THIS TASK (one-line type widening, regression-safe since it's optional). This is the only edit Task 3 makes to `exercises.ts`:

```typescript
// apps/api/src/services/exercises.ts — widen the scoreClip type (line ~15)
scoreClip?: { pieceId: string; bars: [number, number]; tempoFactor?: number; transpose?: number };
```

Also ensure `apps/api/tsconfig.json` allows JSON imports (`resolveJsonModule`). If it is not already enabled, the build agent verifies via `cd apps/api && bunx tsc --noEmit` in Step 4; if tsc reports the JSON import error, set `"resolveJsonModule": true` in `apps/api/tsconfig.json`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/corpus-drill.test.ts && bunx tsc --noEmit
```
Expected: PASS (4 tests) and clean typecheck.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/api/src/services/corpus-drill.ts apps/api/src/services/corpus-drill.test.ts apps/api/src/services/exercises.ts
git commit -m "feat(corpus-drill): buildCorpusDrillClip primitive selection + whole-primitive clip assembly"
```

---

### Task 4: resolveTranspose — load passage key from R2 and compute the interval
**Group:** B (sequential; depends on Task 3 — same file `corpus-drill.ts`)

**Behavior being verified:** When a `pieceId` is supplied and `scores/v1/{pieceId}.json` has a parseable `key_signature`, the assembled `scoreClip.transpose` is `interval(primitiveKey -> passageKey)`; genuinely-unresolvable cases (404, null/garbage key) degrade to `transpose = 0` with a structured warn.
**Interface under test:** `buildCorpusDrillClip(ctx, decision, pieceId)` with a fake `ctx.env.SCORES`.

**Files:**
- Modify: `apps/api/src/services/corpus-drill.ts`
- Test: `apps/api/src/services/corpus-drill.test.ts` (append a describe block)

- [ ] **Step 1: Write the failing test**

```typescript
// append to apps/api/src/services/corpus-drill.test.ts
import { vi } from "vitest";

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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/corpus-drill.test.ts
```
Expected: FAIL — the new transpose cases get `0` for all of them (the Task 3 stub `resolveTranspose` always returns 0 for non-null pieceId too), so `transpose === 2`, `-3`, `+6` assertions FAIL.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the stub `resolveTranspose` in `corpus-drill.ts` with the real R2-reading implementation. (Do NOT change `buildCorpusDrillClip` or `selectPrimitive`.)

```typescript
async function resolveTranspose(
  ctx: ServiceContext,
  pieceId: string | null,
  primitiveKey: string,
): Promise<number> {
  // Best-effort: any unresolvable input -> 0 + a structured warn (NOT a silent
  // catch{return null} like loadScoreContext). The drill still renders, untransposed.
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
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/corpus-drill.test.ts && bunx tsc --noEmit
```
Expected: PASS (all Task 3 + Task 4 cases) and clean typecheck.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/api/src/services/corpus-drill.ts apps/api/src/services/corpus-drill.test.ts
git commit -m "feat(corpus-drill): resolveTranspose loads passage key from R2 and computes interval"
```

---

### Task 5: Wire assignPendingExercise to buildCorpusDrillClip + FLIP the corpus_drill test
**Group:** C (sequential; depends on Group B)

**Behavior being verified:** `assignPendingExercise` corpus_drill path now returns a real `scoreClip` (pieceId = primitive, bars = `[1, N]`, `transpose` present) instead of a text stub.
**Interface under test:** `assignPendingExercise(ctx, args)`.

**Files:**
- Modify: `apps/api/src/services/exercises.ts`
- Modify: `apps/api/src/services/exercises.test.ts` (FLIP the corpus_drill test at lines 222-238)

- [ ] **Step 1: Write the failing test (FLIP the existing assertion)**

Replace the existing `test("corpus_drill produces text stub, no scoreClip", ...)` (exercises.test.ts lines ~222-238) with:

```typescript
	test("corpus_drill now produces a scoreClip of a matched primitive (no longer a text stub)", async () => {
		const mockCtx = buildMockCtxWithPendingRow({
			routingJson: CORPUS_DRILL_ROUTING, // target_dimension: "timing"
			focusDimension: "timing",
			previewTitle: "Timing drill",
			title: "Timing corpus drill",
			instruction: "Timing drill — bars 1-8",
			pieceId: "chopin.ballade.1",
		});
		const payload = await assignPendingExercise(mockCtx, {
			studentId: "stu-1",
			sessionId: "sess-1",
			exerciseId: "pending-row-id",
		});
		// timing matches hanon_001..020 + czerny_001. Under the FAITHFUL sort
		// (source_exercise_number, primitive_id) == (suffixNum(id), id), all sources
		// have a suffix-1 member so the id tiebreak decides: "czerny_001" < "hanon_001".
		// Stable-first for timing is therefore czerny_001, NOT hanon_001.
		// czerny_001.totalBars == 22 (empirically verified via partitura;
		// build agent: assert against the manifest's actual totalBars, not on faith).
		expect(payload.scoreClip?.pieceId).toBe("czerny_001");
		expect(payload.scoreClip?.bars).toEqual([1, 22]);
		expect(payload.scoreClip).toHaveProperty("transpose");
		expect(typeof payload.scoreClip?.transpose).toBe("number");
	});
```

This flip also requires `buildMockCtxWithPendingRow` to give `ctx.env.SCORES.get` a callable. Update its `env: {}` to `env: { SCORES: { get: async () => null } }` (the corpus_drill path calls `resolveTranspose`, which reads `ctx.env.SCORES`; `get -> null` makes it a benign 404 -> transpose 0). Change line ~307-308:

```typescript
	return {
		db: mockDb as unknown as import("../lib/types").ServiceContext["db"],
		env: { SCORES: { get: async () => null } },
	} as unknown as import("../lib/types").ServiceContext;
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/exercises.test.ts
```
Expected: FAIL — `assignPendingExercise` still emits the text stub, so `payload.scoreClip` is `undefined` (`expected undefined to have property "pieceId"`).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/exercises.ts`, add the import and replace the corpus_drill text-stub block (lines ~254-272) with a delegation to `buildCorpusDrillClip`. Add at the top with the other imports:

```typescript
import { buildCorpusDrillClip } from "./corpus-drill";
```

Replace the trailing corpus_drill block (everything after the `if (routing.kind === "own_passage_loop") { ... }` block, i.e. lines ~254-272) with:

```typescript
	// corpus_drill — render a matched primitive clip (transposed into the
	// student's key when resolvable). pieceId is the STUDENT passage piece used
	// only for key resolution; the clip itself is the primitive.
	return buildCorpusDrillClip(ctx, routing, pendingRow.pieceId ?? null);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/exercises.test.ts && bunx tsc --noEmit
```
Expected: PASS (all exercises.test.ts cases, with the flipped corpus_drill assertion green).

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/api/src/services/exercises.ts apps/api/src/services/exercises.test.ts
git commit -m "feat(corpus-drill): wire assignPendingExercise to buildCorpusDrillClip"
```

---

### Task 6: Wire processPrescribeExercise (chat tool) to buildCorpusDrillClip
**Group:** C (sequential; after Task 5)

**Behavior being verified:** The mid-chat `prescribe_exercise` tool's corpus_drill branch returns an `exercise_set` whose config carries a real `scoreClip` (pieceId = primitive, bars = `[1, N]`, `transpose`) instead of a text stub.
**Interface under test:** `processToolUse(ctx, studentId, "prescribe_exercise", input)` — the public tool dispatch entry point (returns a `ToolResult` with `componentsJson`). This is the SAME public interface `tool-processor.test.ts` already uses (`processToolUse`, `TOOL_REGISTRY`).

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Test: `apps/api/src/services/tool-processor.corpus-drill.test.ts` (create)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/tool-processor.corpus-drill.test.ts
import { describe, expect, test } from "vitest";
import { processToolUse } from "./tool-processor";

// A ServiceContext whose SCORES.get always 404s, so resolveTranspose -> 0.
function ctxNoScores() {
  return { db: {} as never, env: { SCORES: { get: async () => null } } as never };
}

describe("prescribe_exercise — corpus_drill", () => {
  test("corpus_drill returns an exercise_set with a primitive scoreClip", async () => {
    const result = await processToolUse(ctxNoScores(), "stu-1", "prescribe_exercise", {
      kind: "corpus_drill",
      target_dimension: "phrasing",
      bar_range: [4, 8],
      tempo_factor: 0.8,
      primitive_id: null,
      piece_id: null,
    });
    expect(result.isError).toBe(false);
    expect(result.componentsJson).toHaveLength(1);
    expect(result.componentsJson[0].type).toBe("exercise_set");
    const cfg = result.componentsJson[0].config as {
      scoreClip?: { pieceId: string; bars: [number, number]; transpose?: number };
    };
    expect(cfg.scoreClip?.pieceId).toBe("burgmuller_001"); // only phrasing match
    expect(cfg.scoreClip?.bars).toEqual([1, 23]);
    expect(cfg.scoreClip).toHaveProperty("transpose");
  });
});
```

This exercises the full public path: `processToolUse` looks up `TOOL_REGISTRY["prescribe_exercise"]`, runs its Zod `safeParse`, and calls `process(ctx, studentId, validatedInput)` — no internal mocking, the real tool runs end to end.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/tool-processor.corpus-drill.test.ts
```
Expected: FAIL — corpus_drill currently returns the text stub with NO `scoreClip`, so `cfg.scoreClip` is `undefined` (`expected undefined to have property "pieceId"`).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`:

(a) Add the import with the others at the top:

```typescript
import { buildCorpusDrillClip } from "./corpus-drill";
import { CorpusDrillSchema } from "../harness/artifacts/exercise-routing";
```

(b) Change `processPrescribeExercise`'s signature to use the real `ctx` (it currently ignores `_ctx`): rename `_ctx` to `ctx`.

(c) Replace the corpus_drill text-stub block (lines ~116-137) with delegation to `buildCorpusDrillClip`. The tool's input has `primitive_id`, `piece_id`, `target_dimension`, `bar_range`, `tempo_factor`. Build a `CorpusDrillDecision` from the tool input (the tool schema is a superset of the routing schema — it adds `piece_id`), then call the shared helper and wrap its `ExerciseSetPayload` into the `exercise_set` InlineComponent:

```typescript
	// corpus_drill — render a matched primitive clip via the shared helper.
	const decision = CorpusDrillSchema.parse({
		kind: "corpus_drill",
		target_dimension: input.target_dimension,
		bar_range: input.bar_range,
		tempo_factor: input.tempo_factor,
		primitive_id: input.primitive_id,
	});
	const payload = await buildCorpusDrillClip(ctx, decision, input.piece_id);
	return [
		{
			type: "exercise_set",
			config: {
				sourcePassage: payload.sourcePassage,
				targetSkill: payload.targetSkill,
				scoreClip: payload.scoreClip,
				exercises: payload.exercises,
			},
		},
	];
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/tool-processor.corpus-drill.test.ts && bunx tsc --noEmit
```
Expected: PASS and clean typecheck.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.corpus-drill.test.ts
git commit -m "feat(corpus-drill): wire prescribe_exercise corpus_drill to buildCorpusDrillClip"
```

---

### Task 7: Web — thread scoreClip.transpose through ExerciseSetCard
**Group:** D (depends on Group C for the contract type; independent web files)

**Behavior being verified:** When `config.scoreClip.transpose` is set, `ExerciseSetCard` passes it to `load`, `getClipPlayback`, and `getClip`, so the rendered clip is transposed.
**Interface under test:** `ExerciseSetCard` render -> the `scoreRenderer` calls it makes (verified via the existing mocked-renderer test convention in `ExerciseSetCard.test.tsx`).

**Files:**
- Modify: `apps/web/src/lib/types.ts`
- Modify: `apps/web/src/components/cards/ExerciseSetCard.tsx`
- Test: `apps/web/src/components/cards/ExerciseSetCard.test.tsx` (add a case)

- [ ] **Step 1: Write the failing test**

READ `apps/web/src/components/cards/ExerciseSetCard.test.tsx` first to match its existing mock convention (it mocks `getClip`/`getClipPlayback` via `mockGetClip`/`mockGetClipPlayback`, each proxied through `(...args) => mock(...args)`).

**Important — existing assertions break on arity.** The existing tests assert 3-arg calls, e.g. `expect(mockGetClip).toHaveBeenCalledWith("chopin.ballades.1", 5, 8)` (lines ~61, 86), `expect(mockGetClipPlayback).toHaveBeenCalledWith("chopin.ballades.1", 5, 8)` (line ~186), `toHaveBeenCalledWith("bach.wtc.1", 1, 4)` (line ~223). Because the mocks proxy `(...args)`, once the component passes a trailing transpose arg those `toHaveBeenCalledWith` calls fail on arity (an explicit 4th arg is NOT equal to an absent one). To keep own_passage byte-identical at the renderer boundary, the component passes `transpose ?? 0` (always a number, never `undefined`), and these existing own_passage assertions are updated to include the trailing `0`. Update them as part of this task:
- `toHaveBeenCalledWith("chopin.ballades.1", 5, 8)` -> `toHaveBeenCalledWith("chopin.ballades.1", 5, 8, 0)` (both getClip cases)
- `mockGetClipPlayback ... toHaveBeenCalledWith("chopin.ballades.1", 5, 8)` -> `..., 5, 8, 0)`
- `toHaveBeenCalledWith("bach.wtc.1", 1, 4)` -> `toHaveBeenCalledWith("bach.wtc.1", 1, 4, 0)`
- if any existing case asserts `load` (via a `mockLoad`), add the trailing `0` there too.

(If the component instead destructures with a default `transpose = 0`, the same trailing-`0` assertions hold. Do NOT pass bare `undefined`; that would force the 3-arg assertions back and re-break this task. Pick `transpose ?? 0` and keep every call 4-arg.)

Then ADD a case asserting the non-zero transpose is forwarded:

```typescript
	it("forwards scoreClip.transpose to load and getClipPlayback", async () => {
		mockGetClipPlayback.mockResolvedValue({ svg: "<svg/>", ir: { bars: [], pageWidth: 1600 }, notes: [] });
		const config = {
			sourcePassage: "bars 1-29",
			targetSkill: "timing",
			scoreClip: { pieceId: "hanon_001", bars: [1, 29] as [number, number], tempoFactor: 0.8, transpose: 2 },
			exercises: [{ title: "t", instruction: "i", focusDimension: "timing" }],
		};
		render(<ExerciseSetCard config={config} />);
		await waitFor(() => expect(mockGetClipPlayback).toHaveBeenCalled());
		// load(pieceId, transpose) and getClipPlayback(pieceId, b0, b1, transpose).
		expect(mockLoad).toHaveBeenCalledWith("hanon_001", 2);
		expect(mockGetClipPlayback).toHaveBeenCalledWith("hanon_001", 1, 29, 2);
	});
```

(If the test file does not already spy on `load` via `mockLoad`, add a `mockLoad` to the existing `vi.mock` of `score-renderer` mirroring the existing `mockGetClipPlayback` pattern — match the file's convention; do not introduce a different mocking style. Keep `mockLoad` resolving a non-`"failed"` value so the clip path does not short-circuit.)

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bunx vitest run src/components/cards/ExerciseSetCard.test.tsx
```
Expected: FAIL — the new case fails because `mockLoad`/`mockGetClipPlayback` are called WITHOUT the transpose arg (the component currently calls `load(pieceId)` and `getClipPlayback(pieceId, bars[0], bars[1])`), so `toHaveBeenCalledWith(..., 2)` fails. (The existing own_passage assertions, updated to expect a trailing `0`, will also fail until Step 3 makes the component pass `transpose ?? 0`.)

- [ ] **Step 3: Implement the minimum to make the test pass**

(a) `apps/web/src/lib/types.ts` line 42 — widen the scoreClip type:

```typescript
	scoreClip?: { pieceId: string; bars: [number, number]; tempoFactor?: number; transpose?: number };
```

(b) `apps/web/src/components/cards/ExerciseSetCard.tsx` clip-loading effect (lines ~160-194):

- Destructure transpose with a 0 default: change `const { pieceId, bars } = config.scoreClip;` to `const { pieceId, bars, transpose = 0 } = config.scoreClip;` (so own_passage, which omits transpose, passes a literal `0` — byte-identical to today since `transpose:0` is a documented Verovio/renderer no-op and the renderer caches by `${pieceId}:${transpose ?? 0}`).
- `scoreRenderer.load(pieceId)` -> `scoreRenderer.load(pieceId, transpose)`
- `scoreRenderer.getClipPlayback(pieceId, bars[0], bars[1])` -> `scoreRenderer.getClipPlayback(pieceId, bars[0], bars[1], transpose)`
- `scoreRenderer.getClip(pieceId, bars[0], bars[1])` -> `scoreRenderer.getClip(pieceId, bars[0], bars[1], transpose)`

Add `config.scoreClip?.transpose` to the effect's dependency array if `pieceId`/`bars` are already dependencies (match the existing dep-array shape; do not restructure the effect).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bunx vitest run src/components/cards/ExerciseSetCard.test.tsx
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/web/src/lib/types.ts apps/web/src/components/cards/ExerciseSetCard.tsx apps/web/src/components/cards/ExerciseSetCard.test.tsx
git commit -m "feat(corpus-drill): thread scoreClip.transpose through ExerciseSetCard"
```

---

### Task 8: Web — transposed playback-MIDI correctness
**Group:** D (parallel with Task 7 — different files)

**Behavior being verified:** `processGetClipPlaybackRequest` against a toolkit loaded WITH a transpose yields playback notes shifted by that transpose relative to an untransposed load of the same bytes — i.e. smplr plays the transposed pitches that match the displayed notation.
**Interface under test:** `loadPiece(bytes, bindings, pieceId, transpose)` + `processGetClipPlaybackRequest(tk, measures, b0, b1)` from `score-worker.ts`, exercised with the real Verovio WASM toolkit.

**Files:**
- Test: `apps/web/src/lib/score-worker.transpose.test.ts` (create)

- [ ] **Step 1: Write the failing test**

READ `apps/web/src/lib/score-worker.ts` and any existing `score-worker.*.test.ts` first to reuse the real-Verovio bootstrapping convention (importing `verovio/wasm` + `verovio/esm`, building `VerovioBindings`) and a committed `.mxl` fixture (e.g. one of the exercise primitives or an existing test fixture). The test loads the SAME bytes twice — once at transpose 0, once at +2 — and asserts every clip-playback note's MIDI is exactly +2.

```typescript
// apps/web/src/lib/score-worker.transpose.test.ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadPiece, processGetClipPlaybackRequest, type VerovioBindings } from "./score-worker";

async function bindings(): Promise<VerovioBindings> {
  const [wasm, esm] = await Promise.all([
    import("verovio/wasm") as Promise<{ default: () => Promise<unknown> }>,
    import("verovio/esm") as Promise<{ VerovioToolkit: new (m: unknown) => unknown }>,
  ]);
  const module = await wasm.default();
  return { module, ToolkitClass: esm.VerovioToolkit as VerovioBindings["ToolkitClass"] };
}

// Committed primitive asset wrapped as .mxl. build-exercise-assets produces these
// under model/data/exercise_primitives/mxl/. Pick hanon_001 (whole-piece clip).
function fixtureBytes(): ArrayBuffer {
  const p = resolve(__dirname, "../../../../model/data/exercise_primitives/mxl/hanon_001.mxl");
  const buf = readFileSync(p);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

describe("score-worker transpose — playback MIDI matches transposed notation", () => {
  it("transposed load shifts clip playback MIDI by the transpose amount", async () => {
    const b = await bindings();
    const bytes = fixtureBytes();

    const base = await loadPiece(bytes.slice(0), b, "hanon_001", 0);
    const shifted = await loadPiece(bytes.slice(0), b, "hanon_001", 2);
    if (base === "failed" || shifted === "failed") throw new Error("load failed");

    const baseClip = await processGetClipPlaybackRequest(base.tk, base.measures, 1, 2);
    const shiftedClip = await processGetClipPlaybackRequest(shifted.tk, shifted.measures, 1, 2);
    if (baseClip === "failed" || shiftedClip === "failed") throw new Error("clip failed");

    expect(baseClip.notes.length).toBeGreaterThan(0);
    expect(shiftedClip.notes.length).toBe(baseClip.notes.length);
    for (let i = 0; i < baseClip.notes.length; i++) {
      expect(shiftedClip.notes[i].midi).toBe(baseClip.notes[i].midi + 2);
    }
  }, 30000);
});
```

- [ ] **Step 2: Run test — verify it FAILS first for the right reason, then confirm it is a real check**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.test.ts
```
Expected behavior to confirm: this test should PASS if and only if the worker already extracts playback notes AFTER the transposed load (it does — `loadPiece` applies `transpose` at `loadData` time and `getMIDIValuesForElement` reads the transposed toolkit). To prove the test is a REAL correctness check and not vacuous, the build agent must first run it with the transpose argument removed from the `shifted` load (call `loadPiece(bytes.slice(0), b, "hanon_001", 0)` for both) and observe it FAIL (`expected X to be X+2`). Restore the +2 transpose and confirm it PASSES. Document this watch-it-fail step in the commit body.

If the test FAILS with the +2 transpose in place, that is a genuine worker bug (playback extracted before transpose) and must be fixed in `score-worker.ts` before committing — but per the spec the worker already does this correctly, so the expected end state is PASS.

- [ ] **Step 3: Implement**

No production code change is expected (the worker already extracts MIDI from the transposed toolkit). This task's deliverable is the regression test that LOCKS that correctness. If Step 2's in-place run fails, fix `processGetClipPlaybackRequest`/`loadPiece` so playback MIDI is read from the transposed toolkit, then re-run.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bunx vitest run src/lib/score-worker.transpose.test.ts
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
git add apps/web/src/lib/score-worker.transpose.test.ts
git commit -m "test(corpus-drill): lock transposed playback-MIDI correctness in score-worker"
```

---

## Post-plan verification (build agent runs after all groups)

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-47-corpus-drill-runtime
# API: full suite green, typecheck clean
cd apps/api && bun run test && bunx tsc --noEmit && cd ../..
# Web: card + worker transpose tests green
cd apps/web && bunx vitest run src/components/cards/ExerciseSetCard.test.tsx src/lib/score-worker.transpose.test.ts && cd ../..
# Model: manifest test green
cd model && uv run python -m pytest tests/exercise_corpus/test_build_render_assets.py -q && cd ..
# Manual click-through: just dev-light, prescribe a corpus_drill in chat, confirm the
# ExerciseSetCard renders a transposed primitive clip with working play/tempo transport.
```

## Spec coverage map

- Manifest (single source of truth, 22-asset filter, verbatim key) -> Task 1.
- keys.py superset port + oracle parity fixture -> Task 2.
- buildCorpusDrillClip selection (explicit / dimension / widen) + whole-primitive clip + honest wording -> Task 3.
- resolveTranspose (R2 load, interval, best-effort 0+warn) -> Task 4.
- `scoreClip.transpose` contract (API) + assignPendingExercise wiring + FLIP exercises.test.ts -> Task 5.
- prescribe_exercise (chat) wiring -> Task 6.
- `scoreClip.transpose` contract (web) + ExerciseSetCard threading -> Task 7.
- Transposed-playback correctness (smplr plays transposed pitches) -> Task 8.

---

## Challenge Review

Reviewed against: the spec, `build_render_assets.py`, `keys.py`, `briefing.py`, `match.py`, `technique_tags.toml`, `exercises.ts` + `exercises.test.ts`, `tool-processor.ts`, `exercise-routing.ts`, `ExerciseSetCard.tsx` + `.test.tsx`, `score-renderer.ts`, `score-worker.ts`, web `types.ts`, the on-disk asset/test layout, the justfile recipe, and `TS_STYLE.md`.

### CEO Pass

**Premise — sound.** corpus_drill is a verified dead end today: both `processPrescribeExercise` (tool-processor.ts:116-137) and `assignPendingExercise` (exercises.ts:254-272) emit a "coming soon" text stub with no `scoreClip`. The retrieval-based exercise system (slices A-C) is invisible at runtime. The most-direct path is a runtime decision layer reusing the already-built `own_passage_loop` renderer, which is exactly what the plan does. The renderer, worker, and renderer API ALREADY accept `transpose` as a trailing arg (`score-renderer.ts` load/getClip/getClipPlayback all take `transpose?`; `score-worker.ts` applies it at `loadData` time and reads `getMIDIValuesForElement` off the transposed toolkit) — so the plan correctly identifies that only the decision layer + threading are missing. High confidence this is the right build.

**Existing coverage / reuse — correctly leveraged.** `keys.py` (oracle), `briefing.py` (offline selection+transpose), and `technique_tags.toml` (dimensions+key) already encode the logic; the plan ports the runtime-relevant slice. Good.

**Scope — appropriately bounded.** Whole-primitive crop (`[1, totalBars]`), no within-bucket rank, 22-asset filter, no deploy. All explicitly scoped out in the spec and respected. No gold-plating.

**12-month alignment — neutral-to-positive.** The TS port of `keys.py` creates a second implementation of key-resolution that must stay in parity with Python. The parity fixture (Task 2) is the right mitigation. Acceptable debt for a Worker that cannot run Python.

[QUESTION] — The spec documents the superset-port trade-off (vs inheriting the oracle bug or back-porting to Python) but does NOT document why selection uses a fresh `(numeric-suffix, id)` stable sort rather than porting `match_by_dimension`'s actual tie-break (`match.py:116` = `(-cosine, primitive_id ascending)`). The chosen sort is defensible (rank is out of scope; deterministic is enough) but a future engineer reading `briefing.py` will expect the Python ordering. One sentence in the spec would prevent confusion.

### Engineering Pass

**Architecture — clean.** One deep async function `buildCorpusDrillClip(ctx, decision, pieceId)` hiding selection + key-resolution + transpose-math + assembly; both call sites delegate. Data flow: decision -> selectPrimitive (manifest lookup) -> resolveTranspose (R2 read, best-effort) -> assemble `ExerciseSetPayload`. Error path on transpose is an explicit `console.log(JSON.stringify(...))` warn + `transpose=0`, not a swallowed catch — matches CLAUDE.md "explicit exceptions over silent fallbacks". `ctx.env.SCORES.get(...)` is accessed as a passed-in argument (ServiceContext), never destructured — TS_STYLE compliant.

**Module depth — DEEP.** `corpus-drill.ts` exposes one function over ~90 LOC of hidden logic. `keys.ts` exposes two pure functions. Both pass.

#### BLOCKERS

[BLOCKER] (confidence: 10/10) — **Selection-algorithm vs asserted-outcome contradiction (timing).** The plan's `selectPrimitive` sorts dimension matches by `(suffixNum(id), id)`. For `target_dimension: "timing"` the built matches are `hanon_001..020` (tagged `["articulation","timing"]`) PLUS `czerny_001` (tagged `["timing","articulation"]`). All three sources have suffix-1 members, so the id tiebreak decides: `"czerny_001" < "hanon_001"`. Verified by executing the plan's own sort: timing stable-first = **`czerny_001`** (bars `[1,22]`), NOT `hanon_001`. This breaks:
  - Task 5 Step 1 flipped test (lines 818-819): asserts `scoreClip.pieceId === "hanon_001"` and `bars === [1,29]` for `CORPUS_DRILL_ROUTING` (target_dimension "timing"). It will get `czerny_001` / `[1,22]` and FAIL — and there is no "minimum implementation" that satisfies the asserted test under the specified sort.
  - Task 3 Step 1 comment (lines 432-434) wrongly claims "among timing matches the stable-first is hanon_001 (suffix 1) as well."
  Fix before executing: either (a) change Task 5's assertion to `czerny_001` / `[1,22]`, or (b) change the selection rule (e.g. tie-break by a source priority that ranks hanon first, or sort by id differently) and update BOTH tasks consistently. Pick one and make the algorithm and every asserted id agree.

[BLOCKER] (confidence: 10/10) — **Widen target is `burgmuller_001`, not `hanon_001`.** The widen branch does `stableSorted(Object.keys(MANIFEST))[0]`. Across all 22 built primitives the global stable-first under `(suffixNum, id)` is **`burgmuller_001`** (suffix 1, "b" sorts before "c"/"h"), verified by execution. But Task 3 Step 1's widen test (lines 454-456) asserts `pieceId === "hanon_001"`, `bars === [1,29]`, and "general warm-up" wording; the spec (Design §1, §Solution) repeatedly states the widen fallback IS hanon_001 ("WIDEN to the global stable-first renderable (hanon_001)"). The implementation as written returns burgmuller_001, so the test FAILS, AND the product behavior is wrong: a "general warm-up" fallback would surface a Burgmüller Romantic character piece instead of the intended neutral Hanon finger drill. Fix: make the widen target an EXPLICIT constant `hanon_001` (not "whatever global stable-first happens to be"), and assert that constant. Relying on alphabetical global-first to "happen to" equal hanon_001 is both wrong here and fragile as the corpus grows.

[BLOCKER] (confidence: 9/10) — **Task 1 overwrites an existing test file with real regression coverage.** `model/tests/exercise_corpus/test_build_render_assets.py` ALREADY EXISTS (so does the package `__init__.py`). It contains three passing tests against the current `build()`: `test_build_emits_valid_mxl_with_matching_note_count` (note-count preservation, ZIP magic, DOCTYPE-stripped), `test_build_is_idempotent`, and `test_build_raises_naming_bad_xml`. Task 1 Step 1 instructs the build agent to *create* a fresh file at that exact path with only the manifest test, and Step 5 `git add`s the file — this silently deletes three existing regression tests. The plan's Open Question ("Does `model/tests/exercise_corpus/` already exist?") was answered wrong (it assumed maybe-not). Fix: Task 1 must APPEND the manifest test to the existing file (and not re-create `__init__.py`), preserving the existing three tests.

[BLOCKER] (confidence: 8/10) — **`build()` will write the committed manifest into the repo as a side effect of the existing tests.** The existing `test_build_emits_valid_mxl_with_matching_note_count` and `test_build_is_idempotent` call `build(xml_dir=_XML_DIR, out_dir=tmp_path)` WITHOUT a `manifest_path`. With Task 1's change, `build()` defaults `manifest_path` to the real `apps/api/src/services/exercise_primitives_manifest.json` and writes it unconditionally at the end of every call. So merely running the pre-existing pytest suite mutates a committed source file in `apps/api/` (and the new manifest test, calling bare `build()`, also writes assets into `model/data/.../mxl/`). This is exactly the "tests pollute the tree" failure mode. Fix: the manifest test must pass an explicit `manifest_path=tmp_path/...` and assert against that, OR `build()` must only write the manifest when `manifest_path` is explicitly provided. Decide and make every `build()` callsite (existing + new) manifest-side-effect-free unless it intends to regenerate the committed artifact. Also note the new test's `build()` (no args) writes into the real `model/data/exercise_primitives/mxl/` — make it tmp_path-scoped too, or accept it as the canonical regeneration step and document that.

#### RISKS

[RISK] (confidence: 6/10) — **`_count_bars` may not equal 29/22/23.** Task 1 asserts `hanon_001.totalBars == 29`, `czerny_001 == 22`, `burgmuller_001 == 23` via `len(list(part.iter_all(partitura.score.Measure)))` on `parts[0]`. These literals are unverified against partitura's actual measure count for these files (repeats, pickup bars, and multi-part scores can shift the count; the existing oracle test counts NOTES, not measures, so it is no precedent). If any literal is wrong the test fails on first run. Mitigation: the build agent should compute the three counts empirically in Step 2 (watch-it-fail) and only then lock the literals — or the plan should mark these three numbers as "verify, then pin". Name the fallback: if `parts[0]` is the wrong part or repeats inflate the count, derive `totalBars` from the timemap/last measure number instead. Low blast radius (test-only) but will stall Task 1.

[RISK] (confidence: 5/10) — **JSON import typing.** `resolveJsonModule: true` is set in `apps/api/tsconfig.json`, but `"include": ["src/**/*.ts"]` lists only `.ts`. Importing `./exercise_primitives_manifest.json` from `corpus-drill.ts` should still typecheck (resolveJsonModule covers imported JSON regardless of `include`), but the manifest must exist on disk before `bunx tsc --noEmit` runs in Task 3 Step 4. Since Task 3 depends on Task 1 (which commits the manifest) this ordering holds — just confirm the manifest is present in the worktree when Task 3 typechecks. The plan's contingency note (set resolveJsonModule if tsc complains) is now moot (already set).

[RISK] (confidence: 5/10) — **`exercises.ts` widening conflicts between Task 3 and Task 5.** Task 3 edits `ExerciseSetPayload.scoreClip` to add `transpose?` AND commits `exercises.ts`. Task 5 also modifies `exercises.ts` (the corpus_drill block) and `exercises.test.ts`. Both are in Group B/C sequential ordering, so no parallel clash — but the plan should be explicit that Task 5 must NOT re-add the `transpose?` field (Task 3 already did) to avoid a confusing no-op diff or merge friction. Minor.

#### Test Philosophy — strong.

All tests exercise public interfaces: `buildCorpusDrillClip` (the module's sole export), `assignPendingExercise`, `processToolUse` (real registry dispatch + Zod parse), the rendered `ExerciseSetCard`, and the real Verovio toolkit (Task 8). R2 is faked via an `R2Bucket`-shaped `ctx.env.SCORES` (external boundary — legitimate), not an internal collaborator. No shape-only tests, no private-method assertions. Task 8's watch-it-fail discipline (remove the +2, observe FAIL, restore) is exactly right for proving a no-prod-code-change regression lock is non-vacuous.

[OBS] — Task 7's test asserts `mockLoad` is spied. The current `ExerciseSetCard.test.tsx` mock (`load: vi.fn().mockResolvedValue(undefined)`) does NOT expose a `mockLoad` handle. The plan's parenthetical already tells the agent to add one mirroring `mockGetClipPlayback`. But note: the existing `load` mock resolves `undefined`, and the component checks `if (loaded === "failed")`; `undefined !== "failed"` so it proceeds — fine. Ensure the new `mockLoad` keeps resolving a non-"failed" value or the clip path short-circuits.

[OBS] — Task 7 changes the component to always pass `transpose` (possibly `undefined`) into `load`/`getClip`/`getClipPlayback`. The existing tests assert `toHaveBeenCalledWith("chopin.ballades.1", 5, 8)` (no 4th arg). Passing `getClip(pieceId, b0, b1, undefined)` makes `toHaveBeenCalledWith(pieceId, 5, 8)` FAIL (arity mismatch: vitest compares the 4th arg `undefined` against an absent arg as not-equal). The build agent must either update those existing assertions to include `undefined`/`0`, or destructure with a default. This is a regression the plan does not call out — verify during Task 7.

#### Failure Modes — acceptable.

`resolveTranspose` degrades every unresolvable input (no pieceId, 404, R2 throw, parse failure, null/garbage key) to `transpose=0` + a structured warn; the clip always renders. No silent failure. `buildCorpusDrillClip` always returns a renderable `scoreClip` (selection cannot fail — it widens to a constant). If a selected primitive's `.mxl` is somehow absent from R2, the renderer surfaces `clipLoadError` in the card (existing behavior). No corrupt-state path.

### Presumption Inventory

```
ASSUMPTION: model/tests/exercise_corpus/test_build_render_assets.py does not yet exist.
VERDICT: RISKY (false)
REASON: It exists with 3 regression tests; Task 1 as written overwrites them. (BLOCKER)

ASSUMPTION: timing dimension stable-first primitive is hanon_001.
VERDICT: RISKY (false)
REASON: (suffixNum, id) sort puts czerny_001 first; Task 5 assertion wrong. (BLOCKER)

ASSUMPTION: widen global-stable-first equals hanon_001.
VERDICT: RISKY (false)
REASON: Global-first is burgmuller_001; widen must use an explicit hanon_001 constant. (BLOCKER)

ASSUMPTION: build() can default manifest_path to the real apps/api path harmlessly.
VERDICT: RISKY
REASON: Existing tests calling build() will write/overwrite the committed manifest. (BLOCKER)

ASSUMPTION: totalBars are 29/22/23 for hanon_001/czerny_001/burgmuller_001.
VERDICT: VALIDATE
REASON: partitura Measure count unverified; compute empirically before pinning. (RISK)

ASSUMPTION: transpose direction is interval(primitiveKey -> passageKey).
VERDICT: SAFE
REASON: briefing.py:190 = transpose_interval(exercise_pc, passage_pc); matches plan and tests.

ASSUMPTION: keys.py raises on lowercase-minor and the TS superset fixes it.
VERDICT: SAFE
REASON: _PC is uppercase-only; czerny_001="c" would raise; first-char-capitalize resolves it correctly.

ASSUMPTION: score-renderer/worker already accept and apply transpose.
VERDICT: SAFE
REASON: load/getClip/getClipPlayback take transpose?; worker applies at loadData, reads MIDI off transposed tk.

ASSUMPTION: resolveJsonModule is needed and may be unset.
VERDICT: SAFE (already set)
REASON: tsconfig already has resolveJsonModule: true.

ASSUMPTION: only *.xml in the primitives dir feed build(); the `segmented/` subdir is ignored.
VERDICT: SAFE
REASON: glob("*.xml") is non-recursive; 22 xml == 22 mxl on disk.

ASSUMPTION: Task 7's added transpose arg won't break existing ExerciseSetCard assertions.
VERDICT: VALIDATE
REASON: existing toHaveBeenCalledWith(pieceId, b0, b1) may fail when a 4th arg is passed. (OBS)
```

### Summary

[BLOCKER] count: 4
[RISK]    count: 3
[QUESTION] count: 1

The architecture, module depth, test philosophy, and failure handling are all sound, and the runtime-reuse instinct (the renderer/worker already do transpose) is correct. But the plan ships with four concrete, execution-blocking defects: two of its own asserted primitive ids contradict its own selection algorithm (timing->czerny not hanon; widen->burgmuller not hanon), and two test-infrastructure assumptions are false (the manifest test file already exists with regression coverage that would be deleted, and the new default `manifest_path` makes existing tests mutate a committed source file). Each is mechanical to fix but each would stop a TDD build agent cold or silently corrupt the tree.

VERDICT: NEEDS_REWORK — (1) reconcile the timing selection: czerny_001 is stable-first, not hanon_001 — fix Task 5's assertion or the sort; (2) make the widen target an explicit hanon_001 constant (global-first is currently burgmuller_001) and fix Task 3's widen + the spec; (3) Task 1 must append to the existing model/tests/exercise_corpus/test_build_render_assets.py, not recreate it (it holds 3 regression tests); (4) prevent build()'s manifest emission from polluting the committed apps/api manifest when existing tests run with a tmp out_dir (pass explicit manifest_path / gate the write). Also (RISK) empirically verify the 29/22/23 bar counts before pinning, and (OBS) update existing ExerciseSetCard assertions for the new trailing transpose arg.
