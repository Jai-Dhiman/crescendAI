# Landing Screen v2 Design

**Goal:** Replace the static marketing sections below the hero with an interactive Exercise Proof Block — three prebaked ProofCards showing real score rendering, bidirectional cursor-audio sync, and per-bar quality score inspection — giving self-learner visitors a direct encounter with the product before they sign up.

**Not in scope:**
- Live inference or runtime API calls from the landing page
- Educator/institution messaging (B2B copy removed)
- Any change to the Hero section (lines 28–71 of `apps/web/src/routes/index.tsx`)
- Native iOS changes
- User account or practice session creation
- Score rendering from a remote piece ID via the API (`api.scores.getData`); all assets are prebaked static files

---

## Problem

The current landing page (`apps/web/src/routes/index.tsx` lines 73–219) shows animated MP4 placeholders, stock photos, and a pull-quote. None of these convey what CrescendAI actually does. Visitors who want to understand the product — "how does it evaluate expression, not notes?" — have no interactive surface to explore before signing up. The result: low activation on the hard metric (record-first-session).

---

## Solution (from the user's perspective)

A visitor scrolls past the hero. They see: "Hear what your teacher hears." Above three full-width cards, each showing a real score for a well-known piano piece (one Romantic, one Baroque, one Contemporary), a playback control, and highlighted bars with quality scores overlaid.

The visitor taps any bar on a card. A small chip appears showing six mini-bars for dynamics, timing, pedaling, articulation, phrasing, and interpretation — the same six dimensions the live product uses.

Below the highlighted bar they read a one-paragraph teacher diagnosis for the focus bar. Below that, a generated exercise card (using the same Artifact/InlineCard component from the live app) shows what the teacher prescribed.

The visitor presses play. The cursor tracks through the score in sync with the audio. They can scrub the audio scrubber to jump to any moment; the cursor follows. They can scrub the cursor; the audio jumps. When they scroll a card into view it begins playing automatically. A manual play button is always visible for keyboard/reduced-motion users.

At the bottom of the page: "Your playing. Heard clearly." with a "Start your first session" button linking to `/app`.

---

## Design

**Static-asset-only data flow.** All artifacts are prebaked files in `apps/web/public/landing/card-{1,2,3}/`. No runtime API calls from the landing page. CF edge cache serves them immutably.

**Zero new inference infrastructure.** The `perBarScores` field in each manifest is populated once by running three recordings through the production pipeline (see Appendix: Artifact Production Recipe) and snapshotting the output.

**Reuse shipped components.** ScoreCursor, ScoreAnnotation, and InlineCard/Artifact internals are already tested and production-verified. ProofCard wraps them behind a single `manifest` prop, hiding coordination complexity.

**Interaction asymmetry is intentional.** All bars are tappable (any `BarScoreChip`); only one bar per card has a teacher diagnosis and exercise. This teaches the encoder-plus-teacher architecture by use: "the encoders score every bar, the teacher diagnoses the one that matters most."

**Bidirectional sync via single timeline state.** A single `currentTime: number` React state drives both the audio element's `currentTime` and the `qstampSource` passed to `ScoreCursor`. Changes from either direction (audio `timeupdate` or scrubber drag) update this shared state.

**Score loading without the API.** The production `ScoreRenderer.load()` calls `api.scores.getData(pieceId)` which hits the API. ProofCard bypasses this: it fetches the prebaked `scoreir.json` directly via `fetch(manifest.scoreIRUrl)` and constructs a `ScoreIR` value from it, then passes that directly to `ScoreCursor`. Score SVG rendering is not needed on the landing page — the score visual is a static SVG pre-rendered from the scoreIR (included as `score.svg` in each card's asset bundle). This keeps the landing page free from the Verovio WASM dependency and the worker protocol.

**Repertoire selection:**
- Card 1 (Romantic): Chopin, Nocturne Op. 9 No. 2 in E-flat major — widely known, expressive dynamics ideal for demonstrating the pedaling and phrasing dimensions
- Card 2 (Baroque): Bach, Prelude in C Major (WTC Book I, BWV 846) — uniform arpeggiation makes timing and articulation deviations clearly visible in bar scores
- Card 3 (Contemporary): Satie, Gymnopédie No. 1 — simple texture that shows interpretation and dynamics scoring for expressive, unhurried playing

**Copy:**
- ResearchCallout: "Music AI that listens for expression, not just notes — trained on competitive performance data from international competitions."
- Footer footnote (ISMIR citation): Foscarin S. et al., "MIDI2Score: Automatic Score Transcription for Piano Music," ISMIR 2024.
- FinalCTA headline: "Your playing. Heard clearly."
- FinalCTA button: "Start your first session"

**Analytics events** (via `window.gtag` — no analytics library currently installed; plan adds a thin event helper that no-ops when gtag is absent):
- `landing_hero_cta_click` — hero "Start Practicing" CTA
- `landing_final_cta_click` — final CTA "Start your first session"
- `landing_proof_card_enter` — ProofCard-N entered viewport (≥60% intersection)
- `landing_proof_card_played_to_end` — audio `ended` event for card N
- `landing_bar_tap` — bar tapped, with `{ cardIndex: number, barNumber: number }`

**UI/UX (userinterface-wiki pass):**
- Audio autoplay: gated on IntersectionObserver ≥60% threshold + `document.visibilityState === 'visible'`; never autoplays on `prefers-reduced-motion: reduce`
- Reduced motion: detects via `window.matchMedia('(prefers-reduced-motion: reduce)')` at mount; disables autoplay, pauses ScoreCursor rAF loop, shows manual play button unconditionally
- Asset prefetch: `<link rel="prefetch">` tags for card-2 and card-3 assets injected into document head once card-1 enters viewport
- Mobile breakpoints: ProofCard stacks score (top) over controls+diagnosis (bottom) on screens < 768px; score SVG scales to container width via `width: 100%; height: auto`

**Appendix: Artifact Production Recipe**

Run once per piece. Produces static files committed to `apps/web/public/landing/card-N/`.

```bash
# 1. Start local services
just dev-muq   # MuQ (8000) + API (8787) + Web (3000)

# 2. Upload a recording through the practice session API
#    (use curl or the web app's record button; save the sessionId)
SESSION_ID=<your-session-id>
PIECE_ID=chopin.nocturnes.9-2   # substitute per card

# 3. Trigger synthesis to get bar-level scores + teacher diagnosis + exercise
curl -X POST http://localhost:8787/api/practice/synthesize \
  -H "Content-Type: application/json" \
  -d "{\"sessionId\": \"$SESSION_ID\"}"

# 4. Export the ScoreIR for the piece
#    (scoreRenderer.load() in the browser writes it; use the debug endpoint)
curl "http://localhost:8787/api/scores/$PIECE_ID/data" \
  --output /tmp/score.midi

# 5. Run the snapshot script (created as part of this plan in Task 0)
bun run apps/web/scripts/snapshot-landing-card.ts \
  --session $SESSION_ID \
  --piece $PIECE_ID \
  --card 1 \
  --out apps/web/public/landing/card-1/

# Produces:
#   recording.opus   — trimmed audio (first 30s)
#   score.svg        — static pre-rendered SVG of the score
#   scoreir.json     — ScoreIR JSON (for ScoreCursor)
#   exercise.json    — InlineComponent (exercise_set) from synthesis output
#   manifest.json    — ProofCardManifest with perBarScores
```

---

## Modules

### ProofCard

**Interface:** `<ProofCard manifest={ProofCardManifest} cardIndex={number} />`

**Hides:**
- Fetch of `scoreir.json` from the static asset URL
- Construction of `ScoreCursor` with `qstampSource` derived from shared `currentTime` state
- IntersectionObserver lifecycle (mount/unmount, threshold callback)
- Bidirectional sync: audio `timeupdate` → state; scrubber drag → state → audio.currentTime
- `prefers-reduced-motion` detection and gating of autoplay/cursor
- Asset prefetch injection for subsequent cards
- Analytics event emission
- Keyboard navigation: Tab cycles bars, Enter opens BarScoreChip, Escape closes

**Depth verdict:** DEEP — single prop, hides substantial coordination between five subsystems (fetch, cursor, audio, IO, analytics)

**Tested through:** Public JSX render behavior, IntersectionObserver mock, audio element mock, keyboard events

---

### BarScoreChip

**Interface:** `<BarScoreChip scores={BarQualityScores} barNumber={number} onClose={() => void} />`
where `BarQualityScores = { dynamics: number; timing: number; pedaling: number; articulation: number; phrasing: number; interpretation: number }`

**Hides:** 6-mini-bar rendering with clamped 0–1 values, label positioning, Escape key listener

**Depth verdict:** SHALLOW — interface roughly mirrors implementation; justified because it is a pure display component with no logic, and the design calls it out as shallow

**Tested through:** Rendered bar heights reflect passed score values; Escape key calls `onClose`

---

### ExerciseProofBlock

**Interface:** `<ExerciseProofBlock manifests={[ProofCardManifest, ProofCardManifest, ProofCardManifest]} />`

**Hides:** ResearchCallout text, three ProofCard instances, stacking layout

**Depth verdict:** SHALLOW — pure layout wrapper; justified as composition-only

**Tested through:** Not tested independently (behavior verified through ProofCard tests + a11y scan of the full LandingPage)

---

### `useProofCardTimeline` (custom hook, internal to ProofCard)

**Interface:** `useProofCardTimeline(audioRef, scoreIR | null): { currentTime: number; setCurrentTime: (t: number) => void; qstampForTime: (t: number) => number | null }`

**Hides:** Mapping from audio `currentTime` (seconds) to qstamp (quarter-note units) via the ScoreIR bar timeline, bidirectional update coordination

**Depth verdict:** DEEP — the qstamp↔time mapping requires traversing the bar timeline; extracted as a hook so it can be tested in isolation

**Tested through:** Unit test verifying qstamp mapping and bidirectional update behavior (Task 8)

---

### `trackLandingEvent` (analytics helper, new file)

**Interface:** `trackLandingEvent(name: string, params?: Record<string, unknown>): void`

**Hides:** `window.gtag` availability check (no-ops when absent)

**Depth verdict:** SHALLOW — three-line helper; justified as a guard wrapper

**Tested through:** Not independently tested (called inside ProofCard; behavior observable in integration)

---

## Verification Architecture

**Canonical success state:** Running `bun run test --run src/components/ProofCard.test.tsx src/components/BarScoreChip.test.tsx src/routes/index.test.tsx` produces all 9 test cases passing.

**Automated check:**
```bash
cd apps/web && bun run test
```

**Harness:** Task 0 creates `apps/web/scripts/snapshot-landing-card.ts` (the artifact production recipe script) and the three manifest JSON fixture files used by all subsequent tests. Task 0 ships independently.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/routes/index.tsx` | Replace lines 73–219 with `<ExerciseProofBlock>`, `<FinalCTA>`, `<LandingFooter>`; add hero CTA analytics event | Modify |
| `apps/web/src/components/ProofCard.tsx` | New deep component | New |
| `apps/web/src/components/ProofCard.test.tsx` | 8 behavior tests (Tasks 1–5, 7–9) | New |
| `apps/web/src/components/BarScoreChip.tsx` | New shallow display component | New |
| `apps/web/src/components/BarScoreChip.test.tsx` | 2 behavior tests (Task 7 shares file) | New |
| `apps/web/src/components/ExerciseProofBlock.tsx` | New layout wrapper | New |
| `apps/web/src/routes/index.test.tsx` | A11y scan test (Task 6) | New |
| `apps/web/src/hooks/useProofCardTimeline.ts` | New custom hook | New |
| `apps/web/src/hooks/useProofCardTimeline.test.ts` | Timeline hook unit test (Task 8) | New |
| `apps/web/src/lib/landing-analytics.ts` | New analytics helper | New |
| `apps/web/src/types/landing.ts` | `ProofCardManifest` type + `BarQualityScores` type | New |
| `apps/web/public/landing/card-1/manifest.json` | Prebaked manifest (Chopin) | New |
| `apps/web/public/landing/card-2/manifest.json` | Prebaked manifest (Bach) | New |
| `apps/web/public/landing/card-3/manifest.json` | Prebaked manifest (Satie) | New |
| `apps/web/scripts/snapshot-landing-card.ts` | Artifact production script | New |

---

## Open Questions

- Q: Should the prebaked `score.svg` be generated from Verovio at snapshot time (committed as static SVG) or rendered via the ScoreRenderer worker at runtime on the landing page?
  Default: Static committed SVG. Avoids Verovio WASM on the critical landing path, keeps TTI bounded by ~200KB asset budget, and is consistent with the "zero runtime API calls" constraint.

- Q: Does the landing page need auth-gated behaviour (redirect logged-in users to `/app`)?
  Default: Yes — keep the existing `beforeLoad` redirect in `index.tsx` unchanged (already in place at lines 6–10).

- Q: What qstamp-to-time mapping strategy is used when the prebaked `scoreir.json` has no absolute timing data?
  Default: The manifest includes a `barTimeline: Array<{ bar: number; tSec: number }>` field mapping bar numbers to audio timestamps from the prebaked recording. `useProofCardTimeline` uses this for qstamp↔time conversion instead of deriving it from the ScoreIR.
