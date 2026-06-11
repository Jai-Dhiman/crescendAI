# Issue #45 — Own-Passage Loop: Excerpt + Reduced-Tempo Playback, Score-First Card Design

**Goal:** When the teacher prescribes an `own_passage_loop` exercise, the student's weak bars render as the HERO element of the exercise card (sheet music central, text secondary) and can be looped at a reduced tempo with a moving playback cursor, an audible metronome, AND synthesized piano audio.

**Not in scope:**
- `corpus_drill` logic (text-stub seam preserved and kept green, S4 owns it)
- `model/`, `apps/evals/`, `apps/api/src/lib/types.ts` (parallel session owns that file)
- The live WebSocket `segment_loop` rep-counter (separate path, do not touch)
- Score LOAD path improvements (S3/#46 owns that)
- Rep counting / auto-advance logic
- CDN-hosted smplr samples (must self-host in `apps/web/public/soundfonts/`)

## Problem

Currently `ExerciseSetCard` renders the clip SVG as a small header panel above text rows. The `scoreClip` object on `ExerciseSetConfig` carries only `{ pieceId, bars }` — no `tempoFactor` — so the tempo the teacher prescribed is silently dropped and the card has no playback capability. There is no synthetic loop clock, no piano audio, no metronome. The card is text-first despite the score being the primary asset.

## Solution (from the user's perspective)

When the teacher card appears for an `own_passage_loop` prescription:

1. The score clip occupies the top ~60% of the card — large, white-background, full-width.
2. Below the score, a transport bar shows Play/Pause/Stop + a continuous tempo slider (0.25x–1.0x) defaulting to the teacher's `tempo_factor`.
3. Pressing Play counts in one bar of metronome clicks, then the passage loops continuously: the cursor sweeps through the clip in sync with the tempo, metronome ticks on every beat, and synthesized piano notes play at the correct pitches and timing.
4. Adjusting the slider during playback rescales tempo live.
5. The exercise title, instruction, and focus dimension appear below the transport — still visible, now secondary.
6. `corpus_drill` cards (no `scoreClip`, no `tempoFactor`) are unchanged — same text-stub layout, no transport rendered.

## Design

### Key decisions

**One LoopClock, three consumers.** A single synthetic clock maps wall-time → qstamp within the clip's bar range. Three consumers read it: (1) `ScoreCursor` via its existing `qstampSource` seam, (2) the Web Audio metronome scheduler, (3) the smplr piano note scheduler. This keeps timing consistent — the cursor cannot drift from the audio.

**Clip-scoped ScoreIR.** The existing `getClip` message returns only an SVG string. The cursor needs an IR whose bars/notes are scoped to the clip's single page (not the full piece). A new worker message `get_clip_playback` runs `select(range) + redoLayout` then calls `parseScoreIR` on the resulting single-page SVG. Without this, the cursor's binary-search over `ir.bars` would find bars that reference pageN > 1, whose overlays are never mounted over the clip SVG.

**smplr `Soundfont` with self-hosted samples.** `smplr` is MIT-licensed, well-maintained, and exposes a clean `piano.start({ note, time, duration, velocity })` API compatible with Web Audio scheduling. The acoustic-grand instrument sample pack is fetched once by a committed script and served from `apps/web/public/soundfonts/` to avoid CDN dependency. `bun add smplr` in `apps/web`.

**LoopClock is pure/no-DOM.** All timing math is in a plain TypeScript class with no `AudioContext`, no `requestAnimationFrame`, no React. This makes it fully unit-testable without stubs.

**LoopPlayer owns AudioContext + smplr.** Lookahead scheduler pattern (25ms poll, 100ms lookahead) schedules notes and metronome ticks ahead of audio time. Same pattern as the existing `useMetronome.ts`.

**Data contract change — `tempoFactor` added to `scoreClip`.** `apps/web/src/lib/types.ts` gains `tempoFactor?: number` on `ExerciseSetConfig.scoreClip`. The two API construction sites in `exercises.ts` (~line 224) and `tool-processor.ts` (~line 94) set it from `routing.tempo_factor`. `apps/api/src/lib/types.ts` is not touched.

### Why not reuse `useMetronome.ts`?

`useMetronome` is a standalone React hook that owns its own AudioContext lifecycle and has no concept of a qstamp-based position source. `LoopPlayer` needs a shared `AudioContext` between metronome clicks and piano notes (to schedule both against the same clock), and must expose a `qstampNow()` source for the cursor. Reusing would require significant restructuring; a new deep module is cleaner.

### Error handling

- smplr load failure → set `audioUnavailable: true` state → render "Audio unavailable" badge; cursor + metronome still run (degraded mode). `Sentry.captureException` called.
- `AudioContext` suspended (gesture-gate) → `play()` calls `ctx.resume()` before scheduling; if still suspended, show "Tap play to start" notice.
- `get_clip_playback` worker failure (corrupt MXL) → extend existing `clipLoadError` path in `ExerciseSetCard` → static clip + text fallback, logged via `console.error`.
- Zero parseable notes in clip IR → metronome-only loop still runs; `console.error` logged.

## Modules

### `apps/web/src/lib/types.ts` — data contract extension

- **Interface:** `ExerciseSetConfig.scoreClip` gains `tempoFactor?: number`
- **Hides:** nothing structural; this is a pure type extension
- **Depth verdict:** SHALLOW (it is a type file, not a logic module — acceptable)

### `apps/web/src/lib/score-worker.ts` + `score-renderer.ts` — `get_clip_playback`

- **Interface:** Worker message `{ type: "get_clip_playback", pieceId, startBar, endBar }` → `{ svg: string; ir: ScoreIR; notes: Array<{ midi: number; startQ: number; endQ: number }> }`. `ScoreRenderer.getClipPlayback(pieceId, start, end)` wraps this.
- **Hides:** select+redoLayout, clip-scoped `parseScoreIR`, timemap note extraction, MIDI pitch extraction via `tk.getMIDIValuesForElement(id)`.
- **Depth verdict:** DEEP — simple request/response surface hides the multi-step Verovio pipeline.
- **Tested through:** `processGetClipPlaybackRequest(tk, measures, start, end)` unit-tested with a fake `tk` stub that returns known timemap + MIDI values; confirmed notes have plausible `midi/startQ/endQ` fields. Real Verovio tested in the existing integration test file (extended with one `get_clip_playback` fixture test).

### `apps/web/src/lib/loop-clock.ts` — synthetic loop timing

- **Interface:** `new LoopClock({ clipStartQ, clipEndQ, beatsPerBar, bpmAtUnity, tempoFactor }); clock.start(nowMs); clock.qstampNow(nowMs): number | null; clock.setTempoFactor(f); clock.stop();`
- **Hides:** count-in offset (one bar of metronome = `beatsPerBar / (bpmAtUnity * tempoFactor) * 60` seconds), wrap arithmetic, elapsed-to-qstamp mapping.
- **Depth verdict:** DEEP — complex wrapping/count-in math behind a `qstampNow(nowMs)` call.
- **Tested through:** pure unit tests on `qstampNow` — wrap correctness, count-in delay, tempo rescale.

### `apps/web/src/lib/loop-player.ts` — audio orchestrator

- **Interface:** `new LoopPlayer({ ctx, instrumentUrl, clipIR, clipNotes, beatsPerBar, bpmAtUnity }); player.play(); player.pause(); player.stop(); player.setTempoFactor(f); player.qstampSource(): number | null; player.state: "idle"|"counting-in"|"playing"|"paused"; player.audioUnavailable: boolean;`
- **Hides:** smplr `Soundfont` lifecycle, lookahead note scheduler (`setInterval` 25ms), metronome oscillator scheduling, `LoopClock` internal, note deduplication across scheduler ticks, smplr error handling.
- **Depth verdict:** DEEP — 5+ internal concerns behind a 5-method public API.
- **Tested through:** fake `AudioContext` + stubbed smplr instrument; assertions on note-on times scaling with `tempoFactor`, metronome beat count per pass, transport state transitions.

### `apps/web/src/hooks/useLoopPlayer.ts` — React adapter

- **Interface:** `useLoopPlayer(config: { clipIR: ScoreIR | null; clipNotes: ClipNote[]; beatsPerBar: number; bpmAtUnity: number; tempoFactor: number }): { isPlaying, isCounting, audioUnavailable, tempoFactor, play, pause, stop, setTempoFactor, qstampSource }`
- **Hides:** `LoopPlayer` mount/unmount lifecycle, `AudioContext` lazy construction, smplr `instrumentUrl` path, `useState` wiring.
- **Depth verdict:** DEEP enough — thin adapter but hides lifecycle and AudioContext construction from the component.
- **Tested through:** `ExerciseSetCard` render test (transport controls present; play/pause toggling); the hook itself is not unit-tested in isolation (thin adapter pattern, tested via component behavior).

### `apps/web/src/components/cards/ExerciseSetCard.tsx` — score-first redesign

- **Interface:** unchanged props `{ config, onExpand, artifactId }` — same API, new layout
- **Hides:** conditional rendering for `own_passage_loop` (has `scoreClip + tempoFactor`) vs `corpus_drill` (no `scoreClip`) vs no-clip fallback; `useLoopPlayer` wiring; `ScoreCursor` over clip IR; transport controls.
- **Depth verdict:** DEEP (single component with non-trivial conditional logic behind stable props)
- **Tested through:** render tests — own_passage renders score hero + transport; corpus_drill renders text-only stub; assign still works; existing corpus_drill stub test stays green.

### `apps/web/scripts/fetch-soundfont.ts` — sample fetcher (one-time utility)

- **Interface:** `bun apps/web/scripts/fetch-soundfont.ts` — downloads acoustic-grand instrument into `apps/web/public/soundfonts/`
- **Hides:** smplr CDN URL discovery, file writing
- **Depth verdict:** SHALLOW (utility script, not a module)

## Verification Architecture

- **Canonical success state:** `bun run test` in `apps/web/` passes all tests including new ones. `just dev-light` shows an exercise card with score as hero, play button triggers count-in then loop, cursor moves, piano plays at 0.5x tempo.
- **Automated check:** `cd apps/web && bun run test` — covers LoopClock wrap/count-in/tempo, LoopPlayer transport + note timing, get_clip_playback fixture, ExerciseSetCard render scenarios.
- **Harness:** No harness buildable before the feature (the loop + audio interaction requires browser APIs). Manual verification step: `just dev-light`, prescribe an exercise in chat, confirm loop plays with cursor + piano.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/types.ts` | Add `tempoFactor?: number` to `scoreClip` in `ExerciseSetConfig` | Modify |
| `apps/api/src/services/exercises.ts` | Pass `tempoFactor: routing.tempo_factor` into `scoreClip` object | Modify |
| `apps/api/src/services/tool-processor.ts` | Pass `tempoFactor: input.tempo_factor` into `scoreClip` object | Modify |
| `apps/web/src/lib/score-worker.ts` | Add `get_clip_playback` message handler + `processGetClipPlaybackRequest` export | Modify |
| `apps/web/src/lib/score-renderer.ts` | Add `getClipPlayback()` method | Modify |
| `apps/web/src/lib/loop-clock.ts` | New pure loop timing module | New |
| `apps/web/src/lib/loop-clock.test.ts` | Unit tests for LoopClock | New |
| `apps/web/src/lib/loop-player.ts` | New audio orchestrator (smplr + metronome + LoopClock) | New |
| `apps/web/src/lib/loop-player.test.ts` | Unit tests for LoopPlayer with stub AudioContext + smplr | New |
| `apps/web/src/hooks/useLoopPlayer.ts` | New React adapter hook | New |
| `apps/web/src/components/cards/ExerciseSetCard.tsx` | Score-first layout + transport + ScoreCursor wiring | Modify |
| `apps/web/src/components/cards/ExerciseSetCard.test.tsx` | Extended render tests; existing corpus_drill stub test kept green | Modify |
| `apps/web/src/components/LoopTransport.tsx` | New transport UI component (play/pause/stop + tempo slider) | New |
| `apps/web/scripts/fetch-soundfont.ts` | Utility script to self-host acoustic-grand samples | New |

## Open Questions

None — all design decisions were resolved in the approved brainstorm.
