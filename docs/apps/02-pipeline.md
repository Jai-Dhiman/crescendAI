# Audio-to-Mark Pipeline

The complete path from microphone to feedback: how audio becomes marks on the
score, a session verdict, and a carry-forward. This is the technical heart of
the delivery system.

> **Status (2026-08-04):** TARGET ARCHITECTURE (epic #154). This describes the
> approved score-first pipeline. The previous pipeline (MuQ HF-endpoint
> scoring, teaching-moment observation streaming, chat delivery) is
> superseded; its removal is tracked in #162/#164. What is already real and
> carried forward: audio capture + chunking, the DO session brain and its
> WebSocket transport, R2 chunk storage, Transkun adoption (#125), offline
> chroma-DTW alignment, the V6 synthesis harness (`docs/harness.md`),
> exercise/segment-loop machinery, and the piece-ID open-set gate (#26).
> Build state lives in #154's sub-issues, not here.

---

## Pipeline Overview

```
mic -> 15s chunks -> WS -> DO session brain
                            1. Transkun AMT (audio -> MIDI)      [always first]
                            2. Piece resolution ladder:
                                 user pick > confident piece_id > pieceless
                            3. Offline score alignment (chunks -> bars;
                                 lags playing by seconds — acceptable,
                                 nothing on screen moves live)
                            4. MPM-style feature extraction (per-bar when
                                 aligned, per-window when pieceless)
                            5. MoonBeam symbolic scoring (MIDI -> 6 dims;
                                 score-conditioned delta when piece known)
                            6. Mark gate: scores + MPM features vs the
                                 student baseline (bandwidth + persistence)
                                 -> 0..1 mark per pause

pause (>= 20s silence)  <-WS-  mark lands on score / timeline canvas
                               (#158 UI is live; no backend emits the `mark`
                                WS event yet — #162 owns the pipeline)
silence >= 60s          ->     soft auto-stop (one-tap resume, same session)

stop -> V6 synthesis   ->      verdict + consolidated marks + one
                               carry-forward, every claim grounded in
                               MPM evidence

review: tap mark -> passage playback (R2 chunks + score cursor)
        "work on this" -> bar-bounded drill (habit-informed variants)
        "ask about this" -> passage-scoped teacher thread

after: student model update — per-dimension EWMA baselines, mark
       lifecycle, habit facts (see 03-memory-system.md)
```

**Streaming is capture, not delivery.** Audio streams continuously so
analysis is ready seconds after the student pauses or stops — but nothing is
*delivered* mid-playing. This preserves the latency asset while respecting
the feedback-timing evidence (marks at boundaries, review at session end).

---

## Stage Notes

### 1. Capture and chunking

Unchanged: MediaRecorder (web) / AVAudioEngine (iOS) produce 15s chunks,
uploaded to the API and stored in R2; the DO session brain owns session
state and the WebSocket. Silence gating before upload survives. If the
WebSocket drops, recording continues locally and chunks re-upload on
reconnect; the review discloses any gap.

### 2. Transkun AMT — always first

Every chunk is transcribed to MIDI by Transkun (#125). Everything downstream
is symbolic. **There is currently no audio-native stream to cross-check the
transcription** (MuQ removed; the audio-native teacher is a separately gated
program, epic #129). Consequence: low-transcription-confidence windows are
**no-comment zones** — no marks, no MPM claims, excluded from baseline
updates, disclosed in the review ("I couldn't hear the fast section clearly
enough to judge it").

### 3. Piece resolution ladder

1. User picked the piece (home -> piece -> record): full certainty.
2. No pick: the open-set piece-ID gate (#26) runs on accumulated notes; a
   confident match surfaces as a dismissible confirm chip in practice mode.
3. No confident match: pieceless session (timestamp anchors, timeline
   canvas). Permanent state, not transitional — the score library is
   copyright-cleared only. Identification enriches, never gates.

Rungs 1 and 2 are implemented and tested (#158) but not yet reachable from a
live Record session: `AppChat` hardcodes `userPickedPieceId={null}` /
`confidentGuess={null}`, and `usePracticeSession`'s `piece_identified`
handler only logs free-text composer/title instead of yielding a catalog
`pieceId`. #160 owns wiring both.

### 4. Offline alignment

Chroma-DTW alignment maps chunks to bars, lagging real playing by seconds.
The follower is an **offline orientation tool** — it decides where marks and
the review cursor land; it never drives a live cursor. Bar anchoring is
gated on per-passage alignment quality: shaky alignment degrades the anchor
to a timestamp. Wrong bar numbers are never shown.

### 5. MPM-style feature extraction

Per-bar interpretable expressive features in the spirit of Music Performance
Markup (Berndt; the MEI-companion performance format): tempo/rubato
decomposition, dynamics transition curves, articulation ratios, pedal
overlap. Computed deterministically from Transkun MIDI + alignment
(per-window when pieceless).

MPM features are the system's **evidence layer**: they ground every mark's
"why", brief the teacher LLM, and define the baseline the arc adapts
against. **Guardrail (standing, from #19):** MPM features are evidence and
auxiliary supervision only — never primary model inputs at inference.

### 6. MoonBeam scoring

MoonBeam-839M with mean pooling (the #138 Phase 0 winner on
Transkun-transcribed audio) scores 6 dimensions from the MIDI;
score-conditioned (delta vs. score MIDI) when the piece is known. Dimension
scores are **internal routing signal** — they rank what gets said and feed
the baseline; they are never shown to the user as numbers.

### 7. Mark gate

A candidate becomes a mark only when it clears the student baseline's
bandwidth-plus-persistence gate (`03-memory-system.md`): outside the
student's own tolerance band, persistently, deduplicated against open marks
on the same passage, paced to at most one mark per pause. Cold-start
sessions use cohort priors with exploratory framing — the cold-start path is
the same code path as the baseline-unavailable fallback.

### 8. Synthesis (V6 harness)

The V6 hook-driven compound loop (`docs/harness.md`) remains the synthesis
engine; its **output contract changes** from chat prose to: one verdict
sentence, the consolidated mark set, and one carry-forward. Every claim must
trace to MPM evidence. A failed synthesis is loud: the review renders an
explicit retry state, never a blank and never a fabricated verdict.

---

## The Three Stable Contracts

Model research swaps in behind these without UI change (see #154):

1. **Mark** — `{anchor: bars|timestamp, taxonomy: needs_work |
   missed_opportunity | strong, dimension, evidence, lifecycle}`
2. **Verdict + carry-forward** — one prose verdict, one next-session focus
3. **Baseline API** — per-dimension EWMA pair + tolerance band + persistence
   state

Planned model milestones land as content upgrades behind these contracts:
#138 fine-tune (better ranking), claim verifier (verified evidence),
follower HMM (more bar anchors), relative-to-score assessment (marks
referencing the score's own markings), temporal reasoning (repetition
comparison), audio-native teacher (no-comment zones shrink).

---

## Open Questions

1. **MoonBeam serving.** Local vs. hosted inference for the encoder; cost
   and latency envelope per session (resolve in #162).
2. **Transcription confidence signal.** What Transkun exposes (or what
   proxy — note density sanity, alignment residual) defines a no-comment
   zone (resolve in #162).
3. **Pause threshold.** 20s mark-delivery / 60s auto-stop are starting
   values, config-tunable.
4. **Per-session cost.** Re-measure once MoonBeam serving is decided; the
   old MuQ HF-endpoint cost model (~$0.30-0.51/session) no longer applies.
