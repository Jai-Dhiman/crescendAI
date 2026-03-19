# UI System: Chat-First Interface and On-Demand Components

How observations are presented to the user. Covers the chat-first interface, on-demand rich components, and the three-stage pipeline extension that configures them.

> **Status (2026-03-19):** Unified artifact container system DESIGNED (CEO review 2026-03-19). Replaces the three-stage UI subagent pipeline with a single artifact system. Artifacts live inline in chat and expand to viewport on demand. Beta ships with exercise artifact type only. Web chat interface COMPLETE. Score panel PARTIAL.

---

## Design Philosophy

### Chat-First

Text is the default medium. The student plays, asks "how was that?", and gets a 1-3 sentence observation in a chat thread. Rich UI components are optional enhancements -- they appear when the teacher decides a visual or interactive aid adds real pedagogical value, not as decoration.

A real teacher reaches for a pencil to mark up the score maybe 30% of the time. The rest is conversation. CrescendAI follows the same ratio.

### Progressive Disclosure

The observation text carries the full teaching insight. Components add depth on demand:

1. **Observation first:** The student reads the text and understands the point.
2. **Component inline:** If a component is attached, it appears below the text as an expandable card. The student can ignore it.
3. **"Tell me more":** The student can ask follow-up questions, request exercises, or explore references -- all within the chat.

### Expected Distribution

- ~70% of observations are text-only (general encouragement, simple corrections, awareness-building)
- ~30% include a component (specific passages needing visual annotation, exercises, or reference recordings)

Start conservative (~20%) and increase based on engagement data.

---

## Artifact System (Unified Container)

The CEO review (2026-03-19) replaced the three-stage UI subagent pipeline with a unified artifact container system. Instead of a separate LLM stage to configure UI components, the teacher LLM declares artifacts directly via tool use or MCP (pattern TBD -- needs research).

### How It Works

1. The teacher LLM (Anthropic / Sonnet 4.6) generates the observation text
2. When a rich component is warranted, the teacher calls a tool (e.g., `generate_exercise`, `highlight_score`) that produces a structured artifact config
3. The API returns both the text and the artifact config to the client
4. The client renders the artifact via the unified `<Artifact>` container component

### Artifact States

Every artifact has three visual states:

```
COLLAPSED (preview)  -->  INLINE (rich card in chat)  -->  EXPANDED (viewport takeover)
```

- **Collapsed:** Minimal preview in chat (title + one-line summary). Default for returned-to observations.
- **Inline:** Rich card in the chat thread. Shows full content. Default for new observations with artifacts.
- **Expanded:** Takes over the viewport. Chat slides away or becomes a sidebar. Triggered by tapping "Expand" or by starting to play an exercise.

### Artifact Config Schema

```json
{
  "text": "Your LH is burying the melody in bars 12-16...",
  "artifact": {
    "type": "exercise",
    "config": {
      "title": "Voice the melody",
      "bars": [12, 16],
      "target_dim": "dynamics",
      "instruction": "Play bars 12-16. Top voice at mf, LH accompaniment at pp.",
      "difficulty": "intermediate"
    }
  }
}
```

When no artifact is warranted:

```json
{
  "text": "Nice phrasing through that transition. The rubato felt natural.",
  "artifact": null
}
```

### Artifact Types

| Type | Beta? | Triggers When | Prerequisites |
|------|-------|---------------|---------------|
| `exercise` | Yes | Corrective feedback where structured practice helps | None (text-based) |
| `score_highlight` | Phase 3 | Observation references specific bars with score available | Score rendering lib, score data |
| `reference_browser` | Phase 3 | Interpretation/phrasing feedback, "what should this sound like?" | Network access |
| `session_review` | Phase 3 | Session ends, synthesis ready | Session brain data |

### Latency Budget (Revised)

| Path | Latency |
|------|---------|
| Text-only observation | ~1.8s (subagent 0.3s + teacher 1.5s) |
| Observation with artifact | ~2.0-2.5s (subagent 0.3s + teacher with tool use 1.7-2.2s) |

The UI subagent stage (previously ~0.3-0.5s) is eliminated. Tool use adds ~0.2-0.5s to the teacher call but removes an entire LLM round-trip.

### Tool Use vs MCP (Open Decision)

The exact pattern for artifact declaration needs research:

- **Option A: Anthropic tool_use** -- Teacher LLM has tools like `generate_exercise(bars, dimension, instruction)`. Schema enforces valid configs. Native to the Anthropic API.
- **Option B: Self-hosted MCP server** -- Artifact types registered as MCP tools. Decouples artifact definitions from the teacher prompt. New types added by registering new tools.
- **Option C: Structured output** -- Teacher outputs JSON with artifact field via JSON mode. Simplest but least reliable.

Decision deferred to implementation phase after research.

---

## Component Library

Four pre-built components render from JSON artifact configs. The unified `<Artifact>` container accepts any type and delegates to the appropriate renderer. Beta ships with `exercise_set` only; other types ship in Phase 3.

### 1. Score Highlight (`score_highlight`) (Phase 3)

Scrolling sheet music with annotations. Shows the student *where* in the music the observation applies.

**Triggers when:** The teacher observation references a specific passage, bar range, or musical moment. Most common for dynamics, phrasing, and articulation feedback.

**Configuration schema:**

```json
{
    "type": "score_highlight",
    "config": {
        "piece": "Chopin Nocturne Op. 9 No. 2",
        "bar_range": [20, 24],
        "highlight_dimension": "dynamics",
        "annotations": [
            { "bar": 21, "beat": 1, "text": "crescendo starts here" },
            { "bar": 23, "beat": 3, "text": "peak should land here" }
        ],
        "show_dynamics_curve": true,
        "show_keyboard": false
    }
}
```

**Prerequisites:**

- Score alignment (chunk timestamps to bar numbers)
- Score rendering engine (MusicXML/Lilypond to notation)
- Score data for the student's piece (initially from a curated library, later user-uploaded)

**Graceful degradation:** If no score data is available for the piece, `score_highlight` is unavailable. The teacher falls back to `text_only`. The observation text still carries the full insight -- the student just does not see the annotated notation.

### 2. Keyboard Guide (`keyboard_guide`) (Phase 3)

Piano keyboard with lit keys, optionally synchronized to scrolling notation. For beginners learning to read music or for anyone learning a new passage.

**Triggers when:** The student is in early learning arc with a piece, or the teacher identifies a fingering/note-reading issue. More common for beginners.

**Configuration schema:**

```json
{
    "type": "keyboard_guide",
    "config": {
        "piece": "Chopin Nocturne Op. 9 No. 2",
        "bar_range": [20, 24],
        "hand": "right",
        "tempo_fraction": 0.5,
        "show_notation": true,
        "highlight_notes": [
            { "bar": 20, "beat": 1, "keys": ["C4", "E4", "G4"], "duration": 0.5 },
            { "bar": 20, "beat": 2, "keys": ["D4", "F4", "A4"], "duration": 0.5 }
        ]
    }
}
```

**Prerequisites:**

- Score alignment
- MIDI data or note-level score data for the piece (MusicXML provides this)
- Piano keyboard component (88-key scrollable, with highlight state per key)

**Graceful degradation:** If score data is unavailable, `keyboard_guide` is unavailable. Falls back to `text_only` or `exercise_set` (text-based exercises do not need notation).

### 3. Exercise Set (`exercise_set`) (Beta)

Takes the specific passage and skill the teacher identified, generates 2-3 targeted practice variations as interactive cards with instructions. Cross-references `04-exercises.md` for the exercise database schema.

**Triggers when:** The teacher observation includes a corrective framing and the student would benefit from structured practice, not just awareness.

**Configuration schema:**

```json
{
    "type": "exercise_set",
    "config": {
        "source_passage": "bars 20-24, Chopin Nocturne Op. 9 No. 2",
        "target_skill": "dynamic control through crescendo",
        "exercises": [
            {
                "title": "Three-level dynamics",
                "instruction": "Play bars 20-24 at pp, then mf, then ff. Feel the difference in arm weight.",
                "focus_dimension": "dynamics",
                "hands": "both"
            },
            {
                "title": "Crescendo isolation",
                "instruction": "Play only the right hand, bars 21-23. Start at pp and arrive at ff by beat 3 of bar 23. No pedal.",
                "focus_dimension": "dynamics",
                "hands": "right"
            },
            {
                "title": "Exaggerated dynamics",
                "instruction": "Play the full passage but exaggerate the crescendo -- make it twice as dramatic as you think it should be. Then scale back.",
                "focus_dimension": "dynamics",
                "hands": "both"
            }
        ]
    }
}
```

**Connections to existing systems:**

- Exercises can be saved to the exercise database (`04-exercises.md`) and tracked for completion
- The condensed reasoning trace records that exercises were generated for this skill
- Synthesized facts can note "student was given dynamics exercises for Nocturne bars 20-24"
- Focus mode sessions can reference these exercises

**Graceful degradation:** Exercise sets are text-based and do not require score data. Always available. If score rendering is unavailable, exercises still render with title and instruction text.

### 4. Reference Browser (`reference_browser`) (Phase 3)

Surfaces professional recordings of the same passage so the student can hear different interpretations. YouTube embeds and Apple Music links.

**Triggers when:** The teacher observation involves interpretation, phrasing, or pedaling -- dimensions where *hearing* the target sound is more useful than *describing* it. Also when the student explicitly asks "what should this sound like?"

**Configuration schema:**

```json
{
    "type": "reference_browser",
    "config": {
        "piece": "Chopin Nocturne Op. 9 No. 2",
        "passage_description": "the crescendo in the second phrase, bars 20-24",
        "references": [
            {
                "artist": "Martha Argerich",
                "source": "youtube",
                "search_query": "Chopin Nocturne Op 9 No 2 Argerich",
                "note": "Builds the crescendo gradually with subtle rubato"
            },
            {
                "artist": "Arthur Rubinstein",
                "source": "youtube",
                "search_query": "Chopin Nocturne Op 9 No 2 Rubinstein",
                "note": "More direct crescendo, arrives suddenly"
            },
            {
                "artist": "Ivo Pogorelich",
                "source": "apple_music",
                "search_query": "Chopin Nocturne Op 9 No 2 Pogorelich",
                "note": "Unconventional dynamics, slower build"
            }
        ],
        "listening_prompt": "Notice how each pianist shapes the crescendo differently. Which approach feels right for how you want to play this?"
    }
}
```

**Prerequisites:**

- Network access (YouTube/Apple Music)
- Search query construction from piece + artist metadata

**Graceful degradation:** If network is unavailable, `reference_browser` is unavailable. Falls back to `text_only`. If a specific search result is not found, skip that reference and show the remaining ones.

---

## Chat Interface

### Web (TanStack Start + React)

- Real-time observations via WebSocket during practice sessions
- React components render inline cards within the chat scroll
- MediaRecorder for audio capture, Web Audio API for processing
- Chat history persists in session state

### iOS (SwiftUI)

- SwiftUI chat view with inline card rendering
- AVAudioEngine for audio capture, chunk upload to API
- SwiftData for local-first chat history persistence
- Cards rendered as native SwiftUI views (not WebView where possible)

### "Tell Me More" Interaction Pattern

Every observation -- text-only or with component -- supports follow-up:

- **"Tell me more"** triggers the existing elaboration flow in the chat
- **"Try exercises for this"** generates an `exercise_set` component for the same passage
- **"How should this sound?"** generates a `reference_browser` component

These actions feed back into the conversation as student messages, and the teacher responds naturally. The chat scroll becomes a timeline of the practice session.

---

## Component Rendering

### How JSON Configs Become UI

The API returns a component configuration (JSON) alongside the observation text. The client is responsible for rendering:

- **Web:** React components keyed by `type`. Each component type (`ScoreHighlightCard`, `KeyboardGuideCard`, `ExerciseSetCard`, `ReferenceBrowserCard`) accepts the `config` object as props and renders accordingly.
- **iOS:** SwiftUI views keyed by `type`. Same component library, native rendering. `ScoreHighlightCard`, `KeyboardGuideCard`, `ExerciseSetCard`, `ReferenceBrowserCard`.

### Chat Layout

Components appear as inline cards in the chat scroll, similar to iMessage rich content:

```
+------------------------------------------+
| [Teacher avatar]                          |
|                                           |
| "The crescendo in the second phrase       |
|  peaked too early -- the sforzando        |
|  didn't land. Try holding back the        |
|  build longer."                           |
|                                           |
| +--------------------------------------+ |
| | [Score Highlight Card]                | |
| |                                       | |
| |  Bars 20-24, Chopin Nocturne Op. 9    | |
| |  [Rendered notation with dynamics     | |
| |   curve and annotations]              | |
| |                                       | |
| |  > "crescendo starts here"   (bar 21) | |
| |  > "peak should land here"  (bar 23)  | |
| +--------------------------------------+ |
|                                           |
| [Try exercises for this] [Tell me more]   |
+------------------------------------------+
```

### Interaction Model

- Cards are scrollable within the chat (the chat scroll is primary)
- Cards can be expanded to full-screen for detail (tap to expand, swipe/click to dismiss)
- When expanded, the artifact takes over the viewport. The chat thread remains accessible via a sidebar or back gesture. This is the "expandable artifact" pattern: the object lives in chat but can grow to fill the screen without leaving the conversation context.
- Cards have action buttons that trigger follow-up conversation turns
- Past cards remain in the chat history for reference

### Graceful Degradation

When a component cannot render (missing score data, network unavailable, unsupported platform capability):

1. The UI subagent (stage 3) is skipped or returns `null`
2. The observation text is delivered without a component
3. The text already contains the full teaching insight -- nothing is lost

No component is ever required. The system always works as a text chat.

### State Between Components

Components in the chat history are snapshots -- they do not update retroactively. If the student improves on a passage, a new observation generates a new card. The chat scroll becomes a timeline of progress.

---

## Platform Differences

| Capability | Web | iOS |
|---|---|---|
| Score rendering | VexFlow/OSMD in DOM | OSMD in WKWebView or native Swift renderer |
| Keyboard guide | Canvas/SVG rendering | Native SwiftUI view (88-key scrollable) |
| Exercise set | React card stack | SwiftUI card stack |
| Reference browser (YouTube) | Embedded iframe player | WKWebView inline player |
| Reference browser (Apple Music) | Link out | MusicKit integration / deep link to Music app |
| Audio capture | MediaRecorder + Web Audio API | AVAudioEngine + ring buffer |
| Chat persistence | Session state (server-backed) | SwiftData local-first |
| Real-time observations | WebSocket | Chunk upload + response polling |
| Offline components | Limited (no service worker yet) | Score highlight + exercise set (cached data) |
| Artifact container | `<Artifact>` React component | `ArtifactView` SwiftUI view |

### V1 vs V2 Component Complexity

Start simple, iterate:

- **Score highlight V1:** Static image of the passage with text annotations overlaid. No interactive scrolling. Generated server-side or via a notation API.
- **Score highlight V2:** Interactive scrolling notation with real-time annotation rendering.
- **Keyboard guide V1:** Static keyboard diagram with highlighted keys. No animation.
- **Keyboard guide V2:** Animated key lighting synchronized to playback.

---

## Open Questions

1. **Component frequency:** How often should the teacher generate a component vs. text-only? Too many cards could feel noisy. Too few and the feature is invisible. The ~30% target from the design philosophy section is confirmed as the right starting point (CEO review 2026-03-19). Increase or decrease based on engagement data.

2. **Score rendering library:** VexFlow (JS, lightweight), OpenSheetMusicDisplay (JS, most capable but heaviest), or a native Swift renderer for iOS? WebView adds latency and feels less native, but notation rendering is hard.

3. **Reference browser content quality:** YouTube search results vary in quality. Should the UI subagent validate results (check video title, duration, relevance) before showing them? Or trust the search and let the student skip irrelevant results?

4. **Preference learning from references:** How quickly can the system learn preferences from reference interactions? After 3 listens of Argerich vs. 1 of Rubinstein, is that enough signal? Or does the student need to explicitly say "I like this one"?

5. **Offline behavior:** Score highlight and exercise set can work offline (data is local or cached). Reference browser requires network. Keyboard guide depends on whether score data is cached. Should offline components degrade or simply not appear?

6. **Component as conversation turn:** When the student interacts with a component (taps "Try it" on an exercise, listens to a reference), should that interaction feed back into the conversation? E.g., "I see you tried the crescendo isolation exercise -- how did it feel?"

7. **Artifact tool use pattern:** Anthropic tool_use vs self-hosted MCP vs structured JSON output. Needs research before implementation.
