# Slice 10: On-Demand UI Components

See `docs/architecture.md` for the full system architecture.
See `docs/apps/06a-subagent-architecture.md` for the two-stage subagent pipeline this extends.
See `docs/apps/09-ios-frontend.md` for the base iOS frontend design.

**Status:** DESIGNED (not implemented)
**Last verified:** 2026-03-03
**Date:** 2026-03-03
**Notes:** Vision/early design. Depends on chat interface (Slice 9), teacher LLM pipeline (Slice 06a), and component rendering infrastructure.

**Goal:** Enable the teacher to generate purpose-built interactive UI components on demand, rendered as inline cards within the chat interface. The chat is the primary surface; components are contextual enhancements that appear when the teacher decides the student needs more than text.

**Tech Stack:** SwiftUI (pre-built component library), structured JSON output from LLM

---

## Vision

The app is a chat interface. The student plays, asks "how was that?", and gets a text observation. But sometimes text isn't enough. A real teacher would grab a pencil and mark up the score, sit down and demonstrate the passage, or say "listen to how Argerich plays this."

CrescendAI can do the same. The teacher LLM decides *what modality* the student needs, and a UI subagent configures the right component. These appear as rich inline cards in the chat -- like iMessage link previews or Apple Pay cards, but interactive and musically aware.

The student scrolls through a conversation that naturally mixes text observations, interactive score views, exercises, and reference recordings. Each card is contextual to the teaching moment.

---

## Three-Stage Pipeline Extension

The existing two-stage pipeline (analysis subagent + teacher LLM) extends to three stages when a visual component is needed:

### Stage 1: Analysis Subagent (Haiku/Flash) -- unchanged

Reasons about which moment matters, why, and what framing to use. Outputs structured handoff.

### Stage 2: Teacher LLM (Sonnet/GPT-4o) -- extended output

Generates the observation text AND declares a **modality**:

```json
{
    "observation": "The crescendo in the second phrase peaked too early...",
    "modality": "score_highlight",
    "modality_context": "Show bars 20-24 with the dynamics curve. Highlight where the crescendo should peak vs. where it actually peaked."
}
```

Possible modality values:

- `text_only` -- no component needed (default, skips stage 3)
- `score_highlight` -- show the passage with annotations
- `keyboard_guide` -- light up keys with timing (beginner-oriented)
- `exercise_set` -- generate targeted practice exercises
- `reference_browser` -- surface professional recordings for comparison

The teacher thinks pedagogically: "they need to *see* this" or "they need to *practice* this" or "they need to *hear* how someone else does this." It doesn't think about UI configuration -- that's stage 3.

### Stage 3: UI Subagent (Haiku/Flash) -- new

Takes the teacher's modality declaration + the analysis context and produces a **component configuration** -- the JSON payload that iOS uses to render the inline card.

The UI subagent thinks like a designer: which component, what data to show, how to configure it, what annotations to add. It has access to the component schema definitions (what fields each component accepts) as its tool/context.

**When stage 3 is skipped:** If `modality` is `text_only`, the Worker returns the observation immediately. No extra latency. Most observations will be text-only -- components are reserved for moments where visual/interactive feedback adds real value.

**Latency:** Stage 3 adds ~0.3-0.5s (Haiku, small structured output). Total for a component response: ~0.5s (analysis) + ~1.5s (teacher) + ~0.4s (UI) = ~2.4s. Within the <3s target.

---

## Component Library

Pre-built SwiftUI components that render from JSON configuration. The LLM selects and configures; it does not generate UI code.

### 1. Score Highlight

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

- Score alignment (chunk timestamps to bar numbers, see `docs/apps/06a-subagent-architecture.md`)
- Score rendering engine (MusicXML/Lilypond to notation -- see Slice 7 exercise rendering)
- Score data for the student's piece (initially from a curated library, later user-uploaded)

**SwiftUI component:** `ScoreHighlightCard` -- renders notation via a rendering library (OSMD/VexFlow in a small WebView, or a native Swift renderer), overlays annotations, optionally shows dynamics curve.

### 2. Keyboard Guide

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
- Piano keyboard SwiftUI component (88-key scrollable, with highlight state per key)

**SwiftUI component:** `KeyboardGuideCard` -- renders a piano keyboard with optional scrolling notation above. Keys light up in sequence at the configured tempo. Student can tap play/pause, adjust speed.

### 3. Exercise Set

Takes the specific passage and skill the teacher identified, generates 2-3 targeted practice variations as interactive cards with instructions.

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

**Connects to existing architecture:**

- Exercises can be saved to the exercise database (Slice 7) and tracked for completion
- The condensed reasoning trace records that exercises were generated for this skill
- Synthesized facts can note "student was given dynamics exercises for Nocturne bars 20-24"
- Future focus mode sessions (Slice 8) can reference these exercises

**SwiftUI component:** `ExerciseSetCard` -- renders a stack of exercise cards, each with title, instruction text, and a "Try it" button. When the student taps "Try it," the app enters a mini-practice mode where MuQ evaluates on the focus dimension only, providing immediate feedback.

### 4. Reference Browser

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

**Context graph connection:** Track which references the student listens to, replays, or asks about. Over time, this feeds into the student model as a preference signal: "student gravitates toward Argerich's dynamic approach" becomes a synthesized fact that informs future teaching and exercise generation.

**SwiftUI component:** `ReferenceBrowserCard` -- renders a list of reference recordings with artist name, note, and a play button. YouTube results open an inline player (WKWebView embed). Apple Music results deep-link to the Music app or show an inline preview if MusicKit is integrated.

---

## Component Rendering in the Chat

Components are inline cards in the chat scroll, similar to iMessage rich content:

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

**Interaction model:**

- Cards are scrollable within the chat (the chat scroll is primary)
- Cards can be expanded to full-screen for detail (tap to expand, swipe to dismiss)
- Cards have action buttons that trigger follow-up: "Try exercises for this" generates an exercise set, "Tell me more" triggers the existing elaboration flow
- Past cards remain in the chat history for reference

**No component is required.** Text-only observations are the default and most common response. Components appear when the teacher decides they add value -- perhaps 20-30% of observations.

---

## Deciding When to Show Components

The teacher LLM (stage 2) makes the modality decision as part of generating the observation. Heuristics the teacher prompt should encode:

| Scenario | Likely modality |
|---|---|
| Observation about a specific bar range with score alignment available | `score_highlight` |
| Beginner learning notes for a new piece | `keyboard_guide` |
| Corrective feedback where practice would help | `exercise_set` |
| Interpretation/phrasing/pedaling where hearing is better than reading | `reference_browser` |
| General encouragement or simple correction | `text_only` |
| No score alignment available (piece not identified) | `text_only` (can't show score) |

The teacher prompt includes the available modalities and when to use them. The decision is pedagogical, not technical: "would a real teacher reach for a visual aid here?"

---

## Connections to Existing Architecture

| Existing concept | How on-demand UI extends it |
|---|---|
| Two-stage subagent pipeline (06a) | Adds stage 3: UI subagent for component configuration |
| Score alignment (06a) | Enables score highlight and keyboard guide components |
| Exercise database (Slice 7) | Exercise builder generates into the same schema |
| Focus mode (Slice 8) | Exercise cards can trigger mini focus-mode sessions |
| Learning arc (06a) | Determines component choice: keyboard guide for beginners, score highlight for advanced |
| Synthesized facts (06a) | Track component interactions (what the student engaged with, which references they liked) |
| iOS frontend (Slice 9) | Chat interface becomes the primary screen; inline cards replace fixed screens |

---

## Implementation Considerations

### Score Data Availability

Most components require score data (MusicXML or Lilypond) for the student's piece. Options:

- **V1: Popular repertoire library.** Curate MusicXML files for the most common student pieces (IMSLP has public domain scores, MuseScore has community-contributed MusicXML). Start with 100-200 pieces covering standard pedagogical repertoire.
- **V2: User upload.** Student uploads their own score (PDF or MusicXML). PDF requires OMR (optical music recognition) -- available via commercial APIs.
- **Graceful degradation:** If no score data is available, `score_highlight` and `keyboard_guide` are unavailable. The teacher falls back to `text_only` or `exercise_set` (text-based exercises don't need notation).

### Component Complexity

Start simple, iterate:

- **Score highlight V1:** Static image of the passage with text annotations overlaid. No interactive scrolling. Generated server-side or via a notation API.
- **Score highlight V2:** Interactive scrolling notation with real-time annotation rendering.
- **Keyboard guide V1:** Static keyboard diagram with highlighted keys. No animation.
- **Keyboard guide V2:** Animated key lighting synchronized to playback.

### State Between Components

Components in the chat history are snapshots -- they don't update retroactively. If the student improves on a passage, a new observation generates a new card. The chat scroll becomes a timeline of progress.

---

## Open Questions

1. **Component frequency:** How often should the teacher generate a component vs. text-only? Too many cards could feel noisy. Too few and the feature is invisible. Hypothesis: start conservative (~20% of observations include a component), increase based on engagement data.

2. **Score rendering library:** VexFlow (JS, WebView), OpenSheetMusicDisplay (JS, WebView), or a native Swift renderer? WebView adds latency and feels less native, but notation rendering is hard. OSMD is the most capable but heaviest.

3. **Reference browser content quality:** YouTube search results vary in quality. Should the UI subagent validate results (check video title, duration, relevance) before showing them? Or trust the search and let the student skip irrelevant results?

4. **Preference learning from references:** How quickly can the system learn preferences from reference interactions? After 3 listens of Argerich vs. 1 of Rubinstein, is that enough signal? Or does the student need to explicitly say "I like this one"?

5. **Offline behavior:** Score highlight and exercise set can work offline (data is local or cached). Reference browser requires network. Keyboard guide depends on whether score data is cached. Should offline components degrade or simply not appear?

6. **Chat history storage:** Component configurations are larger than text observations. How much does this bloat SwiftData? Should component configs be stored separately and referenced by ID?

7. **Component as conversation turn:** When the student interacts with a component (taps "Try it" on an exercise, listens to a reference), should that interaction feed back into the conversation? E.g., "I see you tried the crescendo isolation exercise -- how did it feel?"
