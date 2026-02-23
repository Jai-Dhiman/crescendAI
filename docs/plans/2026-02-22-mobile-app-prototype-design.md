# CrescendAI Mobile App Prototype Design

**Date:** 2026-02-22
**Approach:** SwiftUI prototype (extends existing iOS scaffold)
**Audience:** Investors, early users (pianists), internal team alignment

## Vision

A piano practice companion that a new user can open and immediately understand. No onboarding screens, no account creation, no setup. Play your piano, get feedback. The app feels like a teacher who is always ready, always listening, and can demonstrate what it hears.

Long-term, the system will identify the piece being practiced, assess skill level, track position in the music, guide exercises, set focus areas (e.g., dynamics-only feedback), and play back reference recordings. This prototype validates the UX foundation for that vision.

## App Structure

Two states, not two screens. No tab bar, no navigation chrome.

### State 1: Listening State (default)

The app opens to this. Full-screen, edge-to-edge.

**Layout:**
- Top: CrescendAI wordmark in editorial typography. Below it, a contextual prompt.
  - First-time user: "Play anything -- I'm listening."
  - Returning user: "Last time: Chopin Nocturne Op. 9 No. 2 -- Bars 12-18 dynamics. Pick up where you left off?"
- Center (~60% of screen): The visual pulse. The hero element.
- Bottom: Subtle "CrescendAI is listening" indicator with mic icon. Small settings gear.

**Visual Pulse Behavior:**
- Silent: A still, thin line or gentle circle. Barely there.
- Audio detected: An organic, abstract shape that breathes in response to amplitude, dynamics, and tempo. Not a literal waveform -- something alive and empathetic.
- Uses the editorial color palette (brand accent color against light background).
- Communicates "I'm hearing you," not "I'm measuring you."

**Focus Mode indicator:** When active, the prompt changes to "Focusing on: Dynamics" and the pulse shifts to respond more strongly to the relevant dimension.

**Transition to feedback:** After 3 seconds of silence, the pulse gradually fades to stillness. The feedback chat then slides up from the bottom.

### State 2: Feedback Chat

A modal sheet covering ~85% of the screen, with the listening pulse faintly visible behind it at the top.

**Message structure:**
- AI messages on the left (no avatar, clean editorial body font).
- User messages on the right (when typed, but most interaction is through suggestion chips).
- Messages flow top to bottom, newest at bottom.

**Rich message elements (inline within AI messages):**

1. **Text paragraphs** -- Conversational, specific, teacher-like: "That was a solid run-through. Your legato in the opening phrase is getting smoother, but the crescendo in bars 12-14 drops off too quickly."

2. **Music snippet images** -- Rendered bars with highlights. For non-score-readers: waveform segments with annotations ("this part" with an arrow).

3. **Reference playback buttons** -- Inline audio player: "Here's how that crescendo could build" with a play button. Plays a reference clip from a recording database or rendered MIDI.

4. **Dimension highlight cards** -- Small inline card showing a specific dimension score: "Dynamics: 5.8 / 10" with a one-line interpretation.

**Suggestion chips (bottom of chat):**
Quick-tap actions that adapt to context.

Default chips:
- "Focus on dynamics"
- "Play it again"
- "Show me a reference"
- "What should I work on?"
- "I'm done for today"

After AI suggests focus: "Start focus mode", "Try something else", "Show me what you mean."

**Focus Mode in chat:** Header shows "Focus: Dynamics." AI feedback narrows to that dimension only. Exit via chip or tapping the header indicator. Dismissing the chat returns to the Listening State with focus mode still active.

## First-Time User Experience

No onboarding screens. The chat IS the onboarding.

1. App opens to Listening State. Prompt: "Play anything -- I'm listening."
2. User plays. Pulse comes alive -- the first "aha" moment.
3. User stops. After 3s silence, feedback chat slides up.
4. AI's first message is warm and contextual: "Nice -- I heard something lyrical in a minor key. Here's what stood out to me..." followed by gentle feedback.
5. First-session suggestion chips include "How does this work?" which triggers a brief explanation from the AI about what it listens for and how focus mode works.
6. The AI guides naturally from there. No tutorials, no popups.

## Visual Identity

Editorial style matching the web landing page:
- Light/off-white background
- Strong editorial typography (display, heading, body, label scales)
- Brand accent color for the pulse and interactive elements
- Clean, typographic, structured -- authoritative without being clinical

## Prototype Scope

### Real (device functionality):
- Microphone input and audio level detection
- Visual pulse animation driven by real audio amplitude
- Silence detection (3s threshold triggers feedback)
- Chat UI with message rendering and suggestion chips
- Focus mode state management
- Inline audio playback of reference clips

### Mocked / Hardcoded:
- AI feedback text (pre-written for 3 demo scenarios)
- Music snippet images (pre-rendered, embedded as assets)
- Reference playback audio (pre-loaded clips)
- Dimension scores (hardcoded values)
- Piece detection text (pre-written)
- Context memory ("last session" prompt is hardcoded)

### Demo Scenarios:

1. **First-time user, general feedback** -- User plays anything. Gets a warm general analysis with encouragement and one specific observation. Suggestion chips guide toward focus mode or "how does this work."

2. **Returning user, focus mode** -- App suggests continuing dynamics work. User enters focus mode via chip. Plays. Gets dynamics-only feedback with specific bar references and a dimension score.

3. **Reference playback** -- AI suggests "listen to how this could sound." User taps inline play button. Hears a reference clip. AI follows up with what to listen for in the reference.

### Out of scope:
- Real ML inference
- Account / authentication
- Actual piece identification
- Progress tracking / history persistence
- Settings screen
- Dark mode

## Existing Infrastructure to Build On

The iOS scaffold at `apps/ios/` provides:
- SwiftUI app structure (iOS 16+)
- Design system tokens (Colors, Typography, Spacing)
- Reusable components (CrescendButton, CrescendCard)
- Audio recording and playback (AudioRecorder, AudioPlayer)
- Analysis view with 19-dimension display
- API models (PerformanceDimensions, AnalysisResult)
- Shared Rust crate with UniFFI bindings

Major rework needed:
- Replace tab-based navigation with the two-state model
- Build the visual pulse animation
- Build the rich chat UI with inline elements
- Build the suggestion chip system
- Wire audio level to the pulse
- Pre-populate demo scenario data
