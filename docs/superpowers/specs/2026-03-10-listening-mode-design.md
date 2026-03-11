# Listening Mode: Full-Screen Practice Takeover

## Overview

Replace the current recording top-bar (`RecordingBar`) with a full-screen listening mode. Clicking the record button triggers a radial expansion animation that takes over the entire UI, creating a focused practice environment with dedicated tools. Stopping reverses the animation and returns to chat.

## Transition

### Open (record button -> listening mode)
1. Capture button position via `getBoundingClientRect()`
2. Button scales to 0
3. A circle element at the button's position expands to cover the viewport (~600ms, cubic-bezier easing)
4. Chat fades out during expansion
5. Once full-screen, listening mode content fades in (~300ms)
6. Mic permission + WebSocket connection happen in parallel with the animation

### Close (stop -> chat)
1. Listening mode content fades out
2. Circle contracts back to button position (reverse of open)
3. Chat fades back in
4. Session summary arrives as a normal chat message (includes notes taken during the session)

**No elapsed time display.** Intentional decision to reduce performance pressure during practice.

## Layout (Centered Stage)

```
+--------------------------------------------------+
| [Metronome]   Now practicing                     |
|  icon/logo    Chopin Nocturne Op.9 No.2          |
|               bars 1-16                   [edit] |
+--------------------------------------------------+
|                                                  |
|         [observation toasts overlay here]        |
|                                                  |
|         ~~~ Waveform Visualization ~~~           |
|         (FlowingWaves canvas, full width)        |
|                                                  |
|      7.2    6.8    8.1    7.5    6.4    7.0      |
|      DYN    TIM    PED    ART    PHR    INT      |
|                                                  |
+--------------------------------------------------+
|                         [Notes]    ( STOP )      |
+--------------------------------------------------+
```

### Top Bar
- **Left: Metronome** -- Compact icon/logo. Tapping expands into a control panel.
- **Center/Right: Piece info** -- Auto-populated from chat context extraction. Edit button opens inline text inputs for piece name and section. Falls back to "Unknown piece" with manual entry.

### Center
- **Waveform:** Existing `FlowingWaves` canvas component, expanded to fill available space.
- **Observation toasts:** Real-time observations from the teacher appear as brief toasts overlaid on the waveform area. Same auto-dismiss behavior as current `ObservationToast` (8 seconds). Max 3 visible at once.
- **Dimension scores:** 6 scores in a horizontal row below the waveform. Update in-place with a subtle color pulse on change (no animated counting, no trend lines).

### Bottom Bar
- **Notes button:** Opens the notepad drawer.
- **Stop button:** Large, red, primary action. Ends the session.

## Tools

### Metronome

**Collapsed:** Metronome logo/icon in top-left. If active, pulses on each beat with BPM displayed beside it. If inactive, static icon.

**Expanded:** Clicking the icon opens a panel containing:
- Large BPM display with +/- buttons
- Tap tempo button
- Time signature selector (4/4, 3/4, 6/8)
- On/off toggle
- Accent first beat toggle

Clicking outside or the icon again collapses the panel.

**Audio:** Web Audio API oscillator scheduled via `AudioContext.currentTime` for precise timing. Click sound with optional accent on beat 1.

### Notepad Drawer

Tapping the Notes icon slides a drawer up from the bottom (~40% screen height). Contains a textarea that auto-focuses. Dismiss via tapping outside or a "Done" button. Notes persist for the session duration and are injected client-side into the summary chat message (no backend change required).

### Dimension Scores

Six dimensions displayed: Dynamics, Timing, Pedaling, Articulation, Phrasing, Interpretation. Updated per chunk as WebSocket `chunk_processed` messages arrive. Visual update is a brief color highlight that fades -- minimal distraction during playing.

### Piece/Section Selector

**Current implementation (v1):** Chat context extraction only. Before entering listening mode, scan the current conversation messages client-side for piece/composer/section mentions via an LLM call to the existing API. Runs in parallel with mic setup. If the call fails or is slow, show "Unknown piece" immediately and update asynchronously if a result arrives. Manual edit always available.

**Future:** Audio recognition model (Shazam-style) to identify the piece from the first audio chunks. Separate project.

## State Management

### New component: `ListeningMode.tsx`
- Receives same data as `RecordingBar`: practice state, observations, analyserNode, latestScores, onStop
- Additional props: `pieceContext` (from chat extraction), `onNoteSave`, `originRect` (button position for radial animation)
- Manages internal state: metronome settings, notepad content, panel expanded/collapsed states
- Rendered as a portal (full viewport overlay, z-50)

### Transition orchestration in `AppChat.tsx`
- `ChatInput` forwards a ref to the record button so `AppChat` can call `getBoundingClientRect()` on it
- On `practice.start()`, capture button rect, pass to `ListeningMode` as radial origin
- Fallback origin: center-bottom of viewport if ref is unavailable
- Chat stays mounted but hidden during listening mode (no remount on return)

### Recording pipeline
No changes to `usePracticeSession` hook. ListeningMode consumes its output the same way RecordingBar did.

### Notepad data
Notes stored in component state. On stop, injected client-side into the summary chat message. No backend changes needed.

### Error handling
- WebSocket disconnection or hook error state: exit listening mode immediately (reverse radial animation), return to chat, show an error message in the chat area.
- Mic permission denied: exit before transition completes, show error in chat.

### Summarizing state
When the user hits Stop, the reverse radial animation plays immediately (does not wait for summary). Chat shows a loading indicator while the `session_summary` WebSocket message arrives. Notes are appended to the summary message once it appears.

### Metronome audio isolation
The metronome uses its own `AudioContext` separate from the recording pipeline's context. Output goes to speakers only. The mic may still pick up click sounds -- this is acknowledged and acceptable. Users can turn off the metronome if it affects inference quality.

### Removed
`RecordingBar.tsx` becomes unused. `ListeningMode` fully replaces it.

## Mobile

- Layout structure unchanged; spacing tightens
- Dimension scores wrap to 2 rows of 3 on narrow screens
- Metronome expanded panel renders as a bottom sheet overlay
- Notepad drawer takes ~60% height (vs ~40% desktop)
- Stop button: minimum 48px touch target
- Screen wake lock (`navigator.wakeLock`) requested on entry, released on exit. Progressive enhancement -- fails silently if unsupported.
- Notepad respects virtual keyboard via `visualViewport` handling
