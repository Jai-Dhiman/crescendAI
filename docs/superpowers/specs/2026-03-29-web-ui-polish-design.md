# Web UI Polish Design

Five targeted UI improvements for the web practice companion.

## 1. Remove Observation Toasts During Listening Mode

**Current state:** `ObservationToast.tsx` renders a stack of up to 3 toasts inside `ListeningMode.tsx` showing dimension and framing (e.g., "pedaling (recognition)"). A `console.log` in `usePracticeSession.ts:213` also fires.

**Change:** Remove the observation toast rendering from listening mode. Keep the `console.log` for debugging. Observations are accumulated by the session brain and synthesized on exit -- surfacing them as toasts during play is distracting and provides no user value.

**Files:**
- `apps/web/src/components/ListeningMode.tsx` -- remove observation toast stack rendering (~lines 274-284)
- `apps/web/src/components/ObservationToast.tsx` -- delete file (no other consumers)

## 2. Chat Search (Sidebar, Cmd+K)

Follows ChatGPT's established pattern: sidebar search field with keyboard shortcut, filtering the conversation list.

**Trigger:**
- Click the existing search icon in the sidebar (currently a no-op at `AppChat.tsx:624-629`)
- `Cmd+K` (Mac) / `Ctrl+K` (Windows) keyboard shortcut

**UI:**
- When activated, a search input replaces the top of the conversation list area in the sidebar
- Conversations filter as the user types (client-side, case-insensitive title match)
- Results use the same visual treatment as the normal conversation list
- Click a result to open that conversation

**Dismiss:**
- Escape key
- Clear the search input
- Click the X button on the search field

**Scope:** Conversation titles only. Conversations are already loaded client-side via `useConversations`. No backend changes required. No debounce needed -- client-side filtering of a small list is instant.

**Files:**
- `apps/web/src/components/AppChat.tsx` -- wire search button, add search input UI, add `Cmd+K` listener, filter conversation list

## 3. Remove "Pianist" Subtitle

**Current state:** Hardcoded `<span className="text-body-xs text-text-tertiary">Pianist</span>` at `AppChat.tsx:718` under the username in the sidebar profile section.

**Change:** Remove the span. All users are pianists -- it communicates nothing.

**Files:**
- `apps/web/src/components/AppChat.tsx` -- delete the "Pianist" span (~line 718)

## 4. Listening Mode Visual: Circular Audio-Reactive Waveform

Replace `ResonanceRipples.tsx` (canvas-based expanding rings) with a new `AudioWaveformRing.tsx`.

**Idle state:** A perfect circle (thin stroke, sage green) with a subtle slow breathing animation -- slight scale oscillation on a ~4 second cycle. Clearly alive but waiting.

**Playing state:** The ring deforms based on audio frequency data from the microphone's `AnalyserNode`. Points around the circle are displaced outward proportional to frequency bin energy. Louder or more complex playing produces more dramatic deformation. Silence mid-performance causes the ring to settle back toward a circle.

**Technical approach:**
- Canvas-based (same as current implementation)
- Sample 64-128 frequency bins from `AnalyserNode.getByteFrequencyData()`
- Map bins to points distributed around the circle
- Smooth displacement with lerp between frames to avoid jitter
- Apply slight continuous rotation for organic feel
- Color: sage green (RGB 122, 154, 130) -- current brand color
- Opacity/brightness subtly responds to overall energy level
- Size: same footprint as current ripples -- centered in the listening overlay, leaves room for stop button below

**Files:**
- `apps/web/src/components/ResonanceRipples.tsx` -- delete (replaced)
- `apps/web/src/components/AudioWaveformRing.tsx` -- new file, canvas-based circular waveform
- `apps/web/src/components/ListeningMode.tsx` -- swap `ResonanceRipples` for `AudioWaveformRing`, pass `AnalyserNode` reference
- `apps/web/src/hooks/usePracticeSession.ts` -- expose `AnalyserNode` from the audio pipeline (may already be available via the audio context)

## 5. Listening Mode Open/Close Animation Refinement

**Current state:** Radial `clip-path` circle expansion from the record button, 600ms with `cubic-bezier(0.4, 0, 0.2, 1)`, defined in `app.css:289-302`. State machine in `ListeningMode.tsx` manages `collapsed -> expanding -> open -> collapsing -> collapsed`.

### Timing changes

- **Open:** 600ms to 750ms, easing `cubic-bezier(0.16, 1, 0.3, 1)` -- fast launch from button, gentle settle
- **Close:** 600ms to 500ms, same easing -- dismissal feels snappier than entry
- **State machine timeouts:** Update expanding timeout from 650ms to 800ms, collapsing timeout from 600ms to 550ms (match new animation durations with small buffer)

### Edge treatment

- Add a pseudo-element ring that follows the expanding/collapsing `clip-path` boundary
- Thin sage-green border (1-2px) with a soft glow (~8-12px outward, via `box-shadow` or radial gradient)
- The ring fades in at the start of expansion and fades out as it reaches full screen
- On close, the ring fades in as the circle begins shrinking and fades out at the very end

### Content transitions

- Keep the existing 300ms content fade-in after expansion completes
- Add a 200ms fade-out at the start of collapse, before the circle begins shrinking

**Files:**
- `apps/web/src/styles/app.css` -- update `clip-path` transition durations and easing, add pseudo-element ring styles with glow, add content fade-out keyframes
- `apps/web/src/components/ListeningMode.tsx` -- update state machine timeouts, add content fade-out class toggling

## Dependencies and Ordering

Changes 1, 2, 3, and 5 are independent of each other. Change 4 (audio waveform) depends on having access to an `AnalyserNode` from the audio pipeline, which needs to be verified/exposed in `usePracticeSession.ts`.

Recommended implementation order:
1. Change 3 (remove "Pianist") -- trivial, one line
2. Change 1 (remove toasts) -- small, delete component + references
3. Change 5 (animation refinement) -- CSS + timing tweaks
4. Change 2 (chat search) -- new UI feature, moderate scope
5. Change 4 (audio waveform) -- largest change, new component + audio wiring
