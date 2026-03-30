<!-- /autoplan restore point: /Users/jdhiman/.gstack/projects/Jai-Dhiman-crescendAI/main-autoplan-restore-20260330-091724.md -->
# Web UI Polish Design

Five targeted UI improvements for the web practice companion.

## 1. Remove Observation Toasts During Listening Mode

**Current state:** `ObservationToast.tsx` renders a stack of up to 3 toasts inside `ListeningMode.tsx` showing dimension and framing (e.g., "pedaling (recognition)"). A `console.log` in `usePracticeSession.ts:213` also fires.

**Change:** Remove the observation toast rendering from listening mode. Keep the `console.log` for debugging. Observations are accumulated by the session brain and synthesized on exit -- surfacing them as toasts during play is distracting and provides no user value.

**Cleanup:** Also remove `dismissedObs` state, `setDismissedObs`, and `activeObs` filter from `ListeningMode.tsx` (dead state after toast removal). The `observations` prop can stay on the interface (typed but unused) for future use -- apply `_observations` naming convention.

**Sequencing note:** Ships independently. The existing ResonanceRipples (or its replacement waveform) provides visual feedback that the mic is active.

**Files:**
- `apps/web/src/components/ListeningMode.tsx` -- remove observation toast stack rendering (~lines 274-284), remove `dismissedObs`/`activeObs` state
- `apps/web/src/components/ObservationToast.tsx` -- delete file (no other consumers)

## 2. Chat Search (Sidebar, Cmd+K)

Follows ChatGPT's established pattern: sidebar search field with keyboard shortcut, filtering the conversation list.

**Trigger:**
- Click the existing search icon in the sidebar (currently a no-op at `AppChat.tsx:624-629`)
- `Cmd+K` (Mac) / `Ctrl+K` (Windows) keyboard shortcut. Note: Chrome intercepts Cmd+K for address bar. Handle gracefully -- if browser intercepts, click-to-search still works. Test on target browsers; if Cmd+K is consistently intercepted, fall back to `Cmd+/`.

**UI:**
- When activated, a search input replaces the search button inline (same position in sidebar, not below it)
- Conversations filter as the user types (client-side, case-insensitive title match)
- In search mode, filter against the FULL conversations array (not the `.slice(0, 8)` capped list)
- Results use the same visual treatment as the normal conversation list
- Click a result to open that conversation; search clears and sidebar returns to normal
- If sidebar is collapsed when Cmd+K fires, auto-expand the sidebar first

**Empty state:** "No conversations matching '[query]'" in muted text, centered in the list area.

**Dismiss:**
- Escape key
- Clear the search input
- Click the X button on the search field

**Scope:** Conversation titles only. Conversations are already loaded client-side via `useConversations`. No backend changes required. No debounce needed -- client-side filtering of a small list is instant.

**Implementation notes:**
- Keyboard listener: `document.addEventListener('keydown', handler)` in `useEffect` with cleanup
- Focus sequencing: when sidebar expands on Cmd+K, use `requestAnimationFrame` to focus the input after the DOM updates (the input doesn't exist yet when the state change fires)

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

**Idle-to-active transition:** When `isPlaying` transitions from false to true, blend from breathing animation to frequency-reactive mode over 300-500ms using a crossfade alpha. Do not snap instantly at the first audio sample.

**Technical approach:**
- Canvas-based (same as current implementation)
- Sample 128 frequency bins from `AnalyserNode.getByteFrequencyData()` (fftSize=256, already configured)
- **Logarithmic bin mapping:** Piano fundamentals (27Hz-4186Hz) concentrate in the first ~24 of 128 linear bins. Use log-scale mapping (`Math.pow(binIndex/128, 2)` or mel-scale) to distribute frequency resolution meaningfully around the circle. Without this, half the ring shows silence.
- Smooth displacement with frame-rate-independent lerp: `alpha = 1 - Math.pow(1 - 0.15, dt / 16.67)` where 0.15 is the lerp coefficient (responsive but not jittery for musical content)
- Continuous rotation: ~0.5 deg/frame counterclockwise (~18 deg/sec at 60fps), frame-rate-normalized via `rotation += 0.5 * (dt / 16.67)`
- Color: sage green (RGB 122, 154, 130), opacity range 0.6 (silence) to 1.0 (loud)
- Size: roughly square container (the current `h-32 sm:h-40 md:h-48` band is wider than tall -- adjust container to be more square so the circle reads properly)
- Self-throttle: 30fps cap on low-end devices (detect via `performance.now()` delta; if frame takes >33ms, skip next frame)
- Idle mode: ~15fps (same as current ResonanceRipples throttle)

**AnalyserNode wiring:**
- `usePracticeSession` currently stores the AnalyserNode in a ref (`analyserRef`, line 98) which is not in the return type
- Add `analyserNode: AnalyserNode | null` as state (via `useState`) to `UsePracticeSessionReturn`
- When `start()` creates the analyser (line 364-367), call `setAnalyserNode(analyser)` alongside the ref assignment
- On cleanup, set back to null
- `AudioWaveformRing` must handle null analyser gracefully (show idle breathing) and stop rAF loop when analyser goes away (AudioContext closed)

**Files:**
- `apps/web/src/components/ResonanceRipples.tsx` -- delete (replaced)
- `apps/web/src/components/AudioWaveformRing.tsx` -- new file, canvas-based circular waveform
- `apps/web/src/components/ListeningMode.tsx` -- swap `ResonanceRipples` for `AudioWaveformRing`, pass `analyserNode` prop; adjust waveform container to be more square
- `apps/web/src/hooks/usePracticeSession.ts` -- expose `analyserNode` as state in return type

## 5. Listening Mode Open/Close Animation Refinement

**Current state:** Radial `clip-path` circle expansion from the record button, 600ms with `cubic-bezier(0.4, 0, 0.2, 1)`, defined in `app.css:289-302`. State machine in `ListeningMode.tsx` manages `collapsed -> expanding -> open -> collapsing -> collapsed`.

### Timing changes

- **Open:** 600ms to 750ms, easing `cubic-bezier(0.16, 1, 0.3, 1)` -- fast launch from button, gentle settle
- **Close:** 600ms to 500ms, same easing -- dismissal feels snappier than entry
- **State machine timeouts:** Update expanding timeout from 650ms to 800ms, collapsing timeout from 600ms to 550ms (match new animation durations with small buffer)

### Edge treatment

**Note:** `clip-path` clips the element AND all children/pseudo-elements. A `::before` pseudo-element cannot extend beyond the clip boundary. `box-shadow` is also clipped. The original pseudo-element approach does not work.

**Corrected approach:** Add a sibling `<div>` in the portal (outside `.listening-overlay`, so it is not clipped). Style it as a circular ring with `border: 1.5px solid rgba(122, 154, 130, 0.6)` and `box-shadow: 0 0 12px rgba(122, 154, 130, 0.3)`. Animate its size via CSS custom properties (`--ring-radius`) that mirror the clip-path radius, using the same transition duration and easing. `pointer-events: none` so it doesn't block interaction.

- The ring fades in at the start of expansion and fades out as it reaches full screen
- On close, the ring fades in as the circle begins shrinking and fades out at the very end

### Content transitions

- Keep the existing 300ms content fade-in after expansion completes
- **Content fade-out on close:** Use a `contentFadingOut` state. At collapse start, set `contentFadingOut = true` (content div gets `opacity-0 transition-opacity duration-150`). After 100ms, begin the clip-path shrink (`setPhase("collapsing")`). This overlaps the fade with the start of the shrink for one fluid motion (not two sequential animations with a dead zone).
- Content div stays mounted during fade-out (current code unmounts immediately on `setContentVisible(false)` -- change to tri-state: `visible | fading | hidden`)

### CSS fixes

- Set `clip-path: circle(150vmax at var(--origin-x) var(--origin-y))` on the `open` state instead of `clip-path: none` (prevents jump at close start when browser interpolates from `none`)
- Extract animation timing constants (750ms open, 500ms close) to CSS custom properties for consistency between CSS transitions and JS setTimeout values

### setTimeout cleanup

- Store all `setTimeout` IDs in refs (both `handleStop` and `handleClose` have nested setTimeouts without cleanup)
- Clear refs on component unmount to prevent firing against unmounted state

**Files:**
- `apps/web/src/styles/app.css` -- update `clip-path` transition durations and easing, add ring sibling styles, add content fade-out transition, fix `open` state clip-path
- `apps/web/src/components/ListeningMode.tsx` -- update state machine timeouts, add ring sibling element in portal, add tri-state content visibility, store setTimeout refs

## Dependencies and Ordering

All 5 changes are independent of each other. No cross-change dependencies.

Recommended implementation order:
1. Change 3 (remove "Pianist") -- trivial, one line
2. Change 1 (remove toasts) -- small, delete component + references
3. Change 5 (animation refinement) -- CSS + timing + ring sibling element
4. Change 2 (chat search) -- new UI feature, moderate scope
5. Change 4 (audio waveform) -- largest change, new component + audio wiring

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|---------------|-----------|-----------|----------|
| 1 | CEO | Keep all 5 changes (vs cut 2) | Mechanical | P6 action | Founder in polish-before-beta phase, all changes serve stated goals | Cut chat search + defer waveform |
| 2 | CEO | Mode: SELECTIVE EXPANSION | Mechanical | P3 pragmatic | Feature enhancement on existing system | EXPANSION, HOLD |
| 3 | CEO | Approach A (spec as written) | Mechanical | P1+P5 | Canvas 2D correct tool, all changes needed | Minimal (B), WebGL (C) |
| 4 | CEO | Ship toast removal independently | Taste (user override) | User decision | User chose independent shipping; existing ripples provide visual feedback | Ship with waveform |
| 5 | Eng | Sibling div for edge glow (not pseudo-element) | Mechanical | P5 explicit | clip-path clips pseudo-elements, approach physically impossible | Pseudo-element, SVG, Canvas |
| 6 | Eng | Expose AnalyserNode via useState | Mechanical | P5 explicit | Refs don't trigger re-renders, state does | Return ref.current, callback |
| 7 | Eng | Log-scale bin mapping for piano | Mechanical | P1 completeness | Linear bins waste 50%+ of ring on silence for piano audio | Linear mapping |
| 8 | Eng | Filter full conversation array in search | Mechanical | P1 completeness | .slice(0,8) makes search silently miss results | Filter sliced array |
| 9 | Eng | Tri-state content visibility for fade-out | Mechanical | P5 explicit | Current boolean unmounts immediately, blocking CSS fade | Boolean + setTimeout |
| 10 | Eng | Set explicit clip-path on open state | Mechanical | P3 pragmatic | clip-path: none causes browser interpolation jump on close | Keep none |
| 11 | Design | Overlap fade-out with clip shrink (100ms offset) | Mechanical | P3 pragmatic | Sequential 200ms fade + shrink reads as two animations | Keep sequential |
| 12 | Design | Add search empty state | Mechanical | P1 completeness | Missing empty state looks broken | No empty state |
| 13 | Design | Inline search input at button position | Mechanical | P5 explicit | Input below button is spatially incoherent | Input in list area |
| 14 | Design | Specify lerp coefficient (0.15) | Mechanical | P5 explicit | Unspecified leads to arbitrary implementation choice | Leave unspecified |
| 15 | Design | Specify idle-to-active crossfade (300-500ms) | Mechanical | P1 completeness | Instant snap at first note is jarring | No transition spec |
| 16 | Design | Auto-expand sidebar on Cmd+K | Mechanical | P1 completeness | Undefined behavior when sidebar collapsed | Ignore collapsed state |

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | CLEAR (via /autoplan) | 0 critical gaps, SELECTIVE_EXPANSION |
| Eng Review | `/plan-eng-review` | Architecture & tests | 1 | CLEAR (PLAN via /autoplan) | 11 issues found, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR (via /autoplan) | 12 findings, all resolved |
| Outside Voice | subagent-only | Independent 2nd opinion | 3 | subagent-only | CEO 1/5 confirmed, Eng 6/6, Design 5/7 |

**VERDICT:** APPROVED -- all three reviews passed, 1 critical finding fixed (pseudo-element approach), 16 decisions logged.
