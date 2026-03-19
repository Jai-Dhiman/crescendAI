# Artifact Container System

Unified `<Artifact>` component with three visual states (collapsed, inline, expanded) for rendering rich content inline in chat messages. Beta ships with `exercise_set` as the only artifact type.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Expanded mode | Overlay with backdrop | Simplest for beta; split-pane deferred to Phase 3 with score_highlight |
| Auto-collapse trigger | IntersectionObserver (scroll out of view) | Driven by actual user attention, not arbitrary rules |
| Collapsed visual | Mini card (~56px) | Visual continuity with inline card; title + count + expand chevron |
| Expand affordance | Header icon (top-right) | Established pattern, no interference with accordion interactions |
| Expanded content | Enhanced (complete + progress) | Inline is for scanning; overlay is the workspace |
| Architecture | Artifact wrapper component + Zustand store | Clean separation; store needed for portal communication |

## Component Architecture

```
ChatMessages
  └─ MessageBubble
       └─ Artifact (state machine wrapper)
            ├─ CollapsedPreview (mini card)
            ├─ InlineCard (existing, routes to type-specific cards)
            └─ ExpandedOverlay (portal to document.body)
                 └─ ExerciseSetExpanded (workspace view)
```

### `<Artifact>`

Wraps each `InlineComponent` in `MessageBubble`. Receives `artifactId` (derived from `messageId + component index`) and the `InlineComponent` config. Reads display state from Zustand store. Renders one of three views based on state. Attaches IntersectionObserver for auto-collapse.

Only renders when `message.streaming === false` to prevent layout shifts during streaming.

### `<CollapsedPreview>`

Mini card (~56px tall):
- Left accent bar (4px, `accent` color)
- `target_skill` in `body-sm font-medium text-cream`
- Exercise count badge in `body-xs text-text-tertiary`
- Chevron-right icon as expand affordance
- `source_passage` as single-line truncated subtitle in `body-xs text-text-secondary`
- `bg-surface-card border border-border rounded-xl px-3 py-2`

Interactions:
- Click card body -> `restore(id)` (transitions to inline)
- Click chevron -> `expand(id)` (transitions to expanded)

Generic by design: receives `title`, `subtitle`, `badge` props so future artifact types can provide their own collapsed summary without modifying CollapsedPreview.

### `<InlineCard>` (existing, modified)

The existing type-router. `ExerciseSetCard` gains an `onExpand` callback prop and renders a maximize icon in the top-right of its header.

### `<ArtifactOverlay>`

React portal to `document.body`. Renders once at the app level (in `AppChat.tsx`). Reads `expandedId` and the corresponding `InlineComponent` config from the Zustand store.

Structure:
- Backdrop: `fixed inset-0 bg-black/60 z-50`, click to close
- Content panel: `max-w-2xl mx-auto mt-16 max-h-[80vh] overflow-y-auto bg-surface-card`
- Close button (X) top-right + Escape key handler
- Entry animation: existing `animate-overlay-in` (500ms)
- Body scroll lock on the chat scroll container while open

Dispatches to type-specific expanded renderers (only `ExerciseSetExpanded` for beta).

### `<ExerciseSetExpanded>`

Workspace view for exercises:
- Header: target_skill title + source_passage
- All exercises rendered expanded (no accordion)
- Each exercise shows full instruction text
- Per-exercise action button with merged assign+complete flow:
  - Not assigned: "Start" -> calls `api.exercises.assign()`, transitions to assigned
  - Assigned: "Complete" -> calls `api.exercises.complete()`, transitions to completed
  - No `exercise_id`: button disabled with "Not yet saved" tooltip
- Per-exercise state: `idle | loading | assigned | completing | completed | error`
- Progress indicator at bottom: "1 of 3 completed"

## Zustand Store (`useArtifactStore`)

```typescript
interface ArtifactEntry {
  state: 'collapsed' | 'inline' | 'expanded';
  component: InlineComponent;
}

interface ArtifactStore {
  states: Record<string, ArtifactEntry>;

  register(id: string, component: InlineComponent): void;
  collapse(id: string): void;
  expand(id: string): void;
  restore(id: string): void;
  closeOverlay(id: string): void;
  unregister(id: string): void;
}
```

Constraints:
- Only one artifact expanded at a time. `expand(id)` sets any currently expanded artifact back to `inline`.
- `collapse()` only transitions from `inline` (not from `expanded`).
- Derived selector: `expandedId` returns the ID of the currently expanded artifact (or null).
- `unregister()` removes the entry on unmount.

## IntersectionObserver

Each `<Artifact>` attaches an observer to its root DOM node.

- **Root:** Chat scroll container, passed via `ArtifactScrollContext` (React context wrapping `ChatMessages`).
- **Threshold:** `[0]` -- fires when visibility crosses 0%.
- **Guard:** Only collapses artifacts in `inline` state.
- **Delay:** 200ms timeout before collapsing. Cleared if the element returns to view within 200ms. Prevents flicker during fast scrolling.

## State Transitions

```
                        ┌──────────────┐
          mount ───────>│    inline    │<────── closeOverlay()
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │ scrolled out   │ expand click    │
              ▼ (200ms delay)  ▼                 │
       ┌──────────────┐  ┌──────────────┐       │
       │  collapsed   │  │   expanded   │       │
       └──────┬───────┘  └──────────────┘       │
              │                                  │
     ┌────────┼────────┐                         │
     │ body   │ chevron│                         │
     │ click  │ click  │                         │
     ▼        ▼        │                         │
  restore()  expand()  └─────────────────────────┘
```

## Animations

| Transition | Animation |
|---|---|
| inline -> collapsed | `animate-collapse`: crossfade (200ms ease-out). Height animation as stretch goal -- fallback is opacity swap. |
| collapsed -> inline | Existing `animate-fade-in` (300ms) |
| any -> expanded | Existing `animate-overlay-in` (500ms) on overlay panel, backdrop opacity fade-in (200ms) |
| expanded -> inline | Backdrop fade-out (200ms), panel opacity out (150ms) |

Height animation approach (stretch goal): measure current height with `getBoundingClientRect()`, set explicit height, transition to target. Container uses `overflow: hidden`. Fallback if complex: simple opacity crossfade.

New CSS addition to `app.css`:
- `@keyframes artifact-collapse` (opacity crossfade, optional height transition)
- Backdrop fade-in/fade-out classes

## Files

| File | Action | Purpose |
|------|--------|---------|
| `components/Artifact.tsx` | New | State machine wrapper, IntersectionObserver |
| `components/ArtifactOverlay.tsx` | New | Portal overlay with backdrop |
| `components/cards/CollapsedPreview.tsx` | New | Mini card collapsed view |
| `components/cards/ExerciseSetExpanded.tsx` | New | Workspace expanded view with completion tracking |
| `stores/artifact.ts` | New | Zustand store for artifact state |
| `components/ChatMessages.tsx` | Modified | Swap `InlineCard` for `Artifact`, add `ArtifactScrollContext` |
| `components/cards/ExerciseSetCard.tsx` | Modified | Add `onExpand` prop, expand icon in header |
| `components/InlineCard.tsx` | Modified | Accept and forward `onExpand` prop |
| `components/AppChat.tsx` | Modified | Add `<ArtifactOverlay />` at component root |
| `styles/app.css` | Modified | Add collapse/backdrop animations |

## Edge Cases

1. **Streaming messages:** Artifacts only render after `streaming === false`. Prevents layout shifts.
2. **Multiple components per message:** Each gets its own `<Artifact>` with independent state.
3. **Overlay open during new message:** Overlay stays open. New messages scroll behind backdrop.
4. **Escape key:** Listener registered only when overlay is open. No conflicts.
5. **Body scroll lock:** `overflow: hidden` on chat scroll container (not `document.body`) while overlay open.
6. **Missing exercise_id:** Expanded view disables Start/Complete buttons with explanation text.

## Not in Scope

- Drag-to-resize overlay
- Split-pane layout (Phase 3 with score_highlight)
- Offline/cached artifact state
- Persistence of display state across page reloads (all start as inline)
- Focus mode integration (Phase 3)
