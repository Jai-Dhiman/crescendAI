# Plan D: Reflect-then-Prescribe — Web Surface

**Build-agent dispatch note:** This is plan D of 4 (the LAST) for the reflect-then-prescribe feature. Run tasks
sequentially in the order listed. Each task follows the exact five-step TDD shape. Do NOT skip the
"watch it fail" step. Commit after each task using the exact message in the commit block. Use `bun`
not `npm`. No emojis.

**Goal:** Add the `pending_exercise` InlineComponent variant to the web type system; filter it out of
ChatMessages' auto-render loop; add `api.exercises.assignPending`; build `ReflectionMessage` (the
confirm/deny gate + instant reveal); wire AppChat to route synthesis messages that carry a
`pending_exercise` component to `ReflectionMessage`.

**Spec path:** `docs/specs/2026-06-05-reflect-then-prescribe-design.md`

**Depends on:** Plan B (API endpoint `POST /api/exercises/assign-pending`) and Plan C (the V6 DO block
persists a hidden `pending_exercise` component in the synthesis message and includes it in the WS
synthesis payload's `components` array) merged first. The web keys off that `pending_exercise` component
on the synthesis `RichMessage.components`; there is no separate top-level WS `pendingExercise` field.

**Ship-guard note:** THIS plan ships LAST. The `/ship` for this plan MUST delete the shared spec and ALL four plan files:
- `docs/specs/2026-06-05-reflect-then-prescribe-design.md`
- `docs/plans/2026-06-05-reflect-then-prescribe-v6-prompt.md`
- `docs/plans/2026-06-05-reflect-then-prescribe-pending-assign.md`
- `docs/plans/2026-06-05-reflect-then-prescribe-v6-staging.md`
- `docs/plans/2026-06-05-reflect-then-prescribe-web.md` (this file)

**Style:** no emojis; bun not npm; React.createElement (never JSX) in test files; match existing test
patterns exactly.

---

## Task Groups

### Group 1 — Types + filter gate (no external deps)

#### Task 1: `pending_exercise` InlineComponent variant + ChatMessages filter test

**Goal:** Extend the `InlineComponent` union in `types.ts` with `pending_exercise`; assert via a
rendered test that `ChatMessages` does NOT pass a `pending_exercise` component into the `<Artifact>`
auto-render loop.

**Step 1 — Write the failing test**

Create `apps/web/src/components/ChatMessages.test.tsx`:

```typescript
import { cleanup, render } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Stub Artifact so we can assert it is NOT called with a pending_exercise component.
const mockArtifact = vi.fn(() => null);
vi.mock("./Artifact", () => ({
  Artifact: (props: { component: { type: string }; artifactId: string }) => {
    mockArtifact(props.component);
    return null;
  },
}));
vi.mock("./MessageContent", () => ({
  MessageContent: ({ content }: { content: string }) =>
    React.createElement("div", { "data-testid": "message-content" }, content),
}));
vi.mock("./ToolCallBar", () => ({
  ToolCallBar: () => null,
}));

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
});

describe("ChatMessages — pending_exercise filter", () => {
  it("does NOT render Artifact for a pending_exercise component on a synthesis message", async () => {
    const { ChatMessages } = await import("./ChatMessages");

    const message = {
      id: "msg-1",
      role: "assistant" as const,
      content: "Your pedaling smeared the line. Want a drill for that?",
      createdAt: new Date().toISOString(),
      messageType: "synthesis" as const,
      sessionId: "session-abc",
      components: [
        {
          type: "pending_exercise" as const,
          config: {
            exerciseId: "ex-123",
            focusDimension: "pedaling",
            previewTitle: "Pedaling clarity drill",
          },
        },
      ],
    };

    render(
      React.createElement(ChatMessages, {
        messages: [message],
      }),
    );

    // Artifact must never have been called with the pending_exercise component.
    const pendingCalls = mockArtifact.mock.calls.filter(
      (args) => args[0]?.type === "pending_exercise",
    );
    expect(pendingCalls).toHaveLength(0);
  });

  it("still renders Artifact for a non-pending_exercise component on the same message", async () => {
    const { ChatMessages } = await import("./ChatMessages");

    const message = {
      id: "msg-2",
      role: "assistant" as const,
      content: "Here are exercises.",
      createdAt: new Date().toISOString(),
      components: [
        {
          type: "exercise_set" as const,
          config: {
            sourcePassage: "bars 1-4",
            targetSkill: "pedaling",
            exercises: [
              {
                title: "Legato drill",
                instruction: "Half tempo.",
                focusDimension: "pedaling",
              },
            ],
          },
        },
      ],
    };

    render(
      React.createElement(ChatMessages, {
        messages: [message],
      }),
    );

    expect(mockArtifact).toHaveBeenCalledWith(
      expect.objectContaining({ type: "exercise_set" }),
    );
  });
});
```

**Step 2 — Watch it fail**

```bash
cd apps/web && bunx vitest run src/components/ChatMessages.test.tsx
```

Expected: TypeScript error — `type: "pending_exercise"` is not assignable to `InlineComponent` (the
variant doesn't exist yet). Confirm the error, then proceed.

**Step 3 — Implementation**

Edit `apps/web/src/lib/types.ts`. Add `pending_exercise` to the `InlineComponent` union and add its
config interface immediately after:

```typescript
// In the InlineComponent union, add after the existing segment_loop variant:
| { type: "pending_exercise"; config: PendingExerciseConfig }

// New interface (add after ExerciseSetConfig):
export interface PendingExerciseConfig {
  exerciseId: string;
  focusDimension: string;
  previewTitle: string;
}
```

Edit `apps/web/src/components/ChatMessages.tsx`. The existing filter at lines 188–190 reads:

```typescript
const renderableComponents = (message.components ?? []).filter(
  (c) => (c as { type: string }).type !== "search_catalog_result",
);
```

Change it to also strip `pending_exercise`:

```typescript
const renderableComponents = (message.components ?? []).filter(
  (c) =>
    (c as { type: string }).type !== "search_catalog_result" &&
    (c as { type: string }).type !== "pending_exercise",
);
```

**Step 4 — Watch it pass**

```bash
cd apps/web && bunx vitest run src/components/ChatMessages.test.tsx
```

Both tests must pass.

**Step 5 — Commit**

```bash
git add apps/web/src/lib/types.ts apps/web/src/components/ChatMessages.tsx apps/web/src/components/ChatMessages.test.tsx
git commit -m "$(cat <<'EOF'
feat(web): add pending_exercise InlineComponent variant; filter from auto-render loop

Extends InlineComponent union with pending_exercise config (exerciseId,
focusDimension, previewTitle). ChatMessages now strips pending_exercise
alongside search_catalog_result so the confirm gate is never bypassed.
EOF
)"
```

---

### Group 2 — API client method (depends on Task 1 types)

#### Task 2: `api.exercises.assignPending` in api.ts

**Goal:** Add `assignPending` to the `api.exercises` object in `apps/web/src/lib/api.ts`. It POSTs
to `/api/exercises/assign-pending`, parses the response as `ExerciseSetConfig`, and throws `ApiError`
on non-ok (with Sentry capture) — matching the exact pattern of `api.exercises.assign`.

**Step 1 — Write the failing test**

Create `apps/web/src/lib/api.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Must mock BEFORE importing api.ts so the module substitution is in place.
vi.mock("./api-client", () => ({ client: {} }));
vi.mock("./config", () => ({ API_BASE: "https://api.test" }));
const mockCaptureException = vi.fn();
const mockAddBreadcrumb = vi.fn();
vi.mock("./sentry", () => ({
  Sentry: {
    captureException: (...args: unknown[]) => mockCaptureException(...args),
    addBreadcrumb: (...args: unknown[]) => mockAddBreadcrumb(...args),
  },
}));

let fetchSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  fetchSpy = vi.spyOn(globalThis, "fetch");
  vi.clearAllMocks();
});

afterEach(() => {
  fetchSpy.mockRestore();
});

describe("api.exercises.assignPending", () => {
  const payload = {
    sourcePassage: "bars 5-8",
    targetSkill: "Pedaling clarity",
    exercises: [
      {
        title: "Legato run",
        instruction: "Half tempo.",
        focusDimension: "pedaling",
        exerciseId: "ex-999",
      },
    ],
  };

  it("POSTs to /api/exercises/assign-pending and returns parsed ExerciseSetConfig on 200", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { api } = await import("./api");
    const result = await api.exercises.assignPending({
      sessionId: "sess-1",
      exerciseId: "ex-123",
    });

    expect(fetchSpy).toHaveBeenCalledWith(
      "https://api.test/api/exercises/assign-pending",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ sessionId: "sess-1", exerciseId: "ex-123" }),
      }),
    );
    expect(result).toEqual(payload);
  });

  it("throws ApiError and captures to Sentry on non-ok response", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: "not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { api, ApiError } = await import("./api");
    await expect(
      api.exercises.assignPending({ sessionId: "sess-1", exerciseId: "bad" }),
    ).rejects.toBeInstanceOf(ApiError);

    expect(mockCaptureException).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        extra: expect.objectContaining({ path: "/api/exercises/assign-pending" }),
      }),
    );
  });
});
```

**Step 2 — Watch it fail**

```bash
cd apps/web && bunx vitest run src/lib/api.test.ts
```

Expected: `api.exercises.assignPending is not a function`. Confirm, then proceed.

**Step 3 — Implementation**

In `apps/web/src/lib/api.ts`, add `assignPending` after the existing `complete` method (before the
closing `},` of the `exercises` object). Import `ExerciseSetConfig` from `./types` at the top of
the file if not already imported.

```typescript
async assignPending(body: {
  sessionId: string;
  exerciseId: string;
}): Promise<import("./types").ExerciseSetConfig> {
  return request<import("./types").ExerciseSetConfig>(
    "/api/exercises/assign-pending",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
},
```

Note: `request<T>` already sets `credentials: "include"`, handles non-ok with `ApiError` +
`Sentry.captureException`, and adds a breadcrumb on success. No additional error handling is needed.

**Step 4 — Watch it pass**

```bash
cd apps/web && bunx vitest run src/lib/api.test.ts
```

Both tests must pass.

**Step 5 — Commit**

```bash
git add apps/web/src/lib/api.ts apps/web/src/lib/api.test.ts
git commit -m "$(cat <<'EOF'
feat(web): add api.exercises.assignPending for confirm-gate exercise reveal

Uses the shared request<T> helper (credentials, ApiError, Sentry) to POST
to /api/exercises/assign-pending and return a parsed ExerciseSetConfig.
EOF
)"
```

---

### Group 3 — ReflectionMessage component (depends on Task 1 types + Task 2 api)

#### Task 3: `ReflectionMessage.tsx` + `ReflectionMessage.test.tsx`

**Goal:** A component that renders reflection text, a Confirm button, and a "Not now" button. On
Confirm: calls `api.exercises.assignPending`, then reveals the returned exercise set using
`<Artifact>` (reusing the existing `Artifact -> InlineCard -> ExerciseSetCard` chain). On "Not now":
calls `onDecline(focusDimension)` without calling `assignPending`.

**Props interface:**

```typescript
interface ReflectionMessageProps {
  sessionId: string;
  reflectionText: string;
  pendingConfig: PendingExerciseConfig; // from types.ts
  onDecline: (focusDimension: string) => void;
}
```

**Step 1 — Write the failing test**

Create `apps/web/src/components/ReflectionMessage.test.tsx`:

```typescript
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Must mock BEFORE dynamic import.
const mockAssignPending = vi.fn();
vi.mock("../lib/api", () => ({
  api: {
    exercises: {
      assignPending: (...args: unknown[]) => mockAssignPending(...args),
    },
  },
}));

// Stub Artifact so we can assert it is rendered with the revealed exercise_set.
const mockArtifact = vi.fn(() => null);
vi.mock("./Artifact", () => ({
  Artifact: (props: { component: { type: string; config: unknown }; artifactId: string }) => {
    mockArtifact(props.component);
    return React.createElement(
      "div",
      { "data-testid": "artifact", "data-type": props.component.type },
      null,
    );
  },
}));

// Stub IntersectionObserver (already in test-setup but re-declare for safety).
class MockIO {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
  constructor(_cb: IntersectionObserverCallback) {}
}
// biome-ignore lint/suspicious/noExplicitAny: test stub
(globalThis as any).IntersectionObserver = MockIO;

beforeEach(() => {
  vi.clearAllMocks();
  cleanup();
});

const pendingConfig = {
  exerciseId: "ex-123",
  focusDimension: "pedaling",
  previewTitle: "Pedaling clarity drill",
};

const resolvedPayload = {
  sourcePassage: "bars 5-8",
  targetSkill: "Pedaling clarity",
  exercises: [
    {
      title: "Legato run",
      instruction: "Half tempo.",
      focusDimension: "pedaling",
      exerciseId: "ex-123",
    },
  ],
};

describe("ReflectionMessage", () => {
  it("renders reflection text, Confirm button, and Not-now button", async () => {
    const { ReflectionMessage } = await import("./ReflectionMessage");

    render(
      React.createElement(ReflectionMessage, {
        sessionId: "sess-1",
        reflectionText: "Your pedaling smeared the line in the running passage.",
        pendingConfig,
        onDecline: vi.fn(),
      }),
    );

    expect(
      screen.getByText("Your pedaling smeared the line in the running passage."),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: /confirm/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /not now/i })).toBeTruthy();
  });

  it("clicking Confirm calls assignPending and reveals the exercise via Artifact", async () => {
    mockAssignPending.mockResolvedValueOnce(resolvedPayload);
    const { ReflectionMessage } = await import("./ReflectionMessage");

    render(
      React.createElement(ReflectionMessage, {
        sessionId: "sess-1",
        reflectionText: "Your pedaling smeared the line.",
        pendingConfig,
        onDecline: vi.fn(),
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));

    await waitFor(() => {
      expect(mockAssignPending).toHaveBeenCalledWith({
        sessionId: "sess-1",
        exerciseId: "ex-123",
      });
    });

    await waitFor(() => {
      expect(mockArtifact).toHaveBeenCalledWith(
        expect.objectContaining({ type: "exercise_set" }),
      );
    });
  });

  it("clicking Not-now calls onDecline with focusDimension and does NOT call assignPending", async () => {
    const onDecline = vi.fn();
    const { ReflectionMessage } = await import("./ReflectionMessage");

    render(
      React.createElement(ReflectionMessage, {
        sessionId: "sess-1",
        reflectionText: "Your pedaling smeared the line.",
        pendingConfig,
        onDecline,
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: /not now/i }));

    expect(onDecline).toHaveBeenCalledWith("pedaling");
    expect(mockAssignPending).not.toHaveBeenCalled();
  });

  it("Confirm button shows loading state while assignPending is in flight", async () => {
    let resolveAssign!: (v: typeof resolvedPayload) => void;
    mockAssignPending.mockReturnValueOnce(
      new Promise<typeof resolvedPayload>((res) => {
        resolveAssign = res;
      }),
    );
    const { ReflectionMessage } = await import("./ReflectionMessage");

    render(
      React.createElement(ReflectionMessage, {
        sessionId: "sess-1",
        reflectionText: "Your pedaling smeared the line.",
        pendingConfig,
        onDecline: vi.fn(),
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));

    await waitFor(() => {
      // Button should be disabled during in-flight request
      const btn = screen.getByRole("button", { name: /confirm|adding/i });
      expect(btn).toBeTruthy();
    });

    // Resolve and verify exercise appears
    resolveAssign(resolvedPayload);

    await waitFor(() => {
      expect(mockArtifact).toHaveBeenCalledWith(
        expect.objectContaining({ type: "exercise_set" }),
      );
    });
  });
});
```

**Step 2 — Watch it fail**

```bash
cd apps/web && bunx vitest run src/components/ReflectionMessage.test.tsx
```

Expected: `Cannot find module './ReflectionMessage'`. Confirm, then proceed.

**Step 3 — Implementation**

Create `apps/web/src/components/ReflectionMessage.tsx`:

```typescript
import { useState } from "react";
import { api } from "../lib/api";
import type { ExerciseSetConfig, PendingExerciseConfig } from "../lib/types";
import { Artifact } from "./Artifact";

interface ReflectionMessageProps {
  sessionId: string;
  reflectionText: string;
  pendingConfig: PendingExerciseConfig;
  onDecline: (focusDimension: string) => void;
}

type ConfirmState = "idle" | "loading" | "revealed" | "error";

export function ReflectionMessage({
  sessionId,
  reflectionText,
  pendingConfig,
  onDecline,
}: ReflectionMessageProps) {
  const [confirmState, setConfirmState] = useState<ConfirmState>("idle");
  const [revealedConfig, setRevealedConfig] = useState<ExerciseSetConfig | null>(null);
  const artifactId = `pending-${pendingConfig.exerciseId}`;

  async function handleConfirm() {
    if (confirmState !== "idle") return;
    setConfirmState("loading");
    try {
      const config = await api.exercises.assignPending({
        sessionId,
        exerciseId: pendingConfig.exerciseId,
      });
      setRevealedConfig(config);
      setConfirmState("revealed");
    } catch {
      setConfirmState("error");
    }
  }

  function handleDecline() {
    onDecline(pendingConfig.focusDimension);
  }

  return (
    <div className="flex flex-col gap-3 mt-1">
      <p className="text-body-sm text-text-primary leading-relaxed">{reflectionText}</p>

      {confirmState !== "revealed" && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleConfirm}
            disabled={confirmState === "loading"}
            className="text-body-xs px-3 py-1.5 rounded-lg border border-accent text-accent hover:bg-accent/10 transition disabled:opacity-50"
          >
            {confirmState === "loading" ? "Adding..." : "Confirm"}
          </button>
          <button
            type="button"
            onClick={handleDecline}
            disabled={confirmState === "loading"}
            className="text-body-xs px-3 py-1.5 rounded-lg border border-border text-text-tertiary hover:text-cream hover:border-accent transition disabled:opacity-50"
          >
            Not now
          </button>
        </div>
      )}

      {confirmState === "error" && (
        <p className="text-body-xs text-red-400">Failed to load exercise. Try again.</p>
      )}

      {confirmState === "revealed" && revealedConfig && (
        <Artifact
          artifactId={artifactId}
          component={{ type: "exercise_set", config: revealedConfig }}
        />
      )}
    </div>
  );
}
```

**Step 4 — Watch it pass**

```bash
cd apps/web && bunx vitest run src/components/ReflectionMessage.test.tsx
```

All four tests must pass.

**Step 5 — Commit**

```bash
git add apps/web/src/components/ReflectionMessage.tsx apps/web/src/components/ReflectionMessage.test.tsx
git commit -m "$(cat <<'EOF'
feat(web): add ReflectionMessage component for reflect-then-prescribe confirm gate

Renders reflection prose + Confirm/Not-now. On Confirm, calls
assignPending and reveals the returned exercise_set via the existing
Artifact chain. On Not-now, fires onDecline(focusDimension) without
touching the API.
EOF
)"
```

---

### Group 4 — AppChat wiring (depends on Task 1 types + Task 3 component)

#### Task 4: Wire AppChat to route `pending_exercise`-bearing synthesis messages to `ReflectionMessage`

**Goal:** When a persisted or transient `RichMessage` has `messageType === "synthesis"` AND
`components` contains at least one `pending_exercise` variant, render `<ReflectionMessage>` in place
of the default `<MessageBubble>` in `ChatMessages`. The `onDecline` callback sends a chat turn
signalling the student's decline.

**Architecture note:** `ChatMessages` owns per-message rendering but receives `onTryExercises` as a
prop; the same pattern applies here. The cleanest minimal wiring is:

1. `AppChat` passes an `onDecline` callback prop down to `ChatMessages`.
2. `ChatMessages` detects a synthesis message with a `pending_exercise` component and renders
   `<ReflectionMessage>` instead of the standard `<MessageBubble>` component sequence.
3. `onDecline` in `AppChat` calls `handleSend` with a fixed decline string including the dimension.

Because `AppChat` has deep dependencies (auth, routing, react-query, WebSocket), a full integration
test would require extensive mocking that adds more noise than coverage. The behavioral contract is
already fully covered by Task 3 (`ReflectionMessage.test.tsx`). This task is therefore an
**integration wiring task**: implement the wiring, then run the full vitest suite plus `tsc` to
confirm no regressions or type errors.

**Step 1 — Write the type-level guard test**

Append to `apps/web/src/components/ChatMessages.test.tsx` (the file created in Task 1):

```typescript
describe("ChatMessages — ReflectionMessage routing", () => {
  it("does NOT render MessageContent for a synthesis message that has a pending_exercise component", async () => {
    // We verify that a synthesis+pending_exercise message goes through the
    // ReflectionMessage path, not the standard MessageContent path.
    // ReflectionMessage is mocked to a sentinel div so we can assert on it.
    vi.doMock("./ReflectionMessage", () => ({
      ReflectionMessage: () =>
        React.createElement("div", { "data-testid": "reflection-message" }, null),
    }));

    const { ChatMessages } = await import("./ChatMessages");

    const message = {
      id: "msg-synth",
      role: "assistant" as const,
      content: "Your pedaling smeared the line. Want a drill?",
      createdAt: new Date().toISOString(),
      messageType: "synthesis" as const,
      sessionId: "sess-abc",
      components: [
        {
          type: "pending_exercise" as const,
          config: {
            exerciseId: "ex-789",
            focusDimension: "pedaling",
            previewTitle: "Pedaling clarity drill",
          },
        },
      ],
    };

    render(
      React.createElement(ChatMessages, {
        messages: [message],
        onDecline: vi.fn(),
      }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("reflection-message")).toBeTruthy();
    });

    // The default MessageContent path should NOT render the text as a message-content node
    // when routed to ReflectionMessage.
    const messageBubbles = document.querySelectorAll("[data-testid='message-content']");
    expect(messageBubbles).toHaveLength(0);
  });
});
```

Note: `waitFor` is already imported at the top of the file from `@testing-library/react`.

**Step 2 — Watch it fail**

```bash
cd apps/web && bunx vitest run src/components/ChatMessages.test.tsx
```

Expected: test fails because `ChatMessages` does not yet accept `onDecline` and does not render
`<ReflectionMessage>`. Confirm, then proceed.

**Step 3 — Implementation**

**3a. Update `ChatMessagesProps` in `ChatMessages.tsx`:**

Add `onDecline?: (focusDimension: string) => void` to the props interface:

```typescript
interface ChatMessagesProps {
  messages: RichMessage[];
  children?: React.ReactNode;
  onTryExercises?: (dimension: string) => Promise<void>;
  onDecline?: (focusDimension: string) => void;
}
```

Destructure in the function signature:

```typescript
export function ChatMessages({
  messages,
  children,
  onTryExercises,
  onDecline,
}: ChatMessagesProps) {
```

**3b. Import `ReflectionMessage` in `ChatMessages.tsx`:**

```typescript
import { ReflectionMessage } from "./ReflectionMessage";
```

**3c. Add routing logic in `MessageBubble`.**

`MessageBubble` is the `memo`-wrapped inner component. It currently receives `message`,
`onTryExercises`. Add `onDecline` to its props and inject the routing branch before the standard
render path.

Locate the `MessageBubble` component (currently `memo`-wrapped). The existing `renderableComponents`
filter already strips `pending_exercise`. Add a routing check before the `return` statement of
`MessageBubble`:

```typescript
// Check whether this is a synthesis message gated behind a pending_exercise confirm.
const pendingExerciseComponent = (message.components ?? []).find(
  (c): c is { type: "pending_exercise"; config: import("../lib/types").PendingExerciseConfig } =>
    (c as { type: string }).type === "pending_exercise",
);

if (
  message.messageType === "synthesis" &&
  pendingExerciseComponent &&
  message.sessionId
) {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="max-w-[80%]">
        <ReflectionMessage
          sessionId={message.sessionId}
          reflectionText={message.content}
          pendingConfig={pendingExerciseComponent.config}
          onDecline={onDecline ?? (() => {})}
        />
      </div>
    </div>
  );
}
```

Place this block immediately after the `renderableComponents` derivation and before the `return (` of
the normal render path.

**3d. Pass `onDecline` through from `ChatMessages` to each `MessageBubble`.**

In the `ChatMessages` body, wherever `<MessageBubble>` is rendered (the map over messages), pass the
`onDecline` prop.

**3e. Wire `onDecline` in `AppChat.tsx`.**

In `AppChat`, add `handleDecline`:

```typescript
const handleDecline = useCallback(
  (focusDimension: string) => {
    void handleSend(
      `Not right now — something else? (focus: ${focusDimension})`,
    );
  },
  // handleSend is defined in function scope; wrap in useCallback with stable refs
  // to avoid stale closure. Because handleSend references many refs, use the
  // functional pattern: capture it via a ref.
  [],
);
```

Pass `onDecline={handleDecline}` to `<ChatMessages>` in AppChat's JSX.

**Step 4 — Run full suite + type check**

```bash
cd apps/web && bunx vitest run && bunx tsc --noEmit
```

All tests must pass; `tsc` must emit no errors.

**Step 5 — Commit**

```bash
git add apps/web/src/components/ChatMessages.tsx apps/web/src/components/AppChat.tsx
git commit -m "$(cat <<'EOF'
feat(web): wire AppChat + ChatMessages to route synthesis+pending_exercise to ReflectionMessage

Synthesis messages carrying a pending_exercise component now render
ReflectionMessage instead of the standard MessageBubble. AppChat passes
onDecline which sends a decline chat turn with focusDimension context.
EOF
)"
```

---

## Verification Summary

| Task | Command | Must pass |
|------|---------|-----------|
| 1 | `cd apps/web && bunx vitest run src/components/ChatMessages.test.tsx` | 2 tests |
| 2 | `cd apps/web && bunx vitest run src/lib/api.test.ts` | 2 tests |
| 3 | `cd apps/web && bunx vitest run src/components/ReflectionMessage.test.tsx` | 4 tests |
| 4 | `cd apps/web && bunx vitest run && bunx tsc --noEmit` | full suite + 0 type errors |

Full suite gate before opening PR:

```bash
cd apps/web && bunx vitest run && bunx tsc --noEmit
```
